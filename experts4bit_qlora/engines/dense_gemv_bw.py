# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""``E4B_DENSE_GEMV_BW=1`` (lane P151; grouped-nf4-gemm K35): route the attention projections' single-row decode GEMVs
through ``dense_gemv.gemv_bf16_bw`` instead of cuBLAS.

At one request the decode step's dense bf16 GEMVs are the attention projections: the fused q/k/v (``qkv_proj``,
5120 x 2048 on Qwen3-30B-A3B) and ``o_proj`` (2048 x 4096), 48 of each per step. P148 (``bench/p148``) timed a
prototype GEMV against cuBLAS at those shapes and read it faster in a microbench, a premise; grouped-nf4-gemm K35
ships that prototype's kernel verbatim as ``dense_gemv.gemv_bf16_bw`` under a tolerance contract (relative RMS at
most 1.05 x cuBLAS's own). P151 is the served read; nothing here is a default.

:func:`apply_dense_gemv_bw` swaps each attention module's ``qkv_proj`` and ``o_proj`` (exactly ``nn.Linear``: an int4
or LoRA projection is left alone) for a :class:`DenseGemvLinear` sharing the same weight Parameter (no copy). Its
forward routes ONE bf16 CUDA row to the kernel; every other call (more rows, so a prefill or a decode bucket above 1;
CPU; another dtype) goes to ``F.linear``, unchanged. Under CUDA graphs the routing is decided at capture, by the
bucket's row count: bucket 1 captures the kernel, larger buckets capture cuBLAS.

The engagement record (:func:`engagement`) counts, per row count, the modules whose calls routed and the modules
whose calls passed to ``F.linear``. A served read checks it: routed at 1 row on every swapped module, never above.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

ENV = "E4B_DENSE_GEMV_BW"
TARGETS = ("qkv_proj", "o_proj")


def _on_device(x) -> bool:
    """The kernel needs a CUDA tensor (a module-level seam: CPU tests substitute it)."""
    return x.is_cuda


class DenseGemvLinear(nn.Linear):
    """An ``nn.Linear`` that shares ``src``'s weight and routes a single bf16 CUDA row through ``gemv``.

    It stays an ``nn.Linear`` (``isinstance`` checks elsewhere still hold) without allocating a weight of its own."""

    def __init__(self, src: nn.Linear, gemv):
        nn.Module.__init__(self)
        self.in_features, self.out_features = src.in_features, src.out_features
        self.weight = src.weight                         # the same Parameter: no copy, no second store
        self.register_parameter("bias", None)
        self._gemv = gemv
        self.routed: dict = {}                           # rows -> calls routed through the kernel
        self.passed: dict = {}                           # rows -> calls passed to F.linear

    def forward(self, x):
        rows = x.numel() // self.in_features if self.in_features else 0
        if rows == 1 and x.dtype == torch.bfloat16 and _on_device(x):
            self.routed[1] = self.routed.get(1, 0) + 1
            out = self._gemv(x.reshape(1, self.in_features), self.weight)
            return out.reshape(*x.shape[:-1], self.out_features)
        self.passed[rows] = self.passed.get(rows, 0) + 1
        return F.linear(x, self.weight)


def apply_dense_gemv_bw(model, mode: str, *, gemv=None) -> dict:
    """Swap every attention module's ``qkv_proj`` / ``o_proj`` ``nn.Linear`` for a :class:`DenseGemvLinear` when
    ``mode == "1"``; return the record. ``mode == "0"`` changes nothing. Refused: another mode, a grouped-nf4-gemm
    without ``dense_gemv`` (K35), a projection with a bias or a weight that is not bf16 ``[N, K]`` with contiguous rows,
    and a model where nothing matched (a vacuous lever)."""
    if mode not in ("0", "1"):
        raise ValueError(f"{ENV}={mode!r}: 0 or 1")
    if mode == "0":
        return {"mode": "0", "modules": 0}
    if gemv is None:
        try:
            from dense_gemv import gemv_bf16_bw as gemv
        except ImportError as e:
            raise RuntimeError(f"{ENV}=1 needs grouped-nf4-gemm with dense_gemv (lane K35): {e}") from e
    swapped = []
    for name, mod in model.named_modules():
        for attr in TARGETS:
            lin = getattr(mod, attr, None)
            if type(lin) is not nn.Linear:               # exactly nn.Linear: int4, LoRA or already swapped stay as they are
                continue
            w = lin.weight
            if lin.bias is not None:
                raise RuntimeError(f"{ENV}=1: {name}.{attr} has a bias; gemv_bf16_bw computes W @ x only")
            if w.dtype != torch.bfloat16 or w.dim() != 2 or w.stride(1) != 1:
                raise RuntimeError(f"{ENV}=1: {name}.{attr} is {w.dtype} {tuple(w.shape)} stride {tuple(w.stride())}, "
                                   "not a bf16 [N, K] weight with contiguous rows")
            setattr(mod, attr, DenseGemvLinear(lin, gemv))
            swapped.append((f"{name}.{attr}" if name else attr, tuple(w.shape)))
    if not swapped:
        raise RuntimeError(f"{ENV}=1 matched no qkv_proj / o_proj nn.Linear -- refusing a vacuous lever")
    shapes = sorted({s for _, s in swapped})
    return {"mode": "1", "modules": len(swapped), "shapes": [list(s) for s in shapes],
            "per_target": {t: sum(1 for n, _ in swapped if n.endswith("." + t) or n == t) for t in TARGETS}}


def engagement(model) -> dict:
    """Per row count, how many swapped modules routed through the kernel and how many passed to ``F.linear`` (after a
    build, the graph captures and their warm-ups are those calls)."""
    mods = [m for m in model.modules() if isinstance(m, DenseGemvLinear)]
    routed, passed = {}, {}
    for m in mods:
        for r in m.routed:
            routed[r] = routed.get(r, 0) + 1
        for r in m.passed:
            passed[r] = passed.get(r, 0) + 1
    return {"modules": len(mods), "routed_modules_by_rows": {str(k): v for k, v in sorted(routed.items())},
            "passed_modules_by_rows": {str(k): v for k, v in sorted(passed.items())}}
