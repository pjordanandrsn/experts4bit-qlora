# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""``E4B_SERVE_ATTN_INT8=1`` (lane P149, bench/p149/PREREG-p149.md; grouped-nf4-gemm K37): int8 weight-only storage
for the SERVING attention projections. Opt-in; nothing here is a default.

Each attention projection is swapped for :class:`Int8Linear`: its bf16 weight is rounded once, at the swap, to
grouped-nf4-gemm's per-output-row int8 grid (``dense_i8.quantize_rows_int8``: ``s = absmax / 127``, the grid P147 ruled
I8ATTN GO on for Mixtral-8x7B), and only the int8 codes and fp32 scales are kept. There is no persistent bf16 copy.

Three paths, by the call's row count (tokens in the call):
- **1 row** (decode, bucket 1): K37's int8 GEMV, through ``dense_i8.linear_i8``.
- **2 to 16 rows** (decode, buckets 2..16): K37's small-M kernel, through the same entry point.
- **more than 16 rows** (prefill): a TRANSIENT dequantization to ``bf16(q * s)``, then ``F.linear``; the bf16 matrix is
  freed after the call. ``bf16(q * s)`` is P147's fake-quant round trip, bit for bit.

A call that is not a bf16 CUDA call (CPU, another dtype) takes the third path too. Under CUDA graphs, bucket 1 captures
the GEMV and buckets 2..16 the small-M kernel; the codes and scales are made at the swap, so capture holds.

The engagement record (:func:`engagement`) counts, per row count, the MODULES whose calls routed through K37's kernels
and those whose calls dequantized. It counts Python calls: graph captures, their warm-ups and eager calls (a prefill),
never a replay.

Refused: a grouped-nf4-gemm without K37 (``dense_i8``), a biased projection (``linear_i8`` computes ``s * (q @ x)``
only), a weight that is not bf16 ``[N, K]``, a model whose attention holds no plain ``nn.Linear`` projection, and on a
family with a registered census (:data:`CENSUS`) any other count than that census.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

ENV = "E4B_SERVE_ATTN_INT8"
#: the most rows K37's kernels serve; above it a call is prefill-sized and dequantizes transiently
ROWS_MAX = 16
#: model_type -> the projections every decoder layer's attention must have swapped, and no others (P149: Mixtral's 4 x
#: 32 = 128). The names stay, so glue_r2's rope fold (``_patch_attention_rope_only``) still matches the module.
CENSUS = {"mixtral": ("q_proj", "k_proj", "v_proj", "o_proj")}


def _on_device(x) -> bool:
    """The kernels need a CUDA tensor (a module-level seam: CPU tests substitute it)."""
    return x.is_cuda


def kernel():
    """``(linear_i8, quantize_rows_int8)`` from grouped-nf4-gemm's ``dense_i8`` (K37)."""
    try:
        from dense_i8 import linear_i8, quantize_rows_int8
    except ImportError as e:
        raise RuntimeError(f"{ENV}=1 needs grouped-nf4-gemm with dense_i8.linear_i8 (lane K37): {e}") from e
    return linear_i8, quantize_rows_int8


class Int8Linear(nn.Module):
    """A frozen serving projection stored as int8 codes ``q [N, K]`` and fp32 row scales ``s [N]`` only."""

    def __init__(self, lin: nn.Linear, linear, quantize):
        super().__init__()
        if lin.bias is not None:
            raise RuntimeError(f"{ENV}=1: a biased attention projection; linear_i8 computes s * (q @ x) only")
        w = lin.weight
        if w.dtype != torch.bfloat16 or w.dim() != 2:
            raise RuntimeError(f"{ENV}=1: the projection is {w.dtype} {tuple(w.shape)}, not a bf16 [N, K] weight")
        self.in_features, self.out_features = lin.in_features, lin.out_features
        with torch.no_grad():
            q, s = quantize(w.detach())
        self.register_buffer("q", q.contiguous(), persistent=False)   # int8 [N, K]
        self.register_buffer("s", s.contiguous(), persistent=False)   # fp32 [N]
        self.register_parameter("bias", None)    # callers that read .bias (fuse_qkv) see none
        self._linear = linear
        self.routed: dict = {}                   # rows -> calls served by K37's kernels (1: the GEMV; 2..16: small-M)
        self.passed: dict = {}                   # rows -> calls that dequantized transiently (prefill, CPU, other dtype)

    def forward(self, x):
        k = self.in_features
        rows = x.numel() // k if k else 0
        if 1 <= rows <= ROWS_MAX and x.dtype == torch.bfloat16 and _on_device(x):
            self.routed[rows] = self.routed.get(rows, 0) + 1
            out = self._linear(x.reshape(rows, k).contiguous(), self.q, self.s)   # at most 16 rows: a copy if strided
            return out.reshape(*x.shape[:-1], self.out_features)
        self.passed[rows] = self.passed.get(rows, 0) + 1
        w = (self.q.float() * self.s[:, None]).to(torch.bfloat16)      # transient: P147's round trip, freed after
        return F.linear(x, w.to(x.dtype))

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}, int8 rows + fp32 scales"


def attention_projections(model):
    """``(parent, name, child)`` for every projection the swap takes: ``engines.int4_attn.attention_linears``'s rule (the
    plain ``nn.Linear`` children of modules whose class name ends in ``Attention``), so both levers count the same
    projections. A generator, so a swap holds one replaced bf16 projection at a time, never all of them."""
    from .int4_attn import attention_linears
    return attention_linears(model)


def _census(model) -> int:
    """The registered census, read BEFORE anything is swapped (a refusal leaves the model as it was); returns how many
    projections the swap will take."""
    by_mod: dict = {}
    for mod, name, _ in attention_projections(model):
        by_mod.setdefault(id(mod), set()).add(name)
    n = sum(len(v) for v in by_mod.values())
    if n == 0:
        raise RuntimeError(f"{ENV}=1 matched no attention projection (no plain nn.Linear under an *Attention module)")
    cfg = getattr(model, "config", None)
    want = CENSUS.get(getattr(cfg, "model_type", None))
    if want is not None:
        layers = int(cfg.num_hidden_layers)
        if len(by_mod) != layers or any(v != set(want) for v in by_mod.values()):
            raise RuntimeError(f"{ENV}=1 on {cfg.model_type}: the census is exactly {len(want)} per layer "
                               f"({', '.join(want)}) on {layers} layers, {len(want) * layers} in all; found {n} on "
                               f"{len(by_mod)} attention modules")
    return n


def enable_serve_attn_int8(model, *, linear=None, quantize=None) -> int:
    """Swap every attention projection for :class:`Int8Linear` and return how many. The output head is never an
    attention child, so it is untouched. ``linear`` / ``quantize`` substitute the kernel (CPU tests)."""
    if linear is None or quantize is None:
        linear_k, quantize_k = kernel()
        linear = linear_k if linear is None else linear
        quantize = quantize_k if quantize is None else quantize
    want = _census(model)
    n = 0
    for mod, name, child in attention_projections(model):
        setattr(mod, name, Int8Linear(child, linear, quantize))
        n += 1
    if n != want:
        raise RuntimeError(f"{ENV}=1: the census counted {want} projections and the swap took {n}")
    return n


def engagement(model) -> dict:
    """Per row count, how many swapped projections routed calls through K37's kernels and how many dequantized
    transiently (after a build, the graph captures, their warm-ups and any eager call are those calls)."""
    mods = [m for m in model.modules() if isinstance(m, Int8Linear)]
    routed, passed = {}, {}
    for m in mods:
        for r in m.routed:
            routed[r] = routed.get(r, 0) + 1
        for r in m.passed:
            passed[r] = passed.get(r, 0) + 1
    return {"modules": len(mods), "routed_modules_by_rows": {str(k): v for k, v in sorted(routed.items())},
            "passed_modules_by_rows": {str(k): v for k, v in sorted(passed.items())}}
