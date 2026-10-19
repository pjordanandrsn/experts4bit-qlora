# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""``E4B_INT8_HEAD_BW=1`` (lane P153; grouped-nf4-gemm K36): serve the lm_head on its per-row int8 grid, and route its
single-row decode GEMV through ``dense_gemv.gemv_i8_bw`` instead of cuBLAS.

At one request the lm_head is the decode step's largest dense GEMV (151936 x 2048 on Qwen3-30B-A3B). P148
(``bench/p148``) read an int8 copy of it, with an fp32 scale per row, faster than cuBLAS on the bf16 head in a
microbench, a premise. P150 (``bench/p150``) ruled that rounding the head to that grid keeps P125's quality gate at
BOTH gates, t1 and k16 (GO, on held-out windows). grouped-nf4-gemm K36 ships the prototype kernel and its quantizer
verbatim as ``dense_gemv.gemv_i8_bw`` and ``dense_gemv.quantize_rows_int8``. P153 is the served read; nothing here is
a default.

**Knob-on changes multi-row outputs too.** :func:`apply_int8_head_bw` rounds the head's bf16 weight IN PLACE to its
int8 grid cast back to bf16, exactly P150's I8HEAD round trip (``bench/i8dense/i8_fakequant.roundtrip_``), on the
weight's own device as P150's box ran it (CUDA's division by a scalar multiplies by the reciprocal, so a CPU round trip
of the same weight sits one ulp away in a few percent of rows: 5698 of 151936 in P153's A2000 smoke). So every row
count (a prefill, a decode bucket above 1, the one-row decode) computes the head P150 ruled on, and the served model
never depends on the batch size. That change is covered by P150's GO at both gates. One-row calls then go to
``gemv_i8_bw`` over the same codes and scales: it accumulates ``s * sum(q * x)`` in fp32, without rounding ``q * s`` to
bf16, so it reads the same grid as every other row count, within bf16's output rounding.

The swap is the head only: :class:`Int8HeadLinear` replaces ``get_output_embeddings()`` (exactly ``nn.Linear``) and
shares its weight Parameter, now on the grid, with no second bf16 copy. It holds the int8 codes and the fp32 scales
(+N x K bytes, about 0.31 GB at Qwen3's head). Those bytes arrive AFTER the server sized its slots (free memory is read
before the weights load): at a fixed slot count that is fine, but with an auto ``max_seqs`` on a tight card they can
push a graph capture over. Its forward routes ONE bf16 CUDA row to the kernel; a one-row input that
is not contiguous is copied (one row of K elements), never passed to ``F.linear``. Every other call (more rows; CPU;
another dtype) goes to ``F.linear`` on the rounded weight. Under CUDA graphs bucket 1 captures the kernel and larger
buckets capture cuBLAS on the grid.

Refused: another mode, a grouped-nf4-gemm without K36, a model with no ``nn.Linear`` output head (an int4 head is an
``Int4Linear``), a head tied to the input embeddings (rounding it would round the embeddings), a bias, and a weight that
is not bf16 ``[N, K]`` with contiguous rows. The engagement record (:func:`engagement`) is ``DenseGemvLinear``'s form:
per row count, the modules whose calls routed and those whose calls passed to ``F.linear``.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

ENV = "E4B_INT8_HEAD_BW"
#: rows per step when the head is quantized and rounded: bounds the fp32 temporaries (a whole Qwen3 head in fp32 is
#: 1.2 GB per temporary); the grid is row-wise, so the result is the whole call's, bit for bit
CHUNK_ROWS = 16384


def _on_device(x) -> bool:
    """The kernel needs a CUDA tensor (a module-level seam: CPU tests substitute it)."""
    return x.is_cuda


class Int8HeadLinear(nn.Linear):
    """An ``nn.Linear`` over the head's weight (rounded in place to its int8 grid) that routes a single bf16 CUDA row
    through ``gemv(x, q, s)``. It stays an ``nn.Linear`` (``get_output_embeddings().weight`` still reads the head)."""

    def __init__(self, src: nn.Linear, q, s, gemv):
        nn.Module.__init__(self)
        self.in_features, self.out_features = src.in_features, src.out_features
        self.weight = src.weight                         # the same Parameter, on the grid: no second bf16 copy
        self.register_parameter("bias", None)
        self.register_buffer("q", q, persistent=False)   # int8 [N, K]: not in the state dict
        self.register_buffer("s", s, persistent=False)   # fp32 [N]
        self._gemv = gemv
        self.routed: dict = {}                           # rows -> calls routed through the kernel
        self.passed: dict = {}                           # rows -> calls passed to F.linear

    def forward(self, x):
        rows = x.numel() // self.in_features if self.in_features else 0
        if rows == 1 and x.dtype == torch.bfloat16 and _on_device(x):
            self.routed[1] = self.routed.get(1, 0) + 1
            xr = x.reshape(1, self.in_features)
            if not xr.is_contiguous():
                xr = xr.contiguous()                     # one row of K elements: copied, never a fallback
            out = self._gemv(xr, self.q, self.s)
            return out.reshape(*x.shape[:-1], self.out_features)
        self.passed[rows] = self.passed.get(rows, 0) + 1
        return F.linear(x, self.weight)


def _grid_(weight, quantize, chunk: int = CHUNK_ROWS):
    """``(q, s)`` for ``weight`` by ``quantize`` (``quantize_rows_int8``), and ``weight`` rounded in place to
    ``bf16(q * s)``: P150's I8HEAD round trip, in row chunks."""
    qs, ss = [], []
    with torch.no_grad():
        for a in range(0, weight.shape[0], chunk):
            w = weight[a:a + chunk]
            q, s = quantize(w)
            w.copy_((q.float() * s[:, None]).to(weight.dtype))
            qs.append(q)
            ss.append(s)
    return torch.cat(qs).contiguous(), torch.cat(ss).contiguous()


def apply_int8_head_bw(model, mode: str, *, gemv=None, quantize=None) -> dict:
    """Round the output head to its int8 grid and swap it for an :class:`Int8HeadLinear` when ``mode == "1"``; return
    the record. ``mode == "0"`` changes nothing. Refusals as in the module docstring."""
    if mode not in ("0", "1"):
        raise ValueError(f"{ENV}={mode!r}: 0 or 1")
    if mode == "0":
        return {"mode": "0", "modules": 0}
    if gemv is None or quantize is None:
        try:
            from dense_gemv import gemv_i8_bw, quantize_rows_int8
        except ImportError as e:
            raise RuntimeError(f"{ENV}=1 needs grouped-nf4-gemm with dense_gemv.gemv_i8_bw (lane K36): {e}") from e
        gemv = gemv_i8_bw if gemv is None else gemv
        quantize = quantize_rows_int8 if quantize is None else quantize
    head = model.get_output_embeddings() if hasattr(model, "get_output_embeddings") else None
    if type(head) is not nn.Linear:
        raise RuntimeError(f"{ENV}=1: the output head is {type(head).__name__}, not nn.Linear (an int4 head is an "
                           "Int4Linear, and a swapped head is already Int8HeadLinear) -- refusing")
    emb = model.get_input_embeddings() if hasattr(model, "get_input_embeddings") else None
    if emb is not None and getattr(emb, "weight", None) is not None and emb.weight.data_ptr() == head.weight.data_ptr():
        raise RuntimeError(f"{ENV}=1: the output head is tied to the input embeddings; rounding it would round them too")
    w = head.weight
    if head.bias is not None:
        raise RuntimeError(f"{ENV}=1: the output head has a bias; gemv_i8_bw computes s * (q @ x) only")
    if w.dtype != torch.bfloat16 or w.dim() != 2 or w.stride(1) != 1:
        raise RuntimeError(f"{ENV}=1: the output head is {w.dtype} {tuple(w.shape)} stride {tuple(w.stride())}, not a "
                           "bf16 [N, K] weight with contiguous rows")
    q, s = _grid_(w, quantize)
    model.set_output_embeddings(Int8HeadLinear(head, q, s, gemv))
    return {"mode": "1", "modules": 1, "shape": list(w.shape), "int8_bytes": int(q.numel()),
            "grid": "quantize_rows_int8 (K36); the weight rounded in place to bf16(q * s), P150's I8HEAD"}


def engagement(model) -> dict:
    """Per row count, how many swapped heads routed through the kernel and how many passed to ``F.linear`` (after a
    build, the graph captures and their warm-ups are those calls)."""
    mods = [m for m in model.modules() if isinstance(m, Int8HeadLinear)]
    routed, passed = {}, {}
    for m in mods:
        for r in m.routed:
            routed[r] = routed.get(r, 0) + 1
        for r in m.passed:
            passed[r] = passed.get(r, 0) + 1
    return {"modules": len(mods), "routed_modules_by_rows": {str(k): v for k, v in sorted(routed.items())},
            "passed_modules_by_rows": {str(k): v for k, v in sorted(passed.items())}}
