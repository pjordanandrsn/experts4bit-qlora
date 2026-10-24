# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""``E4B_INT8_HEAD_BW=1`` (lane P153; grouped-nf4-gemm K36): serve the lm_head on its per-row int8 grid, and route its
single-row decode GEMV through ``dense_gemv.gemv_i8_bw`` instead of cuBLAS.

At one request the lm_head is the decode step's largest dense GEMV (151936 x 2048 on Qwen3-30B-A3B). P148
(``bench/p148``) read an int8 copy of it, with an fp32 scale per row, faster than cuBLAS on the bf16 head in a
microbench, a premise. P150 (``bench/p150``) ruled that rounding the head to that grid keeps P125's quality gate at
BOTH gates, t1 and k16 (GO, on held-out windows). grouped-nf4-gemm K36 ships the prototype kernel and its quantizer
verbatim as ``dense_gemv.gemv_i8_bw`` and ``dense_gemv.quantize_rows_int8``. P153 (``bench/p153``) read it FLIP
served; ``serve_paged`` defaults it on for qwen3_moe at that head's shape (``INT8_HEAD_BW_DEFAULT_FAMILIES``). This
module chooses nothing: ``mode`` is the caller's.

**Knob-on changes multi-row outputs too.** :func:`apply_int8_head_bw` rounds the head's bf16 weight IN PLACE to its
int8 grid cast back to bf16, exactly P150's I8HEAD round trip (``bench/i8dense/i8_fakequant.roundtrip_``), on the
weight's own device as P150's box ran it (CUDA's division by a scalar multiplies by the reciprocal, so a row's scale can
differ from the CPU's in its last bit, and a CPU round trip of the same weight differs in a few percent of rows: 5698
of 151936 in P153's A2000 smoke). So every row
count (a prefill, a decode bucket above 1, the one-row decode) computes the head P150 ruled on, and the served model
never depends on the batch size. That change is covered by P150's GO at both gates. One-row calls then go to
``gemv_i8_bw`` over the same codes and scales: it accumulates ``s * sum(q * x)`` in fp32, without rounding ``q * s`` to
bf16, so it reads the same grid as every other row count, within bf16's output rounding.

The swap is the head only: :class:`Int8HeadLinear` replaces ``get_output_embeddings()`` (exactly ``nn.Linear``) and
shares its weight Parameter, now on the grid, with no second bf16 copy. It holds the int8 codes and the fp32 scales
(:func:`copy_bytes`, about 0.31 GB at Qwen3's head). Those bytes arrive AFTER the server sized its slots (free memory
is read before the weights load), so ``serve_paged.resolve_max_seqs`` adds them to the footprint an auto ``max_seqs``
sizes against whenever the knob resolves on. Its forward routes ONE bf16 CUDA row to the kernel; a one-row input that
is not contiguous is copied (one row of K elements), never passed to ``F.linear``. Every other call (more rows; CPU;
another dtype) goes to ``F.linear`` on the rounded weight. Under CUDA graphs bucket 1 captures the kernel and larger
buckets capture cuBLAS on the grid.

``E4B_INT8_HEAD_ROWS`` (K2b, ``bench/dense-small-m/DESIGN-K2.md``) widens the int8 route to 2..value rows (value <= 8):
those calls go to grouped-nf4-gemm K37's ``dense_i8.linear_i8`` over the SAME codes and scales, instead of
``F.linear`` on the bf16 rounding (two bytes an element, twice the int8 head's). One row keeps K36, and more rows keep
``F.linear``. ``linear_i8`` accumulates ``x . q`` in fp32 through ``tl.dot`` and stores ``bf16(s * acc)``: on the same
grid as every other row count, under K37's tolerance contract, never bitwise a row's one-row output. ``1`` is the
one-row-only route and never imports ``dense_i8``. Above 1, it needs the head swapped (``E4B_INT8_HEAD_BW`` resolved
on) and a grouped-nf4-gemm with K37 (0.49.0 or later); each is refused by name. P232 (``bench/p232``) read 8
FLIP_ROWS_8 served; ``serve_paged`` resolves an unset knob to 8 for qwen3_moe at that head's shape when the head is on
and K37 is importable (``INT8_HEAD_ROWS_DEFAULT_FAMILIES``), and to 1 everywhere else. :func:`apply_int8_head_rows`
chooses nothing: ``value`` is the caller's. It records, per row count it routes, the plan ``linear_i8`` runs at the
head's shape (K37's table, else its untimed default).

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
ROWS_ENV = "E4B_INT8_HEAD_ROWS"
#: the most rows ``E4B_INT8_HEAD_ROWS`` may route to K37: the most its reads cover (DESIGN-K2.md section 5), never more
MAX_ROWS = 8
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
        self.rows_max = 1                                # E4B_INT8_HEAD_ROWS: rows 2..rows_max go to self._linear (K37)
        self._linear = None
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
        if 2 <= rows <= self.rows_max and x.dtype == torch.bfloat16 and _on_device(x):
            self.routed[rows] = self.routed.get(rows, 0) + 1
            xr = x.reshape(rows, self.in_features)
            if not xr.is_contiguous():
                xr = xr.contiguous()                     # at most MAX_ROWS rows of K elements: copied, never a fallback
            out = self._linear(xr, self.q, self.s)
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


def copy_bytes(n: int, k: int) -> int:
    """Device bytes the swap adds for an ``[N, K]`` head: the int8 codes (``N x K``) and the fp32 scales (``4 x N``).
    311,772,672 at Qwen3-30B-A3B's 151936 x 2048 head, the rise in ``memory_allocated`` P153's records read."""
    return int(n) * int(k) + 4 * int(n)


def kernel():
    """``(gemv_i8_bw, quantize_rows_int8)`` from grouped-nf4-gemm's ``dense_gemv`` (K36, first released in 0.48.0)."""
    try:
        from dense_gemv import gemv_i8_bw, quantize_rows_int8
    except ImportError as e:
        raise RuntimeError(f"{ENV}=1 needs grouped-nf4-gemm with dense_gemv.gemv_i8_bw (lane K36): {e}") from e
    return gemv_i8_bw, quantize_rows_int8


def kernel_missing() -> str | None:
    """Why :func:`kernel` cannot import here (a grouped-nf4-gemm older than 0.48.0, e.g. under ``--no-deps``), or None."""
    try:
        kernel()
    except RuntimeError as e:
        return f"grouped-nf4-gemm without dense_gemv.gemv_i8_bw (K36, 0.48.0 or later): {e.__cause__}"
    return None


def rows_value(value: str) -> int:
    """``E4B_INT8_HEAD_ROWS`` read as a row bound: empty or ``1`` is ``1`` (one row on K36, more on ``F.linear``);
    ``2`` to :data:`MAX_ROWS` widen it. Anything else is refused. The server reads an unset knob first, as its
    per-family default (``serve_paged._int8_head_rows_env``, lane P232); an explicit value comes here."""
    v = (value or "").strip()
    if v in ("", "1"):
        return 1
    if v.isdigit() and 2 <= int(v) <= MAX_ROWS:
        return int(v)
    raise ValueError(f"{ROWS_ENV}={value!r}: expected 1 to {MAX_ROWS} (1: one row on K36 and more on F.linear; unset: "
                     "per family and head shape, lane P232)")


def rows_kernel():
    """``(linear_i8, plan)`` from grouped-nf4-gemm's ``dense_i8`` (K37, first released in 0.49.0)."""
    try:
        from dense_i8 import linear_i8, plan
    except ImportError as e:
        raise RuntimeError(f"{ROWS_ENV} needs grouped-nf4-gemm with dense_i8.linear_i8 (K37, 0.49.0 or later): {e}") from e
    return linear_i8, plan


def rows_kernel_missing() -> str | None:
    """Why :func:`rows_kernel` cannot import here (a grouped-nf4-gemm older than 0.49.0, e.g. ``[fast]``'s floor or a
    ``--no-deps`` install), or None."""
    try:
        rows_kernel()
    except RuntimeError as e:
        return f"grouped-nf4-gemm without dense_i8.linear_i8 (K37, 0.49.0 or later): {e.__cause__}"
    return None


def apply_int8_head_rows(model, value: int, *, linear=None, plan=None) -> dict:
    """Widen the swapped head's int8 route to 2..``value`` rows through K37's ``linear_i8`` (``rows_kernel``), and
    return the record. ``value == 1`` changes nothing and imports nothing. Refused: a value outside 1..MAX_ROWS, a head
    that is not an :class:`Int8HeadLinear` (``E4B_INT8_HEAD_BW`` did not resolve on), and a package without K37."""
    if not (isinstance(value, int) and 1 <= value <= MAX_ROWS):
        raise ValueError(f"{ROWS_ENV}={value!r}: 1 to {MAX_ROWS}")
    if value == 1:
        return {"rows_max": 1}
    head = model.get_output_embeddings() if hasattr(model, "get_output_embeddings") else None
    if not isinstance(head, Int8HeadLinear):
        raise RuntimeError(f"{ROWS_ENV}={value}: the output head is {type(head).__name__}, not the int8 head "
                           f"({ENV} did not resolve on); this knob only widens that head's route")
    if linear is None:
        linear, plan = rows_kernel()
    head._linear = linear
    head.rows_max = value
    n, k = head.out_features, head.in_features
    return {"rows_max": value, "kernel": "dense_i8.linear_i8 (K37)",
            "plan": {str(m): list(plan(n, k, m)) for m in range(2, value + 1)} if plan is not None else None}


def head_refusal(model) -> str | None:
    """Why ``model``'s output head cannot take the swap, or None: the checks :func:`apply_int8_head_bw` refuses on (the
    text after ``E4B_INT8_HEAD_BW=1: ``), which ``serve_paged``'s per-family default reads as default-off instead."""
    head = model.get_output_embeddings() if hasattr(model, "get_output_embeddings") else None
    if type(head) is not nn.Linear:
        return (f"the output head is {type(head).__name__}, not nn.Linear (an int4 head is an Int4Linear, and a swapped "
                "head is already Int8HeadLinear) -- refusing")
    emb = model.get_input_embeddings() if hasattr(model, "get_input_embeddings") else None
    if emb is not None and getattr(emb, "weight", None) is not None and emb.weight.data_ptr() == head.weight.data_ptr():
        return "the output head is tied to the input embeddings; rounding it would round them too"
    w = head.weight
    if head.bias is not None:
        return "the output head has a bias; gemv_i8_bw computes s * (q @ x) only"
    if w.dtype != torch.bfloat16 or w.dim() != 2 or w.stride(1) != 1:
        return (f"the output head is {w.dtype} {tuple(w.shape)} stride {tuple(w.stride())}, not a bf16 [N, K] weight "
                "with contiguous rows")
    return None


def apply_int8_head_bw(model, mode: str, *, gemv=None, quantize=None) -> dict:
    """Round the output head to its int8 grid and swap it for an :class:`Int8HeadLinear` when ``mode == "1"``; return
    the record. ``mode == "0"`` changes nothing. Refusals as in the module docstring (:func:`head_refusal`)."""
    if mode not in ("0", "1"):
        raise ValueError(f"{ENV}={mode!r}: 0 or 1")
    if mode == "0":
        return {"mode": "0", "modules": 0}
    if gemv is None or quantize is None:
        gemv_k, quantize_k = kernel()
        gemv = gemv_k if gemv is None else gemv
        quantize = quantize_k if quantize is None else quantize
    why = head_refusal(model)
    if why:
        raise RuntimeError(f"{ENV}=1: {why}")
    head = model.get_output_embeddings()
    w = head.weight
    q, s = _grid_(w, quantize)
    model.set_output_embeddings(Int8HeadLinear(head, q, s, gemv))
    return {"mode": "1", "modules": 1, "shape": list(w.shape), "int8_bytes": int(q.numel()),
            "grid": "quantize_rows_int8 (K36); the weight rounded in place to bf16(q * s), P150's I8HEAD"}


def engagement(model) -> dict:
    """Per row count, how many swapped heads routed through a kernel and how many passed to ``F.linear`` (after a
    build, the graph captures and their warm-ups are those calls); with ``E4B_INT8_HEAD_ROWS`` above 1, also which
    kernel each routed row count took."""
    mods = [m for m in model.modules() if isinstance(m, Int8HeadLinear)]
    routed, passed = {}, {}
    for m in mods:
        for r in m.routed:
            routed[r] = routed.get(r, 0) + 1
        for r in m.passed:
            passed[r] = passed.get(r, 0) + 1
    rec = {"modules": len(mods), "routed_modules_by_rows": {str(k): v for k, v in sorted(routed.items())},
           "passed_modules_by_rows": {str(k): v for k, v in sorted(passed.items())}}
    if any(m.rows_max > 1 for m in mods):              # E4B_INT8_HEAD_ROWS: which kernel each routed row count took
        rec["routed_kernel_by_rows"] = {str(k): "K36 gemv_i8_bw" if k == 1 else "K37 linear_i8" for k in sorted(routed)}
    return rec
