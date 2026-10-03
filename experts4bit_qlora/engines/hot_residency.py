"""Hot-expert residency — the constrained-card MoE serving path.

Pins each MoE layer's *hottest* experts permanently in VRAM (computed by the
fused ``grouped-nf4-gemm`` kernel, zero per-token transfer) and streams only
the *cold* tail from pinned host RAM on demand. This is finer-grained than the
whole-layer residency that GGUF runtimes place at (they cannot split experts
within a fused per-layer tensor), and it exploits the empirical fact that MoE
routing, while near-uniform globally, concentrates per layer: on gpt-oss-120b
a per-layer top-16 (12% of experts) captures ~30% of that layer's hits
out-of-sample, so ~12% of the expert VRAM carries ~30% of the traffic at zero
transfer cost.

Regime: this wins where the host CPU is weak and VRAM is small but the link is
not the sole bottleneck (edge boxes, small discrete GPUs). On a strong-CPU
server, computing the cold experts on the host (a GGUF runtime's path) is
faster — that instrument is :mod:`.cold_engine` (``enable_cold_engine``),
which shares this partition and replaces the streaming cold branch with
host-side compute.

The math is identical to the reference ``ExpertsNbit`` forward: the hot and
cold experts are the same NF4 values, merely partitioned by residence, and
both paths decode through the fused kernel with fp32 accumulation. Inference
only (no backward); training forwards fall back to the reference path.

Usage::

    from experts4bit_qlora import enable_hot_residency
    # hot_sets[i] = 1-D LongTensor of hot expert ids for the i-th MoE layer,
    # e.g. derived from a routing-frequency histogram.
    n = enable_hot_residency(model, hot_sets, device="cuda")
"""
from __future__ import annotations

from typing import Sequence

import os

import torch


# PREREG-s2lite: set by a harness that is about to CAPTURE a T > 1
# step, where unique_consecutive's sync is illegal. Off by default --
# grouped routing stays the eager/prefill path's choice.
FORCE_SINGLETON_GROUPS = [False]
# PREREG-s3-grouped-verify: capture-safe DEVICE grouping for T > 1 --
# expert-weight reuse WITH a static, sync-free launch. Takes precedence
# over FORCE_SINGLETON_GROUPS when both are set.
DEVICE_GROUPING = [False]


def _k19_mode_env() -> str:
    """``E4B_INT4_GROUPED_SMALLM``: ``auto`` (the default, also when unset) uses K19 for int4 decode rows above
    T == 1 when the kernel package carries it; ``1`` requires it and routes T == 1 to it too; ``0`` keeps the split-K
    GEMV. Anything else is refused rather than read as one of these."""
    v = os.environ.get("E4B_INT4_GROUPED_SMALLM", "auto").strip().lower() or "auto"
    if v not in ("auto", "0", "1"):
        raise ValueError(f"E4B_INT4_GROUPED_SMALLM={v!r}: expected 'auto', '0' or '1'")
    return v


def _int4_prefill_mode_env() -> str:
    """``E4B_INT4_PREFILL`` (e4b#916): the route of a uniform-int4 store's T > 1 calls that would otherwise take HOST
    grouping. With :data:`DEVICE_GROUPING` off -- the library default, and every ``max_seqs == 1`` server -- that is
    every prefill chunk.

    * ``loop`` (the default, also when unset): the host-grouped branch as it has always been -- one
      ``dequant_int4_ref`` per routed expert per projection, cast to bf16, one matmul each. About 14 launches per
      expert, so about 170k per 512-token chunk on Qwen3-30B-A3B, paid again on every chunk.
    * ``batched``: the same arithmetic, with each projection's routed experts decoded together in slices of
      :data:`_INT4_PREFILL_SLICE` (:func:`_dequant_int4_bf16`). The bf16 weights and the matmuls are the same, so the
      output is bit-identical to ``loop``.
    * ``k19``: device grouping for those calls, served by K19 (``int4_smallm.gemm_int4_b32_grouped_smallm``) at every
      row count: bf16 activations, the int4 weights decoded in registers, bf16 MMA. The operands are the loop's, so
      each output row is within one bf16 ulp of it (K19's contract); only the summation order differs.
    * ``mtile``: device grouping for those calls, served by the grouped int4-b32 M-tile GEMM batched decode uses above
      256 rows (``gemm_int4_b32_grouped_captured``). Its activations are int8 (``quant_x_rows``), so the numerics
      change.

    Under ``k19`` and ``mtile`` a call of at most 256 rows takes the device-grouped decode route it always has (K19 by
    default, see :func:`_k19_mode_env`).

    With :data:`DEVICE_GROUPING` ON (a server capturing batched decode graphs, ``max_seqs > 1``), every T > 1 call is
    already device-grouped, so its prefill rows above 256 have always taken the M-tile: ``loop``, ``batched`` and
    ``mtile`` leave that unchanged, and ``k19`` moves those rows to K19 as well. Anything else is refused rather than
    read as one of these."""
    v = os.environ.get("E4B_INT4_PREFILL", "loop").strip().lower() or "loop"
    if v not in INT4_PREFILL_ROUTES:
        raise ValueError(f"E4B_INT4_PREFILL={v!r}: expected one of {INT4_PREFILL_ROUTES}")
    return v


#: ``E4B_INT4_PREFILL``'s values (e4b#916); the device-grouped ones are the last two.
INT4_PREFILL_ROUTES = ("loop", "batched", "k19", "mtile")


#: Experts decoded per batched pass under ``E4B_INT4_PREFILL=batched``: bounds the transient fp32 decode buffer
#: (16 x 1536 x 2048 x 4 B ~ 0.2 GB for Qwen3-30B-A3B's gate_up) while cutting the launches per expert ~16-fold.
_INT4_PREFILL_SLICE = 16


def _dequant_int4_bf16(packed, scales, ids, N: int, K: int):
    """``dequant_int4_ref(packed[e], scales[e], N, K).to(torch.bfloat16)`` for every ``e`` in ``ids`` at once,
    ``[len(ids), N, K]``. The same elementwise operations in the same precision -- int16 nibbles minus 8, an fp32
    product with the fp16 scale widened to fp32, one rounding to bf16 -- so each slice is bit-identical to the
    per-expert decode (``tests/test_int4_prefill_route.py`` pins it against the reference itself)."""
    G = ids.numel()
    p = packed.index_select(0, ids).to(torch.int16)
    lo = (p & 0xF) - 8
    hi = ((p >> 4) & 0xF) - 8
    q = torch.stack([lo, hi], dim=-1).reshape(G, N, K).float()
    s = scales.index_select(0, ids).float()
    blk = K // s.shape[-1]
    return (q.reshape(G, N, K // blk, blk) * s[..., None]).reshape(G, N, K).to(torch.bfloat16)


def _lean_glue_mode_env() -> str:
    """``E4B_INT4_LEAN_GLUE`` (lane K23): ``auto`` (the default, also when unset) folds the grouping glue around K19 into
    the two kernels that bracket it when the kernel package carries K23's options, and keeps the separate launches when
    it predates them; ``1`` requires K23 (absent is a refusal); ``0`` keeps the separate launches. Lane P89 LICENSED it
    on K19's rows (see :func:`_lean_glue_env`). Anything else is refused."""
    v = os.environ.get("E4B_INT4_LEAN_GLUE", "auto").strip().lower() or "auto"
    if v not in ("auto", "0", "1"):
        raise ValueError(f"E4B_INT4_LEAN_GLUE={v!r}: expected 'auto', '0' or '1'")
    return v


def _lean_glue_env() -> bool:
    """Whether the lean glue is wanted at all (``auto`` or ``1``; see :func:`_lean_glue_mode_env`). The glue folds
    around K19 into the two kernels that bracket it. The tile table is built in ONE launch: no host int32 cast, no pre-zero fills, and the sorted ids from
    the same launch instead of an index_select. gate_up reads the all-resident collapse's token rows directly
    (``gather_div``), so the ``[T * top_k, H]`` expansion is never made. The down projection is stored straight into
    the caller's row order, with no index_copy_ unsort. Bit-identical by construction (grouped-nf4-gemm's K23 contract
    tests; lane P89: tokens identical in all 16 rows of a B=16 step). It applies to K19's rows only, and not to
    gpt-oss's epilogue, which reads the sorted down output."""
    return _lean_glue_mode_env() != "0"


_LEAN_GLUE_SUPPORT: dict = {}


def _lean_glue_supported(builder, k19) -> bool:
    """Whether the installed kernel package carries K23's options (``lean=`` / ``sorted_ids=`` on the builder,
    ``scatter=`` / ``gather_div=`` on K19). Read once per function pair, not per call."""
    key = (builder, k19)
    if key not in _LEAN_GLUE_SUPPORT:
        import inspect
        _LEAN_GLUE_SUPPORT[key] = ({"lean", "sorted_ids"} <= set(inspect.signature(builder).parameters)
                                   and {"scatter", "gather_div"} <= set(inspect.signature(k19).parameters))
    return _LEAN_GLUE_SUPPORT[key]


def _k21_mode_env() -> str:
    """``E4B_MXFP4_GROUPED_SMALLM`` (K21, grouped-nf4-gemm#422): serves the native MXFP4 store's device-grouped decode
    rows through K21 -- K19's grouped small-M tensor-core GEMM on the store's own bytes, over the 16-row device tile
    table -- instead of the kept NF4 stacks (batched rows above :data:`_MXFP4_GEMV_ROWS`). Lane P90 LICENSED it
    (e4b.serve.p90.gptoss.mxfp4.k21-b16.5090.2026-10-01: gpt-oss-20b B=16 x0.581 on an RTX 5090, the store's KL from
    the reference 0.00192 -> 0.00147), with B=1 SLOWER (x1.075). So ``auto`` (the default, also when unset) takes K21
    for rows above T == 1 when the kernel package carries it with its masked K tail, and keeps today's routes when it
    does not; T == 1 stays on the split-K GEMV. ``1`` requires K21 (absent is a refusal) and routes T == 1 to it too
    (:func:`_collapsed_grouping`), which is what lets the decode-shaped KL instrument read the kernel. ``0`` keeps
    today's routes. Anything else is refused."""
    v = os.environ.get("E4B_MXFP4_GROUPED_SMALLM", "auto").strip().lower() or "auto"
    if v not in ("auto", "0", "1"):
        raise ValueError(f"E4B_MXFP4_GROUPED_SMALLM={v!r}: expected 'auto', '0' or '1'")
    return v


#: K21's plan on this route: grouped-nf4-gemm K24's best on gpt-oss-20b's recorded B=16 routing (RTX 5090; BLOCK_N 32,
#: KC 128 with the masked K tail, 4 warps, 3 stages: 7.62 ms/step against the served NF4 route's 15.18). K24 read VOID
#: by its instrument, so this is the treatment an end-to-end lane measures, not a licensed number. Every K24 plan was
#: bit-identical to K21's default, so the plan moves no output bit.
_K21_PLAN = {"block_n": 32, "kc": 128, "warps": 4, "stages": 3}


def _k21_has_masked_tail(mod) -> bool:
    """K21 with the masked K tail (grouped-nf4-gemm#425): KC 128 on gpt-oss's K = 2880 runs as planned instead of being
    lowered or refused."""
    kern = getattr(mod, "_gemm_mxfp4_grouped_smallm", None)
    return "EVEN_K" in (getattr(kern, "arg_names", None) or ())


def _k25_mode_env() -> str:
    """``E4B_NF4_GROUPED_SMALLM`` (K25, grouped-nf4-gemm#429): serves the NF4 store's device-grouped decode rows through
    K25 -- K19's grouped small-M tensor-core GEMM with the NF4 codebook dequant, over the 16-row device tile table --
    instead of the served NF4 M-tile GEMM (``gemm_4bit_grouped_captured``, TF32 on fp32-dequantised weights). Lane P91
    sized it: that GEMM is 62 % (Granite) and 72 % (OLMoE) of B=16 decode kernel time on an RTX 5090. The route runs
    K25 at the served precision (:data:`_K25_PLAN`). Lane P93 read its speed (B=16 x0.594 Granite, x0.598 OLMoE), and
    lane P96 licensed its K8 against the M-tile it replaces, under P95's windowed gate, so ``auto`` is the default (also
    when unset): K25 for rows above T == 1 when the kernel package carries it, the previous routes when it does not.
    ``0`` keeps the previous NF4 routes; ``1`` requires K25 (absent is a refusal) and routes T == 1 to it too
    (:func:`_collapsed_grouping`), which is what lets the decode-shaped K8 instrument read the kernel. Anything else is
    refused."""
    v = os.environ.get("E4B_NF4_GROUPED_SMALLM", "auto").strip().lower() or "auto"
    if v not in ("auto", "0", "1"):
        raise ValueError(f"E4B_NF4_GROUPED_SMALLM={v!r}: expected 'auto', '0' or '1'")
    return v


#: K25's plan on this route: the select-tree codebook decode at the served kernel's weight precision -- fp32 weights
#: through TF32 MMA (``dot_bf16=False``) -- at lane K27's best TF32 plan (BLOCK_N 32, KC 64, 4 warps, 3 stages).
#: grouped-nf4-gemm's K26 read the codebook lookup as about 80 % of K25's time and the exact tree as its fix; K27 (RTX
#: 5090, NF4 families' B=16 shapes) read this arithmetic at 0.448 / 0.502 of the served NF4 GEMM's time with the served
#: kernel's error (ratio 1.000). Lane P92 read K25 in bf16, and OLMoE's K8 failed there; this route is the served
#: precision class, which a lane reads before any default. A kernel package without the tree decode gets the paired
#: lookup (:func:`_k25_plan`), the same fp32 weights.
_K25_PLAN = {"block_n": 32, "kc": 64, "warps": 4, "stages": 3, "lut": "tree", "dot_bf16": False}


def _k25_plan(nf4_smallm_mod) -> dict:
    """:data:`_K25_PLAN`, with the codebook decode the installed kernel package carries: ``"tree"`` when its
    ``_LUT_MODES`` names it, else ``"pair"`` (bit-identical; lane K26)."""
    modes = getattr(nf4_smallm_mod, "_LUT_MODES", None) or {}
    return _K25_PLAN if "tree" in modes else dict(_K25_PLAN, lut="pair")


def _collapsed_grouping(T: int, int4_stores) -> tuple[bool, bool]:
    """``(singleton_groups, device_grouping)`` for the all-resident collapse at ``T`` tokens.

    T == 1 takes singleton groups and T > 1 takes device grouping under :data:`DEVICE_GROUPING` -- except that
    K19's opt-in (``E4B_INT4_GROUPED_SMALLM=1`` with a uniform-int4 store) names every int4 DECODE row, T == 1
    included, so T == 1 takes the device tile table too (capture-legal, no host sync) and its rows reach K19.
    Without that the singleton GEMV would keep T == 1, and K8, which scores through the T == 1 loop, would read
    the GEMV instead of the kernel it gates (lane P87). K21's opt-in does the same on the MXFP4 store, and K25's
    (``E4B_NF4_GROUPED_SMALLM=1``) on the NF4 store.

    A T > 1 call that would take HOST grouping on a uniform-int4 store -- every prefill chunk with
    :data:`DEVICE_GROUPING` off -- takes device grouping instead under ``E4B_INT4_PREFILL=k19`` or ``mtile`` (e4b#916;
    see :func:`_int4_prefill_mode_env`). Singleton groups, when chosen, are left alone."""
    k19_t1 = (T == 1 and int4_stores is not None and int4_stores.get("kind") != "mxfp4"
              and _k19_mode_env() == "1")
    k21_t1 = (T == 1 and int4_stores is not None and int4_stores.get("kind") == "mxfp4"
              and _k21_mode_env() == "1")
    k25_t1 = T == 1 and int4_stores is None and _k25_mode_env() == "1"
    nf4_t1 = T == 1 and int4_stores is None and _nf4_t1_device_grouping_env()
    if k19_t1 or k21_t1 or k25_t1 or nf4_t1:
        return False, True
    singleton = T == 1 or (FORCE_SINGLETON_GROUPS[0] and not DEVICE_GROUPING[0])
    device = DEVICE_GROUPING[0] and T > 1
    if (not singleton and not device and int4_stores is not None and int4_stores.get("kind") != "mxfp4"
            and _int4_prefill_mode_env() in ("k19", "mtile")):
        device = True
    return singleton, device


def _nf4_t1_device_grouping_env() -> bool:
    """``E4B_NF4_T1_DEVICE_GROUPING=1`` (an instrument; off by default, also when unset): the NF4 store's T == 1 decode
    rows take the device tile table, as batched rows do, instead of the singleton decode GEMV. With
    ``E4B_NF4_GROUPED_SMALLM=0`` set explicitly (the default is ``auto`` since lane P96) they then run the served NF4
    M-tile GEMM (``gemm_4bit_grouped_captured``, TF32 on
    the fp32 dequant) -- the arithmetic B=16 decode serves today -- so the decode-shaped K8 can read it at T == 1. It
    exists for lane P94's calibration (how far the production path's own arithmetic variants move K8, on OLMoE in
    particular; lanes P92/P93) and changes no default route. Anything other than ``0``/``1`` is refused."""
    v = os.environ.get("E4B_NF4_T1_DEVICE_GROUPING", "0").strip() or "0"
    if v not in ("0", "1"):
        raise ValueError(f"E4B_NF4_T1_DEVICE_GROUPING={v!r}: expected '0' or '1'")
    return v == "1"


def _decode_a16_default() -> bool:
    """``E4B_INT4_DECODE_A16=1`` (lane P64, e4b#709): the default of
    :data:`DECODE_A16`, read once at import."""
    return os.environ.get("E4B_INT4_DECODE_A16", "0") == "1"


# P64 (e4b#709): a QUALITY INSTRUMENT, off by default. When set, the int4-b32
# store's singleton-groups calls -- T == 1 decode in every default
# configuration, which would otherwise run gemv_int4_b32 on quant_x_rows'
# int8 activations -- take the prefill branch instead: dequant each routed
# expert, bf16 matmul. Same int4 bytes, bf16 activations, no new kernel. The
# device-grouped batched-decode routes and the MXFP4 store are NOT covered.
# The dequant loop reads the expert ids on the host, so the flag is
# eager-only (it refuses under CUDA-graph capture) and slower. The
# environment sets the default at import; a harness may flip it in-process.
DECODE_A16 = [_decode_a16_default()]


#: Rows up to which the MXFP4 store's decode GEMV beats the alternatives
#: (bo3n: x1.22 over NF4 at 4 rows, x0.81 at 64); above it the NF4 stacks,
#: when kept, serve the call.
_MXFP4_GEMV_ROWS = 16

_SWIGLU = {}


def _swiglu_kernel():
    """The kernel side's ``swiglu_rows`` (silu(gate) * up over a
    gate-block-then-up-block row, one launch) or None. Cached; the
    ``E4B_FUSE_SWIGLU=0`` A/B arm disables it."""
    if "k" not in _SWIGLU:
        k = None
        if os.environ.get("E4B_FUSE_SWIGLU", "1") == "1":
            try:
                from int4_b32 import swiglu_rows as k
            except ImportError:
                k = None
        _SWIGLU["k"] = k
    return _SWIGLU["k"]


def _is_silu(act_fn):
    if act_fn is torch.nn.functional.silu:
        return True
    return type(act_fn).__name__ in ("SiLU", "SiLUActivation")


def _swiglu_or(act_fn, gu, gate, up):
    """``silu(gate) * up`` through the fused kernel when the activation
    IS silu, the rows are bf16 on a CUDA device and the kernel exists;
    the torch chain (chunk, silu, mul: three launches) otherwise. The
    kernel computes in fp32 and rounds once, like the chain."""
    k = _swiglu_kernel()
    if (k is not None and gu.is_cuda and gu.dtype == torch.bfloat16
            and _is_silu(act_fn)):
        return k(gu)
    return act_fn(gate) * up


_COMBINE = {}


def _combine_kernel():
    """The kernel side's ``combine_rows`` (top-k weight, sum, bf16 cast in
    one launch) or None; ``E4B_FUSE_COMBINE=0`` is the A/B arm."""
    if "k" not in _COMBINE:
        k = None
        if os.environ.get("E4B_FUSE_COMBINE", "1") == "1":
            try:
                from int4_b32 import combine_rows as k
            except ImportError:
                k = None
        _COMBINE["k"] = k
    return _COMBINE["k"]

def _int4_part_or_none(st, rows: int, device):
    """The store's preallocated split-K partials buffer when it fits THIS call, else None.

    ``enable_serve_experts_int4`` sizes ``st["part"]`` for one token's ``top_k`` rows (the
    B=1 capture pattern: no allocation inside a captured decode step). The singleton branch
    also runs at T > 1 under ``FORCE_SINGLETON_GROUPS`` -- the S2 verify default -- where the
    GEMV has ``T * top_k`` rows and plans its split count from them. Handed the small buffer
    there, it wrote past the end (lane P63, measured on an A2000: 16,384 fp32 elements past a
    128-row buffer at T = 17). The kernel uses the first ``sk * rows`` rows, so a buffer with at
    least that many is kept, exactly as before; a shorter one is replaced by None, and the wrapper
    allocates its own -- through the graph pool under capture, the pattern the device-grouping
    GEMV branch already uses. The arithmetic is the same either way: ``sk`` is the call's own plan."""
    part = st.get("part")
    if part is None:
        return None
    # memoised per (rows, device): this runs on every decode expert call, and the eager B=1 step is
    # host-bound, so the plan is computed once per shape, then it costs one dict lookup
    fits_by = st.setdefault("_part_fits", {})
    key = (rows, str(device))
    fits = fits_by.get(key)
    if fits is None:
        from int4_b32 import _plan
        try:
            from int4_b32 import _sm_count      # grouped-nf4-gemm >= 0.31: the plan takes the row count
            _bn, _wp, sk, _ku = _plan(st["N"], st["K"], rows, _sm_count(device))
        except ImportError:                     # 0.30.x (the [fast] floor): the N-only plan, as its wrapper uses
            _bn, _wp, sk, _ku = _plan(st["N"], st["K"])
        fits = part.dim() == 2 and part.shape[1] == st["N"] and part.shape[0] >= sk * rows
        fits_by[key] = fits
    return part if fits else None


#: Calibration tap for the expert Hessians (engines/int4_experts):
#: ``sink(gu_p, sorted_ids, x_sorted, h)`` is called once per MoE call
#: with the gate/up input rows and the down-projection input rows, in
#: expert-sorted order with their local expert ids -- the two matrices a
#: per-expert GPTQ pack needs. ``None`` in serving: no cost, no branch
#: inside the kernels.
_CALIB_SINK = None


def _fused_over_stack(x_rows, local_ids, gu_p, gu_a, dn_p, dn_a, shapes, has_gate,
                      act_fn, gptoss=None, clamp_limit=None,
                      singleton_groups=False, device_grouping=False,
                      int4_stores=None, x_tokens=None):
    """Down-projection outputs for each (token,slot) row, computed on the device
    the packed stack lives on. ``local_ids`` index into the G-expert stack
    (``gu_p`` is ``[G, n1, k1//2]`` etc.). Returns ``[R, H]`` in the input row
    order (unweighted; the caller applies router scores and scatters).

    ``gptoss``, when given, is ``(gu_bias, dn_bias, alpha, limit)`` where the
    bias stacks are aligned to the SAME local indexing as ``gu_p``/``dn_p``.
    It selects the gpt-oss expert epilogue — per-expert biases + the clamped
    GLU ``(up+1)*(gate*sigmoid(alpha*gate))`` — instead of the plain
    ``act_fn(gate)*up``. gpt-oss weights are de-interleaved to a contiguous
    ``[gate; up]`` layout at load (``gptoss.py``), so ``chunk(2)`` is the
    correct split here (NOT ``[...::2]``). Mirrors ``_GptOssForwardMixin.forward``
    exactly (the correctness oracle).

    ``clamp_limit``, when given, selects the DeepSeek-V4 epilogue: no biases, but
    ``gate.clamp(max=L)`` / ``up.clamp(-L, L)`` before the ordinary ``act_fn(gate)*up``,
    evaluated in **fp32** and cast back for the down GEMM. It is mutually exclusive with
    ``gptoss`` — same clamps, different combination — and mirrors
    ``_DeepseekV4ForwardMixin.forward`` in both structure and precision. Threading it
    matters: an expert module whose forward is allowlisted for patching but whose
    epilogue this function does not reproduce gets silently served plain SwiGLU.

    ``x_tokens``, when given with ``x_rows=None``, is ``(x, row_token, top_k)``: the step's TOKEN rows and the
    (token, slot) -> token map, from the all-resident collapse. K23's lean K19 route reads the token rows directly
    (``gather_div=top_k``); every other route gets ``x.index_select(0, row_token)``, the rows it has always been given."""
    from nf4_grouped import gemm_4bit_grouped

    n1, k1, n2, k2 = shapes
    # A native-MXFP4 store (gpt-oss) serves EVERY shape through the kernel
    # side's grouped MXFP4 GEMM on the singleton contract (one row per
    # group, ids as a device tensor): no int4-b32 kernels, no tile table,
    # no unsort. The mxfp4 kernel has no captured M-tile variant yet, so
    # the device-grouping branch is not taken for this store kind.
    if x_rows is None and x_tokens is None:
        raise ValueError("_fused_over_stack needs x_rows or x_tokens")
    R_rows = x_rows.shape[0] if x_rows is not None else local_ids.numel()
    _mxfp4_store = int4_stores is not None and int4_stores.get("kind") == "mxfp4"
    # K21 (E4B_MXFP4_GROUPED_SMALLM, auto since P90): the store's device-grouped decode rows through the grouped
    # small-M MXFP4 GEMM, at every row count up to the decode bound -- no NF4 fallback. Under auto only rows above
    # T == 1 are device-grouped on this store (T == 1 stays on the GEMV, see _collapsed_grouping); "1" adds T == 1.
    _k21 = None
    _k21_mode = _k21_mode_env()
    if _mxfp4_store and device_grouping and R_rows <= 256 and _k21_mode != "0":
        try:
            import mxfp4_grouped as _mx
            _k21 = _mx.gemm_mxfp4_grouped_smallm
        except (ImportError, AttributeError) as e:
            if _k21_mode == "1":
                raise RuntimeError(
                    "E4B_MXFP4_GROUPED_SMALLM=1 needs grouped-nf4-gemm with K21 "
                    "(mxfp4_grouped.gemm_mxfp4_grouped_smallm, grouped-nf4-gemm#422)") from e
            _mx = None                    # auto: the kernel package predates K21 -> today's route
        if _mx is not None and not _k21_has_masked_tail(_mx):
            if _k21_mode == "1":
                raise RuntimeError(
                    "E4B_MXFP4_GROUPED_SMALLM=1 needs K21's masked K tail (grouped-nf4-gemm#425): "
                    f"the route's plan runs KC {_K21_PLAN['kc']} on K that it need not divide")
            _k21 = None                   # auto: K21 without the masked tail would be lowered -> today's route
    if (_mxfp4_store and _k21 is None and R_rows > _MXFP4_GEMV_ROWS
            and gu_p is not None and gu_p.numel() > 0):
        # Batched rows on the MXFP4 store: the decode GEMV re-streams the
        # weights per row and loses to NF4 above a handful of rows
        # (lane bo3n: x0.81 at 64 rows), and the v1 grouped GEMM loses
        # more (x0.41). With the NF4 stacks kept (E4B_INT4_KEEP_NF4=1)
        # those rows take the NF4 path; the store is a decode lever until
        # an M-tile MXFP4 GEMM exists.
        int4_stores = None
        _mxfp4_store = False
    _int4_gemv_decode = (not _mxfp4_store and device_grouping
                         and int4_stores is not None and R_rows <= 256)
    # K19 (grouped-nf4-gemm#419): the same decode rows through the grouped small-M tensor-core GEMM instead of the
    # split-K GEMV -- K16's arithmetic (bf16 activations, in-register int4 dequant, bf16 MMA) over the device tile
    # table, gate_up's gather folded into the kernel. Lane P88 LICENSED it for these rows on an RTX 5090
    # (e4b.serve.p88.qwen3.int4.k19-b16.5090.2026-10-01: B=16 step 0.905x, K8 +0.0062 nats), so it is the DEFAULT
    # here when the kernel package carries it (E4B_INT4_GROUPED_SMALLM unset or "auto"). "0" keeps the split-K GEMV;
    # "1" requires K19 (absent is a refusal) and also routes T == 1 to it (see _collapsed_grouping). B=1 read SLOWER
    # (x1.103), which is why auto leaves T == 1 on the singleton GEMV.
    _k19 = None
    _k19_mode = _k19_mode_env()
    _lean_mode = _lean_glue_mode_env()    # read (and refused if malformed) on every call, like the K19 mode
    _lean = False
    if _int4_gemv_decode and _k19_mode != "0":
        try:
            from int4_smallm import gemm_int4_b32_grouped_smallm as _k19
        except ImportError as e:
            if _k19_mode == "1":
                raise RuntimeError(
                    "E4B_INT4_GROUPED_SMALLM=1 needs grouped-nf4-gemm with K19 "
                    "(int4_smallm.gemm_int4_b32_grouped_smallm, grouped-nf4-gemm#419)") from e
            _k19 = None                   # auto: the kernel package predates K19 -> the split-K GEMV, as before
        if _k19 is not None:
            _int4_gemv_decode = False     # take the device-grouping branch: it builds the 16-row tile table
    # K25 (E4B_NF4_GROUPED_SMALLM, default auto since lane P96): the NF4 store's device-grouped decode rows through
    # the grouped small-M tensor-core GEMM with the NF4 dequant, instead of the served M-tile GEMM. Not under a
    # calibration sink: the sink reads x_sorted beside sorted_ids, and K25 gathers inside the kernel.
    _k25 = None
    _k25_mode = _k25_mode_env()
    if (int4_stores is None and device_grouping and R_rows <= 256 and _k25_mode != "0"
            and _CALIB_SINK is None):
        try:
            import nf4_smallm as _k25_mod
            _k25 = _k25_mod.gemm_nf4_grouped_smallm
            _k25_kw = _k25_plan(_k25_mod)
        except (ImportError, AttributeError) as e:
            if _k25_mode == "1":
                raise RuntimeError(
                    "E4B_NF4_GROUPED_SMALLM=1 needs grouped-nf4-gemm with K25 "
                    "(nf4_smallm.gemm_nf4_grouped_smallm, grouped-nf4-gemm#429)") from e
            _k25 = None                   # auto: the kernel package predates K25 -> today's NF4 route
    _small = _k19 if _k19 is not None else _k25
    _fused_tiles = None
    if _lean_mode != "0" and _small is not None and gptoss is None:
        try:
            from int4_b32 import build_group_tiles_fused as _fused_tiles
        except ImportError:
            _fused_tiles = None
        if _fused_tiles is not None and _lean_glue_supported(_fused_tiles, _small):
            _lean = True
        elif _lean_mode == "1":
            raise RuntimeError(
                "E4B_INT4_LEAN_GLUE=1 needs grouped-nf4-gemm with K23 "
                "(build_group_tiles_fused lean=/sorted_ids=, "
                "gemm_int4_b32_grouped_smallm scatter=/gather_div=)")
        else:
            _fused_tiles = None           # auto: the kernel package predates K23 -> the separate launches, as before
    if (_k19 is None and device_grouping and int4_stores is not None and not _mxfp4_store and R_rows > 256
            and _int4_prefill_mode_env() == "k19"):
        # E4B_INT4_PREFILL=k19 (e4b#916): K19 serves the device-grouped PREFILL rows too, over the chained 16-row tile
        # table built below (K23's lean glue stays off: its one-launch builder is a decode-shape builder). Without K19
        # in the kernel package the route is refused, not silently replaced by the int8 M-tile.
        try:
            from int4_smallm import gemm_int4_b32_grouped_smallm as _k19
        except ImportError as e:
            raise RuntimeError(
                "E4B_INT4_PREFILL=k19 needs grouped-nf4-gemm with K19 "
                "(int4_smallm.gemm_int4_b32_grouped_smallm, grouped-nf4-gemm#419)") from e
    _tok = None
    if x_rows is None:
        x_t, row_token, top_k = x_tokens
        if _lean and _CALIB_SINK is None:
            _tok = (x_t, top_k)           # K23: K19's (or K25's) gate_up reads the token rows itself (gather_div)
        else:
            x_rows = x_t.index_select(0, row_token)
    if _mxfp4_store and _k21 is None:
        singleton_groups = True
        device_grouping = False
    if _int4_gemv_decode:
        # batched DECODE on the int4 store: the split-K GEMV serves rows
        # in INPUT order (P7: 1.92x/1.28x over the M-tile at ~1-2 rows
        # per expert -- the tile padding wastes ~90% of the MMA lanes,
        # which split-K cannot recover). No tile table, no gather, no
        # unsort: this branch deliberately builds NONE of the grouping
        # machinery.
        order = None
        sorted_ids = local_ids
        x_sorted = x_rows
        sizes = None
        eids = local_ids
    elif device_grouping:
        # PREREG-s3-grouped-verify: expert-major grouping built entirely
        # on device (gnf4 build_group_tiles_device -- exact-parity vs
        # the host builder, CI-gated there), executed through the
        # captured M-tile wrapper. Same math as the grouped path below,
        # same sort/unsort contract (order feeds the shared tail);
        # capture-legal because nothing round-trips to the host. Rows
        # sharing an expert share its packed-weight read -- the reuse
        # the singleton path deliberately forfeits.
        from nf4_grouped import (build_group_tiles_device,
                                 gemm_4bit_grouped_captured)
        # Expert count from the int4 stores when the NF4 stacks are
        # FREED (the int4 lane's default drops them to 0-sized
        # sentinels): gu_p.shape[0] is then 0, and a 0-expert tile
        # table detonates as a storage-size-0 expand in the builder --
        # hit live on the first composed bv3+int4 arm.
        if int4_stores is None and gu_p is not None and gu_p.numel() == 0:
            raise RuntimeError(
                "expert stacks are freed (int4 serve lane active) but the "
                "device-grouping path carries no int4_stores -- a 0-expert "
                "tile table would detonate deep in the builder")
        _n_exp = ((int4_stores["gu"]["blocks"] if _k21 is not None else int4_stores["gu"]["packed"]).shape[0]
                  if int4_stores is not None else gu_p.shape[0])
        # ONE-launch tile table on decode shapes when the kernel side
        # ships it (census: the chained builder is ~10 launches/layer,
        # 48 radix sorts per B=16 step); chained builder otherwise and
        # for prefill chunks, where launches amortize.
        if not _lean and local_ids.numel() <= 256:      # (under _lean the builder was resolved above)
            try:
                from int4_b32 import build_group_tiles_fused as _fused_tiles
            except ImportError:
                _fused_tiles = None
        if _lean:
            # K23: one launch for the table AND the sorted ids
            t_row0, t_rows, t_grp, order, _counts, sorted_ids = _fused_tiles(
                local_ids, _n_exp, 16, lean=True, sorted_ids=True)
        elif _fused_tiles is not None:
            t_row0, t_rows, t_grp, order, _counts = _fused_tiles(
                local_ids, _n_exp, 16)
        else:
            t_row0, t_rows, t_grp, order, _counts = \
                build_group_tiles_device(local_ids, _n_exp, 16)
        if not _lean:
            sorted_ids = local_ids.index_select(0, order)
        if int4_stores is not None or _k25 is not None:
            # the int4 _mm quantises with the gather FOLDED IN (or takes
            # the already-sorted epilogue output), and K25 gathers in its
            # activation load; the [R, K] gather here would be pure waste
            # on those branches
            x_sorted = x_rows
        else:
            x_sorted = x_rows.index_select(0, order).contiguous()
        sizes = None                      # the captured path has no host sizes
        eids = None
    elif singleton_groups:
        # AMENDMENT-b1d-capture: the grouping step's unique_consecutive
        # SYNCS to produce host-side group sizes -- structurally illegal
        # inside CUDA-graph capture. One row per group instead: sizes is
        # a host CONSTANT, the ids ride as a device tensor (the wrapper
        # accepts one), and no sort/unsort is needed. At T == 1 a token's
        # top-k ids are distinct, so every group has one row either way:
        # on the NF4 stack both branches reach the same decode GEMV and
        # agree bit for bit (held by the on-box hash gate). On the int4
        # store they do NOT: this branch is the int8 x int4 GEMV and the
        # grouped branch is dequant + bf16 matmul, different functions
        # (lane P63). tests/test_singleton_groups.py pins the dispatch
        # algebra through a mocked GEMM, not the kernels' arithmetic.
        #
        # PREREG-s2lite: at T > 1 every row is still its own group, so it
        # takes the T == 1 kernel and its output does not depend on the
        # row count. Measured, not argued (lane P63, #708: Qwen3-30B-A3B
        # on an RTX 5090, 48/48 experts modules bit-equal to their T == 1
        # calls at 16, 17 and 160 tokens, int4 with _int4_part_or_none and
        # NF4 on the dot-pad GEMV; tests/test_p63_row_exact_gpu.py). It is
        # NOT bit-equal to the grouped branch at T > 1: there the group
        # size picks the kernel (NF4 M-tile, TF32 weights; int4 dequant +
        # bf16), a different function, not a reordering -- "the M
        # dimension does not participate in any reduction" is true of the
        # math only. And the row-invariance is the kernel's plan at a
        # shape: GNF4_GEMV_DOTPAD=0 plans its split-K from the rows, which
        # P63 read reorder-class. What
        # it COSTS at T > 1 is different: rows sharing an expert no
        # longer share that expert's weight read, so the MoE side reads
        # ~R distinct expert weight-sets instead of ~E(distinct). A
        # captured verify step has no legal alternative absent a
        # device-side grouping kernel, so this is what a graphed
        # verify actually costs today -- and it is an UPPER bound on
        # what a device-grouped one would.
        order = None
        sorted_ids = local_ids
        x_sorted = x_rows.contiguous()
        sizes = [1] * x_rows.shape[0]
        eids = local_ids
    else:
        order = torch.argsort(local_ids)
        sorted_ids = local_ids.index_select(0, order)
        x_sorted = x_rows.index_select(0, order).contiguous()
        uniq, counts = torch.unique_consecutive(sorted_ids,
                                                return_counts=True)
        sizes = counts.tolist()
        # sizes must come to the host (build_group_tiles walks it and the
        # launch grid needs the tile count), but uniq does NOT: tolist()
        # here was a D2H sync whose list gemm_4bit_grouped immediately
        # shipped back with a syncing pageable H2D -- twice, once per
        # GEMM. The wrapper passes a CUDA tensor straight through, and
        # the singleton branch already hands it one (eids = local_ids).
        # Measured at B=16 decode: this plus gnf4's tile-build memo took
        # to_device_i32 traffic /4 and the step 1.206x, tokens identical.
        eids = uniq
    if device_grouping and int4_stores is None and _k25 is not None:
        # K25 (opt-in): the grouped small-M tensor-core GEMM on the NF4 stacks against the SAME 16-row device tiles. The
        # first call gets the UNSORTED x_rows and gathers through `order` in the kernel; the epilogue output is already
        # in sorted order. Outputs in sorted order, so gpt-oss's per-expert biases index by `sorted_ids` and the unsort
        # at the end applies unchanged. Under K23 (_lean) gate_up reads the token rows (gather_div) and down stores into
        # the caller's row order (scatter=order), and the unsort is skipped -- K19's glue, on K25.
        def _mm(xr, pk, am):
            if pk is not None and pk.numel() == 0:
                raise RuntimeError(
                    "expert stacks are freed (int4 serve lane active) "
                    "but the NF4 K25 path was selected -- "
                    "int4_stores did not reach _fused_over_stack")
            if pk is gu_p and _tok is not None:
                return _k25(_tok[0].to(torch.bfloat16), pk, am, t_row0, t_rows, t_grp, order,
                            gather_div=_tok[1], **_k25_kw)
            kw = {"scatter": order} if (_lean and pk is dn_p) else {}
            return _k25(xr.to(torch.bfloat16), pk, am, t_row0, t_rows, t_grp,
                        order if xr is x_rows else None, **kw, **_k25_kw)
    elif device_grouping and int4_stores is None:
        def _mm(xr, pk, am):
            if pk is not None and pk.numel() == 0:
                # freed int4-lane stacks reaching the NF4 captured path
                # produce a SILENT [R, 0] output (E reads as 0), which
                # detonates far away as a size-0 view -- raise here like
                # the host-grouped branch always has
                raise RuntimeError(
                    "expert stacks are freed (int4 serve lane active) "
                    "but the NF4 captured path was selected -- "
                    "int4_stores did not reach _fused_over_stack")
            return gemm_4bit_grouped_captured(xr, pk, am, t_row0, t_rows,
                                              t_grp, 16)
    elif _mxfp4_store and _k21 is not None:
        # K21 (opt-in): the store's own e2m1/e8m0 bytes, dequantised exactly to bf16 in registers, bf16 MMA over the
        # SAME 16-row device tiles; the first call gathers the unsorted rows through `order`, the epilogue output is
        # already sorted. Outputs in sorted order, so gpt-oss's per-expert biases index by `sorted_ids` and the unsort
        # at the end applies unchanged.
        def _mm(xr, pk, am):
            st = int4_stores["gu" if pk is gu_p else "dn"]
            return _k21(xr.to(torch.bfloat16), st["blocks"], st["scales"], t_row0, t_rows, t_grp,
                        order if xr is x_rows else None, **_K21_PLAN)
    elif _mxfp4_store:
        import mxfp4_grouped
        _sizes_mx = [1] * x_rows.shape[0]           # host constant: capture-safe
        _eids_mx = local_ids.to(torch.int32)
        # Decode rows go through the decode-grade GEMV when the kernel
        # side has it (grouped-nf4-gemm >= 0.28, `gemv_mxfp4_b32`: the
        # int4-b32 split-K structure on the MXFP4 bytes, int8 activation
        # rows); the v1 grouped GEMM's M==1 reduction (one program per
        # 64 rows, no split-K) measured SLOWER than NF4 at B=1 on
        # gpt-oss. E4B_MXFP4_GEMV=0 forces the v1 route (the A/B arm).
        _gemv_mx = getattr(mxfp4_grouped, "gemv_mxfp4_b32", None)
        _use_gemv_mx = (_gemv_mx is not None
                        and x_rows.shape[0] <= _MXFP4_GEMV_ROWS
                        and os.environ.get("E4B_MXFP4_GEMV", "1") == "1")
        if _use_gemv_mx:
            from int4_b32 import quant_x_rows

            def _mm(xr, pk, am):
                st = int4_stores["gu" if pk is gu_p else "dn"]
                xq, xs = quant_x_rows(xr.to(torch.bfloat16).contiguous())
                return _gemv_mx(xq, xs, st["blocks"], st["scales"],
                                _eids_mx, st["N"], st["K"])
        else:
            def _mm(xr, pk, am):
                st = int4_stores["gu" if pk is gu_p else "dn"]
                return mxfp4_grouped.gemm_mxfp4_grouped(
                    xr.to(torch.bfloat16).contiguous(),
                    st["blocks"], st["scales"], _sizes_mx, _eids_mx)
    elif int4_stores is not None:
        # Opt-in uniform-int4 expert store (engines/int4_experts). The
        # NF4 stacks may already be FREED, so BOTH branches must serve
        # from the int4 bytes. Identity dispatch: _mm is called with the
        # (possibly empty) NF4 stacks it replaces, so `pk is gu_p` names
        # the slot.
        if _int4_gemv_decode:
            # part=None allocates through the graph private pool (the
            # certified capture pattern); R here is B*top_k, so the
            # B=1 stores' preallocated part buffers do not fit and are
            # deliberately not used.
            from int4_b32 import gemv_int4_b32, quant_x_rows
            e32d = local_ids.to(torch.int32)

            def _mm(xr, pk, am):
                st = int4_stores["gu" if pk is gu_p else "dn"]
                xq, xs = quant_x_rows(xr)
                return gemv_int4_b32(xq, xs, st["packed"], st["scales"],
                                     e32d, st["N"], st["K"])
        elif device_grouping and _k19 is not None:
            # K19 (opt-in): the grouped small-M tensor-core GEMM against the SAME 16-row device tiles. The first
            # call gets the UNSORTED x_rows and gathers through `order` in the kernel; the epilogue output is
            # already in sorted order. bf16 activations, no quantise; outputs in sorted order, as K14's.
            # Under K23 (_lean) the down call stores into the caller's row order (scatter=order), and the unsort at
            # the end is skipped.
            def _mm(xr, pk, am):
                slot = "gu" if pk is gu_p else "dn"
                st = int4_stores[slot]
                if slot == "gu" and _tok is not None:
                    # K23: sorted row i reads token row order[i] // top_k, no [T * top_k, H] expansion
                    return _k19(_tok[0].to(torch.bfloat16), st["packed"], st["scales"], t_row0, t_rows, t_grp,
                                order, gather_div=_tok[1])
                kw = {"scatter": order} if (_lean and slot == "dn") else {}
                return _k19(xr.to(torch.bfloat16), st["packed"], st["scales"], t_row0, t_rows, t_grp,
                            order if xr is x_rows else None, **kw)
        elif device_grouping:
            # batched decode (bv3): the grouped int4-b32 GEMM against
            # the SAME prebuilt device tiles the NF4 captured path uses
            # -- measured 1.95x (gate_up) / 4.50x (down) over the NF4
            # grouped kernel on the B=16 census cells, graph metric.
            # Rows arrive expert-major (the tile builder's order); the
            # activation quantise runs per call because gu and dn see
            # different inputs (x vs the epilogue output). Both the
            # quantise and the GEMM allocate only through the graph
            # private pool -- the b1d-certified capture pattern.
            from int4_b32 import gemm_int4_b32_grouped_captured, quant_x_rows
            try:
                from int4_b32 import quant_x_rows_gathered
            except ImportError:
                quant_x_rows_gathered = None

            def _mm(xr, pk, am):
                st = int4_stores["gu" if pk is gu_p else "dn"]
                if xr is x_rows:
                    # first call: UNSORTED activations -- fold the
                    # expert-major gather into the quantise when the
                    # kernel ships it (kills the per-layer [R, K]
                    # index_select the census priced at 0.46 ms/step)
                    if quant_x_rows_gathered is not None:
                        xq, xs = quant_x_rows_gathered(xr, order)
                    else:
                        xq, xs = quant_x_rows(
                            xr.index_select(0, order).contiguous())
                else:
                    # epilogue output: already in sorted order
                    xq, xs = quant_x_rows(xr)
                return gemm_int4_b32_grouped_captured(
                    xq, xs, st["packed"], st["scales"],
                    t_row0, t_rows, t_grp)
        elif singleton_groups and not DECODE_A16[0]:
            # decode: the int4-b32 grouped GEMV -- measured 2.7-3.3x
            # over the NF4 path at the census cells, grid +0.007 ppl
            from int4_b32 import gemv_int4_b32, quant_x_rows
            e32 = eids.to(torch.int32)

            def _mm(xr, pk, am):
                st = int4_stores["gu" if pk is gu_p else "dn"]
                xq, xs = quant_x_rows(xr)
                return gemv_int4_b32(xq, xs, st["packed"], st["scales"],
                                     e32, st["N"], st["K"],
                                     part=_int4_part_or_none(st, e32.numel(),
                                                             xq.device))
        else:
            # Host-grouped T > 1 calls (prefill chunks, verify) with
            # DEVICE_GROUPING off: dequant each routed expert and matmul in
            # bf16. It runs on every call -- once per prefill CHUNK per
            # layer, never amortised over a request -- a host loop over
            # the ~E_active groups of each projection (e4b#916: ~2.07 s per
            # 512-token chunk on Qwen3-30B-A3B, ~170k launches; lane P100).
            # E4B_INT4_PREFILL picks the route (_int4_prefill_mode_env):
            # `loop` below, `batched` = the same weights decoded in slices,
            # bit-identical; `k19` and `mtile` never reach this branch.
            # Under DECODE_A16 (P64) the singleton decode rows land here
            # too, one row per group: bf16 activations, no quant_x_rows.
            if (singleton_groups and x_rows.is_cuda
                    and torch.cuda.is_current_stream_capturing()):
                raise RuntimeError(
                    "E4B_INT4_DECODE_A16=1 routes decode through the "
                    "prefill branch, which reads the expert ids on the "
                    "host -- illegal under CUDA-graph capture. The flag "
                    "is an eager-only quality instrument; unset it for "
                    "a graphed decode.")
            from int4_pack_ref import dequant_int4_ref
            eids_l = (eids.tolist() if torch.is_tensor(eids) else list(eids))
            row0 = [0]
            for m_ in sizes[:-1]:
                row0.append(row0[-1] + m_)
            _batched = _int4_prefill_mode_env() == "batched"
            if _batched:
                _eids_t = (eids if torch.is_tensor(eids) else torch.as_tensor(eids_l)).to(
                    device=int4_stores["gu"]["packed"].device, dtype=torch.long)

            def _mm_batched(xr, pk, am):
                st = int4_stores["gu" if pk is gu_p else "dn"]
                N_, K_ = st["N"], st["K"]
                out = torch.empty(xr.shape[0], N_, dtype=torch.bfloat16,
                                  device=xr.device)
                for j0 in range(0, len(eids_l), _INT4_PREFILL_SLICE):
                    w = _dequant_int4_bf16(st["packed"], st["scales"],
                                           _eids_t[j0:j0 + _INT4_PREFILL_SLICE], N_, K_)
                    for j, (m_, r0) in enumerate(zip(sizes[j0:j0 + _INT4_PREFILL_SLICE],
                                                     row0[j0:j0 + _INT4_PREFILL_SLICE])):
                        out[r0:r0 + m_] = (xr[r0:r0 + m_].to(torch.bfloat16)
                                           @ w[j].t())
                return out

            def _mm_loop(xr, pk, am):
                st = int4_stores["gu" if pk is gu_p else "dn"]
                N_, K_ = st["N"], st["K"]
                out = torch.empty(xr.shape[0], N_, dtype=torch.bfloat16,
                                  device=xr.device)
                for gi, (m_, e_, r0) in enumerate(zip(sizes, eids_l, row0)):
                    w = dequant_int4_ref(st["packed"][e_],
                                         st["scales"][e_], N_, K_)
                    out[r0:r0 + m_] = (xr[r0:r0 + m_].to(torch.bfloat16)
                                       @ w.to(torch.bfloat16).t())
                return out
            _mm = _mm_batched if _batched else _mm_loop
    else:
        def _mm(xr, pk, am):
            if pk is not None and pk.numel() == 0:
                raise RuntimeError(
                    "expert stacks are freed (int4 serve lane active) but "
                    "this forward carries no int4_stores -- a tiered/"
                    "baseline path reached _fused_over_stack after "
                    "enable_serve_experts_int4. That enable is collapsed-"
                    "path-only; re-enable with all-VRAM placement.")
            return gemm_4bit_grouped(xr, pk, am, sizes, eids)
    gu = _mm(x_sorted, gu_p, gu_a)
    if gptoss is not None:
        gu_bias, dn_bias, alpha, limit = gptoss
        gu = gu + gu_bias.index_select(0, sorted_ids).to(gu.dtype)  # per-expert bias by local id
        gate, up = gu.chunk(2, dim=-1)                              # de-interleaved at load
        gate = gate.clamp(max=limit)
        up = up.clamp(min=-limit, max=limit)
        h = (up + 1) * (gate * torch.sigmoid(gate * alpha))
        if _CALIB_SINK is not None:
            _CALIB_SINK(gu_p, sorted_ids, x_sorted, h)
        dn = _mm(h.contiguous(), dn_p, dn_a)
        dn = dn + dn_bias.index_select(0, sorted_ids).to(dn.dtype)
    elif has_gate:
        gate, up = gu.chunk(2, dim=-1)
        if clamp_limit is not None:
            # fp32, then back to compute dtype for the down GEMM — mirroring
            # `_DeepseekV4ForwardMixin._apply_gate` + its `gated.to(cd)`. The other two
            # branches deliberately stay in compute dtype because THEIR references do
            # (stock `ExpertsNbit.forward` and `_GptOssForwardMixin.forward` both run the
            # epilogue at `cd`); V4's reference is the one that promotes, so reproducing
            # its epilogue means reproducing the precision too. Clamping is exactly where
            # that bites: at the limit the clamped operand is a constant and all the
            # remaining signal is in the other one.
            gate, up = gate.float(), up.float()
            gate = gate.clamp(max=clamp_limit)                      # one-sided, by design
            up = up.clamp(min=-clamp_limit, max=clamp_limit)
            h = (act_fn(gate) * up).to(gu.dtype)
        else:
            h = _swiglu_or(act_fn, gu, gate, up)
        if _CALIB_SINK is not None:
            _CALIB_SINK(gu_p, sorted_ids, x_sorted, h)
        dn = _mm(h.contiguous(), dn_p, dn_a)
    else:
        h = act_fn(gu)
        if _CALIB_SINK is not None:
            _CALIB_SINK(gu_p, sorted_ids, x_sorted, h)
        dn = _mm(h.contiguous(), dn_p, dn_a)
    if order is None:                  # singleton path: input order kept
        return dn
    if _lean:                          # K23: K19's (or K25's) down already stored each row at order[i]
        return dn
    out = torch.empty_like(dn)
    out.index_copy_(0, order, dn)  # unsort back to caller's row order
    return out


def _partition_by_mask(hot_row):
    """``(n_cold, cr, hr)`` carrying EXACTLY the index content and order of
    ``(~hot_row).nonzero()`` / ``hot_row.nonzero()``, for ONE small host
    read instead of two variable-size ``nonzero`` device→host syncs.

    A stable ascending sort of the mask puts the False (cold) positions
    first and the True (hot) positions after, each group in original index
    order — which is precisely ``nonzero``'s order — so the split is a
    slice of one permutation. The all-hot / all-cold edges skip even the
    sort. Equivalence to the ``nonzero`` pair is pinned across all three
    regimes in ``tests/test_dispatch_diet.py``."""
    n = hot_row.numel()
    n_cold = n - int(hot_row.sum())          # the ONE device→host read
    if n_cold == 0 or n_cold == n:
        full = torch.arange(n, device=hot_row.device)
        return ((0, full[:0], full) if n_cold == 0
                else (n, full, full[:0]))
    perm = torch.argsort(hot_row.to(torch.uint8), stable=True)
    return n_cold, perm[:n_cold], perm[n_cold:]


class _HotResidency:
    """Per-module state: a resident GPU hot-stack + a pinned-CPU cold-stack, and
    the global<->local id maps needed to dispatch each routed expert."""

    # Subclasses that never H2D-stream the cold weights (the cold engine computes
    # them on the host) set this False and keep the cold stacks pageable.
    _PIN_COLD = True

    # T5 dispatch diet (off by default — `enable_hybrid_tier` plumbs it):
    # one host sync per layer instead of 2–4 `nonzero` syncs, cached
    # row_token/row_slot index algebra, de-duplicated gathers, and a
    # partition-free all-hot fast path. The baseline `forward` body is
    # untouched when this is False.
    dispatch_diet = False
    _rt_cache = None                # {(T, k, device): (row_token, row_slot)}; see _row_index

    def _row_index(self, T, k, dev):
        """``(row_token, row_slot)`` for ``T`` tokens of ``k`` experts: built once per (T, k, device) and KEPT for the
        module's life. A captured CUDA graph reads these by address and holds no reference to them, and
        ``PagedModelRunner.enable_decode_graphs`` warms each bucket eagerly before capturing it. A single-entry cache
        freed the earlier bucket's index when the next bucket warmed up, and once the allocator handed that block out
        again the earlier bucket's replay gathered token rows through garbage (#913, lane P98: an out-of-range
        ``index_select`` at bucket 1). Bounded by the distinct row counts a model sees: the decode buckets plus the
        prefill chunk lengths, each a few KB."""
        c = self._rt_cache
        if c is None:
            c = self._rt_cache = {}
        key = (int(T), int(k), str(dev))
        hit = c.get(key)
        if hit is None:
            rt = torch.arange(T * k, device=dev) // k
            hit = c[key] = (rt, torch.arange(T * k, device=dev) - rt * k)
        return hit

    # B1c collapse (PREREG-b1c, the BRANCH-2 optimization): when the
    # PLACEMENT is all-VRAM, the token-critical path runs no dispatch
    # algebra at all — no mask, no partition, no cold checks, no
    # scatter. Placement-static predicate, cached; `swap_expert`
    # invalidates. Off by default until its cert.
    collapse_resident = False
    _all_hot_cache = None

    def _all_hot(self):
        """Every expert hot AND the hot stack in identity order — the
        two facts `_forward_collapsed` relies on to pass `flat` directly
        as local ids. Checked once per placement, never per step; any
        future hot-stack reordering makes this False and the baseline
        path runs instead of mis-indexing."""
        c = self._all_hot_cache
        if c is None:
            E = self.is_hot.numel()
            c = bool(self.is_hot.all()) and bool(
                (self.g2h == torch.arange(E, device=self.g2h.device))
                .all())
            self._all_hot_cache = c
        return c

    def __init__(self, mod, hot_ids, device):
        self.mod = mod
        self.device = torch.device(device)
        E = mod.num_experts
        hot_ids = torch.as_tensor(hot_ids, dtype=torch.long).unique()
        if hot_ids.numel() and (hot_ids.min() < 0 or hot_ids.max() >= E):
            raise ValueError(f"hot ids must lie in [0, {E}); got range "
                             f"[{int(hot_ids.min())}, {int(hot_ids.max())}]")
        cold_ids = torch.tensor([e for e in range(E) if e not in set(hot_ids.tolist())],
                                dtype=torch.long)
        self.hot_ids, self.cold_ids = hot_ids, cold_ids
        n1, k1 = mod._gate_up_shape
        n2, k2 = mod._down_shape
        self.shapes = (n1, k1, n2, k2)
        self.has_gate = mod.has_gate
        self.act_fn = mod.act_fn
        self.compute_dtype = mod.compute_dtype

        # gpt-oss epilogue: per-expert biases (de-interleaved to contiguous
        # [gate;up] at load) + clamped GLU. Biases are tiny — keep the hot AND
        # cold sub-stacks resident on the compute device, aligned to the same
        # local id order as the packed hot/cold stacks below.
        self.gptoss = getattr(mod, "alpha", None) is not None and hasattr(mod, "gate_up_bias")
        if self.gptoss:
            self.alpha, self.limit = float(mod.alpha), float(mod.limit)
        # DeepSeek-V4 shares gpt-oss's CLAMPS but not its GLU, and carries no biases:
        # `limit` without `alpha`/`gate_up_bias` is exactly that family. Left None for
        # the stock SwiGLU modules, which have no `limit` at all.
        _lim = getattr(mod, "limit", None)
        self.clamp_limit = (float(_lim) if _lim and not self.gptoss and _lim > 0
                            else None)

        # per-expert flattened packed storage -> [E, n, k/2] / [E, n, k/64]
        gu_p = mod.gate_up_proj.view(E, n1, k1 // 2)
        gu_a = mod.gate_up_absmax.view(E, n1, k1 // 64).float()
        dn_p = mod.down_proj.view(E, n2, k2 // 2)
        dn_a = mod.down_absmax.view(E, n2, k2 // 64).float()

        # index on the weights' own device (the model is typically CUDA-resident)
        hi = hot_ids.to(gu_p.device)
        ci = cold_ids.to(gu_p.device)
        # HOT: resident on the GPU (never transferred again). Hooked for the same
        # reason as _build_cold: a subclass may source these from somewhere other
        # than the module's own [E, ...] buffers, which lets the module itself be
        # built on `meta` and allocate no expert storage at all.
        self._build_hot(gu_p, gu_a, dn_p, dn_a, hi)
        if self.gptoss:
            gub, dnb = mod.gate_up_bias, mod.down_bias           # [E, 2I] / [E, H], contiguous
            # Index on the BIASES' own device, not the packed stacks'. The two
            # are deliberately different tiers on the arena path: the packed
            # weights stay on `meta` (served from the arena) while these small
            # stacks are resident, so reusing `hi`/`ci` -- built on the packed
            # weights' device -- would index a real tensor with meta indices and
            # kill the attach for every gpt-oss layer. A no-op on the direct
            # path, where both live on the same device.
            bhi = hot_ids.to(gub.device)
            bci = cold_ids.to(gub.device)
            self.h_gu_b = gub.index_select(0, bhi).contiguous().to(self.device)
            self.h_dn_b = dnb.index_select(0, bhi).contiguous().to(self.device)
            self.c_gu_b = gub.index_select(0, bci).contiguous().to(self.device)
            self.c_dn_b = dnb.index_select(0, bci).contiguous().to(self.device)
        # COLD: pinned host RAM, streamed per token (only the routed subset).
        # Factored into a hook so a subclass can back the cold tail with
        # something other than a fully-materialized host stack — see
        # `nvme_experts._NvmeResidency`, which serves it from an NVMe arena
        # because a checkpoint like Kimi K3 has 1.446 TB of experts and no host
        # holds them. Everything downstream only ever calls
        # `.index_select(0, routed)` then `.to(dev)` on these four attributes,
        # so that is the whole contract a replacement must honour.
        self._build_cold(gu_p, gu_a, dn_p, dn_a, ci)

        # global expert id -> (is_hot, local index within its stack)
        self._finish_ids(E, hot_ids, cold_ids)

    def _build_hot(self, gu_p, gu_a, dn_p, dn_a, hi):
        """Materialize the hot partition on the compute device.

        Override to source hot experts from elsewhere (e.g. an NVMe arena). The
        four ``h_*`` attributes must end up as device tensors shaped
        ``[len(hot), n, k/2]`` / ``[len(hot), n, k/64]``.
        """
        self.h_gu_p = gu_p.index_select(0, hi).contiguous().to(self.device)
        self.h_gu_a = gu_a.index_select(0, hi).contiguous().to(self.device)
        self.h_dn_p = dn_p.index_select(0, hi).contiguous().to(self.device)
        self.h_dn_a = dn_a.index_select(0, hi).contiguous().to(self.device)

    def _build_cold(self, gu_p, gu_a, dn_p, dn_a, ci):
        """Materialize the cold tail as four (pinned) host stacks.

        Override to serve the cold tail from elsewhere. The contract is narrow:
        each attribute must support ``.index_select(0, routed)`` returning a
        tensor that then accepts ``.to(dev, non_blocking=True)``.
        """
        self.c_gu_p = gu_p.index_select(0, ci).contiguous().cpu()
        self.c_gu_a = gu_a.index_select(0, ci).contiguous().cpu()
        self.c_dn_p = dn_p.index_select(0, ci).contiguous().cpu()
        self.c_dn_a = dn_a.index_select(0, ci).contiguous().cpu()
        if self._PIN_COLD:
            try:
                self.c_gu_p = self.c_gu_p.pin_memory()
                self.c_gu_a = self.c_gu_a.pin_memory()
                self.c_dn_p = self.c_dn_p.pin_memory()
                self.c_dn_a = self.c_dn_a.pin_memory()
            except (RuntimeError, AssertionError):
                pass  # pageable fallback is correct, just synchronous H2D

    def _finish_ids(self, E, hot_ids, cold_ids):
        g2h = torch.full((E,), -1, dtype=torch.long)
        g2h[hot_ids] = torch.arange(hot_ids.numel())
        g2c = torch.full((E,), -1, dtype=torch.long)
        g2c[cold_ids] = torch.arange(cold_ids.numel())
        self.is_hot = torch.zeros(E, dtype=torch.bool, device=self.device)
        self.is_hot[hot_ids.to(self.device)] = True
        self.g2h = g2h.to(self.device)
        self.g2c_cpu = g2c  # cold local ids resolved on CPU (stack is on CPU)

    def forward(self, hidden_states, top_k_index, top_k_weights):
        input_dtype = hidden_states.dtype
        input_dev = hidden_states.device
        # read compute_dtype LIVE off the module (a later change must be honored)
        cd = self.mod.compute_dtype if self.mod.compute_dtype is not None else input_dtype
        dev = self.device
        x = hidden_states.to(device=dev, dtype=cd)
        top_k_weights = top_k_weights.to(dev)
        T, H = x.shape
        k = top_k_index.shape[1]

        flat = top_k_index.reshape(-1).to(dev)                 # [T*k] global expert per assignment
        if (self.collapse_resident and getattr(self, "amort", None) is None
                and self._all_hot()):
            # amort-armed runs keep the instrumented baseline path (the
            # per-bus event bracket lives there); the collapse serves
            # the production shape
            return self._forward_collapsed(x, flat, top_k_weights, T, k,
                                           H, dev, input_dev, input_dtype)
        if self.dispatch_diet:
            return self._forward_diet(x, flat, top_k_weights, T, k, H, dev,
                                      input_dev, input_dtype)
        row_token = torch.arange(T * k, device=dev) // k
        row_slot = torch.arange(T * k, device=dev) - row_token * k
        hot_row = self.is_hot[flat]                            # [T*k] bool
        # [T, k, H] slot landing, NOT [T, H] + index_add_: every assignment
        # owns a unique (token, slot) cell, so branches WRITE (index_put_,
        # no accumulate) instead of atomically adding — CUDA index_add_ with
        # duplicate indices orders its adds by warp scheduling, and at
        # batch=1 all k contributions collide on one row, flipping output
        # bits run to run (the determinism-invariant defect filed with the
        # G3 formal results). The fixed-order sum(dim=1) at the end is the
        # ONE reduction, deterministic for a fixed shape. Costs k× the
        # transient (fp32); trivial at decode, bounded at prefill.
        out = torch.zeros(T, k, H, dtype=torch.float32, device=dev)

        # --- HOT: resident GPU stack, zero transfer ---
        hr = hot_row.nonzero(as_tuple=False).view(-1)
        # per-bus probe (hybrid Phase 8): armed only when a subclass is
        # counting, and it brackets ONLY the hot branch — the GPU bus's
        # completion time is measured, never inferred by subtracting arms
        _amort = getattr(self, "amort", None)
        _ev = None
        if _amort is not None and hr.numel() and dev.type == "cuda":
            _ev = (torch.cuda.Event(enable_timing=True),
                   torch.cuda.Event(enable_timing=True))
            _ev[0].record()
        if hr.numel():
            local = self.g2h[flat.index_select(0, hr)]
            xr = x.index_select(0, row_token.index_select(0, hr))
            gptoss = ((self.h_gu_b, self.h_dn_b, self.alpha, self.limit)
                      if self.gptoss else None)
            dn = _fused_over_stack(xr, local, self.h_gu_p, self.h_gu_a, self.h_dn_p,
                                   self.h_dn_a, self.shapes, self.has_gate, self.act_fn,
                                   gptoss=gptoss, clamp_limit=self.clamp_limit)
            # Router weight AFTER the down projection. Stock `ExpertsNbit.forward` and
            # gpt-oss both do this, so those are exact; V4's reference applies it to the
            # gated activation BEFORE w2. That reordering is exact too, not an
            # approximation: w2 is a bias-free linear map (`_project` is `F.linear(x, W)`
            # with no bias), so `w2(h * s) == s * w2(h)` for the scalar router weight.
            # The only difference is where a rounding lands, and bf16 error is relative,
            # so neither order is nearer the truth. Applying it here instead keeps ONE
            # code path for all three epilogues and lets the down GEMM stay batched.
            w = top_k_weights[row_token.index_select(0, hr), row_slot.index_select(0, hr)].to(torch.float32)
            out.index_put_((row_token.index_select(0, hr), row_slot.index_select(0, hr)),
                           dn.to(torch.float32) * w[:, None])

        if _ev is not None:
            _ev[1].record()
            _ev[1].synchronize()
            _amort["gpu_ns"] += int(_ev[0].elapsed_time(_ev[1]) * 1e6)

        # --- COLD: stream ONLY the routed cold experts from pinned host RAM ---
        cr = (~hot_row).nonzero(as_tuple=False).view(-1)
        if cr.numel():
            self._cold_contrib(x, flat, row_token, row_slot, cr, top_k_weights, out, dev)

        return out.sum(dim=1).to(device=input_dev, dtype=input_dtype)

    def _forward_collapsed(self, x, flat, top_k_weights, T, k, H, dev,
                           input_dev, input_dtype):
        """The all-resident collapse (PREREG-b1c). Fires only under the
        placement-static `_all_hot()` predicate, so `flat` IS the local
        id (identity g2h, asserted at cache time) and every row lands in
        a unique (token, slot) cell in row-major order, so the scatter
        is a reshape and the reduction is the same fixed-order
        `sum(dim=1)` the baseline runs. Same kernel, same row order,
        same per-cell values ⇒ bitwise-equal output — the on-box G1
        gate holds this claim to account."""
        rt = self._row_index(T, k, dev)[0]
        gptoss = ((self.h_gu_b, self.h_dn_b, self.alpha, self.limit)
                  if self.gptoss else None)
        # K23 (E4B_INT4_LEAN_GLUE auto / 1): hand over the token rows; the (token, slot) expansion is made inside only
        # for a route that reads it (the lean K19 route gathers token rows itself)
        if _lean_glue_env():
            xr, xtok = None, (x, rt, k)
        else:
            xr, xtok = x.index_select(0, rt), None
        singleton, grouped = _collapsed_grouping(T, getattr(self, "_int4_stores", None))
        dn = _fused_over_stack(xr, flat, self.h_gu_p, self.h_gu_a,
                               self.h_dn_p, self.h_dn_a, self.shapes,
                               self.has_gate, self.act_fn, gptoss=gptoss,
                               clamp_limit=self.clamp_limit,
                               singleton_groups=singleton,
                               device_grouping=grouped,
                               int4_stores=getattr(self, "_int4_stores",
                                                   None),
                               x_tokens=xtok)
        w = top_k_weights.reshape(-1).to(torch.float32)
        ck = _combine_kernel()
        if (ck is not None and dn.is_cuda and dn.dtype == torch.bfloat16
                and input_dtype == torch.bfloat16
                and dn.device == torch.device(input_dev)):
            # one launch: fp32 weight-and-sum over the k slots, bf16 out.
            # NOT bitwise the chain below: the kernel sums in slot order
            # (with a fused multiply-add, exactly so on sm_86), the chain
            # rounds each product and sums in torch's order. Both are within the error bound of a
            # correct fp32 sum (grouped-nf4-gemm lane B393, #393: 144/144
            # census cases, RTX 5090); E4B_FUSE_COMBINE=0 is the chain.
            return ck(dn, w, k)
        out = (dn.to(torch.float32) * w[:, None]).view(T, k, H)
        return out.sum(dim=1).to(device=input_dev, dtype=input_dtype)

    def _forward_diet(self, x, flat, top_k_weights, T, k, H, dev,
                      input_dev, input_dtype):
        """The baseline forward with its dispatch algebra on a diet (T5).

        Same arithmetic, same index order, bit-identical output — the
        registered claim, gated by the cross-arm token-identity check:
        * ``_partition_by_mask`` replaces the hr/cr double-``nonzero``
          (2 syncs) with one small host read;
        * ``row_token``/``row_slot`` are cached per (T, k) instead of two
          ``arange`` builds per layer per step;
        * each ``index_select`` on them happens once per branch, not 3×;
        * an all-hot step skips the partition AND the ``index_put_``: with
          every (token, slot) cell written exactly once in row-major
          order, the scatter is a reshape.
        """
        row_token, row_slot = self._row_index(T, k, dev)
        hot_row = self.is_hot[flat]                            # [T*k] bool
        n_cold, cr, hr = _partition_by_mask(hot_row)

        _amort = getattr(self, "amort", None)
        _ev = None
        if _amort is not None and hr.numel() and dev.type == "cuda":
            _ev = (torch.cuda.Event(enable_timing=True),
                   torch.cuda.Event(enable_timing=True))
            _ev[0].record()
        gptoss = ((self.h_gu_b, self.h_dn_b, self.alpha, self.limit)
                  if self.gptoss else None)
        if n_cold == 0:
            local = self.g2h[flat]
            xr = x.index_select(0, row_token)
            dn = _fused_over_stack(xr, local, self.h_gu_p, self.h_gu_a,
                                   self.h_dn_p, self.h_dn_a, self.shapes,
                                   self.has_gate, self.act_fn, gptoss=gptoss,
                                   clamp_limit=self.clamp_limit)
            # top_k_weights[row_token, row_slot] over ALL rows is exactly
            # its row-major flattening; same values, no gather kernel
            w = top_k_weights.reshape(-1).to(torch.float32)
            out = (dn.to(torch.float32) * w[:, None]).view(T, k, H)
        else:
            out = torch.zeros(T, k, H, dtype=torch.float32, device=dev)
            if hr.numel():
                rt_h = row_token.index_select(0, hr)
                rs_h = row_slot.index_select(0, hr)
                local = self.g2h[flat.index_select(0, hr)]
                xr = x.index_select(0, rt_h)
                dn = _fused_over_stack(xr, local, self.h_gu_p, self.h_gu_a,
                                       self.h_dn_p, self.h_dn_a, self.shapes,
                                       self.has_gate, self.act_fn,
                                       gptoss=gptoss,
                                       clamp_limit=self.clamp_limit)
                w = top_k_weights[rt_h, rs_h].to(torch.float32)
                out.index_put_((rt_h, rs_h),
                               dn.to(torch.float32) * w[:, None])
        if _ev is not None:
            _ev[1].record()
            _ev[1].synchronize()
            _amort["gpu_ns"] += int(_ev[0].elapsed_time(_ev[1]) * 1e6)
        if n_cold:
            self._cold_contrib(x, flat, row_token, row_slot, cr,
                               top_k_weights, out, dev)
        return out.sum(dim=1).to(device=input_dev, dtype=input_dtype)

    def _cold_contrib(self, x, flat, row_token, row_slot, cr, top_k_weights, out, dev):
        """Accumulate the routed cold experts' contributions into ``out`` (fp32,
        on ``dev``). This class streams the routed NF4 to the device and runs the
        fused kernel; the cold engine overrides it to compute on the host."""
        cold_glob = flat.index_select(0, cr).cpu()
        cold_local_full = self.g2c_cpu.index_select(0, cold_glob)     # local id in the full cold stack
        routed, compact = torch.unique(cold_local_full, return_inverse=True)  # only the ones used now
        # gather + stream the routed cold experts' NF4 to the GPU
        gu_p = self.c_gu_p.index_select(0, routed).to(dev, non_blocking=True)
        gu_a = self.c_gu_a.index_select(0, routed).to(dev, non_blocking=True)
        dn_p = self.c_dn_p.index_select(0, routed).to(dev, non_blocking=True)
        dn_a = self.c_dn_a.index_select(0, routed).to(dev, non_blocking=True)
        xr = x.index_select(0, row_token.index_select(0, cr))
        # bias sub-stack aligned to the streamed `routed` subset (compact indexes into
        # it); `routed` is a CPU index, the bias stacks are device-resident.
        gptoss = None
        if self.gptoss:
            r_dev = routed.to(dev)
            gptoss = (self.c_gu_b.index_select(0, r_dev), self.c_dn_b.index_select(0, r_dev),
                      self.alpha, self.limit)
        dn = _fused_over_stack(xr, compact.to(dev), gu_p, gu_a, dn_p, dn_a,
                               self.shapes, self.has_gate, self.act_fn, gptoss=gptoss,
                               clamp_limit=self.clamp_limit)
        w = top_k_weights[row_token.index_select(0, cr), row_slot.index_select(0, cr)].to(torch.float32)
        out.index_put_((row_token.index_select(0, cr), row_slot.index_select(0, cr)),
                       dn.to(torch.float32) * w[:, None])


def hot_residency_available() -> bool:
    try:
        from nf4_grouped import gemm_4bit_grouped  # noqa: F401
    except ImportError:
        return False
    return torch.cuda.is_available()


def _eligible(mod):
    if getattr(mod, "bits", None) != 4 or getattr(mod, "quant_type", None) != "nf4":
        return "storage is not nf4-4bit"
    if getattr(mod, "blocksize", None) != 64:
        return f"blocksize {getattr(mod, 'blocksize', None)} != 64"
    n1, k1 = mod._gate_up_shape
    n2, k2 = mod._down_shape
    if k1 % 64 or k2 % 64:
        return "K not divisible by 64"
    return None


def wrapped_bases(model) -> set:
    """ids of every ``ExpertsNbit`` that is an ``ExpertsLoRA.base``."""
    try:
        from experts4bit_qlora.lora import ExpertsLoRA
        return {id(m.base) for m in model.modules()
                if isinstance(m, ExpertsLoRA) and hasattr(m, "base")}
    except ImportError:
        return set()


def target_modules(model) -> list:
    """The MoE modules a residency patch targets, in dispatch order.

    Every ``ExpertsNbit`` under ``model``, **including** those that are an
    ``ExpertsLoRA.base``. A wrapped base used to be excluded here, on the
    grounds that ``ExpertsLoRA.forward`` re-implements the expert math inline
    (to inject the low-rank delta before the nonlinearity) and so never calls
    ``base.forward`` — making a patch on it dead code. That is no longer true:
    ``ExpertsLoRA._delegate_to_base`` hands the whole forward to the base when
    an engine is attached and the adapter provably contributes nothing (``B``
    is zero-initialised, so an untrained adapter is *identically* zero), which
    is exactly the base-model inference and benchmarking case.

    Excluding them made this function return ``[]`` for every model from
    ``load_moe_4bit_streaming`` — whose experts are always ``ExpertsLoRA`` —
    so the loader path, the one most callers take, could not enable residency
    at all.

    Being in this list means "targetable, and index-bearing", not "reachable by
    every engine". An engine that does not participate in that delegation — the
    v0 :func:`enable_hot_residency`, whose ``_e4b_hot_ref`` marker
    ``_delegate_to_base`` does not look for — must still skip wrapped bases
    itself, consuming their ``hot_sets`` entry like any other skip.

    Shared so that everything keying off ``hot_sets[i]`` agrees on what ``i``
    means. Re-deriving this list independently is how a caller ends up stamping
    per-module state onto the wrong layers when LoRA-wrapped and bare modules are
    interleaved.

    To PATCH these modules, use this list. To OBSERVE them — a forward hook, a
    routing histogram for an informed hot set — use :func:`dispatched_modules`
    instead: a wrapped base here is not called until an engine is attached, so
    hooks on it silently never fire.
    """
    from .. import ExpertsNbit
    return [m for m in model.modules() if isinstance(m, ExpertsNbit)]


def dispatched_modules(model) -> list:
    """The modules actually CALLED, aligned one-to-one with :func:`target_modules`.

    :func:`target_modules` is the list to **patch** — an engine installs its forward
    on the frozen ``ExpertsNbit`` base. It is the wrong list to **observe**: a wrapped
    base is only ever called once an engine is attached and
    ``ExpertsLoRA._delegate_to_base`` hands the forward down, so a
    ``register_forward_pre_hook`` on one fires zero times until that happens.

    The edge is sharp because the usual reason to hook these modules is to build a
    routing histogram for an INFORMED hot set — and that calibration pass necessarily
    runs *before* the engine is enabled. Hooking the bases yields all-zero counts;
    ``topk`` of an all-zero tensor returns ``0..K-1``; and the "informed" hot set
    silently becomes the by-index set the whole mechanism exists to beat. It fails as
    a plausible null result, not as an error — measured 2026-08-04 on granite, where
    it reported informed and index as indistinguishable because they were the same
    set. Guard a calibration pass by asserting it counted something.

    Returns the wrapper where one exists and the bare module otherwise, in
    ``target_modules`` order — so the list you hook and the list you patch agree on
    what ``hot_sets[i]`` means.

    >>> mods = target_modules(model)                     # what an engine patches
    >>> hooks = dispatched_modules(model)                # what to hook to observe
    >>> assert len(mods) == len(hooks)
    """
    try:
        from experts4bit_qlora.lora import ExpertsLoRA
        wrappers = {id(m.base): m for m in model.modules()
                    if isinstance(m, ExpertsLoRA) and hasattr(m, "base")}
    except ImportError:
        wrappers = {}
    return [wrappers.get(id(m), m) for m in target_modules(model)]


def enable_hot_residency(model, hot_sets: Sequence, device: str = "cuda",
                         verbose: bool = False, state_cls=None,
                         reach_wrapped: bool = False) -> int:
    """Partition every eligible ``ExpertsNbit`` under ``model`` into a resident
    GPU hot-stack + a streamed CPU cold-stack, in MoE-layer order.

    .. deprecated:: 0.6.2
        Superseded by :func:`enable_pipelined_residency` — same capability
        (hot-resident + streamed cold, gpt-oss included) with K as config
        (an empty hot set is pure streaming, all experts is fully resident,
        one code path). This v0 engine is kept through 0.6 so the stamped v0
        receipts stay reproducible; removal in 0.7.

    ``hot_sets`` must carry exactly one entry per targeted ``ExpertsNbit`` module
    in module order (a 1-D array/list of hot expert ids each); a wrong length
    raises. Re-enabling rebuilds the partition from the module's *current*
    weights (never a stale cache). **gpt-oss experts (custom clamped-GLU +
    per-expert biases) ARE supported** — the hot path reproduces their
    epilogue. Modules with OTHER custom ``forward`` overrides, ineligible
    storage, ``[fast]``-enabled, or an ``ExpertsLoRA`` base (the
    streaming-loader path — not yet supported) are skipped; ids outside
    ``[0, num_experts)`` raise.

    Memory model: on a fully-resident module the hot (GPU) and cold (CPU) stacks
    are *added* — the module keeps its original packed weights so the reference
    fallback still works, so VRAM is not reduced in that configuration. The VRAM
    win is realized when the base experts are offloaded (streaming loader): the
    resident stack is then the only GPU copy. Standalone Experts4bit is the
    correctness-supported path today."""
    import warnings
    warnings.warn(
        "enable_hot_residency is superseded by enable_pipelined_residency "
        "(same capability; K is config); kept through 0.6 to reproduce the "
        "v0 receipts; removal in 0.7",
        DeprecationWarning, stacklevel=2)
    # The hot/cold forward runs on the fused grouped-GEMM kernel — fail here,
    # not mid-decode inside _fused_over_stack (2026-07-20 pod A/B: a [train]-only
    # install crashed on the first forward after a full model load).
    try:
        from nf4_grouped import gemm_4bit_grouped  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "enable_hot_residency runs on the fused grouped-GEMM kernel "
            "(module nf4_grouped); install it with: pip install 'experts4bit-qlora[fast]'"
        ) from e
    from experts4bit_qlora import Experts4bit, ExpertsNbit

    stock_forwards = {ExpertsNbit.forward, Experts4bit.forward}
    # gpt-oss experts override forward (clamped-GLU + biases); the hot path now
    # reproduces that epilogue (_fused_over_stack gptoss branch), so treat their
    # forwards as supported rather than skipping them as "custom".
    try:
        from experts4bit_qlora.arch.gptoss import GptOssExperts4bit, GptOssExpertsNbit
        stock_forwards |= {GptOssExperts4bit.forward, GptOssExpertsNbit.forward}
        # Same bargain for DeepSeek-V4: allowlisted ONLY because _fused_over_stack
        # reproduces its clamped epilogue via `clamp_limit`. Allowlisting a forward
        # the fused path does not reproduce is worse than skipping it.
        from experts4bit_qlora.arch.deepseek_v4 import (
            DeepseekV4Experts4bit, DeepseekV4ExpertsNbit)
        stock_forwards |= {DeepseekV4Experts4bit.forward, DeepseekV4ExpertsNbit.forward}
    except ImportError:
        pass
    if hasattr(model, "modules"):
        mods = target_modules(model)
        # `target_modules` includes ExpertsLoRA bases, because the pipelined engine
        # reaches them through `ExpertsLoRA._delegate_to_base`. This v0 engine does
        # NOT: that predicate keys off `_e4b_fast_ref` / `_e4b_pipe_ref` and never
        # looks for `_e4b_hot_ref`, so a patch installed here would sit on a forward
        # nothing calls. Wrapped bases stay in `mods` — so `hot_sets[i]` keeps the
        # same meaning it has for every other engine — and are skipped per-module in
        # the loop below, consuming their entry like any other ineligible layer.
        # `reach_wrapped` is the hybrid tier's claim that patches on wrapped
        # bases ARE reachable: its training seam sits on the ExpertsLoRA
        # wrapper and reads this engine's state directly, and
        # `_delegate_to_base` knows `_e4b_hot_ref`, so zero-adapter eval
        # delegates here too. The v0 engine reached directly keeps refusing.
        wrapped = set() if reach_wrapped else wrapped_bases(model)
        if wrapped and not [m for m in mods if id(m) not in wrapped]:
            raise NotImplementedError(
                "every ExpertsNbit here is an ExpertsLoRA.base (the streaming-loader / "
                "offload path). ExpertsLoRA.forward bypasses base.forward, and this "
                "deprecated v0 engine is not among the ones it delegates to. Use "
                "enable_pipelined_residency, which is reached through the wrapper.")
    else:
        mods = [model]
        wrapped = set()
    if len(hot_sets) != len(mods):
        raise ValueError(
            f"hot_sets has {len(hot_sets)} entries but the model has {len(mods)} "
            f"ExpertsNbit modules — exactly one entry per MoE layer in module "
            f"order is required (skipped layers still consume their entry, so "
            f"alignment never silently shifts and trailing entries are never "
            f"dropped)")
    patched = 0
    for i, mod in enumerate(mods):  # hot_sets[i] belongs to mods[i], patched or not
        if id(mod) in wrapped:
            if verbose:
                print(f"[hot_residency] skip {type(mod).__name__}: ExpertsLoRA base — "
                      "this v0 engine is not reached through the wrapper "
                      "(use enable_pipelined_residency)")
            continue
        if type(mod).forward not in stock_forwards and not hasattr(mod, "_e4b_hot_ref"):
            if verbose:
                print(f"[hot_residency] skip {type(mod).__name__}: custom forward")
            continue
        reason = _eligible(mod)
        if reason is not None:
            if verbose:
                print(f"[hot_residency] skip {type(mod).__name__}: {reason}")
            continue
        if hasattr(mod, "_e4b_fast_ref"):
            if verbose:
                print(f"[hot_residency] skip {type(mod).__name__}: [fast] enabled — disable it first")
            continue
        if hasattr(mod, "_e4b_cold_ref"):
            if verbose:
                print(f"[hot_residency] skip {type(mod).__name__}: cold engine enabled — disable it first")
            continue
        if hasattr(mod, "_e4b_pipe_ref"):
            if verbose:
                print(f"[hot_residency] skip {type(mod).__name__}: pipelined residency enabled — disable it first")
            continue
        if hasattr(mod, "_e4b_mxfp4_ref"):
            if verbose:
                print(f"[hot_residency] skip {type(mod).__name__}: mxfp4 NVMe residency enabled — disable it first")
            continue
        if hasattr(mod, "_hot_residency"):
            # rebuild every time — the base weights are frozen NF4, but a caller may
            # have reloaded a checkpoint; a cached partition must never go stale.
            mod._hot_residency = (state_cls or _HotResidency)(mod, hot_sets[i], device)
            patched += 1
            continue
        state = (state_cls or _HotResidency)(mod, hot_sets[i], device)
        mod._e4b_hot_ref = mod.forward
        mod._hot_residency = state

        def _fwd(hidden, top_k_index, top_k_weights, _m=mod):
            st = _m._hot_residency
            cd = _m.compute_dtype if _m.compute_dtype is not None else hidden.dtype
            if cd not in (torch.bfloat16, torch.float16):
                return _m._e4b_hot_ref(hidden, top_k_index, top_k_weights)
            if torch.is_grad_enabled() and (
                hidden.requires_grad or any(p.requires_grad for p in _m.parameters())
            ):
                return _m._e4b_hot_ref(hidden, top_k_index, top_k_weights)
            return st.forward(hidden, top_k_index, top_k_weights)

        mod.forward = _fwd
        patched += 1
    return patched


def disable_hot_residency(model) -> int:
    """Undo :func:`enable_hot_residency`; returns the number of modules restored."""
    mods = model.modules() if hasattr(model, "modules") else [model]
    restored = 0
    for mod in mods:
        if hasattr(mod, "_e4b_hot_ref") and hasattr(mod, "_hot_residency"):
            mod.forward = mod._e4b_hot_ref
            del mod._e4b_hot_ref, mod._hot_residency
            restored += 1
    return restored
