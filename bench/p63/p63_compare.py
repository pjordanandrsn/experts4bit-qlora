"""bench/p63/p63_compare.py -- lane P63's comparison arithmetic, pure torch (CPU-testable; no kernels, no model).

Everything that turns captured tensors into a verdict lives here, so ``tests/test_p63_compare.py`` can check the ruler
on synthetic tensors with a known answer: a planted first differing layer, planted argmax flips, known ULP distances,
correct sums inside the accuracy bound and a planted gross error outside it.

Three kinds of reading use it (bench/p63/P63-PREREG.md, "Instrument"):

* **end to end** -- per token, per decoder layer, per site, the T = 1 control against the same token inside a W-row
  verify or an L-row prefill: bit equality, max |d|, relative L2, and the bf16 ULP distance over the SIGNIFICANT
  elements only (see ``sig_frac``);
* **module replay** -- a module's own T = 1 inputs stacked into one n-row call: is each row's output the T = 1 output;
* **kernel census** -- a kernel path on captured activations against the exact fp64 result of the same logical
  operands, normalised by the accuracy bound its declared operand model allows.

What the bound can and cannot say (found while building this ruler, before any run): it is a WORST-CASE bound, and its
fp32-accumulation term grows as K * sum|x w|. At K = 2048 that term is larger than the whole error a bf16-rounded weight
typically adds (~sqrt(K) * 2^-8 * |t|), so a path that rounds its weight to bf16 sits inside even the exact-weight
bound. The bound therefore cannot tell operand models apart at served shapes, and it is not used to. It is the
DEFECT line only: a path outside the bound of its OWN declared model is worse than any correct implementation of that
model can be (B393's case C). Whether two paths are REORDER (same operands) or PRECISION (different operands) is read
from the kernels' source -- ``OPERAND_MODELS`` -- and registered; ``rel_rms_err`` is reported beside it as the
descriptive magnitude (an int8-activation path shows several times the error of a bf16 one), never decided on.

Why ULPs are not the metric (B393's lesson, carried over): a bf16 value near zero has a tiny ULP, so two correct
roundings of one sum that cancels can sit thousands of ULPs apart. Hidden states have near-zero elements in every row.
Bit equality is the exactness line; magnitudes are reported relative to the row (``rel_l2``, ``max_abs``), ULPs only
over elements at least ``sig_frac`` of the row RMS; and the correct/defect line for a single GEMM or sum is accuracy
against fp64, never a ULP count.
"""
from __future__ import annotations

import math

import torch

#: Sites captured per decoder layer, in forward order. The first differing site of a token is searched in this order,
#: layer by layer, then ``final_norm``, then the logits.
SITES = ("attn_in", "attn_core", "attn_out", "mlp_in", "mlp_out", "layer_out")

U_BF16 = 2.0 ** -8      # unit roundoff, round to nearest (p = 8)
U_TF32 = 2.0 ** -11     # p = 11
#: fp32 ACCUMULATION unit, deliberately 2^-23 rather than 2^-24: tensor-core MMA adds are not guaranteed
#: round-to-nearest (truncating alignment), and a bound that a correct tensor-core sum can exceed is not a bound.
U_ACC = 2.0 ** -23

#: Declared operand model of every kernel path the census reads: (activation model, weight unit roundoff).
#: "bf16" = the bf16 activation as given (exact); "q8" = quant_x_rows' per-(row, 32-block) int8 grid, error <= s/2
#: with s = max|x_block| / 127. Weight u = 0 means the stored grid value is used exactly (fp32 products), 2^-8 means
#: it is rounded to bf16 before the MMA, 2^-11 to TF32. Two paths with the SAME model differ only in summation order
#: (REORDER); different models are different arithmetic (PRECISION). Read from the kernels' source, registered in
#: P63-PREREG.md's route table; the census checks only that each path lies inside its own model's bound (the DEFECT line).
OPERAND_MODELS = {
    "int4.gemv": ("q8", 0.0),               # gnf4 int4_b32._gemv_int4_b32: int8 x, exact int32 block dots
    "int4.grouped_gemm": ("q8", 0.0),       # int4_b32._gemm_int4_b32_grouped: int8 tl.dot, same operands
    "int4.deq_bf16": ("bf16", U_BF16),      # hot_residency prefill/verify: dequant, .to(bf16), matmul
    "int4.k16": ("bf16", U_BF16),           # int4_smallm: in-register dequant, bf16 MMA
    "int4.cublas_cache": ("bf16", U_BF16),  # Int4Linear > 16 rows: cuBLAS on the cached bf16 weight
    "nf4.gemv_scalar": ("bf16", 0.0),       # nf4_grouped._gemv_nf4_grouped: fp32 lut*a, * absmax
    "nf4.dotpad": ("bf16", U_BF16),         # nf4_grouped._gemv_nf4_dotpad: (lut*am).to(bf16), bf16 MMA
    "nf4.mtile": ("bf16", U_TF32),          # nf4_grouped._gemm_nf4_grouped variant 1: fp32 operands, TF32 tl.dot
    "dense.cublas": ("bf16", 0.0),          # nn.Linear bf16: operands exact, fp32 accumulate
}

#: Paths whose GEMM is cuBLAS called through torch. torch's default
#: ``torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True`` lets cuBLAS round split-K partials to
#: bf16, which no fp32-accumulation bound covers: the NAS A2000 rehearsal read bound ratios up to 7.35 on bf16 attention
#: projections at 16 and 160 rows under the default flag (bench/p63/rehearsal-a2000/). For these paths the probe also
#: reruns the call with the flag OFF; the DEFECT line reads that ratio, and the flag-on ratio is reported beside it.
CUBLAS_PATHS = frozenset({"int4.deq_bf16", "int4.cublas_cache", "dense.cublas"})


# --------------------------------------------------------------------------------------------- bf16 ULP distance --
def bf16_order_key(x: torch.Tensor) -> torch.Tensor:
    """Map bf16 values to int32 keys whose order is the numeric order and whose difference is the ULP distance.
    +0 and -0 share key 0. NaN is not special-cased here (callers mask it)."""
    if x.dtype != torch.bfloat16:
        raise TypeError(f"bf16_order_key takes bf16, got {x.dtype}")
    b = x.contiguous().view(torch.int16).to(torch.int32)
    return torch.where(b >= 0, b, -(b + 32768))


def bf16_ulp(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Elementwise bf16 ULP distance |key(a) - key(b)| (int64). A NaN on either side reads as 2**20."""
    d = (bf16_order_key(a).to(torch.int64) - bf16_order_key(b).to(torch.int64)).abs()
    nan = torch.isnan(a) | torch.isnan(b)
    return torch.where(nan, torch.full_like(d, 1 << 20), d)


def bit_equal(a: torch.Tensor, b: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Bitwise equality reduced over ``dim`` (a -0 against a +0 is NOT equal: this is the exactness line)."""
    if a.dtype == torch.bfloat16:
        return (a.contiguous().view(torch.int16) == b.contiguous().view(torch.int16)).all(dim)
    if a.dtype == torch.float32:
        return (a.contiguous().view(torch.int32) == b.contiguous().view(torch.int32)).all(dim)
    return (a == b).all(dim)


# ------------------------------------------------------------------------------------------------ row statistics --
def row_stats(ref: torch.Tensor, test: torch.Tensor, sig_frac: float = 2.0 ** -4) -> dict:
    """Per-row comparison over the last axis of two bf16 tensors of one shape.

    ``equal`` bitwise; ``max_abs`` and ``rel_l2`` (||test - ref|| / ||ref||, 0 when both are zero) in fp64;
    ``ulp_sig`` the max ULP distance over elements with |ref| >= sig_frac * rms(ref row); ``ulp_all`` over every
    element (reported, never decided on)."""
    if ref.shape != test.shape:
        raise ValueError(f"shape mismatch {tuple(ref.shape)} vs {tuple(test.shape)}")
    r, t = ref.double(), test.double()
    d = (t - r).abs()
    rn = r.pow(2).sum(-1).sqrt()
    dn = d.pow(2).sum(-1).sqrt()
    rel = torch.where(rn > 0, dn / rn.clamp_min(1e-300), torch.where(dn > 0, torch.full_like(dn, math.inf),
                                                                       torch.zeros_like(dn)))
    ulp = bf16_ulp(ref, test)
    rms = r.pow(2).mean(-1, keepdim=True).sqrt()
    sig = r.abs() >= sig_frac * rms
    ulp_sig = torch.where(sig, ulp, torch.zeros_like(ulp)).amax(-1)
    return {"equal": bit_equal(ref, test), "max_abs": d.amax(-1), "rel_l2": rel,
            "ulp_sig": ulp_sig, "ulp_all": ulp.amax(-1)}


def compare_sites(ref: dict, test: dict, sites=SITES) -> dict:
    """``ref[site]`` / ``test[site]``: bf16 ``[P, Ly, H]``. Returns ``[P, Ly, S]`` tensors (S in ``sites`` order)."""
    per = [row_stats(ref[s], test[s]) for s in sites]
    return {k: torch.stack([p[k] for p in per], dim=-1) for k in per[0]}


def logits_stats(ref: torch.Tensor, test: torch.Tensor) -> dict:
    """``[P, V]`` logits (any float dtype). Bitwise equality, argmax agreement, max |d|, KL(P_ref || P_test) in fp64
    over the full vocabulary, and the control's top-1 minus top-2 margin (a flip at a small margin is the
    reorder-class signature; a flip at a large one is not)."""
    eq = bit_equal(ref, test)
    a_r, a_t = ref.argmax(-1), test.argmax(-1)
    lr = torch.log_softmax(ref.double(), -1)
    lt = torch.log_softmax(test.double(), -1)
    kl = (lr.exp() * (lr - lt)).sum(-1).clamp_min(0.0)
    top2 = ref.double().topk(2, dim=-1).values
    return {"equal": eq, "argmax_equal": a_r == a_t, "argmax_ref": a_r, "argmax_test": a_t,
            "max_abs": (test.double() - ref.double()).abs().amax(-1), "kl": kl,
            "margin_ref": top2[:, 0] - top2[:, 1]}


def first_difference(equal: torch.Tensor, final_equal: torch.Tensor | None = None,
                     logits_equal: torch.Tensor | None = None, sites=SITES) -> list:
    """Per position, the first (layer, site) whose rows are not bit-equal, searching layers in order and the sites
    of a layer in forward order, then ``("final", "final_norm")``, then ``("logits", "logits")``. None when every
    compared tensor is bit-equal. ``equal`` is ``[P, Ly, S]`` bool."""
    P, Ly, S = equal.shape
    if S != len(sites):
        raise ValueError(f"{S} site columns for {len(sites)} site names")
    flat = (~equal).reshape(P, Ly * S)
    out = []
    for p in range(P):
        idx = torch.nonzero(flat[p], as_tuple=False)
        if idx.numel():
            i = int(idx[0, 0])
            out.append((i // S, sites[i % S]))
        elif final_equal is not None and not bool(final_equal[p]):
            out.append(("final", "final_norm"))
        elif logits_equal is not None and not bool(logits_equal[p]):
            out.append(("logits", "logits"))
        else:
            out.append(None)
    return out


def summarize_mode(site_cmp: dict, final_cmp: dict | None, logit_cmp: dict, positions, rows_in_window=None,
                   sites=SITES) -> dict:
    """One comparison mode (a verify width or the prefill) of one sub-arm, JSON-able.

    Per layer and site: the fraction of compared positions bit-equal, max rel_l2, max |d|, max significant ULP.
    Per position: the first differing (layer, site), logits bit equality, argmax agreement, KL, control margin.
    Totals: exact positions (every site of every layer, the final norm and the logits bit-equal), argmax flips,
    KL max/mean, the lowest first-differing layer, and a histogram of first-differing sites."""
    eq = site_cmp["equal"]
    fe = final_cmp["equal"] if final_cmp is not None else None
    fd = first_difference(eq, fe, logit_cmp["equal"], sites)
    P, Ly, S = eq.shape
    per_layer = {s: {"frac_equal": [round(float(v), 6) for v in eq[:, :, j].float().mean(0)],
                     "max_rel_l2": [float(v) for v in site_cmp["rel_l2"][:, :, j].amax(0)],
                     "max_abs": [float(v) for v in site_cmp["max_abs"][:, :, j].amax(0)],
                     "max_ulp_sig": [int(v) for v in site_cmp["ulp_sig"][:, :, j].amax(0)]}
                 for j, s in enumerate(sites)}
    rows = []
    for p in range(P):
        r = {"pos": int(positions[p]), "first_diff": (None if fd[p] is None else [fd[p][0], fd[p][1]]),
             "logits_equal": bool(logit_cmp["equal"][p]), "argmax_equal": bool(logit_cmp["argmax_equal"][p]),
             "kl": float(logit_cmp["kl"][p]), "max_abs_logit": float(logit_cmp["max_abs"][p]),
             "margin_ref": float(logit_cmp["margin_ref"][p])}
        if rows_in_window is not None:
            r["row"] = int(rows_in_window[p])
        rows.append(r)
    layers_hit = [f[0] for f in fd if f is not None and isinstance(f[0], int)]
    hist = {}
    for f in fd:
        key = "none" if f is None else (f[1] if isinstance(f[0], int) else f[0])
        hist[key] = hist.get(key, 0) + 1
    flips = int((~logit_cmp["argmax_equal"]).sum())
    return {"positions": P,
            "exact_positions": sum(1 for f in fd if f is None),
            "argmax_flips": flips,
            "flip_margins": [float(m) for m, a in zip(logit_cmp["margin_ref"], logit_cmp["argmax_equal"]) if not a],
            "kl_max": float(logit_cmp["kl"].max()) if P else 0.0,
            "kl_mean": float(logit_cmp["kl"].mean()) if P else 0.0,
            "max_abs_logit": float(logit_cmp["max_abs"].max()) if P else 0.0,
            "min_first_diff_layer": (min(layers_hit) if layers_hit else None),
            "first_diff_site_hist": hist,
            "final_norm_frac_equal": (None if fe is None else round(float(fe.float().mean()), 6)),
            "per_layer": per_layer, "per_position": rows}


# ------------------------------------------------------------------------------------------ accuracy against fp64 --
def q8_error_per_element(x: torch.Tensor, block: int = 32) -> torch.Tensor:
    """Upper bound on |x - dequant(quant_x_rows(x))| per element: half the block scale, s = max|x_block| / 127
    (+ the kernel's 1e-12), with a 2^-20 relative margin for the fp32 arithmetic that computes s."""
    R, K = x.shape
    xb = x.double().reshape(R, K // block, block)
    s = xb.abs().amax(-1, keepdim=True) / 127.0 + 1e-12
    return (0.5 * s * (1 + 2.0 ** -20)).expand_as(xb).reshape(R, K)


def gemm_truth(x: torch.Tensor, w_exact: torch.Tensor, model: str, chunk: int = 16384) -> dict:
    """Exact fp64 ``y = x @ w.T`` of the LOGICAL operands (x as given, the stored weight grid exactly) and the bound
    the declared operand model allows per element:

        bound = 2^-8 |y| + (1 + 2^-8) * [ (K + 2) u_acc S + u_w S + (1 + u_w) E ]

    S = sum_k |x_k w_k|, E = sum_k |w_k| e_k with e_k the activation model's per-element error (0 for bf16, half the
    int8 block scale for q8), u_w the weight rounding. Computed in N-chunks so an lm_head-sized weight fits.
    Returns fp64 ``truth`` and ``bound`` of shape [R, N]."""
    xm, u_w = OPERAND_MODELS[model]
    xd = x.double()
    ex = q8_error_per_element(x) if xm == "q8" else None
    R, K = xd.shape
    N = w_exact.shape[0]
    truth = torch.empty(R, N, dtype=torch.float64, device=x.device)
    bound = torch.empty_like(truth)
    for n0 in range(0, N, chunk):
        wd = w_exact[n0:n0 + chunk].double()
        y = xd @ wd.t()
        S = xd.abs() @ wd.abs().t()
        E = (ex @ wd.abs().t()) if ex is not None else torch.zeros_like(y)
        truth[:, n0:n0 + chunk] = y
        bound[:, n0:n0 + chunk] = U_BF16 * y.abs() + (1 + U_BF16) * ((K + 2) * U_ACC * S + u_w * S + (1 + u_w) * E)
    return {"truth": truth, "bound": bound}


def rel_rms_err(y: torch.Tensor, truth: torch.Tensor) -> float:
    """||y - truth|| / ||truth|| in fp64: the descriptive error magnitude of a path (reported, never decided on)."""
    tn = float(truth.pow(2).sum().sqrt())
    en = float((y.double() - truth).pow(2).sum().sqrt())
    return en / tn if tn > 0 else (0.0 if en == 0 else math.inf)


def bound_ratio(y: torch.Tensor, truth: torch.Tensor, bound: torch.Tensor) -> float:
    """max |y - truth| / bound over elements. <= 1: the path is inside its declared operand model with SOME fp32
    summation order. A zero bound with an exact match reads 0; with a mismatch, inf."""
    err = (y.double() - truth).abs()
    zero = bound == 0
    r = torch.where(zero, torch.where(err == 0, torch.zeros_like(err), torch.full_like(err, math.inf)),
                    err / bound.clamp_min(1e-300))
    return float(r.max()) if r.numel() else 0.0


def combine_truth(dn: torch.Tensor, w: torch.Tensor, k: int) -> dict:
    """fp64 of the top-k combine ``sum_j dn[t*k + j] * w[t*k + j]`` and B393's bound for any fp32 order plus the bf16
    cast (products rounded once, k - 1 adds, with U_ACC): 2^-8 |y| + (k + 1) u_acc S (1 + 2^-8)."""
    TK, H = dn.shape
    T = TK // k
    t = dn.double() * w.double()[:, None]
    y = t.view(T, k, H).sum(1)
    S = t.abs().view(T, k, H).sum(1)
    return {"truth": y, "bound": U_BF16 * y.abs() + (k + 1) * U_ACC * S * (1 + U_BF16)}


def classify_pair(bit_equal_all: bool, path_a: str, path_b: str, ratio_a: float | None = None,
                  ratio_b: float | None = None) -> str:
    """The route class of a (T = 1 path, T > 1 path) pair.

    EXACT: every row bit-equal. DEFECT?: a path outside its OWN declared model's bound (worse than its operands allow
    under any fp32 order -- B393's case C; inspected, never auto-filed). REORDER: same declared operand model, so only
    the summation order differs. PRECISION: different declared operand models -- the two row counts compute different
    functions. The operand models are the kernels' source, not a measurement (see the module docstring)."""
    if bit_equal_all:
        return "EXACT"
    if any(r is not None and r > 1.0 for r in (ratio_a, ratio_b)):
        return "DEFECT?"
    return "REORDER" if OPERAND_MODELS[path_a] == OPERAND_MODELS[path_b] else "PRECISION"


def rows_equal_to_singles(batched: torch.Tensor, singles: torch.Tensor) -> dict:
    """Module replay / kernel census: row i of an n-row call against the 1-row call of row i. Both ``[n, ...]``,
    flattened past the first axis.

    A DTYPE difference is itself a row-count dependence (the NAS rehearsal found one: the fused router epilogue returns
    fp32 routing weights at <= 64 rows and the upstream router bf16 above), so it is recorded, every row reads unequal,
    and the magnitudes are taken in fp64 -- never a crash on a bit view of mismatched widths."""
    b = batched.reshape(batched.shape[0], -1)
    s = singles.reshape(singles.shape[0], -1)
    if b.dtype != s.dtype or b.shape != s.shape:
        d = ((b.double() - s.double()).abs() if b.shape == s.shape else None)
        return {"rows": int(b.shape[0]), "rows_bit_equal": 0, "dtype_batched": str(b.dtype),
                "dtype_single": str(s.dtype), "shape_batched": list(b.shape), "shape_single": list(s.shape),
                "max_abs": (float(d.max()) if d is not None and d.numel() else None)}
    eq = bit_equal(b, s)
    out = {"rows": int(b.shape[0]), "rows_bit_equal": int(eq.sum())}
    if b.dtype == torch.bfloat16:
        st = row_stats(s, b)
        out.update(max_ulp_sig=int(st["ulp_sig"].max()), max_rel_l2=float(st["rel_l2"].max()),
                   max_abs=float(st["max_abs"].max()))
    else:
        d = (b.double() - s.double()).abs()
        out.update(max_abs=float(d.max()) if d.numel() else 0.0)
    return out
