"""Lane P63's ruler, on synthetic tensors with a known answer (bench/p63/p63_compare.py; bench/p63/P63-PREREG.md).

A planted first differing layer and site is found; planted argmax flips are counted; ULP distances are right at the
edges (+0/-0, across zero, NaN); the ULP-near-zero trap B393 hit is excluded by the significance mask; correct fp32
sums sit inside the accuracy bound of their declared operand model and a planted gross error does not; the bound's
blind spot (a bf16-rounded weight at a served K) is pinned so it is never read as a classifier; and every class of
``classify_pair`` fires.
"""
import importlib.util
import math
import pathlib

import pytest
import torch

REPO = pathlib.Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("p63_compare", REPO / "bench" / "p63" / "p63_compare.py")
C = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(C)


def _bf(*v):
    return torch.tensor(v, dtype=torch.float32).to(torch.bfloat16)


def _next_up(x: torch.Tensor) -> torch.Tensor:
    b = x.view(torch.int16) + 1
    return b.view(torch.bfloat16)


def test_ulp_distance_edges():
    one = _bf(1.0)
    assert int(C.bf16_ulp(one, _next_up(one))) == 1
    z, nz = _bf(0.0), _bf(-0.0)
    assert int(C.bf16_ulp(z, nz)) == 0
    assert not bool(C.bit_equal(z, nz, dim=0))                     # exactness is bitwise: -0 is not +0
    tiny = torch.tensor([1], dtype=torch.int16).view(torch.bfloat16)          # smallest positive subnormal
    ntiny = torch.tensor([-32767], dtype=torch.int16).view(torch.bfloat16)    # smallest negative subnormal (0x8001)
    assert int(C.bf16_ulp(tiny, ntiny)) == 2
    assert int(C.bf16_ulp(_bf(1.0), _bf(-1.0))) == 2 * int(C.bf16_order_key(_bf(1.0)))
    assert int(C.bf16_ulp(_bf(float("nan")), one)) == 1 << 20
    with pytest.raises(TypeError):
        C.bf16_order_key(torch.zeros(2))


def test_ulps_near_zero_are_unbounded_and_the_significance_mask_drops_them():
    """B393's trap: a value near zero can sit thousands of ULPs from a value a tiny |d| away."""
    ref = _bf(*([1.0] * 31 + [1e-20]))[None]
    test = ref.clone()
    test[0, -1] = _bf(-1e-20)
    st = C.row_stats(ref, test)
    assert int(st["ulp_all"]) > 10_000
    assert int(st["ulp_sig"]) == 0
    assert float(st["max_abs"]) < 1e-19
    assert not bool(st["equal"])


def _planted():
    torch.manual_seed(0)
    P, Ly, H = 6, 5, 32
    ref = {s: torch.randn(P, Ly, H).to(torch.bfloat16) for s in C.SITES}
    test = {s: v.clone() for s, v in ref.items()}
    for p in (2, 4):
        test["mlp_out"][p, 3, 7] = _next_up(test["mlp_out"][p, 3, 7:8])[0]
    test["attn_core"][5, 1, 0] = _next_up(test["attn_core"][5, 1, 0:1])[0]
    test["layer_out"][5, 4] += 1                                    # later than the first difference: not reported
    final_ref = torch.randn(P, H).to(torch.bfloat16)
    final_test = final_ref.clone()
    final_test[0, 3] = _next_up(final_test[0, 3:4])[0]
    V = 50
    lg_ref = torch.randn(P, V)
    lg_test = lg_ref.clone()
    lg_test[1, int(lg_ref[1].argmin())] += 1e-3          # cannot become the argmax
    return ref, test, final_ref, final_test, lg_ref, lg_test


def test_first_difference_finds_the_planted_layer_and_site_in_forward_order():
    ref, test, fr, ft, lr, lt = _planted()
    site = C.compare_sites(ref, test)
    fin = C.row_stats(fr, ft)
    lg = C.logits_stats(lr, lt)
    fd = C.first_difference(site["equal"], fin["equal"], lg["equal"])
    assert fd == [("final", "final_norm"), ("logits", "logits"), (3, "mlp_out"), None, (3, "mlp_out"),
                  (1, "attn_core")]
    assert int(site["ulp_sig"][2, 3, C.SITES.index("mlp_out")]) <= 1


def test_logits_flips_kl_and_margins():
    torch.manual_seed(1)
    ref = torch.randn(8, 100)
    test = ref.clone()
    for p in (2, 5):                                   # swap the top two: a flip at the control's own margin
        top = ref[p].topk(2).indices
        test[p, top[0]], test[p, top[1]] = ref[p, top[1]], ref[p, top[0]]
    st = C.logits_stats(ref, test)
    assert int((~st["argmax_equal"]).sum()) == 2
    assert float(st["kl"][0]) == 0.0 and bool(st["equal"][0])
    assert float(st["kl"][2]) > 0
    self_kl = C.logits_stats(ref, ref.clone())["kl"]
    assert float(self_kl.max()) == 0.0


def test_summarize_mode_counts_what_the_prereg_reads():
    ref, test, fr, ft, lr, lt = _planted()
    site = C.compare_sites(ref, test)
    fin = C.row_stats(fr, ft)
    lg = C.logits_stats(lr, lt)
    s = C.summarize_mode(site, fin, lg, positions=[10, 11, 12, 13, 14, 15], rows_in_window=[0, 1, 2, 3, 4, 5])
    assert s["positions"] == 6 and s["exact_positions"] == 1
    assert s["min_first_diff_layer"] == 1
    assert s["first_diff_site_hist"] == {"final": 1, "logits": 1, "mlp_out": 2, "none": 1, "attn_core": 1}
    assert s["per_position"][2]["first_diff"] == [3, "mlp_out"] and s["per_position"][3]["first_diff"] is None
    assert s["per_layer"]["mlp_out"]["frac_equal"][3] == pytest.approx(4 / 6)
    assert s["argmax_flips"] == 0


@pytest.mark.parametrize("K", [256, 2048])
def test_correct_sums_are_inside_the_bound_and_a_planted_gross_error_is_not(K):
    """The DEFECT line: exact-operand and bf16-weight paths sit inside the bound of their declared model at a small
    and a served K; a result 1 % off on one element does not."""
    torch.manual_seed(2)
    R, N = 6, 40
    x = torch.randn(R, K).to(torch.bfloat16)
    w = torch.randn(N, K) * 0.05                      # a full-precision fp32 "stored grid"
    y_fp32 = (x.float() @ w.t()).to(torch.bfloat16)                      # exact operands, fp32 sum
    y_wbf16 = (x.float() @ w.to(torch.bfloat16).float().t()).to(torch.bfloat16)   # weight rounded to bf16
    exact = C.gemm_truth(x, w, "dense.cublas")
    assert C.bound_ratio(y_fp32, exact["truth"], exact["bound"]) <= 1.0
    declared = C.gemm_truth(x, w, "int4.cublas_cache")
    assert C.bound_ratio(y_wbf16, declared["truth"], declared["bound"]) <= 1.0
    bad = y_fp32.clone()
    i = int(exact["truth"].abs().argmax())
    bad.view(-1)[i] = (bad.view(-1)[i].float() * 1.01).to(torch.bfloat16)
    assert C.bound_ratio(bad, exact["truth"], exact["bound"]) > 1.0


def test_the_bound_cannot_see_a_bf16_weight_at_served_k_which_is_why_it_is_not_a_classifier():
    """Pinned so nobody reads a bound ratio <= 1 as 'exact operands': at a served K (o_proj's 4096) the worst-case
    accumulation term admits a bf16-rounded weight under the EXACT-weight model (p63_compare's docstring). The
    descriptive rel_rms_err still sees the extra rounding."""
    torch.manual_seed(2)
    x = torch.randn(6, 4096).to(torch.bfloat16)
    w = torch.randn(40, 4096) * 0.05
    y_wbf16 = (x.float() @ w.to(torch.bfloat16).float().t()).to(torch.bfloat16)
    exact = C.gemm_truth(x, w, "dense.cublas")
    assert C.bound_ratio(y_wbf16, exact["truth"], exact["bound"]) <= 1.0
    assert C.rel_rms_err(y_wbf16, exact["truth"]) > C.rel_rms_err((x.float() @ w.t()).to(torch.bfloat16),
                                                                  exact["truth"])


def _q8(x):
    """quant_x_rows' arithmetic in torch: per (row, 32-block) s = max|x| / 127 + 1e-12, q = floor(x / s + 0.5)."""
    R, K = x.shape
    xb = x.float().reshape(R, K // 32, 32)
    s = xb.abs().amax(-1, keepdim=True) / 127.0 + 1e-12
    q = torch.floor(xb / s + 0.5).clamp(-127, 127)
    return (q * s).reshape(R, K)


def test_the_int8_activation_model_covers_the_int8_path_and_the_bf16_model_does_not():
    torch.manual_seed(3)
    R, K, N = 5, 256, 24
    x = torch.randn(R, K).to(torch.bfloat16)
    w = torch.randint(-8, 8, (N, K)).float() * 0.01                    # an int4 grid times a scale
    y_q8 = (_q8(x) @ w.t()).to(torch.bfloat16)
    q8 = C.gemm_truth(x, w, "int4.gemv")
    assert C.bound_ratio(y_q8, q8["truth"], q8["bound"]) <= 1.0
    bf = C.gemm_truth(x, w, "dense.cublas")
    assert C.bound_ratio(y_q8, bf["truth"], bf["bound"]) > 1.0


def test_gemm_truth_is_chunk_invariant():
    torch.manual_seed(4)
    x = torch.randn(3, 64).to(torch.bfloat16)
    w = torch.randn(50, 64)
    a = C.gemm_truth(x, w, "dense.cublas", chunk=7)
    b = C.gemm_truth(x, w, "dense.cublas", chunk=10_000)
    assert torch.equal(a["truth"], b["truth"]) and torch.equal(a["bound"], b["bound"])


def test_combine_bound_accepts_two_correct_orders_many_ulps_apart():
    """B393's cancellation case: two correct fp32 orders round to bf16 values many ULPs apart; both are inside."""
    k = 3
    dn = torch.tensor([[1.0], [1.0 / 3.0], [-1.0]], dtype=torch.float32).to(torch.bfloat16)
    w = torch.tensor([1.0, 1e-3, 1.0])
    t = dn.float() * w[:, None]
    fwd = ((t[0] + t[1]) + t[2]).to(torch.bfloat16)[None]
    rev = ((t[2] + t[0]) + t[1]).to(torch.bfloat16)[None]
    tb = C.combine_truth(dn, w, k)
    assert C.bound_ratio(fwd, tb["truth"], tb["bound"]) <= 1.0
    assert C.bound_ratio(rev, tb["truth"], tb["bound"]) <= 1.0


def test_bound_ratio_zero_bound_edges():
    z = torch.zeros(1, 1, dtype=torch.float64)
    assert C.bound_ratio(torch.zeros(1, 1), z, z) == 0.0
    assert math.isinf(C.bound_ratio(torch.ones(1, 1), z, z))


@pytest.mark.parametrize("eq,a,b,ra,rb,want", [
    (True, "int4.gemv", "int4.deq_bf16", 0.5, 0.5, "EXACT"),
    (False, "int4.gemv", "int4.gemv", 0.5, 0.7, "REORDER"),
    (False, "int4.gemv", "int4.grouped_gemm", 0.5, 0.7, "REORDER"),
    (False, "int4.gemv", "int4.deq_bf16", 0.5, 0.7, "PRECISION"),
    (False, "nf4.dotpad", "nf4.mtile", 0.2, 0.3, "PRECISION"),
    (False, "int4.k16", "int4.cublas_cache", 0.2, 0.3, "REORDER"),
    (False, "int4.gemv", "int4.gemv", 0.5, 1.5, "DEFECT?"),
    (False, "dense.cublas", "dense.cublas", None, None, "REORDER"),
])
def test_classify_pair_branches(eq, a, b, ra, rb, want):
    assert C.classify_pair(eq, a, b, ra, rb) == want


def test_rows_equal_to_singles():
    torch.manual_seed(5)
    s = torch.randn(4, 8).to(torch.bfloat16)
    b = s.clone()
    b[2, 1] = _next_up(b[2, 1:2])[0]
    r = C.rows_equal_to_singles(b, s)
    assert r["rows"] == 4 and r["rows_bit_equal"] == 3 and r["max_ulp_sig"] <= 1
    ints = torch.arange(6).reshape(3, 2)
    assert C.rows_equal_to_singles(ints, ints.clone())["rows_bit_equal"] == 3


def test_a_dtype_difference_is_a_recorded_row_count_dependence_not_a_crash():
    """The rehearsal's case: fp32 routing weights at one row count, bf16 at another (a bit view would mismatch)."""
    w32 = torch.rand(5, 8)
    w16 = w32.to(torch.bfloat16)
    r = C.rows_equal_to_singles(w16, w32)
    assert r["rows_bit_equal"] == 0 and r["dtype_batched"] == "torch.bfloat16" and r["dtype_single"] == "torch.float32"
    assert 0 < r["max_abs"] < 2 ** -8
    assert C.rows_equal_to_singles(torch.zeros(2, 3), torch.zeros(2, 4))["rows_bit_equal"] == 0
