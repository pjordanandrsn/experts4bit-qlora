"""P65's activation entropy (bench/p65/expert_entropy.py) and its census plumbing (bench/p65/p65_census.py), on CPU.

Known distributions give known ratios; the ratio read from the tap's moments equals the ratio of the raw rows; the
calibration sink's new first moment leaves its Hessians bitwise unchanged; a P65 census row carries the entropy field
with P44's ``rel_act`` untouched; two halves combine exactly; the c4val1 windows follow ``calib_batches``' rule."""
import hashlib
import importlib.util
import math
import pathlib
import sys
import types

import pytest
import torch

REPO = pathlib.Path(__file__).resolve().parents[1]
P65 = REPO / "bench" / "p65"
P44 = REPO / "bench" / "p44"
for _p in (str(P65), str(P44)):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ent = _load("expert_entropy", P65 / "expert_entropy.py")
pytest.importorskip("int4_pack_ref")          # census_row packs through the kernel package's reference packer
cen = _load("p65_census", P65 / "p65_census.py")
res = _load("expert_residuals", P44 / "expert_residuals.py")


# ------------------------------------------------------------------------------------------- known distributions
def test_constant_channels_with_distinct_means_have_rho_zero():
    # every channel constant, means differ: all variance is between channels -> sigma2_within = 0
    mu = torch.tensor([1.0, -2.0, 3.0, 0.5], dtype=torch.float64)
    r = ent.entropy_ratio(mu * mu, mu)
    assert r["sigma2_within"] == 0.0 and r["rho"] == 0.0 and r["dH_nats"] is None


def test_identical_channels_have_rho_one():
    # every channel N(mu, s2) with the SAME mu: no between-channel variance -> within == total
    s2, mu = 0.7, 1.3
    m2 = torch.full((64,), s2 + mu * mu, dtype=torch.float64)
    r = ent.entropy_ratio(m2, torch.full((64,), mu, dtype=torch.float64))
    assert r["rho"] == pytest.approx(1.0, abs=1e-12) and r["dH_nats"] == pytest.approx(0.0, abs=1e-12)


def test_analytic_two_channel_case():
    # means +1 / -1, unit variances: within 1, total 1 + Var_c(mu) = 2 -> rho 1/2, dH = 1/2 ln 1/2
    mu = torch.tensor([1.0, -1.0], dtype=torch.float64)
    m2 = torch.tensor([2.0, 2.0], dtype=torch.float64)
    r = ent.entropy_ratio(m2, mu)
    assert r["sigma2_within"] == pytest.approx(1.0) and r["sigma2_total"] == pytest.approx(2.0)
    assert r["rho"] == pytest.approx(0.5) and r["dH_nats"] == pytest.approx(0.5 * math.log(0.5))


def test_rho_is_the_gaussian_entropy_ratio():
    # Colla-Q Eq. 6: rho = exp(2 H_within) / exp(2 H_total) with H = 1/2 ln(2 pi e s2)
    torch.manual_seed(0)
    mu = torch.randn(32, dtype=torch.float64)
    var = torch.rand(32, dtype=torch.float64) + 0.1
    r = ent.entropy_ratio(var + mu * mu, mu)
    H = lambda s2: 0.5 * math.log(2 * math.pi * math.e * s2)  # noqa: E731
    assert r["rho"] == pytest.approx(math.exp(2 * H(r["sigma2_within"])) / math.exp(2 * H(r["sigma2_total"])), rel=1e-12)
    assert r["dH_nats"] == pytest.approx(H(r["sigma2_within"]) - H(r["sigma2_total"]), rel=1e-12)


def test_energy_entropy_extremes():
    even = ent.entropy_ratio(torch.ones(16, dtype=torch.float64), torch.zeros(16, dtype=torch.float64))
    assert even["h_energy"] == pytest.approx(1.0)
    one_hot = torch.zeros(16, dtype=torch.float64)
    one_hot[3] = 5.0
    peaked = ent.entropy_ratio(one_hot, torch.zeros(16, dtype=torch.float64))
    assert peaked["h_energy"] == pytest.approx(0.0, abs=1e-12)


def test_cancelled_channel_is_clamped_and_counted():
    mu = torch.tensor([1.0, 2.0], dtype=torch.float64)
    m2 = mu * mu - torch.tensor([1e-12, 0.0], dtype=torch.float64)     # channel 0 rounds below zero
    r = ent.entropy_ratio(m2, mu)
    assert r["neg_var_channels"] == 1 and r["sigma2_within"] >= 0.0


def test_shape_mismatch_refuses():
    with pytest.raises(ValueError):
        ent.entropy_ratio(torch.ones(3), torch.ones(4))


# ------------------------------------------------------------------------------------------- moments == raw rows
def _rows(T=4000, K=48, C=24, seed=1):
    g = torch.Generator().manual_seed(seed)
    h = torch.randn(T, K, generator=g, dtype=torch.float64) * 0.5 + torch.randn(K, generator=g, dtype=torch.float64)
    W = torch.randn(C, K, generator=g, dtype=torch.float64) * 0.1
    return h, W


def _hessian(h):
    return (2.0 / h.shape[0]) * (h.t() @ h)      # the running mean gptq_pack.HessianAccumulator converges to


def test_output_ratio_from_moments_equals_the_direct_ratio():
    h, W = _rows()
    m2, mu = ent.output_moments(W.float(), _hessian(h).float(), h.mean(0))
    got = ent.entropy_ratio(m2, mu)
    want = ent.direct_entropy_ratio(h @ W.t())
    assert got["rho"] == pytest.approx(want["rho"], rel=1e-5)
    assert got["sigma2_total"] == pytest.approx(want["sigma2_total"], rel=1e-5)


def test_input_ratio_from_moments_equals_the_direct_ratio():
    h, _W = _rows()
    m2, mu = ent.input_moments(_hessian(h), h.mean(0))
    assert ent.entropy_ratio(m2, mu)["rho"] == pytest.approx(ent.direct_entropy_ratio(h)["rho"], rel=1e-10)


def test_entropy_fields_roles_and_no_rows():
    h, W = _rows(K=16, C=8)
    H, m = _hessian(h), h.mean(0)
    f_dn, (m2, mu) = ent.entropy_fields("dn", W, H, m)
    f_gu, _ = ent.entropy_fields("gu", W, H, m)
    assert f_dn["of"].startswith("output") and f_dn["channels"] == 8 and m2.shape == (8,)
    assert f_gu["of"].startswith("input") and f_gu["channels"] == 16
    assert ent.entropy_fields("dn", W, None, None) is None
    with pytest.raises(ValueError):
        ent.entropy_fields("up", W, H, m)


def test_the_ratio_is_deterministic():
    h, W = _rows()
    a = ent.entropy_fields("dn", W.float(), _hessian(h).float(), h.mean(0))[0]
    b = ent.entropy_fields("dn", W.float(), _hessian(h).float(), h.mean(0))[0]
    assert a == b                                   # bit-for-bit, every field


def test_two_halves_combine_exactly():
    h, W = _rows(T=3000)
    a, b = h[:1700], h[1700:]
    parts = []
    for part in (a, b):
        m2, mu = ent.output_moments(W, _hessian(part), part.mean(0))
        parts.append((part.shape[0], m2, mu))
    n, m2, mu = ent.combine_moments(parts)
    assert n == 3000
    # exact algebra; the only error is output_moments' fp32 diagonal (~1e-7 relative)
    assert ent.entropy_ratio(m2, mu)["rho"] == pytest.approx(ent.direct_entropy_ratio(h @ W.t())["rho"], rel=1e-6)


# ------------------------------------------------------------------------------------------- the calibration sink
def _sink_feed(means):
    from experts4bit_qlora.engines.int4_experts import _ExpertHessianSink
    gu_p = torch.ones(4, dtype=torch.uint8)
    sink = _ExpertHessianSink({id(gu_p): 5}, layers=[5], hessian_device="cpu", means=means)
    g = torch.Generator().manual_seed(7)
    for _ in range(3):
        ids = torch.randint(0, 4, (37,), generator=g)
        x = torch.randn(37, 12, generator=g).to(torch.bfloat16)
        h = torch.randn(37, 6, generator=g).to(torch.bfloat16)
        sink(gu_p, ids, x, h)
        sink.__dict__.setdefault("_seen", []).append((ids, x, h))
    return sink


def test_sink_means_leave_the_hessians_bitwise_unchanged():
    pytest.importorskip("gptq_pack")
    plain, with_means = _sink_feed(False), _sink_feed(True)
    hp, hm = plain.hessians(), with_means.hessians()
    assert set(hp) == set(hm) == {5} and set(hp[5]) == set(hm[5])
    for e in hp[5]:
        assert torch.equal(hp[5][e][0], hm[5][e][0]) and torch.equal(hp[5][e][1], hm[5][e][1])
        assert hp[5][e][2] == hm[5][e][2]
    with pytest.raises(RuntimeError):
        plain.means()


def test_sink_means_are_the_exact_mean_of_the_routed_rows():
    pytest.importorskip("gptq_pack")
    sink = _sink_feed(True)
    ids = torch.cat([s[0] for s in sink._seen])
    x = torch.cat([s[1] for s in sink._seen]).to(torch.float64)
    h = torch.cat([s[2] for s in sink._seen]).to(torch.float64)
    got = sink.means()[5]
    for e in got:
        assert torch.allclose(got[e][0], x[ids == e].mean(0), rtol=0, atol=1e-12)
        assert torch.allclose(got[e][1], h[ids == e].mean(0), rtol=0, atol=1e-12)


def test_calibrate_fills_activation_means_and_returns_the_same_hessians(monkeypatch):
    pytest.importorskip("gptq_pack")
    from experts4bit_qlora.engines import hot_residency as hr
    from experts4bit_qlora.engines import int4_experts as ie

    class _W:
        h_gu_p = torch.ones(2, dtype=torch.uint8)

    w = _W()
    monkeypatch.setattr(ie, "_expert_layers", lambda *a, **k: (None, [(0, w)]))

    class _Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.p = torch.nn.Parameter(torch.zeros(1))
            self.config = types.SimpleNamespace()

        def forward(self, ids):
            g = torch.Generator().manual_seed(int(ids.sum()))
            sid = torch.randint(0, 3, (20,), generator=g)
            hr._CALIB_SINK(w.h_gu_p, sid, torch.randn(20, 8, generator=g), torch.randn(20, 4, generator=g))

    batches = [torch.full((1, 4), i) for i in range(3)]
    a = ie.calibrate_expert_hessians(_Model(), "unused", batches, layers_per_pass=1, device="cpu")
    means = {}
    b = ie.calibrate_expert_hessians(_Model(), "unused", batches, layers_per_pass=1, device="cpu",
                                     activation_means=means)
    assert set(a) == set(b) == set(means) == {0} and set(means[0]) == set(b[0])
    for e in a[0]:
        assert torch.equal(a[0][e][0], b[0][e][0]) and torch.equal(a[0][e][1], b[0][e][1]) and a[0][e][2] == b[0][e][2]
        assert means[0][e][0].shape == (8,) and means[0][e][1].shape == (4,)
    assert hr._CALIB_SINK is None                   # the tap is restored


# ------------------------------------------------------------------------------------------- the census row
def test_census_row_carries_entropy_without_changing_rel_act():
    h, W = _rows(T=800, K=64, C=32)
    W32, H = W.float(), _hessian(h).float()
    rows = h.shape[0]
    plain = res.census_row(W32, H, rows, min_rows=cen.NO_GPTQ, damp=0.01, dev="cpu")
    row = res.census_row(W32, H, rows, min_rows=cen.NO_GPTQ, damp=0.01, dev="cpu")
    row["entropy"] = ent.entropy_fields("dn", W32, H, h.mean(0))[0]      # what p65_census.main does
    assert row["rtn"] == plain["rtn"] and row["denom_act"] == plain["denom_act"]
    assert row["recipe_method"] == "rtn" and row["gptq"] is None      # min_rows above any count: no GPTQ solve
    assert row["entropy"]["rho"] is not None and 0.0 < row["entropy"]["rho"] <= 1.0
    # and rel_act is still P44's definition: sqrt(tr(D H D^T) / tr(W H W^T)) for the RTN pack
    from int4_pack_ref import dequant_int4_ref, pack_int4_b32
    p, s = pack_int4_b32(W32)
    D = dequant_int4_ref(p, s, *W32.shape) - W32
    want = math.sqrt(float(((D @ H) * D).sum()) / float(((W32 @ H) * W32).sum()))
    assert row["rtn"]["rel_act"] == pytest.approx(want, rel=1e-6)


def test_p44_census_script_is_untouched():
    """P65 imports P44's census row; it must not edit it. `expert_residuals.py` is still the bytes P44-a ran
    (bench/p44/staged-a.sha256). (`serve_stack.py` is not: P47-P53 appended arms to it after P44 ran. The functions
    P65 reuses from it -- calib_batches, build_served_model, apply_env, MODELS -- are unchanged since P44's pin,
    checked when P65 was registered; P65's own staged.sha256 pins the file as P65 stages it.)"""
    want = {name: sha for sha, name in (ln.split() for ln in (P44 / "staged-a.sha256").read_text().splitlines()
                                        if ln.strip() and not ln.startswith("#"))}
    assert hashlib.sha256((P44 / "expert_residuals.py").read_bytes()).hexdigest() == want["expert_residuals.py"]


def test_full_row_is_the_exact_combination_of_its_halves():
    h, W = _rows(T=2000, K=64, C=32)
    W32 = W.float()
    a, b = h[:1100], h[1100:]
    ra = res.census_row(W32, _hessian(a).float(), a.shape[0], min_rows=cen.NO_GPTQ, damp=0.01, dev="cpu")
    rb = res.census_row(W32, _hessian(b).float(), b.shape[0], min_rows=cen.NO_GPTQ, damp=0.01, dev="cpu")
    for r, hi in ((ra, 0), (rb, 1)):
        r.update({"layer": 0, "expert": 0, "role": "dn", "text": "wikitext", "half": hi})
    full = cen.combine_rows(ra, rb)
    one = res.census_row(W32, _hessian(h).float(), h.shape[0], min_rows=cen.NO_GPTQ, damp=0.01, dev="cpu")
    assert full["half"] == "full" and full["rows"] == 2000
    assert full["rtn"]["rel_act"] == pytest.approx(one["rtn"]["rel_act"], rel=1e-5)
    assert full["rtn"]["rel_frob"] == one["rtn"]["rel_frob"]


# ------------------------------------------------------------------------------------------- texts
class _Tok:
    def __call__(self, text, return_tensors=None):
        ids = torch.tensor([[ord(c) % 997 for c in text]])
        return types.SimpleNamespace(input_ids=ids)


def test_windows_is_calib_batches_rule(monkeypatch):
    docs = [f"document {i} " + "lorem ipsum " * (i % 17 + 3) for i in range(5000)]
    fake = types.ModuleType("datasets")
    fake.load_dataset = lambda *a, **k: {"text": docs}
    monkeypatch.setitem(sys.modules, "datasets", fake)
    import serve_stack
    want = serve_stack.calib_batches(_Tok(), 16, "c4")
    text = "\n\n".join(docs[:4000])[:6_000_000]
    got = cen.windows(_Tok()(text).input_ids[0], 16)
    assert len(got) == len(want) and all(torch.equal(x, y) for x, y in zip(got, want))


def test_c4val1_is_k8s_text(monkeypatch):
    seen = {}
    docs = [f"c4 doc {i} " * 50 for i in range(3000)]

    def load_dataset(repo, data_files=None, split=None):
        seen.update(repo=repo, data_files=data_files, split=split)
        return {"text": docs}

    fake = types.ModuleType("datasets")
    fake.load_dataset = load_dataset
    monkeypatch.setitem(sys.modules, "datasets", fake)
    assert cen.c4val1_text() == "\n\n".join(docs[:2000])
    assert seen == {"repo": "allenai/c4", "data_files": {"v": "en/c4-validation.00001-of-00008.json.gz"}, "split": "v"}
    k8 = (REPO / "bench" / "hybrid-g9" / "step_decomp.py").read_text()
    assert '"en/c4-validation.00001-of-00008.json.gz"' in k8 and 'ds["text"][:2000]' in k8


def test_halves_are_disjoint_and_cover_the_text():
    batches = [torch.full((4, 3), i) for i in range(8)]
    a, b = cen.split_halves(batches)
    assert [int(x[0, 0]) for x in a] == [0, 2, 4, 6] and [int(x[0, 0]) for x in b] == [1, 3, 5, 7]
    d = cen.batches_digest(a)
    assert d["windows"] == 16 and d["tokens"] == 48 and len(d["ids_sha256"]) == 64
