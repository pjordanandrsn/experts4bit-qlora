"""DQ11 MLP-adapter BUILD Phase 1's per-module reads (bench/dq11/dq11_mlp_phase1.py), on CPU.

The MLPs are PEFT LoRA on real bitsandbytes NF4 ``Linear4bit`` bases (bnb quantizes on CPU), so PEFT dispatches its
``Linear4bit`` LoRA path, the fixture's: the delta is cast to bf16 and then added. The engine fuses them without a
``base_ok`` override."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn

peft = pytest.importorskip("peft")
bnb = pytest.importorskip("bitsandbytes")
from transformers import MistralConfig  # noqa: E402
from transformers.models.mistral.modeling_mistral import MistralMLP  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "bench" / "dq11"))
import dq11_mlp_phase1 as p1  # noqa: E402

from experts4bit_qlora.engines import dense_lora_fuse as dlf  # noqa: E402


def _nf4(lin):
    q = bnb.nn.Linear4bit(lin.in_features, lin.out_features, bias=False, compute_dtype=torch.bfloat16, quant_type="nf4")
    q.weight = bnb.nn.Params4bit(lin.weight.data.clone(), requires_grad=False, quant_type="nf4")
    return q.to("cpu")


def _model(layers=2, seed=0):
    torch.manual_seed(seed)
    cfg = MistralConfig(hidden_size=64, intermediate_size=128, hidden_act="silu")
    m = nn.ModuleDict({"layers": nn.ModuleList(nn.ModuleDict({"mlp": MistralMLP(cfg)}) for _ in range(layers))})
    m = m.to(torch.bfloat16)
    for layer in m["layers"]:
        for p in p1.PROJECTIONS:
            setattr(layer["mlp"], p, _nf4(getattr(layer["mlp"], p)))
    m.is_loaded_in_4bit = True                                              # PEFT dispatches its bnb LoRA class only then
    m = peft.inject_adapter_in_model(peft.LoraConfig(r=4, lora_alpha=8, target_modules=list(p1.PROJECTIONS)), m)
    for n, p in m.named_parameters():
        p.requires_grad_("lora_" in n)
        if "lora_" in n:
            p.data = p.data.float()
    return m


def _mlps(m):
    return [layer["mlp"] for layer in m["layers"]]


def _xg(seed=7, tokens=16):
    g = torch.Generator().manual_seed(seed)
    return (torch.randn(1, tokens, 64, generator=g).to(torch.bfloat16),
            torch.randn(1, tokens, 64, generator=g).to(torch.bfloat16))


@pytest.fixture
def model():
    m = _model()
    assert type(_mlps(m)[0].gate_proj).__module__.endswith("lora.bnb")    # the fixture's PEFT path, not lora.Linear
    p1.set_b(_mlps(m), 0.05)
    return m


def test_the_reference_with_no_redraw_is_peft_bit_for_bit(model):
    x, gy = _xg()
    for mlp in _mlps(model):
        truth, ref = p1.peft_read(mlp, x, gy), p1.ref_read(mlp, x, gy)
        assert set(truth) == set(ref) == set(p1.TENSORS)
        bad = [k for k in p1.TENSORS if not torch.equal(truth[k], ref[k])]
        assert bad == [], bad


def test_thirty_one_draws_and_what_each_redraw_moves(model):
    sets = p1.draw_sets()
    assert len(sets) == 31 and len(set(sets)) == 31 and all(sets)
    x, gy = _xg(tokens=64)
    mlp = _mlps(model)[0]
    truth = p1.peft_read(mlp, x, gy)
    single = {r: p1.ref_read(mlp, x, gy, {r}) for r in p1.DRAWS}
    moved = {r: {k for k in p1.TENSORS if not torch.equal(single[r][k], truth[k])} for r in p1.DRAWS}
    assert {"out_gate", "out_up", "out_down"} <= moved["R2"]                # the cast after an FP32 add
    assert set(p1.TENSORS[5:]) <= moved["R3"]                              # the token-split gradient GEMMs
    assert "dx" in moved["R5"]                                             # the LoRA terms in the other order
    assert "dh" not in set().union(*moved.values())                        # dh: no redraw moves it here, so bitwise
    fl = p1.floor(truth, {d: p1.ref_read(mlp, x, gy, d) for d in sets})
    assert fl["dh"] == {"max_abs": 0.0, "rel_rms": 0.0, "moved_by": []} and p1.coverage(fl)["dh"] == []
    assert set(p1.coverage(fl)["dx"]) >= {"R5"}


def test_r4_and_r3_change_the_fp32_gemms_they_name():
    g = torch.Generator().manual_seed(11)
    xf, w = torch.randn(64, 256, generator=g), torch.randn(16, 256, generator=g)
    gt = torch.randn(64, 16, generator=g)
    grads = {}
    for flags in ((False, False, False), (False, True, False), (False, False, True)):
        i_, w_ = xf.clone().requires_grad_(True), w.clone().requires_grad_(True)
        p1._LoraLinear.apply(i_, w_, *flags).backward(gt)
        grads[flags] = (i_.grad, w_.grad)
    plain, tokens, rank = grads[(False, False, False)], grads[(False, True, False)], grads[(False, False, True)]
    assert torch.equal(tokens[0], plain[0]) and not torch.equal(tokens[1], plain[1])      # R3: the weight gradient only
    assert torch.equal(rank[1], plain[1]) and not torch.equal(rank[0], plain[0])          # R4: the input gradient only
    torch.testing.assert_close(rank[0], plain[0])
    torch.testing.assert_close(tokens[1], plain[1])


def test_sigma_is_the_smallest_qualifying_and_the_outcome_names_what_it_must():
    low, ok = {k: 0.005 for k in p1.TENSORS}, {k: 0.02 for k in p1.TENSORS}
    assert p1.choose_sigma({0.01: [low, ok], 0.02: [ok, ok], 0.05: [ok, ok]}) == 0.02
    assert p1.choose_sigma({s: [low] for s in p1.SIGMAS}) is None                       # VOID_LORA_SHARE
    row = {"pass": True, "e_min": 0.01}
    rows = {0: {k: dict(row) for k in p1.TENSORS}, 15: {k: dict(row) for k in p1.TENSORS}}
    assert p1.outcome(rows, 16) == "PHASE1_PASS"
    rows[15]["dx"]["e_min"] = 0.2
    assert p1.outcome(rows, 16) == "PHASE1_PASS, RANK1_UNRESOLVED: 15:dx"               # never a silent PASS
    rows[0]["dB_up"]["pass"] = False
    assert p1.outcome(rows, 16) == "PHASE1_FAIL: 0:dB_up"
    z = {k: torch.ones(2) for k in p1.TENSORS}
    z["dA_up"] = torch.zeros(2)
    z["dB_gate"] = torch.tensor([float("nan"), 1.0])
    assert p1.vacuous(z) == ["dB_gate", "dA_up"]


def test_the_fused_read_is_the_engine_and_a_wrong_fusion_fails_the_bar(model):
    x, gy = _xg()
    assert dlf.enable_dense_mlp_lora_fuse(model) == 6                      # real NF4 bases: no base_ok override
    mlp = _mlps(model)[1]
    fused = p1.fused_read(mlp, x, gy)
    torch.testing.assert_close(fused["out_down"].float(), mlp(x).float(), rtol=0, atol=0)
    dlf.disable_dense_mlp_lora_fuse(model)
    truth = p1.peft_read(mlp, x, gy)
    fl = p1.floor(truth, {d: p1.ref_read(mlp, x, gy, d) for d in p1.draw_sets()})
    terms = p1.adapter_terms(mlp, x, gy)
    rows = p1.bar(truth, fused, fl, terms)
    assert set(rows) == set(p1.TENSORS)
    for k in p1.TENSORS:
        torch.testing.assert_close(fused[k].float(), truth[k].float(), rtol=3e-2, atol=3e-2)
    wrong = dict(fused, dB_up=fused["dB_up"] * 1.01)                       # a 1 % error in one adapter gradient
    assert p1.bar(truth, wrong, fl, terms)["dB_up"]["pass"] is False


def test_shares_are_zero_at_peft_init_and_set_b_is_seeded(model):
    x, gy = _xg()
    fresh = _model()
    sh0 = p1.shares(p1.adapter_terms(_mlps(fresh)[0], x, gy))
    assert sh0["out_gate"] == 0 and sh0["dx"] == 0 and sh0["dA_gate"] == 0  # B = 0: the bar would test nothing
    sh = p1.shares(p1.adapter_terms(_mlps(model)[0], x, gy))
    assert all(v > 0 for v in sh.values()) and sh["dA_gate"] == sh["dB_down"] == 1.0
    again = _model()
    p1.set_b(_mlps(again), 0.05)
    assert all(torch.equal(a, b) for a, b in zip(p1._params(_mlps(model)[0]), p1._params(_mlps(again)[0])))


def test_the_lora_forward_counter_sees_peft_calls_not_fused_ones_and_restores_the_class(model):
    x, _gy = _xg()
    cls = type(_mlps(model)[0].gate_proj)
    before = cls.__dict__.get("forward")
    with p1.lora_forward_counter(model) as unfused:
        _mlps(model)[0](x)
    assert unfused == {"layers.0.mlp.gate_proj": 1, "layers.0.mlp.up_proj": 1, "layers.0.mlp.down_proj": 1}
    assert dlf.enable_dense_mlp_lora_fuse(model) == 6
    with p1.lora_forward_counter(model) as fused:
        _mlps(model)[0](x)
    assert fused == {} and dlf.DENSE_MLP_LORA_STATS.get("fallback", 0) == 0     # a class wrapper is not a hook
    assert cls.__dict__.get("forward") is before


FEW = [frozenset({"R2"}), frozenset({"R3"}), frozenset({"R5"})]     # orchestration tests: the coverage test runs all 31


def _read_layers(m, xg, **kw):
    mlps = dict(enumerate(_mlps(m)))
    xs = {i: xg[0] for i in mlps}
    gys = {i: xg[1] for i in mlps}
    return p1.read_layers(mlps, xs, gys, enable=lambda: dlf.enable_dense_mlp_lora_fuse(m),
                          disable=lambda: dlf.disable_dense_mlp_lora_fuse(m), stats=dlf.DENSE_MLP_LORA_STATS, **kw)


def test_read_layers_runs_the_ladder_the_self_checks_and_the_bar_on_the_real_engine(monkeypatch):
    monkeypatch.setattr(p1, "draw_sets", lambda: FEW)
    m = _model(layers=1)
    rec = _read_layers(m, _xg(tokens=16), adapters=3)
    assert rec["sigma"] in p1.SIGMAS and rec["ladder"][rec["sigma"]]
    assert all(min(sh.values()) >= p1.SHARE_MIN for sh in rec["ladder"][rec["sigma"]].values())
    assert rec["outcome"].startswith(("PHASE1_PASS", "PHASE1_FAIL")), rec["outcome"]
    for layer in rec["layers"].values():
        assert set(layer["rows"]) == set(p1.TENSORS)
        assert layer["rows"]["dh"]["bitwise_required"] == {"differing": 0, "sign_flips": 0, "max_ulp": 0}   # PEFT's own ops
    assert not any(hasattr(mlp, "_e4b_dense_lora") for mlp in _mlps(m))                    # disabled after each read


def test_read_layers_names_engage_count_and_ref_mismatch(monkeypatch):
    monkeypatch.setattr(p1, "draw_sets", lambda: FEW)
    assert _read_layers(_model(layers=1), _xg(), adapters=96)["outcome"].startswith("ENGAGE_COUNT: 3")
    monkeypatch.setattr(p1, "DX_ORDER", ("gate_base", "gate_lora", "up_lora", "up_base"))
    assert _read_layers(_model(layers=1), _xg(tokens=64), adapters=3)["outcome"].startswith("REF_MISMATCH")


def test_the_streamed_floor_equals_the_floor_and_the_bit_diagnostics_count(model, monkeypatch):
    monkeypatch.setattr(p1, "draw_sets", lambda: FEW)
    x, gy = _xg()
    mlp = _mlps(model)[0]
    truth = p1.peft_read(mlp, x, gy)
    reads = {d: p1.ref_read(mlp, x, gy, d) for d in p1.draw_sets()}
    assert p1.floor_streamed(truth, reads.__getitem__) == p1.floor(truth, reads)
    a = torch.tensor([1.0, 2.0, 3.0], dtype=torch.bfloat16)
    b = (a.view(torch.int16) + torch.tensor([0, 1, 0], dtype=torch.int16)).view(torch.bfloat16)   # one bf16 step up
    assert p1.bit_diagnostics(a, a) == {"differing": 0, "sign_flips": 0, "max_ulp": 0}
    assert p1.bit_diagnostics(b, a) == {"differing": 1, "sign_flips": 0, "max_ulp": 1}
    c = a.clone()
    c[0] = -1.0                                                             # a sign flip is counted, never measured
    assert p1.bit_diagnostics(c, a) == {"differing": 1, "sign_flips": 1, "max_ulp": 0}
    assert p1.bit_diagnostics(torch.stack([c[0], b[1], a[2]]), a) == {"differing": 2, "sign_flips": 1, "max_ulp": 1}


# --- Amendment 4 ------------------------------------------------------------------------------------------------------

def test_the_post_a_set_drops_every_draw_with_the_epilogue_redraw():
    post = p1.post_a_draws({"R2"})
    assert len(post) == 15 and not any("R2" in d for d in post) and len(p1.post_a_draws({"R1", "R2", "R3", "R4", "R5"})) == 0


def test_r2_lands_on_the_fused_read_and_the_fp64_read_is_the_same_function(model):
    x, gy = _xg(tokens=32)
    mlp = _mlps(model)[0]
    today = p1.peft_read(mlp, x, gy)
    assert dlf.enable_dense_mlp_lora_fuse(model) == 6
    fused = p1.fused_read(mlp, x, gy)
    dlf.disable_dense_mlp_lora_fuse(model)
    assert "R2" in p1.landed(mlp, x, gy, today, fused)                       # the epilogue's own arithmetic
    ref = p1.fp64_read(mlp, x, gy)
    assert all(ref[k].dtype == torch.float64 for k in p1.TENSORS)
    for k in p1.TENSORS:                                                     # today's path is the same function, rounded
        assert p1.error(today[k], ref[k])["rel_rms"] < 5e-2, k


def test_calibration_and_every_outcome_branch():
    e = lambda v: {k: {"rel_rms": v, "max_abs": v} for k in p1.TENSORS}  # noqa: E731
    errs = {0: {"today": e(1.0), "R1": e(1.01), "R3": e(0.99)}}
    assert p1.calibrate(errs) == []
    errs[0]["R4"] = e(1.2)                                                    # one neutral path 19 % above the rest
    assert p1.calibrate(errs) and all(f.endswith(":R4") for f in p1.calibrate(errs))
    row = lambda ok, e_min=0.01: {"pass": ok, "e_min": e_min}  # noqa: E731
    good = {0: {k: row(True) for k in p1.TENSORS}}
    assert p1.outcome_a4(good, good, 16, [], 16) == "PHASE1_PASS"
    a_fail = {0: dict(good[0], out_gate=row(False))}
    assert p1.outcome_a4(a_fail, good, 16, [], 16) == "PHASE1_PASS, ACCURACY_OVER_REORDER: 0:out_gate"
    assert p1.outcome_a4({}, good, 16, [], 16).startswith("PHASE1_PASS, ACCURACY_OVER_REORDER: 0:*")  # NO_FLOOR
    b_fail = {0: dict(good[0], dx=row(False))}
    assert p1.outcome_a4(good, b_fail, 16, [], 16) == "PHASE1_FAIL: 0:dx"
    assert p1.outcome_a4(good, b_fail, 16, ["0:dx:R4"], 16).startswith("INDETERMINATE: calibration")  # never a FAIL
    assert p1.outcome_a4(good, good, 16, [], 1).startswith("INDETERMINATE: |U| = 1")
    assert p1.outcome_a4(good, good, 16, [], 2).startswith("INDETERMINATE: |U| = 2")       # 4a 1: |U| <= 2
    assert p1.outcome_a4(good, good, 16, [], 3) == "PHASE1_PASS"
    assert p1.outcome_a4(good, b_fail, 16, [], 8).startswith("INDETERMINATE: bar too weak at |U| = 8")  # 4a 3
    assert p1.outcome_a4(good, b_fail, 16, [], 9) == "PHASE1_FAIL: 0:dx"                  # bound 1/10: a FAIL
    hidden = {0: dict(good[0], dh=row(True, 0.2))}
    assert p1.outcome_a4(good, hidden, 16, [], 16) == "PHASE1_PASS, RANK1_UNRESOLVED: 0:dh"


def test_read_layers_a4_end_to_end_on_the_real_engine(monkeypatch):
    monkeypatch.setattr(p1, "draw_sets", lambda: FEW + [frozenset({"R1", "R3"})])
    m = _model(layers=1)
    mlps = dict(enumerate(_mlps(m)))
    x, gy = _xg(tokens=16)
    rec = p1.read_layers_a4(mlps, {0: x}, {0: gy}, enable=lambda: dlf.enable_dense_mlp_lora_fuse(m),
                            disable=lambda: dlf.disable_dense_mlp_lora_fuse(m), stats=dlf.DENSE_MLP_LORA_STATS, adapters=3)
    assert "R2" in rec["excluded"] and rec["draws"] == len([d for d in p1.draw_sets() if not (d & set(rec["excluded"]))])
    assert rec["outcome"].startswith(("PHASE1_PASS", "PHASE1_FAIL", "INDETERMINATE")), rec["outcome"]
    layer = rec["layers"][0]
    assert set(layer["b"]) == set(p1.TENSORS) and "today" in layer["errors"]
    assert all(r["margin"] == p1.MARGIN_B for r in layer["b"].values())
    assert not any(hasattr(mlp, "_e4b_dense_lora") for mlp in _mlps(m))


def test_r2_leaves_the_floor_by_its_arithmetic_even_when_nothing_lands(monkeypatch):
    monkeypatch.setattr(p1, "draw_sets", lambda: FEW)
    monkeypatch.setattr(p1, "landed", lambda *a: [])                          # no redraw lands, mechanically
    m = _model(layers=1)
    x, gy = _xg(tokens=16)
    rec = p1.read_layers_a4(dict(enumerate(_mlps(m))), {0: x}, {0: gy}, enable=lambda: dlf.enable_dense_mlp_lora_fuse(m),
                            disable=lambda: dlf.disable_dense_mlp_lora_fuse(m), stats=dlf.DENSE_MLP_LORA_STATS, adapters=3)
    assert rec["excluded"] == ["R2"] and rec["epilogue_landed"] is False and rec["draws"] == 2      # FEW less {R2}


def test_bar_b_holds_the_fused_error_to_the_band_times_one_plus_the_margin():
    ref = {k: torch.ones(1000, dtype=torch.float64) for k in p1.TENSORS}
    errs = {"today": {k: {"rel_rms": 1e-3, "max_abs": 1e-3} for k in p1.TENSORS}}
    terms = {k: {"adapter_norm": 1.0, "norm": 1.0} for k in p1.TENSORS}
    inside = {k: torch.full((1000,), 1 + 1.04e-3) for k in p1.TENSORS}       # 4 % over the band: inside m = 0.05
    outside = {k: torch.full((1000,), 1 + 1.06e-3) for k in p1.TENSORS}      # 6 % over: outside
    assert all(r["pass"] for r in p1.bar_b(ref, inside, errs, terms).values())
    assert not any(r["pass"] for r in p1.bar_b(ref, outside, errs, terms).values())


def test_the_fp64_premise_refuses_where_the_reference_would_not_be_the_base_arithmetic(model):
    mlp = _mlps(model)[0]
    assert p1.fp64_premise(mlp, 2048, torch.bfloat16, cuda=True) == []           # the fixture: 2048 rows, bf16
    assert p1.fp64_premise(mlp, 16, torch.bfloat16, cuda=False) == []            # CPU: no fused 4-bit kernel
    small = p1.fp64_premise(mlp, 1536, torch.bfloat16, cuda=True)
    assert len(small) == 3 and all("fused 4-bit GEMM" in w for w in small)
    assert any("dequant dtype" in w for w in p1.fp64_premise(mlp, 2048, torch.float16, cuda=True))
    mlp.up_proj.base_layer.bias = torch.nn.Parameter(torch.zeros(128))
    assert any("up_proj: the base has a bias" in w for w in p1.fp64_premise(mlp, 2048, torch.bfloat16, cuda=True))
