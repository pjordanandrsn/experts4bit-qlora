"""DQ11 MLP-adapter BUILD Phase 2's (b) part (bench/dq11/dq11_mlp_phase2.py, Amendment 5 sections 2-4), on CPU.

The fixture is Phase 1's: PEFT LoRA on real bitsandbytes NF4 ``Linear4bit`` bases, so PEFT dispatches its
``Linear4bit`` LoRA path (the delta is cast to bf16 and then added), and the engine fuses them."""
from __future__ import annotations

import itertools
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
import dq11_mlp_phase2 as p2  # noqa: E402

from experts4bit_qlora.engines import dense_lora_fuse as dlf  # noqa: E402


def _nf4(lin):
    q = bnb.nn.Linear4bit(lin.in_features, lin.out_features, bias=False, compute_dtype=torch.bfloat16, quant_type="nf4")
    q.weight = bnb.nn.Params4bit(lin.weight.data.clone(), requires_grad=False, quant_type="nf4")
    return q.to("cpu")


def _model(layers=1, seed=0):
    torch.manual_seed(seed)
    cfg = MistralConfig(hidden_size=64, intermediate_size=128, hidden_act="silu")
    m = nn.ModuleDict({"layers": nn.ModuleList(nn.ModuleDict({"mlp": MistralMLP(cfg)}) for _ in range(layers))})
    m = m.to(torch.bfloat16)
    for layer in m["layers"]:
        for p in p1.PROJECTIONS:
            setattr(layer["mlp"], p, _nf4(getattr(layer["mlp"], p)))
    m.is_loaded_in_4bit = True
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
def mlp():
    m = _model()
    p1.set_b(_mlps(m), 0.05)
    return _mlps(m)[0]


GRADIENT_ONLY = [d for d in p2.single_levels() if d[0][0] in ("R9", "R3", "R4", "R5")]


def test_today_is_peft_bit_for_bit_directly_and_through_the_function(mlp):
    x, gy = _xg()
    truth = p1.peft_read(mlp, x, gy)
    for via in (False, True):
        ref = p2.ref_read(mlp, x, gy, via_function=via)
        assert [k for k in p1.TENSORS if not torch.equal(truth[k], ref[k])] == [], via


def test_the_leveled_reference_reproduces_phase_1s_own_redraws_bit_for_bit(mlp):
    x, gy = _xg(tokens=32)
    for r1, d in p2.PHASE1_EQUAL.items():
        old, new = p1.ref_read(mlp, x, gy, {r1}), p2.ref_read(mlp, x, gy, d)
        assert [k for k in p1.TENSORS if not torch.equal(old[k], new[k])] == [], r1


def test_what_each_family_can_move_follows_the_ops_it_splits(mlp):
    """Structure, not the BLAS's luck: a gradient-only family never moves a forward output, R3 (weight gradients only)
    moves only dA/dB, and R5 (dx's sum order) moves only dx."""
    x, gy = _xg(tokens=32)
    today = p2.ref_read(mlp, x, gy)
    for d in GRADIENT_ONLY:
        r = p2.ref_read(mlp, x, gy, d)
        assert all(torch.equal(r[k], today[k]) for k in p2.OUTPUTS), p2.name(d)
        if d[0][0] == "R3":
            assert all(torch.equal(r[k], today[k]) for k in ("dx", "dh") + p2.OUTPUTS), p2.name(d)
        if d[0][0] == "R5":
            assert [k for k in p1.TENSORS if not torch.equal(r[k], today[k])] in ([], ["dx"]), p2.name(d)


def test_every_split_keeps_each_ops_dtype_and_no_cast_moves(mlp):
    """Every leveled draw returns the same dtypes as today's path: bf16 outputs, dx and dh, FP32 adapter gradients."""
    x, gy = _xg()
    today = p2.ref_read(mlp, x, gy)
    for d in p2.single_levels():
        r = p2.ref_read(mlp, x, gy, d)
        assert {k: r[k].dtype for k in p1.TENSORS} == {k: today[k].dtype for k in p1.TENSORS}, p2.name(d)
    assert today["out_gate"].dtype == today["dx"].dtype == today["dh"].dtype == torch.bfloat16
    assert today["dA_gate"].dtype == today["dB_down"].dtype == torch.float32


def test_chunked_linear_sums_equal_chunks_left_to_right_and_refuses_an_uneven_split():
    g = torch.Generator().manual_seed(1)
    inp, w = torch.randn(5, 16, generator=g), torch.randn(3, 16, generator=g)
    want = ((torch.nn.functional.linear(inp[:, :4], w[:, :4]) + torch.nn.functional.linear(inp[:, 4:8], w[:, 4:8]))
            + torch.nn.functional.linear(inp[:, 8:12], w[:, 8:12])) + torch.nn.functional.linear(inp[:, 12:], w[:, 12:])
    assert torch.equal(p2._chunked_linear(inp, w, 4), want)
    assert torch.equal(p2._chunked_linear(inp, w, 1), torch.nn.functional.linear(inp, w))
    with pytest.raises(ValueError, match="equal chunks"):
        p2._chunked_linear(inp, w, 3)


def test_the_stratified_draws_hold_every_grid_never_r2_and_are_fixed():
    draws = p2.stratified()
    assert draws == p2.stratified() and draws == sorted(set(draws)) and () not in draws
    have = set(draws)
    for grid in p2.GRIDS:
        for combo in itertools.product(*(p2.FAMILIES[f] for f in grid)):
            d = p2.draw(**dict(zip(grid, combo)))
            assert d == () or d in have, (grid, d)
    assert all(f != "R2" for d in draws for f, _c in d)
    grids = p2.stratified(seeded=0)
    assert len(draws) == len(grids) + p2.SEEDED and set(grids) <= have
    assert len([d for d in grids if {f for f, _ in d} <= {"R1", "R6"}]) == 4 * 3 - 1        # the outputs' grid less today
    assert len([d for d in grids if {f for f, _ in d} <= {"R9", "R4"}]) == 4 * 3 - 1        # dh's
    with pytest.raises(ValueError):
        p2.draw(R2=2)


def test_the_digest_is_the_bytes_dtype_and_shape():
    a = torch.tensor([1.0, 2.0], dtype=torch.bfloat16)
    b = a.clone()
    assert p2.digest(a) == p2.digest(b)
    b.view(torch.int16)[0] += 1                                                  # one ULP
    assert p2.digest(a) != p2.digest(b)
    assert p2.digest(a) != p2.digest(a.float()) and p2.digest(a) != p2.digest(a.reshape(2, 1))


def _paths(errs):
    return {f"h{i:02d}": {"err": {"rel_rms": e, "max_abs": e}} for i, e in enumerate(errs)}


def test_judge_unjudged_below_nine_calibrates_only_at_nine_and_bars_at_the_margin():
    fused = lambda e: {"rel_rms": e, "max_abs": e}                              # noqa: E731
    eight = p2.judge(_paths([1.0] * 7 + [5.0]), fused(9.0))                      # an outlier, but k = 8: never calibrated
    assert eight["status"] == p2.UNJUDGED and eight["k"] == 8 and eight["calibration_fails"] == []
    assert eight["false_reject_bound"] == 1 / 9
    nine = [1.0, 1.01, 1.02, 0.99, 1.0, 1.03, 0.98, 1.0, 1.04]
    ok = p2.judge(_paths(nine), fused(1.04 * 1.05))
    assert ok["status"] == "PASS" and ok["k"] == 9 and ok["band_rel_rms"] == 1.04
    assert p2.judge(_paths(nine), fused(1.04 * 1.0501))["status"] == "FAIL"
    outlier = p2.judge(_paths(nine[:-1] + [1.2]), fused(0.5))                    # 1.2 > 1.05 x 1.03: calibration fails
    assert outlier["status"] == p2.INDETERMINATE and len(outlier["calibration_fails"]) == 1


def test_the_outcome_lists_each_status_and_rank1():
    rows = {0: {"dx": {"status": "PASS", "e_min": 0.01, "k": 12, "a": {"pass": False}},
                "dh": {"status": p2.UNJUDGED, "e_min": 0.3, "k": 8, "a": {"pass": True}}},
            15: {"dx": {"status": "FAIL", "e_min": 0.01, "k": 9, "a": {"pass": True}},
                 "dh": {"status": p2.INDETERMINATE, "e_min": 0.01, "k": 10, "a": {"pass": True}}}}
    out = p2.outcome(rows, rank=16)
    assert out["fail"] == ["15:dx"] and out["unjudged_at_k"] == ["0:dh"] and out["calibration_indeterminate"] == ["15:dh"]
    assert out["passed"] == ["0:dx"] and out["rank1_unresolved"] == ["0:dh"] and out["accuracy_over_reorder"] == ["0:dx"]
    assert out["k_min_seen"] == 8


def test_a_level_that_lands_on_a_tensor_leaves_only_that_tensors_paths(mlp):
    """If the candidate were R5 (dx's other order), R5 lands on dx and on nothing else, and every R5 draw leaves U_dx
    but stays in every other tensor's U."""
    x, gy = _xg(tokens=32)
    today = p2.ref_read(mlp, x, gy)
    fused = p2.ref_read(mlp, x, gy, p2.draw(R5=True))
    if torch.equal(fused["dx"], today["dx"]):
        pytest.skip("R5 does not move dx on this CPU fixture")
    rows = p2.landing(mlp, x, gy, today, fused)
    assert set(rows) == {p2.name(d) for d in p2.single_levels()} and all(set(r) == set(p1.TENSORS) for r in rows.values())
    assert all({"to_fused", "to_today", "lands"} <= set(v) for r in rows.values() for v in r.values())    # F2
    landed = p2.landed_by_tensor({0: rows})
    assert landed["dx"] == ["R5"] and all(v == [] for k, v in landed.items() if k != "dx")
    ref = p1.fp64_read(mlp, x, gy)
    draws = [p2.draw(R5=True), p2.draw(R5=True, R4=2), p2.draw(R4=2)]
    streamed = p2.paths_streamed(today, ref, lambda d: p2.ref_read(mlp, x, gy, d), draws, landed)
    assert streamed["dx"]["excluded_draws"] == 2 and streamed["dh"]["excluded_draws"] == 0
    assert all("R5" not in n for v in streamed["dx"]["paths"].values() for n in v["draws"])


def test_distinct_paths_count_bytes_not_draws(mlp):
    """Gradient-only draws leave the outputs bit for bit as today's: they are one path, not many."""
    x, gy = _xg(tokens=32)
    today = p2.ref_read(mlp, x, gy)
    ref = p1.fp64_read(mlp, x, gy)
    streamed = p2.paths_streamed(today, ref, lambda d: p2.ref_read(mlp, x, gy, d), GRADIENT_ONLY,
                                 {k: [] for k in p1.TENSORS})
    for k in p2.OUTPUTS:
        assert len(streamed[k]["paths"]) == 1
        (only,) = streamed[k]["paths"].values()
        assert only["draws"][0] == "today" and len(only["draws"]) == 1 + len(GRADIENT_ONLY)


def test_read_layers_end_to_end_on_the_real_engine():
    m = _model(layers=1)
    mlps = dict(enumerate(_mlps(m)))
    x, gy = _xg(tokens=16)
    rec = p2.read_layers(mlps, {0: x}, {0: gy}, enable=lambda: dlf.enable_dense_mlp_lora_fuse(m),
                         disable=lambda: dlf.disable_dense_mlp_lora_fuse(m), stats=dlf.DENSE_MLP_LORA_STATS,
                         counter=lambda: p1.lora_forward_counter(m), adapters=3, draws=p2.stratified(seeded=4))
    assert "void" not in rec, rec.get("void")
    assert rec["engagement"]["ok"] and rec["engagement"]["adapters"] == [3] and rec["engagement"]["mlp_lora_forwards"] == 0
    rows = rec["layers"][0]
    assert set(rows) == set(p1.TENSORS)
    for k, v in rows.items():
        assert v["status"] in ("PASS", "FAIL", p2.UNJUDGED, p2.INDETERMINATE), k
        assert v["k"] == len(v["paths"]) and v["false_reject_bound"] == 1 / (v["k"] + 1)
        assert v["ref_norm"] > 0 and "fused_to_today" in v["a"]                                       # F3, (a)
        assert (v["status"] == p2.UNJUDGED) == (v["k"] < p2.K_MIN)
    assert set(rec["landing"][0]) == {p2.name(d) for d in p2.single_levels()}                          # F2
    assert rec["outcome"]["k_min_seen"] == min(v["k"] for v in rows.values())
    assert not any(hasattr(mlp, "_e4b_dense_lora") for mlp in _mlps(m))


def test_read_layers_voids_on_a_reference_that_is_not_phase_1s(monkeypatch):
    m = _model(layers=1)
    x, gy = _xg(tokens=16)
    real = p2.ref_read

    def off_by_one(mlp, x, gy, d=(), **kw):
        out = real(mlp, x, gy, d, **kw)
        if d == p2.draw(R4=2):
            out["dx"] = out["dx"] * 2
        return out
    monkeypatch.setattr(p2, "ref_read", off_by_one)
    rec = p2.read_layers(dict(enumerate(_mlps(m))), {0: x}, {0: gy}, enable=lambda: dlf.enable_dense_mlp_lora_fuse(m),
                         disable=lambda: dlf.disable_dense_mlp_lora_fuse(m), stats=dlf.DENSE_MLP_LORA_STATS,
                         counter=lambda: p1.lora_forward_counter(m), adapters=3, draws=[])
    assert rec["void"].startswith("REF_MISMATCH: 0:") and "R4:dx" in rec["void"]
