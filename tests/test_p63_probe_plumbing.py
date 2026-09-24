"""Lane P63's capture plumbing on a real (tiny, random) HF Qwen3-MoE on CPU (bench/p63/p63_probe.py).

The GPU probe compares a token's T = 1 control against the same token inside a verify window and a prefill. Its
reading is only as good as its alignment: row j of a window at p0 must be token p0 + j, a verify must not consume the
control's cache snapshot, the hooks must see every site of every layer, and a module replayed at n = 1 must reproduce
the output recorded during the control. These run the probe's own functions on CPU; nothing here needs a kernel.
"""
import importlib.util
import pathlib

import pytest
import torch

pytest.importorskip("transformers")
from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeConfig, Qwen3MoeForCausalLM  # noqa: E402

REPO = pathlib.Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("p63_probe", REPO / "bench" / "p63" / "p63_probe.py")
P = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(P)
C = P.C

L, W, WINDOWS = 24, 5, (8, 16)


@pytest.fixture(scope="module")
def model():
    cfg = Qwen3MoeConfig(hidden_size=64, intermediate_size=128, moe_intermediate_size=32, num_experts=8,
                         num_experts_per_tok=2, num_hidden_layers=3, num_attention_heads=4, num_key_value_heads=2,
                         head_dim=16, vocab_size=128, max_position_embeddings=128)
    cfg._attn_implementation = "eager"
    torch.manual_seed(0)
    return Qwen3MoeForCausalLM(cfg).to(torch.bfloat16).eval()


@pytest.fixture(scope="module")
def ids():
    return torch.randint(0, 128, (1, L), generator=torch.Generator().manual_seed(7))


def _eq(a, b):
    return all(bool(C.bit_equal(a[s], b[s]).all()) for s in C.SITES)


def test_every_site_of_every_layer_is_captured_and_the_control_repeats_bit_for_bit(model, ids):
    cap = P.SiteCapture(model)
    try:
        with torch.no_grad():
            lg, snaps = P.run_control_hf(model, cap, ids, L, set(WINDOWS))
            ctrl = cap.take()
            lg2, _ = P.run_control_hf(model, cap, ids, L, set())
            again = cap.take()
    finally:
        cap.remove()
    assert set(snaps) == set(WINDOWS)
    for s in C.SITES:
        assert ctrl[s].shape == (L, 3, 64), (s, ctrl[s].shape)
    assert ctrl["final_norm"].shape == (L, 64) and lg.shape == (L, 128)
    assert _eq(ctrl, again) and bool(C.bit_equal(lg, lg2).all())


def test_a_verify_window_aligns_token_by_token_and_does_not_consume_the_snapshot(model, ids):
    cap = P.SiteCapture(model)
    try:
        with torch.no_grad():
            _lg, snaps = P.run_control_hf(model, cap, ids, L, set(WINDOWS))
            ctrl = cap.take()
            for p0 in WINDOWS:
                P.run_verify_hf(model, cap, ids, p0, W, snaps[p0])
                v1 = cap.take()
                P.run_verify_hf(model, cap, ids, p0, W, snaps[p0])
                v2 = cap.take()
                assert _eq(v1, v2), "a verify changed the snapshot it started from"
                pidx = torch.arange(p0, p0 + W)
                site, _fin = P._cmp_mode(ctrl, v1, pidx)
                assert float(site["rel_l2"].max()) < 0.05          # same tokens: rounding-level differences at most
                shifted, _ = P._cmp_mode(ctrl, v1, pidx + 1)       # an off-by-one pairing is not rounding-level
                assert float(shifted["rel_l2"][:, 0, 0].max()) > 0.2
            P.run_prefill_hf(model, cap, ids, L)
            pre = cap.take()
    finally:
        cap.remove()
    site, _fin = P._cmp_mode(ctrl, pre, torch.arange(L))
    assert float(site["rel_l2"].max()) < 0.05


def test_a_planted_difference_is_found_at_its_layer_and_site(model, ids):
    """A second control whose layer-1 experts output is nudged from token 10 on: tokens 0..9 read exact, every later
    token's first difference is (1, mlp_out) -- mlp_in of layer 1 is upstream of the nudge and stays equal."""
    step = {"n": 0}

    def nudge(mod, args, out):
        step["n"] += 1
        return out + 0.5 if step["n"] > 10 else out
    cap = P.SiteCapture(model)
    try:
        with torch.no_grad():
            lg, _ = P.run_control_hf(model, cap, ids, L, set())
            ctrl = cap.take()
            h = model.model.layers[1].mlp.experts.register_forward_hook(nudge)
            try:
                lg2, _ = P.run_control_hf(model, cap, ids, L, set())
            finally:
                h.remove()
            bumped = cap.take()
    finally:
        cap.remove()
    site = C.compare_sites({s: ctrl[s] for s in C.SITES}, {s: bumped[s] for s in C.SITES})
    fd = C.first_difference(site["equal"], C.row_stats(ctrl["final_norm"], bumped["final_norm"])["equal"],
                            C.logits_stats(lg, lg2)["equal"])
    assert fd[:10] == [None] * 10
    assert fd[10:] == [(1, "mlp_out")] * (L - 10)


def test_module_replay_reproduces_the_control_at_one_row_and_scores_dense_gemms(model, ids):
    cap = P.SiteCapture(model)
    rec = P.ModuleRecorder(P.replay_targets(model, full=True))
    try:
        rec.active = True
        with torch.no_grad():
            P.run_control_hf(model, cap, ids, L, set())
        rec.active = False
        cap.take()
        res = P.replay(rec, (1, 4, L), fp64_layers=(0,), device="cpu")
    finally:
        rec.remove()
        cap.remove()
    kinds = {n.split(".", 1)[1] if n != "lm_head" else n for n in res}
    assert {"experts", "router", "input_layernorm", "attn.q_proj", "attn.o_proj", "lm_head"} <= kinds
    for name, r in res.items():
        assert r["calls"] == L, name
        assert r["replay_faithful_at_1"], f"{name}: replay at n=1 does not reproduce the recorded output"
        assert set(r["by_n"]) == {"1", "4", str(L)}, name
    q = res["L0.attn.q_proj"]["by_n"][str(L)]
    assert q["path"] == "dense.cublas" and q["bound_ratio"] == q["bound_ratio"]      # finite, not NaN
    assert "path" not in res["L1.attn.q_proj"]["by_n"][str(L)]                    # fp64 only on the named layers


def test_subarm_toggles_are_configuration_and_are_undone(model, monkeypatch):
    from experts4bit_qlora.engines import hot_residency as hr
    monkeypatch.delenv("E4B_FUSE_COMBINE", raising=False)
    impl = model.config._attn_implementation
    sa = {"name": "hf.singleton.dotpad0", "attn": "hf", "grouping": "singleton", "combine": 0, "dotpad": "0"}
    P.apply_subarm(model, sa, None, impl)
    try:
        assert hr.FORCE_SINGLETON_GROUPS[0] and not hr.DEVICE_GROUPING[0]
        import os
        assert os.environ["E4B_FUSE_COMBINE"] == "0" and os.environ["GNF4_GEMV_DOTPAD"] == "0"
    finally:
        P.reset_subarm(model, impl)
    import os
    assert not hr.FORCE_SINGLETON_GROUPS[0] and not hr.DEVICE_GROUPING[0]
    assert "E4B_FUSE_COMBINE" not in os.environ and "GNF4_GEMV_DOTPAD" not in os.environ
    assert model.config._attn_implementation == impl


def test_the_int4_singleton_subarm_refuses_on_a_cut_without_the_buffer_fix(monkeypatch):
    from experts4bit_qlora.engines import hot_residency as hr
    sa = {"name": "hf.singleton", "attn": "hf", "grouping": "singleton", "combine": 1, "dotpad": None}
    assert P.refusal("int4", sa) is None and P.refusal("nf4", sa) is None
    monkeypatch.delattr(hr, "_int4_part_or_none")
    assert P.refusal("int4", sa).startswith("REFUSED")
    assert P.refusal("nf4", sa) is None


class _DtypeByRows(torch.nn.Module):
    """A router-shaped stand-in: fp32 weights at <= 4 rows, bf16 above (the fused router epilogue's shape)."""

    def forward(self, x):
        w = torch.softmax(x.float(), -1)
        return x, (w if x.shape[0] <= 4 else w.to(torch.bfloat16))


class _Boom(torch.nn.Module):
    def forward(self, x):
        if x.shape[0] > 1:
            raise RuntimeError("only one row")
        return x * 2


def test_replay_records_a_dtype_change_and_survives_a_module_that_raises():
    mods = {"L0.router": _DtypeByRows(), "L0.boom": _Boom()}
    rec = P.ModuleRecorder(mods)
    rec.active = True
    with torch.no_grad():
        for _ in range(8):
            x = torch.randn(1, 6).to(torch.bfloat16)
            for m in mods.values():
                m(x)
    rec.active = False
    res = P.replay(rec, (1, 4, 8), device="cpu")
    rec.remove()
    r = res["L0.router"]["by_n"]
    assert res["L0.router"]["replay_faithful_at_1"]
    assert "dtype_differs" not in r["4"] and r["4"]["rows_bit_equal"] == 4
    assert r["8"]["dtype_differs"] and r["8"]["rows_bit_equal"] == 0
    b = res["L0.boom"]
    assert b["replay_faithful_at_1"] and "error" in b["by_n"]["4"] and "error" in b["by_n"]["8"]
