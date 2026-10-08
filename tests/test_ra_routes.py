"""Declared CPU source/counter stand-ins; no NF4 GPU kernel is executed."""
import copy
import importlib.util
import sys
import types
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("ra_routes", ROOT / "bench/ra/ra_routes.py")
routes = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = routes
spec.loader.exec_module(routes)
STATE = {}
_BW_SHAPES = {(4, 32)}
_DOTPAD_CONFIGS = {(8, 32): (1, 1, 1)}


def dispatch_counts():
    return dict(STATE["counts"])


def _sm_count(device):
    return STATE["sm"]


def _bw():
    return STATE["bw"]


def _dotpad():
    return STATE["dotpad"]


def _bw_plan(N, K):
    return (1, 64, 1, STATE["split"])


def _splitk_plan_sk(N, K):
    return STATE["split"]


def _decode_plan(N, K, T, sm):
    return (1, 1, STATE["split"])


def _seen_route(route, rows):
    STATE["seen"].append(route)


def gemm_4bit_grouped(a_cat, B, absmax, sizes, expert_ids, decode_config=None, split_k=None, bw_config=None):
    for key, amount in STATE["increments"].items():
        STATE["counts"][key] += amount
    return a_cat



def _fused_forward(self, hidden_states):
    return self.qkv_proj(hidden_states)


def observer():
    global STATE
    STATE = dict(counts=dict.fromkeys(routes.COUNTERS, 0), sm=170, bw="auto", dotpad=True,
                 split=1, increments={"bw_prmt32": 1}, seen=[])
    module = sys.modules[__name__]
    digest = routes.module_digest(module)
    adapters = dict(schema=1, modules=[dict(module=k, ast_sha256=digest) for k in ("nf4", "hot", "qkv")])
    return routes.RouteObserver(module, module, module, adapters=adapters)


def record():
    return dict(single_calls=0, batch_calls=0, reference_single=0, reference_batch=0,
                gemv_calls=0, route_mismatches=0, selected_routes={}, gemv_routes={})


def call(obs, rec, *, N=4, T=1, tokens=1, sizes=None, **kwargs):
    obs.active = (rec, tokens)
    try:
        return obs.observe_gemm(gemm_4bit_grouped, torch.ones(T, 32), torch.ones(2, N, 16), None,
                                sizes or [1] * T, None, **kwargs)
    finally:
        obs.active = None


@pytest.mark.parametrize("N,sm,bw,dotpad,split,expected", [
    (4, 170, "auto", True, 1, "bw_prmt32"), (4, 170, "auto", True, 3, "bw_prmt32"),
    (8, 170, "auto", True, 1, "dotpad"), (8, 170, "auto", True, 2, "dotpad_splitk"),
    (8, 120, "auto", True, 1, "scalar"), (8, 170, "0", False, 2, "scalar_splitk"),
])
def test_expected_default_route_and_primary_dispatch_accounting(N, sm, bw, dotpad, split, expected):
    obs, rec = observer(), record()
    STATE.update(sm=sm, bw=bw, dotpad=dotpad, split=split,
                 increments={expected: 1, **({"bw_splitk": 1} if expected.startswith("bw_") and split > 1 else {})})
    call(obs, rec, N=N)
    assert rec["gemv_calls"] == 1 and rec["route_mismatches"] == 0
    assert rec["gemv_routes"] == {expected: 1}  # Split-K supplementary count is not a second launch.


@pytest.mark.parametrize("increments", [{}, {"bw_prmt32": 2}, {"bw_prmt32": 1, "scalar": 1},
                                          {"bw_prmt32": 1, "bw_splitk": 1}, {"bw_prmt32": -1}])
def test_missing_duplicate_split_or_reset_dispatch_is_rejected(increments):
    obs = observer()
    STATE["increments"] = increments
    with pytest.raises(ValueError):
        call(obs, record())


def test_actual_alternative_route_is_retained_as_fallback():
    obs, rec = observer(), record()
    STATE["increments"] = {"scalar": 1}
    call(obs, rec)
    assert rec["route_mismatches"] == 1 and rec["gemv_routes"] == {"scalar": 1}


@pytest.mark.parametrize("override", ["decode_config", "split_k", "bw_config"])
def test_plan_overrides_are_refused(override):
    obs = observer()
    with pytest.raises(ValueError, match="override"):
        call(obs, record(), **{override: 1})


@pytest.mark.parametrize("tokens,sizes,T", [(12, [1], 1), (12, [12], 12)])
def test_batched_or_prefill_singleton_counter_leak_refused(tokens, sizes, T):
    obs = observer()
    with pytest.raises(ValueError, match="unexpected"):
        call(obs, record(), tokens=tokens, sizes=sizes, T=T)


def test_hook_restoration_and_unknown_route():
    obs = observer()
    module = sys.modules[__name__]
    original, seen = module.gemm_4bit_grouped, module._seen_route
    obs.start()
    assert module.gemm_4bit_grouped is not original
    obs.active = (record(), 1)
    with pytest.raises(ValueError, match="unregistered"):
        module._seen_route("dequant_loop", 1)
    obs.close()
    assert module.gemm_4bit_grouped is original and module._seen_route is seen


def test_unreviewed_module_and_changed_counter_schema_refuse():
    obs = observer()
    module = sys.modules[__name__]
    with pytest.raises(ValueError, match="unreviewed"):
        routes.RouteObserver(module, module, module, adapters=dict(schema=1, modules=[]))
    STATE["counts"]["unknown"] = 0
    with pytest.raises(ValueError, match="schema"):
        routes.counters(obs.nf4)


def test_per_pass_census_calls_and_fallback_evidence():
    obs = observer()
    obs.modules["moe"] = record()
    obs.projections["attention"] = 4
    before = obs.snapshot()
    obs.modules["moe"].update(single_calls=1, batch_calls=1, gemv_calls=2, route_mismatches=1, reference_batch=1)
    obs.projections["attention"] += 3
    info = dict(moe_layers=1, fuse_qkv_n=1)
    e = obs.evidence(info, 3, before)
    assert e["decode_gemv"]["calls"] == 2 and e["decode_gemv"]["fallback_calls"] == 2
    with pytest.raises(ValueError, match="accounting"):
        obs.evidence(info, 4, before)
    with pytest.raises(ValueError, match="census"):
        obs.evidence(dict(moe_layers=2, fuse_qkv_n=1), 3, before)


def test_qkv_candidates_and_partial_fusion_fail():
    obs = observer()
    model = torch.nn.Module()
    model.attention = torch.nn.Module()
    a = model.attention
    a.head_dim = 32
    for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
        setattr(a, name, torch.nn.Linear(32, 32, bias=False))
    for name in ("q_norm", "k_norm"):
        setattr(a, name, torch.nn.LayerNorm(32))
    candidates = obs.candidates(model)
    assert candidates == ["attention"]
    with pytest.raises(ValueError, match="incomplete"):
        obs.assemble(model, candidates, {"E4B_PAGED_FUSE_QKV": "auto"})
    a.qkv_proj = torch.nn.Linear(32, 96, bias=False)
    with pytest.raises(ValueError, match="half-fused"):
        obs.assemble(model, candidates, {"E4B_PAGED_FUSE_QKV": "auto"})
    for name in ("q_proj", "k_proj", "v_proj"):
        delattr(a, name)
    with pytest.raises(ValueError, match="unknown fused"):
        obs.assemble(model, candidates, {"E4B_PAGED_FUSE_QKV": "auto"})
    a.forward = types.MethodType(_fused_forward, a)
    obs.assemble(model, candidates, {"E4B_PAGED_FUSE_QKV": "auto"})
    a.qkv_proj(torch.ones(1, 32))
    assert obs.projections["attention"] == 1
    obs.close()


def test_unknown_placement_refuses():
    obs = observer()
    model = torch.nn.Module()
    model.experts = torch.nn.Module()
    model.experts._hot_residency = types.SimpleNamespace(device=torch.device("cpu"))
    with pytest.raises(ValueError, match="placement"):
        obs.assemble(model, [], {"E4B_PAGED_FUSE_QKV": "0"})


def test_per_module_snapshot_mutation_is_not_accepted():
    obs = observer()
    obs.modules["moe"] = record()
    before = copy.deepcopy(obs.snapshot())
    before["modules"]["moe"]["gemv_calls"] = 1
    with pytest.raises(ValueError, match="reset"):
        obs.evidence(dict(moe_layers=1, fuse_qkv_n=0), 0, before)


def standin_forward(hidden, *args, _m, **kwargs):
    if STATE.get("reference"):
        return _m._e4b_hot_ref(hidden, *args, **kwargs)
    _seen_route("nf4_singleton", hidden.shape[0])
    for _ in range(STATE.get("projections", 2)):
        gemm_4bit_grouped(hidden, torch.ones(2, 4, 16), None, [1], None)
    return hidden


def moe_model():
    model = torch.nn.Module()
    mod = torch.nn.Module()
    model.moe = mod
    mod.num_experts = 2
    mod._hot_residency = types.SimpleNamespace(device=torch.device("cuda"), hot_ids=torch.arange(2))
    mod._e4b_hot_ref = lambda hidden, *a, **k: hidden
    mod.forward = types.FunctionType(standin_forward.__code__, globals(), standin_forward.__name__)
    mod.forward.__qualname__ = "enable_hot_residency.<locals>._fwd"
    mod.forward.__kwdefaults__ = {"_m": mod}
    return model


@pytest.mark.parametrize("reference", [False, True])
def test_observed_moe_entry_and_retained_reference_branch(reference):
    obs, model = observer(), moe_model()
    original, ref = model.moe.forward, model.moe._e4b_hot_ref
    STATE["reference"] = reference
    obs.start()
    try:
        obs.assemble(model, [], {"E4B_PAGED_FUSE_QKV": "0"})
        x = torch.ones(1, 32)
        assert torch.equal(model.moe(x), x)
        e = obs.evidence(dict(moe_layers=1, fuse_qkv_n=0), 0)["decode_gemv"]
        assert e["calls"] == (0 if reference else 2)
        assert e["fallback_calls"] == int(reference)
    finally:
        obs.close()
    assert model.moe.forward is original and model.moe._e4b_hot_ref is ref


def test_missing_one_projection_and_exception_restore_scope():
    obs, model = observer(), moe_model()
    STATE["projections"] = 1
    obs.start()
    try:
        obs.assemble(model, [], {"E4B_PAGED_FUSE_QKV": "0"})
        with pytest.raises(ValueError, match="two observed"):
            model.moe(torch.ones(1, 32))
        assert obs.active is None
    finally:
        obs.close()
