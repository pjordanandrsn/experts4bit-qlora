"""CPU mutation controls over native glue forwards, with CPU kernel stand-ins."""
import ast
import copy
import importlib.util
import json
import os
import sys
import types
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("ra_fallback", ROOT / "bench/ra/ra_fallback.py")
fallback = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = fallback
spec.loader.exec_module(fallback)


class ToyRMSNorm(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(32, dtype=torch.bfloat16))
        self.variance_epsilon = 1e-6

    def forward(self, hidden_states):
        x = hidden_states.float()
        return (x * torch.rsqrt(x.square().mean(-1, keepdim=True) + self.variance_epsilon)
                * self.weight.float()).to(hidden_states.dtype)


def patched(monkeypatch):
    from experts4bit_qlora.engines.glue_fuse import fuse_t1_glue
    kernel = types.ModuleType("int4_b32")
    kernel.rmsnorm_rows = lambda x, w, eps: (x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + eps)
                                           * w.float()).to(x.dtype)
    monkeypatch.setitem(sys.modules, "int4_b32", kernel)
    model = torch.nn.Module()
    model.norm = ToyRMSNorm()
    assert fuse_t1_glue(model, mode="1") == 1
    return model


def census(n=1):
    return dict(rms_glue=n, residual_glue=0, rope_glue=0, router_epilogue=0)


def test_real_native_forward_preserves_outputs_and_original_function(monkeypatch):
    model = patched(monkeypatch)
    original = model.norm.forward
    original_defaults = original.__defaults__
    x = torch.ones(1, 32, dtype=torch.bfloat16)
    ref = original(x)
    observer = fallback.GlueObserver().install(model)
    assert torch.equal(model.norm(x), ref)
    assert original.__defaults__ is original_defaults  # Installed release function itself is untouched.
    e = observer.evidence(census())["rms_glue"]
    assert e["small_calls"] == 1 and e["fallback_calls"] == 0 and e["observed_modules"] == 1


@pytest.mark.parametrize("n", [1, 12, 16, 64, 65, 512])
def test_small_wrong_dtype_fallback_is_observed_large_prefill_separate(monkeypatch, n):
    model = patched(monkeypatch)
    observer = fallback.GlueObserver().install(model)
    x = torch.ones(n, 32)  # Native small-row float32 guard falls back.
    expected = ToyRMSNorm()(x)
    assert torch.equal(model.norm(hidden_states=x), expected)
    e = observer.evidence(census())["rms_glue"]
    assert e["fallback_calls"] == int(n <= 64)
    assert e["large_fallbacks"] == int(n > 64)


def test_per_pass_counts_do_not_inherit_previous_fallback(monkeypatch):
    model = patched(monkeypatch)
    observer = fallback.GlueObserver().install(model)
    model.norm(torch.ones(1, 32))
    before = observer.snapshot()
    model.norm(torch.ones(12, 32, dtype=torch.bfloat16))
    assert observer.evidence(census(), before)["rms_glue"]["fallback_calls"] == 0
    assert observer.evidence(census())["rms_glue"]["fallback_calls"] == 1


@pytest.mark.parametrize("n", [0, 2, -1, True])
def test_partial_or_false_native_census_is_rejected(monkeypatch, n):
    observer = fallback.GlueObserver().install(patched(monkeypatch))
    with pytest.raises(ValueError, match="census"):
        observer.evidence(census(n))


def test_unreviewed_body_does_not_get_zero_fallback_claim(monkeypatch):
    model = patched(monkeypatch)
    adapters = json.loads((ROOT / "bench/ra/fallback-adapters.json").read_text())
    adapters["adapters"][0]["ast_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="unreviewed"):
        fallback.GlueObserver(adapters).install(model)


def test_unobserved_modules_and_reset_counts_fail(monkeypatch):
    observer = fallback.GlueObserver().install(patched(monkeypatch))
    before = observer.snapshot()
    before["norm"]["small_calls"] = 1
    with pytest.raises(ValueError, match="accounting"):
        observer.evidence(census(), before)
    with pytest.raises(ValueError, match="coverage changed"):
        observer.evidence(census(), {"foreign": before["norm"]})


def test_duplicate_or_missing_install_rejected(monkeypatch):
    observer = fallback.GlueObserver()
    with pytest.raises(ValueError, match="uninstalled"):
        observer.snapshot()
    observer.install(patched(monkeypatch))
    with pytest.raises(ValueError, match="already"):
        observer.install(patched(monkeypatch))


def test_missing_retained_original_is_not_zero_fallback():
    def forward(hidden_states):
        return hidden_states
    with pytest.raises(ValueError, match="retained original"):
        fallback.clone_orig(forward, {})


@pytest.mark.parametrize("change", ["graphs", "mode", "missing", "invalid"])
def test_independent_defaults_reject_config_and_mode_drift(change):
    modes = dict.fromkeys(("E4B_PAGED_FUSE_QKV", "E4B_FUSE_T1_GLUE", "E4B_FUSE_T1_GLUE_R2", "E4B_FUSE_ROUTER_EPI"), "auto")
    fresh = types.SimpleNamespace(graphs=True, fusion_modes=modes, token="synthetic-secret")
    cfg = copy.deepcopy(fresh)
    if change == "graphs":
        cfg.graphs = False
    elif change == "mode":
        cfg.fusion_modes["E4B_FUSE_T1_GLUE"] = "0"
    elif change == "missing":
        del cfg.fusion_modes["E4B_FUSE_T1_GLUE"]
        fresh = copy.deepcopy(cfg)
    else:
        cfg.fusion_modes["E4B_FUSE_T1_GLUE"] = "UNKNOWN"
        fresh = copy.deepcopy(cfg)
    server = types.SimpleNamespace(PagedServeConfig=types.SimpleNamespace(from_env=lambda: fresh))
    with pytest.raises(ValueError, match="defaults"):
        fallback.resolved_defaults(server, cfg)


def test_defaults_never_emit_token():
    cfg = types.SimpleNamespace(fusion_modes=dict.fromkeys(("E4B_PAGED_FUSE_QKV", "E4B_FUSE_T1_GLUE",
                                                          "E4B_FUSE_T1_GLUE_R2", "E4B_FUSE_ROUTER_EPI"), "0"),
                                token="synthetic-secret")
    server = types.SimpleNamespace(PagedServeConfig=types.SimpleNamespace(from_env=lambda: cfg))
    assert "token" not in fallback.resolved_defaults(server, cfg)


@pytest.mark.parametrize("mutation", ["missing", "mode", "skipped", "kernel_gaps", "no_kernel_mode", "patched"])
def test_native_auto_gap_and_report_mutations_cannot_claim_inapplicable(mutation):
    modes = dict.fromkeys(("E4B_FUSE_T1_GLUE", "E4B_FUSE_T1_GLUE_R2", "E4B_FUSE_ROUTER_EPI"), "auto")
    reports = {k: {"mode": "auto", "patched": [0, 0] if k == "E4B_FUSE_T1_GLUE_R2" else 0} for k in modes}
    info = dict(fuse_t1_glue_n=0, fuse_t1_glue_r2_n=[0, 0], fuse_router_epilogue_n=0,
                fusion_modes=modes, fusion_report=reports)
    fallback.check_fold_reports(info)
    if mutation == "missing":
        del reports["E4B_FUSE_T1_GLUE"]
    elif mutation == "mode":
        reports["E4B_FUSE_T1_GLUE"]["mode"] = "0"
    elif mutation == "patched":
        reports["E4B_FUSE_T1_GLUE"]["patched"] = 1
    else:
        reports["E4B_FUSE_T1_GLUE"][mutation] = 1
    with pytest.raises(ValueError):
        fallback.check_fold_reports(info)


def test_duplicate_adapter_and_tampered_snapshot_fail(monkeypatch):
    data = json.loads((ROOT / "bench/ra/fallback-adapters.json").read_text())
    data["adapters"].append(data["adapters"][0])
    with pytest.raises(ValueError, match="duplicate"):
        fallback.GlueObserver(data)
    observer = fallback.GlueObserver().install(patched(monkeypatch))
    before = observer.snapshot()
    before["norm"]["ast_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="identity"):
        observer.evidence(census(), before)


def native_toys(name):
    spec = importlib.util.spec_from_file_location("ra_native_" + name, ROOT / "tests" / (name + ".py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.parametrize("factory", ["ToyDecoderLayer", "GraniteMoeShapedDecoderLayer", "ToyFusedAttention",
                                     "ToyUnfusedAttention", "ToyNoNormAttention"])
def test_all_round_two_forward_adapters_count_actual_fallback_branch(monkeypatch, factory):
    from experts4bit_qlora.engines.glue_r2 import fuse_t1_glue_r2
    toys = native_toys("test_glue_r2")
    toys._stub_rope_only(monkeypatch, {"resid": 0, "rope": 0})
    model = torch.nn.Module()
    model.subject = getattr(toys, factory)()
    layers, attns = fuse_t1_glue_r2(model, mode="1")
    assert layers + attns == 1
    # Substitute only the test object's retained original. The native branch
    # and complete AST body stay intact; no transformer/GPU attention executes.
    fn = model.subject.forward
    defaults = list(fn.__defaults__)
    names = fn.__code__.co_varnames[:fn.__code__.co_argcount]
    index = names[len(names) - len(defaults):].index("_orig")
    defaults[index] = lambda hidden_states, **kw: hidden_states
    fn.__defaults__ = tuple(defaults)
    observer = fallback.GlueObserver().install(model)
    if layers:
        x = torch.ones(1, 1, 32)  # Wrong dtype enters native retained-original branch.
    else:
        x = torch.ones(1, 1, 32, dtype=torch.bfloat16)  # Absent position embeddings enters that branch.
    assert torch.equal(model.subject(x), x)
    counts = dict(rms_glue=0, residual_glue=layers, rope_glue=attns, router_epilogue=0)
    e = observer.evidence(counts)["residual_glue" if layers else "rope_glue"]
    assert e["small_calls"] == e["fallback_calls"] == 1


@pytest.mark.parametrize("factory", ["ToyRouter", "GptOssLikeRouter", "Gemma4TextRouter"])
def test_all_router_forward_variants_report_large_prefill_fallback(monkeypatch, factory):
    from experts4bit_qlora.engines.router_epilogue import fuse_router_epilogue
    toys = native_toys("test_router_epilogue")
    toys._stub(monkeypatch, {"fused": 0})
    model = torch.nn.Module()
    model.subject = getattr(toys, factory)()
    assert fuse_router_epilogue(model, mode="1") == 1
    observer = fallback.GlueObserver().install(model)
    x = torch.ones(65, toys.HID)
    model.subject(x)
    e = observer.evidence(dict(rms_glue=0, residual_glue=0, rope_glue=0, router_epilogue=1))["router_epilogue"]
    assert e["large_fallbacks"] == 1 and e["fallback_calls"] == 0


def test_portable_ast_ignores_display_and_empty_optional_fields(monkeypatch):
    tree = ast.parse("def f(x): return x + 1").body[0]
    expected = fallback.ast_digest(tree)
    if hasattr(tree, "type_params"):
        del tree.type_params
    assert fallback.ast_digest(tree) == expected
    tree.type_params = []
    assert fallback.ast_digest(tree) == expected
    tree.type_params = [ast.Name(id="T", ctx=ast.Load())]
    assert fallback.ast_digest(tree) != expected
    tree.type_params = []
    tree.body[0].value.right.value = 2
    assert fallback.ast_digest(tree) != expected

    def unavailable(*args, **kwargs):
        raise AssertionError("ast.dump is a display format, not the wire identity")

    monkeypatch.setattr(ast, "dump", unavailable)
    fallback.GlueObserver().install(patched(monkeypatch))
    legacy = json.loads((ROOT / "bench/ra/fallback-adapters.json").read_text())
    legacy["schema"] = 1
    with pytest.raises(ValueError, match="schema"):
        fallback.GlueObserver(legacy)


@pytest.mark.parametrize("model_type,mode,source", [("qwen3_moe", "auto", "default-allowlisted"),
                                                   ("granitemoe", "0", "default-off")])
def test_real_release_from_env_resolves_unset_by_family(monkeypatch, model_type, mode, source):
    from experts4bit_qlora import serve_paged as native
    assert Path(native.__file__).resolve() == ROOT / "experts4bit_qlora/serve_paged.py"
    monkeypatch.setattr(os, "environ", {"E4B_PAGED_DEVICE": "cpu"})
    cfg = native.PagedServeConfig.from_env()  # No stub; no fusion overrides.
    assert set(cfg.fusion_modes.values()) == {native.FUSION_UNSET}
    result = fallback.resolved_defaults(native, cfg, model_type=model_type)
    assert set(result["fusion_modes"].values()) == {mode}
    assert set(result["fusion_sources"].values()) == {source}
    assert result["fusion_modes_unresolved"] == cfg.fusion_modes
    assert result["model_type"] == model_type
    with pytest.raises(ValueError, match="model type"):
        fallback.resolved_defaults(native, cfg)
