# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""The fusion knobs' third setting, ``auto`` (e4b#1313): apply where the module structure and the installed kernels
license it, patch nothing -- without raising -- where they do not. ``1`` keeps the lanes' refusals; ``0`` (the default,
also unset) is off. Lane P115 registers the read that would make ``auto`` ``serve_paged``'s default; this file pins the
semantics before any default moves.

- ``serve_paged._fusion_env`` / ``glue_fuse.fold_mode``: the parser, and a typo refused rather than read as off;
- ``PagedServeConfig.from_env``: the four modes, ``fuse_qkv`` true for ``auto`` and ``1``, defaults unchanged;
- ``_apply_fusions``: ``auto`` on a model with no Qwen3-MoE attention does not raise, the folds still run with their
  modes, and a config built without modes calls everything exactly as before;
- each fold under ``auto``: no matching module, no kernel module, a kernel cut that lacks what a matched structure
  needs -- zero patched and a report, never an error; under ``1`` the same cases raise as they always did;
- the per-family census under ``auto`` on tiny real models with the glue kernels stood in by torch functions:
  Qwen3-MoE, GraniteMoe, gpt-oss and Qwen3.5-MoE (lane P115 Phase C's predictions at small layer counts).
"""
import sys
import types

import pytest
import torch

from experts4bit_qlora import serve_paged
from experts4bit_qlora.engines import glue_fuse, glue_r2, qkv_fuse, router_epilogue
from experts4bit_qlora.serve_paged import FUSION_ENV, FUSION_KNOBS, PagedServeConfig, _apply_fusions, _fusion_env

CENSUS = ("fuse_qkv_n", "fuse_t1_glue_n", "fuse_t1_glue_r2_n", "fuse_router_epilogue_n")


# ------------------------------------------------------------------ parser --

@pytest.mark.parametrize("raw,want", [("", "0"), ("0", "0"), ("1", "1"), ("auto", "auto"), (" AUTO ", "auto"),
                                      ("Auto", "auto")])
def test_the_parser_reads_three_settings(raw, want):
    assert _fusion_env("E4B_FUSE_T1_GLUE", raw) == want
    assert glue_fuse.fold_mode("E4B_FUSE_T1_GLUE", raw) == want


@pytest.mark.parametrize("raw", ["2", "true", "on", "yes"])
def test_the_parser_refuses_anything_else(raw):
    with pytest.raises(ValueError, match="expected 'auto', '0' or '1'"):
        _fusion_env("E4B_PAGED_FUSE_QKV", raw)


def test_fold_mode_reads_the_environment_when_no_value_is_given(monkeypatch):
    monkeypatch.delenv("E4B_FUSE_ROUTER_EPI", raising=False)
    assert glue_fuse.fold_mode("E4B_FUSE_ROUTER_EPI") == "0"
    monkeypatch.setenv("E4B_FUSE_ROUTER_EPI", "auto")
    assert glue_fuse.fold_mode("E4B_FUSE_ROUTER_EPI") == "auto"
    assert glue_fuse.fold_mode("E4B_FUSE_ROUTER_EPI", "0") == "0"          # a value overrides the environment


def _clear(monkeypatch):
    for k in FUSION_KNOBS + ("E4B_PAGED_GRAPHS",):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("E4B_PAGED_DEVICE", "cpu")


def test_from_env_defaults_are_unchanged(monkeypatch):
    _clear(monkeypatch)
    cfg = PagedServeConfig.from_env()
    assert cfg.fuse_qkv is False and cfg.fusion_modes == {k: "0" for k in FUSION_KNOBS}
    assert PagedServeConfig().fuse_qkv is False and PagedServeConfig().fusion_modes == {}


@pytest.mark.parametrize("raw,fused", [("auto", True), ("1", True), ("0", False)])
def test_from_env_reads_every_knob(monkeypatch, raw, fused):
    _clear(monkeypatch)
    for k in FUSION_KNOBS:
        monkeypatch.setenv(k, raw)
    cfg = PagedServeConfig.from_env()
    assert cfg.fuse_qkv is fused and cfg.fusion_modes == {k: raw for k in FUSION_KNOBS}


def test_from_env_refuses_a_typo(monkeypatch):
    _clear(monkeypatch)
    monkeypatch.setenv("E4B_FUSE_T1_GLUE_R2", "ture")
    with pytest.raises(ValueError, match="E4B_FUSE_T1_GLUE_R2"):
        PagedServeConfig.from_env()


# ---------------------------------------------------------- _apply_fusions --

def _recording_folds(monkeypatch, returns=(3, (2, 1), 4)):
    calls = []

    def make(name, ret):
        def fold(model, **kw):
            calls.append((name, dict(kw)))
            if "report" in kw:
                kw["report"]["seen"] = name
            return ret
        return fold
    monkeypatch.setattr(glue_fuse, "fuse_t1_glue", make("glue", returns[0]))
    monkeypatch.setattr(glue_r2, "fuse_t1_glue_r2", make("r2", returns[1]))
    monkeypatch.setattr(router_epilogue, "fuse_router_epilogue", make("epi", returns[2]))
    return calls


def test_auto_qkv_on_a_model_without_qwen3_moe_attention_does_not_raise(monkeypatch):
    calls = _recording_folds(monkeypatch)
    modes = {k: "auto" for k in FUSION_KNOBS}
    rep = {}
    out = _apply_fusions(torch.nn.Sequential(torch.nn.Linear(2, 2)), PagedServeConfig(fuse_qkv=True, fusion_modes=modes),
                         report=rep)
    assert out == {"fuse_qkv_n": 0, "fuse_t1_glue_n": 3, "fuse_t1_glue_r2_n": [2, 1], "fuse_router_epilogue_n": 4}
    assert [n for n, _ in calls] == ["glue", "r2", "epi"], "the folds still run, through fuse_qkv's assembly point"
    assert all(kw["mode"] == "auto" for _, kw in calls)
    assert rep["modes"] == modes and set(rep["folds"]) == set(FUSION_ENV)


def test_qkv_at_1_still_refuses_a_vacuous_fusion(monkeypatch):
    _recording_folds(monkeypatch)
    with pytest.raises(RuntimeError, match="E4B_PAGED_FUSE_QKV=1 matched no attention module"):
        _apply_fusions(torch.nn.Sequential(torch.nn.Linear(2, 2)),
                       PagedServeConfig(fuse_qkv=True, fusion_modes={"E4B_PAGED_FUSE_QKV": "1"}))


def test_the_unfused_branch_passes_each_fold_its_mode(monkeypatch):
    calls = _recording_folds(monkeypatch)
    modes = {"E4B_PAGED_FUSE_QKV": "0", "E4B_FUSE_T1_GLUE": "auto", "E4B_FUSE_T1_GLUE_R2": "1", "E4B_FUSE_ROUTER_EPI": "0"}
    rep = {}
    _apply_fusions(object(), PagedServeConfig(fusion_modes=modes), report=rep)
    assert calls == [("glue", {"mode": "auto", "report": {"seen": "glue"}}),
                     ("r2", {"mode": "1", "report": {"seen": "r2"}}),
                     ("epi", {"mode": "0", "report": {"seen": "epi"}})]
    assert rep["modes"] == modes


def test_a_config_without_modes_calls_everything_exactly_as_before(monkeypatch):
    """Bench harnesses and older callers build ``PagedServeConfig(fuse_qkv=...)`` and patch the folds with one-argument
    stand-ins: no mode and no report reach them, and the folds read their own environment variables."""
    calls = _recording_folds(monkeypatch)
    _apply_fusions(object(), PagedServeConfig(fuse_qkv=False))
    assert calls == [("glue", {}), ("r2", {}), ("epi", {})]


def test_fuse_qkv_forwards_fold_modes_and_reports(monkeypatch):
    calls = _recording_folds(monkeypatch)
    reports = {}
    n = qkv_fuse.fuse_qkv(torch.nn.Sequential(), fold_modes={"E4B_FUSE_T1_GLUE": "auto"}, fold_reports=reports)
    assert n == 0
    assert calls == [("glue", {"mode": "auto", "report": {"seen": "glue"}}),
                     ("r2", {"report": {"seen": "r2"}}), ("epi", {"report": {"seen": "epi"}})]
    assert set(reports) == set(FUSION_ENV)


# ------------------------------------------------------------- the folds --

def _kernels(*, scaled=True, rope_heads=True):
    """The glue kernels stood in by torch functions (the CPU suites' stubs); ``scaled`` / ``rope_heads`` False is an
    older grouped-nf4-gemm cut without the scaled residual fold / the norm-less rotary."""
    stub = types.ModuleType("int4_b32")

    def _norm(x, w, eps):
        xf = x.float()
        return xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps) * w.float()

    def rmsnorm_rows(x, w, eps):
        return _norm(x, w, eps).to(torch.bfloat16)

    def rmsnorm_resid_rows_legacy(x, resid, w, eps):
        s = (x.float() + resid.float()).to(torch.bfloat16)
        return _norm(s, w, eps).to(torch.bfloat16), s

    def rmsnorm_resid_rows(x, resid, w, eps, scale=1.0):
        return rmsnorm_resid_rows_legacy(x * scale if scale != 1.0 else x, resid, w, eps)

    def _rope(xn, cos, sin):
        half = xn.shape[-1] // 2
        rot = torch.cat([-xn[..., half:], xn[..., :half]], dim=-1)
        return (xn * cos.float().unsqueeze(1) + rot * sin.float().unsqueeze(1)).to(torch.bfloat16)

    def rope_norm_heads(x, w, cos, sin, eps):
        return _rope(_norm(x, w, eps).to(torch.bfloat16).float(), cos, sin)

    def router_epilogue(logits, k, norm, *, select_on_logits=False, bias=None):
        if select_on_logits:
            x = logits.float() + (bias.float() if bias is not None else 0.0)
            top, i = torch.topk(x, k, dim=-1)
            return x, torch.softmax(top, dim=-1), i
        probs = torch.softmax(logits.float(), dim=-1)
        v, i = torch.topk(probs, k, dim=-1)
        if norm:
            v = v / v.sum(dim=-1, keepdim=True)
        return probs, v, i

    stub.rmsnorm_rows, stub.rope_norm_heads, stub.router_epilogue = rmsnorm_rows, rope_norm_heads, router_epilogue
    if scaled:
        stub.rmsnorm_resid_rows = rmsnorm_resid_rows
        stub.scaled_resid_add_rows = lambda x, resid, scale: resid + x * scale
    else:
        stub.rmsnorm_resid_rows = rmsnorm_resid_rows_legacy
    if rope_heads:
        stub.rope_heads = lambda x, cos, sin: _rope(x.float(), cos, sin)
    return stub


def _granite(layers=2):
    pytest.importorskip("transformers.models.granitemoe", reason="needs transformers with GraniteMoe")
    from transformers import GraniteMoeConfig
    from transformers.models.granitemoe.modeling_granitemoe import GraniteMoeForCausalLM
    cfg = GraniteMoeConfig(vocab_size=256, hidden_size=128, intermediate_size=64, num_hidden_layers=layers,
                           num_attention_heads=4, num_key_value_heads=2, num_local_experts=4, num_experts_per_tok=2,
                           max_position_embeddings=512, residual_multiplier=0.22, embedding_multiplier=12.0,
                           attention_multiplier=0.015625, logits_scaling=6.0)
    torch.manual_seed(0)
    return GraniteMoeForCausalLM(cfg).to(torch.bfloat16).eval()


def _qwen3_moe(layers=2):
    pytest.importorskip("transformers.models.qwen3_moe", reason="needs transformers with Qwen3-MoE")
    from transformers import Qwen3MoeConfig
    from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeForCausalLM
    cfg = Qwen3MoeConfig(vocab_size=256, hidden_size=128, intermediate_size=128, moe_intermediate_size=64,
                         num_hidden_layers=layers, num_attention_heads=4, num_key_value_heads=2, head_dim=32,
                         num_experts=4, num_experts_per_tok=2, max_position_embeddings=512, decoder_sparse_step=1,
                         mlp_only_layers=[])
    torch.manual_seed(0)
    return Qwen3MoeForCausalLM(cfg).to(torch.bfloat16).eval()


@pytest.mark.parametrize("fold", ["glue", "r2", "epi"])
def test_auto_with_no_matching_module_patches_nothing_and_1_refuses(monkeypatch, fold):
    monkeypatch.setitem(sys.modules, "int4_b32", _kernels())
    fn = {"glue": glue_fuse.fuse_t1_glue, "r2": glue_r2.fuse_t1_glue_r2, "epi": router_epilogue.fuse_router_epilogue}[fold]
    rep = {}
    got = fn(torch.nn.Sequential(torch.nn.Linear(2, 2)), mode="auto", report=rep)
    assert got in (0, (0, 0)) and rep["mode"] == "auto"
    with pytest.raises(RuntimeError, match="vacuous enable"):
        fn(torch.nn.Sequential(torch.nn.Linear(2, 2)), mode="1")
    assert fn(torch.nn.Sequential(), mode="0") in (0, (0, 0))


@pytest.mark.parametrize("fold", ["glue", "r2", "epi"])
def test_auto_without_the_kernel_module_patches_nothing_and_1_refuses(monkeypatch, fold):
    monkeypatch.setitem(sys.modules, "int4_b32", None)              # the import fails
    fn = {"glue": glue_fuse.fuse_t1_glue, "r2": glue_r2.fuse_t1_glue_r2, "epi": router_epilogue.fuse_router_epilogue}[fold]
    model = _qwen3_moe()
    rep = {}
    assert fn(model, mode="auto", report=rep) in (0, (0, 0))
    assert rep["skipped"].startswith("no kernel")
    with pytest.raises(RuntimeError, match="install the matching cut or unset the flag"):
        fn(model, mode="1")


def test_auto_on_an_older_kernel_cut_skips_the_structures_it_cannot_patch(monkeypatch):
    """GraniteMoe's scaled residual needs ``rmsnorm_resid_rows(scale=)`` and its norm-less attention ``rope_heads``.
    On a cut without them, ``1`` refuses (as it always did) and ``auto`` leaves those modules on their own forward."""
    monkeypatch.setitem(sys.modules, "int4_b32", _kernels(scaled=False, rope_heads=False))
    model = _granite(2)
    rep = {}
    assert glue_r2.fuse_t1_glue_r2(model, mode="auto", report=rep) == (0, 0)
    assert rep["kernel_gaps"] == 4 and rep["patched"] == [0, 0]       # two layers, two attentions
    with pytest.raises(RuntimeError, match="needs the kernel side's"):
        glue_r2.fuse_t1_glue_r2(_granite(2), mode="1")


# --------------------------------------------------- the census, per family --

def _gpt_oss(layers=2):
    pytest.importorskip("transformers.models.gpt_oss", reason="needs transformers with gpt-oss")
    from transformers import GptOssConfig
    from transformers.models.gpt_oss.modeling_gpt_oss import GptOssForCausalLM
    cfg = GptOssConfig(vocab_size=256, hidden_size=128, intermediate_size=64, num_hidden_layers=layers, head_dim=32,
                       num_attention_heads=4, num_key_value_heads=2, num_local_experts=4, num_experts_per_tok=2,
                       max_position_embeddings=512, sliding_window=64)
    torch.manual_seed(0)
    return GptOssForCausalLM(cfg).to(torch.bfloat16).eval()


def _qwen3_5_moe(layers=4):
    pytest.importorskip("transformers.models.qwen3_5_moe", reason="needs transformers with Qwen3.5-MoE")
    from transformers.models.qwen3_5_moe import modeling_qwen3_5_moe as m
    from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import Qwen3_5MoeTextConfig
    cfg = Qwen3_5MoeTextConfig(vocab_size=256, hidden_size=128, num_hidden_layers=layers, num_attention_heads=4,
                               num_key_value_heads=2, head_dim=32, num_experts=4, num_experts_per_tok=2,
                               moe_intermediate_size=64, shared_expert_intermediate_size=64,
                               max_position_embeddings=512)
    torch.manual_seed(0)
    return m.Qwen3_5MoeForCausalLM(cfg).to(torch.bfloat16).eval()


AUTO = {k: "auto" for k in FUSION_KNOBS}


@pytest.mark.parametrize("family,build,want", [
    ("qwen3_moe", _qwen3_moe, {"fuse_qkv_n": 2, "fuse_t1_glue_n": 9, "fuse_t1_glue_r2_n": [2, 2], "fuse_router_epilogue_n": 2}),
    ("granitemoe", _granite, {"fuse_qkv_n": 0, "fuse_t1_glue_n": 5, "fuse_t1_glue_r2_n": [2, 2], "fuse_router_epilogue_n": 2}),
    # lane P115 Phase C's gpt-oss-20b prediction 0 / 49 / [24, 0] / 24 at two layers: the attention carries sinks
    ("gpt_oss", _gpt_oss, {"fuse_qkv_n": 0, "fuse_t1_glue_n": 5, "fuse_t1_glue_r2_n": [2, 0], "fuse_router_epilogue_n": 2}),
])
def test_auto_census_per_family(monkeypatch, family, build, want):
    monkeypatch.setitem(sys.modules, "int4_b32", _kernels())
    model = build()
    rep = {}
    out = _apply_fusions(model, PagedServeConfig(fuse_qkv=True, fusion_modes=AUTO), report=rep)
    assert {k: out[k] for k in CENSUS} == want, out
    assert rep["modes"] == AUTO


def test_auto_census_on_the_qwen3_5_hybrid_engages_the_router_only(monkeypatch):
    """Lane P115 Phase C's Qwen3.6-35B-A3B prediction, 0 / 0 / [0, 0] / 40, at four layers: its norms are centered
    (``1 + w``) and fail the probe; its router is the softmax-top-k-renormalise kind."""
    monkeypatch.setitem(sys.modules, "int4_b32", _kernels())
    try:
        model = _qwen3_5_moe(4)
    except Exception as e:                                   # a config this transformers cannot build is a skip, said
        pytest.skip(f"tiny Qwen3.5-MoE not constructible here: {type(e).__name__}: {e}")
    out = _apply_fusions(model, PagedServeConfig(fuse_qkv=True, fusion_modes=AUTO))
    assert {k: out[k] for k in CENSUS} == {"fuse_qkv_n": 0, "fuse_t1_glue_n": 0, "fuse_t1_glue_r2_n": [0, 0],
                                           "fuse_router_epilogue_n": 4}, out


def test_1_on_gpt_oss_refuses_the_q_k_v_fusion(monkeypatch):
    """Phase C's EXPLICIT control: all four at ``1`` on a family with no Qwen3-MoE attention must refuse."""
    monkeypatch.setitem(sys.modules, "int4_b32", _kernels())
    with pytest.raises(RuntimeError, match="matched no attention module"):
        _apply_fusions(_gpt_oss(), PagedServeConfig(fuse_qkv=True, fusion_modes={k: "1" for k in FUSION_KNOBS}))


def test_serve_paged_exports_the_knob_names():
    assert serve_paged.FUSION_KNOBS == ("E4B_PAGED_FUSE_QKV", "E4B_FUSE_T1_GLUE", "E4B_FUSE_T1_GLUE_R2",
                                        "E4B_FUSE_ROUTER_EPI")
