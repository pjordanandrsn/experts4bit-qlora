# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""The B=1 fusion knobs' family-scoped default (lane P115, ``bench/p115/PREREG-p115.md`` Amendment 3, mechanism (B)).

An UNSET knob (``E4B_PAGED_FUSE_QKV``, ``E4B_FUSE_T1_GLUE``, ``E4B_FUSE_T1_GLUE_R2``, ``E4B_FUSE_ROUTER_EPI``) resolves at
the build, per the model's ``config.model_type``: ``auto`` on a family with a SANE read at T == 1 at reading size
(Qwen3-MoE, #1366; Qwen3.5/3.6-MoE, lane FAM's quality and speed reads, #1362; Mixtral, FAM's quality read and P135's
speed read, #1732) and ``0`` everywhere else. Explicit
``auto`` stays structural, as Phase C measured it, and off that list logs one warning naming the read the family lacks
or failed. Every key is a ``model_type`` a real transformers config carries. ``1`` and ``0`` keep their meanings.
``tests/test_fusion_modes.py`` (staged by Phase C) pins the three settings themselves; this file pins the default.
"""
import types

import pytest

from experts4bit_qlora import serve_paged
from experts4bit_qlora.engines import glue_fuse, glue_r2, router_epilogue
from experts4bit_qlora.serve_paged import (FUSION_DEFAULT_FAMILIES, FUSION_KNOBS, FUSION_UNSET, PagedServeConfig,
                                           _apply_fusions, resolve_fusion_modes)

UNSET = {k: FUSION_UNSET for k in FUSION_KNOBS}


def _model(model_type):
    """A stand-in carrying only what ``_apply_fusions`` reads: its ``config.model_type``."""
    return types.SimpleNamespace(config=types.SimpleNamespace(model_type=model_type))


@pytest.fixture
def folds(monkeypatch):
    """Record each fold's mode; the q/k/v fusion is recorded too and reports nothing fused (a stand-in model)."""
    calls = []

    def make(name):
        def fold(model, **kw):
            calls.append((name, kw.get("mode")))
            return (0, 0) if name == "r2" else 0
        return fold

    def fuse_qkv(model, *, fold_modes=None, fold_reports=None):
        calls.append(("qkv", None))
        for mod, fname, env in ((glue_fuse, "fuse_t1_glue", "E4B_FUSE_T1_GLUE"),
                                (glue_r2, "fuse_t1_glue_r2", "E4B_FUSE_T1_GLUE_R2"),
                                (router_epilogue, "fuse_router_epilogue", "E4B_FUSE_ROUTER_EPI")):
            getattr(mod, fname)(model, mode=(fold_modes or {}).get(env))
        return 0

    monkeypatch.setattr(glue_fuse, "fuse_t1_glue", make("glue"))
    monkeypatch.setattr(glue_r2, "fuse_t1_glue_r2", make("r2"))
    monkeypatch.setattr(router_epilogue, "fuse_router_epilogue", make("epi"))
    import experts4bit_qlora.engines.qkv_fuse as qkv_fuse
    monkeypatch.setattr(qkv_fuse, "fuse_qkv", fuse_qkv)
    monkeypatch.setattr(serve_paged, "_FUSION_WARNED", set())
    return calls


def test_the_allowlist_is_the_families_with_a_sane_read_at_t1():
    assert set(FUSION_DEFAULT_FAMILIES) == {"qwen3_moe", "qwen3_5_moe_text", "qwen3_5_moe", "mixtral"}
    assert "#1328" in FUSION_DEFAULT_FAMILIES["qwen3_moe"] and "T == 1" in FUSION_DEFAULT_FAMILIES["qwen3_moe"]
    for f in ("qwen3_5_moe_text", "qwen3_5_moe"):                 # lane FAM: quality at T == 1 and the speed it buys
        why = FUSION_DEFAULT_FAMILIES[f]
        assert "e4b.serve.fam.fused-stack-t1.qw36.5090.2026-10-09" in why and "router-epilogue-speed" in why
    why = FUSION_DEFAULT_FAMILIES["mixtral"]                      # FAM's quality read and P135's speed read, both
    assert "e4b.serve.fam.fused-stack-t1.mixtral.5090.2026-10-09" in why and "#1362" in why
    assert "e4b.serve.p135.fold-speed.mixtral.5090.2026-10-18" in why and "#1732" in why
    unlicensed = serve_paged.FUSION_UNLICENSED
    assert not set(unlicensed) & set(FUSION_DEFAULT_FAMILIES)
    assert "0.924" in unlicensed["gpt_oss"]
    assert all("lane FAM" in unlicensed[f] and "#1362" in unlicensed[f] and "T == 1" in unlicensed[f]
               for f in ("gpt_oss", "granitemoe"))


@pytest.mark.parametrize("family, module, cls", [
    ("qwen3_moe", "qwen3_moe.configuration_qwen3_moe", "Qwen3MoeConfig"),
    ("qwen3_5_moe_text", "qwen3_5_moe.configuration_qwen3_5_moe", "Qwen3_5MoeTextConfig"),
    ("qwen3_5_moe", "qwen3_5_moe.configuration_qwen3_5_moe", "Qwen3_5MoeConfig"),
    ("gpt_oss", "gpt_oss.configuration_gpt_oss", "GptOssConfig"),
    ("granitemoe", "granitemoe.configuration_granitemoe", "GraniteMoeConfig"),
    ("mixtral", "mixtral.configuration_mixtral", "MixtralConfig"),
])
def test_every_key_is_a_real_config_model_type(family, module, cls):
    """A key no served model carries would never match: serve_paged builds Qwen3.5/3.6-MoE's text tower, whose
    ``model_type`` is ``qwen3_5_moe_text``, so a ``qwen3_5_moe`` entry alone did nothing on it."""
    assert family in FUSION_DEFAULT_FAMILIES or family in serve_paged.FUSION_UNLICENSED
    mod = pytest.importorskip(f"transformers.models.{module}", reason="needs that transformers family")
    assert getattr(mod, cls).model_type == family


def test_a_qwen3_6_text_model_resolves_auto_and_patches_only_its_routers(monkeypatch):
    """The real text tower, unset knobs: the build resolves all four to ``auto`` (``default-allowlisted``), and only the
    router epilogue engages -- one fused router per MoE layer (P115's stand-in kernels on CPU)."""
    import sys
    from pathlib import Path
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers.models.qwen3_5_moe.configuration_qwen3_5_moe")
    from transformers import Qwen3_5MoeTextConfig
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from hybrid_reference import reference_modeling
    q35 = reference_modeling("qwen3_5_moe")
    import test_p115_quality_box as t
    monkeypatch.setitem(sys.modules, "int4_b32", t._kernel_stub())
    lin, att = "linear_attention", "full_attention"
    cfg = Qwen3_5MoeTextConfig(vocab_size=128, hidden_size=64, num_hidden_layers=4, num_attention_heads=4,
                               num_key_value_heads=2, head_dim=32, moe_intermediate_size=32,
                               shared_expert_intermediate_size=32, num_experts=4, num_experts_per_tok=2,
                               linear_num_key_heads=2, linear_num_value_heads=4, linear_key_head_dim=16,
                               linear_value_head_dim=16, linear_conv_kernel_dim=4, layer_types=[lin, lin, lin, att],
                               max_position_embeddings=256)
    torch.manual_seed(0)
    model = q35.Qwen3_5MoeForCausalLM(cfg).to(torch.bfloat16).eval()
    assert model.config.model_type == "qwen3_5_moe_text"
    report = {}
    census = _apply_fusions(model, PagedServeConfig(fusion_modes=dict(UNSET)), report=report)
    assert report["model_type"] == "qwen3_5_moe_text"
    assert report["modes"] == {k: "auto" for k in FUSION_KNOBS}
    assert set(report["sources"].values()) == {"default-allowlisted"}
    assert [census[k] for k in ("fuse_qkv_n", "fuse_t1_glue_n", "fuse_t1_glue_r2_n", "fuse_router_epilogue_n")] == \
        [0, 0, [0, 0], 4]


def test_a_mixtral_model_resolves_auto_and_engages_its_three_folds(monkeypatch):
    """Unset knobs on a real Mixtral: the build resolves all four to ``auto`` (``default-allowlisted``) -- FAM's ON_auto,
    the configuration P135 timed -- and engages the census P135 registered at depth L: no fused q/k/v, the glue fold's
    2L + 1 RMSNorms, round 2's L decoder layers and L attentions, and the router epilogue's L routers on their fp32
    path (P115's stand-in kernels on CPU). The prefill glue knob is not a fusion knob and stays off."""
    import sys
    from pathlib import Path
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers.models.mixtral", reason="needs transformers with Mixtral")
    from transformers import MixtralConfig, MixtralForCausalLM
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import test_p115_quality_box as t
    monkeypatch.setitem(sys.modules, "int4_b32", t._kernel_stub())
    layers = 4
    cfg = MixtralConfig(vocab_size=256, hidden_size=128, intermediate_size=64, num_hidden_layers=layers,
                        num_attention_heads=4, num_key_value_heads=2, head_dim=32, num_local_experts=4,
                        num_experts_per_tok=2, max_position_embeddings=512)
    torch.manual_seed(0)
    model = MixtralForCausalLM(cfg).to(torch.bfloat16).eval()
    assert model.config.model_type == "mixtral"
    report = {}
    census = _apply_fusions(model, PagedServeConfig(fusion_modes=dict(UNSET)), report=report)
    assert report["model_type"] == "mixtral"
    assert report["modes"] == {k: "auto" for k in FUSION_KNOBS}
    assert set(report["sources"].values()) == {"default-allowlisted"}
    assert [census[k] for k in ("fuse_qkv_n", "fuse_t1_glue_n", "fuse_t1_glue_r2_n", "fuse_router_epilogue_n")] == \
        [0, 2 * layers + 1, [layers, layers], layers]
    epi = report["folds"]["E4B_FUSE_ROUTER_EPI"]
    assert epi["patched"] == epi["fp32_upstream"] == layers, epi
    assert serve_paged.resolve_prefill_glue(FUSION_UNSET, "mixtral") == ("0", "default-off")


def test_unset_knobs_come_from_env_as_default(monkeypatch):
    for k in FUSION_KNOBS:
        monkeypatch.delenv(k, raising=False)
        monkeypatch.setenv(k, "  ")                       # empty counts as unset, as before
    monkeypatch.setenv("E4B_PAGED_DEVICE", "cpu")
    cfg = PagedServeConfig.from_env()
    assert cfg.fusion_modes == UNSET and cfg.fuse_qkv is True


@pytest.mark.parametrize("family", sorted(FUSION_DEFAULT_FAMILIES))
def test_unset_resolves_to_auto_on_an_allowlisted_family(family):
    resolved, sources = resolve_fusion_modes(dict(UNSET), family)
    assert resolved == {k: "auto" for k in FUSION_KNOBS}
    assert sources == {k: "default-allowlisted" for k in FUSION_KNOBS}


@pytest.mark.parametrize("family", ["gpt_oss", "granitemoe", "olmoe", "granitemoehybrid", "gemma4_text", None])
def test_unset_resolves_to_0_everywhere_else(family):
    resolved, sources = resolve_fusion_modes(dict(UNSET), family)
    assert resolved == {k: "0" for k in FUSION_KNOBS}
    assert sources == {k: "default-off" for k in FUSION_KNOBS}


def test_explicit_settings_pass_through_unchanged():
    modes = {"E4B_PAGED_FUSE_QKV": "1", "E4B_FUSE_T1_GLUE": "0", "E4B_FUSE_T1_GLUE_R2": "auto", "E4B_FUSE_ROUTER_EPI": "0"}
    for family in ("qwen3_moe", "gpt_oss"):
        resolved, sources = resolve_fusion_modes(dict(modes), family)
        assert resolved == modes and set(sources.values()) == {"explicit"}


def test_explicit_auto_off_the_list_warns_once_naming_the_failed_read(monkeypatch, capsys):
    monkeypatch.setattr(serve_paged, "_FUSION_WARNED", set())
    modes = {k: "auto" for k in FUSION_KNOBS}
    resolved, _ = resolve_fusion_modes(dict(modes), "gpt_oss")
    assert resolved == modes                               # structural: it still engages where the structure matches
    out = capsys.readouterr().out
    assert "WARNING" in out and "gpt_oss" in out and "lane FAM" in out and "0.924" in out and "#1362" in out
    resolve_fusion_modes(dict(modes), "gpt_oss")
    assert capsys.readouterr().out == ""                   # once per family per process
    resolve_fusion_modes(dict(modes), "olmoe")
    assert "no registered read" in capsys.readouterr().out
    resolve_fusion_modes(dict(modes), "granitemoe")
    out = capsys.readouterr().out
    assert "WARNING" in out and "T == 1" in out and "#1362" in out
    resolve_fusion_modes(dict(modes), "qwen3_moe")
    assert capsys.readouterr().out == ""                   # allowlisted: no warning


@pytest.mark.parametrize("family,want", [("qwen3_moe", "auto"), ("qwen3_5_moe_text", "auto"), ("qwen3_5_moe", "auto"),
                                         ("mixtral", "auto"), ("granitemoe", "0"), ("gpt_oss", "0"), (None, "0")])
def test_apply_fusions_resolves_the_default_per_family(folds, family, want):
    rep = {}
    _apply_fusions(_model(family), PagedServeConfig(fuse_qkv=True, fusion_modes=dict(UNSET)), report=rep)
    assert [m for n, m in folds if n != "qkv"] == [want, want, want]
    assert ("qkv", None) in folds if want == "auto" else ("qkv", None) not in folds
    assert rep["modes"] == {k: want for k in FUSION_KNOBS}
    src = "default-allowlisted" if want == "auto" else "default-off"
    assert rep["sources"] == {k: src for k in FUSION_KNOBS} and rep["model_type"] == family


def test_zero_is_the_way_back_on_an_allowlisted_family(folds):
    rep = {}
    _apply_fusions(_model("qwen3_moe"), PagedServeConfig(fuse_qkv=False, fusion_modes={k: "0" for k in FUSION_KNOBS}),
                   report=rep)
    assert [m for n, m in folds] == ["0", "0", "0"] and set(rep["sources"].values()) == {"explicit"}


def test_a_config_without_modes_keeps_its_old_reading(folds):
    """Bench harnesses build ``PagedServeConfig(fuse_qkv=...)`` without modes: nothing resolves, nothing warns."""
    rep = {}
    _apply_fusions(_model("gpt_oss"), PagedServeConfig(fuse_qkv=False), report=rep)
    assert [m for n, m in folds] == [None, None, None] and rep["sources"] == {}
