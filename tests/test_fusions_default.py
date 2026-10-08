# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""The B=1 fusion knobs' family-scoped default (lane P115, ``bench/p115/PREREG-p115.md`` Amendment 3, mechanism (B)).

An UNSET knob (``E4B_PAGED_FUSE_QKV``, ``E4B_FUSE_T1_GLUE``, ``E4B_FUSE_T1_GLUE_R2``, ``E4B_FUSE_ROUTER_EPI``) resolves at
the build, per the model's ``config.model_type``: ``auto`` on a family with a registered passing read (Qwen3-MoE,
Qwen3.5/3.6-MoE, Granite-MoE) and ``0`` everywhere else. Explicit ``auto`` stays structural, as Phase C measured it, and
off that list logs one warning naming the read the family lacks or failed. ``1`` and ``0`` keep their meanings.
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


def test_the_allowlist_is_the_registered_reads():
    assert set(FUSION_DEFAULT_FAMILIES) == {"qwen3_moe", "qwen3_5_moe", "granitemoe"}
    assert "#1328" in FUSION_DEFAULT_FAMILIES["qwen3_moe"] and "#1342" in FUSION_DEFAULT_FAMILIES["granitemoe"]
    assert "gpt_oss" not in FUSION_DEFAULT_FAMILIES and "0.924" in serve_paged.FUSION_FAILED_READS["gpt_oss"]


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


@pytest.mark.parametrize("family", ["gpt_oss", "olmoe", "mixtral", "granitemoehybrid", None])
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
    assert "WARNING" in out and "gpt_oss" in out and "0.924 < 0.95" in out and "#1342" in out
    resolve_fusion_modes(dict(modes), "gpt_oss")
    assert capsys.readouterr().out == ""                   # once per family per process
    resolve_fusion_modes(dict(modes), "olmoe")
    assert "no registered read" in capsys.readouterr().out
    resolve_fusion_modes(dict(modes), "qwen3_moe")
    assert capsys.readouterr().out == ""                   # allowlisted: no warning


@pytest.mark.parametrize("family,want", [("qwen3_moe", "auto"), ("granitemoe", "auto"), ("gpt_oss", "0"), (None, "0")])
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
