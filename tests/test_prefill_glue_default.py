# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""``E4B_FUSE_PREFILL_GLUE``'s serving default (lane P130, bench/p130/RESULTS-p130.md): unset resolves per family -- ``1``
on ``qwen3_moe``, the family read, ``0`` elsewhere -- while ``0`` / ``1`` stay explicit. ``serve_paged`` sets the knob for
its own fusion assembly alone, because the decode folds read it when they patch; the library's fold functions keep unset
= ``0`` for the bench harnesses that call them directly."""
import os
import types

import pytest

from experts4bit_qlora import serve_paged as sp

ENV = "E4B_FUSE_PREFILL_GLUE"


def test_the_server_reads_unset_as_per_family_and_refuses_anything_but_zero_or_one(monkeypatch):
    assert sp._prefill_glue_env("") == sp.FUSION_UNSET and sp._prefill_glue_env("  ") == sp.FUSION_UNSET
    assert sp._prefill_glue_env("0") == "0" and sp._prefill_glue_env(" 1 ") == "1"
    for bad in ("auto", "yes", "2"):
        with pytest.raises(ValueError, match=ENV):
            sp._prefill_glue_env(bad)
    monkeypatch.delenv(ENV, raising=False)
    assert sp.PagedServeConfig.from_env().prefill_glue == sp.FUSION_UNSET
    monkeypatch.setenv(ENV, "0")
    assert sp.PagedServeConfig.from_env().prefill_glue == "0"


def test_the_default_is_on_for_the_family_read_and_off_elsewhere():
    assert sp.resolve_prefill_glue(sp.FUSION_UNSET, "qwen3_moe") == ("1", "default-allowlisted")
    for other in ("granitemoe", "gpt_oss", "qwen3_5_moe_text", None):
        assert sp.resolve_prefill_glue(sp.FUSION_UNSET, other) == ("0", "default-off"), other
    assert sp.resolve_prefill_glue("0", "qwen3_moe") == ("0", "explicit")
    assert sp.resolve_prefill_glue("1", "granitemoe") == ("1", "explicit")
    assert list(sp.PREFILL_GLUE_DEFAULT_FAMILIES) == ["qwen3_moe"]


def _model(model_type):
    return types.SimpleNamespace(config=types.SimpleNamespace(model_type=model_type))


@pytest.mark.parametrize("preset", [None, "0", "1"])
@pytest.mark.parametrize("model_type,cfg_value,want", [("qwen3_moe", sp.FUSION_UNSET, "1"),
                                                      ("granitemoe", sp.FUSION_UNSET, "0"),
                                                      ("qwen3_moe", "0", "0"), ("granitemoe", "1", "1")])
def test_the_folds_see_the_resolved_knob_for_the_assembly_alone(monkeypatch, preset, model_type, cfg_value, want):
    seen = []
    monkeypatch.setattr(sp, "_assemble_fusions", lambda model, cfg, report=None: seen.append(os.environ.get(ENV)) or {})
    if preset is None:
        monkeypatch.delenv(ENV, raising=False)
    else:
        monkeypatch.setenv(ENV, preset)
    report = {}
    sp._apply_fusions(_model(model_type), sp.PagedServeConfig(prefill_glue=cfg_value), report=report)
    assert seen == [want]                                    # what glue_fuse.rows_cap reads at patch time
    assert os.environ.get(ENV) == preset                     # restored, or still unset
    assert report["prefill_glue"]["mode"] == want
    assert report["prefill_glue"]["source"] == ("explicit" if cfg_value != sp.FUSION_UNSET
                                                else "default-allowlisted" if want == "1" else "default-off")


def test_the_knob_is_restored_when_the_assembly_raises(monkeypatch):
    def boom(model, cfg, report=None):
        assert os.environ.get(ENV) == "1"
        raise RuntimeError("a fold refused")
    monkeypatch.setattr(sp, "_assemble_fusions", boom)
    monkeypatch.delenv(ENV, raising=False)
    with pytest.raises(RuntimeError, match="a fold refused"):
        sp._apply_fusions(_model("qwen3_moe"), sp.PagedServeConfig())
    assert ENV not in os.environ
