# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""``E4B_K25_PLAN`` (lane P160; qwen3_moe's default since lane P161, test_k25_plan_default.py): K25's block shapes
per projection. ``today`` (also when unset and no family default applies) is the route's
plan for both projections, unchanged; ``c1`` is P156's first carried pair, read by P160 under K39's decode, and is
served ONLY under ``lut="prmt"``: any other decode is refused at build and in the forward, never run (the tree at c1's
shapes is slower, P155). An asked-for ``c1`` is never dropped with K25. On CPU with the route tests' kernel stubs; the
bits are ``test_k25_lut_gpu.py``'s."""
import importlib.util
import pathlib

import pytest

pytest.importorskip("nf4_grouped")

from test_nf4_grouped_smallm_route import _install_stubs, _run, _set_mode  # noqa: E402

from experts4bit_qlora.engines import hot_residency as hr  # noqa: E402

TODAY = {"block_n": 32, "kc": 64, "warps": 4, "stages": 3}
C1_GU = {"block_n": 64, "kc": 256, "warps": 4, "stages": 2}
C1_DN = {"block_n": 128, "kc": 128, "warps": 8, "stages": 3}


@pytest.fixture(autouse=True)
def _fresh(monkeypatch):
    monkeypatch.setattr(hr, "_K25_LUT_DEFAULT", ["tree"])
    monkeypatch.setattr(hr, "K25_LUT_SEEN", {})
    monkeypatch.setattr(hr, "K25_PLAN_SEEN", {})
    monkeypatch.setattr(hr, "_K25_PLAN_DEFAULT", ["today"])
    monkeypatch.delenv("E4B_K25_LUT", raising=False)
    monkeypatch.delenv("E4B_K25_PLAN", raising=False)
    _set_mode(monkeypatch, None)


def _shapes(calls):
    return [{k: p[k] for k in TODAY} for p in calls["k25_plan"]]


def test_c1_is_p160s_carried_pair_byte_for_byte():
    """The plan served is the plan P160 read: its reducer's CARRIED[0], never re-typed."""
    path = pathlib.Path(__file__).resolve().parents[1] / "bench" / "p160" / "p160_reduce.py"
    spec = importlib.util.spec_from_file_location("p160_reduce_for_plan", path)
    red = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(red)
    assert hr._K25_PLANS["c1"] == (red.CARRIED[0]["gu"], red.CARRIED[0]["dn"])
    assert red.TODAY == {k: hr._K25_PLAN[k] for k in TODAY} == TODAY
    assert hr._K25_PLANS["today"] == ({}, {})


@pytest.mark.parametrize("env", [None, "", "today", " Today "], ids=["unset", "empty", "today", "spaced"])
def test_today_is_the_routes_plan_for_both_projections(monkeypatch, env):
    if env is not None:
        monkeypatch.setenv("E4B_K25_PLAN", env)
    monkeypatch.setenv("E4B_K25_LUT", "prmt")
    calls = _install_stubs(monkeypatch, prmt=True)
    _run(24)
    assert _shapes(calls) == [TODAY, TODAY] and hr.K25_PLAN_SEEN == {"today|le256": 1}


@pytest.mark.parametrize("value", ["c2", "C1x", "1", "tree", "prmt"])
def test_an_unknown_plan_is_refused(monkeypatch, value):
    monkeypatch.setenv("E4B_K25_PLAN", value)
    with pytest.raises(ValueError, match=r"E4B_K25_PLAN=.*expected 'today' or 'c1'"):
        hr._k25_plan_raw()


@pytest.mark.parametrize("mode", [None, "1"], ids=["auto", "k25-on"])
def test_c1_under_explicit_prmt_gives_each_projection_its_own_shape(monkeypatch, mode):
    _set_mode(monkeypatch, mode)
    monkeypatch.setenv("E4B_K25_LUT", "prmt")
    monkeypatch.setenv("E4B_K25_PLAN", "c1")
    calls = _install_stubs(monkeypatch, prmt=True)
    out, ref = _run(24)
    assert _shapes(calls) == [C1_GU, C1_DN], calls["k25_plan"]                    # gate_up first, then down
    assert [p["lut"] for p in calls["k25_plan"]] == ["prmt", "prmt"]
    assert [p["dot_bf16"] for p in calls["k25_plan"]] == [False, False]           # the served precision, as P160 read
    assert hr.K25_PLAN_SEEN == {"c1|le256": 1} and hr.K25_LUT_SEEN == {"prmt|le256": 1}
    assert (out.float() - ref).abs().max() < 0.05                                 # the stub's arithmetic is the plan's


def test_c1_under_qwen3_moes_default_prmt_resolves_at_build_and_the_forward_follows(monkeypatch):
    monkeypatch.setenv("E4B_K25_PLAN", "c1")
    calls = _install_stubs(monkeypatch, prmt=True)
    r = hr.resolve_k25_plan(hr.resolve_k25_lut("qwen3_moe"))
    assert r == {"env": "c1", "plan": "c1", "why": None, "source": "explicit", "gu": C1_GU, "dn": C1_DN}
    _run(24)
    assert _shapes(calls) == [C1_GU, C1_DN] and hr.K25_PLAN_SEEN == {"c1|le256": 1}


def test_unset_plan_without_a_family_resolves_today_and_records_both_shapes(monkeypatch):
    """No model_type, no default: today, with the reason (the family's default is test_k25_plan_default.py's)."""
    _install_stubs(monkeypatch, prmt=True)
    r = hr.resolve_k25_plan(hr.resolve_k25_lut("qwen3_moe"))
    assert r == {"env": None, "plan": "today", "why": None, "source": "default-off: no read licenses model_type None",
                 "gu": TODAY, "dn": TODAY}


def test_c1_under_an_explicit_tree_is_refused_before_the_family_and_in_the_forward(monkeypatch):
    monkeypatch.setenv("E4B_K25_LUT", "tree")
    monkeypatch.setenv("E4B_K25_PLAN", "c1")
    calls = _install_stubs(monkeypatch, prmt=True)
    with pytest.raises(RuntimeError, match=r"E4B_K25_PLAN=c1 is lane P160's plan for K39's decode.*'tree'"):
        hr.resolve_k25_plan(hr.resolve_k25_lut(), family_known=False)           # the check before the config
    with pytest.raises(RuntimeError, match=r"E4B_K25_PLAN=c1 .*'tree' \(E4B_K25_LUT=tree\)"):
        _run(24)
    assert calls["k25_plan"] == []                                               # never run at the tree


@pytest.mark.parametrize("mt", ["mixtral", "olmoe", None])
def test_c1_on_a_family_whose_default_is_the_tree_is_refused_once_the_family_is_known(monkeypatch, mt):
    monkeypatch.setenv("E4B_K25_PLAN", "c1")
    _install_stubs(monkeypatch, prmt=True)
    early = hr.resolve_k25_plan(hr.resolve_k25_lut(), family_known=False)        # an unset decode: not yet decidable
    assert early["plan"] == "c1"
    with pytest.raises(RuntimeError, match=r"E4B_K25_PLAN=c1 .*'tree' \(default-off: no read licenses"):
        hr.resolve_k25_plan(hr.resolve_k25_lut(mt))


def test_c1_on_an_older_kernel_package_is_refused_never_run_at_the_tree(monkeypatch):
    monkeypatch.setenv("E4B_K25_PLAN", "c1")
    calls = _install_stubs(monkeypatch, tree=True)                               # 0.49-era: no K39
    with pytest.raises(RuntimeError, match=r"E4B_K25_PLAN=c1 .*'tree' \(default-off: the installed grouped-nf4-gemm"):
        hr.resolve_k25_plan(hr.resolve_k25_lut("qwen3_moe"))
    with pytest.raises(RuntimeError, match=r"E4B_K25_PLAN=c1 .*'tree' \(unset E4B_K25_LUT, as resolved at build\)"):
        _run(24)
    assert calls["k25_plan"] == []


def test_c1_on_rocm_is_refused(monkeypatch):
    import torch
    monkeypatch.setattr(torch.version, "hip", "6.4.0", raising=False)
    monkeypatch.setenv("E4B_K25_PLAN", "c1")
    _install_stubs(monkeypatch, prmt=True)
    with pytest.raises(RuntimeError, match=r"E4B_K25_PLAN=c1 .*ROCm"):
        hr.resolve_k25_plan(hr.resolve_k25_lut("qwen3_moe"))


def test_c1_with_k25_off_is_refused_before_the_family(monkeypatch):
    _set_mode(monkeypatch, "0")
    monkeypatch.setenv("E4B_K25_PLAN", "c1")
    _install_stubs(monkeypatch, prmt=True)
    with pytest.raises(RuntimeError, match=r"E4B_K25_PLAN=c1 names K25's plan, but K25 does not run here "
                                           r"\(E4B_NF4_GROUPED_SMALLM=0\)"):
        hr.resolve_k25_plan(hr.resolve_k25_lut(), family_known=False)


def test_today_with_k25_off_resolves_no_plan_without_a_refusal(monkeypatch):
    _set_mode(monkeypatch, "0")
    monkeypatch.setenv("E4B_K25_PLAN", "today")
    _install_stubs(monkeypatch, prmt=True)
    r = hr.resolve_k25_plan(hr.resolve_k25_lut("qwen3_moe"))
    assert r == {"env": "today", "plan": None, "why": "E4B_NF4_GROUPED_SMALLM=0", "source": "explicit"}


@pytest.mark.parametrize("mode", [None, "auto"], ids=["unset", "auto"])
def test_c1_with_a_kernel_package_without_k25_is_refused_under_auto(monkeypatch, mode):
    """Auto drops K25 silently on a package that predates it; an asked-for c1 is not dropped with it."""
    _set_mode(monkeypatch, mode)
    monkeypatch.setenv("E4B_K25_PLAN", "c1")
    _install_stubs(monkeypatch, with_k25=False)
    with pytest.raises(RuntimeError, match=r"E4B_K25_PLAN=c1 names K25's plan, but K25 does not run here "
                                           r"\(the kernel package predates K25\)"):
        hr.resolve_k25_plan(hr.resolve_k25_lut("qwen3_moe"))
    with pytest.raises(RuntimeError, match=r"E4B_K25_PLAN=c1 names K25's plan, but the installed kernel package has "
                                           r"no K25"):
        _run(24)


def test_today_with_a_kernel_package_without_k25_still_falls_back_under_auto(monkeypatch):
    monkeypatch.setenv("E4B_K25_PLAN", "today")
    calls = _install_stubs(monkeypatch, with_k25=False)
    _run(24)
    assert calls["k25_plan"] == [] and hr.K25_PLAN_SEEN == {}


def test_prefill_rows_take_no_plan_and_are_not_counted(monkeypatch):
    monkeypatch.setenv("E4B_K25_LUT", "prmt")
    monkeypatch.setenv("E4B_K25_PLAN", "c1")
    calls = _install_stubs(monkeypatch, prmt=True)
    _run(300)                                                                    # above K25's 256 rows
    assert calls["k25_plan"] == [] and hr.K25_PLAN_SEEN == {}
