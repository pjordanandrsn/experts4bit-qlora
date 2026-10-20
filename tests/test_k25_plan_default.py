# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""``E4B_K25_PLAN``'s per-family default (lane P161, FLIP; bench/p161/RESULTS-p161.md): an unset knob resolves at engine
build to ``c1`` on qwen3_moe where K25's decode resolved to ``prmt`` (the decode c1 was read with), and to ``today``
everywhere else with the reason recorded. A default never refuses a build or a forward: where the decode is not
``prmt`` (an older kernel package, ROCm, an explicit tree, another family's build in the same process), an unset knob
serves today's plan. ``E4B_K25_PLAN=today`` opts out; an explicit ``c1`` keeps its refusals. On CPU with the route
tests' kernel stubs; the bits are ``test_k25_lut_gpu.py``'s (c1 == today == the tree)."""
import pytest

pytest.importorskip("nf4_grouped")

from test_nf4_grouped_smallm_route import _install_stubs, _run, _set_mode  # noqa: E402

from experts4bit_qlora.engines import hot_residency as hr  # noqa: E402

TODAY = {"block_n": 32, "kc": 64, "warps": 4, "stages": 3}
C1_GU = {"block_n": 64, "kc": 256, "warps": 4, "stages": 2}
C1_DN = {"block_n": 128, "kc": 128, "warps": 8, "stages": 3}


@pytest.fixture(autouse=True)
def _fresh(monkeypatch):
    """Each test starts from an unresolved process (both defaults are today's until a build resolves them)."""
    monkeypatch.setattr(hr, "_K25_LUT_DEFAULT", ["tree"])
    monkeypatch.setattr(hr, "_K25_PLAN_DEFAULT", ["today"])
    monkeypatch.setattr(hr, "K25_LUT_SEEN", {})
    monkeypatch.setattr(hr, "K25_PLAN_SEEN", {})
    monkeypatch.delenv("E4B_K25_LUT", raising=False)
    monkeypatch.delenv("E4B_K25_PLAN", raising=False)
    _set_mode(monkeypatch, None)


def _shapes(calls):
    return [{k: p[k] for k in TODAY} for p in calls["k25_plan"]]


def _build(model_type):
    """The two resolutions as build_engine makes them, in its order."""
    return hr.resolve_k25_plan(hr.resolve_k25_lut(model_type), model_type=model_type)


def test_only_qwen3_moe_is_licensed_and_its_plan_is_c1():
    assert hr.K25_PLAN_DEFAULT_FAMILIES == {"qwen3_moe": "c1"}
    assert set(hr.K25_PLAN_DEFAULT_FAMILIES.values()) <= set(hr._K25_PLANS)


def test_qwen3_moe_unset_takes_c1_under_its_default_prmt_and_the_forward_follows(monkeypatch):
    calls = _install_stubs(monkeypatch, prmt=True)
    r = _build("qwen3_moe")
    assert r == {"env": None, "plan": "c1", "why": None, "source": "default-allowlisted", "gu": C1_GU, "dn": C1_DN}
    assert hr._K25_PLAN_DEFAULT == ["c1"]
    _run(24)
    assert _shapes(calls) == [C1_GU, C1_DN] and [p["lut"] for p in calls["k25_plan"]] == ["prmt", "prmt"]
    assert hr.K25_PLAN_SEEN == {"c1|le256": 1} and hr.K25_LUT_SEEN == {"prmt|le256": 1}


def test_qwen3_moe_unset_under_an_explicit_prmt_takes_c1_too(monkeypatch):
    monkeypatch.setenv("E4B_K25_LUT", "prmt")
    _install_stubs(monkeypatch, prmt=True)
    r = _build("qwen3_moe")
    assert (r["plan"], r["source"]) == ("c1", "default-allowlisted")


def test_today_opts_out_on_qwen3_moe(monkeypatch):
    monkeypatch.setenv("E4B_K25_PLAN", "today")
    calls = _install_stubs(monkeypatch, prmt=True)
    r = _build("qwen3_moe")
    assert (r["env"], r["plan"], r["source"]) == ("today", "today", "explicit") and hr._K25_PLAN_DEFAULT == ["today"]
    _run(24)
    assert _shapes(calls) == [TODAY, TODAY] and hr.K25_PLAN_SEEN == {"today|le256": 1}


def test_qwen3_moe_unset_under_an_explicit_tree_stays_today_never_a_refusal(monkeypatch):
    monkeypatch.setenv("E4B_K25_LUT", "tree")
    calls = _install_stubs(monkeypatch, prmt=True)
    r = _build("qwen3_moe")
    assert r["plan"] == "today" and r["why"] is None and hr._K25_PLAN_DEFAULT == ["today"]
    assert r["source"] == "default-off: K25's decode is 'tree' (explicit), and c1 was read only under 'prmt'"
    _run(24)
    assert _shapes(calls) == [TODAY, TODAY] and [p["lut"] for p in calls["k25_plan"]] == ["tree", "tree"]


def test_qwen3_moe_unset_on_an_older_kernel_package_stays_today(monkeypatch):
    calls = _install_stubs(monkeypatch, tree=True)                           # 0.49-era: no K39, so no prmt
    r = _build("qwen3_moe")
    assert r["plan"] == "today" and r["source"].startswith("default-off: K25's decode is 'tree' (default-off: the ")
    _run(24)
    assert _shapes(calls) == [TODAY, TODAY]


def test_qwen3_moe_unset_on_rocm_stays_today(monkeypatch):
    import torch
    monkeypatch.setattr(torch.version, "hip", "6.4.0", raising=False)
    _install_stubs(monkeypatch, prmt=True)
    r = _build("qwen3_moe")
    assert r["plan"] == "today" and "ROCm" in r["source"] and hr._K25_PLAN_DEFAULT == ["today"]


def test_qwen3_moe_unset_with_k25_off_resolves_no_plan_and_names_why(monkeypatch):
    _set_mode(monkeypatch, "0")
    _install_stubs(monkeypatch, prmt=True)
    r = _build("qwen3_moe")
    assert r == {"env": None, "plan": None, "why": "E4B_NF4_GROUPED_SMALLM=0",
                 "source": "default-off: E4B_NF4_GROUPED_SMALLM=0"}
    assert hr._K25_PLAN_DEFAULT == ["today"]


@pytest.mark.parametrize("mt", ["mixtral", "olmoe", "qwen2_moe", None])
def test_other_families_stay_on_today_with_the_reason(monkeypatch, mt):
    _install_stubs(monkeypatch, prmt=True)
    monkeypatch.setenv("E4B_K25_LUT", "prmt")                              # even under prmt: no read licenses them
    r = _build(mt)
    assert (r["plan"], r["source"]) == ("today", f"default-off: no read licenses model_type {mt!r}")
    assert hr._K25_PLAN_DEFAULT == ["today"]


def test_a_later_build_resets_the_process_default(monkeypatch):
    """One engine per process is the assumption, but a second build re-resolves rather than inheriting c1."""
    _install_stubs(monkeypatch, prmt=True)
    assert _build("qwen3_moe")["plan"] == "c1" and hr._K25_PLAN_DEFAULT == ["c1"]
    assert _build("mixtral")["plan"] == "today" and hr._K25_PLAN_DEFAULT == ["today"]


def test_a_default_c1_never_refuses_in_the_forward_where_the_decode_is_not_prmt(monkeypatch):
    """Another family's build left the decode at the tree while this process's plan default reads c1: the forward serves
    today's plan rather than refusing (only an EXPLICIT c1 refuses there, test_k25_plan.py)."""
    calls = _install_stubs(monkeypatch, prmt=True)
    monkeypatch.setattr(hr, "_K25_PLAN_DEFAULT", ["c1"])
    monkeypatch.setattr(hr, "_K25_LUT_DEFAULT", ["tree"])
    _run(24)
    assert _shapes(calls) == [TODAY, TODAY] and hr.K25_PLAN_SEEN == {"today|le256": 1}


def test_the_check_before_the_config_resolves_no_default(monkeypatch):
    _install_stubs(monkeypatch, prmt=True)
    monkeypatch.setattr(hr, "_K25_PLAN_DEFAULT", ["c1"])                    # whatever a build left
    r = hr.resolve_k25_plan(hr.resolve_k25_lut(), family_known=False)
    assert r["plan"] == "today" and r["source"].startswith("unresolved") and hr._K25_PLAN_DEFAULT == ["c1"]


def test_the_default_cites_p161s_register_row_and_the_record_exists():
    import json
    import pathlib
    root = pathlib.Path(__file__).resolve().parents[1]
    src = (root / "experts4bit_qlora" / "engines" / "hot_residency.py").read_text(encoding="utf-8")
    cid = "e4b.serve.p161.k25-plan-c1.qwen3.5090.2026-10-20"
    assert f'"qwen3_moe": "c1",   # {cid}' in src
    reg = json.loads((root / "docs" / "claims.json").read_text(encoding="utf-8"))
    row = [c for c in reg["claims"] if c["id"] == cid]
    assert len(row) == 1 and row[0]["status"] == "measured" and "bench/p161/RESULTS-p161.md" in row[0]["evidence"]
