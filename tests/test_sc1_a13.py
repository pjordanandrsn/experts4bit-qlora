# Copyright (c) 2026 Cerin Amroth LLC. MIT.
"""SC1 amendment A13 (#846): the box script pins the int4 store's prefill route to ``loop``.

#937 made ``E4B_INT4_PREFILL=auto`` (K19 wherever it can run) the library default after all three SC1 boxes had run under
``loop``. A box-A redraw on main would otherwise read a different e4b on every arm whose int4 store sees a T > 1 call: the
scheduler's prefill, TTFT, and the K8 licence's 512-token prompt. The box exports the pin once, after the scrub, so every e4b
process inherits it, and FOLDS / SPEEDENV / ROUTEENV stay byte-identical to the lanes that pin them to SC1's (P100, P102).
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

RUN = Path(__file__).resolve().parents[1] / "bench" / "sc1" / "sc1_run.sh"
EXPORT = "export E4B_INT4_PREFILL=loop"


def _src() -> str:
    return RUN.read_text()


def test_a13_the_box_exports_the_registered_prefill_route():
    # fails on the registered script: nothing set the prefill route, so main's `auto` would reach every arm
    assert re.search(rf"^{re.escape(EXPORT)}$", _src(), re.M)


def test_a13_the_export_follows_the_scrub_and_precedes_the_tripwire_and_every_arm():
    src = _src()
    scrub = src.index("unset E4B_SERVE_EXP_INT4 ")
    end = src.index("\n", src.index("E4B_PAGED_FUSE_QKV", scrub))
    assert "E4B_INT4_PREFILL" in src[scrub:end], "an inherited E4B_INT4_PREFILL would survive the scrub"
    exp = src.index(EXPORT)
    assert end < exp < src.index("<<'PYT'"), "the export must sit between the scrub and the e4b tripwire"
    first_arm = min(m.start() for m in re.finditer(r'"\$PY" \$W/(step_decomp|sc1_e4b_sched)\.py', src))
    assert exp < first_arm


def test_a13_no_arm_sets_another_prefill_route():
    hits = [ln for ln in _src().splitlines() if "E4B_INT4_PREFILL" in ln and not ln.lstrip().startswith("#")]
    for ln in hits:
        assert ln.strip() == EXPORT or "E4B_PAGED_FUSE_QKV E4B_INT4_PREFILL" in ln or "_int4_prefill_mode_env" in ln \
            or 'environ.get("E4B_INT4_PREFILL")' in ln or 'environ\\.get\\("E4B_INT4_PREFILL"' in ln \
            or "ROUTE_DEFAULT E4B_INT4_PREFILL" in ln or "export could not pin" in ln, ln


def test_a13_the_lanes_pinned_to_sc1s_env_strings_are_untouched():
    # P100 / P102 assert their FOLDS / SPEEDENV / ROUTEENV equal SC1's byte for byte; A13 changes none of them
    src = _src()
    want = {"ROUTEENV": "E4B_INT4_GROUPED_SMALLM=auto E4B_INT4_LEAN_GLUE=auto E4B_NF4_GROUPED_SMALLM=0 E4B_MXFP4_GROUPED_SMALLM=auto",
            "FOLDS": "E4B_FUSE_T1_GLUE=1 E4B_FUSE_T1_GLUE_R2=1 E4B_FUSE_ROUTER_EPI=1"}
    for var, val in want.items():
        assert re.search(rf'^{var}="([^"]*)"', src, re.M).group(1) == val, var


def test_a13_the_tripwire_checks_the_exported_pin():
    src = _src()
    tw = src[src.index("<<'PYT'"): src.index("\nPYT\n")]
    assert 'environ\\.get\\("E4B_INT4_PREFILL"' in tw
    assert 'os.environ.get("E4B_INT4_PREFILL") == "loop"' in tw
    assert 'hr._int4_prefill_mode_env() == "loop"' in tw


def test_a13_loop_holds_where_auto_would_take_k19(monkeypatch):
    hr = pytest.importorskip("experts4bit_qlora.engines.hot_residency")
    m = re.search(r'environ\.get\("E4B_INT4_PREFILL",\s*"([^"]*)"\)', Path(hr.__file__).read_text())
    assert m and m.group(1) == "auto", "the library default the pin guards against"
    monkeypatch.setattr(hr, "_k19_prefill_available", lambda: True)
    monkeypatch.delenv("E4B_INT4_PREFILL", raising=False)
    assert hr._int4_prefill_mode_env() == "k19"          # what an unpinned redraw would run
    monkeypatch.setenv(*EXPORT.split()[1].split("=", 1))
    assert hr._int4_prefill_mode_env() == "loop"         # what every SC1 box ran
