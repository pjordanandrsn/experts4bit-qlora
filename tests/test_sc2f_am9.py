"""SC2f Amendment 9: the qualify path follows TC1 amendment 83's card_clock.

- (a) the burn names ``--mode clock`` (amendment 83's clock rule) at the 2700 floor, and the launch template's guard strings on
  the wrapper are unchanged;
- (b) the in-run judge is amendment 83's own ``passes(..., mode="clock")`` at ``SERVING_JUDGE_FLOOR_MHZ = 0``: bits only, the
  clock recorded, never ruled; a serving-shaped run (busy median 2,400 MHz, clean bits) passes, and a 2,700 floor would VOID it;
- the PREREG states the serving counts the rule was set from.
"""
import ast
import importlib.util
import pathlib
import re

import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
SC2 = REPO / "bench" / "sc2"
_spec = importlib.util.spec_from_file_location("sc2_qualify_am9", SC2 / "sc2_qualify.py")
Q = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(Q)
DRIVE = (SC2 / "sc2_qualify_drive.sh").read_text(encoding="utf-8")
PREREG = (SC2 / "SC2f-PREREG.md").read_text(encoding="utf-8")
AM9 = PREREG[PREREG.index("### Amendment 9"):]
AM9_FLAT = re.sub(r"\s+", " ", AM9)
HDR = "timestamp, clocks.sm [MHz], clocks_event_reasons.active, power.draw [W]\n"


def row(mask="0x0000000000000000", mhz=2800, w=540.0):
    return f"2026/10/26 10:00:00.000, {mhz} MHz, {mask}, {w:.2f} W\n"


def test_a_serving_shaped_run_passes_and_a_clock_floor_would_void_it(monkeypatch):
    """The ruled test: a serving run sits at a busy median under the 5090 floor on a healthy card. At the registered floor of 0 it
    PASSES; a mutant floor of 2,700 VOIDs it (so a floor here would refuse healthy serving runs)."""
    text = HDR + row(mhz=2400, w=75.0) * 30
    rec = Q.judge_run(text)
    assert Q.SERVING_JUDGE_FLOOR_MHZ == 0
    assert rec["verdict"] == "PASS" and rec["busy_sm_median"] == 2400, rec
    monkeypatch.setattr(Q, "SERVING_JUDGE_FLOOR_MHZ", 2700)
    bad = Q.judge_run(text)
    assert bad["verdict"] == "REFUSE" and "2400 MHz < floor 2700" in bad["why"][0], bad


@pytest.mark.parametrize("label,text,want", [
    ("one Reliability sample (recorded)", HDR + row() * 29 + row("0x0000000000000400"), "PASS"),
    ("Reliability on 19 of 20 busy samples (95 %, persistent)", HDR + row() + row("0x0000000000000400") * 19, "REFUSE"),
    ("Reliability on 94 of 100 busy samples", HDR + row() * 6 + row("0x0000000000000400") * 94, "PASS"),
    ("SwPowerCap on every busy sample (recorded)", HDR + row("0x0000000000000004") * 30, "PASS"),
    ("HwSlowdown on one IDLE sample", HDR + row("0x0000000000000008", 210) + row() * 30, "REFUSE"),
    ("BoardLimit on one busy sample", HDR + row() * 29 + row("0x0000000000000200"), "REFUSE"),
    ("half idle plus Reliability on 95 % of the busy samples", HDR + row("0x0000000000000001", 210) * 100 + row() * 5
     + row("0x0000000000000400") * 95, "REFUSE"),
    ("fewer than 10 busy samples", HDR + row(mhz=210) * 40 + row() * 9, "UNREAD"),
])
def test_each_bit_clause(label, text, want):
    assert Q.judge_run(text)["verdict"] == want, label


def test_the_judge_records_the_shares_and_reports_the_clock():
    rec = Q.judge_run(HDR + row(mhz=2400) * 20 + row("0x0000000000000004", 2400) * 10)
    assert rec["verdict"] == "PASS" and rec["busy_swpowercap_fraction"] == round(10 / 30, 4)
    assert rec["busy_reliability_fraction"] == 0.0 and rec["any_sample_bits_seen"] == [] and rec["busy_sm_median"] == 2400


def test_the_judge_is_card_clocks_own_rule_by_keyword():
    """No copy of the rule: judge_run calls card_clock's passes with mode="clock" and the named floor, and no longer reads amendment
    80's bit constants."""
    tree = ast.parse((SC2 / "sc2_qualify.py").read_text(encoding="utf-8"))
    fn = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "judge_run")
    calls = [c for c in ast.walk(fn) if isinstance(c, ast.Call) and isinstance(c.func, ast.Attribute) and c.func.attr == "passes"]
    assert len(calls) == 1
    kw = {k.arg: k.value for k in calls[0].keywords}
    assert getattr(kw.get("mode"), "value", None) == "clock"
    assert isinstance(calls[0].args[1], ast.Name) and calls[0].args[1].id == "SERVING_JUDGE_FLOOR_MHZ"
    names = {n.attr for n in ast.walk(fn) if isinstance(n, ast.Attribute)}
    assert not names & {"HARD_BITS", "SOFT_BITS", "REFUSE_BITS"}, names & {"HARD_BITS", "SOFT_BITS", "REFUSE_BITS"}
    assert not hasattr(Q, "SOFT_MIN_CLEAN")


def test_the_burn_names_the_clock_mode_and_the_template_guards_stand():
    burn = next(ln for ln in DRIVE.splitlines() if "card_clock.py burn" in ln and not ln.lstrip().startswith("#"))
    assert "--floor 2700 --mode clock" in burn
    assert "judge-run --sampler" in DRIVE and "exited 0 without this run's success marker" in DRIVE   # the launch template's guards


def test_the_prereg_states_the_counts_the_rule_was_set_from():
    for s in ("TC1 amendment 83 (e4b#2152)", "14 serving runs", "51 rows are busy", "busy SM median 2,415 MHz",
              "32 of 51 sit under 2,700 MHz", "Reliability on 4 of 51", "SwPowerCap on 3 of 51", "on 0 of 51",
              "`SERVING_JUDGE_FLOOR_MHZ = 0`", "`card_clock.py burn --floor 2700 --mode clock`", "attempts 0 of 3",
              "SC2h (Amendment 5) and SC2i (Amendment 4) take this with Amendment 7, by reference"):
        assert s in AM9_FLAT, s
