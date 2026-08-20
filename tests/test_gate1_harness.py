# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""The gate-1 harness's pure core: metric, verdict, plan.

These are the parts that decide whether a measurement PASSES, so they are
worth more scrutiny than the plumbing around them. Each test below pins a
way the harness could report a number that is not a measurement.
"""
import importlib.util
import sys
from pathlib import Path

import pytest

_p = Path(__file__).resolve().parents[1] / "bench" / "hybrid-g9" / "gate1_cold_sweep.py"
_spec = importlib.util.spec_from_file_location("gate1_cold_sweep", _p)
g1 = importlib.util.module_from_spec(_spec)
sys.modules["gate1_cold_sweep"] = g1
_spec.loader.exec_module(g1)


# ------------------------------------------------------------- the metric --

def test_fully_hidden_and_fully_exposed_are_the_endpoints():
    assert g1.hide_ratio(0.0, 100.0) == 1.0        # nothing reached the wall
    assert g1.hide_ratio(100.0, 100.0) == 0.0      # all of it did


def test_partial_hiding_is_the_fraction_kept_off_the_wall():
    assert g1.hide_ratio(30.0, 100.0) == pytest.approx(0.70)


def test_no_cold_work_is_none_not_a_number():
    """A point with no cold path neither hid anything nor failed to. A 0.0
    or 1.0 here would be read as a measurement of the mechanism."""
    assert g1.hide_ratio(0.0, 0.0) is None
    assert g1.hide_ratio(5.0, None) is None


def test_negative_exposure_clamps_rather_than_exceeding_one():
    """An arm can measure faster than its control inside noise. Reporting
    hide_ratio > 1 would claim it hid more latency than existed."""
    assert g1.hide_ratio(-20.0, 100.0) == 1.0


def test_over_exposure_is_reported_as_negative_not_floored():
    """Cold work costing MORE than its isolated cost is a real signal
    (contention), and flooring it at 0 would hide the knee."""
    assert g1.hide_ratio(150.0, 100.0) == pytest.approx(-0.5)


# ------------------------------------------------------------ the verdict --

def _point(**over):
    dyn = {"equivalent": True, "hide_ratio": 0.8, "exposed_ns": 10,
           "destination_flips": 3, "proportional_slowdown": 0.01}
    dyn.update(over)
    return {"arms": {"dynamic": dyn,
                     "cold-gpu": {"exposed_ns": 50},
                     "cold-cpu": {"exposed_ns": 40}}}


def test_all_clauses_passing_is_a_pass():
    assert g1.gate1_verdict(_point())["verdict"] == "PASS"


def test_numerical_inequivalence_fails_however_fast_it_was():
    """Equivalence is not tradeable against timing. A fast wrong answer is
    not a result."""
    v = g1.gate1_verdict(_point(equivalent=False, hide_ratio=0.99,
                                proportional_slowdown=0.0))
    assert v["verdict"] == "MISS"
    assert v["clauses"]["numerically_equivalent"] is False


def test_losing_to_a_fixed_arm_fails():
    v = g1.gate1_verdict(_point(exposed_ns=45))   # worse than cold-cpu's 40
    assert v["verdict"] == "MISS" and not v["clauses"]["beats_both_fixed"]


def test_never_flipping_destination_fails():
    """A scheduler that always picks the same side is a placement rule."""
    v = g1.gate1_verdict(_point(destination_flips=0))
    assert v["verdict"] == "MISS" and not v["clauses"]["destination_flipped"]


def test_hide_ratio_below_the_floor_fails():
    assert g1.gate1_verdict(_point(hide_ratio=0.69))["verdict"] == "MISS"
    assert g1.gate1_verdict(_point(hide_ratio=0.70))["verdict"] == "PASS"


def test_missing_measurements_fail_rather_than_pass_by_default():
    """An absent number must never read as a satisfied clause."""
    v = g1.gate1_verdict({"arms": {"dynamic": {}}})
    assert v["verdict"] == "MISS"
    assert not any(v["clauses"].values())


def test_every_clause_is_reported_including_the_passing_ones():
    v = g1.gate1_verdict(_point(hide_ratio=0.1))
    assert set(v["clauses"]) == {
        "numerically_equivalent", "hide_ratio_ge_floor", "beats_both_fixed",
        "destination_flipped", "slowdown_under_ceiling"}
    assert v["clauses"]["numerically_equivalent"] is True
    assert v["thresholds"]["hide_floor"] == 0.70


# --------------------------------------------------------------- the plan --

def test_the_plan_covers_every_cell_and_is_emitted_before_measuring():
    plan = g1.build_plan((0.01, 0.05), g1.ARMS, threshold=4.0)
    assert len(plan) == 2 * len(g1.ARMS)
    assert {c["arm"] for c in plan} == set(g1.ARMS)


def test_control_carries_no_forced_cold_mass():
    """That is what makes it a baseline rather than a fourth destination."""
    assert g1.arm_config("control", 4.0)["_forced_cold"] is False
    for arm in ("cold-gpu", "cold-cpu", "dynamic"):
        assert g1.arm_config(arm, 4.0)["_forced_cold"] is True


def test_the_arms_differ_only_in_destination():
    assert g1.arm_config("cold-gpu", 4.0)["cold_dest"] == "gpu"
    assert g1.arm_config("cold-cpu", 4.0)["cold_dest"] == "cpu"
    assert g1.arm_config("dynamic", 7.5)["cold_dest"] == 7.5


def test_unknown_arm_is_a_named_error():
    with pytest.raises(ValueError, match="unknown arm"):
        g1.arm_config("magic", 4.0)


def test_self_pair_reports_the_instruments_own_spread():
    sp = g1.self_pair([100, 102, 101], [101, 100, 103])
    assert sp["ratio"] >= 1.0 and sp["abs_diff_ns"] >= 0


def test_summarize_on_no_steps_does_not_invent_a_median():
    assert g1.summarize([]) == {"n": 0}
