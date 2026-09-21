"""The P56 reducer's refusals, driven. A reducer that has never rejected anything
is not evidence — the whole lane turns on distinguishing two patterns and on
refusing an arm whose number would be a lie, so both are exercised here on
synthetic receipts."""

import importlib.util
import json
import os

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "p56_reduce", os.path.join(os.path.dirname(__file__), "..", "bench", "p56", "p56_reduce.py"))
p56 = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(p56)

N = 20


def _arm(tag, losses, **kw):
    r = {"tag": tag, "status": "ok", "C1_bit_exact": True, "init_sha": "aa" * 32,
         "trainable_params": 487280640, "tokens": {"sha256": "bb" * 32},
         "losses": losses, "loss_last": losses[-1], "s_per_step_median_11plus": 1.0,
         "n_patched": 30}
    r.update(kw)
    return r


def _write(tmp_path, arms):
    for a in arms:
        (tmp_path / f"{a['tag']}.json").write_text(json.dumps(a))
    return str(tmp_path)


def _curve(base, delta):
    return [round(base - 0.01 * i + delta, 5) for i in range(N)]


def test_ladder_that_tracks_the_arithmetic_is_read_as_a_defect(tmp_path):
    """Smallest rung inside the band, largest outside: the divergence is a
    function of the arithmetic, so the fused path's own error is implicated."""
    ref = _arm("reference_attn4", _curve(1.0, 0.0), n_patched=0)
    d = _write(tmp_path, [
        ref,
        _arm("batched_attn4", _curve(1.0, 0.001),
             batched_stats={"modules": 30, "calls": 2400, "batched": 2400,
                            "fallback_calls": 0, "by_reason": {}}),
        _arm("fused_attn4_nodgrad", _curve(1.0, 0.03), dgrad=False),
        _arm("fused_attn4", _curve(1.0, 0.09), dgrad=True),
    ])
    res = p56.reduce_run(d)
    assert res["verdict"] == "TRACKS-ARITHMETIC", res
    assert [r["status"] for r in res["rows"]] == ["PASS", "PASS", "FAIL"]


def test_flat_ladder_is_read_as_a_floor_not_a_defect(tmp_path):
    """Every rung outside the band and within 2x of each other, despite per-op
    errors an order of magnitude apart: the delta does not track the arithmetic."""
    ref = _arm("reference_attn4", _curve(1.0, 0.0), n_patched=0)
    d = _write(tmp_path, [
        ref,
        _arm("batched_attn4", _curve(1.0, 0.075),
             batched_stats={"modules": 30, "calls": 2400, "batched": 2400,
                            "fallback_calls": 0, "by_reason": {}}),
        _arm("fused_attn4_nodgrad", _curve(1.0, 0.088), dgrad=False),
        _arm("fused_attn4", _curve(1.0, 0.083), dgrad=True),
    ])
    res = p56.reduce_run(d)
    assert res["verdict"] == "FLOOR-NOT-ARITHMETIC", res
    assert all(r["status"] == "FAIL" for r in res["rows"])


def test_a_batched_arm_that_fell_back_is_VOID_not_a_passing_number(tmp_path):
    """THE control. A batched arm that fell back to the reference on every call
    agrees with the reference perfectly — a 0.000 parity that measures nothing.
    Read as a number it is the strongest possible evidence for the wrong
    conclusion, so it must be refused, and refused for the right reason."""
    ref = _arm("reference_attn4", _curve(1.0, 0.0), n_patched=0)
    d = _write(tmp_path, [
        ref,
        _arm("batched_attn4", _curve(1.0, 0.0),        # identical to the reference
             batched_stats={"modules": 30, "calls": 2400, "batched": 0,
                            "fallback_calls": 2400, "by_reason": {"pad_waste": 2400}}),
        _arm("fused_attn4", _curve(1.0, 0.09), dgrad=True),
    ])
    res = p56.reduce_run(d)
    row = next(r for r in res["rows"] if r["arm"] == "batched_attn4")
    assert row["status"] == "VOID" and "fell back" in row["why"], row
    # and it must not be counted as the smallest rung
    assert res["verdict"] != "TRACKS-ARITHMETIC", res


def test_batched_arm_without_engagement_evidence_is_VOID(tmp_path):
    """An arm carrying no `batched_stats` has not proven the path ran at all."""
    ref = _arm("reference_attn4", _curve(1.0, 0.0), n_patched=0)
    d = _write(tmp_path, [ref, _arm("batched_attn4", _curve(1.0, 0.001))])
    row = next(r for r in p56.reduce_run(d)["rows"] if r["arm"] == "batched_attn4")
    assert row["status"] == "VOID" and "engagement unproven" in row["why"]


@pytest.mark.parametrize("field,value,needle", [
    ("init_sha", "cc" * 32, "init_sha differs"),
    ("trainable_params", 123, "trainable"),
    ("C1_bit_exact", False, "C1"),
    ("n_patched", 0, "n_patched == 0"),
])
def test_an_arm_that_did_not_start_from_the_reference_is_VOID(tmp_path, field, value, needle):
    ref = _arm("reference_attn4", _curve(1.0, 0.0), n_patched=0)
    bad = _arm("fused_attn4", _curve(1.0, 0.001), dgrad=True)
    bad[field] = value
    row = next(r for r in p56.reduce_run(_write(tmp_path, [ref, bad]))["rows"]
               if r["arm"] == "fused_attn4")
    assert row["status"] == "VOID" and needle in row["why"], row


def test_no_reference_arm_is_NO_REF_not_a_verdict(tmp_path):
    d = _write(tmp_path, [_arm("fused_attn4", _curve(1.0, 0.09), dgrad=True)])
    res = p56.reduce_run(d)
    assert res["verdict"] == "NO-REF" and not res["readable"]


def test_a_ladder_that_all_passes_is_a_non_reproduction_not_a_fix(tmp_path):
    """The outcome neither registered pattern covers. If every rung lands inside
    the band, the standing 0.08257 did not reproduce on this box — which is a
    result about the standing row, not evidence that anything was fixed. It must
    not fall through to MIXED, and it must not be read as TRACKS-ARITHMETIC just
    because the smallest rung happens to be smallest."""
    ref = _arm("reference_attn4", _curve(1.0, 0.0), n_patched=0)
    d = _write(tmp_path, [
        ref,
        _arm("batched_attn4", _curve(1.0, 0.001),
             batched_stats={"modules": 30, "calls": 2400, "batched": 2400,
                            "fallback_calls": 0, "by_reason": {}}),
        _arm("fused_attn4_nodgrad", _curve(1.0, 0.002), dgrad=False),
        _arm("fused_attn4", _curve(1.0, 0.003), dgrad=True),
    ])
    res = p56.reduce_run(d)
    assert res["verdict"] == "DID-NOT-REPRODUCE", res
    assert all(r["status"] == "PASS" for r in res["rows"])
    assert "not evidence that anything was fixed" in res["why"]
