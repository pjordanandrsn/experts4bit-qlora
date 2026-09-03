# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""The K8 gate rule, pinned to the receipts that decided it."""
import pytest

from experts4bit_qlora.k8_gate import Arm, verdict

WT = "5e3656e01e50aaaa"
C4 = "9f6289665fb9bbbb"


def _wt(ppl, src="wikitext"):
    return Arm(ppl, WT, 8192, src)


def _c4(ppl):
    return Arm(ppl, C4, 8192, "c4val1")


def test_uncalibrated_is_symmetric():
    ok, _ = verdict([(_wt(7.97594), _wt(7.97594 + 0.0558))], calibrated=False)
    assert not ok                                   # P13 RTN: +0.0558 FAIL
    ok, _ = verdict([(_wt(7.97594), _wt(7.97594 - 0.078))], calibrated=False)
    assert not ok                                   # too good = broken instrument
    ok, _ = verdict([(_wt(7.97594), _wt(7.97594 + 0.049))], calibrated=False)
    assert ok


def test_calibrated_one_sided_needs_corroboration():
    base_wt, base_c4 = _wt(7.97594), _c4(15.37804)
    # P22b: the C4 pack, -0.042 on wikitext and -0.115 on C4-val -> PASS
    ok, lines = verdict([(base_wt, _wt(7.93409)), (base_c4, _c4(15.26311))],
                        calibrated=True, calibration_domain="c4val1")
    assert ok and lines[-1].startswith("K8 VERDICT PASS")
    # P22: the wikitext-calibrated pack, -0.078 on ONE text -> not corroborated
    ok, lines = verdict([(base_wt, _wt(7.89796))], calibrated=True,
                        calibration_domain="wikitext")
    assert not ok and "moves with the calibration text" in lines[-1]
    # an improvement only on the calibration-domain text is not corroborated
    ok, _ = verdict([(base_wt, _wt(7.93409)), (base_c4, _c4(15.40))],
                    calibrated=True, calibration_domain="wikitext")
    assert not ok
    # a regression beyond budget fails one-sided too
    ok, _ = verdict([(base_wt, _wt(7.97594 + 0.06)), (base_c4, _c4(15.40))],
                    calibrated=True, calibration_domain="c4val1")
    assert not ok


def test_mismatched_text_or_steps_never_compare():
    with pytest.raises(ValueError, match="different text"):
        verdict([(_wt(8.0), _c4(8.0))], calibrated=False)
    with pytest.raises(ValueError, match="step counts"):
        verdict([(Arm(8.0, WT, 8192), Arm(8.0, WT, 4096))], calibrated=False)
