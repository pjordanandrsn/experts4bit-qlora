"""bench/hp1/hp1_calib.py (HP1-PREREG Amendment 4): the 48-layer zero-adapter tolerance from a calibration that can fail."""

import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "bench" / "hp1"))
import hp1_calib  # noqa: E402

TAG = "hp1cal_k00_s3407"
REAL_9 = REPO / "tests" / "fixtures" / "hp1" / "eval_early_hp1k-5090-9.json"


def _ev(gaps, digest_ok=True, tag=TAG):
    return {"bases": [{"dir": "/root/tc1/hp1/base_e4b_qwen3hp1cal", "status": "VOID", "digest_matches_training_run": digest_ok,
                       "zero_adapter_row_gaps": {tag: gaps}}]}


def _drift(center=0.006, spread=0.002, n=48):
    """Per-row gaps that look like numeric drift: all near one value."""
    return [round(center + spread * ((i % 7) - 3) / 3, 6) for i in range(n)]


def test_drift_like_gaps_pass_with_tol_twice_the_mean_abs_gap():
    gaps = _drift(0.003, 0.001)
    res = hp1_calib.calibrate(_ev(gaps), TAG, ceiling=0.01)
    mean = sum(abs(g) for g in gaps) / len(gaps)
    assert res["verdict"] == "PASS" and res["tol"] == pytest.approx(2 * mean, abs=1e-6) and res["tol"] >= 2 * mean
    assert res["n_rows"] == 48 and res["max_over_median"] <= hp1_calib.SHAPE_FACTOR


def test_one_outlier_row_is_a_path_difference_and_closes():
    gaps = _drift(0.003, 0.0005)
    gaps[17] = 0.05                                    # one row 15x the rest: template / truncation / masking, not numerics
    res = hp1_calib.calibrate(_ev(gaps), TAG, ceiling=0.01)
    assert res["verdict"] == "CLOSE_SHAPE" and "row 17" in res["reason"]


def test_tol_above_the_pre_box_ceiling_closes():
    res = hp1_calib.calibrate(_ev(_drift(0.008, 0.002)), TAG, ceiling=0.01)   # tol ~ 0.016 > 0.01
    assert res["verdict"] == "CLOSE_CEILING" and res["tol"] > 0.01


def test_the_shape_check_runs_before_the_ceiling():
    gaps = _drift(0.008, 0.001)
    gaps[3] = 0.2
    assert hp1_calib.calibrate(_ev(gaps), TAG, ceiling=0.01)["verdict"] == "CLOSE_SHAPE"


@pytest.mark.parametrize("ev, why", [
    (_ev(_drift()[:47]), "48 finite"),                                    # a row short
    (_ev(_drift()[:47] + [float("nan")]), "48 finite"),                   # a non-finite gap
    (_ev(_drift(), digest_ok=False), "digest"),                           # the base is not the training run's
    (_ev(_drift(), tag="other"), "exactly one base"),                     # no gaps for this tag
    ({"bases": []}, "exactly one base"),
])
def test_unusable_input_refuses_rather_than_guessing(ev, why):
    with pytest.raises(ValueError, match=why):
        hp1_calib.calibrate(ev, TAG, ceiling=0.01)


def test_hp1k_5090_9s_real_early_check_refuses_cleanly(tmp_path, capsys):
    """The receipt from the draw that motivated Amendment 4 carries row_mean only (no per-row losses): the calibration must refuse it,
    not derive a tolerance from one number."""
    ev = json.loads(REAL_9.read_text())
    assert ev["bases"][0]["zero_adapter"]["row_mean"] == 1.921029 and "zero_adapter_row_gaps" not in ev["bases"][0]
    with pytest.raises(ValueError, match="exactly one base"):
        hp1_calib.calibrate(ev, "hp1_k00_s3407", ceiling=0.01)
    out = tmp_path / "calib.json"
    rc = hp1_calib.main([str(REAL_9), "--tag", "hp1_k00_s3407", "--ceiling", "0.01", "--out", str(out)])
    assert rc == 2 and not out.exists() and capsys.readouterr().out.startswith("HP1_CALIB=UNUSABLE")


def test_cli_writes_the_verdict_and_exits_by_it(tmp_path, capsys):
    src = tmp_path / "eval_cal.json"
    src.write_text(json.dumps(_ev(_drift(0.003, 0.001))))
    out = tmp_path / "calib.json"
    assert hp1_calib.main([str(src), "--tag", TAG, "--ceiling", "0.01", "--out", str(out)]) == 0
    assert json.loads(out.read_text())["verdict"] == "PASS" and capsys.readouterr().out.startswith("HP1_CALIB=PASS tol=")
    src.write_text(json.dumps(_ev(_drift(0.008, 0.002))))
    assert hp1_calib.main([str(src), "--tag", TAG, "--ceiling", "0.01", "--out", str(out)]) == 3
