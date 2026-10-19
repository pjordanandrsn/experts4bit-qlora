"""bench/hp2/hp2_reduce.py (HP2-PREREG): the G1 gate on fixtures built from REAL calibration records, and the verdict."""

import json
import math
import random
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "bench" / "hp2"))
import hp2_reduce as R  # noqa: E402

REAL = json.loads((REPO / "tests" / "fixtures" / "hp1" / "eval_cal_hp1cal-5090-2.json").read_text())
REAL_GAPS = REAL["bases"][0]["zero_adapter_row_gaps"]["hp1cal_k00_s3407"]      # 48 real per-row gaps (e4b, 48 layers)
N = len(REAL_GAPS)


def _cal(e4b_gaps, uns_gaps, digest_ok=True):
    return {"bases": [
        {"dir": "hp1/base_e4b_qwen3hp2", "digest_matches_training_run": digest_ok, "status": "OK",
         "zero_adapter_row_gaps": {R.CAL_TAGS["e4b"]: e4b_gaps}},
        {"dir": "hp1/base_unsloth_qwen3hp2", "digest_matches_training_run": True, "status": "OK",
         "zero_adapter_row_gaps": {R.CAL_TAGS["unsloth"]: uns_gaps}}]}


def _tracking(seed=1, noise=0.003, shift=0.0):
    """An Unsloth side that shares the real rows' structure (row-paired): the real gaps plus small independent noise."""
    rng = random.Random(seed)
    return [round(g + shift + rng.gauss(0, noise), 6) for g in REAL_GAPS]


def test_g1_passes_when_the_frameworks_share_the_real_rows_structure():
    rec = R.g1(_cal(REAL_GAPS, _tracking(noise=0.001)), rows=N)
    assert rec["verdict"] == "PASS" and rec["reach"] <= R.G1_BOUND and rec["null"] == 0.0 and rec["m_need"] <= R.M_EXPECTED
    assert rec["se_delta0"] == pytest.approx(rec["sd_d"] / math.sqrt(N), abs=1e-6)


def test_g1_closes_when_the_pairing_does_not_cancel_the_real_per_row_spread():
    """The same real gaps in another row order: the per-row spread (sd 0.030) no longer cancels, SE ~0.0062, G1 closes."""
    shuffled = REAL_GAPS[:]
    random.Random(7).shuffle(shuffled)
    rec = R.g1(_cal(REAL_GAPS, shuffled), rows=N)
    assert rec["verdict"] == "CLOSE" and rec["reach"] > R.G1_BOUND and "not a strike" in rec["reason"]


def test_g1_closes_on_a_real_mean_offset_between_the_frameworks():
    rec = R.g1(_cal(REAL_GAPS, _tracking(shift=-0.012)), rows=N)          # Unsloth's base read 0.012 lower on every row
    assert rec["verdict"] == "CLOSE" and rec["delta0"] == pytest.approx(0.012, abs=0.002)


def test_g1_records_the_zero_centred_6b_tolerance_and_reports_each_z_band():
    """#1793 ruling: 6(b) is centred on each run's own z_f at the registered 0.01 (AXO1767's MATCH band); z_f - b_f is reported
    against 3 x sd_f x sqrt(1/48 + 1/R) (the spread of a 48-row mean about an R-row mean), never a VOID."""
    rec = R.g1(_cal(REAL_GAPS, _tracking(shift=0.004)), rows=N)
    assert rec["roundtrip_zero_centred_tol"] == R.ROUNDTRIP_MATCH == 0.01
    for fw in ("e4b", "unsloth"):
        v = rec["per_framework"][fw]
        half = 3 * v["sd"] * math.sqrt(1 / R.SCORED_ROWS + 1 / N)
        assert rec["z_bands"][fw][0] == v["b"] and rec["z_bands"][fw][1] == pytest.approx(half, abs=2e-6)
        assert rec["z_bands"][fw][1] >= half - 1e-6
    assert rec["z_bands"]["unsloth"][0] == pytest.approx(rec["z_bands"]["e4b"][0] + 0.004, abs=0.002)
    assert "never a VOID" in rec["false_void"]


def test_g1_stops_when_m_need_exceeds_the_expected_margin_though_the_hard_bound_passes():
    """#1793 ruling: record m_need = 2(|delta0| + 2 SE); if it exceeds the expected margin (the 0.005 floor), the box stops at G1
    rather than spending on a G2 closure that could be predicted."""
    rec = R.g1(_cal(REAL_GAPS, _tracking(noise=0.012)), rows=N)
    assert rec["reach"] <= R.G1_BOUND and rec["m_need"] > R.M_EXPECTED and rec["verdict"] == "STOP" and "not a strike" in rec["reason"]


@pytest.mark.parametrize("ev, why", [
    (_cal(REAL_GAPS, _tracking(), digest_ok=False), "digest"),
    (_cal(REAL_GAPS[:47], _tracking()), "finite per-row gaps"),
    ({"bases": []}, "exactly one base"),
])
def test_g1_refuses_unusable_input(ev, why):
    with pytest.raises(ValueError, match=why):
        R.g1(ev, rows=N)


def test_the_g1_stop_is_an_exit_code_and_its_record_is_written_first(tmp_path):
    src, out = tmp_path / "eval_cal.json", tmp_path / "g1.json"
    src.write_text(json.dumps(_cal(REAL_GAPS, _tracking(noise=0.001))))
    assert R.main(["g1", str(src), "--rows", str(N), "--out", str(out)]) == R.EXIT_OK and json.loads(out.read_text())["verdict"] == "PASS"
    shuffled = REAL_GAPS[:]
    random.Random(7).shuffle(shuffled)
    src.write_text(json.dumps(_cal(REAL_GAPS, shuffled)))
    assert R.main(["g1", str(src), "--rows", str(N), "--out", str(out)]) == R.EXIT_G1_CLOSE == 4
    assert json.loads(out.read_text())["verdict"] == "CLOSE"
    src.write_text("{}")
    assert R.main(["g1", str(src), "--rows", str(N), "--out", str(tmp_path / "x.json")]) == R.EXIT_UNUSABLE


def test_the_registered_row_count_is_300():
    with pytest.raises(ValueError, match="300 finite"):
        R.g1(_cal(REAL_GAPS, _tracking()))


# ------------------------------------------------------------------------------------------------------------- verdict
G1_PASS = {"verdict": "PASS", "delta0": 0.0004, "se_delta0": 0.0009, "m_need": 0.0044}   # |delta0| + 2 SE = 0.0022 <= m/2 at the floor


def _arm(fw, seed, y, rho=0.001, status="SCORED", order_sha=None):
    tag = f"hp2_k00_s{seed}" if fw == "e4b" else f"hp2_unsloth_s{seed}"
    return {"tag": tag, "seed": seed, "status": status, "order": {"order_seed": seed, "order_sha": order_sha or f"o{seed}"},
            "evaluator": {"row_mean": y} if status == "SCORED" else None, "evaluator_minus_inloop_final": rho}


def _ev(deltas, base=0.86, spread=(0.0, 0.004, -0.004), rho_e=(0.001, 0.0012, 0.0008), rho_u=(0.0011, 0.0009, 0.001)):
    arms = []
    for i, s in enumerate(R.SEEDS):
        u = base + spread[i]
        arms += [_arm("unsloth", s, u, rho_u[i]), _arm("e4b", s, u + deltas[i], rho_e[i])]
    bases = [{"dir": "/root/tc1/hp1/base_e4b_qwen3hp2", "zero_adapter": {"row_mean": 1.9210}},
             {"dir": "/root/tc1/hp1/base_unsloth_qwen3hp2", "zero_adapter": {"row_mean": 1.9172}}]
    return {"arms": arms, "bases": bases}


def test_z_minus_b_is_reported_per_framework_never_gated():
    ev = _ev((0.0002, -0.0003, 0.0001))
    for a in ev["arms"]:
        a["zero_gap"] = 0.0070 if a["tag"].startswith("hp2_k00") else 0.0130
    g = dict(G1_PASS, z_bands={"e4b": [-0.0013, 0.014], "unsloth": [0.0300, 0.014]})   # unsloth: |z| inside, |z - b| = 0.017 outside
    rec, code = R.verdict([ev], g)
    assert rec["z_vs_b"]["e4b"]["z_minus_b"] == pytest.approx(0.0083) and rec["z_vs_b"]["e4b"]["inside"]
    assert rec["z_vs_b"]["unsloth"]["inside"] is False and code == 0 and rec["verdict"] == "PARITY"   # reported, not a gate


def test_delta_base_is_reported_beside_delta():
    rec, _ = R.verdict([_ev((0.0002, -0.0003, 0.0001))], G1_PASS)
    assert rec["delta_base"] == pytest.approx(0.0038, abs=1e-6)


def test_a_null_read_inside_the_margin_is_parity_worded_as_no_measurable_difference():
    rec, code = R.verdict([_ev((0.0002, -0.0003, 0.0001))], G1_PASS)
    assert code == 0 and rec["verdict"] == "PARITY" and "no measurable difference" in rec["wording"] and "equal" not in rec["wording"]
    assert rec["headline"]["ci90"][0] >= -rec["m"] and rec["headline"]["ci90"][1] <= rec["m"]


def test_a_clear_lower_loss_reads_beats():
    rec, code = R.verdict([_ev((-0.03, -0.031, -0.029))], G1_PASS)
    assert code == 0 and rec["verdict"] == "BEATS"


def test_a_wide_null_read_is_inconclusive_with_its_interval():
    rec, code = R.verdict([_ev((0.004, -0.004, 0.012), spread=(0.0, 0.0005, -0.0005))], G1_PASS)
    assert rec["verdict"] == "INCONCLUSIVE" and "no measurable effect, with its interval" in rec["wording"]


def test_g2_closes_when_the_trained_differential_is_wider_than_half_the_margin():
    rec, code = R.verdict([_ev((0.0, 0.0, 0.0), rho_e=(0.01, -0.01, 0.012))], G1_PASS)
    assert code == 3 and rec["verdict"] == "CLOSED_G2" and not rec["g2"]["deltaN_within_half_m"]
    assert "could not read margin" in rec["wording"]


def test_g2_also_holds_delta0_to_half_the_realized_margin():
    rec, code = R.verdict([_ev((0.0, 0.0, 0.0))], dict(G1_PASS, delta0=0.004, se_delta0=0.001))   # 0.006 > m/2 = 0.0025
    assert code == 3 and rec["verdict"] == "CLOSED_G2" and not rec["g2"]["delta0_within_half_m"]


def test_a_g1_closure_reads_no_headline():
    rec, code = R.verdict([_ev((0.0, 0.0, 0.0))], dict(G1_PASS, verdict="CLOSE"))
    assert code == 3 and rec["verdict"] == "G1_CLOSED" and "headline" not in rec


def test_a_missing_or_void_run_is_incomplete():
    ev = _ev((0.0, 0.0, 0.0))
    ev["arms"][1]["status"] = "VOID"
    rec, code = R.verdict([ev], G1_PASS)
    assert code == 3 and rec["verdict"] == "INCOMPLETE" and any("e4b seed 3410" in m for m in rec["missing"])


def test_runs_that_do_not_share_one_data_order_are_incomplete():
    ev = _ev((0.0, 0.0, 0.0))
    ev["arms"][0]["order"]["order_sha"] = "different"
    rec, code = R.verdict([ev], G1_PASS)
    assert code == 3 and rec["verdict"] == "INCOMPLETE" and rec["order_problems"]
