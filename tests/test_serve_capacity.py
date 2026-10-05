# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""The capacity model (:mod:`experts4bit_qlora.serve_capacity`): its plans are SC2's, it reproduces SC2b's measured
attainment from SC2b's own step costs, it reads costs from a step trace without inventing any, and its mechanics move
the right way. CPU only, no torch."""
import importlib.util
import pathlib

import pytest

from experts4bit_qlora.serve_capacity import StepCosts, Workload, ceiling, simulate

REPO = pathlib.Path(__file__).resolve().parents[1]


def _sc2_driver():
    spec = importlib.util.spec_from_file_location("sc2_driver", REPO / "bench" / "sc2" / "sc2_driver.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_plans_are_sc2_drivers_plans_exactly():
    drv = _sc2_driver()
    for rate, seed in ((1.0, 101), (8.0, 208), (None, 1)):
        mode = "serial" if rate is None else "poisson"
        assert Workload(n=120, rate=rate, seed=seed).plan() == drv.plan(mode, rate or 0.0, 120, seed, 64, 64, 256)


# SC2b's two ON servers (bench/h2h-2026-10-02/sc2b; sc2b-5090-1): prefill step = the request trace's admission ->
# first token p50 (serial); decode terms and first-decode claims from the bucket-controlled fit
# (bench/sc2/sc2c_census.py fit: decode 4.562 / 4.861 ms per token + 0.357 / 0.330 ms per bucket row; stall 0.218 /
# 0.224 s minus the prefill step). Measured attainment at 1 / 2 / 4 / 8 req/s from RESULTS-sc2b.md.
SC2B_ON = {1: (StepCosts(0.157, 0.055, 4.562e-3, 0.357e-3, source="sc2b on d1"), (1.00, 0.90, 0.12, 0.06)),
           2: (StepCosts(0.168, 0.055, 4.861e-3, 0.330e-3, source="sc2b on d2"), (1.00, 1.00, 0.29, 0.06))}


@pytest.mark.parametrize("draw", [1, 2])
def test_sc2bs_measured_attainment_is_reproduced_from_its_own_step_costs(draw):
    costs, measured = SC2B_ON[draw]
    got = [simulate(costs, Workload(n=120, rate=r, seed=draw * 100 + r)).attainment for r in (1, 2, 4, 8)]
    assert all(abs(g - m) <= 0.10 for g, m in zip(got, measured)), (got, measured)
    # and SC2b's reading of the ceiling, 1 req/s
    assert ceiling(costs, draws=(draw,))["ceiling"] == (1 if draw == 1 else 2)


def test_costs_are_read_from_a_step_trace_and_never_invented():
    rows = []
    for i in range(3):                                   # three prefill steps: 120, 110, 130 ms to the first token
        rows.append({"step": len(rows), "step_ms": 130.0 + 10 * i, "prefill_replays": 1, "prefill_tokens": 512,
                     "seg": {"ops": 0.1, "plan": 0.9, "pf_forward": 50.0 + 10 * (i % 2) - 10 * (i == 2), "pf_flush": 60.0,
                             "pf_sync": 9.0, "dec_issue": 10.0}})
    for b, ms in ((1, 5.0), (4, 6.0), (16, 10.0), (16, 10.0), (4, 6.0)):
        rows.append({"step": len(rows), "step_ms": ms, "decode_rows": b, "bucket": b, "seg": {"dec_ready": 0.0}})
    rows[3]["seg"]["dec_ready"] = 30.0                   # one first decode's claims: 30 ms over 3 prompts
    c = StepCosts.from_step_trace(rows)
    assert c.prefill_step_s == pytest.approx(0.120) and c.first_decode_s == pytest.approx(0.010)
    assert c.decode_base_s == pytest.approx(0.004666, rel=1e-3) and c.decode_per_row_s == pytest.approx(0.000333, rel=1e-2)
    assert "3 prefill steps, 5 decode-only steps, 3 prompts" in c.source
    with pytest.raises(ValueError, match="no prefill step"):
        StepCosts.from_step_trace(rows[3:])
    with pytest.raises(ValueError, match="no decode-only step"):
        StepCosts.from_step_trace(rows[:3])


def test_bad_costs_are_refused():
    with pytest.raises(ValueError, match="prefill_step_s"):
        StepCosts(-0.1, 0.0, 0.004, 0.0003)
    with pytest.raises(ValueError, match="buckets"):
        StepCosts(0.1, 0.0, 0.004, 0.0003, buckets=(4, 2))
    with pytest.raises(ValueError, match="max_seqs"):
        simulate(StepCosts(0.1, 0.0, 0.004, 0.0003), Workload(n=4, rate=1.0), max_seqs=0)


def test_the_mechanics_move_the_right_way():
    base = StepCosts(0.165, 0.055, 4.7e-3, 0.34e-3)
    cheap = StepCosts(0.050, 0.001, 4.7e-3, 0.34e-3)
    assert ceiling(cheap)["ceiling"] >= ceiling(base)["ceiling"]
    # more slots can only help a saturated server
    w = Workload(n=120, rate=8.0, seed=108)
    assert simulate(cheap, w, max_seqs=64).attainment >= simulate(cheap, w, max_seqs=16).attainment
    # serial TTFT is the prefill step plus the client's side
    s = simulate(cheap, Workload(n=6, rate=None, seed=1), http_s=0.003)
    assert all(t == pytest.approx(0.053) for t in s.ttft_s)
    # a 2048-token prompt takes four chunks: its first token comes three steps later
    long = simulate(cheap, Workload(n=6, rate=None, seed=1, prompt_tokens=2048), http_s=0.003)
    assert all(t == pytest.approx(0.053 * 4 - 0.003 * 3) for t in long.ttft_s)
    # an active set past the largest bucket runs as consecutive replays
    assert base.decode_step_s(24) == pytest.approx(2 * 4.7e-3 + 0.34e-3 * (16 + 8))
