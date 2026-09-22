"""P61's reducer applies the pre-registration literally: the fit recovers known coefficients, and each verdict branch
fires on the rows that should trigger it (bench/p61/p61_reduce.py; bench/p61/P61-PREREG.md)."""
import importlib.util
import pathlib

import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("p61_reduce", REPO / "bench" / "p61" / "p61_reduce.py")
red = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(red)

BPE = {"gate_up": 1536 * 1024 + 1536 * 64 * 2, "down": 2048 * 384 + 2048 * 24 * 2}


def _rows(b=(0.10, 0.05), c=(1.2, 0.6), a=(2.0, 1.5), l2_gbs=10000.0, shared=6.479, gap=None, removed=3500.0):
    grid = []
    for proj, bb, cc, aa in zip(BPE, b, c, a):
        for R in (16, 32, 64, 128):
            for D in (1, 2, 4, 8, 16, 32, 64, 128):
                if D <= R and R % D == 0:
                    t = aa + bb * R + cc * D
                    grid.append({"proj": proj, "R": R, "D": D, "graph_us_per_call": t + 0.3,
                                 "gemv_us_per_call": t, "reduce_us_per_call": 0.3})
    gap = sum(b) * removed / 1000.0 if gap is None else gap
    per_layer = {"served": {"step_ms_median": shared + 0.1, "gemv_kernel_ms_median": 6.0},
                 "dedup": {"step_ms_median": shared + 0.1 - gap, "gemv_kernel_ms_median": 5.0}}
    return {"device": "SYNTH", "sm_count": 170, "torch": "x", "l2_bytes": 96 << 20, "layers": 48, "steps": 128,
            "eids_sha256": "c050961e7f7d9f12", "plans": {}, "bytes_per_expert": BPE,
            "bandwidth": {"dram_copy_gbs": 1500.0, "l2_copy_gbs": l2_gbs, "l2_buf_bytes": 24 << 20},
            "grid": grid, "distinct_per_call": {"mean": 54.7, "min": 48, "max": 71, "rows_removed_per_step_mean": removed},
            "recorded": {"per_layer": per_layer,
                         "shared": {"served": {"step_ms_median": shared, "gemv_kernel_ms_median": 5.9},
                                    "dedup": {"step_ms_median": shared - 0.9, "gemv_kernel_ms_median": 4.9}}}}


def test_fit_recovers_known_coefficients():
    v = red.verdicts(_rows())
    a, b, c, r2, rel = v["proj"]["gate_up"]["fit_gemv"]
    assert (a, b, c) == pytest.approx((2.0, 0.10, 1.2), abs=1e-9)
    assert r2 == pytest.approx(1.0) and rel == pytest.approx(0.0, abs=1e-9)
    assert v["proj"]["down"]["b1"] == pytest.approx(0.05)


def test_p0_gate_refuses_everything():
    v = red.verdicts(_rows(shared=7.0))
    assert not v["p0"] and v["decision"].startswith("¬P0")


def test_p1_bands():
    assert red.verdicts(_rows())["p1"] == "HELD"
    assert red.verdicts(_rows(gap=0.525 * 1.4))["p1"] == "NOT HELD"          # predicted 0.525 vs measured 0.735
    v = red.verdicts(_rows(gap=0.525 * 3))
    assert v["p1"] == "REFUTED" and "no lever lane" in v["decision"]


@pytest.mark.parametrize("l2_gbs,branch", [(1.0e5, "NOT-BYTES"), (1.2e4, "L2-BOUND"), (2.5e4, "MIXED")])
def test_p2_branches(l2_gbs, branch):
    # gate_up floor = 1769472 B / l2 -> at 1e5 GB/s 0.0177 us (b1 0.10 -> 5.7x), at 1.2e4 0.147 us (0.68x);
    # down floor = 884736 B / l2 -> at 1.2e4 0.0737 us (b1 0.05 -> 0.68x); at 2.5e4: gate_up 1.41x, down 1.41x -> MIXED
    v = red.verdicts(_rows(l2_gbs=l2_gbs))
    assert v["p2"] == branch and branch in v["decision"]


def test_render_runs():
    assert "Decision rule" in red.render(_rows())
