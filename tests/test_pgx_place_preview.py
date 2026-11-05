"""PGX Z4: E-hyb's placement at the registered budgets, from the real solver on the committed fit profile (CPU)."""
import json
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "bench", "pgx"))
import pgx_place_preview as pp  # noqa: E402

PROFILE = os.path.join(ROOT, "bench", "pgx", "fit_profile.jsonl")


def _calib(scatter_gbs):
    return {"cpu_bench": {"scatter_best": {"gbs": scatter_gbs}},
            "gpu_bench": {"devices": [{"b_vram_triad_gbs": 1571.7}]}}


def test_vram_knob_is_the_decimal_budget_in_gib():
    assert [pp.vram_gib(b) for b in (4, 8, 12)] == [3.725, 7.451, 11.176]


@pytest.fixture(scope="module", params=[496.48, 60.0])
def preview(request):
    return request.param, pp.preview(_calib(request.param), PROFILE)


def test_every_placement_fits_its_budget_with_the_nvme_tier_empty(preview):
    _, p = preview
    for b in (4, 8, 12):
        for arm in ("E-hyb", "E-hyb-u"):
            v = p[f"{arm}|{b}"]
            assert v["vram_bytes"] <= v["budget_bytes"], (arm, b)
            assert v["nvme_experts"] == 0 and v["vram_experts"] + v["dram_experts"] == 48 * 128
        assert p[f"E-hyb|{b}"]["profile_kind"] == "measured"
        assert p[f"E-hyb-u|{b}"]["profile_kind"] == "uniform-assumed"


def test_the_profile_beats_uniform_and_stays_under_the_ceiling(preview):
    _, p = preview
    for b in (4, 8, 12):
        e, u, c = (p[f"{k}|{b}"]["vram_mass_share"] for k in ("E-hyb", "E-hyb-u", "ceiling"))
        assert u < e <= c + 1e-12, b


def test_a_slower_host_cpu_keeps_more_mass_on_the_gpu():
    """The solver balances GPU against CPU time: the host's scatter bandwidth moves E-hyb's coverage, which is why the
    registration computes E-hyb's expectation from the box's own calib.json."""
    fast = pp.preview(_calib(496.48), PROFILE)
    slow = pp.preview(_calib(60.0), PROFILE)
    for b in (8, 12):
        assert slow[f"E-hyb|{b}"]["vram_mass_share"] > fast[f"E-hyb|{b}"]["vram_mass_share"] + 0.05


def test_main_writes_the_preview_and_refuses_nothing_at_the_registered_budgets(tmp_path):
    c = tmp_path / "calib.json"
    c.write_text(json.dumps(_calib(60.0)))
    out = tmp_path / "p.json"
    assert pp.main(["--calib", str(c), "--out", str(out)]) == 0
    assert json.loads(out.read_text())["E-hyb|8"]["profile_kind"] == "measured"
