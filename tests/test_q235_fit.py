# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""Q235's gate 2 (bench/q235/q235_fit.py): each time-left guard's budget is scaled from the A2000 rehearsal to the registered model,
and the sealed caps, the wallclock and the run's cost must fit it."""
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
Q = ROOT / "bench" / "q235"


def _mod(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _fit():
    return _mod("q235_fit", Q / "q235_fit.py")


def test_the_fit_reads_every_sealed_cap_from_the_runner():
    f = _fit()
    caps = f.sealed_caps(Q / "q235_run.sh")
    assert set(caps) == {"require-git", "create-venv", "install", "hostrate", "fetch", "tokens", "box", "repeat"}, caps
    assert all(v > 0 for v in caps.values())


def test_the_fit_scales_each_phase_explicitly_and_judges_caps_and_cost():
    f = _fit()
    timings = {"require-git": 2, "create-venv": 4, "install": 150, "hostrate": 20, "fetch": 30, "tokens": 7, "box": 600, "repeat": 400}
    fetch = {"bytes": 27_047_001_621, "hash_seconds": 23.7}
    caps = {"require-git": 240, "create-venv": 120, "install": 1800, "hostrate": 600, "fetch": 14400, "tokens": 300, "box": 30000,
            "repeat": 20000}
    rep = f.fit(timings, 178.0, fetch, 4_000_000_000, 470_211_497_053, 500.0, 1.6, caps)
    p = rep["phases"]
    assert p["box"]["scaled_s"] == 600 * 94 / 4 and p["repeat"]["scaled_s"] == 400 * 94 / 4
    assert abs(p["fetch"]["scaled_s"] - (470_211_497_053 / 62.5e6 + 470_211_497_053 / (27_047_001_621 / 23.7))) < 1
    assert p["install"]["scaled_s"] == 4_000_000_000 / 62.5e6 + 178.0
    assert p["tokens"]["scaled_s"] == 7 and p["box"]["budget_s"] == int(-(-600 * 94 / 4 * 1.5 // 1))
    assert p["box"]["fits"] and rep["usd_at_wallclock"] == round((sum(caps.values()) + 300 + 1800) / 3600 * 1.6, 2)
    short = f.fit(timings, 178.0, fetch, 4_000_000_000, 470_211_497_053, 500.0, 1.6, {**caps, "box": 5400})
    assert not short["phases"]["box"]["fits"] and not short["fits"]
    dear = f.fit(timings, 178.0, fetch, 4_000_000_000, 470_211_497_053, 500.0, 6.0, caps)
    assert dear["usd_at_wallclock"] > f.RUN_CAP_USD and not dear["fits"] and dear["cost_tier"] == "above the per-run cap"
    assert rep["fits"] and rep["cost_tier"].startswith("above the no-ask tier") and f.NO_ASK_USD < rep["usd_at_wallclock"] <= 35
    cheap = f.fit(timings, 178.0, fetch, 4_000_000_000, 470_211_497_053, 500.0, 0.5, caps)
    assert cheap["cost_tier"] == "within the no-ask tier" and cheap["fits"]


def test_the_config_bytes_are_the_locked_checkpoint():
    """The slice and full tensor bytes the disk floor uses come from the config; they must add up to the locked shards."""
    f = _fit()
    lock = json.loads((Q / "model_files.json").read_text())
    full = f.LAYERS * f.LAYER_BYTES + f.OUTSIDE_BYTES
    assert abs(lock["safetensors_bytes"] - full) / full < 1e-4        # the rest is the shards' safetensors headers


def test_the_floors_are_the_rehearsal_s_link_and_its_load_s_effective_disk_rate():
    f = _fit()
    hr = {"h2d_gb_s": 21.5, "host_copy_gb_s": 5.14, "disk_read_gb_s": 1.9, "disk_method": "O_DIRECT (dd iflag=direct)",
          "pcie_gen": "4", "pcie_width": "16"}
    fl = f.floors(hr, 534.0)                                                   # no profile: bytes over the whole load
    assert fl["h2d_gb_s"] == 21.5 and fl["host_copy_gb_s"] == 5.14
    assert fl["disk_read_gb_s"] == round((4 * f.LAYER_BYTES + f.OUTSIDE_BYTES) / 534.0 / 1e9, 3)
    assert fl["basis"]["rehearsal_direct_read_gb_s"] == 1.9                  # reported beside, never the floor
    prof = {"read": 30.12, "assemble": 65.68, "quantize": 1.14, "offload": 7.38, "other_s": 11.51, "read_bytes": 22391367680}
    fl = f.floors(hr, 126.2, prof)                                             # with the load profile: the reads' own rate
    assert fl["disk_read_gb_s"] == round(22391367680 / 30.12 / 1e9, 3) and fl["basis"]["dominant"] == "assemble"
    assert set(f.sealed_floors(Q / "q235_run.sh")) == {"h2d_gb_s", "disk_read_gb_s", "host_copy_gb_s", "fetch_gb_s"}


def test_the_fit_refuses_a_rehearsal_that_is_not_a_clean_a2000_run(tmp_path):
    (tmp_path / "receipts").mkdir()
    (tmp_path / "receipts" / "run.json").write_text(json.dumps({"mode": "cpu"}))
    (tmp_path / "receipts" / "phase-timings.jsonl").write_text(json.dumps({"phase": "box", "elapsed_s": 1, "rc": 0}) + "\n")
    r = subprocess.run([sys.executable, str(Q / "q235_fit.py"), "--rehearsal", str(tmp_path), "--warm-install-s", "1",
                        "--bandwidth-mbps", "500", "--usd-per-hour", "2"], capture_output=True, text=True, timeout=60)
    assert r.returncode != 0 and "not a clean A2000 rehearsal" in r.stderr


def test_the_pull_floor_is_the_rate_that_fits_the_fetch_cap():
    f = _fit()
    fetch = {"bytes": 27_047_001_621, "hash_seconds": 23.7}
    model = 470_211_497_053
    floor = f.fetch_floor(model, fetch, 12000)
    hash_s = model / (fetch["bytes"] / fetch["hash_seconds"])
    assert floor >= model / (12000 / f.MARGIN - hash_s) / 1e9 and floor - model / (12000 / f.MARGIN - hash_s) / 1e9 < 1e-3
    assert f.fetch_floor(model, fetch, 300) == float("inf")                  # no time left after hashing: no rate fits
    assert f.sealed_floors(Q / "q235_run.sh")["fetch_gb_s"] >= floor           # the sealed floor is at least the fitted one
