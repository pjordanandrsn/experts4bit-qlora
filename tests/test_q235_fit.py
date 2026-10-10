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
    assert set(caps) == {"require-git", "create-venv", "install", "fetch", "tokens", "box", "repeat"}, caps
    assert all(v > 0 for v in caps.values())


def test_the_fit_scales_each_phase_explicitly_and_judges_caps_and_cost():
    f = _fit()
    timings = {"require-git": 2, "create-venv": 4, "install": 150, "fetch": 30, "tokens": 7, "box": 600, "repeat": 400}
    fetch = {"bytes": 27_047_001_621, "hash_seconds": 23.7}
    caps = {"require-git": 240, "create-venv": 120, "install": 1800, "fetch": 14400, "tokens": 300, "box": 30000, "repeat": 20000}
    rep = f.fit(timings, 178.0, fetch, 4_000_000_000, 470_211_497_053, 500.0, 2.2, caps)
    p = rep["phases"]
    assert p["box"]["scaled_s"] == 600 * 94 / 4 and p["repeat"]["scaled_s"] == 400 * 94 / 4
    assert abs(p["fetch"]["scaled_s"] - (470_211_497_053 / 62.5e6 + 470_211_497_053 / (27_047_001_621 / 23.7))) < 1
    assert p["install"]["scaled_s"] == 4_000_000_000 / 62.5e6 + 178.0
    assert p["tokens"]["scaled_s"] == 7 and p["box"]["budget_s"] == int(-(-600 * 94 / 4 * 1.5 // 1))
    assert p["box"]["fits"] and rep["usd_at_wallclock"] == round((sum(caps.values()) + 300 + 900) / 3600 * 2.2, 2)
    short = f.fit(timings, 178.0, fetch, 4_000_000_000, 470_211_497_053, 500.0, 2.2, {**caps, "box": 5400})
    assert not short["phases"]["box"]["fits"] and not short["fits"]
    dear = f.fit(timings, 178.0, fetch, 4_000_000_000, 470_211_497_053, 500.0, 6.0, caps)
    assert dear["usd_at_wallclock"] > f.RUN_CAP_USD and not dear["fits"]


def test_the_fit_refuses_a_rehearsal_that_is_not_a_clean_a2000_run(tmp_path):
    (tmp_path / "receipts").mkdir()
    (tmp_path / "receipts" / "run.json").write_text(json.dumps({"mode": "cpu"}))
    (tmp_path / "receipts" / "phase-timings.jsonl").write_text(json.dumps({"phase": "box", "elapsed_s": 1, "rc": 0}) + "\n")
    r = subprocess.run([sys.executable, str(Q / "q235_fit.py"), "--rehearsal", str(tmp_path), "--warm-install-s", "1",
                        "--bandwidth-mbps", "500", "--usd-per-hour", "2"], capture_output=True, text=True, timeout=60)
    assert r.returncode != 0 and "not a clean A2000 rehearsal" in r.stderr
