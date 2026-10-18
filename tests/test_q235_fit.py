# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""Q235's gate 2 (bench/q235/q235_fit.py): each time-left guard's budget is scaled from the A2000 rehearsal to the registered model,
and the sealed caps, the wallclock and the run's cost must fit it."""
from __future__ import annotations

import importlib.util
import json
import math
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

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
    assert set(caps) == {"require-git", "create-venv", "install", "hostrate", "fetch", "tokens", "box"}, caps   # repeat: Amendment 15
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


def _rehearsal_from_receipts(d):
    """gate2/'s committed A2000 receipts laid out as a rehearsal workspace (the box record carries only its load)."""
    g = Q / "gate2"
    (d / "receipts" / "box").mkdir(parents=True)
    for name in ("run.json", "phase-timings.jsonl", "hostrate.json"):
        (d / "receipts" / name).write_text((g / name).read_text())
    (d / "warm-fetch.json").write_text((g / "warm-fetch.json").read_text())
    load = json.loads((g / "a2000-leg-summary.json").read_text())["load"]
    (d / "receipts" / "box" / "box.json").write_text(json.dumps({"load": load}))
    return d


def test_the_in_process_plants_add_only_their_measured_forwards_to_the_repeat(tmp_path):
    """Amendment 3: the repeat's budget is the rehearsal's repeat plus the two plant forwards a later A2000 leg measured, scaled."""
    reh = _rehearsal_from_receipts(tmp_path / "reh")
    later = tmp_path / "later"
    (later / "receipts" / "repeat").mkdir(parents=True)
    (later / "receipts" / "run.json").write_text(json.dumps({"mode": "a2000", "e4b_sha": "f" * 40, "nonce": "n"}))
    (later / "receipts" / "repeat" / "box.json").write_text(json.dumps(
        {"plants": {"routing": {"forward_s": 1.689}, "epilogue": {"forward_s": 1.693}}}))
    cmd = [sys.executable, str(Q / "q235_fit.py"), "--rehearsal", str(reh), "--warm-install-s", "178", "--bandwidth-mbps", "500",
           "--usd-per-hour", "1.6"]
    r = subprocess.run(cmd + ["--plants-from", str(later)], capture_output=True, text=True, timeout=60)
    rep = json.loads(r.stdout)
    base = {x["phase"]: x["elapsed_s"] for x in map(json.loads, (Q / "gate2" / "phase-timings.jsonl").read_text().splitlines())}
    assert abs(rep["phases"]["repeat"]["scaled_s"] - round((base["repeat"] + 3.382) * 94 / 4, 1)) < 0.1, rep["phases"]["repeat"]
    assert rep["rehearsal"]["in_process_plants"]["added_to_repeat_s"] == 3.382
    assert rep["phases"]["repeat"]["budget_s"] == 4702        # the A2000 record, reported: Amendment 15 sizes the repeat on the box
    assert rep["phases"]["repeat"]["sealed_cap_s"] is None and rep["phases"]["repeat"]["fits"]
    (later / "receipts" / "repeat" / "box.json").write_text(json.dumps({"repeat": {"loss": 1.0}}))
    r = subprocess.run(cmd + ["--plants-from", str(later)], capture_output=True, text=True, timeout=60)
    assert r.returncode != 0 and "ran both in-process plants" in r.stderr


def _leg(d, box_s, load_s, mode="a2000", sha="e"):
    """An A2000 leg's receipts: its run record, its phases (rc 0) and its box record's load."""
    (d / "receipts" / "box").mkdir(parents=True, exist_ok=True)
    (d / "receipts" / "run.json").write_text(json.dumps({"mode": mode, "e4b_sha": sha * 40, "nonce": f"n-{sha}"}))
    (d / "receipts" / "phase-timings.jsonl").write_text(
        "\n".join(json.dumps({"phase": p, "elapsed_s": t, "rc": 0}) for p, t in (("box", box_s), ("repeat", 50.0))) + "\n")
    (d / "receipts" / "box" / "box.json").write_text(json.dumps({"load": {"seconds": load_s}}))
    return d


def test_the_changed_step_adds_only_its_warm_against_warm_increment_to_the_cold_box(tmp_path):
    """Amendments 6 and 7 (the maintainer's rule): the rehearsal keeps its whole cold box phase, and only the change is added,
    measured warm against warm: max(0, (later phase - later load) - (baseline phase - baseline load))."""
    reh = _rehearsal_from_receipts(tmp_path / "reh")
    own = {x["phase"]: x["elapsed_s"] for x in map(json.loads, (Q / "gate2" / "phase-timings.jsonl").read_text().splitlines())}["box"]
    base = _leg(tmp_path / "base", 136.0, 56.8, sha="9")                 # outside the load: 79.2 s (the am4 leg's own numbers)
    cmd = [sys.executable, str(Q / "q235_fit.py"), "--rehearsal", str(reh), "--warm-install-s", "178", "--bandwidth-mbps", "500",
           "--usd-per-hour", "1.6", "--box-baseline", str(base)]

    def run(later):
        return subprocess.run(cmd + ["--box-from", str(later)], capture_output=True, text=True, timeout=60)
    later = _leg(tmp_path / "later", 136.0 + 10.0, 56.8, sha="e")         # 10 s over the baseline outside the load
    rep = json.loads(run(later).stdout)
    step = rep["rehearsal"]["changed_step"]
    assert step["increment_s"] == 10.0 and step["own_box_phase_s"] == own and step["box_basis_s"] == round(own + 10.0, 3)
    assert step["baseline_e4b_sha"] == "9" * 40 and step["from_e4b_sha"] == "e" * 40
    assert abs(rep["phases"]["box"]["scaled_s"] - round((own + 10.0) * 94 / 4, 1)) < 0.1, rep["phases"]["box"]
    later = _leg(tmp_path / "later", 140.0, 70.0)                          # 70 s outside the load, under the baseline's 79.2
    assert json.loads(run(later).stdout)["rehearsal"]["changed_step"]["increment_s"] == 0.0
    later = _leg(tmp_path / "later", 136.0, 56.8)                          # exactly the baseline
    assert json.loads(run(later).stdout)["rehearsal"]["changed_step"]["box_basis_s"] == round(own, 3)


def test_the_changed_step_refuses_by_name(tmp_path):
    """Each refusal names itself (no traceback): a leg that is not a clean A2000 run, no box phase, no load.seconds, a box phase
    not longer than its load, and --box-from without its baseline or the other way round."""
    reh = _rehearsal_from_receipts(tmp_path / "reh")
    base = _leg(tmp_path / "base", 136.0, 56.8, sha="9")
    cmd = [sys.executable, str(Q / "q235_fit.py"), "--rehearsal", str(reh), "--warm-install-s", "178", "--bandwidth-mbps", "500",
           "--usd-per-hour", "1.6"]

    def refused(extra, msg):
        r = subprocess.run(cmd + extra, capture_output=True, text=True, timeout=60)
        assert r.returncode != 0 and msg in r.stderr and "Traceback" not in r.stderr, (extra, r.stderr[-400:])
    later = _leg(tmp_path / "later", 146.0, 56.8)
    refused(["--box-from", str(later)], "--box-from and --box-baseline go together")
    refused(["--box-baseline", str(base)], "--box-from and --box-baseline go together")
    both = ["--box-from", str(later), "--box-baseline", str(base)]
    (later / "receipts" / "phase-timings.jsonl").write_text(json.dumps({"phase": "repeat", "elapsed_s": 50.0, "rc": 0}) + "\n")
    refused(both, "--box-from has no box phase in its phase timings")
    _leg(later, 146.0, 56.8)
    (later / "receipts" / "box" / "box.json").write_text(json.dumps({"steps": {}}))
    refused(both, "--box-from's box record carries no load.seconds")
    _leg(later, 50.0, 56.8)
    refused(both, "--box-from's box phase is not longer than its own load")
    _leg(later, 146.0, 56.8, mode="cpu")
    refused(both, "--box-from is not a clean A2000 leg")
    _leg(later, 146.0, 56.8)
    (base / "receipts" / "box" / "box.json").unlink()
    refused(both, "--box-baseline's box record carries no load.seconds")


def _leg_from_gate2(d, tag):
    """A gate2/ A2000 leg (run-TAG.json, phase-timings-TAG.jsonl, a2000-leg-TAG-summary.json) laid out as receipts."""
    g = Q / "gate2"
    (d / "receipts" / "box").mkdir(parents=True)
    (d / "receipts" / "repeat").mkdir()
    (d / "receipts" / "run.json").write_text((g / f"run-{tag}.json").read_text())
    (d / "receipts" / "phase-timings.jsonl").write_text((g / f"phase-timings-{tag}.jsonl").read_text())
    s = json.loads((g / f"a2000-leg-{tag}-summary.json").read_text())
    (d / "receipts" / "box" / "box.json").write_text(json.dumps({"load": s["box"]["load"]}))
    (d / "receipts" / "repeat" / "box.json").write_text(json.dumps({"plants": s["repeat"]["plants"]}))
    return d


def test_amendment_7s_refit_is_kept_as_its_record(tmp_path):
    """Amendments 6 and 7's refit, from the committed gate2/ legs: the first leg's whole box phase plus the change measured warm
    against warm (f17f0744 against the Amendment 4 leg, 222.0 - 79.2 = 142.8 s), scaled to 94 layers with the margin, is 13,847 s.
    Amendment 12 seals the box cap from the RTX PRO 6000 leg instead (2,997 s), so the A2000 refit no longer fits the sealed caps."""
    reh = _rehearsal_from_receipts(tmp_path / "reh")
    cmd = [sys.executable, str(Q / "q235_fit.py"), "--rehearsal", str(reh), "--warm-install-s", "178", "--bandwidth-mbps", "3000",
           "--usd-per-hour", "1.70", "--plants-from", str(_leg_from_gate2(tmp_path / "a3", "a3")),
           "--box-baseline", str(_leg_from_gate2(tmp_path / "am4", "am4")), "--box-from", str(_leg_from_gate2(tmp_path / "a7", "a7"))]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    rep = json.loads(r.stdout)
    step = rep["rehearsal"]["changed_step"]
    assert step["increment_s"] == 142.8 and step["box_basis_s"] == 392.8, step
    box = rep["phases"]["box"]
    assert box["budget_s"] == 13847 and box["sealed_cap_s"] == 2997 == _fit().sealed_caps(Q / "q235_run.sh")["box"], box
    assert not box["fits"] and r.returncode == 1


def test_a_leg_s_floors_are_max_of_sealed_a2000_and_the_slower_sample_over_the_slack():
    """Amendment 11's worked example: samples 24.0 and 23.0 GB/s H2D give 23.0 / 1.2 = 19.167; host copy 36 and 35 give 29.167; the
    cold box load's 3.0 GB/s reads give 2.5. A sealed floor above a leg's stays (floors only move up)."""
    f = _fit()
    sealed = {"h2d_gb_s": 5.8, "host_copy_gb_s": 5.14, "disk_read_gb_s": 0.743, "fetch_gb_s": 0.062}
    h1 = {"h2d_gb_s": 24.0, "host_copy_gb_s": 36.0, "pcie_gen": "5", "pcie_width": "16", "host_threads": 32}
    h2 = {"h2d_gb_s": 23.0, "host_copy_gb_s": 35.0}
    fl = f.leg_floors(h1, h2, 3.0, sealed)
    assert (fl["h2d_gb_s"], fl["host_copy_gb_s"], fl["disk_read_gb_s"]) == (19.167, 29.167, 2.5), fl
    up = f.leg_floors(h1, h2, 3.0, {**sealed, "h2d_gb_s": 30.0})
    assert up["h2d_gb_s"] == 30.0                                               # never below a sealed floor
    slow = f.leg_floors({**h1, "h2d_gb_s": 6.0}, {**h2, "h2d_gb_s": 5.9}, 1.0, sealed)
    assert slow["h2d_gb_s"] == 5.83 and slow["disk_read_gb_s"] == 1.145          # the A2000's measured rates bind a slow leg
    assert f.LEG_SLACK == 1.2 and f.MARGIN / f.LEG_SLACK == 1.25
    assert f.leg_floors(h1, h2, None, sealed)["disk_read_gb_s"] == 1.145      # Amendment 12: no cold-read sample, no leg term


def test_a_leg_whose_two_samples_disagree_is_unusable():
    f = _fit()
    sealed = {"h2d_gb_s": 5.8, "host_copy_gb_s": 5.14, "disk_read_gb_s": 0.743}
    with pytest.raises(SystemExit, match="unusable"):
        f.leg_floors({"h2d_gb_s": 24.0, "host_copy_gb_s": 36.0}, {"h2d_gb_s": 20.0, "host_copy_gb_s": 36.0}, 3.0, sealed)
    f.leg_floors({"h2d_gb_s": 24.0, "host_copy_gb_s": 36.0}, {"h2d_gb_s": 20.5, "host_copy_gb_s": 36.0}, 3.0, sealed)   # 14.6 %


def test_the_fit_reports_the_expected_run_against_the_six_hour_guard():
    f = _fit()
    timings = {"require-git": 2, "create-venv": 4, "install": 150, "hostrate": 20, "fetch": 30, "tokens": 7, "box": 100, "repeat": 60}
    fetch = {"bytes": 27_047_001_621, "hash_seconds": 23.7}
    caps = f.sealed_caps(Q / "q235_run.sh")
    rep = f.fit(timings, 178.0, fetch, 4_000_000_000, 470_211_497_053, 800.0, 1.6, caps)
    want = (2 + 4 + 20 + 7 + 4_000_000_000 / 1e8 + 178.0 + 470_211_497_053 / 1e8 + 470_211_497_053 / (27_047_001_621 / 23.7)
            + (100 + 60) * 94 / 4 + f.RESERVE_S + f.CONTROLLER_S)
    assert rep["expected_h"] == round(want / 3600, 2) and rep["expected_within_guard"] == (want <= 0.8 * 6 * 3600)


def _leg_dir(tmp_path, h2=None, box=110, repeat=70):
    d = tmp_path / "leg"
    (d / "receipts" / "box").mkdir(parents=True)
    (d / "receipts" / "run.json").write_text(json.dumps({"mode": "pro6000", "nonce": "n", "e4b_sha": "a" * 40}))
    phases = {"require-git": 3, "create-venv": 4, "install": 160, "hostrate": 25, "fetch": 300, "tokens": 6, "box": box,
              "repeat": repeat, "hostrate2": 25}
    (d / "receipts" / "phase-timings.jsonl").write_text("".join(json.dumps({"phase": k, "elapsed_s": v, "rc": 0}) + "\n"
                                                                 for k, v in phases.items()))
    (d / "receipts" / "fetch.json").write_text(json.dumps({"bytes": 27_047_001_621, "hash_seconds": 30.0}))
    prof = {"read": 9.0, "read_bytes": 22_391_367_680, "assemble": 10.0, "quantize": 0.5, "offload": 1.0, "other_s": 2.0}
    (d / "receipts" / "box" / "box.json").write_text(json.dumps({"load": {"seconds": 25.0, "profile": prof}}))
    (d / "receipts" / "hostrate.json").write_text(json.dumps({"h2d_gb_s": 24.0, "host_copy_gb_s": 36.0}))
    (d / "receipts" / "hostrate2.json").write_text(json.dumps(h2 or {"h2d_gb_s": 23.0, "host_copy_gb_s": 35.0}))
    return d


def test_the_fit_reads_a_pro6000_leg_end_to_end(tmp_path):
    d = _leg_dir(tmp_path)
    r = subprocess.run([sys.executable, str(Q / "q235_fit.py"), "--rehearsal", str(d), "--warm-install-s", "160",
                        "--bandwidth-mbps", "800", "--usd-per-hour", "1.6", "--out", str(tmp_path / "fit.json")],
                       capture_output=True, text=True)
    rep = json.loads((tmp_path / "fit.json").read_text())
    sealed = _fit().sealed_floors(Q / "q235_run.sh")
    assert rep["floors"]["h2d_gb_s"] == max(19.167, sealed["h2d_gb_s"])          # max(sealed, A2000, 23.0 / 1.2): floors only up
    fit_hc = 35.0 * (10.0 * 94 / 4) / (10.0 * 94 / 4 + 0.2 * 110 * 94 / 4)            # Amendment 13: min(samples) x A / (A + 0.2 x box)
    assert rep["floors"]["host_copy_gb_s"] == math.ceil(max(5.14, 4.34, fit_hc) * 1000 - 1e-9) / 1000
    assert rep["floors"]["disk_read_gb_s"] == max(1.145, sealed["disk_read_gb_s"])   # Amendment 12: no cold-read sample from a leg
    assert rep["phases"]["box"]["scaled_s"] == 110 * 94 / 4 and "expected_within_guard" in rep
    assert not rep["phases"]["box"]["fits"] and r.returncode == 1            # 110 s x 94/4 x 1.5 = 3,878 s > CAP_BOX 2,997
    bad = _leg_dir(tmp_path / "b", h2={"h2d_gb_s": 18.0, "host_copy_gb_s": 35.0})
    r = subprocess.run([sys.executable, str(Q / "q235_fit.py"), "--rehearsal", str(bad), "--warm-install-s", "160",
                        "--bandwidth-mbps", "800", "--usd-per-hour", "1.6"], capture_output=True, text=True)
    assert r.returncode != 0 and "unusable" in r.stderr
    r = subprocess.run([sys.executable, str(Q / "q235_fit.py"), "--rehearsal", str(d), "--warm-install-s", "160",
                        "--bandwidth-mbps", "800", "--usd-per-hour", "1.6", "--box-from", str(d), "--box-baseline", str(d)],
                       capture_output=True, text=True)
    assert r.returncode != 0 and "A2000 tools" in r.stderr


def test_a_leg_s_fit_binds_the_expected_run_to_the_guard(tmp_path, monkeypatch):
    """Every cap holds (box 390 s and repeat 130 s scale inside CAP_BOX / CAP_REPEAT; the fetch at 580 Mbps inside CAP_FETCH), yet
    the expected run (about 6.0 h) is above 0.8 x 6 h, so a leg's fit refuses. With the floors sealed as the leg implies them (what
    Amendment 12 does), the same leg at box 110 s / repeat 70 s fits."""
    f = _fit()
    sealed = {"h2d_gb_s": 19.167, "host_copy_gb_s": 29.167, "disk_read_gb_s": 1.145, "fetch_gb_s": 10.0}
    monkeypatch.setattr(f, "sealed_floors", lambda run_sh: dict(sealed))
    # Amendment 11's caps (box 13,847, repeat 4,710): under Amendment 12's tighter caps every cap fitting already implies an
    # expected run below 4.8 h, so the binding is exercised against the caps it was registered beside.
    a11_caps = {"require-git": 240, "create-venv": 120, "install": 1800, "hostrate": 600, "fetch": 12000, "tokens": 300, "box": 13847,
                "repeat": 4710}
    monkeypatch.setattr(f, "sealed_caps", lambda run_sh: dict(a11_caps))

    def run(d):
        out = d / "fit.json"
        b = json.loads((d / "receipts" / "box" / "box.json").read_text())
        t = {r["phase"]: r["elapsed_s"] for r in map(json.loads, (d / "receipts" / "phase-timings.jsonl").read_text().splitlines())}
        hr = [json.loads((d / "receipts" / n).read_text()) for n in ("hostrate.json", "hostrate2.json")]
        derived = f.leg_floors(hr[0], hr[1], None, dict(sealed), b["load"]["profile"]["assemble"] * 94 / 4, t["box"] * 94 / 4)
        monkeypatch.setattr(f, "sealed_floors", lambda run_sh: {**sealed, **{k: derived[k] for k in
                                                                              ("h2d_gb_s", "host_copy_gb_s", "disk_read_gb_s")}})
        monkeypatch.setattr(sys, "argv", ["q235_fit.py", "--rehearsal", str(d), "--warm-install-s", "160", "--bandwidth-mbps", "580",
                                          "--usd-per-hour", "1.6", "--out", str(out)])
        with pytest.raises(SystemExit) as e:
            f.main()
        return e.value.code, json.loads(out.read_text())

    code, rep = run(_leg_dir(tmp_path / "long", box=390, repeat=130))
    assert all(p["fits"] for p in rep["phases"].values()) and rep["floors"]["sealed_match"], rep["phases"]
    assert rep["expected_within_guard"] is False and rep["fits"] is False and code == 1
    code, rep = run(_leg_dir(tmp_path / "short"))
    assert rep["expected_within_guard"] is True and rep["fits"] is True and code == 0


def test_the_sealed_floors_and_caps_are_the_refit_of_the_pro6000_leg(tmp_path):
    """Amendment 12, mechanical: Amendment 11's formulas on the committed leg receipt (bench/q235/am12/leg, q235-leg-pro6000-2)
    give exactly the sealed floors (H2D 28.33 / 1.2 -> 23.609; the disk 1.145, because the leg's load was warm and its cold-read
    sample is absent; host copy by Amendment 13's fit, 70.20 x 34.075 / (34.075 + 0.2 x 1997.5) -> 5.518) and the sealed box and
    repeat budgets (box 85 s x 94/4 x 1.5 -> 2,997, sealed; the repeat by Amendment 15's box formula on the leg's record, 1.5 x
    (15.1 + 5 x 1.717) x 94/4 -> 835, not sealed), with the expected run inside 0.8 x 6 h at the pin's 806 Mbps."""
    import hashlib
    am12 = Q / "am12"
    for line in (am12 / "SHA256SUMS").read_text().splitlines():
        digest, name = line.split()
        assert hashlib.sha256((am12 / name).read_bytes()).hexdigest() == digest, name
    leg = tmp_path / "leg"
    shutil.copytree(am12 / "leg", leg)
    out = tmp_path / "fit.json"
    r = subprocess.run([sys.executable, str(Q / "q235_fit.py"), "--rehearsal", str(leg), "--warm-install-s", "155",
                        "--bandwidth-mbps", "806", "--usd-per-hour", "1.627", "--out", str(out)], capture_output=True, text=True)
    rep = json.loads(out.read_text())
    fl, sealed = rep["floors"], _fit().sealed_floors(Q / "q235_run.sh")
    assert (fl["h2d_gb_s"], fl["host_copy_gb_s"], fl["disk_read_gb_s"]) == (23.609, 5.518, 1.145) == (
        sealed["h2d_gb_s"], sealed["host_copy_gb_s"], sealed["disk_read_gb_s"]), (fl, sealed)
    assert fl["basis"]["per_floor"]["disk_read_gb_s"]["sample"] is None
    assert fl["basis"]["per_floor"]["disk_read_gb_s"]["warm_load_read_gb_s_not_used"] > 10      # the page cache, not the disk
    caps = _fit().sealed_caps(Q / "q235_run.sh")
    assert rep["phases"]["box"]["budget_s"] == caps["box"] == 2997 and rep["phases"]["repeat"]["budget_s"] == 835
    assert rep["phases"]["repeat"]["on_box_basis_s"] == round((15.1 + 5 * 1.717) * 94 / 4, 1)
    assert "repeat" not in caps and rep["phases"]["repeat"]["sealed_cap_s"] is None and rep["phases"]["repeat"]["fits"]
    assert rep["expected_within_guard"] and fl["sealed_match"] and rep["fits"] and r.returncode == 0, rep


def test_amendment_13s_host_copy_floor_is_from_the_fit_not_the_leg_rate():
    """Amendment 13, motivated by q235-pro6000-5's refusal (host copy 17.69 GB/s against Amendment 12's 58.5): the host-copy floor is
    the slowest host copy at which the scaled assemble part grows by at most 0.2 x the scaled box. On the leg: 70.20 x 34.075 /
    (34.075 + 399.5) = 5.5171 -> 5.518. A host at 17.69 passes it; a host at 5.0 is refused. The base 5.14 still binds a leg whose
    fit-derived value is lower."""
    f = _fit()
    sealed = {"h2d_gb_s": 23.609, "host_copy_gb_s": 58.5, "disk_read_gb_s": 1.145}
    h1, h2 = {"h2d_gb_s": 28.36, "host_copy_gb_s": 76.64}, {"h2d_gb_s": 28.33, "host_copy_gb_s": 70.20}
    fl = f.leg_floors(h1, h2, None, sealed, assemble_scaled_s=1.45 * 94 / 4, box_scaled_s=85 * 94 / 4)
    assert fl["host_copy_gb_s"] == 5.518 and fl["h2d_gb_s"] == 23.609 and fl["disk_read_gb_s"] == 1.145, fl
    assert 17.69 >= fl["host_copy_gb_s"] > 5.0
    low = f.leg_floors(h1, h2, None, sealed, assemble_scaled_s=0.1, box_scaled_s=85 * 94 / 4)
    assert low["host_copy_gb_s"] == 5.14                                       # never below the floor sealed before Amendment 12
    assert f.leg_floors(h1, h2, None, sealed)["host_copy_gb_s"] == 58.5        # without the fit's parts: Amendment 11's leg / 1.2
    assert (f.HC_BOX_SHARE, f.HOST_COPY_BASE) == (0.2, 5.14)
