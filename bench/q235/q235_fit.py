# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""Q235's gate 2, the time-left fit (bench/q235/PREREG-q235.md, "Before any rental"): every time-left guard's budget is scaled from
the A2000 rehearsal's measured phases to the registered model, explicitly, and each guard must fit what it guards.

The scaling, phase by phase (a measured time on the RTX A2000 is taken as an upper bound for the registered class, which is not
slower at anything the box does; the depth-truncated slice has ``SLICE_LAYERS`` of the model's ``LAYERS`` decoder layers):

- **require-git, create-venv, tokens**: the same work at any depth, kept as measured.
- **install**: the wheel bytes over the host's listed download bandwidth, plus the rehearsal's measured install from a cold cache.
- **fetch**: the checkpoint's bytes over the host's listed download bandwidth, plus their hashing at the rehearsal's measured rate.
- **box, repeat**: each scaled by LAYERS / SLICE_LAYERS. The load, the frozen-base hashes and the step's per-layer staging grow
  with depth; scaling the whole phase also scales its constant parts (embedding, head), which overestimates.

Each phase's budget is the scaled time times ``MARGIN``; a sealed cap in ``q235_run.sh`` below its budget refuses. The run's wallclock
is the sum of the caps, the runner's 300 s reserve and the controller's staging and receipt fetch; at the host's rate it must be
within the launcher's per-run cap.

    python q235_fit.py --rehearsal DIR --warm-install-s S --bandwidth-mbps M --usd-per-hour R [--out FIT.json]
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
LAYERS, SLICE_LAYERS = 94, 4
MARGIN = 1.5
RESERVE_S = 300                     # q235_run.sh's deadline reserve
CONTROLLER_S = 900                  # tc1_drive.sh: staging, the nonce handshake, the receipt fetch (the per-layer outputs, ~2 GB)
RUN_CAP_USD = 35.0                  # the launcher's per-run hard cap
FIXED = ("require-git", "create-venv", "tokens")


def sealed_caps(run_sh: Path) -> dict:
    """The caps q235_run.sh seals, by phase name."""
    text = run_sh.read_text()
    found = dict(re.findall(r"\b(CAP_[A-Z]+)=(\d+)\b", text))
    names = {"CAP_GIT": "require-git", "CAP_VENV": "create-venv", "CAP_INSTALL": "install", "CAP_FETCH": "fetch",
             "CAP_TOKENS": "tokens", "CAP_BOX": "box", "CAP_REPEAT": "repeat"}
    return {phase: int(found[var]) for var, phase in names.items() if var in found}


def fit(timings: dict, warm_install_s: float, fetch: dict, wheel_bytes: int, model_bytes: int, bandwidth_mbps: float,
        usd_per_hour: float, caps: dict) -> dict:
    """The fitted budget per phase against the sealed caps, the wallclock and the run's cost at the host's rate."""
    bw = bandwidth_mbps * 1e6 / 8                                          # bytes per second
    hash_rate = fetch["bytes"] / max(fetch["hash_seconds"], 1e-9)
    scaled = {p: float(timings[p]) for p in FIXED}
    scaled["install"] = wheel_bytes / bw + warm_install_s
    scaled["fetch"] = model_bytes / bw + model_bytes / hash_rate
    for p in ("box", "repeat"):
        scaled[p] = float(timings[p]) * LAYERS / SLICE_LAYERS
    phases = {}
    for p, s in scaled.items():
        budget = math.ceil(max(s * MARGIN, 60))
        phases[p] = {"scaled_s": round(s, 1), "budget_s": budget, "sealed_cap_s": caps.get(p), "fits": caps.get(p, 0) >= budget}
    wallclock_s = sum(caps.get(p, 0) for p in phases) + RESERVE_S + CONTROLLER_S
    usd = wallclock_s / 3600 * usd_per_hour
    return {"schema": "q235-fit/1", "layers": LAYERS, "slice_layers": SLICE_LAYERS, "margin": MARGIN,
            "bandwidth_mbps": bandwidth_mbps, "usd_per_hour": usd_per_hour, "hash_rate_gb_s": round(hash_rate / 1e9, 3),
            "phases": phases, "wallclock_h": round(wallclock_s / 3600, 2), "usd_at_wallclock": round(usd, 2),
            "run_cap_usd": RUN_CAP_USD, "fits": all(v["fits"] for v in phases.values()) and usd <= RUN_CAP_USD}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--rehearsal", type=Path, required=True, help="the A2000 rehearsal's workspace (receipts/)")
    ap.add_argument("--warm-install-s", type=float, required=True, help="the rehearsal's install from a cold wheel cache")
    ap.add_argument("--bandwidth-mbps", type=float, required=True, help="the pinned host's listed download bandwidth")
    ap.add_argument("--usd-per-hour", type=float, required=True, help="the pinned host's rate")
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()
    rows = [json.loads(ln) for ln in (a.rehearsal / "receipts" / "phase-timings.jsonl").read_text().splitlines()]
    run = json.loads((a.rehearsal / "receipts" / "run.json").read_text())
    if run.get("mode") != "a2000" or any(r["rc"] != 0 for r in rows):
        sys.exit("refusing: not a clean A2000 rehearsal (mode a2000, every phase rc 0)")
    timings = {r["phase"]: r["elapsed_s"] for r in rows}
    fetch = json.loads((a.rehearsal / "warm-fetch.json").read_text())       # the cold fetch (the rehearsal's own re-verified warm files)
    wheels = json.loads((HERE / "wheels.json").read_text())["packages"]
    model = json.loads((HERE / "model_files.json").read_text())
    rep = fit(timings, a.warm_install_s, fetch, sum(w["size"] for w in wheels), model["total_bytes"], a.bandwidth_mbps,
              a.usd_per_hour, sealed_caps(HERE / "q235_run.sh"))
    rep["rehearsal"] = {"nonce": run.get("nonce"), "e4b_sha": run.get("e4b_sha"), "timings": timings}
    text = json.dumps(rep, indent=1)
    if a.out:
        a.out.write_text(text + "\n")
    print(text)
    sys.exit(0 if rep["fits"] else 1)


if __name__ == "__main__":
    main()
