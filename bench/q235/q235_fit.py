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
within the launcher's per-run cap, and the fit says whether it is also within the no-ask tier (above it, the ACK needs approval).

**The premise is enforced, not assumed.** Scaling A2000 times holds only on a host whose link and disk are no slower than the
rehearsal's, so the fit derives two floors the box checks before the fetch (``q235_hostrate.py``):
- the link: the rehearsal's measured pinned host-to-device rate;
- the disk: the rehearsal load's effective read rate, the slice's tensor bytes over the load's seconds (a box at that rate reads the
  checkpoint within the scaled load). The rehearsal's own O_DIRECT figure is reported beside it; on a cached filesystem it overstates
  the disk, which is why it is not the floor.

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
NO_ASK_USD = 15.0                   # above this, a run's ACK needs approval
FIXED = ("require-git", "create-venv", "tokens", "hostrate")
# The checkpoint's tensor bytes from its config at ac9c66cc (bf16): per decoder layer q (64 x 128 x 4096), k and v (4 x 128 x 4096 each),
# o (4096 x 8192), the router (128 x 4096), the 128 experts' gate, up and down (3 x 1536 x 4096 each) and four norms (2 x 4096 +
# 2 x 128); outside the layers the embedding and the head (151936 x 4096 each) and the final norm (4096).
LAYER_BYTES = 2 * (64 * 128 * 4096 + 2 * 4 * 128 * 4096 + 4096 * 8192 + 128 * 4096 + 128 * 3 * 1536 * 4096 + 2 * 4096 + 2 * 128)
OUTSIDE_BYTES = 2 * (2 * 151936 * 4096 + 4096)


def sealed_caps(run_sh: Path) -> dict:
    """The caps q235_run.sh seals, by phase name."""
    text = run_sh.read_text()
    found = dict(re.findall(r"\b(CAP_[A-Z]+)=(\d+)\b", text))
    names = {"CAP_GIT": "require-git", "CAP_VENV": "create-venv", "CAP_INSTALL": "install", "CAP_HOSTRATE": "hostrate", "CAP_FETCH": "fetch",
             "CAP_TOKENS": "tokens", "CAP_BOX": "box", "CAP_REPEAT": "repeat"}
    return {phase: int(found[var]) for var, phase in names.items() if var in found}


def floors(hostrate: dict, load_s: float) -> dict:
    """The host-rate floors the box enforces: the rehearsal's pinned H2D, and its load's effective disk read rate."""
    slice_bytes = SLICE_LAYERS * LAYER_BYTES + OUTSIDE_BYTES
    return {"h2d_gb_s": hostrate["h2d_gb_s"], "disk_read_gb_s": round(slice_bytes / load_s / 1e9, 3),
            "basis": {"slice_tensor_bytes": slice_bytes, "rehearsal_load_s": load_s,
                      "rehearsal_direct_read_gb_s": hostrate.get("disk_read_gb_s"), "rehearsal_disk_method": hostrate.get("disk_method"),
                      "rehearsal_pcie": f"gen {hostrate.get('pcie_gen')} x{hostrate.get('pcie_width')}"}}


def sealed_floors(run_sh: Path) -> dict:
    """The host-rate floors q235_run.sh seals for a science box."""
    found = dict(re.findall(r"\b(MIN_H2D_GBS|MIN_DISK_GBS)=([0-9.]+)\b", run_sh.read_text()))
    return {"h2d_gb_s": float(found.get("MIN_H2D_GBS", 0)), "disk_read_gb_s": float(found.get("MIN_DISK_GBS", 0))}


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
    tier = "within the no-ask tier" if usd <= NO_ASK_USD else ("above the no-ask tier: the ACK needs approval"
                                                                 if usd <= RUN_CAP_USD else "above the per-run cap")
    return {"schema": "q235-fit/1", "layers": LAYERS, "slice_layers": SLICE_LAYERS, "margin": MARGIN,
            "bandwidth_mbps": bandwidth_mbps, "usd_per_hour": usd_per_hour, "hash_rate_gb_s": round(hash_rate / 1e9, 3),
            "phases": phases, "wallclock_h": round(wallclock_s / 3600, 2), "usd_at_wallclock": round(usd, 2),
            "run_cap_usd": RUN_CAP_USD, "no_ask_usd": NO_ASK_USD, "cost_tier": tier,
            "fits": all(v["fits"] for v in phases.values()) and usd <= RUN_CAP_USD}


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
    box = json.loads((a.rehearsal / "receipts" / "box" / "box.json").read_text())
    hostrate = json.loads((a.rehearsal / "receipts" / "hostrate.json").read_text())
    rep["floors"] = floors(hostrate, float(box["load"]["seconds"]))
    sealed = sealed_floors(HERE / "q235_run.sh")
    rep["floors"]["sealed"] = sealed
    rep["floors"]["sealed_match"] = (sealed.get("h2d_gb_s") == rep["floors"]["h2d_gb_s"]
                                     and sealed.get("disk_read_gb_s") == rep["floors"]["disk_read_gb_s"])
    rep["fits"] = rep["fits"] and rep["floors"]["sealed_match"]
    rep["rehearsal"] = {"nonce": run.get("nonce"), "e4b_sha": run.get("e4b_sha"), "timings": timings}
    text = json.dumps(rep, indent=1)
    if a.out:
        a.out.write_text(text + "\n")
    print(text)
    sys.exit(0 if rep["fits"] else 1)


if __name__ == "__main__":
    main()
