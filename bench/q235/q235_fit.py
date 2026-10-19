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

**The premise is enforced, not assumed.** Scaling A2000 times holds only on a host no slower than the rehearsal at what bounds the
work, so the fit reads the box's load profile (``q235_box.LoadProfile``: read, assemble, quantize, offload, other) and derives a
floor for each host-bound part, which the box checks before the fetch (``q235_hostrate.py``):
- the link (the step's staging, the offload and the stacks' copy in): the rehearsal's measured pinned host-to-device rate;
- the host memory (the expert assembly): the rehearsal's measured host-copy rate;
- the pull (the checkpoint fetch): the rate at which its bytes, hashed at the rehearsal's rate, fit the fetch cap / ``MARGIN``; the
  fetch's timed first slice enforces it;
- the disk (the shard reads): the rehearsal load's effective read rate, the bytes its reads returned over their seconds (without a
  profile, the slice's tensor bytes over the whole load). The rehearsal's own O_DIRECT figure is reported beside it; on a cached
  filesystem it overstates the disk, which is why it is not the floor.
Quantization runs on the card, whose class is registered. The profile's dominant part is named in the report.

Amendment 3's in-process plants add two forwards to the repeat. ``--plants-from DIR`` takes their measured time from a later A2000
leg (its repeat's ``plants``: each plant's synchronized forward) and adds it to the rehearsal's repeat before the scaling. Only those
forwards are taken from it: its load ran on a warm page cache, so its phases would understate a cold host.

Amendments 6 and 7 change the box's step: Shape B's accumulation, rule 8's deterministic check passes and the timed MoE pass.
The rehearsal keeps its whole cold box phase (load and the rest), and only the change is added, measured warm against warm (the
maintainer's rule): ``--box-from DIR`` is a later A2000 leg that ran the changed step and ``--box-baseline DIR`` a warm A2000 leg
at the code before the change. Each leg's box time outside its load is its box phase's elapsed seconds less its box record's
``load.seconds``, and

    box basis = the rehearsal's box phase + max(0, (later phase - later load) - (baseline phase - baseline load)).

The time outside the load is itself cache-sensitive (imports, compile caches, the frozen-base hash reads) and shrinks on a warm
host, so a warm leg's step alone would understate the cold box; the increment does not, and it is never negative.

Amendment 11 moves the TIMING leg to a rented RTX PRO 6000 (``q235_run.sh`` mode ``pro6000``, run ids ``q235-leg-pro6000-N``); the
A2000 keeps gate 1's correctness. ``--rehearsal`` may then be that leg, whose box and repeat are the registered class's own, scaled by
LAYERS / SLICE_LAYERS and ``MARGIN`` as before. Its fetch record is the box's own ``receipts/fetch.json`` (the slice's download and
hashing). Because its budgets come from a fast host, its floors are re-derived (``leg_floors``): for the link and the host copy,

    floor = max(sealed, the A2000's measured, min(the leg's two hostrate samples) / LEG_SLACK),

the two samples (before the fetch and after the repeat) agreeing within LEG_AGREE or the leg is unusable; for the disk, the same with
the leg's cold box load's effective read rate as its sample. Floors only move up. A box host at a floor runs a host-bound part at
most LEG_SLACK slower than the leg did, inside the budget's ``MARGIN`` (1.5 / 1.2 = 1.25 is left for everything else). A leg also
binds the maintainer's six-hour rule: the expected run (the scaled phases, no ``MARGIN``, plus the reserve and the controller) is at
most EXPECTED_SHARE of GUARD_H.

    python q235_fit.py --rehearsal DIR --warm-install-s S --bandwidth-mbps M --usd-per-hour R [--plants-from DIR2]
                       [--box-from DIR3 --box-baseline DIR4] [--out FIT.json]
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
CONTROLLER_S = 1800                 # tc1_drive.sh: staging, the nonce handshake, the receipt fetch (per-layer outputs of the box, the
                                    # null and two plants: 4 x 94 x 2,648 x 4,096 x 2 B, about 8.2 GB, at >= 4.6 MB/s)
# Amendments 15 and 16: phases whose cap the box sizes on the box (q235_run.sh's repeat_cap and box_cap), so nothing is sealed for
# them. The fit reports their scaled budget and counts it in the wallclock. It never refuses on the repeat's; the box's budget must fit
# the box's half-cap on the expected timeline: floor((guard - controller - the expected phases before it - reserve) / BOX_SHARE).
ON_BOX_CAPS = ("box", "repeat")
REPEAT_STEPS = 5                    # Amendment 15: the box's cap is MARGIN x (load + REPEAT_STEPS x step 0), as q235_run.sh's repeat_cap
BOX_SHARE = 2                       # Amendment 16: the box's cap is floor((time left at its start - RESERVE_S) / BOX_SHARE)
RUN_CAP_USD = 35.0                  # the launcher's per-run hard cap
NO_ASK_USD = 15.0                   # above this, a run's ACK needs approval
FIXED = ("require-git", "create-venv", "tokens", "hostrate")
# Amendment 11 (the PRO 6000 timing leg's floors). The A2000's measured rates: gate 1's leg at d2d6b292 (2026-10-18, contended): the
# hostrate probe's pinned H2D and host copy, and its box load's reads (22,391,367,680 bytes in 19.55 s).
A2000_MEASURED = {"h2d_gb_s": 5.83, "host_copy_gb_s": 4.34, "disk_read_gb_s": 1.145}
LEG_SLACK = 1.2                     # a box host may be at most this much slower than the leg host on a host-bound part
LEG_AGREE = 0.15                    # the leg's two hostrate samples agree within this share, or the leg is unusable
GUARD_H, EXPECTED_SHARE = 6.0, 0.8  # the science box's guard and the share of it the expected run may use (2026-10-18 07:51Z)
# Amendment 13: the host-copy floor from the fit, per host-bound part. The load's assemble (host copies) may add at most this share
# of the box's scaled time on a slower host; the floor never goes below the one sealed before Amendment 12.
HC_BOX_SHARE, HOST_COPY_BASE = 0.2, 5.14
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


def floors(hostrate: dict, load_s: float, profile: dict | None = None) -> dict:
    """The host-rate floors the box enforces: the rehearsal's pinned H2D, its host-copy rate, and its load's effective read rate."""
    slice_bytes = SLICE_LAYERS * LAYER_BYTES + OUTSIDE_BYTES
    parts = {k: profile[k] for k in ("read", "assemble", "quantize", "offload", "other_s")} if profile else {}
    if profile and profile.get("read") and profile.get("read_bytes"):
        disk, disk_basis = round(profile["read_bytes"] / profile["read"] / 1e9, 3), "the load's reads: bytes returned over their seconds"
    else:
        disk, disk_basis = round(slice_bytes / load_s / 1e9, 3), "the slice's tensor bytes over the whole load"
    return {"h2d_gb_s": hostrate["h2d_gb_s"], "host_copy_gb_s": hostrate.get("host_copy_gb_s", 0.0), "disk_read_gb_s": disk,
            "basis": {"slice_tensor_bytes": slice_bytes, "rehearsal_load_s": load_s, "disk_floor_from": disk_basis,
                      "load_profile_s": parts, "dominant": max(parts, key=parts.get) if parts else None,
                      "rehearsal_direct_read_gb_s": hostrate.get("disk_read_gb_s"), "rehearsal_disk_method": hostrate.get("disk_method"),
                      "rehearsal_host_threads": hostrate.get("host_threads"),
                      "rehearsal_pcie": f"gen {hostrate.get('pcie_gen')} x{hostrate.get('pcie_width')}"}}


def _rate(x) -> float:
    """A floor rounded UP to 1e-3 GB/s, so a rounded floor never admits a host below the derived one."""
    return math.ceil(x * 1000 - 1e-9) / 1000


def leg_floors(hostrate: dict, hostrate2: dict, leg_disk_gb_s: float | None, sealed: dict, assemble_scaled_s: float | None = None,
               box_scaled_s: float | None = None) -> dict:
    """Amendment 11: the floors a PRO 6000 timing leg implies; refuses (SystemExit) when its two hostrate samples disagree.
    ``leg_disk_gb_s`` None: the leg did not observe a cold read, so the disk floor takes no leg sample (Amendment 12).
    ``assemble_scaled_s`` and ``box_scaled_s`` (Amendment 13): the host-copy floor comes from the fit instead of leg / 1.2 --
    the slowest host copy at which the scaled assemble part grows by at most HC_BOX_SHARE of the scaled box,
    ``min(samples) x A / (A + HC_BOX_SHARE x box)``, never below HOST_COPY_BASE or the A2000's measured rate."""
    out, basis = {}, {}
    for k in ("h2d_gb_s", "host_copy_gb_s"):
        a, b = float(hostrate[k]), float(hostrate2[k])
        spread = abs(a - b) / max(a, b)
        if spread > LEG_AGREE:
            sys.exit(f"refusing: the leg's two {k} samples ({a}, {b}) differ by {spread:.1%} > {LEG_AGREE:.0%}: the leg is unusable "
                     "for floors (one re-run on another host, then RED)")
        leg = min(a, b) / LEG_SLACK
        if k == "host_copy_gb_s" and assemble_scaled_s is not None and box_scaled_s is not None:
            fit = min(a, b) * assemble_scaled_s / (assemble_scaled_s + HC_BOX_SHARE * box_scaled_s)
            out[k] = _rate(max(HOST_COPY_BASE, A2000_MEASURED[k], fit))
            basis[k] = {"samples": [a, b], "spread": round(spread, 4), "rule": "Amendment 13: min(samples) x A / (A + "
                        f"{HC_BOX_SHARE} x box)", "assemble_scaled_s": assemble_scaled_s, "box_scaled_s": box_scaled_s,
                        "fit_derived": round(fit, 4), "base": HOST_COPY_BASE, "a2000_measured": A2000_MEASURED[k]}
            continue
        out[k] = _rate(max(sealed[k], A2000_MEASURED[k], leg))
        basis[k] = {"samples": [a, b], "spread": round(spread, 4), "leg_over_slack": round(leg, 4), "sealed": sealed[k],
                    "a2000_measured": A2000_MEASURED[k]}
    leg = None if leg_disk_gb_s is None else leg_disk_gb_s / LEG_SLACK
    out["disk_read_gb_s"] = _rate(max(sealed["disk_read_gb_s"], A2000_MEASURED["disk_read_gb_s"], leg or 0.0))
    basis["disk_read_gb_s"] = {"sample": leg_disk_gb_s, "leg_over_slack": None if leg is None else round(leg, 4),
                               "sealed": sealed["disk_read_gb_s"], "a2000_measured": A2000_MEASURED["disk_read_gb_s"]}
    out["basis"] = {"rule": f"max(sealed, A2000 measured, leg / {LEG_SLACK}); floors only move up", "per_floor": basis,
                    "pcie": f"gen {hostrate.get('pcie_gen')} x{hostrate.get('pcie_width')}", "host_threads": hostrate.get("host_threads")}
    return out


def sealed_floors(run_sh: Path) -> dict:
    """The host-rate floors q235_run.sh seals for a science box."""
    found = dict(re.findall(r"\b(MIN_H2D_GBS|MIN_DISK_GBS|MIN_HOST_COPY_GBS|MIN_FETCH_GBS)=([0-9.]+)\b", run_sh.read_text()))
    return {"h2d_gb_s": float(found.get("MIN_H2D_GBS", 0)), "disk_read_gb_s": float(found.get("MIN_DISK_GBS", 0)),
            "host_copy_gb_s": float(found.get("MIN_HOST_COPY_GBS", 0)), "fetch_gb_s": float(found.get("MIN_FETCH_GBS", 0))}


def fetch_floor(model_bytes: int, fetch: dict, fetch_cap_s: int) -> float:
    """The pull rate (GB/s, rounded up to 1e-3) at which the checkpoint, hashed at the rehearsal's rate, fits ``fetch_cap_s`` / MARGIN."""
    hash_s = model_bytes / (fetch["bytes"] / max(fetch["hash_seconds"], 1e-9))
    left = fetch_cap_s / MARGIN - hash_s
    if left <= 0:
        return math.inf
    return math.ceil(model_bytes / left / 1e9 * 1000) / 1000


def fit(timings: dict, warm_install_s: float, fetch: dict, wheel_bytes: int, model_bytes: int, bandwidth_mbps: float,
        usd_per_hour: float, caps: dict, box_record: dict | None = None) -> dict:
    """The fitted budget per phase against the sealed caps, the wallclock and the run's cost at the host's rate. ``box_record``
    ({"load_s", "step_s"}, the rehearsal box's load.seconds and step.elapsed_s): Amendment 15's repeat budget, scaled like the box."""
    bw = bandwidth_mbps * 1e6 / 8                                          # bytes per second
    hash_rate = fetch["bytes"] / max(fetch["hash_seconds"], 1e-9)
    scaled = {p: float(timings[p]) for p in FIXED}
    scaled["install"] = wheel_bytes / bw + warm_install_s
    scaled["fetch"] = model_bytes / bw + model_bytes / hash_rate
    for p in ("box", "repeat"):
        scaled[p] = float(timings[p]) * LAYERS / SLICE_LAYERS
    phases = {}
    left_at_box = GUARD_H * 3600 - CONTROLLER_S - sum(v for p, v in scaled.items() if p not in ("box", "repeat"))
    half_cap = math.floor((left_at_box - RESERVE_S) / BOX_SHARE)
    for p, s in scaled.items():
        budget = math.ceil(max(s * MARGIN, 60))
        if p == "box" and p not in caps:
            phases[p] = {"scaled_s": round(s, 1), "budget_s": budget, "sealed_cap_s": None, "cap": "sized on the box (Amendment 16)",
                         "expected_left_at_start_s": round(left_at_box, 1), "half_cap_s": half_cap, "fits": budget <= half_cap}
        elif p in ON_BOX_CAPS and p not in caps:
            basis = None
            if box_record is not None:              # the box's own formula, on the rehearsal box's record scaled to the full depth
                basis = (float(box_record["load_s"]) + REPEAT_STEPS * float(box_record["step_s"])) * LAYERS / SLICE_LAYERS
                budget = math.ceil(max(basis * MARGIN, 60))
            phases[p] = {"scaled_s": round(s, 1), "budget_s": budget, "sealed_cap_s": None, "cap": "sized on the box (Amendment 15)",
                         "on_box_basis_s": None if basis is None else round(basis, 1), "fits": True}
        else:
            phases[p] = {"scaled_s": round(s, 1), "budget_s": budget, "sealed_cap_s": caps.get(p), "fits": caps.get(p, 0) >= budget}
    wallclock_s = sum(caps.get(p, phases[p]["budget_s"] if p in ON_BOX_CAPS else 0) for p in phases) + RESERVE_S + CONTROLLER_S
    expected_s = sum(scaled.values()) + RESERVE_S + CONTROLLER_S          # no MARGIN: the run as expected, not its alarms
    usd = wallclock_s / 3600 * usd_per_hour
    tier = "within the no-ask tier" if usd <= NO_ASK_USD else ("above the no-ask tier: the ACK needs approval"
                                                                 if usd <= RUN_CAP_USD else "above the per-run cap")
    return {"schema": "q235-fit/1", "layers": LAYERS, "slice_layers": SLICE_LAYERS, "margin": MARGIN,
            "bandwidth_mbps": bandwidth_mbps, "usd_per_hour": usd_per_hour, "hash_rate_gb_s": round(hash_rate / 1e9, 3),
            "phases": phases, "wallclock_h": round(wallclock_s / 3600, 2), "usd_at_wallclock": round(usd, 2),
            "expected_h": round(expected_s / 3600, 2), "expected_within_guard": expected_s <= EXPECTED_SHARE * GUARD_H * 3600,
            "run_cap_usd": RUN_CAP_USD, "no_ask_usd": NO_ASK_USD, "cost_tier": tier,
            "fits": all(v["fits"] for v in phases.values()) and usd <= RUN_CAP_USD}


def leg_box(d, flag):
    """A clean A2000 leg's run record, box phase seconds and box load seconds, or a named refusal."""
    rows = [json.loads(ln) for ln in (d / "receipts" / "phase-timings.jsonl").read_text().splitlines()]
    run = json.loads((d / "receipts" / "run.json").read_text())
    if run.get("mode") != "a2000" or any(r["rc"] != 0 for r in rows):
        sys.exit(f"refusing: {flag} is not a clean A2000 leg (mode a2000, every phase rc 0)")
    phase = {r["phase"]: r["elapsed_s"] for r in rows}.get("box")
    if phase is None:
        sys.exit(f"refusing: {flag} has no box phase in its phase timings")
    try:
        load = float(json.loads((d / "receipts" / "box" / "box.json").read_text())["load"]["seconds"])
    except (OSError, KeyError, TypeError, ValueError):
        sys.exit(f"refusing: {flag}'s box record carries no load.seconds")
    if float(phase) - load <= 0:
        sys.exit(f"refusing: {flag}'s box phase is not longer than its own load")
    return run, float(phase), load


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--rehearsal", type=Path, required=True, help="the A2000 rehearsal's workspace (receipts/)")
    ap.add_argument("--warm-install-s", type=float, required=True, help="the rehearsal's install from a cold wheel cache")
    ap.add_argument("--bandwidth-mbps", type=float, required=True, help="the pinned host's listed download bandwidth")
    ap.add_argument("--usd-per-hour", type=float, required=True, help="the pinned host's rate")
    ap.add_argument("--plants-from", type=Path, default=None, help="an A2000 leg whose repeat ran the in-process plants")
    ap.add_argument("--box-from", type=Path, default=None, help="an A2000 leg whose box ran the changed step (Amendments 6 and 7)")
    ap.add_argument("--box-baseline", type=Path, default=None,
                    help="with --box-from: a warm A2000 leg at the code before the change, the increment's baseline")
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()
    rows = [json.loads(ln) for ln in (a.rehearsal / "receipts" / "phase-timings.jsonl").read_text().splitlines()]
    run = json.loads((a.rehearsal / "receipts" / "run.json").read_text())
    if run.get("mode") not in ("a2000", "pro6000") or any(r["rc"] != 0 for r in rows):
        sys.exit("refusing: not a clean A2000 rehearsal or Amendment 11 pro6000 leg (every phase rc 0)")
    leg = run["mode"] == "pro6000"
    if leg and (a.plants_from or a.box_from):
        sys.exit("refusing: a pro6000 leg ran the current step and its plants itself; --plants-from / --box-from are A2000 tools")
    timings = {r["phase"]: r["elapsed_s"] for r in rows}
    plants = None
    if a.plants_from:
        later = json.loads((a.plants_from / "receipts" / "repeat" / "box.json").read_text())
        lrun = json.loads((a.plants_from / "receipts" / "run.json").read_text())
        if lrun.get("mode") != "a2000" or set(later.get("plants", {})) != {"routing", "epilogue"}:
            sys.exit("refusing: --plants-from is not an A2000 leg whose repeat ran both in-process plants")
        plants = {k: v["forward_s"] for k, v in later["plants"].items()}
        timings["repeat"] = timings["repeat"] + sum(plants.values())
    box_step = None
    if (a.box_from is None) != (a.box_baseline is None):
        sys.exit("refusing: --box-from and --box-baseline go together (the change is measured warm against warm)")
    if a.box_from:
        lrun_b, later_phase, later_load = leg_box(a.box_from, "--box-from")
        base_run, base_phase, base_load = leg_box(a.box_baseline, "--box-baseline")
        own_phase = timings["box"]
        later_step, base_step = later_phase - later_load, base_phase - base_load
        increment = max(0.0, later_step - base_step)
        box_step = {"from_e4b_sha": lrun_b.get("e4b_sha"), "from_nonce": lrun_b.get("nonce"),
                    "baseline_e4b_sha": base_run.get("e4b_sha"), "baseline_nonce": base_run.get("nonce"),
                    "later_phase_s": later_phase, "later_load_s": later_load, "later_outside_load_s": round(later_step, 3),
                    "baseline_phase_s": base_phase, "baseline_load_s": base_load, "baseline_outside_load_s": round(base_step, 3),
                    "increment_s": round(increment, 3), "own_box_phase_s": own_phase}
        timings["box"] = own_phase + increment
        box_step["box_basis_s"] = round(timings["box"], 3)
    fetch_file = (a.rehearsal / "receipts" / "fetch.json") if leg else (a.rehearsal / "warm-fetch.json")
    fetch = json.loads(fetch_file.read_text())         # A2000: its own re-verified warm files; a leg: the box's slice fetch (Amendment 11)
    wheels = json.loads((HERE / "wheels.json").read_text())["packages"]
    model = json.loads((HERE / "model_files.json").read_text())
    box = json.loads((a.rehearsal / "receipts" / "box" / "box.json").read_text())
    try:                                           # Amendment 15: the box's own repeat formula, when the record carries both parts
        box_record = {"load_s": float(box["load"]["seconds"]), "step_s": float(box["step"]["elapsed_s"])}
    except (KeyError, TypeError, ValueError):      # a record from before the step's timing: the leg-scaled repeat, as before
        box_record = None
    rep = fit(timings, a.warm_install_s, fetch, sum(w["size"] for w in wheels), model["total_bytes"], a.bandwidth_mbps,
              a.usd_per_hour, sealed_caps(HERE / "q235_run.sh"), box_record=box_record)
    hostrate = json.loads((a.rehearsal / "receipts" / "hostrate.json").read_text())
    rep["floors"] = floors(hostrate, float(box["load"]["seconds"]), box["load"].get("profile"))
    sealed = sealed_floors(HERE / "q235_run.sh")
    if leg:                                            # Amendment 11: the rented leg's floors replace the A2000 rehearsal's
        hostrate2 = json.loads((a.rehearsal / "receipts" / "hostrate2.json").read_text())
        warm_read = rep["floors"]["disk_read_gb_s"]
        # Amendment 12: an Amendment 11 leg cannot observe a cold read (its fetch has just read and hashed the slice, so the box
        # load reads from the page cache); the registered cold-read sample is absent and contributes nothing. Reported, not used.
        prof = box["load"].get("profile") or {}
        rep["floors"] = leg_floors(hostrate, hostrate2, None, sealed,      # Amendment 13: host copy from the fit's own parts
                                   assemble_scaled_s=float(prof.get("assemble", 0.0)) * LAYERS / SLICE_LAYERS,
                                   box_scaled_s=float(timings["box"]) * LAYERS / SLICE_LAYERS)
        rep["floors"]["basis"]["per_floor"]["disk_read_gb_s"]["warm_load_read_gb_s_not_used"] = warm_read
        rep["fits"] = rep["fits"] and rep["expected_within_guard"]
    rep["floors"]["sealed"] = sealed
    caps = sealed_caps(HERE / "q235_run.sh")
    rep["floors"]["fetch_gb_s"] = fetch_floor(model["total_bytes"], fetch, caps["fetch"])
    rep["floors"]["sealed_match"] = (all(sealed.get(k) == rep["floors"][k] for k in ("h2d_gb_s", "host_copy_gb_s", "disk_read_gb_s"))
                                     and sealed.get("fetch_gb_s", 0) >= rep["floors"]["fetch_gb_s"])
    rep["fits"] = rep["fits"] and rep["floors"]["sealed_match"]
    rep["rehearsal"] = {"nonce": run.get("nonce"), "e4b_sha": run.get("e4b_sha"), "timings": timings}
    if plants is not None:
        rep["rehearsal"]["in_process_plants"] = {"from_e4b_sha": lrun.get("e4b_sha"), "from_nonce": lrun.get("nonce"),
                                                 "forward_s": plants, "added_to_repeat_s": round(sum(plants.values()), 3)}
    if box_step is not None:
        rep["rehearsal"]["changed_step"] = box_step
    text = json.dumps(rep, indent=1)
    if a.out:
        a.out.write_text(text + "\n")
    print(text)
    sys.exit(0 if rep["fits"] else 1)


if __name__ == "__main__":
    main()
