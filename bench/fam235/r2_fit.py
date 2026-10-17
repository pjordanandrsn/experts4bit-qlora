# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""R2's time-left fit: Q235's (bench/q235/q235_fit.py), run unchanged, with R2's depth and checkpoint in place of Q235's, after one
R2 pre-step that slows the rehearsal to the registered cold floors.

**Why the pre-step.** Q235's fit takes the rehearsal's measured times as an upper bound for the registered class, and its own host
rates as the floors. R2's A2000 leg read the slice it had just fetched, warm in the host's cache (2.71 GB/s, above the host's own
O_DIRECT 1.88), so its read rate would make a floor that refuses its own host. R2's floors are therefore Q235's sealed cold ones,
measured on the same RTX A2000 host (``FLOORS``), and each of the rehearsal's box and repeat processes is first slowed to them,
part by part from its load profile, before Q235's fit scales it:
- the read: the bytes it read at the disk floor, when that is slower than measured;
- the expert assembly and the offload (host-copy bound): times the measured host-copy rate over the floor's;
- the staging (host to device): the bytes staged at the H2D floor, when that is slower than measured. The box stages the experts in
  four passes, its in-process pair and its two steps, each counted at a step's bytes; the repeat records no staged bytes, so its
  three forwards are bounded by the box step's bytes each (a forward stages less than a step: conservative);
- then times ``SPREAD``, the slow end of the observed RTX 5090 per-step spread.
Every slowed time is at least the measured one: a rehearsal that read faster than a floor is slowed to it, never sped up.

The constants that differ from Q235's (Mixtral-8x22B-Instruct-v0.1 at cc88a6cc, from its config): 56 decoder layers against the
rehearsal's 4; per layer, bf16, q and o (6,144 x 6,144 each), k and v (1,024 x 6,144 each), the router (8 x 6,144), the 8 experts'
w1, w3 and w2 (3 x 16,384 x 6,144 each) and two norms (2 x 6,144); outside the layers the embedding and the head (32,768 x 6,144
each) and the final norm (6,144). 56 layers and the outside sum to 281,260,142,592 B, the checkpoint index's ``total_size``.

Its reads: the lock and the runner are R2's (``model_files.json``, ``r2_run.sh``), the wheels Q235's.

    python r2_fit.py --rehearsal DIR --warm-install-s S --bandwidth-mbps M --usd-per-hour R [--out FIT.json]
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
Q235 = REPO / "bench" / "q235"
LAYERS, SLICE_LAYERS = 56, 4
H, FF, V, E, KV = 6144, 16384, 32768, 8, 1024           # hidden, expert intermediate, vocabulary, experts, k/v width
LAYER_BYTES = 2 * (H * H + 2 * KV * H + H * H + E * H + E * 3 * FF * H + 2 * H)
OUTSIDE_BYTES = 2 * (2 * V * H + H)
FLOORS = {"h2d_gb_s": 5.8, "host_copy_gb_s": 5.14, "disk_read_gb_s": 0.743}   # Q235's sealed cold floors, this A2000 host (gate2)
SPREAD = 1.18                                            # the slow end of the observed RTX 5090 per-step spread (6-18 %)
POLICY_MAX_H = 6.0                                       # the launcher's teardown.max_wallclock_h
REPEAT_FORWARDS = 3                                      # the repeat's null and its two in-process plants
BOX_PASSES = 4                                           # the box's in-process pair (g0, g1) and its two steps, each staging the experts


class _Here:
    """Q235's ``HERE`` with R2's own files where R2 has them: the lock and the runner are R2's, the wheels Q235's."""
    own = {"model_files.json": HERE / "model_files.json", "q235_run.sh": HERE / "r2_run.sh"}

    def __truediv__(self, name):
        return self.own.get(name, Q235 / name)


def fit_module():
    """Q235's fit as a private module instance, with R2's depth, checkpoint bytes and files."""
    spec = importlib.util.spec_from_file_location("q235_fit_for_r2", Q235 / "q235_fit.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.HERE, mod.LAYERS, mod.SLICE_LAYERS = _Here(), LAYERS, SLICE_LAYERS
    mod.LAYER_BYTES, mod.OUTSIDE_BYTES = LAYER_BYTES, OUTSIDE_BYTES
    return mod


def slowed(measured_s: float, profile: dict, hostrate: dict, staged_bytes: float) -> dict:
    """One process's measured seconds slowed to ``FLOORS`` part by part (see the module's docstring); never below measured."""
    read = max(0.0, profile["read_bytes"] / (FLOORS["disk_read_gb_s"] * 1e9) - profile["read"])
    ratio = max(1.0, hostrate["host_copy_gb_s"] / FLOORS["host_copy_gb_s"])
    copy = (profile["assemble"] + profile["offload"]) * (ratio - 1.0)
    staging = max(0.0, staged_bytes / (FLOORS["h2d_gb_s"] * 1e9) - staged_bytes / (hostrate["h2d_gb_s"] * 1e9))
    out = measured_s + read + copy + staging
    if not out >= measured_s:
        raise AssertionError(f"a slowed time ({out}) is below its measurement ({measured_s})")
    return {"measured_s": measured_s, "read_s": round(read, 2), "host_copy_s": round(copy, 2), "staging_s": round(staging, 2),
            "slowed_s": round(out, 2), "with_spread_s": round(out * SPREAD, 2)}


def cold_timings(timings: dict, box: dict, repeat: dict, hostrate: dict) -> tuple:
    """The rehearsal's phase times with the box and the repeat slowed to the floors and the spread; the record of each."""
    step_bytes = [box[s]["h2d_bytes_offload"] for s in ("step", "step1")]
    box_staged = sum(step_bytes) + (BOX_PASSES - len(step_bytes)) * max(step_bytes)   # the pair stages as a step does
    rec = {"box": slowed(float(timings["box"]), box["load"]["profile"], hostrate, box_staged),
           "repeat": slowed(float(timings["repeat"]), repeat["load"]["profile"], hostrate, REPEAT_FORWARDS * max(step_bytes))}
    out = dict(timings)
    for p in ("box", "repeat"):
        out[p] = rec[p]["with_spread_s"]
    return out, rec


def sealed_host_floors(run_sh: Path) -> dict:
    """The science line's class and host floors, as ``r2_run.sh`` seals them (empty while the science mode refuses)."""
    m = re.search(r"^  science\) CARD_RE='([^']*)'; MIN_VRAM_MIB=(\d+); MIN_RAM_GB=(\d+); MIN_DISK_GB=(\d+)", run_sh.read_text(), re.M)
    return {"card_re": m.group(1), "min_vram_mib": int(m.group(2)), "min_ram_gb": int(m.group(3)), "min_disk_gb": int(m.group(4))} \
        if m else {}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--rehearsal", type=Path, required=True, help="the A2000 rehearsal's workspace (receipts/)")
    ap.add_argument("--warm-install-s", type=float, required=True, help="the rehearsal's install from a cold wheel cache")
    ap.add_argument("--bandwidth-mbps", type=float, required=True, help="the pinned host's listed download bandwidth")
    ap.add_argument("--usd-per-hour", type=float, required=True, help="the pinned host's rate")
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()
    q = fit_module()
    rd = a.rehearsal / "receipts"
    rows = [json.loads(ln) for ln in (rd / "phase-timings.jsonl").read_text().splitlines()]
    run = json.loads((rd / "run.json").read_text())
    if run.get("mode") != "a2000" or any(r["rc"] != 0 for r in rows):
        sys.exit("refusing: not a clean A2000 rehearsal (mode a2000, every phase rc 0)")
    box, repeat = json.loads((rd / "box" / "box.json").read_text()), json.loads((rd / "repeat" / "box.json").read_text())
    if set(repeat.get("plants", {})) != {"routing", "epilogue"}:
        sys.exit("refusing: the rehearsal's repeat did not run both in-process plants")
    hostrate = json.loads((rd / "hostrate.json").read_text())
    timings, slowed_rec = cold_timings({r["phase"]: r["elapsed_s"] for r in rows}, box, repeat, hostrate)
    fetch = json.loads((a.rehearsal / "warm-fetch.json").read_text())
    wheels = json.loads((q.HERE / "wheels.json").read_text())["packages"]
    model = json.loads((q.HERE / "model_files.json").read_text())
    caps = q.sealed_caps(q.HERE / "q235_run.sh")
    rep = q.fit(timings, a.warm_install_s, fetch, sum(w["size"] for w in wheels), model["total_bytes"], a.bandwidth_mbps,
                a.usd_per_hour, caps)
    rep["schema"] = "r2-fit/1"
    sealed = q.sealed_floors(q.HERE / "q235_run.sh")
    floors = {**FLOORS, "fetch_gb_s": q.fetch_floor(model["total_bytes"], fetch, caps["fetch"]),
              "basis": "Q235's sealed cold floors from the same RTX A2000 host; the pull floor from R2's fetch cap"}
    floors["sealed"] = sealed
    floors["sealed_match"] = (all(sealed.get(k) == FLOORS[k] for k in FLOORS) and sealed.get("fetch_gb_s", 0) >= floors["fetch_gb_s"])
    rep["floors"] = floors
    rep["host_floors"] = sealed_host_floors(q.HERE / "q235_run.sh")
    rep["within_policy_max"] = rep["wallclock_h"] <= POLICY_MAX_H
    rep["fits"] = rep["fits"] and floors["sealed_match"] and rep["within_policy_max"]
    rep["rehearsal"] = {"nonce": run.get("nonce"), "e4b_sha": run.get("e4b_sha"), "timings": {r["phase"]: r["elapsed_s"] for r in rows},
                        "slowed_to_floors": slowed_rec, "spread": SPREAD,
                        "measured_rates": {k: hostrate.get(k) for k in ("h2d_gb_s", "host_copy_gb_s", "disk_read_gb_s")},
                        "load_read_gb_s": {"box": box["load"]["profile"].get("read_gb_s"),
                                           "repeat": repeat["load"]["profile"].get("read_gb_s")}}
    text = json.dumps(rep, indent=1)
    if a.out:
        a.out.write_text(text + "\n")
    print(text)
    sys.exit(0 if rep["fits"] else 1)


if __name__ == "__main__":
    main()
