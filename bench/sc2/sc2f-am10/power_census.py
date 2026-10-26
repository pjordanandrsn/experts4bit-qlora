#!/usr/bin/env python3
# Copyright (c) 2026 Cerin Amroth LLC. MIT.
"""power_census.py -- SC2f Amendment 10's evidence, counts only (reads nothing but its two inputs; rents nothing).

  power_census.py --reads READS.json --store <receipts root> [--out census.json]

- market: the screen's RTX 5090 reads (sc2_qualify.py admit's READS format) by listed power: machines seen, machines
  with a qualifying offer (<= $1.05/h all-in, >= 320 GB), machines reaching an admit point, and admit points (3
  consecutive reads sharing a qualifying offer id), per machine;
- burns: every pre-install card_clock burn in the store (``card_clock_burn.json`` / ``card_clock.json``) with its
  run, machine, enforced power and busy SM median;
- training: per run, the median of its ``gpuclk_*.txt`` busy medians (busy = SM > 1000 MHz, >= 10 busy samples), grouped
  by the run's enforced power limit (``forensics.txt`` line 1, else ``host_facts.json``);
- listed vs enforced: machines present in both the reads and the store, and whether the powers agree;
- the recorded-clock census (``sc2_qualify.recorded_clocks``: machines with a loaded clock record, and how many pass), and
  Amendment 10's admission replayed read by read over the reads with no attempts used and no clean runs (ADMIT reads by
  machine)."""
import argparse
import collections
import importlib.util
import json
import re
import statistics
import sys
from pathlib import Path

CEILING, DISK, FLOOR, BUSY, MIN_BUSY = 1.05, 320, 2700, 1000, 10


def market(reads):
    pw, qual = collections.defaultdict(set), collections.defaultdict(dict)
    for i, rd in enumerate(reads):
        for o in rd:
            m = str(o["machine_id"])
            pw[m].add(o.get("listed_power_w"))
            if isinstance(o.get("usd_h"), (int, float)) and o["usd_h"] <= CEILING and (o.get("disk_gb") or 0) >= DISK:
                qual[m].setdefault(i, set()).add(str(o["offer_id"]))
    per, by = {}, collections.defaultdict(lambda: {"machines": 0, "qualified": 0, "with_admit_point": 0, "admit_points": 0})
    for m, ps in pw.items():
        known = sorted(p for p in ps if p is not None)
        key = "/".join(f"{p:g}" for p in known) or "unknown"
        q = qual.get(m, {})
        pts = sum(1 for i in q if i - 1 in q and i - 2 in q and q[i] & q[i - 1] & q[i - 2])
        b = by[key]
        b["machines"] += 1
        b["qualified"] += bool(q)
        b["with_admit_point"] += pts > 0
        b["admit_points"] += pts
        if pts:
            per[m] = {"listed_w": key, "admit_points": pts, "qualifying_reads": len(q)}
    return {"reads": len(reads), "by_listed_w": dict(sorted(by.items())), "admit_points_by_machine": per}


def _power(rd: Path):
    out = set()
    for f in rd.rglob("forensics.txt"):
        first = f.read_text(errors="replace").split("\n", 1)[0]
        if "5090" in first and (x := re.search(r",\s*([\d.]+)\s*W\s*,", first)):
            out.add(float(x.group(1)))
    for f in rd.rglob("host_facts.json"):
        try:
            v = json.loads(f.read_text()).get("gpu_power_limit_w")
        except ValueError:
            v = None
        if isinstance(v, (int, float)):
            out.add(float(v))
    return sorted(out)


def _log_median(path: Path):
    sm = []
    for line in path.read_text(errors="replace").splitlines():
        p = line.split("|", 1)[0].split()
        if len(p) > 1:
            try:
                v = float(p[1].rstrip(","))
            except ValueError:
                continue
            if v > BUSY:
                sm.append(v)
    return statistics.median(sm) if len(sm) >= MIN_BUSY else None


def store_scan(store: Path):
    burns, train, machine_w = [], collections.defaultdict(list), collections.defaultdict(set)
    for rc in sorted(store.glob("*/*/receipt.json")):
        try:
            r = json.loads(rc.read_text())
        except ValueError:
            continue
        env = r.get("environment") or {}
        if "5090" not in str(env.get("vast_gpu_name") or r.get("gpu_model") or ""):
            continue
        rd, m = rc.parent, str(env.get("vast_machine_id") or "?")
        pw = _power(rd)
        key = "/".join(f"{p:g}" for p in pw) or "unknown"
        machine_w[m].update(pw)
        for f in sorted([*rd.rglob("card_clock_burn.json"), *rd.rglob("card_clock.json")]):
            try:
                b = json.loads(f.read_text())
            except ValueError:
                continue
            if isinstance(b, dict) and (b.get("busy_samples") or 0) >= MIN_BUSY and b.get("sm_median") is not None:
                burns.append({"run": f"{rd.parent.name}/{rd.name}", "machine": m, "power_w": key, "sm_median": b["sm_median"],
                              "pass_floor": b["sm_median"] >= FLOOR})
        meds = [x for x in (_log_median(f) for f in rd.rglob("gpuclk*.txt")) if x is not None]
        if meds:
            train[key].append((m, statistics.median(meds)))
    tr = {}
    for key, v in train.items():
        ms = collections.defaultdict(list)
        for m, x in v:
            ms[m].append(x)
        meds = sorted(x for _, x in v)
        tr[key] = {"runs": len(v), "machines": len(ms), "runs_at_floor": sum(x >= FLOOR for x in meds),
                   "machines_every_run_at_floor": sum(all(x >= FLOOR for x in xs) for xs in ms.values()),
                   "run_median_min": meds[0], "run_median_median": statistics.median(meds), "run_median_max": meds[-1]}
    return burns, dict(sorted(tr.items(), key=lambda kv: (kv[0] == "unknown", kv[0]))), machine_w


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--reads", required=True)
    ap.add_argument("--store", required=True)
    ap.add_argument("--out")
    a = ap.parse_args(argv)
    reads = json.loads(Path(a.reads).read_text())
    mk = market(reads)
    burns, train, machine_w = store_scan(Path(a.store))
    listed = collections.defaultdict(set)
    for rd in reads:
        for o in rd:
            if o.get("listed_power_w") is not None:
                listed[str(o["machine_id"])].add(float(o["listed_power_w"]))
    both = sorted(set(listed) & {m for m, w in machine_w.items() if w}, key=int)
    differ = {m: {"listed": sorted(listed[m]), "enforced": sorted(machine_w[m])} for m in both if listed[m] != machine_w[m]}
    spec = importlib.util.spec_from_file_location("sc2_qualify", Path(__file__).resolve().parents[1] / "sc2_qualify.py")
    q = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(q)
    clocks = q.recorded_clocks(Path(a.store))
    if clocks is None:
        print("the recorded-clock census is UNREAD (a store error)", file=sys.stderr)
        return 1
    replay = collections.Counter()
    for i in range(q.STABLE_READS, len(reads) + 1):
        r = q.admit([], reads[:i], False, clocks=clocks)
        if r["verdict"] == "ADMIT":
            replay[r["machine"]] += 1
    out = {"market": mk, "burns": burns, "training_by_enforced_w": train,
           "listed_vs_enforced": {"machines": len(both), "agree": len(both) - len(differ), "differ": differ},
           "recorded_clocks": {"machines": len(clocks), "pass": sum(1 for v in clocks.values() if v["ok"])},
           "am10_replay_admit_reads": dict(sorted(replay.items()))}
    text = json.dumps(out, indent=1, sort_keys=True)
    if a.out:
        Path(a.out).write_text(text + "\n")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
