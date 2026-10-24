# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""D0s writer probe (EXPLORE, never data): how long ``merged_arena.write`` takes on this host, and what bounds it.

The base bake on the card, then the writer for one fixture at both sources (MERGED-Q from the arena, MERGED-W from the
checkpoint), each timed, with the writer's own log time-stamped. Beside them: a profile of this process's cores (CPU
ticks per wall second, every 15 s), the host facts the launcher records (bench/common/host_facts.py: the cgroup CPU
quota, affinity, the single-thread loop), torch's thread count and the peak RSS. Writes ``<out>/probe.json``.

    python bench/d0s/d0s_writer_probe.py --snapshot SNAP --adapters DIR --out DIR [--fixture nodq]
"""
from __future__ import annotations

import argparse
import json
import os
import resource
import sys
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))


class Profile(threading.Thread):
    """Cores this process used per interval: (utime + stime) ticks over wall time, every thread included."""

    def __init__(self, period: float = 15.0):
        super().__init__(daemon=True)
        self.period, self.rows, self.halt = period, [], threading.Event()

    @staticmethod
    def _cpu_s() -> float:
        fields = open("/proc/self/stat").read().rsplit(")", 1)[1].split()
        return (int(fields[11]) + int(fields[12])) / os.sysconf("SC_CLK_TCK")      # utime, stime (fields 14, 15)

    def run(self):
        t0, c0 = time.monotonic(), self._cpu_s()
        while not self.halt.wait(self.period):
            t, c = time.monotonic(), self._cpu_s()
            self.rows.append([round(time.time(), 1), round((c - c0) / (t - t0), 2)])
            t0, c0 = t, c


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--snapshot", required=True)
    ap.add_argument("--adapters", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--fixture", default="nodq", choices=("nodq", "dq"))
    a = ap.parse_args(argv)
    sys.path.insert(0, os.path.join(ROOT, "bench", "common"))
    import host_facts
    import nvme_bake_nf4 as nb
    import torch

    from experts4bit_qlora import merged_arena as MA

    os.makedirs(a.out, exist_ok=True)
    rec = {"scheme": "d0s-writer-probe/1", "fixture": a.fixture, "host_facts": host_facts.gather(),
           "torch_threads": torch.get_num_threads(), "os_cpu_count": os.cpu_count(),
           "affinity": len(os.sched_getaffinity(0)), "cells": {}}
    prof = Profile()
    prof.start()
    base = os.path.join(a.out, "base.arena")
    t = time.time()
    nb.bake_nf4(a.snapshot, base, log=lambda *x, **k: None)
    rec["bake"] = {"start": round(t, 1), "seconds": round(time.time() - t, 1)}
    ad = os.path.join(a.adapters, f"fixture-{a.fixture}-v2")
    for cell, source, ckpt in (("MERGED-Q", "arena", None), ("MERGED-W", "checkpoint", a.snapshot)):
        log = []
        t = time.time()
        row = {"source": source, "start": round(t, 1), "log": log}
        try:
            block = MA.write(base, ad, os.path.join(a.out, f"{a.fixture}-{cell}.arena"), source=source,
                             checkpoint=ckpt, log=lambda *x: log.append([round(time.time(), 1), " ".join(map(str, x))]))
            row.update(status="WRITTEN", retention=block.get("retention_summary"))
        except MA.MergeRefused as e:
            row.update(status="WRITER_REFUSED", why=str(e))
        row["seconds"] = round(time.time() - t, 1)
        rec["cells"][cell] = row
        print(f"{cell} {row['status']} {row['seconds']} s", flush=True)
    prof.halt.set()
    prof.join(timeout=prof.period + 5)
    rec["profile"] = prof.rows
    rec["peak_rss_gb"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20, 2)     # Linux: kB
    with open(os.path.join(a.out, "probe.json"), "w", encoding="utf-8") as f:
        f.write(json.dumps(rec, indent=1, sort_keys=True) + "\n")
    cores = [c for _t, c in prof.rows]
    hf = rec["host_facts"]
    print(f"PROBE quota={hf.get('cpu_quota_cpus')} mops={hf.get('cpu_loop_mops')} torch_threads={rec['torch_threads']} "
          f"bake={rec['bake']['seconds']} s " + " ".join(f"{c}={r['seconds']} s" for c, r in rec["cells"].items())
          + f" cores_mean={sum(cores) / max(len(cores), 1):.1f} peak_rss={rec['peak_rss_gb']} GB", flush=True)
    return 0 if all(r["status"] == "WRITTEN" for r in rec["cells"].values()) else 3


if __name__ == "__main__":
    raise SystemExit(main())
