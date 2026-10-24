# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""D0s writer probe (EXPLORE, never data): how long ``merged_arena.write`` takes on this host, and what bounds it.

The base bake on the card, then the writer for one fixture at both sources (MERGED-Q from the arena, MERGED-W from the
checkpoint), each timed, with the writer's own log time-stamped. Beside them: a profile of this process's cores (CPU
ticks per wall second, every 15 s), the host facts the launcher records (bench/common/host_facts.py: the cgroup CPU
quota, affinity, the single-thread loop), torch's thread count and the peak RSS. Writes ``<out>/probe.json``.

    python bench/d0s/d0s_writer_probe.py --snapshot SNAP --adapters DIR --out DIR [--fixture nodq] [--mode both|confirm]

``--mode confirm`` tests the oversubscription cause: after the bake, MERGED-Q runs twice, each in its own process: first
with OMP_NUM_THREADS = MKL_NUM_THREADS = floor(min(cgroup quota, affinity)) (the disadvantaged, cold-cache slot), then at
torch's default, capped at min(10x the pinned run, 900 s). Per run: torch's thread count and the env, the cgroup's
cpu.stat (nr_periods, nr_throttled, throttled time) before and after, the .tmp growth, the cores, and the output qfp.
"""
from __future__ import annotations

import argparse
import json
import os
import math
import resource
import signal
import subprocess
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


CPU_STAT = ("/sys/fs/cgroup/cpu.stat", "/sys/fs/cgroup/cpu/cpu.stat", "/sys/fs/cgroup/cpu,cpuacct/cpu.stat")


def cpu_stat() -> dict:
    """The cgroup's CFS counters (v2 throttled_usec, or v1 throttled_time in ns), from the first readable source."""
    for path in CPU_STAT:
        try:
            rows = dict(line.split() for line in open(path).read().splitlines() if len(line.split()) == 2)
            return {"source": path, **{k: int(v) for k, v in rows.items() if v.isdigit()}}
        except OSError:
            continue
    return {"source": None}


def one(base: str, adapter: str, dst: str) -> int:
    """One MERGED-Q write in this process; prints a JSON line: threads, env, seconds, merged qfp."""
    import torch

    from experts4bit_qlora import merged_arena as MA
    t = time.time()
    block = MA.write(base, adapter, dst, source="arena", log=lambda *x: None)
    print("ONE " + json.dumps({"torch_threads": torch.get_num_threads(), "seconds": round(time.time() - t, 1),
                               "env": {k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS")},
                               "merged_qfp_v1": block.get("merged_qfp_v1")}), flush=True)
    return 0


def confirm_run(label: str, base: str, adapter: str, dst: str, threads: int | None, cap_s: float) -> dict:
    """Run :func:`one` in a child (pinned threads, or torch's default), sampling the .tmp, the child's cores and cpu.stat;
    kill it at ``cap_s``."""
    env = {k: v for k, v in os.environ.items() if k not in ("OMP_NUM_THREADS", "MKL_NUM_THREADS")}
    if threads is not None:
        env.update(OMP_NUM_THREADS=str(threads), MKL_NUM_THREADS=str(threads))
    before = cpu_stat()
    t0 = time.time()
    child = subprocess.Popen([sys.executable, os.path.abspath(__file__), "--one", "--base", base, "--adapter", adapter,
                              "--dst", dst], env=env, stdout=subprocess.PIPE, text=True, start_new_session=True)
    hz, samples, last = os.sysconf("SC_CLK_TCK"), [], None
    while child.poll() is None and time.time() - t0 < cap_s:
        time.sleep(5)
        try:
            f = open(f"/proc/{child.pid}/stat").read().rsplit(")", 1)[1].split()
            cpu = (int(f[11]) + int(f[12])) / hz
        except (OSError, IndexError, ValueError):
            cpu = None
        size = os.path.getsize(dst + ".tmp") if os.path.exists(dst + ".tmp") else None
        now = time.time()
        cores = round((cpu - last[1]) / (now - last[0]), 2) if cpu is not None and last and last[1] is not None else None
        samples.append([round(now - t0, 1), size, cores])
        last = (now, cpu)
    capped = child.poll() is None
    if capped:
        os.killpg(child.pid, signal.SIGKILL)
    out = child.communicate()[0] or ""
    after = cpu_stat()
    res = next((json.loads(line[4:]) for line in out.splitlines() if line.startswith("ONE ")), None)
    sizes = [(t, s) for t, s, _c in samples if s]
    rate = (sizes[-1][1] - sizes[0][1]) / (sizes[-1][0] - sizes[0][0]) / 2**20 if len(sizes) > 1 and sizes[-1][0] > sizes[0][0] else None
    return {"label": label, "threads_env": threads, "cap_s": round(cap_s, 1), "capped": capped,
            "wall_s": round(time.time() - t0, 1), "result": res, "tmp_mb_per_s": round(rate, 3) if rate else None,
            "cpu_stat_before": before, "cpu_stat_after": after, "samples": samples}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--snapshot")
    ap.add_argument("--adapters")
    ap.add_argument("--out")
    ap.add_argument("--fixture", default="nodq", choices=("nodq", "dq"))
    ap.add_argument("--mode", default="both", choices=("both", "confirm"))
    ap.add_argument("--one", action="store_true")
    ap.add_argument("--base")
    ap.add_argument("--adapter")
    ap.add_argument("--dst")
    a = ap.parse_args(argv)
    if a.one:
        return one(a.base, a.adapter, a.dst)
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
    if a.mode == "confirm":
        quota = rec["host_facts"].get("cpu_quota_cpus")
        n = max(1, math.floor(min(quota if isinstance(quota, (int, float)) else rec["affinity"], rec["affinity"])))
        pinned = confirm_run("pinned", base, ad, os.path.join(a.out, f"{a.fixture}-MERGED-Q-pinned.arena"), n, 1800)
        cap = min(10 * pinned["wall_s"], 900)
        default = confirm_run("default", base, ad, os.path.join(a.out, f"{a.fixture}-MERGED-Q-default.arena"), None, cap)
        rec["confirm"] = {"threads_pinned": n, "runs": [pinned, default]}
        for r in (pinned, default):
            res = r["result"] or {}
            print(f"{r['label']}: threads_env={r['threads_env']} torch_threads={res.get('torch_threads')} wall={r['wall_s']} s "
                  f"capped={r['capped']} tmp_rate={r['tmp_mb_per_s']} MB/s qfp={str(res.get('merged_qfp_v1'))[:16]} "
                  f"throttled {r['cpu_stat_before'].get('nr_throttled')}->{r['cpu_stat_after'].get('nr_throttled')}", flush=True)
    for cell, source, ckpt in ([] if a.mode == "confirm" else (("MERGED-Q", "arena", None), ("MERGED-W", "checkpoint", a.snapshot))):
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
    if a.mode == "confirm":
        return 0 if (rec["confirm"]["runs"][0]["result"] or {}).get("merged_qfp_v1") else 3
    return 0 if all(r["status"] == "WRITTEN" for r in rec["cells"].values()) else 3


if __name__ == "__main__":
    raise SystemExit(main())
