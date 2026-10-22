#!/usr/bin/env python3
"""p225_probe.py -- lane P225's on-box host screen (bench/p225/PREREG-p225.md, "The host screen"). Pure Python plus the
standard library, so it runs before anything is installed.

``--floors TIER`` measures the registered floors and refuses mechanically: it prints one ``P225_SCREEN {json}`` line
and exits 0 when every floor passes, or 19 (HOST_REFUSED, never a strike) when any fails. The refusal reads ONLY the
floor values in :data:`FLOORS` (tests/test_p225.py asserts it): single-thread speed (the median of 5 runs of the
launcher's own ``cpu_loop_mops`` loop, each at least 2 s), the sustained download, host RAM and free disk. Everything
else it measures is recorded, never refusing: the cgroup CPU quota, the visible CPUs and the /proc/stat steal share.

``--boundary NAME`` re-probes single-thread speed and steal at a phase boundary (one ``P225_BOUNDARY {json}`` line,
never refusing), so a speed change during a run shows in the receipt.

    python p225_probe.py --floors a --work /root/p225 --model Qwen/Qwen3-30B-A3B --revision <sha>
    python p225_probe.py --boundary train
    python p225_probe.py --self-test
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import statistics
import sys
import time
import urllib.request

#: the registered floors (PREREG "The host screen"): one tier, the ruling; no fallback tier at the registered N
FLOORS = {
    "a": {"mops": 30.0, "download_mb_s": 60.0, "ram_gib": 60, "disk_gb": 100},
}
MOPS_RUNS, MOPS_MIN_S = 5, 2.0              # the median of 5 runs, each at least 2 s
DL_MIN_BYTES, DL_MIN_S, DL_MAX_S = 256 << 20, 20.0, 60.0   # at least 256 MiB or 20 s; give up at 60 s
HOST_REFUSED = 19


LOOP_N = 1_000_000   # adertha.compute.host_probe.LOOP_N: the same code and count as the launcher's cpu_loop_mops


def _loop() -> int:
    """The launcher's host_facts ``cpu_loop_mops`` body, verbatim (adertha/compute/host_probe.py): comparable only at
    the same Python minor, which is recorded beside it."""
    x = 0
    for i in range(LOOP_N):
        x = (x + i * 7) & 0xFFFF
    return x


def mops_once(min_s: float = MOPS_MIN_S) -> float:
    """Mops over at least ``min_s`` seconds of back-to-back loops (iterations per microsecond)."""
    t0 = time.perf_counter()
    n = 0
    while True:
        _loop()
        n += 1
        dt = time.perf_counter() - t0
        if dt >= min_s:
            return n * LOOP_N / dt / 1e6


def mops_median(runs: int = MOPS_RUNS, min_s: float = MOPS_MIN_S) -> dict:
    vals = [mops_once(min_s) for _ in range(runs)]
    return {"median": round(statistics.median(vals), 2), "runs": [round(v, 2) for v in vals], "min_s": min_s,
            "python": "%d.%d" % sys.version_info[:2]}


def cpu_quota():
    """The cgroup CPU quota in CPUs (v2 ``cpu.max``, else v1), or None when unlimited or unreadable. Recorded only."""
    try:
        a, b = open("/sys/fs/cgroup/cpu.max").read().split()
        return None if a == "max" else round(int(a) / int(b), 2)
    except (OSError, ValueError):
        pass
    try:
        a = int(open("/sys/fs/cgroup/cpu/cpu.cfs_quota_us").read())
        b = int(open("/sys/fs/cgroup/cpu/cpu.cfs_period_us").read())
        return None if a < 0 else round(a / b, 2)
    except (OSError, ValueError):
        return None


def _cpu_times():
    f = open("/proc/stat").readline().split()
    return [int(x) for x in f[1:]]


def steal_share(seconds: float = 1.0):
    """The /proc/stat steal share over ``seconds`` (field 8 of the cpu line), or None where unreadable. Recorded only."""
    try:
        a = _cpu_times()
        time.sleep(seconds)
        b = _cpu_times()
    except (OSError, ValueError, IndexError):
        return None
    d = [y - x for x, y in zip(a, b)]
    tot = sum(d)
    return round(d[7] / tot, 4) if tot > 0 and len(d) > 7 else None


def ram_gib():
    try:
        for line in open("/proc/meminfo"):
            if line.startswith("MemTotal:"):
                return int(line.split()[1]) // 1048576
    except OSError:
        pass
    return None


def free_gb(path: str):
    try:
        return shutil.disk_usage(path).free // 10**9
    except OSError:
        return None


def download(model: str, revision: str) -> dict:
    """One sustained GET of the checkpoint's first shard from the HF endpoint the fetch uses: it reads until 256 MiB
    AND 20 s have passed, or the body ends, or 60 s. A non-2xx status, or an error, is a refusal reason, never a rate."""
    try:
        meta = json.load(urllib.request.urlopen(f"https://huggingface.co/api/models/{model}/revision/{revision}",
                                                timeout=30))
        shard = sorted(s["rfilename"] for s in meta["siblings"] if s["rfilename"].endswith(".safetensors"))[0]
        req = urllib.request.Request(f"https://huggingface.co/{model}/resolve/{revision}/{shard}")
        t0, n = time.time(), 0
        with urllib.request.urlopen(req, timeout=60) as r:
            status = r.status
            while True:
                dt = time.time() - t0
                if (n >= DL_MIN_BYTES and dt >= DL_MIN_S) or dt >= DL_MAX_S:
                    break
                b = r.read(1 << 20)
                if not b:
                    break
                n += len(b)
        dt = time.time() - t0
        return {"status": status, "bytes": n, "seconds": round(dt, 2), "mb_s": round(n / dt / 1e6, 1) if dt else None}
    except Exception as e:  # noqa: BLE001 -- a refusal reason, recorded by name
        return {"status": getattr(e, "code", None), "error": f"{type(e).__name__}: {str(e)[:120]}", "mb_s": None}


def judge(tier: str, measured: dict) -> list:
    """The refusal reasons: a pure function of the FLOORS and the four floor measurements, nothing else."""
    f = FLOORS[tier]
    why = []
    m = (measured.get("mops") or {}).get("median")
    if m is None or m < f["mops"]:
        why.append(f"single-thread {m} Mops < {f['mops']} (median of {MOPS_RUNS})")
    dl = measured.get("download") or {}
    if dl.get("mb_s") is None or not (200 <= (dl.get("status") or 0) < 300) or dl["mb_s"] < f["download_mb_s"]:
        why.append(f"download {dl.get('mb_s')} MB/s (status {dl.get('status')}) < {f['download_mb_s']}")
    r = measured.get("ram_gib")
    if r is None or r < f["ram_gib"]:
        why.append(f"host RAM {r} GiB < {f['ram_gib']}")
    d = measured.get("free_gb")
    if d is None or d < f["disk_gb"]:
        why.append(f"free disk {d} GB < {f['disk_gb']}")
    return why


def floors(tier: str, work: str, model: str, revision: str) -> int:
    measured = {"tier": tier, "floors": FLOORS[tier], "mops": mops_median(), "download": download(model, revision),
                "ram_gib": ram_gib(), "free_gb": free_gb(work),
                "recorded": {"cpu_quota": cpu_quota(), "visible_cpus": os.cpu_count(), "steal": steal_share()}}
    measured["refused"] = judge(tier, measured)
    print("P225_SCREEN " + json.dumps(measured), flush=True)
    return HOST_REFUSED if measured["refused"] else 0


def boundary(name: str) -> int:
    print("P225_BOUNDARY " + json.dumps({"phase": name, "mops": mops_median(runs=3), "steal": steal_share(),
                                          "cpu_quota": cpu_quota()}), flush=True)
    return 0


def self_test() -> int:
    n = 0

    def ok(c, msg):
        nonlocal n
        assert c, msg
        n += 1
    good = {"mops": {"median": 31.0}, "download": {"status": 206, "mb_s": 70.0}, "ram_gib": 251, "free_gb": 120}
    ok(judge("a", good) == [], "a host above every floor passes tier a")
    ok(len(judge("a", dict(good, mops={"median": 29.9}))) == 1, "29.9 Mops refuses under tier a")
    ok(len(judge("a", dict(good, mops={"median": 18.0}))) == 1, "18.0 Mops refuses: there is no fallback tier")
    ok(len(judge("a", dict(good, download={"status": 429, "mb_s": 500.0}))) == 1, "a 429 is a refusal, never a rate")
    ok(len(judge("a", dict(good, download={"status": 206, "mb_s": None, "error": "x"}))) == 1, "a failed GET refuses")
    ok(len(judge("a", dict(good, ram_gib=None))) == 1, "an unread RAM refuses (unread is not passing)")
    ok(len(judge("a", dict(good, free_gb=99))) == 1, "99 GB free refuses")
    ok(len(judge("a", {"mops": {}, "download": {}, "ram_gib": None, "free_gb": None})) == 4, "nothing read: four reasons")
    r = mops_median(runs=3, min_s=0.05)
    ok(len(r["runs"]) == 3 and r["median"] > 0, "the speed probe runs and reports a median")
    ok(set(FLOORS) == {"a"} and FLOORS["a"]["mops"] == 30.0, "the registered floors")
    print(f"p225_probe self-test OK ({n} checks)")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--floors", choices=sorted(FLOORS))
    ap.add_argument("--boundary")
    ap.add_argument("--work", default=".")
    ap.add_argument("--model")
    ap.add_argument("--revision")
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--check-args", action="store_true", help="apply the argument checks a real run applies, then stop")
    a = ap.parse_args(argv)
    if a.self_test:
        return self_test()
    if a.floors:
        if not (a.model and a.revision):
            ap.error("--floors needs --model and --revision")
        if a.check_args:
            print("ARGS OK")
            return 0
        return floors(a.floors, a.work, a.model, a.revision)
    if a.boundary:
        if a.check_args:
            print("ARGS OK")
            return 0
        return boundary(a.boundary)
    ap.error("one of --floors, --boundary or --self-test")
    return 2


if __name__ == "__main__":
    sys.exit(main())
