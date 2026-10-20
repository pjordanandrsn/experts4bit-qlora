#!/usr/bin/env python3
"""host_probe.py OUT|- -- the box's own measured host facts, written as ``host_facts.json`` beside a run.

ONE producer, two byte-identical copies: ``adertha/compute/host_probe.py`` in adertha-agents (the source of record,
beside the registry that reads its keys) and ``bench/common/host_facts.py`` in experts4bit-qlora. adertha-agents' CI
fails when the two differ. Edits land in adertha-agents first; experts4bit-qlora then copies the file byte for byte.
The launcher runs it on every rented box after the pre-flight (``rent.collect_host_facts``); a lane may also run it.

The host registry reads this file from a run's receipt directory. It keeps only the numeric keys it knows, ignores
the rest, and rejects a file whose ``measured_at`` falls outside the run's own window, so a copy shipped with a kit is
never taken for a reading.

Rules:
- every fact is measured here, on the box, from the kernel or the driver: /proc, /sys and nvidia-smi;
- ``host_*`` facts describe the WHOLE host, as /proc and /sys see it, which in a container is larger than what the run
  may use. The run's own share is ``affinity_cpus`` and the cgroup quota (``cpu_quota_cpus`` or
  ``cpu_quota_unlimited``);
- every value is a number or a small enum, and ``sources`` names where each one came from;
- no identifiers: no hostname, container id, serial, UUID or CPU model string;
- a fact that cannot be read is left out. The helper never fails a run: it prints why and exits 0.

Usage, on the box (one call, before or after the work):
    python3 host_probe.py "$OUT_DIR/host_facts.json"
    python3 host_probe.py -          one line on stdout, ``HOST_FACTS_JSON {...}`` (the launcher reads it over ssh)

Facts read from the ENVIRONMENT (``omp_num_threads``, ``thp_mem_alloc_enable``) are the calling process's own: run it
with the run's environment, or read them as the caller's. The launcher's call records them as its ssh session's.
"""
from __future__ import annotations

import datetime as dt
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

THP_MODES = ("always", "madvise", "never", "defer", "defer+madvise")


def _read(root: Path, rel: str) -> str | None:
    try:
        return (root / rel).read_text()
    except OSError:
        return None


def _bracketed(text: str | None) -> str | None:
    """The selected value of a /sys enum file, e.g. 'always [madvise] never' -> 'madvise'."""
    m = re.search(r"\[([^\]]+)\]", text or "")
    return m.group(1) if m and m.group(1) in THP_MODES else None


LOOP_N = 1_000_000   # the fixed single-thread loop's iterations: the same code and count in every kit


def cpu_loop_mops() -> float:
    """A fixed single-thread pure-Python loop, best of three, in million iterations per second: what THIS run got from
    one core (clock, co-tenants, quota), not the part's rated boost. Comparable only at the same Python minor
    (``cpu_loop_python``); well under a second on any current host. Tests replace it."""
    best = None
    for _ in range(3):
        t0, x = time.perf_counter(), 0
        for i in range(LOOP_N):
            x = (x + i * 7) & 0xFFFF
        dt = time.perf_counter() - t0
        best = dt if best is None or dt < best else best
    return round(LOOP_N / best / 1e6, 2)


def affinity() -> set:
    """The CPUs this process may run on (Linux only; tests replace it)."""
    return os.sched_getaffinity(0)


def gather(root: Path = Path("/"), smi: str = "nvidia-smi") -> dict:
    """The measured facts and their sources. ``root`` lets tests point it at a fake /proc and /sys."""
    facts: dict = {}
    sources: dict = {}
    cpuinfo = _read(root, "proc/cpuinfo")
    if cpuinfo:
        logical = len(re.findall(r"(?m)^processor\s*:", cpuinfo))
        cores = set()
        for block in cpuinfo.split("\n\n"):
            phys = re.search(r"(?m)^physical id\s*:\s*(\d+)", block)
            core = re.search(r"(?m)^core id\s*:\s*(\d+)", block)
            if phys and core:
                cores.add((phys.group(1), core.group(1)))
        if logical:
            facts["host_logical_cpus"] = logical
            sources["host_logical_cpus"] = "/proc/cpuinfo: processor entries"
        if cores:
            facts["host_physical_cores"] = len(cores)
            sources["host_physical_cores"] = "/proc/cpuinfo: distinct (physical id, core id) pairs"
            if logical:
                facts["host_threads_per_core"] = round(logical / len(cores), 3)
                sources["host_threads_per_core"] = "host_logical_cpus / host_physical_cores"
    node_dir = root / "sys/devices/system/node"
    nodes = list(node_dir.glob("node[0-9]*")) if node_dir.is_dir() else []
    if nodes:
        facts["host_numa_nodes"] = len(nodes)
        sources["host_numa_nodes"] = "/sys/devices/system/node/node*"
    for key, rel in (("thp_enabled", "sys/kernel/mm/transparent_hugepage/enabled"),
                     ("thp_defrag", "sys/kernel/mm/transparent_hugepage/defrag")):
        v = _bracketed(_read(root, rel))
        if v:
            facts[key] = v
            sources[key] = f"/{rel}"
    bal = (_read(root, "proc/sys/kernel/numa_balancing") or "").strip()
    if bal.isdigit():
        facts["numa_balancing"] = int(bal)
        sources["numa_balancing"] = "/proc/sys/kernel/numa_balancing"
    # the run's CPU allotment, which in a container is smaller than the host /proc describes: the cgroup quota (v2,
    # else v1) and the affinity mask. "Unlimited" is recorded as such, so it is never confused with "unreadable".
    quota = None
    cpu_max = (_read(root, "sys/fs/cgroup/cpu.max") or "").split()
    if len(cpu_max) == 2 and cpu_max[1].isdigit() and int(cpu_max[1]) > 0:
        if cpu_max[0] == "max":
            quota = ("unlimited", "/sys/fs/cgroup/cpu.max: max")
        elif cpu_max[0].isdigit():
            quota = (round(int(cpu_max[0]) / int(cpu_max[1]), 3), "/sys/fs/cgroup/cpu.max: quota / period")
    if quota is None:
        for d in ("sys/fs/cgroup/cpu", "sys/fs/cgroup/cpu,cpuacct"):
            q = (_read(root, f"{d}/cpu.cfs_quota_us") or "").strip()
            per = (_read(root, f"{d}/cpu.cfs_period_us") or "").strip()
            if per.isdigit() and int(per) > 0 and (q == "-1" or q.isdigit()):
                quota = ("unlimited", f"/{d}/cpu.cfs_quota_us: -1") if q == "-1" else (
                    round(int(q) / int(per), 3), f"/{d}/cpu.cfs_quota_us / cpu.cfs_period_us (cgroup v1)")
                break
    if quota is not None:
        key = "cpu_quota_unlimited" if quota[0] == "unlimited" else "cpu_quota_cpus"
        facts[key] = 1 if quota[0] == "unlimited" else quota[0]
        sources[key] = quota[1]
    # per-core speed: the rated maximum (a label) and a measured single-thread rate (what the run got), and the run's
    # OpenMP thread count; a framework's speed ratio moved with these, not with core count (the TC1 receipts, 10-20)
    # each of the run's CPUs (affinity; cpu0 when there is no affinity call): a hybrid part's P and E cores differ, so
    # cpu0 alone would label the host with one core type. The highest and the lowest per-CPU maximum are recorded.
    try:
        cpus = sorted(affinity())
    except (AttributeError, OSError):
        cpus = [0]
    maxes = []
    for c in cpus:
        v = (_read(root, f"sys/devices/system/cpu/cpu{c}/cpufreq/cpuinfo_max_freq") or "").strip()
        if v.isdigit() and int(v) > 0:
            maxes.append(int(v))
    if maxes:
        src = (f"/sys/devices/system/cpu/cpu*/cpufreq/cpuinfo_max_freq (kHz / 1000) over the run's {len(maxes)} "
               "readable CPU(s): the rated maximum")
        facts["cpu_max_mhz"], sources["cpu_max_mhz"] = round(max(maxes) / 1000, 1), src + ", highest"
        facts["cpu_max_mhz_lo"], sources["cpu_max_mhz_lo"] = round(min(maxes) / 1000, 1), src + ", lowest"
    try:
        facts["cpu_loop_mops"] = cpu_loop_mops()
        sources["cpu_loop_mops"] = f"a fixed {LOOP_N}-iteration single-thread Python loop, best of 3 (Mops/s)"
        facts["cpu_loop_python"] = sys.version_info[0] * 100 + sys.version_info[1]
        sources["cpu_loop_python"] = "sys.version_info: the loop is comparable only at the same Python (e.g. 312)"
    except Exception:  # noqa: BLE001 -- a missing fact never costs the run
        pass
    # the helper's OWN environment: call it with the run's env set. Unset means the runtime's default, which differs by
    # runtime (libgomp: the affinity count; torch intra-op: the physical cores), so absent is recorded as absent.
    thp_env = os.environ.get("THP_MEM_ALLOC_ENABLE", "").strip()
    if thp_env.isdigit():
        facts["thp_mem_alloc_enable"] = int(thp_env)
        sources["thp_mem_alloc_enable"] = "the calling process's THP_MEM_ALLOC_ENABLE (left out when unset: absent is not 0)"
    omp = os.environ.get("OMP_NUM_THREADS", "").strip()
    if omp.isdigit() and int(omp) > 0:
        facts["omp_num_threads"] = int(omp)
        sources["omp_num_threads"] = "the calling process's OMP_NUM_THREADS (left out when unset: absent is not 0)"
    try:
        aff = len(affinity())
        if aff:
            facts["affinity_cpus"] = aff
            sources["affinity_cpus"] = "os.sched_getaffinity(0)"
    except (AttributeError, OSError):
        pass                                  # not Linux: no affinity call
    try:
        query = ["--query-gpu=power.limit,clocks.max.sm,memory.total", "--format=csv,noheader,nounits"]
        out = subprocess.run([smi, *query], capture_output=True, text=True, timeout=20)
        first = out.stdout.strip().splitlines()[0] if out.returncode == 0 and out.stdout.strip() else ""
        vals = [v.strip() for v in first.split(",")]
        if len(vals) == 3:
            for key, v, cast in (("gpu_power_limit_w", vals[0], float), ("gpu_max_sm_mhz", vals[1], float),
                                 ("gpu_mem_total_mib", vals[2], float)):
                try:
                    facts[key] = cast(v)
                    sources[key] = "nvidia-smi --query-gpu=power.limit,clocks.max.sm,memory.total (GPU 0)"
                except ValueError:
                    pass
    except (OSError, subprocess.SubprocessError):
        pass
    facts["measured_at"] = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    facts["sources"] = sources
    return facts


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 1:
        print("usage: host_probe.py OUT|-", file=sys.stderr)
        return 0
    try:
        facts = gather(Path(os.environ.get("HOST_FACTS_ROOT", "/")), os.environ.get("HOST_FACTS_SMI", "nvidia-smi"))
        if argv[0] == "-":
            print("HOST_FACTS_JSON " + json.dumps(facts, sort_keys=True), flush=True)
            return 0
        Path(argv[0]).write_text(json.dumps(facts, indent=1, sort_keys=True) + "\n")
        print(f"HOST_FACTS {len(facts) - 2} facts -> {argv[0]}")
    except Exception as e:  # noqa: BLE001 -- a missing fact never costs the run
        print(f"HOST_FACTS not written: {type(e).__name__}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
