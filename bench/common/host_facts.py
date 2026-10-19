#!/usr/bin/env python3
"""bench/common/host_facts.py OUT -- the box's own measured host facts, written as ``host_facts.json`` beside a run.

The host registry reads this file from a run's receipt directory. It keeps only the numeric keys it knows, ignores the
rest, and rejects a file whose ``measured_at`` falls outside the run's own window, so a copy shipped with a kit is never
taken for a reading.

Rules:
- every fact is measured here, on the box, from the kernel or the driver: /proc, /sys and nvidia-smi;
- every value is a number or a small enum, and ``sources`` names where each one came from;
- no identifiers: no hostname, container id, serial, UUID or CPU model string;
- a fact that cannot be read is left out. The helper never fails a run: it prints why and exits 0.

Usage, on the box, from the lane's runner (one call, before or after the work):
    python3 host_facts.py "$OUT_DIR/host_facts.json"
"""
from __future__ import annotations

import datetime as dt
import json
import os
import re
import subprocess
import sys
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
            facts["logical_cpus"] = logical
            sources["logical_cpus"] = "/proc/cpuinfo: processor entries"
        if cores:
            facts["physical_cores"] = len(cores)
            sources["physical_cores"] = "/proc/cpuinfo: distinct (physical id, core id) pairs"
            if logical:
                facts["threads_per_core"] = round(logical / len(cores), 3)
                sources["threads_per_core"] = "logical_cpus / physical_cores"
    nodes = [p for p in (root / "sys/devices/system/node").glob("node[0-9]*")] if (root / "sys/devices/system/node").is_dir() else []
    if nodes:
        facts["numa_nodes"] = len(nodes)
        sources["numa_nodes"] = "/sys/devices/system/node/node*"
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
    cpu_max = (_read(root, "sys/fs/cgroup/cpu.max") or "").split()
    if len(cpu_max) == 2 and cpu_max[0].isdigit() and cpu_max[1].isdigit() and int(cpu_max[1]) > 0:
        facts["cpu_quota_cpus"] = round(int(cpu_max[0]) / int(cpu_max[1]), 3)
        sources["cpu_quota_cpus"] = "/sys/fs/cgroup/cpu.max: quota / period"
    try:
        out = subprocess.run([smi, "--query-gpu=power.limit,clocks.max.sm,memory.total", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=20)
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
        print("usage: host_facts.py OUT", file=sys.stderr)
        return 0
    try:
        facts = gather(Path(os.environ.get("HOST_FACTS_ROOT", "/")), os.environ.get("HOST_FACTS_SMI", "nvidia-smi"))
        Path(argv[0]).write_text(json.dumps(facts, indent=1, sort_keys=True) + "\n")
        print(f"HOST_FACTS {len(facts) - 2} facts -> {argv[0]}")
    except Exception as e:  # noqa: BLE001 -- a missing fact never costs the run
        print(f"HOST_FACTS not written: {type(e).__name__}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
