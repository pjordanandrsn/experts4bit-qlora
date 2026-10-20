#!/usr/bin/env python3
"""HOST-FACTS PROBE, box side (bench/hostprobe/PREREG-hostprobe.md): print the facts that bound other lanes' reads, and
nothing else. Standard library only: no torch, no model, no measurement of any kind.

    python3 hostprobe_box.py            -> one line, ``HOSTPROBE_JSON {...}``, on stdout; always exits 0
    python3 hostprobe_box.py --root R   -> read /proc and /sys under R instead of / (tests)

A fact that cannot be read is left out, never guessed. A kernel built without THP (no transparent_hugepage directory)
reads ``thp_enabled``/``thp_defrag`` = ``absent``, which is a fact, not a failed read. Every value is a number or a short
enum except ``kernel`` (the release string). ``thp_mem_alloc_enable`` and ``omp_num_threads`` are read from THIS
process's environment, i.e. the ssh session's, not any lane's. GPU 0's power limit, enforced limit and maximum SM clock
come from ``nvidia-smi``'s query, bounded at 20 s."""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

THP_ENABLED = ("always", "madvise", "never")
THP_DEFRAG = ("always", "defer", "defer+madvise", "madvise", "never")


def _read(root: Path, rel: str) -> str | None:
    try:
        return (root / rel).read_text().strip()
    except OSError:
        return None


def _bracketed(text: str | None, allowed: tuple[str, ...]) -> str | None:
    """The selected mode of a sysfs choice line (``always [madvise] never`` -> ``madvise``), only if it is a known one."""
    m = re.search(r"\[([^\]]+)\]", text or "")
    return m.group(1) if m and m.group(1) in allowed else None


def _env_int(name: str) -> int | None:
    v = os.environ.get(name, "").strip()
    return int(v) if v.isdigit() else None


GPU_QUERY = ("power.limit", "enforced.power.limit", "clocks.max.sm")
GPU_KEYS = ("gpu_power_limit_w", "gpu_enforced_power_limit_w", "gpu_max_sm_mhz")


def gpu_facts(smi: str = "nvidia-smi") -> dict:
    """GPU 0's power limit, enforced power limit and maximum SM clock (numbers only; '[N/A]' or a failed call is left
    out). Bounded at 20 s: a hung driver never holds the box."""
    try:
        r = subprocess.run([smi, "--id=0", "--query-gpu=" + ",".join(GPU_QUERY), "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, timeout=20)
    except (OSError, subprocess.SubprocessError):
        return {}
    if r.returncode != 0 or not r.stdout.strip():
        return {}
    out = {}
    for key, raw in zip(GPU_KEYS, r.stdout.strip().splitlines()[0].split(",")):
        try:
            out[key] = float(raw.strip())
        except ValueError:
            continue
    return out


def facts(root: Path = Path("/"), smi: str | None = "nvidia-smi") -> dict:
    out: dict = {}
    if not (root / "sys/kernel/mm/transparent_hugepage").is_dir():
        out["thp_enabled"] = out["thp_defrag"] = "absent"      # a kernel without THP: a fact, not a failed read
    for key, rel, allowed in (("thp_enabled", "sys/kernel/mm/transparent_hugepage/enabled", THP_ENABLED),
                              ("thp_defrag", "sys/kernel/mm/transparent_hugepage/defrag", THP_DEFRAG),
                              ("khugepaged_defrag", "sys/kernel/mm/transparent_hugepage/khugepaged/defrag", None)):
        if key in out:
            continue
        raw = _read(root, rel)
        if allowed is None:
            if raw is not None and raw.isdigit():
                out[key] = int(raw)
        else:
            mode = _bracketed(raw, allowed)
            if mode:
                out[key] = mode
    kernel = _read(root, "proc/sys/kernel/osrelease")
    if kernel and re.fullmatch(r"[0-9A-Za-z._+~-]{1,64}", kernel):
        out["kernel"] = kernel
    bal = _read(root, "proc/sys/kernel/numa_balancing")
    if bal is not None and bal.isdigit():
        out["numa_balancing"] = int(bal)
    meminfo = _read(root, "proc/meminfo") or ""
    m = re.search(r"^MemTotal:\s+(\d+) kB", meminfo, re.M)
    if m:
        out["mem_total_bytes"] = int(m.group(1)) * 1024
    mx = _read(root, "sys/fs/cgroup/memory.max")                                  # cgroup v2
    if mx == "max":
        out["cgroup_memory_max"] = "max"
    elif mx is not None and mx.isdigit():
        out["cgroup_memory_max"] = int(mx)
    else:
        v1 = _read(root, "sys/fs/cgroup/memory/memory.limit_in_bytes")           # cgroup v1
        if v1 is not None and v1.isdigit():
            out["cgroup_memory_max"] = "max" if int(v1) >= 1 << 62 else int(v1)
    cpus = _read(root, "sys/devices/system/cpu/online")
    if cpus and re.fullmatch(r"[0-9,\-]+", cpus):
        n = 0
        for part in cpus.split(","):
            a, _, b = part.partition("-")
            n += (int(b) - int(a) + 1) if b else 1
        out["online_cpus"] = n
    nodes = [p for p in (root / "sys/devices/system/node").glob("node[0-9]*")] if (root / "sys/devices/system/node").is_dir() else []
    if nodes:
        out["numa_nodes"] = len(nodes)
    if str(root) == "/" and hasattr(os, "sched_getaffinity"):
        out["affinity_cpus"] = len(os.sched_getaffinity(0))
    for key, name in (("thp_mem_alloc_enable", "THP_MEM_ALLOC_ENABLE"), ("omp_num_threads", "OMP_NUM_THREADS")):
        v = _env_int(name)
        if v is not None:
            out[key] = v
    if smi:
        out.update(gpu_facts(smi))
    out["measured_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    return out


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    root = Path(argv[argv.index("--root") + 1]) if "--root" in argv else Path("/")
    smi = argv[argv.index("--smi") + 1] if "--smi" in argv else "nvidia-smi"
    try:
        f = facts(root, smi)
    except Exception as e:  # noqa: BLE001 -- a probe never fails its box; the drive records what came back
        f = {"error": f"{type(e).__name__}"}
    print("HOSTPROBE_JSON " + json.dumps(f, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
