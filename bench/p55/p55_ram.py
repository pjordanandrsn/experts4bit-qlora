#!/usr/bin/env python3
# Copyright (c) 2026 Cerin Amroth LLC. MIT.
"""P55 Amendment 1: how much memory a process in THIS container can actually have.

P55's subject is a host-memory class. On Vast an offer's ``cpu_ram`` is the container's allotment, while
``/proc/meminfo`` inside the container reports the whole host (no lxcfs). A 63 GB rental on a 512 GB host reads
MemTotal 512 GB, so the registered STOP-1, which read MemTotal, would have called the class "not drawn". The effective
memory is min(MemTotal, the cgroup memory limit when one is set). The headroom at an instant is the same minimum taken
over (MemAvailable, limit - the cgroup's usage).

    python3 p55_ram.py [--class-max-gib 72] [--meminfo PATH] [--cgroup-dir DIR]

prints ``key: value`` lines (into forensics.txt) and exits 0. It exits 2 when MemTotal cannot be read. A host is in the
class when its effective memory is at most ``--class-max-gib``.

    python3 p55_ram.py --sample [--meminfo PATH] [--cgroup-dir DIR]

prints one CSV row ``epoch,mem_available_kb,mem_free_kb,cgroup_usage_bytes,cgroup_limit_bytes`` (empty fields when
unreadable), for the C_headroom trace.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

GIB = 1 << 30
# cgroup v1 reports "no limit" as a page-rounded 2**63-1; anything this large is no limit at all.
V1_UNLIMITED = 1 << 60
#: the largest shard in google/gemma-4-26B-A4B-it @ 4d7ae49: model-00001-of-00002.safetensors, from the Hub's blob
#: listing. 49.91 GB decimal is 46.48 GiB. The registration's "49.9 GiB" mixed the two units (Amendment 1).
SHARD_BYTES = 49_907_246_508
SHARD_GIB = SHARD_BYTES / GIB


def meminfo_kib(path: str) -> dict[str, int]:
    out: dict[str, int] = {}
    try:
        with open(path) as fh:
            for line in fh:
                parts = line.split()
                if len(parts) >= 2 and parts[0].endswith(":") and parts[1].isdigit():
                    out[parts[0][:-1]] = int(parts[1])
    except OSError:
        pass
    return out


def _read_int(path: str) -> int | None:
    try:
        with open(path) as fh:
            v = fh.read().strip()
    except OSError:
        return None
    return int(v) if v.isdigit() else None


def cgroup_limit(cgroup_dir: str) -> tuple[str, int | None]:
    """(what was read, the limit in bytes or None for no limit). v2 `memory.max` says `max` for none."""
    v2 = os.path.join(cgroup_dir, "memory.max")
    if os.path.exists(v2):
        try:
            with open(v2) as fh:
                raw = fh.read().strip()
        except OSError:
            return ("unreadable", None)
        if raw == "max":
            return (v2 + " = max", None)
        if raw.isdigit():
            return (v2, int(raw))
        return (f"{v2} = {raw!r}", None)
    v1 = os.path.join(cgroup_dir, "memory", "memory.limit_in_bytes")
    n = _read_int(v1)
    if n is None:
        return ("none readable", None)
    return (v1, None if n >= V1_UNLIMITED else n)


def cgroup_usage(cgroup_dir: str) -> int | None:
    for p in (os.path.join(cgroup_dir, "memory.current"), os.path.join(cgroup_dir, "memory", "memory.usage_in_bytes")):
        n = _read_int(p)
        if n is not None:
            return n
    return None


def effective_bytes(mem_total_kib: int, limit: int | None) -> int:
    total = mem_total_kib * 1024
    return total if limit is None else min(total, limit)


def headroom_bytes(mem_available_kib: int | None, usage: int | None, limit: int | None) -> int | None:
    """min(MemAvailable, limit - usage); the cgroup term only when both are known."""
    terms = []
    if mem_available_kib is not None:
        terms.append(mem_available_kib * 1024)
    if limit is not None and usage is not None:
        terms.append(max(0, limit - usage))
    return min(terms) if terms else None


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="p55_ram.py")
    ap.add_argument("--meminfo", default="/proc/meminfo")
    ap.add_argument("--cgroup-dir", default="/sys/fs/cgroup")
    ap.add_argument("--class-max-gib", type=float, default=72.0)
    ap.add_argument("--sample", action="store_true")
    a = ap.parse_args(argv)
    mi = meminfo_kib(a.meminfo)
    src, limit = cgroup_limit(a.cgroup_dir)
    if a.sample:
        usage = cgroup_usage(a.cgroup_dir)
        cells = [str(int(time.time())), str(mi.get("MemAvailable", "")), str(mi.get("MemFree", "")),
                 "" if usage is None else str(usage), "" if limit is None else str(limit)]
        print(",".join(cells))
        return 0
    if "MemTotal" not in mi:
        print(f"mem_total: unreadable ({a.meminfo})")
        return 2
    eff = effective_bytes(mi["MemTotal"], limit)
    print(f"mem_total_gib: {mi['MemTotal'] * 1024 / GIB:.1f}")
    print(f"cgroup_limit_source: {src}")
    print(f"cgroup_limit_bytes: {'none' if limit is None else limit}")
    print(f"effective_ram_gib: {eff / GIB:.1f}")
    print(f"largest_shard_gib: {SHARD_GIB:.2f}")
    print(f"class_max_gib: {a.class_max_gib:g}")
    print(f"class_drawn: {1 if eff <= a.class_max_gib * GIB else 0}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
