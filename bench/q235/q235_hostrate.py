# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""The host rates Q235's time-left fit rests on (gate 2): the step is link-bound and the load reads the checkpoint from disk, and
both belong to the host, not the card. Measured the same way on the A2000 rehearsal and on the box, after install and before the
checkpoint fetch, so the box can refuse a host slower than the fit assumed.

- **link**: pinned host-to-device and device-to-host GB/s, timed with CUDA events over ``--copies`` copies of a 1 GiB pinned
  buffer after a warm-up, and the PCIe generation and width the card negotiated (``nvidia-smi``, recorded beside the copy, which
  is the arbiter);
- **host copy**: the host's memory-copy rate, a multi-threaded tensor copy of 2 GiB (what the loader's host-side expert assembly
  does), against which a CPU-bound load is checked;
- **disk**: the read rate of a freshly written file of ``--disk-gib`` random bytes in the working directory, with ``O_DIRECT``
  (``dd iflag=direct``) where the filesystem supports it, else after evicting its pages (``posix_fadvise DONTNEED``). The method
  is recorded; a cache-served read overstates the disk, so ``--require-direct`` (the box) refuses rather than fall back.
- **link, by buffer** (Amendment 9; REPORTED only, no floor reads it): pinned host-to-device GB/s from ``--multi-buffers`` separately
  pinned 1 GiB buffers, each allocated with the process bound to the CPUs of one NUMA node in turn and timed alone, beside the NUMA
  node each buffer's pages report, the card's own NUMA node and its IOMMU group's mode (sysfs). On ``q235-pro6000-3``'s host the
  single buffer above read 20.39 GB/s while staging, which reads 94 layers' separately pinned homes, ran at 26.6; one buffer is one
  sample of placement. Buffer 0 is timed first and again last, so a link still warming up is not read as placement. ``h2d_gb_s``
  stays the single-buffer figure, so every floor and fit reads what it read before.

    python q235_hostrate.py --out hostrate.json [--min-h2d-gbs X --min-host-copy-gbs Z --min-disk-gbs Y]
                                                  (exit 18 below the link or host-copy floor, 13 below the disk's)
"""
from __future__ import annotations

import argparse
import contextlib
import json
import os
import re
import statistics
import subprocess
import sys
import time
from pathlib import Path

GIB = 1 << 30


def link(copies: int) -> dict:
    import torch
    if not torch.cuda.is_available():
        return {"h2d_gb_s": None, "d2h_gb_s": None, "why": "no CUDA device"}
    host = torch.empty(GIB, dtype=torch.uint8).pin_memory()
    dev = torch.empty(GIB, dtype=torch.uint8, device="cuda")
    out = {}
    for name, dst, src in (("h2d_gb_s", dev, host), ("d2h_gb_s", host, dev)):
        for _ in range(2):
            dst.copy_(src, non_blocking=True)
        torch.cuda.synchronize()
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(copies):
            dst.copy_(src, non_blocking=True)
        end.record()
        torch.cuda.synchronize()
        out[name] = round(copies * GIB / (start.elapsed_time(end) / 1e3) / 1e9, 2)
    try:                                                                # Amendment 9, reported: where the buffer above sits
        out["single_buffer_pages"] = page_nodes(host.data_ptr(), GIB)
    except Exception as e:                                              # noqa: BLE001
        out["single_buffer_pages"] = {"method": None, "why": f"error: {type(e).__name__}: {e}"[:300]}
    q = "pcie.link.gen.current,pcie.link.gen.max,pcie.link.width.current,pcie.link.width.max"
    try:
        gen, gen_max, width, width_max = [v.strip() for v in subprocess.check_output(
            ["nvidia-smi", f"--query-gpu={q}", "--format=csv,noheader"], text=True).splitlines()[0].split(",")]
        out.update(pcie_gen=gen, pcie_gen_max=gen_max, pcie_width=width, pcie_width_max=width_max)
    except (OSError, subprocess.CalledProcessError, ValueError) as e:
        out["pcie_why"] = f"{type(e).__name__}: {e}"
    bus = None
    with contextlib.suppress(OSError, subprocess.CalledProcessError, IndexError):
        bus = subprocess.check_output(["nvidia-smi", "--query-gpu=pci.bus_id", "--format=csv,noheader"], text=True).splitlines()[0]
    try:
        out["card"] = card_placement(bus)                               # Amendment 9, reported
    except Exception as e:                                              # noqa: BLE001
        out["card"] = {"pci_bus_id": bus, "why": f"error: {type(e).__name__}: {e}"[:300]}
    return out


def _cpulist(s: str) -> list[int]:
    """A sysfs CPU list (``0-3,8,10-11``) as CPU ids."""
    out = []
    for part in s.strip().split(","):
        if "-" in part:
            a, b = part.split("-")
            out += range(int(a), int(b) + 1)
        elif part:
            out.append(int(part))
    return out


def numa_topology(root: Path = Path("/sys")) -> dict:
    """The NUMA nodes that have CPUs, from ``devices/system/node``: each node's CPU ids and memory. Empty where sysfs does not expose
    them (a single implicit node)."""
    nodes = {}
    for d in sorted((root / "devices" / "system" / "node").glob("node[0-9]*"), key=lambda d: int(d.name[4:])):
        try:
            cpus = _cpulist((d / "cpulist").read_text())
        except (OSError, ValueError):
            continue
        if not cpus:
            continue
        mem = None
        with contextlib.suppress(OSError):
            m = re.search(r"MemTotal:\s+(\d+) kB", (d / "meminfo").read_text())
            mem = int(m.group(1)) * 1024 if m else None
        nodes[int(d.name[4:])] = {"cpus": cpus, "mem_bytes": mem}
    return nodes


def sysfs_bdf(bus_id: str) -> str:
    """``nvidia-smi``'s ``00000000:41:00.0`` as sysfs's ``0000:41:00.0``."""
    dom, rest = bus_id.strip().lower().split(":", 1)
    return f"{dom[-4:]}:{rest}"


def card_placement(bus_id: str | None, root: Path = Path("/sys"), cmdline: Path = Path("/proc/cmdline")) -> dict:
    """The card's NUMA node (sysfs ``numa_node``; -1 = the platform reports none), its IOMMU group's default-domain type (``identity``
    = passthrough, ``DMA`` / ``DMA-FQ`` = translated; no group = no IOMMU in use or none exposed) and the kernel command line's IOMMU
    tokens. Each field is None, with a reason, where the container does not expose it."""
    out: dict = {"pci_bus_id": bus_id}
    with contextlib.suppress(OSError):
        out["kernel_iommu_args"] = [w for w in cmdline.read_text().split() if "iommu" in w.lower()]
    if not bus_id:
        out["why"] = "no PCI bus id for the card"
        return out
    dev = root / "bus" / "pci" / "devices" / sysfs_bdf(bus_id)
    if not dev.exists():
        out["why"] = f"{dev} is not exposed here"
        return out
    try:
        out["gpu_numa_node"] = int((dev / "numa_node").read_text())
    except (OSError, ValueError) as e:
        out["gpu_numa_node"] = None
        out["gpu_numa_why"] = f"no numa_node in sysfs (a kernel without NUMA, or not exposed): {type(e).__name__}"
    group = dev / "iommu_group"
    if group.exists():
        out["iommu_group"] = os.path.basename(os.path.realpath(group))
        with contextlib.suppress(OSError):
            out["iommu_type"] = (Path(os.path.realpath(group)) / "type").read_text().strip()
    else:
        out["iommu_type"] = None
        out["iommu_why"] = "the card has no IOMMU group (no IOMMU in use, or none exposed)"
    return out


def numa_maps_nodes(ptr: int, maps: str, numa_maps: str) -> dict | None:
    """The page counts per node (``N<node>=<pages>``) that ``/proc/self/numa_maps`` gives for the mapping holding ``ptr``, found via
    ``/proc/self/maps``. None when the mapping is absent or lists no node counts (a driver mapping often lists none)."""
    start = None
    for line in maps.splitlines():
        lo, hi = (int(x, 16) for x in line.split()[0].split("-"))
        if lo <= ptr < hi:
            start = lo
            break
    if start is None:
        return None
    for line in numa_maps.splitlines():
        f = line.split()
        if f and int(f[0], 16) == start:
            counts = {k[1:]: int(v) for k, v in (w.split("=", 1) for w in f[2:] if re.fullmatch(r"N\d+=\d+", w))}
            return {"policy": f[1], "pages": counts} if counts else None
    return None


def page_nodes(ptr: int, nbytes: int, samples: int = 16) -> dict:
    """Where a buffer's pages sit: ``move_pages(2)`` queried (no move) at ``samples`` pages spread over the buffer; where the kernel
    refuses that (a driver-pinned mapping), ``numa_maps``'s counts for the mapping. ``method`` names which answered; ``why`` says
    when neither could."""
    if sys.platform != "linux":
        return {"method": None, "why": f"not Linux ({sys.platform})"}
    import ctypes
    import platform
    nr = {"x86_64": 279, "aarch64": 239}.get(platform.machine())
    out: dict = {}
    if nr is not None:
        page = os.sysconf("SC_PAGE_SIZE")
        first, last = -(-ptr // page) * page, (ptr + nbytes) // page * page - page
        addrs = sorted({first + (last - first) * i // max(samples - 1, 1) // page * page for i in range(samples)})
        arr, status = (ctypes.c_void_p * len(addrs))(*addrs), (ctypes.c_int * len(addrs))()
        libc = ctypes.CDLL(None, use_errno=True)
        rc = libc.syscall(ctypes.c_long(nr), ctypes.c_int(0), ctypes.c_ulong(len(addrs)), arr, None, status, ctypes.c_int(0))
        if rc == 0:
            nodes, errs = {}, {}
            for s in status:
                if s >= 0:
                    nodes[str(s)] = nodes.get(str(s), 0) + 1
                else:
                    errs[str(-s)] = errs.get(str(-s), 0) + 1
            if nodes:
                return {"method": "move_pages", "sampled": len(addrs), "nodes": nodes, "errno": errs}
            out["move_pages"] = {"sampled": len(addrs), "errno": errs}
        else:
            err = ctypes.get_errno()
            out["move_pages"] = {"rc": rc, "errno": err}
            if err == 1:
                out["move_pages"]["why"] = "EPERM: a seccomp filter refuses move_pages (a container's default needs CAP_SYS_NICE)"
    try:
        got = numa_maps_nodes(ptr, Path("/proc/self/maps").read_text(), Path("/proc/self/numa_maps").read_text())
    except OSError as e:
        got, out["numa_maps_why"] = None, f"{type(e).__name__}: {e}"
    if got:
        return {"method": "numa_maps", **got, **out}
    return {"method": None, "why": "neither move_pages nor numa_maps reports this buffer's pages", **out}


def link_by_buffer(buffers: int, copies: int) -> dict:
    """Amendment 9: pinned H2D from ``buffers`` separately pinned 1 GiB buffers, each allocated with this thread bound to one NUMA
    node's CPUs in turn (round robin; unbound where the host exposes no nodes or no affinity call) and timed alone with CUDA events
    after one warm-up copy. Buffer 0 is timed again after the rest (``first_last_gb_s``). Reported only."""
    import torch
    if not torch.cuda.is_available() or buffers <= 0:
        return {"h2d_by_buffer": None, "h2d_by_buffer_why": "no CUDA device" if buffers > 0 else "disabled (--multi-buffers 0)"}
    emptied = hasattr(torch._C, "_host_emptyCache")
    if emptied:
        torch._C._host_emptyCache()        # no buffer below may be a cached block from the single-buffer copy above
    topo = numa_topology()
    ids = sorted(topo)
    can_bind = bool(ids) and hasattr(os, "sched_setaffinity")
    dev = torch.empty(GIB, dtype=torch.uint8, device="cuda")
    bufs, per = [], []
    for i in range(buffers):
        node = ids[i % len(ids)] if can_bind else None
        old = os.sched_getaffinity(0) if can_bind else None
        try:
            if node is not None:
                os.sched_setaffinity(0, topo[node]["cpus"])
            h = torch.empty(GIB, dtype=torch.uint8, pin_memory=True)
            h.fill_(i + 1)                                     # content only: the pages were placed when pinned, above
        finally:
            if old is not None:
                os.sched_setaffinity(0, old)
        bufs.append(h)
        per.append({"bound_node": node, "pages": page_nodes(h.data_ptr(), GIB)})

    def timed(h):
        dev.copy_(h, non_blocking=True)
        torch.cuda.synchronize()
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(copies):
            dev.copy_(h, non_blocking=True)
        end.record()
        torch.cuda.synchronize()
        return round(copies * GIB / (start.elapsed_time(end) / 1e3) / 1e9, 2)

    for p, h in zip(per, bufs):
        p["gb_s"] = timed(h)
    first_last = [per[0]["gb_s"], timed(bufs[0])]
    rates = sorted(p["gb_s"] for p in per)
    by_bound: dict = {}
    by_observed: dict = {}
    for p in per:
        key = p["pages"].get("nodes") or p["pages"].get("pages")
        p["observed_node"] = max(key, key=key.get) if key else None
        if p["bound_node"] is not None:
            by_bound.setdefault(str(p["bound_node"]), []).append(p["gb_s"])
        if p["observed_node"] is not None:
            by_observed.setdefault(str(p["observed_node"]), []).append(p["gb_s"])
    del bufs, h                            # every pinned buffer (h: the allocation loop's last), so the cache can let them go
    if emptied:
        torch._C._host_emptyCache()        # hand the 8 GiB back before the host-copy and disk reads
    return {"h2d_by_buffer": {
        "buffers": buffers, "bytes_each": GIB, "copies_each": copies, "per_buffer": per, "first_last_gb_s": first_last,
        "gb_s_min": rates[0], "gb_s_median": round(statistics.median(rates), 2), "gb_s_max": rates[-1],
        "by_bound_node_median_gb_s": {k: round(statistics.median(v), 2) for k, v in sorted(by_bound.items())},
        "by_observed_node_median_gb_s": ({k: round(statistics.median(v), 2) for k, v in sorted(by_observed.items())}
                                         if all(p["observed_node"] is not None for p in per) else None),
        "numa_nodes": {str(k): {"cpus": len(v["cpus"]), "mem_bytes": v["mem_bytes"]} for k, v in topo.items()},
        "bound": can_bind, "host_cache_emptied": emptied,
        "numa_why": None if ids else "no NUMA nodes in sysfs (a kernel without NUMA, or a container that hides them): buffers unbound"}}


def host_copy(gib: int = 2, copies: int = 3) -> dict:
    import torch
    src = torch.ones(gib * GIB, dtype=torch.uint8)
    dst = torch.empty_like(src)
    dst.copy_(src)
    t0 = time.perf_counter()
    for _ in range(copies):
        dst.copy_(src)
    return {"host_copy_gb_s": round(copies * gib * GIB / (time.perf_counter() - t0) / 1e9, 2), "host_threads": torch.get_num_threads()}


def disk(directory: Path, gib: int, require_direct: bool = False) -> dict:
    path = directory / ".q235-hostrate.bin"
    block = 16 << 20
    try:
        with open(path, "wb") as f:
            for _ in range(gib * GIB // block):
                f.write(os.urandom(block))
            f.flush()
            os.fsync(f.fileno())
        nbytes = path.stat().st_size
        r = subprocess.run(["dd", f"if={path}", "of=/dev/null", "bs=16M", "iflag=direct"], capture_output=True, text=True)
        if r.returncode == 0:
            m = re.search(r"copied, ([0-9.]+) s", r.stderr)
            if m:
                return {"disk_read_gb_s": round(nbytes / float(m.group(1)) / 1e9, 2), "disk_method": "O_DIRECT (dd iflag=direct)",
                        "disk_bytes": nbytes}
        if require_direct:
            return {"disk_read_gb_s": None, "disk_method": f"O_DIRECT unavailable here: {r.stderr.strip()[-200:]}", "disk_bytes": nbytes}
        fd = os.open(path, os.O_RDONLY)
        try:
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
            t0 = time.perf_counter()
            while os.read(fd, block):
                pass
            dt = time.perf_counter() - t0
        finally:
            os.close(fd)
        return {"disk_read_gb_s": round(nbytes / dt / 1e9, 2), "disk_method": "buffered after POSIX_FADV_DONTNEED (may be cache-served)",
                "disk_bytes": nbytes}
    finally:
        path.unlink(missing_ok=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--dir", type=Path, default=Path.cwd())
    ap.add_argument("--copies", type=int, default=10)
    ap.add_argument("--multi-buffers", type=int, default=8, help="Amendment 9: separately pinned 1 GiB buffers (0 = off)")
    ap.add_argument("--multi-copies", type=int, default=5)
    ap.add_argument("--disk-gib", type=int, default=4)
    ap.add_argument("--min-h2d-gbs", type=float, default=0.0)
    ap.add_argument("--min-disk-gbs", type=float, default=0.0)
    ap.add_argument("--min-host-copy-gbs", type=float, default=0.0)
    ap.add_argument("--require-direct", action="store_true", help="the box: an O_DIRECT read or a refusal, never a cached one")
    a = ap.parse_args()
    single = link(a.copies)                # FIRST: the floors' single-buffer read, before any by-buffer block exists to be reused
    try:                                   # Amendment 9 is reported only: its failure is recorded and never fails the phase
        by_buffer = link_by_buffer(a.multi_buffers, a.multi_copies)
    except Exception as e:                 # noqa: BLE001
        by_buffer = {"h2d_by_buffer": None, "h2d_by_buffer_why": f"error: {type(e).__name__}: {e}"[:300]}
    rec = {"schema": "q235-hostrate/1", **single, **by_buffer, **host_copy(), **disk(a.dir, a.disk_gib, a.require_direct),
           "floors": {"h2d_gb_s": a.min_h2d_gbs, "disk_read_gb_s": a.min_disk_gbs, "host_copy_gb_s": a.min_host_copy_gbs}}
    a.out.write_text(json.dumps(rec, indent=1) + "\n")
    print(json.dumps(rec))
    if a.min_h2d_gbs and (rec.get("h2d_gb_s") or 0) < a.min_h2d_gbs:
        print(f"refusing: pinned H2D {rec.get('h2d_gb_s')} GB/s < {a.min_h2d_gbs}", file=sys.stderr)
        sys.exit(18)
    if a.min_host_copy_gbs and rec["host_copy_gb_s"] < a.min_host_copy_gbs:
        print(f"refusing: host copy {rec['host_copy_gb_s']} GB/s < {a.min_host_copy_gbs}", file=sys.stderr)
        sys.exit(18)
    if a.require_direct and rec["disk_read_gb_s"] is None:
        print(f"refusing: no O_DIRECT read of the disk ({rec['disk_method']})", file=sys.stderr)
        sys.exit(13)
    if a.min_disk_gbs and (rec["disk_read_gb_s"] or 0) < a.min_disk_gbs:
        print(f"refusing: disk read {rec['disk_read_gb_s']} GB/s < {a.min_disk_gbs}", file=sys.stderr)
        sys.exit(13)


if __name__ == "__main__":
    main()
