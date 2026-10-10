# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""The host rates Q235's time-left fit rests on (gate 2): the step is link-bound and the load reads the checkpoint from disk, and
both belong to the host, not the card. Measured the same way on the A2000 rehearsal and on the box, after install and before the
checkpoint fetch, so the box can refuse a host slower than the fit assumed.

- **link**: pinned host-to-device and device-to-host GB/s, timed with CUDA events over ``--copies`` copies of a 1 GiB pinned
  buffer after a warm-up, and the PCIe generation and width the card negotiated (``nvidia-smi``, recorded beside the copy, which
  is the arbiter);
- **disk**: the read rate of a freshly written file of ``--disk-gib`` random bytes in the working directory, with ``O_DIRECT``
  (``dd iflag=direct``) where the filesystem supports it, else after evicting its pages (``posix_fadvise DONTNEED``). The method
  is recorded; a cache-served read overstates the disk, so ``--require-direct`` (the box) refuses rather than fall back.

    python q235_hostrate.py --out hostrate.json [--min-h2d-gbs X --min-disk-gbs Y]   (exit 18 below the link floor, 13 below the disk's)
"""
from __future__ import annotations

import argparse
import json
import os
import re
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
    q = "pcie.link.gen.current,pcie.link.gen.max,pcie.link.width.current,pcie.link.width.max"
    try:
        gen, gen_max, width, width_max = [v.strip() for v in subprocess.check_output(
            ["nvidia-smi", f"--query-gpu={q}", "--format=csv,noheader"], text=True).splitlines()[0].split(",")]
        out.update(pcie_gen=gen, pcie_gen_max=gen_max, pcie_width=width, pcie_width_max=width_max)
    except (OSError, subprocess.CalledProcessError, ValueError) as e:
        out["pcie_why"] = f"{type(e).__name__}: {e}"
    return out


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
    ap.add_argument("--disk-gib", type=int, default=4)
    ap.add_argument("--min-h2d-gbs", type=float, default=0.0)
    ap.add_argument("--min-disk-gbs", type=float, default=0.0)
    ap.add_argument("--require-direct", action="store_true", help="the box: an O_DIRECT read or a refusal, never a cached one")
    a = ap.parse_args()
    rec = {"schema": "q235-hostrate/1", **link(a.copies), **disk(a.dir, a.disk_gib, a.require_direct),
           "floors": {"h2d_gb_s": a.min_h2d_gbs, "disk_read_gb_s": a.min_disk_gbs}}
    a.out.write_text(json.dumps(rec, indent=1) + "\n")
    print(json.dumps(rec))
    if a.min_h2d_gbs and (rec.get("h2d_gb_s") or 0) < a.min_h2d_gbs:
        print(f"refusing: pinned H2D {rec.get('h2d_gb_s')} GB/s < {a.min_h2d_gbs}", file=sys.stderr)
        sys.exit(18)
    if a.require_direct and rec["disk_read_gb_s"] is None:
        print(f"refusing: no O_DIRECT read of the disk ({rec['disk_method']})", file=sys.stderr)
        sys.exit(13)
    if a.min_disk_gbs and (rec["disk_read_gb_s"] or 0) < a.min_disk_gbs:
        print(f"refusing: disk read {rec['disk_read_gb_s']} GB/s < {a.min_disk_gbs}", file=sys.stderr)
        sys.exit(13)


if __name__ == "__main__":
    main()
