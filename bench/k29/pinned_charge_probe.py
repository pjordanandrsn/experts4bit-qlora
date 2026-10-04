#!/usr/bin/env python3
"""What does a page-locked host byte cost the container's memory cgroup? (grouped-nf4-gemm#71)

#71 measured cgroup v1 with a cap ladder (smallest surviving `--memory` cap): cap = 1.004 * alloc + 325 MB. A cap
ladder needs a cgroup the process can create and limit, which rented containers never grant. This instrument needs only
the container's OWN cgroup charge, readable from inside: v2 `memory.current`, v1 `memory.usage_in_bytes`. Each
(mode, size) runs in a fresh process: build the CUDA context, read the charge, allocate and touch N MB (pinned:
`torch.empty(N, dtype=uint8, pin_memory=True)`; pageable: the same without pinning), read the charge again. The
per-byte cost is the slope of the charge delta against N, fitted per mode.

    python3 k29_pinned_charge.py [--sizes 0,512,1024,2048,4096] [--reps 2] [--out k29.json]
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

MB = 1 << 20


def cgroup_charge_file() -> tuple[str, str]:
    """(version, path) of the charge counter for THIS process's own cgroup, as mounted in the container."""
    mounts = open("/proc/mounts").read()
    if os.path.exists("/sys/fs/cgroup/memory.current") and " cgroup2 " in mounts:
        return "v2", "/sys/fs/cgroup/memory.current"
    if os.path.exists("/sys/fs/cgroup/memory/memory.usage_in_bytes"):
        return "v1", "/sys/fs/cgroup/memory/memory.usage_in_bytes"
    raise SystemExit("no readable cgroup memory charge (neither v2 memory.current nor v1 memory.usage_in_bytes)")


def read_int(path: str) -> int:
    with open(path) as fh:
        return int(fh.read().strip())


def child(mode: str, size_mb: int, charge_path: str) -> dict:
    import torch
    torch.empty(1, device="cuda")                    # the CUDA context, which the intercept carries
    torch.cuda.synchronize()
    time.sleep(0.5)
    c0 = read_int(charge_path)
    t = None
    if size_mb:
        n = size_mb * MB
        t = torch.empty(n, dtype=torch.uint8, pin_memory=(mode == "pinned"))
        t.fill_(1)                                   # touch every page: pageable pages are charged on first touch
    time.sleep(0.5)
    c1 = read_int(charge_path)
    return {"mode": mode, "size_mb": size_mb, "charge_before": c0, "charge_after": c1,
            "delta_mb": (c1 - c0) / MB, "is_pinned": bool(t.is_pinned()) if t is not None else None}


def fit(xs: list[float], ys: list[float]) -> tuple[float, float, float]:
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    a = sxy / sxx
    b = my - a * mx
    ss_res = sum((y - (a * x + b)) ** 2 for x, y in zip(xs, ys))
    ss_tot = sum((y - my) ** 2 for y in ys)
    return a, b, (1 - ss_res / ss_tot) if ss_tot else 1.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", default="0,512,1024,2048,4096")
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--out", default="k29.json")
    ap.add_argument("--child", nargs=3, metavar=("MODE", "SIZE_MB", "CHARGE_PATH"))
    a = ap.parse_args()
    if a.child:
        print(json.dumps(child(a.child[0], int(a.child[1]), a.child[2])))
        return 0
    version, path = cgroup_charge_file()
    import torch
    env = {"cgroup": version, "charge_file": path, "proc_self_cgroup": open("/proc/self/cgroup").read().strip(),
           "torch": torch.__version__, "cuda": torch.version.cuda,
           "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}
    try:
        env["driver"] = subprocess.run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                                       capture_output=True, text=True, timeout=30).stdout.strip()
    except Exception as e:  # noqa: BLE001
        env["driver"] = f"unreadable: {e!r}"
    for f in ("memory.max", "memory/memory.limit_in_bytes"):
        p = os.path.join("/sys/fs/cgroup", f)
        if os.path.exists(p):
            env["limit_" + f.replace("/", "_")] = open(p).read().strip()
    rows = []
    sizes = [int(s) for s in a.sizes.split(",")]
    for rep in range(a.reps):
        for mode in ("pinned", "pageable"):
            for size in sizes:
                r = subprocess.run([sys.executable, __file__, "--child", mode, str(size), path],
                                   capture_output=True, text=True, timeout=600)
                if r.returncode != 0:
                    rows.append({"mode": mode, "size_mb": size, "rep": rep, "error": r.stderr.strip()[-400:]})
                    continue
                row = json.loads(r.stdout.strip().splitlines()[-1])
                row["rep"] = rep
                rows.append(row)
                print(json.dumps(row), flush=True)
    fits = {}
    for mode in ("pinned", "pageable"):
        pts = [(r["size_mb"], r["delta_mb"]) for r in rows if r.get("mode") == mode and "delta_mb" in r]
        if len({x for x, _ in pts}) >= 2:
            s, i, r2 = fit([x for x, _ in pts], [y for _, y in pts])
            fits[mode] = {"slope": round(s, 4), "intercept_mb": round(i, 1), "r2": round(r2, 5), "points": len(pts)}
    out = {"env": env, "rows": rows, "fits": fits}
    json.dump(out, open(a.out, "w"), indent=1)
    print("FITS", json.dumps(fits))
    print("ENV", json.dumps({k: env[k] for k in ("cgroup", "torch", "driver", "gpu")}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
