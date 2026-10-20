#!/usr/bin/env python3
"""HBD's sibling probe (bench/hbd/PREREG-hbd.md): in a fresh process with a cell's own interpreter and environment, load
bitsandbytes the way the cell does, run one CUDA quantize, and record the libbitsandbytes files mapped into this process.
It cross-checks the cell's in-process read (tc1_arm --hbd-record); the reducer VOIDs a cell whose two reads disagree.

    python hbd_probe.py --framework e4b|unsloth --out probe.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os


def libraries():
    paths = []
    for line in open("/proc/self/maps"):
        parts = line.split()
        if len(parts) >= 6 and "libbitsandbytes" in os.path.basename(parts[-1]) and parts[-1] not in paths:
            paths.append(parts[-1])
    return [{"path": p, "sha256": hashlib.sha256(open(p, "rb").read()).hexdigest()} for p in paths]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--framework", choices=("e4b", "unsloth"), required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    if a.framework == "unsloth":
        import unsloth  # noqa: F401  (the cell imports unsloth before torch, so the probe does too)
    import torch
    import bitsandbytes
    import bitsandbytes.functional as F
    rec = {"framework": a.framework, "torch": torch.__version__, "torch_cuda": torch.version.cuda, "bitsandbytes": bitsandbytes.__version__,
           "BNB_CUDA_VERSION": os.environ.get("BNB_CUDA_VERSION"), "LD_LIBRARY_PATH": os.environ.get("LD_LIBRARY_PATH")}
    if torch.cuda.is_available():
        F.quantize_4bit(torch.randn(256, 128, dtype=torch.bfloat16, device="cuda"), blocksize=64, quant_type="nf4")
        torch.cuda.synchronize()
        rec.update(device=torch.cuda.get_device_name(0), capability=".".join(map(str, torch.cuda.get_device_capability(0))))
    else:
        rec["device"] = None
    rec["libraries"] = libraries()
    with open(a.out, "w") as f:
        json.dump(rec, f, indent=1, sort_keys=True)
    print("HBD PROBE " + json.dumps({"framework": a.framework, "libraries": [x["path"] for x in rec["libraries"]]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
