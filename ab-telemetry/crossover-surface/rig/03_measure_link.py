#!/usr/bin/env python3
"""03_measure_link.py — measure L: pinned host->device bandwidth (the GEMM-feed lane the knee
formula divides by). Pinned memcpy loop, event-bracketed, best of N. Records receipts/link.json.
This is G3's denominator: the RAID-0 stripe S must reach >= this L or the f-axis measures the
wrong bottleneck."""
import json, os, sys, torch
assert torch.cuda.is_available(), "need CUDA"
dev = torch.device("cuda")
N_MB = int(os.environ.get("LINK_MB", "512"))
n = N_MB * 1024 * 1024
host = torch.empty(n, dtype=torch.uint8, pin_memory=True)
d = torch.empty(n, dtype=torch.uint8, device=dev)
for _ in range(3):  # warm
    d.copy_(host, non_blocking=True); torch.cuda.synchronize()
best = 0.0
for _ in range(20):
    s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
    s.record(); d.copy_(host, non_blocking=True); e.record(); torch.cuda.synchronize()
    gbps = n / (s.elapsed_time(e) / 1e3) / 1e9
    best = max(best, gbps)
rec = {"pinned_h2d_GBps": round(best, 2), "bytes_MB": N_MB,
       "gpu": torch.cuda.get_device_name(0),
       "pcie_gen": torch.cuda.get_device_properties(0).__dict__.get("pci_bus_id", "?")}
out = sys.argv[1] if len(sys.argv) > 1 else "receipts"
os.makedirs(out, exist_ok=True)
json.dump(rec, open(os.path.join(out, "link.json"), "w"), indent=2)
print(f"pinned H2D L = {best:.2f} GB/s"); print("wrote", os.path.join(out, "link.json"))
