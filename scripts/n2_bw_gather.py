"""N2 Phase-0.b — BW_gather probe (docs/N2_PHASE01_RECONSTRUCTION.md).

Pinned H2D bandwidth at expert-gather granularity: for each precision's true per-expert slab
size (OLMoE: gate_up+down packed + absmax), time a "gather" of k=8 slab-sized pinned->device
copies, 100 reps, median GB/s. Plus the 256 MB single-block ceiling for reference. This is the
denominator of t_fetch(p) = 16 layers x 8 experts x slab(p) / BW_gather(p).

Usage:
    python scripts/n2_bw_gather.py --job-dir runs/jobs/n2_phase0_bw
"""

import argparse
import json
import os
import statistics
import sys
from datetime import datetime, timezone

# OLMoE-1B-7B per-expert bytes (gate_up 2048x2048 + down 2048x1024 = 6,291,456 params).
SLAB_BYTES = {
    "nf4": 6_291_456 // 2 + (6_291_456 // 64) * 4,   # packed uint8 (2/byte) + fp32 absmax
    "int8": 6_291_456 + (6_291_456 // 64) * 4,
    "bf16": 6_291_456 * 2,
}
K_EXPERTS = 8
N_LAYERS = 16


def time_gather(nbytes, k, reps):
    import torch

    srcs = [torch.empty(nbytes, dtype=torch.uint8).pin_memory() for _ in range(k)]
    dsts = [torch.empty(nbytes, dtype=torch.uint8, device="cuda") for _ in range(k)]
    for d, s in zip(dsts, srcs):  # warm
        d.copy_(s, non_blocking=True)
    torch.cuda.synchronize()
    times = []
    for _ in range(reps):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for d, s in zip(dsts, srcs):
            d.copy_(s, non_blocking=True)
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))  # ms for k copies
    med_ms = statistics.median(times)
    gb = k * nbytes / 1e9
    return gb / (med_ms / 1e3), med_ms


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--job-dir", required=True)
    ap.add_argument("--reps", type=int, default=100)
    args = ap.parse_args()
    os.makedirs(args.job_dir, exist_ok=True)

    import torch

    out = {"job_type": "n2_bw_gather", "status": "pass", "k_experts": K_EXPERTS,
           "n_layers": N_LAYERS, "reps": args.reps, "bw": {}, "t_fetch_ms_per_token": {}}
    for p, nbytes in SLAB_BYTES.items():
        gbps, med_ms = time_gather(nbytes, K_EXPERTS, args.reps)
        out["bw"][p] = {"slab_bytes": nbytes, "gather_gbps": round(gbps, 2),
                        "gather_ms_per_layer": round(med_ms, 4)}
        t_fetch_ms = N_LAYERS * med_ms
        out["t_fetch_ms_per_token"][p] = round(t_fetch_ms, 2)
        print(f"{p:>5}: slab {nbytes/1e6:.2f} MB | gather(k=8) {gbps:.2f} GB/s | "
              f"t_fetch {t_fetch_ms:.1f} ms/token")
    gbps_ceiling, _ = time_gather(256 * 1024 * 1024, 1, 20)
    out["ceiling_256mb_gbps"] = round(gbps_ceiling, 2)
    print(f"256MB-block ceiling: {gbps_ceiling:.2f} GB/s")
    out.update(torch_version=torch.__version__, gpu_name=torch.cuda.get_device_name(0),
               timestamp=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"))
    with open(os.path.join(args.job_dir, "result.json"), "w") as f:
        json.dump(out, f, indent=2, sort_keys=True)
        f.write("\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
