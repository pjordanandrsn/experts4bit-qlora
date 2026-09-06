"""Training-side box-acceptance anchor.

Serving has one (kernel/decode_anchor.py, M2: 7.37 ms +/-4.2%). Training has
none, so a slow BOX is indistinguishable from a slow BUILD. Measured
2026-08-26/27: identical config, identical stack, 3.7 s/step on one RTX 5090
and 6.1 s/step on another -- 1.65x, against M2's 8.5% serving dispersion.

Both ran TRAIN_VRAM_FRAC=1.0 (all experts VRAM-resident, no PCIe expert
traffic), so transfer bandwidth does NOT explain it. TR1's census found the
training path launch-bound (2.92M launches/step at 8-11% GPU busy), which
makes host/driver launch throughput the prime suspect. Hence three probes,
not one: a box can be fast in FLOPs and slow in launches.

Emits JSON only. Classing is the caller's job (see train_anchor_gate.py).
"""
import json, os, statistics, sys, time
import torch

REPS = int(os.environ.get("ANCHOR_REPS", "5"))
OUT = os.environ.get("ANCHOR_OUT", "/root/anchor.json")


def _sync():
    torch.cuda.synchronize()


def probe_flops(n=4096, iters=50):
    """Dense bf16 GEMM: raw SM throughput. Insensitive to launch cost."""
    a = torch.randn(n, n, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(n, n, device="cuda", dtype=torch.bfloat16)
    for _ in range(5):
        a @ b
    _sync(); t = time.perf_counter()
    for _ in range(iters):
        a @ b
    _sync()
    dt = time.perf_counter() - t
    return {"tflops": round(2 * n**3 * iters / dt / 1e12, 2),
            "ms_per_gemm": round(dt / iters * 1e3, 4)}


def probe_launch(iters=20000):
    """Many trivial kernels: host+driver launch throughput.

    THE discriminator for this workload. Tensors are tiny so device time is
    negligible and the number is dominated by launch overhead -- which is
    what TR1's census says bounds the bnb training path.
    """
    x = torch.ones(1, device="cuda")
    for _ in range(1000):
        x.add_(0.0)
    _sync(); t = time.perf_counter()
    for _ in range(iters):
        x.add_(0.0)
    _sync()
    dt = time.perf_counter() - t
    return {"launches_per_s": round(iters / dt), "us_per_launch": round(dt / iters * 1e6, 3)}


def probe_h2d(mb=256, iters=20):
    """Pinned host->device bandwidth: the offload/DRAM-tier path."""
    n = mb * 1024 * 1024 // 4
    host = torch.empty(n, dtype=torch.float32, pin_memory=True)
    dev = torch.empty(n, dtype=torch.float32, device="cuda")
    for _ in range(3):
        dev.copy_(host, non_blocking=False)
    _sync(); t = time.perf_counter()
    for _ in range(iters):
        dev.copy_(host, non_blocking=False)
    _sync()
    dt = time.perf_counter() - t
    return {"h2d_gb_s": round(mb / 1024 * iters / dt, 2)}


def main():
    if not torch.cuda.is_available():
        json.dump({"status": "NO_CUDA"}, open(OUT, "w")); return
    r = {"status": "OK", "reps": REPS,
         "gpu": torch.cuda.get_device_name(0),
         "capability": list(torch.cuda.get_device_capability(0)),
         "torch": torch.__version__, "cuda": torch.version.cuda}
    try:
        import subprocess
        r["nvidia_smi"] = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version,power.limit,clocks.max.sm,memory.total",
             "--format=csv,noheader"], capture_output=True, text=True, timeout=30).stdout.strip()
    except Exception as e:
        r["nvidia_smi"] = "ERR " + repr(e)[:80]

    # A/A: each probe repeated; report median AND spread. A box whose own
    # repeats disagree is unusable regardless of how it compares to others.
    for name, fn in (("flops", probe_flops), ("launch", probe_launch), ("h2d", probe_h2d)):
        runs = [fn() for _ in range(REPS)]
        keys = runs[0].keys()
        agg = {}
        for k in keys:
            vals = [x[k] for x in runs]
            agg[k] = {"median": statistics.median(vals),
                      "min": min(vals), "max": max(vals),
                      "spread": round(max(vals) / min(vals), 4) if min(vals) else None}
        r[name] = agg
        r[name + "_runs"] = runs
    json.dump(r, open(OUT, "w"), indent=1)
    print(json.dumps({k: r[k] for k in ("gpu", "flops", "launch", "h2d")}, indent=1))


if __name__ == "__main__":
    main()
