"""Diagnostic count, not a registered measurement (TC1 amendment 53's Why): kernels, launch calls, sync-like calls and
cudaMalloc/cudaFree per forward + backward of grouped-nf4-gemm's lora_delta_grouped, the single padded block
(NF4_QLORA_PAD_BUCKETS=0) against buckets (=1), with fp32 and bf16 adapters, at Qwen3-30B-A3B's packed-row shape
(4,096 tokens x top-8 of 128 experts, Zipf(0.6) routing, r 16; gate_up 2048 -> 1536, down 768 -> 2048).

NF4_QLORA_LORA_PATH=padded is forced: under `auto` the 2 GiB pad-byte rule sends calls this skewed to the per-expert loop.
Written for a correctness-only testbed (an RTX A2000): it also records host and GPU milliseconds, which are NOT readings
and are never quoted -- only the counts are read.

usage: python bucket_count.py <grouped-nf4-gemm kernel/ dir> <out.json>"""
import os, sys, json, random, time, statistics as S
KDIR = sys.argv[1]; out = sys.argv[2]
sys.path.insert(0, KDIR)
os.environ["NF4_QLORA_LORA_PATH"] = "padded"   # the byte rule would send the single block to the loop
import torch
import nf4_qlora as Q, nf4_grouped as NG  # noqa: F401 -- imported as the run imported it

def zipf_sizes(tokens=4096, experts=128, k=8, s=0.6, seed=0):
    rng = random.Random(seed)
    w = [1.0 / (r + 1) ** s for r in range(experts)]
    rng.shuffle(w)
    sizes = [0] * experts
    for _ in range(tokens):
        keys = sorted(range(experts), key=lambda e: -(rng.random() ** (1.0 / w[e])))[:k]
        for e in keys:
            sizes[e] += 1
    return sizes

dev = "cuda"
E, R = 128, 16
SHAPES = {"gate_up": (2048, 1536), "down": (768, 2048)}
routings = [zipf_sizes(seed=s) for s in range(6)]
res = {"torch": torch.__version__, "triton": __import__("triton").__version__,
       "gpu": torch.cuda.get_device_name(), "allow_tf32": torch.backends.cuda.matmul.allow_tf32, "zipf_s": 0.6, "lora_path": "padded", "rows": []}

def one_iter(a_cats, As, Bs, sizes, eids):
    for name in ("gate_up", "down"):      # one projection's block alive at a time (the A2000 has 12 GB)
        d = Q.lora_delta_grouped(a_cats[name], As[name], Bs[name], sizes, eids, scaling=1.0)
        d.float().sum().backward()

def run_row(ad_dtype, mode, As, Bs, acts, eids, stats0):

        # warm-up over every routing (cuBLAS heuristics, triton, allocator)
        for it in range(12):
            one_iter(acts, As, Bs, routings[it % len(routings)], eids)
        torch.cuda.synchronize()
        host, gpu = [], []
        for it in range(24):
            sizes = routings[it % len(routings)]
            # host enqueue time with the GPU kept busy, so the queue never blocks the host
            torch.cuda._sleep(int(2e9))
            t0 = time.perf_counter(); one_iter(acts, As, Bs, sizes, eids); host.append(time.perf_counter() - t0)
            torch.cuda.synchronize()
            e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            e0.record(); one_iter(acts, As, Bs, sizes, eids); e1.record(); torch.cuda.synchronize()
            gpu.append(e0.elapsed_time(e1) / 1e3)
        from torch.profiler import profile, ProfilerActivity
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            for it in range(6):
                one_iter(acts, As, Bs, routings[it], eids)
            torch.cuda.synchronize()
        ev = prof.key_averages()
        kern = sum(e.count for e in ev if getattr(e, "device_type", None) is not None and str(e.device_type).endswith("CUDA"))
        launches = sum(e.count for e in ev if e.key in ("cudaLaunchKernel", "cuLaunchKernel", "cudaLaunchKernelExC", "cuLaunchKernelEx"))
        syncs = sum(e.count for e in ev if "Synchronize" in e.key or e.key in ("cudaMemcpy", "cudaStreamWaitEvent"))
        mallocs = sum(e.count for e in ev if e.key in ("cudaMalloc", "cudaFree"))
        top_cpu = sorted(((e.key, round(e.self_cpu_time_total / 6 / 1e3, 3), e.count // 6) for e in ev), key=lambda x: -x[1])[:12]
        paths = {k: Q.LORA_PATH_STATS[k] - stats0.get(k, 0) for k in Q.LORA_PATH_STATS}
        row = {"adapters": str(ad_dtype).split(".")[-1], "buckets": mode, "host_ms_med": round(S.median(host) * 1e3, 3),
               "gpu_ms_med": round(S.median(gpu) * 1e3, 3), "kernels_per_iter": kern / 6, "launch_calls_per_iter": launches / 6,
               "sync_like_per_iter": syncs / 6, "malloc_free_per_iter": mallocs / 6, "paths": paths,
               "top_self_cpu_ms_per_iter": top_cpu}
        res["rows"].append(row)
        print(json.dumps({k: v for k, v in row.items() if k != "top_self_cpu_ms_per_iter"}), flush=True)

for ad_dtype in (torch.float32, torch.bfloat16):
    for mode in ("0", "1"):
        os.environ["NF4_QLORA_PAD_BUCKETS"] = mode
        torch.manual_seed(0)
        As = {n: (torch.randn(E, R, K, device=dev, dtype=ad_dtype) * 0.02).requires_grad_() for n, (K, N) in SHAPES.items()}
        Bs = {n: (torch.randn(E, N, R, device=dev, dtype=ad_dtype) * 0.02).requires_grad_() for n, (K, N) in SHAPES.items()}
        acts = {n: torch.randn(4096 * 8, K, device=dev, dtype=torch.bfloat16, requires_grad=True) for n, (K, N) in SHAPES.items()}
        eids = list(range(E))
        stats0 = dict(Q.LORA_PATH_STATS)
        try:
            run_row(ad_dtype, mode, As, Bs, acts, eids, stats0)
        except torch.cuda.OutOfMemoryError as ex:
            row = {"adapters": str(ad_dtype).split(".")[-1], "buckets": mode, "oom": str(ex)[:160]}
            res["rows"].append(row); print(json.dumps(row), flush=True)
        del As, Bs, acts; torch.cuda.empty_cache()
json.dump(res, open(out, "w"), indent=1)

