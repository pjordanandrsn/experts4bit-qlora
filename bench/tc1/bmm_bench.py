"""LoRA-delta bmm host-cost replay (TC1-PREREG amendment 24): the padded LoRA delta's two batched products and their backward
(grouped-nf4-gemm kernel/nf4_qlora.py:_lora_delta_padded) at the shapes e4b's fused training step produces on recorded real-router
calls -- routecalls-qwen3.json's op == "fwd" calls (gate_up N 1536 K 2048, down N 2048 K 768), r 16. No model.

Per variant (fp32, bf16, fp32 under preferred_blas_library("cublaslt"), fp32 with TF32), each in a fresh process: the host time of every
forward bmm (perf_counter around the call, the queue drained first so a launch cannot block), the host time of the backward, and the
backward's device time (CUDA events), over five shape regimes in this order:
  recorded        the recorded shapes, first exposure (after a warm-up on one shape no call uses)
  recorded_again  the same shapes again
  cold            per call, the nearest widest >= the recorded one that this process has not used yet (--reps passes, pooled)
  fixed           every call at the median (groups, widest)
  bucket          the widest rounded up to a multiple of --bucket (groups as recorded); bucket_again: the same again
`fwd_host_us_median` pools both forward products of every call in the regime. The parent writes OUT with one row per variant.
usage: bmm_bench.py routecalls-qwen3.json OUT.json [--label t28] [--reps 3] [--bucket 32] [--r 16]
"""
import argparse, json, statistics, subprocess, sys, time
ap = argparse.ArgumentParser(); ap.add_argument("calls"); ap.add_argument("out"); ap.add_argument("--label", default="")
ap.add_argument("--reps", type=int, default=3); ap.add_argument("--bucket", type=int, default=32); ap.add_argument("--r", type=int, default=16)
ap.add_argument("--child", nargs=2)
a = ap.parse_args()
VARIANTS = [("fp32", "default"), ("bf16", "default"), ("fp32", "cublaslt"), ("fp32tf32", "default")]
REGIMES = ("recorded", "recorded_again", "cold", "fixed", "bucket", "bucket_again")


def child(dtype_name, blas):
    import torch
    if blas == "cublaslt":
        torch.backends.cuda.preferred_blas_library("cublaslt")
    torch.backends.cuda.matmul.allow_tf32 = dtype_name == "fp32tf32"
    dt = torch.bfloat16 if dtype_name == "bf16" else torch.float32
    dev = "cuda"
    calls = [c for c in json.load(open(a.calls))["calls"] if c["op"] == "fwd"]
    shapes = [(sum(1 for s in c["sizes"] if s), max(c["sizes"]), c["N"], c["K"]) for c in calls]
    Gs, Ws = sorted(s[0] for s in shapes), sorted(s[1] for s in shapes)
    med = (Gs[len(Gs) // 2], Ws[len(Ws) // 2])
    up = lambda w: -(-w // a.bucket) * a.bucket
    seen = set()
    torch.manual_seed(0)

    def delta(G, W, N, K):
        x = torch.randn(G, W, K, dtype=dt, device=dev, requires_grad=True)
        A = (torch.randn(G, a.r, K, dtype=dt, device=dev) * 0.02).requires_grad_()
        B = (torch.randn(G, N, a.r, dtype=dt, device=dev) * 0.02).requires_grad_()
        torch.cuda.synchronize()
        t0 = time.perf_counter(); h = torch.bmm(x, A.transpose(1, 2)); t1 = time.perf_counter()
        torch.cuda.synchronize()
        t2 = time.perf_counter(); d = torch.bmm(h, B.transpose(1, 2)); t3 = time.perf_counter()
        g = torch.ones_like(d)
        torch.cuda.synchronize()
        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        t4 = time.perf_counter(); e0.record(); d.backward(g); e1.record(); t5 = time.perf_counter()
        torch.cuda.synchronize()
        seen.add((G, W, N, K))
        return (t1 - t0) * 1e6, (t3 - t2) * 1e6, (t5 - t4) * 1e3, e0.elapsed_time(e1)

    def shape_of(regime, G, W, N, K):
        if regime == "fixed":
            return med[0], med[1], N, K
        if regime.startswith("bucket"):
            return G, up(W), N, K
        if regime == "cold":
            while (G, W, N, K) in seen:
                W += 1
        return G, W, N, K

    def run(regime, passes=1):
        fh, bw, dv, used = [], [], [], set()
        for _ in range(passes):
            for s in shapes:
                sh = shape_of(regime, *s); used.add(sh)
                r = delta(*sh); fh += [r[0], r[1]]; bw.append(r[2]); dv.append(r[3])
        m = statistics.median
        return {"fwd_host_us_median": round(m(fh), 1), "fwd_host_us_p90": round(sorted(fh)[int(0.9 * len(fh))], 1),
                "bwd_host_ms_median": round(m(bw), 3), "bwd_device_ms_median": round(m(dv), 3), "calls": len(bw), "distinct_shapes": len(used)}

    for N, K in {(s[2], s[3]) for s in shapes}:          # warm-up: cuBLAS init on a shape no regime uses
        for _ in range(3):
            delta(1, 1, N, K)
    out = {"label": a.label, "dtype": dtype_name, "blas": blas, "torch": torch.__version__, "cuda": torch.version.cuda,
           "gpu": torch.cuda.get_device_name(), "cap": list(torch.cuda.get_device_capability()), "n_calls": len(shapes),
           "median_shape": list(med), "bucket_multiple": a.bucket, "r": a.r}
    for reg in REGIMES:
        out[reg] = run(reg, passes=a.reps if reg == "cold" else 1)
    print(json.dumps(out))


if a.child:
    child(*a.child)
    sys.exit(0)
res = []
for dt, blas in VARIANTS:
    p = subprocess.run([sys.executable, "-u", __file__, a.calls, a.out, "--label", a.label, "--reps", str(a.reps), "--bucket", str(a.bucket),
                        "--r", str(a.r), "--child", dt, blas], capture_output=True, text=True)
    line = p.stdout.strip().splitlines()[-1] if p.stdout.strip() else ""
    try:
        r = json.loads(line)
    except Exception:
        r = {"label": a.label, "dtype": dt, "blas": blas, "error": ((p.stderr or "") + (p.stdout or ""))[-800:]}
    res.append(r)
    print(f"BMM {a.label} {dt} {blas} torch {r.get('torch')}: " + (" ".join(f"{k}={r[k]['fwd_host_us_median']}us" for k in REGIMES if k in r)
                                                                 or f"ERROR {r.get('error', '')[-200:]}"), flush=True)
json.dump({"label": a.label, "results": res}, open(a.out, "w"), indent=1)
