"""Kernel-route replay: grouped-nf4-gemm's fused NF4 grouped GEMM (forward and dgrad) against a whole-stack bitsandbytes
dequantize_4bit followed by torch._grouped_mm, on recorded real-router calls of e4b's training step (Qwen3-30B-A3B shapes).

Per unique call: device time of each route (CUDA events, median of --reps after warm-up) and the grouped_mm route's error against the
fused route. Weights are random NF4 stacks built by e4b's Experts4bit.from_float (the training layout); activations random bf16.
usage: route_bench.py routecalls-qwen3.json ROUTEBENCH.json [--reps 10]  (TC1c amendment 3's `routebench` token runs exactly this)
The calls file was recorded by route_record.py on an RTX A2000 (a 4-layer slice of the real checkpoint; the router decides the sizes).
"""
import argparse, json, statistics
import torch
ap = argparse.ArgumentParser(); ap.add_argument("calls"); ap.add_argument("out"); ap.add_argument("--reps", type=int, default=10)
a = ap.parse_args()
import nf4_grouped as ng
from experts4bit_qlora import Experts4bit
from experts4bit_qlora.engines.batched import _dequant_whole
dev = "cuda"
calls = json.load(open(a.calls))["calls"]
seen, uniq = set(), []
for c in calls:
    key = (c["op"], c["N"], c["K"], tuple(c["sizes"]), tuple(c["eids"]))
    if key not in seen:
        seen.add(key); uniq.append(c)
E, H, I = calls[0]["E"], 2048, 768
torch.manual_seed(0)
base = Experts4bit.from_float(torch.randn(E, 2 * I, H, dtype=torch.bfloat16, device=dev) * 0.02,
                              torch.randn(E, H, I, dtype=torch.bfloat16, device=dev) * 0.02, quant_type="nf4", compute_dtype=torch.bfloat16)
STACK = {(2 * I, H): (base.gate_up_proj.view(E, 2 * I, H // 2), base.gate_up_absmax.view(E, 2 * I, H // 64).float()),
         (H, I): (base.down_proj.view(E, H, I // 2), base.down_absmax.view(E, H, I // 64).float())}
def deq(N, K):
    B, am = STACK[(N, K)]
    return _dequant_whole(B, am, E, N, K, "nf4", 64, torch.bfloat16)
# the dequant matches grouped-nf4-gemm's oracle on one expert (layout check)
for (N, K), (B, am) in STACK.items():
    ref = ng.dequant_ref(B[3], am[3], N, K).to(torch.bfloat16)
    assert torch.equal(deq(N, K)[3], ref), ("layout", N, K, (deq(N, K)[3].float() - ref.float()).abs().max().item())
cap = torch.cuda.get_device_capability()
gm_ok, gm_why = True, ""
try:
    t = torch.randn(32, 64, dtype=torch.bfloat16, device=dev); w = torch.randn(2, 64, 48, dtype=torch.bfloat16, device=dev)
    torch._grouped_mm(t, w, offs=torch.tensor([16, 32], dtype=torch.int32, device=dev)); torch.cuda.synchronize()
except Exception as e:
    gm_ok, gm_why = False, f"{type(e).__name__}: {str(e)[:160]}"
def timed(fn):
    for _ in range(3): fn()
    torch.cuda.synchronize(); ts = []
    for _ in range(a.reps):
        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        e0.record(); fn(); e1.record(); torch.cuda.synchronize(); ts.append(e0.elapsed_time(e1))
    return statistics.median(ts)
rows = []; tot = {"fused": 0.0, "dequant": 0.0, "grouped_mm": 0.0}
for i, c in enumerate(uniq):
    N, K, sizes, eids = c["N"], c["K"], c["sizes"], c["eids"]
    T = sum(sizes); B, am = STACK[(N, K)]
    g = torch.Generator(device=dev).manual_seed(i)
    full = [0] * E
    for s, e in zip(sizes, eids): full[e] = s
    offs = torch.tensor(full, dtype=torch.int32, device=dev).cumsum(0, dtype=torch.int32)
    if c["op"] == "fwd":
        x = torch.randn(T, K, dtype=torch.bfloat16, device=dev, generator=g) * 0.5
        fused = lambda: ng.gemm_4bit_grouped(x, B, am, sizes, eids)
        gm = lambda W: torch._grouped_mm(x, W.transpose(1, 2), offs=offs)
    else:
        x = torch.randn(T, N, dtype=torch.bfloat16, device=dev, generator=g) * 0.5
        fused = lambda: ng.dgrad_4bit_grouped(x, B, am, sizes, eids)
        gm = lambda W: torch._grouped_mm(x, W, offs=offs)
    r = {"op": c["op"], "N": N, "K": K, "G": len(sizes), "T": T, "max": max(sizes), "fused_ms": timed(fused), "dequant_ms": timed(lambda: deq(N, K))}
    if not gm_ok and i < 6:                 # where grouped_mm is unavailable: check the orientation with a per-group loop on the same W
        W = deq(N, K); yf = fused().float(); parts, row = [], 0
        for s, e in zip(sizes, eids):
            xs = x[row:row + s]; parts.append(xs @ W[e].T if c["op"] == "fwd" else xs @ W[e]); row += s
        yl = torch.cat(parts).float()
        r["loop_fro_err_rel"] = ((yl - yf).norm() / yf.norm().clamp_min(1e-12)).item()
    if gm_ok:
        W = deq(N, K)
        r["grouped_mm_ms"] = timed(lambda: gm(W))
        yf, yg = fused().float(), gm(W).float()
        r["max_abs_err_rel"] = (yg - yf).abs().max().item() / max(yf.abs().max().item(), 1e-12)
        r["fro_err_rel"] = ((yg - yf).norm() / yf.norm().clamp_min(1e-12)).item()
        tot["grouped_mm"] += r["grouped_mm_ms"]
    tot["fused"] += r["fused_ms"]; tot["dequant"] += r["dequant_ms"]
    rows.append(r)
by = {}
for op in ("fwd", "dgrad"):
    rs = [r for r in rows if r["op"] == op]
    by[op] = {"calls": len(rs), "fused_ms": round(sum(r["fused_ms"] for r in rs), 3), "dequant_ms": round(sum(r["dequant_ms"] for r in rs), 3)}
    if gm_ok:
        by[op]["grouped_mm_ms"] = round(sum(r["grouped_mm_ms"] for r in rs), 3)
        by[op]["dequant_plus_grouped_mm_over_fused"] = round((by[op]["dequant_ms"] + by[op]["grouped_mm_ms"]) / by[op]["fused_ms"], 4)
        by[op]["grouped_mm_only_over_fused"] = round(by[op]["grouped_mm_ms"] / by[op]["fused_ms"], 4)
        by[op]["fro_err_rel_max"] = max(r["fro_err_rel"] for r in rs)
loop = [r["loop_fro_err_rel"] for r in rows if "loop_fro_err_rel" in r]
res = {"loop_check_fro_err_rel_max": max(loop) if loop else None, "gpu": torch.cuda.get_device_name(), "capability": list(cap), "torch": torch.__version__, "grouped_mm_supported": gm_ok, "grouped_mm_why": gm_why,
       "unique_calls": len(uniq), "by_op": by, "rows": rows}
json.dump(res, open(a.out, "w"), indent=1)
print(json.dumps({k: v for k, v in res.items() if k != "rows"}, indent=1))
