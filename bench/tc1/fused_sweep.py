"""Config replay of grouped-nf4-gemm's fused training kernels on recorded real-router calls (the routecalls file route_bench.py reads):
the forward M-tile kernel over (prefill_variant, prefill_groups, BLOCK_N, num_warps, num_stages) at the cost-rule M-tile, and the dgrad
kernel over (BLOCK_M, BLOCK_N, BLOCK_K, num_warps). Per config: summed device time over the unique calls (CUDA events, median of --reps)
against the default, and whether its outputs are bit-identical to the default's (else the max relative Frobenius difference). A config
that fails to compile or launch on the card is recorded with its error, not hidden.
usage: fused_sweep.py routecalls-qwen3.json FUSEDSWEEP.json [--reps 5]  (TC1c amendment 5's `fusedsweep` token runs exactly this)
"""
import argparse, json, statistics
import torch
ap = argparse.ArgumentParser(); ap.add_argument("calls"); ap.add_argument("out"); ap.add_argument("--reps", type=int, default=5)
a = ap.parse_args()
import nf4_grouped as ng
from experts4bit_qlora import Experts4bit
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
FWD = [None]                                                    # None = the shipped default (variant 1, groups 1, 128/w4/s3)
for var in (1, 3):
    for grp in (1, 2):
        for bn, w, st in ((128, 4, 3), (128, 8, 3), (128, 8, 4), (256, 8, 3), (64, 4, 4), (128, 4, 4)):
            FWD.append((var, grp, bn, w, st))
DG = [None] + [(32, 64, 64, 2), (64, 64, 64, 4), (64, 128, 64, 4), (128, 128, 64, 8), (64, 64, 64, 8), (128, 64, 64, 4), (32, 128, 64, 4), (64, 64, 32, 4)]
def timed(fn):
    for _ in range(2): fn()
    torch.cuda.synchronize(); ts = []
    for _ in range(a.reps):
        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        e0.record(); fn(); e1.record(); torch.cuda.synchronize(); ts.append(e0.elapsed_time(e1))
    return statistics.median(ts)
inputs = []
for i, c in enumerate(uniq):
    g = torch.Generator(device=dev).manual_seed(i)
    x = torch.randn(sum(c["sizes"]), c["K"] if c["op"] == "fwd" else c["N"], dtype=torch.bfloat16, device=dev, generator=g) * 0.5
    inputs.append(x)
res = {"gpu": torch.cuda.get_device_name(), "capability": list(torch.cuda.get_device_capability()), "torch": torch.__version__,
       "unique_calls": len(uniq), "fwd": [], "dgrad": []}
for op, configs in (("fwd", FWD), ("dgrad", DG)):
    ref = {}
    for cfg in configs:
        tot, ok, err, worst, ident = 0.0, True, "", 0.0, True
        for i, c in enumerate(uniq):
            if c["op"] != op: continue
            B, am = STACK[(c["N"], c["K"])]; x = inputs[i]
            if op == "fwd":
                kw = {} if cfg is None else dict(prefill_variant=cfg[0], prefill_groups=cfg[1], prefill_config=cfg[2:])
                fn = lambda: ng.gemm_4bit_grouped(x, B, am, c["sizes"], c["eids"], **kw)
            else:
                fn = lambda: ng.dgrad_4bit_grouped(x, B, am, c["sizes"], c["eids"], config=cfg)
            try:
                y = fn()
                tot += timed(fn)
            except Exception as e:
                ok, err = False, f"{type(e).__name__}: {str(e)[:200]}"; break
            if cfg is None:
                ref[i] = y
            elif not torch.equal(y, ref[i]):
                ident = False
                worst = max(worst, ((y.float() - ref[i].float()).norm() / ref[i].float().norm().clamp_min(1e-12)).item())
        row = {"config": cfg, "ok": ok}
        if ok:
            row.update(total_ms=round(tot, 3), bit_identical_to_default=ident, max_fro_rel_vs_default=worst)
        else:
            row["error"] = err
        res[op].append(row)
        print(op, cfg, row.get("total_ms"), row.get("bit_identical_to_default"), round(row.get("max_fro_rel_vs_default", 0), 5), row.get("error", ""), flush=True)
for op in ("fwd", "dgrad"):
    d = res[op][0]["total_ms"]
    for r in res[op]:
        if r["ok"]: r["over_default"] = round(r["total_ms"] / d, 4)
    best = min((r for r in res[op] if r["ok"]), key=lambda r: r["total_ms"])
    best_ident = min((r for r in res[op] if r["ok"] and r["bit_identical_to_default"]), key=lambda r: r["total_ms"])
    res[op + "_best"] = best; res[op + "_best_bit_identical"] = best_ident
    print(op, "default", d, "| best", best["config"], best["over_default"], "| best bit-identical", best_ident["config"], best_ident["over_default"])
json.dump(res, open(a.out, "w"), indent=1)
