"""Diagnostic: is the scatter-combine's forward and backward byte-identical when computed over row chunks? Shape: Qwen3-30B-A3B packed row,
4,096 tokens x top-8 (32,768 routed rows), hidden 2048. Compares the whole-tensor composite (engines/fast.py today) with row chunks."""
import json, sys, torch
dev = sys.argv[1] if len(sys.argv) > 1 else "cuda"
out_path = sys.argv[2] if len(sys.argv) > 2 else None
T, k, H = 4096, 8, 2048
R = T * k
g0 = torch.Generator(device="cpu").manual_seed(0)
down = (torch.randn(R, H, generator=g0) * 0.5).to(torch.bfloat16).to(dev)
w = torch.rand(R, generator=g0).to(dev)
order = torch.randperm(R, generator=g0).to(dev)
g = (torch.randn(T, H, generator=g0) * 0.1).to(torch.bfloat16).to(dev)

def fwd_whole():
    buf = torch.zeros(R, H, dtype=torch.float32, device=dev)
    buf[order] = down.to(torch.float32) * w[:, None]
    return buf.view(T, k, H).sum(1).to(torch.bfloat16)

def fwd_chunk(c):
    buf = torch.zeros(R, H, dtype=torch.float32, device=dev)
    for s in range(0, R, c):
        buf[order[s:s + c]] = down[s:s + c].to(torch.float32) * w[s:s + c, None]
    return buf.view(T, k, H).sum(1).to(torch.bfloat16)

def bwd_whole():
    gprod = g.to(torch.float32)[torch.div(order, k, rounding_mode="floor")]
    gw = (gprod * down.to(torch.float32)).sum(1, keepdim=True).squeeze(1)
    gdown = gprod.mul_(w[:, None]).to(down.dtype)
    return gw, gdown

def bwd_chunk(c):
    gw = torch.empty(R, dtype=torch.float32, device=dev)
    gdown = torch.empty(R, H, dtype=down.dtype, device=dev)
    g32 = g.to(torch.float32)
    for s in range(0, R, c):
        gp = g32[torch.div(order[s:s + c], k, rounding_mode="floor")]
        gw[s:s + c] = (gp * down[s:s + c].to(torch.float32)).sum(1, keepdim=True).squeeze(1)
        gdown[s:s + c] = gp.mul_(w[s:s + c, None]).to(down.dtype)
    return gw, gdown

def peak(fn):
    if dev != "cuda":
        return None, fn()
    torch.cuda.synchronize(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    b = torch.cuda.memory_allocated()
    r = fn()
    torch.cuda.synchronize()
    return round((torch.cuda.max_memory_allocated() - b) / 2**20, 1), r

res = {"device": dev, "torch": torch.__version__, "gpu": torch.cuda.get_device_name() if dev == "cuda" else None,
       "shape": {"tokens": T, "k": k, "hidden": H}, "rows": []}
pf, F0 = peak(fwd_whole)
pb, (GW0, GD0) = peak(bwd_whole)
res["whole"] = {"fwd_peak_mib": pf, "bwd_peak_mib": pb}
for c in (1024, 4096, 8192, 16384, 32768):
    qf, F1 = peak(lambda: fwd_chunk(c))
    qb, (GW1, GD1) = peak(lambda: bwd_chunk(c))
    row = {"chunk_rows": c, "fwd_equal": bool(torch.equal(F0, F1)), "gw_equal": bool(torch.equal(GW0, GW1)),
           "gdown_equal": bool(torch.equal(GD0, GD1)), "fwd_peak_mib": qf, "bwd_peak_mib": qb,
           "gw_max_abs_diff": float((GW0 - GW1).abs().max())}
    res["rows"].append(row)
    print(json.dumps(row), flush=True)
print(json.dumps(res["whole"]))
if out_path:
    json.dump(res, open(out_path, "w"), indent=1)
