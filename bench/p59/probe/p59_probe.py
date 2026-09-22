"""P59 follow-up probe (A2000, free): is the K16 small-M GEMM bitwise invariant to fusing q/k/v along N?

Synthetic weights at Qwen3-30B-A3B's attention shapes (K=2048; q 4096, k 512, v 512), three Int4Linear vs
Int4Linear.fuse of them, rows 1..64, with the K16 route on and off. Output: probe.json (committed beside this file).
"""
import json

import torch
from experts4bit_qlora.engines.int4_attn import Int4Linear, resolve_smallm
import int4_smallm  # noqa: F401  (the K16 kernel must be installed, or smallm=True refuses)
torch.manual_seed(0)
dev = "cuda"
K = 2048
dims = {"q": 4096, "k": 512, "v": 512}
res = {"smallm_resolved_auto": resolve_smallm(None)}
for smallm in (True, False):
    parts = []
    for n, N in dims.items():
        lin = torch.nn.Linear(K, N, bias=False, device=dev, dtype=torch.bfloat16)
        torch.nn.init.normal_(lin.weight, std=K ** -0.5)
        parts.append(Int4Linear(lin, smallm=smallm))
    fused = Int4Linear.fuse(parts)
    plans = {n: (p._smallm_cfg if p._smallm is not None else None) for n, p in zip(dims, parts)}
    plans["fused"] = fused._smallm_cfg if fused._smallm is not None else None
    for R in (1, 2, 8, 16, 17, 64):
        x = (torch.randn(R, K, device=dev) / 4).to(torch.bfloat16)
        with torch.no_grad():
            sep = torch.cat([p(x) for p in parts], dim=-1)
            fus = fused(x)
        d = (sep.float() - fus.float()).abs()
        res[f"smallm={smallm} R={R}"] = {"bitwise_equal": bool(torch.equal(sep, fus)), "max_abs": float(d.max()), "n_diff": int((d > 0).sum())}
    res[f"smallm={smallm} plans (bn,kc,sk)"] = plans
print(json.dumps(res, indent=1))
