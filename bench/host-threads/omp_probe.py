#!/usr/bin/env python3
"""Mechanism check (no number travels): does torch's intra-op pool sized to the VISIBLE CPUs slow the loader's host-side
expert assembly when a CFS quota allows fewer? Times experts4bit_qlora's own fuse_experts on one Qwen3-30B-A3B-shaped layer
(128 experts; gate/up [768, 2048], down [2048, 768]; bf16, about 1.2 GB) at each thread count given on the command line."""
import json
import os
import sys
import time

import torch  # noqa: E402

from experts4bit_qlora.arch.moe_conventions import fuse_experts  # noqa: E402


def quota():
    from experts4bit_qlora.util import container_cpu_quota
    return container_cpu_quota()[0]


E, I, H = 128, 768, 2048
g = torch.Generator().manual_seed(0)
gate = [torch.randn(I, H, generator=g).to(torch.bfloat16) for _ in range(E)]
up = [torch.randn(I, H, generator=g).to(torch.bfloat16) for _ in range(E)]
down = [torch.randn(H, I, generator=g).to(torch.bfloat16) for _ in range(E)]
out = {"visible": len(os.sched_getaffinity(0)), "quota": quota(), "runs": {}}
for t in [int(x) for x in sys.argv[1:]]:
    torch.set_num_threads(t)
    walls = []
    for _ in range(4):
        t0 = time.perf_counter()
        gu, dn = fuse_experts(gate, up, down)
        walls.append(time.perf_counter() - t0)
        del gu, dn
    out["runs"][t] = {"walls": [round(w, 3) for w in walls], "median": round(sorted(walls)[2], 3)}
print("OMP_PROBE " + json.dumps(out), flush=True)
