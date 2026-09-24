"""Host-side cost of gptq_pack.HessianAccumulator.add at Mixtral's expert shapes (CPU only): the part of a Hessian
pass that a faster GPU does not shrink -- the K x K fp32 gram landing on the host (a copy) and the running-mean
scale + add. Emulates the box's hessian_device="cpu" path; the gram itself is computed on the GPU there."""
import json
import time

import torch

out = {"threads": torch.get_num_threads()}
for name, K in (("gate_up_in (hidden 4096)", 4096), ("down_in (inter 14336)", 14336)):
    H = torch.zeros(K, K)
    gram_dev = torch.randn(K, K)          # stands in for the GPU gram arriving over PCIe
    n = 1000
    ts_copy, ts_acc = [], []
    for i in range(4):
        t0 = time.perf_counter()
        g = gram_dev.clone()              # the landing copy (pinned-less D2H lands in fresh host memory)
        t1 = time.perf_counter()
        b = 512
        H *= n / (n + b)
        n += b
        H.add_(g, alpha=2.0 / n)
        t2 = time.perf_counter()
        ts_copy.append(t1 - t0)
        ts_acc.append(t2 - t1)
        del g
    out[name] = {"K": K, "bytes": K * K * 4, "copy_s_median": sorted(ts_copy)[1], "scale_add_s_median": sorted(ts_acc)[1]}
print(json.dumps(out, indent=1))
