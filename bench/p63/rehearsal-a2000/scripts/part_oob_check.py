"""Does the int4 singleton branch's preallocated split-K buffer fit a T > 1 call?

enable_serve_experts_int4 sizes st["part"] as [sk(N, K, R=1) * top_k, N]. The singleton
branch of hot_residency._fused_over_stack passes it to gemv_int4_b32 at EVERY T, where
the call has R = T * top_k rows and plans sk from (N, K, R). The partials buffer is a view
into a larger buffer filled with a sentinel, so a write past the view lands in memory this
script owns and is counted, instead of corrupting a live tensor.
"""
import json

import torch
from int4_b32 import _plan, _sm_count, gemv_int4_b32, quant_x_rows
from int4_pack_ref import pack_int4_b32

torch.manual_seed(0)
dev = "cuda"
res = {"device": torch.cuda.get_device_name(0), "sm_count": torch.cuda.get_device_properties(0).multi_processor_count}
E, k = 8, 8
for (N, K) in ((2048, 2048), (1536, 2048), (2048, 768)):
    w = torch.randn(E, N, K) * 0.02
    pk, sc = zip(*(pack_int4_b32(w[e]) for e in range(E)))
    packed, scales = torch.stack(pk).to(dev), torch.stack(sc).to(dev)
    _, _, sk_store, _ = _plan(N, K)                      # enable_serve_experts_int4's sizing
    store_rows = sk_store * k
    for T in (1, 2, 17):
        R = T * k
        x = torch.randn(R, K, device=dev).to(torch.bfloat16)
        eids = torch.randint(0, E, (R,), device=dev, dtype=torch.int32)
        xq, xs = quant_x_rows(x)
        ref = gemv_int4_b32(xq, xs, packed, scales, eids, N, K)            # part=None: sized by the wrapper
        _, _, sk_call, _ = _plan(N, K, R, _sm_count(xq.device))
        need_rows = sk_call * R
        SENT = 12345.0
        buf = torch.full(((max(need_rows, store_rows) + 64) * N,), SENT, dtype=torch.float32, device=dev)
        part = buf[: store_rows * N].view(store_rows, N)
        out = gemv_int4_b32(xq, xs, packed, scales, eids, N, K, part=part)
        torch.cuda.synchronize()
        tail = buf[store_rows * N:]
        res[f"N={N} K={K} T={T}"] = {
            "rows_R": R, "sk_store": sk_store, "sk_call": sk_call,
            "part_rows_allocated": store_rows, "part_rows_the_call_indexes": need_rows,
            "sentinel_elements_overwritten_past_the_buffer": int((tail != SENT).sum()),
            "out_equals_part_None_out": bool(torch.equal(out, ref)),
        }
print(json.dumps(res, indent=1))
