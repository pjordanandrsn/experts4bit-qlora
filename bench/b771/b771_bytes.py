"""Lane B771, stage 1 (bench/b771/PREREG-b771.md; e4b#771): the fused fp8 KV append against its reference, byte
for byte, at scale, with the hardware e4m3 cast (sm_89+).

gnf4's fp8_kv_append_bt1 -- the append the CUDA-graph bucket step uses -- must write exactly the bytes
quantize_kv_fp8 -- the eager step's quantize -- produces. gnf4's own gate compares a few thousand values; the
A2000 probe (fp32 intermediates, torch's cast as a stand-in) put the shipped kernel's byte mismatch rate at ~6e-8.
This counts it on real hardware over ~1.3e8 values per group layout, for whichever grouped-nf4-gemm is installed.

    python b771_bytes.py --out bytes_<tag>.json
"""
import argparse
import json
import time

import torch


def run(groups: int, mul: float, calls: int, S: int = 4096, H: int = 8, D: int = 128, seed: int = 0,
        dev: str = "cuda", kv=None):
    """``kv`` is the fp8_kv module (injectable, so CI can check this harness's layout math on CPU)."""
    if kv is None:
        import fp8_kv as kv
    fp8_kv_append_bt1, quantize_kv_fp8 = kv.fp8_kv_append_bt1, kv.quantize_kv_fp8
    torch.manual_seed(seed)
    bt, bps = 1, calls                       # one token per block: every append lands at fill 0 of its own row
    pay = bt * H * D
    srow = H * groups * 4
    row_bytes = pay + bt * srow
    pool = torch.zeros(S * bps * row_bytes, dtype=torch.uint8, device=dev)
    want = torch.zeros(S * bps, row_bytes, dtype=torch.uint8, device=dev)
    tbl = torch.arange(S * bps, dtype=torch.int32, device=dev).reshape(S, bps).contiguous()
    slot_idx = torch.randperm(S, device=dev).to(torch.int32)      # non-identity slot order, as in serving
    lens = torch.zeros(S, dtype=torch.int32, device=dev)
    ones = torch.ones(S, dtype=torch.int32, device=dev)
    for c in range(calls):
        x = (torch.randn(S, H, D, device=dev) * mul).to(torch.bfloat16)
        fp8_kv_append_bt1(x, pool, tbl, slot_idx, lens, row_bytes, pay, bt, groups)
        lens.index_add_(0, slot_idx.long(), ones)                 # the caller publishes, as Fp8PagedKV does
        q, s = quantize_kv_fp8(x, group=None if groups == 1 else D // groups)
        rows = slot_idx.long() * bps + c                         # row of (slot, block c), fill 0
        want[rows, :pay] = q.view(torch.uint8).reshape(S, H * D)
        want[rows, pay:] = s.float().reshape(S, -1).view(torch.uint8)
    got = pool.view(S * bps, row_bytes)
    pay_bad = int((got[:, :pay] != want[:, :pay]).sum())
    sc_bad = int((got[:, pay:].view(torch.int32) != want[:, pay:].view(torch.int32)).sum())
    return {"groups": groups, "mul": mul, "values": S * H * D * calls, "scales": S * H * groups * calls,
            "payload_bytes_differing": pay_bad, "scales_differing": sc_bad}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--calls", type=int, default=32)
    a = ap.parse_args()
    import importlib.metadata as md
    import fp8_kv
    t0 = time.time()
    rows = [run(g, m, a.calls, seed=i) for i, (g, m) in enumerate(((1, 1.0), (4, 1.0), (1, 30.0), (4, 30.0)))]
    rep = {"lane": "B771", "stage": "bytes", "gnf4": md.version("grouped-nf4-gemm"),
           "gnf4_has_e4m3_group": hasattr(fp8_kv, "_e4m3_group"),
           "device": torch.cuda.get_device_name(), "cc": list(torch.cuda.get_device_capability()),
           "rows": rows, "values": sum(r["values"] for r in rows),
           "payload_bytes_differing": sum(r["payload_bytes_differing"] for r in rows),
           "scales_differing": sum(r["scales_differing"] for r in rows), "seconds": time.time() - t0}
    json.dump(rep, open(a.out, "w"), indent=1)
    print(f"B771_BYTES gnf4={rep['gnf4']} e4m3_group={rep['gnf4_has_e4m3_group']} values={rep['values']} "
          f"payload_bytes_differing={rep['payload_bytes_differing']} scales_differing={rep['scales_differing']}",
          flush=True)


if __name__ == "__main__":
    main()
