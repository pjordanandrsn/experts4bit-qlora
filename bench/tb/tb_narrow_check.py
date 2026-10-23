#!/usr/bin/env python
"""T-B option C's narrowing premise, the box's first step before any timing (EXPLORE; Map 1, e4b#2043 section 5).

Both prototype kernels narrow fp32 to bf16: the fused pad's ``unpad_rows`` (grouped-nf4-gemm ``lora_pad``) and the lean
combine's down gradient (e4b ``combine_lean``). On CUDA each uses Triton's plain cast; under the Triton interpreter (whose
cast truncates) each uses round to nearest even in integer arithmetic. Their bitwise proof so far ran on CPU only. Here, on
the card, every fp32 high half (all 65,536, so every exponent, sign and NaN class) with the low halves at each rounding
edge, plus 4 M random patterns, goes through:

  in use      each kernel as the wrapper calls it on CUDA (plain cast)
  integer     each kernel with its integer narrowing forced on (reported beside: the interpreter's rule on this sm)

and is compared with the composite each kernel replaces, run by torch on the same card: ``x.index_select(..).to(bf16)``
for the pad and ``(fp32(g) * w).to(bf16)`` for the combine. torch's CPU bits are reported too (its NaN pattern depends
on the code path). The gate is the IN-USE paths only: every bit equal to torch on this card, NaNs included. Anything else
exits 31, and the box stops before timing: a finding, not a retry.
"""
from __future__ import annotations

import argparse
import json
import sys

LOW_EDGES = (0x0000, 0x0001, 0x7FFF, 0x8000, 0x8001, 0xFFFF)


def patterns(torch, n_random: int, seed: int = 0):
    hi = torch.arange(1 << 16, dtype=torch.int64) << 16
    lo = torch.tensor(LOW_EDGES, dtype=torch.int64)
    edge = (hi[:, None] | lo[None, :]).reshape(-1)
    g = torch.Generator().manual_seed(seed)
    rnd = torch.randint(0, 1 << 32, (n_random,), generator=g, dtype=torch.int64)
    bits = torch.cat([edge, rnd])
    bits = torch.where(bits >= 1 << 31, bits - (1 << 32), bits).to(torch.int32)
    return bits.view(torch.float32)


def compare(got, want, x):
    """Mismatch counts between two bf16 tensors' bits, all and NaN-input only, with up to 5 examples."""
    import torch
    gb, wb = got.view(torch.int16).cpu(), want.view(torch.int16).cpu()
    bad = gb != wb
    nan = torch.isnan(x.cpu())
    idx = bad.nonzero().flatten()[:5].tolist()
    xb = x.cpu().view(torch.int32)
    ex = [f"{int(xb[i]) & 0xFFFFFFFF:08x}: got {int(gb[i]) & 0xFFFF:04x} want {int(wb[i]) & 0xFFFF:04x}" for i in idx]
    return {"n": int(x.numel()), "mismatch": int(bad.sum()), "mismatch_nan_inputs": int((bad & nan).sum()),
            "mismatch_finite_or_inf": int((bad & ~nan).sum()), "examples": ex}


def nan_bits(t, x):
    import torch
    b = t.view(torch.int16).cpu()[torch.isnan(x.cpu())]
    return sorted({f"{int(v) & 0xFFFF:04x}" for v in b.unique().tolist()})


def run(dev: str, n_random: int):
    import torch
    import triton
    from experts4bit_qlora.engines import combine_lean as cl
    import lora_pad as lp

    x = patterns(torch, n_random).to(dev)
    N = x.numel()
    out = {"device": dev, "n_patterns": N}
    if dev == "cuda":
        out["gpu"] = torch.cuda.get_device_name()
        out["sm"] = "sm_%d%d" % torch.cuda.get_device_capability()
    out["torch"], out["triton"] = torch.__version__, triton.__version__

    # the pad's narrowing: rows of H values, unpad_rows over an identity row map
    H = 4096
    rows = (N + H - 1) // H
    src = torch.zeros(rows * H, dtype=torch.float32, device=dev)
    src[:N] = x
    src = src.view(rows, H)
    flat = torch.arange(rows, device=dev)
    want_pad = src.index_select(0, flat).to(torch.bfloat16)
    res = {}
    for name, forced in (("pad_in_use", None), ("pad_integer", True)):
        dst = torch.empty(rows, H, dtype=torch.bfloat16, device=dev)
        if forced is None:
            lp.unpad_rows(src, dst, flat)
        else:
            bn = lp._block(H)
            lp._unpad_rows_kernel[(rows, triton.cdiv(H, bn))](src, dst, flat, H, src.stride(0), dst.stride(0), BLOCK_N=bn,
                                                             F32_TO_BF16=True)
        res[name] = compare(dst.view(-1)[:N], want_pad.view(-1)[:N], x)
        res[name]["nan_bits"] = nan_bits(dst.view(-1)[:N], x)

    # the combine's down gradient: one row per H values, w = 1 (an exact multiply), order = identity, k = 1
    g = src
    down = torch.zeros(rows, H, dtype=torch.bfloat16, device=dev)
    w = torch.ones(rows, dtype=torch.float32, device=dev)
    order = torch.arange(rows, device=dev)
    want_cmb = (g.to(torch.float32)[order] * w[:, None]).to(torch.bfloat16)
    for name, forced in (("combine_in_use", None), ("combine_integer", True)):
        if forced is None:
            _, gd = cl.grad_products(g, down, w, order, 1, False, True)
        else:
            gd = torch.empty_like(down)
            bh = cl._block(H)
            cl._grad_products_kernel[(rows, triton.cdiv(H, bh))](
                g, down, w, order, down, gd, 1, H, g.stride(0), down.stride(0), down.stride(0), gd.stride(0),
                NEED_W=False, NEED_DOWN=True, INT_RNE=True, BLOCK_H=bh)
        res[name] = compare(gd.view(-1)[:N], want_cmb.view(-1)[:N], x)
        res[name]["nan_bits"] = nan_bits(gd.view(-1)[:N], x)

    out["results"] = res
    out["torch_device_nan_bits"] = {"pad": nan_bits(want_pad.view(-1)[:N], x), "combine": nan_bits(want_cmb.view(-1)[:N], x)}
    cpu = x.cpu().to(torch.bfloat16)
    out["torch_cpu_nan_bits"] = nan_bits(cpu, x)
    out["torch_cpu_vs_device_finite_mismatch"] = int(((cpu.view(torch.int16) != want_pad.view(-1)[:N].view(torch.int16).cpu())
                                                      & ~torch.isnan(x.cpu())).sum())
    in_use = [res["pad_in_use"]["mismatch"], res["combine_in_use"]["mismatch"]]
    out["gate"] = "PASS" if all(m == 0 for m in in_use) else "FAIL"
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--random", type=int, default=4_000_000)
    ap.add_argument("--out", default="tb_narrow_check.json")
    a = ap.parse_args()
    out = run(a.device, a.random)
    with open(a.out, "w") as f:
        json.dump(out, f, indent=1)
    for k, v in out["results"].items():
        print(f"[tb-narrow] {k}: {v['mismatch']}/{v['n']} mismatches ({v['mismatch_nan_inputs']} on NaN inputs); "
              f"NaN out {v['nan_bits']}", flush=True)
    print(f"[tb-narrow] torch NaN bits: device {out['torch_device_nan_bits']}, cpu {out['torch_cpu_nan_bits']}; "
          f"cpu vs device finite mismatches {out['torch_cpu_vs_device_finite_mismatch']}", flush=True)
    print(f"[tb-narrow] GATE {out['gate']} ({out.get('gpu')}, {out.get('sm')})", flush=True)
    return 0 if out["gate"] == "PASS" else 31


if __name__ == "__main__":
    sys.exit(main())
