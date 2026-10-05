#!/usr/bin/env python3
"""sc1g_mtile_check.py -- lane SC1g amendment A3 (#846), a $0 side check: does grouped-nf4-gemm's NF4 M-tile path
(gemm_4bit_grouped, e4b's `nf4_mtile_host` route) lose accuracy when ONE call carries a whole window's rows?

Box J read e4b's chunk-free full forward (`--ppl-oracle full`: 2561 tokens x top-4 = 10244 expert rows per layer in one
call) at 0.811 nats on conv1 against 0.731 for the same weights in 128-token chunks (512 rows per call). This reads one
layer's experts both ways against an exact reference:

  one       gemm_4bit_grouped on all the window's rows in ONE call
  chunked   the same rows in 128-token chunks (<= 512 rows per call), as the chunked prefill runs
  ref       per group, x_g (fp32) @ dequant_ref(B[e], absmax[e]).T in fp32, rounded to bf16 (the kernel returns bf16)

Weights are gpt-oss-20b's real experts (MXFP4 dequantized, then NF4-packed by gnf4's own quantize_pack_nf4, the layout
the M-tile reads); activations are synthetic (normal, and normal with 4 channels x 100); routing is synthetic top-4 of 32
(uniform, and skewed so one expert takes most tokens -- the large-group case). Per (layer, which, routing, kind): the
worst and mean per-row ||a - ref|| / ||ref|| of `one` and `chunked`, and one vs chunked. --mutate is the positive control
(the reference reads the next expert) and must read MTILE_DISAGREES. Correctness only: no timing is read.

  sc1g_mtile_check.py --model-dir SNAPSHOT --out OUT.json [--layers 0,12] [--tokens 2561] [--chunk 128] [--tol 5e-3]
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
TOPK = 4


def nf4_stack(blocks, scales, dev):
    """The MXFP4 store -> dequantized fp32 per expert -> gnf4's NF4 pack: (B [E, N, K/2] u8, absmax [E, N, K/64] f32)."""
    from mxfp4_pack_ref import dequant_mxfp4
    from nf4_pack_ref import quantize_pack_nf4
    E, N, K2 = blocks.shape
    K = K2 * 2
    Bs, As = [], []
    for e in range(E):
        w = dequant_mxfp4(blocks[e].to(dev).view(N, K // 32, 16), scales[e].to(dev)).float()
        # row slices: the pack's argmin materializes [rows, K, 16] fp32, ~1 GB for a whole gate_up expert; absmax is
        # per (row, 64-block), so slicing rows is exact
        parts = [quantize_pack_nf4(w[r:r + 512]) for r in range(0, N, 512)]
        Bs.append(torch.cat([q for q, _ in parts]))
        As.append(torch.cat([s for _, s in parts]))
    return torch.stack(Bs), torch.stack(As), N, K


def route(T, E, kind, g):
    """[T, TOPK] distinct expert ids per token: uniform, or skewed (expert e drawn with weight 1 / (e + 1)^2)."""
    w = torch.ones(E) if kind == "uniform" else 1.0 / torch.arange(1, E + 1, dtype=torch.float32) ** 2
    return torch.stack([torch.multinomial(w, TOPK, replacement=False, generator=g) for _ in range(T)])


def grouped(x, ids):
    """Rows (token, slot) sorted by expert, as e4b's dispatch sorts them: (x_sorted, sizes, eids, order)."""
    flat = ids.reshape(-1)
    order = torch.argsort(flat, stable=True)
    eids, sizes = torch.unique_consecutive(flat[order], return_counts=True)
    return x.repeat_interleave(TOPK, dim=0)[order], [int(s) for s in sizes], [int(e) for e in eids], order


def run(x, ids, B, A, N, K, dev, chunk, mutate):
    from nf4_grouped import dequant_ref, gemm_4bit_grouped
    T = x.shape[0]
    xs, sizes, eids, order = grouped(x, ids)
    xs = xs.to(dev, torch.bfloat16).contiguous()
    one = torch.empty(T * TOPK, N, dtype=torch.float32, device=dev)
    one[order.to(dev)] = gemm_4bit_grouped(xs, B, A, sizes, eids).float()
    ref = torch.empty_like(one)
    r0, refs = 0, torch.empty(T * TOPK, N, dtype=torch.float32, device=dev)
    E = B.shape[0]
    for m, e in zip(sizes, eids):
        ew = (e + 1) % E if mutate else e
        w = dequant_ref(B[ew], A[ew], N, K).float()
        refs[r0:r0 + m] = xs[r0:r0 + m].float() @ w.t()
        r0 += m
    ref[order.to(dev)] = refs.to(torch.bfloat16).float()
    chk = torch.empty_like(one)
    for t0 in range(0, T, chunk):
        cx, cs, ce, co = grouped(x[t0:t0 + chunk], ids[t0:t0 + chunk])
        y = gemm_4bit_grouped(cx.to(dev, torch.bfloat16).contiguous(), B, A, cs, ce).float()
        base = t0 * TOPK
        chk[base + co.to(dev)] = y
    def rows(a, b):
        r = (a - b).norm(dim=1) / b.norm(dim=1).clamp_min(1e-30)
        return {"max": float(r.max()), "mean": float(r.mean())}
    return {"rows_one_call": T * TOPK, "max_group": max(sizes), "groups": len(sizes),
            "one_vs_ref": rows(one, ref), "chunked_vs_ref": rows(chk, ref), "one_vs_chunked": rows(one, chk)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--layers", default="0,12")
    ap.add_argument("--tokens", type=int, default=2561)
    ap.add_argument("--chunk", type=int, default=128)
    ap.add_argument("--tol", type=float, default=5e-3, help="worst per-row one-call error above this = MTILE_DISAGREES")
    ap.add_argument("--mutate", action="store_true", help="positive control: the reference reads the next expert")
    a = ap.parse_args(argv)
    from sc1g_gemv_check import load_store
    dev = "cuda"
    cap = {"name": torch.cuda.get_device_name(0), "cc": list(torch.cuda.get_device_capability(0))}
    g = torch.Generator().manual_seed(0)
    out = []
    for layer in (int(v) for v in a.layers.split(",")):
        st = load_store(a.model_dir, layer)
        for which in ("gu", "dn"):
            B, A, N, K = nf4_stack(*st[which], dev)
            for rk in ("uniform", "skewed"):
                ids = route(a.tokens, B.shape[0], rk, g)
                for kind in ("normal", "outlier"):
                    x = torch.randn(a.tokens, K, generator=g)
                    if kind == "outlier":
                        x[:, torch.randperm(K, generator=g)[:4]] *= 100.0
                    r = run(x, ids, B, A, N, K, dev, a.chunk, a.mutate)
                    out.append({"layer": layer, "which": which, "routing": rk, "kind": kind, **r})
                    print(f"L{layer} {which} {rk} {kind}: rows={r['rows_one_call']} max_group={r['max_group']} "
                          f"one={r['one_vs_ref']['max']:.3e} chunked={r['chunked_vs_ref']['max']:.3e} "
                          f"one_vs_chunked={r['one_vs_chunked']['max']:.3e}", flush=True)
            del B, A
            torch.cuda.empty_cache()
    worst_one = max(r["one_vs_ref"]["max"] for r in out)
    worst_chk = max(r["chunked_vs_ref"]["max"] for r in out)
    verdict = "MTILE_AGREES" if worst_one <= a.tol and worst_chk <= a.tol else "MTILE_DISAGREES"
    rec = {"device": cap, "verdict": verdict, "worst_one_call": worst_one, "worst_chunked": worst_chk, "tol": a.tol,
           "tokens": a.tokens, "chunk": a.chunk, "mutate": a.mutate, "rows": out}
    json.dump(rec, open(a.out, "w"), indent=1)
    print(f"SC1G_MTILE_CHECK {verdict} device={cap['name']} cc={cap['cc']} worst_one_call={worst_one:.3e} "
          f"worst_chunked={worst_chk:.3e} mutate={a.mutate}", flush=True)
    return 0 if verdict == "MTILE_AGREES" else 1


if __name__ == "__main__":
    sys.exit(main())
