#!/usr/bin/env python3
"""sc1g_mxfp4_prefill_check.py -- lane SC1g (#846), A6's conv2 lead, CORRECTNESS ONLY (A2000-safe, no timings): is e4b's
KEEP_NF4=0 prompt path out of line with the other expert paths' kernel error at > 256 rows?

What e4b runs (engines/hot_residency.py): with the MXFP4 store and E4B_INT4_KEEP_NF4=0 the prompt's MoE rows go through
gnf4's `gemm_mxfp4_grouped(x, blocks, scales, sizes=[1]*M, eids)` -- every size is 1, so the launcher takes its
`max(sizes) == 1` branch: the per-row `_gemv_mxfp4_grouped` reduction (bf16 activations read as fp32, e2m1 x e8m0 decoded
exactly, fp32 accumulate, one bf16 rounding at the store). The route counter calls it `mxfp4_grouped_v1`.

Paths, each against ITS OWN exact reference (fp64 dequantise-then-matmul of the same weight bytes), on the same activations:
  grouped_v1   gemm_mxfp4_grouped(x, blocks, scales, [1]*M, eids)              e4b's KEEP_NF4=0 prompt route
  grouped_tile gemm_mxfp4_grouped(x_sorted, blocks, scales, sizes, eids_g)     the tiled kernel (context; e4b does not call it)
  mxfp4_gemv   gemv_mxfp4_b32(*quant_x_rows(x), blocks, scales, eids, N, K)    the decode route (int8 activations)
  nf4_mtile    gemm_4bit_grouped(x_sorted, nf4 B, absmax, sizes, eids_g)      the KEEP_NF4=1 prompt route, vs its NF4 reference
  bf16_mm      per expert x_bf16 @ dequant(W).bf16.T (cuBLAS)                   what box R's bf16 reference computes
plus the floor every bf16-output path shares: the exact reference rounded to bf16.

Weights: gpt-oss-20b's released MXFP4 expert stacks (gate_up and down) of a few layers, through e4b's `_mxfp4_store_layout`
transform (row de-interleave of gate/up, no re-quantisation). Activations: synthetic, heavy-tailed (Student-t df 4 plus a
few x20 outlier channels), bf16; routing: a Zipf-skewed histogram over the 32 experts. Rows M in {16, 256, 2048}.

Metrics per (layer, proj, M, path): max_rel = max|y - ref| / max|ref|, mean_rel = mean|y - ref| / mean|ref|; for the
MXFP4 kernel paths also eq_floor = the fraction of outputs bit-equal to the exact reference rounded to bf16.
The question (the maintainer's): is grouped_v1|gt256's error OUT OF LINE with the other paths' -- not whether it is non-zero.
`verdict` answers it (OUT_OF_LINE when grouped_v1|gt256's worst mean_rel > 2x the other paths' worst). `floor_verdict` is the
absolute read: grouped_v1|gt256's mean_rel over the bf16-output floor's, ABOVE_FLOOR when the worst ratio > 1.5.

  sc1g_mxfp4_prefill_check.py --model DIR --out OUT.json [--layers 0,11,23] [--mutate]

--mutate is the positive control: the reference decodes the HIGH nibble first, which must make grouped_v1 read wrong.
The relative `verdict` CANNOT see it -- every MXFP4 kernel path reads the bytes the same way, so all of them move against the
mutated reference together and stay in line with each other. The control is read by `floor_verdict`, which must flip to
ABOVE_FLOOR (bf16_mm and nf4_mtile build their references from the same decode, so they stay at the floor).
Exit 0 only on IN_LINE and AT_FLOOR, else 1 -- so the --mutate run must exit 1.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

import torch

H = 2880                 # gpt-oss-20b hidden; intermediate is also 2880 (gate_up N = 5760)
E = 32
ROWS = (16, 256, 2048)


def store_layout(gu_blocks, gu_scales, dn_blocks, dn_scales):
    """e4b's engines/int4_experts._mxfp4_store_layout, verbatim in effect: de-interleave gate/up rows, flatten groups."""
    gub = torch.cat([gu_blocks[:, 0::2], gu_blocks[:, 1::2]], dim=1)
    gus = torch.cat([gu_scales[:, 0::2], gu_scales[:, 1::2]], dim=1)
    e, two_i, g_h, _ = gu_blocks.shape
    e2, h, g_i, _ = dn_blocks.shape
    return (gub.reshape(e, two_i, g_h * 16).contiguous(), gus.contiguous(),
            dn_blocks.reshape(e2, h, g_i * 16).contiguous(), dn_scales.contiguous())


def dequant_mxfp4(blocks_e, scales_e, fp4_values, mutate=False):
    """One expert: blocks [N, K//2] uint8, scales [N, K//32] uint8 -> W [N, K] float64. Element 2j = LOW nibble."""
    lut = torch.tensor(fp4_values, dtype=torch.float64, device=blocks_e.device)
    b = blocks_e.to(torch.int64)
    lo, hi = b & 0xF, (b >> 4) & 0xF
    first, second = (hi, lo) if mutate else (lo, hi)
    nib = torch.stack([first, second], dim=-1).reshape(b.shape[0], -1)
    w = lut[nib]
    sc = torch.exp2(scales_e.to(torch.float64) - 127.0).repeat_interleave(32, dim=1)
    return w * sc


def acts(m, k, g, dev):
    x = torch.distributions.StudentT(4.0).sample((m, k)).to(torch.float32)
    ch = torch.randperm(k, generator=g)[:8]
    x[:, ch] *= 20.0
    return x.to(dev, torch.bfloat16)


def zipf_ids(m, g):
    p = 1.0 / torch.arange(1, E + 1, dtype=torch.float64) ** 1.1
    perm = torch.randperm(E, generator=g)
    return perm[torch.multinomial(p, m, replacement=True, generator=g)].to(torch.int32)


def err(y, ref, fl=None):
    d = (y.double() - ref).abs()
    r = {"max_rel": float(d.max() / ref.abs().max().clamp_min(1e-30)),
         "mean_rel": float(d.mean() / ref.abs().mean().clamp_min(1e-30))}
    if fl is not None:                                   # bit-equal to the exact result rounded to bf16
        r["eq_floor"] = float((y.to(torch.bfloat16) == fl).double().mean())
    return r


def check_proj(blocks, scales, nf4_b, nf4_a, x, eids, mods, mutate):
    """One projection, one row count: every path against its exact reference."""
    mx, nf, i4, dref, fp4 = mods
    dev = x.device
    n = blocks.shape[1]
    k = x.shape[1]
    order = torch.argsort(eids, stable=True)
    xs, es = x[order], eids[order]
    uniq, counts = torch.unique_consecutive(es, return_counts=True)
    sizes, eids_g = [int(c) for c in counts], [int(u) for u in uniq]
    ref = torch.empty(x.shape[0], n, dtype=torch.float64, device=dev)          # MXFP4 exact, in x's row order
    ref_nf4 = torch.empty(x.shape[0], n, dtype=torch.float64, device=dev)      # NF4 exact, in sorted order
    bf16 = torch.empty(x.shape[0], n, dtype=torch.float64, device=dev)         # cuBLAS bf16, in x's row order
    r0 = 0
    for e, c in zip(eids_g, sizes):
        rows = order[r0:r0 + c]
        w = dequant_mxfp4(blocks[e], scales[e], fp4, mutate)
        ref[rows] = x[rows].double() @ w.t()
        bf16[rows] = (x[rows] @ w.to(torch.bfloat16).t()).double()
        wn = dref(nf4_b[e], nf4_a[e], n, k).to(torch.float64)
        ref_nf4[r0:r0 + c] = xs[r0:r0 + c].double() @ wn.t()
        r0 += c
    fl = ref.to(torch.bfloat16)
    out = {"floor_bf16_output": err(fl, ref)}
    out["grouped_v1"] = err(mx.gemm_mxfp4_grouped(x, blocks, scales, [1] * x.shape[0], eids), ref, fl)
    yt = mx.gemm_mxfp4_grouped(xs, blocks, scales, sizes, torch.tensor(eids_g, dtype=torch.int32, device=dev))
    out["grouped_tile"] = err(yt, ref[order], fl[order])
    xq, xsc = i4.quant_x_rows(x)
    out["mxfp4_gemv"] = err(mx.gemv_mxfp4_b32(xq, xsc, blocks, scales, eids, n, k), ref, fl)
    out["nf4_mtile"] = err(nf.gemm_4bit_grouped(xs, nf4_b, nf4_a, sizes, eids_g), ref_nf4)
    out["bf16_mm"] = err(bf16, ref)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--layers", default="0,11,23")
    ap.add_argument("--mutate", action="store_true")
    a = ap.parse_args(argv)
    import mxfp4_grouped as mx
    import nf4_grouped as nf
    import int4_b32 as i4
    from nf4_pack_ref import quantize_pack_nf4
    from safetensors import safe_open
    dev = torch.device("cuda")
    mods = (mx, nf, i4, nf.dequant_ref, mx.FP4_VALUES)
    wmap = json.load(open(os.path.join(a.model, "model.safetensors.index.json")))["weight_map"]
    g = torch.Generator().manual_seed(0)
    torch.manual_seed(0)
    rec = {"device": {"name": torch.cuda.get_device_name(0), "cc": list(torch.cuda.get_device_capability(0))},
           "torch": torch.__version__, "mutate": a.mutate, "rows": list(ROWS), "layers": {}, "shards": {}}
    for layer in [int(v) for v in a.layers.split(",")]:
        pre = f"model.layers.{layer}.mlp.experts."
        t = {}
        for name in ("gate_up_proj_blocks", "gate_up_proj_scales", "down_proj_blocks", "down_proj_scales"):
            shard = wmap[pre + name]
            with safe_open(os.path.join(a.model, shard), framework="pt") as f:
                t[name] = f.get_tensor(pre + name)
            if shard not in rec["shards"]:
                h = hashlib.sha256()
                with open(os.path.join(a.model, shard), "rb") as fh:
                    for chunk in iter(lambda: fh.read(1 << 24), b""):
                        h.update(chunk)
                rec["shards"][shard] = h.hexdigest()
        u8 = {k: (v if v.dtype == torch.uint8 else v.view(torch.uint8)) for k, v in t.items()}
        gub, gus, dnb, dns = (s.to(dev) for s in store_layout(u8["gate_up_proj_blocks"], u8["gate_up_proj_scales"],
                                                              u8["down_proj_blocks"], u8["down_proj_scales"]))
        rec["layers"][layer] = {}
        for proj, (blocks, scales) in (("gate_up", (gub, gus)), ("down", (dnb, dns))):
            n, k = blocks.shape[1], blocks.shape[2] * 2
            pk, am = [], []
            for e in range(E):                                   # NF4 of the same weights (box R's fake-quant recipe)
                p_, a_ = quantize_pack_nf4(dequant_mxfp4(blocks[e], scales[e], mx.FP4_VALUES).float())
                pk.append(p_.to(dev))
                am.append(a_.to(dev))
            nf4_b, nf4_a = torch.stack(pk), torch.stack(am)
            rec["layers"][layer][proj] = {"N": n, "K": k}
            for m in ROWS:
                x = acts(m, k, g, dev)
                eids = zipf_ids(m, g).to(dev)
                r = check_proj(blocks, scales, nf4_b, nf4_a, x, eids, mods, a.mutate)
                rec["layers"][layer][proj][f"M{m}"] = r
                print(f"SC1G_MXFP4_PREFILL layer={layer} {proj} M={m} " + " ".join(
                    f"{p}={v['max_rel']:.2e}/{v['mean_rel']:.2e}" for p, v in r.items()), flush=True)
            del nf4_b, nf4_a, pk, am
            torch.cuda.empty_cache()

    # the verdict on the maintainer's question: grouped_v1 at > 256 rows against the worst of the other kernel paths
    gt, others, ratio, eq = [], [], [], []
    for lr in rec["layers"].values():
        for pr in lr.values():
            for mk, r in pr.items():
                if not mk.startswith("M"):
                    continue
                if int(mk[1:]) > 256:
                    gt.append(r["grouped_v1"]["mean_rel"])
                    ratio.append(r["grouped_v1"]["mean_rel"] / r["floor_bf16_output"]["mean_rel"])
                    eq.append(r["grouped_v1"]["eq_floor"])
                others.append(max(r[p]["mean_rel"] for p in ("grouped_tile", "mxfp4_gemv", "nf4_mtile", "bf16_mm")))
    worst_gt, worst_other, worst_ratio = max(gt), max(others), max(ratio)
    verdict = ("OUT_OF_LINE" if worst_gt > 2.0 * worst_other else "IN_LINE")
    floor_verdict = ("ABOVE_FLOOR" if worst_ratio > 1.5 else "AT_FLOOR")
    rec.update(verdict=verdict, grouped_v1_gt256_mean_rel_max=worst_gt, other_paths_mean_rel_max=worst_other,
               floor_verdict=floor_verdict, grouped_v1_gt256_floor_ratio_max=worst_ratio,
               grouped_v1_gt256_eq_floor_min=min(eq))
    json.dump(rec, open(a.out, "w"), indent=1)
    print(f"SC1G_MXFP4_PREFILL_CHECK {verdict} floor={floor_verdict} mutate={a.mutate} grouped_v1|gt256 mean_rel max "
          f"{worst_gt:.3e} vs other paths max {worst_other:.3e}; over its floor max x{worst_ratio:.3f}; bit-equal to the "
          f"floor min {min(eq):.6f}", flush=True)
    return 0 if verdict == "IN_LINE" and floor_verdict == "AT_FLOOR" else 1


if __name__ == "__main__":
    sys.exit(main())
