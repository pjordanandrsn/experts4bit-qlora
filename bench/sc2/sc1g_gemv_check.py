#!/usr/bin/env python3
"""sc1g_gemv_check.py -- lane SC1g amendment A1 (#846): a direct correctness read of grouped-nf4-gemm's gemv_mxfp4_b32 on
gpt-oss-20b's own MXFP4 experts, separating the KERNEL from the int8 activation SCHEME.

For each call (activations x [R, K] bf16, expert ids e [R], one layer's gate_up or down store):
  kernel     gemv_mxfp4_b32(quant_x_rows(x))                          -- what e4b serves at T == 1
  exact      dequant(quant_x_rows(x)) @ dequant_mxfp4(W_e)^T, fp32     -- the same quantised x, through an exact path
  raw        x @ dequant_mxfp4(W_e)^T, fp32                            -- no activation quantisation
The kernel returns bf16, so the verdict compares it with the exact reference ROUNDED TO bf16 (kernel_vs_exact_bf16): a
correct kernel differs from it only where fp32 summation order flips a rounding. kernel_vs_exact (unrounded) is reported and
sits at the bf16 rounding floor (~1e-3). exact vs raw is the int8 per-32 scheme's error on THESE activations. All as
||a - b|| / ||b|| per call, aggregated per (layer, which).

The store is rebuilt from the checkpoint with e4b's own _mxfp4_store_layout (the path the serve lane takes), so the 5090
box and the A2000 run the same code on the same bytes; only the activations travel. Activations come from
sc1g_k8.py's capture (--acts), or are synthetic (--synthetic: N(0, 1) rows, and the same rows with 4 channels per row
scaled 100x, as an outlier probe).

  sc1g_gemv_check.py --model-dir SNAPSHOT --out OUT.json (--acts CAPTURE.pt | --synthetic --layers 0,12)
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import torch


def load_store(model_dir: str, layer: int):
    from safetensors import safe_open
    from experts4bit_qlora.engines.int4_experts import _mxfp4_store_layout
    pre = f"model.layers.{layer}.mlp.experts."
    want = {k: None for k in ("gate_up_proj_blocks", "gate_up_proj_scales", "down_proj_blocks", "down_proj_scales")}
    idx = json.load(open(os.path.join(model_dir, "model.safetensors.index.json")))["weight_map"]
    for k in want:
        with safe_open(os.path.join(model_dir, idx[pre + k]), "pt") as f:
            want[k] = f.get_tensor(pre + k)
    gub, gus, dnb, dns = _mxfp4_store_layout(want["gate_up_proj_blocks"], want["gate_up_proj_scales"],
                                             want["down_proj_blocks"], want["down_proj_scales"])
    return {"gu": (gub, gus), "dn": (dnb, dns)}


def rel(a, b):
    return float((a - b).norm() / b.norm().clamp_min(1e-30))


def check_call(x, eids, blocks, scales, dev, mutate=False):
    from int4_b32 import quant_x_rows
    from mxfp4_grouped import gemv_mxfp4_b32
    from mxfp4_pack_ref import dequant_mxfp4
    E, N, K2 = blocks.shape
    K = K2 * 2
    x = x.to(dev, torch.bfloat16).contiguous()
    eids = eids.to(dev, torch.int32).contiguous()
    b, s = blocks.to(dev), scales.to(dev)
    xq, xs = quant_x_rows(x)
    kern = gemv_mxfp4_b32(xq, xs, b, s, eids, N, K).float()
    xd = (xq.float().view(x.shape[0], K // 32, 32) * xs[..., None]).view(x.shape[0], K)
    exact, raw = torch.empty_like(kern), torch.empty_like(kern)
    for r, e in enumerate(eids.tolist()):
        if mutate:                    # the detector's positive control: the references read the NEXT expert
            e = (e + 1) % E
        w = dequant_mxfp4(b[e].view(N, K // 32, 16), s[e])          # [N, K] fp32
        exact[r] = xd[r] @ w.t()
        raw[r] = x[r].float() @ w.t()
    xf = x.float().view(x.shape[0], K // 32, 32).abs()
    crest = (xf.amax(-1) / xf.mean(-1).clamp_min(1e-30)).amax().item()     # worst per-32 block max/mean
    return {"kernel_vs_exact_bf16": rel(kern, exact.to(torch.bfloat16).float()), "kernel_vs_exact": rel(kern, exact),
            "scheme_exact_vs_raw": rel(exact, raw), "kernel_vs_raw": rel(kern, raw),
            "x_absmax": float(x.float().abs().max()), "x_rms": float(x.float().pow(2).mean().sqrt()), "block_crest_max": crest}


def summarize(rows):
    out = {}
    for r in rows:
        k = f"L{r['layer']}_{r['which']}"
        o = out.setdefault(k, {"n": 0, "kernel_vs_exact_bf16_max": 0.0, "kernel_vs_exact_max": 0.0, "scheme_mean": 0.0,
                               "scheme_max": 0.0, "crest_max": 0.0})
        o["n"] += 1
        o["kernel_vs_exact_bf16_max"] = max(o["kernel_vs_exact_bf16_max"], r["kernel_vs_exact_bf16"])
        o["kernel_vs_exact_max"] = max(o["kernel_vs_exact_max"], r["kernel_vs_exact"])
        o["scheme_mean"] += (r["scheme_exact_vs_raw"] - o["scheme_mean"]) / o["n"]
        o["scheme_max"] = max(o["scheme_max"], r["scheme_exact_vs_raw"])
        o["crest_max"] = max(o["crest_max"], r["block_crest_max"])
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--acts", default="")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--layers", default="0,12")
    ap.add_argument("--mutate", action="store_true", help="positive control: the references use the wrong expert; "
                    "the verdict must be KERNEL_DISAGREES")
    ap.add_argument("--kernel-tol", type=float, default=1e-3,
                    help="kernel vs the bf16-rounded exact reference above this on any call is KERNEL_DISAGREES")
    a = ap.parse_args(argv)
    dev = "cuda"
    cap = {"name": torch.cuda.get_device_name(0), "cc": list(torch.cuda.get_device_capability(0))}
    rows = []
    if a.acts:
        acts = torch.load(a.acts, weights_only=False)["rows"]
        stores = {}
        for c in acts:
            if c["layer"] not in stores:
                stores[c["layer"]] = load_store(a.model_dir, c["layer"])
            b, s = stores[c["layer"]][c["which"]]
            rows.append({"layer": c["layer"], "which": c["which"], "step": c["step"],
                         **check_call(c["x"], c["eids"], b, s, dev, a.mutate)})
    if a.synthetic:
        g = torch.Generator().manual_seed(0)
        for layer in (int(v) for v in a.layers.split(",")):
            st = load_store(a.model_dir, layer)
            for which in ("gu", "dn"):
                b, s = st[which]
                K = b.shape[2] * 2
                for kind in ("normal", "outlier"):
                    x = torch.randn(4, K, generator=g)
                    if kind == "outlier":
                        x[:, torch.randperm(K, generator=g)[:4]] *= 100.0
                    eids = torch.randint(0, b.shape[0], (4,), generator=g)
                    rows.append({"layer": layer, "which": which, "step": kind, **check_call(x, eids, b, s, dev, a.mutate)})
    worst = max((r["kernel_vs_exact_bf16"] for r in rows), default=None)
    verdict = "NO_ROWS" if worst is None else ("KERNEL_AGREES" if worst <= a.kernel_tol else "KERNEL_DISAGREES")
    rec = {"device": cap, "verdict": verdict, "kernel_vs_exact_worst": worst, "metric": "kernel_vs_exact_bf16", "kernel_tol": a.kernel_tol,
           "summary": summarize(rows), "rows": rows, "acts": a.acts or None, "synthetic": a.synthetic, "mutate": a.mutate}
    json.dump(rec, open(a.out, "w"), indent=1)
    print(f"SC1G_GEMV_CHECK {verdict} device={cap['name']} cc={cap['cc']} calls={len(rows)} kernel_vs_exact_worst={worst} "
          f"scheme_mean={sum(r['scheme_exact_vs_raw'] for r in rows) / max(len(rows), 1):.4g}", flush=True)
    return 0 if verdict == "KERNEL_AGREES" else 1


if __name__ == "__main__":
    sys.exit(main())
