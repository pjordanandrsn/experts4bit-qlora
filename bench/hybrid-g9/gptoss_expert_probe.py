# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""gpt-oss expert path probe: where does e4b's gpt-oss forward diverge
from the checkpoint's own math?

Stages, each printed as one ``PROBE`` line:

* ``dequant``   e4b's :func:`dequantize_mxfp4` against an independent
                MXFP4 decode of the same bytes (e2m1 nibbles, e8m0 scales).
* ``expert``    e4b's :class:`GptOssExperts4bit` (NF4) forward for a few
                experts against the reference forward (bf16 dequantised
                weights, HF's interleave / clamp / ``(up + 1)`` / biases)
                on random inputs -- cosine and relative error.
* ``layer``     with ``--capture`` (a ``{x, y}`` file recorded from layer
                0's MLP inside the served model): the reference MoE output
                for ``x`` -- HF's router (top-k then softmax) over all
                experts of layer 0 -- against the served ``y``.

An NF4 expert path sits at cosine > 0.99 against bf16; a wrong
interleave, orientation or epilogue sits far below it."""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import torch

FP4_LUT = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
                        -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0])


def independent_mxfp4_decode(blocks: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    """``blocks`` uint8 [..., rows, G, 16] (two e2m1 nibbles per byte, low
    first), ``scales`` uint8 [..., rows, G] (e8m0, bias 127) -> float32
    [..., rows, G * 32]."""
    lo = (blocks & 0x0F).long()
    hi = (blocks >> 4).long()
    vals = torch.stack([FP4_LUT[lo], FP4_LUT[hi]], dim=-1).reshape(*blocks.shape[:-1], 32)
    sc = torch.ldexp(torch.ones_like(scales, dtype=torch.float32),
                     scales.to(torch.int32) - 127)
    out = vals * sc[..., None]
    return out.reshape(*blocks.shape[:-2], -1)


def _snapshot(model: str) -> str:
    from huggingface_hub import snapshot_download
    return snapshot_download(model, allow_patterns=["*.json", "*.safetensors"])


def _tensors(snap: str, names):
    from safetensors import safe_open
    idx = json.load(open(os.path.join(snap, "model.safetensors.index.json")))["weight_map"]
    out = {}
    files = {}
    for n in names:
        fn = os.path.join(snap, idx[n])
        if fn not in files:
            files[fn] = safe_open(fn, "pt", device="cpu")
        out[n] = files[fn].get_tensor(n)
    return out


def reference_expert(x, W_gu, b_gu, W_dn, b_dn, alpha=1.702, limit=7.0):
    """HF's GptOssExperts math for ONE expert. ``W_gu`` [hidden, 2I]
    input-major interleaved, ``W_dn`` [I, hidden] input-major."""
    gate_up = x @ W_gu + b_gu
    gate, up = gate_up[..., ::2], gate_up[..., 1::2]
    gate = gate.clamp(max=limit)
    up = up.clamp(min=-limit, max=limit)
    glu = gate * torch.sigmoid(gate * alpha)
    return ((up + 1) * glu) @ W_dn + b_dn


def _stats(got, ref):
    g, r = got.float().reshape(-1), ref.float().reshape(-1)
    cos = torch.nn.functional.cosine_similarity(g, r, dim=0).item()
    rel = ((g - r).norm() / r.norm().clamp_min(1e-12)).item()
    return cos, rel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="openai/gpt-oss-20b")
    ap.add_argument("--layer", type=int, default=0)
    ap.add_argument("--experts", type=int, default=4)
    ap.add_argument("--capture", default=None)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    dev = torch.device(a.device)
    snap = _snapshot(a.model)
    cfg = json.load(open(os.path.join(snap, "config.json")))
    alpha = float(cfg.get("swiglu_alpha", 1.702))
    limit = float(cfg.get("swiglu_limit", 7.0))
    top_k = int(cfg.get("num_experts_per_tok", 4))
    pfx = f"model.layers.{a.layer}.mlp.experts."
    names = [pfx + n for n in ("gate_up_proj_blocks", "gate_up_proj_scales", "gate_up_proj_bias",
                               "down_proj_blocks", "down_proj_scales", "down_proj_bias")]
    T = _tensors(snap, names)
    gu_blocks, gu_scales, gu_bias = (T[pfx + "gate_up_proj_blocks"], T[pfx + "gate_up_proj_scales"],
                                     T[pfx + "gate_up_proj_bias"])
    dn_blocks, dn_scales, dn_bias = (T[pfx + "down_proj_blocks"], T[pfx + "down_proj_scales"],
                                     T[pfx + "down_proj_bias"])
    print(f"PROBE shapes gate_up_blocks={tuple(gu_blocks.shape)} scales={tuple(gu_scales.shape)} "
          f"bias={tuple(gu_bias.shape)} down_blocks={tuple(dn_blocks.shape)} "
          f"alpha={alpha} limit={limit} top_k={top_k}", flush=True)
    E = gu_blocks.shape[0]
    n_e = min(a.experts, E)

    # ---- stage 1: dequant. Independent decode is [E, rows, K] (output-major,
    # nn.Linear orientation); e4b returns input-major [E, K, rows].
    from experts4bit_qlora.formats.mxfp4 import dequantize_mxfp4
    ind_gu = independent_mxfp4_decode(gu_blocks[:n_e], gu_scales[:n_e])       # [e, 2I, H]
    ind_dn = independent_mxfp4_decode(dn_blocks[:n_e], dn_scales[:n_e])       # [e, H, I]
    e4b_gu = dequantize_mxfp4(gu_blocks[:n_e], gu_scales[:n_e], dtype=torch.float32)
    e4b_dn = dequantize_mxfp4(dn_blocks[:n_e], dn_scales[:n_e], dtype=torch.float32)
    for nm, ind, ours in (("gate_up", ind_gu, e4b_gu), ("down", ind_dn, e4b_dn)):
        cand = {"input-major(transposed)": ind.transpose(-1, -2), "output-major": ind}
        best = None
        for lay, ref in cand.items():
            if ref.shape == ours.shape:
                d = (ref - ours).abs().max().item()
                if best is None or d < best[1]:
                    best = (lay, d)
        print(f"PROBE dequant {nm}: e4b={tuple(ours.shape)} matches {best[0] if best else 'NOTHING'} "
              f"maxabs={best[1] if best else float('nan'):.3g} "
              f"({'EXACT' if best and best[1] == 0 else 'MISMATCH' if not best or best[1] > 1e-6 else 'close'})",
              flush=True)

    # ---- stage 2: the e4b module's forward vs the reference math
    from experts4bit_qlora.arch.gptoss import GptOssExperts4bit
    W_gu = ind_gu.transpose(-1, -2).contiguous().to(torch.bfloat16)   # [e, H, 2I] input-major interleaved
    W_dn = ind_dn.transpose(-1, -2).contiguous().to(torch.bfloat16)   # [e, I, H]
    mod = GptOssExperts4bit.from_gptoss(W_gu, gu_bias[:n_e].to(torch.bfloat16), W_dn,
                                        dn_bias[:n_e].to(torch.bfloat16), alpha=alpha,
                                        limit=limit, quant_type="nf4",
                                        compute_dtype=torch.bfloat16).to(dev)
    torch.manual_seed(0)
    H = W_gu.shape[1]
    x = (torch.randn(16, H) * 0.5).to(torch.bfloat16).to(dev)
    for e in range(n_e):
        idx = torch.full((16, 1), e, dtype=torch.long, device=dev)
        w = torch.ones(16, 1, dtype=torch.bfloat16, device=dev)
        got = mod(x, idx, w)
        ref = reference_expert(x.float(), W_gu[e].float().to(dev), gu_bias[e].float().to(dev),
                               W_dn[e].float().to(dev), dn_bias[e].float().to(dev), alpha, limit)
        cos, rel = _stats(got, ref)
        # a deliberately WRONG epilogue for scale: plain silu(gate)*up, no bias
        gate_up = x.float() @ W_gu[e].float().to(dev)
        wrong = (torch.nn.functional.silu(gate_up[..., ::2]) * gate_up[..., 1::2]) @ W_dn[e].float().to(dev)
        wcos, _ = _stats(wrong, ref)
        print(f"PROBE expert e={e}: cos={cos:.5f} rel={rel:.4f}  (plain-silu-no-bias control cos={wcos:.3f})",
              flush=True)

    # ---- stage 3: the served layer against the reference MoE
    if a.capture:
        cap = torch.load(a.capture, map_location="cpu")
        xs, ys = cap["x"].float(), cap["y"].float()
        rn = _tensors(snap, [f"model.layers.{a.layer}.mlp.router.weight",
                             f"model.layers.{a.layer}.mlp.router.bias"])
        Wr = rn[f"model.layers.{a.layer}.mlp.router.weight"].float()
        br = rn[f"model.layers.{a.layer}.mlp.router.bias"].float()
        logits = xs @ Wr.t() + br
        topv, topi = torch.topk(logits, top_k, dim=-1)
        topw = torch.softmax(topv, dim=-1)
        full_gu = independent_mxfp4_decode(gu_blocks, gu_scales).transpose(-1, -2).contiguous()  # [E, H, 2I]
        full_dn = independent_mxfp4_decode(dn_blocks, dn_scales).transpose(-1, -2).contiguous()  # [E, I, H]
        ref = torch.zeros_like(xs)
        for e in range(E):
            rows = (topi == e).any(-1).nonzero(as_tuple=True)[0]
            if rows.numel() == 0:
                continue
            we = (topw * (topi == e)).sum(-1)[rows][:, None]
            ref[rows] += we * reference_expert(xs[rows], full_gu[e], gu_bias[e].float(),
                                               full_dn[e], dn_bias[e].float(), alpha, limit)
        cos, rel = _stats(ys, ref)
        per_tok = torch.nn.functional.cosine_similarity(ys, ref, dim=-1)
        print(f"PROBE layer{a.layer} served-vs-reference: T={xs.shape[0]} cos={cos:.5f} rel={rel:.4f} "
              f"per-token cos min={per_tok.min():.4f} median={per_tok.median():.4f} "
              f"|y|={ys.norm(dim=-1).mean():.3f} |ref|={ref.norm(dim=-1).mean():.3f}", flush=True)
    print("PROBE done", flush=True)


if __name__ == "__main__":
    main()
