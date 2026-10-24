"""Plumbing only: a synthetic fixture-base (layer 0, CPU-quantized like e4b's _quantize_stack, plus the DQ payload as
_compress_one builds it) and a random adapter, so the D0 tools run end to end before the real fixtures exist."""
import json
import os
import sys

import torch
import bitsandbytes.functional as F
from safetensors.torch import save_file

import d0lib

out, layers, E = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
ck = d0lib.Checkpoint("/home/node/work/d0/olmoe-instruct-7f1c97f4")
cfg = json.load(open(os.path.join(ck.path, "config.json")))
H, inter, r = cfg["hidden_size"], cfg["intermediate_size"], 8
g = torch.Generator().manual_seed(0)
stacks, dq, ad = {}, {}, {}
for L in range(layers):
    for which, (n, k) in (("gate_up", (2 * inter, H)), ("down", (H, inter))):
        packed, absmax = [], []
        for e in range(cfg["num_experts"]):
            w = ck.expert(L, e, which) if e < E else torch.zeros(n, k)
            q, st = F.quantize_4bit(w.contiguous(), blocksize=64, quant_type="nf4", compress_statistics=False)
            packed.append(q.reshape(-1))
            absmax.append(st.absmax.reshape(-1))
        stacks[f"{L}.{which}_proj"], am = torch.stack(packed), torch.stack(absmax)
        stacks[f"{L}.{which}_absmax"] = am
        flat = am.reshape(-1)
        off = flat.mean()
        qa, s2 = F.quantize_blockwise(flat - off, blocksize=256)
        dq.update({f"{L}.{which}_absmax_q": qa.reshape(am.shape).contiguous(), f"{L}.{which}_absmax_s": s2.absmax,
                   f"{L}.{which}_absmax_off": off.reshape(1).float(), f"{L}.{which}_absmax_code": s2.code.float()})
        nE = cfg["num_experts"]
        ad[f"base.model.layers.{L}.mlp.experts.{which}_lora_A"] = (torch.randn(nE, r, k, generator=g) / r).bfloat16()
        ad[f"base.model.layers.{L}.mlp.experts.{which}_lora_B"] = (torch.randn(nE, n, r, generator=g) * 2e-3).bfloat16()
os.makedirs(os.path.join(out, "fixture-base"), exist_ok=True)
os.makedirs(os.path.join(out, "adapter"), exist_ok=True)
save_file(stacks, os.path.join(out, "fixture-base", "stacks.safetensors"))
save_file(dq, os.path.join(out, "fixture-base", "absmax_dq.safetensors"))
save_file(ad, os.path.join(out, "adapter", "adapter.safetensors"))
json.dump({"setup": {"r": r, "alpha": 16}, "synthetic": True}, open(os.path.join(out, "adapter", "adapter_manifest.json"), "w"))
print("synthetic fixture written", out)
