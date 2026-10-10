# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""The NF4 side of Q235's fidelity read: e4b's training path as the box runs it -- experts NF4 in pinned host RAM, NF4 attention, LoRA
(B zero-initialized, so the forward is the base model's), ``enable_fast_train`` at the release's defaults -- one training forward
(gradients enabled, as the step's step-0 forward is) on the given token ids. Hooks record each decoder layer's output and its router's
top-k; the loss is the model's own. Run it twice in separate processes for the reproducibility floor."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time

import torch


def run(path, tokens, out_dir, *, r=16, alpha=16, adapter_dtype="bf16", offload=True):
    from experts4bit_qlora import enable_fast_train, load_moe_4bit_streaming
    from experts4bit_qlora.lora import add_attention_lora, quantize_attention_projections_4bit
    dt = {"bf16": torch.bfloat16, "fp32": torch.float32}[adapter_dtype]
    t0 = time.time()
    torch.manual_seed(0)
    model, _ = load_moe_4bit_streaming(path, "cuda", torch.bfloat16, r, alpha, offload=offload, pin=True, prefetch=False,
                                       quant_type="nf4")
    quantize_attention_projections_4bit(model)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.config.use_cache = False
    add_attention_lora(model, r, alpha, dt)
    n = enable_fast_train(model)
    assert n > 0, "enable_fast_train patched nothing"
    load_s = time.time() - t0
    model.train()
    layers = model.model.layers
    outs, topk = [None] * len(layers), [None] * len(layers)
    hooks = []
    for i, lay in enumerate(layers):
        hooks.append(lay.register_forward_hook(
            lambda m, a, o, i=i: outs.__setitem__(i, (o[0] if isinstance(o, tuple) else o).detach().to("cpu").contiguous())
            if outs[i] is None else None))
        hooks.append(lay.mlp.gate.register_forward_hook(
            lambda m, a, o, i=i: topk.__setitem__(i, o[2].detach().to("cpu")) if topk[i] is None else None))
    ids, mask, labels = (tokens[k].to("cuda") for k in ("input_ids", "attention_mask", "labels"))
    torch.cuda.synchronize()
    t1 = time.time()
    out = model(input_ids=ids, attention_mask=mask, labels=labels)
    torch.cuda.synchronize()
    fwd_s = time.time() - t1
    for h in hooks:
        h.remove()
    os.makedirs(out_dir, exist_ok=True)
    sha = []
    for i, o in enumerate(outs):
        torch.save(o, os.path.join(out_dir, f"layer{i:03d}.pt"))
        sha.append(hashlib.sha256(o.view(torch.uint8).numpy().tobytes()).hexdigest())
    torch.save({"router_topk": topk}, os.path.join(out_dir, "nf4.pt"))
    rec = {"loss": float(out.loss), "layers": len(layers), "layer_sha256": sha, "adapter_dtype": adapter_dtype, "offload": offload,
           "patched": n, "load_s": round(load_s, 1), "forward_s": round(fwd_s, 2), "max_memory_allocated": torch.cuda.max_memory_allocated(),
           "torch": torch.__version__}
    json.dump(rec, open(os.path.join(out_dir, "nf4.json"), "w"), indent=1)
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint")
    ap.add_argument("tokens")
    ap.add_argument("--out", required=True)
    ap.add_argument("--adapter-dtype", default="bf16", choices=("bf16", "fp32"))
    a = ap.parse_args()
    rec = run(a.checkpoint, torch.load(a.tokens), a.out, adapter_dtype=a.adapter_dtype)
    print(json.dumps({k: rec[k] for k in ("loss", "layers", "patched", "load_s", "forward_s")}))


if __name__ == "__main__":
    main()
