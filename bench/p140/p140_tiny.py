#!/usr/bin/env python3
"""p140_tiny.py -- lane P140's DRY fixture (bench/p140/PREREG-p140.md): a tiny random Qwen3-MoE checkpoint with Qwen3-30B-A3B's
tokenizer, so the dry run drives the load with expert offload, enable_fast_train, both accumulation loops and the reducer
on any CUDA card at no cost. Never a reading. P133's dims (hidden 512, expert width 256), which the grouped kernels take.

    python p140_tiny.py --out DIR/p140-tiny
"""
import argparse
import hashlib
import json
import os
import sys

QWEN, QWEN_REV = "Qwen/Qwen3-30B-A3B", "ad44e777bcd18fa416d9da3bd8f70d33ebb85d39"
DIMS = dict(hidden_size=512, intermediate_size=1024, moe_intermediate_size=256, num_hidden_layers=2,
            num_attention_heads=8, num_key_value_heads=4, head_dim=64, num_experts=8, num_experts_per_tok=2,
            max_position_embeddings=4096, tie_word_embeddings=False, initializer_range=0.2)
NAME = "p140-tiny"


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    a = p.parse_args(argv)
    if os.path.basename(a.out.rstrip("/")) != NAME:
        raise SystemExit(f"REFUSED: the fixture directory must be named {NAME} (the reducer reads it by that name)")
    import torch
    from transformers import AutoTokenizer, Qwen3MoeConfig, Qwen3MoeForCausalLM
    tok = AutoTokenizer.from_pretrained(QWEN, revision=QWEN_REV)
    cfg = Qwen3MoeConfig(vocab_size=len(tok), **DIMS)
    torch.manual_seed(0)
    model = Qwen3MoeForCausalLM(cfg).to(torch.bfloat16)
    # The real checkpoint's on-disk layout: one gate/up/down per expert (as tests/test_layer_major.py writes it), which the
    # streaming loader fuses -- not transformers 5's in-memory fused stack.
    from safetensors.torch import save_file
    new = {}
    for k, v in model.state_dict().items():
        if k.endswith("experts.gate_up_proj"):
            base = k[: -len("gate_up_proj")]
            for e in range(v.shape[0]):
                g, u = v[e].chunk(2, dim=0)
                new[f"{base}{e}.gate_proj.weight"] = g
                new[f"{base}{e}.up_proj.weight"] = u
        elif k.endswith("experts.down_proj"):
            base = k[: -len("down_proj")]
            for e in range(v.shape[0]):
                new[f"{base}{e}.down_proj.weight"] = v[e]
        else:
            new[k] = v
    os.makedirs(a.out, exist_ok=True)
    save_file({k: v.contiguous().clone() for k, v in new.items()}, os.path.join(a.out, "model.safetensors"))
    with open(os.path.join(a.out, "model.safetensors.index.json"), "w") as f:
        json.dump({"weight_map": {k: "model.safetensors" for k in new}}, f)
    cfg.save_pretrained(a.out)
    tok.save_pretrained(a.out)
    h = hashlib.sha256()
    for f in sorted(os.listdir(a.out)):
        if f.endswith(".safetensors"):
            h.update(open(os.path.join(a.out, f), "rb").read())
    print("P140_TINY " + json.dumps({"dir": a.out, "dims": DIMS, "vocab": len(tok), "weights_sha256": h.hexdigest(),
                                     "tokenizer": f"{QWEN}@{QWEN_REV}"}), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
