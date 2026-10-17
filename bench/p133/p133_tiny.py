#!/usr/bin/env python3
"""p133_tiny.py -- lane P133's DRY fixture (bench/p133/PREREG-p133.md): a tiny random Qwen3-MoE checkpoint with Qwen3-30B-A3B's
tokenizer, so the dry run drives the default ``serve_paged`` server, the bake, every arm, the quality instrument and the
reducer on any CUDA card at no cost. Never a reading.

Every expert GEMV's K (hidden 512, expert width 256) is a multiple of 256, so both decode routes (the scalar GEMV and
``GNF4_GEMV_BW=1``'s default plan) take their full K-steps. ``initializer_range`` 0.2 (10x the default) gives the random
model peaked logits, so P110's scale mutant moves the NLL measurably and the reducer reaches its gain rungs.

    python p133_tiny.py --out DIR/p133-tiny
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
NAME = "p133-tiny"


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
    Qwen3MoeForCausalLM(cfg).to(torch.bfloat16).save_pretrained(a.out)
    tok.save_pretrained(a.out)
    h = hashlib.sha256()
    for f in sorted(os.listdir(a.out)):
        if f.endswith(".safetensors"):
            h.update(open(os.path.join(a.out, f), "rb").read())
    print("P133_TINY " + json.dumps({"dir": a.out, "dims": DIMS, "vocab": len(tok), "weights_sha256": h.hexdigest(),
                                     "tokenizer": f"{QWEN}@{QWEN_REV}"}), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
