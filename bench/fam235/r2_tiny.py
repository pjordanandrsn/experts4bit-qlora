# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""The CPU rehearsal's model: a tiny random Mixtral of the subject's architecture (4 decoder layers, 8 experts, top-2), saved with
``save_pretrained`` beside the subject checkpoint's tokenizer files, so the box script runs end to end on a CPU host. Never a
scientific input: it is not the registered model.

    python r2_tiny.py --out DIR [--tokenizer-from DIR]   (the tokenizer files default to the staged closure's)
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import torch

TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json", "tokenizer.model", "special_tokens_map.json")


def build(out: Path, tokenizer_from: Path) -> None:
    from transformers import MixtralConfig, MixtralForCausalLM
    cfg = MixtralConfig(vocab_size=32768, hidden_size=128, intermediate_size=64, num_hidden_layers=4, num_attention_heads=4,
                        num_key_value_heads=2, head_dim=32, num_local_experts=8, num_experts_per_tok=2, tie_word_embeddings=False,
                        max_position_embeddings=4096)
    torch.manual_seed(0)
    MixtralForCausalLM(cfg).to(torch.bfloat16).save_pretrained(out)
    for name in TOKENIZER_FILES:
        src = tokenizer_from / name
        if src.is_file():
            shutil.copyfile(src, out / name)
    if not (out / "tokenizer.json").is_file():
        raise SystemExit(f"no tokenizer.json in {tokenizer_from}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--tokenizer-from", type=Path, default=Path.cwd())
    a = ap.parse_args()
    build(a.out, a.tokenizer_from)
    print(f"tiny Mixtral -> {a.out}")


if __name__ == "__main__":
    main()
