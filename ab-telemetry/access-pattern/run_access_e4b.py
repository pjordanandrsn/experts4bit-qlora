#!/usr/bin/env python3
"""Phase-0 access sweep via the e4b streaming 4-bit loader — measures routing of the ACTUAL
quantized model the SSD-tier thesis deploys (fused experts in NF4), which fits a 24GB card.

  MODEL=Qwen/Qwen3-30B-A3B OUT=access_qwen30b.jsonl python run_access_e4b.py
  MODEL=allenai/OLMoE-1B-7B-0924 OUT=access_olmoe.jsonl python run_access_e4b.py

Forward-only, counting. No training, no offload needed (experts NF4-resident). The reframed grid
maxes at one 2048-token forward, so this is minutes on the GPU.
"""

import os
import sys

import torch
from transformers import AutoConfig, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from access_counter import ExpertAccessCounter, attach

from experts4bit_qlora.loader import load_moe_4bit_streaming

MODEL = os.environ.get("MODEL", "Qwen/Qwen3-30B-A3B")
OUT = os.environ.get("OUT", "access_pattern.jsonl")
DEV = "cuda"

# Reframed token-count grid (see access_predictions.json): (batch, seq).
CELLS = [
    (1, 1), (1, 2), (1, 4), (1, 8), (1, 16), (1, 32), (1, 64),
    (1, 128), (1, 256), (1, 512), (1, 1024), (1, 2048),
    (8, 8), (8, 64),  # batch>1 vs long-seq at matched eff_tokens 64 / 512
]


def _expert_count(cfg):
    t = getattr(cfg, "text_config", cfg)
    for k in ("num_experts", "num_local_experts", "n_routed_experts"):
        if getattr(t, k, None):
            return getattr(t, k)
    raise SystemExit("no expert count")


def _top_k(cfg):
    t = getattr(cfg, "text_config", cfg)
    for k in ("num_experts_per_tok", "experts_per_token", "top_k_experts", "moe_top_k"):
        if getattr(t, k, None):
            return getattr(t, k)
    raise SystemExit("no top_k")


def main():
    cfg = AutoConfig.from_pretrained(MODEL, trust_remote_code=True)
    E, k = _expert_count(cfg), _top_k(cfg)
    print(f"model={MODEL} E={E} k={k}", flush=True)

    model, _cfg = load_moe_4bit_streaming(
        MODEL, device=DEV, dtype=torch.bfloat16, r=8, alpha=16, offload=False, quant_type="nf4"
    )
    model.eval()

    counter = ExpertAccessCounter(E, k)
    n = attach(model, counter)
    print(f"hooked {n} router gates", flush=True)
    assert n > 0, "no gates hooked"

    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    corpus = (
        "The frozen-expert memory hierarchy question is where placement and access pattern cross. "
        "Mixture-of-experts routing gathers a token-dependent subset of experts per layer, and the "
        "union over a batch grows toward the total as effective tokens increase. "
    ) * 500
    ids0 = tok(corpus, return_tensors="pt").input_ids[0]

    with torch.no_grad():
        for step, (b, s) in enumerate(CELLS):
            need = b * s
            ids = (ids0.repeat(need // ids0.numel() + 1)[:need] if ids0.numel() < need else ids0[:need])
            model(ids.reshape(b, s).to(DEV), use_cache=False)
            counter.close_step(step, batch=b, seq=s)
            rows = [r for r in counter.records if r["step"] == step]
            mrf = sum(r["read_fraction"] for r in rows) / len(rows)
            print(f"  eff={b*s:>6} (b{b}xs{s})  mean_read_fraction={mrf:.4f}", flush=True)

    counter.dump(OUT)
    print(f"wrote {OUT} ({len(counter.records)} rows)", flush=True)


if __name__ == "__main__":
    main()
