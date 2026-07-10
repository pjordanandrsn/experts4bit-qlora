#!/usr/bin/env python3
"""Phase-0 access-pattern sweep on the REAL model — forward-only, counting, no training.

Loads the base model (4-bit if CUDA, bf16 on CPU), attaches the gate-Linear counter, runs one
forward per (batch, seq) cell over real tokens, and dumps access_pattern.jsonl. The reframed grid
(from access_predictions.json) is token-count-centric: the useful signal is at eff_tokens <= a few
hundred, so the biggest single forward is ~2048 tokens — cheap on GPU, minutes on CPU.

  MODEL=Qwen/Qwen3-30B-A3B DEVICE=cuda OUT=access_pattern.jsonl python run_access_sweep.py

DEVICE=cuda loads load_in_4bit (fits a 24GB card resident, ~19GB); DEVICE=cpu loads bf16 (needs
~61GB host RAM). No offload, no e4b — the base router is all that is exercised.
"""

import json
import os
import sys

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from access_counter import ExpertAccessCounter, attach

MODEL = os.environ.get("MODEL", "Qwen/Qwen3-30B-A3B")
DEVICE = os.environ.get("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
OUT = os.environ.get("OUT", "access_pattern.jsonl")

# Reframed grid: token-count sweep (see access_predictions.json). (batch, seq).
CELLS = [
    (1, 1), (1, 2), (1, 4), (1, 8), (1, 16), (1, 32), (1, 64),
    (1, 128), (1, 256), (1, 512), (1, 1024), (1, 2048),
    (8, 8), (8, 64),  # batch>1 vs long-seq at matched eff_tokens 64 / 512
]


def _n_experts(cfg):
    for k in ("num_experts", "num_local_experts", "n_routed_experts"):
        if getattr(cfg, k, None):
            return getattr(cfg, k)
    raise SystemExit("could not find expert count on config")


def _top_k(cfg):
    for k in ("num_experts_per_tok", "experts_per_token", "top_k_experts", "moe_top_k"):
        if getattr(cfg, k, None):
            return getattr(cfg, k)
    raise SystemExit("could not find top_k on config")


def main():
    cfg = AutoConfig.from_pretrained(MODEL, trust_remote_code=True)
    tcfg = getattr(cfg, "text_config", cfg)
    E, k = _n_experts(tcfg), _top_k(tcfg)
    print(f"model={MODEL} device={DEVICE} E={E} k={k}", flush=True)

    load_kw = dict(trust_remote_code=True, low_cpu_mem_usage=True)
    if DEVICE == "cuda":
        from transformers import BitsAndBytesConfig

        load_kw.update(
            device_map="cuda",
            quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4"),
        )
    else:
        load_kw.update(torch_dtype=torch.bfloat16, device_map="cpu")

    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(MODEL, **load_kw).eval()

    counter = ExpertAccessCounter(E, k)
    n_hooked = attach(model, counter)
    print(f"hooked {n_hooked} router gates", flush=True)
    if n_hooked == 0:
        raise SystemExit("no gates hooked — check the router seam for this model_type")

    # Real tokens: a fixed deterministic slice, so the routing reflects real content correlation
    # (the whole reason the measured curve can deviate from the uniform null).
    corpus = (
        "The frozen-expert memory hierarchy question is where placement and access pattern cross. "
        "Mixture-of-experts routing gathers a token-dependent subset of experts per layer. "
    ) * 400
    all_ids = tok(corpus, return_tensors="pt").input_ids[0]

    dev = "cuda" if DEVICE == "cuda" else "cpu"
    with torch.no_grad():
        for step, (b, s) in enumerate(CELLS):
            need = b * s
            if all_ids.numel() < need:
                reps = need // all_ids.numel() + 1
                ids = all_ids.repeat(reps)[:need]
            else:
                ids = all_ids[:need]
            batch_ids = ids.reshape(b, s).to(dev)
            model(batch_ids, use_cache=False)
            counter.close_step(step, batch=b, seq=s)
            rows = [r for r in counter.records if r["step"] == step]
            mean_rf = sum(r["read_fraction"] for r in rows) / len(rows)
            print(f"  eff_tokens={b*s:>6} (b{b}xs{s})  mean_read_fraction={mean_rf:.4f}", flush=True)

    counter.dump(OUT)
    print(f"wrote {OUT} ({len(counter.records)} rows)", flush=True)


if __name__ == "__main__":
    main()
