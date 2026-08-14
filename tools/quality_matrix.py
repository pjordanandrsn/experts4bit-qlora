#!/usr/bin/env python3
"""Does the accuracy survive the config? Standardized tasks, real weights.

Everything else in this project compares our loader against
``from_pretrained`` on identical weights, where the answer is supposed to be
exactly 0.00 — and is. That proves the plumbing. It cannot tell you whether a
4-bit expert stack served through the fused kernel still *answers questions*
as well as the model its publisher released.

So this runs the EleutherAI harness — the same tasks and metrics model cards
report — across three nested comparisons, each answering a different question:

1. ``bf16_reference`` vs the **published** score. Does our harness protocol
   reproduce the model card at all? This is the loosest comparison and the one
   most likely to disagree for boring reasons: published numbers depend on
   harness version, shot count, and acc-vs-acc_norm, and publishers sometimes
   use their own eval code. A gap here indicts the protocol, not the loader.

2. ``nf4_*`` vs ``bf16_reference``. What does 4-bit quantization actually cost
   in accuracy? A real, expected, non-zero number worth knowing per model.

3. **``nf4_fast`` / ``nf4_pipelined`` vs ``nf4_baseline`` — the one that can
   fail.** An execution config is a performance optimization over identical
   weights, so it must be accuracy-NEUTRAL. Any non-zero delta here is a bug in
   the kernel or the residency split, not a tradeoff. This is the column to
   read first.

Each row records the ``enable_*`` patch count. A config that patched zero
modules never engaged, and its scores are the baseline's wearing another
name — the fused kernel is NF4-only, so every fp4 config does exactly that.
"""
from __future__ import annotations

import argparse
import gc
import json
import time
import traceback

import torch

import experts4bit_qlora as E

DEV = "cuda" if torch.cuda.is_available() else "cpu"

#: Scores as published by the model's own authors. Sourced from the model card
#: at the commit noted; they are a SANITY reference, not ground truth, because
#: the publisher's harness version / shot count / metric may differ from ours.
#: An empty dict means "we did not find a published number for this task" —
#: which is reported as such rather than silently omitted.
#: Only numbers actually READ off a model card go here. Anything recalled
#: rather than read is a fabrication risk and is better shown as "-".
PUBLISHED = {
    # allenai/OLMoE-1B-7B-0924 README.md, "LMs with ~1B active parameters" row.
    # CAVEAT: the OLMoE authors report via their own eval setup, and the card
    # does not state shot counts or acc-vs-acc_norm per task. So a gap against
    # these is far more likely to be a protocol difference than a defect —
    # which is exactly why the config-neutrality column, not this one, is the
    # one that can actually fail.
    "OLMoE-1B-7B-0924": {
        "mmlu": 54.1, "hellaswag": 80.0, "arc_challenge": 62.1,
        "arc_easy": 84.2, "piqa": 79.8, "winogrande": 70.2,
    },
    # Qwen/Qwen1.5-MoE-A2.7B ships a minimal card with NO benchmark table.
    # Empty on purpose: reported as "-" rather than invented.
    "Qwen1.5-MoE-A2.7B": {},
}

TASKS = ["arc_challenge", "arc_easy", "hellaswag", "winogrande", "piqa",
         "openbookqa", "boolq"]

MODELS = [
    ("allenai/OLMoE-1B-7B-0924", "olmoe", 64, 8),
    ("Qwen/Qwen1.5-MoE-A2.7B", "qwen2_moe", 60, 4),
]


def evaluate(model, repo, tasks, limit, batch_size=8):
    import lm_eval
    from lm_eval.models.huggingface import HFLM
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(repo, trust_remote_code=True)
    lm = HFLM(pretrained=model, tokenizer=tok, batch_size=batch_size, device=DEV)
    res = lm_eval.simple_evaluate(model=lm, tasks=tasks, limit=limit,
                                  verbosity="ERROR")
    out = {}
    for task, m in res["results"].items():
        for key in ("acc_norm,none", "acc,none", "exact_match,none"):
            if key in m:
                out[task] = round(float(m[key]) * 100, 2)
                break
    return out


def load_bf16(repo, **_):
    from transformers import AutoModelForCausalLM
    m = AutoModelForCausalLM.from_pretrained(
        repo, dtype=torch.bfloat16, trust_remote_code=True).to(DEV)
    return m.eval(), -1


def load_quant(repo, quant_type, config=None, n_experts=64, top_k=8):
    """Real weights, experts quantized on the way in, then the config applied."""
    model, _cfg = E.load_moe_4bit_streaming(
        repo, device=DEV, dtype=torch.bfloat16, r=8, alpha=16,
        quant_type=quant_type, trust_remote_code=True)
    model.eval()          # BEFORE enabling: the fused path self-bypasses in train mode
    # hot_sets is PER MoE LAYER — exactly one entry per ExpertsNbit module in
    # module order, not one entry total. The synthetic config matrix ran a
    # single bare module, so a one-entry list worked there and hid this; a real
    # 16-layer model raises. Count the layers from the model itself.
    n_layers = len(E.dispatched_modules(model)) or sum(
        1 for _ in model.modules() if type(_).__name__ == "ExpertsLoRA")
    all_hot = [list(range(n_experts))] * n_layers
    patched = -1
    if config == "fast":
        patched = E.enable_fast(model)
    elif config == "pipelined":
        patched = E.enable_pipelined_residency(model, all_hot, k_slots=top_k)
    elif config == "hot":
        patched = E.enable_hot_residency(model, all_hot)
    return model, patched


ARMS = [
    ("bf16_reference", lambda r, ne, k: load_bf16(r)),
    ("nf4_baseline", lambda r, ne, k: load_quant(r, "nf4", None, ne, k)),
    ("nf4_fast", lambda r, ne, k: load_quant(r, "nf4", "fast", ne, k)),
    ("nf4_pipelined", lambda r, ne, k: load_quant(r, "nf4", "pipelined", ne, k)),
    ("fp4_baseline", lambda r, ne, k: load_quant(r, "fp4", None, ne, k)),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=400,
                    help="examples per task; smaller is faster but noisier")
    ap.add_argument("--tasks", default=",".join(TASKS))
    ap.add_argument("--out", default="/workspace/results/quality.jsonl")
    ap.add_argument("--models", default="")
    a = ap.parse_args()
    tasks = a.tasks.split(",")
    models = MODELS
    if a.models:
        want = set(a.models.split(","))
        models = [m for m in MODELS if m[0].split("/")[-1] in want]

    for repo, _mt, n_experts, top_k in models:
        short = repo.split("/")[-1]
        base_scores = {}
        for arm, loader in ARMS:
            row = {"model": short, "arm": arm, "tasks": tasks,
                   "limit": a.limit, "ts": time.strftime("%FT%T")}
            try:
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                t0 = time.time()
                model, patched = loader(repo, n_experts, top_k)
                row["load_s"] = round(time.time() - t0, 1)
                row["patched"] = patched
                row["peak_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 2)
                row["scores"] = evaluate(model, repo, tasks, a.limit)
                # A config that patched nothing did not run; say so on the row
                # rather than letting equal scores read as "config is neutral".
                row["engaged"] = patched != 0
                row["status"] = "OK" if patched != 0 else "NOT-ENGAGED"
                if arm == "bf16_reference":
                    base_scores["bf16"] = row["scores"]
                if arm == "nf4_baseline":
                    base_scores["nf4"] = row["scores"]
                ref = base_scores.get("bf16")
                if ref:
                    row["vs_bf16"] = {t: round(row["scores"].get(t, 0) - ref[t], 2)
                                      for t in ref if t in row["scores"]}
                pub = PUBLISHED.get(short) or {}
                if pub and arm == "bf16_reference":
                    row["vs_published"] = {
                        t: round(row["scores"][t] - pub[t], 2)
                        for t in pub if t in row["scores"]}
                nf4 = base_scores.get("nf4")
                if nf4 and arm.startswith("nf4_") and arm != "nf4_baseline":
                    # THE column that can fail: same weights, different path.
                    row["vs_nf4"] = {t: round(row["scores"].get(t, 0) - nf4[t], 2)
                                     for t in nf4 if t in row["scores"]}
                    row["config_neutral"] = all(
                        abs(v) < 1e-9 for v in row["vs_nf4"].values())
                del model
            except Exception as e:
                row["status"] = f"FAIL {type(e).__name__}: {str(e)[:110]}"
                traceback.print_exc()
            finally:
                gc.collect()
                torch.cuda.empty_cache()
            with open(a.out, "a") as f:
                f.write(json.dumps(row) + "\n")
            print(f"{short:24s} {arm:16s} {row.get('status')} "
                  f"{row.get('scores', '')}", flush=True)
    print("QUALITY_MATRIX_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
