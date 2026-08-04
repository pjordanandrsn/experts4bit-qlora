"""kl_sweep.py — locate the churn-to-loss threshold with a dense within-model KL sweep.

STATUS: BLOCKED, kept as a record of why. Every k failed with
``assert module.weight.shape[1] == 1`` inside bitsandbytes: granite's MoE experts are
FUSED 3D tensors ([n_experts, out, in]) and ``Linear4bit`` cannot quantize them — which
is the reason experts4bit-qlora exists in the first place. Partial quantization therefore
has to go through e4b's own loader, which does not currently expose a per-layer subset.

This also qualifies the int8 anchor measured earlier (KL 8.09e-03): bitsandbytes int8 hit
the same limitation and can only have quantized the NON-expert Linears, leaving the MoE
experts in bf16. That anchor is still a valid (KL, accuracy) pair — both axes were
measured, not assumed — but it is NOT "int8 of the whole model", and must not be
described that way.

The three prior runs bound the threshold between 2.2e-02 (clean) and 1.41e-01
(destructive) but cannot locate it: no quantization SCHEME lands in that gap. The dial
that does is PARTIAL quantization — quantize the first k transformer layers to NF4 and
leave the rest bf16, and KL interpolates smoothly from 0 to the fully-quantized value.

CAVEAT, stated up front because it limits the claim: partial quantization concentrates
damage in some layers rather than spreading it uniformly. If the threshold depends on the
DISTRIBUTION of damage and not only its aggregate KL, a partial-quant sweep locates the
threshold for concentrated damage specifically. Uniform-damage dials (scheme, blocksize)
cannot fill this gap at all, so this is the available instrument, not the ideal one.

Scored on T1-T3 only (583 probes). T4-T7 sat at floor in every prior run and contribute
noise, not signal; the registered tier floor already excluded them from directional
statements. Declared deviation from the 1311-probe prereg default.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kl_fidelity import teacher_forced_logits  # noqa: E402
from kl_ikp import is_refusal, load_probes  # noqa: E402
from kl_paths import _gate_on_k0, score_pair  # noqa: E402

SWEEP_TIERS = ("T1", "T2", "T3")


def load_partial_nf4(model_id: str, k: int, n_layers: int):
    """Quantize layers [0, k) to NF4; keep [k, n_layers) in bf16.

    Skip-list matching is substring-based in transformers, so "model.layers.4" would also
    match "model.layers.40". Safe here because granite has 24 layers; assert it.
    """
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig
    assert n_layers < 100, "substring skip-list is ambiguous past 100 layers"
    skip = [f"model.layers.{i}" for i in range(k, n_layers)]
    if k == 0:
        return AutoModelForCausalLM.from_pretrained(
            model_id, dtype=torch.bfloat16, device_map="auto").eval()
    cfg = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=torch.bfloat16,
                             llm_int8_skip_modules=skip or None)
    return AutoModelForCausalLM.from_pretrained(
        model_id, quantization_config=cfg, dtype=torch.bfloat16,
        device_map="auto").eval()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="ibm-granite/granite-3.0-1b-a400m-instruct")
    ap.add_argument("--k0-receipt", required=True)
    ap.add_argument("--k", type=int, required=True, help="how many layers to quantize")
    ap.add_argument("--probes", default="/root/ikp_clean.json")
    ap.add_argument("--max-len", type=int, default=320)
    ap.add_argument("--out-prefix", required=True)
    a = ap.parse_args()

    _gate_on_k0(a.k0_receipt)
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    dev = "cuda"
    tok = AutoTokenizer.from_pretrained(a.model)
    n_layers = AutoConfig.from_pretrained(a.model).num_hidden_layers

    # ---- x first: KL of this partial quantization against the bf16 base
    base = AutoModelForCausalLM.from_pretrained(
        a.model, dtype=torch.bfloat16).to(dev).eval()
    part = load_partial_nf4(a.model, a.k, n_layers)
    t0 = time.time()
    r = score_pair(lambda i: teacher_forced_logits(base, i),
                   lambda i: teacher_forced_logits(part, i), tok, a.max_len, dev)
    r.update({"row": f"partial NF4: first {a.k}/{n_layers} layers quantized",
              "reference": f"the bf16 base ({a.model})",
              "test": f"NF4 on layers [0,{a.k}), bf16 on [{a.k},{n_layers})",
              "k": a.k, "n_layers": n_layers, "wall_s": round(time.time() - t0, 1)})
    json.dump(r, open(f"{a.out_prefix}_kl.json", "w"), indent=1)
    print(f"  k={a.k}  KL={r['kl_mean']:.6e}  top1={r['top1_agreement']:.4f}", flush=True)
    del base
    torch.cuda.empty_cache()

    # ---- y second: IKP on the signal-bearing tiers
    probes = [p for p in load_probes(a.probes) if p["tier"] in SWEEP_TIERS]
    out, ans_f = [], f"{a.out_prefix}_ans.json"
    if os.path.exists(ans_f):
        out = json.load(open(ans_f))["answers"]
    for p in probes[len(out):]:
        msgs = [{"role": "system", "content": "You are answering factual knowledge "
                 "questions. Answer concisely with just the factual answer."},
                {"role": "user", "content": p["question"]}]
        enc = tok.apply_chat_template(msgs, add_generation_prompt=True,
                                      return_tensors="pt", return_dict=True)
        ids = enc["input_ids"].to(part.device)
        with torch.no_grad():
            g = part.generate(ids, max_new_tokens=48, do_sample=False,
                              pad_token_id=tok.eos_token_id)
        t = tok.decode(g[0][ids.shape[-1]:], skip_special_tokens=True).strip()
        out.append({"id": p["id"], "tier": p["tier"], "question": p["question"],
                    "gold": p["answer"], "response": t, "refusal": is_refusal(t)})
        if len(out) % 100 == 0:
            json.dump({"path": f"k{a.k}", "answers": out}, open(ans_f, "w"))
            print(f"    {len(out)}/{len(probes)}", flush=True)
    json.dump({"path": f"k{a.k}", "answers": out}, open(ans_f, "w"))
    print(f"  k={a.k} generated {len(out)} -> {ans_f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
