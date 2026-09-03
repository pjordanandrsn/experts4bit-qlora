# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""Does the gpt-oss arm gap reverse on in-distribution text? (2x2)

The arm-diff established that the two oracle arms do NOT diverge
structurally: identical attention config, bit-identical MXFP4 dequant,
and a per-layer cross-arm cosine that drifts smoothly from layer 1
(0.9997 -> 0.966 at layer 24) instead of breaking at one layer. What
remains is NF4 experts vs bf16, and the e4b arm scores LOWER NLL
(4.9552 vs 5.0900 on a full forward).

**Hypothesis.** Far out of distribution (raw wikitext, ppl in the
hundreds to thousands) the bf16 model is confidently wrong, and
quantization noise raises predictive entropy, which is rewarded there.
NF4 acts as a temperature increase, not as an improvement.

**Pre-registered prediction (fixed before running).** Each arm wins on
text IT generated, and each loses on the other's, because greedy
self-generated text is the arm's own argmax path:

    nll[up   on up-text] < nll[e4b on up-text]
    nll[e4b  on e4b-text] < nll[up  on e4b-text]

If instead ONE arm wins on BOTH texts, the entropy explanation is
falsified and that arm is defective.

Scoring is a single FULL forward per (arm, text) -- never the chunked
scorer, which is itself wrong on this family by ~0.02 nats
(KL(full||chunk)=0.0165, top1 93.9%).

Every line is prefixed ``INDIST``."""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys

import torch

PROMPTS = [
    "Explain why a bicycle stays upright when it is moving.",
    "Summarise how a compiler turns source code into machine code.",
    "Describe the water cycle for a curious ten-year-old.",
    "What makes sourdough bread rise, and why does it taste sour?",
    "Give three reasons a bridge might be built as a suspension bridge.",
    "How does noise-cancelling in headphones actually work?",
    "Explain the difference between weather and climate.",
    "Why do metals conduct electricity better than plastics?",
]


def _harmony_ids(tok, prompt, suffix="<|channel|>final<|message|>"):
    ids = tok.apply_chat_template([{"role": "user", "content": prompt}],
                                  add_generation_prompt=True, tokenize=True)
    if hasattr(ids, "input_ids"):
        ids = ids["input_ids"]
    ids = list(ids) + list(tok(suffix, add_special_tokens=False)["input_ids"])
    return torch.tensor(ids, dtype=torch.long)


@torch.no_grad()
def _generate(model, tok, n_new, dev):
    """Greedy continuations of each harmony prompt: {prompt_len, ids}."""
    out = []
    for p in PROMPTS:
        pre = _harmony_ids(tok, p)
        gen = model.generate(pre[None].to(dev), max_new_tokens=n_new,
                             do_sample=False, temperature=None, top_p=None,
                             top_k=None, use_cache=True,
                             pad_token_id=tok.eos_token_id)
        ids = gen[0].cpu()
        out.append({"prompt_len": int(pre.numel()), "ids": ids})
    return out


@torch.no_grad()
def _score_full(model, samples, dev):
    """Teacher-forced NLL over the GENERATED positions only, one full
    forward per sample. Returns (mean nll, [logits per sample])."""
    tot, n, keep = 0.0, 0, []
    for s in samples:
        ids = s["ids"].to(dev)
        pl = s["prompt_len"]
        lg = model(input_ids=ids[None]).logits[0].cpu().float()
        pred = lg[pl - 1:-1]                       # predicts ids[pl:]
        tgt = s["ids"][pl:]
        tot += float(-torch.log_softmax(pred, -1)
                     .gather(1, tgt[:, None]).sum())
        n += int(tgt.numel())
        keep.append(pred)
        del lg
        gc.collect()
        torch.cuda.empty_cache()
    return tot / max(1, n), keep


def _kl_top1(P, Q):
    kl, agree, n = 0.0, 0, 0
    for p, q in zip(P, Q):
        lp, lq = torch.log_softmax(p, -1), torch.log_softmax(q, -1)
        kl += float((lp.exp() * (lp - lq)).sum(-1).sum())
        agree += int((p.argmax(-1) == q.argmax(-1)).sum())
        n += int(p.shape[0])
    return kl / max(1, n), agree / max(1, n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="openai/gpt-oss-20b")
    ap.add_argument("--arena", required=True)
    ap.add_argument("--calib", default="/root/ctrl/calib.json")
    ap.add_argument("--n-new", type=int, default=96)
    ap.add_argument("--vram-gb", type=float, default=22.0)
    ap.add_argument("--dram-gb", type=float, default=0.0)
    ap.add_argument("--hot-rows", type=int, default=64)
    ap.add_argument("--work", default="/root/ctrl/indist")
    ap.add_argument("--stage", required=True,
                    choices=["upstream", "e4b", "upstream_on_e4b", "report"])
    a = ap.parse_args()
    os.makedirs(a.work, exist_ok=True)
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model)

    if a.stage == "upstream":
        from transformers import AutoModelForCausalLM
        _, total = torch.cuda.mem_get_info()
        budget = max(4, int(total / 2**30) - 6)
        m = AutoModelForCausalLM.from_pretrained(
            a.model, dtype=torch.bfloat16, device_map="auto",
            attn_implementation="eager",
            max_memory={0: f"{budget}GiB", "cpu": "120GiB"})
        m.eval()
        dev = next(m.parameters()).device
        samples = _generate(m, tok, a.n_new, dev)
        torch.save(samples, f"{a.work}/text_upstream.pt")
        lens = [int(s["ids"].numel() - s["prompt_len"]) for s in samples]
        print(f"INDIST generated[upstream]: {len(samples)} samples, "
              f"new tokens {lens}", flush=True)
        nll, P = _score_full(m, samples, dev)
        torch.save(P, f"{a.work}/logits_up_on_up.pt")
        print(f"INDIST score[up on up-text]: nll={nll:.5f}", flush=True)
        json.dump({"up_on_up": nll}, open(f"{a.work}/up.json", "w"))
        return

    if a.stage == "upstream_on_e4b":
        # the fourth cell: the bf16 arm on the text the NF4 arm generated.
        # A fresh load is required -- the e4b stage owned the card.
        from transformers import AutoModelForCausalLM
        _, total = torch.cuda.mem_get_info()
        budget = max(4, int(total / 2**30) - 6)
        m = AutoModelForCausalLM.from_pretrained(
            a.model, dtype=torch.bfloat16, device_map="auto",
            attn_implementation="eager",
            max_memory={0: f"{budget}GiB", "cpu": "120GiB"})
        m.eval()
        dev = next(m.parameters()).device
        e4b_text = torch.load(f"{a.work}/text_e4b.pt")
        nll, _ = _score_full(m, e4b_text, dev)
        rec = json.load(open(f"{a.work}/up.json"))
        rec["up_on_e4b"] = nll
        json.dump(rec, open(f"{a.work}/up.json", "w"))
        print(f"INDIST score[up on e4b-text]: nll={nll:.5f}", flush=True)
        return

    if a.stage == "e4b":
        from gptoss_arm_diff import _load_e4b, _set_eager
        m = _load_e4b(a)
        _set_eager(m)
        dev = torch.device("cuda")
        samples_e = _generate(m, tok, a.n_new, dev)
        torch.save(samples_e, f"{a.work}/text_e4b.pt")
        print(f"INDIST generated[e4b]: {len(samples_e)} samples", flush=True)
        up_text = torch.load(f"{a.work}/text_upstream.pt")
        nll_e_on_up, Q_up = _score_full(m, up_text, dev)
        nll_e_on_e, _ = _score_full(m, samples_e, dev)
        torch.save(Q_up, f"{a.work}/logits_e4b_on_up.pt")
        json.dump({"e4b_on_up": nll_e_on_up, "e4b_on_e4b": nll_e_on_e},
                  open(f"{a.work}/e4b.json", "w"))
        print(f"INDIST score[e4b on up-text]: nll={nll_e_on_up:.5f}", flush=True)
        print(f"INDIST score[e4b on e4b-text]: nll={nll_e_on_e:.5f}", flush=True)
        return

    # ---- report: needs upstream scored on e4b's text too
    up = json.load(open(f"{a.work}/up.json"))
    e4 = json.load(open(f"{a.work}/e4b.json"))
    up_on_e4b = up.get("up_on_e4b")
    P = torch.load(f"{a.work}/logits_up_on_up.pt")
    Q = torch.load(f"{a.work}/logits_e4b_on_up.pt")
    kl, top1 = _kl_top1(P, Q)
    print(f"INDIST 2x2 nll: up_on_up={up['up_on_up']:.5f} "
          f"e4b_on_up={e4['e4b_on_up']:.5f} "
          f"up_on_e4b={up_on_e4b if up_on_e4b is None else round(up_on_e4b, 5)} "
          f"e4b_on_e4b={e4['e4b_on_e4b']:.5f}", flush=True)
    win_up = e4["e4b_on_up"] > up["up_on_up"]
    print(f"INDIST prediction[up wins on up-text]: {win_up} "
          f"(margin {e4['e4b_on_up'] - up['up_on_up']:+.5f} nats)", flush=True)
    if up_on_e4b is not None:
        win_e4 = up_on_e4b > e4["e4b_on_e4b"]
        print(f"INDIST prediction[e4b wins on e4b-text]: {win_e4} "
              f"(margin {up_on_e4b - e4['e4b_on_e4b']:+.5f} nats)", flush=True)
        print(f"INDIST verdict: {'ENTROPY EXPLANATION HOLDS (each arm wins its own text)' if win_up and win_e4 else 'ONE ARM WINS BOTH -- that arm is DEFECTIVE, investigate it'}",
              flush=True)
    print(f"INDIST gate quantities on in-distribution text: "
          f"mean full-vocab KL(up||e4b)={kl:.5f} nats/token top1={top1:.4f} "
          f"(pre-registered gate: KL <= 0.01, top1 >= 0.99)", flush=True)


if __name__ == "__main__":
    main()
