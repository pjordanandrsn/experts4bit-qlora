"""kl_ikp.py — K3: does KL predict knowledge loss on fused-MoE 4-bit?

READ bench/K3-PREREG.md FIRST. It was committed before this file produced a number, and
it constrains what may be claimed — in particular it forbids reporting a correlation
coefficient, because three points (one of them definitionally zero) cannot support one.

Three phases, so the work survives a contended GPU:
  --phase generate --path {bf16,nf4,fp4}   greedy answers for the pinned probe set
  --phase judge                            verbatim judge prompt from IKP's own scorer
  --phase score                            tier accuracy + binomial SE + kill switch

Benchmark: IKP (Incompressible Knowledge Probes), Bojie Li / Pine AI, arXiv 2604.24827,
repo 19PINE-AI/ikp. Probe set CC BY 4.0; code MIT. Motivation: Quesma 2026-08-03, which
reports r=-0.981 over 55 *dense GGUF k-quants* — a different quantization family from ours.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import urllib.request

# The exact cleaned subset the paper scored. A different file is a different benchmark,
# so this is verified, not trusted.
PROBE_SHA256 = "a64ab7f3c06c919986298ffdaa501843ce4238494219c4c74d8f108f7d65c1db"
PROBE_N = 1311
TIERS = ["T1", "T2", "T3", "T4", "T5", "T6", "T7"]

# Registered in the prereg, restated here so the thresholds live next to the code that
# applies them rather than only in prose.
KILL_SWITCH_OVERALL = 0.10   # bf16 below this => under-powered, no KL<->accuracy inference
TIER_FLOOR = 0.05            # tier below this => excluded from directional statements

_judge_local_fn = None       # set by --judge-local


def load_probes(path: str) -> list[dict]:
    raw = open(path, "rb").read()
    got = hashlib.sha256(raw).hexdigest()
    if got != PROBE_SHA256:
        raise SystemExit(f"probe set hash mismatch\n  expected {PROBE_SHA256}\n  got      {got}\n"
                         "Refusing to score: a different probe file is a different benchmark.")
    d = json.loads(raw)
    items = d if isinstance(d, list) else (d.get("probes") or list(d.values())[0])
    if len(items) != PROBE_N:
        raise SystemExit(f"probe count {len(items)} != {PROBE_N}")
    return items


def is_refusal(response: str) -> bool:
    """Verbatim from IKP src/scorer.py — reproduced, not reinterpreted."""
    if not response or not response.strip():
        return True
    r = response.lower().strip()
    for phrase in ["i don't know", "i do not know", "i'm not sure", "i am not sure",
                   "i cannot", "i can't", "i'm unable", "i am unable", "not familiar",
                   "no information", "not aware", "unknown"]:
        if phrase in r:
            return True
    return False


def judge_prompt(question: str, gold: str, response: str) -> str:
    """Verbatim from IKP src/scorer.py score_with_llm_judge."""
    return f"""You are a strict factual judge. Does the model's response contain the correct answer?

Question: {question}
Correct answer: {gold}
Model's response: {response}

Strict rules:
1. If the correct answer is a YEAR (like 1931), the model must state that EXACT year. A different year (e.g. 1960 instead of 1931) is INCORRECT.
2. If the correct answer is a NUMBER, it must match exactly or be very close (within 1-2%).
3. If the correct answer is a NAME, the model must name the same entity. Minor spelling differences are OK.
4. If the correct answer is a RESEARCH FIELD, accept synonyms (e.g. "networking" = "computer networking") but reject unrelated fields. The model must demonstrate it actually knows the person's field, not just guess randomly.
5. If the model clearly does not know the answer, gives an unrelated response, or says "I don't know", that is INCORRECT.
6. If the model gives a DIFFERENT year, a DIFFERENT name, or a DIFFERENT field from the correct answer, that is INCORRECT.

Reply with exactly one word: CORRECT or INCORRECT"""


def parse_verdict(text: str) -> bool:
    """IKP's rule, including its explicit warning: do not match CORRECT inside INCORRECT."""
    r = (text or "").strip().upper()
    return r.startswith("CORRECT")


# ----------------------------------------------------------------------------- generate

def phase_generate(a) -> int:
    import torch
    from transformers import AutoTokenizer

    probes = load_probes(a.probes)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(a.model)

    if a.path == "dqbf16":
        # The reference path: shipped MXFP4 bytes dequantized to bf16.
        from kl_gptoss import load_dequant_bf16
        model = load_dequant_bf16(a.model, dev)
    elif a.path == "mxfp4":
        # Guarded: refuses if transformers silently dequantized instead of running native.
        from kl_gptoss import load_native_mxfp4
        model = load_native_mxfp4(a.model, dev)
    elif a.path == "bf16":
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(
            a.model, torch_dtype=torch.bfloat16, trust_remote_code=True).to(dev).eval()
    else:
        from experts4bit_qlora.loader import load_moe_4bit_streaming
        model, _ = load_moe_4bit_streaming(a.model, dev, torch.bfloat16, 8, 16,
                                           offload=False, prefetch=False, quant_type=a.path)
        model.eval()

    out, done = [], 0
    if os.path.exists(a.out) and a.resume:
        out = json.load(open(a.out))["answers"]
        done = len(out)
        print(f"  resuming at {done}/{len(probes)}")

    for p in probes[done:]:
        # Prereg: the direct-question phrasing only (1 of IKP's 3), declared deviation.
        msgs = [{"role": "system",
                 "content": "You are answering factual knowledge questions. "
                            "Answer concisely with just the factual answer."},
                {"role": "user", "content": p["question"]}]
        # return_dict=True then index: apply_chat_template hands back a BatchEncoding on
        # current transformers, and passing that straight to generate() dies inside
        # generate() with a bare AttributeError that names neither the cause nor this call.
        if getattr(tok, "chat_template", None):
            enc = tok.apply_chat_template(msgs, add_generation_prompt=True,
                                          return_tensors="pt", return_dict=True)
            ids = enc["input_ids"].to(dev)
        else:
            # BASE models (e.g. OLMoE-1B-7B-0924) ship no chat template, and
            # apply_chat_template then raises. Fall back to a plain completion prompt —
            # the same prompt for every path, so cross-path comparison stays valid.
            ids = tok(f"Question: {p['question']}\nAnswer:",
                      return_tensors="pt")["input_ids"].to(dev)
        with torch.no_grad():
            gen = model.generate(ids, max_new_tokens=48, do_sample=False,
                                 pad_token_id=tok.eos_token_id)
        text = tok.decode(gen[0][ids.shape[-1]:], skip_special_tokens=True).strip()
        out.append({"id": p["id"], "tier": p["tier"], "question": p["question"],
                    "gold": p["answer"], "response": text, "refusal": is_refusal(text)})
        if len(out) % 50 == 0:
            json.dump({"path": a.path, "model": a.model, "answers": out},
                      open(a.out, "w"))
            print(f"  {len(out)}/{len(probes)}", flush=True)

    json.dump({"path": a.path, "model": a.model, "answers": out}, open(a.out, "w"))
    print(f"  wrote {len(out)} answers -> {a.out}")
    return 0


# -------------------------------------------------------------------------------- judge

def _judge_http(url: str, model: str, prompt: str, timeout: int = 120) -> str:
    body = json.dumps({"model": model, "temperature": 0, "max_tokens": 8,
                       "messages": [{"role": "user", "content": prompt}]}).encode()
    req = urllib.request.Request(url, data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)["choices"][0]["message"]["content"]


def _judge_local(model_id: str):
    """In-process judge. IKP's own scorer defaults to a small local model
    (create_ollama_judge("qwen3:4b")), so judging locally is the method's design, not a
    deviation from it. Absolute accuracies still are not comparable to the published
    Gemini-judged numbers — only across our own paths, which is the comparison we need."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tk = AutoTokenizer.from_pretrained(model_id)
    md = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=torch.bfloat16, device_map="auto").eval()

    def call(prompt: str) -> str:
        enc = tk.apply_chat_template([{"role": "user", "content": prompt}],
                                     add_generation_prompt=True, return_tensors="pt",
                                     return_dict=True)
        ids = enc["input_ids"].to(md.device)
        with torch.no_grad():
            out = md.generate(ids, max_new_tokens=6, do_sample=False,
                              pad_token_id=tk.eos_token_id)
        return tk.decode(out[0][ids.shape[-1]:], skip_special_tokens=True)
    return call


def phase_judge(a) -> int:
    d = json.load(open(a.answers))
    answers = d["answers"]
    global _judge_local_fn
    if a.judge_local:
        _judge_local_fn = _judge_local(a.judge_local)
    verdicts = {}
    if os.path.exists(a.out) and a.resume:
        verdicts = json.load(open(a.out))["verdicts"]
        print(f"  resuming with {len(verdicts)} verdicts")

    n_skip = 0
    for i, ans in enumerate(answers):
        if ans["id"] in verdicts:
            continue
        # IKP excludes refusals from scoring rather than counting them wrong.
        if ans["refusal"]:
            verdicts[ans["id"]] = {"correct": False, "excluded": True}
            n_skip += 1
            continue
        p = judge_prompt(ans["question"], ans["gold"], ans["response"])
        raw = (_judge_local_fn(p) if _judge_local_fn
               else _judge_http(a.judge_url, a.judge_model, p))
        verdicts[ans["id"]] = {"correct": parse_verdict(raw), "excluded": False,
                               "raw": raw.strip()[:40]}
        if len(verdicts) % 50 == 0:
            json.dump({"path": d["path"], "verdicts": verdicts}, open(a.out, "w"))
            print(f"  judged {len(verdicts)}/{len(answers)}", flush=True)

    json.dump({"path": d["path"], "judge_model": a.judge_model, "verdicts": verdicts},
              open(a.out, "w"))
    print(f"  {len(verdicts)} verdicts ({n_skip} refusals excluded) -> {a.out}")
    return 0


# -------------------------------------------------------------------------------- score

def _wilson_se(k: int, n: int) -> float:
    if n == 0:
        return float("nan")
    p = k / n
    return (p * (1 - p) / n) ** 0.5


def phase_score(a) -> int:
    paths = {}
    for spec in a.score:
        name, ans_f, ver_f, kl = spec.split(":", 3)
        answers = json.load(open(ans_f))["answers"]
        verdicts = json.load(open(ver_f))["verdicts"]
        per_tier = {t: [0, 0] for t in TIERS}   # [correct, scored]
        for ans in answers:
            v = verdicts.get(ans["id"])
            if v is None or v["excluded"]:
                continue
            per_tier[ans["tier"]][1] += 1
            per_tier[ans["tier"]][0] += int(v["correct"])
        tot_c = sum(v[0] for v in per_tier.values())
        tot_n = sum(v[1] for v in per_tier.values())
        paths[name] = {
            "kl_mean": float(kl),
            "overall": {"correct": tot_c, "scored": tot_n,
                        "acc": tot_c / tot_n if tot_n else float("nan"),
                        "se": _wilson_se(tot_c, tot_n)},
            "tiers": {t: {"correct": c, "scored": n, "acc": c / n if n else float("nan"),
                          "se": _wilson_se(c, n)} for t, (c, n) in per_tier.items()},
        }

    # The reference is the KL==0 path, identified by its KL rather than by a name — for
    # gpt-oss it is the dequant path, and calling it "bf16" would imply an original that
    # does not exist.
    ref = paths.get("bf16") or next((p for p in paths.values() if p["kl_mean"] == 0), None)
    report = {"paths": paths, "prereg": "bench/K3-PREREG.md"}
    if ref is None:
        report["verdict"] = "no bf16 reference scored — cannot apply the kill switch"
    elif ref["overall"]["acc"] < KILL_SWITCH_OVERALL:
        report["verdict"] = (
            f"UNDER-POWERED (registered kill switch): bf16 scored "
            f"{ref['overall']['acc']:.1%} < {KILL_SWITCH_OVERALL:.0%}. Tier accuracies are "
            f"reported; NO KL-to-accuracy inference is drawn, per bench/K3-PREREG.md.")
        report["usable_tiers"] = []
    else:
        usable = [t for t in TIERS if ref["tiers"][t]["acc"] >= TIER_FLOOR]
        report["usable_tiers"] = usable
        report["verdict"] = (
            f"bf16 scored {ref['overall']['acc']:.1%} >= {KILL_SWITCH_OVERALL:.0%}; "
            f"tiers usable for directional statements: {usable or 'none'}")

    json.dump(report, open(a.out, "w"), indent=1)
    print(f"\n  prereg: bench/K3-PREREG.md  (registered before any of this existed)")
    hdr = f"  {'path':<6} {'KL':>10}  {'overall':>16}"
    print(hdr + "".join(f"{t:>14}" for t in TIERS))
    for name, p in paths.items():
        row = (f"  {name:<6} {p['kl_mean']:>10.4e}  "
               f"{p['overall']['acc']:>7.1%}±{p['overall']['se']:>5.1%}  ")
        for t in TIERS:
            row += f"{p['tiers'][t]['acc']:>8.1%}±{p['tiers'][t]['se']:<4.1%}"
        print(row)
    print(f"\n  {report['verdict']}")
    print(f"  receipt -> {a.out}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="K3: IKP accuracy against KL-measured paths")
    ap.add_argument("--phase", required=True, choices=["generate", "judge", "score"])
    ap.add_argument("--probes", default="ikp_clean.json")
    ap.add_argument("--model", default="ibm-granite/granite-3.0-1b-a400m-instruct")
    ap.add_argument("--path", choices=["bf16", "nf4", "fp4", "dqbf16", "mxfp4"])
    ap.add_argument("--answers")
    ap.add_argument("--judge-url", default="http://127.0.0.1:8781/v1/chat/completions")
    ap.add_argument("--judge-model", default="qwen")
    ap.add_argument("--judge-local", help="HF model id to judge with in-process")
    ap.add_argument("--score", nargs="*", default=[],
                    help="name:answers.json:verdicts.json:kl_mean")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    return {"generate": phase_generate, "judge": phase_judge, "score": phase_score}[a.phase](a)


if __name__ == "__main__":
    raise SystemExit(main())
