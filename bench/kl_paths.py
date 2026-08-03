"""kl_paths.py — K2: measure KL-from-reference for each serving path.

GATED ON K0. Run bench/kl_fidelity.py --controls first; this driver refuses to measure
unless a passing control receipt is supplied, because a fidelity number from an
unvalidated instrument is worse than no number (it looks authoritative).

EVERY ROW NAMES ITS REFERENCE. A KL with an unstated reference is meaningless, and the
native-4-bit rows are the ones most likely to be misread: gpt-oss and K3 ship AS 4-bit,
there is no bf16 original, so their reference is the DEQUANT path of the same shipped
bytes — never an imagined bf16 model.

ON THE ZERO ROWS: `tests/test_nvme_residency_equivalence.py` already proves bit-identity
with `torch.equal`. KL of identical tensors is trivially 0, so these rows are
CONFIRMATORY, not novel. Their value is putting identity on the same axis as everyone
else's degradation — not discovering anything. Say that wherever they are quoted.

SCOPE: fused-MoE NF4/MXFP4 on this stack. The motivating correlation (Quesma 2026-08-03,
r=-0.981 between mean KL and IKP accuracy) is over dense GGUF k-quants and is THEIRS
until K3 tests it on our formats.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kl_fidelity import (KLAccumulator, METRIC_VERSION,  # noqa: E402
                         teacher_forced_logits)
from kl_prompts import PROMPTS, digest as prompt_digest, strata_counts  # noqa: E402


def _gate_on_k0(receipt_path: str) -> dict:
    """Refuse to measure without a PASSING K0 control receipt."""
    if not os.path.exists(receipt_path):
        raise SystemExit(f"K0 gate: no control receipt at {receipt_path}. "
                         "Run `python bench/kl_fidelity.py --controls --out <path>` first.")
    r = json.load(open(receipt_path))
    if not r.get("all_passed"):
        raise SystemExit("K0 gate: control receipt reports a FAILING control — refusing "
                         "to measure. Fix the instrument before measuring any path.")
    if r.get("metric_version") != METRIC_VERSION:
        raise SystemExit(f"K0 gate: receipt metric {r.get('metric_version')} != "
                         f"instrument {METRIC_VERSION}; numbers would be incomparable.")
    return r


def _tokenize(tok, text: str, max_len: int):
    ids = tok(text, return_tensors="pt", truncation=True, max_length=max_len)["input_ids"]
    return ids


def score_pair(ref_fwd, test_fwd, tok, max_len: int, device) -> dict:
    """Token-weighted KL over the committed prompt set, plus per-stratum breakdown.

    Per-stratum is REQUIRED, not decorative: aggregation is token-weighted and the
    long-context stratum is ~58% of scored tokens, so a pooled mean alone would hide a
    format that is clean on short prompts and degrades over distance (or the reverse).
    """
    pooled = KLAccumulator()
    per_stratum: dict[str, KLAccumulator] = {}
    tokens_per_prompt: list[int] = []
    for p in PROMPTS:
        ids = _tokenize(tok, p["text"], max_len).to(device)
        tokens_per_prompt.append(int(ids.numel()))
        ref = ref_fwd(ids)
        test = test_fwd(ids)
        pooled.add(ref, test)
        per_stratum.setdefault(p["stratum"], KLAccumulator()).add(ref, test)
    out = pooled.summary()
    out["per_stratum"] = {k: v.summary() for k, v in per_stratum.items()}
    out["tokens_min"] = min(tokens_per_prompt)
    out["tokens_max"] = max(tokens_per_prompt)
    out["tokens_total"] = sum(tokens_per_prompt)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="K2 path table (KL from named references)")
    ap.add_argument("--model", default="allenai/OLMoE-1B-7B-0924")
    ap.add_argument("--k0-receipt", required=True)
    ap.add_argument("--rows", default="nvme,bf16nf4",
                    help="comma list: nvme,hot,bf16nf4")
    ap.add_argument("--max-len", type=int, default=320)
    ap.add_argument("--limit", type=int, default=0, help="debug: first N prompts only")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    k0 = _gate_on_k0(a.k0_receipt)
    if a.limit:
        del PROMPTS[a.limit:]

    from transformers import AutoTokenizer

    from experts4bit_qlora.loader import load_moe_4bit_streaming

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(a.model)
    rows: list[dict] = []
    want = {r.strip() for r in a.rows.split(",") if r.strip()}

    # ---------------------------------------------------------------- row: NF4 vs bf16
    if "bf16nf4" in want:
        t0 = time.time()
        from transformers import AutoModelForCausalLM
        base = AutoModelForCausalLM.from_pretrained(
            a.model, torch_dtype=torch.bfloat16, trust_remote_code=True).to(dev).eval()
        nf4, _ = load_moe_4bit_streaming(a.model, dev, torch.bfloat16, 8, 16,
                                         offload=False, prefetch=False, quant_type="nf4")
        nf4.eval()
        r = score_pair(lambda i: teacher_forced_logits(base, i),
                       lambda i: teacher_forced_logits(nf4, i), tok, a.max_len, dev)
        r.update({
            "row": "our NF4 quantization of a bf16 base",
            "reference": f"the bf16 base ({a.model}, torch_dtype=bfloat16)",
            "test": "experts4bit NF4 (load_moe_4bit_streaming, quant_type=nf4)",
            "expectation": "> 0 — a real quantization, measured not assumed",
            "wall_s": round(time.time() - t0, 1),
        })
        rows.append(r)
        del base, nf4
        torch.cuda.empty_cache() if dev == "cuda" else None

    # ------------------------------------------- row: DRAM-resident vs NVMe-streamed
    if "nvme" in want:
        t0 = time.time()
        resident, _ = load_moe_4bit_streaming(a.model, dev, torch.bfloat16, 8, 16,
                                              offload=False, prefetch=False,
                                              quant_type="nf4")
        resident.eval()
        streamed, _ = load_moe_4bit_streaming(a.model, dev, torch.bfloat16, 8, 16,
                                              offload=True, prefetch=True,
                                              quant_type="nf4")
        streamed.eval()
        r = score_pair(lambda i: teacher_forced_logits(resident, i),
                       lambda i: teacher_forced_logits(streamed, i), tok, a.max_len, dev)
        r.update({
            "row": "tier transition: DRAM-resident vs streamed",
            "reference": "DRAM-resident (offload=False), same model, same run",
            "test": "streamed cold tier (offload=True, prefetch=True)",
            "expectation": "exactly 0.000 — CONFIRMATORY; torch.equal already proved "
                           "bit-identity in tests/test_nvme_residency_equivalence.py",
            "wall_s": round(time.time() - t0, 1),
        })
        rows.append(r)
        del resident, streamed
        torch.cuda.empty_cache() if dev == "cuda" else None

    receipt = {
        "metric_version": METRIC_VERSION,
        "k0_gate": {"receipt": a.k0_receipt, "all_passed": k0["all_passed"],
                    "torch_at_control": k0.get("torch")},
        "torch": torch.__version__,
        "device": dev,
        "model": a.model,
        "prompt_set": {"sha256": prompt_digest(), "n": len(PROMPTS),
                       "strata": strata_counts(), "max_len": a.max_len},
        "aggregation": "token-weighted (see kl_prompts docstring: longctx ~58% of tokens)",
        "rows": rows,
        "not_measured": {
            "native MXFP4 vs dequant-to-bf16 of the same shipped bytes": "gpt-oss not in "
                "the offline hf-cache on this host; needs a download. Reference would be "
                "the DEQUANT path of the shipped bytes — there is no bf16 original.",
            "MXFP4 -> NF4 requant (the 'requant tax')": "same blocker; reference would be "
                "dequant-to-bf16 of the shipped MXFP4.",
        },
    }
    with open(a.out, "w") as f:
        json.dump(receipt, f, indent=1)
    for r in rows:
        print(f"\n== {r['row']}")
        print(f"   reference : {r['reference']}")
        print(f"   test      : {r['test']}")
        print(f"   expected  : {r['expectation']}")
        print(f"   KL mean={r['kl_mean']:.6e} median={r['kl_median']:.6e} "
              f"p95={r['kl_p95']:.6e} max={r['kl_max_per_token']:.6e}")
        print(f"   exactly_zero={r['exactly_zero']}  top1={r['top1_agreement']:.6f}  "
              f"tokens={r['n_tokens_scored']}  vocab={r['vocab_size']}")
        for s, v in r["per_stratum"].items():
            print(f"     {s:>10}: mean={v['kl_mean']:.3e} tok={v['n_tokens_scored']}")
    print(f"\nreceipt -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
