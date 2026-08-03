"""kl_gptoss.py — the two K2 rows that need a natively-4-bit model.

gpt-oss ships AS MXFP4. There is no bf16 original, so the reference for BOTH rows is the
DEQUANT path of the same shipped bytes — never an imagined bf16 model. Every printed row
says so, because this is the cell most likely to be misread.

  row A  native MXFP4 on the shipped bytes  vs  dequant-to-bf16 of those same bytes
  row B  MXFP4 -> NF4 requant (e4b)         vs  dequant-to-bf16 of those same bytes

Row B is the "requant tax": what a stack pays for converting a shipped MXFP4 checkpoint
into its own 4-bit format instead of serving the original bytes.

Gated on a K0 control receipt produced ON THIS HOST — the instrument is validated per
machine, not once globally.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kl_fidelity import METRIC_VERSION, teacher_forced_logits  # noqa: E402
from kl_paths import _gate_on_k0, score_pair  # noqa: E402
from kl_prompts import PROMPTS, digest as prompt_digest, strata_counts  # noqa: E402

MID = "openai/gpt-oss-20b"


def load_dequant_bf16(mid: str, dev: str):
    """The reference: the shipped MXFP4 bytes dequantized to bf16."""
    from transformers import AutoModelForCausalLM, Mxfp4Config
    m = AutoModelForCausalLM.from_pretrained(
        mid, quantization_config=Mxfp4Config(dequantize=True),
        torch_dtype=torch.bfloat16, device_map=dev)
    return m.eval()


def load_native_mxfp4(mid: str, dev: str):
    """The shipped bytes, computed on as MXFP4 (no dequant to a wider dtype).

    GUARDED. Without the `kernels` package transformers silently falls back to
    "dequantizing the model to bf16" and returns a perfectly working model. That model
    would score KL = 0 against the bf16 reference and I would have reported native MXFP4
    as distributionally identical to its own dequant — a fabricated headline result caused
    by a missing pip package. So the fallback is detected and raised, not tolerated.
    """
    from transformers import AutoModelForCausalLM
    m = AutoModelForCausalLM.from_pretrained(mid, dtype="auto", device_map="auto")
    qc = getattr(m.config, "quantization_config", None)
    method = getattr(qc, "quant_method", None) or (qc or {}).get("quant_method")
    if qc is None or "mxfp4" not in str(method).lower():
        raise RuntimeError(
            f"refusing to call this native MXFP4: quantization_config={qc!r}. "
            "transformers dequantized to bf16 (usually a missing `kernels` package), "
            "which would make this row a bf16-vs-bf16 comparison reported as MXFP4.")
    if getattr(qc, "dequantize", False):
        raise RuntimeError("quantization_config has dequantize=True — not the native path")
    return m.eval()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MID)
    ap.add_argument("--k0-receipt", required=True)
    ap.add_argument("--rows", default="mxfp4,requant")
    ap.add_argument("--max-len", type=int, default=320)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    k0 = _gate_on_k0(a.k0_receipt)
    if a.limit:
        del PROMPTS[a.limit:]

    from transformers import AutoTokenizer
    dev = "cuda"
    tok = AutoTokenizer.from_pretrained(a.model)
    want = {r.strip() for r in a.rows.split(",") if r.strip()}
    rows, not_measured = [], {}

    # One row per process: a row that raises AFTER loading its model leaks ~13-42 GB, and
    # the next row then OOMs for reasons that have nothing to do with the next row.
    if len(want) > 1:
        raise SystemExit("run one row per process (--rows mxfp4 | --rows requant): a failed "
                         "row leaks its model and OOMs the next one")

    ref = load_dequant_bf16(a.model, dev)
    REF_NAME = ("dequant-to-bf16 of the SAME shipped MXFP4 bytes "
                "(Mxfp4Config(dequantize=True)) — there is no bf16 original of gpt-oss")

    if "mxfp4" in want:
        t0 = time.time()
        try:
            nat = load_native_mxfp4(a.model, dev)
            r = score_pair(lambda i: teacher_forced_logits(ref, i),
                           lambda i: teacher_forced_logits(nat, i), tok, a.max_len, dev)
            r.update({
                "row": "native MXFP4 computed on the shipped bytes",
                "reference": REF_NAME,
                "test": "native MXFP4 (transformers, no dequant)",
                "expectation": "measure — this is the format the model actually ships in",
                "wall_s": round(time.time() - t0, 1),
            })
            rows.append(r)
            del nat
            torch.cuda.empty_cache()
        except Exception as e:  # honest failure beats a fabricated row
            not_measured["native MXFP4 vs dequant-bf16"] = (
                f"{type(e).__name__}: {str(e)[:300]}")

    if "requant" in want:
        t0 = time.time()
        try:
            from experts4bit_qlora.loader import load_moe_4bit_streaming
            nf4, _ = load_moe_4bit_streaming(a.model, dev, torch.bfloat16, 8, 16,
                                             offload=False, prefetch=False,
                                             quant_type="nf4")
            nf4.eval()
            r = score_pair(lambda i: teacher_forced_logits(ref, i),
                           lambda i: teacher_forced_logits(nf4, i), tok, a.max_len, dev)
            r.update({
                "row": "MXFP4 -> NF4 requant (the requant tax)",
                "reference": REF_NAME,
                "test": "experts4bit NF4 requantized from the shipped MXFP4",
                "expectation": "measure — the cost of converting a shipped MXFP4 "
                               "checkpoint into another 4-bit format",
                "wall_s": round(time.time() - t0, 1),
            })
            rows.append(r)
            del nf4
            torch.cuda.empty_cache()
        except Exception as e:
            not_measured["MXFP4->NF4 requant vs dequant-bf16"] = (
                f"{type(e).__name__}: {str(e)[:300]}")

    receipt = {
        "metric_version": METRIC_VERSION,
        "k0_gate": {"receipt": a.k0_receipt, "all_passed": k0["all_passed"]},
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(0),
        "model": a.model,
        "reference_note": REF_NAME,
        "prompt_set": {"sha256": prompt_digest(), "n": len(PROMPTS),
                       "strata": strata_counts(), "max_len": a.max_len},
        "rows": rows,
        "not_measured": not_measured,
    }
    json.dump(receipt, open(a.out, "w"), indent=1)
    for r in rows:
        print(f"\n== {r['row']}")
        print(f"   reference : {r['reference']}")
        print(f"   test      : {r['test']}")
        print(f"   KL mean={r['kl_mean']:.6e} median={r['kl_median']:.6e} "
              f"p95={r['kl_p95']:.6e}")
        print(f"   exactly_zero={r['exactly_zero']}  top1={r['top1_agreement']:.6f}  "
              f"tokens={r['n_tokens_scored']}  vocab={r['vocab_size']}")
        for s, v in r["per_stratum"].items():
            print(f"     {s:>10}: mean={v['kl_mean']:.3e} tok={v['n_tokens_scored']}")
    for k, v in not_measured.items():
        print(f"\n== NOT MEASURED: {k}\n   {v}")
    print(f"\nreceipt -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
