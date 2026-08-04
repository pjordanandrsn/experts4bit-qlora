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
                         decode_teacher_forced_logits, teacher_forced_logits)
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
                    help="comma list: nvme,hot,bf16nf4,bf16fp4")
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

    # ---------------------------------------------------------------- row: FP4 vs bf16
    # A second real quantization, so the K3 prereg has more than one nonzero x-value.
    # Measured BEFORE any accuracy is scored: the x-axis is fixed before the y-axis exists.
    if "bf16fp4" in want:
        t0 = time.time()
        from transformers import AutoModelForCausalLM
        base = AutoModelForCausalLM.from_pretrained(
            a.model, torch_dtype=torch.bfloat16, trust_remote_code=True).to(dev).eval()
        fp4, _ = load_moe_4bit_streaming(a.model, dev, torch.bfloat16, 8, 16,
                                         offload=False, prefetch=False, quant_type="fp4")
        fp4.eval()
        r = score_pair(lambda i: teacher_forced_logits(base, i),
                       lambda i: teacher_forced_logits(fp4, i), tok, a.max_len, dev)
        r.update({
            "row": "our FP4 quantization of a bf16 base",
            "reference": f"the bf16 base ({a.model}, torch_dtype=bfloat16)",
            "test": "experts4bit FP4 (load_moe_4bit_streaming, quant_type=fp4)",
            "expectation": "> 0 — a real quantization; same reference as the NF4 row so "
                           "the two are directly comparable",
            "wall_s": round(time.time() - t0, 1),
        })
        rows.append(r)
        del base, fp4
        torch.cuda.empty_cache() if dev == "cuda" else None

    # --------------------------------------------------------------- row: int8 vs bf16
    # A deliberately LOW-KL point on the same model, to break the size-vs-KL confound:
    # comparing gpt-oss (small KL, 21B) with granite (large KL, 1.3B) varied both at once.
    # Not an experts4bit path — bitsandbytes int8, declared as such wherever quoted.
    if "bf16int8" in want:
        t0 = time.time()
        from transformers import AutoModelForCausalLM, BitsAndBytesConfig
        base = AutoModelForCausalLM.from_pretrained(
            a.model, torch_dtype=torch.bfloat16, trust_remote_code=True).to(dev).eval()
        i8 = AutoModelForCausalLM.from_pretrained(
            a.model, quantization_config=BitsAndBytesConfig(load_in_8bit=True),
            torch_dtype=torch.bfloat16, device_map="auto").eval()
        r = score_pair(lambda i: teacher_forced_logits(base, i),
                       lambda i: teacher_forced_logits(i8, i), tok, a.max_len, dev)
        r.update({
            "row": "int8 (bitsandbytes) of a bf16 base — low-KL anchor",
            "reference": f"the bf16 base ({a.model}, torch_dtype=bfloat16)",
            "test": "bitsandbytes load_in_8bit (NOT an experts4bit path)",
            "expectation": "> 0 but far below the 4-bit rows — supplies the small-KL end "
                           "of a within-model sweep",
            "wall_s": round(time.time() - t0, 1),
        })
        rows.append(r)
        del base, i8
        torch.cuda.empty_cache() if dev == "cuda" else None

    # --------------------------------------- row: VRAM-resident vs host-streamed (hot)
    if "hot" in want:
        t0 = time.time()
        # enable_hot_residency is deprecated since 0.6.2; enable_pipelined_residency is
        # the supported path and takes k_slots (the model's routed top-k) explicitly.
        from experts4bit_qlora.hot_residency import (hot_residency_available,
                                                     target_modules)
        from experts4bit_qlora.pipelined import enable_pipelined_residency
        if not hot_residency_available():
            rows.append({
                "row": "tier transition: VRAM-resident vs host-streamed",
                "skipped": "hot residency unavailable (needs nf4_grouped + CUDA)",
            })
        else:
            resident, _ = load_moe_4bit_streaming(a.model, dev, torch.bfloat16, 8, 16,
                                                  offload=False, prefetch=False,
                                                  quant_type="nf4")
            resident.eval()
            split, _ = load_moe_4bit_streaming(a.model, dev, torch.bfloat16, 8, 16,
                                               offload=False, prefetch=False,
                                               quant_type="nf4")
            split.eval()
            cfg = split.config
            n_exp = getattr(cfg, "num_local_experts", None) or cfg.num_experts
            k_slots = getattr(cfg, "num_experts_per_tok", None) or cfg.num_experts_per_token
            n_layers = len(target_modules(split))

            # BOTH sides run the engine; only the HOT SET differs. That is the whole
            # design of this row, and the original version got it wrong: it compared a
            # model with no engine against the engine with half its experts streamed,
            # which varies two things at once — where the weight lives (the variable the
            # row names) and which kernel computes it (grouped 4-bit GEMM over a gathered
            # slot store vs ExpertsLoRA's per-expert GEMV decode path). Measured, that
            # conflated comparison gives 3.16e-3, and attributing it to residency would
            # have been wrong: with the kernel held fixed the answer is exactly 0.000.
            #
            # K is data, not a code path — enable_pipelined_residency's own docstring:
            # "pass a 0-length set for pure streaming, all experts for fully resident —
            # same code path" — so all-hot vs half-hot is free to construct and is the
            # clean experiment.
            n_ref = enable_pipelined_residency(
                resident, [list(range(n_exp)) for _ in range(n_layers)],
                device=dev, k_slots=int(k_slots))
            n_patched = enable_pipelined_residency(
                split, [list(range(n_exp // 2)) for _ in range(n_layers)],
                device=dev, k_slots=int(k_slots))
            if n_patched != n_layers or n_ref != n_layers:
                raise RuntimeError(
                    f"pipelined residency patched {n_patched}/{n_ref} of {n_layers} MoE "
                    "layers — an unpatched layer is silently identical to the reference")
            # DECODE-shaped scoring, and this row is the only one that uses it. The
            # engine serves T==1 forwards and hands a multi-token prefill straight back
            # to the reference path (see enable_pipelined_residency's docstring), so the
            # prefill scorer every other row uses would engage it exactly zero times and
            # report a flawless 0.000 that measures nothing. Both sides of the comparison
            # use the same scorer, so the pair stays apples-to-apples; the cost is that
            # this row's absolute KL is not directly comparable to the prefill rows'.
            r = score_pair(lambda i: decode_teacher_forced_logits(resident, i),
                           lambda i: decode_teacher_forced_logits(split, i),
                           tok, a.max_len, dev)
            # PROOF OF EXECUTION, and it is not optional here. This row's expectation is
            # exactly 0.000, so an engine that never ran satisfies it perfectly: the
            # unsplit reference IS what a dead patch returns. `enable_pipelined_residency`
            # reaches these modules through `ExpertsLoRA._delegate_to_base`, which is
            # conditional (eval + no_grad + zero adapter), so "patched" is not "ran".
            # The engine's own device-side fetch counters settle it.
            def _traffic(model):
                t = [m._pipelined.traffic() for m in target_modules(model)
                     if hasattr(m, "_pipelined")]
                return (sum(x["cold_pcie_bytes"] for x in t),
                        sum(x["hot_d2d_bytes"] for x in t))

            cold_ref, _ = _traffic(resident)
            cold, hot_d2d = _traffic(split)
            # The ASYMMETRY is the tier transition. Both numbers are load-bearing: the
            # test side must stream (or nothing was tiered) and the reference side must
            # not (or it was not actually all-resident, and the two sides are the same
            # experiment). Checking only one of them lets a 0.000 mean "nothing moved".
            if cold == 0:
                raise RuntimeError(
                    "pipelined residency moved zero cold-tier bytes: the engine never ran "
                    "(the wrapper did not delegate), so this row would report a 0.000 that "
                    "measures nothing. Check model.eval() and that the adapter is untrained.")
            if cold_ref != 0:
                raise RuntimeError(
                    f"the all-resident reference streamed {cold_ref} cold bytes — it is not "
                    "fully resident, so this row is not isolating weight location")
            r.update({
                "row": "tier transition: VRAM-resident vs host-streamed",
                "reference": f"pipelined engine, hot_set = ALL {n_exp} experts "
                             f"(nothing streamed), same model, same run",
                "test": f"pipelined engine, hot_set = {n_exp // 2}/{n_exp} experts pinned "
                        f"in VRAM, remainder streamed from host DRAM "
                        f"({n_patched} layers patched)",
                "scoring": "decode-shaped teacher forcing (one token per forward, KV "
                           "cache) on BOTH sides — not the prefill scorer the other rows "
                           "use; this engine only engages at T==1",
                "expectation": "exactly 0.000 — CONFIRMATORY; where a weight lives must "
                               "not change the arithmetic performed on it. Same bytes, "
                               "same GEMM, reached via a different source address: the "
                               "engine holds the kernel fixed and takes K as data, so "
                               "only location varies between these two sides.",
                "amendment": "Twice-amended, both disclosed. (1) The row could not run at "
                             "all — enable_pipelined_residency refused ExpertsLoRA bases, "
                             "which is every model load_moe_4bit_streaming returns. (2) It "
                             "then scored with the prefill scorer, which engages this "
                             "decode-only engine zero times and would have reported a "
                             "flawless 0.000 that measured nothing; scoring is now "
                             "decode-shaped. A third change was considered and REJECTED on "
                             "evidence: the 0.000 expectation was briefly withdrawn after "
                             "the original reference side (a model with NO engine) measured "
                             "3.16e-3. That comparison varied weight location AND compute "
                             "kernel together, so it could not test the row's claim. With "
                             "the reference moved onto the engine at hot_set=ALL, the "
                             "kernel is held fixed, and the registered 0.000 is met exactly "
                             "— see granite-hot-kernel-vs-reference.json for the 3.16e-3, "
                             "which is the KERNEL's cost and is a separate finding.",
                "engine_cold_pcie_bytes": cold,
                "engine_hot_d2d_bytes": hot_d2d,
                "engine_cold_pcie_bytes_reference": cold_ref,
                "witness": "the ASYMMETRY is the measurement: the test side must stream "
                           "(nonzero cold bytes — otherwise nothing was tiered and the "
                           "engine may not even have run) and the reference side must not "
                           "(zero — otherwise it is not all-resident and both sides are "
                           "the same experiment). A 0.000 is only meaningful with both.",
                "wall_s": round(time.time() - t0, 1),
            })
            rows.append(r)
            del resident, split
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
        if "skipped" in r:
            print(f"   SKIPPED   : {r['skipped']}")
            continue
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
