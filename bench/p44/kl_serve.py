#!/usr/bin/env python3
"""bench/p44/kl_serve.py -- P44-b: KL-from-bf16 for e4b's SERVING stacks (the arms bo7 timed), per stratum.

GATED ON K0 (`kl_paths._gate_on_k0`): no KL row without a passing control receipt produced ON THIS HOST.

EVERY ROW NAMES ITS REFERENCE (kl_paths.py's rule). Gemma-4's reference is the bf16 checkpoint; gpt-oss ships AS
MXFP4, so its reference is the DEQUANT path of the same shipped bytes (`Mxfp4Config(dequantize=True)`), never an
imagined bf16 model.

THE SCORER IS DECODE-SHAPED ON BOTH SIDES (`kl_fidelity.decode_teacher_forced_logits`: one token per forward, KV
cache, position i conditioned on 0..i). The arms are SERVING stacks whose levers engage only at T == 1 -- the int4
attention GEMV (`Int4Linear.GEMV_ROWS_MAX = 1`), the glue folds and the router epilogue ("off decode shapes every
patch falls through to the original chain"), the MXFP4 store's single-row GEMV. Scored with the prefill forward,
every one of them would report a flawless 0.000 that means "the lever never ran" (the trap kl_paths.py's hot row
documents). Same scorer on the reference side, so the pair stays apples-to-apples; the cost is that these absolute
KLs are not directly comparable to the prefill rows in KL-FINDINGS.md. `--scorer prefill` exists for a debugging
control and is RECORDED on every row.

THE REFERENCE IS SCORED ONCE AND FREED. Its per-prompt logits are cached to disk (`--ref-cache`, the model's own
output dtype, lossless), then the reference model is deleted BEFORE any arm is built, so a 52 GB bf16 Gemma-4 and a
served arm never share the card. Arms run one at a time; a failed arm is a `not_measured` entry with its error, and
the receipt is rewritten after every arm so a crash loses nothing already measured.

PROOF OF EXECUTION per arm: `serve_stack.build_served_model` returns a census (Int4Linear count, int4 expert layers
and store kinds, modules each fusion patched); a set lever with a zero count REFUSES the row -- a KL against a stack
whose lever silently did not apply is the number this lane must never publish.

The READING RULE (P44-PREREG.md) is applied by `p44_reduce.py`, not here; this file measures.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
import traceback

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
for cand in (HERE, os.path.join(HERE, "..")):              # staged flat on the box; bench/ in the repo
    if os.path.exists(os.path.join(cand, "kl_fidelity.py")):
        sys.path.insert(0, cand)
        break
from kl_fidelity import (METRIC_VERSION, KLAccumulator,  # noqa: E402
                         decode_teacher_forced_logits, teacher_forced_logits)
from kl_paths import _gate_on_k0  # noqa: E402
from kl_prompts import PROMPTS, digest as prompt_digest, strata_counts  # noqa: E402
from serve_stack import ARMS, MODELS, apply_env, arm_env, build_served_model  # noqa: E402

REFERENCE = {
    "gemma4": "the bf16 checkpoint (AutoModelForCausalLM, dtype=bfloat16) at the pinned revision, resident on the same card",
    "gptoss": "dequant-to-bf16 of the SAME shipped MXFP4 bytes (Mxfp4Config(dequantize=True)) -- there is no bf16 "
              "original of gpt-oss (kl_paths.py's rule)",
}
SCORERS = {"decode": decode_teacher_forced_logits, "prefill": teacher_forced_logits}


def load_reference(family: str, model_id: str, revision: str, dev: str):
    from transformers import AutoModelForCausalLM
    if family == "gptoss":
        from transformers import Mxfp4Config
        m = AutoModelForCausalLM.from_pretrained(model_id, revision=revision, dtype=torch.bfloat16, device_map=dev,
                                                 quantization_config=Mxfp4Config(dequantize=True))
        qc = getattr(m.config, "quantization_config", None)
        deq = getattr(qc, "dequantize", None) if qc is not None else None
        if deq is not True and not (isinstance(qc, dict) and qc.get("dequantize") is True):
            raise RuntimeError(f"gpt-oss reference is not the dequant path: quantization_config={qc!r}")
    else:
        m = AutoModelForCausalLM.from_pretrained(model_id, revision=revision, dtype=torch.bfloat16, device_map=dev)
    m.config.use_cache = True
    return m.eval()


def _tokenize(tok, text: str, max_len: int):
    return tok(text, return_tensors="pt", truncation=True, max_length=max_len)["input_ids"]


def reference_pass(family, model_id, revision, tok, prompts, cache_dir, max_len, score, dev) -> dict:
    """Fill `cache_dir/<prompt id>.pt` for every prompt that lacks one; load the reference only if needed; free it."""
    os.makedirs(cache_dir, exist_ok=True)
    missing = [p for p in prompts if not os.path.exists(os.path.join(cache_dir, p["id"] + ".pt"))]
    rep = {"cache_dir": cache_dir, "n_cached_before": len(prompts) - len(missing), "n_computed": len(missing)}
    if missing:
        t0 = time.time()
        ref = load_reference(family, model_id, revision, dev)
        rep["load_s"] = round(time.time() - t0, 1)
        rep["reference_class"] = type(ref).__name__
        rep["reference_dtype"] = str(next(ref.parameters()).dtype)
        t1 = time.time()
        with torch.no_grad():
            for i, p in enumerate(missing):
                ids = _tokenize(tok, p["text"], max_len).to(dev)
                logits = score(ref, ids)
                torch.save({"logits": logits.detach().cpu(), "n": int(ids.numel()), "id": p["id"]},
                           os.path.join(cache_dir, p["id"] + ".pt"))
                if i % 20 == 0:
                    print(f"  reference {i + 1}/{len(missing)} ({time.time() - t1:.0f}s)", flush=True)
        rep["score_s"] = round(time.time() - t1, 1)
        rep["logits_dtype"] = str(logits.dtype)
        del ref
        gc.collect()
        torch.cuda.empty_cache()
    return rep


def _lever_check(env: dict, info: dict) -> None:
    """A set lever must have engaged. Zero counts under a set flag refuse the row."""
    if env.get("E4B_SERVE_EXP_INT4") == "1" and info.get("int4_expert_layers", 0) == 0:
        raise RuntimeError("E4B_SERVE_EXP_INT4=1 but no expert layer carries an int4 store -- the hook did not apply")
    if env.get("E4B_SERVE_ATTN_INT4_CALIB") == "1" and info.get("int4_attn_projections", 0) == 0:
        raise RuntimeError("E4B_SERVE_ATTN_INT4_CALIB=1 but no attention projection is Int4Linear -- the hook did not apply")
    for flag, key in (("E4B_FUSE_T1_GLUE", "fuse_t1_glue_n"), ("E4B_FUSE_T1_GLUE_R2", "fuse_t1_glue_r2_n"),
                      ("E4B_FUSE_ROUTER_EPI", "fuse_router_epilogue_n")):
        if env.get(flag) == "1" and not info.get(key):
            raise RuntimeError(f"{flag}=1 but the fusion patched nothing")


def score_arm(model, tok, prompts, cache_dir, max_len, score, dev) -> dict:
    pooled = KLAccumulator()
    per_stratum: dict[str, KLAccumulator] = {}
    tokens = []
    with torch.no_grad():
        for i, p in enumerate(prompts):
            ids = _tokenize(tok, p["text"], max_len).to(dev)
            ref = torch.load(os.path.join(cache_dir, p["id"] + ".pt"))
            if ref["n"] != int(ids.numel()):
                raise RuntimeError(f"{p['id']}: reference scored {ref['n']} tokens, arm has {int(ids.numel())} -- tokenizer drift")
            ref_logits = ref["logits"].to(dev)
            test = score(model, ids)
            pooled.add(ref_logits, test)
            per_stratum.setdefault(p["stratum"], KLAccumulator()).add(ref_logits, test)
            tokens.append(int(ids.numel()))
            if i % 20 == 0:
                print(f"  arm {i + 1}/{len(prompts)}", flush=True)
    out = pooled.summary()
    out["per_stratum"] = {k: v.summary() for k, v in per_stratum.items()}
    out["tokens_min"], out["tokens_max"], out["tokens_total"] = min(tokens), max(tokens), sum(tokens)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="P44-b: KL-from-bf16 for the served arms, per stratum")
    ap.add_argument("--family", required=True, choices=sorted(REFERENCE))
    ap.add_argument("--model", default=None)
    ap.add_argument("--revision", default=None)
    ap.add_argument("--arena", required=True)
    ap.add_argument("--calib", required=True, help="placement calibration json (P39's calib.json)")
    ap.add_argument("--k0-receipt", required=True)
    ap.add_argument("--arms", default=None, help="comma list; default: every registered arm of the family, in order")
    ap.add_argument("--max-len", type=int, default=320)
    ap.add_argument("--limit", type=int, default=0, help="debug: first N prompts only (RECORDED)")
    ap.add_argument("--ref-cache", required=True)
    ap.add_argument("--scorer", default="decode", choices=sorted(SCORERS))
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    k0 = _gate_on_k0(a.k0_receipt)
    model_id, revision = MODELS[a.family]
    model_id = a.model or model_id
    revision = a.revision or revision
    arms = [x.strip() for x in (a.arms or ",".join(ARMS[a.family])).split(",") if x.strip()]
    for arm in arms:
        if arm not in ARMS[a.family]:
            raise SystemExit(f"unregistered arm {arm!r} for {a.family}; registered: {list(ARMS[a.family])}")
    prompts = PROMPTS[:a.limit] if a.limit else PROMPTS
    dev = "cuda"
    score = SCORERS[a.scorer]
    try:
        import usercustomize  # noqa: F401  (the staged lane hook; the lever census below is the real check)
        hook = getattr(usercustomize, "__file__", "?")
    except ImportError:
        hook = None

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_id, revision=revision)

    receipt = {
        "lane": "P44-b", "metric_version": METRIC_VERSION,
        "k0_gate": {"receipt": a.k0_receipt, "all_passed": k0["all_passed"], "torch_at_control": k0.get("torch")},
        "torch": torch.__version__, "device": dev,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "family": a.family, "model": model_id, "revision": revision,
        "reference": REFERENCE[a.family],
        "scorer": {"name": a.scorer, "both_sides": True,
                   "note": "decode-shaped teacher forcing on BOTH sides (one token per forward, KV cache): the "
                           "serving levers engage only at T == 1" if a.scorer == "decode" else
                           "PREFILL scorer -- a debugging control; serving levers that engage only at T == 1 "
                           "report 0.000 here and that means 'never ran', not 'faithful'"},
        "prompt_set": {"sha256": prompt_digest(), "n_registered": len(PROMPTS), "n_scored": len(prompts),
                       "limit": a.limit, "strata": strata_counts(), "max_len": a.max_len},
        "aggregation": "token-weighted (see kl_prompts docstring: longctx ~58% of tokens)",
        "hook": hook, "arms_requested": arms, "rows": [], "not_measured": {},
    }

    def flush():
        with open(a.out, "w") as f:
            json.dump(receipt, f, indent=1)

    flush()
    t0 = time.time()
    receipt["reference_pass"] = reference_pass(a.family, model_id, revision, tok, prompts, a.ref_cache,
                                               a.max_len, score, dev)
    receipt["reference_pass"]["wall_s"] = round(time.time() - t0, 1)
    flush()

    for arm in arms:
        env = arm_env(a.family, arm, model_id)
        apply_env(env)
        t1 = time.time()
        model = None
        try:
            model, info = build_served_model(model_id, a.arena, a.calib, device=dev)
            _lever_check(env, info)
            r = score_arm(model, tok, prompts, a.ref_cache, a.max_len, score, dev)
            r.update({"arm": arm, "row": f"{a.family}/{arm} vs the family's reference", "env": env,
                      "engagement": info, "reference": REFERENCE[a.family], "scorer": a.scorer,
                      "wall_s": round(time.time() - t1, 1)})
            receipt["rows"].append(r)
            print(f"== {a.family}/{arm}: KL mean={r['kl_mean']:.6e} p95={r['kl_p95']:.6e} top1={r['top1_agreement']:.5f} "
                  f"tokens={r['n_tokens_scored']} ({r['wall_s']} s)", flush=True)
            for s, v in r["per_stratum"].items():
                print(f"     {s:>10}: mean={v['kl_mean']:.3e} tok={v['n_tokens_scored']}", flush=True)
        except Exception as e:  # an honest hole beats a fabricated row
            receipt["not_measured"][arm] = {"error": f"{type(e).__name__}: {str(e)[:600]}",
                                            "traceback_tail": traceback.format_exc()[-1500:],
                                            "wall_s": round(time.time() - t1, 1)}
            print(f"== {a.family}/{arm}: NOT MEASURED -- {type(e).__name__}: {str(e)[:300]}", flush=True)
        finally:
            del model
            gc.collect()
            torch.cuda.empty_cache()
            flush()
    receipt["wall_s_total"] = round(time.time() - t0, 1)
    flush()
    print(f"receipt -> {a.out} ({len(receipt['rows'])} rows, {len(receipt['not_measured'])} not measured)")
    return 0 if receipt["rows"] and not receipt["not_measured"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
