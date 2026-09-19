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
from serve_stack import ARMS, LANE_KEYS, MODELS, arm_env, build_arm_model, control_arm  # noqa: E402

REFERENCE = {
    "gemma4": "the bf16 checkpoint (AutoModelForCausalLM, dtype=bfloat16) at the pinned revision, resident on the same card",
    "gptoss": "dequant-to-bf16 of the SAME shipped MXFP4 bytes (Mxfp4Config(dequantize=True)) -- there is no bf16 "
              "original of gpt-oss (kl_paths.py's rule)",
    "gemma4diag": "the bf16 checkpoint (AutoModelForCausalLM, dtype=bfloat16) at the pinned revision, resident on the same "
                  "card -- P44-b's Gemma-4 reference, reused unchanged for the P47 builders (#597)",
    "gemma4layer": "the bf16 checkpoint (AutoModelForCausalLM, dtype=bfloat16) at the pinned revision -- P44-b's Gemma-4 "
                   "reference, reused unchanged for the P48 one-layer-at-a-time builders (#597)",
}
SCORERS = {"decode": decode_teacher_forced_logits, "prefill": teacher_forced_logits}
SELF_CONSISTENCY_MAX = 1e-2      # amendment 5: the reference must agree with itself decode-vs-prefill or the decode scorer is refused


def load_reference(family: str, model_id: str, revision: str, dev: str):
    from transformers import AutoModelForCausalLM
    if family == "gptoss":
        from transformers import Mxfp4Config
        m = AutoModelForCausalLM.from_pretrained(model_id, revision=revision, dtype=torch.bfloat16, device_map=dev,
                                                 quantization_config=Mxfp4Config(dequantize=True))
        # transformers 5.x DROPS quantization_config from the config once it has dequantised, so the proof is the
        # weights: no packed byte tensor may remain (run 2 refused a correct dequant on the missing config).
        low = [n for n, t in list(m.named_parameters()) + list(m.named_buffers())
               if t.dtype in (torch.uint8, torch.int8) or "blocks" in n.split(".")[-1] and t.dtype not in (torch.bfloat16, torch.float32, torch.float16)]
        if low:
            raise RuntimeError(f"gpt-oss reference still carries packed/quantised tensors ({len(low)}: {low[:3]}) -- not the dequant path")
        for n, t in m.named_parameters():
            if "experts" in n and t.dtype != torch.bfloat16:
                raise RuntimeError(f"gpt-oss reference expert tensor {n} is {t.dtype}, expected bf16 after dequantize=True")
    else:
        m = AutoModelForCausalLM.from_pretrained(model_id, revision=revision, dtype=torch.bfloat16, device_map=dev)
    m.config.use_cache = True
    return m.eval()


def _tokenize(tok, text: str, max_len: int):
    return tok(text, return_tensors="pt", truncation=True, max_length=max_len)["input_ids"]


def self_consistency(ref, tok, prompts, max_len, dev, n: int) -> dict:
    """Control (i), amendment 2/5: the REFERENCE scored decode-shaped vs prefill-shaped on the same prompts. An HF-side
    cache/positions fault shows here; over SELF_CONSISTENCY_MAX the decode scorer is refused for the family."""
    acc = KLAccumulator()
    t0 = time.time()
    with torch.no_grad():
        for p in prompts[:n]:
            ids = _tokenize(tok, p["text"], max_len).to(dev)
            acc.add(decode_teacher_forced_logits(ref, ids), teacher_forced_logits(ref, ids))
    s_ = acc.summary()
    return {"n_prompts": min(n, len(prompts)), "kl_mean": s_["kl_mean"], "kl_max": s_["kl_max_per_token"], "top1": s_["top1_agreement"],
            "tokens": s_["n_tokens_scored"], "threshold": SELF_CONSISTENCY_MAX, "passes": s_["kl_mean"] < SELF_CONSISTENCY_MAX,
            "wall_s": round(time.time() - t0, 1)}


def reference_pass(family, model_id, revision, tok, prompts, cache_dir, max_len, scorer_name, dev, controls_n) -> tuple:
    """Fill `cache_dir/<prompt id>.pt` for every prompt that lacks one; load the reference only if needed; free it.
    With ``scorer_name == "auto"`` the self-consistency control runs FIRST on the loaded reference and picks the scorer:
    decode when the reference agrees with itself, prefill when it does not (P44-b run 3: Gemma-4's reference read
    0.279 nats against itself under decode, so every decode row of that run was the scorer, not the model).
    Returns ``(report, scorer_name_used)``; the cache dir is suffixed by the scorer so the two never mix."""
    chosen = scorer_name
    rep = {"requested_scorer": scorer_name}
    ref = None
    if scorer_name == "auto":
        t0 = time.time()
        ref = load_reference(family, model_id, revision, dev)
        rep["load_s"] = round(time.time() - t0, 1)
        rep["self_consistency"] = self_consistency(ref, tok, prompts, max_len, dev, controls_n)
        chosen = "decode" if rep["self_consistency"]["passes"] else "prefill"
        rep["scorer_selected_by_control"] = chosen
        print(f"  control (i) reference decode-vs-prefill: KL {rep['self_consistency']['kl_mean']:.4e} -> scorer {chosen}", flush=True)
    cache_dir = f"{cache_dir}.{chosen}"
    score = SCORERS[chosen]
    os.makedirs(cache_dir, exist_ok=True)
    missing = [p for p in prompts if not os.path.exists(os.path.join(cache_dir, p["id"] + ".pt"))]
    rep.update({"cache_dir": cache_dir, "scorer": chosen, "n_cached_before": len(prompts) - len(missing), "n_computed": len(missing)})
    if missing:
        t0 = time.time()
        if ref is None:
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
    if ref is not None:
        del ref
        gc.collect()
        torch.cuda.empty_cache()
    return rep, chosen


def _builder_check(info: dict) -> None:
    """P47: a loader builder's layer set must have applied -- the quantised / unquantised stack counts `verify_moe_4bit`
    reports must equal what the builder asked for, or the row is refused (a half that did not quantise is not a half)."""
    if info.get("builder", "served") == "served":
        return
    for got, want in (("n_quantized", "expected_quantized"), ("n_unquantized", "expected_unquantized")):
        if info.get(got) != info.get(want):
            raise RuntimeError(f"builder {info['builder']!r}: {got}={info.get(got)} but the builder expects {info.get(want)} "
                               f"(quantize_layers={info.get('quantize_layers')}) -- the layer set did not apply; row refused")


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


def run_controls(a, model_id, revision, tok, prompts, dev, ref_cache, scorer_used) -> dict:
    """Amendment 2/5 controls, after the arms. (i) the reference decode-vs-prefill self-KL (already run and decisive
    when --scorer auto; re-run here so the receipt carries it in every mode). (ii) the nf4 control arm scored with the
    OTHER shape against the same cached reference -- with the reference's own shape gap (i) disclosed beside it."""
    n = min(a.controls_n, len(prompts))
    out = {"n_prompts": n}
    try:
        ref = load_reference(a.family, model_id, revision, dev)
        out["reference_decode_vs_prefill"] = self_consistency(ref, tok, prompts, a.max_len, dev, n)
        del ref
    except Exception as e:
        out["reference_decode_vs_prefill"] = {"error": f"{type(e).__name__}: {str(e)[:400]}"}
    gc.collect()
    torch.cuda.empty_cache()
    import subprocess
    ctrl_arm = control_arm(a.family)
    other = "prefill" if scorer_used == "decode" else "decode"
    part = f"{a.out}.{ctrl_arm}.{other}.part.json"
    env = dict(os.environ)
    for k in LANE_KEYS:
        env.pop(k, None)
    env.update(arm_env(a.family, ctrl_arm, model_id))
    cmd = [sys.executable, "-u", os.path.abspath(__file__), "--family", a.family, "--model", model_id, "--revision", revision,
           "--arena", a.arena, "--calib", a.calib, "--k0-receipt", a.k0_receipt, "--arms", ctrl_arm, "--max-len", str(a.max_len),
           "--limit", str(n), "--ref-cache", ref_cache, "--scorer", other, "--out", part, "--child", "--controls", "0"]
    t1 = time.time()
    rc = subprocess.call(cmd, env=env)
    got = json.load(open(part)) if os.path.exists(part) else {}
    rows = got.get("rows", [])
    out["control_arm_other_shape"] = ({"arm": ctrl_arm, "arm_scorer": other, "reference_scorer": scorer_used, "kl_mean": rows[0]["kl_mean"],
                                       "kl_p95": rows[0]["kl_p95"], "top1": rows[0]["top1_agreement"], "tokens": rows[0]["n_tokens_scored"],
                                       "per_stratum": {k: v["kl_mean"] for k, v in rows[0]["per_stratum"].items()},
                                       "note": "the control arm under the OTHER forward shape vs the cached reference: includes the reference's own shape gap (i)"}
                                      if rows else {"arm": ctrl_arm, "error": f"child rc={rc}: {got.get('not_measured')}"})
    out["control_arm_other_shape"]["wall_s"] = round(time.time() - t1, 1)
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
    ap.add_argument("--scorer", default="auto", choices=["auto"] + sorted(SCORERS),
                    help="auto (amendment 5): control (i) picks decode when the reference agrees with itself, prefill otherwise")
    ap.add_argument("--out", required=True)
    ap.add_argument("--child", action="store_true", help="internal: one arm, env already set by the parent")
    ap.add_argument("--controls", type=int, default=1,
                    help="amendment 2: after the arms, the instrument controls -- reference decode-vs-prefill self-KL and the nf4 "
                         "control arm scored with the PREFILL scorer against the same cached reference (first --controls-n prompts)")
    ap.add_argument("--controls-n", type=int, default=40)
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
    if a.child and a.scorer == "auto":
        raise SystemExit("child needs a resolved --scorer (decode|prefill)")
    score = SCORERS[a.scorer] if a.scorer != "auto" else None
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
                   "note": {"decode": "decode-shaped teacher forcing on BOTH sides (one token per forward, KV cache): the serving levers engage only at T == 1",
                            "prefill": "PREFILL scorer on BOTH sides: the weight-format levers (int4 experts, calibrated int4 attention) engage; the decode-only fusions (folds, router epilogue) do not and their rows equal the control's by construction",
                            "auto": "control (i) picks the scorer per family: decode when the reference agrees with itself to < 1e-2 nats, prefill otherwise (amendment 5)"}[a.scorer]},
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
    if not a.child:
        receipt["reference_pass"], scorer_used = reference_pass(a.family, model_id, revision, tok, prompts, a.ref_cache,
                                                                a.max_len, a.scorer, dev, a.controls_n)
        receipt["reference_pass"]["wall_s"] = round(time.time() - t0, 1)
        receipt["scorer"]["used"] = scorer_used
        ref_cache = receipt["reference_pass"]["cache_dir"]
        score = SCORERS[scorer_used]
        flush()
    else:
        scorer_used, ref_cache = a.scorer, a.ref_cache
        missing = [p["id"] for p in prompts if not os.path.exists(os.path.join(ref_cache, p["id"] + ".pt"))]
        if missing:
            raise SystemExit(f"child: reference cache incomplete ({len(missing)} prompts missing) -- the parent scores the reference first")

    if not a.child:
        # ONE CHILD PER ARM, env set before its interpreter starts (the hook arms itself at import time -- run 2's fault).
        import subprocess
        for arm in arms:
            env = dict(os.environ)
            for k in LANE_KEYS:
                env.pop(k, None)
            env.update(arm_env(a.family, arm, model_id))
            part = f"{a.out}.{arm}.part.json"
            cmd = [sys.executable, "-u", os.path.abspath(__file__), "--family", a.family, "--model", model_id, "--revision", revision,
                   "--arena", a.arena, "--calib", a.calib, "--k0-receipt", a.k0_receipt, "--arms", arm, "--max-len", str(a.max_len),
                   "--limit", str(a.limit), "--ref-cache", ref_cache, "--scorer", scorer_used, "--out", part, "--child"]
            t1 = time.time()
            rc = subprocess.call(cmd, env=env)
            got = json.load(open(part)) if os.path.exists(part) else {"rows": [], "not_measured": {arm: {"error": f"child exited rc={rc} with no partial receipt", "wall_s": round(time.time() - t1, 1)}}}
            receipt["rows"] += got.get("rows", [])
            receipt["not_measured"].update(got.get("not_measured", {}))
            for r in got.get("rows", []):
                print(f"== {a.family}/{arm}: KL mean={r['kl_mean']:.6e} p95={r['kl_p95']:.6e} top1={r['top1_agreement']:.5f} tokens={r['n_tokens_scored']} ({r['wall_s']} s)", flush=True)
            for k, v in got.get("not_measured", {}).items():
                print(f"== {a.family}/{k}: NOT MEASURED -- {v.get('error', '')[:300]}", flush=True)
            flush()
        if a.controls:
            receipt["controls"] = run_controls(a, model_id, revision, tok, prompts, dev, ref_cache, scorer_used)
            flush()
        receipt["wall_s_total"] = round(time.time() - t0, 1)
        flush()
        print(f"receipt -> {a.out} ({len(receipt['rows'])} rows, {len(receipt['not_measured'])} not measured)")
        return 0 if receipt["rows"] and not receipt["not_measured"] else 3

    # ---- child: exactly one arm, the env already set by the parent; the hook banner proves it armed
    for arm in arms:
        env = arm_env(a.family, arm, model_id)
        for k, v in env.items():
            if os.environ.get(k) != v:
                raise SystemExit(f"child env mismatch for {k}: {os.environ.get(k)!r} != {v!r} -- the parent must set the arm's env before the interpreter starts")
        t1 = time.time()
        model = None
        try:
            model, info = build_arm_model(a.family, arm, model_id, a.arena, a.calib, device=dev)
            info["hook_loaded"] = hook
            _lever_check(env, info)
            _builder_check(info)
            r = score_arm(model, tok, prompts, ref_cache, a.max_len, score, dev)
            r.update({"arm": arm, "builder": info.get("builder", "served"), "row": f"{a.family}/{arm} vs the family's reference", "env": env,
                      "engagement": info, "reference": REFERENCE[a.family], "scorer": scorer_used,
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
    return 0 if receipt["rows"] and not receipt["not_measured"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
