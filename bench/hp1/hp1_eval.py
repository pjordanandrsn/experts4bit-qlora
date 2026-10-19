#!/usr/bin/env python3
"""HP1's one evaluator (HP1-PREREG.md, "The one evaluator"): every arm's step-N adapter, scored by this code on the base that
arm trained on, with this file's own eager forward. No framework's eval loop and no framework's fused kernel is on this path.

Inputs:
  --receipts   the TC1 receipts of the arms to score (each carries `hp1.adapter.path`, `hp1.train_base.digest`, its tokens sha and
               its in-loop eval curve);
  --bases      base exports (`--hp1-export-base 1` runs: base_<framework>_<fam>/index.json); an arm is scored on the base whose
               re-derived digest equals the arm's training-base digest. No match -> the arm is NOT scored here and the readout
               uses its own-base number (the harness's eval_loss at step N, same rows), labelled so (ruling item 3);
  --tokens     the TC1 tokens file (its eval rows; the first --eval-n are scored).

Per base: the skeleton is the model's own HF class built on `meta`, each layer's experts and q/k/v/o replaced by hp1_lib's
EagerExperts / EagerLinear reading the packed base, every other weight (embeddings, norms, routers, lm_head) loaded from the
export, the rotary buffers rebuilt. STRICT self-checks, per base: the export's digest equals its training run's; a zero adapter
scores within 0.005 of each matched arm's in-loop step-0 loss; a perturbed adapter changes the score. Output: one JSON with
per-arm per-row losses, the row mean (the readout's y) and the token-weighted mean.
"""

import argparse
import glob
import hashlib
import importlib.util
import json
import os
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import hp1_lib  # noqa: E402

ZERO_ADAPTER_TOL = 0.005     # TC1's SAME-BYTES-CLASS bound on step-0 held-out across forwards
ROUNDTRIP_TOL = 0.02         # self-check 6(b): the evaluator's step-N score vs the arm's own in-loop step-N score (TC1b's fixed band)
EXPECT_TRAINABLE = 642514944  # the registered trainable count on every Qwen3-30B-A3B arm (HP1-PREREG "Fixture")
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def load_tc1_arm():
    flat = os.path.join(HERE, "tc1_arm.py")              # a box stages the pieces flat; the repo keeps them in bench/tc1
    path = flat if os.path.exists(flat) else os.path.join(HERE, "..", "tc1", "tc1_arm.py")
    spec = importlib.util.spec_from_file_location("tc1_arm_for_hp1", path)
    m = importlib.util.module_from_spec(spec)
    sys.modules["tc1_arm_for_hp1"] = m
    spec.loader.exec_module(m)
    return m


def build_model(store, model_id, revision):
    """The evaluator's model over one BaseStore: HF's class from the pinned config on `meta`, eager modules swapped in, the rest
    materialised from the export. Returns (model, {L: (experts_module, {kind: EagerLinear})})."""
    import torch.nn as nn
    from transformers import AutoConfig, AutoModelForCausalLM
    cfg = AutoConfig.from_pretrained(model_id, revision=revision)
    cfg._attn_implementation = "eager"
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(cfg, dtype=torch.bfloat16)
    keys = set(store.keys())
    hooks = {}
    for L, layer in enumerate(model.model.layers):
        mlp, attn = layer.mlp, layer.self_attn
        E = int(cfg.num_experts)
        if (L, "gate_up", 0) not in keys:
            raise RuntimeError(f"base {store.dir}: layer {L} has no expert slots")
        mlp.experts = hp1_lib.EagerExperts((lambda kind, e, L=L: store.deq((L, kind, e))), E, mlp.experts.act_fn)
        lins = {}
        for pn, kind in (("q_proj", "q"), ("k_proj", "k"), ("v_proj", "v"), ("o_proj", "o")):
            old = getattr(attn, pn)
            if getattr(old, "bias", None) is not None:
                raise RuntimeError(f"layer {L} {pn}: a biased projection; the export carries no bias")
            lin = hp1_lib.EagerLinear(lambda L=L, kind=kind: store.deq((L, kind, -1)))
            setattr(attn, pn, lin)
            lins[kind] = lin
        hooks[L] = (mlp.experts, lins)
    model.to_empty(device=DEV)
    loaded = set()

    def put(param, key):
        w = store.deq(key)
        if tuple(w.shape) != tuple(param.shape):
            raise RuntimeError(f"{key}: export {list(w.shape)} vs model {list(param.shape)}")
        with torch.no_grad():
            param.copy_(w)
        loaded.add(id(param))
    put(model.model.embed_tokens.weight, (-1, "embed", -1))
    put(model.lm_head.weight, (-1, "lm_head", -1))
    put(model.model.norm.weight, (-1, "final_norm", -1))
    for L, layer in enumerate(model.model.layers):
        put(layer.mlp.gate.weight, (L, "router", -1))
        for name, m in layer.named_modules():
            last = name.split(".")[-1]
            if "norm" in last.lower() and isinstance(getattr(m, "weight", None), nn.Parameter):
                put(m.weight, (L, "norm:" + last, -1))
    missing = [n for n, p in model.named_parameters() if id(p) not in loaded]
    if missing:
        raise RuntimeError(f"parameters not covered by the base export: {missing[:6]}")
    model.model.rotary_emb = type(model.model.rotary_emb)(config=cfg, device=DEV)
    model.eval()
    return model, hooks


def set_adapter(hooks, canon):
    """Attach a canonical adapter (hp1_lib.load_canonical output) to every eager module, or detach all (canon None)."""
    for L, (experts, lins) in hooks.items():
        if canon is None:
            experts.adapter = None
            for lin in lins.values():
                lin.adapter = None
            continue
        t, sc = canon
        experts.adapter = {k: (t[f"{L}.{k}.A"], t[f"{L}.{k}.B"], sc[f"{L}.{k}"]) for k in ("gate_up", "down")}
        for kind, lin in lins.items():
            lin.adapter = (t[f"{L}.{kind}.A"], t[f"{L}.{kind}.B"], sc[f"{L}.{kind}"])
    if canon is not None:
        t = canon[0]
        want = {f"{L}.{k}.{ab}" for L in hooks for k in ("gate_up", "down", "q", "k", "v", "o") for ab in "AB"}
        extra, absent = set(t) - want, want - set(t)
        if extra or absent:
            raise RuntimeError(f"canonical adapter does not cover the model: extra {sorted(extra)[:4]} missing {sorted(absent)[:4]}")


def score(model, rows, tc1):
    mean, per_row = tc1.eval_loss(model, rows, lambda x: {}, True)
    ntok = [max(len(r) - 1, 1) for r in rows]
    tw = sum(loss * n for loss, n in zip(per_row, ntok)) / sum(ntok)
    return {"row_mean": round(mean, 6), "token_weighted": round(tw, 6), "row_losses": per_row, "n_rows": len(rows)}


def sha_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def _bounds_arg(v):
    """--roundtrip-bounds: inline JSON, or @path to a JSON file (the driver passes a file: no JSON in a shell word)."""
    if not v:
        return None
    return json.load(open(v[1:])) if v.startswith("@") else json.loads(v)


def receipt_arm(path):
    """One arm as the evaluator reads it from its TC1 receipt."""
    r = json.load(open(path))
    hp1 = r.get("hp1") or {}
    curve = r.get("eval_curve") or []
    step0 = [c["heldout_loss"] for c in curve if c.get("step") == 0]
    step0_rows = [c.get("row_losses") for c in curve if c.get("step") == 0]
    return {"receipt": path, "framework": r.get("framework"), "tag": r.get("tag"), "seed": r.get("seed"), "order": r.get("order"),
            "tokens_sha": (r.get("tokens") or {}).get("sha256"), "adapter": (hp1.get("adapter") or {}).get("path"),
            "adapter_sha256": (hp1.get("adapter") or {}).get("sha256"), "train_digest": (hp1.get("train_base") or {}).get("digest"),
            "inloop_step0": step0[0] if step0 else None, "inloop_step0_rows": step0_rows[0] if step0_rows else None, "inloop_final": curve[-1]["heldout_loss"] if curve else None,
            "inloop_final_step": curve[-1]["step"] if curve else None, "steps": r.get("steps"),
            "hp1_errors": {k: v for k, v in hp1.items() if k.endswith("_error")}, "raw_ok": (hp1.get("train_base") or {}).get("raw_ok"),
            "status_receipt": r.get("status"), "c1_bit_exact": r.get("C1_bit_exact"), "trainable_params": r.get("trainable_params"),
            "lora_init": r.get("lora_init"), "adapter_dtype": r.get("adapter_dtype"), "matched_init": r.get("matched_init"),
            "adapter_dtypes_after": r.get("adapter_dtypes_after") or {}}


def tc1_invalid(arm, tokens_sha, expect_trainable):
    """TC1's validity predicates (HP1-PREREG "Validity"), read from the receipt; the first that fails, or None. A run that is not
    TC1-valid is never scored."""
    if arm["status_receipt"] != "ok":
        return f"receipt status {arm['status_receipt']!r} (not ok)"
    if arm["c1_bit_exact"] is not True:
        return "C1: the frozen base changed during training (C1_bit_exact is not true)"
    if arm["tokens_sha"] != tokens_sha:
        return f"tokens sha {str(arm['tokens_sha'])[:12]} != the evaluator's {str(tokens_sha)[:12]}"
    if expect_trainable is not None and arm["trainable_params"] != expect_trainable:
        return f"trainable {arm['trainable_params']} != the registered {expect_trainable}"
    if str(arm["lora_init"] or "").startswith("matched:") and not (arm["matched_init"] or {}).get("complete"):
        return "matched init incomplete"
    dts = set(arm["adapter_dtypes_after"])
    if arm["framework"] == "e4b" and arm["adapter_dtype"] == "native" and "torch.bfloat16" not in dts:
        return f"dtype census: a native e4b cell must carry bf16 expert adapters, found {sorted(dts)}"
    if arm["adapter_dtype"] == "fp32" and dts != {"torch.float32"}:
        return f"dtype census: an fp32 cell must carry fp32 adapters only, found {sorted(dts)}"
    if not arm["order"] or arm["order"].get("order_seed") != arm["seed"]:
        return f"order: {arm['order']!r} does not carry this run's seed {arm['seed']}"
    if arm["hp1_errors"] or not arm["adapter"] or not arm["train_digest"]:
        return f"no HP1 export in the receipt ({arm['hp1_errors'] or 'adapter/digest missing'})"
    if arm["raw_ok"] is not True:
        return "the training-base digest's stored-slot dequant check did not pass (raw_ok)"
    if arm["inloop_step0"] is None:
        return "no in-loop step-0 held-out read: the zero-adapter check cannot run (fails closed)"
    if arm["inloop_final"] is None or arm["inloop_final_step"] != arm["steps"]:
        return f"no in-loop step-{arm['steps']} held-out read: the round-trip check cannot run (fails closed)"
    return None


def evaluate(arms, base_dirs, rows, open_base, make_scorer, load_adapter, file_sha, tokens_sha, expect_trainable, write=None,
             release=None, zero_tol=ZERO_ADAPTER_TOL, roundtrip_tol=ROUNDTRIP_TOL, zero_report_only=False, roundtrip_bounds=None):
    """The evaluator's rules, with the model and the scoring injected (``open_base(dir) -> store`` with .digest() and .meta;
    ``make_scorer(store) -> (set_adapter(canon | None), score() -> {"row_mean", ...})``; ``load_adapter(path) -> (tensors, scalings)``).
    Every STRICT failure VOIDs: a TC1-invalid run is never scored; a base whose digest or zero-adapter check fails VOIDs every arm
    that trained on it; the round-trip check 6(b) and the perturbation control VOID their arm / base. Every verdict uses evaluator
    numbers only, so an arm that no export claims is VOID too (it has no evaluator number). Each VOID carries `void_level`: "run"
    (the driver redraws it once) or "base" (a systematic miss: never redrawn). ``write(bases, arms)`` is called after every base;
    ``release()`` after every base, once its store and scorer are dropped (one model on the device at a time)."""
    bases = []
    for arm in arms:
        why = tc1_invalid(arm, tokens_sha, expect_trainable)
        if why:
            arm["status"], arm["reason"], arm["void_level"] = "VOID", f"TC1-invalid: {why}", "run"
    for bdir in base_dirs:
        t0 = time.perf_counter()
        set_adapter = score_fn = None
        store = open_base(bdir)
        digest, n_slots = store.digest()
        want = store.meta.get("train_base_digest")
        brec = {"dir": bdir, "meta": store.meta, "digest": digest, "n_slots": n_slots, "digest_matches_training_run": digest == want}
        bases.append(brec)
        claim = [x for x in arms if x.get("status") is None and x["train_digest"] in (digest, want)]
        if digest != want:
            brec["status"], brec["reason"] = "VOID", f"STRICT: the export's re-derived digest {digest[:12]} != its training run's {str(want)[:12]}"
        mine = [x for x in claim if x["train_digest"] == digest]
        if brec.get("status") is None and not mine:
            brec["status"] = "UNUSED"
        if brec.get("status") is None:
            set_adapter, score_fn = make_scorer(store)
            set_adapter(None)
            zero = score_fn()
            # HP1-PREREG Amendment 4: the per-row zero-adapter losses and, where the receipt carries them, the per-row gaps against the
            # in-loop step-0 rows, so a VOID can be diagnosed from the receipt (hp1k-5090-9 could not be); the tolerance used is recorded.
            brec["zero_adapter"] = {k: zero[k] for k in ("row_mean", "token_weighted")}
            brec["zero_adapter"]["row_losses"] = zero.get("row_losses")
            brec["zero_adapter_tol"] = zero_tol
            brec["zero_adapter_vs_inloop_step0"] = {x["tag"]: round(zero["row_mean"] - x["inloop_step0"], 6) for x in mine}
            brec["zero_adapter_row_gaps"] = {x["tag"]: [round(e - i, 6) for e, i in zip(zero["row_losses"], x["inloop_step0_rows"])]
                                             for x in mine if zero.get("row_losses") and x.get("inloop_step0_rows")
                                             and len(x["inloop_step0_rows"]) == len(zero["row_losses"])}
            bad = [x["tag"] for x in mine if abs(zero["row_mean"] - x["inloop_step0"]) > zero_tol]
            brec["zero_adapter_report_only"] = bool(zero_report_only)
            if bad and zero_report_only:          # HP2: the gap is deterministic per base, so it is a reported fact, never a VOID
                brec["zero_adapter_outside_tol"] = bad
            elif bad:
                brec["status"], brec["reason"] = "VOID", f"STRICT zero-adapter check: |evaluator - in-loop step 0| > {zero_tol} for {bad}"
        if brec.get("status") is None:
            flip_done = False
            for arm in mine:
                if file_sha(arm["adapter"]) != arm["adapter_sha256"]:
                    arm["status"], arm["reason"], arm["void_level"] = "VOID", "adapter file sha differs from the receipt's", "run"
                    continue
                canon = load_adapter(arm["adapter"])
                set_adapter(canon)
                res = score_fn()
                arm.update(base=bdir, evaluator=res, evaluator_minus_inloop_final=round(res["row_mean"] - arm["inloop_final"], 6))
                rb = (roundtrip_bounds or {}).get(arm["framework"])                # HP2: [centre, half-width] per framework
                if rb is not None and abs(arm["evaluator_minus_inloop_final"] - rb[0]) > rb[1]:
                    arm["status"], arm["reason"], arm["void_level"] = "VOID", (f"STRICT round-trip 6(b): |evaluator - in-loop step N "
                                                                               f"- {rb[0]}| > {rb[1]}"), "run"
                elif rb is None and abs(arm["evaluator_minus_inloop_final"]) > roundtrip_tol:     # self-check 6(b): the export round-trips its layout
                    arm["status"], arm["reason"], arm["void_level"] = "VOID", f"STRICT round-trip 6(b): |evaluator - in-loop step N| > {roundtrip_tol}", "run"
                else:
                    arm["status"] = "SCORED"
                if not flip_done:                   # STRICT control: a perturbed adapter must move the score
                    t, sc = canon
                    k = sorted(x for x in t if x.endswith(".B"))[0]
                    t = dict(t)
                    t[k] = t[k].clone()
                    t[k].view(-1)[0] += 1.0
                    set_adapter((t, sc))
                    flipped = score_fn()["row_mean"]
                    brec["perturbed_adapter_control"] = {"tensor": k, "row_mean": flipped, "changed": flipped != res["row_mean"]}
                    flip_done = True
                    if flipped == res["row_mean"]:
                        brec["status"], brec["reason"] = "VOID", "STRICT control: a perturbed adapter did not change the score"
            brec.setdefault("status", "OK")
        if brec["status"] == "VOID":                # every arm that trained on this base (by either digest) is VOID with it
            for arm in claim:
                arm["status"], arm["reason"], arm["void_level"] = "VOID", f"base {bdir}: {brec['reason']}", "base"
        brec["seconds"] = round(time.perf_counter() - t0, 1)
        del store, set_adapter, score_fn            # the scorer closes over the model: drop it before the next base loads
        if release:
            release()
        if write:
            write(bases, arms)
    for arm in arms:
        if arm.get("status") is None:               # no evaluator number: every verdict uses evaluator numbers only
            arm["status"], arm["void_level"] = "VOID", "run"
            arm["reason"] = "no base export claims this arm's training-base digest, so it has no evaluator number (its redraw re-exports the base)"
    return bases, arms


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--receipts", nargs="+", required=True)
    ap.add_argument("--bases", nargs="+", required=True)
    ap.add_argument("--tokens", required=True)
    ap.add_argument("--eval-n", type=int, default=48)
    ap.add_argument("--model", default="Qwen/Qwen3-30B-A3B")
    ap.add_argument("--revision", default="ad44e777bcd18fa416d9da3bd8f70d33ebb85d39")
    ap.add_argument("--expect-trainable", type=int, default=EXPECT_TRAINABLE,
                    help="the registered trainable count every arm must carry (0 = not checked: the shakedown's truncated model)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--zero-report-only", type=int, default=0,
                    help="1: record the zero-adapter gap (and any excess over --zero-tol) but never VOID on it (HP2: it is deterministic per base)")
    ap.add_argument("--roundtrip-bounds", default=None,
                    help='HP2: JSON {framework: [centre, half-width]}; an arm VOIDs iff |evaluator - in-loop step N - centre| > half-width')
    ap.add_argument("--roundtrip-tol", type=float, default=ROUNDTRIP_TOL,
                    help="the 6(b) round-trip tolerance (HP2: the calibrated mean-level bound; default TC1b's 0.02)")
    ap.add_argument("--zero-tol", type=float, default=ZERO_ADAPTER_TOL,
                    help="the zero-adapter tolerance (HP1-PREREG Amendment 4: set by the 48-layer calibration; default the 2-layer 0.005)")
    ap.add_argument("--require-all-scored", type=int, default=0,
                    help="the box-side shakedown: exit 3 unless every arm is SCORED by this evaluator and every base is OK")
    a = ap.parse_args()
    tc1 = load_tc1_arm()
    tc1.DEV = DEV
    tk = json.load(open(a.tokens))
    rows = tk["eval"][: a.eval_n]
    if len(rows) != a.eval_n:
        sys.exit(f"tokens file has {len(rows)} eval rows, need {a.eval_n}")
    arms = [receipt_arm(p) for p in sorted({q for pat in a.receipts for q in glob.glob(pat)})]
    out = {"evaluator": "bench/hp1/hp1_eval.py", "tokens": {"path": a.tokens, "sha256": tk.get("sha256"), "eval_n": a.eval_n},
           "model": a.model, "revision": a.revision, "device": torch.cuda.get_device_name(0) if DEV == "cuda" else "cpu",
           "torch": torch.__version__, "zero_adapter_tol": a.zero_tol, "roundtrip_tol": a.roundtrip_tol,
           "zero_adapter_report_only": bool(a.zero_report_only), "roundtrip_bounds": _bounds_arg(a.roundtrip_bounds),
           "expect_trainable": a.expect_trainable or None, "bases": [], "arms": []}
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)

    def write(bases, arms_now):                     # after every base, so a deadline mid-pass keeps what was scored
        out["bases"], out["arms"] = bases, arms_now
        json.dump(out, open(a.out + ".tmp", "w"), indent=1)
        os.replace(a.out + ".tmp", a.out)

    def release():
        import gc
        gc.collect()
        if DEV == "cuda":
            torch.cuda.empty_cache()

    def make_scorer(store):
        model, hooks = build_model(store, a.model, a.revision)
        return (lambda canon: set_adapter(hooks, None if canon is None else canon[:2])), (lambda: score(model, rows, tc1))

    bases, arms = evaluate(arms, a.bases, rows, lambda d: hp1_lib.BaseStore(d, DEV), make_scorer,
                           lambda p: hp1_lib.load_canonical(p, DEV)[:2], sha_file, tk.get("sha256"), a.expect_trainable or None, write,
                           release, zero_tol=a.zero_tol, roundtrip_tol=a.roundtrip_tol, zero_report_only=bool(a.zero_report_only),
                           roundtrip_bounds=_bounds_arg(a.roundtrip_bounds))
    write(bases, arms)
    print(json.dumps({"bases": [(b["dir"], b.get("status")) for b in bases],
                      "arms": [(x["tag"], x["status"], (x.get("evaluator") or {}).get("row_mean")) for x in arms]}))
    ok_digests = {b["digest"] for b in bases if b.get("status") == "OK"}

    def base_fine(b):                               # an UNUSED base is fine when it is a verified duplicate of a base that scored
        return b.get("status") == "OK" or (b.get("status") == "UNUSED" and b.get("digest_matches_training_run") and b["digest"] in ok_digests)
    if a.require_all_scored and (not arms or any(x["status"] != "SCORED" for x in arms) or not all(map(base_fine, bases))):
        sys.exit(3)


if __name__ == "__main__":
    main()
