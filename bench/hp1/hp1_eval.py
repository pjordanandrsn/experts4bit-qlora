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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--receipts", nargs="+", required=True)
    ap.add_argument("--bases", nargs="+", required=True)
    ap.add_argument("--tokens", required=True)
    ap.add_argument("--eval-n", type=int, default=48)
    ap.add_argument("--model", default="Qwen/Qwen3-30B-A3B")
    ap.add_argument("--revision", default="ad44e777bcd18fa416d9da3bd8f70d33ebb85d39")
    ap.add_argument("--out", required=True)
    ap.add_argument("--require-all-scored", type=int, default=0,
                    help="the box-side shakedown: exit 3 unless every arm is SCORED by this evaluator and every base is OK")
    a = ap.parse_args()
    tc1 = load_tc1_arm()
    tc1.DEV = DEV
    tk = json.load(open(a.tokens))
    rows = tk["eval"][: a.eval_n]
    if len(rows) != a.eval_n:
        sys.exit(f"tokens file has {len(rows)} eval rows, need {a.eval_n}")
    arms = []
    for p in sorted({q for pat in a.receipts for q in glob.glob(pat)}):
        r = json.load(open(p))
        hp1 = r.get("hp1") or {}
        curve = r.get("eval_curve") or []
        arms.append({"receipt": p, "framework": r.get("framework"), "tag": r.get("tag"), "seed": r.get("seed"), "order": r.get("order"),
                     "tokens_sha": (r.get("tokens") or {}).get("sha256"), "adapter": (hp1.get("adapter") or {}).get("path"),
                     "adapter_sha256": (hp1.get("adapter") or {}).get("sha256"), "train_digest": (hp1.get("train_base") or {}).get("digest"),
                     "inloop_step0": curve[0]["heldout_loss"] if curve else None, "inloop_final": curve[-1]["heldout_loss"] if curve else None,
                     "inloop_final_step": curve[-1]["step"] if curve else None, "hp1_errors": {k: v for k, v in hp1.items() if k.endswith("_error")}})
    out = {"evaluator": "bench/hp1/hp1_eval.py", "tokens": {"path": a.tokens, "sha256": tk.get("sha256"), "eval_n": a.eval_n},
           "model": a.model, "revision": a.revision, "device": torch.cuda.get_device_name(0) if DEV == "cuda" else "cpu",
           "torch": torch.__version__, "bases": [], "arms": []}
    for arm in arms:
        if arm["tokens_sha"] != tk.get("sha256"):
            arm["status"] = "VOID"
            arm["reason"] = f"tokens sha {str(arm['tokens_sha'])[:12]} != evaluator tokens {str(tk.get('sha256'))[:12]}"
        elif arm["hp1_errors"] or not arm["adapter"] or not arm["train_digest"]:
            arm["status"] = "VOID"
            arm["reason"] = f"no HP1 export in the receipt ({arm['hp1_errors'] or 'adapter/digest missing'})"
    for bdir in a.bases:
        t0 = time.perf_counter()
        store = hp1_lib.BaseStore(bdir, DEV)
        digest, n_slots = store.digest()
        want = store.meta.get("train_base_digest")
        brec = {"dir": bdir, "meta": store.meta, "digest": digest, "n_slots": n_slots, "digest_matches_training_run": digest == want}
        out["bases"].append(brec)
        if digest != want:
            brec["status"] = "VOID"
            brec["reason"] = f"STRICT: the export's re-derived digest {digest[:12]} != its training run's {str(want)[:12]}"
            del store
            continue
        mine = [x for x in arms if x.get("status") is None and x["train_digest"] == digest]
        if not mine:
            brec["status"] = "UNUSED"
            del store
            continue
        model, hooks = build_model(store, a.model, a.revision)
        set_adapter(hooks, None)
        zero = score(model, rows, tc1)
        brec["zero_adapter"] = {k: zero[k] for k in ("row_mean", "token_weighted")}
        bad = [x["tag"] for x in mine if x["inloop_step0"] is not None and abs(zero["row_mean"] - x["inloop_step0"]) > ZERO_ADAPTER_TOL]
        brec["zero_adapter_vs_inloop_step0"] = {x["tag"]: None if x["inloop_step0"] is None else round(zero["row_mean"] - x["inloop_step0"], 6) for x in mine}
        if bad:
            brec["status"] = "VOID"
            brec["reason"] = f"STRICT zero-adapter check: |evaluator - in-loop step 0| > {ZERO_ADAPTER_TOL} for {bad}"
            del model, hooks, store
            continue
        flip_done = False
        for arm in mine:
            if sha_file(arm["adapter"]) != arm["adapter_sha256"]:
                arm["status"], arm["reason"] = "VOID", "adapter file sha differs from the receipt's"
                continue
            canon = hp1_lib.load_canonical(arm["adapter"], DEV)
            set_adapter(hooks, canon[:2])
            res = score(model, rows, tc1)
            arm.update(status="SCORED", base=bdir, evaluator=res)
            if arm["inloop_final"] is not None:
                arm["evaluator_minus_inloop_final"] = round(res["row_mean"] - arm["inloop_final"], 6)     # REPORTING
            if not flip_done:                       # STRICT control: a perturbed adapter must move the score
                t, sc = canon[0], canon[1]
                k = next(iter(sorted(x for x in t if x.endswith(".B"))))
                t[k] = t[k].clone()
                t[k].view(-1)[0] += 1.0
                set_adapter(hooks, (t, sc))
                flipped = score(model, rows, tc1)["row_mean"]
                brec["perturbed_adapter_control"] = {"tensor": k, "row_mean": flipped, "changed": flipped != res["row_mean"]}
                flip_done = True
                if flipped == res["row_mean"]:
                    brec["status"] = "VOID"
                    brec["reason"] = "STRICT control: a perturbed adapter did not change the score"
            del canon
        brec.setdefault("status", "OK")
        if brec["status"] == "VOID":
            for arm in mine:
                arm["status"], arm["reason"] = "VOID", f"base {bdir}: {brec['reason']}"
        brec["seconds"] = round(time.perf_counter() - t0, 1)
        del model, hooks, store
        torch.cuda.empty_cache() if DEV == "cuda" else None
    for arm in arms:
        if arm.get("status") is None:
            arm["status"] = "OWN_BASE_ONLY"
            arm["reason"] = "no base export matches this arm's training-base digest: the readout uses its own-base score (in-loop eval_loss at step N, same rows)"
        out["arms"].append(arm)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=1)
    print(json.dumps({"bases": [(b["dir"], b.get("status")) for b in out["bases"]],
                      "arms": [(x["tag"], x["status"], (x.get("evaluator") or {}).get("row_mean")) for x in out["arms"]]}))
    ok_digests = {b["digest"] for b in out["bases"] if b.get("status") == "OK"}

    def base_fine(b):                               # an UNUSED base is fine when it is a verified duplicate of a base that scored
        return b.get("status") == "OK" or (b.get("status") == "UNUSED" and b.get("digest_matches_training_run") and b["digest"] in ok_digests)
    if a.require_all_scored and (not out["arms"] or any(x["status"] != "SCORED" for x in out["arms"]) or not all(map(base_fine, out["bases"]))):
        sys.exit(3)


if __name__ == "__main__":
    main()
