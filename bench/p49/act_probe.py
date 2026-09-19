#!/usr/bin/env python3
"""bench/p49/act_probe.py -- P49's activation probe (P49-PREREG.md, P5): WHERE does a ~9 % relative weight error become a
0.9-nat KL? On the SAME tokens, compare each quantised layer's expert-branch output against the bf16 reference's:

  raw    = experts(hidden_states_2)                 -- the MoE branch before Gemma-4's post_feedforward_layernorm_2
  norm2  = post_feedforward_layernorm_2(raw)        -- what is added (with the dense branch) to the residual

Reference pass (HF bf16, hooks on every text layer) saves raw/norm2 per prompt per layer and records per-layer rms of the
residual input, the raw MoE output, its post-norm, and the dense branch (post_feedforward_layernorm_1). Then per arm the
model is built through `serve_stack.build_arm_model` and the quantised layer's raw/norm2 are captured on the same prompts;
layers below it are bf16 in both models, so the layer's inputs are identical and the difference is the store alone.

Receipt: <out>.json with per-layer reference stats and per-arm relative errors (token-weighted over the probe prompts):
rel_raw, rel_norm2, amplification = rel_norm2 / rel_raw, and the reference's contrib = rms(norm2) / rms(residual_in).
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import re
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from kl_prompts import PROMPTS  # noqa: E402
from kl_serve import load_reference  # noqa: E402
from serve_stack import ARMS, LANE_KEYS, MODELS, apply_env, arm_env, build_arm_model, builder_for, parse_only  # noqa: E402

_LAYER_EXPERTS = re.compile(r"(?:^|\.)layers\.(\d+)\.experts$")


def moe_layers(model) -> dict:
    """layer index -> (layer module, experts module) for every text decoder layer that carries an MoE block."""
    mods = dict(model.named_modules())
    out = {}
    for name, m in mods.items():
        mm = _LAYER_EXPERTS.search(name)   # SEARCH: the name is dotted ("model.language_model.layers.0.experts"), so match() anchors at 0 and finds nothing
        if not mm:
            continue
        layer = mods[name[: -len(".experts")]]
        if hasattr(layer, "post_feedforward_layernorm_2"):
            out[int(mm.group(1))] = (layer, m)
    if not out:
        seen = [n for n in mods if n.endswith(".experts")][:6]
        raise RuntimeError("no `layers.<i>.experts` module with a post_feedforward_layernorm_2 parent found -- not Gemma-4's "
                           f"text tower? (modules ending in '.experts': {seen})")
    return out


def _rms(t: torch.Tensor) -> float:
    return float(t.float().pow(2).mean().sqrt())


def pick_prompts(n: int) -> list:
    """The first n//4 prompts of each stratum (so a probe of 8 sees every stratum), in registered order."""
    per = max(1, n // 4)
    out, seen = [], {}
    for p in PROMPTS:
        if seen.get(p["stratum"], 0) < per:
            out.append(p)
            seen[p["stratum"]] = seen.get(p["stratum"], 0) + 1
    return out[:n]


class Capture:
    def __init__(self, layers: dict, want: set):
        self.store = {}
        self.handles = []
        for i, (layer, experts) in layers.items():
            if i not in want:
                continue
            self.store[i] = {}
            self.handles.append(layer.register_forward_pre_hook(self._mk(i, "resid_in"), with_kwargs=False))
            self.handles.append(experts.register_forward_hook(self._mk_out(i, "raw")))
            self.handles.append(layer.post_feedforward_layernorm_2.register_forward_hook(self._mk_out(i, "norm2")))
            self.handles.append(layer.post_feedforward_layernorm_1.register_forward_hook(self._mk_out(i, "norm1")))

    def _mk(self, i, key):
        def hook(mod, args):
            x = args[0]
            self.store[i][key] = x.detach().reshape(-1, x.shape[-1]).to(torch.float32).cpu()
        return hook

    def _mk_out(self, i, key):
        def hook(mod, args, out):
            x = out[0] if isinstance(out, tuple) else out
            self.store[i][key] = x.detach().reshape(-1, x.shape[-1]).to(torch.float32).cpu()
        return hook

    def clear(self):
        for i in self.store:
            self.store[i] = {}

    def remove(self):
        for h in self.handles:
            h.remove()


def reference_pass(model_id, revision, tok, prompts, max_len, dev, cache_dir) -> dict:
    os.makedirs(cache_dir, exist_ok=True)
    t0 = time.time()
    ref = load_reference("gemma4", model_id, revision, dev)
    layers = moe_layers(ref)
    cap = Capture(layers, set(layers))
    stats = {i: {"resid_in": [], "raw": [], "norm2": [], "norm1": [], "tokens": 0} for i in layers}
    with torch.no_grad():
        for p in prompts:
            ids = tok(p["text"], return_tensors="pt", truncation=True, max_length=max_len)["input_ids"].to(dev)
            cap.clear()
            ref(input_ids=ids, use_cache=False)
            for i, d in cap.store.items():
                torch.save({k: d[k].to(torch.float16) for k in ("raw", "norm2")}, os.path.join(cache_dir, f"{p['id']}_L{i}.pt"))
                n = d["raw"].shape[0]
                for k in ("resid_in", "raw", "norm2", "norm1"):
                    stats[i][k].append((_rms(d[k]), n))
                stats[i]["tokens"] += n
    cap.remove()
    del ref
    gc.collect()
    torch.cuda.empty_cache()
    out = {}
    for i, s in stats.items():
        tw = {k: sum(r * r * n for r, n in v) / max(1, sum(n for _, n in v)) for k, v in s.items() if k != "tokens"}
        rms = {k: tw[k] ** 0.5 for k in tw}
        out[i] = {"rms_resid_in": rms["resid_in"], "rms_raw": rms["raw"], "rms_norm2": rms["norm2"], "rms_norm1": rms["norm1"],
                  "contrib_norm2_over_resid": rms["norm2"] / max(rms["resid_in"], 1e-30),
                  "norm_gain_norm2_over_raw": rms["norm2"] / max(rms["raw"], 1e-30),
                  "dense_over_moe": rms["norm1"] / max(rms["norm2"], 1e-30), "tokens": s["tokens"]}
    return {"layers": out, "wall_s": round(time.time() - t0, 1), "n_prompts": len(prompts)}


def arm_pass(family, arm, model_id, tok, prompts, max_len, dev, cache_dir, arena, calib) -> dict:
    t0 = time.time()
    builder = builder_for(family, arm)
    po = parse_only(builder)
    if not po:
        raise RuntimeError(f"{arm}: the activation probe reads one-layer builders only (got {builder!r})")
    layer_i, qt, bs = po
    apply_env(arm_env(family, arm, model_id))
    model, info = build_arm_model(family, arm, model_id, arena, calib, device=dev)
    if info.get("n_quantized") != 1:
        raise RuntimeError(f"{arm}: expected exactly one quantised stack, verify reports {info.get('n_quantized')}")
    layers = moe_layers(model)
    if layer_i not in layers:
        raise RuntimeError(f"{arm}: layer {layer_i} not found among the model's MoE layers {sorted(layers)[:5]}...")
    cap = Capture(layers, {layer_i})
    acc = {"raw_err2": 0.0, "raw_ref2": 0.0, "n2_err2": 0.0, "n2_ref2": 0.0, "tokens": 0, "cos_raw": []}
    with torch.no_grad():
        for p in prompts:
            ids = tok(p["text"], return_tensors="pt", truncation=True, max_length=max_len)["input_ids"].to(dev)
            cap.clear()
            model(input_ids=ids, use_cache=False)
            q = cap.store[layer_i]
            ref = torch.load(os.path.join(cache_dir, f"{p['id']}_L{layer_i}.pt"))
            for k, a, b in (("raw", q["raw"], ref["raw"].float()), ("norm2", q["norm2"], ref["norm2"].float())):
                if a.shape != b.shape:
                    raise RuntimeError(f"{arm}: {k} shape {tuple(a.shape)} vs reference {tuple(b.shape)} -- tokenizer or layer drift")
            acc["raw_err2"] += float((q["raw"] - ref["raw"].float()).pow(2).sum())
            acc["raw_ref2"] += float(ref["raw"].float().pow(2).sum())
            acc["n2_err2"] += float((q["norm2"] - ref["norm2"].float()).pow(2).sum())
            acc["n2_ref2"] += float(ref["norm2"].float().pow(2).sum())
            acc["cos_raw"].append(float(torch.nn.functional.cosine_similarity(q["raw"].reshape(1, -1), ref["raw"].float().reshape(1, -1))))
            acc["tokens"] += q["raw"].shape[0]
    cap.remove()
    del model
    gc.collect()
    torch.cuda.empty_cache()
    rel_raw = (acc["raw_err2"] / max(acc["raw_ref2"], 1e-30)) ** 0.5
    rel_n2 = (acc["n2_err2"] / max(acc["n2_ref2"], 1e-30)) ** 0.5
    return {"arm": arm, "builder": builder, "layer": layer_i, "quant_type": qt, "blocksize": bs, "engagement": info,
            "rel_err_raw": rel_raw, "rel_err_norm2": rel_n2, "amplification_norm2_over_raw": rel_n2 / max(rel_raw, 1e-30),
            "cos_raw_mean": sum(acc["cos_raw"]) / len(acc["cos_raw"]), "tokens": acc["tokens"], "wall_s": round(time.time() - t0, 1)}


def main() -> int:
    ap = argparse.ArgumentParser(description="P49 activation probe: per-layer expert-branch error, raw vs post-norm, vs the bf16 reference")
    ap.add_argument("--family", required=True, choices=[f for f in ARMS if f.startswith("gemma4")])
    ap.add_argument("--arms", default=None, help="comma list; default every registered arm of the family")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--max-len", type=int, default=320)
    ap.add_argument("--cache-dir", required=True)
    ap.add_argument("--arena", default="")
    ap.add_argument("--calib", default="")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    model_id, revision = MODELS[a.family]
    arms = [x for x in (a.arms or ",".join(ARMS[a.family])).split(",") if x]
    for k in LANE_KEYS:
        os.environ.pop(k, None)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_id, revision=revision)
    prompts = pick_prompts(a.n)
    dev = "cuda"
    rec = {"lane": "P49", "family": a.family, "model": model_id, "revision": revision, "n_prompts": len(prompts),
           "prompt_ids": [p["id"] for p in prompts], "max_len": a.max_len, "torch": torch.__version__,
           "gpu": torch.cuda.get_device_name(0), "reference": None, "arms": [], "not_measured": {}}
    def flush():
        json.dump(rec, open(a.out, "w"), indent=1)
    rec["reference"] = reference_pass(model_id, revision, tok, prompts, a.max_len, dev, a.cache_dir)
    flush()
    print(f"reference captured: {len(rec['reference']['layers'])} layers, {rec['reference']['wall_s']} s", flush=True)
    for i, s in sorted(rec["reference"]["layers"].items()):
        print(f"  L{i:02d} rms resid {s['rms_resid_in']:.3f} raw {s['rms_raw']:.4f} norm2 {s['rms_norm2']:.3f} norm1 {s['rms_norm1']:.3f} "
              f"contrib {s['contrib_norm2_over_resid']:.3f} gain {s['norm_gain_norm2_over_raw']:.1f} dense/moe {s['dense_over_moe']:.2f}", flush=True)
    for arm in arms:
        try:
            r = arm_pass(a.family, arm, model_id, tok, prompts, a.max_len, dev, a.cache_dir, a.arena, a.calib)
            rec["arms"].append(r)
            print(f"== {arm}: L{r['layer']} {r['quant_type']} b{r['blocksize']}  rel_raw {r['rel_err_raw']:.4f}  rel_norm2 {r['rel_err_norm2']:.4f}  "
                  f"amp {r['amplification_norm2_over_raw']:.2f}  cos {r['cos_raw_mean']:.4f} ({r['wall_s']} s)", flush=True)
        except Exception as e:
            rec["not_measured"][arm] = f"{type(e).__name__}: {str(e)[:600]}"
            print(f"== {arm}: NOT MEASURED -- {type(e).__name__}: {str(e)[:300]}", flush=True)
            gc.collect()
            torch.cuda.empty_cache()
        flush()
    return 0 if rec["arms"] and not rec["not_measured"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
