#!/usr/bin/env python3
"""Measured inference + training matrix across MoE and dense architectures.

For every (architecture x dtype) cell this builds a REAL module tree of that
architecture (small, so it runs anywhere), invents a checkpoint matching it,
plans and executes the load, then measures both halves:

  inference — a forward pass; records logits finiteness and tokens/sec;
  training  — LoRA adapters on the attention projections, then N optimizer
              steps; records gradient finiteness and whether the loss actually
              moved.

The point is numbers, not assertions. A cell that fails prints WHY and the run
continues, because "which configurations do not work" is the useful half of a
compatibility matrix. Emits a markdown table on stdout.

    python3 tools/moe_matrix.py                 # default matrix, CPU
    python3 tools/moe_matrix.py --device cuda   # same on a GPU
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import traceback

import torch

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))

from experts4bit_qlora.moe_load import execute_moe_plan  # noqa: E402
from experts4bit_qlora.moe_plan import plan_moe_checkpoint  # noqa: E402

# (label, model_type, is_moe, extra config kwargs)
ARCHITECTURES = [
    ("qwen3_moe", "qwen3_moe", True,
     dict(moe_intermediate_size=32, num_experts=4, num_experts_per_tok=2,
          decoder_sparse_step=1)),
    ("mixtral", "mixtral", True,
     dict(num_local_experts=4, num_experts_per_tok=2)),
    ("phimoe", "phimoe", True,
     dict(num_local_experts=4, num_experts_per_tok=2)),
    ("olmoe", "olmoe", True,
     dict(num_experts=4, num_experts_per_tok=2)),
    ("qwen3 (dense)", "qwen3", False, {}),
    ("llama (dense)", "llama", False, {}),
    ("mistral (dense)", "mistral", False, {}),
]

BASE = dict(hidden_size=32, intermediate_size=64, num_hidden_layers=2,
            num_attention_heads=4, num_key_value_heads=2, head_dim=8,
            vocab_size=64, tie_word_embeddings=False)


def build(model_type, extra):
    from transformers import AutoConfig, AutoModelForCausalLM
    cfg = AutoConfig.for_model(model_type, **{**BASE, **extra})
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(cfg)
    return model, cfg


def fake_checkpoint(model, cfg, is_moe):
    keys, store = [], {}
    for name, t in model.state_dict().items():
        if name.endswith("experts.gate_up_proj") or name.endswith("experts.down_proj"):
            continue
        if "rotary_emb" in name or name.endswith("inv_freq"):
            continue
        keys.append(name)
        store[name] = torch.randn(tuple(t.shape)) * 0.02
    if not is_moe:
        return keys, store
    n_exp = getattr(cfg, "num_experts", None) or getattr(cfg, "num_local_experts")
    inter = getattr(cfg, "moe_intermediate_size", None) or cfg.intermediate_size
    hid = cfg.hidden_size
    for layer in range(cfg.num_hidden_layers):
        for e in range(n_exp):
            if cfg.model_type in ("mixtral", "phimoe"):
                spec = [("w1", (inter, hid)), ("w3", (inter, hid)), ("w2", (hid, inter))]
                base = f"model.layers.{layer}.block_sparse_moe.experts.{e}"
            else:
                spec = [("gate_proj", (inter, hid)), ("up_proj", (inter, hid)),
                        ("down_proj", (hid, inter))]
                base = f"model.layers.{layer}.mlp.experts.{e}"
            for proj, shape in spec:
                k = f"{base}.{proj}.weight"
                keys.append(k)
                store[k] = torch.randn(shape) * 0.02
    return keys, store


def attach_lora(model, rank=4, dtype=torch.float32, device="cpu"):
    """Minimal LoRA on every attention projection — enough to prove gradients
    flow through a loaded model without pulling in peft."""
    n = 0
    for name, mod in model.named_modules():
        if not isinstance(mod, torch.nn.Linear):
            continue
        if ".self_attn." not in name:
            continue
        a = torch.nn.Parameter(torch.randn(rank, mod.in_features, dtype=dtype,
                                           device=device) * 0.01)
        b = torch.nn.Parameter(torch.zeros(mod.out_features, rank, dtype=dtype,
                                           device=device))
        mod.register_parameter("lora_a", a)
        mod.register_parameter("lora_b", b)
        base_fwd = mod.forward

        def fwd(x, _m=mod, _f=base_fwd):
            return _f(x) + torch.nn.functional.linear(
                torch.nn.functional.linear(x, _m.lora_a), _m.lora_b)
        mod.forward = fwd
        n += 1
    return n


def run_cell(label, model_type, is_moe, extra, dtype, device, steps=25):
    out = {"arch": label, "dtype": str(dtype).replace("torch.", ""),
           "load": "-", "infer": "-", "tok_s": "-", "train": "-",
           "loss0": "-", "loss_n": "-", "drop": "-", "note": ""}
    try:
        model, cfg = build(model_type, extra)
        keys, store = fake_checkpoint(model, cfg, is_moe)
        plan = plan_moe_checkpoint(keys, model, model_type, dense_ok=not is_moe)
        execute_moe_plan(plan, model, store.__getitem__, device=device, dtype=dtype)
        out["load"] = f"OK ({len(plan.passthrough)}p/{plan.n_expert_stacks}s)"
    except Exception as e:
        out["load"] = "FAIL"
        out["note"] = f"{type(e).__name__}: {str(e)[:70]}"
        return out

    ids = torch.randint(0, cfg.vocab_size, (1, 16), device=device)
    try:
        model.eval()
        with torch.no_grad():
            t0 = time.time()
            logits = model(ids).logits
            dt = time.time() - t0
        out["infer"] = "OK" if torch.isfinite(logits).all() else "NON-FINITE"
        out["tok_s"] = f"{ids.numel()/dt:.0f}"
    except Exception as e:
        out["infer"] = "FAIL"
        out["note"] = f"{type(e).__name__}: {str(e)[:70]}"
        return out

    try:
        n_lora = attach_lora(model, dtype=dtype, device=device)
        params = [p for n_, p in model.named_parameters() if "lora_" in n_]
        for p in model.parameters():
            p.requires_grad_(False)
        for p in params:
            p.requires_grad_(True)
        opt = torch.optim.AdamW(params, lr=1e-2)
        model.train()
        losses = []
        for _ in range(steps):
            o = model(ids, labels=ids)
            o.loss.backward()
            if not all(torch.isfinite(p.grad).all() for p in params if p.grad is not None):
                out["train"] = "NON-FINITE GRAD"
                return out
            opt.step()
            opt.zero_grad(set_to_none=True)
            losses.append(float(o.loss.detach()))
        # A stepping optimizer with finite grads proves PLUMBING. Training is
        # only demonstrated if the loss actually falls, so that is the verdict.
        drop = losses[0] - losses[-1]
        learned = drop > 1e-3
        out["train"] = (f"OK ({n_lora} adapters)" if learned
                        else f"NO-LEARN ({n_lora} adapters)")
        out["loss0"], out["loss_n"] = f"{losses[0]:.4f}", f"{losses[-1]:.4f}"
        out["drop"] = f"{drop:+.4f}"
    except Exception as e:
        out["train"] = "FAIL"
        out["note"] = f"{type(e).__name__}: {str(e)[:70]}"
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--dtypes", default="float32,bfloat16")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()
    dtypes = [getattr(torch, d.strip()) for d in a.dtypes.split(",")]
    rows = []
    for label, mt, is_moe, extra in ARCHITECTURES:
        for dt in dtypes:
            rows.append(run_cell(label, mt, is_moe, extra, dt, a.device))
            print(f"  ran {label:16s} {str(dt):16s} -> {rows[-1]['load']:14s} "
                  f"infer={rows[-1]['infer']:6s} train={rows[-1]['train']}",
                  file=sys.stderr)
    if a.json:
        print(json.dumps(rows, indent=1))
        return
    cols = ["arch", "dtype", "load", "infer", "tok_s", "train", "loss0", "loss_n",
            "drop", "note"]
    print("| " + " | ".join(cols) + " |")
    print("|" + "|".join("---" for _ in cols) + "|")
    for r in rows:
        print("| " + " | ".join(str(r[c]) for c in cols) + " |")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
