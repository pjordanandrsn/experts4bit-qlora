#!/usr/bin/env python3.12
"""Flagship matrix per docs/PREREG-flagship-matrix.md (registered b08747f/b33c553).

Per cell: model x dataset x arm(reference|fused).
Measures B1 bit-exactness, B2 loss parity, B3 cost (s/step, tok/s, peak VRAM,
J/step). Receipts written after EVERY cell so a death mid-matrix keeps what ran.
"""
import gc, hashlib, json, os, statistics, subprocess, threading, time, traceback
import torch

OUT = "/workspace/n9/n9_matrix.json"
DS_DIR = "/workspace/n9/data"
STEPS = int(os.environ.get("N9_STEPS", "200"))
SEQ, R, ALPHA, LR = 512, 8, 16, 1e-4
RES = {"prereg": "docs/PREREG-flagship-matrix.md", "steps": STEPS, "cells": {}}


def save():
    json.dump(RES, open(OUT, "w"), indent=1, default=str)


class PowerSampler:
    """Board power at 200 ms, per the prereg. Idle baseline subtracted."""
    def __init__(self):
        self.samples, self._run = [], False

    def _loop(self):
        while self._run:
            try:
                w = subprocess.run(
                    ["nvidia-smi", "--query-gpu=power.draw", "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=5).stdout.strip().split("\n")[0]
                self.samples.append(float(w))
            except Exception:
                pass
            time.sleep(0.2)

    def __enter__(self):
        self._run = True
        self.t = threading.Thread(target=self._loop, daemon=True)
        self.t.start()
        return self

    def __exit__(self, *a):
        self._run = False
        self.t.join(timeout=2)

    def stats(self, idle_w):
        if not self.samples:
            return None
        return {"mean_w": round(statistics.mean(self.samples), 1),
                "max_w": round(max(self.samples), 1),
                "n_samples": len(self.samples),
                "idle_w": round(idle_w, 1)}


def idle_power(n=10):
    vals = []
    for _ in range(n):
        try:
            vals.append(float(subprocess.run(
                ["nvidia-smi", "--query-gpu=power.draw", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5).stdout.strip().split("\n")[0]))
        except Exception:
            pass
        time.sleep(0.15)
    return statistics.median(vals) if vals else 0.0


def expert_hashes(model):
    """B1: SHA-256 of every frozen expert's packed bytes."""
    h = {}
    for name, m in model.named_modules():
        for attr in ("gate_up_proj", "down_proj", "gate_up_absmax", "down_absmax"):
            t = getattr(m, attr, None)
            if t is None:
                continue
            b = t.detach().to("cpu").contiguous().numpy().tobytes()
            h[f"{name}.{attr}"] = hashlib.sha256(b).hexdigest()
    return h


def encode(tok, rows, seq):
    out = []
    for r in rows:
        text = f"### Instruction:\n{r['instruction']}\n\n### Response:\n{r['output']}"
        ids = tok(text, return_tensors="pt", truncation=True,
                  max_length=seq).input_ids[0]
        if ids.numel() >= 8:
            out.append(ids)
    return out


@torch.no_grad()
def eval_loss(model, data, limit=48):
    model.eval()
    tot, n = 0.0, 0
    for ids in data[:limit]:
        x = ids.unsqueeze(0).cuda()
        tot += float(model(input_ids=x, labels=x).loss)
        n += 1
    model.train()
    return tot / max(n, 1)


def run_cell(model_id, loader, ds_name, arm, idle_w):
    from experts4bit_qlora.fast import enable_fast_train, disable_fast_train
    from experts4bit_qlora.lora import add_attention_lora
    from transformers import AutoTokenizer

    torch.manual_seed(0)
    gc.collect(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()

    model, _ = loader(model_id, "cuda", torch.bfloat16, R, ALPHA,
                      offload=True, pin=True, prefetch=False, quant_type="nf4")
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.config.use_cache = False
    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    ds = json.load(open(f"{DS_DIR}/ds_{ds_name}.json"))
    train = encode(tok, ds["train"], SEQ)
    ev = encode(tok, ds["eval"], SEQ)

    add_attention_lora(model, R, ALPHA, torch.float32)
    n_fused = enable_fast_train(model, verbose=False) if arm == "fused" else 0
    if arm == "reference":
        disable_fast_train(model)

    h_before = expert_hashes(model)
    ev0 = eval_loss(model, ev)

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=LR)
    model.train()
    torch.cuda.reset_peak_memory_stats()

    losses, tokens = [], 0
    with PowerSampler() as ps:
        torch.cuda.synchronize(); t0 = time.time()
        for i in range(STEPS):
            ids = train[i % len(train)].unsqueeze(0).cuda()
            out = model(input_ids=ids, labels=ids)
            out.loss.backward()
            opt.step(); opt.zero_grad(set_to_none=True)
            losses.append(round(float(out.loss), 5))
            tokens += ids.numel()
            if (i + 1) % 50 == 0:
                print(f"    [{ds_name}/{arm}] step {i+1}/{STEPS} loss {losses[-1]}", flush=True)
        torch.cuda.synchronize(); wall = time.time() - t0

    ev1 = eval_loss(model, ev)
    h_after = expert_hashes(model)
    changed = [k for k in h_before if h_before[k] != h_after.get(k)]
    pw = ps.stats(idle_w)
    net_w = (pw["mean_w"] - idle_w) if pw else None

    cell = {
        "model": model_id, "dataset": ds_name, "arm": arm,
        "fused_modules": n_fused,
        "B1_experts_hashed": len(h_before),
        "B1_experts_changed": len(changed),
        "B1_bit_exact": len(changed) == 0,
        "loss_first": losses[0], "loss_last": losses[-1],
        "loss_mean_last20": round(statistics.mean(losses[-20:]), 5),
        "eval_loss_step0": round(ev0, 5), "eval_loss_final": round(ev1, 5),
        "eval_rel_improvement": round(ev1 / ev0, 5) if ev0 else None,
        "s_per_step": round(wall / STEPS, 4),
        "tokens_per_s": round(tokens / wall, 1),
        "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 1e9, 3),
        "power": pw,
        "joules_per_step": round(net_w * (wall / STEPS), 2) if net_w else None,
        "losses": losses,
    }
    del model, opt, params
    gc.collect(); torch.cuda.empty_cache()
    return cell


def main():
    from experts4bit_qlora.loader import load_moe_4bit_streaming
    import importlib.metadata as md

    RES["env"] = {"gpu": torch.cuda.get_device_name(0),
                  "cap": list(torch.cuda.get_device_capability()),
                  "torch": torch.__version__,
                  "gnf4": md.version("grouped-nf4-gemm"),
                  "e4b": md.version("experts4bit-qlora")}
    idle_w = idle_power()
    RES["idle_power_w"] = round(idle_w, 1)
    save()
    print(json.dumps(RES["env"], indent=1), f"idle {idle_w:.1f} W", flush=True)

    models = json.load(open("/workspace/n9/models.json"))
    datasets = ["clinical", "code", "finance", "legal", "support"]

    for model_id in models:
        for ds in datasets:
            for arm in ("reference", "fused"):
                key = f"{model_id.split('/')[-1]}|{ds}|{arm}"
                if key in RES["cells"]:
                    continue
                print(f"\n=== {key} ===", flush=True)
                t0 = time.time()
                try:
                    RES["cells"][key] = run_cell(model_id, load_moe_4bit_streaming,
                                                 ds, arm, idle_w)
                    c = RES["cells"][key]
                    print(f"  ok  B1_exact={c['B1_bit_exact']} "
                          f"loss {c['loss_first']}->{c['loss_last']} "
                          f"eval {c['eval_loss_step0']}->{c['eval_loss_final']} "
                          f"{c['s_per_step']}s/step {c['joules_per_step']}J/step "
                          f"{c['peak_vram_gb']}GB", flush=True)
                except Exception as e:
                    RES["cells"][key] = {"ok": False, "err": f"{type(e).__name__}: {e}",
                                         "tb": traceback.format_exc()[-1500:],
                                         "wall_s": round(time.time() - t0, 1)}
                    print(f"  FAILED: {RES['cells'][key]['err']}", flush=True)
                    gc.collect(); torch.cuda.empty_cache()
                save()

    save()
    print("\nMATRIX COMPLETE", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        RES["fatal"] = traceback.format_exc()[-2000:]
        save()
        raise
