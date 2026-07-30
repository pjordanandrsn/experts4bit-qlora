#!/usr/bin/env python3.12
"""ONE matrix cell, then exit. Per docs/PREREG-flagship-matrix.md.

Why a subprocess per cell: the in-process driver leaked ~13 GB across cells and
OOMed at 22.41 GB, while the phase diagnostic showed a single clean cell peaks
at 9.13 GB and the eval loop accumulates nothing. Process exit is the only
VRAM free that is guaranteed, and it also means one cell's failure cannot
poison the next.

Costs ~4.3 min of reload per cell (the streaming NF4 quantize). That is the
price of isolation and it is worth it: the alternative is a matrix whose later
cells are measured under memory pressure the earlier ones did not have.
"""
import argparse, gc, hashlib, json, statistics, subprocess, threading, time
import torch

SEQ, R, ALPHA, LR = 512, 8, 16, 1e-4


class PowerSampler:
    def __init__(self):
        self.samples, self._run = [], False

    def _loop(self):
        while self._run:
            try:
                self.samples.append(float(subprocess.run(
                    ["nvidia-smi", "--query-gpu=power.draw", "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=5).stdout.strip().split("\n")[0]))
            except Exception:
                pass
            time.sleep(0.2)

    def __enter__(self):
        self._run = True
        self.t = threading.Thread(target=self._loop, daemon=True); self.t.start(); return self

    def __exit__(self, *a):
        self._run = False; self.t.join(timeout=2)


def idle_power(n=10):
    v = []
    for _ in range(n):
        try:
            v.append(float(subprocess.run(
                ["nvidia-smi", "--query-gpu=power.draw", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5).stdout.strip().split("\n")[0]))
        except Exception:
            pass
        time.sleep(0.15)
    return statistics.median(v) if v else 0.0


def expert_hashes(model):
    h = {}
    for name, m in model.named_modules():
        for attr in ("gate_up_proj", "down_proj", "gate_up_absmax", "down_absmax"):
            t = getattr(m, attr, None)
            if t is None:
                continue
            h[f"{name}.{attr}"] = hashlib.sha256(
                t.detach().to("cpu").contiguous().numpy().tobytes()).hexdigest()
    return h


def encode(tok, rows, seq):
    out = []
    for r in rows:
        ids = tok(f"### Instruction:\n{r['instruction']}\n\n### Response:\n{r['output']}",
                  return_tensors="pt", truncation=True, max_length=seq).input_ids[0]
        if ids.numel() >= 8:
            out.append(ids)
    return out


@torch.no_grad()
def eval_loss(model, data, limit=48):
    model.eval()
    tot = n = 0
    for ids in data[:limit]:
        x = ids.unsqueeze(0).cuda()
        tot += float(model(input_ids=x, labels=x).loss); n += 1
    model.train()
    return tot / max(n, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--arm", required=True, choices=["reference", "fused"])
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    from experts4bit_qlora.loader import load_moe_4bit_streaming
    from experts4bit_qlora.fast import enable_fast_train, disable_fast_train
    from experts4bit_qlora.lora import add_attention_lora
    from transformers import AutoTokenizer
    import importlib.metadata as md

    idle_w = idle_power()
    torch.manual_seed(0)
    t_load = time.time()
    model, _ = load_moe_4bit_streaming(a.model, "cuda", torch.bfloat16, R, ALPHA,
                                       offload=True, pin=True, prefetch=False,
                                       quant_type="nf4")
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.config.use_cache = False
    load_s = time.time() - t_load

    tok = AutoTokenizer.from_pretrained(a.model)
    ds = json.load(open(f"/workspace/n9/data/ds_{a.dataset}.json"))
    train, ev = encode(tok, ds["train"], SEQ), encode(tok, ds["eval"], SEQ)

    add_attention_lora(model, R, ALPHA, torch.float32)
    n_fused = enable_fast_train(model, verbose=False) if a.arm == "fused" else disable_fast_train(model)

    h_before = expert_hashes(model)
    ev0 = eval_loss(model, ev)

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=LR)
    model.train()
    gc.collect(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()

    losses, tokens = [], 0
    with PowerSampler() as ps:
        torch.cuda.synchronize(); t0 = time.time()
        for i in range(a.steps):
            ids = train[i % len(train)].unsqueeze(0).cuda()
            out = model(input_ids=ids, labels=ids)
            out.loss.backward(); opt.step(); opt.zero_grad(set_to_none=True)
            losses.append(round(float(out.loss), 5)); tokens += ids.numel()
            if (i + 1) % 50 == 0:
                print(f"    step {i+1}/{a.steps} loss {losses[-1]}", flush=True)
        torch.cuda.synchronize(); wall = time.time() - t0

    ev1 = eval_loss(model, ev)
    h_after = expert_hashes(model)
    changed = [k for k in h_before if h_before[k] != h_after.get(k)]
    mean_w = statistics.mean(ps.samples) if ps.samples else None
    net_w = (mean_w - idle_w) if mean_w else None

    cell = {
        "model": a.model, "dataset": a.dataset, "arm": a.arm, "steps": a.steps,
        "gnf4": md.version("grouped-nf4-gemm"), "e4b": md.version("experts4bit-qlora"),
        "gpu": torch.cuda.get_device_name(0),
        "cap": list(torch.cuda.get_device_capability()),
        "fused_modules": n_fused,
        "B1_experts_hashed": len(h_before), "B1_experts_changed": len(changed),
        "B1_bit_exact": len(changed) == 0,
        "B1_changed_sample": changed[:3],
        "loss_first": losses[0], "loss_last": losses[-1],
        "loss_mean_last20": round(statistics.mean(losses[-20:]), 5),
        "eval_loss_step0": round(ev0, 5), "eval_loss_final": round(ev1, 5),
        "eval_rel_improvement": round(ev1 / ev0, 5) if ev0 else None,
        "load_s": round(load_s, 1),
        "s_per_step": round(wall / a.steps, 4),
        "tokens_per_s": round(tokens / wall, 1),
        "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 1e9, 3),
        "idle_w": round(idle_w, 1),
        "mean_w": round(mean_w, 1) if mean_w else None,
        "power_samples": len(ps.samples),
        "joules_per_step": round(net_w * (wall / a.steps), 2) if net_w else None,
        "losses": losses,
    }
    json.dump(cell, open(a.out, "w"), indent=1)
    print("CELL OK " + json.dumps({k: v for k, v in cell.items() if k != "losses"}), flush=True)


if __name__ == "__main__":
    main()
