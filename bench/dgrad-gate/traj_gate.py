"""Real-data trajectory gate: does any accelerated lane train DIFFERENTLY?

The composed-gradient campaign (bench/dgrad-gate/) showed every lane on the bf16 noise
floor at step 0. This is the complementary instrument: 20 optimizer steps on REAL text
(Alpaca — the dataset this repo's own METHODOLOGY eval uses), scored on a held-out slice
the optimizer never sees, one model load, adapter snapshot/restore between arms so every
arm starts bit-identical and sees identical data in identical order. The protocol and the
0.05 median-|Δ| train band are bench/fused-train-gate's.

Arms: reference, fast_train+dgrad (the new default), batched (the fallback).
"""
import argparse
import json
import time

import torch


def trainable_snapshot(model):
    return {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}


def restore(model, snap):
    with torch.no_grad():
        for n, p in model.named_parameters():
            if n in snap:
                p.copy_(snap[n])


def build_batches(tok, model_id, n_train, n_eval, seq, device):
    from datasets import load_dataset

    ds = load_dataset("tatsu-lab/alpaca", split="train")
    # Fixed slices, fixed order: the comparison is arm-vs-arm, and any shuffling
    # would have to be identical across arms anyway — so there is none.
    def fmt(row):
        ins = row["instruction"] + (("\n\n" + row["input"]) if row["input"] else "")
        return f"### Instruction:\n{ins}\n\n### Response:\n{row['output']}"

    def encode(rows):
        out = []
        for r in rows:
            ids = tok(fmt(r), truncation=True, max_length=seq,
                      return_tensors="pt")["input_ids"]
            if ids.shape[1] >= 16:                 # skip degenerate rows
                out.append(ids.to(device))
        return out

    train = encode([ds[i] for i in range(0, n_train * 2)])[:n_train]
    heldout = encode([ds[i] for i in range(5000, 5000 + n_eval * 2)])[:n_eval]
    assert len(train) == n_train and len(heldout) == n_eval
    return train, heldout


@torch.no_grad()
def eval_loss(model, batches):
    model.eval()
    tot = 0.0
    for ids in batches:
        tot += float(model(input_ids=ids, labels=ids).loss)
    model.train()
    return tot / len(batches)


def run_arm(model, name, train_b, eval_b, lr, enable, disable):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    n_patched = enable(model) if enable else 0
    before = eval_loss(model, eval_b)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr)
    losses, t0 = [], time.time()
    for ids in train_b:
        opt.zero_grad(set_to_none=True)
        out = model(input_ids=ids, labels=ids)
        out.loss.backward()
        opt.step()
        losses.append(float(out.loss.detach()))
    torch.cuda.synchronize()
    dt = (time.time() - t0) / len(train_b)
    after = eval_loss(model, eval_b)
    if disable:
        disable(model)
    return dict(arm=name, n_patched=n_patched, eval_before=before, eval_after=after,
                losses=losses, s_per_step=round(dt, 3),
                train_peak_gb=round(torch.cuda.max_memory_allocated() / 2**30, 2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-30B-A3B")
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--eval-n", type=int, default=32)
    ap.add_argument("--seq", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--out", default="traj_gate.json")
    a = ap.parse_args()

    import experts4bit_qlora as e4b
    from experts4bit_qlora import (disable_batched_train, disable_fast_train,
                                   enable_batched_train, enable_fast_train,
                                   load_moe_4bit_streaming)
    from transformers import AutoTokenizer

    env = dict(gpu=torch.cuda.get_device_name(0), torch=torch.__version__,
               e4b=e4b.__version__)
    print("env:", json.dumps(env), flush=True)
    tok = AutoTokenizer.from_pretrained(a.model, trust_remote_code=True)
    model, cfg = load_moe_4bit_streaming(a.model, "cuda", torch.bfloat16,
                                         r=8, alpha=16, quant_type="nf4", offload=False)
    model.to("cuda")
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.train()
    for n, p in model.named_parameters():
        p.requires_grad_("lora" in n and "experts" in n)

    train_b, eval_b = build_batches(tok, a.model, a.steps, a.eval_n, a.seq, "cuda")
    print(f"train {len(train_b)} steps | heldout {len(eval_b)}", flush=True)
    snap = trainable_snapshot(model)

    ARMS = [
        ("reference", None, None),
        ("fast_train_dgrad", lambda m: enable_fast_train(m, dgrad=True), disable_fast_train),
        ("batched", enable_batched_train, disable_batched_train),
    ]
    results = {}
    for name, en, dis in ARMS:
        restore(model, snap)
        print(f"--- arm {name} ---", flush=True)
        results[name] = run_arm(model, name, train_b, eval_b, a.lr, en, dis)
        print(json.dumps({k: v for k, v in results[name].items() if k != "losses"}),
              flush=True)

    ref = results["reference"]
    print("\n=== SUMMARY (band: train median |dL| <= 0.05, per fused-train-gate) ===")
    print(f"{'arm':<20}{'s/step':>8}{'eval before':>13}{'eval after':>12}"
          f"{'Δeval vs ref':>14}{'train med|Δ|':>14}")
    for k, v in results.items():
        if k == "reference":
            print(f"{k:<20}{v['s_per_step']:>8.2f}{v['eval_before']:>13.4f}"
                  f"{v['eval_after']:>12.4f}{'—':>14}{'—':>14}")
            continue
        dl = sorted(abs(x - y) for x, y in zip(v["losses"], ref["losses"]))
        med = dl[len(dl) // 2]
        v["train_median_abs_delta"] = med
        v["eval_after_delta_vs_ref"] = v["eval_after"] - ref["eval_after"]
        verdict = "PASS" if med <= 0.05 else "FAIL"
        print(f"{k:<20}{v['s_per_step']:>8.2f}{v['eval_before']:>13.4f}"
              f"{v['eval_after']:>12.4f}{v['eval_after_delta_vs_ref']:>+14.4f}"
              f"{med:>11.4f} {verdict}")
    with open(a.out, "w") as f:
        json.dump(dict(model=a.model, steps=a.steps, seq=a.seq, env=env,
                       dataset="tatsu-lab/alpaca", arms=results), f, indent=1)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
