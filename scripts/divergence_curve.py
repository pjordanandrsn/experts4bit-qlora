"""Divergence-onset probe (final gate) — flip-annotated trajectory divergence curve.

Three legs from byte-identical initial state (seed, first-batch order, fresh AdamW, LoRA B=0):
  A = resident twin 1, B = resident twin 2 (the run-to-run null), C = offload.
Per step, logs vs twin-A: adapter-weight L2 divergence and routed-set flip count (symmetric
difference vs A's routing that step). Separates the two live hypotheses:
  - A vs B diverges like A vs C  → nondeterministic accumulation; placement incidental.
  - A vs B stays ~0 while A vs C grows → placement (staging/scheduling) causes divergence.
D3 boundary: step 1 must be bitwise across all legs (it was, in the grid logs).

Writes result.json with per-step curves + first-divergence/first-flip steps per leg.

Example:
  QUANT_TYPE=bf16 python scripts/divergence_curve.py --job-dir runs/jobs/divergence_bf16 --steps 60
"""

import argparse
import json
import os
import sys
from datetime import datetime, timezone


def build_batches(n, seq, n_train, grad_accum):
    import experts4bit_qlora.train as train
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(train.MODEL)
    data = train.encode_alpaca(tok, f"train[:{n_train}]")
    it = iter(data)
    batches = []
    for _ in range(n * grad_accum):
        try:
            ex = next(it)
        except StopIteration:
            it = iter(data)
            ex = next(it)
        batches.append({"input_ids": list(ex["input_ids"]), "labels": list(ex["labels"])})
    return batches


def run_leg(quant_type, offload, seed, steps, seq, grad_accum, lr, batches, subset_idx):
    """Run one leg; per step, keep only a fixed random SUBSET of the flat LoRA weights (the
    same ``subset_idx`` across all legs), fp32, in a RAM list. rev1 OOM'd on full fp64
    snapshots; rev2 OOM'd on a network-FS memmap whose page cache charged the cgroup. A fixed
    random subset (unbiased sample) preserves divergence onset / growth shape / A-vs-B-vs-C
    ratio at ~0.5 GB/leg. Returns (subset_snapshots [steps, K] fp32, routed_sets, losses)."""
    import numpy as np
    import torch
    from experts4bit_qlora.loader import load_moe_4bit_streaming
    from experts4bit_qlora.lora import add_attention_lora, ExpertsLoRA
    import experts4bit_qlora.train as train
    from transformers import get_cosine_schedule_with_warmup

    torch.manual_seed(seed)
    model, _ = load_moe_4bit_streaming(train.MODEL, "cuda", torch.bfloat16, 8, 16,
                                       offload=offload, pin=True, quant_type=quant_type)
    if not offload:
        model.to("cuda")
    add_attention_lora(model, 8, 16, torch.bfloat16)
    lora = [(n, p) for n, p in model.named_parameters()
            if "lora" in n and ("experts" in n or "self_attn" in n)]
    for n, p in model.named_parameters():
        p.requires_grad_("lora" in n and ("experts" in n or "self_attn" in n))
    params = [p for _, p in lora]
    opt = torch.optim.AdamW([{"params": params, "lr": lr}], lr=lr)
    sched = get_cosine_schedule_with_warmup(opt, num_warmup_steps=5, num_training_steps=150)
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    model.train()

    # capture routing on the FIRST micro-batch of each step, layer-by-layer
    cur_routed = {}
    handles = []
    for li, mod in enumerate(m for m in model.modules() if isinstance(m, ExpertsLoRA)):
        def mk(idx):
            def hook(module, args):
                if module.training:
                    tki = args[1]
                    cur_routed.setdefault(idx, set()).update(int(x) for x in tki.reshape(-1).tolist())
            return hook
        handles.append(mod.register_forward_pre_hook(mk(li)))

    idx_t = torch.as_tensor(subset_idx, device="cuda")
    snaps = np.empty((steps, len(subset_idx)), dtype=np.float32)
    routed_per_step, losses = [], []
    bi = 0
    for step in range(steps):
        opt.zero_grad()
        cur_routed.clear()
        step_routed = None
        lacc = 0.0
        for g in range(grad_accum):
            ex = batches[bi]
            bi += 1
            ids = torch.tensor([ex["input_ids"]], device="cuda")
            lbl = torch.tensor([ex["labels"]], device="cuda")
            out = model(input_ids=ids, labels=lbl)
            if g == 0:
                step_routed = {k: set(v) for k, v in cur_routed.items()}
            (out.loss / grad_accum).backward()
            lacc += out.loss.item() / grad_accum
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        sched.step()
        losses.append(lacc)
        routed_per_step.append(step_routed)
        flat = torch.cat([p.detach().float().reshape(-1) for _, p in lora])  # on GPU
        snaps[step] = flat[idx_t].cpu().numpy()  # keep only the K-subset
        del flat
    for h in handles:
        h.remove()
    del model, opt, sched
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    return snaps, routed_per_step, losses


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--job-dir", required=True)
    ap.add_argument("--quant-type", default="bf16")
    ap.add_argument("--steps", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--seq", type=int, default=256)
    ap.add_argument("--n-train", type=int, default=10000)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--subset", type=int, default=2_000_000, help="fixed random weight-subset size (RAM-bounded)")
    args = ap.parse_args()
    os.makedirs(args.job_dir, exist_ok=True)
    os.environ["SEQ"] = str(args.seq)
    os.environ["N_TRAIN"] = str(args.n_train)

    import numpy as np
    import torch  # noqa
    batches = build_batches(args.steps, args.seq, args.n_train, args.grad_accum)

    # Fixed random subset of LoRA params, SAME across legs (seeded, independent of the leg
    # torch.manual_seed). LoRA count for OLMoE r=8 = 60,817,408; sample K=2M.
    K = min(args.subset, 60_817_408)
    rng = np.random.default_rng(12345)
    subset_idx = np.sort(rng.choice(60_817_408, size=K, replace=False)).astype(np.int64)

    snaps = {}
    routed = {}
    losses = {}
    for tag, offload in (("A_resident", False), ("B_resident", False), ("C_offload", True)):
        print(f"[leg {tag}] offload={offload}", flush=True)
        s, r, lo = run_leg(args.quant_type, offload, args.seed, args.steps, args.seq,
                           args.grad_accum, args.lr, batches, subset_idx)
        snaps[tag] = s
        routed[tag] = r
        losses[tag] = lo

    def weight_div(a, b):  # scaled L2 on the K-subset (unbiased estimate of full L2 up to sqrt(N/K))
        return [float(np.linalg.norm(a[k].astype(np.float64) - b[k].astype(np.float64)))
                for k in range(a.shape[0])]

    def flip_curve(x, y):
        return [sum(len(x[k].get(li, set()) ^ y[k].get(li, set())) for li in set(x[k]) | set(y[k]))
                for k in range(len(x))]

    ab_w = weight_div(snaps["A_resident"], snaps["B_resident"])
    ac_w = weight_div(snaps["A_resident"], snaps["C_offload"])
    ab_f = flip_curve(routed["A_resident"], routed["B_resident"])
    ac_f = flip_curve(routed["A_resident"], routed["C_offload"])

    def first_nonzero(v):
        for i, x in enumerate(v):
            if x > 0:
                return i + 1  # 1-indexed step
        return None

    result = {
        "job_type": "divergence_curve", "status": "pass",
        "storage_mode": args.quant_type, "seed": args.seed, "steps": args.steps,
        "subset_K": int(K), "n_params_total": 60_817_408,
        "weight_div_note": "L2 on a fixed random K-subset; multiply by sqrt(60817408/K) for a full-vector estimate",
        "loss_A": losses["A_resident"], "loss_B": losses["B_resident"], "loss_C": losses["C_offload"],
        "weight_div_AB": ab_w, "weight_div_AC": ac_w,
        "flip_AB": ab_f, "flip_AC": ac_f,
        "first_weight_div_step_AB": first_nonzero(ab_w),
        "first_weight_div_step_AC": first_nonzero(ac_w),
        "first_flip_step_AB": first_nonzero(ab_f),
        "first_flip_step_AC": first_nonzero(ac_f),
        "final_weight_div_AB": ab_w[-1], "final_weight_div_AC": ac_w[-1],
        "torch_version": torch.__version__, "cuda_version": torch.version.cuda,
        "gpu_name": torch.cuda.get_device_name(0),
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    try:
        import bitsandbytes
        result["bitsandbytes_version"] = bitsandbytes.__version__
    except Exception:
        result["bitsandbytes_version"] = None
    with open(os.path.join(args.job_dir, "result.json"), "w") as f:
        json.dump(result, f, indent=2, sort_keys=True)
        f.write("\n")
    print(f"first weight-div step: A-vs-B {result['first_weight_div_step_AB']}, "
          f"A-vs-C {result['first_weight_div_step_AC']}")
    print(f"first flip step: A-vs-B {result['first_flip_step_AB']}, A-vs-C {result['first_flip_step_AC']}")
    print(f"final weight L2 div: A-vs-B {ab_w[-1]:.3e}, A-vs-C {ac_w[-1]:.3e}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
