"""One-step train placement certificate (debt D3, queue item T1.1/T1.2).

Runs three legs in one process, with full model teardown and identical re-seeding between
them: (a) resident step, (b) resident step repeat, (c) offload step — from identical state
(same seed, same first GRAD_ACCUM micro-batches in the trainer's sequential order, fresh
AdamW, LoRA B=0 by construction). Compares, in the order the decision tree demands
(docs/POST_AUDIT_WORK_QUEUE.md T1): forward losses/logits -> adapter grads -> post-step
adapter weights -> optimizer state. (a) vs (b) is the run-to-run null; (a) vs (c) is the
placement question. All comparisons are computed in-job and written to result.json; raw
tensors stay job-local (leg_*.pt).

Design notes (docs/TRAIN_PLACEMENT_CERTIFICATE.md):
- The production path has NO dropout and consumes no RNG after LoRA-A init (T1.0), so the
  handoff's dropout-ON leg is a counterfactual: --dropout forces attention dropout onto the
  loaded model (attribute-level, count attested in the result) to test the
  preserve_rng_state/recompute hypothesis anyway.
- --deterministic wraps the step in torch.use_deterministic_algorithms(warn_only=True) +
  math-only SDPA + cudnn.deterministic, and records which ops still warned. CUBLAS needs
  CUBLAS_WORKSPACE_CONFIG set before CUDA init — the script sets :4096:8 itself when
  --deterministic is passed (before importing torch).

Example:
    python scripts/one_step_certificate.py --job-dir runs/jobs/cert_olmoe_bf16_dropoutOFF_default \\
        --quant-type bf16 --seed 1337
"""

import argparse
import gc
import hashlib
import json
import os
import sys
import warnings
from datetime import datetime, timezone


def _sha(t) -> str:
    """SHA-256 of a tensor's raw bytes (native dtype — bitwise comparison, not float compare).

    The uint8 dtype-view is load-bearing: ``Tensor.numpy()`` rejects bf16 outright
    (rev1 red: runs/results/postaudit_cert_rev1_FAILED.md), and hashing raw bytes must not
    round-trip through a float cast anyway."""
    import torch

    return hashlib.sha256(
        t.detach().cpu().contiguous().view(-1).view(torch.uint8).numpy().tobytes()
    ).hexdigest()


def _utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def build_batches(seq, n_train, grad_accum):
    """First GRAD_ACCUM micro-batches, exactly the trainer's sequential order (no shuffle)."""
    import experts4bit_qlora.train as train

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(train.MODEL)
    data = train.encode_alpaca(tok, f"train[:{n_train}]")
    batches = []
    it = iter(data)
    for _ in range(grad_accum):
        ex = next(it)
        batches.append({"input_ids": list(ex["input_ids"]), "labels": list(ex["labels"])})
    return batches


def force_attention_dropout(model, p):
    """Counterfactual dropout-ON leg: set attention dropout on every module that carries the
    attribute (OLMoE attention reads self.attention_dropout at forward). Returns how many
    modules were touched — 0 means the leg is vacuous and the result says so."""
    n = 0
    for mod in model.modules():
        if hasattr(mod, "attention_dropout") and not hasattr(mod, "base"):
            mod.attention_dropout = p
            n += 1
    if hasattr(model, "config") and hasattr(model.config, "attention_dropout"):
        model.config.attention_dropout = p
    return n


def run_leg(tag, quant_type, offload, seed, seq, grad_accum, lr, batches, dropout_p, job_dir):
    """One full leg: seed -> load -> LoRA -> 1 optimizer step. Returns the artifact dict and
    saves raw tensors to <job_dir>/leg_<tag>.pt."""
    import torch

    from experts4bit_qlora.loader import load_moe_4bit_streaming
    from experts4bit_qlora.lora import add_attention_lora
    import experts4bit_qlora.train as train
    from transformers import get_cosine_schedule_with_warmup

    torch.manual_seed(seed)
    model, _ = load_moe_4bit_streaming(
        train.MODEL, "cuda", torch.bfloat16, 8, 16, offload=offload, pin=True, quant_type=quant_type
    )
    if not offload:
        model.to("cuda")
    add_attention_lora(model, 8, 16, torch.bfloat16)

    n_dropout_mods = force_attention_dropout(model, dropout_p) if dropout_p > 0 else 0

    lora_params = []
    for n, p in model.named_parameters():
        if "lora" in n and ("experts" in n or "self_attn" in n):
            p.requires_grad_(True)
            lora_params.append((n, p))
        else:
            p.requires_grad_(False)
    params = [p for _, p in lora_params]

    opt = torch.optim.AdamW([{"params": params, "lr": lr}], lr=lr)
    sched = get_cosine_schedule_with_warmup(opt, num_warmup_steps=5, num_training_steps=150)
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    model.train()

    losses, logits_mb0 = [], None
    opt.zero_grad()
    for i, ex in enumerate(batches):
        ids = torch.tensor([ex["input_ids"]], device="cuda")
        lbl = torch.tensor([ex["labels"]], device="cuda")
        out = model(input_ids=ids, labels=lbl)
        if i == 0:
            logits_mb0 = out.logits.detach().clone()
        (out.loss / grad_accum).backward()
        losses.append(float(out.loss.item()))
    grads = {n: p.grad.detach().clone() for n, p in lora_params if p.grad is not None}
    torch.nn.utils.clip_grad_norm_(params, 1.0)
    opt.step()
    sched.step()
    weights = {n: p.detach().clone() for n, p in lora_params}
    opt_state = {}
    for n, p in lora_params:
        st = opt.state.get(p, {})
        for k in ("exp_avg", "exp_avg_sq"):
            if k in st:
                opt_state[f"{n}.{k}"] = st[k].detach().clone()

    torch.cuda.synchronize()
    art = {
        "tag": tag,
        "offload": offload,
        "losses": losses,
        "n_dropout_modules": n_dropout_mods,
        "config_attention_dropout": float(getattr(model.config, "attention_dropout", -1.0)),
        "n_grad_tensors": len(grads),
        "logits_mb0_sha256": _sha(logits_mb0),
        "grads_sha256": {n: _sha(g) for n, g in grads.items()},
        "weights_sha256": {n: _sha(w) for n, w in weights.items()},
        "opt_state_sha256": {n: _sha(t) for n, t in opt_state.items()},
        "peak_gpu_gb": round(torch.cuda.max_memory_allocated() / 1e9, 3),
    }
    torch.save(
        {"logits_mb0": logits_mb0.cpu(), "grads": {n: g.cpu() for n, g in grads.items()},
         "weights": {n: w.cpu() for n, w in weights.items()},
         "opt_state": {n: t.cpu() for n, t in opt_state.items()}},
        os.path.join(job_dir, f"leg_{tag}.pt"),
    )
    tensors = {"logits_mb0": logits_mb0, "grads": grads, "weights": weights, "opt_state": opt_state}

    del model, opt, sched, params, lora_params
    gc.collect()
    import torch as _t

    _t.cuda.empty_cache()
    _t.cuda.reset_peak_memory_stats()
    return art, tensors


def compare(x, y):
    """Diff two legs' tensor dicts: bitwise-equal counts + fp64 max-abs-diff / rel-norm stats."""
    import torch

    out = {}
    for group in ("logits_mb0", "grads", "weights", "opt_state"):
        a, b = x[group], y[group]
        if group == "logits_mb0":
            a, b = {"logits_mb0": a}, {"logits_mb0": b}
        n_bitwise = 0
        max_abs = 0.0
        max_rel = 0.0
        for k in a:
            ta, tb = a[k].double(), b[k].double()
            if torch.equal(a[k], b[k]):
                n_bitwise += 1
                continue
            d = (ta - tb).abs().max().item()
            max_abs = max(max_abs, d)
            denom = ta.norm().item()
            if denom > 0:
                max_rel = max(max_rel, (ta - tb).norm().item() / denom)
        out[group] = {
            "n_tensors": len(a),
            "n_bitwise_equal": n_bitwise,
            "all_bitwise": n_bitwise == len(a),
            "max_abs_diff": max_abs,
            "max_rel_norm_diff": max_rel,
        }
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--job-dir", required=True)
    ap.add_argument("--quant-type", required=True)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--seq", type=int, default=256)
    ap.add_argument("--n-train", type=int, default=10000)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--dropout", type=float, default=0.0, help=">0 forces attention dropout (counterfactual ON leg)")
    ap.add_argument("--deterministic", action="store_true")
    args = ap.parse_args()
    os.makedirs(args.job_dir, exist_ok=True)

    # Env the trainer module reads at import — set before importing anything torch-side.
    os.environ["SEQ"] = str(args.seq)
    os.environ["N_TRAIN"] = str(args.n_train)
    os.environ["SEED"] = str(args.seed)
    if args.deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    import torch

    det_warnings = []
    if args.deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)

    batches = build_batches(args.seq, args.n_train, args.grad_accum)

    legs = [("a", False), ("b", False), ("c", True)]
    arts, tensors = {}, {}
    with warnings.catch_warnings(record=True) as wrec:
        warnings.simplefilter("always")
        for tag, offload in legs:
            print(f"[leg {tag}] offload={offload} quant={args.quant_type} det={args.deterministic} "
                  f"dropout={args.dropout}", flush=True)
            arts[tag], tensors[tag] = run_leg(
                tag, args.quant_type, offload, args.seed, args.seq, args.grad_accum, args.lr,
                batches, args.dropout, args.job_dir,
            )
    det_warnings = sorted({str(w.message) for w in wrec if "deterministic" in str(w.message).lower()})

    null_ab = compare(tensors["a"], tensors["b"])
    placement_ac = compare(tensors["a"], tensors["c"])

    result = {
        "job_type": "cert",
        "status": "pass",
        "model": os.environ.get("MODEL", "allenai/OLMoE-1B-7B-0924"),
        "storage_mode": args.quant_type,
        "seed": args.seed,
        "seq": args.seq,
        "grad_accum": args.grad_accum,
        "lr": args.lr,
        "dropout": args.dropout,
        "deterministic": args.deterministic,
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "deterministic_op_warnings": det_warnings,
        "legs": {t: {k: v for k, v in a.items() if not k.endswith("sha256")} for t, a in arts.items()},
        "losses": {t: arts[t]["losses"] for t in arts},
        "loss_delta_ab": [abs(x - y) for x, y in zip(arts["a"]["losses"], arts["b"]["losses"])],
        "loss_delta_ac": [abs(x - y) for x, y in zip(arts["a"]["losses"], arts["c"]["losses"])],
        "null_ab": null_ab,
        "placement_ac": placement_ac,
        "verdict": {
            "null_bitwise": all(g["all_bitwise"] for g in null_ab.values()),
            "placement_bitwise": all(g["all_bitwise"] for g in placement_ac.values()),
            "placement_within_null": all(
                placement_ac[g]["max_abs_diff"] <= 10 * max(null_ab[g]["max_abs_diff"], 1e-300)
                or placement_ac[g]["all_bitwise"]
                for g in placement_ac
            ),
        },
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "gpu_name": torch.cuda.get_device_name(0),
        "timestamp": _utc(),
    }
    try:
        import bitsandbytes

        result["bitsandbytes_version"] = bitsandbytes.__version__
    except Exception:
        result["bitsandbytes_version"] = None
    with open(os.path.join(args.job_dir, "result.json"), "w") as f:
        json.dump(result, f, indent=2, sort_keys=True)
        f.write("\n")
    v = result["verdict"]
    print(f"CERT {args.quant_type} det={args.deterministic} dropout={args.dropout}: "
          f"null_bitwise={v['null_bitwise']} placement_bitwise={v['placement_bitwise']} "
          f"placement_within_null={v['placement_within_null']}")
    for g in ("logits_mb0", "grads", "weights", "opt_state"):
        print(f"  {g:<12} null max|d|={null_ab[g]['max_abs_diff']:.3e} "
              f"placement max|d|={placement_ac[g]['max_abs_diff']:.3e}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
