"""Shared audit/parity/train-probe helpers for falsifying unsloth-zoo's MoE bnb-4bit fix.

Everything here is measurement code: it inspects a loaded model and reports facts.
The verdict logic lives in run_falsify.py.
"""

import json
import torch
import torch.nn as nn
import torch.nn.functional as F


def find_expert_modules(model):
    """All modules holding fused expert tensors (gate_up_proj + down_proj attributes)."""
    hits = []
    for name, module in model.named_modules():
        if hasattr(module, "gate_up_proj") and hasattr(module, "down_proj"):
            hits.append((name, module))
    return hits


def _param_facts(p):
    facts = {
        "class": type(p).__name__,
        "dtype": str(p.dtype),
        "stored_shape": list(p.shape),
        "nbytes": p.numel() * p.element_size(),
        "quant_state": None,
        "original_shape": None,
    }
    qs = getattr(p, "quant_state", None)
    if qs is not None:
        facts["quant_state"] = {
            "quant_type": getattr(qs, "quant_type", "?"),
            "blocksize": getattr(qs, "blocksize", "?"),
            "logical_shape": list(qs.shape) if getattr(qs, "shape", None) is not None else None,
        }
    orig = getattr(p, "_original_shape", None)
    if orig is not None:
        facts["original_shape"] = list(orig)
    return facts


def audit_experts(model):
    """Per-experts-module facts: is the fused tensor a bnb Params4bit or a plain bf16 Parameter?"""
    rows = []
    for name, module in find_expert_modules(model):
        rows.append({
            "module": name,
            "module_class": type(module).__name__,
            "gate_up_proj": _param_facts(module.gate_up_proj),
            "down_proj": _param_facts(module.down_proj),
        })
    quantized = [r for r in rows if r["gate_up_proj"]["quant_state"] is not None]
    summary = {
        "n_expert_modules": len(rows),
        "n_quantized": len(quantized),
        "expert_bytes_total": sum(
            r["gate_up_proj"]["nbytes"] + r["down_proj"]["nbytes"] for r in rows
        ),
        "cuda_allocated_gb": round(torch.cuda.memory_allocated() / 2**30, 3)
        if torch.cuda.is_available() else None,
    }
    return rows, summary


def _dequant_expert_weight(p):
    """Independent dequant of a Params4bit expert tensor to its logical 3-D fp32 weight.

    Uses bnb's own dequantize_4bit on the STORED quant_state (never re-quantizes),
    so the reference sees exactly the weights the quantized path should be using.
    """
    import bitsandbytes as bnb
    qs = getattr(p, "quant_state", None)
    if qs is None:
        return p.data.float()  # not quantized: reference == stored bf16 weight
    w = bnb.functional.dequantize_4bit(p.data, qs)
    orig = getattr(p, "_original_shape", None) or (qs.shape if getattr(qs, "shape", None) is not None else None)
    if orig is not None and tuple(w.shape) != tuple(orig):
        w = w.reshape(tuple(orig))
    return w.float()


def parity_check(experts_module, hidden_dim, num_experts, top_k, act_fn, device="cuda", n_tokens=64, seed=0):
    """Compare the module's (possibly patched/quantized) forward against an
    independent fp32 loop over dequantized weights. Same routing, same math:
    gate,up = W_gu[e] @ x; act(gate)*up; W_d[e] @ .; x top_k weight."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    hs = torch.randn(n_tokens, hidden_dim, generator=g).to(device=device, dtype=torch.bfloat16)
    idx = torch.randint(0, num_experts, (n_tokens, top_k), generator=g).to(device)
    w = torch.rand(n_tokens, top_k, generator=g).to(device=device, dtype=torch.bfloat16)
    w = w / w.sum(-1, keepdim=True)

    def ref_forward(dtype):
        w_gu = _dequant_expert_weight(experts_module.gate_up_proj).to(device=device, dtype=dtype)
        w_d = _dequant_expert_weight(experts_module.down_proj).to(device=device, dtype=dtype)
        h, wt = hs.to(dtype), w.to(dtype)
        out = torch.zeros(n_tokens, hidden_dim, device=device, dtype=dtype)
        for t in range(n_tokens):
            for k in range(top_k):
                e = idx[t, k].item()
                gate, up = F.linear(h[t], w_gu[e]).chunk(2, dim=-1)
                out[t] += wt[t, k].to(dtype) * F.linear(act_fn(gate) * up, w_d[e])
        return out.float()

    with torch.no_grad():
        out_theirs = experts_module(hs, idx, w).float()
        out_ref32 = ref_forward(torch.float32)   # ground truth on THEIR dequantized weights
        out_ref16 = ref_forward(torch.bfloat16)  # precision-noise control: same math in bf16

    scale = out_ref32.abs().max().item()
    err_theirs = (out_theirs - out_ref32).abs().max().item()
    err_control = (out_ref16 - out_ref32).abs().max().item()
    return {
        "ref_max_abs": scale,
        "err_theirs_vs_ref32": err_theirs,
        "err_bf16control_vs_ref32": err_control,
        "rel_err_theirs": err_theirs / max(scale, 1e-12),
        "rel_err_control": err_control / max(scale, 1e-12),
        # >> 1 means their forward diverges beyond bf16 precision noise = real mismatch
        "excess_over_precision_noise": err_theirs / max(err_control, 1e-12),
        "out_finite": bool(torch.isfinite(out_theirs).all().item()),
    }


def lora_placement(model):
    """Bucket every LoRA parameter by whether it adapts experts or something else."""
    expert_markers = ("experts", "gate_up_proj", "down_proj")
    buckets = {"expert": [], "other": []}
    for name, p in model.named_parameters():
        if "lora" not in name.lower():
            continue
        is_expert = ".experts." in name or any(
            m in name for m in ("gate_up_proj", "shared_expert")
        )
        buckets["expert" if is_expert else "other"].append(name)
    return buckets


def train_probe(model, vocab_size, device="cuda", seq_len=64, batch=2, seed=0):
    torch.cuda.empty_cache()
    """One fwd/bwd/step. Reports loss, finiteness, and — the real question —
    which LoRA params actually received nonzero gradients."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    ids = torch.randint(0, vocab_size, (batch, seq_len), generator=g).to(device)
    out = model(input_ids=ids, labels=ids)
    loss = out.loss
    loss.backward()

    grad_report = {"expert_lora_grads": 0, "expert_lora_nonzero": 0,
                   "other_lora_grads": 0, "other_lora_nonzero": 0,
                   "expert_grad_norm": 0.0, "other_grad_norm": 0.0}
    for name, p in model.named_parameters():
        if "lora" not in name.lower() or p.grad is None:
            continue
        is_expert = ".experts." in name or "gate_up_proj" in name or (
            "down_proj" in name and ".experts" in name)
        key = "expert" if is_expert else "other"
        grad_report[f"{key}_lora_grads"] += 1
        norm = p.grad.float().norm().item()
        grad_report[f"{key}_grad_norm"] += norm
        if norm > 0:
            grad_report[f"{key}_lora_nonzero"] += 1

    trainable = [p for p in model.parameters() if p.requires_grad]
    torch.optim.SGD(trainable, lr=1e-4).step()

    return {
        "loss": loss.item(),
        "loss_finite": bool(torch.isfinite(loss).item()),
        "grads": grad_report,
    }


def dump(path, obj):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=str)
    print(f"[wrote {path}]")
