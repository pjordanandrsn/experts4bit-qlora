"""Optional fused-GEMM fast path — ``pip install "experts4bit-qlora[fast]"``.

Routes the frozen-expert projections of :class:`ExpertsNbit` through
``grouped-nf4-gemm``'s single-launch grouped kernel (``gemm_4bit_grouped``:
NF4 decoded to fp32 in registers inside the GEMM, fp32 accumulation, bf16
epilogue) instead of the per-expert dequantize-then-``linear`` loop. At bs=1
decode the reference loop pays ~3 kernel launches per active expert plus a
full bf16 weight materialization; the fused path is one launch over all
active experts and reads only packed bytes.

Inference-only by design: the fused kernel has no backward, so any forward
that requires grad (QLoRA training) silently uses the reference path — the
recompute-backward training semantics are untouched. The two paths dequantize
the *same* NF4 values; the fused path accumulates in fp32, which measured
*more* accurate than the reference's bf16 materialization on every cell of
the kernel's stamped property suite.

Usage::

    from experts4bit_qlora import enable_fast
    n = enable_fast(model)      # patches eligible ExpertsNbit modules
    # ... generate / evaluate; training steps automatically fall back ...
    disable_fast(model)         # restore, if wanted

Eligibility (checked per module, ineligible modules are left untouched):
NF4 4-bit storage, blocksize 64, K divisible by 64 on both projections, CUDA
storage, and the module still uses the stock ``ExpertsNbit`` forward
(subclasses that override ``forward`` — e.g. the gpt-oss clamped-activation
experts — are skipped rather than silently mis-activated).
"""
from __future__ import annotations

from typing import Optional

import torch


def fast_available() -> bool:
    """True iff the fused kernel package is importable and CUDA is up."""
    try:
        from nf4_grouped import gemm_4bit_grouped  # noqa: F401
    except ImportError:
        return False
    return torch.cuda.is_available()


def _eligible(mod) -> Optional[str]:
    """Return None if ``mod`` can take the fast path, else the reason not."""
    if getattr(mod, "bits", None) != 4 or getattr(mod, "quant_type", None) != "nf4":
        return "storage is not nf4-4bit"
    if getattr(mod, "blocksize", None) != 64:
        return f"blocksize {getattr(mod, 'blocksize', None)} != 64"
    n1, k1 = mod._gate_up_shape
    n2, k2 = mod._down_shape
    if k1 % 64 or k2 % 64:
        return "K not divisible by 64"
    if not mod.gate_up_proj.is_cuda:
        return "storage not on CUDA"
    return None


def fused_experts_forward(mod, hidden_states, top_k_index, top_k_weights):
    """Fused inference forward with the reference signature and semantics.

    Falls back to the module's reference forward whenever grad is required
    (training) or the input dtype isn't a kernel epilogue dtype.
    """
    if torch.is_grad_enabled() and (
        hidden_states.requires_grad or any(p.requires_grad for p in mod.parameters())
    ):
        return mod._e4b_reference_forward(hidden_states, top_k_index, top_k_weights)

    compute_dtype = mod.compute_dtype if mod.compute_dtype is not None else hidden_states.dtype
    if compute_dtype not in (torch.bfloat16, torch.float16):
        return mod._e4b_reference_forward(hidden_states, top_k_index, top_k_weights)

    from nf4_grouped import gemm_4bit_grouped

    input_dtype = hidden_states.dtype
    tokens, hidden = hidden_states.shape
    k = top_k_index.shape[1]
    n1, k1 = mod._gate_up_shape
    n2, k2 = mod._down_shape
    E = mod.num_experts

    # token->expert sort: one (token, slot) row per assignment, grouped by expert
    flat = top_k_index.reshape(-1)
    order = torch.argsort(flat)                      # [tokens*k], expert-grouped
    token_rows = order // k                          # source token per row
    top_pos = order - token_rows * k                 # which top-k slot it was
    counts = torch.bincount(flat, minlength=E)       # tokens per expert
    active = torch.nonzero(counts, as_tuple=False).view(-1)
    sizes = counts[active].tolist()
    expert_ids = active.to(torch.int32).tolist()

    a_cat = hidden_states.index_select(0, token_rows).to(compute_dtype).contiguous()

    up = gemm_4bit_grouped(
        a_cat,
        mod.gate_up_proj.view(E, n1, k1 // 2),
        mod.gate_up_absmax.view(E, n1, k1 // 64).float(),
        sizes,
        expert_ids,
    )
    if mod.has_gate:
        gate, up_h = up.chunk(2, dim=-1)
        h = mod.act_fn(gate) * up_h
    else:
        h = mod.act_fn(up)

    down = gemm_4bit_grouped(
        h.to(compute_dtype).contiguous(),
        mod.down_proj.view(E, n2, k2 // 2),
        mod.down_absmax.view(E, n2, k2 // 64).float(),
        sizes,
        expert_ids,
    )

    w = top_k_weights[token_rows, top_pos].to(torch.float32)
    out = torch.zeros(tokens, hidden, dtype=torch.float32, device=hidden_states.device)
    out.index_add_(0, token_rows, down.to(torch.float32) * w[:, None])
    return out.to(input_dtype)


def enable_fast(model, verbose: bool = False) -> int:
    """Patch every eligible ``ExpertsNbit`` under ``model`` (or ``model`` itself).

    Returns the number of modules patched. Modules whose class overrides
    ``forward`` (custom-activation experts) or whose storage is ineligible are
    skipped — pass ``verbose=True`` to print each skip reason once.
    """
    from experts4bit_qlora import Experts4bit, ExpertsNbit

    stock_forwards = {ExpertsNbit.forward, Experts4bit.forward}
    mods = model.modules() if hasattr(model, "modules") else [model]
    patched = 0
    for mod in mods:
        if not isinstance(mod, ExpertsNbit):
            continue
        if type(mod).forward not in stock_forwards:
            if verbose:
                print(f"[e4b.fast] skip {type(mod).__name__}: custom forward")
            continue
        reason = _eligible(mod)
        if reason is not None:
            if verbose:
                print(f"[e4b.fast] skip {type(mod).__name__}: {reason}")
            continue
        if not hasattr(mod, "_e4b_reference_forward"):
            mod._e4b_reference_forward = mod.forward
        mod.forward = fused_experts_forward.__get__(mod)
        patched += 1
    return patched


def disable_fast(model) -> int:
    """Undo :func:`enable_fast`; returns the number of modules restored."""
    mods = model.modules() if hasattr(model, "modules") else [model]
    restored = 0
    for mod in mods:
        if hasattr(mod, "_e4b_reference_forward"):
            mod.forward = mod._e4b_reference_forward
            del mod._e4b_reference_forward
            restored += 1
    return restored
