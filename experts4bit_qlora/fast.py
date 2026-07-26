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
recompute-backward training semantics are untouched.

Accuracy, measured rather than asserted. The two paths dequantize the *same*
NF4 values and the fused path accumulates in fp32, which measured *more*
accurate than the reference's bf16 materialization on every cell of the
kernel's stamped **per-op** property suite. **That does not survive
composition**: through 16 layers of OLMoE-1B-7B the fused path is consistently
*worse*, costing **+0.023% perplexity** (7.45645 vs the reference's 7.45474 on
24 independent 2048-token chunks). A per-op accuracy result is not a
model-level one, and this docstring previously implied it was. For scale, the
NF4 KV cache costs ~2.1% — about 92x more. Deeper models compound further;
94 layers is unmeasured.

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
        return mod._e4b_fast_ref(hidden_states, top_k_index, top_k_weights)

    compute_dtype = mod.compute_dtype if mod.compute_dtype is not None else hidden_states.dtype
    if compute_dtype not in (torch.bfloat16, torch.float16):
        return mod._e4b_fast_ref(hidden_states, top_k_index, top_k_weights)

    from nf4_grouped import gemm_4bit_grouped

    input_dtype = hidden_states.dtype
    tokens, hidden = hidden_states.shape
    k = top_k_index.shape[1]
    n1, k1 = mod._gate_up_shape
    n2, k2 = mod._down_shape
    E = mod.num_experts

    # token->expert sort: one (token, slot) row per assignment, grouped by expert
    flat = top_k_index.reshape(-1)
    order = torch.argsort(flat, stable=True)                      # [tokens*k], expert-grouped
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
    return _scatter_combine(down, w, order, token_rows, tokens, k, hidden,
                            hidden_states.device, input_dtype)


def fused_experts_lora_forward(mod, hidden_states, top_k_index, top_k_weights):
    """Grouped-GEMM forward for :class:`ExpertsLoRA` — the *streaming* expert path.

    ``ExpertsLoRA`` is not an ``ExpertsNbit`` subclass and never calls
    ``base.forward()`` (it reads ``base.gate_up_proj`` and calls
    ``base._project`` per expert), so patching the base is dead code on any
    model built by ``load_moe_4bit_streaming``. That left the grouped kernel
    unreachable in exactly the configuration the big-model claims run in. This
    is the same fusion as :func:`fused_experts_forward` — one launch over all
    active experts, reading only packed bytes — with the trainable low-rank
    delta applied on the already-expert-sorted rows.

    Offload-safe: the offload pre-hook sits on ``ExpertsLoRA``, so by the time
    this runs the layer's experts are staged. The ``numel() == 0`` check is a
    guard, not a policy — it routes an evicted read back to the reference path
    so the failure carries ``offload.py``'s explanatory error rather than a
    shape mismatch from inside the kernel.
    """
    base = mod.base
    if torch.is_grad_enabled() and (
        hidden_states.requires_grad or any(p.requires_grad for p in mod.parameters())
    ):
        return mod._e4b_fast_ref(hidden_states, top_k_index, top_k_weights)
    if mod.training:
        # A reentrant-checkpoint initial forward is no_grad but training; the
        # reference path preserves the exact summation order its grad-enabled
        # recompute will reproduce. Do not fuse under it.
        return mod._e4b_fast_ref(hidden_states, top_k_index, top_k_weights)

    compute_dtype = base.compute_dtype if base.compute_dtype is not None else hidden_states.dtype
    if compute_dtype not in (torch.bfloat16, torch.float16):
        return mod._e4b_fast_ref(hidden_states, top_k_index, top_k_weights)
    if base.gate_up_proj.numel() == 0 or base.down_proj.numel() == 0:
        return mod._e4b_fast_ref(hidden_states, top_k_index, top_k_weights)

    from nf4_grouped import gemm_4bit_grouped

    input_dtype = hidden_states.dtype
    hidden_states = hidden_states.to(compute_dtype)
    tokens, hidden = hidden_states.shape
    k = top_k_index.shape[1]
    n1, k1 = base._gate_up_shape
    n2, k2 = base._down_shape
    E = base.num_experts

    flat = top_k_index.reshape(-1)
    order = torch.argsort(flat, stable=True)
    token_rows = order // k
    top_pos = order - token_rows * k
    counts = torch.bincount(flat, minlength=E)
    active = torch.nonzero(counts, as_tuple=False).view(-1)
    sizes = counts[active].tolist()
    expert_ids = active.to(torch.int32).tolist()

    a_cat = hidden_states.index_select(0, token_rows).contiguous()

    up = gemm_4bit_grouped(
        a_cat,
        base.gate_up_proj.view(E, n1, k1 // 2),
        base.gate_up_absmax.view(E, n1, k1 // 64).float(),
        sizes,
        expert_ids,
    )
    # The rows are already grouped by expert, so each adapter applies to one
    # contiguous slice. The rank is tiny (r=8); the heavy GEMM is what fused.
    off = 0
    for e, n in zip(expert_ids, sizes):
        up[off:off + n] += mod._lora(
            a_cat[off:off + n], mod.gate_up_lora_A[e], mod.gate_up_lora_B[e])
        off += n

    if base.has_gate:
        gate, up_h = up.chunk(2, dim=-1)
        h = base.act_fn(gate) * up_h
    else:
        h = base.act_fn(up)
    h = h.contiguous()

    down = gemm_4bit_grouped(
        h.to(compute_dtype),
        base.down_proj.view(E, n2, k2 // 2),
        base.down_absmax.view(E, n2, k2 // 64).float(),
        sizes,
        expert_ids,
    )
    off = 0
    for e, n in zip(expert_ids, sizes):
        down[off:off + n] += mod._lora(
            h[off:off + n], mod.down_lora_A[e], mod.down_lora_B[e])
        off += n

    w = top_k_weights[token_rows, top_pos].to(torch.float32)
    return _scatter_combine(down, w, order, token_rows, tokens, k, hidden,
                            hidden_states.device, input_dtype)


def _scatter_combine(down, w, order, token_rows, tokens, k, hidden, device, out_dtype):
    """Weighted combine of the expert-sorted rows back into token order.

    ``index_add_`` is the obvious way to do this and is what this path used
    first. On CUDA it accumulates with atomics, so the summation ORDER varies
    run to run and the result is nondeterministic: measured as a 1.2e-2 %
    perplexity spread across repeats on OLMoE, against a bit-stable 0.0 % for
    the reference path. A stable sort alone does not fix it — the atomics do it
    on their own.

    ``order`` is a permutation, so every destination index is written exactly
    once: scattering by assignment is deterministic, and the per-token reduction
    then becomes a fixed-axis ``sum`` rather than an atomic accumulation. Costs
    one ``[tokens*k, hidden]`` fp32 buffer, which at decode (tokens=1) is
    negligible.
    """
    buf = torch.zeros(tokens * k, hidden, dtype=torch.float32, device=device)
    buf[order] = down.to(torch.float32) * w[:, None]
    return buf.view(tokens, k, hidden).sum(1).to(out_dtype)


def enable_fast(model, verbose: bool = False) -> int:
    """Patch every eligible ``ExpertsNbit`` under ``model`` (or ``model`` itself).

    Also patches ``ExpertsLoRA`` wrappers — the streaming loader's expert
    module, which bypasses ``ExpertsNbit.forward`` entirely, so patching only
    the base would leave the grouped kernel unreachable on every offloaded
    model. A wrapped base is skipped rather than patched twice: its forward is
    never called.

    Returns the number of modules patched. Modules whose class overrides
    ``forward`` (custom-activation experts) or whose storage is ineligible are
    skipped — pass ``verbose=True`` to print each skip reason once.
    """
    from experts4bit_qlora import Experts4bit, ExpertsNbit
    from experts4bit_qlora.lora import ExpertsLoRA

    stock_forwards = {ExpertsNbit.forward, Experts4bit.forward}
    mods = list(model.modules() if hasattr(model, "modules") else [model])
    patched = 0

    wrapped_bases = {id(m.base) for m in mods if isinstance(m, ExpertsLoRA)}
    for mod in mods:
        if not isinstance(mod, ExpertsLoRA):
            continue
        if type(mod).forward is not ExpertsLoRA.forward:
            if verbose:
                print(f"[e4b.fast] skip {type(mod).__name__}: custom forward")
            continue
        reason = _eligible(mod.base)
        if reason is not None:
            if verbose:
                print(f"[e4b.fast] skip {type(mod).__name__}: {reason}")
            continue
        if hasattr(mod, "_e4b_fast_ref"):
            continue  # already enabled; idempotent
        mod._e4b_fast_ref = mod.forward
        mod.forward = fused_experts_lora_forward.__get__(mod)
        patched += 1

    for mod in mods:
        if not isinstance(mod, ExpertsNbit):
            continue
        if id(mod) in wrapped_bases:
            if verbose:
                print(f"[e4b.fast] skip {type(mod).__name__}: wrapped by a patched "
                      "ExpertsLoRA (its forward is never called)")
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
        if hasattr(mod, "_hot_residency"):
            if verbose:
                print(f"[e4b.fast] skip {type(mod).__name__}: hot_residency enabled — disable it first")
            continue
        if hasattr(mod, "_e4b_cold_ref"):
            if verbose:
                print(f"[e4b.fast] skip {type(mod).__name__}: cold engine enabled — disable it first")
            continue
        if hasattr(mod, "_e4b_pipe_ref"):
            if verbose:
                print(f"[e4b.fast] skip {type(mod).__name__}: pipelined residency enabled — disable it first")
            continue
        if hasattr(mod, "_e4b_fast_ref"):
            continue  # already enabled; idempotent
        mod._e4b_fast_ref = mod.forward
        mod.forward = fused_experts_forward.__get__(mod)
        patched += 1
    return patched


def disable_fast(model) -> int:
    """Undo :func:`enable_fast`; returns the number of modules restored."""
    mods = model.modules() if hasattr(model, "modules") else [model]
    restored = 0
    for mod in mods:
        if hasattr(mod, "_e4b_fast_ref"):
            mod.forward = mod._e4b_fast_ref
            del mod._e4b_fast_ref
            restored += 1
    return restored
