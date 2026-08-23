"""Optional fused-GEMM fast path — ``pip install "experts4bit-qlora[fast]"``.

Routes the frozen-expert projections of :class:`ExpertsNbit` through
``grouped-nf4-gemm``'s single-launch grouped kernel (``gemm_4bit_grouped``:
NF4 decoded to fp32 in registers inside the GEMM, fp32 accumulation, bf16
epilogue) instead of the per-expert dequantize-then-``linear`` loop. At bs=1
decode the reference loop pays ~3 kernel launches per active expert plus a
full bf16 weight materialization; the fused path is one launch over all
active experts and reads only packed bytes.

``enable_fast`` is inference-only by design: ``gemm_4bit_grouped`` has no
backward, so any forward that requires grad (QLoRA training) silently uses the
reference path — the recompute-backward training semantics are untouched.

**That is a statement about ``enable_fast``, not about this module.** Training
has its own fused entry point here: ``enable_fast_train`` (see below) routes the
``ExpertsLoRA`` *training* forward through ``nf4_qlora.fused_grouped_lora``,
whose backward recomputes one decoded expert at a time, and the expert LoRA
delta trains through it. This paragraph used to read "inference-only by design"
with no such qualifier, describing the whole module; it was written before that
path existed and was then quoted back at the maintainers as evidence the package
had no training accelerator (issue #38). If you add another entry point, say so
here — a reader who trusts a stale module header does not go looking further
down the file.

Note that ``enable_fast_train`` needs ``grouped-nf4-gemm >= 0.2.4`` for the
``nf4_qlora`` module, and returns ``0`` when it is missing. Pass ``verbose=True``
to be told which, and assert the return is non-zero: a silent ``0`` is
indistinguishable from "there is no fast training path".

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
storage, and an epilogue this path can reproduce. A subclass that overrides
``forward`` is skipped rather than silently mis-activated — unless it exposes
``_apply_gate`` (DeepSeek-V4's clamped SwiGLU), which the fused paths call, so
they cannot drift from the reference. gpt-oss stays skipped: its forward also
adds per-expert biases, which no fused path here applies.
"""
from __future__ import annotations

from typing import Optional

import types

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
    # Checked BEFORE quant_type, because an MXFP4-arena module still reports
    # `quant_type="nf4"` -- that is the class the loader built, and only its
    # BUFFERS were re-declared to MXFP4 geometry. So the nf4 test below passes it,
    # and the `is_cuda` test below rejects it only while it is still on meta. The
    # kernel here reads fp32 absmax per 64 elements; MXFP4 carries uint8 e8m0 per
    # 32, so the reshape would either raise mid-forward or, at the sizes where the
    # element counts happen to divide, compute nonsense.
    if getattr(mod, "_e4b_mxfp4_arena", False):
        return ("storage is MXFP4 (arena), which the NF4 grouped kernel cannot read — "
                "the MXFP4 forward is wired by nvme_experts and needs no patch here")
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
    from ..lora import _epilogue

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
    # The base's OWN epilogue, not an assumed SwiGLU: `_epilogue` is the same hook
    # `ExpertsLoRA.forward` uses, so the fused and reference paths cannot drift apart.
    # DeepSeek-V4 loads INSIDE an ExpertsLoRA, and a plain `act_fn(gate) * up` here
    # silently drops its clamps while the frozen base still applies them.
    h = _epilogue(mod, up)

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
    from ..lora import _epilogue

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

    # The base's OWN epilogue, not an assumed SwiGLU: `_epilogue` is the same hook
    # `ExpertsLoRA.forward` uses, so the fused and reference paths cannot drift apart.
    # DeepSeek-V4 loads INSIDE an ExpertsLoRA, and a plain `act_fn(gate) * up` here
    # silently drops its clamps while the frozen base still applies them.
    h = _epilogue(base, up)
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
    mods = list(model.modules()) if hasattr(model, "modules") else [model]
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
        # A base whose forward this path cannot reproduce must be SKIPPED, not fused.
        # `_epilogue` covers a custom *activation* (V4's clamps) via `_apply_gate`; it
        # cannot cover gpt-oss, whose forward also adds per-expert biases the grouped
        # path never applies. The bare-module loop below already makes this check --
        # the wrapper loop only checked the WRAPPER's forward, so a custom base reached
        # the kernel whenever it was LoRA-wrapped, which is how V4 loads.
        if (type(mod.base).forward not in stock_forwards
                and not hasattr(mod.base, "_apply_gate")):
            if verbose:
                print(f"[e4b.fast] skip {type(mod).__name__}: base "
                      f"{type(mod.base).__name__} has a custom forward")
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
        # Same rule as the wrapper loop above: a custom forward is fine when it is only
        # a custom ACTIVATION, because `fused_experts_forward` now calls `_epilogue`.
        # Without this a BARE V4 module was skipped while the LoRA-wrapped one was fused.
        if (type(mod).forward not in stock_forwards
                and not hasattr(mod, "_apply_gate")):
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
        if hasattr(mod, "_e4b_mxfp4_ref"):
            if verbose:
                print(f"[e4b.fast] skip {type(mod).__name__}: mxfp4 NVMe residency enabled — disable it first")
            continue
        if hasattr(mod, "_e4b_fast_ref"):
            continue  # already enabled; idempotent
        mod._e4b_fast_ref = mod.forward
        mod.forward = fused_experts_forward.__get__(mod)
        patched += 1
    # Train mode still costs you the kernel, but NOT for the reason this warning used
    # to give. The old text said "ExpertsLoRA only hands off to the patched base under
    # eval + no_grad", which described the era when `enable_fast` patched the BASE and
    # a wrapped base was reached only via `ExpertsLoRA._delegate_to_base`. The wrapper
    # loop above changed that: an `ExpertsLoRA` gets `fused_experts_lora_forward` on
    # ITSELF and its base is skipped. The conclusion survives anyway, because that
    # forward short-circuits to the reference path under `mod.training` (see its own
    # body) to preserve the summation order a reentrant-checkpoint recompute will
    # reproduce.
    #
    # Counting gemm_4bit_grouped itself on an A2000, wrapped module, all four states:
    #
    #   mode   adapter   fused kernel calls
    #   eval   zero      2
    #   eval   trained   2      <- reachable; a trained adapter does NOT kill it
    #   train  zero      0
    #   train  trained   0      <- the case this warns about
    #
    # Count the KERNEL, not the wrapper: once patched the wrapper is always invoked,
    # so counting calls to it reports 1 in every cell above and says nothing.
    if patched and getattr(model, "training", False):
        import warnings

        warnings.warn(
            f"[e4b.fast] model is in TRAINING mode: fused_experts_lora_forward falls back "
            f"to the reference path while `training` is set, so all {patched} patch(es) "
            "will be bypassed and the fused kernel will not run. Call model.eval() before "
            "inference.",
            RuntimeWarning,
            stacklevel=2,
        )
    # The non-zero-adapter warning that used to sit here is GONE, on both counts.
    # Obsolete: eval + a trained adapter fuses fine (row 2 above) — the wrapper's fused
    # forward applies the low-rank delta on the expert-sorted rows rather than skipping
    # it. And DEAD CODE besides: `unreachable` was only appended in the bare-module
    # loop, which `continue`s on every wrapped base, while `lora_parent` was keyed BY
    # wrapped bases — so the lookup could never hit and the warning could never fire.
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


def _dgrad_supported() -> bool:
    """Whether the installed ``grouped-nf4-gemm`` takes the ``dgrad_kernel`` flag.

    Checked rather than assumed: the flag landed in 0.7.0, and this package's
    ``[fast]`` extra admits older ones. Passing an unknown kwarg would raise from
    inside a training forward, which is a worse failure than not accelerating.
    """
    import inspect

    try:
        from nf4_qlora import fused_grouped_lora
    except ImportError:
        return False
    try:
        return "dgrad_kernel" in inspect.signature(fused_grouped_lora).parameters
    except (TypeError, ValueError):
        return False


def _dgrad_kwarg(lora_mod) -> dict:
    """``{"dgrad_kernel": True}`` only when this module opted in AND the installed
    kernel package understands it. Empty dict otherwise, so the call is unchanged."""
    if getattr(lora_mod, "_e4b_dgrad", False):
        return {"dgrad_kernel": True}
    return {}


def fused_experts_train_forward(lora_mod, hidden_states, top_k_index, top_k_weights):
    """Training-capable fused forward for an ``ExpertsLoRA``.

    The inference fast path patches the frozen ``ExpertsNbit`` base, which
    training can never reach: ``ExpertsLoRA`` must inject its low-rank delta
    *before* the SwiGLU, and the base kernel was forward-only anyway, so no
    ``dL/dx`` could flow through it.

    This composes both halves in the grouped layout: the frozen projection goes
    through ``grouped-nf4-gemm``'s differentiable wrapper (recompute-decode
    backward, one expert live at a time) and the trainable ``B(Ax)`` delta is
    added to the SAME pre-activation tensor -- which is the only place it is
    correct, since ``act(Wx + BAx) != act(Wx) + d``.

    Numerics note: this accumulates in fp32 and scatters exactly like the
    reference loop, but it sums experts in group-sorted order rather than
    ascending-expert-id order. That is an ulp-level reordering, the same class
    of difference ``_forward_decode`` already documents -- which is why this
    path is OPT-IN via ``enable_fast_train`` rather than a silent default.
    """
    from nf4_qlora import fused_grouped_lora
    from ..lora import _epilogue

    base = lora_mod.base
    input_dtype = hidden_states.dtype
    compute_dtype = base.compute_dtype if base.compute_dtype is not None else input_dtype
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

    # Set by `nvme_train.enable_nvme_train_residency` when this layer's frozen
    # experts come off an NVMe arena. Routed staging then fills only the routed
    # rows of a full-shaped stack, so a backward reading a row the recompute did
    # not re-stage would get uninitialized memory rather than an error. The
    # closures below run in BACKWARD, which is the one place that can happen, so
    # the check belongs here and nowhere else. `None` (the host-RAM home, whose
    # stage is always the whole layer) costs one attribute lookup per forward.
    guard = getattr(base, "_e4b_stage_guard", None)

    # weights_fn is load-bearing, not optional: under expert offload the views
    # below are TRANSIENT staged copies. nf4_qlora must not hold them on the
    # autograd ctx (that pinned all 48 layers and OOMed at 22.41 GB); it calls
    # this closure in backward instead, re-reading whatever is staged then --
    # which under gradient checkpointing is this layer, because the recompute
    # forward re-stages it first.
    def _gate_up_now():
        if guard is not None:
            guard(expert_ids)
        return (base.gate_up_proj.view(E, n1, k1 // 2),
                base.gate_up_absmax.view(E, n1, k1 // 64).float())

    proj = fused_grouped_lora(
        a_cat,
        base.gate_up_proj.view(E, n1, k1 // 2),
        base.gate_up_absmax.view(E, n1, k1 // 64).float(),
        sizes, expert_ids,
        lora_mod.gate_up_lora_A, lora_mod.gate_up_lora_B,
        weights_fn=_gate_up_now,
        scaling=lora_mod.scaling,     # alpha/r -- the reference applies it too
        **_dgrad_kwarg(lora_mod),
    )
    # The base's OWN epilogue, not an assumed SwiGLU: `_epilogue` is the same hook
    # `ExpertsLoRA.forward` uses, so the fused and reference paths cannot drift apart.
    # DeepSeek-V4 loads INSIDE an ExpertsLoRA, and a plain `act_fn(gate) * up` here
    # silently drops its clamps while the frozen base still applies them.
    h = _epilogue(base, proj)

    def _down_now():
        if guard is not None:
            guard(expert_ids)
        return (base.down_proj.view(E, n2, k2 // 2),
                base.down_absmax.view(E, n2, k2 // 64).float())

    down = fused_grouped_lora(
        h.to(compute_dtype).contiguous(),
        base.down_proj.view(E, n2, k2 // 2),
        base.down_absmax.view(E, n2, k2 // 64).float(),
        sizes, expert_ids,
        lora_mod.down_lora_A, lora_mod.down_lora_B,
        weights_fn=_down_now,
        scaling=lora_mod.scaling,
        **_dgrad_kwarg(lora_mod),
    )

    w = top_k_weights[token_rows, top_pos].to(torch.float32)
    return _scatter_combine(down, w, order, token_rows, tokens, k, hidden,
                            hidden_states.device, input_dtype)


def enable_fast_train(model, verbose: bool = False, dgrad: bool = False) -> int:
    """Route ``ExpertsLoRA`` TRAINING through the fused grouped kernel.

    ``enable_fast`` patches the frozen base and is inference-only. This patches
    the ``ExpertsLoRA`` wrapper -- the module the model actually calls -- so the
    kernel is reached with gradients enabled.

    ``dgrad=True`` additionally routes the BACKWARD through
    ``grouped-nf4-gemm``'s single-launch dgrad kernel (>= 0.7.0) instead of its
    per-expert decode loop. Measured on an A2000: the loop is 78-84% of a training
    step, and the kernel cuts the two grad_x loops from 117.3 ms to 9.2 ms per
    layer at E=256 while materializing nothing. A second opt-in rather than part of
    the first, because it is a second numerics change: the loop decodes with the
    same oracle the reference uses and is EXACT, the kernel accumulates fp32 in a
    different order and lands near 2.9e-3 -- inside the bf16 budget, not zero.
    Layer-composed fidelity is MEASURED (bench/dgrad-gate/): at 48 layers on
    Qwen3-30B-A3B, dgrad=True adds nothing to this lane's composed gradient error
    (4.97e-2 -> 4.99e-2 mean vs the reference) and is the fastest option at
    real width (2.52x, vs 1.72x without). The error this lane carries is the
    FORWARD fusion's, present with or without this flag.

    Requested but unsupported (``grouped-nf4-gemm`` older than 0.7.0) it is turned
    OFF with a warning rather than raising from inside a forward.

    Opt-in on purpose: it changes the expert summation ORDER (group-sorted vs
    ascending expert id), an ulp-level difference that should be a deliberate
    choice in a training run, not a silent one. Returns the number patched.
    """
    try:
        from nf4_qlora import fused_grouped_lora  # noqa: F401
    except ImportError:
        if verbose:
            print("[e4b.fast] grouped-nf4-gemm has no nf4_qlora: need >= 0.2.4")
        return 0
    from experts4bit_qlora import Experts4bit, ExpertsNbit
    from experts4bit_qlora.lora import ExpertsLoRA

    if dgrad and not _dgrad_supported():
        import warnings
        warnings.warn(
            "[e4b.fast] dgrad=True ignored: the installed grouped-nf4-gemm has no "
            "`dgrad_kernel` argument (it landed in 0.7.0). Upgrade with "
            "`pip install -U 'grouped-nf4-gemm>=0.14.0'`; the fused training path "
            "still runs, with its per-expert decode backward.",
            RuntimeWarning, stacklevel=2)
        dgrad = False

    stock_forwards = {ExpertsNbit.forward, Experts4bit.forward}
    patched = 0
    for mod in model.modules():
        if isinstance(mod, ExpertsLoRA) and hasattr(mod, "_e4b_train_ref"):
            # Already patched. The loop below would skip it silently, so a second
            # call asking for a DIFFERENT backward — the natural thing to do after
            # upgrading grouped-nf4-gemm — would return 0 patched and leave the
            # decode loop running, with nothing said. Re-patching in place would
            # capture our own forward as the reference, so say what to do instead.
            if getattr(mod, "_e4b_dgrad", False) != dgrad:
                import warnings
                warnings.warn(
                    f"[e4b.fast] enable_fast_train(dgrad={dgrad}) on a module already "
                    f"patched with dgrad={getattr(mod, '_e4b_dgrad', False)}: the "
                    "existing patch is kept and the flag is NOT applied. Call "
                    "disable_fast_train first to change the backward.",
                    RuntimeWarning, stacklevel=2)
            continue
        if isinstance(mod, ExpertsLoRA):
            # The kernel-free batched path patches the same forward. Patching over it
            # would capture ITS forward as this one's reference, so a later
            # disable_batched_train could not unwind and disable_fast_train would
            # reinstate the batched path. `enable_batched_train` refuses the mirror
            # case; this is the other half of that guard.
            if hasattr(mod, "_e4b_batched_ref"):
                if verbose:
                    print("[e4b.fast] skip: enable_batched_train already patched this "
                          "module; call disable_batched_train first")
                continue
            # Same bargain as `enable_fast`; this loop had no eligibility gate at all,
            # so every ExpertsLoRA was fused regardless of what its base computes.
            if (type(mod.base).forward not in stock_forwards
                    and not hasattr(mod.base, "_apply_gate")):
                if verbose:
                    print(f"[e4b.fast] skip {type(mod).__name__}: base "
                          f"{type(mod.base).__name__} has a custom forward")
                continue
            # An MXFP4-arena base passes every check above -- it is `quant_type="nf4"`
            # by class and carries `_apply_gate` (V4) -- and then dies inside the
            # forward on `gate_up_absmax.view(E, n1, k1 // 64)`, because e8m0 scales
            # are one byte per 32 elements, not fp32 per 64. Refuse at patch time,
            # where the message can say why, rather than at step 1 of a rented run.
            if getattr(mod.base, "_e4b_mxfp4_arena", False):
                if verbose:
                    print(f"[e4b.fast] skip {type(mod).__name__}: base storage is "
                          "MXFP4 (arena); the NF4 grouped kernel cannot read it")
                continue
            mod._e4b_train_ref = mod.forward
            mod._e4b_dgrad = dgrad
            mod.forward = types.MethodType(
                lambda self, hs, tki, tkw: fused_experts_train_forward(self, hs, tki, tkw),
                mod)
            patched += 1
    if verbose:
        print(f"[e4b.fast] fused TRAINING path on {patched} ExpertsLoRA module(s)"
              + (" (dgrad kernel backward)" if dgrad else ""))
    return patched


def disable_fast_train(model) -> int:
    n = 0
    for mod in model.modules():
        ref = getattr(mod, "_e4b_train_ref", None)
        if ref is not None:
            mod.forward = ref
            del mod._e4b_train_ref
            # Leave nothing behind that a later enable would silently inherit.
            if hasattr(mod, "_e4b_dgrad"):
                del mod._e4b_dgrad
            n += 1
    return n
