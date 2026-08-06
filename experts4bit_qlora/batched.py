"""Kernel-free batched training path — stock torch + bitsandbytes, no extras.

``enable_fast_train`` (see :mod:`experts4bit_qlora.fast`) needs
``grouped-nf4-gemm``, which has to build and is arch-gated. Without it that call
returns ``0`` and training falls back to ``ExpertsLoRA.forward``'s per-expert
Python loop: at 256 experts over 40 layers that is ~10k sync-gated iterations per
forward, and the GPU idles through most of it.

This module was written as the fallback for that case and then measured faster
than the thing it was falling back from. One training step, E=256, 512 tokens,
top_k 8, hidden 512, inter 768, RTX A2000:

===========================  ==========  =========  ==========
path                           step ms    vs loop     peak MB
===========================  ==========  =========  ==========
reference per-expert loop        601.2      1.00x          59
``enable_fast_train``            132.6      4.53x         108
this module                       25.0     24.01x         417
===========================  ==========  =========  ==========

**The table above is a MICROBENCH, and its ranking does not survive real width.**
A prior revision of this docstring predicted a bigger card "should narrow this";
measured, it did not narrow — it reversed. Qwen3-30B-A3B (48 layers, hidden 2048)
on an A6000, one training step, from the published wheels
(``bench/dgrad-gate/RESULTS-dgrad-gate.md``):

===================================  ==========  =========  ==========
path                                   s/step      vs loop    peak GB
===================================  ==========  =========  ==========
reference per-expert loop               12.781      1.00x      23.13
``enable_fast_train``                    7.420      1.72x      24.31
``enable_fast_train(dgrad=True)``        5.075      2.52x      25.51
this module                             12.219      1.05x      26.71
===================================  ==========  =========  ==========

At toy width the per-expert Python loop is the cost and batching anything wins;
at real width the matmuls dominate, and the whole-stack dequant this path pays on
every forward AND backward stops being amortizable — while also costing the most
peak memory of any lane. What survives at scale is the one thing the kernel lane
cannot offer: **no extras**. Reach for this module when ``grouped-nf4-gemm`` will
not build for your arch; default to ``enable_fast_train(dgrad=True)`` otherwise.

One consolation result: this path's composed gradient error against the reference
is ~4e-3 at 48 layers where the fused lane's is ~5e-2 — the tightest of the
accelerated lanes, if fidelity-to-the-reference ever matters more than speed.

**Why batching a frozen expert stack needs no kernel.** The experts are frozen,
so the dequantized stack is a constant with respect to autograd — there is no
``dW`` to accumulate and nothing to keep. And it can be produced in ONE call:
``_quantize_stack`` quantizes each expert with ``compress_statistics=False``,
and the constructor refuses shapes where the blocksize does not divide an
expert's rows, so blocks never straddle an expert boundary and the flattened
absmax is an exact concatenation. A single ``dequantize_4bit`` over the whole
buffer with ``QuantState(shape=(E*out, in))`` therefore reproduces the
per-expert loop **bit for bit** — pinned by
``test_fused_train_parity.test_whole_stack_dequant_equals_per_expert_loop``,
which is what fails first if double-quant is ever enabled or a straddling shape
is admitted. Measured 32x against the per-expert decode loop at E=256.

**Credit.** The approach is not ours. @jiwoon-ahn proposed it in issue #38 — batch
the frozen expert projections with a single whole-stack dequant, sort token/expert
pairs, run the groups as ``bmm``s — with measurements, a working implementation, and
an offer to upstream it. This module is that idea. What we verified before building
on it: the whole-stack dequant is not merely close to the per-expert loop, it is
bit-identical, for the structural reason above.

Two things this implementation does differently from the design in #38:

* **Recompute-decode backward.** The dequantized stack is regenerated inside
  ``backward`` rather than saved, so it is never held across the
  forward-to-backward window. #38 lets autograd save it and therefore *requires*
  gradient checkpointing to keep 40 layers of it from being resident. Here the
  stack is transient in each direction, so checkpointing is an option rather
  than a precondition. Same guarantee ``nf4_qlora`` makes for the fused lane,
  one whole stack at a time instead of one expert at a time.
* **Trainable expert LoRA.** #38 requires the expert adapter be zero and frozen,
  which restricts it to ``TRAIN_EXPERTS=0``. Batching the low-rank delta as a
  padded double-``bmm`` costs little and keeps the delta where it has to be —
  added to the *pre-activation* projection, since
  ``act(Wx + BAx) != act(Wx) + d`` for any cheap ``d`` — so the package's
  default ``TRAIN_EXPERTS=1`` works on this path.

Numerics: expert contributions are summed in group-sorted order rather than
ascending expert id, an ulp-level reordering — the same class of difference
``ExpertsLoRA._forward_decode`` already documents. Opt-in for that reason, like
``enable_fast_train``.

Usage::

    from experts4bit_qlora import enable_batched_train
    n = enable_batched_train(model)     # assert n > 0, or you are on the loop
    ...                                 # train
    disable_batched_train(model)
"""
from __future__ import annotations

import types

import torch

# Padded-bmm cutoff: group sizes come from the router, so one hot expert makes the
# padded block `G * max(sizes)` rows wide however few rows are real. Past this
# multiple of the real rows, fall back to the reference loop — pathological routing
# must not cost more than it did without this path. A guard, not a tuned optimum;
# under even remotely balanced routing the ratio sits near 1.
_PAD_WASTE_LIMIT = 4.0


def _dequant_whole(packed, absmax, n_exp, out_f, in_f, quant_type, blocksize, dtype):
    """The whole expert stack in one ``dequantize_4bit``. See the module docstring for
    why the flattened absmax is exact rather than approximately right."""
    import bitsandbytes.functional as F
    from bitsandbytes.functional import QuantState

    state = QuantState(
        absmax=absmax.reshape(-1),
        shape=torch.Size((n_exp * out_f, in_f)),
        dtype=dtype,
        blocksize=blocksize,
        quant_type=quant_type,
        code=F.get_4bit_type(quant_type, device=packed.device),
    )
    return F.dequantize_4bit(packed.reshape(-1, 1), quant_state=state).view(n_exp, out_f, in_f)


class _FrozenGroupedProj(torch.autograd.Function):
    """``x_pad @ W_e.T`` for every group at once, decoding W in both directions.

    ``x_pad`` is ``[G, maxn, in]``; the result is ``[G, maxn, out]``. Only the packed
    bytes are saved — saving the decoded stack is what would force gradient
    checkpointing, and saving it per layer is what would not fit.
    """

    @staticmethod
    def forward(ctx, x_pad, packed, absmax, n_exp, out_f, in_f, eids, quant_type, blocksize):
        W = _dequant_whole(packed, absmax, n_exp, out_f, in_f, quant_type, blocksize, x_pad.dtype)
        # `eids is None` means every expert is active in ascending order — the common
        # case at any real token count. Skipping the gather there avoids a second
        # [G, out, in] copy of the stack at exactly the moment it is largest.
        Wg = W if eids is None else W[eids]
        out = torch.bmm(x_pad, Wg.transpose(1, 2))
        ctx.save_for_backward(packed, absmax, eids)
        ctx.meta = (n_exp, out_f, in_f, quant_type, blocksize)
        return out

    @staticmethod
    def backward(ctx, grad_out):
        packed, absmax, eids = ctx.saved_tensors
        n_exp, out_f, in_f, quant_type, blocksize = ctx.meta
        grad_x = None
        if ctx.needs_input_grad[0]:
            grad_out = grad_out.contiguous()
            W = _dequant_whole(packed, absmax, n_exp, out_f, in_f, quant_type, blocksize,
                               grad_out.dtype)
            Wg = W if eids is None else W[eids]
            grad_x = torch.bmm(grad_out, Wg)
            del W, Wg
        return (grad_x,) + (None,) * 8


def _lora_delta_padded(x_pad, lora_A, lora_B, eids, scaling):
    """``scaling * B_e(A_e x)`` for every group at once. Adapters may sit in a higher
    precision than the activations (fp32 adapters over bf16 compute is the shipped
    configuration), so compute in the adapter dtype and cast the delta back."""
    A = lora_A if eids is None else lora_A[eids]      # [G, r, in]
    B = lora_B if eids is None else lora_B[eids]      # [G, out, r]
    d = torch.bmm(torch.bmm(x_pad.to(A.dtype), A.transpose(1, 2)), B.transpose(1, 2))
    return (scaling * d).to(x_pad.dtype)


def batched_experts_train_forward(mod, hidden_states, top_k_index, top_k_weights):
    """Batched replacement for ``ExpertsLoRA.forward``. Falls back to the reference
    forward — not to a slower version of itself — whenever the batched shape would be
    wasteful or the storage is not there."""
    from .lora import _epilogue

    base = mod.base
    reference = mod._e4b_batched_ref
    input_dtype = hidden_states.dtype
    compute_dtype = base.compute_dtype if base.compute_dtype is not None else input_dtype

    # Under expert offload the packed buffers are 0-element placeholders between
    # forwards. Entering the dequant with one produces a shaped-but-empty stack rather
    # than an error, so check before, not after.
    if base.gate_up_proj.numel() == 0 or base.down_proj.numel() == 0:
        return reference(hidden_states, top_k_index, top_k_weights)

    hs = hidden_states.to(compute_dtype)
    tokens, hidden = hs.shape
    k = top_k_index.shape[1]
    n_exp = base.num_experts
    dev = hs.device

    flat = top_k_index.reshape(-1)
    order = torch.argsort(flat, stable=True)
    token_rows = torch.div(order, k, rounding_mode="floor")
    top_pos = order - token_rows * k
    counts = torch.bincount(flat, minlength=n_exp)
    active = torch.nonzero(counts, as_tuple=False).view(-1)
    sizes = counts[active]
    n_grp = int(active.numel())
    if n_grp == 0:
        # No token routed anywhere — an empty batch. `sizes.max()` would raise on the
        # empty tensor, where the reference simply returns its zero accumulator.
        return reference(hidden_states, top_k_index, top_k_weights)
    widest = int(sizes.max())
    total = int(sizes.sum())
    if n_grp * widest > _PAD_WASTE_LIMIT * total:
        return reference(hidden_states, top_k_index, top_k_weights)

    grp = torch.repeat_interleave(torch.arange(n_grp, device=dev), sizes)
    slot = torch.arange(total, device=dev) - (torch.cumsum(sizes, 0) - sizes)[grp]
    # None means "all experts, in order" — see _FrozenGroupedProj.forward.
    eids = None if (n_grp == n_exp) else active

    a_cat = hs.index_select(0, token_rows)
    x_pad = a_cat.new_zeros(n_grp, widest, hidden)
    x_pad[grp, slot] = a_cat

    gu_out, gu_in = base._gate_up_shape
    proj = _FrozenGroupedProj.apply(
        x_pad, base.gate_up_proj, base.gate_up_absmax, n_exp, gu_out, gu_in,
        eids, base.quant_type, base.blocksize,
    ) + _lora_delta_padded(x_pad, mod.gate_up_lora_A, mod.gate_up_lora_B, eids, mod.scaling)

    # The base's OWN epilogue, never an assumed SwiGLU: a hardcoded act(gate)*up here
    # would silently drop DeepSeek-V4's clamps while the frozen base still applies them.
    h = _epilogue(base, proj)

    dn_out, dn_in = base._down_shape
    out_pad = _FrozenGroupedProj.apply(
        h, base.down_proj, base.down_absmax, n_exp, dn_out, dn_in,
        eids, base.quant_type, base.blocksize,
    ) + _lora_delta_padded(h, mod.down_lora_A, mod.down_lora_B, eids, mod.scaling)

    rows = out_pad[grp, slot] * top_k_weights[token_rows, top_pos, None]
    # fp32 accumulation, same as the reference: bf16 routing weights over many
    # contributions is where a plain sum loses digits.
    final = torch.zeros(tokens, hidden, dtype=torch.float32, device=dev)
    final.index_add_(0, token_rows, rows.float())
    return final.to(input_dtype)


def batched_train_available() -> bool:
    """True on any install of this package — the point of this path. Present so
    callers can branch symmetrically with ``fast_available()``."""
    return True


def enable_batched_train(model, verbose: bool = False) -> int:
    """Route ``ExpertsLoRA`` training through the batched path. Returns the number
    patched; assert it, because ``0`` and "silently still on the loop" look the same
    from the caller's side.

    Choosing between this and ``enable_fast_train``: this one measured faster on the
    box in the module docstring, and spends peak memory to get there (a whole decoded
    stack rather than a single decoded expert). Take the kernel lane when VRAM is the
    binding constraint or experts are offloaded; take this one otherwise, or when
    ``grouped-nf4-gemm`` will not build.

    A module already carrying the fused training patch is SKIPPED rather than wrapped.
    Both patch ``ExpertsLoRA.forward``, so stacking them would make this path's
    fallbacks — pad-waste, evicted storage — land on the fused forward instead of the
    reference, which is not what "fall back" is supposed to mean. Call
    ``disable_fast_train`` first if you want to switch lanes.
    """
    from experts4bit_qlora import Experts4bit, ExpertsNbit
    from experts4bit_qlora.lora import ExpertsLoRA

    stock_forwards = {ExpertsNbit.forward, Experts4bit.forward}
    patched = 0
    for mod in model.modules():
        if not isinstance(mod, ExpertsLoRA) or hasattr(mod, "_e4b_batched_ref"):
            continue
        # Mutual exclusion with the fused training patch: wrapping it would leave this
        # path's fallbacks pointing at the fused forward rather than the reference.
        if hasattr(mod, "_e4b_train_ref"):
            if verbose:
                print("[e4b.batched] skip: enable_fast_train already patched this "
                      "module; call disable_fast_train first")
            continue
        base = mod.base
        # Same bargain as enable_fast/enable_fast_train: a base whose forward this
        # path cannot reproduce is SKIPPED, not silently mis-accelerated. gpt-oss
        # lands here — its forward adds per-expert biases that nothing below applies.
        if type(base).forward not in stock_forwards and not hasattr(base, "_apply_gate"):
            if verbose:
                print(f"[e4b.batched] skip: base {type(base).__name__} has a custom forward")
            continue
        if getattr(base, "bits", None) != 4:
            if verbose:
                print(f"[e4b.batched] skip: {getattr(base, 'bits', '?')}-bit storage, "
                      "the whole-stack dequant is 4-bit only")
            continue
        mod._e4b_batched_ref = mod.forward
        mod.forward = types.MethodType(
            lambda self, hs, tki, tkw: batched_experts_train_forward(self, hs, tki, tkw), mod)
        patched += 1
    if verbose:
        print(f"[e4b.batched] batched training path on {patched} ExpertsLoRA module(s)")
    return patched


def disable_batched_train(model) -> int:
    """Restore the reference forward. Returns the number restored.

    Refuses to unpatch a module something else has patched ON TOP of, and says so.
    ``enable_fast_train`` predates this module and does not know to skip a
    batched-patched module, so ``enable_batched_train`` then ``enable_fast_train``
    stacks: the fused patch captures THIS path's forward as its reference. Restoring
    from underneath that would hand the module back the true reference while the layer
    above still holds ours, and a later ``disable_fast_train`` would reinstate exactly
    the forward this call was meant to remove — leaving the module on a path with no
    ``_e4b_batched_ref`` to undo it. Unwind the outer patch first.
    """
    import warnings

    n = 0
    for mod in model.modules():
        ref = getattr(mod, "_e4b_batched_ref", None)
        if ref is None:
            continue
        if hasattr(mod, "_e4b_train_ref"):
            warnings.warn(
                "[e4b.batched] not restoring: enable_fast_train patched over this "
                "module, so its reference is this path's forward. Call "
                "disable_fast_train first, then disable_batched_train.",
                RuntimeWarning, stacklevel=2)
            continue
        mod.forward = ref
        del mod._e4b_batched_ref
        n += 1
    return n
