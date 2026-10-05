# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""Fused RMSNorm for TRAINING with a frozen weight (on by default since TC1 amendment 15; ``E4B_FUSED_RMSNORM=0`` turns it
off): one Triton launch forward, one backward (dx only).

A Qwen3-MoE layer runs four Hugging Face RMSNorms (input and post-attention, and the per-head q / k norms). Each is a composite
of about eight kernels forward and ten backward; under gradient checkpointing the forward runs twice. On TC1's RTX 5090 profile
(``tc1-5090-41``) that is roughly 20,000 of the shipped step's 134,000 device events and 17 % of its CPU op time. The decode-only
fusion (:mod:`experts4bit_qlora.engines.glue_fuse`) has no backward; this one does, for the frozen-weight case QLoRA trains in.

Mirrors the Hugging Face composite's casts: statistics in fp32 (``x.float()``, ``mean(x^2)``, ``rsqrt(var + eps)``), the
normalised value rounded to the input dtype, then ``weight * that`` rounded once. The backward reproduces the composite's
chain: ``g * w`` rounded to the input dtype (the bf16 multiply's own rounding), then ``dx = r * gw - x * r^3 * sum(gw * x) / N``
in fp32, rounded once. Not bit-identical to the composite -- the row reductions run in a different order -- so callers gate it
behind an equivalence reading, not an exactness claim."""
from __future__ import annotations

import os

import torch

try:                                    # Triton is a Linux-only dependency; without it the patcher refuses, it never half-applies
    import triton
    import triton.language as tl
except ImportError:                     # pragma: no cover - exercised on macOS installs
    triton = tl = None

from .triton_prebind import prebind

__all__ = ["rmsnorm_frozen", "enable_fused_rmsnorm_train", "RMSNORM_TRAIN_STATS"]

#: Patched module count and fused calls (forward launches), so a training census can say the fusion served the step.
RMSNORM_TRAIN_STATS = {"patched": 0, "calls": 0, "fallback_calls": 0}


def _jit(f):
    return triton.jit(f) if triton is not None else f


@_jit
def _rms_fwd(X, W, Y, R, stride, N, eps, ROWS: tl.constexpr, BLOCK: tl.constexpr,
             OFFSET: tl.constexpr = 0.0, MUL_FP32: tl.constexpr = False):
    pid = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    cmask = cols < N
    w = tl.load(W + cols, mask=cmask, other=0.0).to(tl.float32) + OFFSET
    if not MUL_FP32:
        w = w.to(Y.dtype.element_ty).to(tl.float32)   # (1 + w) rounded as the composite's input-dtype operand (identity at 0)
    for i in range(ROWS):
        row = pid * ROWS + i
        x = tl.load(X + row * stride + cols, mask=cmask, other=0.0).to(tl.float32)
        var = tl.sum(x * x, axis=0) / N
        r = tl.math.rsqrt(var + eps)
        if MUL_FP32:
            xh = x * r                                # the Gemma / Qwen3.5 composite: (x * r) * w in fp32, rounded once
        else:
            xh = (x * r).to(Y.dtype.element_ty).to(tl.float32)   # the Llama composite: x * r rounded, then w * that
        tl.store(Y + row * stride + cols, (xh * w).to(Y.dtype.element_ty), mask=cmask)
        tl.store(R + row, r)


@_jit
def _rms_bwd(DY, X, W, R, DX, stride, N, ROWS: tl.constexpr, BLOCK: tl.constexpr,
             OFFSET: tl.constexpr = 0.0, MUL_FP32: tl.constexpr = False):
    pid = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    cmask = cols < N
    w = tl.load(W + cols, mask=cmask, other=0.0).to(tl.float32) + OFFSET
    if not MUL_FP32:
        w = w.to(DX.dtype.element_ty).to(tl.float32)
    for i in range(ROWS):
        row = pid * ROWS + i
        x = tl.load(X + row * stride + cols, mask=cmask, other=0.0).to(tl.float32)
        dy = tl.load(DY + row * stride + cols, mask=cmask, other=0.0).to(tl.float32)
        r = tl.load(R + row)
        if MUL_FP32:
            gw = dy * w                               # the composite's multiply is fp32: no rounding before the reduction
        else:
            gw = (dy * w).to(DX.dtype.element_ty).to(tl.float32)
        s = tl.sum(gw * x, axis=0)
        dx = gw * r - x * (r * r * r) * (s / N)
        tl.store(DX + row * stride + cols, dx.to(DX.dtype.element_ty), mask=cmask)


# E4B_TRITON_PREBIND (on unless =0): the same kernels, launched without Triton's per-call argument binding (engines/triton_prebind.py)
_rms_fwd_launch = prebind(_rms_fwd)
_rms_bwd_launch = prebind(_rms_bwd)


def _rows_per_prog(n_rows: int, N: int) -> int:
    if N >= 1024:
        return 1
    for k in (8, 4, 2):
        if n_rows % k == 0:
            return k
    return 1


class RMSNormFrozen(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, eps, offset=0.0, mul_fp32=False):
        RMSNORM_TRAIN_STATS["calls"] += 1
        shape = x.shape
        N = shape[-1]
        x2 = x.reshape(-1, N)
        if not x2.is_contiguous():
            x2 = x2.contiguous()
        M = x2.shape[0]
        y = torch.empty_like(x2)
        r = torch.empty(M, dtype=torch.float32, device=x.device)
        k = _rows_per_prog(M, N)
        _rms_fwd_launch[(M // k,)](x2, weight, y, r, x2.stride(0), N, eps, ROWS=k, BLOCK=triton.next_power_of_2(N), num_warps=4 if N >= 1024 else 1,
                                      OFFSET=float(offset), MUL_FP32=bool(mul_fp32))
        ctx.save_for_backward(x2, weight, r)
        ctx.shape, ctx.k, ctx.offset, ctx.mul_fp32 = shape, k, float(offset), bool(mul_fp32)
        return y.view(shape)

    @staticmethod
    def backward(ctx, dy):
        x2, weight, r = ctx.saved_tensors
        N = x2.shape[1]
        dy2 = dy.reshape(-1, N)
        if not dy2.is_contiguous():
            dy2 = dy2.contiguous()
        dx = torch.empty_like(x2)
        k = ctx.k
        _rms_bwd_launch[(x2.shape[0] // k,)](dy2, x2, weight, r, dx, x2.stride(0), N, ROWS=k, BLOCK=triton.next_power_of_2(N), num_warps=4 if N >= 1024 else 1,
                                               OFFSET=ctx.offset, MUL_FP32=ctx.mul_fp32)
        return dx.view(ctx.shape), None, None, None, None


def rmsnorm_frozen(x, weight, eps, offset=0.0, mul_fp32=False):
    """``offset`` 0 is ``w * norm(x)``; 1 is the centered ``(1 + w) * norm(x)`` (Gemma, Qwen3.5/3.6). ``mul_fp32`` False is the
    Llama composite (normalised value rounded, then multiplied); True multiplies in fp32 and rounds once."""
    return RMSNormFrozen.apply(x, weight, eps, offset, mul_fp32)


def enable_fused_rmsnorm_train(model, verbose: bool = False, strict: bool = True) -> int:
    """Patch every structurally-matched RMSNorm whose weight is FROZEN to :func:`rmsnorm_frozen` on CUDA bf16 / fp16 inputs.

    Matching reuses the decode fusion's rules (:mod:`glue_fuse`): a 1-D ``weight``, a float epsilon under either upstream
    spelling, an ``...RMSNorm`` class name, and a semantic probe through the module's own forward, so a centered variant
    (``x_norm * (1 + weight)``) is never patched. A norm whose weight trains keeps the original chain (this backward returns
    no weight gradient). Other dtypes, devices and shapes fall through to the original forward. Not bit-identical to the
    composite (row reductions run in another order): on an RTX A2000 about 1 element in 100,000 differs, by one bf16 ulp, in
    the forward and in dx. Returns the number patched. ``strict`` (an explicit ``E4B_FUSED_RMSNORM=1``, or a direct call)
    refuses a zero-match enable and a missing Triton; the default-on path (``strict=False``) returns 0 instead, so a model
    without a frozen RMSNorm trains on its composite as before."""
    if triton is None:
        if not strict:
            return 0
        raise RuntimeError("enable_fused_rmsnorm_train needs Triton (Linux); set E4B_FUSED_RMSNORM=0")
    from .glue_fuse import _is_rmsnorm, _norm_eps, _probe_variant
    n = skipped = 0
    variants = {}
    for mod in model.modules():
        if not _is_rmsnorm(mod) or getattr(mod, "_e4b_rmsnorm_train", False):
            continue
        if mod.weight.requires_grad:
            skipped += 1
            continue
        eps = _norm_eps(mod)
        var = _probe_variant(mod, eps)
        if var is None:
            skipped += 1
            continue
        offset, mul_fp32 = var
        variants[var] = variants.get(var, 0) + 1
        orig = mod.forward

        def _fwd(hidden_states, _m=mod, _orig=orig, _eps=eps, _off=offset, _f32=mul_fp32):
            if (not hidden_states.is_cuda or hidden_states.dtype not in (torch.bfloat16, torch.float16)
                    or hidden_states.shape[-1] != _m.weight.numel() or _m.weight.dtype != hidden_states.dtype
                    or _m.weight.requires_grad):
                RMSNORM_TRAIN_STATS["fallback_calls"] += 1
                return _orig(hidden_states)
            return rmsnorm_frozen(hidden_states, _m.weight, _eps, _off, _f32)

        mod.forward = _fwd
        mod._e4b_rmsnorm_variant = var
        mod._e4b_rmsnorm_train = True
        n += 1
    if n == 0 and not strict:
        return 0
    if n == 0:
        raise RuntimeError(f"E4B_FUSED_RMSNORM=1 patched no RMSNorm ({skipped} name-matched but trainable or failing the probe) -- "
                           "refusing a vacuous enable")
    RMSNORM_TRAIN_STATS["patched"] += n
    if verbose:
        names = {(0.0, False): "w*round(norm)", (0.0, True): "w*norm fp32", (1.0, False): "(1+w)*round(norm)",
                 (1.0, True): "(1+w)*norm fp32"}
        detail = ", ".join(f"{c} {names.get(v, v)}" for v, c in sorted(variants.items()))
        print(f"[e4b.rmsnorm] fused {n} frozen RMSNorm(s) for training [{detail}] ({skipped} left on the composite)")
    return n


def fused_rmsnorm_requested() -> bool:
    """On unless ``E4B_FUSED_RMSNORM=0`` (the default since TC1 amendment 15's 5090 A/B, ``tc1-5090-45``)."""
    return os.environ.get("E4B_FUSED_RMSNORM", "1").strip() != "0"


def fused_rmsnorm_explicit() -> bool:
    """``E4B_FUSED_RMSNORM=1`` set by hand: a zero-match enable is then refused rather than skipped."""
    return os.environ.get("E4B_FUSED_RMSNORM", "").strip() == "1"
