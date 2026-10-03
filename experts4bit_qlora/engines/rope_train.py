# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""Fused rotary embedding for q and k in TRAINING (on by default; ``E4B_FUSED_ROPE=0`` turns it off): one Triton launch forward,
one backward per tensor, BIT-IDENTICAL to Hugging Face's
``apply_rotary_pos_emb`` (``q * cos + rotate_half(q) * sin`` in the input dtype).

Every op in the composite is elementwise, and the product of two bf16 (or fp16) values is exact in fp32, so computing each
product in fp32 and rounding it to the input dtype -- then the sum, rounded once -- reproduces the composite's per-op rounding
byte for byte. The backward is the same: ``dx = round(round(g * cos) + rotate_half^T(round(g * sin)))``, where the transpose of
``rotate_half`` moves the second half's gradient forward and negates the first half's. Autograd's accumulation of x's three uses
(the product and the two rotate_half slices, each zero off its half) adds exact zeros, so the order does not matter."""
from __future__ import annotations

import os
import sys

import torch

try:
    import triton
    import triton.language as tl
except ImportError:                                   # pragma: no cover
    triton = tl = None


def _jit(f):
    return triton.jit(f) if triton is not None else f


@_jit
def _rne_bf16(x):
    """fp32 -> the nearest bf16 (ties to even), returned as fp32, by integer arithmetic on the bits. Triton 3.4 folds a
    ``.to(bfloat16).to(float32)`` round trip that feeds an add (measured: 14 % of a product-plus-add's elements came out
    unrounded), so the intermediate rounding the composite performs is done here where the compiler cannot elide it.
    Finite inputs only (the composite's operands are finite)."""
    b = x.to(tl.uint32, bitcast=True)
    b = (b + 0x7FFF + ((b >> 16) & 1)) & 0xFFFF0000
    return b.to(tl.float32, bitcast=True)


@_jit
def _rope_fwd(X, C, S, Y, sxb, sxh, sxs, scb, scs, syb, syh, sys_, H, L, D: tl.constexpr, HALF: tl.constexpr, BWD: tl.constexpr):
    # one program per (b, h, s) row of D elements; x / y rows contiguous in d
    pid = tl.program_id(0)
    s = pid % L
    h = (pid // L) % H
    b = pid // (L * H)
    i = tl.arange(0, HALF)
    xb = X + b * sxb + h * sxh + s * sxs
    cb = C + b * scb + s * scs
    sb = S + b * scb + s * scs
    x1 = tl.load(xb + i).to(tl.float32)
    x2 = tl.load(xb + HALF + i).to(tl.float32)
    c1 = tl.load(cb + i).to(tl.float32)
    c2 = tl.load(cb + HALF + i).to(tl.float32)
    s1 = tl.load(sb + i).to(tl.float32)
    s2 = tl.load(sb + HALF + i).to(tl.float32)
    dt = Y.dtype.element_ty
    if BWD:
        # g = x here.  dx1 = r(g1*c1) + r(g2*s2) ; dx2 = r(g2*c2) - r(g1*s1)
        a1 = _rne_bf16(x1 * c1)
        a2 = _rne_bf16(x2 * c2)
        r1 = _rne_bf16(x2 * s2)
        r2 = _rne_bf16(x1 * s1)
        y1 = (a1 + r1).to(dt)
        y2 = (a2 - r2).to(dt)
    else:
        # y1 = r(x1*c1) + r(-x2*s1) ; y2 = r(x2*c2) + r(x1*s2)
        a1 = _rne_bf16(x1 * c1)
        a2 = _rne_bf16(x2 * c2)
        r1 = _rne_bf16(-x2 * s1)
        r2 = _rne_bf16(x1 * s2)
        y1 = (a1 + r1).to(dt)
        y2 = (a2 + r2).to(dt)
    yb = Y + b * syb + h * syh + s * sys_
    tl.store(yb + i, y1)
    tl.store(yb + HALF + i, y2)


def _launch(x, cos, sin, bwd):
    B, H, L, D = x.shape
    if x.stride(3) != 1:
        x = x.contiguous()
    y = torch.empty_like(x)                       # same strides as x (empty_like preserves a dense non-contiguous layout)
    _rope_fwd[(B * H * L,)](x, cos, sin, y, x.stride(0), x.stride(1), x.stride(2), cos.stride(0), cos.stride(1),
                            y.stride(0), y.stride(1), y.stride(2), H, L, D=D, HALF=D // 2, BWD=bwd, num_warps=1)
    return y


class _RopeQK(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, cos, sin):
        ctx.save_for_backward(cos, sin)
        return _launch(q, cos, sin, False), _launch(k, cos, sin, False)

    @staticmethod
    def backward(ctx, gq, gk):
        cos, sin = ctx.saved_tensors
        return _launch(gq, cos, sin, True), _launch(gk, cos, sin, True), None, None


def rope_qk(q, k, cos, sin):
    """q, k: [B, H, L, D]; cos, sin: [B, L, D] (or [1, L, D]) in q's dtype. Returns (q_embed, k_embed)."""
    ROPE_TRAIN_STATS["calls"] += 1
    B = q.shape[0]
    if cos.shape[0] != B:
        cos, sin = cos.expand(B, -1, -1), sin.expand(B, -1, -1)
    return _RopeQK.apply(q, k, cos, sin)


#: Patched modules and fused calls, so a training census can say the fusion served the step.
ROPE_TRAIN_STATS = {"patched_modules": 0, "calls": 0}


def _eligible(q, k, cos, sin) -> bool:
    if triton is None or not q.is_cuda:
        return False
    if not (q.dtype == k.dtype == cos.dtype == sin.dtype == torch.bfloat16):
        return False                                  # the exact rounding helper is bf16's; fp16 keeps the composite
    if q.dim() != 4 or k.dim() != 4 or cos.dim() != 3 or sin.shape != cos.shape:
        return False
    D = q.shape[-1]
    half = D // 2
    if D % 2 or half & (half - 1) or k.shape[-1] != D or cos.shape[-1] != D:
        return False
    if q.shape[2] != cos.shape[1] or k.shape[2] != cos.shape[1] or k.shape[0] != q.shape[0] or cos.shape[0] not in (1, q.shape[0]):
        return False
    return q.stride(-1) == 1 and k.stride(-1) == 1 and cos.stride(-1) == 1 and sin.stride(-1) == 1 and cos.stride() == sin.stride()


def fused_rope_requested() -> bool:
    return os.environ.get("E4B_FUSED_ROPE", "1").strip() != "0"


def enable_fused_rope(model, verbose: bool = False) -> int:
    """Point the model's attention modules' ``apply_rotary_pos_emb`` at the fused kernel. Only the modules that define the
    model's own attention classes are touched, only a function with the Hugging Face ``(q, k, cos, sin, unsqueeze_dim=1)``
    contract is replaced, and any call outside the fused path's shapes, dtypes or arguments goes to the original. Returns the
    number of modules patched (0 is legal: a model without that function keeps its own rotary)."""
    if triton is None:
        return 0
    mods = {type(m).__module__ for m in model.modules() if type(m).__name__.endswith("Attention")}
    n = 0
    for name in sorted(mods):
        mod = sys.modules.get(name)
        orig = getattr(mod, "apply_rotary_pos_emb", None)
        if orig is None or getattr(orig, "_e4b_fused_rope", False) or not callable(orig):
            continue
        try:
            import inspect
            params = list(inspect.signature(orig).parameters)
        except (TypeError, ValueError):
            continue
        if params[:4] != ["q", "k", "cos", "sin"]:
            continue

        def fused(q, k, cos, sin, *args, _orig=orig, **kwargs):
            unsq = kwargs.get("unsqueeze_dim", 1)
            if args or set(kwargs) - {"unsqueeze_dim"} or unsq != 1 or not _eligible(q, k, cos, sin):
                return _orig(q, k, cos, sin, *args, **kwargs)
            return rope_qk(q, k, cos, sin)

        fused._e4b_fused_rope = True
        fused._e4b_orig = orig
        mod.apply_rotary_pos_emb = fused
        n += 1
    ROPE_TRAIN_STATS["patched_modules"] += n
    if verbose:
        print(f"[e4b.rope] fused rotary in {n} attention module(s)")
    return n


def disable_fused_rope(model) -> int:
    n = 0
    for name in {type(m).__module__ for m in model.modules() if type(m).__name__.endswith("Attention")}:
        mod = sys.modules.get(name)
        f = getattr(mod, "apply_rotary_pos_emb", None)
        if getattr(f, "_e4b_fused_rope", False):
            mod.apply_rotary_pos_emb = f._e4b_orig
            n += 1
    return n
