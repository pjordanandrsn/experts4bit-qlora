"""The fused training combine's glue in fewer launches (``E4B_TRAIN_LEAN_COMBINE=1``; opt-in, EXPLORE: Map 1 T-B, target 2).

``_ScatterCombine`` scatters the expert-sorted rows back into token slots and sums each token's ``k`` slots. Its forward runs
an fp32 ``zeros`` buffer, a cast, a broadcast multiply and an ``index_put``; its backward runs a cast, an index divide, a gather,
a cast, a multiply, the sum, an in-place multiply and a narrowing cast. With this knob:

* the buffer is ``empty``: ``order`` is a permutation of the ``tokens * k`` slots, so every row is written (the zero pass was
  dead), and ``_PermGather``'s backward ``new_zeros`` is ``new_empty`` for the same reason;
* ``scatter_mul(buf, down, w, order)``: ``buf[order[i]] = fp32(down[i]) * w[i]``, one launch for the cast, multiply and
  ``index_put``;
* ``grad_products(g, down, w, order, k)``: ``prod[i] = fp32(g[order[i] // k]) * fp32(down[i])`` for the weight gradient's
  ``.sum(1)`` (kept as the ATen reduction, so its order is unchanged), and ``gdown[i] = bf16(fp32(g[order[i] // k]) * w[i])``
  narrowed as torch narrows on that device: the plain cast on CUDA (``cvt.rn``, the instruction torch's own CUDA cast
  uses) and, under the Triton interpreter, whose cast truncates, round to nearest even in integer arithmetic (a NaN
  to the quiet 0x7FC0; torch's own CPU cast writes 0xFFFF or 0x7FC0 by code path, so no NaN pattern is torch's there).

Every value is the composite's: the same single IEEE multiply on the same operands, exact widening casts, and the same
narrowing rule. On a tensor neither on CUDA nor under ``TRITON_INTERPRET=1`` each function runs the torch ops it replaces.
The chunked combine (``E4B_COMBINE_CHUNK``, images of 128 MiB and more) is left as it is.
"""
from __future__ import annotations

import os

import torch

try:
    import triton
    import triton.language as tl
except ImportError:                                   # pragma: no cover
    triton = tl = None


def lean_combine_enabled() -> bool:
    """``E4B_TRAIN_LEAN_COMBINE=1``: anything else is off."""
    return os.environ.get("E4B_TRAIN_LEAN_COMBINE", "0").strip() == "1"


def _jit(f):
    return triton.jit(f) if triton is not None else f


def _use_triton(t: torch.Tensor) -> bool:
    return triton is not None and (t.is_cuda or os.environ.get("TRITON_INTERPRET") == "1")


@_jit
def _rne_bf16(x):
    """fp32 -> bf16 by round to nearest even, a NaN to 0x7FC0, in integer arithmetic (the interpreter's cast truncates)."""
    b = x.to(tl.uint32, bitcast=True)
    h = ((b + 0x7FFF + ((b >> 16) & 1)) >> 16).to(tl.uint16)
    h = tl.where(x != x, 0x7FC0, h).to(tl.uint16)
    return h.to(tl.bfloat16, bitcast=True)


@_jit
def _scatter_mul_kernel(buf, down, w, order, H, s_buf, s_down, BLOCK_H: tl.constexpr):
    i = tl.program_id(0)
    offs = tl.program_id(1) * BLOCK_H + tl.arange(0, BLOCK_H)
    mh = offs < H
    dst = tl.load(order + i).to(tl.int64)
    wi = tl.load(w + i)
    x = tl.load(down + i.to(tl.int64) * s_down + offs, mask=mh).to(tl.float32)
    tl.store(buf + dst * s_buf + offs, x * wi, mask=mh)


@_jit
def _grad_products_kernel(g, down, w, order, prod, gdown, k, H, s_g, s_down, s_prod, s_gdown, NEED_W: tl.constexpr,
                          NEED_DOWN: tl.constexpr, INT_RNE: tl.constexpr, BLOCK_H: tl.constexpr):
    i = tl.program_id(0)
    offs = tl.program_id(1) * BLOCK_H + tl.arange(0, BLOCK_H)
    mh = offs < H
    t = (tl.load(order + i) // k).to(tl.int64)
    gp = tl.load(g + t * s_g + offs, mask=mh).to(tl.float32)
    if NEED_W:
        x = tl.load(down + i.to(tl.int64) * s_down + offs, mask=mh).to(tl.float32)
        tl.store(prod + i.to(tl.int64) * s_prod + offs, gp * x, mask=mh)
    if NEED_DOWN:
        y = gp * tl.load(w + i)
        if INT_RNE:
            y = _rne_bf16(y)
        else:
            y = y.to(gdown.dtype.element_ty)
        tl.store(gdown + i.to(tl.int64) * s_gdown + offs, y, mask=mh)


def _block(n: int) -> int:
    return max(16, min(1024, triton.next_power_of_2(int(n))))


def scatter_mul(buf: torch.Tensor, down: torch.Tensor, w: torch.Tensor, order: torch.Tensor) -> None:
    """``buf[order] = down.to(float32) * w[:, None]`` in one launch (every row of ``buf`` written: ``order`` is a permutation)."""
    rows, H = down.shape
    if not _use_triton(buf):
        buf[order] = down.to(torch.float32) * w[:, None]
        return
    bh = _block(H)
    _scatter_mul_kernel[(rows, triton.cdiv(H, bh))](buf, down, w, order, H, buf.stride(0), down.stride(0), BLOCK_H=bh)


def grad_products(g, down, w, order, k, need_w, need_down):
    """``(gw, gdown)``: the weight gradient (``fp32(g[order // k]) * fp32(down)``, summed by ATen's ``.sum(1)``) and the down
    gradient (``bf16(fp32(g[order // k]) * w)`` in ``down``'s dtype), as ``_ScatterCombine.backward`` computes them."""
    rows, H = down.shape
    if not _use_triton(down):
        gprod = g.to(torch.float32)[torch.div(order, k, rounding_mode="floor")]
        gw = (gprod * down.to(torch.float32)).sum(1, keepdim=True).squeeze(1) if need_w else None
        gdown = gprod.mul_(w[:, None]).to(down.dtype) if need_down else None
        return gw, gdown
    g = g.contiguous()
    prod = torch.empty(rows, H, dtype=torch.float32, device=down.device) if need_w else down
    gdown = torch.empty_like(down) if need_down else down
    bh = _block(H)
    _grad_products_kernel[(rows, triton.cdiv(H, bh))](
        g, down, w, order, prod, gdown, k, H, g.stride(0), down.stride(0), prod.stride(0), gdown.stride(0),
        NEED_W=bool(need_w), NEED_DOWN=bool(need_down), INT_RNE=down.dtype == torch.bfloat16 and not down.is_cuda,
        BLOCK_H=bh)
    gw = prod.sum(1, keepdim=True).squeeze(1) if need_w else None
    return gw, (gdown if need_down else None)
