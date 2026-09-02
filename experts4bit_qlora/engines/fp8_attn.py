# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""Opt-in fp8 (e4m3) storage for the SERVING attention projections.

Callers opt in with :func:`enable_serve_attn_fp8` after any q/k/v
fusion pass, the ``enable_fast``/``enable_nvme_residency`` pattern --
free when unused. Harnesses gate the call on ``E4B_SERVE_ATTN_FP8=1``.

**Why a second format on the same projections.** The int4 lane
(:mod:`int4_attn`) is faster still, and it was refused here on quality:
+0.0558 ppl over 8,192 teacher-forced tokens, well outside the gate.
The attention projections are memory bound at decode, so what matters
is bytes per weight, and e4m3 sits between the two: half of bf16, twice
int4, with an EXPONENT that tracks a row's dynamic range rather than a
4-bit linear grid. Measured on the weight distribution directly, e4m3
at group 128 carries ~4.6x lower mean absolute error than int4-b32 at
group 32 -- a smaller error from a coarser group, because the format
and not the group size is what was binding.

That argument sets up the gate; it does not pass it. The quality gate
runs FIRST in this lane's campaign, before any speed work, exactly
because the int4 attempt showed speed here is cheap to get and easy to
lose.

Decode (M = 1) is served by the fp8 GEMV; prefill (M > 1) dequantises
once and runs a dense matmul -- the large-M side of the crossover, paid
per request rather than per token. The lm_head is NOT touched: it was
ineligible on quality even for int4 (+0.18 ppl) and is a separate
question from these projections.
"""
from __future__ import annotations

import torch
from torch import nn

__all__ = ["Fp8Linear", "enable_serve_attn_fp8"]

_GROUP = 128


def _kernels():
    """Lazy, loud: absence degrades at ENABLE time with a sentence,
    never silently at forward time (the fp8-append pattern)."""
    from int4_b32 import gemv_fp8_rows, quant_fp8_rows  # noqa: F401
    return gemv_fp8_rows, quant_fp8_rows


class Fp8Linear(nn.Module):
    """Frozen serving projection stored as e4m3 with per-group scales."""

    def __init__(self, lin: nn.Linear):
        super().__init__()
        if lin.bias is not None:
            raise RuntimeError(
                "E4B_SERVE_ATTN_FP8: projection carries a bias; this path "
                "stores weight-only fp8 -- refusing rather than dropping it")
        gemv, quant = _kernels()
        self._gemv, self._group = gemv, _GROUP
        self.N, self.K = lin.out_features, lin.in_features
        if self.K % _GROUP:
            raise RuntimeError(
                f"E4B_SERVE_ATTN_FP8: K={self.K} is not a multiple of the "
                f"scale group {_GROUP}; refusing rather than padding a "
                "projection into a layout the kernel cannot address")
        # quantise FROM the source bf16 weight, never from an already
        # quantised grid: composing two lossy grids measured ~7x the
        # single grid's ppl cost on the int4 lane
        q, s = quant(lin.weight.detach().to(torch.float32), group=_GROUP)
        dev = lin.weight.device
        self.register_buffer("q", q.to(dev), persistent=False)
        self.register_buffer("scales", s.to(dev), persistent=False)

    def _deq(self) -> torch.Tensor:
        return (self.q.float().reshape(self.N, self.K // self._group,
                                       self._group)
                * self.scales.float()[..., None]
                ).reshape(self.N, self.K).to(torch.bfloat16)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rows = x.reshape(-1, self.K)
        if rows.shape[0] == 1:
            out = self._gemv(rows.to(torch.bfloat16), self.q, self.scales,
                             group=self._group)
            return out.reshape(*x.shape[:-1], self.N).to(x.dtype)
        w = self._deq()
        return (rows.to(torch.bfloat16) @ w.t()).reshape(
            *x.shape[:-1], self.N).to(x.dtype)


def enable_serve_attn_fp8(model) -> int:
    """Swap every structural attention projection for :class:`Fp8Linear`.

    Returns the count; refuses a vacuous enable. lm_head untouched."""
    try:
        _kernels()
    except ImportError as e:
        raise RuntimeError(
            "E4B_SERVE_ATTN_FP8=1 needs grouped-nf4-gemm with the fp8 GEMV "
            f"(missing: {e}); install the matching cut or unset the flag"
        ) from e
    n = 0
    for mod in model.modules():
        if type(mod).__name__.endswith("Attention"):
            for name, child in list(mod.named_children()):
                if type(child) is nn.Linear:
                    setattr(mod, name, Fp8Linear(child))
                    n += 1
    if n == 0:
        raise RuntimeError("E4B_SERVE_ATTN_FP8=1 matched no attention "
                           "projections -- refusing a vacuous enable")
    return n
