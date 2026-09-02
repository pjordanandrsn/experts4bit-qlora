# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""Opt-in uniform-int4 storage for the SERVING attention projections.

Callers opt in with :func:`enable_serve_attn_int4` (after any q/k/v
fusion pass), the ``enable_fast``/``enable_nvme_residency`` pattern --
free when unused. Harnesses conventionally gate the call on
``E4B_SERVE_ATTN_INT4=1``. Each projection becomes :class:`Int4Linear`: weights packed to the
int4-b32 grid at load FROM THE SOURCE bf16 tensor (never from an
already-quantised grid -- composition measured ~7x the pure grid's ppl
cost), decode (M = 1) served by the grouped int4 GEMV, prefill (M > 1)
by dequant-then-matmul -- the winning regime per the fused/dequant
crossover, paid once per request.

Measured basis (receipts: int4port P1, INT4GATE/INT4SPLIT): the GEMV
runs the qkv shape at 1,044 GB/s -- 6.9x over the NF4 register-LUT path
and 2.6x over the bf16 dense baseline -- and the int4 grid on attention
costs -0.006 ppl over 8,192 teacher-forced tokens. The lm_head is NOT
eligible (+0.18 ppl measured); this module never touches it.

Capture-legality: each module preallocates its split-K partials buffer
and its activation-quant outputs at swap time, so a captured decode
step allocates nothing here.
"""
from __future__ import annotations

import torch
from torch import nn


def _kernels():
    """Lazy, loud: the int4 kernels ship in grouped-nf4-gemm >= the cut
    carrying ``int4_b32``. Absence degrades at ENABLE time with a
    sentence, never silently at forward time (the fp8-append pattern)."""
    from int4_b32 import gemv_int4_b32, quant_x_rows  # noqa: F401
    from int4_pack_ref import dequant_int4_ref, pack_int4_b32  # noqa: F401
    return gemv_int4_b32, quant_x_rows, dequant_int4_ref, pack_int4_b32


class Int4Linear(nn.Module):
    """Frozen serving projection stored on the int4-b32 grid."""

    def __init__(self, lin: nn.Linear, packer=None):
        """``packer(w_fp32_cpu) -> (packed, scales)`` defaults to the
        shipped round-to-nearest packer; the calibrated lane passes one
        closed over that projection's Hessian. Same bytes either way."""
        super().__init__()
        if lin.bias is not None:
            raise RuntimeError(
                "E4B_SERVE_ATTN_INT4: projection carries a bias; this path "
                "stores weight-only int4 -- refusing rather than dropping it")
        gemv, qx, dref, pack = _kernels()
        self._gemv, self._qx, self._dref = gemv, qx, dref
        self.N, self.K = lin.out_features, lin.in_features
        dev = lin.weight.device
        packed, scales = (packer or pack)(lin.weight.detach().float().cpu())
        self.register_buffer("packed",
                             packed.reshape(1, self.N, self.K // 2).to(dev),
                             persistent=False)
        self.register_buffer("scales",
                             scales.reshape(1, self.N, self.K // 32).to(dev),
                             persistent=False)
        self.register_buffer("_eid0",
                             torch.zeros(1, dtype=torch.int32, device=dev),
                             persistent=False)
        from int4_b32 import _plan
        _bn, _wp, sk, _ku = _plan(self.N, self.K)
        self.register_buffer("_part",
                             torch.empty(sk, self.N, dtype=torch.float32,
                                         device=dev),
                             persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rows = x.reshape(-1, self.K)
        if rows.shape[0] == 1:
            xq, xs = self._qx(rows)
            out = self._gemv(xq, xs, self.packed, self.scales, self._eid0,
                             self.N, self.K, part=self._part)
            return out.reshape(*x.shape[:-1], self.N).to(x.dtype)
        # prefill / verify (M > 1): dequant once, dense matmul -- the
        # large-M side of the crossover; per-request cost, not per-token
        w = self._deq()
        return (rows.to(torch.bfloat16) @ w.t()).reshape(
            *x.shape[:-1], self.N).to(x.dtype)

    def _deq(self) -> torch.Tensor:
        lo = (self.packed.to(torch.int16) & 0xF) - 8
        hi = ((self.packed.to(torch.int16) >> 4) & 0xF) - 8
        q = torch.stack([lo, hi], dim=-1).reshape(self.N, self.K).float()
        w = (q.reshape(self.N, self.K // 32, 32)
             * self.scales.reshape(self.N, self.K // 32).float()[..., None])
        return w.reshape(self.N, self.K).to(torch.bfloat16)


def enable_serve_attn_int4(model) -> int:
    """Swap every structural attention projection for Int4Linear.
    Returns the count; refuses a vacuous enable. lm_head untouched."""
    try:
        _kernels()
    except ImportError as e:
        raise RuntimeError(
            "E4B_SERVE_ATTN_INT4=1 needs grouped-nf4-gemm with int4_b32 "
            f"(missing: {e}); install the matching cut or unset the flag"
        ) from e
    n = 0
    for mod in model.modules():
        if type(mod).__name__.endswith("Attention"):
            for name, child in list(mod.named_children()):
                if type(child) is nn.Linear:
                    setattr(mod, name, Int4Linear(child))
                    n += 1
    if n == 0:
        raise RuntimeError("E4B_SERVE_ATTN_INT4=1 matched no attention "
                           "projections -- refusing a vacuous enable")
    return n
