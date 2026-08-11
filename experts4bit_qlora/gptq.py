"""Dequantize GPTQ checkpoints.

GPTQ is the other ubiquitous community 4-bit format, and it is a trap next to
AWQ: identical ``qweight`` / ``qzeros`` / ``scales`` key names, but a different
packing axis, a different bit order, a ``+1`` zero-point offset, and a ``g_idx``
row permutation. Decoding one as the other produces a correctly-shaped tensor of
scrambled weights, so the planner keys on the ``g_idx`` sibling to tell them
apart and this decoder is verified against gptqmodel's own math.

Layout (bits=4), note it is packed along **IN**, the opposite of AWQ:

* ``qweight`` ``[in // (32 // bits), out]`` int32 — sequential shifts, no interleave
* ``qzeros``  ``[groups, out // (32 // bits)]`` int32
* ``scales``  ``[groups, out]``
* ``g_idx``   ``[in]`` int32 — which group each input row belongs to

Dequant, verbatim from gptqmodel's ``dequantize_weight``::

    weights = scales[g_idx] * (weight - zeros[g_idx])

``g_idx`` indexes rows directly, so act-order (``desc_act=true``) and plain
group-order checkpoints go through the same path with no special case.
"""
from __future__ import annotations

import torch


def _unpack_dim(packed: torch.Tensor, bits: int, dim: int) -> torch.Tensor:
    """Expand `packed` along `dim`, extracting `32//bits` fields per int32 in
    SEQUENTIAL order (GPTQ's order — unlike AWQ there is no interleave)."""
    per = 32 // bits
    shifts = torch.arange(0, 32, bits, device=packed.device, dtype=torch.int32)
    if dim == 0:
        out = (packed.unsqueeze(1) >> shifts.view(1, per, 1)) & ((1 << bits) - 1)
        return out.reshape(packed.shape[0] * per, packed.shape[1])
    out = (packed.unsqueeze(2) >> shifts.view(1, 1, per)) & ((1 << bits) - 1)
    return out.reshape(packed.shape[0], packed.shape[1] * per)


def dequantize_gptq(
    qweight: torch.Tensor,   # [in//(32//bits), out] int32
    qzeros: torch.Tensor,    # [groups, out//(32//bits)] int32
    scales: torch.Tensor,    # [groups, out]
    g_idx: torch.Tensor,     # [in] int32
    *,
    bits: int = 4,
    dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """-> dense ``[out, in]`` (the orientation a Linear declares)."""
    if qweight.dtype is not torch.int32 or qzeros.dtype is not torch.int32:
        raise ValueError("GPTQ qweight/qzeros must be int32")
    w = _unpack_dim(qweight, bits, dim=0)            # [in, out]
    z = _unpack_dim(qzeros, bits, dim=1)             # [groups, out]
    if z.shape != scales.shape:
        raise ValueError(
            f"unpacked zeros {tuple(z.shape)} != scales {tuple(scales.shape)}")
    # GPTQ stores zero_point - 1; restore it before subtracting.
    z = z + 1
    gi = g_idx.long()
    # g_idx selects a group per input row. Two failure modes are silent without
    # these checks: a WRONG-LENGTH g_idx broadcasts into a truncated weight, and
    # a NEGATIVE entry wraps to the last group via Python indexing, yielding a
    # full-shaped tensor built from the wrong scales. Both produce plausible
    # output and no error, so both are rejected here.
    if gi.numel() != w.shape[0]:
        raise ValueError(
            f"g_idx has {gi.numel()} entries but the unpacked weight has "
            f"{w.shape[0]} input rows — refusing to index a mismatched g_idx")
    if int(gi.min()) < 0 or int(gi.max()) >= z.shape[0]:
        raise ValueError(
            f"g_idx values span [{int(gi.min())}, {int(gi.max())}] but there "
            f"are only {z.shape[0]} groups — a negative index would silently "
            f"wrap to the last group and load the wrong scales")
    dense = scales.to(torch.float32)[gi] * (w.to(torch.float32) - z.to(torch.float32)[gi])
    return dense.t().contiguous().to(dtype)          # [in, out] -> [out, in]
