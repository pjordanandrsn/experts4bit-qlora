"""Dequantize AWQ (Activation-aware Weight Quantization) checkpoints.

AWQ is one of the two most common community 4-bit formats. Unlike every other
format this loader handles, it is **asymmetric** — each group carries a
zero-point as well as a scale — and its nibbles are stored in an interleaved
order that no amount of shape inspection would reveal:

    AWQ_REVERSE_ORDER = [0, 4, 1, 5, 2, 6, 3, 7]

That permutation is taken verbatim from autoawq's ``packing_utils``; getting it
wrong yields a tensor of exactly the right shape full of scrambled weights,
which loads clean and computes garbage. It is pinned by a test that reproduces
autoawq's own unpack.

Layout (bits=4, group_size=g), and note it is TRANSPOSED relative to the
compressed-tensors formats — packing runs along the OUT axis:

* ``qweight``  ``[in, out // (32 // bits)]`` int32
* ``qzeros``   ``[in // g, out // (32 // bits)]`` int32
* ``scales``   ``[in // g, out]`` fp16

Dequant is ``(value - zero) * scale`` per group, and the result is transposed
to the ``[out, in]`` a torch Linear declares.
"""
from __future__ import annotations

import torch

#: Verbatim from autoawq.utils.packing_utils — the interleave AWQ packs with.
AWQ_REVERSE_ORDER = [0, 4, 1, 5, 2, 6, 3, 7]


def _unpack_awq_int32(packed: torch.Tensor, bits: int) -> torch.Tensor:
    """int32 [..., n] -> int8 [..., n * (32//bits)] in AWQ's stored order."""
    shifts = torch.arange(0, 32, bits, device=packed.device)
    out = torch.bitwise_right_shift(packed[:, :, None], shifts[None, None, :])
    return out.view(out.shape[0], -1).to(torch.int8)


def _reverse_awq_order(x: torch.Tensor, bits: int) -> torch.Tensor:
    order = torch.arange(x.shape[-1], dtype=torch.int32, device=x.device)
    order = order.view(-1, 32 // bits)[:, AWQ_REVERSE_ORDER].reshape(-1)
    return x[:, order.long()]


def dequantize_awq(
    qweight: torch.Tensor,   # [in, out//(32//bits)] int32
    qzeros: torch.Tensor,    # [in//group, out//(32//bits)] int32
    scales: torch.Tensor,    # [in//group, out]
    *,
    bits: int = 4,
    dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """-> dense ``[out, in]`` (the orientation a Linear declares).

    ``group_size`` is derived from the shapes rather than trusted from a config:
    it is ``in_features // n_groups``.
    """
    if qweight.dtype is not torch.int32 or qzeros.dtype is not torch.int32:
        raise ValueError("AWQ qweight/qzeros must be int32")
    mask = (1 << bits) - 1
    in_f, groups = qweight.shape[0], qzeros.shape[0]
    if in_f % groups:
        raise ValueError(f"{in_f} in-features not divisible by {groups} groups")
    group = in_f // groups

    iw = _reverse_awq_order(_unpack_awq_int32(qweight, bits), bits) & mask
    iz = _reverse_awq_order(_unpack_awq_int32(qzeros, bits), bits) & mask
    if iw.shape[-1] != scales.shape[-1]:
        raise ValueError(
            f"unpacked out-features {iw.shape[-1]} != scales {scales.shape[-1]}")

    # Broadcast each group's zero/scale across its `group` input rows.
    z = iz.to(torch.float32).repeat_interleave(group, dim=0)
    s = scales.to(torch.float32).repeat_interleave(group, dim=0)
    dense = (iw.to(torch.float32) - z) * s        # [in, out]
    return dense.t().contiguous().to(dtype)       # -> [out, in]
