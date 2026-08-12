"""Dequantize compressed-tensors int-packed weights (llm-compressor / vLLM).

The ``pack-quantized`` format ships each matrix as three tensors:
``X.weight_packed`` (int32, ``num_bits`` values densely packed per word),
``X.weight_scale`` (per-group scales), and ``X.weight_shape`` (the logical
``[out, in]`` dims). It is one of the most common ways open weights are shipped
4-bit, so decoding it covers many models, not just the one that prompted it.

``num_bits`` and ``group_size`` are DERIVED from the three tensors' shapes, not
read from the config — several checkpoints (Kimi-K2.5) ship an empty
``quantization_config``, and the shapes are unambiguous: packed columns imply
the bit width, scale columns imply the group. The unpack matches
``compressed_tensors.unpack_from_int32`` bit-for-bit (verified in tests):
element ``i`` lives at global bit position ``i*num_bits``, values are signed
(subtract ``2**(num_bits-1)``).
"""
from __future__ import annotations

import torch


def _derive(packed_cols: int, in_features: int, scale_cols: int):
    num_bits = (32 * packed_cols) // in_features
    if num_bits < 1 or num_bits > 8 or (32 * packed_cols) % in_features:
        raise ValueError(
            f"cannot derive a whole 1..8 bit width from packed {packed_cols} "
            f"words and {in_features} in-features")
    if in_features % scale_cols:
        raise ValueError(
            f"{in_features} in-features not divisible by {scale_cols} scale groups")
    return num_bits, in_features // scale_cols


def dequantize_compressed_int(
    packed: torch.Tensor,      # [..., out, in//(32//num_bits)] int32
    scale: torch.Tensor,       # [..., out, in//group_size]
    shape: torch.Tensor,       # [out, in] (int) — the logical dense dims
    *,
    dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """-> dense ``[..., out, in]`` in ``dtype``.

    Signed group-symmetric integer quantization: unpack the little-end-first
    ``num_bits`` fields, recentre to signed, and multiply by the group scale.
    Leading batch dims (a stacked expert axis) are handled by flattening.
    """
    if packed.dtype is not torch.int32:
        raise ValueError(f"packed must be int32, got {packed.dtype}")
    out_f, in_f = int(shape[-2]), int(shape[-1])
    num_bits, group_size = _derive(packed.shape[-1], in_f, scale.shape[-1])

    lead = packed.shape[:-2]
    p = packed.reshape(-1, out_f, packed.shape[-1])
    mask = (1 << num_bits) - 1
    per_word = 32 // num_bits
    # Vectorized: extract every field in one shift/mask over a broadcast offset
    # axis, rather than looping over in_features. The loop version was correct
    # but ran the unpack element-by-element in Python — minutes per large tensor.
    offs = (torch.arange(per_word, device=p.device, dtype=torch.int32) * num_bits)
    fields = (p.unsqueeze(-1) >> offs) & mask         # [n, out, words, per_word]
    unpacked = fields.reshape(p.shape[0], out_f, -1)[:, :, :in_f]
    signed = unpacked - (1 << (num_bits - 1))         # symmetric recentre

    sc = scale.reshape(-1, out_f, scale.shape[-1]).to(torch.float32)
    sc = sc.repeat_interleave(group_size, dim=-1)      # [n, out, in]
    dense = signed.to(torch.float32) * sc
    return dense.reshape(*lead, out_f, in_f).to(dtype)
