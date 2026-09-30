"""Dequantize NVFP4 weights (compressed-tensors nvfp4, NVIDIA ModelOpt FP4).

NVIDIA's 4-bit float (E2M1) with two-level scaling, as shipped by
llm-compressor's ``nvfp4-pack-quantized`` format (Solar-Open-100B-NVFP4,
Kimi-K2.7-Code-NVFP4, growing). Each matrix is:

* ``X.weight_packed`` — uint8, two E2M1 nibbles per byte (low nibble first).
* ``X.weight_scale`` — per-GROUP scale (group_size 16), in fp8 (e4m3).
* ``X.weight_global_scale`` — a single per-TENSOR fp32 scale.

ModelOpt FP4 ships the same bytes under ``X.weight`` / ``X.weight_scale`` /
``X.weight_scale_2``. The two per-tensor scales are RECIPROCAL conventions, and
nothing in the tensors says which one a file uses — only the key name does:

* compressed-tensors: ``weight_global_scale = 448 * 6 / amax`` is a DIVISOR —
  ``E2M1[nibble] * (group_scale / global_scale)`` (compressed-tensors
  ``generate_gparam`` and ``_dequantize``).
* ModelOpt: ``weight_scale_2 = amax / (448 * 6)`` is a MULTIPLIER —
  ``E2M1[nibble] * group_scale * weight_scale_2`` (ModelOpt
  ``NVFP4QTensor.get_weights_scaling_factor_2``).

Multiplying by a compressed-tensors global scale decodes every weight
``global_scale**2`` too large (~5e7 on a real Qwen3-30B-A3B tensor) and still
loads clean, so the caller must name the convention (#788). The E2M1 codebook,
nibble order and scale convention match compressed-tensors' own decompress
exactly (verified in tests). As with the int pack format, ``num_bits`` is fixed
(4) and ``group_size`` is derived from the packed vs scale shapes rather than
trusted from a config that several checkpoints ship empty.
"""
from __future__ import annotations

import torch

# E2M1 magnitude codebook (index 0..7); the 4th bit is the sign.
_E2M1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])

#: The per-tensor scale's meaning, by the format that wrote it (module docstring).
CONVENTIONS = ("compressed-tensors", "modelopt")


def dequantize_nvfp4(
    packed: torch.Tensor,        # [..., out, in//2] uint8
    scale: torch.Tensor,         # [..., out, in//group_size] (fp8/any float)
    global_scale: torch.Tensor,  # scalar-ish per-tensor fp32
    *,
    convention: str,
    dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """-> dense ``[..., out, in]`` in ``dtype``.

    Two-level symmetric fp4: decode each E2M1 nibble and apply its group scale
    and the single per-tensor scale. ``convention`` is required and says what
    that per-tensor scale is: ``"compressed-tensors"`` (``weight_global_scale``,
    a divisor) or ``"modelopt"`` (``weight_scale_2``, a multiplier). A
    non-finite per-tensor scale is refused, and so is a zero divisor, rather
    than decoded to inf/nan.
    Leading batch dims (a stacked expert axis) are handled by flattening.
    """
    if convention not in CONVENTIONS:
        raise ValueError(
            f"nvfp4 convention must be one of {CONVENTIONS}, got {convention!r}")
    if packed.dtype is not torch.uint8:
        raise ValueError(f"nvfp4 packed must be uint8, got {packed.dtype}")
    out_f, half = packed.shape[-2], packed.shape[-1]
    in_f = half * 2
    if in_f % scale.shape[-1]:
        raise ValueError(
            f"{in_f} in-features not divisible by {scale.shape[-1]} scale groups")
    group = in_f // scale.shape[-1]

    lead = packed.shape[:-2]
    p = packed.reshape(-1, out_f, half)
    low = (p & 0x0F).to(torch.long)
    high = ((p & 0xF0) >> 4).to(torch.long)
    # interleave low, high along the in-features axis -> [..., out, in]
    nib = torch.stack((low, high), dim=-1).reshape(p.shape[0], out_f, in_f)
    signs = (nib & 0x08).to(torch.bool)
    mag = nib & 0x07
    cb = _E2M1.to(device=packed.device)
    vals = cb[mag] * torch.where(signs, -1.0, 1.0)

    sc = scale.reshape(-1, out_f, scale.shape[-1]).to(torch.float32)
    sc = sc.repeat_interleave(group, dim=-1)
    gs = global_scale.to(torch.float32).reshape(())
    if not bool(torch.isfinite(gs)):
        raise ValueError(f"nvfp4 per-tensor scale must be finite, got {float(gs)}")
    if convention == "compressed-tensors":
        if float(gs) == 0.0:      # a divisor: 0 would decode to inf/nan
            raise ValueError("nvfp4 compressed-tensors global scale is a divisor "
                             "and must be non-zero, got 0.0")
        # compressed-tensors' own order: scale / global_scale, then x_q * scale.
        dense = vals * (sc / gs)
    else:
        # A multiplier; 0 is legal (ModelOpt's amax / 2688 of an all-zero tensor).
        dense = vals * sc * gs
    return dense.reshape(*lead, out_f, in_f).to(dtype)
