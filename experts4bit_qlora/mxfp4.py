"""MXFP4 dequantization for GPT-OSS fused expert stacks.

GPT-OSS ships its MoE experts in **MXFP4** (OCP Microscaling FP4): each expert
projection is stored as two on-disk tensors,

    {gate_up,down}_proj_blocks   uint8   [..., G, B]   two packed fp4 nibbles/byte
    {gate_up,down}_proj_scales   uint8   [..., G]      one e8m0 exponent per block

where a "block" is B bytes = 2*B fp4 values sharing a single power-of-two scale
(bias 127). The fp4 code is the e2m1 grid ``FP4_VALUES`` (sign * {0,.5,1,1.5,2,3,4,6}).

:func:`dequantize_mxfp4` is a clean-room reimplementation of the reference math
in ``transformers.integrations.mxfp4.convert_moe_packed_tensors`` — it is
verified **bit-identical** to that reference (bf16 and fp32) across the real
gpt-oss-20b/120b expert shapes; see ``tests/test_mxfp4_dequant.py``. This
package owns the primitive so the loader does not depend on a specific
transformers version, and so the "dequantized from the exact released bytes"
step is auditable in one place.

This module only *reads* MXFP4. The training-side primitive that stores the
fused stack back in MXFP4 (``ExpertsMxfp4``) is separate.
"""
from __future__ import annotations

import torch

# e2m1 fp4 value grid, low index = low nibble. Index order matches the OCP MXFP4
# code points and the transformers reference LUT exactly.
FP4_VALUES = (
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
    -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
)


def dequantize_mxfp4(
    blocks: torch.Tensor,
    scales: torch.Tensor,
    *,
    dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Dequantize an MXFP4 fused-expert projection to a dense ``dtype`` tensor.

    Args:
        blocks: uint8 ``[..., G, B]`` — two fp4 nibbles per byte (low nibble first).
        scales: ``[..., G]`` e8m0 exponents (bias 127); ``blocks.shape[:-1] == scales.shape``.
        dtype: output dtype (bf16 or fp32).

    Returns:
        ``[..., 2*B*G-transposed]`` dense tensor matching
        ``transformers`` ``convert_moe_packed_tensors`` bit-for-bit — i.e. the
        layout ``GptOssExperts`` expects for its ``{gate_up,down}_proj`` params.
    """
    if blocks.shape[:-1] != scales.shape:
        raise ValueError(
            f"blocks.shape[:-1]={tuple(blocks.shape[:-1])} != scales.shape={tuple(scales.shape)}"
        )
    blocks = blocks.to(torch.uint8)
    exp = scales.to(torch.int32) - 127  # e8m0 bias
    lut = torch.tensor(FP4_VALUES, dtype=dtype, device=blocks.device)

    *prefix, G, B = blocks.shape
    rows = 1
    for d in prefix:
        rows *= d
    rows *= G

    blk = blocks.reshape(rows, B)
    out = torch.empty(rows, B * 2, dtype=dtype, device=blocks.device)
    out[:, 0::2] = lut[(blk & 0x0F).int()]  # low nibble
    out[:, 1::2] = lut[(blk >> 4).int()]    # high nibble
    torch.ldexp(out, exp.reshape(rows, 1), out=out)  # * 2**exp, per-block scale

    out = out.reshape(*prefix, G, B * 2).view(*prefix, G * B * 2)
    return out.transpose(1, 2).contiguous()
