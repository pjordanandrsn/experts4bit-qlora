"""A single Triton kernel that fully dequantizes a **double-quantized** NF4 tensor
(bitsandbytes ``compress_statistics=True``) to fp16/bf16 — the Unsloth Puzzles "Task A" shape.

Both dequantizations happen in **one** kernel launch, with no intermediate absmax buffer:

* the per-64-block ``absmax`` is itself quantized (nested), so it is dequantized on the fly as
  ``state2.code[absmax_u8[blk]] * state2.absmax[blk // 256] + offset`` (fp32);
* the NF4 weight nibble is looked up in the 16-entry codebook and scaled by that absmax.

bitsandbytes packs weights row-major, element ``2i`` in the **high** nibble of byte ``i`` and
``2i+1`` in the **low** nibble; ``absmax`` has one entry per 64 weights, and ``state2.absmax`` one
fp32 per 256 absmax entries (i.e. per ``64*256 = 16384`` weights). Verified bit-exact against
``bitsandbytes.functional.dequantize_4bit`` for fp16 and bf16 (``tests/test_triton_nf4.py``), and
measured ~1.3x faster than it on an RTX A2000. The nibble unpack uses inline PTX (``bfe.u32``).

``your_dequantize_nf4(module)`` matches the Unsloth-puzzle harness and drops into its
``test_dequantize``; ``dequantize_nf4_compiled`` is a ``torch.compile``-safe variant (registered as a
``triton_op`` so Dynamo traces it as one opaque op, no graph break).
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _dequantize_nf4_kernel(
    w_ptr,  # *u8   packed NF4 weights, numel // 2
    absmax_ptr,  # *u8   nested-quantized per-64-block absmax, numel // 64
    s2code_ptr,  # *fp32 second-level codebook (256)
    s2absmax_ptr,  # *fp32 second-level scales, numel // 16384
    nf4_ptr,  # *fp32 NF4 codebook (16)
    out_ptr,  # *out  numel
    offset,  # fp32 scalar added back after the nested dequant
    numel,
    NESTED: tl.constexpr,  # blocksize * state2.blocksize (= 64 * 256 = 16384)
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < numel

    # --- dequant 1: reconstruct the fp32 block scale (nested / double dequant), fused, no buffer ---
    blk = offs // 64  # which per-64-block this element belongs to
    a_idx = tl.load(absmax_ptr + blk, mask=mask, other=0).to(tl.int32)  # u8 index into s2code
    s2c = tl.load(s2code_ptr + a_idx)  # gather 256-entry codebook
    s2a = tl.load(s2absmax_ptr + (offs // NESTED), mask=mask, other=0.0)
    absmax = s2c * s2a + offset  # fp32 per-element block scale

    # --- dequant 2: NF4 nibble -> codebook value, scaled by absmax ---
    byte = tl.load(w_ptr + (offs // 2), mask=mask, other=0, eviction_policy="evict_first").to(tl.int32)
    start = tl.where((offs % 2) == 0, 4, 0).to(tl.int32)  # high nibble for even element, low for odd
    # custom PTX: bit-field-extract 4 bits at `start` in a single instruction (register start pos),
    # vs a shift + mask. Correct + ~same speed; verified against bnb in tests/test_triton_nf4.py.
    nibble = tl.inline_asm_elementwise(
        asm="bfe.u32 $0, $1, $2, 4;", constraints="=r,r,r", args=[byte, start],
        dtype=tl.int32, is_pure=True, pack=1,
    )
    value = tl.load(nf4_ptr + nibble)  # gather 16-entry NF4 codebook

    tl.store(out_ptr + offs, (value * absmax).to(out_ptr.dtype.element_ty), mask=mask)


_BLOCK = 1024


def _launch(w, absmax, s2code, s2absmax, nf4code, out, offset, numel, nested):
    _dequantize_nf4_kernel[(triton.cdiv(numel, _BLOCK),)](
        w, absmax, s2code, s2absmax, nf4code, out, offset, numel, NESTED=nested, BLOCK=_BLOCK
    )


def _your_dequantize_nf4(weight: torch.Tensor, quant_state) -> torch.Tensor:
    """Dequantize a bitsandbytes double-quantized NF4 ``weight`` given its ``quant_state``."""
    qs = quant_state
    out_features, in_features = qs.shape
    numel = out_features * in_features
    out = torch.empty(numel, dtype=qs.dtype, device=weight.device)
    _launch(
        weight.reshape(-1), qs.absmax, qs.state2.code, qs.state2.absmax, qs.code, out,
        float(qs.offset), numel, qs.blocksize * qs.state2.blocksize,
    )
    return out.view(out_features, in_features)


def your_dequantize_nf4(weight) -> torch.Tensor:
    """Unsloth-puzzle entry point: dequantize a ``bnb.nn.Linear4bit``'s weight to its compute dtype."""
    return _your_dequantize_nf4(weight.weight.data, weight.weight.quant_state)


# ---------------------------------------------------------------------------------------------------
# torch.compile-safe variant: register the kernel as a triton_op so Dynamo treats the whole thing as
# one opaque custom op (no graph break), with a fake/meta impl for shape+dtype propagation.
# ---------------------------------------------------------------------------------------------------
try:
    from torch.library import triton_op, wrap_triton

    @triton_op("experts4bit::dequantize_nf4", mutates_args={})
    def _dequantize_nf4_op(
        weight: torch.Tensor,
        absmax: torch.Tensor,
        s2code: torch.Tensor,
        s2absmax: torch.Tensor,
        nf4code: torch.Tensor,
        offset: float,
        out_features: int,
        in_features: int,
        nested: int,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        numel = out_features * in_features
        out = torch.empty(numel, dtype=dtype, device=weight.device)
        wrap_triton(_dequantize_nf4_kernel)[(triton.cdiv(numel, _BLOCK),)](
            weight.reshape(-1), absmax, s2code, s2absmax, nf4code, out, offset, numel,
            NESTED=nested, BLOCK=_BLOCK,
        )
        return out.view(out_features, in_features)

    @_dequantize_nf4_op.register_fake
    def _(weight, absmax, s2code, s2absmax, nf4code, offset, out_features, in_features, nested, dtype):
        return weight.new_empty((out_features, in_features), dtype=dtype)

    def dequantize_nf4_compiled(weight) -> torch.Tensor:
        """``torch.compile``-safe dequant (routes through the registered ``triton_op``)."""
        qs = weight.weight.quant_state
        o, i = qs.shape
        return _dequantize_nf4_op(
            weight.weight.data.reshape(-1), qs.absmax, qs.state2.code, qs.state2.absmax, qs.code,
            float(qs.offset), o, i, qs.blocksize * qs.state2.blocksize, qs.dtype,
        )
except (ImportError, AttributeError):  # older torch without triton_op
    dequantize_nf4_compiled = None
