"""The aligned FP8 dequantize path must be BIT-EXACT, not merely close.

`dequantize_fp8_blocks` grew a fast path because the general one holds four full
`[out, in]` tensors at once — an expanded fp32 scale, `weight.float()`, the fp32
product and the result — 12-14 bytes per parameter where the class docstring
claimed 2. On DeepSeek-V4's largest dense weight that is ~403 MB against the ~67 MB
advertised, and it is the allocation that OOMs a 24 GB card on the 284B run.

The fast path decodes in `dtype` with a broadcast scale. That is exact rather than
approximate, and for a reason worth stating: e4m3 -> bf16 is lossless (3 mantissa
bits into 7) and an e8m0 scale is a power of two, so the multiply cannot round.
"Cannot round" is still a claim, so it is tested — including where the product
overflows or goes subnormal, which is the only place the argument could fail.
"""
from __future__ import annotations

import pytest
import torch

from experts4bit_qlora.formats.fp8_blocks import (
    BLOCK,
    _scale_values,
    dequantize_fp8_blocks,
    fp8_block_scale_shape,
)


def _general(weight, scale, *, block_size=BLOCK, dtype=torch.bfloat16):
    """The fp32 route, inlined, as the oracle the fast path must match."""
    if weight.dtype in (torch.uint8, torch.int8):
        weight = weight.view(torch.float8_e4m3fn)
    out, inn = weight.shape
    s = _scale_values(scale.to(weight.device))
    s = s.repeat_interleave(block_size, 0)[:out].repeat_interleave(block_size, 1)[:, :inn]
    return (weight.float() * s).to(dtype)


def _fixture(out, inn, lo, hi, seed, block_size=BLOCK):
    g = torch.Generator().manual_seed(seed)
    w = torch.randint(0, 256, (out, inn), generator=g, dtype=torch.uint8)
    # Exclude the e4m3fn NaN encodings (0x7F / 0xFF). A real checkpoint stores no
    # NaNs, and `torch.equal` is False at every NaN regardless of correctness —
    # which made the first version of this comparison report five bogus failures.
    w[(w & 0x7F) == 0x7F] = 0x10
    s = torch.randint(lo, hi, fp8_block_scale_shape((out, inn), block_size),
                      generator=g, dtype=torch.uint8)
    return w, s


@pytest.mark.parametrize("out,inn,lo,hi,label", [
    (256, 256, 120, 135, "typical exponents"),
    (256, 256, 100, 155, "wide exponents"),
    (256, 256, 60, 200, "extreme — over/underflow territory"),
    (128, 1024, 118, 138, "non-square"),
    (512, 1024, 120, 134, "wq_b-shaped"),
])
def test_aligned_fast_path_is_bit_identical(out, inn, lo, hi, label):
    w, s = _fixture(out, inn, lo, hi, seed=3)
    got, want = dequantize_fp8_blocks(w, s), _general(w, s)
    assert got.dtype == want.dtype and got.shape == want.shape
    assert torch.equal(torch.isnan(got), torch.isnan(want)), f"{label}: NaN positions moved"
    fin = ~torch.isnan(want)
    assert torch.equal(got[fin], want[fin]), (
        f"{label}: the aligned path is not bit-identical — max finite delta "
        f"{(got[fin].float() - want[fin].float()).abs().max():.3e}")


def test_ragged_shapes_still_work_and_agree():
    """The fast path is skipped when a dimension is not a multiple of the block.

    V4 never hits this — every dense tensor is a multiple of 128 — but the
    function is general, and a fast path that silently mis-tiled a ragged shape
    would be worse than no fast path.
    """
    for out, inn in ((300, 256), (256, 300), (130, 130)):
        w, s = _fixture(out, inn, 120, 135, seed=7)
        got, want = dequantize_fp8_blocks(w, s), _general(w, s)
        fin = ~torch.isnan(want)
        assert torch.equal(got[fin], want[fin]), f"ragged {(out, inn)} disagrees"


def test_the_callers_weight_is_never_mutated():
    """The fast path multiplies IN PLACE, so it must own the tensor it scales.

    `weight.to(dtype)` allocates when the dtypes differ — which is always, in
    practice — but "in practice" is exactly the assumption that turns into a
    corrupted frozen base when someone calls this with `dtype` already matching.
    """
    w, s = _fixture(256, 256, 120, 135, seed=11)
    before = w.clone()
    dequantize_fp8_blocks(w, s)
    assert torch.equal(w, before), "dequantize mutated the caller's packed weight"


def test_output_is_not_a_view_of_the_input():
    w, s = _fixture(256, 256, 120, 135, seed=13)
    out = dequantize_fp8_blocks(w, s)
    assert out.data_ptr() != w.data_ptr(), "result aliases the packed storage"
