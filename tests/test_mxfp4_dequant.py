"""MXFP4 dequant: self-contained golden + transformers bit-parity (identity gate)."""
import pytest
import torch

from experts4bit_qlora.mxfp4 import FP4_VALUES, dequantize_mxfp4


def test_fp4_grid():
    # e2m1: sign * {0, .5, 1, 1.5, 2, 3, 4, 6}
    assert len(FP4_VALUES) == 16
    assert FP4_VALUES[:8] == (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
    assert FP4_VALUES[8:] == (-0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0)


def test_golden_vector():
    """Hand-checkable case (self-contained; no transformers needed).

    E=1, dim=2, G=1, B=2 bytes/block:
      row0 bytes 0x21,0x30 -> nibbles (1,2,0,3) -> [.5,1,0,1.5], exp +1 -> [1,2,0,3]
      row1 bytes 0x00,0x54 -> nibbles (0,0,4,5) -> [0,0,2,3],   exp -1 -> [0,0,1,1.5]
    then transpose(1,2) -> [1,4,2].
    """
    blocks = torch.tensor([[[[0x21, 0x30]], [[0x00, 0x54]]]], dtype=torch.uint8)
    scales = torch.tensor([[[128], [126]]], dtype=torch.uint8)  # exp +1 / -1
    out = dequantize_mxfp4(blocks, scales, dtype=torch.float32)
    expected = torch.tensor([[[1.0, 0.0], [2.0, 0.0], [0.0, 1.0], [3.0, 1.5]]])
    assert out.shape == (1, 4, 2)
    assert torch.equal(out, expected)


def test_shape_mismatch_raises():
    with pytest.raises(ValueError):
        dequantize_mxfp4(
            torch.zeros(2, 3, 4, dtype=torch.uint8),
            torch.zeros(2, 5, dtype=torch.uint8),
        )


# --- identity gate: bit-parity with the transformers reference on real shapes ---
_HAS_REF = False
try:
    from transformers.integrations.mxfp4 import convert_moe_packed_tensors  # noqa: E402

    _HAS_REF = True
except Exception:  # pragma: no cover - reference not installed in minimal CI
    pass


@pytest.mark.skipif(not _HAS_REF, reason="transformers gpt-oss mxfp4 reference not installed")
@pytest.mark.parametrize("shape", [(32, 2880, 90, 16), (32, 5760, 90, 16), (4, 128, 8, 16)])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_bit_parity_with_transformers(shape, dtype):
    """Our dequant must equal transformers' convert_moe_packed_tensors bit-for-bit.

    Scales are drawn in a non-overflowing e8m0 range: uniform 0..255 exponents push
    values past the dtype max, producing inf/NaN where torch.equal is meaningless
    (NaN != NaN). Real gpt-oss scales are calibrated, so [-20, 20] is representative.
    """
    torch.manual_seed(0)
    blocks = torch.randint(0, 256, shape, dtype=torch.uint8)
    scales = torch.randint(107, 148, shape[:-1], dtype=torch.uint8)  # exp in [-20, 20]
    ref = convert_moe_packed_tensors(blocks, scales, dtype=dtype)
    ours = dequantize_mxfp4(blocks, scales, dtype=dtype)
    assert not torch.isnan(ref).any(), "test scale range should not overflow"
    assert torch.equal(ref, ours)


# --- real-bytes identity gate: OpenAI's actual gpt-oss-20b released weights ---
import glob
import os

_SHARD = sorted(glob.glob(os.path.expanduser(
    os.environ.get("GPTOSS20B_SHARD_GLOB",
                   "~/hf-cache/models--openai--gpt-oss-20b/snapshots/*/model-00000-of-00002.safetensors")
)))


@pytest.mark.skipif(not (_HAS_REF and _SHARD), reason="gpt-oss-20b shard 0 not cached (set GPTOSS20B_SHARD_GLOB)")
@pytest.mark.parametrize("proj", ["gate_up_proj", "down_proj"])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_real_gptoss20b_bytes(proj, dtype):
    """Dequant of the EXACT released MXFP4 bytes == transformers reference, bit-for-bit.

    The strongest form of the identity gate: not synthetic shapes but OpenAI's
    shipped layer-0 expert tensors. Requires the first shard cached locally.
    """
    from safetensors import safe_open

    with safe_open(_SHARD[0], framework="pt") as f:
        b = f.get_tensor(f"model.layers.0.mlp.experts.{proj}_blocks")
        s = f.get_tensor(f"model.layers.0.mlp.experts.{proj}_scales")
    ref = convert_moe_packed_tensors(b, s, dtype=dtype)
    ours = dequantize_mxfp4(b, s, dtype=dtype)
    assert torch.equal(ref, ours)
