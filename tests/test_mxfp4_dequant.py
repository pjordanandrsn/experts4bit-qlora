"""MXFP4 dequant: self-contained golden + transformers bit-parity (identity gate)."""
import glob
import os

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("bitsandbytes")

from experts4bit_qlora.mxfp4 import FP4_VALUES, dequantize_mxfp4  # noqa: E402


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


# ------------------------------------------------------------ rank handling --
# `transpose(1, 2)` hardcoded gpt-oss's rank-4 blocks. A single expert's
# `[rows, G, B]` — the per-expert layout of a DeepSeek-V3-lineage checkpoint
# like released Kimi K3 — raised IndexError. `transpose(-2, -1)` is equivalent
# for the rank-4 case and correct for both.
def test_single_expert_rank3_blocks_dequantize():
    rows, G, B = 8, 4, 16
    g = torch.Generator().manual_seed(1689)
    blocks = torch.randint(0, 256, (rows, G, B), generator=g, dtype=torch.uint8)
    scales = torch.randint(112, 123, (rows, G), generator=g, dtype=torch.uint8)
    out = dequantize_mxfp4(blocks, scales, dtype=torch.bfloat16)
    assert out.shape == (G * B * 2, rows)          # [K, rows] — transposed
    assert torch.isfinite(out).all()


def test_rank3_matches_the_rank4_path_expert_by_expert():
    """A [E, rows, G, B] call and E separate [rows, G, B] calls must agree."""
    E, rows, G, B = 3, 8, 4, 16
    g = torch.Generator().manual_seed(7)
    blocks = torch.randint(0, 256, (E, rows, G, B), generator=g, dtype=torch.uint8)
    scales = torch.randint(112, 123, (E, rows, G), generator=g, dtype=torch.uint8)
    fused = dequantize_mxfp4(blocks, scales, dtype=torch.float32)
    for e in range(E):
        one = dequantize_mxfp4(blocks[e], scales[e], dtype=torch.float32)
        assert torch.equal(one, fused[e])


def test_rank2_blocks_raise_a_useful_error():
    with pytest.raises(ValueError, match="at least one leading row axis"):
        dequantize_mxfp4(torch.zeros(4, 16, dtype=torch.uint8),
                         torch.zeros(4, dtype=torch.uint8))


def test_kimi_k3_shaped_expert_projection():
    """Real released-K3 geometry: w1 packed [3072, 1792] uint8 with a
    [3072, 112] e8m0 scale (group_size 32) -> W^T [3584, 3072]."""
    R, KH, G = 96, 1792, 112          # R reduced; KH/G/B are the real values
    B = KH // G
    g = torch.Generator().manual_seed(30)
    packed = torch.randint(0, 256, (R, KH), generator=g, dtype=torch.uint8)
    scale = torch.randint(112, 123, (R, G), generator=g, dtype=torch.uint8)
    out = dequantize_mxfp4(packed.reshape(R, G, B), scale, dtype=torch.bfloat16)
    assert out.shape == (KH * 2, R)
    assert B * 2 == 32               # group_size 32, as K3's config declares
    assert torch.isfinite(out).all()
