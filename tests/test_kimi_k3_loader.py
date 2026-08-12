"""Gates for the Kimi K3 loader seam.

K3 is 1.5 TB, so nothing here downloads it. These pin the parts that are decidable
without the weights: the registration, the multimodal prefix (K3 reverses Gemma-4's
order), the trust_remote_code plumbing, and — most importantly — the ORIENTATION of
the per-expert MXFP4 dequant, which is the one mistake that would produce a model
that loads with correct shapes and computes garbage.
"""
import os

import pytest
import torch

# `experts4bit_qlora.loader` transitively imports bitsandbytes, so on a machine
# without it this module raised at COLLECTION time and vanished from the run --
# reading as "nothing to test" rather than "not verified". conftest's collection
# guard flags exactly that. Declare the dependency so it reports SKIPPED.
pytest.importorskip("bitsandbytes")

from experts4bit_qlora.loader import (
    K3_PER_EXPERT_MXFP4, MULTIMODAL_CKPT_PREFIX, SUPPORTED_ARCHITECTURES,
)
from experts4bit_qlora.formats.mxfp4 import FP4_VALUES, dequantize_mxfp4


def test_kimi_k3_is_registered_under_block_sparse_moe():
    assert SUPPORTED_ARCHITECTURES.get("kimi_k3") == "block_sparse_moe.experts"


def test_k3_multimodal_prefix_reverses_gemma4_order():
    """Gemma-4 nests `model.language_model.`; K3 ships `language_model.model.`.
    Deriving the prefix from the mere presence of `text_config` gets K3 wrong."""
    assert MULTIMODAL_CKPT_PREFIX["kimi_k3"] == "language_model.model."
    assert MULTIMODAL_CKPT_PREFIX.get("gemma4") is None  # falls back to the default


def test_k3_projection_map_is_w1_gate_w3_up_w2_down():
    (gate, up, down), packed, scale = K3_PER_EXPERT_MXFP4["kimi_k3"]
    assert (gate, up, down) == ("w1", "w3", "w2")
    assert (packed, scale) == ("weight_packed", "weight_scale")


def test_trust_remote_code_defaults_off_and_reads_env(monkeypatch):
    """Executing checkpoint-supplied code must never be the default."""
    import experts4bit_qlora.loader as L
    monkeypatch.delenv("E4B_TRUST_REMOTE_CODE", raising=False)
    with pytest.raises(Exception):
        # bogus id: we only care that it does not silently enable remote code.
        L.load_moe_4bit_streaming("does/not/exist", "cpu", torch.bfloat16, 8, 16)
    assert os.environ.get("E4B_TRUST_REMOTE_CODE") is None


# --------------------------------------------------------------- orientation --
def _pack_mxfp4(rows, K, *, seed=0):
    """Build (blocks, scales) for a [rows, K] logical matrix, MXFP4 group-32."""
    assert K % 32 == 0
    g = torch.Generator().manual_seed(seed)
    G, B = K // 32, 16
    blocks = torch.randint(0, 256, (rows, G, B), generator=g, dtype=torch.uint8)
    scales = torch.randint(120, 128, (rows, G), generator=g, dtype=torch.uint8)
    return blocks, scales


def test_dequant_then_T_recovers_stored_orientation():
    """`dequantize_mxfp4` returns [K, rows] (the GptOssExperts layout). The loader's
    per-expert K3 branch needs STORED [rows, K] for `from_float`, so it transposes.
    This pins that the transpose is required and lands the right way round."""
    rows, K = 12, 64
    blocks, scales = _pack_mxfp4(rows, K)
    w = dequantize_mxfp4(blocks, scales, dtype=torch.float32)
    assert w.shape == (K, rows), "dequant returns the transpose of the stored matrix"
    stored = w.T.contiguous()
    assert stored.shape == (rows, K)

    # element-exact: byte b of block g holds two fp4 nibbles, low nibble first,
    # scaled by 2**(e8m0 - 127).
    lut = torch.tensor(FP4_VALUES, dtype=torch.float32)
    for r in (0, rows - 1):
        for gi in (0, K // 32 - 1):
            for b in (0, 15):
                byte = int(blocks[r, gi, b])
                exp = int(scales[r, gi]) - 127
                lo = lut[byte & 0x0F] * (2.0 ** exp)
                hi = lut[byte >> 4] * (2.0 ** exp)
                col = gi * 32 + 2 * b
                assert stored[r, col] == lo, (r, gi, b, "low nibble")
                assert stored[r, col + 1] == hi, (r, gi, b, "high nibble")


def test_k3_expert_fusion_shapes_match_from_float_contract():
    """gate_up must be [2*inter, latent] and down [latent, inter] per expert, using
    the real released K3 geometry (inter 3072, latent 3584) at reduced expert count."""
    INTER, LATENT, E = 3072, 3584, 2
    gate_up_rows, down_rows = [], []
    for e in range(E):
        # w1/w3 are stored [inter, latent]; w2 is stored [latent, inter]
        gu = []
        for seed, _proj in ((e * 3 + 0, "w1"), (e * 3 + 1, "w3")):
            bl, sc = _pack_mxfp4(INTER, LATENT, seed=seed)
            gu.append(dequantize_mxfp4(bl, sc, dtype=torch.float32).T.contiguous())
        gate_up_rows.append(torch.cat(gu, dim=0))
        bl, sc = _pack_mxfp4(LATENT, INTER, seed=e * 3 + 2)
        down_rows.append(dequantize_mxfp4(bl, sc, dtype=torch.float32).T.contiguous())
    gate_up = torch.stack(gate_up_rows)
    down = torch.stack(down_rows)
    assert gate_up.shape == (E, 2 * INTER, LATENT)
    assert down.shape == (E, LATENT, INTER)
    assert torch.isfinite(gate_up).all() and torch.isfinite(down).all()
