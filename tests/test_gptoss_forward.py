"""GptOssExperts4bit.forward parity vs the real transformers GptOssExperts, on real bytes.

bf16-passthrough isolates the structural logic (layout transpose, gate/up de-interleave,
biases, clamped GLU) from quant error; the nf4 run checks the shipped probe/kernel config
within a bounded tolerance. Skips unless transformers' gpt_oss + a cached shard are present.
"""
import glob
import os

import pytest
import torch

_SHARD = sorted(glob.glob(os.path.expanduser(os.environ.get(
    "GPTOSS20B_SHARD_GLOB",
    "~/hf-cache/models--openai--gpt-oss-20b/snapshots/*/model-00000-of-00002.safetensors",
))))
try:
    from transformers.models.gpt_oss.modeling_gpt_oss import GptOssConfig, GptOssExperts
    _HAS = True
except Exception:
    _HAS = False


@pytest.mark.skipif(not (_HAS and _SHARD), reason="gpt-oss shard/transformers not available")
@pytest.mark.parametrize("quant_type,max_rel_l2", [("bf16", 0.02), ("nf4", 0.12)])
def test_forward_parity_real_bytes(quant_type, max_rel_l2):
    from safetensors import safe_open
    from experts4bit_qlora.mxfp4 import dequantize_mxfp4
    from experts4bit_qlora.gptoss import GptOssExperts4bit

    torch.manual_seed(0)
    with safe_open(_SHARD[0], framework="pt") as f:
        gu = dequantize_mxfp4(f.get_tensor("model.layers.0.mlp.experts.gate_up_proj_blocks"),
                              f.get_tensor("model.layers.0.mlp.experts.gate_up_proj_scales"), dtype=torch.bfloat16)
        gub = f.get_tensor("model.layers.0.mlp.experts.gate_up_proj_bias").to(torch.bfloat16)
        dn = dequantize_mxfp4(f.get_tensor("model.layers.0.mlp.experts.down_proj_blocks"),
                              f.get_tensor("model.layers.0.mlp.experts.down_proj_scales"), dtype=torch.bfloat16)
        dnb = f.get_tensor("model.layers.0.mlp.experts.down_proj_bias").to(torch.bfloat16)

    E, H, twoI = gu.shape
    # reference: transformers GptOssExperts filled with the dequantized weights
    cfg = GptOssConfig(num_local_experts=E, hidden_size=H, intermediate_size=twoI // 2)
    ref = GptOssExperts(cfg).to(torch.bfloat16).eval()
    with torch.no_grad():
        ref.gate_up_proj.copy_(gu); ref.gate_up_proj_bias.copy_(gub)
        ref.down_proj.copy_(dn); ref.down_proj_bias.copy_(dnb)

    ours = GptOssExperts4bit.from_gptoss(gu, gub, dn, dnb, quant_type=quant_type, compute_dtype=torch.bfloat16).eval()

    n, k = 16, cfg.num_experts_per_tok
    x = torch.randn(n, H, dtype=torch.bfloat16)
    idx = torch.stack([torch.randperm(E)[:k] for _ in range(n)])
    scores = torch.softmax(torch.randn(n, k), dim=-1).to(torch.bfloat16)

    with torch.no_grad():
        a = ref(x, idx, scores).float()
        b = ours(x, idx, scores).float()
    # relative Frobenius error: robust to the few large-magnitude GLU outliers that make
    # elementwise atol misleading. bf16 storage (~0.4-0.7%) confirms the structure exactly
    # (a layout/interleave/bias bug would be O(1)); nf4 adds bounded quant error.
    rel_l2 = ((a - b).norm() / (a.norm() + 1e-6)).item()
    max_abs = (a - b).abs().max().item()
    print(f"[{quant_type}] rel_l2={rel_l2:.4g} max_abs={max_abs:.4g}")
    assert rel_l2 < max_rel_l2, f"{quant_type}: rel_l2 {rel_l2:.4g} >= {max_rel_l2}"
