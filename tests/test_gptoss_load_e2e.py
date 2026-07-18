"""End-to-end: load the full gpt-oss-20b on a single small GPU via expert offload
and generate. Heavy + gated (needs CUDA + the full model cached); skips in CI.

Validated 2026-07-17 on an A2000 12GB: 24/24 MoE layers -> GptOssExperts4bit,
3.61 GB resident after load, 5.14 GB peak in generation, coherent output.
"""
import glob
import os

import pytest
import torch

_SNAP = sorted(glob.glob(os.path.expanduser(os.environ.get(
    "GPTOSS20B_DIR", "~/hf-cache/models--openai--gpt-oss-20b/snapshots/*/config.json"))))
_RUN = os.environ.get("RUN_GPTOSS_E2E") == "1"


@pytest.mark.skipif(not (_RUN and _SNAP and torch.cuda.is_available()),
                    reason="set RUN_GPTOSS_E2E=1 with gpt-oss-20b cached + CUDA (slow: full-model load)")
def test_gptoss20b_offload_load_and_generate():
    from transformers import AutoTokenizer
    from experts4bit_qlora.loader import load_moe_4bit_streaming
    from experts4bit_qlora.gptoss import GptOssExperts4bit

    torch.cuda.reset_peak_memory_stats()
    model, cfg = load_moe_4bit_streaming(
        "openai/gpt-oss-20b", device="cuda:0", dtype=torch.bfloat16,
        r=8, alpha=16, offload=True, quant_type="nf4",
    )
    n = sum(isinstance(m, GptOssExperts4bit) for m in model.modules())
    assert n == cfg.num_hidden_layers, f"{n} experts layers != {cfg.num_hidden_layers}"

    tok = AutoTokenizer.from_pretrained("openai/gpt-oss-20b")
    ids = tok("The capital of France is", return_tensors="pt").input_ids.to("cuda:0")
    model.eval()
    with torch.no_grad():
        out = model.generate(input_ids=ids, max_new_tokens=12, do_sample=False)
    text = tok.decode(out[0], skip_special_tokens=True)
    peak = torch.cuda.max_memory_allocated() / 1e9
    assert len(text) > len("The capital of France is"), "no tokens generated"
    assert peak < 11.0, f"offload should keep peak well under the ~11GB resident size; got {peak:.2f}GB"
