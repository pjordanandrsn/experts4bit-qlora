"""Structural tests: the generalized streaming loader handles each supported fused-MoE architecture.

For each architecture, build a tiny model, write it as a checkpoint in the on-disk expert layout that
architecture's real checkpoints use (per-expert for OLMoE/Qwen3, fused for Gemma-4), run
``load_moe_4bit_streaming`` end-to-end, and assert experts were quantized to ``Experts4bit`` +
``ExpertsLoRA``, attention LoRA attached, no meta tensors remain, and a forward pass runs. Requires a
CUDA GPU + bitsandbytes (``Experts4bit`` is a 4-bit GPU primitive).
"""

import json
import os

import pytest
import torch

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="Experts4bit loader path is CUDA-only")


def _olmoe():
    from transformers.models.olmoe.configuration_olmoe import OlmoeConfig
    from transformers.models.olmoe.modeling_olmoe import OlmoeForCausalLM

    return OlmoeForCausalLM(
        OlmoeConfig(
            hidden_size=64,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=4,
            num_experts=8,
            num_experts_per_tok=2,
            vocab_size=128,
        )
    )


def _qwen3_moe():
    from transformers.models.qwen3_moe.configuration_qwen3_moe import Qwen3MoeConfig
    from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeForCausalLM

    return Qwen3MoeForCausalLM(
        Qwen3MoeConfig(
            hidden_size=64,
            intermediate_size=128,
            moe_intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            num_experts=8,
            num_experts_per_tok=2,
            vocab_size=128,
            decoder_sparse_step=1,
            mlp_only_layers=[],
            head_dim=16,
        )
    )


def _gemma4():
    from transformers.models.gemma4.configuration_gemma4 import Gemma4TextConfig
    from transformers.models.gemma4.modeling_gemma4 import Gemma4ForCausalLM

    return Gemma4ForCausalLM(
        Gemma4TextConfig(
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            vocab_size=128,
            head_dim=16,
            num_experts=8,
            top_k_experts=2,
            moe_intermediate_size=64,
            enable_moe_block=True,
        )
    )


def _write_ckpt(model, d, per_expert):
    """Save a checkpoint. per_expert=True splits fused experts back to per-expert Linears (OLMoE/Qwen3
    on-disk layout); per_expert=False keeps them fused (Gemma-4 on-disk layout)."""
    from safetensors.torch import save_file

    new = {}
    for k, v in model.state_dict().items():
        if per_expert and k.endswith("experts.gate_up_proj"):  # [n_exp, 2*inter, hidden] -> per-expert
            base = k[: -len("gate_up_proj")]
            for e in range(v.shape[0]):
                g, u = v[e].chunk(2, dim=0)
                new[f"{base}{e}.gate_proj.weight"] = g.contiguous()
                new[f"{base}{e}.up_proj.weight"] = u.contiguous()
        elif per_expert and k.endswith("experts.down_proj"):  # [n_exp, hidden, inter] -> per-expert
            base = k[: -len("down_proj")]
            for e in range(v.shape[0]):
                new[f"{base}{e}.down_proj.weight"] = v[e].contiguous()
        else:
            new[k] = v  # keep fused (Gemma-4) or non-expert tensors as-is
    new = {k: v.to(torch.bfloat16).contiguous() for k, v in new.items()}
    save_file(new, os.path.join(d, "model.safetensors"))
    json.dump(
        {"weight_map": {k: "model.safetensors" for k in new}},
        open(os.path.join(d, "model.safetensors.index.json"), "w"),
    )
    model.config.save_pretrained(d)


@cuda
@pytest.mark.parametrize(
    "build,per_expert",
    [(_olmoe, True), (_qwen3_moe, True), (_gemma4, False)],
    ids=["olmoe", "qwen3_moe", "gemma4"],
)
def test_loader_handles_architecture(build, per_expert, tmp_path):
    from experts4bit_qlora import ExpertsLoRA
    from experts4bit_qlora.loader import load_moe_4bit_streaming
    from experts4bit_qlora.lora import add_attention_lora

    torch.manual_seed(0)
    _write_ckpt(build(), str(tmp_path), per_expert=per_expert)
    model, cfg = load_moe_4bit_streaming(str(tmp_path), "cuda", torch.bfloat16, r=4, alpha=8)
    n_attn = add_attention_lora(model, 4, 8, torch.bfloat16)

    n_expert_mods = sum(isinstance(m, ExpertsLoRA) for m in model.modules())
    assert 1 <= n_expert_mods <= cfg.num_hidden_layers  # experts replaced on the MoE layers
    assert n_attn == cfg.num_hidden_layers * 4  # q/k/v/o per layer
    assert not [n for n, t in list(model.named_parameters()) + list(model.named_buffers()) if t.is_meta]

    model.config.use_cache = False
    out = model(input_ids=torch.randint(0, cfg.vocab_size, (1, 8), device="cuda"))
    assert tuple(out.logits.shape) == (1, 8, cfg.vocab_size)


def test_unsupported_model_type_errors():
    """A non-fused-MoE architecture (e.g. a dense model) fails fast with a clear message."""
    from experts4bit_qlora.loader import SUPPORTED_MODEL_TYPES, load_moe_4bit_streaming

    assert {"olmoe", "qwen3_moe", "gemma4"} <= SUPPORTED_MODEL_TYPES
    with pytest.raises(NotImplementedError, match="Unsupported model_type"):
        load_moe_4bit_streaming("gpt2", "cuda", torch.bfloat16, r=4, alpha=8)
