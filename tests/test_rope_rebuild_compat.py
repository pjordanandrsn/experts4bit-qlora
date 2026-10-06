"""Both loaders' meta ``inv_freq`` rebuild works on every transformers the package supports.

transformers 5.19 changed ``<Model>RotaryEmbedding.compute_default_rope_parameters`` from ``(config, device=None, ...)``
to ``(config, **kwargs)``; a positional ``device`` raised ``TypeError`` there (#1280 fixed moe_load; glimmer_load carried
the same call). A rotary module built under ``torch.device("meta")`` must come back with the bytes its own constructor
computes on the CPU.
"""
import pytest
import torch
from torch import nn

transformers = pytest.importorskip("transformers")
qm = pytest.importorskip("transformers.models.qwen3_moe.modeling_qwen3_moe")


def _cfg():
    from transformers import Qwen3MoeConfig
    return Qwen3MoeConfig(hidden_size=64, num_attention_heads=4, num_key_value_heads=2, head_dim=16,
                          intermediate_size=32, moe_intermediate_size=16, num_experts=4, num_experts_per_tok=2,
                          num_hidden_layers=1, vocab_size=128)


def _meta_rotary(cfg):
    with torch.device("meta"):
        return qm.Qwen3MoeRotaryEmbedding(cfg)


@pytest.mark.parametrize("loader", ["moe_load", "glimmer_load"])
def test_meta_inv_freq_is_rebuilt_to_the_constructors_bytes(loader):
    import importlib
    mod = importlib.import_module(f"experts4bit_qlora.arch.{loader}")
    cfg = _cfg()
    ref = qm.Qwen3MoeRotaryEmbedding(cfg)
    outer = nn.Module()
    outer.model = nn.Module()
    outer.model.language_model = nn.Module()          # glimmer_load rebuilds the text tower only
    outer.model.language_model.rotary_emb = _meta_rotary(cfg)
    rebuilt = mod._materialize_computed_buffers(outer, torch.device("cpu"))
    rot = outer.model.language_model.rotary_emb
    assert "model.language_model.rotary_emb.inv_freq" in rebuilt
    assert not rot.inv_freq.is_meta and rot.inv_freq.device.type == "cpu"
    assert torch.equal(rot.inv_freq, ref.inv_freq)
