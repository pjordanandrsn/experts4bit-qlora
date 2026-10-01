# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""bench/p60/record_eids.py records routed expert ids for every MoE family, not only Qwen3.

The recorder hooks each block's ``experts`` call and reads its ``top_k_index`` argument, so it needs no knowledge of
the router's name (``gate`` or ``router``) or of its return shape. That shape differs by family:
- Qwen3-MoE and OLMoE return ``(logits, weights, index)``;
- Granite-MoE returns ``(index, weights, logits)``;
- gpt-oss returns ``(logits, scores, index)``.

For each family, a tiny random model decodes a few steps at batch 5. The recorded ids must equal, call for call, the
indices the family's own router returned. The test also pins the shape and the prefill-call accounting.
"""
import importlib.util
import os

import pytest
import torch

pytest.importorskip("transformers")

HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("record_eids", os.path.join(HERE, "..", "bench", "p60", "record_eids.py"))
record_eids = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(record_eids)

COMMON = dict(hidden_size=64, num_hidden_layers=3, num_attention_heads=4, num_key_value_heads=2, vocab_size=512,
              max_position_embeddings=256)


def _qwen3():
    from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeConfig, Qwen3MoeForCausalLM
    cfg = Qwen3MoeConfig(intermediate_size=128, moe_intermediate_size=32, num_experts=16, num_experts_per_tok=4,
                         head_dim=16, **COMMON)
    return Qwen3MoeForCausalLM, cfg, 4, "gate", 2


def _olmoe():
    from transformers.models.olmoe.modeling_olmoe import OlmoeConfig, OlmoeForCausalLM
    cfg = OlmoeConfig(intermediate_size=32, num_experts=16, num_experts_per_tok=4, **COMMON)
    return OlmoeForCausalLM, cfg, 4, "gate", 2


def _granite():
    from transformers.models.granitemoe.modeling_granitemoe import GraniteMoeConfig, GraniteMoeForCausalLM
    cfg = GraniteMoeConfig(intermediate_size=32, num_local_experts=8, num_experts_per_tok=2, **COMMON)
    return GraniteMoeForCausalLM, cfg, 2, "router", 0


def _gpt_oss():
    from transformers.models.gpt_oss.modeling_gpt_oss import GptOssConfig, GptOssForCausalLM
    cfg = GptOssConfig(intermediate_size=32, num_local_experts=8, num_experts_per_tok=2, head_dim=16,
                       sliding_window=64, **COMMON)
    return GptOssForCausalLM, cfg, 2, "router", 2


@pytest.mark.parametrize("build", [_qwen3, _olmoe, _granite, _gpt_oss], ids=["qwen3_moe", "olmoe", "granitemoe", "gpt_oss"])
def test_recorded_ids_are_the_routers_own(build):
    cls, cfg, k, router_attr, idx_pos = build()
    cfg._attn_implementation = "eager"
    torch.manual_seed(0)
    model = cls(cfg).eval()
    B, T, prefix = 5, 14, 8
    rec = record_eids.RouterRecorder(model, top_k=k, batch=B)
    assert len(rec.names) == cfg.num_hidden_layers, rec.names
    seen = [[] for _ in rec.names]
    routers = [m for n, m in model.named_modules() if n.split(".")[-1] == router_attr]
    assert len(routers) == cfg.num_hidden_layers

    def router_hook(li):
        def hook(_m, _i, out):
            ids = out[idx_pos].reshape(-1, k)
            if ids.shape[0] == B:
                seen[li].append(ids.to(torch.int16))
        return hook
    hs = [r.register_forward_hook(router_hook(i)) for i, r in enumerate(routers)]
    ids = torch.randint(0, cfg.vocab_size, (B, T), generator=torch.Generator().manual_seed(1))
    record_eids.run(model, ids, prefix, 4, "cpu")
    rec.remove()
    for h in hs:
        h.remove()
    e = rec.tensor()
    assert e.shape == (T - prefix, cfg.num_hidden_layers, B, k), e.shape
    assert rec.other == [prefix // 4] * cfg.num_hidden_layers, rec.other     # prefill chunks are not B-row calls
    for li in range(cfg.num_hidden_layers):
        assert torch.equal(e[:, li], torch.stack(seen[li])), f"layer {li}: recorded ids differ from the router's"


def test_a_model_without_experts_is_refused():
    with pytest.raises(RuntimeError, match="no MoE experts module"):
        record_eids.RouterRecorder(torch.nn.Linear(4, 4), top_k=2, batch=5)
