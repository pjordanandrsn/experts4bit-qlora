"""Checkpoint tensors the text model does not build: skipped only when the modeling class
declares them, refused by name otherwise (e4b#529).

ERNIE-4.5-21B-A3B-PT's released index carries a multi-token-prediction block — tensors under
``model.mtp_block.0.*``, ``model.mtp_emb_norm.*``, ``model.mtp_hidden_norm.*`` and
``model.mtp_linear_proj.*`` — that ``Ernie4_5_MoeModel`` does not build. The loader walked to the
first of them and died with ``Ernie4_5_MoeModel has no attribute `mtp_block```, which named
neither the checkpoint key nor a remedy; the family's ARCHITECTURE_SUPPORT row read ``validated``
on a fixture that had no MTP weights, so nothing here could see it. transformers ignores those
keys through the class's own ``_keys_to_ignore_on_load_unexpected = ["mtp"]`` ("Not supporting
multi-token prediction (MTP) atm"), and the loader now honours that same declaration.

The fixture is the family's own tiny model plus tensors under the released prefixes. It proves the
mechanism; it does NOT prove the released checkpoint loads — that is the support probe's row.
"""
import json
import os

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("bitsandbytes")

from quant_guard import load_or_skip  # noqa: E402

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if torch.cuda.is_available() else torch.float32

#: The four prefixes are the released ERNIE-4.5-21B-A3B-PT index's (e4b#529 quotes them); the
#: expansion below is a stand-in of the same shape and count (12), not a copy of the index.
MTP_KEYS = (
    "model.mtp_block.0.input_layernorm.weight",
    "model.mtp_block.0.post_attention_layernorm.weight",
    "model.mtp_block.0.self_attn.q_proj.weight",
    "model.mtp_block.0.self_attn.k_proj.weight",
    "model.mtp_block.0.self_attn.v_proj.weight",
    "model.mtp_block.0.self_attn.o_proj.weight",
    "model.mtp_block.0.mlp.gate_proj.weight",
    "model.mtp_block.0.mlp.up_proj.weight",
    "model.mtp_block.0.mlp.down_proj.weight",
    "model.mtp_emb_norm.0.weight",
    "model.mtp_hidden_norm.0.weight",
    "model.mtp_linear_proj.0.weight",
)


def _ernie():
    pytest.importorskip("transformers.models.ernie4_5_moe", reason="this transformers has no ernie4_5_moe")
    from transformers.models.ernie4_5_moe.configuration_ernie4_5_moe import Ernie4_5_MoeConfig
    from transformers.models.ernie4_5_moe.modeling_ernie4_5_moe import Ernie4_5_MoeForCausalLM

    return Ernie4_5_MoeForCausalLM(
        Ernie4_5_MoeConfig(
            hidden_size=64,
            intermediate_size=128,
            moe_intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=16,
            vocab_size=128,
            moe_num_experts=8,
            moe_k=2,
            moe_num_shared_experts=1,
            moe_layer_start_index=0,
        )
    )


def _write_ckpt(model, d, extra):
    """The family's on-disk layout (per-expert Linears) plus ``extra`` tensors the model does not build."""
    from safetensors.torch import save_file

    new = {}
    for k, v in model.state_dict().items():
        if k.endswith("experts.gate_up_proj"):
            base = k[: -len("gate_up_proj")]
            for e in range(v.shape[0]):
                g, u = v[e].chunk(2, dim=0)
                new[f"{base}{e}.gate_proj.weight"] = g.contiguous()
                new[f"{base}{e}.up_proj.weight"] = u.contiguous()
        elif k.endswith("experts.down_proj"):
            base = k[: -len("down_proj")]
            for e in range(v.shape[0]):
                new[f"{base}{e}.down_proj.weight"] = v[e].contiguous()
        else:
            new[k] = v
    new = {k: v.to(DTYPE).contiguous().clone() for k, v in new.items()}
    for k, v in extra.items():
        assert k not in new, k
        new[k] = v.to(DTYPE).contiguous()
    save_file(new, os.path.join(d, "model.safetensors"))
    json.dump({"weight_map": {k: "model.safetensors" for k in new}},
              open(os.path.join(d, "model.safetensors.index.json"), "w"))
    model.config.save_pretrained(d)


def test_declared_unbuilt_tensors_are_skipped_counted_and_the_model_runs(tmp_path, capsys):
    torch.manual_seed(0)
    _write_ckpt(_ernie(), str(tmp_path), {k: torch.randn(8, 8) for k in MTP_KEYS})
    model, cfg = load_or_skip(str(tmp_path), DEVICE, DTYPE, r=4, alpha=8)
    out = capsys.readouterr().out
    assert f"skipped {len(MTP_KEYS)} checkpoint tensor(s) the text model does not build" in out, out
    assert "_keys_to_ignore_on_load_unexpected" in out
    # Nothing declared-and-skipped leaked into the tree, and everything the model builds arrived.
    assert not [n for n, t in list(model.named_parameters()) + list(model.named_buffers()) if t.is_meta]
    assert not hasattr(model.model, "mtp_block")
    model.config.use_cache = False
    logits = model(input_ids=torch.randint(0, cfg.vocab_size, (1, 8), device=DEVICE)).logits
    assert tuple(logits.shape) == (1, 8, cfg.vocab_size)
    assert torch.isfinite(logits).all()


def test_an_undeclared_unplaceable_tensor_is_refused_by_name(tmp_path):
    """The false-accept probe: the skip is gated on the class's declaration, not on
    "anything with no module" — a tensor no pattern covers must stop the load and say which."""
    from experts4bit_qlora.loader import UnplaceableTensorError

    torch.manual_seed(0)
    _write_ckpt(_ernie(), str(tmp_path), {"model.speculative_head.weight": torch.randn(8, 8)})
    with pytest.raises(UnplaceableTensorError) as ei:
        load_or_skip(str(tmp_path), DEVICE, DTYPE, r=4, alpha=8)
    msg = str(ei.value)
    assert "model.speculative_head.weight" in msg
    assert "_keys_to_ignore_on_load_unexpected" in msg and "CKPT_KEY_REWRITERS" in msg
    assert isinstance(ei.value, AttributeError)  # what callers used to catch still catches it


def test_patterns_come_from_the_modeling_class():
    """ERNIE declares `mtp`; Qwen3-MoE declares nothing, so a stray key there is refused."""
    from experts4bit_qlora.loader import _unbuilt_key_patterns

    pats = _unbuilt_key_patterns(_ernie())
    assert pats and all(any(p.search(k) for p in pats) for k in MTP_KEYS)
    assert not any(p.search("model.layers.0.mlp.gate.weight") for p in pats)

    from transformers.models.qwen3_moe.configuration_qwen3_moe import Qwen3MoeConfig
    from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeForCausalLM
    qwen = Qwen3MoeForCausalLM(Qwen3MoeConfig(
        hidden_size=32, intermediate_size=64, moe_intermediate_size=32, num_hidden_layers=1,
        num_attention_heads=2, num_key_value_heads=1, num_experts=4, num_experts_per_tok=2,
        vocab_size=64, decoder_sparse_step=1, mlp_only_layers=[], head_dim=16))
    assert _unbuilt_key_patterns(qwen) == []


def test_assign_refuses_by_name():
    from experts4bit_qlora.loader import UnplaceableTensorError, _assign

    m = torch.nn.Sequential(torch.nn.Linear(2, 2))
    with pytest.raises(UnplaceableTensorError, match=r"model\.mtp_block\.0\.foo\.weight"):
        _assign(m, "model.mtp_block.0.foo.weight", torch.zeros(2))
    _assign(m, "0.weight", torch.ones(2, 2))  # a real destination still places
    assert torch.equal(m[0].weight.detach(), torch.ones(2, 2))
