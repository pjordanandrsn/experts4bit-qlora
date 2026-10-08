"""Tied synthetic shards must survive strict real-model loading, without untied substitution."""
from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1] / "bench" / "dq10"
SPEC = importlib.util.spec_from_file_location("dq10_subject", ROOT / "dq10_subject.py")
subject = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(subject)


def test_public_configs_are_unchanged_and_new_families():
    for name, digest in subject.CONFIG_HASHES.items():
        raw = (ROOT / "configs" / (name + ".json")).read_bytes()
        assert hashlib.sha256(raw).hexdigest() == digest
    assert subject.config_for("mistral7b_v03").model_type == "mistral"
    cfg = subject.config_for("smollm3_3b")
    assert cfg.model_type == "smollm3" and cfg.vocab_size == 128256 and cfg.tie_word_embeddings


def test_tied_alias_seed_and_strict_model_load(tmp_path):
    import torch
    from safetensors.torch import load_file
    from transformers import AutoModelForCausalLM, SmolLM3Config

    cfg = SmolLM3Config(hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                       num_attention_heads=8, num_key_value_heads=4, vocab_size=128,
                       tie_word_embeddings=True, no_rope_layers=[1, 1],
                       layer_types=["full_attention"] * 2, bos_token_id=1, eos_token_id=1, pad_token_id=2)
    manifest = subject.write_tensors(cfg, "alias-check", tmp_path)
    assert manifest["tied_aliases"] == {"lm_head.weight": "model.embed_tokens.weight"}
    weights = {key: value for file in tmp_path.glob("*.safetensors") for key, value in load_file(file).items()}
    assert torch.equal(weights["lm_head.weight"], weights["model.embed_tokens.weight"])
    aliases = [row for row in manifest["files"].values() if row["tensor"] in
               {"lm_head.weight", "model.embed_tokens.weight"}]
    assert len(aliases) == 2 and len({row["seed"] for row in aliases}) == 1
    assert all(row["seed_source"] == "model.embed_tokens.weight" for row in aliases)
    model = AutoModelForCausalLM.from_config(cfg, dtype=torch.bfloat16, attn_implementation="sdpa")
    model.load_state_dict(weights, strict=True)
    assert model.get_input_embeddings().weight is model.get_output_embeddings().weight
    assert torch.equal(model.get_input_embeddings().weight, weights["model.embed_tokens.weight"])
    ids = torch.arange(16).unsqueeze(0) + 3
    assert torch.isfinite(model(input_ids=ids, labels=ids, use_cache=False).loss)


def test_changed_or_unknown_config_refused(tmp_path):
    (tmp_path / "smollm3_3b.json").write_text("{}")
    with pytest.raises(ValueError, match="checksum"):
        subject.config_for("smollm3_3b", tmp_path)
    with pytest.raises(ValueError, match="unregistered"):
        subject.config_for("unregistered", tmp_path)
