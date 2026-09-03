# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""The upstream control shares the oracle's scorer and window and is
labelled as its own arm, so a verdict can never mistake it for the
e4b-loaded oracle."""
import importlib.util
import json
import os
import sys
import types

import pytest
import torch

transformers = pytest.importorskip("transformers")

_HERE = os.path.dirname(__file__)


def _step_decomp():
    spec = importlib.util.spec_from_file_location(
        "step_decomp", os.path.join(_HERE, "..", "bench", "hybrid-g9", "step_decomp.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules["step_decomp"] = mod
    spec.loader.exec_module(mod)
    return mod


def _tiny_llama():
    from transformers import LlamaConfig, LlamaForCausalLM
    cfg = LlamaConfig(vocab_size=64, hidden_size=32, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2,
                      intermediate_size=64, max_position_embeddings=128)
    torch.manual_seed(0)
    return LlamaForCausalLM(cfg).eval()


def test_upstream_arm_is_labelled_and_runs_eager(tmp_path):
    sd = _step_decomp()
    model = _tiny_llama()
    model.config._attn_implementation = "sdpa"
    ids = torch.randint(0, 64, (40,))
    a = types.SimpleNamespace(ppl_oracle="upstream", ppl_steps=16, prompt_len=8,
                              prompt_offset=0, ppl_source="wikitext",
                              out=str(tmp_path / "up" / "o.json"))
    sd._ppl_oracle_main(a, model, ids, "f" * 64)
    rec = json.load(open(a.out))
    assert rec["attn_path"] == "upstream-eager-oracle"
    assert rec["tokens_scored"] == 16
    assert model.config._attn_implementation == "eager"
    # the same scorer as the e4b-loaded oracle: identical nll on identical input
    a.ppl_oracle = "eager"
    a.out = str(tmp_path / "eager.json")
    sd._ppl_oracle_main(a, model, ids, "f" * 64)
    rec2 = json.load(open(a.out))
    assert rec2["attn_path"] == "eager-oracle"
    assert abs(rec2["mean_nll"] - rec["mean_nll"]) < 1e-6


def test_k8_window_is_a_function_of_args_and_tokenizer():
    """The window helper returns the corpus, the row stride, the prompts,
    the scored ids and the digest -- what every arm consumes."""
    sd = _step_decomp()
    assert callable(sd._k8_window) and callable(sd._upstream_oracle_main)
    import inspect
    assert list(inspect.signature(sd._k8_window).parameters) == ["a", "tok"]


class _StubTok:
    """A tokenizer with a chat template that prefixes a fixed header."""
    def apply_chat_template(self, msgs, add_generation_prompt=True, tokenize=True):
        assert msgs[0]["role"] == "user" and add_generation_prompt and tokenize
        return [1, 2, 3]

    def __call__(self, text, add_special_tokens=True, return_tensors=None):
        assert add_special_tokens is False
        return {"input_ids": [9] * len(text.split("|"))}


def test_chat_prefix_ids_template_then_suffix():
    sd = _step_decomp()
    pre = sd._chat_prefix_ids(_StubTok())
    assert pre.tolist() == [1, 2, 3]
    pre = sd._chat_prefix_ids(_StubTok(), "<|channel|>final<|message|>")
    assert pre.tolist()[:3] == [1, 2, 3] and len(pre) > 3
    assert pre.dtype == torch.long
