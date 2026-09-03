# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""The chunked, cache-driven oracle scores exactly what one full forward
scores: for every scored position the prediction sees exactly the tokens
before it. A tiny Llama on the CPU is enough to pin the arithmetic."""
import importlib.util
import os
import sys

import pytest
import torch

transformers = pytest.importorskip("transformers")

_HERE = os.path.dirname(__file__)
_spec = importlib.util.spec_from_file_location(
    "step_decomp", os.path.join(_HERE, "..", "bench", "hybrid-g9", "step_decomp.py"))


def _load_harness():
    mod = importlib.util.module_from_spec(_spec)
    sys.modules["step_decomp"] = mod
    _spec.loader.exec_module(mod)
    return mod


def _tiny_llama():
    from transformers import LlamaConfig, LlamaForCausalLM
    cfg = LlamaConfig(vocab_size=97, hidden_size=32, intermediate_size=64,
                      num_hidden_layers=2, num_attention_heads=4,
                      num_key_value_heads=2, max_position_embeddings=256)
    cfg._attn_implementation = "eager"
    torch.manual_seed(0)
    return LlamaForCausalLM(cfg).eval()


@pytest.mark.parametrize("chunk", [1, 7, 64])
def test_chunked_oracle_equals_full_forward(chunk):
    h = _load_harness()
    model = _tiny_llama()
    torch.manual_seed(1)
    ids = torch.randint(0, 97, (120,))
    prompt_len, steps = 16, 40
    got = h.ppl_oracle_score(model, ids, prompt_len, steps, chunk=chunk, device="cpu")
    with torch.no_grad():
        lg = torch.log_softmax(model(input_ids=ids[None, :prompt_len + steps + 1]).logits[0].float(), -1)
    # the paged instrument scores ids[prompt_len+1 .. prompt_len+steps]
    # from the logits at positions prompt_len .. prompt_len+steps-1
    rows = torch.arange(prompt_len, prompt_len + steps)
    ref = -lg[rows, ids[rows + 1]].mean().item()
    assert abs(got - ref) < 1e-4, (got, ref)


def test_full_forward_scorer_matches_the_chunked_one_on_a_dense_model():
    """The positive control for `--ppl-oracle full`: with no MoE router,
    chunking is a pure reordering and the two scorers must agree to float
    noise. (They DISagree on a mixture-of-experts model, where the
    reordering flips router top-k choices — that is the floor
    METHODOLOGY 13.1 measures, not a bug in either scorer.)"""
    h = _load_harness()
    model = _tiny_llama()
    torch.manual_seed(1)
    ids = torch.randint(0, 97, (120,))
    prompt_len, steps = 16, 40
    chunked = h.ppl_oracle_score(model, ids, prompt_len, steps, chunk=7,
                                 device="cpu")
    full = h.ppl_oracle_score_full(model, ids, prompt_len, steps, device="cpu")
    assert abs(chunked - full) < 1e-4, (chunked, full)


def test_full_forward_scorer_scores_the_documented_window():
    """It must score ids[prompt_len+1 .. prompt_len+steps] from the logits
    at positions prompt_len .. prompt_len+steps-1 — the same window the
    paged instrument scores. An off-by-one would silently compare
    different text between arms."""
    h = _load_harness()
    model = _tiny_llama()
    torch.manual_seed(3)
    ids = torch.randint(0, 97, (120,))
    prompt_len, steps = 16, 40
    got = h.ppl_oracle_score_full(model, ids, prompt_len, steps, device="cpu")
    with torch.no_grad():
        lg = torch.log_softmax(
            model(input_ids=ids[None, :prompt_len + steps + 1]).logits[0].float(), -1)
    rows = torch.arange(prompt_len, prompt_len + steps)
    ref = -lg[rows, ids[rows + 1]].mean().item()
    assert abs(got - ref) < 1e-4, (got, ref)
