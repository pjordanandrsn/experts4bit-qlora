# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""The paged attention's UNBOUND fallback (a registered model run with no paged context bound, e.g. an instrument's
transformers reference) against transformers' own attention, on a tiny Gemma-4 whose sliding window binds.

transformers builds no attention mask for an implementation it has no mask function for, and ``e4b_paged`` registers
none (one would run HF's mask preprocessing inside the bound, graph-captured forwards). Before the fix the fallback
therefore ran SDPA with no mask at all. It dropped Gemma-4's sliding window: the last logits of a 40-token prompt moved
by 0.85 at window 16. In a cached chunked prefill it also took torch's top-left ``is_causal`` alignment. The bound
paged path builds its own masks and is untouched.
"""
import pytest
import torch

pytest.importorskip("transformers.models.gemma4", reason="needs transformers with Gemma-4")

P, WINDOW, CHUNK = 40, 16, 16


def _tiny(seed=0, window=WINDOW):
    from transformers import Gemma4TextConfig
    from transformers.models.gemma4.modeling_gemma4 import Gemma4ForCausalLM
    cfg = Gemma4TextConfig(vocab_size=256, hidden_size=128, intermediate_size=128, num_hidden_layers=6,
                           num_attention_heads=4, num_key_value_heads=2, head_dim=32, global_head_dim=64,
                           num_global_key_value_heads=1, layer_types=["sliding_attention"] * 5 + ["full_attention"],
                           sliding_window=window, enable_moe_block=True, num_experts=4, top_k_experts=2,
                           moe_intermediate_size=64, attention_k_eq_v=True, hidden_size_per_layer_input=0,
                           vocab_size_per_layer_input=256, max_position_embeddings=512)
    torch.manual_seed(seed)
    return Gemma4ForCausalLM(cfg).eval()


def _ids():
    return torch.randint(0, 256, (1, P), generator=torch.Generator().manual_seed(1))


def _reference(ids):
    """transformers' own attention (its default implementation, with its own masks), one forward."""
    with torch.no_grad():
        return _tiny()(input_ids=ids).logits[0].float()


def test_the_window_binds_in_this_fixture():
    """The same weights with a window that never binds: the last logits move by far more than the tolerance below, so
    a fallback that dropped the window could not pass."""
    ids = _ids()
    with torch.no_grad():
        unbound_window = _tiny(window=4096)(input_ids=ids).logits[0, -1].float()
    assert float((_reference(ids)[-1] - unbound_window).abs().max()) > 0.1


def test_the_unbound_fallback_keeps_the_sliding_window():
    from experts4bit_qlora.engines import paged_attention
    ids = _ids()
    want = _reference(ids)
    m = _tiny()
    paged_attention.register(m)
    assert paged_attention.current_context() is None
    with torch.no_grad():
        got = m(input_ids=ids).logits[0].float()
    torch.testing.assert_close(got, want, atol=1e-4, rtol=1e-4)


def test_the_unbound_fallback_aligns_a_cached_chunk_to_its_last_key():
    """Chunked prefill through transformers' DynamicCache, then cached decode steps: every chunk's queries sit after
    the cache, so the causal mask must align to the last key (full layer) and keep the window (sliding layers)."""
    from transformers.cache_utils import DynamicCache

    from experts4bit_qlora.engines import paged_attention
    ids = _ids()
    want = _reference(ids)
    m = _tiny()
    paged_attention.register(m)
    cache = DynamicCache(config=m.config)
    rows = []
    with torch.no_grad():
        for start in range(0, P - 4, CHUNK):
            stop = min(start + CHUNK, P - 4)
            rows.append(m(input_ids=ids[:, start:stop], past_key_values=cache, use_cache=True).logits[0].float())
        for t in range(P - 4, P):
            rows.append(m(input_ids=ids[:, t:t + 1], past_key_values=cache, use_cache=True).logits[0].float())
    torch.testing.assert_close(torch.cat(rows), want, atol=1e-4, rtol=1e-4)
