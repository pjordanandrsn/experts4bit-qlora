"""Layer-major accumulation beyond Qwen3 (bench/layer-major-families/DESIGN.md): Mixtral, OLMoE and Gemma-4's text model,
each against the stock accumulation loop on a tiny random model (fp32, both checkpoint kinds).

STRICT, per family:
- each micro-batch's loss is ``torch.equal`` to the stock forward's, which is the drift guard itself, run and passed;
- the gradients are the same weighted sum (fp32, so summation order only: the band test_layer_major.py uses for Qwen3);
- the new refusals: Mixtral's training-time ``router_jitter_noise``, Gemma-4's per-layer embeddings, and a family that is
  still unsupported.
Gemma-4's tiny model mixes sliding and full layers (``sliding_window`` below the sequence length) and has KV-shared reader
layers; a mutant that shares ONE ``shared_kv_states`` dict across micro-batches must fail the equality check, or that
check would pass vacuously.
"""
import pytest

torch = pytest.importorskip("torch")
tr = pytest.importorskip("transformers")

from experts4bit_qlora.engines import layer_major as lm  # noqa: E402
from experts4bit_qlora.engines.layer_major import LayerMajorDriftError, layer_major_step  # noqa: E402

VOCAB, M = 97, 3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
WEIGHTS = [1.0 / M] * M


def _mixtral():
    return tr.MixtralForCausalLM(tr.MixtralConfig(
        hidden_size=64, intermediate_size=128, num_hidden_layers=3, num_attention_heads=4, num_key_value_heads=2,
        num_local_experts=8, num_experts_per_tok=2, vocab_size=VOCAB, max_position_embeddings=64))


def _olmoe():
    return tr.OlmoeForCausalLM(tr.OlmoeConfig(
        hidden_size=64, intermediate_size=32, num_hidden_layers=3, num_attention_heads=4, num_key_value_heads=4,
        num_experts=8, num_experts_per_tok=2, vocab_size=VOCAB, max_position_embeddings=64))


def _gemma4(kv_shared=2, ple=0):
    from transformers import Gemma4TextConfig
    from transformers.models.gemma4.modeling_gemma4 import Gemma4ForCausalLM
    return Gemma4ForCausalLM(Gemma4TextConfig(
        vocab_size=VOCAB, hidden_size=128, intermediate_size=128, num_hidden_layers=6, num_attention_heads=4,
        num_key_value_heads=2, head_dim=32, global_head_dim=64, num_global_key_value_heads=1,
        layer_types=["sliding_attention", "full_attention"] * 3, sliding_window=4, num_kv_shared_layers=kv_shared,
        enable_moe_block=True, num_experts=4, top_k_experts=2, moe_intermediate_size=64, attention_k_eq_v=True,
        hidden_size_per_layer_input=ple, vocab_size_per_layer_input=VOCAB, max_position_embeddings=512))


FAMILIES = {"mixtral": _mixtral, "olmoe": _olmoe, "gemma4": _gemma4}


def _tiny(family, reentrant, seed=0, **kw):
    torch.manual_seed(seed)
    m = FAMILIES[family](**kw).to(torch.float32).to(DEVICE)
    m.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": reentrant})
    if reentrant:
        m.enable_input_require_grads()
    m.config.use_cache = False
    m.train()
    return m


def _batches(widths, seed=3):
    g = torch.Generator().manual_seed(seed)
    out = []
    for w in widths:
        ids = torch.randint(1, VOCAB, (2, w), generator=g)
        att = torch.ones(2, w, dtype=torch.long)
        att[1, w - 2:] = 0
        out.append({"input_ids": ids.to(DEVICE), "labels": ids.masked_fill(att == 0, -100).to(DEVICE),
                    "attention_mask": att.to(DEVICE)})
    return out


# Gemma-4's KV-shared layers read the keys an earlier layer stored for the same forward: equal widths, so a mutant that
# mixes micro-batches fails on values (a loss mismatch), not on a shape error.
WIDTHS = {"mixtral": (9, 14, 6), "olmoe": (9, 14, 6), "gemma4": (12, 12, 12)}


def _stock(model, batches):
    losses = []
    for b, w in zip(batches, WEIGHTS):
        out = model(**b)
        (out.loss * w).backward()
        losses.append(out.loss.detach())
    return losses


def _grads(model):
    return {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}


@pytest.mark.parametrize("reentrant", [False, True], ids=["nonreentrant", "reentrant"])
@pytest.mark.parametrize("family", sorted(FAMILIES))
def test_losses_equal_the_stock_forward_and_grads_are_the_same_sum(family, reentrant):
    model = _tiny(family, reentrant)
    ref_losses = _stock(model, _batches(WIDTHS[family]))
    ref = _grads(model)
    model.zero_grad(set_to_none=True)
    losses = layer_major_step(model, _batches(WIDTHS[family]), WEIGHTS, handles=[])     # the drift guard runs here
    assert model in lm._VERIFIED
    for got, want in zip(losses, ref_losses):
        assert torch.equal(got, want)
    got = _grads(model)
    assert got.keys() == ref.keys() and len(got) > 0
    for n in ref:
        torch.testing.assert_close(got[n], ref[n], rtol=1e-5, atol=1e-7, msg=n)
    assert len(model.model.layers) == model.config.num_hidden_layers


def test_the_gemma4_tiny_model_really_shares_kv_and_mixes_masks():
    model = _tiny("gemma4", reentrant=False)
    attn = [layer.self_attn for layer in model.model.layers]
    assert any(a.is_kv_shared_layer for a in attn) and any(a.store_full_length_kv for a in attn)
    assert set(model.config.layer_types) == {"sliding_attention", "full_attention"}
    assert model.config.sliding_window < min(WIDTHS["gemma4"])


def test_one_shared_kv_dict_across_micro_batches_fails_the_equality_check(monkeypatch):
    """The mutant: every micro-batch's layers read and write ONE dict, so a KV-shared layer reads another micro-batch's
    keys. The drift guard must catch it before any backward."""
    model = _tiny("gemma4", reentrant=False)
    shared = lm._shared_kv_states()
    monkeypatch.setattr(lm, "_shared_kv_states", lambda: shared)
    with pytest.raises(LayerMajorDriftError, match="lose torch.equal"):
        layer_major_step(model, _batches(WIDTHS["gemma4"]), WEIGHTS, handles=[])
    assert all(p.grad is None for p in model.parameters())                      # no backward ran
    assert model not in lm._VERIFIED


def test_the_new_refusals():
    m = _tiny("mixtral", reentrant=False)
    m.config.router_jitter_noise = 0.1
    with pytest.raises(NotImplementedError, match="router_jitter_noise"):
        layer_major_step(m, _batches(WIDTHS["mixtral"]), WEIGHTS, handles=[])
    g = _tiny("gemma4", reentrant=False, ple=16)
    with pytest.raises(NotImplementedError, match="per-layer embeddings"):
        layer_major_step(g, _batches(WIDTHS["gemma4"]), WEIGHTS, handles=[])
    cfg = tr.GraniteMoeConfig(hidden_size=32, intermediate_size=64, num_hidden_layers=1, num_attention_heads=4,
                              num_key_value_heads=2, num_local_experts=4, num_experts_per_tok=2, vocab_size=VOCAB)
    with pytest.raises(NotImplementedError, match="GraniteMoeForCausalLM is not supported"):
        layer_major_step(tr.GraniteMoeForCausalLM(cfg), _batches((5, 5, 5)), WEIGHTS, handles=[])
