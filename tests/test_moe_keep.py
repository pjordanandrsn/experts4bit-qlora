"""Two value-identical memory trims for the fused MoE training step.

``_ScatterCombine``: the training forward's combine as one autograd node saving the bf16 ``down`` instead of its fp32 copy -- forward
and both gradients ``torch.equal`` to ``_scatter_combine`` under autograd.

``keep_moe_activations`` (E4B_MOE_KEEP_LAYERS): attention-only checkpointing in the last n decoder layers -- every trainable gradient
``torch.equal`` to whole-layer checkpointing on a tiny Qwen3-MoE, the right layers changed, and an exact unwind.
"""
import pytest
import torch

from experts4bit_qlora.engines.fast import _scatter_combine, _ScatterCombine
from experts4bit_qlora.engines import moe_keep

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


@pytest.mark.parametrize("dev", DEVICES)
@pytest.mark.parametrize("tokens,k,hidden", [(1, 8, 64), (7, 2, 32), (190, 8, 128), (33, 4, 96)])
@pytest.mark.parametrize("down_dtype,out_dtype", [(torch.bfloat16, torch.bfloat16), (torch.float32, torch.float32), (torch.bfloat16, torch.float32)])
def test_scatter_combine_node_is_bit_identical(dev, tokens, k, hidden, down_dtype, out_dtype):
    g = torch.Generator().manual_seed(tokens * 7 + k)
    order = torch.randperm(tokens * k, generator=g).to(dev)
    down0 = torch.randn(tokens * k, hidden, generator=g).to(down_dtype).to(dev)
    w0 = torch.rand(tokens * k, generator=g).to(dev)
    gout = torch.randn(tokens, hidden, generator=g).to(out_dtype).to(dev)
    d1, w1 = down0.clone().requires_grad_(True), w0.clone().requires_grad_(True)
    a = _scatter_combine(d1, w1, order, None, tokens, k, hidden, dev, out_dtype)
    (a.float() * gout.float()).sum().backward()
    d2, w2 = down0.clone().requires_grad_(True), w0.clone().requires_grad_(True)
    b = _ScatterCombine.apply(d2, w2, order, tokens, k, out_dtype)
    (b.float() * gout.float()).sum().backward()
    assert torch.equal(a, b)
    assert torch.equal(d1.grad, d2.grad) and torch.equal(w1.grad, w2.grad)


def test_env_parsing(monkeypatch):
    for v, want in (("", None), ("0", None), ("off", None), ("all", "all"), ("ALL", "all"), ("12", 12)):
        monkeypatch.setenv("E4B_MOE_KEEP_LAYERS", v)
        assert moe_keep.moe_keep_layers_requested() == want, v
    monkeypatch.setenv("E4B_MOE_KEEP_LAYERS", "-1")
    with pytest.raises(ValueError):
        moe_keep.moe_keep_layers_requested()


def _tiny(seed=0):
    tr = pytest.importorskip("transformers")
    cfg = tr.Qwen3MoeConfig(hidden_size=64, intermediate_size=128, moe_intermediate_size=32, num_experts=8, num_experts_per_tok=2,
                            num_hidden_layers=3, num_attention_heads=4, num_key_value_heads=2, head_dim=16, vocab_size=97,
                            max_position_embeddings=64, decoder_sparse_step=1, norm_topk_prob=True)
    torch.manual_seed(seed)
    m = tr.Qwen3MoeForCausalLM(cfg).float()
    m.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    m.config.use_cache = False
    m.train()
    return m


def _grads(m):
    ids = torch.randint(0, 97, (2, 24), generator=torch.Generator().manual_seed(3))
    m.zero_grad(set_to_none=True)
    m(input_ids=ids, labels=ids).loss.backward()
    return {n: p.grad.clone() for n, p in m.named_parameters() if p.grad is not None}


def test_keep_moe_activations_changes_the_last_n_layers_and_gradients_stay_equal():
    ref = _grads(_tiny())
    m = _tiny()
    assert moe_keep.keep_moe_activations(m, 2) == 2
    layers = moe_keep._decoder_layers(m)
    assert [lay.gradient_checkpointing for lay in layers] == [True, False, False]
    assert [hasattr(lay, "_e4b_keep_refs") for lay in layers] == [False, True, True]
    assert moe_keep.keep_moe_activations(m, "all") == 1                 # the remaining layer; already-kept layers are not wrapped twice
    got = _grads(m)
    assert ref.keys() == got.keys()
    for n in ref:
        assert torch.equal(ref[n], got[n]), n
    assert moe_keep.release_moe_activations(m) == 3
    assert all(lay.gradient_checkpointing and not hasattr(lay, "_e4b_keep_refs") for lay in moe_keep._decoder_layers(m))
    back = _grads(m)
    for n in ref:
        assert torch.equal(ref[n], back[n]), n


def test_nothing_to_keep_without_checkpointing():
    m = _tiny()
    m.gradient_checkpointing_disable()
    assert moe_keep.keep_moe_activations(m, "all") == 0


def _tiny_hybrid(family, seed=0):
    """A tiny hybrid whose MoE block is NOT named ``mlp`` and whose other layers are not attention: LFM2-MoE (``feed_forward``,
    short-conv + attention layers, dense leading layers) and Qwen3.5-MoE (Gated DeltaNet + attention, ``mlp`` with a shared
    expert)."""
    tr = pytest.importorskip("transformers")
    torch.manual_seed(seed)
    if family == "lfm2_moe":
        cfg = tr.Lfm2MoeConfig(vocab_size=97, hidden_size=64, intermediate_size=128, moe_intermediate_size=32, num_hidden_layers=4,
                               num_attention_heads=4, num_key_value_heads=2, num_experts=8, num_experts_per_tok=2,
                               num_dense_layers=1, layer_types=["conv", "full_attention", "conv", "full_attention"],
                               max_position_embeddings=64)
        m = tr.Lfm2MoeForCausalLM(cfg).float()
    else:
        cfg = tr.Qwen3_5MoeTextConfig(vocab_size=97, hidden_size=64, moe_intermediate_size=32, shared_expert_intermediate_size=32,
                                      num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2, head_dim=16,
                                      num_experts=8, num_experts_per_tok=2, linear_num_value_heads=4, linear_num_key_heads=2,
                                      linear_key_head_dim=16, linear_value_head_dim=16,
                                      layer_types=["linear_attention", "linear_attention", "linear_attention", "full_attention"],
                                      max_position_embeddings=64)
        m = tr.Qwen3_5MoeForCausalLM(cfg).float()
    m.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    m.config.use_cache = False
    m.train()
    return m


@pytest.mark.parametrize("family,want", [("lfm2_moe", 3), ("qwen3_5_moe", 4)])
def test_keep_is_structural_across_hybrid_families(family, want):
    """Every MoE-bearing layer is kept whatever its children are called; dense layers are never touched; the non-MoE weighted
    children (attention, short-conv, Gated DeltaNet) stay checkpointed on their own; gradients stay exactly equal."""
    try:
        ref = _grads(_tiny_hybrid(family))
    except Exception as e:                                   # a fast-path kernel package this tiny CPU model cannot use
        pytest.skip(f"{family} tiny model does not run here: {type(e).__name__}: {e}")
    m = _tiny_hybrid(family)
    assert moe_keep.keep_moe_activations(m, "all") == want
    for lay in moe_keep._decoder_layers(m):
        wrapped = [c for c, _o in lay._e4b_keep_refs]
        assert wrapped and all(not moe_keep._holds_experts(c) for c in wrapped)
    got = _grads(m)
    assert ref.keys() == got.keys()
    for n in ref:
        assert torch.equal(ref[n], got[n]), n
    assert moe_keep.release_moe_activations(m) == want
