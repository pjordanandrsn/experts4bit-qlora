"""The structural matchers that let a new MoE family inherit the Qwen training stack (moe-generalize).

* Attention discovery admits LFM2's ``out_proj`` output projection beside ``q_proj``/``k_proj`` -- and NOT a short-conv /
  Mamba / Gated-DeltaNet ``out_proj`` beside an ``in_proj``, nor a biased CLIP-style tower.
* Trainables and the router are found by owning module and shape, never by parameter name: a router called ``gate``,
  ``router.layer`` or anything else beside an ``ExpertsLoRA`` is found; a shared expert's ``[1, H]`` gate is not.
* ``enable_fast_train``'s storage gate skips a positively non-NF4 / non-64 base and counts it, and ignores attributes a base
  does not carry.
"""
import types

import pytest
import torch
from torch import nn

from experts4bit_qlora.engines.fast import _train_storage_reason
from experts4bit_qlora.lora import detect_attention_projections


class _Attn(nn.Module):
    def __init__(self, H, out_name, bias=False):
        super().__init__()
        self.q_proj = nn.Linear(H, H, bias=bias)
        self.k_proj = nn.Linear(H, H // 2, bias=bias)
        self.v_proj = nn.Linear(H, H // 2, bias=bias)
        setattr(self, out_name, nn.Linear(H, H, bias=bias))


class _Conv(nn.Module):                      # LFM2 short conv / Mamba / GDN: in_proj + out_proj, no q/k
    def __init__(self, H):
        super().__init__()
        self.in_proj = nn.Linear(H, 3 * H, bias=False)
        self.out_proj = nn.Linear(H, H, bias=False)


def test_out_proj_attention_is_detected_and_mixers_are_not():
    H = 64
    m = nn.ModuleDict({"a0": _Attn(H, "o_proj"), "a1": _Attn(H, "out_proj"), "conv": _Conv(H),
                       "clip": _Attn(H, "out_proj", bias=True)})
    found = detect_attention_projections(m, exact_linear=True).candidates
    got = sorted((name, id(mod)) for mod, name in found)
    want = sorted([(n, id(m["a0"])) for n in ("q_proj", "k_proj", "v_proj", "o_proj")]
                  + [(n, id(m["a1"])) for n in ("q_proj", "k_proj", "v_proj", "out_proj")])
    assert got == want


def test_real_lfm2_tree_census_counts_attention_layers_only():
    tr = pytest.importorskip("transformers")
    cfg = tr.Lfm2MoeConfig(vocab_size=97, hidden_size=64, intermediate_size=128, moe_intermediate_size=32, num_hidden_layers=4,
                           num_attention_heads=4, num_key_value_heads=2, num_experts=8, num_experts_per_tok=2,
                           num_dense_layers=1, layer_types=["conv", "full_attention", "conv", "full_attention"],
                           max_position_embeddings=64)
    from hybrid_reference import reference_modeling
    m = reference_modeling("lfm2_moe").Lfm2MoeForCausalLM(cfg)
    assert detect_attention_projections(m, exact_linear=True).expected_count == 2 * 4


def _base(**kw):
    b = types.SimpleNamespace(quant_type="nf4", bits=4, blocksize=64, _gate_up_shape=(256, 128), _down_shape=(128, 128))
    for k, v in kw.items():
        setattr(b, k, v)
    return b


@pytest.mark.parametrize("kw,want", [
    ({}, None),
    ({"quant_type": "fp4"}, "not nf4"),
    ({"bits": 8}, "8-bit"),
    ({"blocksize": 128}, "blocksize 128"),
    ({"_down_shape": (128, 96)}, "K not divisible"),
])
def test_storage_gate(kw, want):
    r = _train_storage_reason(_base(**kw))
    assert (r is None) if want is None else (want in r), r


def test_storage_gate_ignores_absent_attributes():
    assert _train_storage_reason(types.SimpleNamespace()) is None


def _experts_lora(E=8, H=64, inter=64):
    from experts4bit_qlora import Experts4bit, ExpertsLoRA
    from quant_guard import require_quantize
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    require_quantize(dev)
    base = Experts4bit.from_float((torch.randn(E, 2 * inter, H) * 0.1).to(dev), (torch.randn(E, H, inter) * 0.1).to(dev),
                                  quant_type="nf4", compute_dtype=torch.bfloat16)
    return ExpertsLoRA(base, r=4, alpha=8, dtype=torch.float32)


@pytest.mark.parametrize("router_path", ["gate", "router.layer", "router.proj"])
def test_router_and_trainables_by_structure(router_path):
    from experts4bit_qlora.lora import LoRALinear, router_param_ids, trainable_lora_param_ids
    E, H = 8, 64
    ex = _experts_lora(E, H)
    block = nn.Module()
    block.experts = ex
    head, _, leaf = router_path.rpartition(".")
    holder = block
    if head:
        sub = nn.Module()
        setattr(block, head, sub)
        holder = sub
    router = nn.Linear(H, E, bias=False)
    setattr(holder, leaf, router)
    block.shared_expert_gate = nn.Linear(H, 1, bias=False)          # [1, H]: never a router
    attn = LoRALinear(nn.Linear(H, H, bias=False), r=4, alpha=8, dtype=torch.float32)
    model = nn.ModuleDict({"mixer": block, "attn": attn})
    assert router_param_ids(model) == {id(router.weight)}
    exp_ids, att_ids = trainable_lora_param_ids(model)
    assert exp_ids == {id(p) for n, p in ex.named_parameters(recurse=False) if "lora" in n}
    assert att_ids == {id(attn.lora_A), id(attn.lora_B)}


def test_router_search_skips_attention_and_refuses_an_ambiguous_block():
    """Gemma-4's experts sit directly in the decoder layer, beside the attention: a ``k_proj`` whose rows happen to equal
    ``num_experts`` must not be taken for the router. Two real ``[E, H]`` candidates in one block are refused, not guessed."""
    from experts4bit_qlora import lora
    E, H = 8, 64
    layer = nn.Module()
    layer.experts = _experts_lora(E, H)
    layer.self_attn = nn.Module()
    layer.self_attn.q_proj = nn.Linear(H, 2 * H, bias=False)
    layer.self_attn.k_proj = nn.Linear(H, E, bias=False)               # [E, H] by coincidence: attention, never a router
    layer.self_attn.v_proj = nn.Linear(H, E, bias=False)
    layer.self_attn.o_proj = nn.Linear(2 * H, H, bias=False)
    layer.router = nn.Module()
    layer.router.proj = nn.Linear(H, E, bias=False)
    model = nn.ModuleDict({"layer": layer})
    assert lora.router_param_ids(model) == {id(layer.router.proj.weight)} and lora.ROUTER_AMBIGUOUS == []

    layer.router.extra = nn.Linear(H, E, bias=False)                  # a second [E, H] candidate: ambiguous
    assert lora.router_param_ids(model) == set()
    assert lora.ROUTER_AMBIGUOUS == ["layer"]


def test_recurrent_kernel_fallbacks_reads_the_wrappers_own_decision(monkeypatch):
    """transformers' kernel wrappers decide at import and swallow the ImportError; the census reads their closure cell."""
    import sys
    from experts4bit_qlora.engines.fast import recurrent_kernel_fallbacks

    def make(is_new):
        is_new_implementation = is_new

        def wrapped(x):
            return x if is_new_implementation else x
        return wrapped

    fake = types.ModuleType("transformers.models.fakehyb.modeling_fakehyb")
    fake.chunk_scan, fake.conv_fn, fake.plain = make(False), make(True), (lambda x: x)

    class Block(nn.Module):
        pass
    Block.__module__ = fake.__name__
    monkeypatch.setitem(sys.modules, fake.__name__, fake)
    assert recurrent_kernel_fallbacks(nn.ModuleDict({"b": Block()})) == ["modeling_fakehyb.chunk_scan"]
