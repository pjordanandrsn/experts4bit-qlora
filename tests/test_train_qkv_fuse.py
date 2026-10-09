# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""P129: E4B_TRAIN_FUSE_QKV=1, one fused q/k/v projection for training attention.

- Off by default: with the knob unset ``enable_fast_train``'s hook does nothing.
- Refusals keep today's path, with the reason recorded.
- A released projection called on its own raises a clear ``RuntimeError`` naming the fusion.
- On CUDA with bitsandbytes NF4 bases (the last two with fp32 and with bf16 adapters, the dtypes training runs):
  - the fused dequantize is bit for bit the three dequantized weights stacked;
  - the fused projection is within TC1's rounding bar of the three ``LoRALinear`` modules on outputs, the input gradient and every
    adapter gradient (``2**-6`` of the largest entry for bf16, ``2**-16`` for fp32);
  - the patched attention forward is within the reorder bar of the unfused one;
  - the adapters stay the parameters.
"""
from __future__ import annotations

import copy

import pytest

torch = pytest.importorskip("torch")
qmod = pytest.importorskip("transformers.models.qwen3_moe.modeling_qwen3_moe")

from torch import nn  # noqa: E402

from experts4bit_qlora.engines import fast  # noqa: E402
from experts4bit_qlora.engines import train_qkv_fuse as tq  # noqa: E402
from experts4bit_qlora.lora import LoRALinear  # noqa: E402

CUDA = torch.cuda.is_available()


def _cfg(hidden=128):
    cfg = qmod.Qwen3MoeConfig(hidden_size=hidden, num_attention_heads=2, num_key_value_heads=1, head_dim=64,
                              intermediate_size=256, moe_intermediate_size=64, num_experts=4, num_experts_per_tok=2,
                              num_hidden_layers=1, attention_bias=False, rms_norm_eps=1e-6)
    cfg._attn_implementation = "eager"
    return cfg


class _Holder(nn.Module):
    def __init__(self, attn):
        super().__init__()
        self.attn = attn


def _lora_wrap(attn, r=4, dtype=torch.float32, seed=5):
    g = torch.Generator().manual_seed(seed)
    for n in ("q_proj", "k_proj", "v_proj", "o_proj"):
        m = LoRALinear(getattr(attn, n), r, r, dtype)
        m.lora_B.data.copy_((torch.randn(m.lora_B.shape, generator=g) * 0.02).to(m.lora_B.device))
        setattr(attn, n, m)
    return attn


# --- CPU: the knob, the hook, the refusals ---

@pytest.mark.parametrize("val,on", [(None, False), ("", False), ("0", False), ("true", False), ("on", False), ("1", True)])
def test_knob(monkeypatch, val, on):
    if val is None:
        monkeypatch.delenv("E4B_TRAIN_FUSE_QKV", raising=False)
    else:
        monkeypatch.setenv("E4B_TRAIN_FUSE_QKV", val)
    assert tq.train_fuse_qkv_requested() is on


def test_hook_does_nothing_unset_or_unpatched(monkeypatch):
    called = []
    monkeypatch.setattr(tq, "enable_train_fuse_qkv", lambda m, verbose=False: called.append(m) or 7)
    monkeypatch.delenv("E4B_TRAIN_FUSE_QKV", raising=False)
    assert fast._maybe_fuse_train_qkv(object(), patched=3) == 0
    monkeypatch.setenv("E4B_TRAIN_FUSE_QKV", "1")
    assert fast._maybe_fuse_train_qkv(object(), patched=0) == 0
    assert called == []
    assert fast._maybe_fuse_train_qkv("m", patched=3) == 7 and called == ["m"]


def test_refusals_keep_todays_path():
    cfg = _cfg()
    attn = qmod.Qwen3MoeAttention(cfg, layer_idx=0)
    assert "LoRALinear" in tq._refusal(attn)                       # unwrapped projections
    _lora_wrap(attn)
    assert "bitsandbytes 4-bit" in tq._refusal(attn)               # plain nn.Linear bases
    a2 = _lora_wrap(qmod.Qwen3MoeAttention(cfg, layer_idx=0))
    a2.k_proj = LoRALinear(a2.k_proj.base, 8, 8, torch.float32)
    assert "rank, dtype or scaling" in tq._refusal(a2)
    holder = _Holder(attn)
    fwd = attn.forward
    assert tq.enable_train_fuse_qkv(holder) == 0
    assert tq.TRAIN_QKV_STATS["refused"] and not hasattr(attn, "qkv_proj") and attn.forward == fwd


def test_other_attention_classes_untouched():
    class Other(nn.Module):
        def __init__(self):
            super().__init__()
            self.q_proj, self.k_proj, self.v_proj = nn.Linear(8, 8), nn.Linear(8, 8), nn.Linear(8, 8)
    m = _Holder(Other())
    assert tq.enable_train_fuse_qkv(m) == 0 and not hasattr(m.attn, "qkv_proj")


def test_a_released_projection_raises_a_clear_error():
    m = LoRALinear(nn.Linear(16, 8, bias=False), 4, 4, torch.float32)
    tq._release_base(m, "model.layers.0.self_attn.q_proj")
    assert m.base is None and m._e4b_fused_into_qkv
    assert [n for n, _ in m.named_parameters()] == ["lora_A", "lora_B"]
    with pytest.raises(RuntimeError, match=r"model\.layers\.0\.self_attn\.q_proj was fused .* E4B_TRAIN_FUSE_QKV=1"):
        m(torch.randn(2, 16))


# --- CUDA: NF4 bases ---

needs_nf4 = pytest.mark.skipif(not CUDA, reason="bitsandbytes NF4 Linear4bit needs CUDA here")


def _nf4_attn(seed=3, adapter_dtype=torch.float32):
    bnb = pytest.importorskip("bitsandbytes")
    cfg = _cfg()
    torch.manual_seed(seed)
    attn = qmod.Qwen3MoeAttention(cfg, layer_idx=0).to(torch.bfloat16)
    for n in ("q_proj", "k_proj", "v_proj", "o_proj"):
        lin = getattr(attn, n)
        q = bnb.nn.Linear4bit(lin.in_features, lin.out_features, bias=False, compute_dtype=torch.bfloat16, quant_type="nf4")
        q.weight = bnb.nn.Params4bit(lin.weight.data.clone(), requires_grad=False, quant_type="nf4")
        setattr(attn, n, q)
    attn = attn.cuda()
    return cfg, _lora_wrap(attn, dtype=adapter_dtype)


#: The adapter dtypes training runs: fp32 (matched init) and bf16 (the shipped default).
ADAPTER_DTYPES = pytest.mark.parametrize("adapter_dtype", [torch.float32, torch.bfloat16], ids=["fp32", "bf16"])


@needs_nf4
def test_dequant_is_bitwise():
    from bitsandbytes.functional import dequantize_4bit
    _, attn = _nf4_attn()
    want = torch.cat([dequantize_4bit(getattr(attn, n).base.weight.data, getattr(attn, n).base.weight.quant_state)
                      for n in ("q_proj", "k_proj", "v_proj")])
    assert torch.equal(tq.FusedQKVLoRA(attn).dequantized(), want)


def _bar(got, ref):
    tol = (2.0 ** -6 if ref.dtype == torch.bfloat16 else 2.0 ** -16) * ref.float().abs().max().item()
    return (got.float() - ref.float()).abs().max().item(), tol


@needs_nf4
@ADAPTER_DTYPES
def test_isolated_projection_within_rounding_bar(adapter_dtype):
    _, attn = _nf4_attn(adapter_dtype=adapter_dtype)
    fused = tq.FusedQKVLoRA(attn)
    g = torch.Generator(device="cuda").manual_seed(7)
    x = torch.randn(2, 9, 128, generator=g, device="cuda").to(torch.bfloat16)
    up = (torch.randn(2, 9, sum(fused.ns), generator=g, device="cuda") * 0.01).to(torch.bfloat16)

    def run(f):
        xx = x.detach().clone().requires_grad_(True)
        for n in ("q_proj", "k_proj", "v_proj"):
            for w in ("lora_A", "lora_B"):
                getattr(getattr(attn, n), w).grad = None
        out = f(xx)
        out.backward(up)
        return [out.detach(), xx.grad.detach()] + [getattr(getattr(attn, n), w).grad.detach().clone()
                                                   for n in ("q_proj", "k_proj", "v_proj") for w in ("lora_A", "lora_B")]
    ref = run(lambda xx: torch.cat([attn.q_proj(xx), attn.k_proj(xx), attn.v_proj(xx)], -1))
    got = run(fused)
    for i, (a, b) in enumerate(zip(got, ref)):
        d, tol = _bar(a, b)
        assert d <= tol, (i, d, tol)


@needs_nf4
@ADAPTER_DTYPES
def test_patched_attention_within_reorder_bar_and_adapters_stay_params(adapter_dtype):
    cfg, attn = _nf4_attn(adapter_dtype=adapter_dtype)
    ref_attn = copy.deepcopy(attn)
    holder = _Holder(attn)
    names = {n: id(p) for n, p in holder.named_parameters() if "lora" in n}
    assert tq.enable_train_fuse_qkv(holder) == 1 and tq.TRAIN_QKV_STATS["refused"] == {}
    assert {n: id(p) for n, p in holder.named_parameters() if "lora" in n} == names
    assert all(getattr(attn, n).base is None for n in ("q_proj", "k_proj", "v_proj"))
    with pytest.raises(RuntimeError, match="E4B_TRAIN_FUSE_QKV"):
        attn.k_proj(torch.randn(1, 3, cfg.hidden_size, device="cuda").to(torch.bfloat16))
    torch.manual_seed(9)
    x = torch.randn(1, 7, cfg.hidden_size, device="cuda").to(torch.bfloat16)
    rot = qmod.Qwen3MoeRotaryEmbedding(cfg).cuda()
    cos, sin = rot(x, torch.arange(7, device="cuda")[None])
    want, _ = ref_attn(x, (cos, sin), None)
    got, _ = attn(x, (cos, sin), None)
    d = (want.float() - got.float()).abs().max().item()
    assert d <= want.float().abs().max().item() * 2.0 ** -6, d
    got.float().sum().backward()
    assert all(getattr(getattr(attn, n), w).grad is not None for n in ("q_proj", "k_proj", "v_proj") for w in ("lora_A", "lora_B"))
