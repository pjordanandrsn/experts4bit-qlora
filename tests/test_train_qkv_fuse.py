# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""P129: one fused q/k/v projection for training attention, on by default (``E4B_TRAIN_FUSE_QKV=0`` keeps the three projections).

- On by default: unset, ``enable_fast_train``'s hook fuses; ``0`` / ``false`` / ``off`` / ``no`` keep today's path, and an
  unpatched model is never touched.
- Refusals keep today's path, with the reason recorded.
- ``disable_fast_train`` on a model with nothing fused changes nothing.
- On CUDA with bitsandbytes NF4 bases (the last two with fp32 and with bf16 adapters, the dtypes training runs):
  - the fused dequantize is bit for bit the three dequantized weights stacked;
  - the fused projection is within TC1's rounding bar of the three ``LoRALinear`` modules on outputs, the input gradient and every
    adapter gradient (``2**-6`` of the largest entry for bf16, ``2**-16`` for fp32);
  - the patched attention forward is within the reorder bar of the unfused one;
  - the adapters stay the parameters;
  - at the default (knob unset) the hook fuses, and the fused attention's forward and backward are within the same bars of the
    unfused one; with the knob at ``0`` the module is untouched;
  - the q/k/v bases stay registered, re-pointed at the fused copy (views, a non-nested fp32 absmax), and a direct call to one is
    bit for bit what it was before the fusion;
  - ``state_dict``: a fused model's carries every key an unfused model's does except each q/k/v base's two nested-statistics keys;
    it loads strict into a freshly loaded unfused model, whose forward is then bit for bit the fused-then-disabled one; loaded
    into a fused model (a resume), it round-trips and writes through the views into the fused bytes;
  - a move of the fused module's bytes re-points the bases at the moved copy;
  - ``disable_fast_train`` is a round trip: the projections and the attention compute bit for bit what they did before the
    fusion; a second enable fuses again, bit for bit as the first, and an enable of a fused module changes nothing.
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

@pytest.mark.parametrize("val,on", [(None, True), ("", True), ("1", True), ("true", True), ("on", True), ("0", False), ("false", False),
                                    ("FALSE", False), ("off", False), ("no", False), (" 0 ", False)])
def test_knob(monkeypatch, val, on):
    if val is None:
        monkeypatch.delenv("E4B_TRAIN_FUSE_QKV", raising=False)
    else:
        monkeypatch.setenv("E4B_TRAIN_FUSE_QKV", val)
    assert tq.train_fuse_qkv_requested() is on


def test_hook_fuses_by_default_and_not_when_off_or_unpatched(monkeypatch):
    called = []
    monkeypatch.setattr(tq, "enable_train_fuse_qkv", lambda m, verbose=False: called.append(m) or 7)
    monkeypatch.setenv("E4B_TRAIN_FUSE_QKV", "0")
    assert fast._maybe_fuse_train_qkv(object(), patched=3) == 0
    monkeypatch.delenv("E4B_TRAIN_FUSE_QKV", raising=False)
    assert fast._maybe_fuse_train_qkv(object(), patched=0) == 0
    assert called == []
    assert fast._maybe_fuse_train_qkv("m", patched=3) == 7 and called == ["m"]       # unset: the default fuses


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
    _assert_views(attn)
    xk = torch.randn(1, 3, cfg.hidden_size, device="cuda").to(torch.bfloat16)
    with torch.no_grad():                                          # a direct call still works, bit for bit as before
        assert all(torch.equal(getattr(attn, n)(xk), getattr(ref_attn, n)(xk)) for n in ("q_proj", "k_proj", "v_proj"))
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


@needs_nf4
@ADAPTER_DTYPES
def test_the_default_fuses_through_the_hook_within_the_bars(monkeypatch, adapter_dtype):
    """The new default end to end through ``enable_fast_train``'s hook: knob unset, the attention is fused and its forward output
    and the adapters' gradients stay within the bars of the unfused module; knob at 0, the module is untouched."""
    cfg, attn = _nf4_attn(adapter_dtype=adapter_dtype)
    ref_attn = copy.deepcopy(attn)
    kept = copy.deepcopy(attn)
    monkeypatch.setenv("E4B_TRAIN_FUSE_QKV", "0")
    assert fast._maybe_fuse_train_qkv(_Holder(kept), patched=1) == 0
    assert all(getattr(kept, n).base is not None for n in ("q_proj", "k_proj", "v_proj")) and not hasattr(kept, "qkv_proj")
    monkeypatch.delenv("E4B_TRAIN_FUSE_QKV", raising=False)
    assert fast._maybe_fuse_train_qkv(_Holder(attn), patched=1) == 1 and hasattr(attn, "qkv_proj")
    torch.manual_seed(9)
    x = torch.randn(1, 7, cfg.hidden_size, device="cuda").to(torch.bfloat16)
    rot = qmod.Qwen3MoeRotaryEmbedding(cfg).cuda()
    cos, sin = rot(x, torch.arange(7, device="cuda")[None])
    outs, grads = [], []
    for a in (ref_attn, attn):
        o, _ = a(x, (cos, sin), None)
        o.float().sum().backward()
        outs.append(o.detach())
        grads.append([getattr(getattr(a, n), w).grad.detach().clone() for n in ("q_proj", "k_proj", "v_proj") for w in ("lora_A", "lora_B")])
    d = (outs[0].float() - outs[1].float()).abs().max().item()
    assert d <= outs[0].float().abs().max().item() * 2.0 ** -6, d
    for g0, g1 in zip(*grads):
        dg, tol = _bar(g1, g0)
        assert dg <= max(tol, 2.0 ** -6 * g0.float().abs().max().item()), (dg, tol)


def test_disable_on_a_model_with_nothing_fused_changes_nothing():
    attn = _lora_wrap(qmod.Qwen3MoeAttention(_cfg(), layer_idx=0))
    holder = _Holder(attn)
    before = {n: id(m) for n, m in holder.named_modules()}
    stats = copy.deepcopy(tq.TRAIN_QKV_STATS)
    assert tq.disable_train_fuse_qkv(holder) == 0
    assert {n: id(m) for n, m in holder.named_modules()} == before and tq.TRAIN_QKV_STATS == stats


@needs_nf4
@ADAPTER_DTYPES
def test_disable_restores_the_three_projections_bit_for_bit(monkeypatch, adapter_dtype):
    """enable then disable is a round trip (TC1's attn_only arm probes enable_fast_train, then disables it and trains): the bases,
    views of the fused bytes with the expanded fp32 absmax, and the attention compute bit for bit what they did before the fusion. A
    second enable fuses again, bit for bit as the first, and an enable of a fused module changes nothing."""
    monkeypatch.delenv("E4B_TRAIN_FUSE_QKV", raising=False)
    cfg, attn = _nf4_attn(adapter_dtype=adapter_dtype)
    names = ("q_proj", "k_proj", "v_proj")
    torch.manual_seed(9)
    x = torch.randn(1, 7, cfg.hidden_size, device="cuda").to(torch.bfloat16)
    rot = qmod.Qwen3MoeRotaryEmbedding(cfg).cuda()
    pe = rot(x, torch.arange(7, device="cuda")[None])
    with torch.no_grad():
        proj0 = [getattr(attn, n)(x) for n in names]
        out0 = attn(x, pe, None)[0]
    holder = _Holder(attn)
    assert fast._maybe_fuse_train_qkv(holder, patched=1) == 1
    with torch.no_grad():
        fused0 = attn(x, pe, None)[0]
    assert tq.disable_train_fuse_qkv(holder) == 1 and tq.TRAIN_QKV_STATS["fused"] == 0
    assert not hasattr(attn, "qkv_proj") and "forward" not in attn.__dict__
    for n in names:
        m = getattr(attn, n)
        assert m.base is not None and "forward" not in m.__dict__ and not m.base.weight.quant_state.nested
    with torch.no_grad():
        assert all(torch.equal(getattr(attn, n)(x), want) for n, want in zip(names, proj0))
        assert torch.equal(attn(x, pe, None)[0], out0)
    o, _ = attn(x, pe, None)
    o.float().sum().backward()
    assert all(getattr(getattr(attn, n), w).grad is not None for n in names for w in ("lora_A", "lora_B"))
    assert fast._maybe_fuse_train_qkv(holder, patched=1) == 1
    with torch.no_grad():
        assert torch.equal(attn(x, pe, None)[0], fused0)
    fq, fwd = attn.qkv_proj, attn.forward
    assert tq.enable_train_fuse_qkv(holder) == 1 and attn.qkv_proj is fq and attn.forward == fwd   # fused already: unchanged


def _assert_views(attn):
    """Each q/k/v base holds a view of the fused bytes and a non-nested quant state over the fused absmax."""
    fq = attn.qkv_proj
    for n in ("q_proj", "k_proj", "v_proj"):
        w = getattr(attn, n).base.weight
        assert w.untyped_storage().data_ptr() == fq.packed.untyped_storage().data_ptr(), n
        assert not w.quant_state.nested and w.quant_state.absmax.untyped_storage().data_ptr() == fq.absmax.untyped_storage().data_ptr()


NESTED_KEYS = ("weight.nested_absmax", "weight.nested_quant_map")


def _clone_sd(m):
    return {k: v.detach().clone() for k, v in m.state_dict().items()}


def _scramble_adapters(attn, seed):
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for n, p in attn.named_parameters():
            if "lora" in n:
                p.copy_(torch.randn(p.shape, generator=g).to(p.device, p.dtype))


@needs_nf4
@ADAPTER_DTYPES
def test_a_fused_state_dict_loads_strict_into_an_unfused_model_bit_for_bit(monkeypatch, adapter_dtype):
    """A full-model save of a fused model (a Trainer checkpoint, ``save_pretrained``) carries every q/k/v base; a freshly loaded unfused
    model takes it strict, and then computes bit for bit what the fused model computes once disabled. The key sets differ only by
    each q/k/v base's two nested-statistics keys: the fused bases' quant state is the non-nested fp32 form."""
    monkeypatch.delenv("E4B_TRAIN_FUSE_QKV", raising=False)
    cfg, attn = _nf4_attn(adapter_dtype=adapter_dtype)
    unfused_keys = set(_Holder(attn).state_dict())
    holder = _Holder(attn)
    assert tq.enable_train_fuse_qkv(holder) == 1
    sd = _clone_sd(holder)
    want_missing = {f"attn.{n}.base.{k}" for n in ("q_proj", "k_proj", "v_proj") for k in NESTED_KEYS}
    assert want_missing <= unfused_keys and set(sd) == unfused_keys - want_missing
    _, fresh = _nf4_attn(adapter_dtype=adapter_dtype)
    _scramble_adapters(fresh, 11)
    res = _Holder(fresh).load_state_dict(sd, strict=True)
    assert not res.missing_keys and not res.unexpected_keys
    assert tq.disable_train_fuse_qkv(holder) == 1
    torch.manual_seed(9)
    x = torch.randn(1, 7, cfg.hidden_size, device="cuda").to(torch.bfloat16)
    pe = qmod.Qwen3MoeRotaryEmbedding(cfg).cuda()(x, torch.arange(7, device="cuda")[None])
    with torch.no_grad():
        assert torch.equal(fresh(x, pe, None)[0], attn(x, pe, None)[0])


@needs_nf4
@ADAPTER_DTYPES
def test_a_resume_into_a_fused_model_round_trips_through_the_views(monkeypatch, adapter_dtype):
    """``load_state_dict`` into a fused model (a resume): the adapters take the saved values, the fused forward is bit for bit the
    saved model's, and a loaded q/k/v weight writes through its view into the fused bytes."""
    monkeypatch.delenv("E4B_TRAIN_FUSE_QKV", raising=False)
    cfg, a = _nf4_attn(adapter_dtype=adapter_dtype)
    ha = _Holder(a)
    assert tq.enable_train_fuse_qkv(ha) == 1
    sd = _clone_sd(ha)
    _, b = _nf4_attn(adapter_dtype=adapter_dtype)
    hb = _Holder(b)
    assert tq.enable_train_fuse_qkv(hb) == 1
    _scramble_adapters(b, 12)
    res = hb.load_state_dict(sd, strict=True)
    assert not res.missing_keys and not res.unexpected_keys
    _assert_views(b)
    assert all(torch.equal(p, sd[n]) for n, p in hb.named_parameters() if "lora" in n)
    torch.manual_seed(9)
    x = torch.randn(1, 7, cfg.hidden_size, device="cuda").to(torch.bfloat16)
    pe = qmod.Qwen3MoeRotaryEmbedding(cfg).cuda()(x, torch.arange(7, device="cuda")[None])
    with torch.no_grad():
        assert torch.equal(b(x, pe, None)[0], a(x, pe, None)[0])
    sd2 = dict(sd)
    k = "attn.k_proj.base.weight"
    sd2[k] = sd[k] ^ 0x11                                              # different NF4 codes: the load must reach the fused bytes
    hb.load_state_dict(sd2, strict=True)
    nq = sd["attn.q_proj.base.weight"].numel()
    assert torch.equal(b.qkv_proj.packed[nq:nq + sd[k].numel()].view(sd[k].shape), sd2[k])


@needs_nf4
def test_a_move_of_the_fused_bytes_re_points_the_bases():
    """``_apply`` (``.to`` / ``.cuda`` / ``.cpu``) gives the fused module new bytes; the bases follow, so nothing is held twice and the
    projections and the fused forward compute what they did."""
    cfg, attn = _nf4_attn()
    holder = _Holder(attn)
    assert tq.enable_train_fuse_qkv(holder) == 1
    torch.manual_seed(9)
    x = torch.randn(1, 7, cfg.hidden_size, device="cuda").to(torch.bfloat16)
    pe = qmod.Qwen3MoeRotaryEmbedding(cfg).cuda()(x, torch.arange(7, device="cuda")[None])
    with torch.no_grad():
        before, k_before = attn(x, pe, None)[0], attn.k_proj(x)
    old = attn.qkv_proj.packed.untyped_storage().data_ptr()
    attn.qkv_proj._apply(lambda t: t.clone())
    assert attn.qkv_proj.packed.untyped_storage().data_ptr() != old
    _assert_views(attn)
    with torch.no_grad():
        assert torch.equal(attn(x, pe, None)[0], before) and torch.equal(attn.k_proj(x), k_before)
