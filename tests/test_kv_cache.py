"""Property suite for the 4-bit KV cache (experts4bit_qlora/kv_cache.py).

The gates that matter: the cache must hand back what was written (within the
measured quantization bound), appending must not re-quantize history, the
footprint claim must be arithmetic, ineligible layers must degrade to bf16
rather than fail, and a real generate() must run through it.
"""
from __future__ import annotations

import pytest
import torch

from experts4bit_qlora import NF4KVCache, kv_nf4_available

kv = pytest.mark.skipif(not kv_nf4_available(),
                        reason="needs [fast] (grouped-nf4-gemm) + CUDA")


def _kv(T, H, D, seed=0):
    g = torch.Generator(device="cpu").manual_seed(seed)
    x = (torch.randn(1, H, T, D, generator=g) * 0.5).cuda().bfloat16()
    return x


@kv
def test_roundtrip_within_quantization_bound():
    """Bound is the MEASURED NF4 round-trip on iid data (~9.1%), with margin.

    Worth recording why this number recurs: the kernel suite measured V-only
    error THROUGH attention at 9.22%, essentially identical to V's round-trip
    error here. That is the mechanism, not a coincidence — attention's weighted
    sum is a convex combination, which preserves relative error magnitude, so V
    error passes through unattenuated while K error gets contracted by the
    softmax (9.2% logit -> 1.4%).
    """
    c = NF4KVCache()
    k, v = _kv(64, 4, 128, 1), _kv(64, 4, 128, 2)
    gk, gv = c.update(k, v, 0)
    assert gk.shape == k.shape and gv.shape == v.shape
    for name, got, want in (("K", gk, k), ("V", gv, v)):
        rel = ((got.float() - want.float()).norm() / want.float().norm()).item()
        assert rel < 0.11, f"{name} round-trip rel err {rel:.4f} (measured ~0.091)"


@kv
def test_append_does_not_requantize_history():
    """Appending must be O(new tokens): the packed prefix has to survive
    byte-identically, or a 32K cache is re-quantized on every decode step."""
    c = NF4KVCache()
    k0, v0 = _kv(32, 4, 128, 3), _kv(32, 4, 128, 4)
    c.update(k0, v0, 0)
    prefix = c._k[0][1][:32].clone()
    k1, v1 = _kv(1, 4, 128, 5), _kv(1, 4, 128, 6)
    c.update(k1, v1, 0)
    assert c.get_seq_length(0) == 33
    assert torch.equal(c._k[0][1][:32], prefix), "history was re-packed on append"


@kv
def test_footprint_is_measured_not_claimed():
    c = NF4KVCache()
    T, H, D = 512, 4, 128
    c.update(_kv(T, H, D, 7), _kv(T, H, D, 8), 0)
    got, fp16 = c.memory_bytes(), c.memory_bytes(fp16=True)
    assert fp16 == 2 * T * H * D * 2                       # K and V, bf16
    ratio = fp16 / got
    assert 3.5 < ratio < 3.6, f"expected ~3.56x, got {ratio:.3f}"


@kv
def test_ineligible_head_dim_degrades_to_bf16_not_failure():
    """96 is not a multiple of the 64-element blocksize; that layer must still
    work, in bf16, and say so."""
    c = NF4KVCache()
    c.update(_kv(8, 2, 96, 9), _kv(8, 2, 96, 10), 0)
    assert c.layer_modes[0] == "K=raw,V=raw"
    assert c.memory_bytes() == c.memory_bytes(fp16=True)


@kv
def test_asymmetric_switches():
    """V dominates the error on the kernel fixture, so keeping V in bf16 while
    quantizing K must be expressible."""
    c = NF4KVCache(quantize_keys=True, quantize_values=False)
    c.update(_kv(16, 4, 128, 11), _kv(16, 4, 128, 12), 0)
    assert c.layer_modes[0] == "K=nf4,V=raw"


@kv
def test_real_generate_runs_through_the_cache():
    """End-to-end: a real model, real generate(), our cache object."""
    transformers = pytest.importorskip("transformers")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    name = "HuggingFaceTB/SmolLM2-135M"
    try:
        tok = AutoTokenizer.from_pretrained(name)
        m = AutoModelForCausalLM.from_pretrained(name, dtype=torch.bfloat16).cuda().eval()
    except Exception as e:                      # no network / not cached
        pytest.skip(f"model unavailable: {type(e).__name__}")
    ids = tok("The capital of France is", return_tensors="pt").input_ids.cuda()
    torch.manual_seed(0)
    ref = m.generate(ids, max_new_tokens=12, do_sample=False)
    cache = NF4KVCache()
    torch.manual_seed(0)
    got = m.generate(ids, max_new_tokens=12, do_sample=False, past_key_values=cache)
    assert cache.get_seq_length(0) > 0, "cache was never populated"
    assert any("nf4" in mode for mode in cache.layer_modes.values()), cache.layer_modes
    saved = cache.memory_bytes(fp16=True) / max(cache.memory_bytes(), 1)
    assert saved > 3.0, f"expected >3x saving, got {saved:.2f}x"
    # text may differ (quantization is lossy) but generation must be well-formed
    assert got.shape[1] == ref.shape[1]
    print("\n  ref:", tok.decode(ref[0][ids.shape[1]:], skip_special_tokens=True))
    print("  nf4:", tok.decode(got[0][ids.shape[1]:], skip_special_tokens=True))
    print(f"  cache {cache.memory_bytes()/1024:.1f} KB vs bf16 "
          f"{cache.memory_bytes(fp16=True)/1024:.1f} KB ({saved:.2f}x)")


@kv
def test_per_channel_keys_roundtrip_and_tail_flushes():
    """Per-channel keys must survive an incremental append pattern identical to
    decode: the residual tail holds < group tokens in bf16 and flushes a group
    at a time, so the reconstructed cache must match a single-shot store."""
    c = NF4KVCache(key_scaling="per_channel", group=64)
    v = _kv(200, 4, 128, 9)
    k = _kv(200, 4, 128, 8)
    # feed in ragged chunks the way generation would
    got = None
    for lo, hi in ((0, 130), (130, 131), (131, 199), (199, 200)):
        got, _ = c.update(k[:, :, lo:hi], v[:, :, lo:hi], 0)
    assert got.shape == k.shape
    assert c.get_seq_length(0) == 200
    rel = ((got.float() - k.float()).norm() / k.float().norm()).item()
    assert rel < 0.11, f"per-channel round-trip {rel:.4f}"
    # 200 tokens = 3 full groups (192) quantized + 8 in the bf16 tail
    assert c._ktail[0].shape[2] == 8
    assert c._k[0][1].shape[0] == 192


@kv
def test_per_channel_costs_the_same_as_per_token():
    """The claim that motivates using it: identical bytes, different fidelity."""
    k, v = _kv(256, 4, 128, 1), _kv(256, 4, 128, 2)
    a = NF4KVCache(); a.update(k, v, 0)
    b = NF4KVCache(key_scaling="per_channel", group=64); b.update(k, v, 0)
    assert b.memory_bytes() == a.memory_bytes()


@kv
def test_mask_sizes_counts_tokens_about_to_be_written():
    """transformers builds the attention mask BEFORE the layers run, so
    kv_length must include this step's queries. Returning only the stored
    length gives a zero-width mask on the first forward — which blows up any
    model using an explicit additive mask (gpt-oss, via its attention sinks)
    while models tolerating a None mask hide it."""
    c = NF4KVCache()
    pos = torch.arange(16, device="cuda")
    assert c.get_mask_sizes(pos, 0) == (16, 0)          # empty cache, 16 new
    c.update(_kv(16, 4, 128, 1), _kv(16, 4, 128, 2), 0)
    nxt = torch.arange(16, 17, device="cuda")
    assert c.get_mask_sizes(nxt, 0) == (17, 0)          # 16 stored + 1 new
    # generate() passes a bare int query length, not a position tensor
    assert c.get_mask_sizes(1, 0) == (17, 0)
    assert c.get_mask_sizes([0, 1, 2], 0) == (19, 0)
