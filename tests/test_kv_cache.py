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


@kv
def test_eviction_keeps_sinks_plus_recent_and_reports_held_length():
    """Sparsity is the other axis: quantization shrinks each token, eviction
    removes tokens. get_seq_length must report what is HELD, not what was seen,
    or the attention mask describes a cache that does not exist."""
    for kw in (dict(quantize_keys=False, quantize_values=False),
               dict(quantize_keys=True, quantize_values=True)):
        c = NF4KVCache(keep_sink=4, keep_recent=64, **kw)
        for lo in range(0, 256, 64):
            k, v = _kv(64, 4, 128, 1), _kv(64, 4, 128, 2)
            got, _ = c.update(k, v, 0)
            c.evict()                                  # between forwards, never during
        assert c.held_length(0) == 68, kw              # 4 sinks + 64 recent HELD
        assert c.get_seq_length(0) == 256, kw          # but 256 tokens SEEN
        # the mask must describe the cache that exists, positions the true stream
        assert c.get_mask_sizes(1, 0) == (69, 0), kw
        assert c.get_query_offset(0) == 68, kw


@kv
def test_eviction_actually_frees_memory():
    """The point of the exercise: bytes must fall, and compose with NF4."""
    def bytes_for(**kw):
        c = NF4KVCache(**kw)
        for _ in range(8):
            c.update(_kv(128, 4, 128, 1), _kv(128, 4, 128, 2), 0)
            c.evict()
        return c.memory_bytes()
    full16 = bytes_for(quantize_keys=False, quantize_values=False)
    full4 = bytes_for(quantize_keys=True, quantize_values=True)
    evict16 = bytes_for(quantize_keys=False, quantize_values=False,
                        keep_sink=4, keep_recent=256)
    evict4 = bytes_for(quantize_keys=True, quantize_values=True,
                       keep_sink=4, keep_recent=256)
    assert full16 / full4 == pytest.approx(3.56, abs=0.05)      # quantization
    assert full16 / evict16 == pytest.approx(1024 / 260, abs=0.1)  # eviction
    # the claim that matters: the two axes multiply
    assert full16 / evict4 > 13.0, full16 / evict4


@kv
def test_chunked_prefill_matches_single_forward():
    """The invariant that a single-forward test cannot express: feeding a
    sequence in chunks must give the same logits as feeding it whole. This is
    the accumulating path — decode and chunked prefill both live here — and it
    is where get_query_offset's value actually matters. Without it the suite
    can be green on four architectures while the multi-call path is broken."""
    from transformers import AutoModelForCausalLM, DynamicCache
    # fp32: bf16 picks different matmul kernels for a 64-row chunk than for a
    # 256-row batch, which alone costs ~1.5e-2 even for transformers' OWN cache.
    # fp32 removes that confound so the assertion is about the cache, not the
    # numerics.
    m = AutoModelForCausalLM.from_pretrained(
        "HuggingFaceTB/SmolLM2-135M", dtype=torch.float32).cuda().eval()
    ids = (torch.arange(256, device="cuda") % 20000).unsqueeze(0)

    def chunked(cache):
        with torch.no_grad():
            return torch.cat([m(ids[:, lo:lo + 64], past_key_values=cache,
                                use_cache=True).logits.float()
                              for lo in range(0, 256, 64)], dim=1)

    with torch.no_grad():
        whole = m(ids, use_cache=False).logits.float()
    mine = chunked(NF4KVCache(quantize_keys=False, quantize_values=False))
    rel = ((mine - whole).norm() / whole.norm()).item()
    assert rel < 1e-4, f"chunked prefill diverged from single forward: {rel:.3e}"
    # and it must track transformers' own cache under the identical protocol
    theirs = chunked(DynamicCache())
    assert ((mine - theirs).norm() / theirs.norm()).item() < 1e-6


@kv
def test_evict_index_keeps_an_arbitrary_set_and_matches_evict_on_its_special_case():
    """The general keep-set. Recency is a prefix plus a suffix; an
    importance-based policy keeps neither, so the store has to survive an
    arbitrary gather along the token axis."""
    for kw in (dict(quantize_keys=False, quantize_values=False),
               dict(quantize_keys=True, quantize_values=True)):
        k, v = _kv(64, 4, 128, 1), _kv(64, 4, 128, 2)
        c = NF4KVCache(**kw)
        c.update(k, v, 0)
        want = torch.tensor([0, 1, 2, 3, 17, 40, 41, 63], device="cuda")
        ref_k = c._load(c._k[0], torch.bfloat16)[:, :, want]      # what we asked to keep
        c.evict_index(want)
        assert c.held_length(0) == 8, kw
        assert c.get_seq_length(0) == 64, kw                      # SEEN is untouched
        got_k = c._load(c._k[0], torch.bfloat16)
        assert torch.equal(got_k, ref_k), kw                      # exactly those rows
        # and the sink+recent case must agree with the special-cased evict()
        a, b = NF4KVCache(keep_sink=4, keep_recent=8, **kw), NF4KVCache(**kw)
        a.update(k, v, 0); b.update(k, v, 0)
        a.evict()
        b.evict_index(torch.cat([torch.arange(4), torch.arange(56, 64)]).cuda())
        assert torch.equal(a._load(a._k[0], torch.bfloat16),
                           b._load(b._k[0], torch.bfloat16)), kw
        assert a.memory_bytes() == b.memory_bytes(), kw


@kv
def test_evict_index_refuses_per_channel_keys():
    """Per-channel absmax is grouped over RUNS of tokens; an arbitrary keep-set
    would leave scales describing groups that no longer exist. Refuse loudly."""
    c = NF4KVCache(key_scaling="per_channel")
    c.update(_kv(128, 4, 128, 1), _kv(128, 4, 128, 2), 0)
    with pytest.raises(NotImplementedError, match="per_channel"):
        c.evict_index(torch.arange(8, device="cuda"))


@kv
def test_host_residence_is_byte_identical_to_resident():
    """Streaming must change WHERE the bytes live, not WHAT they are.

    Quantization runs on the GPU in both paths and only the packed result moves
    host-side, so this is exact equality -- not a tolerance. If it ever needs a
    tolerance, something quantized on the wrong device and a residence change
    quietly became a fidelity change.
    """
    for kw in (dict(quantize_keys=False, quantize_values=False),
               dict(quantize_keys=True, quantize_values=True),
               dict(quantize_keys=False, quantize_values=True)):
        gpu = NF4KVCache(**kw)
        host = NF4KVCache(residence="host", max_context=512, **kw)
        for lo in range(0, 256, 64):
            k, v = _kv(64, 4, 128, lo + 1), _kv(64, 4, 128, lo + 2)
            gk, gv = gpu.update(k, v, 0)
            hk, hv = host.update(k, v, 0)
            assert hk.is_cuda and hv.is_cuda, "streamed layer must land on device"
            assert torch.equal(gk, hk), (kw, lo, "keys diverged")
            assert torch.equal(gv, hv), (kw, lo, "values diverged")
        assert host.held_length(0) == gpu.held_length(0) == 256, kw
        assert host.memory_bytes() == gpu.memory_bytes(), kw   # same cache size
        assert gpu.device_bytes() == gpu.memory_bytes(), kw
        assert host.device_bytes() == 0, kw                    # ...none of it resident


@kv
def test_host_residence_holds_no_vram_and_survives_eviction():
    """The capacity claim, and the arena invariant that eviction can break.

    Eviction builds a fresh tensor of the survivors, which is no longer a view
    into the pinned arena -- so without recompaction the next append writes at
    an offset the data no longer occupies and the cache silently corrupts.
    """
    for kw in (dict(quantize_keys=True, quantize_values=True),
               dict(quantize_keys=False, quantize_values=False)):
        gpu = NF4KVCache(keep_sink=4, keep_recent=64, **kw)
        host = NF4KVCache(keep_sink=4, keep_recent=64, residence="host",
                          max_context=512, **kw)
        for lo in range(0, 256, 64):
            k, v = _kv(64, 4, 128, lo + 1), _kv(64, 4, 128, lo + 2)
            gk, _ = gpu.update(k, v, 0)
            hk, _ = host.update(k, v, 0)
            assert torch.equal(gk, hk), (kw, lo)      # matches across evictions
            gpu.evict(); host.evict()
        assert host.held_length(0) == gpu.held_length(0) == 68, kw
        assert host.get_seq_length(0) == 256, kw
        assert host.device_bytes() == 0, kw
        assert host.memory_bytes() == gpu.memory_bytes(), kw
        # and the arbitrary keep-set path recompacts too
        want = torch.tensor([0, 1, 2, 3, 30, 60, 67], device="cuda")
        gpu.evict_index(want); host.evict_index(want)
        assert torch.equal(gpu._load(gpu._k[0], torch.bfloat16),
                           host._load(host._k[0], torch.bfloat16)), kw
        k, v = _kv(32, 4, 128, 9), _kv(32, 4, 128, 10)
        assert torch.equal(gpu.update(k, v, 0)[0], host.update(k, v, 0)[0]), kw


@kv
def test_host_residence_refuses_what_it_cannot_represent():
    """Two loud failures beat two silent ones: an unsized arena, and an
    overflow. Growing on demand is not offered -- that is the O(T^2) cat this
    design exists to avoid."""
    with pytest.raises(ValueError, match="max_context"):
        NF4KVCache(residence="host")
    with pytest.raises(ValueError, match="residence"):
        NF4KVCache(residence="disk")
    c = NF4KVCache(residence="host", max_context=64)
    c.update(_kv(64, 4, 128, 1), _kv(64, 4, 128, 2), 0)
    with pytest.raises(ValueError, match="arena overflow"):
        c.update(_kv(8, 4, 128, 3), _kv(8, 4, 128, 4), 0)


@kv
def test_host_residence_matches_resident_through_a_real_model():
    """The end-to-end gate: LOGITS, not cache objects.

    The unit tests above compare what the cache hands back. This compares what
    the MODEL computes from it, across chunked prefill -- the accumulating path
    where get_query_offset lives and where a residence bug (a stale arena view,
    a slice that stopped being pinned, a copy landing on the wrong stream) would
    actually surface. fp32 so the comparison is about residence, not about bf16
    picking different kernels for different chunk shapes.
    """
    from transformers import AutoModelForCausalLM
    m = AutoModelForCausalLM.from_pretrained(
        "HuggingFaceTB/SmolLM2-135M", dtype=torch.float32).cuda().eval()
    ids = (torch.arange(256, device="cuda") % 20000).unsqueeze(0)

    def chunked(cache):
        with torch.no_grad():
            return torch.cat([m(ids[:, lo:lo + 64], past_key_values=cache,
                                use_cache=True).logits
                              for lo in range(0, 256, 64)], dim=1)

    for kw in (dict(quantize_keys=True, quantize_values=True),
               dict(quantize_keys=False, quantize_values=False)):
        resident = chunked(NF4KVCache(**kw))
        streamed = chunked(NF4KVCache(residence="host", max_context=256, **kw))
        assert torch.equal(resident, streamed), (
            kw, (resident - streamed).abs().max().item())
