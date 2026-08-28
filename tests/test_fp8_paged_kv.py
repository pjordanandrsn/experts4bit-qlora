# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""Batched FP8 paged KV (Phase 7): the pool-side plumbing that hands the
fused kernel its arguments. The round-trip oracle here is quantize-direct —
whatever ``Fp8PagedKV`` stores through block rows must dequantize to
EXACTLY what quantizing the same tokens directly produces, block splits,
partial tails, and interleaved sequences included; any drift is plumbing,
not format. The kernel-vs-reference numerics live in gnf4's
test_fp8_paged_attn; the CUDA test here closes the loop end to end
(append -> pools -> kernel vs reference dequant -> SDPA) under the
serving path's documented tolerance (invariant 4')."""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("row_pool", reason="needs grouped-nf4-gemm N-series")
pytest.importorskip("fp8_kv", reason="needs grouped-nf4-gemm N-series")

from experts4bit_qlora.engines.fp8_paged_kv import Fp8PagedKV  # noqa: E402
from fp8_kv import dequant_kv_fp8_ref, quantize_kv_fp8  # noqa: E402

L, H, D = 2, 2, 32
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def _tokens(t, seed):
    g = torch.Generator().manual_seed(seed)
    return (torch.randn(t, H, D, generator=g) * 1.5).to(DEV), \
        torch.randn(t, H, D, generator=g).to(DEV)


def _direct(x, groups):
    q, s = quantize_kv_fp8(x, group=None if groups == 1 else D // groups)
    return dequant_kv_fp8_ref(q, s)


@pytest.mark.parametrize("k_groups", [1, 4])
@pytest.mark.parametrize("schedule", [
    [5, 1, 1, 1],           # prefill + decodes inside one block
    [16, 16],               # exact block boundaries
    [23, 1] + [1] * 40,     # tail crossing two boundaries
    [50],                   # multi-block prefill in one call
])
def test_roundtrip_matches_direct_quantization(k_groups, schedule):
    kv = Fp8PagedKV(L, H, D, batch=1, max_tokens_per_seq=128,
                    k_groups=k_groups, device=DEV)
    ks, vs = [], []
    pos = 0
    for t_new in schedule:
        k, v = _tokens(t_new, seed=100 + pos)
        for layer in range(L):
            kv.append(layer, 0, k, v)
        ks.append(k)
        vs.append(v)
        pos += t_new
    k_all, v_all = torch.cat(ks), torch.cat(vs)
    for layer in range(L):
        got_k, got_v = kv.reference_kv(layer, 0)
        assert torch.equal(got_k, _direct(k_all, k_groups)), \
            f"K drift layer {layer}: pool bytes != direct quantization"
        assert torch.equal(got_v, _direct(v_all, 1))
        assert int(kv.seq_lens[layer, 0]) == k_all.shape[0]


def test_interleaved_sequences_stay_separate():
    """Appends alternating between sequences allocate interleaved pool rows;
    the block table must keep each sequence's story straight."""
    kv = Fp8PagedKV(1, H, D, batch=3, max_tokens_per_seq=64, device=DEV)
    per_seq = {0: [], 1: [], 2: []}
    step = 0
    for rounds in range(4):
        for seq, t_new in ((0, 7), (1, 16), (2, 3)):
            k, v = _tokens(t_new, seed=1000 + step)
            kv.append(0, seq, k, v)
            per_seq[seq].append((k, v))
            step += 1
    tbl = kv.block_table[0]
    used = tbl[kv.seq_lens[0] > 0]
    for seq, chunks in per_seq.items():
        k_all = torch.cat([c[0] for c in chunks])
        v_all = torch.cat([c[1] for c in chunks])
        got_k, got_v = kv.reference_kv(0, seq)
        assert torch.equal(got_k, _direct(k_all, 4))
        assert torch.equal(got_v, _direct(v_all, 1))
    # rows really are interleaved across sequences (this test would pass
    # trivially if each sequence's blocks were contiguous)
    n0 = -(-sum(c[0].shape[0] for c in per_seq[0]) // 16)
    assert not torch.equal(tbl[0, :n0].cpu(),
                           torch.arange(n0, dtype=torch.int32)), \
        "expected interleaved allocation; schedule no longer exercises it"
    del used


def test_overflow_refused():
    kv = Fp8PagedKV(1, H, D, batch=1, max_tokens_per_seq=32, device=DEV)
    k, v = _tokens(32, seed=5)
    kv.append(0, 0, k, v)
    with pytest.raises(ValueError, match="overflows"):
        kv.append(0, 0, *_tokens(1, seed=6))


def test_bytes_per_token_honest():
    kv = Fp8PagedKV(1, H, D, batch=1, max_tokens_per_seq=32,
                    k_groups=4, device=DEV)
    # K: payload + 4 fp32 group scales per (token, head); V: payload + 1
    expect = (H * D + H * 4 * 4) + (H * D + H * 4)
    assert kv.bytes_per_token() == expect


def test_append_batch_matches_per_seq():
    kv1 = Fp8PagedKV(1, H, D, batch=2, max_tokens_per_seq=32, device=DEV)
    kv2 = Fp8PagedKV(1, H, D, batch=2, max_tokens_per_seq=32, device=DEV)
    k, v = _tokens(2, seed=9)
    kb = k[:, :, None].permute(0, 1, 2, 3)  # -> [B, H, 1, D]
    vb = v[:, :, None].permute(0, 1, 2, 3)
    kv1.append_batch(0, kb, vb)
    for b in range(2):
        kv2.append(0, b, k[b:b + 1], v[b:b + 1])
    for b in range(2):
        a = kv1.reference_kv(0, b)
        c = kv2.reference_kv(0, b)
        assert torch.equal(a[0], c[0]) and torch.equal(a[1], c[1])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_kernel_attention_end_to_end():
    """append -> pools -> fused kernel, against SDPA over the reference
    dequant of the same pools. Same bytes both ways, so the tolerance is
    softmax accumulation order only — documented serving tolerance."""
    pytest.importorskip("fp8_paged_attn")
    hq, hkv, d = 8, 2, 64
    kv = Fp8PagedKV(1, hkv, d, batch=3, max_tokens_per_seq=128,
                    k_groups=2, device="cuda")
    lens = [100, 37, 128]
    for seq, t in enumerate(lens):
        k, v = (torch.randn(t, hkv, d) * 1.5, torch.randn(t, hkv, d))
        kv.append(0, seq, k.cuda(), v.cuda())
    q = (torch.randn(3, hq, d) * 0.5).to(torch.bfloat16).cuda()
    out = kv.attention(0, q)
    for seq, t in enumerate(lens):
        kr, vr = kv.reference_kv(0, seq)
        ref = torch.nn.functional.scaled_dot_product_attention(
            q[seq][None, :, None].float(),
            kr.permute(1, 0, 2)[None].float(),
            vr.permute(1, 0, 2)[None].float(),
            enable_gqa=True)[0, :, 0]
        assert torch.allclose(out[seq].float(), ref, atol=2e-2, rtol=2e-2), \
            f"seq {seq}: kernel vs reference-SDPA beyond serving tolerance"


def test_failed_append_leaves_pools_in_lockstep(monkeypatch):
    """A throw mid-append (e.g. OOM inside quantization) must not desync
    the two pools' tails: both sides quantize BEFORE either pool is
    touched, so a failed append changes nothing and the next append's K
    and V land at the same shared block-table row (review finding)."""
    import experts4bit_qlora.engines.fp8_paged_kv as mod
    kv = Fp8PagedKV(1, H, D, batch=1, max_tokens_per_seq=64, device=DEV)
    k0, v0 = _tokens(16, seed=1)
    kv.append(0, 0, k0, v0)
    tails = (kv.kp.tail[0], kv.vp.tail[0])

    calls = {"n": 0}
    orig = mod.Fp8PagedKV._quant_bytes

    def failing(self, x, groups):
        calls["n"] += 1
        if calls["n"] == 2:            # the K side of this append
            raise RuntimeError("simulated OOM")
        return orig(self, x, groups)

    monkeypatch.setattr(mod.Fp8PagedKV, "_quant_bytes", failing)
    k1, v1 = _tokens(16, seed=2)
    with pytest.raises(RuntimeError, match="simulated OOM"):
        kv.append(0, 0, k1, v1)
    monkeypatch.setattr(mod.Fp8PagedKV, "_quant_bytes", orig)

    assert (kv.kp.tail[0], kv.vp.tail[0]) == tails, \
        "failed append advanced a pool tail"
    assert kv._seen[0][0] == 16 and int(kv.seq_lens[0, 0]) == 16
    # and the NEXT append still round-trips both sides at the same rows
    kv.append(0, 0, k1, v1)
    got_k, got_v = kv.reference_kv(0, 0)
    assert torch.equal(got_k, _direct(torch.cat([k0, k1]), 4))
    assert torch.equal(got_v, _direct(torch.cat([v0, v1]), 1))


def test_slot_reset_recycles_blocks_and_isolates_the_next_sequence():
    """A serving loop recycles slots continuously. Without reclaim the
    pool is a one-shot arena that dies at `blocks` sequences no matter
    how few ran at once; with a leaky reclaim the next tenant inherits
    the last one's context, which reads as fluent nonsense rather than a
    crash."""
    kv = Fp8PagedKV(2, H, D, batch=2, max_tokens_per_seq=64, device=DEV)
    free0 = kv.free_blocks(0)
    k1, v1 = _tokens(33, seed=1)
    kv.append(0, 0, k1, v1)
    kv.append(1, 0, k1, v1)
    assert kv.free_blocks(0) < free0, "append did not consume blocks"

    kv.reset(0)
    assert kv.free_blocks(0) == free0, "reset leaked blocks"
    assert int(kv.seq_lens[0, 0]) == 0
    got_k, got_v = kv.reference_kv(0, 0)
    assert got_k.shape[0] == 0, "reset left the old sequence readable"

    # the recycled slot serves a fresh sequence with no trace of the old
    k2, v2 = _tokens(5, seed=2)
    kv.append(0, 0, k2, v2)
    got_k, got_v = kv.reference_kv(0, 0)
    assert torch.equal(got_k, _direct(k2, 4))
    assert torch.equal(got_v, _direct(v2, 1))


def test_every_slot_can_reach_its_cap_simultaneously():
    """The sizing invariant, and the reason block exhaustion is not a
    runtime hazard: the pool holds batch x blocks_per_seq rows, so all
    slots filling to max_tokens_per_seq at once still fits — with reuse
    in the mix, which is where an off-by-one in reclaim would show."""
    kv = Fp8PagedKV(1, H, D, batch=3, max_tokens_per_seq=32, device=DEV)
    for slot in range(3):
        kv.append(0, slot, *_tokens(32, seed=10 + slot))
    assert kv.free_blocks(0) == 0
    kv.reset(1)
    kv.append(0, 1, *_tokens(32, seed=20))          # recycled tenant
    assert kv.free_blocks(0) == 0
    for slot in range(3):
        assert int(kv.seq_lens[0, slot]) == 32
    # a sequence past its own cap is refused by the per-sequence check,
    # which fires before the pool can ever run dry
    with pytest.raises(ValueError, match="overflows"):
        kv.append(0, 0, *_tokens(1, seed=30))


def test_exhausted_free_list_is_refused_loudly():
    """White-box guard: if reclaim ever leaked, allocation must fail
    loudly rather than hand out a row another sequence is still
    reading."""
    kv = Fp8PagedKV(1, H, D, batch=2, max_tokens_per_seq=64, device=DEV)
    kv._free[0].clear()
    with pytest.raises(RuntimeError, match="out of KV blocks"):
        kv.append(0, 0, *_tokens(4, seed=31))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_attention_maps_rows_to_slots_not_positions():
    """A serving loop decodes SUBSETS — sequences finish at different
    times — so the kernel's batch rows must be selected by slot. Passing
    the full tables works only while the active set is slots 0..B-1 in
    order; otherwise a sequence attends over another's KV, which reads as
    a model that starts coherent and degenerates rather than as a
    crash."""
    pytest.importorskip("fp8_paged_attn")
    hq, hkv, d = 8, 2, 64
    kv = Fp8PagedKV(1, hkv, d, batch=3, max_tokens_per_seq=64,
                    k_groups=2, device="cuda")
    for slot, n in ((0, 8), (1, 24), (2, 40)):
        k = (torch.randn(n, hkv, d) * 1.5).cuda()
        v = torch.randn(n, hkv, d).cuda()
        kv.append(0, slot, k, v)
    q = (torch.randn(1, hq, d) * 0.5).to(torch.bfloat16).cuda()

    # decoding ONLY slot 2 must attend over slot 2's 40 tokens
    got = kv.attention(0, q, slots=[2])
    kr, vr = kv.reference_kv(0, 2)
    ref = torch.nn.functional.scaled_dot_product_attention(
        q[0][None, :, None].float(), kr.permute(1, 0, 2)[None].float(),
        vr.permute(1, 0, 2)[None].float(), enable_gqa=True)[0, :, 0]
    torch.testing.assert_close(got[0].float(), ref, rtol=5e-2, atol=5e-2)

    # and the mismatch is refused rather than silently mis-mapped
    with pytest.raises(ValueError, match="pass slots="):
        kv.attention(0, q)


# ---------------------------------------------------------------------------
# Batched row writes (append_many's scatter fast path). The oracle is the
# per-sequence loop itself: E4B_BATCHED_KV_WRITE=0 builds a cache through
# _write_side, and the batched cache must be BYTE-identical — pools,
# seq_lens and _seen — after the same append schedule. The scatter is a
# pure re-addressing of the same quantized bytes, so any drift is index
# arithmetic, not numerics.

def _pair(monkeypatch, batch, max_tokens):
    monkeypatch.setenv("E4B_BATCHED_KV_WRITE", "0")
    loop = Fp8PagedKV(L, H, D, batch=batch, max_tokens_per_seq=max_tokens,
                      k_groups=4, device=DEV)
    monkeypatch.setenv("E4B_BATCHED_KV_WRITE", "1")
    fast = Fp8PagedKV(L, H, D, batch=batch, max_tokens_per_seq=max_tokens,
                      k_groups=4, device=DEV)
    assert not loop._batched_write and fast._batched_write
    return loop, fast


def _drive(kv, schedule):
    for i, (layer, seqs, t_new) in enumerate(schedule):
        g = torch.Generator().manual_seed(1000 + i)
        k = (torch.randn(len(seqs), t_new, H, D, generator=g) * 1.5).to(DEV)
        v = torch.randn(len(seqs), t_new, H, D, generator=g).to(DEV)
        kv.append_many(layer, seqs, k, v)


def _assert_identical(loop, fast):
    assert torch.equal(loop.kp.dev, fast.kp.dev)
    assert torch.equal(loop.vp.dev, fast.vp.dev)
    assert torch.equal(loop.seq_lens, fast.seq_lens)
    assert loop._seen == fast._seen


@pytest.mark.parametrize("schedule", [
    # uniform decode: every sequence at the same fill
    [(l, [0, 1, 2, 3], 1) for _ in range(40) for l in range(L)],
    # prefill then decode across a block boundary
    [(l, [0, 1, 2, 3], 8) for l in range(L)]
    + [(l, [0, 1, 2, 3], 1) for _ in range(20) for l in range(L)],
    # multi-token appends that stay inside one block
    [(l, [0, 1, 2, 3], 4) for _ in range(6) for l in range(L)],
])
def test_batched_write_matches_loop(monkeypatch, schedule):
    loop, fast = _pair(monkeypatch, batch=4, max_tokens=128)
    _drive(loop, schedule)
    _drive(fast, schedule)
    _assert_identical(loop, fast)


def test_batched_write_ragged_fills(monkeypatch):
    """Sequences at DIFFERENT in-block offsets — the case the fill-grouped
    form handled with one scatter per group and this one handles with one
    scatter per region regardless of spread."""
    loop, fast = _pair(monkeypatch, batch=4, max_tokens=256)
    sched = []
    for q in range(4):                      # stagger starts by q tokens
        sched += [(l, [q], 1) for l in range(L) for _ in range(q)]
    sched += [(l, [0, 1, 2, 3], 1) for _ in range(60) for l in range(L)]
    _drive(loop, sched)
    _drive(fast, sched)
    _assert_identical(loop, fast)


def test_batched_write_straddle_falls_back(monkeypatch):
    """A multi-token append that crosses a block boundary must take the
    per-sequence loop — and appends AFTER the fallback must stay correct,
    which is exactly what a stale advance-cached index would corrupt
    (the loop advances _seen without touching the cache)."""
    loop, fast = _pair(monkeypatch, batch=4, max_tokens=256)
    sched = []
    for _ in range(3):
        sched += [(l, [0, 1, 2, 3], 1) for _ in range(14) for l in range(L)]
        sched += [(l, [0, 1, 2, 3], 5) for l in range(L)]   # 14+5 straddles
    _drive(loop, sched)
    _drive(fast, sched)
    _assert_identical(loop, fast)


def test_batched_write_survives_reset_and_slot_reuse(monkeypatch):
    """reset() hands blocks back and the next sequence may land on the
    SAME pool rows at different fills. The flat index is a pure function
    of (slots, fills, T), so no invalidation hook is needed — this pins
    that reasoning."""
    loop, fast = _pair(monkeypatch, batch=2, max_tokens=64)
    first = [(l, [0, 1], 1) for _ in range(20) for l in range(L)]
    _drive(loop, first)
    _drive(fast, first)
    loop.reset(0)
    fast.reset(0)
    second = [(l, [0, 1], 1) for _ in range(10) for l in range(L)]
    _drive(loop, second)
    _drive(fast, second)
    assert torch.equal(loop.seq_lens, fast.seq_lens)
    assert loop._seen == fast._seen
    for b in range(2):
        for layer in range(L):
            a = loop.reference_kv(layer, b)
            c = fast.reference_kv(layer, b)
            assert torch.equal(a[0], c[0]) and torch.equal(a[1], c[1])


def test_batched_write_t_change_rebuilds_index(monkeypatch):
    """T changing between calls must rebuild the cached index (its length
    is T-dependent); a same-fill coincidence must not advance it."""
    loop, fast = _pair(monkeypatch, batch=2, max_tokens=128)
    sched = ([(l, [0, 1], 2) for l in range(L)]
             + [(l, [0, 1], 1) for _ in range(4) for l in range(L)]
             + [(l, [0, 1], 3) for l in range(L)]
             + [(l, [0, 1], 1) for _ in range(4) for l in range(L)])
    _drive(loop, sched)
    _drive(fast, sched)
    _assert_identical(loop, fast)


def test_batched_write_roundtrip_matches_direct(monkeypatch):
    """The scatter must also hold the file's own oracle: bytes written
    through the batched path dequantize to exactly what quantizing the
    same tokens directly produces."""
    monkeypatch.setenv("E4B_BATCHED_KV_WRITE", "1")
    kv = Fp8PagedKV(L, H, D, batch=2, max_tokens_per_seq=64,
                    k_groups=4, device=DEV)
    ks = {0: [], 1: []}
    vs = {0: [], 1: []}
    for i in range(24):
        g = torch.Generator().manual_seed(3000 + i)
        k = (torch.randn(2, 1, H, D, generator=g) * 1.5).to(DEV)
        v = torch.randn(2, 1, H, D, generator=g).to(DEV)
        for layer in range(L):
            kv.append_many(layer, [0, 1], k, v)
        for b in range(2):
            ks[b].append(k[b, 0])
            vs[b].append(v[b, 0])
    for b in range(2):
        kr, vr = kv.reference_kv(0, b)
        torch.testing.assert_close(
            kr, _direct(torch.stack(ks[b]), 4), rtol=0, atol=0)
        torch.testing.assert_close(
            vr, _direct(torch.stack(vs[b]), 1), rtol=0, atol=0)
