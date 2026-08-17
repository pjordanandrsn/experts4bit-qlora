# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""Paged FP8 attention implementation (Phase 9).

The property that matters is substitutability: with the context bound,
this must compute what stock attention computes for the same tokens —
including ACROSS a chunk boundary, which is the case chunked prefill
introduces and the one a naive implementation gets wrong by letting a
later chunk attend only to itself. Unbound, it must behave exactly like
SDPA so an ordinary forward is unaffected by the registration.
"""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from experts4bit_qlora.engines.paged_attention import (  # noqa: E402
    PagedAttentionContext,
    paged_attention_forward,
    set_context,
)

H_Q, H_KV, D = 8, 2, 64


class _Mod(torch.nn.Module):
    def __init__(self, layer_idx=0):
        super().__init__()
        self.layer_idx = layer_idx
        self.num_key_value_groups = H_Q // H_KV
        self.is_causal = True


def _qkv(B, T, seed=0, device="cpu"):
    g = torch.Generator(device="cpu").manual_seed(seed)
    q = torch.randn(B, H_Q, T, D, generator=g).to(device)
    k = torch.randn(B, H_KV, T, D, generator=g).to(device)
    v = torch.randn(B, H_KV, T, D, generator=g).to(device)
    return q, k, v


def _reference(q, k, v):
    """Stock causal SDPA over the whole sequence."""
    T = q.shape[2]
    mask = torch.ones(T, T, dtype=torch.bool,
                      device=q.device).tril()[None, None]
    o = torch.nn.functional.scaled_dot_product_attention(
        q, k, v, attn_mask=mask, scale=D ** -0.5, enable_gqa=True)
    return o.transpose(1, 2).contiguous()


def test_unbound_context_is_plain_sdpa():
    q, k, v = _qkv(2, 6)
    set_context(None)
    got, _ = paged_attention_forward(_Mod(), q, k, v, None,
                                     scaling=D ** -0.5, is_causal=True)
    assert got.shape == (2, 6, H_Q, D)
    torch.testing.assert_close(got, _reference(q, k, v))


def test_prefill_across_a_chunk_boundary_matches_whole_sequence():
    """The chunked-prefill property. Feeding a prompt in two chunks must
    equal feeding it whole: the second chunk's queries have to see the
    first chunk's keys, which is precisely what staging exists for."""
    q, k, v = _qkv(1, 8, seed=3)
    want = _reference(q, k, v)

    ctx = PagedAttentionContext(kv=None, slots=[0], mode="prefill")
    set_context(ctx)
    try:
        o1, _ = paged_attention_forward(
            _Mod(), q[:, :, :5], k[:, :, :5], v[:, :, :5], None,
            scaling=D ** -0.5)
        o2, _ = paged_attention_forward(
            _Mod(), q[:, :, 5:], k[:, :, 5:], v[:, :, 5:], None,
            scaling=D ** -0.5)
    finally:
        set_context(None)
    got = torch.cat([o1, o2], dim=1)
    torch.testing.assert_close(got, want, rtol=2e-5, atol=2e-5)


def test_staging_is_per_sequence_not_shared():
    """Two slots prefilling in the same step must not see each other's
    keys — the failure mode is silent and produces fluent nonsense."""
    q, k, v = _qkv(2, 4, seed=11)
    ctx = PagedAttentionContext(kv=None, slots=[0, 1], mode="prefill")
    set_context(ctx)
    try:
        got, _ = paged_attention_forward(_Mod(), q, k, v, None,
                                         scaling=D ** -0.5)
    finally:
        set_context(None)
    want = _reference(q, k, v)
    torch.testing.assert_close(got, want, rtol=2e-5, atol=2e-5)
    assert set(ctx.staging) == {(0, 0), (0, 1)}


def test_batch_slot_mismatch_is_refused():
    q, k, v = _qkv(3, 2)
    ctx = PagedAttentionContext(kv=None, slots=[0], mode="prefill")
    set_context(ctx)
    try:
        with pytest.raises(ValueError, match="binds 1 slots"):
            paged_attention_forward(_Mod(), q, k, v, None, scaling=1.0)
    finally:
        set_context(None)


def test_decode_regime_refuses_multi_token_queries():
    q, k, v = _qkv(1, 4)
    ctx = PagedAttentionContext(kv=None, slots=[0], mode="decode")
    set_context(ctx)
    try:
        with pytest.raises(ValueError, match="one query token"):
            paged_attention_forward(_Mod(), q, k, v, None, scaling=1.0)
    finally:
        set_context(None)


def test_flush_and_drop_release_staging():
    ctx = PagedAttentionContext(kv=None, slots=[0], mode="prefill")
    k = torch.randn(3, H_KV, D)
    ctx.stage(0, 0, k, k)
    ctx.stage(1, 0, k, k)
    ctx.stage(0, 1, k, k)
    kk, _ = ctx.flush(0, 0)
    assert kk.shape[0] == 3 and (0, 0) not in ctx.staging
    ctx.drop(1)
    assert set(ctx.staging) == {(1, 0)}
