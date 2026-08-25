# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""PREREG-s2lite mechanics that run on CPU: rewind semantics,
lens_override validation, and the shim's verify-mode length stagger.

The paged read kernel itself is CUDA; its correctness gate is the
on-box `--s2-verify gate` arm. What CPU can check is everything the
gate would blame on plumbing: length arithmetic, causality-by-lengths,
rewind's refusal to move forward, and shape contracts.
"""

import pytest

torch = pytest.importorskip("torch")
mod = pytest.importorskip("experts4bit_qlora.engines.fp8_paged_kv")
pa = pytest.importorskip("experts4bit_qlora.engines.paged_attention")

_CLS = mod.Fp8PagedKV


def _kv(**kw):
    args = dict(n_layers=2, n_kv_heads=1, head_dim=64, batch=1,
                max_tokens_per_seq=64, device="cpu")
    args.update(kw)
    return _CLS(**args)


def _append(kv, layer, seq, t):
    kv.append(layer, seq, torch.randn(t, 1, 64), torch.randn(t, 1, 64))


class TestRewind:
    def test_rewind_updates_both_mirrors_on_all_layers(self):
        kv = _kv()
        for layer in range(2):
            _append(kv, layer, 0, 10)
        kv.rewind(0, 6)
        for layer in range(2):
            assert kv._seen[layer][0] == 6
            assert int(kv.seq_lens[layer][0]) == 6

    def test_rewind_forward_refuses(self):
        kv = _kv()
        for layer in range(2):
            _append(kv, layer, 0, 4)
        with pytest.raises(ValueError, match="rewind forward"):
            kv.rewind(0, 9)

    def test_rewind_then_append_overwrites_the_tail(self):
        """The whole point: after rewind, the next append lands at the
        rewound position, so rejected draft bytes become unreachable
        AND get overwritten by the next real tokens."""
        kv = _kv(n_layers=1)
        _append(kv, 0, 0, 8)
        kv.rewind(0, 5)
        _append(kv, 0, 0, 2)
        assert kv._seen[0][0] == 7
        assert int(kv.seq_lens[0][0]) == 7


class TestLensOverride:
    def test_shape_mismatch_refuses(self):
        kv = _kv(n_layers=1)
        _append(kv, 0, 0, 4)
        q = torch.randn(3, 1, 64)
        bad = torch.tensor([1, 2], dtype=torch.int32)
        with pytest.raises(ValueError, match="lens_override"):
            kv.attention(0, q, slots=[0, 0, 0], lens_override=bad)


class _RecordingKV:
    """Stands in for Fp8PagedKV under the shim: records what verify
    mode appends and what lengths it reads with."""

    def __init__(self, base):
        self.base = base
        self.seq_lens = torch.zeros(1, 4, dtype=torch.int32)
        self.seq_lens[0, 2] = base
        self.appended = None
        self.read_lens = None
        self.read_slots = None

    def append(self, layer, slot, k, v):
        self.appended = (layer, slot, k.shape[0])
        self.seq_lens[layer, slot] += k.shape[0]

    def attention(self, layer, q, slots=None, lens_override=None):
        self.read_slots = list(slots)
        self.read_lens = lens_override.clone()
        return torch.zeros(q.shape[0], q.shape[1], q.shape[2])


def test_verify_mode_staggers_lengths_causally():
    """Row i must read base + i + 1 tokens: its own draft position and
    everything before it, never a later draft token."""
    base = 37
    kv = _RecordingKV(base)
    ctx = pa.PagedAttentionContext(kv=kv, slots=[2], mode="verify")
    prev = pa.set_context(ctx)
    try:
        T, hq, d = 5, 2, 64
        q = torch.randn(1, hq, T, d)
        k = torch.randn(1, 1, T, d)
        v = torch.randn(1, 1, T, d)

        class _M:
            layer_idx = 0

        out, _ = pa.paged_attention_forward(_M(), q, k, v, None)
    finally:
        pa.set_context(prev)
    assert kv.appended == (0, 2, T)
    assert kv.read_slots == [2] * T
    want = torch.tensor([base + i + 1 for i in range(T)],
                        dtype=torch.int32)
    assert torch.equal(kv.read_lens.to(torch.int32), want), (
        kv.read_lens, want)
    assert out.shape == (1, T, hq, d)


def test_verify_mode_refuses_batch():
    kv = _RecordingKV(1)
    ctx = pa.PagedAttentionContext(kv=kv, slots=[0, 1], mode="verify")
    prev = pa.set_context(ctx)
    try:
        q = torch.randn(2, 1, 3, 64)

        class _M:
            layer_idx = 0

        with pytest.raises(ValueError, match="single-sequence"):
            pa.paged_attention_forward(_M(), q, q, q, None)
    finally:
        pa.set_context(prev)
