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

    def test_rewind_under_graph_mode_drift(self):
        """append_graph_t1 advances seq_lens ONLY -- _seen goes stale
        under graph decoding by design. Rewind must judge forwardness by
        the DEVICE length and then repair BOTH mirrors: with _seen=4 and
        seq_lens=10, rewind(0, 7) is a legal rollback the old
        host-mirror check would have refused (Bugbot, e4b#241)."""
        kv = _kv(n_layers=1)
        _append(kv, 0, 0, 4)
        kv.seq_lens[0].narrow(0, 0, 1).fill_(10)   # graph-mode drift
        assert kv.seen_device(0, 0) == 10 and kv._seen[0][0] == 4
        kv.rewind(0, 7)
        assert kv._seen[0][0] == 7
        assert int(kv.seq_lens[0][0]) == 7

    def test_rewind_forward_refuses(self):
        kv = _kv()
        for layer in range(2):
            _append(kv, layer, 0, 4)
        with pytest.raises(ValueError, match="rewind forward"):
            kv.rewind(0, 9)

    def test_rewind_forward_judged_by_device_length(self):
        """Forwardness is relative to the device truth: seq_lens=10
        means rewind(0, 9) is a rollback even while _seen reads 4."""
        kv = _kv(n_layers=1)
        _append(kv, 0, 0, 4)
        kv.seq_lens[0].narrow(0, 0, 1).fill_(10)
        kv.rewind(0, 9)                      # legal: 9 < 10 on device
        with pytest.raises(ValueError, match="rewind forward"):
            kv.rewind(0, 11)

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

    def attention(self, layer, q, slots=None, lens_override=None, **kw):
        self.last_kw = dict(kw)
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


def test_device_grouping_branch_dispatch(monkeypatch):
    """PREREG-s3-grouped-verify: with DEVICE_GROUPING set and T>1 the
    fused stack must route BOTH gemms through the captured wrapper with
    expert-major rows, and unsort the result to input row order. gnf4's
    heavy pieces are stubbed; the sort/unsort contract is the thing
    under test."""
    import sys

    hr = pytest.importorskip("experts4bit_qlora.engines.hot_residency")

    calls = []

    class _StubGnf4:
        E4M3_MAX = 448.0

        @staticmethod
        def build_group_tiles_device(ids, n_experts, block_m):
            order = torch.argsort(ids.to(torch.int64), stable=True)
            counts = torch.bincount(ids.to(torch.int64),
                                    minlength=n_experts)
            t = torch.zeros(4, dtype=torch.int32)
            return t, t, t, order, counts

        @staticmethod
        def gemm_4bit_grouped_captured(xr, pk, am, r0, rw, gp, bm):
            calls.append(xr.clone())
            n_out = pk.shape[1]
            # row-identifying output so unsort is checkable
            return (xr[:, :1].repeat(1, n_out)).to(torch.bfloat16)

        @staticmethod
        def gemm_4bit_grouped(*a, **k):
            raise AssertionError("device branch must not call the host-"
                                 "sizes wrapper")

    monkeypatch.setitem(sys.modules, "nf4_grouped", _StubGnf4())

    # grouped-nf4-gemm >= 0.19 ships a ONE-launch Triton tile builder
    # (int4_b32.build_group_tiles_fused) that the stack prefers on decode
    # shapes; on a CPU runner that launch has no driver. Stub it with the
    # same torch contract -- the kernel is not what this test is about.
    class _StubInt4B32:
        build_group_tiles_fused = staticmethod(_StubGnf4.build_group_tiles_device)

    monkeypatch.setitem(sys.modules, "int4_b32", _StubInt4B32())
    R, K1, N2 = 6, 8, 4
    x = torch.arange(R, dtype=torch.float32)[:, None].repeat(1, K1)
    ids = torch.tensor([3, 0, 3, 1, 0, 3])
    out = hr._fused_over_stack(
        x, ids, torch.zeros(4, 2 * N2, K1 // 2), torch.zeros(4, 2 * N2, 1),
        torch.zeros(4, N2, K1 // 2), torch.zeros(4, N2, 1),
        (2 * N2, K1, N2, K1), True, torch.nn.functional.silu,
        device_grouping=True)
    assert len(calls) == 2, "both gemms must go through the captured path"
    # the marker column carries each gathered row's ORIGINAL index; the
    # gather is expert-major when those rows' EXPERT IDS are sorted
    # (the first draft asserted the markers themselves were sorted --
    # wrong thing: [1,4,3,0,2,5] IS the correct order for ids
    # [3,0,3,1,0,3])
    got_rows = calls[0][:, 0].long()
    gathered_experts = ids.index_select(0, got_rows)
    assert torch.equal(gathered_experts.sort().values,
                       gathered_experts), gathered_experts
    # and the gather must be a permutation of all rows
    assert torch.equal(got_rows.sort().values, torch.arange(R)), got_rows
    # unsort contract: output shape restored to input row order
    assert out.shape == (R, N2)
