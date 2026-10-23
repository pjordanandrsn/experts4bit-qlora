# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""Admission by reserved KV blocks (serving roadmap lever G2, ``bench/kv-pool-g/DESIGN.md``).

What is pinned: with ``kv_blocks=None`` (the default) admission is exactly as before, and a pool too large to bind makes
the same runner calls as no pool; a binding pool makes the queue head wait for blocks, in FIFO order (nothing behind a
waiting head is admitted first); a request that could never fit is refused when it is added; a reservation is
``ceil((prompt + max_new) / block_tokens)`` and is released when the slot is freed, by a finish or an abort; under the
decode lookahead a binding pool still yields the synchronous path's streams.
"""
import collections

import pytest

from experts4bit_qlora.engines.scheduler import ContinuousScheduler


class Runner:
    """Each token is a function of its sequence's history (the same whenever it is computed); records every call."""

    def __init__(self):
        self.hist, self.plen, self.calls = {}, {}, []
        self.queued = collections.deque()

    @staticmethod
    def _next(h):
        return (sum((i + 1) * t for i, t in enumerate(h)) * 31 + len(h)) % 97

    def bind(self, rid, slot, prompt):
        self.hist[rid], self.plen[rid] = list(prompt), len(prompt)
        self.calls.append(("bind", rid, slot))

    def run_prefill(self, chunks):
        self.calls.append(("prefill", list(chunks)))
        first = {}
        for rid, start, take in chunks:
            if start + take >= self.plen[rid]:
                first[rid] = self._next(self.hist[rid])
                self.hist[rid].append(first[rid])
        return first

    def run_decode(self, rids):
        self.calls.append(("decode", list(rids)))
        got = {}
        for rid in rids:
            got[rid] = self._next(self.hist[rid])
            self.hist[rid].append(got[rid])
        return got

    def issue_decode(self, rids):
        self.calls.append(("issue", list(rids)))
        h = {}
        for rid in rids:
            h[rid] = self._next(self.hist[rid])
            self.hist[rid].append(h[rid])
        self.queued.append(h)
        return h

    def collect_decode(self, handle):
        assert self.queued and self.queued[0] is handle
        return dict(self.queued.popleft())

    def free_slot(self, rid):
        self.calls.append(("free", rid))
        self.hist.pop(rid)


PROMPTS = [list(range(1, 10)), list(range(1, 6)), list(range(1, 13)), list(range(1, 8)), list(range(1, 5))]
MAX_NEW = [14, 10, 6, 3, 8]


def _serve(**kw):
    r = Runner()
    kw.setdefault("max_seqs", 4)
    kw.setdefault("chunk_tokens", 4)
    s = ContinuousScheduler(runner=r, clock=lambda: 0.0, **kw)
    rids = [s.add_request(p, m) for p, m in zip(PROMPTS, MAX_NEW)]
    s.run_until_idle(max_steps=2000)
    assert not s.active and not s.queue
    done = {q.rid: q for q in s.done}
    return [(done[i].out, done[i].finish_reason) for i in rids], s, r


def test_no_pool_and_a_pool_that_never_binds_make_the_same_calls():
    out0, s0, r0 = _serve()
    out1, s1, r1 = _serve(kv_blocks=10 ** 6)
    assert out0 == out1 and r0.calls == r1.calls
    assert "kv_blocks" not in s0.stats()
    st = s1.stats()
    assert st["kv_blocks"] == 10 ** 6 and st["kv_blocks_reserved"] == 0 and st["kv_blocks_waits"] == 0


def test_a_binding_pool_makes_the_head_wait_and_keeps_fifo():
    # block 4: needs are ceil((9+14)/4)=6, ceil((5+10)/4)=4, ceil((12+6)/4)=5, ceil((7+3)/4)=3, ceil((4+8)/4)=3
    r = Runner()
    s = ContinuousScheduler(runner=r, clock=lambda: 0.0, max_seqs=4, chunk_tokens=4, kv_blocks=10, block_tokens=4)
    for p, m in zip(PROMPTS, MAX_NEW):
        s.add_request(p, m)
    s.step()
    assert sorted(s.active) == [0, 1] and s.kv_blocks_reserved == 10      # 6 + 4: request 2 (needs 5) waits
    assert [q.rid for q in s.queue] == [2, 3, 4]                           # 3 and 4 (3 each) would fit 0, but FIFO
    assert s.stats()["kv_blocks_waits"] >= 1
    peak = 0
    order = []
    while s.active or s.queue:
        s.step()
        peak = max(peak, s.kv_blocks_reserved)
        for kind, *rest in r.calls:
            if kind == "bind" and rest[0] not in order:
                order.append(rest[0])
    assert peak <= 10
    assert order == [0, 1, 2, 3, 4]                                         # admitted in arrival order
    assert s.kv_blocks_reserved == 0 and not s._reserved


def test_binding_changes_when_requests_run_not_what_they_emit():
    out_free, _, _ = _serve()
    out_tight, s, _ = _serve(kv_blocks=7, block_tokens=4)
    assert out_tight == out_free
    assert s.kv_blocks_waits > 0


def test_a_request_that_can_never_fit_is_refused_when_added():
    s = ContinuousScheduler(runner=Runner(), max_seqs=2, kv_blocks=4, block_tokens=16)
    s.add_request(list(range(1, 33)), 32)                                   # 64 tokens = 4 blocks: fits exactly
    with pytest.raises(ValueError, match="could never be admitted"):
        s.add_request(list(range(1, 33)), 33)                               # 65 tokens = 5 blocks


def test_abort_releases_an_active_reservation_and_a_queued_abort_holds_none():
    r = Runner()
    s = ContinuousScheduler(runner=r, clock=lambda: 0.0, max_seqs=4, chunk_tokens=8, kv_blocks=6, block_tokens=4)
    a = s.add_request(list(range(1, 9)), 12)       # 5 blocks
    b = s.add_request(list(range(1, 5)), 4)        # 2 blocks: waits behind a
    s.step()
    assert a in s.active and s.kv_blocks_reserved == 5 and [q.rid for q in s.queue] == [b]
    assert s.abort(b) and s.kv_blocks_reserved == 5                       # queued: nothing was reserved
    c = s.add_request(list(range(1, 5)), 4)
    s.step()
    assert c not in s.active
    assert s.abort(a) and s.kv_blocks_reserved == 0
    s.step()
    assert c in s.active and s.kv_blocks_reserved == 2


def test_the_lookahead_under_a_binding_pool_emits_the_synchronous_streams():
    out_sync, _, _ = _serve(kv_blocks=8, block_tokens=4)
    out_look, s, r = _serve(kv_blocks=8, block_tokens=4, lookahead=True)
    assert out_look == out_sync
    assert s._pending is None and not s._inflight and s.kv_blocks_reserved == 0
    assert any(c[0] == "issue" for c in r.calls)


@pytest.mark.parametrize("kw", [{"kv_blocks": 0}, {"kv_blocks": -3}, {"block_tokens": 0}])
def test_nonsense_pools_refuse(kw):
    with pytest.raises(ValueError):
        ContinuousScheduler(runner=Runner(), **kw)


def test_blocks_of_is_the_ceiling_over_prompt_plus_new_tokens():
    s = ContinuousScheduler(runner=Runner(), kv_blocks=100, block_tokens=16)
    s.add_request(list(range(1, 17)), 16)
    s.add_request(list(range(1, 18)), 16)
    q = s.queue
    assert s.blocks_of(q[0]) == 2 and s.blocks_of(q[1]) == 3


def test_every_exit_releases_the_reservation_exactly_once():
    """Finish by length, finish on a stop id, abort mid-prefill, abort mid-decode: each frees its blocks once, so the
    pool never leaks or double-frees over many cycles."""
    r = Runner()
    s = ContinuousScheduler(runner=r, clock=lambda: 0.0, max_seqs=2, chunk_tokens=2, kv_blocks=8, block_tokens=4)
    for cycle in range(20):
        a = s.add_request(list(range(1, 9)), 4)                  # 3 blocks; prefill takes 4 steps at chunk 2
        b = s.add_request(list(range(1, 5)), 6, stop_ids=None)   # 3 blocks
        s.step()
        assert s.kv_blocks_reserved == 6
        if cycle % 3 == 0:
            assert s.abort(a)                                    # mid-prefill
        for _ in range(3):
            s.step()
        if cycle % 3 == 1 and b in s.active:
            assert s.abort(b)                                    # mid-decode
        s.run_until_idle(max_steps=100)
        assert s.kv_blocks_reserved == 0 and not s._reserved, cycle
    stop_tok = Runner._next(list(range(1, 5)))                  # the first token request c will emit
    c = s.add_request(list(range(1, 5)), 8, stop_ids=[stop_tok])
    s.run_until_idle(max_steps=100)
    done = {q.rid: q for q in s.done}
    assert done[c].finish_reason == "stop" and s.kv_blocks_reserved == 0


@pytest.mark.parametrize("lookahead", [False, True])
def test_a_binding_pool_composes_with_the_interleave_knobs(lookahead):
    """G2's reservation and A1's prefill_every / loaded_chunk_tokens act on the same plan: together they change when
    requests run, never what they emit, and every reservation is released at idle."""
    out_free, _, _ = _serve()
    out_both, s, _ = _serve(kv_blocks=8, block_tokens=4, prefill_every=2, loaded_chunk_tokens=2, lookahead=lookahead)
    assert out_both == out_free
    assert s.kv_blocks_waits > 0 and s.prefill_deferred > 0 and s.kv_blocks_reserved == 0
