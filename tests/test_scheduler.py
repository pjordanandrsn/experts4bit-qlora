# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""Continuous-batching scheduler (Phase 9).

These tests drive a deterministic fake runner, which is the point of the
runner seam: scheduling defects are decision defects, and mixing them
with GPU timing is how they hide. What is pinned here is what the gate's
claims rest on — no wait-for-slowest, prompts chunked so decode is not
starved, admission bounded by REAL KV capacity, and TTFT measured from
arrival rather than from admission (the flattering version a loaded
server would otherwise report).
"""

import pytest

from experts4bit_qlora.engines.scheduler import (
    ContinuousScheduler,
    Phase,
)


class FakeRunner:
    """Records every call; emits token = 1000 + len(out) per sequence."""

    def __init__(self):
        self.prefill_calls: list[list[tuple[int, int, int]]] = []
        self.decode_calls: list[list[int]] = []
        self.freed: list[int] = []
        self.counts: dict[int, int] = {}

    def run_prefill(self, chunks):
        self.prefill_calls.append(list(chunks))
        return {rid: 1000 for rid, _, _ in chunks}

    def run_decode(self, rids):
        self.decode_calls.append(list(rids))
        out = {}
        for rid in rids:
            self.counts[rid] = self.counts.get(rid, 0) + 1
            out[rid] = 2000 + self.counts[rid]
        return out

    def free_slot(self, rid):
        self.freed.append(rid)


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        self.t += 1.0
        return self.t


def _sched(**kw):
    r = FakeRunner()
    kw.setdefault("max_seqs", 4)
    kw.setdefault("chunk_tokens", 4)
    return ContinuousScheduler(runner=r, **kw), r


def test_short_and_long_prompts_do_not_wait_for_each_other():
    """The headline claim. A long prompt must not block a short one's
    tokens: with continuous batching the short sequence starts decoding
    while the long one is still ingesting."""
    s, r = _sched(chunk_tokens=4, max_prefill_tokens_per_step=4)
    short = s.add_request([1, 2], max_new_tokens=3)
    long = s.add_request(list(range(40)), max_new_tokens=3)
    for _ in range(6):
        s.step()
    short_req = next(q for q in list(s.active.values()) + s.done
                     if q.rid == short)
    long_req = next(q for q in list(s.active.values()) + s.done
                    if q.rid == long)
    assert short_req.out, "short sequence produced nothing while long ran"
    assert long_req.phase is Phase.PREFILL
    assert not long_req.out
    # and the short one's decode really did ride along in shared steps
    assert any(short in call for call in r.decode_calls)


def test_prompt_is_chunked_not_swallowed_whole():
    s, r = _sched(chunk_tokens=8, max_prefill_tokens_per_step=8)
    s.add_request(list(range(30)), max_new_tokens=1)
    s.step()
    assert r.prefill_calls[0] == [(0, 0, 8)]
    s.step()
    assert r.prefill_calls[1] == [(0, 8, 8)]
    # four chunks of 8 cover 30 tokens: 8, 8, 8, 6
    s.step()
    s.step()
    assert r.prefill_calls[3] == [(0, 24, 6)]


def test_step_prefill_budget_is_respected_across_sequences():
    """The budget is per STEP, not per sequence: two prompts sharing a
    step must not together exceed it, or a 'chunked' prefill silently
    becomes a full one whenever concurrency rises."""
    s, r = _sched(chunk_tokens=8, max_prefill_tokens_per_step=10)
    s.add_request(list(range(20)), max_new_tokens=1)
    s.add_request(list(range(20)), max_new_tokens=1)
    s.step()
    assert sum(n for _, _, n in r.prefill_calls[0]) == 10


def test_admission_is_bounded_by_kv_slots_not_optimism():
    s, r = _sched(max_seqs=8, kv_slots=2, chunk_tokens=4)
    for _ in range(5):
        s.add_request([1, 2], max_new_tokens=2)
    s.step()
    assert len(s.active) == 2, "admitted past KV capacity"
    assert len(s.queue) == 3


def test_slot_is_recycled_after_completion():
    s, r = _sched(max_seqs=8, kv_slots=1, chunk_tokens=4)
    a = s.add_request([1], max_new_tokens=1)
    b = s.add_request([2], max_new_tokens=1)
    s.run_until_idle()
    assert r.freed == [a, b], "slot was not recycled between sequences"
    assert len(s.done) == 2
    assert s.stats()["kv_slots_free"] == 1


def test_ttft_measures_from_arrival_including_queue_wait():
    clock = Clock()
    r = FakeRunner()
    s = ContinuousScheduler(runner=r, max_seqs=1, kv_slots=1,
                            chunk_tokens=4, clock=clock)
    s.add_request([1, 2], max_new_tokens=4, now=0.0)   # arrives at t=0
    s.add_request([3, 4], max_new_tokens=1, now=0.0)   # waits behind it
    s.run_until_idle()
    first, second = s.done[0], s.done[1]
    assert second.queue_wait > 0, "queued request reported no wait"
    assert second.ttft > first.ttft, \
        "TTFT ignored queue wait — a loaded server would look fast while " \
        "callers waited"


def test_finished_sequences_leave_the_batch_immediately():
    """No wait-for-slowest: a sequence that hits its token budget must
    retire in the same step, not at the end of the batch."""
    s, r = _sched(max_seqs=4, chunk_tokens=4)
    s.add_request([1], max_new_tokens=1)
    s.add_request([2], max_new_tokens=5)
    s.step()                       # both prefill; both emit first token
    assert len(s.done) == 1        # the max_new_tokens=1 one is finished
    assert len(s.active) == 1
    assert r.freed == [0]


def test_idle_scheduler_reports_empty_plan_not_error():
    s, r = _sched()
    assert s.step().is_empty
    assert s.stats()["completed"] == 0


def test_stats_pairs_aggregate_with_per_stream_and_latency():
    s, r = _sched(max_seqs=4, chunk_tokens=4)
    for _ in range(3):
        s.add_request([1, 2, 3], max_new_tokens=2)
    s.run_until_idle()
    st = s.stats()
    assert st["completed"] == 3
    assert st["tokens_emitted"] == 6
    assert st["prefill_tokens"] == 9
    # the protocol's requirement: never a throughput number alone
    for key in ("ttft_p50", "ttft_p99", "per_stream_tok_s_mean",
                "queue_wait_p50"):
        assert st[key] is not None, f"{key} missing from stats"


def test_runner_that_forgets_a_first_token_is_an_error_not_a_hang():
    """A runner that completes a prompt but returns no token would leave
    the sequence in prefill forever, looking like a slow model rather
    than a broken contract."""
    class Forgetful(FakeRunner):
        def run_prefill(self, chunks):
            super().run_prefill(chunks)
            return {}

    s = ContinuousScheduler(runner=Forgetful(), max_seqs=2, chunk_tokens=8)
    s.add_request([1, 2], max_new_tokens=1)
    with pytest.raises(RuntimeError, match="without returning its first"):
        s.step()


def test_rejects_nonsense_configuration():
    r = FakeRunner()
    with pytest.raises(ValueError):
        ContinuousScheduler(runner=r, max_seqs=0)
    with pytest.raises(ValueError):
        ContinuousScheduler(runner=r, chunk_tokens=0)
    with pytest.raises(ValueError):
        ContinuousScheduler(runner=r, kv_slots=0)
    s, _ = _sched()
    with pytest.raises(ValueError):
        s.add_request([], max_new_tokens=1)
    with pytest.raises(ValueError):
        s.add_request([1], max_new_tokens=0)
