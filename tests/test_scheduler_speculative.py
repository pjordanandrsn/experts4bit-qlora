# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""The scheduler's side of speculative decoding (lane SD2, ``bench/sd2/PREREG-sd2.md`` B5).

A speculative runner returns several tokens a step for a sequence. The scheduler emits them in order and stops at the
token that finishes the sequence: a stop id once ``min_tokens`` are out, or its length. Later tokens are dropped and
counted. Before each decode it tells the runner how many tokens each sequence may still emit, so a verify never runs
past a request's length. A plain runner is unchanged: these tests drive fakes, as ``test_scheduler.py`` does.
"""

import pytest

from experts4bit_qlora.engines.scheduler import ContinuousScheduler, Phase


class SpecRunner:
    """Emits ``per_step`` tokens a decode step (fewer when the runner is told less is left, as a real one is):
    1000 for the first token, then 2000 + n for the n-th decoded token of each sequence."""

    speculative = True

    def __init__(self, per_step=3, honour_budget=True):
        self.per_step = per_step
        self.honour_budget = honour_budget
        self.budgets: list[dict] = []
        self.counts: dict[int, int] = {}
        self.freed: list[int] = []

    def run_prefill(self, chunks):
        return {rid: 1000 for rid, _, _ in chunks}

    def decode_budgets(self, left):
        self.budgets.append(dict(left))

    def run_decode(self, rids):
        out = {}
        for rid in rids:
            n = self.per_step
            if self.honour_budget:
                n = min(n, self.budgets[-1][rid])
            toks = []
            for _ in range(n):
                self.counts[rid] = self.counts.get(rid, 0) + 1
                toks.append(2000 + self.counts[rid])
            out[rid] = toks
        return out

    def free_slot(self, rid):
        self.freed.append(rid)


def _run(runner, **add):
    s = ContinuousScheduler(runner=runner, max_seqs=2, chunk_tokens=8)
    rid = s.add_request([1, 2, 3], **add)
    s.run_until_idle()
    (req,) = s.done
    assert req.rid == rid
    return s, req


def test_a_speculative_step_emits_its_tokens_in_order():
    s, req = _run(SpecRunner(per_step=3), max_new_tokens=7)
    assert req.out == [1000, 2001, 2002, 2003, 2004, 2005, 2006]
    assert req.finish_reason == "length" and s.tokens_emitted == 7


def test_the_runner_is_told_what_is_left_before_each_decode():
    r = SpecRunner(per_step=3)
    _run(r, max_new_tokens=7)
    # one token from prefill, then 6 left: 3 + 3
    assert [b[0] for b in r.budgets] == [6, 3]


def test_tokens_past_the_length_are_dropped_and_counted():
    # a runner that ignores the budget computes past the length; the scheduler keeps the length exact
    s, req = _run(SpecRunner(per_step=4, honour_budget=False), max_new_tokens=6)
    assert req.out == [1000, 2001, 2002, 2003, 2004, 2005]
    assert req.finish_reason == "length"
    assert s.spec_dropped == 3 and s.stats()["spec_dropped"] == 3


def test_a_stop_id_mid_list_ends_the_sequence_there():
    s, req = _run(SpecRunner(per_step=4), max_new_tokens=20, stop_ids=[2002])
    assert req.out == [1000, 2001, 2002] and req.finish_reason == "stop"
    assert s.spec_dropped == 2


def test_a_stop_id_before_min_tokens_does_not_end_the_sequence():
    s, req = _run(SpecRunner(per_step=4), max_new_tokens=6, stop_ids=[2002], min_tokens=4)
    assert req.out == [1000, 2001, 2002, 2003, 2004, 2005] and req.finish_reason == "length"


def test_a_finished_sequence_frees_its_slot_once():
    r = SpecRunner(per_step=4)
    _run(r, max_new_tokens=3, stop_ids=[2001])
    assert r.freed == [0]


def test_an_empty_list_is_refused():
    class Empty(SpecRunner):
        def run_decode(self, rids):
            return {rid: [] for rid in rids}

    s = ContinuousScheduler(runner=Empty(), max_seqs=1, chunk_tokens=8)
    s.add_request([1, 2], max_new_tokens=4)
    s.step()                                  # prefill: the first token
    with pytest.raises(RuntimeError, match="no token"):
        s.step()


def test_a_speculative_runner_needs_decode_budgets():
    class NoBudget:
        speculative = True

    with pytest.raises(ValueError, match="decode_budgets"):
        ContinuousScheduler(runner=NoBudget())


def test_the_lookahead_and_a_speculative_runner_do_not_combine():
    class Both(SpecRunner):
        def issue_decode(self, rids):
            return rids

        def collect_decode(self, handle):
            return {}

    with pytest.raises(ValueError, match="do not combine"):
        ContinuousScheduler(runner=Both(), lookahead=True)


def test_a_plain_runner_reports_no_speculative_stats():
    class Plain:
        def run_prefill(self, chunks):
            return {rid: 7 for rid, _, _ in chunks}

        def run_decode(self, rids):
            return {rid: 8 for rid in rids}

        def free_slot(self, rid):
            pass

    s = ContinuousScheduler(runner=Plain(), max_seqs=1, chunk_tokens=8)
    s.add_request([1], max_new_tokens=3)
    s.run_until_idle()
    assert s.done[0].out == [7, 8, 8] and "spec_dropped" not in s.stats()


def test_two_sequences_take_their_own_lists():
    r = SpecRunner(per_step=2)
    s = ContinuousScheduler(runner=r, max_seqs=2, chunk_tokens=8)
    a = s.add_request([1, 2], max_new_tokens=4)
    b = s.add_request([3, 4], max_new_tokens=2)
    s.run_until_idle()
    out = {q.rid: q.out for q in s.done}
    assert out[a] == [1000, 2001, 2002, 2003] and out[b] == [1000, 2001]
    assert all(q.phase is Phase.DONE for q in s.done)
