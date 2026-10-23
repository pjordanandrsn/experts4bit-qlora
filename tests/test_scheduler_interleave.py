# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""The prefill/decode interleave knobs (serving roadmap lever A1): ``prefill_every`` and ``loaded_chunk_tokens``.

What is pinned: the defaults compose exactly the plans the scheduler made before the knobs existed (checked against
that planner, kept here verbatim); ``prefill_every=N`` puts prefill on at most one step in N while a decode row is
live, and never delays a prompt when nothing decodes; ``loaded_chunk_tokens`` caps the chunk AND the step's budget
only while decoding; nonsense values refuse; the server reads both from the environment and reports them.
"""
import pytest

from experts4bit_qlora.engines.scheduler import ContinuousScheduler, Phase, StepPlan


class FakeRunner:
    """Records every call; a finished prompt's first token is 1000, decode tokens count up from 2001."""

    def __init__(self):
        self.calls: list = []
        self.counts: dict = {}

    def run_prefill(self, chunks):
        self.calls.append(("prefill", list(chunks)))
        return {rid: 1000 for rid, _, _ in chunks}

    def run_decode(self, rids):
        self.calls.append(("decode", list(rids)))
        out = {}
        for rid in rids:
            self.counts[rid] = self.counts.get(rid, 0) + 1
            out[rid] = 2000 + self.counts[rid]
        return out

    def free_slot(self, rid):
        self.calls.append(("free", rid))


class PreA1Scheduler(ContinuousScheduler):
    """``plan()`` exactly as it was before the interleave knobs (e4b 7acc6ef7): the reference the defaults must match."""

    def plan(self) -> StepPlan:
        self._admit()
        plan = StepPlan()
        budget = self.max_prefill_tokens
        for req in self.active.values():
            if req.phase is not Phase.PREFILL or budget <= 0:
                continue
            take = min(self.chunk_tokens, budget,
                       req.prompt_len - req.prompt_pos)
            if take > 0:
                plan.prefill.append((req.rid, req.prompt_pos, take))
                budget -= take
        for req in self.active.values():
            if req.phase is Phase.DECODE:
                plan.decode.append(req.rid)
        return plan


# (step at which it arrives, prompt length, max_new_tokens): short and long prompts, arrivals while others decode
WORKLOAD = [(0, 3, 6), (0, 21, 4), (2, 9, 8), (3, 1, 2), (5, 30, 3), (5, 4, 9), (9, 12, 5), (14, 2, 1)]


def _drive(sched, workload=WORKLOAD, steps=80):
    for t in range(steps):
        for at, n, m in workload:
            if at == t:
                sched.add_request(list(range(1, n + 1)), max_new_tokens=m, now=float(t))
        sched.step()
    return sched


def _make(cls=ContinuousScheduler, **kw):
    r = FakeRunner()
    kw.setdefault("max_seqs", 3)
    kw.setdefault("chunk_tokens", 8)
    kw.setdefault("max_prefill_tokens_per_step", 10)
    return cls(runner=r, clock=lambda: 0.0, **kw), r


def test_the_defaults_make_the_pre_a1_plans_call_for_call():
    for kw in ({}, {"prefill_every": 1, "loaded_chunk_tokens": 0}):
        ref_s, ref_r = _make(PreA1Scheduler)
        s, r = _make(**kw)
        _drive(ref_s)
        _drive(s)
        assert r.calls == ref_r.calls and r.calls
        assert [(q.rid, q.out) for q in s.done] == [(q.rid, q.out) for q in ref_s.done]
        assert "prefill_deferred" not in s.stats()


def _decoding(s):
    return any(q.phase is Phase.DECODE for q in s.active.values())


def test_prefill_every_n_spaces_prefill_while_a_decode_row_is_live():
    s, r = _make(prefill_every=3, chunk_tokens=4, max_prefill_tokens_per_step=4)
    s.add_request([1, 2], max_new_tokens=40)        # decodes for the whole test
    s.step()                                         # its prompt (nothing else decodes yet)
    assert r.calls[0] == ("prefill", [(0, 0, 2)])
    s.add_request(list(range(1, 25)), max_new_tokens=1)   # 24 tokens: 6 chunks of 4
    since, gaps, prefill_steps, held = 0, [], 0, 0
    for _ in range(40):
        pending = (any(q.rid == 1 for q in s.queue)   # admitted inside this step's plan, or already prefilling
                   or (1 in s.active and s.active[1].phase is Phase.PREFILL))
        plan = s.step()
        assert plan.decode == [0]                    # the resident decode advanced on EVERY step
        if plan.prefill:
            prefill_steps += 1
            gaps.append(since)
            since = 0
        else:
            since += 1
            held += pending
        if 1 not in s.active:
            break
    assert prefill_steps == 6
    assert all(g == 2 for g in gaps[1:])             # exactly one prefill step in 3 while a chunk is pending
    assert gaps[0] <= 2                              # and the first chunk waited at most N - 1 steps
    assert s.stats()["prefill_deferred"] == held


def test_prefill_every_never_delays_a_prompt_when_nothing_decodes():
    s, r = _make(prefill_every=4, chunk_tokens=4, max_prefill_tokens_per_step=4)
    s.add_request(list(range(1, 17)), max_new_tokens=1)
    for _ in range(4):
        s.step()
    assert [c for c in r.calls if c[0] == "prefill"] == [
        ("prefill", [(0, 0, 4)]), ("prefill", [(0, 4, 4)]), ("prefill", [(0, 8, 4)]), ("prefill", [(0, 12, 4)])]
    assert s.prefill_deferred == 0


def test_an_arrival_after_a_long_decode_run_is_prefilled_at_once():
    s, r = _make(prefill_every=5, chunk_tokens=4, max_prefill_tokens_per_step=4)
    s.add_request([1], max_new_tokens=50)
    for _ in range(10):
        s.step()                                     # decode-only steps saturate the counter
    s.add_request([1, 2, 3], max_new_tokens=1)
    plan = s.step()
    assert plan.prefill == [(1, 0, 3)] and plan.decode == [0]


def test_loaded_chunk_caps_chunk_and_budget_only_while_decoding():
    s, r = _make(loaded_chunk_tokens=2, chunk_tokens=8, max_prefill_tokens_per_step=8)
    s.add_request(list(range(1, 11)), max_new_tokens=30)
    plan = s.step()
    assert plan.prefill == [(0, 0, 8)]               # nothing decodes: the full chunk
    s.step()                                         # its last 2 tokens; it now decodes
    assert _decoding(s)
    s.add_request(list(range(1, 7)), max_new_tokens=1)
    s.add_request(list(range(1, 7)), max_new_tokens=1)
    plan = s.step()
    assert plan.prefill == [(1, 0, 2)]               # chunk AND budget capped: one 2-token chunk, not 2 + 2 + ...
    assert plan.decode == [0]
    plan = s.step()
    assert plan.prefill == [(1, 2, 2)]


def test_loaded_chunk_wider_than_the_chunk_changes_nothing():
    ref_s, ref_r = _make(PreA1Scheduler)
    s, r = _make(loaded_chunk_tokens=64)
    _drive(ref_s)
    _drive(s)
    assert r.calls == ref_r.calls


def test_both_knobs_together_still_finish_every_request_with_its_tokens():
    s, r = _make(prefill_every=2, loaded_chunk_tokens=3)
    pending = list(WORKLOAD)
    for t in range(200):
        for at, n, m in pending:
            if at == t:
                s.add_request(list(range(1, n + 1)), max_new_tokens=m, now=float(t))
        plan = s.step()
        if plan.decode:                              # a decode row was live: chunk and step budget are capped
            assert sum(n for _, _, n in plan.prefill) <= 3, (t, plan)
    assert not s.active and not s.queue
    assert sorted(q.rid for q in s.done) == list(range(len(WORKLOAD)))
    for q in s.done:
        assert q.out == [1000] + [2001 + i for i in range(WORKLOAD[q.rid][2] - 1)]


@pytest.mark.parametrize("kw", [{"prefill_every": 0}, {"prefill_every": -2}, {"loaded_chunk_tokens": -1}])
def test_nonsense_knobs_refuse(kw):
    with pytest.raises(ValueError):
        ContinuousScheduler(runner=FakeRunner(), **kw)


def test_the_server_reads_both_knobs_and_refuses_nonsense(monkeypatch):
    from experts4bit_qlora.serve_paged import PagedServeConfig
    for k in ("E4B_PAGED_PREFILL_EVERY", "E4B_PAGED_LOADED_CHUNK_TOKENS"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("E4B_PAGED_MODEL", "tiny/moe")
    cfg = PagedServeConfig.from_env()
    assert (cfg.prefill_every, cfg.loaded_chunk_tokens) == (1, 0)
    monkeypatch.setenv("E4B_PAGED_PREFILL_EVERY", "4")
    monkeypatch.setenv("E4B_PAGED_LOADED_CHUNK_TOKENS", "128")
    cfg = PagedServeConfig.from_env()
    assert (cfg.prefill_every, cfg.loaded_chunk_tokens) == (4, 128)
    monkeypatch.setenv("E4B_PAGED_PREFILL_EVERY", "0")
    with pytest.raises(ValueError, match="E4B_PAGED_PREFILL_EVERY"):
        PagedServeConfig.from_env().validate()
