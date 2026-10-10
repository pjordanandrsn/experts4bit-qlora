# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""Speculative decode's loop (lane SD2, ``bench/sd2/PREREG-sd2.md`` B1, B2, B5; the PREREG's CPU test list).

:class:`SpecDecoder` is driven here by a toy target whose arithmetic does not depend on the row count, so speculation
must emit exactly the stream plain greedy decode emits, for k = 1..3 and drafts right, wrong and mixed. The rest:
- the budgets and the slot's capacity shrink k, and a request with no room drops to T == 1;
- the lengths each step writes, the rollback last;
- a speculating request that becomes batched mid-flight keeps the T == 1 stream (the maintainer's requirement);
- the runner's dispatch, with fakes: it speculates only for a request decoding alone.
The real verify, a decode bucket on alias slots, runs on the card (``tests/test_spec_decode_gpu.py``).
"""
import types

import pytest
import torch

from experts4bit_qlora.engines.paged_runner import PagedModelRunner
from experts4bit_qlora.engines.spec_decode import SpecDecoder

V = 97


def f(ctx):
    """The toy target's greedy next token after ``ctx``."""
    return (sum(ctx[-3:]) * 7 + 3 * len(ctx) + ctx[-1] * ctx[-1]) % V


def greedy(prompt, n):
    out = list(prompt)
    for _ in range(n):
        out.append(f(out))
    return out[len(prompt):]


class Aux:
    def __init__(self, k):
        self.dec = torch.zeros(k + 1, 4)
        self.pre = torch.zeros(64, 4)
        self.mode = "decode"


class Drafter:
    """Proposes the true continuation from the oracle stream, wrong from position ``right`` on (a cycle of patterns)."""

    def __init__(self, k, stream, pattern):
        self.k, self.stream, self.pattern, self.calls = k, stream, list(pattern), 0

    def _drafts(self, pos):                                   # pos: the position of the last token the target emitted
        p = self.pattern[self.calls % len(self.pattern)]
        self.calls += 1
        d = [self.stream[pos + 1 + i] if pos + 1 + i < len(self.stream) else 0 for i in range(self.k)]
        # an int r: right before r, wrong from r on; ("hole", j): wrong at j only, so later drafts match again and
        # only the LEADING matches may be accepted
        wrong = [j for j in range(self.k) if (j == p[1] if isinstance(p, tuple) else j >= p)]
        for i in wrong:
            d[i] = (d[i] + 1) % V
        return torch.tensor(d)

    def prefill(self, tok, aux):
        return self._drafts(len(tok))                          # tok = x[1..P]; x[P] sits at position P

    def extend_and_draft(self, g, aux_v, base, a):
        return self._drafts(int(base) + int(a) + 1)


class KV:
    def __init__(self):
        self.lens, self.notes = [], []

    def set_len_device(self, slot, n):
        self.lens.append((slot, int(n.reshape(()))))

    def note_len(self, slot, n):
        self.notes.append((slot, n))


def _spec(k, stream, pattern, capacity=10_000):
    kv = KV()
    dec = SpecDecoder(drafter=Drafter(k, stream, pattern), aux=Aux(k), kv=kv, k=k, capacity=capacity, device="cpu")
    seen = []

    def verify(slot, n, ids, pos):
        """Row i's argmax given the stream through position pos[0] - 1, then ids[0..i]: the toy target at T == 1."""
        p0 = int(pos[0])
        ctx = list(stream[:p0])
        rows = []
        for i in range(n):
            ctx.append(int(ids[i]))
            rows.append(f(ctx))
        seen.append((n, p0, [int(t) for t in ids]))
        return torch.tensor(rows)

    dec.verify = verify
    return dec, kv, seen


@pytest.mark.parametrize("k", [1, 2, 3])
@pytest.mark.parametrize("pattern", [[0], [99], [0, 1, 2, 3, 99], [2, 0, 99, 1], [("hole", 0), ("hole", 1), 99, ("hole", 2)]],
                         ids=["wrong", "right", "mixed", "mixed2", "holes"])
def test_speculation_emits_the_greedy_stream(k, pattern):
    prompt = [5, 11, 2, 40]
    n_new = 41
    ref = greedy(prompt, n_new)
    stream = prompt + ref                                      # the oracle stream, positions 0..
    dec, kv, _ = _spec(k, stream, pattern)
    tokens = prompt + [ref[0]]                                 # prefill emitted the first token
    dec.start(0, 3, tokens)
    out = [ref[0]]
    plain = 0
    while len(out) < n_new:
        got = dec.step(0, n_new - len(out)) if dec.eligible(0) else None
        if got is None:                                        # no room: T == 1 from here (the runner's plain path)
            out.append(f(prompt + out))
            plain += 1
            continue
        out.extend(got)
    assert out == ref
    c = dec.census()
    assert c["emitted"] + plain == len(out) - 1                # every token after the first, speculative or plain
    assert c["emitted"] == c["accepted"] + c["steps"]          # each step: its accepted drafts plus the target's token
    assert c["accepted"] <= c["drafted"] and c["steps"] > 0
    if pattern == [99]:                                        # always right: every step emits k + 1 until the end
        assert c["accepted"] == c["drafted"]


@pytest.mark.parametrize("k", [2, 3])
def test_only_the_leading_matches_are_accepted(k):
    """A first draft that is wrong, then drafts that are the target's own continuation of that wrong prefix: they
    match the verify's later rows, and accepting any of them would emit tokens conditioned on a token never emitted."""
    prompt = [8, 6, 7, 5]
    ref = greedy(prompt, 20)
    stream = prompt + ref

    class SelfConsistentWrong(Drafter):
        def _drafts(self, pos):
            ctx = list(stream[:pos + 1])
            d = [(stream[pos + 1] + 1) % V]                    # wrong
            for _ in range(1, self.k):
                d.append(f(ctx + d))                           # what the target says after the wrong prefix
            return torch.tensor(d)

    dec, _, _ = _spec(k, stream, [0])
    dec.drafter = SelfConsistentWrong(k, stream, [0])
    dec.start(0, 0, prompt + [ref[0]])
    out = [ref[0]]
    for _ in range(6):
        got = dec.step(0, 100)
        assert len(got) == 1                                   # a = 0: only the target's own token
        out.extend(got)
    assert out == ref[:len(out)]


def test_the_lengths_written_are_base_plus_accepted_plus_one_and_the_mirror_follows():
    prompt = [1, 2, 3]
    ref = greedy(prompt, 30)
    dec, kv, seen = _spec(3, prompt + ref, [0, 3, 1, 2])
    dec.start(0, 7, prompt + [ref[0]])
    base = len(prompt)
    for _ in range(4):
        got = dec.step(0, 100)
        (slot, length), note = kv.lens[-1], kv.notes[-1]
        assert slot == 7 and length == base + len(got) and note == (7, length)
        base = length
    # every verify ran k + 1 rows from the last emitted token at its position
    assert all(n == 4 for n, _, _ in seen)
    assert [p0 for _, p0, _ in seen] == [3] + [length for (_, length) in kv.lens[:-1]]


def test_the_budget_shrinks_k_and_never_runs_past_the_length():
    prompt = [9, 9]
    ref = greedy(prompt, 10)
    dec, kv, seen = _spec(3, prompt + ref, [99])               # always right
    dec.start(0, 0, prompt + [ref[0]])
    assert dec.step(0, 3) == ref[1:4]                          # budget 3: k shrinks to 2, emits exactly 3
    assert seen[-1][0] == 3
    assert dec.step(0, 1) is None                              # budget 1: no room for a draft
    assert not dec.eligible(0) and dec.census()["dropped_end"] == 1


def test_the_slots_capacity_shrinks_k():
    prompt = [4, 4, 4]
    ref = greedy(prompt, 10)
    dec, kv, seen = _spec(3, prompt + ref, [99], capacity=6)
    dec.start(0, 0, prompt + [ref[0]])                         # base 3: positions 3..5 fit, 6 does not
    got = dec.step(0, 100)
    assert seen[-1][0] == 3 and got == ref[1:4]
    assert dec.k_eff(6, 100) < 1


def test_a_request_batched_mid_flight_keeps_the_greedy_stream():
    """The maintainer's requirement: a speculating request becomes batched. Its draft state is dropped, the census
    counts it, and its T == 1 stream after the transition is the stream a never-speculating run emits."""
    prompt = [3, 1, 4, 1, 5]
    ref = greedy(prompt, 30)
    dec, kv, _ = _spec(2, prompt + ref, [1, 2, 0])
    dec.start(0, 2, prompt + [ref[0]])
    out = [ref[0]]
    for _ in range(3):
        out.extend(dec.step(0, 100))
    dec.drop(0, "batched")                                     # a second request was admitted
    assert not dec.eligible(0) and dec.census()["dropped_batched"] == 1
    while len(out) < 30:                                       # T == 1 from the state the speculation left
        out.append(f(prompt + out))
    assert out == ref


def test_start_needs_a_prompt_and_k_must_be_positive():
    dec, _, _ = _spec(1, [1, 2, 3], [0])
    with pytest.raises(ValueError, match="at least one token"):
        dec.start(0, 0, [5])
    with pytest.raises(ValueError, match="k must be"):
        SpecDecoder(drafter=None, aux=None, kv=None, k=0, capacity=8, device="cpu")


# ------------------------------------------------------------------------------------- the runner's dispatch --

class FakeSpec:
    def __init__(self, k=2, steps=None):
        self.k, self.aux = k, types.SimpleNamespace(mode=None)
        self.state = {}
        self.steps = list(steps or [])
        self.dropped = []
        self.started = []

    def eligible(self, rid):
        return rid in self.state

    def drop(self, rid, why):
        if self.state.pop(rid, None) is not None:
            self.dropped.append((rid, why))

    def step(self, rid, budget):
        return self.steps.pop(0)

    def start(self, rid, slot, tokens):
        self.state[rid] = {"slot": slot}
        self.started.append(rid)


def _runner(n_alias=3):
    kv = types.SimpleNamespace(L=1, B=4, scratch=[4, 5, 6, 7], n_alias=n_alias)
    r = PagedModelRunner(torch.nn.Linear(1, 1), kv, device="cpu")
    plain = []

    def bucketed(rids):
        plain.append(list(rids))
        out = {}
        for rid in rids:
            out[rid] = 50 + len(r.tokens[rid])
            r.tokens[rid].append(out[rid])
            r.pos_of[rid] += 1
        return out

    r._run_decode_bucketed = bucketed
    return r, plain


def _bind(r, rid, slot, prompt):
    r.slot_of[rid], r.pos_of[rid], r.tokens[rid] = slot, len(prompt) + 1, list(prompt) + [7]


def test_enable_speculation_comes_before_the_graphs_and_needs_alias_slots():
    r, _ = _runner(n_alias=1)
    with pytest.raises(ValueError, match="alias slots"):
        r.enable_speculation(FakeSpec(k=2))
    r, _ = _runner(n_alias=3)
    spec = FakeSpec(k=3)
    assert r.enable_speculation(spec) == 3
    assert r.speculative and r._verify_buckets == (2, 3, 4) and spec.aux.mode == "decode"
    assert spec.verify == r._spec_verify
    r2, _ = _runner()
    r2._graphs = {}
    with pytest.raises(RuntimeError, match="before the decode and prefill graphs"):
        r2.enable_speculation(FakeSpec())


def test_a_request_alone_speculates_and_its_tokens_and_positions_advance():
    r, plain = _runner()
    spec = FakeSpec(steps=[[11, 12, 13]])
    r.enable_speculation(spec)
    r._graphs = {}
    _bind(r, 0, 0, [1, 2])
    spec.state[0] = {}
    r.decode_budgets({0: 9})
    assert r.run_decode([0]) == {0: [11, 12, 13]}
    assert r.tokens[0][-3:] == [11, 12, 13] and r.pos_of[0] == len(r.tokens[0]) and not plain


def test_a_second_bound_request_drops_the_draft_and_decodes_at_t1():
    r, plain = _runner()
    spec = FakeSpec(steps=[[11, 12]])
    r.enable_speculation(spec)
    r._graphs = {}
    _bind(r, 0, 0, [1, 2])
    spec.state[0] = {}
    _bind(r, 1, 1, [3])                                        # admitted: request 0 is not alone any more
    got = r.run_decode([0])
    assert plain == [[0]] and got == {0: 50 + 3} and spec.dropped == [(0, "batched")]
    assert spec.steps == [[11, 12]]                            # it never speculated
    r.run_decode([0, 1])                                       # batched from here
    assert plain[-1] == [0, 1] and r.pos_of[0] == len(r.tokens[0])


def test_no_room_falls_back_to_the_plain_step():
    r, plain = _runner()
    spec = FakeSpec(steps=[None])
    r.enable_speculation(spec)
    r._graphs = {}
    _bind(r, 0, 0, [1, 2])
    spec.state[0] = {}
    assert r.run_decode([0]) == {0: 53} and plain == [[0]]


def test_speculation_needs_the_decode_graphs():
    r, _ = _runner()
    spec = FakeSpec(steps=[[1]])
    r.enable_speculation(spec)
    _bind(r, 0, 0, [1])
    spec.state[0] = {}
    with pytest.raises(RuntimeError, match="decode graphs"):
        r.run_decode([0])


def test_freeing_a_slot_drops_its_draft():
    r, _ = _runner()
    spec = FakeSpec()
    r.enable_speculation(spec)
    _bind(r, 0, 0, [1])
    spec.state[0] = {}
    r.kv._seen = None
    r._reset = lambda slot: None
    r.free_slot(0)
    assert spec.dropped == [(0, "freed")]


def test_the_verify_buckets_are_captured_but_plain_steps_never_pick_one():
    """The maintainer's aliasing condition 4: bucket 3 exists for k = 2's verify only; plain batched steps keep
    bucket_for over the decode buckets (1, 2, 4, ...), so ON2's batched path is the shipped default's."""
    from experts4bit_qlora.engines.fp8_paged_kv import Fp8PagedKV
    from experts4bit_qlora.engines.paged_runner import bucket_for

    kv = Fp8PagedKV(2, 2, 32, batch=2, max_tokens_per_seq=32, device="cpu", scratch_slots=4, alias_slots=2)
    r = PagedModelRunner(torch.nn.Linear(1, 1), kv, device="cpu")
    r.enable_speculation(FakeSpec(k=2))
    r.enable_decode_graphs((1, 2, 4), capture=False, verbose=False)
    assert r._buckets == (1, 2, 4) and sorted(r._bufs) == [1, 2, 3, 4] and 3 in r.graph_stats
    assert bucket_for(3, r._buckets) == 4
