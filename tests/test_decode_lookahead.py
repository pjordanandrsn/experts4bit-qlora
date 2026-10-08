# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""The decode lookahead (``E4B_PAGED_DECODE_LOOKAHEAD``, lane P118): each step issues the next decode step before it
reads the previous one back, so the GPU has a step queued while the host emits, retires and plans.

What must hold, and is tested here without the fp8 kernels:

* the scheduler emits the synchronous path's sequences, tokens and finish reasons; it never issues a sequence past
  ``max_new_tokens``; a sequence that stops on a stop id has one more step computed and discarded, and keeps its slot
  until that step is collected; an abort while a step is queued frees the slot once it is collected;
* the runner's ``issue_decode`` / ``collect_decode`` feed a step the inputs ``run_decode`` would (ids, positions, slots,
  padding) through a real :class:`Fp8PagedKV` and the bucket buffers ``enable_decode_graphs`` builds, with a stand-in
  for the captured graph that folds every input into its slot's history: identical streams and identical bucket
  statistics, chained replays included; a host edit of ``tokens`` with nothing queued resyncs, one with a step queued is
  refused, as are freeing a slot or calling ``run_decode`` with a step queued.

On a CUDA device the same runner tests run with pinned staging and events, the stand-in graph made slow so the host runs
ahead of it. The tiny Qwen3 through captured graphs on sm_89+ is ``tests/test_decode_lookahead_gpu.py``.
"""
import collections
import types

import pytest
import torch

from experts4bit_qlora.engines.fp8_paged_kv import Fp8PagedKV
from experts4bit_qlora.engines.paged_runner import PagedModelRunner
from experts4bit_qlora.engines.scheduler import ContinuousScheduler, Phase

V = 97
PROMPTS = [[3, 17, 42, 9, 11, 5, 88, 23, 7], [5, 1, 99, 64, 2], [120, 7, 7, 31, 2, 9, 14, 60, 3, 3, 8, 1],
           [15, 250, 4, 4, 81, 6, 12], [44, 45, 46, 47]]
MAX_NEW = [14, 10, 6, 3, 8]


# ------------------------------------------------------------------ the scheduler, on a fake runner --
class LookRunner:
    """Each token is a function of its sequence's history, so a stream is the same whenever it is computed. Tokens are
    computed at issue (the "device") and handed back at collect; the fake checks the protocol as it goes."""

    def __init__(self):
        self.hist, self.plen = {}, {}
        self.queued = collections.deque()
        self.issued = collections.Counter()
        self.freed = []
        self.depth = []          # steps outstanding at each collect: 2 = a newer step was queued behind it

    @staticmethod
    def _next(h):
        return (sum((i + 1) * t for i, t in enumerate(h)) * 31 + len(h)) % V

    def bind(self, rid, slot, prompt):
        self.hist[rid], self.plen[rid] = list(prompt), len(prompt)

    def run_prefill(self, chunks):
        first = {}
        for rid, start, take in chunks:
            if start + take >= self.plen[rid]:
                first[rid] = self._next(self.hist[rid])
                self.hist[rid].append(first[rid])
        return first

    def run_decode(self, rids):
        assert not self.queued, "run_decode with a lookahead step queued"
        got = {}
        for rid in rids:
            got[rid] = self._next(self.hist[rid])
            self.hist[rid].append(got[rid])
        return got

    def issue_decode(self, rids):
        assert len(self.queued) < 2, "more than two steps queued"
        h = {}
        for rid in rids:
            assert rid in self.hist, f"rid {rid} issued after its slot was freed"
            h[rid] = self._next(self.hist[rid])
            self.hist[rid].append(h[rid])
            self.issued[rid] += 1
        self.queued.append(h)
        return h

    def collect_decode(self, handle):
        assert self.queued and self.queued[0] is handle, "steps collected out of issue order"
        self.depth.append(len(self.queued))
        return dict(self.queued.popleft())

    def free_slot(self, rid):
        assert all(rid not in h for h in self.queued), f"rid {rid}'s slot freed with a step queued"
        self.hist.pop(rid)
        self.freed.append(rid)


def _serve(lookahead, *, max_seqs=2, stop=None, prompts=PROMPTS, max_new=MAX_NEW):
    run = LookRunner()
    s = ContinuousScheduler(runner=run, max_seqs=max_seqs, chunk_tokens=4, lookahead=lookahead)
    rids = [s.add_request(p, m, stop_ids=stop) for p, m in zip(prompts, max_new)]
    s.run_until_idle(max_steps=1000)
    assert not s.active and not s.queue
    done = {r.rid: r for r in s.done}
    return [(done[r].out, done[r].finish_reason) for r in rids], s, run


def test_the_lookahead_emits_the_synchronous_streams():
    sync, s0, _ = _serve(False)
    look, s1, run = _serve(True)
    assert look == sync and [len(o) for o, _ in look] == MAX_NEW
    assert s1._pending is None and not s1._inflight and s1.lookahead_discarded == 0
    # the first token comes from prefill; the lookahead never issues past max_new_tokens
    assert all(run.issued[r] == m - 1 for r, m in enumerate(MAX_NEW)), run.issued
    assert s1.tokens_emitted == s0.tokens_emitted == sum(MAX_NEW)
    assert s1.stats()["lookahead_discarded"] == 0 and "lookahead_discarded" not in s0.stats()
    # it overlapped: most collects had a newer step queued behind them
    assert sum(d == 2 for d in run.depth) >= len(run.depth) // 2 and run.depth[-1] == 1, run.depth


@pytest.mark.parametrize("max_seqs", [8, 2])
def test_a_stop_costs_at_most_one_discarded_step(max_seqs):
    """With every request resident, a sequence that stops has had one step issued beyond its last token. With requests
    waiting for its slot, its step is collected before planning, so the stop is seen before another is issued."""
    sync, _, _ = _serve(False, max_seqs=max_seqs)
    # stop on a token some streams emit before their length: the stream through it is the synchronous one
    stop = {sync[0][0][4], sync[1][0][2]}
    want, _, _ = _serve(False, max_seqs=max_seqs, stop=stop)
    got, s, run = _serve(True, max_seqs=max_seqs, stop=stop)
    assert got == want
    stopped = [i for i, (o, why) in enumerate(want) if why == "stop" and len(o) < MAX_NEW[i]]
    assert stopped, want
    beyond = {i: run.issued[i] - (len(o) - 1) for i, (o, _) in enumerate(want)}   # the first token is prefill's
    assert set(beyond.values()) <= {0, 1} and sum(beyond.values()) == s.lookahead_discarded
    assert all(beyond[i] == 0 for i in range(len(want)) if i not in stopped)
    if max_seqs >= len(PROMPTS):
        assert s.lookahead_discarded == len(stopped)


def test_a_full_house_with_a_queue_still_drains():
    """Every slot held by a stopped sequence waiting for its queued step, and requests still queued: the collect-only
    step must count as work, or run_until_idle would stop with the queue full."""
    first = LookRunner._next(PROMPTS[0])
    prompts = [PROMPTS[0]] * 3
    nxt = LookRunner._next(PROMPTS[0] + [first])
    got, s, _ = _serve(True, max_seqs=1, stop={nxt}, prompts=prompts, max_new=[5, 5, 5])
    assert got == [([first, nxt], "stop")] * 3 and s.lookahead_discarded == 3


def test_an_abort_with_a_step_queued_frees_the_slot_once_it_is_collected():
    run = LookRunner()
    s = ContinuousScheduler(runner=run, max_seqs=2, chunk_tokens=64, lookahead=True)
    a = s.add_request(PROMPTS[0], 20)
    b = s.add_request(PROMPTS[1], 20)
    for _ in range(50):
        s.step()
        q = s.active.get(a)
        if q is not None and q.phase is Phase.DECODE and a in s._pending_rids:
            break
    assert s.abort(a)
    assert a in s.active and run.freed == []             # its queued step still writes its slot
    s.step()
    assert a not in s.active and run.freed == [a] and [r.rid for r in s.aborted] == [a]
    assert s.lookahead_discarded == 1
    s.run_until_idle()
    assert [r.rid for r in s.done] == [b] and len(s.done[0].out) == 20


def test_a_runner_without_the_entry_points_is_refused():
    class Plain:
        def run_prefill(self, chunks):
            return {}

        def run_decode(self, rids):
            return {}

        def free_slot(self, rid):
            pass

    with pytest.raises(ValueError, match="issue_decode"):
        ContinuousScheduler(runner=Plain(), lookahead=True)
    ContinuousScheduler(runner=Plain())


# ------------------------------------------------------------ the runner, on the real KV and buckets --
class StandInDecode:
    """A captured decode graph's contract: reads the bucket's static ids and positions and, through its device selector,
    the slots' KV lengths (pre-step + 1, as attention does under the step selection), folds them into each slot's
    history (the KV the slot attends to), and writes one token per row. ``spin`` makes each replay slow on a GPU, so
    a host that runs ahead would rewrite inputs a queued step had not read yet."""

    def __init__(self, runner, b, hist, spin=0):
        self.buf, self.hist, self.spin = runner._bufs[b], hist, spin

    def replay(self):
        buf = self.buf
        st = buf["st"]
        if self.spin:
            x = torch.ones(512, 512, device=self.hist.device)
            for _ in range(self.spin):
                x = x @ x * 1e-3
            self.hist.add_(x[0, :1].to(torch.long) * 0)
        sl = st["slot_l"]
        lens = st["lens"][0].to(torch.long)
        h = self.hist.index_select(0, sl)
        h = (h * 1_000_003 + buf["ids"].view(-1) * 7_919 + buf["pos"].view(-1) * 104_729 + lens * 13) % 2_147_483_647
        self.hist.index_copy_(0, sl, h)
        buf["tok"].copy_(h % V)


DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


def _runner(device, buckets, monkeypatch, spin=0):
    monkeypatch.setenv("E4B_KV_STEP_SELECT", "1")
    kv = Fp8PagedKV(2, 2, 32, batch=4, max_tokens_per_seq=64, device=device, scratch_slots=max(buckets))
    r = PagedModelRunner(torch.nn.Module(), kv, device=device)
    status = r.enable_decode_graphs(buckets, capture=False, verbose=False)
    assert set(status) == set(buckets)
    hist = torch.zeros(kv.B + kv.n_scratch, dtype=torch.long, device=device)
    for b in buckets:
        r._graphs[b] = StandInDecode(r, b, hist, spin)

    def prefill(chunks):
        """The prompt's K/V into the pool (its length), its history, and its first token."""
        first = {}
        for rid, start, take in chunks:
            slot, n = r.slot_of[rid], len(r.tokens[rid])
            if start + take >= n:
                kv.seq_lens[:, slot] = n
                hist[slot] = sum((i + 1) * t for i, t in enumerate(r.tokens[rid])) % 2_147_483_647
                first[rid] = int(hist[slot]) % V
                r.tokens[rid].append(first[rid])
                r.pos_of[rid] = n + 1
        return first

    r.run_prefill = prefill
    return r, kv


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("buckets", [(1, 2, 4), (1, 2)])      # (1, 2): four rows run as chained replays
def test_the_runner_decodes_the_synchronous_streams(device, buckets, monkeypatch):
    out, stats = {}, {}
    for look in (False, True):
        r, _ = _runner(device, buckets, monkeypatch, spin=(40 if device == "cuda" else 0))
        s = ContinuousScheduler(runner=r, max_seqs=4, kv_slots=4, chunk_tokens=64, lookahead=look)
        rids = [s.add_request(p, m) for p, m in zip(PROMPTS, MAX_NEW)]
        s.run_until_idle(max_steps=500)
        done = {q.rid: q.out for q in s.done}
        out[look] = [done[i] for i in rids]
        stats[look] = r.graph_stats
        if look:
            assert not r._la["inflight"] and not any(e["open"] for e in r._la["ring"])
    assert out[True] == out[False] and [len(o) for o in out[True]] == MAX_NEW
    assert stats[True] == stats[False]                 # the same rows in the same buckets, step for step
    assert stats[True][max(buckets)]["pad_rows"] >= 0 and sum(v["replays"] for v in stats[True].values()) > 0


def _bound(device, monkeypatch, n=2):
    r, kv = _runner(device, (1, 2, 4), monkeypatch)
    for rid in range(n):
        r.bind(rid, rid, PROMPTS[rid])
        r.run_prefill([(rid, 0, len(PROMPTS[rid]))])
    return r


@pytest.mark.parametrize("device", DEVICES)
def test_a_host_edit_resyncs_when_nothing_is_queued(device, monkeypatch):
    toks = {}
    for look in (False, True):
        r = _bound(device, monkeypatch)
        step = (lambda rids: r.collect_decode(r.issue_decode(rids))) if look else r.run_decode
        step([0, 1])
        r.tokens[1].append(5)                           # a host edit between steps (its position advances with it)
        r.pos_of[1] += 1
        step([0, 1])
        toks[look] = (list(r.tokens[0]), list(r.tokens[1]))
    assert toks[True] == toks[False]


@pytest.mark.parametrize("device", DEVICES)
def test_misuse_with_a_step_queued_is_refused(device, monkeypatch):
    r = _bound(device, monkeypatch)
    h1 = r.issue_decode([0, 1])
    h2 = r.issue_decode([0, 1])
    with pytest.raises(RuntimeError, match="already queued"):
        r.issue_decode([0])
    with pytest.raises(RuntimeError, match="older queued step"):
        r.collect_decode(h2)
    with pytest.raises(RuntimeError, match="queued"):
        r.run_decode([0])
    with pytest.raises(RuntimeError, match="in flight"):
        r.free_slot(0)
    r.tokens[0].append(9)                               # edited while its steps are queued
    a = r.collect_decode(h1)
    with pytest.raises(RuntimeError, match="already collected"):
        r.collect_decode(h1)
    with pytest.raises(RuntimeError, match="device token is not"):
        r.issue_decode([0])
    b = r.collect_decode(h2)
    assert set(a) == set(b) == {0, 1}
    r.free_slot(1)
    with pytest.raises(RuntimeError, match="slot controller"):
        r.slot_controller = types.SimpleNamespace(on_decode_step=lambda: None)
        r.issue_decode([0])


def test_the_lookahead_needs_decode_graphs():
    kv = Fp8PagedKV(2, 2, 32, batch=2, max_tokens_per_seq=64, device="cpu", scratch_slots=0)
    r = PagedModelRunner(torch.nn.Module(), kv, device="cpu")
    r.bind(0, 0, [1, 2])
    with pytest.raises(RuntimeError, match="decode graphs"):
        r.issue_decode([0])


# ------------------------------------------------------------------------------------ the server switch --
def test_the_switch_parses_refuses_and_needs_decode_graphs(monkeypatch):
    from experts4bit_qlora import serve_paged
    from experts4bit_qlora.serve_paged import PagedServeConfig, _lookahead_env

    assert _lookahead_env(None) is False and _lookahead_env("") is False and _lookahead_env(" 0 ") is False
    assert _lookahead_env("1") is True
    for bad in ("2", "on", "auto", "true"):
        with pytest.raises(ValueError, match="E4B_PAGED_DECODE_LOOKAHEAD"):
            _lookahead_env(bad)
    monkeypatch.setattr(serve_paged, "_capability", lambda device: None)      # host-independent: no GPU facts
    monkeypatch.setenv("E4B_PAGED_MODEL", "Qwen/Qwen3-30B-A3B")
    monkeypatch.delenv("E4B_PAGED_DECODE_LOOKAHEAD", raising=False)
    monkeypatch.setenv("E4B_PAGED_GRAPHS", "1")
    assert PagedServeConfig.from_env().decode_lookahead is False             # off unless set
    monkeypatch.setenv("E4B_PAGED_DECODE_LOOKAHEAD", "1")
    assert PagedServeConfig.from_env().decode_lookahead is True
    monkeypatch.setenv("E4B_PAGED_GRAPHS", "0")
    with pytest.raises(ValueError, match="needs bucketed decode graphs"):
        PagedServeConfig.from_env()


def test_the_server_answers_the_same_with_the_lookahead():
    """One completion that stops on EOS before max_tokens, through the HTTP layer, both ways: the same text, finish
    reason and usage; the lookahead computed one more decode row for it, discarded."""
    from fastapi.testclient import TestClient

    import test_serve_paged as tsp
    from experts4bit_qlora.serve_paged import EngineParts, PagedEngine, PagedServeConfig, create_app

    class LookScripted(tsp.ScriptedRunner):
        def issue_decode(self, rids):
            return self.run_decode(rids)

        def collect_decode(self, handle):
            return dict(handle)

    body = {"model": "tiny/moe", "prompt": tsp.PROMPT, "max_tokens": 8}
    got = {}
    for look in (False, True):
        runner = LookScripted(scripts={tsp.PROMPT_IDS: [tsp.HELLO, tsp.WORLD, tsp.USER, tsp.EOS]})
        cfg = PagedServeConfig(model="tiny/moe", max_seqs=4, max_tokens_per_seq=64, chunk_tokens=8,
                               graphs=look, decode_lookahead=look)
        sched = ContinuousScheduler(runner=runner, max_seqs=4, kv_slots=4, chunk_tokens=8,
                                    max_prefill_tokens_per_step=8, lookahead=look)
        parts = EngineParts(scheduler=sched, tokenizer=tsp.TinyTokenizer(), eos_ids=frozenset({tsp.EOS}),
                            info={"model_id": "tiny/moe", "moe_layers": 1}, runner=runner)
        with TestClient(create_app(cfg, engine=PagedEngine(cfg, parts))) as c:
            r = c.post("/v1/completions", json=body)
            assert r.status_code == 200, r.text
            h = c.get("/health").json()
        j = r.json()
        got[look] = (j["choices"][0]["text"], j["choices"][0]["finish_reason"], j["usage"])
        assert h["engine"]["decode_lookahead"] is look
        rows = sum(len(b) for b in runner.decode_batches)
        assert rows == (4 if look else 3) and runner.freed == [0]
        assert sched.stats().get("lookahead_discarded", 0) == (1 if look else 0)
    assert got[True] == got[False] and got[False][1] == "stop" and got[False][2]["completion_tokens"] == 4
