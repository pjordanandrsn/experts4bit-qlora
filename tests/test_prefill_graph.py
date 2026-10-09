# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""The first-chunk prefill graph's routing in ``PagedModelRunner.run_prefill`` (``E4B_PAGED_PREFILL_GRAPH``), on the CPU.

A stand-in graph and a stub model, both writing K/V and logits that are functions of the chunk's tokens, so every
staged and flushed value says which tokens produced it. The first chunk of exactly ``T`` tokens must take the
replay; later chunks and other first-chunk lengths run the model, counted by reason. A prompt that continues past a
replayed chunk must keep a COPY of the graph's K/V: the next replay (another request's first chunk) overwrites the
graph's outputs before the prompt's later chunks and flush. The mutation arm stages without the copy and must be
caught by the same comparison. The real graph, capture and startup check run in ``tests/test_prefill_graph_gpu.py``.
"""
import types

import pytest
import torch

from experts4bit_qlora.engines import paged_attention
from experts4bit_qlora.engines.paged_runner import PagedModelRunner, PrefillGraphRefused

T, V = 4, 64


def kv_of(ids):
    """[n] tokens -> the K and V a chunk of them stages: [n, H=1, D=2], distinct per token."""
    f = ids.to(torch.float32)[:, None, None]
    return torch.cat([f, f + 0.5], -1), torch.cat([-f, f * 2], -1)


def logits_of(ids):
    """[n] tokens -> [1, n, V]: argmax of position i is (ids[i] * 7 + 3) % V."""
    out = torch.zeros(1, ids.numel(), V)
    out[0, torch.arange(ids.numel()), (ids * 7 + 3) % V] = 1.0
    return out


class StubModel(torch.nn.Module):
    """Stages its chunk's K/V through the bound paged context, as the attention implementation does."""

    def __init__(self):
        super().__init__()
        self.calls = 0

    def forward(self, input_ids, position_ids=None, use_cache=False):
        self.calls += 1
        ctx = paged_attention._CTX
        ids = input_ids[0]
        k, v = kv_of(ids)
        ctx.stage(0, ctx.slots[0], k, v)
        return types.SimpleNamespace(logits=logits_of(ids))


class StandInGraph:
    """Writes the same functions of its static input into its static outputs on every replay."""

    def __init__(self, ids, logits, k, v):
        self.ids, self.logits, self.k, self.v = ids, logits, k, v

    def replay(self):
        k, v = kv_of(self.ids[0])
        self.k.copy_(k)
        self.v.copy_(v)
        self.logits.copy_(logits_of(self.ids[0]))


class RecordingKV:
    L, B, scratch = 1, 8, []

    def __init__(self):
        self.appended = {}

    def reset(self, slot):
        self.appended.pop(slot, None)

    def append(self, layer, slot, k, v):
        self.appended[slot] = (k.clone(), v.clone())


def _runner():
    r = PagedModelRunner(StubModel(), RecordingKV(), device="cpu")
    ids = torch.zeros(1, T, dtype=torch.long)
    k, v = kv_of(ids[0])
    g = StandInGraph(ids, torch.zeros(1, T, V), k.clone(), v.clone())
    r._prefill_graph = {"T": T, "graph": g, "ids": ids, "logits": g.logits, "staged": {0: (g.k, g.v)}}
    return r


# A (4: one replayed chunk, complete), B (10: a replayed chunk that continues, then two eager later chunks),
# D (4: replayed BETWEEN B's chunks, overwriting the graph's outputs), C (3: a short first chunk)
PROMPTS = {"A": [5, 9, 2, 11], "B": [1, 2, 3, 4, 20, 21, 22, 23, 40, 41], "D": [60, 61, 62, 63], "C": [7, 8, 9]}
SLOTS = {"A": 0, "B": 1, "D": 2, "C": 3}
ORDER = [("A", 0, 4), ("B", 0, 4), ("D", 0, 4), ("B", 4, 4), ("C", 0, 3), ("B", 8, 2)]


def _drive(r):
    rid = {name: i for i, name in enumerate(PROMPTS)}
    for name, prompt in PROMPTS.items():
        r.bind(rid[name], SLOTS[name], prompt)
    first = {}
    for name, start, take in ORDER:
        got = r.run_prefill([(rid[name], start, take)])
        first.update({name: tok for _, tok in got.items()})
    return first


def _expected(prompt):
    ids = torch.tensor(prompt)
    k, v = kv_of(ids)
    return k, v, int((ids[-1] * 7 + 3) % V)


def test_first_chunks_of_T_tokens_replay_and_the_rest_run_eagerly():
    r = _runner()
    first = _drive(r)
    for name, prompt in PROMPTS.items():
        k, v, tok = _expected(prompt)
        got_k, got_v = r.kv.appended[SLOTS[name]]
        assert torch.equal(got_k, k) and torch.equal(got_v, v), name
        assert first[name] == tok, name
    st = r.prefill_graph_stats()
    assert {k: st[k] for k in ("status", "T", "replays", "eager_chunks", "eager_reasons")} == {
        "status": "on", "T": T, "replays": 3, "eager_chunks": 3, "eager_reasons": {"later_chunk": 2, "short_chunk": 1}}
    assert r.model.calls == 3                 # B's two later chunks and C's short first chunk


def test_without_the_copy_a_continuing_prompt_flushes_the_next_replays_kv():
    """The mutation arm: stage the graph's outputs uncopied. D's replay overwrites them before B's flush, so B's pool
    K/V begin with D's tokens. The comparison above must see it."""
    r = _runner()

    def no_copy(self, rid, slot, take, done):
        pg = self._prefill_graph
        pg["ids"].copy_(torch.tensor(self.tokens[rid][:take], dtype=torch.long)[None])
        pg["graph"].replay()
        self.ctx.drop(slot)
        for layer, (k, v) in pg["staged"].items():
            self.ctx.staging[(layer, slot)] = ([k], [v])
        return pg["logits"]

    r._replay_prefill_graph = types.MethodType(no_copy, r)
    _drive(r)
    k, _, _ = _expected(PROMPTS["B"])
    got_k, _ = r.kv.appended[SLOTS["B"]]
    assert not torch.equal(got_k, k)
    assert torch.equal(got_k[:T], kv_of(torch.tensor(PROMPTS["D"]))[0])


def test_an_auto_refusal_is_recorded_and_reported_with_its_reason():
    r = _runner()
    r.note_prefill_graph_refused("memory: 3300 MiB pool, 1000 MiB free")
    assert r.prefill_graph_stats() == {"status": "refused", "why": "memory: 3300 MiB pool, 1000 MiB free"}
    assert r._prefill_graph is None


def test_stats_read_off_until_enabled_and_cpu_refuses():
    r = PagedModelRunner(StubModel(), RecordingKV(), device="cpu")
    assert r.prefill_graph_stats() == {"status": "off"}
    with pytest.raises(PrefillGraphRefused, match="needs a CUDA device") as e:
        r.enable_prefill_graph(T)
    assert e.value.why.startswith("needs a CUDA device")
    assert r.prefill_graph_stats() == {"status": "off"}


def test_the_capture_reads_the_head_s_width_on_a_dense_and_an_int4_head():
    """The capture's warm-up prompts are drawn below the head's output width. The int4 head
    (``E4B_SERVE_LMHEAD_INT4_CALIB=1``) is an ``Int4Linear``: its packed grid and ``N``, never a ``weight``. Reading
    ``.weight`` crashed the default graph server's build with that head (lane P125's proof, ``p125-prove-1``)."""
    import inspect

    from experts4bit_qlora.engines import paged_runner
    from experts4bit_qlora.engines.int4_attn import Int4Linear

    class Headed(torch.nn.Module):
        def __init__(self, head):
            super().__init__()
            self.lm_head = head

        def get_output_embeddings(self):
            return self.lm_head

    assert paged_runner._output_width(Headed(torch.nn.Linear(8, 50, bias=False))) == 50
    int4 = Int4Linear.__new__(Int4Linear)       # the real class; its constructor needs grouped-nf4-gemm's kernels
    torch.nn.Module.__init__(int4)
    int4.N, int4.K = 151936, 2048
    assert not hasattr(int4, "weight"), "the int4 head carries no weight: the read this fixes"
    assert paged_runner._output_width(Headed(int4)) == 151936
    src = inspect.getsource(paged_runner.PagedModelRunner.enable_prefill_graph)
    assert "_output_width(self.model)" in src and ".weight.shape" not in src
