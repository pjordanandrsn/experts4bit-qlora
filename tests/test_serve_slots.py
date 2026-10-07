# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""``serve_paged`` above 16 slots (no GPU): decode-graph buckets that end at ``max_seqs`` (``E4B_PAGED_BUCKETS=auto``),
and the step trace's count of the replays a decode step took.

Kept apart from ``tests/test_decode_graph_buckets.py``, whose bytes lanes P109-P115 pin in their staged sets.
"""
import collections
import types

import torch

from experts4bit_qlora.engines.paged_runner import PagedModelRunner, bucket_for, chunk_rows
from experts4bit_qlora.serve_recipe import default_buckets


def test_auto_buckets_put_a_wide_step_in_one_bucket():
    b = default_buckets(64)
    assert [bucket_for(n, b) for n in (16, 17, 32, 33, 64)] == [16, 32, 32, 64, 64]
    assert chunk_rows(range(64), b[-1]) == [list(range(64))]
    assert len(chunk_rows(range(64), 16)) == 4          # the default list: four replays, a host sync after each


class _Tracer:
    def __init__(self):
        self.counts, self.notes = {}, {}

    def count(self, name, n=1):
        self.counts[name] = self.counts.get(name, 0) + n

    def note(self, **kw):
        self.notes.update(kw)

    def mark(self, name, event=False):
        pass


def _bucketed_step(buckets, n_rows):
    """``_run_decode_bucketed`` on a stand-in carrying exactly what it reads; each replay emits token 7 per row."""
    bufs = {b: {"ids": torch.zeros(b, 1, dtype=torch.long), "pos": torch.zeros(b, 1, dtype=torch.long),
                "tok": torch.zeros(b, dtype=torch.long), "st": None} for b in buckets}
    graphs = {b: types.SimpleNamespace(replay=(lambda b=b: bufs[b]["tok"].fill_(7))) for b in buckets}
    ready = set()
    kv = types.SimpleNamespace(scratch=list(range(1000, 1000 + max(buckets))),
                               graph_bucket_load=lambda st, slots: None, graph_bucket_publish=lambda st: None,
                               _seen={0: collections.defaultdict(int)})
    fake = types.SimpleNamespace(
        _buckets=tuple(buckets), _bufs=bufs, _graphs=graphs, kv=kv, pool_layers=[0], tracer=_Tracer(),
        graph_stats={b: {"replays": 0, "eager_steps": 0, "rows": 0, "pad_rows": 0} for b in buckets},
        slot_of={r: r for r in range(n_rows)}, tokens={r: [5] for r in range(n_rows)},
        pos_of={r: 1 for r in range(n_rows)}, _graph_ready=ready, _ensure_graph_ready=ready.add)
    got = PagedModelRunner._run_decode_bucketed(fake, list(range(n_rows)))
    assert got == {r: 7 for r in range(n_rows)} and all(fake.pos_of[r] == 2 for r in range(n_rows))
    return fake


def test_the_step_trace_counts_the_replays_a_wide_step_takes():
    """``dec_pieces``: one per replay. Above the largest bucket a step runs in consecutive replays, and the trace's
    ``bucket`` names only the last one, so a reader needs the count to tell a chained step from a native one."""
    chained = _bucketed_step((1, 2, 4, 8, 16), 40)
    assert chained.tracer.counts["dec_pieces"] == 3 and chained.tracer.counts["decode_rows"] == 40
    assert [chained.graph_stats[b]["replays"] for b in (8, 16)] == [1, 2] and chained.tracer.notes["bucket"] == 8
    native = _bucketed_step(default_buckets(40), 40)
    assert native.tracer.counts["dec_pieces"] == 1 and native.graph_stats[40]["replays"] == 1
    assert native.graph_stats[40]["pad_rows"] == 0 and native.tracer.notes["bucket"] == 40
