# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""A bound decode-graph bucket of ONE row appends through the bucket's slot (lane B771).

``enable_decode_graphs`` initialises graph mode on a SCRATCH slot (``graph_mode_init(seq=scratch[0])``)
and binds each bucket's real slots on device. The paged-attention shim sent any single-row decode to
``append_graph_t1``, which writes to that init-time slot: at bucket 1 -- exactly one active row -- every
new token landed in scratch, the row's own length never advanced, and attention (which reads the row's
slot through the bucket selector) ran without the tokens of the one-row phase. B771 found it on an RTX
5090: with every other difference removed, the bucket step left the eager step at the first one-row
step and nowhere else. The replay-vs-padded-eager gate could not see it: both run the same shim.

These pin the routing through the real shim on CPU, with a recording stand-in for the KV cache: a
bound bucket takes the batch form whatever its size; an unbound single-sequence graph (the b1d path,
whose ``_g_seq`` IS the sequence) keeps the single-slot form.
"""
import types

import pytest

torch = pytest.importorskip("torch")
pa = pytest.importorskip("experts4bit_qlora.engines.paged_attention")


class _RecordingKV:
    graph_t1 = True

    def __init__(self, slots, bound):
        self.calls = []
        self._g_slots = list(slots)
        self._g_sel = torch.tensor(slots) if bound else None

    def append_graph_bt1(self, layer, k, v):
        self.calls.append(("bt1", layer, tuple(k.shape)))

    def append_graph_t1(self, layer, k, v):
        self.calls.append(("t1", layer, tuple(k.shape)))

    def attention(self, layer, q, slots=None, **kw):
        return torch.zeros_like(q)


def _decode(kv, slots, h_q=4, h_kv=2, d=16):
    B = len(slots)
    q = torch.randn(B, h_q, 1, d)
    k = torch.randn(B, h_kv, 1, d)
    v = torch.randn(B, h_kv, 1, d)
    module = types.SimpleNamespace(layer_idx=3)
    prev = pa.set_context(pa.PagedAttentionContext(kv=kv, slots=list(slots), mode="decode"))
    try:
        pa.paged_attention_forward(module, q, k, v, None, scaling=d ** -0.5)
    finally:
        pa.set_context(prev)
    return kv.calls


def test_a_bound_bucket_of_one_row_appends_through_the_bucket_slot():
    calls = _decode(_RecordingKV([7], bound=True), [7])
    assert calls == [("bt1", 3, (1, 2, 16))], calls


def test_a_bound_bucket_of_several_rows_still_takes_the_batch_form():
    calls = _decode(_RecordingKV([7, 2, 5], bound=True), [7, 2, 5])
    assert [c[0] for c in calls] == ["bt1"], calls


def test_an_unbound_single_sequence_graph_keeps_the_single_slot_form():
    """The b1d path: graph_mode_init(seq=<the sequence>) and no bucket bound -- its _g_seq is right."""
    calls = _decode(_RecordingKV([0], bound=False), [0])
    assert calls == [("t1", 3, (1, 2, 16))], calls   # key[0].permute(1, 0, 2): [T, H, D]
