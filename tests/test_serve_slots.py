# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""``serve_paged`` above 16 slots (no GPU): decode-graph buckets that end at ``max_seqs`` (``E4B_PAGED_BUCKETS=auto``),
and the step trace's count of the replays a decode step took.

Kept apart from ``tests/test_decode_graph_buckets.py``, whose bytes lanes P109-P115 pin in their staged sets.
"""
import collections
import types

import pytest
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


# ------------------------------------------------- E4B_PAGED_MAX_SEQS=auto --

GIB = 2**30
FREE_5090 = int(31.3 * GIB)            # an RTX 5090's free memory before load (32,607 MiB less the CUDA context)
FREE_4090 = int(23.4 * GIB)            # an RTX 4090's (24,564 MiB less the context)


def _qwen3_30b():
    """Qwen3-30B-A3B's shape (the model lane SC2e read), on the meta device: no weights."""
    tr = pytest.importorskip("transformers")
    from experts4bit_qlora.arch.topology import describe_moe
    return describe_moe(tr.Qwen3MoeConfig(
        hidden_size=2048, intermediate_size=6144, moe_intermediate_size=768, num_experts=128, num_experts_per_tok=8,
        num_hidden_layers=48, num_attention_heads=32, num_key_value_heads=4, head_dim=128, vocab_size=151936,
        max_position_embeddings=40960, decoder_sparse_step=1, tie_word_embeddings=False))


def test_auto_takes_64_slots_where_sc2e_read_them_and_the_estimate_fits():
    from experts4bit_qlora.serve_recipe import MAX_SEQS_AUTO_WIDTHS, ServeSetup, choose_max_seqs
    topo = _qwen3_30b()
    r = choose_max_seqs(topo, ServeSetup(max_tokens_per_seq=2048, exp_int4=True, attn_int4=True), FREE_5090)
    assert r["max_seqs"] == 64 and MAX_SEQS_AUTO_WIDTHS == (64, 16)
    assert [c["max_seqs"] for c in r["candidates"]] == [64, 16] and r["candidates"][0]["fits"]
    # each width is priced whole: estimate (weights, pool, scratch, bulk flush) + the prefill-graph reserve + 1 GiB
    c = r["candidates"][0]
    assert c["need_bytes"] == c["estimate_bytes"] + c["prefill_graph_reserve_bytes"] + GIB


def test_auto_never_takes_64_at_4096_tokens_a_slot_on_a_32_gb_card():
    """The server's default 4,096 tokens a slot doubles the pool: 64 slots need ~34 GiB on Qwen3-30B-A3B int4. On the
    default bucket list the next width SC2e read is 16; with E4B_PAGED_BUCKETS=auto it is 32 (s32a)."""
    from experts4bit_qlora.serve_recipe import MAX_SEQS_AUTO_WIDTHS_AUTO_BUCKETS, ServeSetup, choose_max_seqs
    topo = _qwen3_30b()
    st = ServeSetup(max_tokens_per_seq=4096, exp_int4=True, attn_int4=True)
    r = choose_max_seqs(topo, st, FREE_5090)
    assert r["max_seqs"] == 16 and not r["candidates"][0]["fits"]
    wide = choose_max_seqs(topo, st, FREE_5090, widths=MAX_SEQS_AUTO_WIDTHS_AUTO_BUCKETS)
    assert wide["max_seqs"] == 32 and [c["fits"] for c in wide["candidates"]] == [False, True, True]


@pytest.mark.parametrize("tokens", [2048, 4096])
def test_auto_keeps_16_on_a_24_gb_card(tokens):
    from experts4bit_qlora.serve_recipe import MAX_SEQS_AUTO_WIDTHS_AUTO_BUCKETS, ServeSetup, choose_max_seqs
    topo = _qwen3_30b()
    for widths in ((64, 16), MAX_SEQS_AUTO_WIDTHS_AUTO_BUCKETS):
        r = choose_max_seqs(topo, ServeSetup(max_tokens_per_seq=tokens, exp_int4=True, attn_int4=True), FREE_4090,
                            widths=widths)
        assert r["max_seqs"] == 16
    if tokens == 4096:                     # nothing fits all-VRAM there: 16, today's default, and the record says so
        assert "no width fits" in r["why"]


def _hybrid():
    tr = pytest.importorskip("transformers")
    from experts4bit_qlora.arch.topology import describe_moe
    return describe_moe(tr.Qwen3_5MoeConfig(text_config=dict(
        vocab_size=128, hidden_size=64, num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2, head_dim=32,
        moe_intermediate_size=64, shared_expert_intermediate_size=64, num_experts=4, num_experts_per_tok=2,
        linear_num_key_heads=2, linear_num_value_heads=4, linear_key_head_dim=16, linear_value_head_dim=24,
        linear_conv_kernel_dim=4, max_position_embeddings=256, layer_types=["linear_attention", "full_attention"] * 2)))


def test_auto_prices_a_hybrids_state_per_slot_and_no_prefill_graph_reserve():
    """A hybrid's linear-attention state is a full slot each (~62 MiB on Qwen3.6-35B-A3B), seqs and scratch alike: 64
    slots cost it 128 slots of state. The estimate carries that, so the choice does; the prefill graph cannot engage on
    a hybrid, so nothing is reserved for it."""
    from experts4bit_qlora.serve_recipe import (ServeSetup, choose_max_seqs, estimate_serve_footprint,
                                                linear_state_pool_bytes, prefill_graph_reserve_bytes)
    topo = _hybrid()
    st = ServeSetup(max_tokens_per_seq=256)
    r = choose_max_seqs(topo, st, 1 << 40)
    need = {c["max_seqs"]: c["need_bytes"] for c in r["candidates"]}
    state = {w: linear_state_pool_bytes(topo.linear_state_layers, 2 * w) for w in (16, 64)}
    est = {w: estimate_serve_footprint(topo, ServeSetup(max_seqs=w, max_tokens_per_seq=256)).device_bytes for w in (16, 64)}
    assert est[64] - est[16] >= state[64] - state[16] > 0
    assert prefill_graph_reserve_bytes(topo, ServeSetup(max_seqs=64)) == 0
    assert choose_max_seqs(topo, st, need[64] - 1)["max_seqs"] == 16
    assert choose_max_seqs(topo, st, need[64])["max_seqs"] == 64


def test_auto_keeps_16_without_a_cuda_device_and_under_the_solver():
    from experts4bit_qlora.serve_recipe import ServeSetup, choose_max_seqs
    topo = _qwen3_30b()
    assert choose_max_seqs(topo, ServeSetup(), None)["max_seqs"] == 16
    r = choose_max_seqs(topo, ServeSetup(placement="solver", graphs=False), FREE_5090)
    assert r["max_seqs"] == 16 and "solver" in r["why"]


def test_the_prefill_graph_reserve_bounds_every_pool_measured():
    """chunk x hidden x layers x 16 B: 0.25 GiB on OLMoE-1B-7B's shape and 0.75 GiB on Qwen3-30B-A3B's, against the
    pools measured at 512 tokens (SV1: 0.24 and 0.57 GiB at NF4; SC2e: 0.42 GiB at int4)."""
    tr = pytest.importorskip("transformers")
    from experts4bit_qlora.arch.topology import describe_moe
    from experts4bit_qlora.serve_recipe import ServeSetup, prefill_graph_reserve_bytes
    olmoe = describe_moe(tr.OlmoeConfig(hidden_size=2048, num_hidden_layers=16, num_attention_heads=16,
                                        num_key_value_heads=16, num_experts=64, num_experts_per_tok=8,
                                        intermediate_size=1024, vocab_size=50304))
    assert prefill_graph_reserve_bytes(olmoe, ServeSetup()) >= int(0.24 * GIB)
    assert prefill_graph_reserve_bytes(_qwen3_30b(), ServeSetup()) >= int(0.57 * GIB)
    assert prefill_graph_reserve_bytes(_qwen3_30b(), ServeSetup(prefill_graph="0")) == 0
