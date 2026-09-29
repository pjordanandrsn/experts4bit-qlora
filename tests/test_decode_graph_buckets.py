# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""Bucketed CUDA-graph decode in ``PagedModelRunner`` (#511).

A step pads its active set to the next bucket with scratch KV slots and replays
that bucket's graph. The oracle is the SAME padded step run eagerly, which must
give the same tokens bit for bit: a bf16 GEMM may round differently at a
different row count, so unpadded eager decode is reported, not asserted.
"""
import types

import pytest
import torch

from experts4bit_qlora.engines.paged_runner import (
    PagedModelRunner, bucket_for, chunk_rows)


# ------------------------------------------------------------- anywhere --

def test_bucket_for_picks_the_smallest_bucket_that_holds_the_rows():
    b = (1, 2, 4, 8, 16)
    assert [bucket_for(n, b) for n in (1, 2, 3, 4, 5, 8, 9, 16)] == [1, 2, 4, 4, 8, 8, 16, 16]
    with pytest.raises(ValueError, match="largest bucket"):
        bucket_for(17, b)


def test_chunk_rows_splits_an_oversized_active_set():
    assert chunk_rows(range(5), 2) == [[0, 1], [2, 3], [4]]
    assert chunk_rows([7], 16) == [[7]]


def _fake_runner(n_scratch, batch=4):
    kv = types.SimpleNamespace(L=1, B=batch, scratch=list(range(batch, batch + n_scratch)))
    model = torch.nn.Linear(1, 1)
    return PagedModelRunner(model, kv, device="cpu")


def test_bind_refuses_a_scratch_slot():
    r = _fake_runner(2)
    with pytest.raises(ValueError, match="scratch slot"):
        r.bind(0, 5, [1, 2])


def test_graphs_refuse_too_few_scratch_slots():
    r = _fake_runner(4)
    with pytest.raises(ValueError, match="scratch_slots >= the largest bucket"):
        r.enable_decode_graphs(buckets=(1, 2, 8))


# --------------------------------------------------------------- on a GPU --

# The fp8 paged KV quantises to e4m3 (Triton ``fp8e4nv``), which compiles only on
# sm_89+. On an sm_86 card (the A2000) the kernel refuses to build; that is a
# fact about the card, so it skips by name rather than failing as a compile error.
needs_fp8 = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() < (8, 9),
    reason="the fp8 paged KV needs native e4m3 (sm_89+)")

# different prompt lengths and generation lengths: the active set shrinks
# 4 -> 3 -> 2 -> 1 (buckets 4, 4 padded, 2, 1), and a fifth request is admitted
# into a recycled slot mid-run
PROMPTS = [[3, 17, 42, 9, 11, 5, 88, 23, 7], [5, 1, 99, 64, 2], [120, 7, 7, 31, 2, 9, 14, 60, 3, 3, 8, 1],
           [15, 250, 4, 4, 81, 6, 12], [44, 45, 46, 47]]
MAX_NEW = [14, 10, 6, 3, 8]


def _run(mode, buckets=(1, 2, 4)):
    pytest.importorskip("fp8_paged_attn", reason="needs grouped-nf4-gemm's fp8 paged attention")
    fp8_kv = pytest.importorskip("fp8_kv", reason="needs grouped-nf4-gemm's fp8 KV")
    if not hasattr(fp8_kv, "fp8_kv_append_bt1"):
        pytest.skip("this grouped-nf4-gemm has no fused batch KV append")
    from transformers.models.qwen3.configuration_qwen3 import Qwen3Config
    from transformers.models.qwen3.modeling_qwen3 import Qwen3ForCausalLM

    from experts4bit_qlora.engines import paged_attention
    from experts4bit_qlora.engines.fp8_paged_kv import Fp8PagedKV
    from experts4bit_qlora.engines.scheduler import ContinuousScheduler

    torch.manual_seed(0)
    cfg = Qwen3Config(hidden_size=256, intermediate_size=512, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2, head_dim=64,
                      vocab_size=256, max_position_embeddings=256)
    model = Qwen3ForCausalLM(cfg).to("cuda", torch.bfloat16).eval()
    paged_attention.register(model)
    kv = Fp8PagedKV(cfg.num_hidden_layers, cfg.num_key_value_heads, cfg.head_dim,
                    batch=4, max_tokens_per_seq=64, scratch_slots=max(buckets))
    runner = PagedModelRunner(model, kv)
    status = None
    if mode in ("graph", "padded-eager"):
        status = runner.enable_decode_graphs(buckets, capture=(mode == "graph"), verbose=False)
    sched = ContinuousScheduler(runner=runner, max_seqs=4, kv_slots=4, chunk_tokens=64)
    rids = [sched.add_request(p, m) for p, m in zip(PROMPTS, MAX_NEW)]
    sched.run_until_idle(max_steps=500)
    out = {r.rid: list(r.out) for r in sched.done}
    return [out[r] for r in rids], status, getattr(runner, "graph_stats", None)


@needs_fp8
def test_a_bucket_replay_decodes_exactly_as_the_padded_eager_step():
    graph, status, stats = _run("graph")
    assert status == {1: "graph", 2: "graph", 4: "graph"}, status
    eager, _, estats = _run("padded-eager")
    assert graph == eager, f"replay != padded eager:\n  graph={graph}\n  eager={eager}"
    assert [len(t) for t in graph] == MAX_NEW
    # every bucket actually replayed, and the padded rows were real padding
    assert all(stats[b]["replays"] > 0 for b in (1, 2, 4)), stats
    assert stats[4]["pad_rows"] > 0, stats
    assert sum(s["eager_steps"] for s in stats.values()) == 0
    assert sum(s["replays"] for s in estats.values()) == 0
    plain, _, _ = _run("plain")
    # reported, not asserted: unpadded eager runs a different row count per GEMM
    print("unpadded-eager agreement:",
          sum(a == b for x, y in zip(graph, plain) for a, b in zip(x, y)), "/", sum(MAX_NEW))


@needs_fp8
@pytest.mark.parametrize("mode", ["graph", "padded-eager"])
def test_every_bucket_step_advances_its_rows_own_kv_length(mode, monkeypatch):
    """Lane B771: bucket 1 appended through the single-slot form, which writes to graph_mode_init's
    slot -- a scratch slot -- so the one active row's own length stopped advancing while the runner's
    host mirror kept counting. The replay-vs-padded-eager test above cannot see that (both share the
    shim), so this checks the invariant itself after every bucketed step, in every layer: each stepped
    row's device length equals the host count. The trace below ends in a one-row phase."""
    from experts4bit_qlora.engines import paged_runner as pr

    seen_one_row = {"n": 0}
    real = pr.PagedModelRunner._run_decode_bucketed

    def checked(self, rids):
        got = real(self, rids)
        if len(rids) == 1:
            seen_one_row["n"] += 1
        torch.cuda.synchronize()
        for r in rids:
            slot = self.slot_of[r]
            for layer in range(self.kv.L):
                dev_len = int(self.kv.seq_lens[layer, slot].item())
                host_len = self.kv._seen[layer][slot]
                assert dev_len == host_len, (
                    f"after a {len(rids)}-row bucket step, slot {slot} layer {layer}: device length "
                    f"{dev_len} != host count {host_len} -- the row's K/V went somewhere else")
        return got

    monkeypatch.setattr(pr.PagedModelRunner, "_run_decode_bucketed", checked)
    _run(mode)
    assert seen_one_row["n"] > 0, "the trace never reached a one-row step; the test would prove nothing"
