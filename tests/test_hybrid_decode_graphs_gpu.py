# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""Bucketed CUDA-graph decode for a HYBRID model (linear attention + attention): the per-slot Gated DeltaNet state is
gathered and scattered through the bound bucket's device selector (``linear_state._bucket_selector``), so a captured
graph reads and writes the step's real slots, not the scratch slots it was captured on.

The oracle is the one ``tests/test_decode_graph_buckets.py`` uses: the SAME padded step run eagerly
(``capture=False``) must give the same tokens bit for bit. A graph that baked the capture's slot list would decode every
row from scratch-slot state and diverge at once. Every bucket must actually capture and replay (a failed capture runs
eagerly and would make the comparison trivial), and a padding row must occur.

A dense Qwen3.5 hybrid (``Qwen3_5ForCausalLM``: Gated DeltaNet + full attention, dense MLP) keeps the MoE engine out of
the capture: transformers' own MoE block routes experts with data-dependent indexing, which no CUDA graph can hold.
The linear-attention module is the same family the Qwen3.5 / 3.6 MoE checkpoints use.

Skips without an sm_89+ card (the fp8 KV's e4m3) or grouped-nf4-gemm's fused batch KV append.
"""
import pytest
import torch

needs_fp8 = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() < (8, 9),
    reason="the fp8 paged KV needs native e4m3 (sm_89+)")

LIN, ATT = "linear_attention", "full_attention"
# the active set shrinks 4 -> 3 -> 2 -> 1 (buckets 4, 4 padded, 2, 1), and a fifth request is admitted into a recycled
# slot mid-run, so a slot's linear state is reset and refilled between graph replays
PROMPTS = [[3, 17, 42, 9, 11, 5, 88, 23, 7], [5, 1, 99, 64, 2], [120, 7, 7, 31, 2, 9, 14, 60, 3, 3, 8, 1],
           [15, 250, 4, 4, 81, 6, 12], [44, 45, 46, 47]]
MAX_NEW = [14, 10, 6, 3, 8]


def _run(mode, buckets=(1, 2, 4)):
    pytest.importorskip("fp8_paged_attn", reason="needs grouped-nf4-gemm's fp8 paged attention")
    fp8_kv = pytest.importorskip("fp8_kv", reason="needs grouped-nf4-gemm's fp8 KV")
    if not hasattr(fp8_kv, "fp8_kv_append_bt1"):
        pytest.skip("this grouped-nf4-gemm has no fused batch KV append")
    pytest.importorskip("transformers.models.qwen3_5", reason="needs transformers with Qwen3.5")
    from transformers import Qwen3_5TextConfig
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM

    from experts4bit_qlora.engines import paged_attention
    from experts4bit_qlora.engines.fp8_paged_kv import Fp8PagedKV
    from experts4bit_qlora.engines.paged_runner import PagedModelRunner, kv_layers
    from experts4bit_qlora.engines.scheduler import ContinuousScheduler

    torch.manual_seed(0)
    cfg = Qwen3_5TextConfig(vocab_size=256, hidden_size=256, intermediate_size=512, num_hidden_layers=4,
                            num_attention_heads=4, num_key_value_heads=2, head_dim=64, linear_num_key_heads=2,
                            linear_num_value_heads=4, linear_key_head_dim=32, linear_value_head_dim=32,
                            linear_conv_kernel_dim=4, layer_types=[LIN, ATT, LIN, ATT],
                            max_position_embeddings=256)
    model = Qwen3_5ForCausalLM(cfg).to("cuda", torch.bfloat16).eval()
    paged_attention.register(model)
    kv = Fp8PagedKV(kv_layers(model, cfg.num_hidden_layers), cfg.num_key_value_heads, cfg.head_dim,
                    batch=4, max_tokens_per_seq=64, scratch_slots=max(buckets))
    runner = PagedModelRunner(model, kv)
    assert runner.linear_state is not None and kv.L == 2 and runner.pool_layers == [0, 1]   # a compact pool
    status = None
    if mode in ("graph", "padded-eager"):
        status = runner.enable_decode_graphs(buckets, capture=(mode == "graph"), verbose=False)
        assert runner.linear_state.frozen and runner.linear_state.allocated([0, 2])
    sched = ContinuousScheduler(runner=runner, max_seqs=4, kv_slots=4, chunk_tokens=64)
    rids = [sched.add_request(p, m) for p, m in zip(PROMPTS, MAX_NEW)]
    sched.run_until_idle(max_steps=500)
    out = {r.rid: list(r.out) for r in sched.done}
    return [out[r] for r in rids], status, getattr(runner, "graph_stats", None)


@needs_fp8
def test_a_hybrid_bucket_replay_decodes_exactly_as_the_padded_eager_step():
    graph, status, stats = _run("graph")
    assert status == {1: "graph", 2: "graph", 4: "graph"}, status
    eager, _, estats = _run("padded-eager")
    assert graph == eager, f"replay != padded eager:\n  graph={graph}\n  eager={eager}"
    assert [len(t) for t in graph] == MAX_NEW
    assert all(stats[b]["replays"] > 0 for b in (1, 2, 4)), stats
    assert stats[4]["pad_rows"] > 0, stats
    assert sum(s["eager_steps"] for s in stats.values()) == 0
    assert sum(s["replays"] for s in estats.values()) == 0
    plain, _, _ = _run("plain")
    # reported, not asserted: unpadded eager runs a different row count per GEMM
    print("unpadded-eager agreement:",
          sum(a == b for x, y in zip(graph, plain) for a, b in zip(x, y)), "/", sum(MAX_NEW))
