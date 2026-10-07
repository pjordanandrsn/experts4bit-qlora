# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""Bucketed CUDA-graph decode with the B=1 glue folds applied (lane P115's premise, e4b#1313).

``tests/test_decode_graph_buckets.py`` proves a bucket replay decodes exactly as the same padded step run eagerly. Lane
P115 reads the registered B=1 fused stack on the default graph server, so the folds' kernels -- glue round 1
(``rmsnorm_rows``) and round 2 (``rmsnorm_resid_rows``, ``rope_norm_heads``) -- must capture and replay exactly too.
The model is that test's tiny dense Qwen3 (its decoder layer is the plain four-child body round 2 folds, and its
attention the separate-projection shape with per-head q/k norms), in bf16 so the folds' decode gate takes the fused
path. The router epilogue and the fused q/k/v need a MoE attention and router; the lane's own engagement checks cover
them on the served model.
"""
import pytest
import torch

needs_fp8 = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() < (8, 9),
    reason="the fp8 paged KV needs native e4m3 (sm_89+)")

PROMPTS = [[3, 17, 42, 9, 11, 5, 88, 23, 7], [5, 1, 99, 64, 2], [120, 7, 7, 31, 2, 9, 14, 60, 3, 3, 8, 1],
           [15, 250, 4, 4, 81, 6, 12], [44, 45, 46, 47]]
MAX_NEW = [14, 10, 6, 3, 8]
KERNELS = ("rmsnorm_rows", "rmsnorm_resid_rows", "rope_norm_heads")


def _run(mode, monkeypatch, buckets=(1, 2, 4)):
    pytest.importorskip("fp8_paged_attn", reason="needs grouped-nf4-gemm's fp8 paged attention")
    fp8_kv = pytest.importorskip("fp8_kv", reason="needs grouped-nf4-gemm's fp8 KV")
    int4_b32 = pytest.importorskip("int4_b32", reason="needs grouped-nf4-gemm's decode glue kernels")
    if not hasattr(fp8_kv, "fp8_kv_append_bt1"):
        pytest.skip("this grouped-nf4-gemm has no fused batch KV append")
    from transformers.models.qwen3.configuration_qwen3 import Qwen3Config
    from transformers.models.qwen3.modeling_qwen3 import Qwen3ForCausalLM

    from experts4bit_qlora.engines import paged_attention
    from experts4bit_qlora.engines.fp8_paged_kv import Fp8PagedKV
    from experts4bit_qlora.engines.glue_fuse import fuse_t1_glue
    from experts4bit_qlora.engines.glue_r2 import fuse_t1_glue_r2
    from experts4bit_qlora.engines.paged_runner import PagedModelRunner
    from experts4bit_qlora.engines.scheduler import ContinuousScheduler

    calls = {k: 0 for k in KERNELS}
    for name in KERNELS:
        orig = getattr(int4_b32, name)

        def counted(*a, _n=name, _o=orig, **k):
            calls[_n] += 1
            return _o(*a, **k)
        monkeypatch.setattr(int4_b32, name, counted)       # before the folds bind them
    monkeypatch.setenv("E4B_FUSE_T1_GLUE", "1")
    monkeypatch.setenv("E4B_FUSE_T1_GLUE_R2", "1")
    torch.manual_seed(0)
    cfg = Qwen3Config(hidden_size=256, intermediate_size=512, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2, head_dim=64,
                      vocab_size=256, max_position_embeddings=256)
    model = Qwen3ForCausalLM(cfg).to("cuda", torch.bfloat16).eval()
    paged_attention.register(model)
    census = (fuse_t1_glue(model), fuse_t1_glue_r2(model))
    kv = Fp8PagedKV(cfg.num_hidden_layers, cfg.num_key_value_heads, cfg.head_dim,
                    batch=4, max_tokens_per_seq=64, scratch_slots=max(buckets))
    runner = PagedModelRunner(model, kv)
    status = runner.enable_decode_graphs(buckets, capture=(mode == "graph"), verbose=False)
    sched = ContinuousScheduler(runner=runner, max_seqs=4, kv_slots=4, chunk_tokens=64)
    rids = [sched.add_request(p, m) for p, m in zip(PROMPTS, MAX_NEW)]
    sched.run_until_idle(max_steps=500)
    out = {r.rid: list(r.out) for r in sched.done}
    return [out[r] for r in rids], status, runner.graph_stats, census, calls


@needs_fp8
def test_the_folds_replay_exactly_as_their_padded_eager_step(monkeypatch):
    graph, status, stats, census, calls = _run("graph", monkeypatch)
    assert census == (9, (2, 2)), census                  # 4 norms per layer + the final norm; 2 layers, 2 attentions
    assert status == {1: "graph", 2: "graph", 4: "graph"}, status
    assert all(stats[b]["replays"] > 0 for b in (1, 2, 4)), stats
    assert sum(s["eager_steps"] for s in stats.values()) == 0
    assert all(calls[k] > 0 for k in KERNELS), calls       # the capture ran the fused kernels
    eager, _, estats, census2, ecalls = _run("padded-eager", monkeypatch)
    assert census2 == census
    assert graph == eager, f"replay != padded eager with the folds:\n  graph={graph}\n  eager={eager}"
    assert [len(t) for t in graph] == MAX_NEW
    assert sum(s["replays"] for s in estats.values()) == 0
    assert all(ecalls[k] > 0 for k in KERNELS), ecalls
