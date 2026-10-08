# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""The decode lookahead (``E4B_PAGED_DECODE_LOOKAHEAD``, lane P118) on a real model, KV and captured graphs (sm_89+).

``tests/test_decode_lookahead.py`` checks the protocol on the CPU with a stand-in for the graph. Here the tiny Qwen3
of ``tests/test_decode_graph_buckets.py`` decodes five requests through four slots (a recycled slot, buckets 4 -> 1)
synchronously and with the lookahead, through captured graphs and through the padded eager step. The lookahead feeds
each step the synchronous step's inputs on the device, so the tokens, the buckets each step used and every slot's KV
length must be the same; and a step must actually be queued while the host works (the lookahead engaged).
"""
import pytest
import torch

needs_fp8 = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() < (8, 9),
    reason="the fp8 paged KV needs native e4m3 (sm_89+)")

PROMPTS = [[3, 17, 42, 9, 11, 5, 88, 23, 7], [5, 1, 99, 64, 2], [120, 7, 7, 31, 2, 9, 14, 60, 3, 3, 8, 1],
           [15, 250, 4, 4, 81, 6, 12], [44, 45, 46, 47]]
MAX_NEW = [14, 10, 6, 3, 8]


def _run(capture, lookahead, buckets=(1, 2, 4), stop=None, n=len(PROMPTS)):
    pytest.importorskip("fp8_paged_attn", reason="needs grouped-nf4-gemm's fp8 paged attention")
    fp8_kv = pytest.importorskip("fp8_kv", reason="needs grouped-nf4-gemm's fp8 KV")
    if not hasattr(fp8_kv, "fp8_kv_append_bt1"):
        pytest.skip("this grouped-nf4-gemm has no fused batch KV append")
    from transformers.models.qwen3.configuration_qwen3 import Qwen3Config
    from transformers.models.qwen3.modeling_qwen3 import Qwen3ForCausalLM

    from experts4bit_qlora.engines import paged_attention
    from experts4bit_qlora.engines.fp8_paged_kv import Fp8PagedKV
    from experts4bit_qlora.engines.paged_runner import PagedModelRunner
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
    status = runner.enable_decode_graphs(buckets, capture=capture, verbose=False)
    sched = ContinuousScheduler(runner=runner, max_seqs=4, kv_slots=4, chunk_tokens=64, lookahead=lookahead)
    rids = [sched.add_request(p, m, stop_ids=stop) for p, m in zip(PROMPTS[:n], MAX_NEW[:n])]
    # at each collect: were two steps outstanding, i.e. was a newer step already queued behind the one read back?
    overlapped, outstanding = [], [0]
    issue, collect = runner.issue_decode, runner.collect_decode

    def issue_counted(rids_):
        outstanding[0] += 1
        return issue(rids_)

    def collect_counted(h):
        overlapped.append(outstanding[0] == 2)
        outstanding[0] -= 1
        return collect(h)

    runner.issue_decode, runner.collect_decode = issue_counted, collect_counted
    sched.run_until_idle(max_steps=500)
    torch.cuda.synchronize()
    out = {r.rid: (list(r.out), r.finish_reason) for r in sched.done}
    lens = [int(kv.seq_lens[layer, s_]) for layer in range(kv.L) for s_ in range(kv.B)]
    return [out[r] for r in rids], status, runner.graph_stats, lens, overlapped


@needs_fp8
@pytest.mark.parametrize("capture", [True, False], ids=["graph", "padded-eager"])
def test_the_lookahead_decodes_the_synchronous_tokens(capture):
    sync, status, s_stats, s_lens, _ = _run(capture, False)
    look, _, l_stats, l_lens, overlapped = _run(capture, True)
    if capture:
        assert status == {1: "graph", 2: "graph", 4: "graph"}, status
    assert look == sync, f"lookahead != synchronous:\n  look={look}\n  sync={sync}"
    assert [len(o) for o, _ in look] == MAX_NEW
    assert l_stats == s_stats                         # the same rows in the same buckets
    assert l_lens == s_lens                           # every slot's KV length where the synchronous path left it
    # the lookahead engaged: most collects had a newer step queued behind them (not the last, nor one taken early
    # because a queued request waited for the slot it frees)
    assert overlapped and sum(overlapped) >= len(overlapped) // 2 and not overlapped[-1], overlapped


@needs_fp8
def test_a_stop_through_the_lookahead_keeps_the_synchronous_streams():
    """A stop id ends a stream early: the lookahead computes one more step for it and discards the token. With every
    request resident (four requests, four slots: no admission waits on the stop) the streams are the synchronous ones."""
    sync, _, _, _, _ = _run(True, False, n=4)
    stop = {sync[0][0][3]}
    want, _, _, _, _ = _run(True, False, stop=stop, n=4)
    got, _, _, _, _ = _run(True, True, stop=stop, n=4)
    assert [w[1] for w in want].count("stop") >= 1
    assert got == want
