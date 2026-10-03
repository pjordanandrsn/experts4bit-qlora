# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""``E4B_KV_STEP_SELECT``: a decode-graph bucket's KV-table selection once per step instead of once per layer.

SC1b (``bench/h2h-2026-10-02/sc1b/README.md``) put about 0.8 ms of e4b's B=16 decode step on an RTX 5090 in
``fp8_paged_kv.py``: two ``index_select`` per layer re-selecting the active set's block-table and seq-lens rows, and one
``seq_lens.index_add_`` per layer. Under the switch, :meth:`Fp8PagedKV.graph_bucket_load` selects every layer's rows in
one launch each, outside the graph, and :meth:`Fp8PagedKV.graph_bucket_publish` advances every layer's lengths once
after the step. The kernels must see the same values, so the tokens must be identical.

On CPU: the stacked block table, the selection, the lengths attention reads, the publish, and the switch's parsing.
On sm_89+ (the fp8 KV): the tiny Qwen3 of ``tests/test_decode_graph_buckets.py`` decodes the same tokens with the
switch on and off, through captured graphs and through the padded eager step.
"""
import pytest
import torch

from experts4bit_qlora.engines import fp8_paged_kv
from experts4bit_qlora.engines.fp8_paged_kv import Fp8PagedKV


def _kv(monkeypatch, on, L=3, batch=4, scratch=4):
    monkeypatch.setenv("E4B_KV_STEP_SELECT", "1" if on else "0")
    return Fp8PagedKV(L, 2, 32, batch=batch, max_tokens_per_seq=64, device="cpu", scratch_slots=scratch)


def test_the_switch_parses_and_refuses():
    f = fp8_paged_kv._step_select_env
    assert f(None) is False and f("") is False and f("0") is False and f(" 1 ") is True
    for bad in ("2", "on", "true"):
        with pytest.raises(ValueError, match="E4B_KV_STEP_SELECT"):
            f(bad)


def test_the_block_table_is_one_tensor_seen_per_layer(monkeypatch):
    kv = _kv(monkeypatch, on=False)
    kv.block_table[1][2, 3] = 77
    assert int(kv._bt_all[1, 2, 3]) == 77
    kv._ensure_blocks(2, 0, 2)
    assert torch.equal(kv._bt_all[2], kv.block_table[2])
    assert all(kv.block_table[layer].is_contiguous() for layer in range(kv.L))


def test_the_step_selection_is_what_each_layer_would_select_after_its_append(monkeypatch):
    kv = _kv(monkeypatch, on=True)
    for layer in range(kv.L):
        for s_ in range(kv.B):
            kv._ensure_blocks(layer, s_, 2)
    kv.seq_lens.copy_(torch.randint(1, 40, kv.seq_lens.shape, dtype=kv.seq_lens.dtype))
    slots = [2, 0, 3] + kv.scratch[:1]                  # an active set out of order, padded with one scratch slot
    st = kv.graph_bucket(len(slots))
    kv.graph_bucket_load(st, slots)
    before = kv.seq_lens.clone()                        # scratch lengths already reset to 0 by the load
    assert int(before[0, kv.scratch[0]]) == 0
    sel = torch.tensor(slots)
    for layer in range(kv.L):
        assert torch.equal(st["tbl"][layer], kv.block_table[layer].index_select(0, sel))
        assert torch.equal(st["lens"][layer], before[layer].index_select(0, sel) + 1)
    # with the bucket's selection in force, kernel_args hands each layer its slice -- what the per-layer path would
    # select once that layer's append had bumped its lengths
    kv._g_step = st
    for layer in range(kv.L):
        _k, _v, tbl, lens = kv.kernel_args(layer, slots)
        assert tbl.data_ptr() == st["tbl"][layer].data_ptr() and torch.equal(lens, before[layer][sel] + 1)
    kv._g_step = None
    kv.graph_bucket_publish(st)
    for layer in range(kv.L):
        assert torch.equal(kv.seq_lens[layer][sel], before[layer][sel] + 1)
        others = [s_ for s_ in range(kv.B + kv.n_scratch) if s_ not in slots]
        assert torch.equal(kv.seq_lens[layer][others], before[layer][others])


def test_without_the_switch_nothing_changes(monkeypatch):
    kv = _kv(monkeypatch, on=False)
    st = kv.graph_bucket(2)
    assert "tbl" not in st and kv._g_step is None
    kv.seq_lens.fill_(5)
    kv.graph_bucket_load(st, [0, 1])
    kv.graph_bucket_publish(st)
    assert int(kv.seq_lens[:, :2].min()) == 5 == int(kv.seq_lens[:, :2].max())


# ------------------------------------------------------------------------------------------- on the card --

needs_fp8 = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() < (8, 9),
    reason="the fp8 paged KV needs native e4m3 (sm_89+)")

PROMPTS = [[3, 17, 42, 9, 11, 5, 88, 23, 7], [5, 1, 99, 64, 2], [120, 7, 7, 31, 2, 9, 14, 60, 3, 3, 8, 1],
           [15, 250, 4, 4, 81, 6, 12], [44, 45, 46, 47]]
MAX_NEW = [14, 10, 6, 3, 8]                 # the active set shrinks 4 -> 3 -> 2 -> 1, and a fifth request takes a slot


def _run(mode, buckets=(1, 2, 4)):
    pytest.importorskip("fp8_paged_attn", reason="needs grouped-nf4-gemm's fp8 paged attention")
    fp8_kv = pytest.importorskip("fp8_kv", reason="needs grouped-nf4-gemm's fp8 KV")
    if not hasattr(fp8_kv, "fp8_kv_append_bt1"):
        pytest.skip("this grouped-nf4-gemm has no fused batch KV append")
    from transformers.models.qwen3.configuration_qwen3 import Qwen3Config
    from transformers.models.qwen3.modeling_qwen3 import Qwen3ForCausalLM

    from experts4bit_qlora.engines import paged_attention
    from experts4bit_qlora.engines.paged_runner import PagedModelRunner
    from experts4bit_qlora.engines.scheduler import ContinuousScheduler

    torch.manual_seed(0)
    cfg = Qwen3Config(hidden_size=256, intermediate_size=512, num_hidden_layers=2, num_attention_heads=4,
                      num_key_value_heads=2, head_dim=64, vocab_size=256, max_position_embeddings=256)
    model = Qwen3ForCausalLM(cfg).to("cuda", torch.bfloat16).eval()
    paged_attention.register(model)
    kv = Fp8PagedKV(cfg.num_hidden_layers, cfg.num_key_value_heads, cfg.head_dim, batch=4, max_tokens_per_seq=64,
                    scratch_slots=max(buckets))
    runner = PagedModelRunner(model, kv)
    runner.enable_decode_graphs(buckets, capture=(mode == "graph"), verbose=False)
    sched = ContinuousScheduler(runner=runner, max_seqs=4, kv_slots=4, chunk_tokens=64)
    rids = [sched.add_request(p, m) for p, m in zip(PROMPTS, MAX_NEW)]
    sched.run_until_idle(max_steps=500)
    out = {r.rid: list(r.out) for r in sched.done}
    return [out[r] for r in rids], kv._step_select


@needs_fp8
@pytest.mark.parametrize("mode", ["graph", "padded-eager"])
def test_the_step_selection_decodes_exactly_as_the_per_layer_selection(mode, monkeypatch):
    monkeypatch.setenv("E4B_KV_STEP_SELECT", "0")
    ref, on = _run(mode)
    assert on is False
    monkeypatch.setenv("E4B_KV_STEP_SELECT", "1")
    got, on = _run(mode)
    assert on is True and got == ref
