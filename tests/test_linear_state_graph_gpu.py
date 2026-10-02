# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""The per-slot linear-attention state under REAL CUDA-graph capture, on any CUDA card (no fp8 KV involved).

An all-linear dense Qwen3.5 model (Gated DeltaNet layers only, dense MLP) keeps attention, the fp8 pool and the MoE
engine out of the graph, so what is captured is exactly the linear wrapper's gather / scatter and transformers' state
update. Two ways a graph can address the pool, each replayed against the same steps run eagerly, bit for bit:

* FIXED slots (``bench/p39/step_decomp.py``'s B=1 and batched lanes): the slot list is constant across replays, so
  baking it is right; what must not happen is a host-to-device copy inside the capture. ``LinearStatePool._index``
  builds the index tensor during the eager warm-up step and the capture reuses it.
* A BUCKET SELECTOR (``PagedModelRunner.enable_decode_graphs``, #907): the graph is captured on scratch slots, and each
  replay addresses whatever slot ids were written into the selector, in whatever order.

Runs on sm_86 (the NAS A2000) as well as the 5090; skips without CUDA.
"""
import types

import pytest
import torch

needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device for graph capture")

LIN = "linear_attention"
NOMASK = {"linear_attention": None, "full_attention": None}
PROMPTS = {0: [5, 9, 2, 7, 1, 33, 8], 1: [3, 3, 8, 120], 2: [11, 4, 6, 6, 90], 5: [2, 2], 6: [7], 7: [1, 4, 4]}
STEPS = 6


def _model():
    pytest.importorskip("transformers.models.qwen3_5", reason="needs transformers with Qwen3.5")
    from transformers import Qwen3_5TextConfig
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM
    torch.manual_seed(0)
    cfg = Qwen3_5TextConfig(vocab_size=256, hidden_size=128, intermediate_size=256, num_hidden_layers=3,
                            num_attention_heads=4, num_key_value_heads=2, head_dim=32, linear_num_key_heads=2,
                            linear_num_value_heads=4, linear_key_head_dim=32, linear_value_head_dim=32,
                            linear_conv_kernel_dim=4, layer_types=[LIN] * 3, max_position_embeddings=256)
    return Qwen3_5ForCausalLM(cfg).to("cuda", torch.bfloat16).eval()


def _ctx(slots, sel=None):
    from experts4bit_qlora.engines import paged_attention
    ctx = paged_attention.PagedAttentionContext(kv=None, slots=list(slots), mode="decode")
    if sel is not None:
        ctx.kv = types.SimpleNamespace(_g_sel=sel)                 # a bound decode-graph bucket's selector
    return ctx


def _forward(model, ctx, ids, pos):
    from experts4bit_qlora.engines import paged_attention
    prev = paged_attention.set_context(ctx)
    try:
        return model(input_ids=ids, position_ids=pos, use_cache=False, attention_mask=NOMASK).logits[:, -1]
    finally:
        paged_attention.set_context(prev)


def _prefilled():
    from experts4bit_qlora.engines import linear_state
    model = _model()
    pool = linear_state.install(model, n_slots=8)
    with torch.no_grad():
        for slot, p in PROMPTS.items():
            _forward(model, _ctx([slot]), torch.tensor([p], device="cuda"),
                     torch.arange(len(p), device="cuda")[None])
            pool.mark([slot])
    return model, pool


def _step_inputs(rows, step):
    ids = torch.tensor([[10 + 7 * step + r] for r in rows], device="cuda")
    pos = torch.tensor([[len(PROMPTS[r]) + step] for r in rows], device="cuda")
    return ids, pos


def _eager(rows_per_step):
    model, pool = _prefilled()
    out = []
    with torch.no_grad():
        for step, rows in enumerate(rows_per_step):
            out.append(_forward(model, _ctx(rows), *_step_inputs(rows, step)).clone())
    return out, pool


def _capture(model, ctx, ids, pos):
    """Record one step (a warm-up forward has already run on the same inputs); nothing executes until replay."""
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.no_grad(), torch.cuda.graph(g):
        out = _forward(model, ctx, ids, pos)
    torch.cuda.synchronize()
    return g, out


@needs_cuda
def test_a_fixed_slot_graph_replays_exactly_as_the_eager_steps():
    rows = [2, 0, 1]
    ref, ref_pool = _eager([rows] * STEPS)
    model, pool = _prefilled()
    with torch.no_grad():
        got = [_forward(model, _ctx(rows), *_step_inputs(rows, 0)).clone()]   # eager warm-up: builds the index
        ids, pos = _step_inputs(rows, 1)
        g, out = _capture(model, _ctx(rows), ids, pos)                         # records; mutates nothing
        for step in range(1, STEPS):
            nid, npos = _step_inputs(rows, step)
            ids.copy_(nid)
            pos.copy_(npos)
            g.replay()
            got.append(out.clone())
    torch.cuda.synchronize()
    for a, b in zip(got, ref):
        assert torch.equal(a, b)
    for layer in ref_pool.conv:
        assert torch.equal(pool.conv[layer], ref_pool.conv[layer]) and torch.equal(pool.rec[layer], ref_pool.rec[layer])


@needs_cuda
def test_a_bucket_graph_captured_on_scratch_slots_replays_the_slots_its_selector_names():
    order = [[2, 0, 1], [2, 0, 1], [1, 2, 0], [0, 1, 2], [2, 0, 1]]   # the rows move between replays
    ref, ref_pool = _eager(order)
    model, pool = _prefilled()
    sel = torch.tensor([5, 6, 7], device="cuda")                    # capture on scratch slots
    ids, pos = _step_inputs([5, 6, 7], 0)
    with torch.no_grad():
        _forward(model, _ctx([5, 6, 7], sel=sel), ids, pos)         # eager warm-up, on the scratch rows only
    scratch_before = {layer: (pool.conv[layer][5:].clone(), pool.rec[layer][5:].clone()) for layer in pool.conv}
    g, out = _capture(model, _ctx([5, 6, 7], sel=sel), ids, pos)
    got = []
    with torch.no_grad():
        for step, rows in enumerate(order):
            sel.copy_(torch.tensor(rows, device="cuda"))
            nid, npos = _step_inputs(rows, step)
            ids.copy_(nid)
            pos.copy_(npos)
            g.replay()
            got.append(out.clone())
    torch.cuda.synchronize()
    for step, (a, b) in enumerate(zip(got, ref)):
        rows = order[step]
        assert torch.equal(a, b), f"step {step}, rows {rows}"
    for layer in ref_pool.conv:
        assert torch.equal(pool.conv[layer][:3], ref_pool.conv[layer][:3])
        assert torch.equal(pool.rec[layer][:3], ref_pool.rec[layer][:3])
        c, r = scratch_before[layer]                                # the capture itself wrote nothing
        assert torch.equal(pool.conv[layer][5:], c) and torch.equal(pool.rec[layer][5:], r)
