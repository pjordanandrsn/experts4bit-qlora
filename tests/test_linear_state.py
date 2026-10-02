# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""Per-slot linear-attention state for the paged runner (``engines/linear_state.py``), on tiny random Qwen3.5-MoE
models built in place (no download), on CPU.

* **The state oracle.** An all-linear model through the pool reproduces transformers' own DynamicCache, per sequence,
  while sequences are prefilled in chunks and decoded together in changing batches and row orders, on non-contiguous
  slots. The state arithmetic is transformers' in both, so the bar is tight (relative 1e-5) and the greedy tokens are
  equal.
* **End to end.** A hybrid model (three linear layers, one full-attention layer) through ``PagedModelRunner``
  (paged attention, the fp8 KV pool, the state pool) against DynamicCache. The fp8 KV moves the logits, so the bar
  comes from a CONTROL: the same comparison on an all-attention model of the same shape, which has no linear state.
  The hybrid model must stay within 2x the control's error, with greedy tokens equal. On CPU the decode attention is
  SDPA over the pool's own dequantized K/V (``Fp8PagedKV.reference_kv``) in place of the Triton kernel.
* **A compact KV pool.** A pool sized to the attention layers only (paged attention maps model layer -> pool layer)
  gives the same logits, bit for bit, as a pool with a layer per index.
* **Refusals.** Rows that mix sequences with and without state; a linear layer the pool cannot drive (transformers
  labels Mamba-style layers ``linear_attention`` too); an unknown state-carrying layer type; decode graphs for a model
  with linear layers. And a recycled slot starts from zero.
"""
import types

import pytest
import torch

pytest.importorskip("transformers.cache_utils", reason="needs transformers")
q35 = pytest.importorskip("transformers.models.qwen3_5_moe.modeling_qwen3_5_moe",
                          reason="needs transformers with Qwen3.5-MoE")
from transformers import Qwen3_5MoeTextConfig  # noqa: E402
from transformers.cache_utils import DynamicCache  # noqa: E402

from experts4bit_qlora.engines import linear_state, paged_attention  # noqa: E402
from experts4bit_qlora.engines.paged_runner import PagedModelRunner, kv_layer_map, kv_layers, layer_plan  # noqa: E402

LIN, ATT = "linear_attention", "full_attention"
NOMASK = {LIN: None, ATT: None}           # an all-linear model: no attention mask to build (no padding)
STEPS = 5


def _model(layer_types, seed=0):
    cfg = Qwen3_5MoeTextConfig(vocab_size=128, hidden_size=64, num_hidden_layers=len(layer_types), num_attention_heads=4,
                               num_key_value_heads=2, head_dim=32, moe_intermediate_size=32,
                               shared_expert_intermediate_size=32, num_experts=4, num_experts_per_tok=2,
                               linear_num_key_heads=2, linear_num_value_heads=4, linear_key_head_dim=16,
                               linear_value_head_dim=16, linear_conv_kernel_dim=4, layer_types=list(layer_types),
                               max_position_embeddings=256)
    torch.manual_seed(seed)
    return q35.Qwen3_5MoeForCausalLM(cfg).eval()


def _prompts():
    g = torch.Generator().manual_seed(1)
    return {0: torch.randint(0, 128, (7,), generator=g).tolist(), 1: torch.randint(0, 128, (12,), generator=g).tolist(),
            2: torch.randint(0, 128, (3,), generator=g).tolist()}


def _reference(model, prompts, mask=None):
    """transformers' DynamicCache, one sequence at a time: last-position logits and greedy tokens per step."""
    logits, toks = {}, {}
    with torch.no_grad():
        for s, p in prompts.items():
            cache = DynamicCache(config=model.config)
            kw = {} if mask is None else {"attention_mask": mask}
            out = model(input_ids=torch.tensor([p]), position_ids=torch.arange(len(p))[None], past_key_values=cache,
                        use_cache=True, **kw)
            lg, tk = [out.logits[0, -1]], [int(out.logits[0, -1].argmax())]
            for t in range(STEPS):
                out = model(input_ids=torch.tensor([[tk[-1]]]), position_ids=torch.tensor([[len(p) + t]]),
                            past_key_values=cache, use_cache=True, **kw)
                lg.append(out.logits[0, -1])
                tk.append(int(out.logits[0, -1].argmax()))
            logits[s], toks[s] = lg, tk
    return logits, toks


def _compare(got, ref):
    worst, toks_equal = 0.0, True
    for s in ref:
        assert len(got[s]) == len(ref[s]), (s, len(got[s]), len(ref[s]))
        for a, b in zip(got[s], ref[s]):
            worst = max(worst, float((a - b).abs().max() / b.abs().max()))
            toks_equal &= int(a.argmax()) == int(b.argmax())
    return worst, toks_equal


def _bound(slots):
    return paged_attention.PagedAttentionContext(kv=None, slots=list(slots), mode="decode")


def test_the_pool_reproduces_transformers_dynamic_cache():
    model = _model([LIN] * 4)
    prompts = _prompts()
    ref, ref_tok = _reference(model, prompts, mask=NOMASK)
    pool = linear_state.install(model, n_slots=8)
    assert pool is not None and model._e4b_linear_state is pool
    slot_of = {0: 5, 1: 2, 2: 7}                                   # non-contiguous slots
    got = {s: [] for s in prompts}
    with torch.no_grad():
        for s, p in prompts.items():                              # seq 1 prefilled in chunks of 5, 5, 2
            for a, b in ([(0, 5), (5, 10), (10, 12)] if s == 1 else [(0, len(p))]):
                prev = paged_attention.set_context(_bound([slot_of[s]]))
                try:
                    out = model(input_ids=torch.tensor([p[a:b]]), position_ids=torch.arange(a, b)[None], use_cache=False,
                                attention_mask=NOMASK)
                finally:
                    paged_attention.set_context(prev)
                pool.mark([slot_of[s]])
            got[s].append(out.logits[0, -1])
        for t in range(STEPS):
            active = [0, 1, 2] if t < 2 else [2, 0]                # a batch, then another row order without seq 1
            prev = paged_attention.set_context(_bound([slot_of[s] for s in active]))
            try:
                out = model(input_ids=torch.tensor([[ref_tok[s][t]] for s in active]),
                            position_ids=torch.tensor([[len(prompts[s]) + t] for s in active]), use_cache=False,
                            attention_mask=NOMASK)
            finally:
                paged_attention.set_context(prev)
            for i, s in enumerate(active):
                got[s].append(out.logits[i, -1])
        for t in range(2, STEPS):                                  # seq 1 alone
            prev = paged_attention.set_context(_bound([slot_of[1]]))
            try:
                out = model(input_ids=torch.tensor([[ref_tok[1][t]]]), position_ids=torch.tensor([[len(prompts[1]) + t]]),
                            use_cache=False, attention_mask=NOMASK)
            finally:
                paged_attention.set_context(prev)
            got[1].append(out.logits[0, -1])
    worst, toks_equal = _compare(got, ref)
    assert toks_equal and worst < 1e-5, worst


def _paged_run(model, prompts, ref_tok, pool_layers=None):
    """PagedModelRunner end to end on CPU, teacher-forced with the reference's greedy tokens. ``pool_layers`` sizes
    the KV pool (default: one layer per index)."""
    pytest.importorskip("row_pool", reason="needs grouped-nf4-gemm N-series")
    pytest.importorskip("fp8_kv", reason="needs grouped-nf4-gemm N-series")
    from experts4bit_qlora.engines.fp8_paged_kv import Fp8PagedKV
    cfg = model.config
    paged_attention.register(model)
    kv = Fp8PagedKV(pool_layers or cfg.num_hidden_layers, cfg.num_key_value_heads, cfg.head_dim, batch=4,
                    max_tokens_per_seq=64, device="cpu")

    def reference_attention(layer, q, slots=None, sm_scale=None, window=None, sinks=None, **_):
        outs = []
        for b, slot in enumerate(slots):
            kr, vr = kv.reference_kv(layer, slot)
            outs.append(torch.nn.functional.scaled_dot_product_attention(
                q[b][None, :, None].float(), kr.permute(1, 0, 2)[None].float(), vr.permute(1, 0, 2)[None].float(),
                scale=sm_scale, enable_gqa=True)[0, :, 0].to(q.dtype))
        return torch.stack(outs)

    kv.attention = reference_attention
    runner = PagedModelRunner(model, kv, device="cpu")
    last = {}
    inner = model.forward

    def keep_logits(*a, **k):
        out = inner(*a, **k)
        last["logits"] = out.logits
        return out

    model.forward = keep_logits
    slot_of = {0: 3, 1: 0, 2: 2}
    for s, p in prompts.items():
        runner.bind(s, slot_of[s], p)
    got = {s: [] for s in prompts}
    with torch.no_grad():
        for s, p in prompts.items():
            for start, take in ([(0, 5), (5, 5), (10, 2)] if s == 1 else [(0, len(p))]):
                runner.run_prefill([(s, start, take)])
            got[s].append(last["logits"][0, -1])
            runner.tokens[s][-1] = ref_tok[s][0]
        for t in range(STEPS):
            active = [0, 1, 2] if t < 2 else [2, 0]
            runner.run_decode(active)
            for i, s in enumerate(active):
                got[s].append(last["logits"][i, -1])
                runner.tokens[s][-1] = ref_tok[s][t + 1]
        for t in range(2, STEPS):
            runner.run_decode([1])
            got[1].append(last["logits"][0, -1])
            runner.tokens[1][-1] = ref_tok[1][t + 1]
    return runner, got


def test_the_paged_runner_serves_a_hybrid_model_within_its_attention_only_control():
    prompts = _prompts()
    control = _model([ATT] * 4)                                    # no linear state: only the fp8 KV moves logits
    c_ref, c_tok = _reference(control, prompts)
    c_runner, c_got = _paged_run(control, prompts, c_tok)
    assert c_runner.linear_state is None and c_runner.attn_layers == [0, 1, 2, 3]
    c_worst, c_toks = _compare(c_got, c_ref)

    hybrid = _model([LIN, LIN, LIN, ATT])
    h_ref, h_tok = _reference(hybrid, prompts)
    h_runner, h_got = _paged_run(hybrid, prompts, h_tok)
    assert h_runner.linear_state is not None and h_runner.attn_layers == [3]
    h_worst, h_toks = _compare(h_got, h_ref)
    assert c_toks and h_toks, (c_toks, h_toks)
    assert h_worst <= 2 * max(c_worst, 1e-3), (h_worst, c_worst)


def test_rows_mixing_state_and_no_state_are_refused():
    pool = linear_state.LinearStatePool(4)
    pool.conv[0], pool.rec[0], pool.kernel[0] = torch.zeros(4, 8, 4), torch.zeros(4, 2, 4, 4), 4
    pool.mark([1])
    with pytest.raises(RuntimeError, match="mix sequences with and without state"):
        pool.view(0, [1, 2])
    assert pool.view(0, [2, 3]).has_previous_state[0] is False      # none carries state: a first prompt chunk
    assert pool.view(0, [1]).has_previous_state[0] is True


def test_a_recycled_slot_starts_from_zero():
    model = _model([LIN] * 2)
    pool = linear_state.install(model, n_slots=2)
    ids = torch.tensor([[5, 9, 2, 7]])
    with torch.no_grad():
        def run(slot):
            prev = paged_attention.set_context(_bound([slot]))
            try:
                out = model(input_ids=ids, position_ids=torch.arange(4)[None], use_cache=False, attention_mask=NOMASK)
            finally:
                paged_attention.set_context(prev)
            pool.mark([slot])
            return out.logits[0, -1]
        fresh = run(0)
        run(0)                                                     # slot 0 now carries a second prompt's state
        pool.reset(0)
        assert torch.equal(run(0), fresh)                          # after reset it is a fresh sequence again


def test_unsupported_state_carrying_layers_are_refused():
    mamba_like = torch.nn.Linear(2, 2)                             # a linear_attention label with no Gated DeltaNet
    mamba_like.config = types.SimpleNamespace(layer_types=[LIN, ATT])
    with pytest.raises(NotImplementedError, match="would run without their state"):
        linear_state.install(mamba_like, n_slots=2)
    other = torch.nn.Linear(2, 2)
    other.config = types.SimpleNamespace(layer_types=["hybrid", ATT])
    with pytest.raises(NotImplementedError, match="carry state the paged runner does not keep"):
        layer_plan(other, 2, 2)
    plain = torch.nn.Linear(2, 2)                                  # no layer_types: every layer is attention
    assert layer_plan(plain, 3, 2) == ([0, 1, 2], None)


def test_decode_graphs_are_refused_for_a_model_with_linear_layers():
    model = _model([LIN, ATT])
    kv = types.SimpleNamespace(L=2, B=2, scratch=[2, 3])
    runner = PagedModelRunner(model, kv, device="cpu")
    assert runner.linear_state is not None and runner.attn_layers == [1]
    with pytest.raises(NotImplementedError, match="not captured yet"):
        runner.enable_decode_graphs(buckets=(1, 2))


def test_a_compact_kv_pool_matches_one_layer_per_index_bit_for_bit():
    prompts = _prompts()
    model = _model([LIN, ATT, LIN, ATT])
    _, ref_tok = _reference(model, prompts)
    full_runner, full = _paged_run(model, prompts, ref_tok)
    model = _model([LIN, ATT, LIN, ATT])                           # a fresh copy: the same seed, the same weights
    assert kv_layers(model, 4) == 2
    compact_runner, compact = _paged_run(model, prompts, ref_tok, pool_layers=2)
    assert full_runner.ctx.layer_map == {} and compact_runner.ctx.layer_map == {1: 0, 3: 1}
    for s in prompts:
        for a, b in zip(full[s], compact[s]):
            assert torch.equal(a, b)


def test_the_kv_layer_map_and_the_pool_size():
    assert kv_layer_map([3, 7], 8) == {} and kv_layer_map([3, 7], 2) == {3: 0, 7: 1}
    assert kv_layer_map([0, 1, 2], 3) == {}                        # a plain model: identity
    with pytest.raises(ValueError, match="size it to 2"):
        kv_layer_map([3, 7], 5)
    plain = torch.nn.Linear(2, 2)
    assert kv_layers(plain, 6) == 6
    sliding = torch.nn.Linear(2, 2)                                # every layer attends (gpt-oss style): no compaction
    sliding.config = types.SimpleNamespace(layer_types=["sliding_attention", ATT, "sliding_attention"])
    assert kv_layers(sliding, 3) == 3


def test_kv_geometry_reads_a_composite_configs_text_config():
    from experts4bit_qlora.serve_paged import _kv_geometry
    vl = types.SimpleNamespace(text_config=types.SimpleNamespace(num_key_value_heads=2, head_dim=256, hidden_size=2048,
                                                                 num_attention_heads=16))
    assert _kv_geometry(vl) == (2, 256)
