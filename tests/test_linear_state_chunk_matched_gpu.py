# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""The per-slot linear state against transformers' DynamicCache at the SAME prefill chunking (e4b#928).

``tests/test_linear_state_gpu.py`` compares the paged path, prefilled in 32-token chunks, with a SINGLE-call
transformers forward. On transformers' torch path a split prefill is (nearly) free, so that comparison measures e4b.
flash-linear-attention's chunk gated delta rule is not invariant to where a prompt is split, and transformers' own
chunked prefill drifts from its single prefill by the same amount (lane P103, bench/p103/RESULTS-p103.md). So with fla
installed the single-call reference charges the KERNEL's split-variance to the code under test. These tests compare
like with like -- the reference prefilled in the same 32-token chunks -- so they read e4b whichever kernels
transformers resolves:

- on any CUDA card: an all-linear tiny Qwen3.5-MoE (no attention, no fp8), the pool driven as ``run_prefill`` /
  ``run_decode`` drive it, one sequence per decode step, must equal transformers' chunk-matched DynamicCache BIT FOR
  BIT at every step;
- on an sm_89+ card: the hybrid (three linear layers, one attention layer, the compact fp8 pool) must stay within 2x
  its all-attention control's worst relative logit error against chunk-matched references, greedy tokens equal (the
  bar of test_linear_state_gpu.py, with the reference chunking fixed).

The files test_linear_state_gpu.py, test_hybrid_decode_graphs_gpu.py and test_linear_state_graph_gpu.py stay as they
are: six lanes pin their bytes.
"""
import pytest
import torch

needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
needs_fp8 = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() < (8, 9),
    reason="the fp8 paged KV needs native e4m3 (sm_89+)")

LIN, ATT = "linear_attention", "full_attention"
STEPS = 6
CHUNK = 32
NOMASK = {LIN: None, ATT: None}


def _model(layer_types, seed=0):
    from transformers import Qwen3_5MoeTextConfig
    from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeForCausalLM
    cfg = Qwen3_5MoeTextConfig(vocab_size=256, hidden_size=128, num_hidden_layers=len(layer_types), num_attention_heads=4,
                               num_key_value_heads=2, head_dim=64, moe_intermediate_size=64,
                               shared_expert_intermediate_size=64, num_experts=4, num_experts_per_tok=2,
                               linear_num_key_heads=2, linear_num_value_heads=4, linear_key_head_dim=32,
                               linear_value_head_dim=32, linear_conv_kernel_dim=4, layer_types=list(layer_types),
                               max_position_embeddings=512)
    torch.manual_seed(seed)
    return Qwen3_5MoeForCausalLM(cfg).to("cuda", torch.bfloat16).eval()


def _prompts(seed=3):
    g = torch.Generator().manual_seed(seed)
    return {0: torch.randint(0, 256, (37,), generator=g).tolist(), 1: torch.randint(0, 256, (70,), generator=g).tolist(),
            2: torch.randint(0, 256, (9,), generator=g).tolist()}


def _reference_chunked(model, prompts, all_linear):
    """transformers' DynamicCache, prefilled in CHUNK-token chunks (the paged path's chunking), then STEPS greedy steps.
    An all-linear cache cannot answer get_seq_length, so positions travel explicitly there."""
    from transformers.cache_utils import DynamicCache
    logits, toks = {}, {}
    with torch.no_grad():
        for s, p in prompts.items():
            cache = DynamicCache(config=model.config)

            def call(ids, start):
                kw = {}
                if all_linear:
                    pos = torch.arange(start, start + len(ids), device="cuda")
                    kw = dict(position_ids=pos[None], cache_position=pos, attention_mask=NOMASK)
                return model(input_ids=torch.tensor([ids], device="cuda"), past_key_values=cache, use_cache=True, **kw)
            for st in range(0, len(p), CHUNK):
                out = call(p[st:st + CHUNK], st)
            lg, tk = [out.logits[0, -1].float()], [int(out.logits[0, -1].argmax())]
            for t in range(STEPS):
                out = call([tk[-1]], len(p) + t)
                lg.append(out.logits[0, -1].float())
                tk.append(int(out.logits[0, -1].argmax()))
            logits[s], toks[s] = lg, tk
    return logits, toks


def _worst(got, ref):
    worst, toks = 0.0, True
    for s in ref:
        for a, b in zip(got[s], ref[s]):
            worst = max(worst, float((a - b).abs().max() / b.abs().max()))
            toks &= int(a.argmax()) == int(b.argmax())
    return worst, toks


@needs_cuda
@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_an_all_linear_pool_equals_transformers_prefilled_in_the_same_chunks(seed):
    pytest.importorskip("transformers.models.qwen3_5_moe", reason="needs transformers with Qwen3.5-MoE")
    from experts4bit_qlora.engines import linear_state, paged_attention
    model = _model([LIN] * 3, seed)
    prompts = _prompts(100 + seed)
    ref, ref_tok = _reference_chunked(model, prompts, all_linear=True)
    pool = linear_state.install(model, n_slots=4)
    slot_of = {0: 2, 1: 0, 2: 3}

    def fwd(slots, mode, ids, pos):
        prev = paged_attention.set_context(paged_attention.PagedAttentionContext(kv=None, slots=slots, mode=mode))
        try:
            return model(input_ids=ids, position_ids=pos, use_cache=False, attention_mask=NOMASK).logits
        finally:
            paged_attention.set_context(prev)

    got = {s: [] for s in prompts}
    with torch.no_grad():
        for s, p in prompts.items():
            pool.reset(slot_of[s])
            for st in range(0, len(p), CHUNK):                       # run_prefill's chunks: mark after each
                ids = torch.tensor([p[st:st + CHUNK]], device="cuda")
                lg = fwd([slot_of[s]], "prefill", ids, torch.arange(st, st + ids.shape[1], device="cuda")[None])
                pool.mark([slot_of[s]])
            got[s].append(lg[0, -1].float())
        for t in range(STEPS):
            for s in (2, 0, 1):                                      # one sequence per step, out of prompt order
                lg = fwd([slot_of[s]], "decode", torch.tensor([[ref_tok[s][t]]], device="cuda"),
                         torch.tensor([[len(prompts[s]) + t]], device="cuda"))
                pool.mark([slot_of[s]])
                got[s].append(lg[0, -1].float())
    for s in prompts:
        for i, (a, b) in enumerate(zip(got[s], ref[s])):
            assert torch.equal(a, b), (f"seq {s} step {i}: the pool differs from transformers' chunk-matched cache "
                                       f"(maxabs {(a - b).abs().max():.3e})")


@needs_fp8
def test_the_hybrid_paged_path_stays_within_its_control_against_chunk_matched_references():
    pytest.importorskip("fp8_paged_attn", reason="needs grouped-nf4-gemm's fp8 paged attention")
    from experts4bit_qlora.engines import paged_attention
    from experts4bit_qlora.engines.fp8_paged_kv import Fp8PagedKV
    from experts4bit_qlora.engines.paged_runner import PagedModelRunner, kv_layers
    prompts = _prompts()

    def paged(model, ref_tok, compact):
        cfg = model.config
        paged_attention.register(model)
        n = kv_layers(model, cfg.num_hidden_layers) if compact else cfg.num_hidden_layers
        kv = Fp8PagedKV(n, cfg.num_key_value_heads, cfg.head_dim, batch=4, max_tokens_per_seq=128, device="cuda")
        runner = PagedModelRunner(model, kv, device="cuda")
        last, inner = {}, model.forward

        def keep(*a, **k):
            out = inner(*a, **k)
            last["logits"] = out.logits
            return out

        model.forward = keep
        slot_of = {0: 2, 1: 0, 2: 3}
        for s, p in prompts.items():
            runner.bind(s, slot_of[s], p)
        got = {s: [] for s in prompts}
        with torch.no_grad():
            for s, p in prompts.items():
                for start in range(0, len(p), CHUNK):
                    runner.run_prefill([(s, start, min(CHUNK, len(p) - start))])
                got[s].append(last["logits"][0, -1].float())
                runner.tokens[s][-1] = ref_tok[s][0]
            for t in range(STEPS):
                active = [0, 1, 2] if t % 2 == 0 else [2, 1, 0]
                runner.run_decode(active)
                for i, s in enumerate(active):
                    got[s].append(last["logits"][i, -1].float())
                    runner.tokens[s][-1] = ref_tok[s][t + 1]
        return runner, got

    control = _model([ATT] * 4)
    c_ref, c_tok = _reference_chunked(control, prompts, all_linear=False)
    c_runner, c_got = paged(control, c_tok, compact=False)
    assert c_runner.linear_state is None
    c_worst, c_toks = _worst(c_got, c_ref)
    hybrid = _model([LIN, ATT, LIN, LIN])
    h_ref, h_tok = _reference_chunked(hybrid, prompts, all_linear=False)
    h_runner, h_got = paged(hybrid, h_tok, compact=True)
    assert h_runner.linear_state is not None and h_runner.attn_layers == [1]
    h_worst, h_toks = _worst(h_got, h_ref)
    print(f"chunk-matched: control worst {c_worst:.3e} tokens {c_toks} | hybrid worst {h_worst:.3e} tokens {h_toks}")
    assert c_toks and h_toks, (c_toks, h_toks)
    assert h_worst <= 2 * max(c_worst, 1e-2), (h_worst, c_worst)
