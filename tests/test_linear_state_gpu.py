# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""The hybrid paged path on a GPU, through the REAL fp8 paged decode kernel (gnf4's Triton), against transformers'
DynamicCache on the same weights. ``tests/test_linear_state.py`` pins the same path on CPU with SDPA standing in for
the kernel; this file is where the kernel, the per-slot linear state on CUDA tensors and the compact KV pool meet.

Tiny random Qwen3.5-MoE models built in place, in bf16. The bar is a CONTROL: an all-attention model of the same
shape through the same comparison, whose only error is the fp8 KV. The hybrid (three linear layers, one attention
layer, a compact pool) must stay within 2x the control's worst relative logit error, with greedy tokens equal.

Skips without an sm_89+ card (the fp8 KV quantizes to e4m3, Triton ``fp8e4nv``); lane P97 runs it on the card as its
premise.
"""
import pytest
import torch

needs_fp8 = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() < (8, 9),
    reason="the fp8 paged KV needs native e4m3 (sm_89+)")

LIN, ATT = "linear_attention", "full_attention"
STEPS = 6


def _model(layer_types):
    from transformers import Qwen3_5MoeTextConfig
    from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeForCausalLM
    cfg = Qwen3_5MoeTextConfig(vocab_size=256, hidden_size=128, num_hidden_layers=len(layer_types), num_attention_heads=4,
                               num_key_value_heads=2, head_dim=64, moe_intermediate_size=64,
                               shared_expert_intermediate_size=64, num_experts=4, num_experts_per_tok=2,
                               linear_num_key_heads=2, linear_num_value_heads=4, linear_key_head_dim=32,
                               linear_value_head_dim=32, linear_conv_kernel_dim=4, layer_types=list(layer_types),
                               max_position_embeddings=512)
    torch.manual_seed(0)
    return Qwen3_5MoeForCausalLM(cfg).to("cuda", torch.bfloat16).eval()


def _prompts():
    g = torch.Generator().manual_seed(3)
    return {0: torch.randint(0, 256, (37,), generator=g).tolist(), 1: torch.randint(0, 256, (70,), generator=g).tolist(),
            2: torch.randint(0, 256, (9,), generator=g).tolist()}


def _reference(model, prompts):
    from transformers.cache_utils import DynamicCache
    logits, toks = {}, {}
    with torch.no_grad():
        for s, p in prompts.items():
            cache = DynamicCache(config=model.config)
            out = model(input_ids=torch.tensor([p], device="cuda"), past_key_values=cache, use_cache=True)
            lg, tk = [out.logits[0, -1].float()], [int(out.logits[0, -1].argmax())]
            for _ in range(STEPS):
                out = model(input_ids=torch.tensor([[tk[-1]]], device="cuda"), past_key_values=cache, use_cache=True)
                lg.append(out.logits[0, -1].float())
                tk.append(int(out.logits[0, -1].argmax()))
            logits[s], toks[s] = lg, tk
    return logits, toks


def _paged(model, prompts, ref_tok, compact):
    from experts4bit_qlora.engines import paged_attention
    from experts4bit_qlora.engines.fp8_paged_kv import Fp8PagedKV
    from experts4bit_qlora.engines.paged_runner import PagedModelRunner, kv_layers
    cfg = model.config
    paged_attention.register(model)
    n = kv_layers(model, cfg.num_hidden_layers) if compact else cfg.num_hidden_layers
    kv = Fp8PagedKV(n, cfg.num_key_value_heads, cfg.head_dim, batch=4, max_tokens_per_seq=128, device="cuda")
    runner = PagedModelRunner(model, kv, device="cuda")
    last = {}
    inner = model.forward

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
            for start in range(0, len(p), 32):                     # 32-token prefill chunks
                runner.run_prefill([(s, start, min(32, len(p) - start))])
            got[s].append(last["logits"][0, -1].float())
            runner.tokens[s][-1] = ref_tok[s][0]
        for t in range(STEPS):
            active = [0, 1, 2] if t % 2 == 0 else [2, 1, 0]        # batched decode, changing row order
            runner.run_decode(active)
            for i, s in enumerate(active):
                got[s].append(last["logits"][i, -1].float())
                runner.tokens[s][-1] = ref_tok[s][t + 1]
    return runner, got


def _worst(got, ref):
    worst, toks = 0.0, True
    for s in ref:
        for a, b in zip(got[s], ref[s]):
            worst = max(worst, float((a - b).abs().max() / b.abs().max()))
            toks &= int(a.argmax()) == int(b.argmax())
    return worst, toks


@needs_fp8
def test_the_hybrid_paged_path_on_the_real_kernel_stays_within_its_control():
    pytest.importorskip("fp8_paged_attn", reason="needs grouped-nf4-gemm's fp8 paged attention")
    prompts = _prompts()
    control = _model([ATT] * 4)
    c_ref, c_tok = _reference(control, prompts)
    c_runner, c_got = _paged(control, prompts, c_tok, compact=False)
    assert c_runner.linear_state is None
    c_worst, c_toks = _worst(c_got, c_ref)

    hybrid = _model([LIN, ATT, LIN, LIN])
    h_ref, h_tok = _reference(hybrid, prompts)
    h_runner, h_got = _paged(hybrid, prompts, h_tok, compact=True)
    assert h_runner.linear_state is not None and h_runner.attn_layers == [1]
    assert h_runner.kv.L == 1 and h_runner.ctx.layer_map == {1: 0}
    assert h_runner.linear_state.conv and all(t.is_cuda for t in h_runner.linear_state.conv.values())
    h_worst, h_toks = _worst(h_got, h_ref)
    print(f"control worst {c_worst:.3e} tokens {c_toks} | hybrid worst {h_worst:.3e} tokens {h_toks}")
    assert c_toks and h_toks, (c_toks, h_toks)
    assert h_worst <= 2 * max(c_worst, 1e-2), (h_worst, c_worst)
