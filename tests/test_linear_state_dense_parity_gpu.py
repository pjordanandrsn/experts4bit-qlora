# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""The hybrid paged path against transformers, on a DENSE model, over many seeds (e4b#928).

``tests/test_linear_state_gpu.py`` (pinned by six lanes; its bytes stay) scores a tiny 4-expert MoE hybrid at ONE seed
by its worst relative logit error. On that model the statistic is a lottery: routing near-ties make it bimodal across
seeds on every kernel set, transformers' torch path included, where 5 of 10 seeds fail its own bound (lane P104,
bench/p104/RESULTS-p104.md). Its pass on the torch path is the luck of seed 0, and fla merely moves that draw.

These tests ask the same question with a statistic that is a distribution:
- a DENSE hybrid (Qwen3.5, three linear layers and one attention layer, no experts) through ``PagedModelRunner`` with
  the compact fp8 KV pool, 32-token prefill chunks and batched decode in changing row order;
- against transformers' DynamicCache prefilled in the SAME chunks;
- for EVERY one of 8 seeds: the hybrid's worst relative logit error within 2x that seed's dense all-attention
  CONTROL (the same comparison with no linear state).

Greedy-token agreement is REPORTED, not gated: with the fp8 KV, the all-attention control itself flips a greedy token
against its bf16 reference on some seeds (argmax near-ties of a tiny random model), so per-seed token equality is a
lottery too. On the A2000 the dense hybrid sits at 0.7-1.6x its control on every seed, whichever kernels transformers
resolves; a pool that never writes its state back fails the bar.

Two variants: on any CUDA card the decode attention is SDPA over the pool's own dequantized fp8 K/V (the stand-in of
``bench/p98/p98_box.py``); on an sm_89+ card it is gnf4's real fp8 paged kernel.
"""
import pytest
import torch

needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
needs_fp8 = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() < (8, 9),
    reason="the fp8 paged KV kernel needs native e4m3 (sm_89+)")

LIN, ATT = "linear_attention", "full_attention"
STEPS, CHUNK, SEEDS = 6, 32, 8


def _model(layer_types, seed):
    from transformers import Qwen3_5TextConfig
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM
    cfg = Qwen3_5TextConfig(vocab_size=256, hidden_size=128, intermediate_size=256, num_hidden_layers=len(layer_types),
                            num_attention_heads=4, num_key_value_heads=2, head_dim=64, linear_num_key_heads=2,
                            linear_num_value_heads=4, linear_key_head_dim=32, linear_value_head_dim=32,
                            linear_conv_kernel_dim=4, layer_types=list(layer_types), max_position_embeddings=512)
    torch.manual_seed(seed)
    return Qwen3_5ForCausalLM(cfg).to("cuda", torch.bfloat16).eval()


def _prompts(seed):
    g = torch.Generator().manual_seed(1000 + seed)
    return {0: torch.randint(0, 256, (37,), generator=g).tolist(), 1: torch.randint(0, 256, (70,), generator=g).tolist(),
            2: torch.randint(0, 256, (9,), generator=g).tolist()}


def _reference(model, prompts):
    from transformers.cache_utils import DynamicCache
    logits, toks = {}, {}
    with torch.no_grad():
        for s, p in prompts.items():
            cache = DynamicCache(config=model.config)
            for st in range(0, len(p), CHUNK):
                out = model(input_ids=torch.tensor([p[st:st + CHUNK]], device="cuda"), past_key_values=cache, use_cache=True)
            lg, tk = [out.logits[0, -1].float()], [int(out.logits[0, -1].argmax())]
            for _ in range(STEPS):
                out = model(input_ids=torch.tensor([[tk[-1]]], device="cuda"), past_key_values=cache, use_cache=True)
                lg.append(out.logits[0, -1].float())
                tk.append(int(out.logits[0, -1].argmax()))
            logits[s], toks[s] = lg, tk
    return logits, toks


def _paged(model, prompts, ref_tok, compact, stand_in):
    from experts4bit_qlora.engines import paged_attention
    from experts4bit_qlora.engines.fp8_paged_kv import Fp8PagedKV
    from experts4bit_qlora.engines.paged_runner import PagedModelRunner, kv_layers
    cfg = model.config
    paged_attention.register(model)
    n = kv_layers(model, cfg.num_hidden_layers) if compact else cfg.num_hidden_layers
    kv = Fp8PagedKV(n, cfg.num_key_value_heads, cfg.head_dim, batch=4, max_tokens_per_seq=128, device="cuda")
    if stand_in:
        def attention(layer, q, slots=None, sm_scale=None, window=None, sinks=None, **_):
            outs = []
            for b, slot in enumerate(slots):
                kr, vr = kv.reference_kv(layer, slot)
                outs.append(torch.nn.functional.scaled_dot_product_attention(
                    q[b][None, :, None].float(), kr.permute(1, 0, 2)[None].float(), vr.permute(1, 0, 2)[None].float(),
                    scale=sm_scale, enable_gqa=True)[0, :, 0].to(q.dtype))
            return torch.stack(outs)
        kv.attention = attention
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
    try:
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
    finally:
        model.forward = inner
    return runner, got


def _worst(got, ref):
    worst, agree, n = 0.0, 0, 0
    for s in ref:
        for a, b in zip(got[s], ref[s]):
            worst = max(worst, float((a - b).abs().max() / b.abs().max()))
            agree += int(int(a.argmax()) == int(b.argmax()))
            n += 1
    return worst, f"{agree}/{n}"


def _every_seed(stand_in):
    pytest.importorskip("transformers.models.qwen3_5", reason="needs transformers with Qwen3.5")
    rows = []
    for seed in range(SEEDS):
        prompts = _prompts(seed)
        control = _model([ATT] * 4, seed)
        c_ref, c_tok = _reference(control, prompts)
        c_runner, c_got = _paged(control, prompts, c_tok, compact=False, stand_in=stand_in)
        assert c_runner.linear_state is None
        c_worst, c_toks = _worst(c_got, c_ref)
        del control
        hybrid = _model([LIN, ATT, LIN, LIN], seed)
        h_ref, h_tok = _reference(hybrid, prompts)
        h_runner, h_got = _paged(hybrid, prompts, h_tok, compact=True, stand_in=stand_in)
        assert h_runner.linear_state is not None and h_runner.attn_layers == [1]
        h_worst, h_toks = _worst(h_got, h_ref)
        del hybrid
        rows.append((seed, c_worst, h_worst, c_toks, h_toks))
    for seed, c_worst, h_worst, c_toks, h_toks in rows:
        print(f"dense seed {seed}: control worst {c_worst:.3e} argmax agree {c_toks} | hybrid worst {h_worst:.3e} "
              f"argmax agree {h_toks} | ratio {h_worst / max(c_worst, 1e-2):.2f}")
    bad = [r[:3] for r in rows if not r[2] <= 2 * max(r[1], 1e-2)]
    assert not bad, f"seeds outside the bar (seed, control worst, hybrid worst): {bad}"


@needs_cuda
def test_the_dense_hybrid_stays_within_its_control_on_every_seed_with_the_stand_in_attention():
    pytest.importorskip("fp8_kv", reason="needs grouped-nf4-gemm's fp8 KV")
    _every_seed(stand_in=True)


@needs_fp8
def test_the_dense_hybrid_stays_within_its_control_on_every_seed_on_the_real_kernel():
    pytest.importorskip("fp8_paged_attn", reason="needs grouped-nf4-gemm's fp8 paged attention")
    _every_seed(stand_in=False)
