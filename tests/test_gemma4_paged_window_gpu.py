# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""Gemma-4's sliding window through e4b's paged path on a GPU, against transformers' forward on the same weights.

A tiny DENSE Gemma-4: five sliding layers at window 16 and one full layer, in bf16. Dense, because a tiny MoE's routing
near-ties make a per-seed statistic a lottery (lane P104). Three 40-token prompts, so the window binds from position
16, are prefilled in 16-token chunks through ``PagedModelRunner`` with the fp8 KV pool and decoded together for 6
teacher-forced steps. The reference is the same model run with a ``DynamicCache`` and no paged context bound,
prefilled in the same 16-token chunks; e4b's unbound fallback now carries the window and the last-key alignment (#966).

The bar comes from a CONTROL: the same comparison on the same weights with a window that never binds (4096), whose
only error is the fp8 KV. With the window binding, on every one of 8 seeds, the mean KL(reference || paged) over every
step (the chunked prefill's last position and the decode steps), on fp32 log-probs, stays within 2x the control's
(floor 1e-4). The prefill step alone is reported, not gated: in bf16 it moves by 5e-5 to 9e-4 between the two SDPA
paths in both arms (NAS A2000), one position per prompt. On the A2000, a stand-in that ignores the window fails the bar
at 13.5-27x on every seed.

Two variants: SDPA standing in for the decode attention over the pool's own fp8 bytes and honouring the window (any CUDA
card), and gnf4's real fp8 paged kernel (sm_89+).
"""
import pytest
import torch

needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
needs_fp8 = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() < (8, 9),
    reason="the fp8 paged KV kernel needs native e4m3 (sm_89+)")

P, STEPS, CHUNK, SEEDS, WINDOW = 40, 6, 16, 8, 16


def _model(seed, window):
    from transformers import Gemma4TextConfig
    from transformers.models.gemma4.modeling_gemma4 import Gemma4ForCausalLM
    cfg = Gemma4TextConfig(vocab_size=256, hidden_size=128, intermediate_size=256, num_hidden_layers=6,
                           num_attention_heads=4, num_key_value_heads=2, head_dim=32, global_head_dim=64,
                           num_global_key_value_heads=1, layer_types=["sliding_attention"] * 5 + ["full_attention"],
                           sliding_window=window, enable_moe_block=False, attention_k_eq_v=True,
                           hidden_size_per_layer_input=0, vocab_size_per_layer_input=256, max_position_embeddings=512)
    torch.manual_seed(seed)
    return Gemma4ForCausalLM(cfg).to("cuda", torch.bfloat16).eval()


def _prompts(seed):
    g = torch.Generator().manual_seed(1000 + seed)
    return [torch.randint(0, 256, (P + STEPS + 1,), generator=g).tolist() for _ in range(3)]


def _reference(model, prompts):
    from transformers.cache_utils import DynamicCache
    out_lp = []
    with torch.no_grad():
        for w in prompts:
            cache = DynamicCache(config=model.config)
            for s in range(0, P, CHUNK):                    # chunk-matched to the paged prefill (lane P104's lesson)
                out = model(input_ids=torch.tensor([w[s:min(s + CHUNK, P)]], device="cuda"), past_key_values=cache,
                            use_cache=True)
            rows = [out.logits[0, -1].float().log_softmax(-1)]
            for t in range(STEPS):
                out = model(input_ids=torch.tensor([[w[P + t]]], device="cuda"), past_key_values=cache, use_cache=True)
                rows.append(out.logits[0, -1].float().log_softmax(-1))
            out_lp.append(torch.stack(rows))
    return out_lp


def _paged(model, prompts, stand_in):
    from experts4bit_qlora.engines.fp8_paged_kv import Fp8PagedKV
    from experts4bit_qlora.engines.paged_runner import PagedModelRunner, kv_layers
    from experts4bit_qlora.serve_paged import _kv_geometry
    hkv, hd = _kv_geometry(model.config)
    kv = Fp8PagedKV(kv_layers(model, 6), hkv, hd, batch=len(prompts), max_tokens_per_seq=P + STEPS + 8, device="cuda")
    windows_seen = set()
    kernel = kv.attention
    if stand_in:
        def kernel(layer, q, slots=None, sm_scale=None, window=None, sinks=None, **_):
            outs = []
            for b, slot in enumerate(slots):
                kr, vr = kv.reference_kv(layer, slot)
                if window:
                    kr, vr = kr[-window:], vr[-window:]
                outs.append(torch.nn.functional.scaled_dot_product_attention(
                    q[b][None, :, None].float(), kr.permute(1, 0, 2)[None].float(),
                    vr.permute(1, 0, 2)[None].float(), scale=sm_scale, enable_gqa=True)[0, :, 0].to(q.dtype))
            return torch.stack(outs)

    def counted(layer, q, *a, window=None, **kw):
        windows_seen.add((int(layer), window))
        return kernel(layer, q, *a, window=window, **kw)

    kv.attention = counted
    runner = PagedModelRunner(model, kv, device="cuda")
    last, inner = {}, model.forward

    def keep(*a, **k):
        o = inner(*a, **k)
        last["logits"] = o.logits
        return o

    model.forward = keep
    lps = [[] for _ in prompts]
    try:
        for i, w in enumerate(prompts):
            runner.bind(i, i, w[:P])
        with torch.no_grad():
            for i in range(len(prompts)):
                for start in range(0, P, CHUNK):
                    runner.run_prefill([(i, start, min(CHUNK, P - start))])
                lps[i].append(last["logits"][0, -1].float().log_softmax(-1))
                runner.tokens[i][-1] = prompts[i][P]
            rows = list(range(len(prompts)))
            for t in range(STEPS):
                runner.run_decode(rows)
                lg = last["logits"][:, -1].float().log_softmax(-1)
                for i in rows:
                    lps[i].append(lg[i])
                    runner.tokens[i][-1] = prompts[i][P + t + 1]
    finally:
        model.forward = inner
    return [torch.stack(x) for x in lps], windows_seen


def _kls(ref, got):
    """(prefill-step mean KL, all-steps mean KL) of KL(ref || got) over the prompts."""
    pre, every = [], []
    for r, g in zip(ref, got):
        kl = (r.exp() * (r - g)).sum(-1)
        pre.append(float(kl[0]))
        every += kl.tolist()
    return sum(pre) / len(pre), sum(every) / len(every)


def _every_seed(stand_in):
    pytest.importorskip("transformers.models.gemma4", reason="needs transformers with Gemma-4")
    pytest.importorskip("fp8_kv", reason="needs grouped-nf4-gemm's fp8 KV")
    from experts4bit_qlora.engines import paged_attention
    rows = []
    for seed in range(SEEDS):
        prompts = _prompts(seed)
        res = {}
        for name, window in (("control", 4096), ("window", WINDOW)):
            model = _model(seed, window)
            paged_attention.register(model)
            ref = _reference(model, prompts)
            got, seen = _paged(model, prompts, stand_in)
            res[name] = _kls(ref, got) + (seen,)
            del model
        rows.append((seed, res))
    for seed, res in rows:
        (cp, cd, _), (wp, wd, seen) = res["control"], res["window"]
        print(f"gemma4 window seed {seed}: control prefill {cp:.2e} all {cd:.3e} | window prefill {wp:.2e} all "
              f"{wd:.3e} | ratio {wd / max(cd, 1e-4):.2f}")
        assert {w for layer, w in seen if layer < 5} == {WINDOW} and {w for layer, w in seen if layer == 5} == {0}, seen
    bad = [(s, r["window"][1], r["control"][1]) for s, r in rows if not r["window"][1] <= 2 * max(r["control"][1], 1e-4)]
    assert not bad, f"seeds outside the bar (seed, window KL, control KL): {bad}"


@needs_cuda
def test_gemma4s_window_through_the_paged_path_with_the_stand_in_attention():
    _every_seed(stand_in=True)


@needs_fp8
def test_gemma4s_window_through_the_paged_path_on_the_real_kernel():
    pytest.importorskip("fp8_paged_attn", reason="needs grouped-nf4-gemm's fp8 paged attention")
    _every_seed(stand_in=False)
