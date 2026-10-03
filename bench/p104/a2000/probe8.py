"""probe7 with an optional DENSE model (MODEL=dense: Qwen3.5 text, no experts) and 10 seeds: is the hybrid metric's
bimodality MoE routing near-ties? Original question: does the fp8 hybrid parity test's extra error under fla reproduce on sm_86 with the stand-in attention (SDPA over the
pool's dequantized fp8 K/V, as p98_box's --stand-in-attention), and which piece drives it? test_linear_state_gpu.py's
models ([ATT]*4 control, [LIN, ATT, LIN, LIN] hybrid) through PagedModelRunner, 32-token prefill chunks; references
from transformers' DynamicCache. Variants (worst relative logit error, control | hybrid):
  V1 batched decode in changing row order (the test)  vs the chunk-matched reference
  V2 one sequence per decode step                      vs the chunk-matched reference
  V3 batched decode (the test)                         vs the single-call reference
over 5 seeds (seed 0 = the test's)."""
import statistics
import torch

LIN, ATT = "linear_attention", "full_attention"
STEPS, CHUNK = 6, 32


import os
DENSE = os.environ.get("MODEL") == "dense"
NSEEDS = int(os.environ.get("NSEEDS", "10"))


def model(layer_types, seed):
    if DENSE:
        from transformers import Qwen3_5TextConfig
        from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM
        cfg = Qwen3_5TextConfig(vocab_size=256, hidden_size=128, intermediate_size=256, num_hidden_layers=len(layer_types),
                                num_attention_heads=4, num_key_value_heads=2, head_dim=64, linear_num_key_heads=2,
                                linear_num_value_heads=4, linear_key_head_dim=32, linear_value_head_dim=32,
                                linear_conv_kernel_dim=4, layer_types=list(layer_types), max_position_embeddings=512)
        torch.manual_seed(seed)
        return Qwen3_5ForCausalLM(cfg).to("cuda", torch.bfloat16).eval()
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


def prompts(seed):
    g = torch.Generator().manual_seed(seed)
    return {0: torch.randint(0, 256, (37,), generator=g).tolist(), 1: torch.randint(0, 256, (70,), generator=g).tolist(),
            2: torch.randint(0, 256, (9,), generator=g).tolist()}


def reference(m, ps, chunk):
    from transformers.cache_utils import DynamicCache
    lg, tk = {}, {}
    with torch.no_grad():
        for s, p in ps.items():
            cache = DynamicCache(config=m.config)
            step = chunk or len(p)
            for st in range(0, len(p), step):
                out = m(input_ids=torch.tensor([p[st:st + step]], device="cuda"), past_key_values=cache, use_cache=True)
            lg[s], tk[s] = [out.logits[0, -1].float()], [int(out.logits[0, -1].argmax())]
            for _ in range(STEPS):
                out = m(input_ids=torch.tensor([[tk[s][-1]]], device="cuda"), past_key_values=cache, use_cache=True)
                lg[s].append(out.logits[0, -1].float())
                tk[s].append(int(out.logits[0, -1].argmax()))
    return lg, tk


def paged(m, ps, ref_tok, compact, batched):
    from experts4bit_qlora.engines import paged_attention
    from experts4bit_qlora.engines.fp8_paged_kv import Fp8PagedKV
    from experts4bit_qlora.engines.paged_runner import PagedModelRunner, kv_layers
    cfg = m.config
    paged_attention.register(m)
    n = kv_layers(m, cfg.num_hidden_layers) if compact else cfg.num_hidden_layers
    kv = Fp8PagedKV(n, cfg.num_key_value_heads, cfg.head_dim, batch=4, max_tokens_per_seq=128, device="cuda")

    def stand_in(layer, q, slots=None, sm_scale=None, window=None, sinks=None, **_):
        outs = []
        for b, slot in enumerate(slots):
            kr, vr = kv.reference_kv(layer, slot)
            outs.append(torch.nn.functional.scaled_dot_product_attention(
                q[b][None, :, None].float(), kr.permute(1, 0, 2)[None].float(), vr.permute(1, 0, 2)[None].float(),
                scale=sm_scale, enable_gqa=True)[0, :, 0].to(q.dtype))
        return torch.stack(outs)

    kv.attention = stand_in
    runner = PagedModelRunner(m, kv, device="cuda")
    last, inner = {}, m.forward

    def keep(*a, **k):
        out = inner(*a, **k)
        last["logits"] = out.logits
        return out

    m.forward = keep
    slot_of = {0: 2, 1: 0, 2: 3}
    for s, p in ps.items():
        runner.bind(s, slot_of[s], p)
    got = {s: [] for s in ps}
    with torch.no_grad():
        for s, p in ps.items():
            for start in range(0, len(p), CHUNK):
                runner.run_prefill([(s, start, min(CHUNK, len(p) - start))])
            got[s].append(last["logits"][0, -1].float())
            runner.tokens[s][-1] = ref_tok[s][0]
        for t in range(STEPS):
            groups = [[0, 1, 2] if t % 2 == 0 else [2, 1, 0]] if batched else [[2], [0], [1]]
            for active in groups:
                runner.run_decode(active)
                for i, s in enumerate(active):
                    got[s].append(last["logits"][i, -1].float())
                    runner.tokens[s][-1] = ref_tok[s][t + 1]
    m.forward = inner
    return got


def worst(got, ref):
    w, toks = 0.0, True
    for s in ref:
        for a, b in zip(got[s], ref[s]):
            w = max(w, float((a - b).abs().max() / b.abs().max()))
            toks &= int(a.argmax()) == int(b.argmax())
    return w, toks


print("MODEL", "dense" if DENSE else "moe", "seeds", NSEEDS)
rows = {k: [] for k in ("c1", "h1", "h2", "h3")}
for seed in range(NSEEDS):
    ps = prompts(3 if seed == 0 else 100 + seed)
    c = model([ATT] * 4, seed)
    c_ref, c_tok = reference(c, ps, CHUNK)
    rows["c1"].append(worst(paged(c, ps, c_tok, False, True), c_ref)[0])
    del c
    h = model([LIN, ATT, LIN, LIN], seed)
    h_ref, h_tok = reference(h, ps, CHUNK)
    h_ref1, h_tok1 = reference(h, ps, None)
    rows["h1"].append(worst(paged(h, ps, h_tok, True, True), h_ref)[0])
    rows["h2"].append(worst(paged(h, ps, h_tok, True, False), h_ref)[0])
    rows["h3"].append(worst(paged(h, ps, h_tok1, True, True), h_ref1)[0])
    print(f"SEED {seed}: control V1 {rows['c1'][-1]:.3e} | hybrid V1 {rows['h1'][-1]:.3e} V2 {rows['h2'][-1]:.3e} V3 {rows['h3'][-1]:.3e}")
    del h
    torch.cuda.empty_cache()
for k, v in rows.items():
    print(f"STATS {k}: median {statistics.median(v):.3e} max {max(v):.3e} values {[f'{x:.2e}' for x in v]}")
