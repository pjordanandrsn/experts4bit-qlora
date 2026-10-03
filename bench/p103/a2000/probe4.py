"""probe2 over many seeds: the distribution of the batched-decode drift (P1 batched, P32s unbatched) per kernel arm.
Where does the hybrid paged path's extra error under fla come from? test_linear_state_gpu.py's model shape, all layers
linear (no attention, so no fp8 needed), driven as PagedModelRunner.run_prefill / run_decode drive the per-slot pool.
Each variant's worst relative logit error against transformers' DynamicCache with a single prefill (R):
  P32  : the pool, 32-token prefill chunks (mark after each chunk), batched decode with changing row order (the test)
  P1   : the pool, the whole prompt in one prefill chunk, the same batched decode
  P32s : the pool, 32-token chunks, decode one sequence per step (no batching)
  R32  : transformers' own DynamicCache, prefilled in 32-token chunks, then decode
"""
import torch

from experts4bit_qlora.engines import linear_state, paged_attention

LIN = "linear_attention"
STEPS = 6
NOMASK = {"linear_attention": None, "full_attention": None}


def model(seed=0):
    from transformers import Qwen3_5MoeTextConfig
    from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeForCausalLM
    cfg = Qwen3_5MoeTextConfig(vocab_size=256, hidden_size=128, num_hidden_layers=3, num_attention_heads=4,
                               num_key_value_heads=2, head_dim=64, moe_intermediate_size=64,
                               shared_expert_intermediate_size=64, num_experts=4, num_experts_per_tok=2,
                               linear_num_key_heads=2, linear_num_value_heads=4, linear_key_head_dim=32,
                               linear_value_head_dim=32, linear_conv_kernel_dim=4, layer_types=[LIN] * 3,
                               max_position_embeddings=512)
    torch.manual_seed(seed)
    return Qwen3_5MoeForCausalLM(cfg).to("cuda", torch.bfloat16).eval()


def prompts(seed=3):
    g = torch.Generator().manual_seed(seed)
    return {0: torch.randint(0, 256, (37,), generator=g).tolist(), 1: torch.randint(0, 256, (70,), generator=g).tolist(),
            2: torch.randint(0, 256, (9,), generator=g).tolist()}


def reference(m, ps, chunk=None):
    from transformers.cache_utils import DynamicCache
    lg, tk = {}, {}
    with torch.no_grad():
        for s, p in ps.items():
            cache = DynamicCache(config=m.config)
            step = chunk or len(p)

            def call(ids, st):
                # positions explicit: an all-linear DynamicCache cannot answer get_seq_length
                pos = torch.arange(st, st + len(ids), device="cuda")
                return m(input_ids=torch.tensor([ids], device="cuda"), position_ids=pos[None], cache_position=pos,
                         past_key_values=cache, use_cache=True, attention_mask=NOMASK)
            for st in range(0, len(p), step):
                out = call(p[st:st + step], st)
            lg[s], tk[s] = [out.logits[0, -1].float()], [int(out.logits[0, -1].argmax())]
            for t in range(STEPS):
                out = call([tk[s][-1]], len(p) + t)
                lg[s].append(out.logits[0, -1].float())
                tk[s].append(int(out.logits[0, -1].argmax()))
    return lg, tk


def fwd(m, ctx, ids, pos):
    prev = paged_attention.set_context(ctx)
    try:
        return m(input_ids=ids, position_ids=pos, use_cache=False, attention_mask=NOMASK).logits
    finally:
        paged_attention.set_context(prev)


def paged(m, pool, ps, ref_tok, chunk, batched):
    slot_of = {0: 2, 1: 0, 2: 3}
    got = {s: [] for s in ps}
    with torch.no_grad():
        for s, p in ps.items():
            pool.reset(slot_of[s])
            step = chunk or len(p)
            for st in range(0, len(p), step):
                ids = torch.tensor([p[st:st + step]], device="cuda")
                pos = torch.arange(st, st + ids.shape[1], device="cuda")[None]
                lg = fwd(m, paged_attention.PagedAttentionContext(kv=None, slots=[slot_of[s]], mode="prefill"), ids, pos)
                pool.mark([slot_of[s]])
            got[s].append(lg[0, -1].float())
        for t in range(STEPS):
            orders = ([[0, 1, 2] if t % 2 == 0 else [2, 1, 0]]) if batched else [[0], [1], [2]]
            for active in orders:
                ids = torch.tensor([[ref_tok[s][t]] for s in active], device="cuda")
                pos = torch.tensor([[len(ps[s]) + t] for s in active], device="cuda")
                lg = fwd(m, paged_attention.PagedAttentionContext(kv=None, slots=[slot_of[s] for s in active], mode="decode"),
                         ids, pos)
                pool.mark([slot_of[s] for s in active])
                for i, s in enumerate(active):
                    got[s].append(lg[i, -1].float())
    return got


def worst(got, ref):
    w, toks = 0.0, True
    for s in ref:
        for a, b in zip(got[s], ref[s]):
            w = max(w, float((a - b).abs().max() / b.abs().max()))
            toks &= int(a.argmax()) == int(b.argmax())
    return w, toks


import statistics
rows = {"P1": [], "P32s": []}
for seed in range(20):
    m = model(seed)
    ps = prompts(100 + seed)
    R, rt = reference(m, ps)
    pool = linear_state.install(m, n_slots=4)
    rows["P1"].append(worst(paged(m, pool, ps, rt, None, True), R)[0])
    rows["P32s"].append(worst(paged(m, pool, ps, rt, 32, False), R)[0])
    del m, pool
    torch.cuda.empty_cache()
for k, v in rows.items():
    print(f"STATS {k}: median {statistics.median(v):.3e} mean {statistics.mean(v):.3e} max {max(v):.3e} min {min(v):.3e} n {len(v)}")
