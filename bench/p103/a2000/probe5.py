"""Chunked prefill under fla: is the chunk gated delta rule invariant to splitting a prompt, and does the layout of the
initial state matter? (1) the rule over T tokens in one call vs split at 32 with the first call's final state carried in
-- as returned, and made contiguous; (2) transformers' own DynamicCache chunked prefill (R32) vs single prefill over 20
seeds of the tiny MoE model (probe4's shape)."""
import inspect
import os
import statistics
import torch
import transformers.models.qwen3_5.modeling_qwen3_5 as m

wrapped = m.torch_chunk_gated_delta_rule
impl = inspect.getclosurevars(wrapped).nonlocals.get("implementation")
print("RESOLVED", getattr(impl, "__module__", None))
g = torch.Generator(device="cuda").manual_seed(0)
for (B, T, H, K, V) in ((1, 70, 4, 32, 32), (1, 300, 32, 128, 128)):
    q = torch.randn(B, T, H, K, generator=g, device="cuda").to(torch.bfloat16)
    k = torch.randn(B, T, H, K, generator=g, device="cuda").to(torch.bfloat16)
    v = torch.randn(B, T, H, V, generator=g, device="cuda").to(torch.bfloat16)
    gg = (-torch.rand(B, T, H, generator=g, device="cuda") * 2).float()
    beta = torch.rand(B, T, H, generator=g, device="cuda").to(torch.bfloat16)
    kw = dict(use_qk_l2norm_in_kernel=True, output_final_state=True)
    o_all, s_all = impl(q, k, v, g=gg, beta=beta, initial_state=None, **kw)
    o1, s1 = impl(q[:, :32], k[:, :32], v[:, :32], g=gg[:, :32], beta=beta[:, :32], initial_state=None, **kw)
    print(f"SHAPE B{B} T{T} H{H} K{K} V{V}: final state shape {tuple(s1.shape)} stride {s1.stride()} contiguous {s1.is_contiguous()} dtype {s1.dtype}")
    for name, init in (("as returned", s1), ("contiguous", s1.contiguous()), ("zeros_like+copy", torch.zeros_like(s1).copy_(s1)),
                       ("fresh contiguous copy", torch.empty(s1.shape, dtype=s1.dtype, device=s1.device).copy_(s1))):
        o2, s2 = impl(q[:, 32:], k[:, 32:], v[:, 32:], g=gg[:, 32:], beta=beta[:, 32:], initial_state=init, **kw)
        o_split = torch.cat([o1, o2], 1)
        print(f"  SPLIT@32 init {name:22s} stride {init.stride()}: out max|split - whole| {(o_split.float() - o_all.float()).abs().max().item():.3e} "
              f"(rel {((o_split.float() - o_all.float()).abs().max() / o_all.float().abs().max()).item():.3e}), final state {(s2 - s_all).abs().max().item():.3e}")

import probe2 as p2  # noqa: E402  (the tiny MoE model, prompts, reference)
rows = []
for seed in range(20):
    mm = p2.model(seed)
    ps = p2.prompts(100 + seed)
    R, _ = p2.reference(mm, ps)
    R32, _ = p2.reference(mm, ps, chunk=32)
    rows.append(p2.worst(R32, R)[0])
    del mm
print(f"STATS R32 (transformers' own chunked prefill vs single): median {statistics.median(rows):.3e} mean {statistics.mean(rows):.3e} "
      f"max {max(rows):.3e} nonzero {sum(r > 0 for r in rows)}/{len(rows)}")
