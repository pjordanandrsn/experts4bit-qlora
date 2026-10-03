"""Is the decode-step (T == 1) recurrent gated delta rule batch-invariant? The same rows run as one batch of B and one
at a time; the per-row outputs and final states are compared, for transformers' torch reference and, when installed,
fla's fused_recurrent (what transformers resolves). Shapes: Qwen3.6's Gated DeltaNet (32 value heads, 128-dim K/V)."""
import inspect
import torch
import transformers.models.qwen3_5.modeling_qwen3_5 as m

wrapped = m.torch_recurrent_gated_delta_rule
impl = inspect.getclosurevars(wrapped).nonlocals.get("implementation")
torch_ref = getattr(wrapped, "__wrapped__", None)
print("RESOLVED", getattr(impl, "__module__", None), "| torch reference available:", torch_ref is not None)
g = torch.Generator(device="cuda").manual_seed(0)
import os
B, H, K, V = (int(x) for x in os.environ.get("SHAPE", "6,32,128,128").split(","))
print("SHAPE", B, H, K, V)


def rnd(*s, dt=torch.bfloat16, scale=1.0):
    return (torch.randn(*s, generator=g, device="cuda") * scale).to(dt)


q, k, v = rnd(B, 1, H, K), rnd(B, 1, H, K), rnd(B, 1, H, V)
gg = (-torch.rand(B, 1, H, generator=g, device="cuda") * 2).float()
beta = torch.rand(B, 1, H, generator=g, device="cuda").to(torch.bfloat16)
s0 = rnd(B, H, K, V, dt=torch.float32, scale=0.1)
outs = {}
for name, fn in (("torch_reference", torch_ref), ("resolved", impl)):
    if fn is None:
        continue
    kw = dict(use_qk_l2norm_in_kernel=True, output_final_state=True)
    ob, sb = fn(q, k, v, g=gg, beta=beta, initial_state=s0.clone(), **kw)
    outs[name] = (ob.float(), sb.float())
    worst_o = worst_s = 0.0
    for i in range(B):
        o1, s1 = fn(q[i:i + 1], k[i:i + 1], v[i:i + 1], g=gg[i:i + 1], beta=beta[i:i + 1], initial_state=s0[i:i + 1].clone(), **kw)
        worst_o = max(worst_o, (ob[i:i + 1].float() - o1.float()).abs().max().item())
        worst_s = max(worst_s, (sb[i:i + 1].float() - s1.float()).abs().max().item())
    perm = torch.randperm(B, generator=g, device="cuda")
    op, sp = fn(q[perm], k[perm], v[perm], g=gg[perm], beta=beta[perm], initial_state=s0[perm].clone(), **kw)
    worst_p = (op.float() - ob[perm].float()).abs().max().item()
    print(f"BATCH_INVARIANCE {name} ({getattr(fn, '__module__', '?')}): out max|B - 1| {worst_o:.3e}, state {worst_s:.3e}, "
          f"permuted rows {worst_p:.3e}, out dtype {ob.dtype}, state dtype {sb.dtype}, out max|.| {ob.float().abs().max().item():.3e}")
if outs.get("torch_reference") is not None and impl is not torch_ref and "resolved" in outs:
    (ot, st), (of, sf) = outs["torch_reference"], outs["resolved"]
    print(f"IMPL_DIFF resolved vs torch reference: out max|d| {(ot - of).abs().max().item():.3e} (rel {((ot - of).abs().max() / ot.abs().max()).item():.3e}), "
          f"state max|d| {(st - sf).abs().max().item():.3e}")
