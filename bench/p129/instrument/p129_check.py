"""P129 Phase 1 check (A2000): exactness (dequant bitwise; loss and grads within TC1's rounding bar) and counts (launches, Python
calls) of E4B_TRAIN_FUSE_QKV against today's path, on the 2-layer random Qwen3-MoE set up as TC1's e4b arm."""
import cProfile, collections, json, pstats
import torch
from experts4bit_qlora import load_moe_4bit_streaming, enable_fast_train
from experts4bit_qlora.lora import add_attention_lora, quantize_attention_projections_4bit
from experts4bit_qlora.engines.train_qkv_fuse import enable_train_fuse_qkv, TRAIN_QKV_STATS, FusedQKVLoRA
from bitsandbytes.functional import dequantize_4bit
import os as _os
D = _os.environ.get("P129_MODEL_DIR", "qwen3moe-2l")             # make_model.py writes it
def build(fuse):
    torch.manual_seed(0)
    model, cfg = load_moe_4bit_streaming(D, "cuda", torch.bfloat16, 16, 16, offload=False, pin=True, prefetch=False, quant_type="nf4")
    model.to("cuda")
    quantize_attention_projections_4bit(model)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.config.use_cache = False
    add_attention_lora(model, 16, 16, torch.float32)
    torch.manual_seed(1)
    for n_, p in model.named_parameters():
        if "lora" in n_:
            p.data = p.data.float()
            if "lora_B" in n_:
                p.data.normal_(0, 0.02)
    enable_fast_train(model, verbose=False)
    deq = None
    if fuse:
        deq = [torch.cat([dequantize_4bit(getattr(L.self_attn, n).base.weight.data, getattr(L.self_attn, n).base.weight.quant_state)
                          for n in ("q_proj", "k_proj", "v_proj")]) for L in model.model.layers]
        nf = enable_train_fuse_qkv(model)
        assert nf == len(model.model.layers), (nf, TRAIN_QKV_STATS)
    model.train()
    return model, deq
torch.autograd.set_multithreading_enabled(False)
g = torch.Generator().manual_seed(2)
ids = torch.randint(0, 4096, (2, 270), generator=g).cuda()
def step(m):
    m.zero_grad(set_to_none=True)
    out = m(input_ids=ids, labels=ids)
    out.loss.backward()
    torch.cuda.synchronize()
    return out.loss.detach().float()
res = {}
ref, _ = build(False)
for _ in range(2):
    step(ref)
lr = step(ref)
gref = {n: p.grad.detach().clone() for n, p in ref.named_parameters() if p.grad is not None}
def counts(m):
    pr = cProfile.Profile(); pr.enable(); step(m); pr.disable()
    py = sum(v[1] for v in pstats.Stats(pr).stats.values())
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as prof:
        step(m)
    la = sum(e.count for e in prof.key_averages() if e.key in ("cudaLaunchKernel", "cuLaunchKernel", "cuLaunchKernelEx", "cudaLaunchKernelExC"))
    return py, la
res["ref_python_calls"], res["ref_launches"] = counts(ref)
del ref; torch.cuda.empty_cache()
fz, deq = build(True)
res["dequant_bitwise"] = all(torch.equal(L.self_attn.qkv_proj.dequantized(), d) for L, d in zip(fz.model.layers, deq))
for _ in range(2):
    step(fz)
lf = step(fz)
res["loss_ref"], res["loss_fused"] = float(lr), float(lf)
res["loss_bitwise"] = bool(torch.equal(lr, lf))
worst, worst_n, nonbit = 0.0, None, 0
for n, p in fz.named_parameters():
    if p.grad is None:
        continue
    r = gref.get(n)
    if r is None:
        res.setdefault("grad_missing_in_ref", []).append(n); continue
    bar = (2.0 ** -6 if r.dtype == torch.bfloat16 else 2.0 ** -16) * r.float().abs().max().item()
    d = (p.grad.float() - r.float()).abs().max().item()
    if d > 0:
        nonbit += 1
    rel = d / max(r.float().abs().max().item(), 1e-30)
    if rel > worst:
        worst, worst_n = rel, n
    if d > bar:
        res.setdefault("grads_over_bar", []).append((n, d, bar))
res["grads_compared"] = len(gref); res["grads_not_bitwise"] = nonbit
res["grad_worst_rel"], res["grad_worst_name"] = worst, worst_n
res["fused_python_calls"], res["fused_launches"] = counts(fz)
res["launch_cut"] = res["ref_launches"] - res["fused_launches"]
res["python_cut_pct"] = round(100 * (res["ref_python_calls"] - res["fused_python_calls"]) / res["ref_python_calls"], 2)
res["stats"] = {k: v for k, v in TRAIN_QKV_STATS.items() if k != "refused"} | {"refused": len(TRAIN_QKV_STATS["refused"])}
print("RESULT " + json.dumps(res, default=str))
