"""P70 cast-proof diagnosis (NAS A2000; not a reading). Separates GEMM row-count variance from the fused kernel."""
import json, os, torch
os.environ["E4B_FUSE_ROUTER_EPI"] = "1"
from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeConfig, Qwen3MoeTopKRouter
from experts4bit_qlora.engines import router_epilogue as re_mod
cfg = Qwen3MoeConfig(hidden_size=2048, num_experts=128, num_experts_per_tok=8, norm_topk_prob=True)
torch.manual_seed(726)
ref = Qwen3MoeTopKRouter(cfg).to("cuda", torch.bfloat16)
with torch.no_grad():
    ref.weight.normal_(0, 0.02)
holder = torch.nn.Module(); holder.gate = Qwen3MoeTopKRouter(cfg).to("cuda", torch.bfloat16)
holder.gate.load_state_dict(ref.state_dict()); n = re_mod.fuse_router_epilogue(holder)
x = (torch.randn(80, 2048) * 0.5).to("cuda", torch.bfloat16)
r = {"gpu": torch.cuda.get_device_name(0), "patched": n}
lin = lambda t: torch.nn.functional.linear(t, ref.weight)
with torch.no_grad():
    L80, L64, L1 = lin(x), lin(x[:64]), lin(x[:1])
    r["gemm_logits_64_eq_80[:64]"] = bool(torch.equal(L64, L80[:64]))
    r["gemm_logits_1_eq_80[:1]"] = bool(torch.equal(L1, L80[:1]))
    r["gemm_logits_1_eq_64[:1]"] = bool(torch.equal(L1, L64[:1]))
    _, rw80, ri80 = ref(x); _, rw64, ri64 = ref(x[:64]); _, rw1, ri1 = ref(x[:1])
    r["upstream_idx_64_eq_80[:64]"] = bool(torch.equal(ri64, ri80[:64]))
    r["upstream_idx_1_eq_64[:1]"] = bool(torch.equal(ri1, ri64[:1]))
    for cast in (False, True):
        re_mod.CAST_WEIGHTS[0] = cast
        _, w64, i64 = holder.gate(x[:64]); _, w1, i1 = holder.gate(x[:1])
        k = "on" if cast else "off"
        r[f"{k}_idx64_eq_upstream64_ordered"] = bool(torch.equal(i64, ri64))
        r[f"{k}_idx64_eq_upstream64_as_sets"] = bool(torch.equal(i64.sort(-1).values, ri64.sort(-1).values))
        r[f"{k}_idx1_eq_upstream1"] = bool(torch.equal(i1, ri1))
        r[f"{k}_w64_bit_equal_upstream64_frac"] = float((w64.to(rw64.dtype) == rw64).float().mean()) if cast else None
        r[f"{k}_w64_max_abs_vs_upstream64"] = float((w64.float() - rw64.float()).abs().max())
        r[f"{k}_one_row_eq_row0_of_64_idx"] = bool(torch.equal(i1, i64[:1]))
        r[f"{k}_one_row_eq_row0_of_64_w"] = bool(torch.equal(w1, w64[:1]))
    re_mod.CAST_WEIGHTS[0] = False
    # the kernel alone, on identical fp32 logits: is it the upstream function?
    from int4_b32 import router_epilogue as kern
    _f, kw, ki = kern(L64.float(), 8, True)
    p = torch.softmax(L64.float(), -1); tv, ti = torch.topk(p, 8, -1); tv = tv / tv.sum(-1, keepdim=True)
    r["kernel_on_same_logits_idx_eq_torch_ordered"] = bool(torch.equal(ki.long(), ti))
    r["kernel_on_same_logits_idx_eq_torch_as_sets"] = bool(torch.equal(ki.long().sort(-1).values, ti.sort(-1).values))
    r["kernel_on_same_logits_w_max_abs"] = float((kw.float() - tv).abs().max())
    r["kernel_on_same_logits_w_bitequal_frac"] = float((kw.float() == tv).float().mean())
    r["kernel_cast_bf16_bitequal_frac"] = float((kw.to(torch.bfloat16) == tv.to(torch.bfloat16)).float().mean())
print("P70DIAG " + json.dumps(r), flush=True)
