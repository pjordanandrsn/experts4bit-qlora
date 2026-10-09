"""P129: the fused projection in isolation -- same input, same upstream gradient -- against the three LoRALinear modules."""
import json, copy
import torch
from experts4bit_qlora import load_moe_4bit_streaming, enable_fast_train
from experts4bit_qlora.lora import add_attention_lora, quantize_attention_projections_4bit
from experts4bit_qlora.engines.train_qkv_fuse import FusedQKVLoRA
import os as _os
D = _os.environ.get("P129_MODEL_DIR", "qwen3moe-2l")             # make_model.py writes it
torch.manual_seed(0)
model, cfg = load_moe_4bit_streaming(D, "cuda", torch.bfloat16, 16, 16, offload=False, pin=True, prefetch=False, quant_type="nf4")
model.to("cuda")
quantize_attention_projections_4bit(model)
add_attention_lora(model, 16, 16, torch.float32)
torch.manual_seed(1)
for n_, p in model.named_parameters():
    if "lora" in n_:
        p.data = p.data.float()
        if "lora_B" in n_:
            p.data.normal_(0, 0.02)
attn = model.model.layers[0].self_attn
fused = FusedQKVLoRA(attn)
g = torch.Generator(device="cuda").manual_seed(3)
x = (torch.randn(2, 270, 2048, generator=g, device="cuda") * 0.5).to(torch.bfloat16)
gup = (torch.randn(2, 270, sum(fused.ns), generator=g, device="cuda") * 0.01).to(torch.bfloat16)
def run(f):
    xx = x.detach().clone().requires_grad_(True)
    for m in (attn.q_proj, attn.k_proj, attn.v_proj):
        for p in (m.lora_A, m.lora_B):
            p.grad = None
    out = f(xx)
    out.backward(gup)
    gr = {f"{n}.{w}": getattr(getattr(attn, n), w).grad.detach().clone() for n in ("q_proj", "k_proj", "v_proj") for w in ("lora_A", "lora_B")}
    return out.detach(), xx.grad.detach(), gr
ref = run(lambda xx: torch.cat([attn.q_proj(xx), attn.k_proj(xx), attn.v_proj(xx)], -1))
fz = run(fused)
def cmp(a, b, name):
    bar = (2.0 ** -6 if b.dtype == torch.bfloat16 else 2.0 ** -16) * b.float().abs().max().item()
    d = (a.float() - b.float()).abs().max().item()
    return {"name": name, "dtype": str(b.dtype), "bitwise": bool(torch.equal(a, b)), "maxdiff": d, "bar": bar, "within": d <= bar,
            "rel": d / max(b.float().abs().max().item(), 1e-30)}
rows = [cmp(fz[0], ref[0], "qkv out"), cmp(fz[1], ref[1], "d x")] + [cmp(fz[2][k], ref[2][k], "d " + k) for k in ref[2]]
print("RESULT " + json.dumps(rows))
