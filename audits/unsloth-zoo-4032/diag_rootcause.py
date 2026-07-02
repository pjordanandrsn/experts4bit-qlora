"""Root-cause clincher for the square-dims transposition bug.

1. Sanity: torch._grouped_mm itself vs a hand loop (is the primitive fine?).
2. Reproduce: parity failure on the tiny square model, default preprocess_weight.
3. Prove: monkeypatch preprocess_weight to always transpose from the known
   F.linear source layout -> parity must drop to exact (== bf16 control).
"""

import json

import unsloth  # noqa: F401
import torch
import torch.nn.functional as F
from transformers.activations import ACT2FN

from audit_lib import find_expert_modules, parity_check

# --- 1. torch._grouped_mm primitive sanity ---
E, T, K, N = 4, 32, 16, 24
torch.manual_seed(0)
x = torch.randn(T, K, device="cuda", dtype=torch.bfloat16)
w = torch.randn(E, K, N, device="cuda", dtype=torch.bfloat16)
counts = torch.tensor([8, 8, 8, 8], device="cuda", dtype=torch.int32)
offs = torch.cumsum(counts, 0, dtype=torch.int32)
out = torch._grouped_mm(x, w, offs=offs)
ref = torch.cat([x[i * 8:(i + 1) * 8] @ w[i] for i in range(E)])
print("[1] torch._grouped_mm max err vs loop:", (out - ref).abs().max().item())

# --- 2 & 3. tiny square model, before/after preprocess fix ---
from unsloth import FastLanguageModel
from unsloth_zoo.temporary_patches import moe_utils

model, _ = FastLanguageModel.from_pretrained(
    model_name="/tmp/claude-tiny-qwen3moe", max_seq_length=512, load_in_4bit=True, dtype=None,
)
model.eval()
cfg = model.config.get_text_config()
_, mod = find_expert_modules(model)[0]
act = getattr(mod, "act_fn", None) or ACT2FN[cfg.hidden_act]

torch.manual_seed(0)
before = parity_check(mod, cfg.hidden_size, cfg.num_experts, cfg.num_experts_per_tok, act)
print("[2] default preprocess_weight :", json.dumps(before))

_orig = moe_utils.preprocess_weight

def fixed_preprocess_weight(weight, proj_type, hidden_dim, model_type=None):
    # transformers v5 stores F.linear layout: gate_up (E, 2I, H), down (E, H, I).
    # grouped_mm wants (E, in, out) -> always transpose; no shape guessing.
    return weight.transpose(-2, -1)

moe_utils.preprocess_weight = fixed_preprocess_weight
try:
    torch.manual_seed(0)
    after = parity_check(mod, cfg.hidden_size, cfg.num_experts, cfg.num_experts_per_tok, act)

    # elementwise bit-identity check (max-abs-ratio 1.0 alone doesn't prove it):
    # rebuild the exact parity inputs and compare fixed-backend output vs bf16 reference
    import torch.nn.functional as F
    from audit_lib import _dequant_expert_weight
    g = torch.Generator(device="cpu").manual_seed(0)
    hs = torch.randn(64, cfg.hidden_size, generator=g).to(device="cuda", dtype=torch.bfloat16)
    idx = torch.randint(0, cfg.num_experts, (64, cfg.num_experts_per_tok), generator=g).cuda()
    w = torch.rand(64, cfg.num_experts_per_tok, generator=g).to(device="cuda", dtype=torch.bfloat16)
    w = w / w.sum(-1, keepdim=True)
    with torch.no_grad():
        out_fixed = mod(hs, idx, w)
        w_gu = _dequant_expert_weight(mod.gate_up_proj).to(device="cuda", dtype=torch.bfloat16)
        w_d = _dequant_expert_weight(mod.down_proj).to(device="cuda", dtype=torch.bfloat16)
        ref16 = torch.zeros_like(out_fixed)
        for t in range(64):
            for k in range(cfg.num_experts_per_tok):
                e = idx[t, k].item()
                gate, up = F.linear(hs[t], w_gu[e]).chunk(2, dim=-1)
                ref16[t] += w[t, k] * F.linear(act(gate) * up, w_d[e])
    bit_identical = bool(torch.equal(out_fixed, ref16))
finally:
    moe_utils.preprocess_weight = _orig
print("[3] transpose-always fix      :", json.dumps(after))
print("[3b] torch.equal(fixed_backend_out, bf16_reference_out):", bit_identical)
print("[verdict] excess over noise, before -> after:",
      round(before["excess_over_precision_noise"], 1), "->",
      round(after["excess_over_precision_noise"], 3))
