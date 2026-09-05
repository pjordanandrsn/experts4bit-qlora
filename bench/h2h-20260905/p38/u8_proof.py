"""P38 amendment 4 proof (fresh process, GPU free): unsloth_zoo's `_moe_uses_bnb4bit_expert_weights` on the experts module
BEFORE get_peft_model, AFTER it on the module named `mlp.experts` (PEFT's wrapper, which U8 inspected), and AFTER it on the
innermost module; plus the classes of the expert parameters at each level. Same load + PEFT call as the harness. Writes
u8_proof.json. No training, no receipt row."""
import os, json, sys, time, torch
from unsloth import FastLanguageModel
from unsloth_zoo.temporary_patches.moe_utils_bnb4bit import _moe_uses_bnb4bit_expert_weights as pred
REV = "ad44e777bcd18fa416d9da3bd8f70d33ebb85d39"
local = os.path.join(os.path.expanduser("~/.cache/huggingface/hub"), "models--Qwen--Qwen3-30B-A3B", "snapshots", REV)
def experts(model): return [(n, m) for n, m in model.named_modules() if n.endswith("mlp.experts")]
def inner(m):
    while hasattr(m, "base_layer"): m = m.base_layer
    return m
def kinds(m): return sorted({type(getattr(m, k, None)).__name__ for k in ("gate_up_proj", "down_proj")})
out = {"revision": REV, "when": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
t0 = time.time()
model, tok = FastLanguageModel.from_pretrained(model_name=local, max_seq_length=512, dtype=torch.bfloat16, load_in_4bit=True)
E = experts(model)
out["before_peft"] = {"n_modules": len(E), "n_pred_true": sum(bool(pred(m)) for _, m in E), "param_kinds": sorted({k for _, m in E for k in kinds(m)}), "example": E[0][0] if E else None}
model = FastLanguageModel.get_peft_model(model, r=8, lora_alpha=16, lora_dropout=0.0, bias="none",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    use_gradient_checkpointing="unsloth", random_state=int(sys.argv[1]) if len(sys.argv) > 1 else 3407)
E = experts(model)
out["after_peft_wrapper"] = {"n_modules": len(E), "n_pred_true": sum(bool(pred(m)) for _, m in E), "wrapper_cls": sorted({type(m).__name__ for _, m in E}), "param_kinds": sorted({k for _, m in E for k in kinds(m)}), "example": E[0][0] if E else None}
I = [(n, inner(m)) for n, m in E]
out["after_peft_innermost"] = {"n_modules": len(I), "n_pred_true": sum(bool(pred(m)) for _, m in I), "inner_cls": sorted({type(m).__name__ for _, m in I}), "param_kinds": sorted({k for _, m in I for k in kinds(m)}), "depth": max(len([1 for _ in iter(lambda m=m: None, 0)]) for _, m in E[:1]) if False else None}
d = 0; m = E[0][1]
while hasattr(m, "base_layer"): m = m.base_layer; d += 1
out["after_peft_innermost"]["wrap_depth"] = d
out["load_s"] = round(time.time() - t0, 1)
json.dump(out, open("/root/p38/u8_proof.json", "w"), indent=1); print(json.dumps(out, indent=1))
