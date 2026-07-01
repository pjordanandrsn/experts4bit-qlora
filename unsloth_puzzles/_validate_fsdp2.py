"""Validate that FSDP2 (fully_shard) accepts + trains a bitsandbytes 4-bit QLoRA model.

Single-rank process group on one GPU: this doesn't test cross-rank comms, but it DOES exercise
FSDP2's parameter flattening/sharding over bnb Params4bit + LoRA params — the exact spot where
FSDP2 + 4-bit usually fails. Tiny Llama (SmolLM2-135M) so it's fast.
"""

import os
import torch
import torch.distributed as dist
from torch.distributed.fsdp import CPUOffloadPolicy, MixedPrecisionPolicy, fully_shard
from transformers import AutoModelForCausalLM, BitsAndBytesConfig, set_seed
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
os.environ.setdefault("MASTER_PORT", "29513")
os.environ.setdefault("RANK", "0")
os.environ.setdefault("WORLD_SIZE", "1")
os.environ.setdefault("LOCAL_RANK", "0")
dist.init_process_group("nccl", device_id=torch.device("cuda:0"))
torch.cuda.set_device(0)
set_seed(3407)

MODEL = os.environ.get("MODEL", "HuggingFaceTB/SmolLM2-135M-Instruct")
DTYPE = torch.bfloat16
bnb = BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=DTYPE, bnb_4bit_quant_storage=DTYPE,  # bf16 storage -> FSDP2-shardable
)
model = AutoModelForCausalLM.from_pretrained(
    MODEL, quantization_config=bnb, torch_dtype=DTYPE, attn_implementation="sdpa", device_map={"": 0},
)
model.config.use_cache = False
model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
model = get_peft_model(model, LoraConfig(
    r=16, lora_alpha=32, lora_dropout=0.0, bias="none", task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
))

# FSDP2: wrap each decoder layer, then the root — with bf16 mixed precision + CPU param offload.
mp = MixedPrecisionPolicy(param_dtype=DTYPE, reduce_dtype=torch.float32)
off = CPUOffloadPolicy()
n_wrapped = 0
for mod in model.modules():
    if type(mod).__name__ == "LlamaDecoderLayer":
        fully_shard(mod, mp_policy=mp, offload_policy=off)
        n_wrapped += 1
fully_shard(model, mp_policy=mp, offload_policy=off)
print(f"[fsdp2] wrapped {n_wrapped} decoder layers + root; offload=CPU, mp=bf16")

trainable = [p for p in model.parameters() if p.requires_grad]
n_train = sum(p.numel() for p in trainable)
n_base_trainable = sum(p.requires_grad for n, p in model.named_parameters() if "lora" not in n)
opt = torch.optim.AdamW(trainable, lr=2e-4)
losses = []
lora_grad = base_grad = None
for step in range(6):
    ids = torch.randint(0, model.config.vocab_size, (2, 64), device="cuda")
    out = model(input_ids=ids, labels=ids)
    out.loss.backward()
    if step == 5:  # inspect grads BEFORE zero_grad clears them
        lora_grad = any(p.grad is not None for n, p in model.named_parameters() if "lora" in n)
        base_grad = any(p.grad is not None for n, p in model.named_parameters() if "lora" not in n)
    opt.step()
    opt.zero_grad()
    losses.append(round(out.loss.item(), 4))

finite = all(x == x for x in losses)
learned = losses[-1] < losses[0]
print(f"[fsdp2] trainable={n_train:,} (base-trainable params={n_base_trainable}, want 0)")
print(f"[fsdp2] losses={losses}  finite={finite}  learned={learned}  lora_has_grad={lora_grad}  base_has_grad={base_grad}")
print("[fsdp2] peak GPU:", round(torch.cuda.max_memory_allocated() / 1e9, 2), "GB")
ok = finite and learned and lora_grad and not base_grad and n_base_trainable == 0
print("RESULT:", "PASS — FSDP2 shards + trains a 4-bit QLoRA model (only LoRA learns)" if ok else "FAIL")
dist.destroy_process_group()
