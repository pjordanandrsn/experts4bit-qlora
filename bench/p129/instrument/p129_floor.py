"""P129 Amendment 1 (A2000): the end-to-end question against a neutral floor. One run per (config, seed): 30 AdamW8bit steps on the
2-layer random Qwen3-MoE set up as TC1's e4b arm; per-step training loss, held-out at 0 and 30, the first step's gradients."""
import json, os, sys
import torch
import bitsandbytes as bnb
from experts4bit_qlora import load_moe_4bit_streaming, enable_fast_train
from experts4bit_qlora.lora import add_attention_lora, quantize_attention_projections_4bit
from experts4bit_qlora.engines.train_qkv_fuse import enable_train_fuse_qkv, TRAIN_QKV_STATS
import os as _os
D = _os.environ.get("P129_MODEL_DIR", "qwen3moe-2l")             # make_model.py writes it
CFG, SEED = sys.argv[1], int(sys.argv[2])
ENV = {"B": {}, "F1": {"NF4_QLORA_SINGLE_LADDER": "0"}, "F2": {"NF4_QLORA_PAD_BUCKETS": "1"}, "F3": {}, "FUSED": {}}[CFG]
os.environ.update(ENV)
torch.manual_seed(0)
model, cfg = load_moe_4bit_streaming(D, "cuda", torch.bfloat16, 16, 16, offload=False, pin=True, prefetch=False, quant_type="nf4")
model.to("cuda")
quantize_attention_projections_4bit(model)
model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
model.config.use_cache = False
add_attention_lora(model, 16, 16, torch.float32)
gi = torch.Generator().manual_seed(SEED)
for n_, p in model.named_parameters():
    if "lora" in n_:
        p.data = p.data.float()
        if "lora_A" in n_:
            p.data.copy_((torch.randn(p.shape, generator=gi) / p.shape[0]).to(p.device))
        if "lora_B" in n_:
            p.data.copy_((torch.randn(p.shape, generator=gi) * 0.02).to(p.device))
enable_fast_train(model, verbose=False)
if CFG == "FUSED":
    assert enable_train_fuse_qkv(model) == len(model.model.layers), TRAIN_QKV_STATS
model.train()
params = [p for p in model.parameters() if p.requires_grad]
opt = bnb.optim.AdamW8bit(params, lr=2e-4, weight_decay=0.001)
sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda i: min(1.0, (i + 1) / 5))
gd = torch.Generator().manual_seed(SEED)
batches = [torch.randint(0, 4096, (2, 270), generator=gd) for _ in range(30)]
gh = torch.Generator().manual_seed(SEED + 1000)
held = [torch.randint(0, 4096, (2, 270), generator=gh).cuda() for _ in range(4)]
def heldout():
    with torch.no_grad():
        return sum(model(input_ids=h, labels=h).loss.float().item() for h in held) / len(held)
out = {"cfg": CFG, "seed": SEED, "held0": heldout(), "loss": []}
model.train()
for t, b in enumerate(batches):
    b = b.cuda()
    if CFG == "F3":
        tot = 0.0
        for r in range(b.shape[0]):
            l = model(input_ids=b[r:r + 1], labels=b[r:r + 1]).loss / b.shape[0]
            l.backward(); tot += l.detach().float().item()
        out["loss"].append(tot)
    else:
        l = model(input_ids=b, labels=b).loss
        l.backward(); out["loss"].append(l.detach().float().item())
    if t == 0:
        torch.save({n: p.grad.detach().float().cpu() for n, p in model.named_parameters() if p.grad is not None},
                   _os.path.join(_os.environ.get("P129_OUT", "p129a1"), f"g0_{CFG}_{SEED}.pt"))
    opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
out["held30"] = heldout()
print("RUN " + json.dumps(out))
