"""experts4bit-qlora#1526: measured allocated peak of fused grouped_nf4 QLoRA training on a reduced Qwen3-30B-A3B,
against experts4bit-qlora's estimate and Loggetta's plan, built the way TC1's e4b arm builds it."""
import argparse
import json
import os
import time

import torch

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--tokens", type=int, required=True)        # packed: one row of this many tokens per micro-batch
ap.add_argument("--adapters", choices=("fp32", "native"), required=True)
ap.add_argument("--accum", type=int, default=4)
ap.add_argument("--steps", type=int, default=3)
ap.add_argument("--snapshot", help="write the allocator history of the last step here")
ap.add_argument("--out", required=True)
a = ap.parse_args()

import bitsandbytes as bnb
import experts4bit_qlora as e4b
from experts4bit_qlora.lora import add_attention_lora, quantize_attention_projections_4bit

GiB = 1 << 30
model, cfg = e4b.load_moe_4bit_streaming(a.model, "cuda", torch.bfloat16, 16, 16, offload=False, pin=True,
                                         prefetch=False, quant_type="nf4")
model.to("cuda")
n_attn4 = quantize_attention_projections_4bit(model)
model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
model.config.use_cache = False
add_attention_lora(model, 16, 16, torch.float32)
n_patched = e4b.enable_fast_train(model, dgrad=True)
cast = 0
if a.adapters == "fp32":                               # TC1 T19: every LoRA parameter fp32
    for n, p in model.named_parameters():
        if "lora" in n and p.dtype != torch.float32:
            p.data = p.data.float()
            cast += 1
params = [p for n, p in model.named_parameters() if p.requires_grad]
dtypes = {}
for n, p in model.named_parameters():
    if "lora" in n:
        dtypes[str(p.dtype)] = dtypes.get(str(p.dtype), 0) + 1
opt = bnb.optim.AdamW8bit(params, lr=2e-4, weight_decay=0.001)
g = torch.Generator(device="cpu").manual_seed(1)
vocab = cfg.vocab_size


def micro():
    ids = torch.randint(0, vocab, (1, a.tokens), generator=g).cuda()
    out = model(input_ids=ids, labels=ids)
    (out.loss / a.accum).backward()


torch.cuda.synchronize()
load_alloc = torch.cuda.memory_allocated()
torch.cuda.reset_peak_memory_stats()
peaks = []
for step in range(a.steps):
    if a.snapshot and step == a.steps - 1:
        torch.cuda.memory._record_memory_history(max_entries=400000, context="all", stacks="python")
    torch.cuda.reset_peak_memory_stats()
    for _ in range(a.accum):
        micro()
    opt.step()
    opt.zero_grad(set_to_none=True)       # as TC1 (tc1_arm.py)
    torch.cuda.synchronize()
    peaks.append(torch.cuda.max_memory_allocated())
if a.snapshot:
    torch.cuda.memory._dump_snapshot(a.snapshot)
    torch.cuda.memory._record_memory_history(enabled=None)

setup = e4b.QLoRASetup(adapter_dtype="fp32" if a.adapters == "fp32" else "bf16", attn_4bit=True,
                       expert_kernel="grouped_nf4", r=16, alpha=16)
topo = e4b.describe_moe(a.model)
fp = e4b.estimate_qlora_footprint(topo, setup, tokens_per_microbatch=a.tokens, optimizer="adamw_8bit")
items = [{"name": i.name, "where": i.where, "bytes": i.bytes} for i in fp.items]
est_device = sum(i["bytes"] for i in items if i["where"] == "device")

loggetta = None
try:
    from loggetta import Constraints, Workload, describe_model, plan
    from loggetta.hardware import GPU, Fact, HardwareProfile, Host
    f = lambda v: Fact(v, "stated")  # noqa: E731
    tot = torch.cuda.get_device_properties(0).total_memory
    gpu = GPU(index=0, vendor="nvidia", name=torch.cuda.get_device_name(0), uuid=None,
              compute_capability=f(torch.cuda.get_device_capability(0)), memory_total=f(tot), memory_free=f(tot),
              driver=f(None), pcie_gen_max=f(None), pcie_width_max=f(None), pcie_gen_current=f(None),
              pcie_width_current=f(None))
    host = Host(cpu_model=f(None), cpus=f(os.cpu_count()), memory_total=f(64 * GiB), memory_available=f(64 * GiB),
                memory_limit=f(64 * GiB))
    p = plan(describe_model(a.model), HardwareProfile(gpus=(gpu,), host=host, platform="Linux"),
             Workload(kind="train", seq_len=a.tokens, micro_batch=1, grad_accum=a.accum, optimizer="adamw_8bit"),
             Constraints(fixed={"expert_kernel": "grouped_nf4", "expert_residency": "device", "attn_4bit": True,
                                "adapter_dtype": setup.adapter_dtype, "r": 16, "alpha": 16}))
    c = p.selected or p.alternatives[0]
    lines = [{"name": ln.name, "basis": ln.basis, "bytes": ln.bytes} for ln in c.lines if ln.where == "device"]
    alloc_est = sum(ln["bytes"] for ln in lines if not ln["name"].startswith(("allocator reserve", "CUDA context")))
    loggetta = {"status": p.status, "lines": lines, "allocator_estimate_bytes": alloc_est}
except Exception as ex:                       # report, do not hide
    loggetta = {"error": repr(ex)}

rec = {"probe": "e4b#1526", "model": a.model, "reduced": json.load(open(os.path.join(a.model, "REDUCED.json"))),
       "tokens_per_microbatch": a.tokens, "packing": "one full row per micro-batch (random ids, labels = ids)",
       "adapters": a.adapters, "accum": a.accum, "steps": a.steps, "lora_dtypes": dtypes, "cast_to_fp32": cast,
       "n_attn4": n_attn4, "n_patched": n_patched, "load_allocated_bytes": load_alloc,
       "peak_allocated_bytes_per_step": peaks, "e4b_estimate_device_bytes": est_device, "e4b_items": items,
       "loggetta": loggetta,
       "versions": {k: __import__(m).__version__ for k, m in (("torch", "torch"), ("experts4bit-qlora", "experts4bit_qlora"),
                                                             ("bitsandbytes", "bitsandbytes"), ("transformers", "transformers"))},
       "gpu": torch.cuda.get_device_name(0), "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
json.dump(rec, open(a.out, "w"), indent=1)
print(json.dumps({k: rec[k] for k in ("tokens_per_microbatch", "adapters", "peak_allocated_bytes_per_step",
                                      "e4b_estimate_device_bytes")}))
