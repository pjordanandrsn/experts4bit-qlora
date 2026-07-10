#!/usr/bin/env python3
"""Phase-0 access sweep — stock transformers + device_map=auto (bf16), the sweep that produced
RESULTS.md. Routes EXACTLY (base model, no quantization); overflow CPU-offloads (OLMoE resident,
Qwen3-30B via CPU offload with ~61GB host RAM). Base routing is the clean characterization; NF4
perturbs logits negligibly for expert-SELECTION counting. Forward-only, counting.

  MODEL=Qwen/Qwen3-30B-A3B OUT=access_qwen30b_diverse.jsonl python run_access_stock.py
"""
import os, sys
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from access_counter import ExpertAccessCounter, attach

MODEL = os.environ["MODEL"]; OUT = os.environ["OUT"]
CELLS = [(1,1),(1,2),(1,4),(1,8),(1,16),(1,32),(1,64),(1,128),(1,256),(1,512),(1,1024),(1,2048),(8,8),(8,64)]

def _ec(c):
    t=getattr(c,"text_config",c)
    for k in ("num_experts","num_local_experts","n_routed_experts"):
        if getattr(t,k,None): return getattr(t,k)
    raise SystemExit("no E")
def _tk(c):
    t=getattr(c,"text_config",c)
    for k in ("num_experts_per_tok","experts_per_token","top_k_experts","moe_top_k"):
        if getattr(t,k,None): return getattr(t,k)
    raise SystemExit("no k")

cfg = AutoConfig.from_pretrained(MODEL, trust_remote_code=True)
E,k=_ec(cfg),_tk(cfg)
print(f"model={MODEL} E={E} k={k}", flush=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL, trust_remote_code=True, torch_dtype=torch.bfloat16, device_map="auto", low_cpu_mem_usage=True,
).eval()
counter = ExpertAccessCounter(E,k)
n = attach(model, counter); print(f"hooked {n} router gates", flush=True); assert n>0
tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
# DIVERSE real text (removes the repetitive-corpus confound that inflates imbalance).
from datasets import load_dataset
ds = load_dataset("wikitext", "wikitext-103-raw-v1", split="train", streaming=True)
buf=[]
for r in ds:
    t=r["text"].strip()
    if len(t)>50: buf.append(t)
    if len(buf)>=4000: break
ids0 = tok("\n".join(buf), return_tensors="pt").input_ids[0]
print(f"diverse corpus tokens available: {ids0.numel()}", flush=True)
in_dev = next(model.parameters()).device
with torch.no_grad():
    for step,(b,s) in enumerate(CELLS):
        need=b*s
        ids=(ids0.repeat(need//ids0.numel()+1)[:need] if ids0.numel()<need else ids0[:need])
        model(ids.reshape(b,s).to(in_dev), use_cache=False)
        counter.close_step(step, batch=b, seq=s)
        rows=[r for r in counter.records if r["step"]==step]
        print(f"  eff={b*s:>6} (b{b}xs{s})  mean_read_fraction={sum(r['read_fraction'] for r in rows)/len(rows):.4f}", flush=True)
counter.dump(OUT); print(f"wrote {OUT} ({len(counter.records)} rows)", flush=True)
