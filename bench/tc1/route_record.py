"""usage: route_record.py tokens.json routecalls-qwen3.json <checkpoint dir>

Record the group sizes and expert ids of every fused forward and dgrad call in e4b's training step (4-layer Qwen3-30B-A3B slice,
TC1's alpaca rows, mb2, 8 micro-batches) to JSON, for a kernel-route replay on other cards."""
import json, sys, warnings
import torch
warnings.filterwarnings("ignore")
tok = json.load(open(sys.argv[1])); out = sys.argv[2]; ckpt = sys.argv[3]   # a local Qwen3-30B-A3B slice (decoder layers truncated), real weights
import nf4_grouped as ng
from experts4bit_qlora import load_moe_4bit_streaming, enable_fast_train
from experts4bit_qlora.lora import add_attention_lora, quantize_attention_projections_4bit
model, cfg = load_moe_4bit_streaming(ckpt, "cuda", torch.bfloat16, 16, 16, offload=False, pin=True, prefetch=False, quant_type="nf4")
model.to("cuda"); quantize_attention_projections_4bit(model)
model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False}); model.config.use_cache = False
add_attention_lora(model, 16, 16, torch.float32); enable_fast_train(model, dgrad=True); model.train()
PAD = tok["pad_id"]; rows = tok["train"]
REC = []
_f, _d = ng.gemm_4bit_grouped, ng.dgrad_4bit_grouped
def ids(e): return [int(x) for x in (e.tolist() if torch.is_tensor(e) else e)]
def f(a_cat, B, absmax, sizes, expert_ids, *a, **kw):
    if max(sizes) > 1: REC.append({"op": "fwd", "N": int(B.shape[1]), "K": int(B.shape[2]) * 2, "E": int(B.shape[0]), "sizes": [int(s) for s in sizes], "eids": ids(expert_ids)})
    return _f(a_cat, B, absmax, sizes, expert_ids, *a, **kw)
def d(go, B, absmax, sizes, expert_ids, *a, **kw):
    REC.append({"op": "dgrad", "N": int(B.shape[1]), "K": int(B.shape[2]) * 2, "E": int(B.shape[0]), "sizes": [int(s) for s in sizes], "eids": ids(expert_ids)})
    return _d(go, B, absmax, sizes, expert_ids, *a, **kw)
ng.gemm_4bit_grouped, ng.dgrad_4bit_grouped = f, d
for i in range(8):
    a, b = rows[2 * i], rows[2 * i + 1]; n = max(len(a), len(b))
    x = torch.tensor([a + [PAD] * (n - len(a)), b + [PAD] * (n - len(b))], device="cuda"); lab = x.clone(); lab[x == PAD] = -100
    model(input_ids=x, attention_mask=(x != PAD).long(), labels=lab).loss.backward()
torch.cuda.synchronize()
json.dump({"source": "e4b fused training step, 4-layer Qwen3-30B-A3B slice (real weights' router), TC1 alpaca rows, mb2, 8 micro-batches",
           "calls": REC}, open(out, "w"))
from collections import Counter
print("calls", len(REC), Counter(r["op"] for r in REC), "rows median", sorted(sum(r["sizes"]) for r in REC)[len(REC) // 2])
