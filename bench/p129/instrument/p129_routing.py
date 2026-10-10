"""P129 Phase 2 read, follow-up (RTX A2000, forward only, report-only): does the router amplify the q/k/v rounding? The real Qwen3-30B-A3B at
the pin, set up as TC1's shipped e4b arm (NF4 attention, bf16 adapters, lora_B zero), the box's eight held-out rows, step 0:
  A        the stock path (its routing recorded: the router's logits, scores and top-8 indices, per layer and row);
  D2       A with q's matmul split in two along N;
  B        the fused q/k/v module with serving's forward;
  B_pin    B with every layer's router output replaced by A's (the same experts and weights as A).
Per layer: the tokens whose top-8 expert set differs from A's. The loss per mode, and how much of A -> B pinning removes."""
import json, os, sys, time
import torch
import torch.nn.functional as F
from bitsandbytes.functional import dequantize_4bit
from experts4bit_qlora import load_moe_4bit_streaming, enable_fast_train
from experts4bit_qlora.lora import add_attention_lora, quantize_attention_projections_4bit
from experts4bit_qlora.engines.train_qkv_fuse import enable_train_fuse_qkv, TRAIN_QKV_STATS
rows = json.load(open(sys.argv[1]))["eval"]
SRC = os.environ.get("P129_MODEL_DIR", "Qwen3-30B-A3B")   # a local copy checked against the revision's sha256s
t0 = time.time()
model, cfg = load_moe_4bit_streaming(SRC, "cuda", torch.bfloat16, 16, 16, offload=True, pin=True, prefetch=False, quant_type="nf4")
print("loaded", round(time.time() - t0), "s", flush=True)
quantize_attention_projections_4bit(model)
model.config.use_cache = False
add_attention_lora(model, 16, 16, torch.bfloat16)
enable_fast_train(model, verbose=False)
layers = model.model.layers
attn = [L.self_attn for L in layers]
S = {"mode": None, "row": 0, "layer": 0, "A": {}, "idx": {}}
def gate_hook(mod, inp, out):
    key = (S["row"], S["layer"]); S["layer"] += 1
    m = S["mode"]
    if m == "A":
        S["A"][key] = tuple(t.detach().cpu() for t in out)
    if m == "B_pin":
        return tuple(t.to(out[0].device) for t in S["A"][key])
    S["idx"].setdefault(m, {})[key] = out[2].detach().cpu()
for L in layers:
    L.mlp.gate.register_forward_hook(gate_hook)
def deq(lin):
    return dequantize_4bit(lin.weight.data, lin.weight.quant_state).to(torch.bfloat16)
class Split(torch.nn.Module):                      # D2: q's 4-bit base as two matmuls along N
    def __init__(self, base):
        super().__init__(); self.b = base
    def forward(self, x):
        w = deq(self.b); h = w.shape[0] // 2
        return torch.cat([F.linear(x, w[:h]), F.linear(x, w[h:])], -1)
@torch.no_grad()
def run(mode):
    S["mode"] = mode
    model.eval()
    per = []
    for i, ids in enumerate(rows):
        S["row"], S["layer"] = i, 0
        x = torch.tensor(ids, dtype=torch.long).unsqueeze(0).cuda()
        per.append(float(model(input_ids=x, labels=x).loss))
    return sum(per) / len(per), per
res = {}
res["A"] = run("A")
orig = [a.q_proj.base for a in attn]
for a in attn:
    a.q_proj.base = Split(a.q_proj.base)
res["D2"] = run("D2")
for a, b in zip(attn, orig):
    a.q_proj.base = b
assert enable_train_fuse_qkv(model) == len(attn) and not TRAIN_QKV_STATS["refused"]
res["B"] = run("B")
res["B_pin"] = run("B_pin")
def flips(m):
    per_layer = []
    for l in range(len(layers)):
        n = d = 0
        for i in range(len(rows)):
            a, b = S["idx"]["A"][(i, l)], S["idx"][m][(i, l)]
            sa, sb = a.sort(-1).values, b.sort(-1).values
            d += int((sa != sb).any(-1).sum()); n += a.shape[0]
        per_layer.append(round(d / n, 5))
    return per_layer
out = {"loss": {k: {"mean": round(v[0], 5), "rows": [round(x, 5) for x in v[1]]} for k, v in res.items()},
       "flip_fraction_per_layer": {m: flips(m) for m in ("D2", "B")}, "tokens_per_layer": sum(len(r) for r in rows),
       "pinning_removes": round((res["B"][0] - res["B_pin"][0]) / (res["B"][0] - res["A"][0]), 4) if res["B"][0] != res["A"][0] else None}
for m in ("D2", "B"):
    f = out["flip_fraction_per_layer"][m]
    out.setdefault("first_flip_layer", {})[m] = next((l for l, v in enumerate(f) if v > 0), None)
    out.setdefault("mean_flip_fraction", {})[m] = round(sum(f) / len(f), 5)
print("P129-ROUTING", json.dumps(out))
