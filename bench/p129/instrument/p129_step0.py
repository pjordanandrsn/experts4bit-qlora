"""P129 Phase 2 read, QUALITY_FAIL investigation (RTX A2000, forward only): the step-0 held-out loss of the real Qwen3-30B-A3B at the
pinned revision, set up as TC1's shipped e4b arm (NF4 attention, bf16 adapters, native init: lora_B == 0), on the box's own eight
held-out rows, under five ways to compute the q/k/v base projections:
  A  the stock training path (three dequantize + matmul);
  D2 A with q's matmul split in two along N (a rounding-class change of the same kind);
  D1 A with the three matmuls in fp32 from the same dequantized weights, rounded to bf16 (the accurate reference);
  C  one matmul over the three dequantized weights concatenated, inside the stock forward (the fused arithmetic, no fused module);
  C2 the stock forward fed q/k/v from a FusedQKVLoRA built on the same projections (B's exact q/k/v, without serving's forward);
  B  E4B_TRAIN_FUSE_QKV's fused module and serving's fused forward (what the box ran as q1).
B == C2 bitwise says serving's forward adds nothing beyond the q/k/v values; A, C, C2, D1, D2 spread the rounding floor.
Localization (one held-out row, A / C2 / B): per layer, q and k after their norms (rope inputs) and after rope, v, the attention
interface output, the module output after o_proj and the decoder layer output; the first layer and tensor where the paths differ,
and how the difference grows with depth."""
import json, os, sys, time
import torch
import torch.nn.functional as F
from bitsandbytes.functional import dequantize_4bit
from experts4bit_qlora import load_moe_4bit_streaming, enable_fast_train
from experts4bit_qlora.lora import add_attention_lora, quantize_attention_projections_4bit
from experts4bit_qlora.engines.train_qkv_fuse import enable_train_fuse_qkv, TRAIN_QKV_STATS, FusedQKVLoRA
from transformers.models.qwen3_moe import modeling_qwen3_moe as MQ
REV = "ad44e777bcd18fa416d9da3bd8f70d33ebb85d39"
rows = json.load(open(sys.argv[1]))["eval"]
t0 = time.time()
SRC = os.environ.get("P129_MODEL_DIR", "Qwen3-30B-A3B")   # the local copy, verified file-for-file against REV's sha256s first
model, cfg = load_moe_4bit_streaming(SRC, "cuda", torch.bfloat16, 16, 16, offload=True, pin=True, prefetch=False, quant_type="nf4")
print("loaded", round(time.time() - t0), "s; commit", getattr(cfg, "_commit_hash", None), flush=True)
quantize_attention_projections_4bit(model)
model.config.use_cache = False
add_attention_lora(model, 16, 16, torch.bfloat16)
enable_fast_train(model, verbose=False)
attn = [L.self_attn for L in model.model.layers]
assert all(float(getattr(a, n).lora_B.abs().max()) == 0.0 for a in attn for n in ("q_proj", "k_proj", "v_proj")), "lora_B not zero"
MODE = {"m": "A"}
FQ = [FusedQKVLoRA(a) for a in attn]                     # B's base and adapter arithmetic, built on the same projections
REC = {"on": False, "calls": [], "attn": [], "o": [], "layer": []}
_rope = MQ.apply_rotary_pos_emb
def rope_rec(q, k, cos, sin, *a, **kw):
    qo, ko = _rope(q, k, cos, sin, *a, **kw)
    if REC["on"]:
        REC["calls"].append({"q_pre": q.detach().float().cpu(), "k_pre": k.detach().float().cpu(),
                             "q_post": qo.detach().float().cpu(), "k_post": ko.detach().float().cpu()})
    return qo, ko
MQ.apply_rotary_pos_emb = rope_rec
IMPL = model.config._attn_implementation
def wrap_attn(fn):
    def w(module, q, k, v, *a, **kw):
        out = fn(module, q, k, v, *a, **kw)
        if REC["on"]:
            REC["attn"].append({"v": v.detach().float().cpu(), "attn": out[0].detach().float().cpu()})
        return out
    return w
if IMPL in MQ.ALL_ATTENTION_FUNCTIONS:
    MQ.ALL_ATTENTION_FUNCTIONS[IMPL] = wrap_attn(MQ.ALL_ATTENTION_FUNCTIONS[IMPL])
else:
    MQ.eager_attention_forward = wrap_attn(MQ.eager_attention_forward)
for a in attn:
    a.register_forward_hook(lambda m, i, o: REC["o"].append(o[0].detach().float().cpu()) if REC["on"] else None)
for L in model.model.layers:
    L.register_forward_hook(lambda m, i, o: REC["layer"].append((o[0] if isinstance(o, tuple) else o).detach().float().cpu()) if REC["on"] else None)
def deq(p):
    return dequantize_4bit(p.base.weight.data, p.base.weight.quant_state).to(torch.bfloat16)
def patch(a, fq_mod):
    q, k, v = a.q_proj, a.k_proj, a.v_proj
    oq, ok, ov = q.forward, k.forward, v.forward
    cache = {}
    def fq(x):
        m = MODE["m"]
        if m == "A":
            return oq(x)
        if m == "C2":
            out = fq_mod(x)
            nq, nk = fq_mod.ns[0], fq_mod.ns[1]
            cache["x"], cache["k"], cache["v"] = x, out[..., nq:nq + nk], out[..., nq + nk:]
            return out[..., :nq]
        Wq, Wk, Wv = deq(q), deq(k), deq(v)
        if m == "C":
            out = F.linear(x, torch.cat([Wq, Wk, Wv]))
            nq, nk = Wq.shape[0], Wk.shape[0]
            cache["x"], cache["k"], cache["v"] = x, out[..., nq:nq + nk], out[..., nq + nk:]
            return out[..., :nq]
        if m == "D1":
            cache["x"] = x
            cache["k"] = F.linear(x.float(), Wk.float()).to(x.dtype)
            cache["v"] = F.linear(x.float(), Wv.float()).to(x.dtype)
            return F.linear(x.float(), Wq.float()).to(x.dtype)
        if m == "D2":
            h = Wq.shape[0] // 2
            cache.clear()
            return torch.cat([F.linear(x, Wq[:h]), F.linear(x, Wq[h:])], -1)
    def fk(x):
        return cache["k"] if MODE["m"] in ("C", "C2", "D1") and cache.get("x") is x else ok(x)
    def fv(x):
        return cache["v"] if MODE["m"] in ("C", "C2", "D1") and cache.get("x") is x else ov(x)
    q.forward, k.forward, v.forward = fq, fk, fv
    return (q, oq), (k, ok), (v, ov)
saved = [patch(a, f) for a, f in zip(attn, FQ)]
@torch.no_grad()
def record(row):
    model.eval()
    for key in ("calls", "attn", "o", "layer"):
        REC[key] = []
    REC["on"] = True
    x = torch.tensor(row, dtype=torch.long).unsqueeze(0).cuda()
    loss = float(model(input_ids=x, labels=x).loss)
    REC["on"] = False
    per = []
    for i in range(len(attn)):
        d = dict(REC["calls"][i]); d.update(REC["attn"][i]); d["o"] = REC["o"][i]; d["layer"] = REC["layer"][i]
        per.append(d)
    return loss, per
def compare(P, Q):
    out, first = [], None
    for i, (p, q) in enumerate(zip(P, Q)):
        row = {}
        for t in ("q_pre", "k_pre", "q_post", "k_post", "v", "attn", "o", "layer"):
            a, b = p[t], q[t]
            same = bool(torch.equal(a, b))
            rel = float((a - b).norm() / a.norm().clamp_min(1e-30))
            row[t] = {"equal": same, "rel": rel, "maxabs": float((a - b).abs().max())}
            if not same and first is None:
                first = (i, t)
        out.append(row)
    return {"first_difference": first, "per_layer": [{t: round(v["rel"], 8) for t, v in r.items()} for r in out],
            "equal_layers": [i for i, r in enumerate(out) if all(v["equal"] for v in r.values())]}
LOC = {}
@torch.no_grad()
def heldout():
    model.eval()
    per = []
    for ids in rows:
        x = torch.tensor(ids, dtype=torch.long).unsqueeze(0).cuda()
        per.append(float(model(input_ids=x, labels=x).loss))
    return sum(per) / len(per), per
res = {}
for m in ("A", "C2"):
    MODE["m"] = m
    LOC[m] = record(rows[0])
for m in ("A", "D2", "D1", "C", "C2"):
    MODE["m"] = m
    t1 = time.time(); mean, per = heldout()
    res[m] = {"mean": round(mean, 5), "rows": [round(v, 5) for v in per], "s": round(time.time() - t1)}
    print(m, res[m], flush=True)
MODE["m"] = "A"
for trio in saved:
    for mod, f in trio:
        mod.forward = f                         # back to the modules' own forwards before the fused module takes their bytes
n = enable_train_fuse_qkv(model)
assert n == len(attn) and not TRAIN_QKV_STATS["refused"], (n, TRAIN_QKV_STATS)
LOC["B"] = record(rows[0])
loc = {"row0_loss": {k: round(v[0], 6) for k, v in LOC.items()}, "A_vs_B": compare(LOC["A"][1], LOC["B"][1]),
       "C2_vs_B": compare(LOC["C2"][1], LOC["B"][1]), "A_vs_C2": compare(LOC["A"][1], LOC["C2"][1]), "attn_impl": IMPL}
print("P129-LOCALIZE", json.dumps(loc))
t1 = time.time(); mean, per = heldout()
res["B"] = {"mean": round(mean, 5), "rows": [round(v, 5) for v in per], "s": round(time.time() - t1), "fused": n}
print("B", res["B"], flush=True)
print("P129-STEP0", json.dumps(res))
