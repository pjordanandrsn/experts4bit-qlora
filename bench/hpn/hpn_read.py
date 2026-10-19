#!/usr/bin/env python3
"""bench/hpn/hpn_read.py -- HPN's fp32 reference reader F (bench/hpn/HPN-DESIGN.md).

F is the one evaluator's forward (bench/hp1/hp1_eval.py + hp1_lib.py) with every bf16 cast removed:
- each stored slot is dequantized by the evaluator's own dequantizer, with the QuantState's dtype set to fp32 and no cast;
- the expert and attention projections, the routing sums, the norms, attention and logits all run in fp32;
- autocast is off.
The NF4 bytes, the routing and the module layout are the evaluator's. Only the arithmetic precision changes, so E(r) - F(r) is the
evaluator's own bf16 rounding on row r.

    python hpn_read.py --base DIR --tokens TOKENS.json --eval-n N --model M --revision R --out OUT.json [--runs 2]

The output carries the per-row losses of every run (the two-reruns control, N0). A diagnostic only: never a scored number.
"""
import argparse
import json
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
for d in (os.path.join(HERE, "..", "hp1"), HERE):          # in the repo; on a box the staged files sit side by side
    sys.path.insert(0, d)
import hp1_eval  # noqa: E402
import hp1_lib  # noqa: E402

DEV = hp1_eval.DEV


def deq32(r, device):
    """hp1_lib.dequant_raw without its bf16 casts: 4-bit slots dequantize with an fp32 QuantState (the product code x absmax kept in
    fp32); bf16 slots (embeddings, norms, routers, lm_head) are upcast exactly."""
    if r["fmt"] == "bf16":
        return r["w"].to(device=device, dtype=torch.float32)
    from bitsandbytes.functional import QuantState, dequantize_4bit, get_4bit_type
    code = r["code"].to(device) if r.get("code") is not None else get_4bit_type(r["fmt"], device=device)
    st = QuantState(absmax=r["absmax"].to(device=device, dtype=torch.float32), shape=torch.Size(r["shape"]), code=code,
                    blocksize=int(r["blocksize"]), quant_type=r["fmt"], dtype=torch.float32)
    w = dequantize_4bit(r["packed"].to(device).reshape(-1, 1), quant_state=st).to(torch.float32)
    return w.t().contiguous() if r.get("transposed") else w


class F32Experts(nn.Module):
    """hp1_lib.EagerExperts in fp32 throughout (same signature, same routing, same chunk(2) split; no adapter: F reads bases only)."""

    def __init__(self, deq, num_experts, act_fn):
        super().__init__()
        self.deq, self.num_experts, self.act_fn = deq, int(num_experts), act_fn

    def forward(self, hidden_states, top_k_index, top_k_weights):
        x_all = hidden_states.to(torch.float32)
        out = torch.zeros_like(x_all)
        mask = F.one_hot(top_k_index, num_classes=self.num_experts).permute(2, 1, 0)
        hit = torch.greater(mask.sum(dim=(-1, -2)), 0).nonzero(as_tuple=False).view(-1).tolist()
        for e in hit:
            pos, tok = torch.where(mask[e])
            h = F.linear(x_all[tok], self.deq("gate_up", e))
            gate, up = h.chunk(2, dim=-1)
            y = F.linear(self.act_fn(gate) * up, self.deq("down", e))
            out.index_add_(0, tok, y * top_k_weights[tok, pos, None].to(torch.float32))
        return out.to(hidden_states.dtype)


class F32Linear(nn.Module):
    """hp1_lib.EagerLinear in fp32 (no bias: the evaluator refuses a biased projection too)."""

    def __init__(self, deq):
        super().__init__()
        self.deq = deq

    def forward(self, x):
        return F.linear(x.to(torch.float32), self.deq()).to(x.dtype)


def build_model_f32(store, model_id, revision, deq=None):
    """hp1_eval.build_model's skeleton in fp32: HF's class from the pinned config, eager attention, the experts and q/k/v/o replaced by
    the fp32 modules above, every other weight loaded from the export through deq32 (exact upcasts), the rotary buffers rebuilt."""
    from transformers import AutoConfig, AutoModelForCausalLM
    deq = deq or (lambda key: deq32(store.raw(key), store.device))
    cfg = AutoConfig.from_pretrained(model_id, revision=revision)
    cfg._attn_implementation = "eager"
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(cfg, dtype=torch.float32)
    keys = set(store.keys())
    for L, layer in enumerate(model.model.layers):
        mlp, attn = layer.mlp, layer.self_attn
        if (L, "gate_up", 0) not in keys:
            raise RuntimeError(f"base {store.dir}: layer {L} has no expert slots")
        mlp.experts = F32Experts((lambda kind, e, L=L: deq((L, kind, e))), int(cfg.num_experts), mlp.experts.act_fn)
        for pn, kind in (("q_proj", "q"), ("k_proj", "k"), ("v_proj", "v"), ("o_proj", "o")):
            if getattr(getattr(attn, pn), "bias", None) is not None:
                raise RuntimeError(f"layer {L} {pn}: a biased projection; the export carries no bias")
            setattr(attn, pn, F32Linear(lambda L=L, kind=kind: deq((L, kind, -1))))
    model.to_empty(device=store.device)
    loaded = set()

    def put(param, key):
        w = deq(key)
        if tuple(w.shape) != tuple(param.shape):
            raise RuntimeError(f"{key}: export {list(w.shape)} vs model {list(param.shape)}")
        with torch.no_grad():
            param.copy_(w)
        loaded.add(id(param))
    put(model.model.embed_tokens.weight, (-1, "embed", -1))
    put(model.lm_head.weight, (-1, "lm_head", -1))
    put(model.model.norm.weight, (-1, "final_norm", -1))
    for L, layer in enumerate(model.model.layers):
        put(layer.mlp.gate.weight, (L, "router", -1))
        for name, m in layer.named_modules():
            last = name.split(".")[-1]
            if "norm" in last.lower() and isinstance(getattr(m, "weight", None), nn.Parameter):
                put(m.weight, (L, "norm:" + last, -1))
    missing = [n for n, p in model.named_parameters() if id(p) not in loaded]
    if missing:
        raise RuntimeError(f"parameters not covered by the base export: {missing[:6]}")
    model.model.rotary_emb = type(model.model.rotary_emb)(config=cfg, device=store.device)
    model.eval()
    return model


def read(model, rows, tc1, runs=2):
    """[per-row losses] for each run: tc1_arm.eval_loss with autocast OFF (fp32), so the losses are recorded as every reader's are."""
    out = []
    with torch.no_grad():
        for _ in range(runs):
            out.append(tc1.eval_loss(model, rows, lambda x: {}, False)[1])
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--base", required=True)
    ap.add_argument("--tokens", required=True)
    ap.add_argument("--eval-n", type=int, required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--revision", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--runs", type=int, default=2)
    a = ap.parse_args(argv)
    tc1 = hp1_eval.load_tc1_arm()
    tc1.DEV = DEV
    tk = json.load(open(a.tokens))
    rows = tk["eval"][: a.eval_n]
    if len(rows) != a.eval_n:
        sys.exit(f"tokens file has {len(rows)} eval rows, need {a.eval_n}")
    store = hp1_lib.BaseStore(a.base, DEV)
    digest, n_slots = store.digest()
    if digest != store.meta.get("train_base_digest"):
        sys.exit(f"STRICT: the export's digest {digest[:12]} != its training run's {str(store.meta.get('train_base_digest'))[:12]}")
    runs = read(build_model_f32(store, a.model, a.revision), rows, tc1, a.runs)
    rec = {"reader": "F", "tool": "bench/hpn/hpn_read.py", "base": a.base, "digest": digest, "n_slots": n_slots,
           "tokens": {"path": a.tokens, "sha256": tk.get("sha256"), "eval_n": a.eval_n}, "model": a.model, "revision": a.revision,
           "device": torch.cuda.get_device_name(0) if DEV == "cuda" else "cpu", "torch": torch.__version__, "runs": runs,
           "row_mean": [round(sum(r) / len(r), 6) for r in runs]}
    json.dump(rec, open(a.out, "w"), indent=1)
    print(json.dumps({"reader": "F", "base": a.base, "row_mean": rec["row_mean"]}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
