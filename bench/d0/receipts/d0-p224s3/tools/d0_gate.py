"""D0 plumbing gate (PREREG-d0.md): on a 2-layer truncation of the base, d0_forward matches transformers'
OlmoeForCausalLM (fp32, the same raw weights, no adapter) to max |dlogit| <= 1e-3 on two held-out rows.

    python d0_gate.py --ckpt DIR --heldout heldout.json [--layers 2]
"""
import argparse
import json
import sys

import torch

import d0_forward as fw
import d0lib


class RawExperts:
    def __init__(self, ck, layer, n):
        self.ck, self.layer, self.n = ck, layer, n

    def __call__(self, e):
        return self.ck.expert(self.layer, e, "gate_up"), self.ck.expert(self.layer, e, "down")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--heldout", required=True)
    ap.add_argument("--layers", type=int, default=2)
    a = ap.parse_args()
    from transformers import OlmoeConfig, OlmoeForCausalLM

    ck = d0lib.Checkpoint(a.ckpt)
    cfg = d0lib.load_cfg(a.ckpt)
    rows = [torch.tensor(r["ids"]) for r in json.load(open(a.heldout))["rows"][:2]]
    hcfg = OlmoeConfig.from_pretrained(a.ckpt)
    hcfg.num_hidden_layers = a.layers
    model = OlmoeForCausalLM(hcfg).float().eval()
    want = {k: v for k, v in model.state_dict().items()}
    sd = {}
    for k in want:
        if ".mlp.experts." in k:
            layer = int(k.split(".")[2])
            stack = [ck.expert(layer, e, "gate_up" if k.endswith("gate_up_proj") else "down") for e in range(cfg["num_experts"])]
            sd[k] = torch.stack(stack)
        else:
            sd[k] = ck.get(k).float()
    model.load_state_dict(sd, strict=True)
    worst = 0.0
    hd = cfg["hidden_size"] // cfg["num_attention_heads"]
    cos, sin = fw.rope_tables(max(len(r) for r in rows), hd, cfg["rope_theta"])
    for ids in rows:
        with torch.no_grad():
            ref = model(ids.unsqueeze(0)).logits[0]
            x = ck.get("model.embed_tokens.weight").float()[ids]
            off = [0, len(ids)]
            for layer in range(a.layers):
                lw = fw.LayerWeights(ck, layer, None)
                x = fw.attention(x, off, lw, cfg, cos, sin, use_lora=False)
                h, tv, ti = fw.route(x, lw, cfg)
                ex = RawExperts(ck, layer, cfg["num_experts"])
                x, _ = fw.moe(x, h, tv, ti, ex)
            mine = torch.nn.functional.linear(fw.rms(x, ck.get("model.norm.weight").float(), cfg["rms_norm_eps"]),
                                              ck.get("lm_head.weight").float())
        d = (mine - ref).abs().max().item()
        worst = max(worst, d)
        print(f"row len {len(ids)}: max |dlogit| {d:.3e}  (logit scale {ref.abs().max().item():.1f})")
    ok = worst <= 1e-3
    print("D0_GATE " + json.dumps({"layers": a.layers, "max_abs_dlogit": worst, "ok": ok}))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
