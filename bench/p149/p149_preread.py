#!/usr/bin/env python3
"""p149_preread.py -- P149 Amendment 4's router pre-read (bench/p149/PREREG-p149.md): a PREDICTION, USD 0, on the lab's RTX
A2000, posted before box Q+S is created. Graded in the read, never ruling.

1. P109's W16 prompts (16 rows of 512 wikitext tokens) are decoded GREEDILY for 160 tokens, eagerly with a KV cache, on
   e4b's streaming loader (experts NF4, offloaded to pinned host RAM, as P147 ran Mixtral on this card): once as S0 (the
   bf16 attention projections), then as S1 (``engines.int8_attn.enable_serve_attn_int8``: K37's kernels, compiled on
   sm_86, at decode; a transient dequantization at the prefill).
2. Both streams are replayed eagerly through S0's model: each sparse-MoE block's input routed by the block's own gate (the
   top ``num_experts_per_tok`` of its logits), at the LONG decode steps' inputs (each row's first 159 tokens).
3. Per stream: the distinct experts per layer-step, the per-expert row histogram, the 16-row tile count and the largest
   per-expert row count (``p149_box.route_stats``, the box's own reduction).
4. Priced two ways, with f the grouped expert GEMM's share of Mixtral's W16 step (P145's per-name census at B = 16:
   0.911): g16_routing = 1 / (1 - f + f * X_B / X_A), X the tile count (the tile model) or the distinct count (the bytes
   model). The band is the two models' span, widened by 0.002 each side.

The A2000's tokens are not the 5090's (the other GEMMs' reduction orders differ by architecture), so this predicts.

    python p149_preread.py --model DIR --prompts prompts.json --out preread.json
    python p149_preread.py --self-test
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import p149_box  # noqa: E402  (route_stats: the box's own reduction)

MOE_SHARE_W16 = 0.911     # P145 (p145-5090-1, PERNAME_TABLE at B = 16): (29.562 + 18.940) / 53.224 ms
LONG, BAND_PAD = 160, 0.002


def predict(a: dict, b: dict, f: float = MOE_SHARE_W16) -> dict:
    """The two models' g16_routing (S0's kernels on S1's tokens over S0's own) and the band."""
    tile = 1 / (1 - f + f * b["tiles_mean"] / a["tiles_mean"])
    byte = 1 / (1 - f + f * b["distinct_mean"] / a["distinct_mean"])
    lo, hi = min(tile, byte) - BAND_PAD, max(tile, byte) + BAND_PAD
    return {"tile_model": round(tile, 5), "bytes_model": round(byte, 5), "band": [round(lo, 4), round(hi, 4)], "f": f}


def greedy(model, rows, n, device):
    """Greedy decode of ``n`` tokens for every row at once (equal-length prompts), eager, with the model's KV cache."""
    import torch
    ids = torch.tensor(rows, device=device)
    out, past = [], None
    with torch.no_grad():
        o = model(input_ids=ids, use_cache=True, logits_to_keep=1)
        for _ in range(n):
            nxt = o.logits[:, -1].argmax(-1)
            out.append(nxt.tolist())
            past = o.past_key_values
            o = model(input_ids=nxt[:, None], past_key_values=past, use_cache=True, logits_to_keep=1)
    return [list(r) for r in zip(*out)]


def replay(model, rows, stream, device):
    """Routing picks per MoE layer and step (LONG - 1 steps), from each block's gate on its own input."""
    import torch
    mods = [m for m in model.modules() if type(m).__name__.endswith("SparseMoeBlock")]
    k_top = int(model.config.num_experts_per_tok)
    got = {i: None for i in range(len(mods))}
    hooks = [m.register_forward_pre_hook(lambda mod, args, _i=i: got.__setitem__(_i, args[0].detach()))
             for i, m in enumerate(mods)]
    picks = {i: [[] for _ in range(LONG - 1)] for i in range(len(mods))}
    try:
        for p, s in zip(rows, stream):
            with torch.no_grad():
                model(input_ids=torch.tensor([list(p) + list(s[:LONG - 1])], device=device), use_cache=False,
                      logits_to_keep=1)
            for i, m in enumerate(mods):
                h = got[i].reshape(-1, got[i].shape[-1])[len(p):len(p) + LONG - 1]
                top = (h.float() @ m.gate.weight.float().T).topk(k_top, dim=-1).indices.tolist()
                for step, pr in enumerate(top):
                    picks[i][step].append(tuple(sorted(pr)))
    finally:
        for h_ in hooks:
            h_.remove()
    return picks


def main_run(a) -> int:
    import torch
    import experts4bit_qlora as e4b
    from experts4bit_qlora.engines import int8_attn
    pf = json.load(open(a.prompts))
    rows = pf["rows"]
    rec = {"lane": "P149 Amendment 4 pre-read", "prompts_sha256": pf["prompts_sha256"], "card": torch.cuda.get_device_name(0),
           "phases": {}}
    t0 = time.time()
    model, _ = e4b.load_moe_4bit_streaming(a.model, "cuda", torch.bfloat16, 16, 16, offload=True, pin=True,
                                            prefetch=True, quant_type="nf4")
    model.eval()
    rec["phases"]["load"] = round(time.time() - t0, 1)
    t0 = time.time()
    s0 = greedy(model, rows, LONG, "cuda")
    rec["phases"]["s0_decode"] = round(time.time() - t0, 1)
    host = {(id(mod), n): c.weight.detach().cpu() for mod in model.modules() if type(mod).__name__.endswith("Attention")
            for n, c in mod.named_children() if type(c) is torch.nn.Linear}       # S0's weights, kept on the host
    rec["s1_swapped"] = int8_attn.enable_serve_attn_int8(model)
    t0 = time.time()
    s1 = greedy(model, rows, LONG, "cuda")
    rec["phases"]["s1_decode"] = round(time.time() - t0, 1)
    rec["s1_engagement"] = int8_attn.engagement(model)
    restored = 0                                                    # back to S0's model for the replay
    for mod in model.modules():
        for n, c in list(mod.named_children()):
            if isinstance(c, int8_attn.Int8Linear):
                lin = torch.nn.Linear(c.in_features, c.out_features, bias=False, device="cuda", dtype=torch.bfloat16)
                with torch.no_grad():
                    lin.weight.copy_(host[(id(mod), n)])
                setattr(mod, n, lin)
                restored += 1
    del host
    torch.cuda.empty_cache()
    rec["s0_restored"] = restored
    first = []
    for x, y in zip(s0, s1):
        i = next((k for k, (p, q) in enumerate(zip(x, y)) if p != q), None)
        first.append(i)
    rec["streams"] = {"rows_identical": sum(f is None for f in first), "first_divergent_step": first}
    t0 = time.time()
    stats = {name: p149_box.route_stats(replay(model, rows, s, "cuda")) for name, s in (("A", s0), ("B", s1))}
    rec["phases"]["replay"] = round(time.time() - t0, 1)
    rec["routing"] = stats
    rec["prediction"] = predict(stats["A"], stats["B"])
    json.dump(rec, open(a.out, "w"), indent=1)
    print("P149_PREREAD " + json.dumps({"streams": rec["streams"]["rows_identical"], "routing": stats,
                                        "prediction": rec["prediction"]}), flush=True)
    return 0


def self_test() -> int:
    cases = []
    a = {"tiles_mean": 8.0, "distinct_mean": 7.9}
    p = predict(a, dict(a))
    cases.append(("equal streams predict 1.0", p["tile_model"] == 1.0 and p["bytes_model"] == 1.0
                  and p["band"] == [0.998, 1.002]))
    p = predict(a, {"tiles_mean": 8.08, "distinct_mean": 7.9})
    cases.append(("1 % more tiles on B predicts S1's tokens slower", p["tile_model"] < 1 and p["bytes_model"] == 1.0
                  and p["band"][0] < p["tile_model"] < p["band"][1]))
    cases.append(("f is P145's B = 16 grouped GEMM share", abs(MOE_SHARE_W16 - (29.562 + 18.940) / 53.224) < 5e-4))
    bad = [c for c, ok in cases if not ok]
    print(f"p149_preread self-test: {len(cases) - len(bad)}/{len(cases)} OK" + (f"; FAILED {bad}" if bad else ""))
    return 1 if bad else 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model")
    ap.add_argument("--prompts")
    ap.add_argument("--out")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args(argv)
    if a.self_test:
        return self_test()
    return main_run(a)


if __name__ == "__main__":
    sys.exit(main())
