"""Q235 Shape B: staged bytes per step, and the box-time effect of two staging layouts. Analytic, from records only.

Inputs (sha256 prefixes in README.md):
- box 1 (q235-pro6000-2) box/box.json: 254,124,490,752 B staged in 187 copies per Shape A step -> bytes per whole-layer staging;
- box 1 box/nf4.pt: the routers' top-k for every layer at Shape A's 8 x 331 positions (the measured routing);
- box 1 tokens.pt: the attention mask, from which Shape B's micro-batches are rebuilt as the box builds them (row pairs, each
  trimmed to its longer row: 134 / 266 / 662 / 158 positions, as the A2000 f17f0744 leg recorded);
- e4b#1680 (bench/q235/am7/PREDICTION-C.md): the PRO 6000's experts compute per micro-batch per layer, 15.13 ms.

    python staged_bytes.py BOX_RECEIPTS_DIR [--h2d 21.76 --h2d 50]

BOX_RECEIPTS_DIR holds box/box.json, box/nf4.pt and tokens.pt. Nothing here runs a model or touches a GPU."""
import argparse
import json
import os
import statistics

import torch

N_LAYERS, MICRO_BATCH_ROWS, N_EXPERTS = 94, 2, 128
COMPUTE_MS = 15.13          # e4b#1680: PRO 6000 experts compute per micro-batch per layer (fwd + recompute + bwd), shape-A ratios
# q235_box.py at 13c05fce, a Shape B box (no --repeat): l0, l1 and ga1 are Shape A passes; rule 8's check at steps 0 and 1
# (separate + token + token, each), the two ga_plants, the two registered steps and Amendment 7's bound timing (warm-up + 4) are
# Shape B passes.
SHAPE_A_PASSES, SHAPE_B_PASSES = 3, 6 + 2 + 2 + 5


def micro_batches(mask):
    lens = mask.sum(1).tolist()
    return [(i, i + 1, max(lens[i], lens[i + 1])) for i in range(0, mask.shape[0], MICRO_BATCH_ROWS)]


def hit_fractions(topk, mask):
    """Per micro-batch: the mean over layers of the fraction of experts its positions route to; and, per layer, the fraction
    the four micro-batches route to together."""
    mbs = micro_batches(mask)
    per_mb, union = [[] for _ in mbs], []
    for t in topk:
        t = t.reshape(mask.shape[0], -1, t.shape[-1])
        u = set()
        for j, (a, b, w) in enumerate(mbs):
            s = set(t[a, :w].reshape(-1).tolist()) | set(t[b, :w].reshape(-1).tolist())
            per_mb[j].append(len(s) / N_EXPERTS)
            u |= s
        union.append(len(u) / N_EXPERTS)
    return [2 * w for _, _, w in mbs], [statistics.mean(f) for f in per_mb], statistics.mean(union)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("receipts")
    ap.add_argument("--h2d", type=float, action="append", default=None, help="GB/s (default: 21.76 measured, and 50)")
    a = ap.parse_args()
    step = json.load(open(os.path.join(a.receipts, "box", "box.json")))["step"]
    per_call = step["h2d_bytes_offload"] / step["h2d_copies_offload"] / 1e9
    topk = torch.load(os.path.join(a.receipts, "box", "nf4.pt"), weights_only=False)["router_topk"]
    mask = torch.load(os.path.join(a.receipts, "tokens.pt"), weights_only=True)["attention_mask"]
    positions, mb_hits, union_hit = hit_fractions(topk, mask)
    m = len(positions)
    calls_a = 2 * N_LAYERS - 1                       # per micro-batch: n-1 forward (layer 0 stays resident) + n recompute
    layouts = {
        "(a) as-is": m * calls_a,
        "(b) layer-major": calls_a,
        "(c) routed experts per micro-batch": calls_a * sum(mb_hits),
        "(b)+(c) layer-major, routed union": calls_a * union_hit,
    }
    compute_s = m * N_LAYERS * COMPUTE_MS / 1e3
    out = {"bytes_per_layer_staging_gb": round(per_call, 6), "micro_batch_positions": positions,
           "experts_hit_mean_per_micro_batch": [round(h, 4) for h in mb_hits], "experts_hit_union_mean": round(union_hit, 4),
           "experts_compute_s_per_pass": round(compute_s, 2), "box_passes": {"shape_a": SHAPE_A_PASSES, "shape_b": SHAPE_B_PASSES},
           "layouts": {}}
    shape_a_gb = SHAPE_A_PASSES * calls_a * per_call
    for name, calls in layouts.items():
        gb = calls * per_call
        row = {"stagings_per_pass": round(calls, 1), "gb_per_pass": round(gb, 1)}
        for h in a.h2d or [21.76, 50.0]:
            row[f"staging_s_per_pass@{h:g}"] = round(gb / h, 2)
            row[f"pass_s@{h:g}"] = round(gb / h + compute_s, 2)
            row[f"box_staging_s@{h:g}"] = round((SHAPE_B_PASSES * gb + shape_a_gb) / h, 1)
        out["layouts"][name] = row
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
