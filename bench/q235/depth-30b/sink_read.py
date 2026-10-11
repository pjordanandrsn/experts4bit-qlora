"""EXPLORATORY (not a registered line): where the full-state error's norm lives. Per layer, over real tokens: the share of the
reference's squared norm and of the box's squared error held by position 0 of each row (the attention sink), that position's own
relative error, the per-token median relative error, and the state statistic with position 0 excluded, for the box and each plant.

    python sink_read.py REF_DIR TOKENS OUT.json RUN=DIR [RUN=DIR ...]
"""
import json
import sys

import torch

ref, tokens, out = sys.argv[1:4]
runs = dict(a.split("=", 1) for a in sys.argv[4:])
m = torch.load(tokens, weights_only=True)["attention_mask"].bool()
m0 = m.clone()
m0[:, 0] = False
sink = torch.zeros_like(m)
sink[:, 0] = True
sink &= m
rec = {"layers": {}, "note": "exploratory; position 0 of each row is the attention sink"}
for L in (3, 10, 20, 23, 24, 25, 30, 40, 47):
    b = torch.load(f"{ref}/layer{L:03d}.pt", weights_only=True).float()
    row = {"ref_norm_sink": float(b[sink].norm(dim=-1).mean()), "ref_norm_median_other": float(b[m0].norm(dim=-1).median()),
           "ref_sq_share_sink": float((b[sink] ** 2).sum() / (b[m] ** 2).sum())}
    for k, d in runs.items():
        a = torch.load(f"{d}/layer{L:03d}.pt", weights_only=True).float()
        e = a - b
        row[k] = {"state": float(e[m].norm() / b[m].norm()), "state_without_sink": float(e[m0].norm() / b[m0].norm()),
                  "err_sq_share_sink": float((e[sink] ** 2).sum() / (e[m] ** 2).sum()),
                  "sink_rel_err": float(e[sink].norm() / b[sink].norm()),
                  "per_token_median_without_sink": float((e[m0].norm(dim=-1) / b[m0].norm(dim=-1)).median())}
    rec["layers"][L] = row
json.dump(rec, open(out, "w"), indent=1)
print(json.dumps({L: {k: round(v["state_without_sink"], 3) for k, v in r.items() if isinstance(v, dict)} for L, r in rec["layers"].items()}))
