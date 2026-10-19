"""p153-w16-router-replay's review-suggested robustness check (not part of the pre-statement): from the replay's saved
router picks, how the 128 row assignments of each bucket-16 step (16 rows x top-8) spread across experts, S1 vs S0.

A grouped small-M GEMM's time depends on the distinct experts (the pre-stated metric) and also on each expert's row count
M. So, summed over every layer and counted step: the distinct experts, the max rows per expert, the experts with M > 1,
M >= 4 and M >= 8, the sum of M^2, and the histogram of M. CPU only; reads nothing but the picks file.

usage: python rr_spread.py rr.json.pt rr_spread.json
"""
import json
import sys

import torch

picks_path, out_path = sys.argv[1], sys.argv[2]
d = torch.load(picks_path, weights_only=True)            # tensors, a dict of floats and a list of ints only
p = d["picks"].long()                                             # [layers, 32 rows (S0 0-15, S1 16-31), steps, top_k]
L, B, S, K = p.shape
E = 128
out = {"layers": L, "steps": S, "top_k": K, "experts": E, "arms": {}}
for arm, rows in (("S0", slice(0, 16)), ("S1", slice(16, 32))):
    x = p[:, rows]
    cnt = torch.zeros(L, S, E, dtype=torch.long)
    for r in range(16):
        cnt.scatter_add_(2, x[:, r], torch.ones_like(x[:, r]))
    assert int(cnt.sum()) == L * S * 16 * K, "every layer-step holds 16 x top-k assignments"
    out["arms"][arm] = {
        "distinct": int((cnt > 0).sum()), "max_rows_sum": int(cnt.max(-1).values.sum()),
        "m_gt_1": int((cnt > 1).sum()), "m_ge_4": int((cnt >= 4).sum()), "m_ge_8": int((cnt >= 8).sum()),
        "m_squared_sum": int((cnt * cnt).sum()), "histogram_m_0_to_16": torch.bincount(cnt.flatten(), minlength=17)[:17].tolist(),
    }
s0, s1 = out["arms"]["S0"], out["arms"]["S1"]
out["ratio_S1_over_S0"] = {k: s1[k] / s0[k] for k in s0 if k != "histogram_m_0_to_16"}
json.dump(out, open(out_path, "w"), indent=1)
print(json.dumps(out["ratio_S1_over_S0"]))
