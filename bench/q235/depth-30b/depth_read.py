"""Q235 depth test reader: the lines registered 2026-10-10 22:41Z, before the runs, applied mechanically.

Inputs: the bf16 48-layer reference, the NF4 box, and the two plants (layer 24), all over the same tokens' real positions.
(1)  Full-state statistic, Amendment 1 bands (q235_gate1.band_failures / plant_fails_as_registered).
     Line 1 FAILS iff either plant does not fail its registered line -> full-state bands VOID for any deep box.
(2a) PRIMARY: the contribution statistic separates iff BOTH plants' u_L >= 2 x the box's u_L (L = planted layer).
(2b) REPORTED: each plant's u_L against 2 x the box's max u_i over i = 4..44.
Also reported: the box's u_i profile (1..47), the plants' u_i for L+1..47, losses, the full-state profiles, and each plant's
full-state rel err at L beside the box's.
"""
import argparse
import importlib.util
import json
import os

import torch


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ap = argparse.ArgumentParser()
ap.add_argument("--instrument", required=True, help="bench/q235/instrument of a tree whose q235_compare reports update_rel_err")
ap.add_argument("--ref", required=True)
ap.add_argument("--box", required=True)
ap.add_argument("--plant-routing", required=True)
ap.add_argument("--plant-epilogue", required=True)
ap.add_argument("--tokens", required=True)
ap.add_argument("--layer", type=int, default=24)
ap.add_argument("--interior", default="4,44")
ap.add_argument("--out", required=True)
a = ap.parse_args()

cmpmod = _load(os.path.join(a.instrument, "q235_compare.py"), "q235_compare")
g1 = _load(os.path.join(a.instrument, "q235_gate1.py"), "q235_gate1")
mask = torch.load(a.tokens, weights_only=True)["attention_mask"]
L = a.layer
lo, hi = (int(x) for x in a.interior.split(","))

runs = {"box": a.box, "routing": a.plant_routing, "epilogue": a.plant_epilogue}
cmp = {k: cmpmod.compare(d, a.ref, mask) for k, d in runs.items()}
for k in ("routing", "epilogue"):
    rec = json.load(open(os.path.join(runs[k], "box.json")))
    assert rec["plant"] == {"kind": k, "layer": L}, (k, rec["plant"])
assert json.load(open(os.path.join(a.box, "box.json")))["plant"]["kind"] is None

u = {k: c["update_rel_err"] for k, c in cmp.items()}
box_int_max = max(u["box"][i] for i in range(lo, hi + 1))

line1 = {k: g1.plant_fails_as_registered(k, cmp[k]) for k in ("routing", "epilogue")}
line2a = {k: u[k][L] >= 2 * u["box"][L] for k in ("routing", "epilogue")}
line2b = {k: u[k][L] >= 2 * box_int_max for k in ("routing", "epilogue")}

out = {
    "line1_full_state": {
        "verdict": "HOLDS" if all(line1.values()) else "FAILS (full-state bands VOID for any deep box)",
        "plant_fails_as_registered": line1,
        "box_band_failures": g1.band_failures(cmp["box"]),
        "plant_band_failures": {k: g1.band_failures(cmp[k]) for k in ("routing", "epilogue")},
        "summary": {k: {"rel_err_median": c["rel_err_median"], "rel_err_max": c["rel_err_max"],
                        "router_overlap_min": c["router_overlap_min"]} for k, c in cmp.items()},
        "rel_err_at_L": {k: c["rel_err"][L] for k, c in cmp.items()},
    },
    "line2a_primary": {"verdict": "SEPARATES" if all(line2a.values()) else "DOES NOT SEPARATE",
                       "u_L": {k: u[k][L] for k in u}, "threshold_2x_box_u_L": 2 * u["box"][L], "per_plant": line2a},
    "line2b_reported": {"box_max_u_interior": box_int_max, "interior": [lo, hi], "threshold": 2 * box_int_max,
                        "per_plant": line2b,
                        "box_argmax_interior": max(range(lo, hi + 1), key=lambda i: u["box"][i])},
    "box_u_profile": u["box"],
    "plants_u_after_L": {k: u[k][L + 1:] for k in ("routing", "epilogue")},
    "full_state_profiles": {k: c["rel_err"] for k, c in cmp.items()},
    "router_overlap_at_L": {k: c["router_overlap"][L] for k, c in cmp.items()},
    "losses": {k: {"run": c["loss_a"], "ref": c["loss_b"], "gap": c["loss_gap"]} for k, c in cmp.items()},
    "real_tokens": cmp["box"]["real_tokens"], "layer": L,
}
json.dump(out, open(a.out, "w"), indent=1)
print(json.dumps({"line1": out["line1_full_state"]["verdict"], "line2a": out["line2a_primary"]["verdict"],
                  "u_L": out["line2a_primary"]["u_L"], "line2b": line2b, "box_max_u_interior": box_int_max,
                  "rel_err_at_L": out["line1_full_state"]["rel_err_at_L"]}))
