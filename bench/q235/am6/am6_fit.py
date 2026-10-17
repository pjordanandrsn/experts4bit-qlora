# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""Q235 Amendment 6's BAR, read mechanically from a calibration's box records by the registered rule: s(n) per depth, the log-log fit
over the 235B cuts extrapolated to 94 layers with the exponent's interval, the 30B at 48 layers against the fit, BAR = 3 x max(s94,
the 30B's s(48)), each plant in units of BAR, and the SOFT stops. p50 and p99 are fitted the same way and reported. It reads all three
forms of rule 8's records (Shape A against B; the same-shapes reference; the deterministic check).

CAL_DIR holds one ``q235_box.py --micro-batch 2`` output per depth: ``b235_2``, ``b235_4``, ``b235_8`` (cuts of the 235B) and
``q30_48`` (Qwen3-30B-A3B), each with its ``box.json``.

    python am6_fit.py CAL_DIR [--out FIT.json]
"""
import argparse
import json
import math
import os

T975_DF1 = 12.706                        # Student t, 97.5 %, 1 degree of freedom (3 points, 2 parameters)


def fit(pts):
    """least squares of log s on log n: (intercept, exponent, exponent SE, max relative residual)"""
    xs, ys = [math.log(n) for n, _ in pts], [math.log(s) for _, s in pts]
    m = len(xs)
    mx, my = sum(xs) / m, sum(ys) / m
    sxx = sum((x - mx) ** 2 for x in xs)
    b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
    a = my - b * mx
    ssr = sum((y - (a + b * x)) ** 2 for x, y in zip(xs, ys))
    se = math.sqrt(ssr / (m - 2) / sxx) if m > 2 else float("nan")
    worst = max(abs(math.exp(y - (a + b * x)) - 1) for x, y in zip(xs, ys))
    return a, b, se, worst


def at(a, b, n):
    return math.exp(a + b * math.log(n))


ap = argparse.ArgumentParser()
ap.add_argument("cal")
ap.add_argument("--out")
x = ap.parse_args()
depth = {}
for name, n in (("b235_2", 2), ("b235_4", 4), ("b235_8", 8), ("q30_48", 48)):
    p = os.path.join(x.cal, name, "box.json")
    if not os.path.exists(p):
        continue
    r = json.load(open(p))
    if "rule8" in r["step"]:                                            # Amendment 6's third form: the deterministic check passes
        q = "rule8.check"
        g = {k: r[k]["rule8"]["check"] for k in ("step", "step1")}
    else:
        q = "ga_mb_check" if "ga_mb_check" in r["step"] else "ga_check"  # the second form; the first form's records lack it
        g = {k: r[k][q] for k in ("step", "step1")}
    ab = {k: r[k]["ga_check"] for k in ("step", "step1")}
    depth[name] = {"layers": n, "rules_all_true": all(r["rules"].values()),
                   "s": max(v["max_rel"] for v in g.values()), "p50": max(v["rel_p50"] for v in g.values()),
                   "p99": max(v["rel_p99"] for v in g.values()),
                   "per_step": {k: {f: v[f] for f in ("max_rel", "max_rel_tensor", "rel_p50", "rel_p99", "cosine", "norm_ratio",
                                                      "compared", "zero_mismatch")} for k, v in g.items()},
                   "per_layer_max_rel": {k: v["per_layer_max_rel"] for k, v in g.items()},
                   "floor": {k: r[k]["rule8"]["floor"]["max_rel"] for k in ("step", "step1")} if q == "rule8.check" else None,
                   "modes": {k: (r[k]["rule8"]["algorithms"], r[k]["rule8"]["restored"], r[k]["algorithms"]) for k in ("step", "step1")}
                   if q == "rule8.check" else None,
                   "plants": {k: {f: (v.get("vs_micro_batches", v) if q != "ga_check" else v)[f] for f in ("max_rel", "cosine", "norm_ratio")}
                              for k, v in r["ga_plants"].items()},
                   "quantity": q, "in_process_spread": r["in_process"]["grad_maxrel"],
                   "reported_a_vs_b": {k: {f: v[f] for f in ("max_rel", "rel_p50", "cosine")} for k, v in ab.items()},
                   "route_flips": {f: r["route_flips"][f] for f in ("flipped", "token_layers", "flipped_fraction", "layers_with_flips")}
                   if "route_flips" in r else None,
                   "route_flips_per_layer": r.get("route_flips", {}).get("per_layer"),
                   "shape_b": r["shape_b"]}
b235 = [depth[k] for k in ("b235_2", "b235_4", "b235_8") if k in depth]
out = {"schema": "q235-am6-bar/1", "depth": depth}
if len(b235) == 3:
    fits = {}
    for q in ("s", "p99", "p50"):
        a, b, se, worst = fit([(d["layers"], d[q]) for d in b235])
        lo, hi = b - T975_DF1 * se, b + T975_DF1 * se
        f = {"exponent": b, "exponent_se": se, "exponent_95": [lo, hi], "max_rel_residual": worst, "at_94": at(a, b, 94),
             "at_94_95": [at(a, lo, 94) if lo == lo else None, at(a, hi, 94) if hi == hi else None], "at_48": at(a, b, 48)}
        if "q30_48" in depth:
            f["q30_48"] = depth["q30_48"][q]
            f["q30_48_over_fit"] = depth["q30_48"][q] / f["at_48"]
        fits[q] = f
    out["fit"] = fits
    s = fits["s"]
    s30 = depth["q30_48"]["s"] if "q30_48" in depth else None
    bar = 3 * max(s["at_94"], s30 or 0.0)
    soft = {"exponent > 0.5": s["exponent"] > 0.5, "a point > 30 % off the fit": s["max_rel_residual"] > 0.30,
            "the 30B > 3x off the fit": (s30 is not None and not (1 / 3 <= s["q30_48_over_fit"] <= 3))}
    plants = {k: {d_name: d["plants"][k]["max_rel"] / bar for d_name, d in depth.items()} for k in ("mean", "overwrite")}
    out.update({"bar": bar, "bar_rule": "3 x max(s94 from the 235B log-log fit, the 30B's s(48))", "soft": soft,
                "soft_any": any(soft.values()), "plants_in_bar_units": plants,
                "plants_registered": {k: min(v.values()) >= 3 for k, v in plants.items()},
                "p99_clean": not (fits["p99"]["exponent"] > 0.5 or fits["p99"]["max_rel_residual"] > 0.30 or
                                  ("q30_48" in depth and not (1 / 3 <= fits["p99"]["q30_48_over_fit"] <= 3)))})
print(json.dumps(out, indent=1))
if x.out:
    json.dump(out, open(x.out, "w"), indent=1)
