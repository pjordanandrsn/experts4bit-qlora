"""D0 reduce (PREREG-d0.md "The labels", "Retention by loss", "Resolution"): one fixture's pass outputs -> summary JSON.

    python d0_reduce.py --pass-dir DIR --out summary.json [--heldout-ok true|false]

Step 1 per cell (T x S): rho_bar = sum<D,dW>/sum|dW|^2, c_bar = sum<D,dW>/sqrt(sum|D|^2 sum|dW|^2), e_bar, n_bar, f_bar,
per-expert rho quantiles (an expert's two projections pooled), rho_bar per layer and per projection, the base
conversion error, and rho_bar_route (per-expert rho weighted by REF's held-out routing counts).
Step 2: L per arm = total NLL / total scored tokens; the improvement L(Q) - L(REF) with its bootstrap SE (2,000
resamples over rows, seed 0, numerator and denominator from the same draw); R per cell only when RESOLVED (> 3 SE);
labels with the CI-crossing note; when unresolved, labels from rho_bar marked "(rho_bar; loss unresolved)".
"""
import argparse
import csv
import glob
import json
import math
import os
from collections import defaultdict

CELLS = [(t, s) for t in ("NF4", "INT4", "MXFP4") for s in ("Q", "W")]
BOUNDS = (0.5, 0.95)


def read(pattern):
    rows = []
    for p in sorted(glob.glob(pattern)):
        rows += list(csv.DictReader(open(p)))
    return rows


def label(x):
    return "RETAINED" if x >= BOUNDS[1] else ("PARTIAL" if x >= BOUNDS[0] else "LOST")


def quantile(xs, q):
    xs = sorted(xs)
    if not xs:
        return None
    i = (len(xs) - 1) * q
    lo, hi = math.floor(i), math.ceil(i)
    return xs[lo] + (xs[hi] - xs[lo]) * (i - lo)


def step1(d):
    cells = read(os.path.join(d, "cells_L*.csv"))
    routing = {(int(r["layer"]), int(r["expert"])): int(r["routed"]) for r in read(os.path.join(d, "routing_L*.csv"))}
    out = {}
    for t, s in CELLS:
        rows = [r for r in cells if r["T"] == t and r["S"] == s]
        if not rows:
            continue
        S = lambda k, rs=rows: sum(float(r[k]) for r in rs)  # noqa: E731
        per_exp = defaultdict(lambda: [0.0, 0.0])
        per_layer, per_proj = defaultdict(lambda: [0.0, 0.0]), defaultdict(lambda: [0.0, 0.0])
        for r in rows:
            key = (int(r["layer"]), int(r["expert"]))
            for acc in (per_exp[key], per_layer[int(r["layer"])], per_proj[r["which"]]):
                acc[0] += float(r["dot"])
                acc[1] += float(r["dd"])
        rho_e = {k: v[0] / v[1] for k, v in per_exp.items() if v[1] > 0}
        wsum = sum(routing.get(k, 0) for k in rho_e)
        out[f"{t}-{s}"] = {
            "rho_bar": S("dot") / S("dd"),
            "c_bar": S("dot") / math.sqrt(S("DD") * S("dd")) if S("DD") > 0 else 0.0,
            "e_bar": math.sqrt(S("ee") / S("dd")), "n_bar": math.sqrt(S("DD") / S("dd")),
            "f_bar": S("changed") / S("numel"),
            "rho_expert_p10_median_p90": [quantile(list(rho_e.values()), q) for q in (0.1, 0.5, 0.9)],
            "rho_bar_layer": {k: v[0] / v[1] for k, v in sorted(per_layer.items())},
            "rho_bar_proj": {k: v[0] / v[1] for k, v in per_proj.items()},
            "base_rel_err": math.sqrt(S("bb") / S("qq")),
            "rho_bar_route": (sum(routing.get(k, 0) * v for k, v in rho_e.items()) / wsum) if wsum else None,
            "n_expert_projections": len(rows)}
    checks = read(os.path.join(d, "checks_L*.csv"))
    out["_checks"] = {"nf4_q_exact_share": sum(r["nf4_q_exact"] == "True" for r in checks) / max(1, len(checks)),
                      "cpu_rebuild_code_diff_share": sum(int(r["cpu_codes_diff"]) for r in checks)
                      / max(1, sum(int(r["codes"]) for r in checks)),
                      "dW_rel_norm": math.sqrt(sum(float(r["dW_norm2"]) for r in checks)
                                               / max(1e-30, sum(float(r["Q_norm2"]) for r in checks)))}
    return out


def step2(d, s1):
    p = os.path.join(d, "arms.csv")
    if not os.path.exists(p):
        return None
    rows = list(csv.DictReader(open(p)))
    arms = sorted({r["arm"] for r in rows})
    nrow = 1 + max(int(r["row"]) for r in rows)
    nll = {a: [0.0] * nrow for a in arms}
    cnt = {a: [0] * nrow for a in arms}
    dmax, dmean = defaultdict(float), defaultdict(float)
    for r in rows:
        i = int(r["row"])
        nll[r["arm"]][i], cnt[r["arm"]][i] = float(r["nll"]), int(r["n"])
        dmax[r["arm"]] = max(dmax[r["arm"]], float(r["dmax"]))
        dmean[r["arm"]] += float(r["dmean"]) * int(r["n"])

    def L(a, idx):
        return sum(nll[a][i] for i in idx) / sum(cnt[a][i] for i in idx)

    full = list(range(nrow))
    import random
    rng = random.Random(0)
    draws = [[rng.randrange(nrow) for _ in range(nrow)] for _ in range(2000)]
    imp = L("Q", full) - L("REF", full)
    imp_bs = [L("Q", ix) - L("REF", ix) for ix in draws]
    mu = sum(imp_bs) / len(imp_bs)
    se = math.sqrt(sum((x - mu) ** 2 for x in imp_bs) / (len(imp_bs) - 1))
    resolved = imp > 3 * se
    out = {"L": {a: L(a, full) for a in arms}, "improvement": imp, "improvement_se": se, "resolved": resolved,
           "dlogit_vs_ref": {a: {"max": dmax[a], "mean": dmean[a] / sum(cnt[a])} for a in arms}, "cells": {}}
    for t, s in CELLS:
        base, merged = f"{t}-{s}-base", f"{t}-{s}-merged"
        c = {"gap_to_ref": L(merged, full) - L("REF", full)}
        if resolved:
            R = (L(base, full) - L(merged, full)) / imp
            Rs = sorted((L(base, ix) - L(merged, ix)) / (L("Q", ix) - L("REF", ix)) for ix in draws
                        if L("Q", ix) - L("REF", ix) != 0)
            lo, hi = quantile(Rs, 0.05), quantile(Rs, 0.95)
            lab = label(R)
            crossed = [b for b in BOUNDS if lo < b <= hi]
            c.update(R=R, R_ci90=[lo, hi], label=lab + (f" (CI crosses {crossed[0]})" if crossed else ""))
        else:
            rb = s1[f"{t}-{s}"]["rho_bar"]
            c.update(R=None, label=f"{label(rb)} (rho_bar; loss unresolved)")
        out["cells"][f"{t}-{s}"] = c
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pass-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--heldout-ok", default="true", help="false when the unseen proof failed: Step 2 VOID")
    a = ap.parse_args()
    s1 = step1(a.pass_dir)
    s2 = step2(a.pass_dir, s1) if a.heldout_ok == "true" else None
    if s2 is None:
        labels = {k: f"{label(v['rho_bar'])} (rho_bar; loss unresolved)" for k, v in s1.items() if not k.startswith("_")}
    else:
        labels = {k: v["label"] for k, v in s2["cells"].items()}
    json.dump({"step1": s1, "step2": s2, "labels": labels}, open(a.out, "w"), indent=1)
    for k, v in labels.items():
        st = s1[k]
        print(f"{k:9s} {v:40s} rho_bar {st['rho_bar']:.3f}  c_bar {st['c_bar']:.3f}  f {st['f_bar']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
