#!/usr/bin/env python3
"""bench/hp1/hp1_calib.py -- HP1-PREREG Amendment 4: the 48-layer zero-adapter tolerance, from a calibration that can FAIL.

Input: the one evaluator's JSON (hp1_eval.py --out) over the calibration arm: one e4b K00 run on 48 CALIBRATION rows (disjoint
from HP1's held-out 48), its base exported as the box builds it. The evaluator records, per base, the per-row zero-adapter gaps
against the arm's in-loop step-0 rows (`zero_adapter_row_gaps`).

Rules (registered in Amendment 4, before any box and before any scored y):
  - shape: the per-row gaps must look like numeric drift. If any row's |gap| > SHAPE_FACTOR x the median |gap|, the two paths
    differ (template, truncation or masking), not just in numerics: CLOSE_SHAPE.
  - tol = TOL_FACTOR x the mean |gap| over the calibration rows.
  - ceiling: tol above --ceiling: CLOSE_CEILING (the evaluator would sit too far from training at depth to read HP1's margin).
Prints one line `HP1_CALIB=<verdict> tol=<tol>` and writes --out. Exit 0 on PASS, 3 on a CLOSE, 2 on unusable input.
"""
import argparse
import json
import math
import statistics
import sys

SHAPE_FACTOR = 5.0
TOL_FACTOR = 2.0
N_ROWS = 48


def calibrate(ev, tag, ceiling, n_rows=N_ROWS):
    """The calibration verdict from an hp1_eval output dict. Raises ValueError on input that cannot be read as a calibration."""
    recs = [b for b in ev.get("bases", []) if tag in (b.get("zero_adapter_row_gaps") or {})]
    if len(recs) != 1:
        raise ValueError(f"expected exactly one base with per-row zero-adapter gaps for {tag!r}, found {len(recs)}")
    b = recs[0]
    if not b.get("digest_matches_training_run"):
        raise ValueError("the calibration base's digest does not match its training run")
    gaps = b["zero_adapter_row_gaps"][tag]
    if len(gaps) != n_rows or not all(isinstance(g, (int, float)) and math.isfinite(g) for g in gaps):
        raise ValueError(f"expected {n_rows} finite per-row gaps, got {len(gaps)}")
    a = [abs(g) for g in gaps]
    med, mean, mx = statistics.median(a), sum(a) / len(a), max(a)
    ratio = (mx / med) if med > 0 else (math.inf if mx > 0 else 0.0)
    tol = math.ceil(TOL_FACTOR * mean * 1e6) / 1e6          # rounded UP to 1e-6: never looser than the rule
    out = {"calibration": "bench/hp1/hp1_calib.py", "tag": tag, "base": b.get("dir"), "n_rows": len(gaps), "row_gaps": gaps,
           "median_abs_gap": round(med, 6), "mean_abs_gap": round(mean, 6), "max_abs_gap": round(mx, 6),
           "max_over_median": None if math.isinf(ratio) else round(ratio, 3), "signed_mean_gap": round(sum(gaps) / len(gaps), 6),
           "shape_factor": SHAPE_FACTOR, "tol_factor": TOL_FACTOR, "tol": tol, "ceiling": ceiling}
    if ratio > SHAPE_FACTOR:
        worst = max(range(len(a)), key=lambda i: a[i])
        out.update(verdict="CLOSE_SHAPE", reason=f"row {worst}: |gap| {a[worst]:.6f} > {SHAPE_FACTOR} x median {med:.6f}: a path "
                                                 "difference, not numeric drift")
    elif tol > ceiling:
        out.update(verdict="CLOSE_CEILING", reason=f"tol {tol} > ceiling {ceiling}")
    else:
        out.update(verdict="PASS", reason=f"tol {tol} = {TOL_FACTOR} x mean |gap| {mean:.6f}; max/median {ratio:.2f} <= {SHAPE_FACTOR}")
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("eval_json", help="hp1_eval.py output over the calibration arm")
    ap.add_argument("--tag", required=True, help="the calibration arm's tag")
    ap.add_argument("--ceiling", type=float, required=True, help="the registered tol ceiling (HP1-PREREG Amendment 4)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    try:
        res = calibrate(json.load(open(a.eval_json)), a.tag, a.ceiling)
    except (ValueError, KeyError, OSError, json.JSONDecodeError) as e:
        print(f"HP1_CALIB=UNUSABLE {type(e).__name__}: {e}")
        return 2
    json.dump(res, open(a.out, "w"), indent=1)
    print(f"HP1_CALIB={res['verdict']} tol={res['tol']} mean_abs={res['mean_abs_gap']} max_over_median={res['max_over_median']}")
    return 0 if res["verdict"] == "PASS" else 3


if __name__ == "__main__":
    sys.exit(main())
