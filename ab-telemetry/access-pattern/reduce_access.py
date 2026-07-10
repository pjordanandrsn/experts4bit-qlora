#!/usr/bin/env python3
"""Reduce access_pattern.jsonl: measured read_fraction vs the occupancy null, and the eff_tokens
where MEASURED read_fraction crosses 0.5 and 0.9 — the vertical slices Session 4 pays to measure."""

import json
import math
import sys
from collections import defaultdict

E, K = 128, 8
Q = 1 - K / E


def null_rf(n):
    return 1 - Q**n


def cross(points, target):
    """Linear-interpolate the eff_tokens where the measured curve crosses `target`."""
    pts = sorted(points)
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        if y0 <= target <= y1 and y1 != y0:
            return round(x0 + (target - y0) * (x1 - x0) / (y1 - y0), 1)
    return None


def main(path):
    by_cell = defaultdict(list)
    meta = {}
    for line in open(path):
        r = json.loads(line)
        by_cell[(r["batch"], r["seq"], r["eff_tokens"])].append(r["read_fraction"])
    curve = []  # (eff_tokens, mean_rf) for the batch=1 sweep
    print(f"{'cell':<14}{'eff_tokens':<12}{'measured':<11}{'null':<9}{'gap'}")
    rows = []
    for (b, s, eff), fr in sorted(by_cell.items(), key=lambda kv: (kv[0][2], kv[0][0])):
        m = sum(fr) / len(fr)
        n = null_rf(eff)
        print(f"b{b}xs{s:<11} {eff:<12}{m:<11.4f}{n:<9.4f}{m-n:+.4f}")
        rows.append({"batch": b, "seq": s, "eff_tokens": eff, "measured_read_fraction": round(m, 4),
                     "null_read_fraction": round(n, 4), "gap_vs_null": round(m - n, 4)})
        if b == 1:
            curve.append((eff, m))
    c50, c90 = cross(curve, 0.5), cross(curve, 0.9)
    out = {
        "model": "Qwen3-30B-A3B", "E": E, "k": K,
        "measured_crossings_eff_tokens": {"0.5": c50, "0.9": c90},
        "null_crossings_eff_tokens": {"0.5": round(math.log(0.5) / math.log(Q), 1),
                                      "0.9": round(math.log(0.1) / math.log(Q), 1)},
        "verdict": None, "cells": rows,
    }
    # Is real routing meaningfully below the null (wide window) or ~= null (narrow / decode-scale)?
    if c90 is not None:
        widening = c90 / (math.log(0.1) / math.log(Q))
        out["verdict"] = (
            f"real 0.9-crossing at {c90} eff_tokens = {widening:.1f}x the null's {out['null_crossings_eff_tokens']['0.9']}; "
            + ("token correlation WIDENS the sparse-read window -> Session 4 slices at the measured crossings"
               if widening > 1.5 else
               "real ~= null: window is decode-scale; SSD-tier useful only at small-batch serving -> D2 (wider-expert model) becomes load-bearing")
        )
    json.dump(out, open("access_reduction.json", "w"), indent=2)
    print("\n" + json.dumps({k: out[k] for k in ("measured_crossings_eff_tokens", "null_crossings_eff_tokens", "verdict")}, indent=2))
    print("wrote access_reduction.json")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "access_pattern.jsonl")
