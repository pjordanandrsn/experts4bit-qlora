# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""PREREG-t5b Phase A verdict: G0 A/A, GS shape-gate (the T5 lesson --
abort on operating-point drift), H-A attribution (coverage + largest
region). Bars hardcoded from the prereg; --self-test runs both
directions before any receipt touches it."""

import argparse
import json
import sys
from pathlib import Path

AA_SPREAD_MAX = 7.5
GS_STEP_LO, GS_STEP_HI = 115.0, 165.0
GS_ATTN_MAX = 55.0
GS_DRAM_MAX = 25.0
HA_COVERAGE_MIN = 80.0        # % of step the brackets must jointly own
HA_TOP_REGION_MIN = 25.0      # % of step the Phase B target must own


def _med(rep):
    return rep["decode_median_ms"]


def verdict(aa1, aa2, attr, region_ops):
    out = {}
    m1, m2 = float(_med(aa1)["step"]), float(_med(aa2)["step"])
    spread = abs(m1 - m2) / min(m1, m2) * 100.0
    out["g0"] = {"aa1_ms": m1, "aa2_ms": m2, "spread_pct": spread,
                 "pass": spread < AA_SPREAD_MAX}
    if not out["g0"]["pass"]:
        out["verdict"] = "NO-VERDICT (G0 fail: box not measurement-grade)"
        return out

    base = _med(aa1) if m1 <= m2 else _med(aa2)
    gs = {"step_ms": float(base["step"]),
          "attn_ms": float(base["attention_host"]),
          "dram_ms": float(base["dram_experts_host"])}
    gs["pass"] = (GS_STEP_LO <= gs["step_ms"] <= GS_STEP_HI
                  and gs["attn_ms"] <= GS_ATTN_MAX
                  and gs["dram_ms"] <= GS_DRAM_MAX)
    out["gs"] = gs
    if not gs["pass"]:
        out["verdict"] = ("ABORT (GS shape-gate: baseline is not the "
                          "certified operating point -- diagnose config "
                          "drift before any attribution is read)")
        return out

    d = _med(attr)
    step = float(d["step"])
    regions = {
        "attn": float(d["attention_host"]),
        "moe": float(d["moe_host"]),
        "lmhead": float(d["lmhead_host"]),
        "sched": float(d["scheduler_python_and_bookkeeping"]),
        "drain": float(d["drain"]),
    }
    # AMENDMENT-t5b-h-a: when the attribution ran with the block region
    # (b1-instruments), router/top-k = block - experts is its own named
    # region instead of landing in the residual. Same bars.
    if "moe_block_host" in d:
        regions["router_topk"] = (float(d["moe_block_host"])
                                  - regions["moe"])
    shares = {k: v / step * 100.0 for k, v in regions.items()}
    coverage = sum(shares.values())
    top = max(shares, key=shares.get)
    out["h_a"] = {"step_ms": step, "regions_ms": regions,
                  "shares_pct": {k: round(v, 1) for k, v in shares.items()},
                  "coverage_pct": coverage,
                  "residual_pct": 100.0 - coverage,
                  "top_region": top, "top_share_pct": shares[top],
                  "region_calls": region_ops.get("region_calls", {}),
                  "pass": (coverage >= HA_COVERAGE_MIN
                           and shares[top] >= HA_TOP_REGION_MIN)}
    if out["h_a"]["pass"]:
        bar = max(10.0, shares[top] / 2.0)
        out["verdict"] = (f"PHASE-A-PASS (target={top} at "
                          f"{shares[top]:.1f}% of step; Phase B wall bar "
                          f"= {bar:.1f}%, partial floor 6%)")
        out["phase_b_bar_pct"] = bar
    else:
        out["verdict"] = ("PHASE-A-STOP (bill is diffuse: coverage "
                          f"{coverage:.1f}%, top {top} "
                          f"{shares[top]:.1f}% -- ladder re-points at T4 "
                          "overlap; record and stop)")
    return out


def _rep(step, attn, dram, moe=None, lmh=None, sched=0.4, drain=0.4):
    d = {"step": step, "attention_host": attn, "dram_experts_host": dram,
         "scheduler_python_and_bookkeeping": sched, "drain": drain}
    if moe is not None:
        d["moe_host"] = moe
        d["lmhead_host"] = lmh
        d["host_residual"] = step - attn - moe - lmh
    return {"decode_median_ms": d}


def self_test():
    ro = {"region_calls": {"e4b::moe": 576, "e4b::attn": 576,
                           "e4b::lmhead": 12}}
    # pass: concentrated bill -> names the target and derives the bar
    v = verdict(_rep(131.0, 40.0, 14.0), _rep(132.0, 41.0, 14.0),
                _rep(135.0, 41.0, 14.0, moe=62.0, lmh=8.0), ro)
    assert v["verdict"].startswith("PHASE-A-PASS (target=moe"), v
    assert abs(v["phase_b_bar_pct"] - 62.0 / 135.0 * 100 / 2) < 0.1, v
    # stop: diffuse bill (nothing >= 25%)
    v = verdict(_rep(131.0, 40.0, 14.0), _rep(132.0, 41.0, 14.0),
                _rep(135.0, 30.0, 14.0, moe=30.0, lmh=28.0), ro)
    assert v["verdict"].startswith("PHASE-A-STOP"), v
    # stop: poor coverage even with a big region
    v = verdict(_rep(131.0, 40.0, 14.0), _rep(132.0, 41.0, 14.0),
                _rep(200.0, 40.0, 14.0, moe=55.0, lmh=5.0), ro)
    assert v["verdict"].startswith("PHASE-A-STOP"), v
    # abort: shape gate (per-seq shape sneaks in -- the T5 accident)
    v = verdict(_rep(215.8, 124.6, 14.0), _rep(216.0, 125.0, 14.0),
                _rep(216.0, 125.0, 14.0, moe=60.0, lmh=8.0), ro)
    assert v["verdict"].startswith("ABORT (GS"), v
    # no-verdict: A/A spread
    v = verdict(_rep(131.0, 40.0, 14.0), _rep(150.0, 41.0, 14.0),
                _rep(135.0, 41.0, 14.0, moe=62.0, lmh=8.0), ro)
    assert v["verdict"].startswith("NO-VERDICT (G0"), v
    # bar formula on a just-over-threshold target: moe 35.1/135 = 26%,
    # attn+lmhead fill coverage without out-topping it; bar = 26/2 = 13
    v = verdict(_rep(131.0, 40.0, 14.0), _rep(132.0, 41.0, 14.0),
                _rep(135.0, 34.0, 14.0, moe=35.1, lmh=34.0,
                     sched=4.0, drain=2.0), ro)
    assert v["verdict"].startswith("PHASE-A-PASS (target=moe"), v
    assert abs(v["phase_b_bar_pct"] - 35.1 / 135.0 * 100 / 2) < 0.1, v
    # AMENDED: block region present -- router_topk named, coverage that
    # missed without it now passes with the SAME bars
    rep_blk = _rep(165.4, 54.4, 14.0, moe=75.7, lmh=0.06,
                   sched=0.7, drain=0.4)
    rep_blk["decode_median_ms"]["moe_block_host"] = 104.0   # router 28.3
    v = verdict(_rep(141.0, 46.0, 14.0), _rep(140.7, 46.0, 14.0),
                rep_blk, ro)
    assert v["h_a"]["shares_pct"]["router_topk"] > 0, v
    assert v["verdict"].startswith("PHASE-A-PASS (target=moe"), v
    # and WITHOUT the block region the same numbers still stop (guards
    # that the amendment changed the instrument, not the bar)
    v = verdict(_rep(141.0, 46.0, 14.0), _rep(140.7, 46.0, 14.0),
                _rep(165.4, 54.4, 14.0, moe=75.7, lmh=0.06,
                     sched=0.7, drain=0.4), ro)
    assert v["verdict"].startswith("PHASE-A-STOP"), v
    print("self-test OK: 8/8 branches exercised")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--aa1")
    ap.add_argument("--aa2")
    ap.add_argument("--attr")
    ap.add_argument("--region-ops")
    a = ap.parse_args()
    if a.self_test:
        self_test()
        sys.exit(0)
    load = lambda f: json.loads(Path(f).read_text())
    print(json.dumps(verdict(load(a.aa1), load(a.aa2), load(a.attr),
                             load(a.region_ops)), indent=2))
