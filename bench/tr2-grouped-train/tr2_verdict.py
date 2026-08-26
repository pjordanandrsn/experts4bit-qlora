# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""Verdict calculator for PREREG-tr2-grouped-train.

Report shape:
  base_a/base_b/hyb_a/hyb_b: census receipts (the amended TR1
    composer consumes each pair; this calculator re-uses it).
  lc: {"base_launches_per_step": int, "hyb_launches_per_step": int}
  evals: {"base_before": f, "base_after": f,
          "hyb_before": f, "hyb_after": f}

Bars verbatim from the prereg: PASS = step cut >= 2x AND launch cut
>= 10x AND quality; PARTIAL >= 1.25x; REFUTED < 1.25x. Baseline
sanity: base median within +/-10% of the TR1 anchor 51.68 s.
Prediction check recorded either direction. Quality: finite losses
(composer), hyb final <= base final + 0.05, hyb improvement >= 80%
of base improvement.
"""

import argparse
import importlib.util
import json
import math
import sys
from pathlib import Path

TR1_ANCHOR_MS = 51680.0
ANCHOR_TOL = 0.10
PASS_SPEEDUP = 2.0
PARTIAL_SPEEDUP = 1.25
# model-check thresholds (NOT pass gates -- bars follow the claim):
# additive model: 2x wall needs >= ~2.3x launch cut; ~8x launches
# should yield >= 2x wall
MODEL_MIN_LAUNCH_FOR_2X = 2.0
MODEL_LAUNCH_IMPLIES_2X = 8.0
EVAL_DELTA_MAX = 0.05
IMPROVE_FRAC = 0.80
EPS = 1e-9

_spec = importlib.util.spec_from_file_location(
    "tr1_compose", Path(__file__).resolve().parent.parent /
    "tr1-census" / "tr1_compose.py")
_tr1 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_tr1)


def verdict(rep):
    for k in ("base_a", "base_b", "hyb_a", "hyb_b", "lc", "evals"):
        if not rep.get(k):
            return ("REFUSE", f"missing {k!r}")
    base, why = _tr1.compose(rep["base_a"], rep["base_b"])
    if why:
        return ("REFUSE", f"base census: {why}")
    # the hybrid pair's same-workload gate carries an ABSOLUTE floor
    # measured from the base pair on the same box (amendment with the
    # TR2 receipts: per-step jitter is ~250 ms absolute regardless of
    # step duration, so a purely relative gate refuses fast-step runs
    # whose absolute agreement beats the baseline's own)
    floor = 1.5 * (base.get("aa_step_delta_ms") or 0.0)
    hyb, why = _tr1.compose(rep["hyb_a"], rep["hyb_b"],
                            aa_abs_floor_ms=floor)
    if why:
        return ("REFUSE", f"hybrid census: {why}")

    b_ms, h_ms = base["anchor_step_ms"], hyb["anchor_step_ms"]
    lo, hi = TR1_ANCHOR_MS * (1 - ANCHOR_TOL), TR1_ANCHOR_MS * (1 + ANCHOR_TOL)
    if not (lo <= b_ms <= hi):
        return ("REFUSE", f"baseline sanity: {b_ms:.0f} ms outside "
                f"[{lo:.0f}, {hi:.0f}] -- not the registered workload")

    lc = rep["lc"]
    lb, lh = lc["base_launches_per_step"], lc["hyb_launches_per_step"]
    if lb <= 0 or lh <= 0:
        return ("REFUSE", "launch-count probe missing or zero")
    launch_cut = lb / lh

    ev = rep["evals"]
    for k in ("base_before", "base_after", "hyb_before", "hyb_after"):
        val = ev.get(k)
        if val is None or not math.isfinite(val):
            return ("REFUSE", f"evals[{k}] = {val!r} -- non-finite "
                    "eval; a diverged run cannot certify (the TR1 "
                    "composer's finite-loss rule, applied to evals)")
    b_impr = ev["base_before"] - ev["base_after"]
    h_impr = ev["hyb_before"] - ev["hyb_after"]
    if b_impr <= 0:
        return ("REFUSE", "base arm did not learn -- quality frame "
                "undefined")
    q = []
    if ev["hyb_after"] > ev["base_after"] + EVAL_DELTA_MAX + EPS:
        return ("REFUSE", f"quality: hyb final {ev['hyb_after']:.4f} > "
                f"base {ev['base_after']:.4f} + {EVAL_DELTA_MAX}")
    if h_impr < IMPROVE_FRAC * b_impr - EPS:
        return ("REFUSE", f"quality: hyb improvement {h_impr:.4f} < "
                f"{IMPROVE_FRAC:.0%} of base's {b_impr:.4f}")
    q.append(f"evals base {ev['base_after']:.3f} / hyb "
             f"{ev['hyb_after']:.3f}")

    speed = b_ms / h_ms
    pred = ""
    if speed >= PASS_SPEEDUP - EPS and launch_cut < MODEL_MIN_LAUNCH_FOR_2X:
        pred = (" MODEL-FALSIFIED: >=2x wall with <2x launch cut -- "
                "the launch-storm account missed something; record")
    if launch_cut >= MODEL_LAUNCH_IMPLIES_2X and speed < PASS_SPEEDUP:
        pred = (" MODEL-FALSIFIED: >=8x launch cut without >=2x wall "
                "-- the wall was not launches; record")
    # implied per-launch host cost, the model residual each arm reports
    cost_b = (b_ms - 5650.0) / lb * 1000.0
    cost_h = (h_ms - 5650.0) / lh * 1000.0
    detail = (f"step {h_ms/1000:.2f} vs {b_ms/1000:.2f} s = "
              f"{speed:.2f}x; launches {lb:,} -> {lh:,} "
              f"({launch_cut:.1f}x; implied us/launch "
              f"{cost_b:.1f} -> {cost_h:.1f}); {'; '.join(q)}.{pred}")
    if speed >= PASS_SPEEDUP - EPS:
        return ("PASS", detail + " -- TRAIN_ARENA becomes the "
                "documented default path for arena-holding models")
    if speed >= PARTIAL_SPEEDUP - EPS:
        return ("PARTIAL", detail + " -- ships opt-in with disclosure")
    return ("REFUTED", detail)


# ---------------------------------------------------------------- self-test
def _census_jitter(step_ms, jit_ms):
    """Pair-of-runs builder where run B's walls differ from run A's by
    +-jit_ms alternating -- absolute jitter, balanced half-medians."""
    a = _census(step_ms)
    b = _census(step_ms)
    for i, st in enumerate(b["steps"]):
        st["step_wall_ms"] += jit_ms if i % 2 else -jit_ms
        for ph in ("data", "forward", "backward", "loss_sync", "optim"):
            st[ph] += (jit_ms if i % 2 else -jit_ms) / 5.0
    return a, b


def _census(step_ms=51680.0):
    # phases scale with the wall so the TR1 closure gate holds on the
    # synthetic (0.2/0.4/0.3/0.05/0.04 of wall = 99% accounted)
    f = step_ms / 1000.0
    return _tr1._run(n=15, wall=step_ms,
                     phases=tuple(x * f for x in
                                  (200.0, 400.0, 300.0, 50.0, 40.0)))


def _mk(base=51680.0, hyb=20000.0, lb=2_900_000, lh=120_000,
        evals=None):
    return {"base_a": _census(base), "base_b": _census(base),
            "hyb_a": _census(hyb), "hyb_b": _census(hyb),
            "lc": {"base_launches_per_step": lb,
                   "hyb_launches_per_step": lh},
            "evals": evals or {"base_before": 2.27, "base_after": 1.62,
                               "hyb_before": 2.27, "hyb_after": 1.60}}


def _self_test():
    v = verdict
    out = v(_mk())
    assert out[0] == "PASS", out
    # fast-step hybrid pair with ~250 ms ABSOLUTE jitter (6%+ relative)
    # passes when the base pair carries the same absolute noise -- the
    # TR2 receipts case; without base noise (floor 0) it still refuses
    r = _mk(hyb=4000.0, lh=126_000)
    ba, bb = _census_jitter(51680.0, 250.0)
    ha, hb = _census_jitter(4000.0, 250.0)
    r["base_a"], r["base_b"] = ba, bb
    r["hyb_a"], r["hyb_b"] = ha, hb
    out = v(r)
    assert out[0] == "PASS", out
    r2 = _mk(hyb=4000.0, lh=126_000)
    ha2, hb2 = _census_jitter(4000.0, 250.0)
    r2["hyb_a"], r2["hyb_b"] = ha2, hb2   # base pair identical -> floor 0
    out = v(r2)
    assert out[0] == "REFUSE" and "per-step" in out[1], out
    # exactly 2x wall passes regardless of launch multiple (bars
    # follow the claim); a 5x launch cut is model-CONSISTENT
    out = v(_mk(hyb=25840.0, lh=600_000))
    assert out[0] == "PASS" and "FALSIFIED" not in out[1], out
    # 2x wall with launch cut only 1.5x: passes the BAR, records the
    # model falsification
    out = v(_mk(hyb=25000.0, lh=1_950_000))
    assert out[0] == "PASS" and "MODEL-FALSIFIED" in out[1], out
    # 8x+ launches but 1.1x wall: falsified the other way, REFUTED
    out = v(_mk(hyb=47000.0, lh=290_000))
    assert out[0] == "REFUTED" and "MODEL-FALSIFIED" in out[1], out
    # PARTIAL band, model-consistent (1.33x wall from a 1.4x launch
    # cut: predicted 5.65 + 2.07M x 15.9us = 38.6 s ~ 39 s)
    out = v(_mk(hyb=39000.0, lh=2_100_000))
    assert out[0] == "PARTIAL" and "FALSIFIED" not in out[1], out
    # REFUTED
    assert v(_mk(hyb=48000.0, lh=200_000))[0] == "REFUTED"
    # baseline sanity two-sided
    for bad in (44000.0, 60000.0):
        out = v(_mk(base=bad))
        assert out[0] == "REFUSE" and "sanity" in out[1], out
    # quality: hyb final too high refuses
    out = v(_mk(evals={"base_before": 2.27, "base_after": 1.62,
                       "hyb_before": 2.27, "hyb_after": 1.70}))
    assert out[0] == "REFUSE" and "quality" in out[1], out
    # quality: hyb under-learns refuses (final inside the +0.05 delta
    # so the improvement gate is what trips: 0.35 < 80% of 0.65)
    out = v(_mk(evals={"base_before": 2.27, "base_after": 1.62,
                       "hyb_before": 2.00, "hyb_after": 1.65}))
    assert out[0] == "REFUSE" and "improvement" in out[1], out
    # non-finite evals refuse regardless of direction (Bugbot, #258:
    # NaN comparisons are False, so they slipped every gate)
    for bad in (float("nan"), float("-inf"), float("inf")):
        out = v(_mk(evals={"base_before": 2.27, "base_after": 1.62,
                           "hyb_before": 2.27, "hyb_after": bad}))
        assert out[0] == "REFUSE" and "non-finite" in out[1], (bad, out)
    out = v(_mk(evals={"base_before": float("nan"), "base_after": 1.62,
                       "hyb_before": 2.27, "hyb_after": 1.60}))
    assert out[0] == "REFUSE" and "non-finite" in out[1], out
    # base didn't learn: frame undefined
    out = v(_mk(evals={"base_before": 1.62, "base_after": 1.62,
                       "hyb_before": 2.27, "hyb_after": 1.60}))
    assert out[0] == "REFUSE" and "did not learn" in out[1], out
    # census refusal propagates
    r = _mk()
    r["hyb_b"]["steps"][7]["loss"] = float("nan")
    out = v(r)
    assert out[0] == "REFUSE" and "hybrid census" in out[1], out
    # missing launch probe refuses
    r = _mk()
    r["lc"]["hyb_launches_per_step"] = 0
    assert v(r)[0] == "REFUSE"
    print("tr2_verdict self-test: OK")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("report", nargs="?")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()
    if a.self_test:
        _self_test()
        return
    rep = json.load(open(a.report))
    v, why = verdict(rep)
    print(f"TR2 VERDICT: {v}\n{why}")
    if v == "REFUSE":
        sys.exit(2)


if __name__ == "__main__":
    main()
