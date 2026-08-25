# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""PREREG-tr1-census composer: turns two census runs (A/A) into the
registered budget deliverable, enforcing the census's refusal gates.

No speed bars here by design -- TR1 registers instruments only. Gates
verbatim from the prereg: phase closure within 5% of step wall,
steady-state spread < 10% of median, A/A phase shares within 3 points
absolute, N warmup steps dropped (recorded).
"""

import argparse
import json
import statistics
import sys

CLOSURE_FRAC = 0.05
STEADY_FRAC = 0.10
AA_POINTS = 3.0
WARMUP_DROP = 3

PHASES = ("data", "forward", "backward", "loss_sync", "optim")


def _shares(run):
    steps = run["steps"][WARMUP_DROP:]
    if len(steps) < 5:
        return None, f"only {len(steps)} steps after warmup drop"
    for i, s in enumerate(steps):
        ls = s.get("loss")
        if ls is None or ls != ls or ls == float("inf"):
            return None, (f"step {WARMUP_DROP + i}: loss {ls!r} -- a "
                          "broken run's budget is not a budget")
    walls = [s["step_wall_ms"] for s in steps]
    med = statistics.median(walls)
    spread = (max(walls) - min(walls))
    if spread >= STEADY_FRAC * med:
        return None, (f"not steady: wall spread {spread:.1f} ms >= "
                      f"{STEADY_FRAC:.0%} of median {med:.1f}")
    tot = {p: sum(s.get(p, 0.0) for s in steps) for p in PHASES}
    phase_sum = sum(tot.values())
    wall_sum = sum(walls)
    gap = abs(wall_sum - phase_sum) / wall_sum
    if gap > CLOSURE_FRAC:
        return None, (f"phase closure fails: |wall - phases| = "
                      f"{gap:.1%} of wall (> {CLOSURE_FRAC:.0%}) -- "
                      "unaccounted time is a finding, not rounding")
    shares = {p: 100.0 * tot[p] / phase_sum for p in PHASES}
    return {"median_step_ms": med, "shares_pct": shares,
            "steps_used": len(steps), "closure_gap": gap}, None


def compose(run_a, run_b):
    a, why = _shares(run_a)
    if why:
        return None, f"run A: {why}"
    b, why = _shares(run_b)
    if why:
        return None, f"run B: {why}"
    for p in PHASES:
        d = abs(a["shares_pct"][p] - b["shares_pct"][p])
        if d > AA_POINTS:
            return None, (f"A/A drift on {p}: {d:.1f} points "
                          f"(> {AA_POINTS}) -- shares not stable")
    return {
        "anchor_step_ms": min(a["median_step_ms"], b["median_step_ms"]),
        "shares_pct": {p: (a["shares_pct"][p] + b["shares_pct"][p]) / 2
                       for p in PHASES},
        "a": a, "b": b,
        "meta": run_a.get("meta", {}),
        "warmup_dropped": WARMUP_DROP,
    }, None


# ---------------------------------------------------------------- self-test
def _run(n=12, wall=1000.0, phases=(200.0, 400.0, 300.0, 50.0, 40.0),
         jitter=0.0, loss=2.0):
    steps = []
    for i in range(n):
        j = jitter if i % 2 else -jitter
        row = dict(zip(PHASES, [x + j / len(PHASES) for x in phases]))
        row["step_wall_ms"] = wall + j
        row["loss"] = loss
        steps.append(row)
    return {"meta": {"model": "m"}, "steps": steps}


def _self_test():
    ok, why = compose(_run(), _run())
    assert why is None, why
    assert abs(sum(ok["shares_pct"].values()) - 100.0) < 1e-6
    assert ok["anchor_step_ms"] == 1000.0
    # closure: phases sum 990 of 1000 wall = 1.0% gap -> passes; open a
    # 6% hole and it must refuse
    bad = _run(phases=(200.0, 400.0, 250.0, 50.0, 40.0))  # sum 940
    _, why = compose(bad, bad)
    assert why and "closure" in why, why
    # steady-state: 12% wall swing refuses
    _, why = compose(_run(jitter=120.0), _run())
    assert why and "not steady" in why, why
    # A/A drift: forward share moved > 3 points between runs
    drift = _run(phases=(200.0, 460.0, 240.0, 50.0, 40.0))
    _, why = compose(_run(), drift)
    assert why and "A/A drift" in why, why
    # too few steps
    _, why = compose(_run(n=6), _run())
    assert why and "steps" in why, why
    # NaN loss inside the timed window refuses; a poisoned WARMUP loss
    # does not (warmup is dropped)
    nan_run = _run()
    nan_run["steps"][7]["loss"] = float("nan")
    _, why = compose(nan_run, _run())
    assert why and "loss" in why, why
    warm_nan = _run()
    warm_nan["steps"][1]["loss"] = float("nan")
    ok, why = compose(warm_nan, _run())
    assert why is None, why
    # missing loss key refuses (an old-format receipt cannot certify)
    old_fmt = _run()
    del old_fmt["steps"][8]["loss"]
    _, why = compose(old_fmt, _run())
    assert why and "loss" in why, why
    # warmup rows are genuinely dropped: poison the first 3 steps of an
    # otherwise-good run; compose must still pass
    poisoned = _run()
    for i in range(WARMUP_DROP):
        poisoned["steps"][i]["step_wall_ms"] = 9999.0
    ok, why = compose(poisoned, _run())
    assert why is None, why
    print("tr1_compose self-test: OK")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_a", nargs="?")
    ap.add_argument("run_b", nargs="?")
    ap.add_argument("--out", default="tr1_budget.json")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()
    if a.self_test:
        _self_test()
        return
    ra = json.load(open(a.run_a))
    rb = json.load(open(a.run_b))
    rep, why = compose(ra, rb)
    if why:
        print(f"TR1 CENSUS: REFUSE\n{why}")
        sys.exit(2)
    json.dump(rep, open(a.out, "w"), indent=1)
    sh = rep["shares_pct"]
    print("TR1 CENSUS: OK  anchor "
          f"{rep['anchor_step_ms']:.0f} ms/step  "
          + "  ".join(f"{p}={sh[p]:.1f}%" for p in PHASES))


if __name__ == "__main__":
    main()
