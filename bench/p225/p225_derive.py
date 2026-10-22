#!/usr/bin/env python3
"""p225_derive.py -- lane P225's alarm and guard derivation (bench/p225/PREREG-p225.md, "The alarms"), from the
design inputs only: the four receipts' measured phase times (p224-5090-1, p224-5090-2, p225x-5090-3, p225x-5090-4;
explore points are design inputs, never results). The runner's per-tier NEED_* and *_ALARM lines are written from
:func:`plan`, and tests/test_p225.py parses them back out of the runner and asserts the invariant.

The model forms, fixed here:
- CPU-dispatch-bound phases (the train; the reference process, REF and HF0 together) are linear in 1/Mops between the
  two measured speeds, 36.4 and 18.6, and used ONLY between them (their intercepts are negative). They carry the
  maintainer's 1.75 margin for p224-5090-2's unnamed extra slowness.
- The reference process scales with the held-out rows: a fixed 22 s load, then per-row work.
- A scoring slot's extra time is GPU-bound: about 0.28 s a row on both a 36.4- and a 17.85-Mops host.
- Fetch, bake and a plain server slot are priced at their slowest measured time.
need = model x (1.75 for CPU-dispatch phases) x 1.5; alarm >= 1.25 x need, rounded up to 10 s;
guard = (sum of needs + gate 60 + reduce 60 + install 450 + reserve 600) x 1.25 + 300 boot;
worst = guard x USD 1.05/h + 66 GB x USD 0.011.

    python p225_derive.py --tier a --rows 1632
    python p225_derive.py --self-test
"""
from __future__ import annotations

import argparse
import json
import math
import sys

MOPS_FAST, MOPS_SLOW = 36.4, 18.6                      # the two measured single-thread speeds (p224-5090-1, p225x-5090-3)
TRAIN_S = {MOPS_FAST: 559.0, MOPS_SLOW: 1184.0}        # train_REF, 80 steps
REF688_S = {MOPS_FAST: 1247.0, MOPS_SLOW: 3497.0}      # the REF + HF0 process at 688 rows (1,247 = 22 + 2 x 0.89 x 688)
REF_LOAD_S = 22.0
FETCH_S, BAKE_S, SLOT_S = 806.0, 362.0, 98.0           # slowest measured: p224-5090-2's fetch and bake, its plain slot
SCORE_S_PER_ROW = 0.28                                  # a scoring slot's extra time per held-out row (GPU-bound)
# Read from each run's FIRST slot, so it carries the first server's cold start: (129 - 57) / 256 on p224-5090-1 and
# (274 - 82) / 688 on p225x-5090-4. Every measured first slot, the slowest being p224-5090-2's 364 s at 688 rows on an
# 18.0-Mops host, sits under need(slot) + need(score) at 688 rows (the self-test checks it).
FIRST_SLOTS_688 = {"p224-5090-2": 364.0, "p225x-5090-3": 264.0, "p225x-5090-4": 274.0}
MARGIN_UNNAMED, NEED_X, ALARM_X, GUARD_X = 1.75, 1.5, 1.25, 1.25
GATE_S, REDUCE_S, INSTALL_S, RESERVE_S, BOOT_S = 60, 60, 450, 600, 300
USD_H, DOWNLOAD_GB, USD_GB = 1.05, 66, 0.011
FLOOR_MOPS = {"a": 30.0}                                # one tier; the 18-Mops fallback was withdrawn (its guard
#                                                         is 8.9 h at the registered N, over the 6 h teardown cap)


def interp(table: dict, mops: float) -> float:
    """Linear in 1/Mops between the two measured speeds; never outside them."""
    if not (MOPS_SLOW <= mops <= MOPS_FAST):
        raise ValueError(f"{mops} Mops is outside the measured {MOPS_SLOW}..{MOPS_FAST}: no extrapolation")
    x0, x1 = 1 / MOPS_FAST, 1 / MOPS_SLOW
    f = (1 / mops - x0) / (x1 - x0)
    return table[MOPS_FAST] + f * (table[MOPS_SLOW] - table[MOPS_FAST])


def _ceil10(x: float) -> int:
    return int(math.ceil(x / 10.0) * 10)


def plan(tier: str, rows: int) -> dict:
    m = FLOOR_MOPS[tier]
    ref688 = interp(REF688_S, m)
    model = {"train": interp(TRAIN_S, m),
             "ref": REF_LOAD_S + (ref688 - REF_LOAD_S) * rows / 688,
             "fetch": FETCH_S, "bake": BAKE_S, "slot": SLOT_S, "score": SCORE_S_PER_ROW * rows}
    cpu = {"train", "ref"}
    need = {k: _ceil10(v * (MARGIN_UNNAMED if k in cpu else 1.0) * NEED_X) for k, v in model.items()}
    alarm = {k: _ceil10(ALARM_X * need[k]) for k in ("fetch", "bake", "train", "ref")}
    alarm["slot"] = _ceil10(ALARM_X * (need["slot"] + need["score"]))     # a slot's alarm covers its scoring term
    work = need["fetch"] + need["bake"] + need["train"] + need["ref"] + 6 * need["slot"] + 2 * need["score"] + GATE_S + REDUCE_S
    fit = INSTALL_S + work + RESERVE_S
    guard_s = fit * GUARD_X + BOOT_S
    worst = guard_s / 3600 * USD_H + DOWNLOAD_GB * USD_GB
    return {"tier": tier, "floor_mops": m, "rows": rows, "model_s": {k: round(v, 1) for k, v in model.items()},
            "need": need, "alarm": alarm, "need_work": work, "fit": fit, "guard_s": guard_s,
            "guard_h": guard_s / 3600, "worst_usd": worst}


def self_test() -> int:
    n = 0

    def ok(c, msg):
        nonlocal n
        assert c, msg
        n += 1
    ok(abs(interp(TRAIN_S, MOPS_FAST) - 559) < 1e-9 and abs(interp(TRAIN_S, MOPS_SLOW) - 1184) < 1e-9, "endpoints exact")
    ok(abs(interp(REF688_S, 30.0) - 1749) < 2, "ref at 30 Mops and 688 rows is about 1,749 s (the review's figure)")
    for bad in (18.0, 40.0):
        try:
            interp(TRAIN_S, bad)
            ok(False, f"{bad} must refuse")
        except ValueError:
            ok(True, "")
    p = plan("a", 688)
    ok(p["need"]["ref"] == 4600 and p["need"]["train"] == 1840, f"688 rows at tier a: {p['need']}")
    ok(all(p["alarm"][k] >= ALARM_X * p["need"][k] for k in ("fetch", "bake", "train", "ref")), "every alarm >= 1.25 x need")
    ok(p["alarm"]["slot"] >= ALARM_X * (p["need"]["slot"] + p["need"]["score"]), "the slot alarm covers scoring")
    big = plan("a", 1632)
    ok(big["need"]["ref"] > p["need"]["ref"] and big["guard_s"] > p["guard_s"], "more rows, more time")
    for t in FLOOR_MOPS:
        q = plan(t, 688)
        ok(max(FIRST_SLOTS_688.values()) <= q["need"]["slot"] + q["need"]["score"], f"every measured first slot fits ({t})")
    print(f"p225_derive self-test OK ({n} checks)")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tier", choices=sorted(FLOOR_MOPS))
    ap.add_argument("--rows", type=int)
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args(argv)
    if a.self_test:
        return self_test()
    if not (a.tier and a.rows):
        ap.error("--tier and --rows (or --self-test)")
    print(json.dumps(plan(a.tier, a.rows), indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
