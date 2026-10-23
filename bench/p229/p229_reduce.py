# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""P229's reducer (bench/p229/PREREG-p229.md): K2b's microbench. On one RTX 5090, does the int8 lm_head's 2..8-row route
(``E4B_INT8_HEAD_ROWS``, grouped-nf4-gemm K37's ``dense_i8.linear_i8`` at its shipped plan) read near the int8 bytes'
cost, against the served path at those rows (``F.linear`` on the bf16 rounding of the grid) and against K36 at one row?

Pure Python, no torch. The registered rules live here; the box imports them (``p229_bench.py``). Per M in 2, 4, 8, 60
reps (15 Williams squares of four arms): **D** (``F.linear`` on the grid, the served path: the CONTROL), **D2** (D
captured again: the self-pair), **I** (the head's installed route: K37 at M rows, the REGISTERED QUESTION) and **K** (the
head's one-row route, K36: the bytes' reference).

:func:`verdict`: GO_ROWS_8 / GO_ROWS_4 / GO_ROWS_2 (the largest M with every read M up to it PASSing: the knob's
licensed bound), NO_GO (M = 2 misses: K2a becomes the explicit next decision), TOLERANCE_FAIL or VOID.

    python p229_reduce.py --bench bench_p229.json --out verdict_p229.json
    python p229_reduce.py --self-test
"""
from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import sys

LANE = "P229"
HEAD = {"N": 151936, "K": 2048}              # Qwen3-30B-A3B's lm_head
MS = (2, 4, 8)                               # every row count the knob may route is read (the review's condition)
READ_STEPS = tuple(range(60))                # 60 reps a cell, 15 whole Williams squares of four arms
ARMS = ["D", "D2", "I", "K"]
SPACER = "S"                                 # D's call with its own output, replayed untimed before every timed replay
ITERS = 10
BAR_D = 0.70                                 # I/D's upper bound (median + 2 SE), raw AND clock-adjusted
BAR_K = 1.25                                 # I/K's upper bound, raw AND clock-adjusted (the same bytes as one row)
TOL = 1.5e-2                                 # max |I - D| / max |D| (the served routes' tolerance)
SHIPPED_PLAN = [32, 256, 4]                  # K37's PLAN_SMALL_DEFAULT: the plan the route runs at the head's shape
ROWS_RECORD = {"rows_max": 8, "kernel": "dense_i8.linear_i8 (K37)",       # the plan EVERY routed M runs: the shipped one
               "plan": {str(m): SHIPPED_PLAN for m in range(2, 9)}}
GNF4_COMMIT = "41ef086e6a3e2be5b158ff1bb38e0de67862a949"   # grouped-nf4-gemm v0.50.0, released
SELF_BAND = (0.99, 1.01)
CLOCK_TOL = 0.02
MIN_POWER_W = 575
BOOT_B, BOOT_SEED = 2000, 229
READ_VERDICTS = ("GO_ROWS_8", "GO_ROWS_4", "GO_ROWS_2", "NO_GO", "TOLERANCE_FAIL")


def _num(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x) and x > 0


def _good(xs, n):
    return isinstance(xs, list) and len(xs) == n and all(_num(x) for x in xs)


def williams(n):
    first, lo, hi = [0], 1, n - 1
    while len(first) < n:
        first.append(lo)
        lo += 1
        if len(first) < n:
            first.append(hi)
            hi -= 1
    rows = [[(x + i) % n for x in first] for i in range(n)]
    return rows if n % 2 == 0 else rows + [row[::-1] for row in rows]


def arm_order(r, arms=ARMS):
    w = williams(len(arms))
    return [arms[j] for j in w[r % len(w)]]


def se_median(q, b=BOOT_B, seed=BOOT_SEED):
    n = len(q)
    normal = math.sqrt(math.pi / 2) * statistics.stdev(q) / math.sqrt(n)
    rng = random.Random(seed)
    boot = statistics.stdev(statistics.median([q[int(rng.random() * n)] for _ in range(n)]) for _ in range(b))
    return max(normal, boot)


def summarize(q):
    med = statistics.median(q)
    se = se_median(q) if len(q) > 1 else math.inf
    return {"n": len(q), "median": med, "se": se, "ub": med + 2 * se, "lb": med - 2 * se, "min": min(q), "max": max(q)}


def clock_tag(M, arm):
    return f"{M}|{arm}"


def engaged_want(M):
    """The engagement each arm's eager pass must add to the head's ``routed`` counts: I routes M rows (K37), K one
    row (K36), D and D2 call ``F.linear`` on the weight and never the head."""
    return {"D": {}, "D2": {}, "I": {str(M): 1}, "K": {"1": 1}}


def _global(rec, why):
    shake = bool(rec.get("shakedown"))
    if rec.get("lane") != LANE or rec.get("arms") != ARMS or rec.get("spacer") != SPACER or rec.get("iters") != ITERS:
        why.append("the lane, the arms, the spacer or the timed replays an arm are not the registered ones")
    if not shake and ("5090" not in str(rec.get("device", "")) or rec.get("cc") != [12, 0]):
        why.append(f"device {rec.get('device')} cc {rec.get('cc')}: the lane registers an RTX 5090 (sm_120)")
    if rec.get("gnf4_commit") != GNF4_COMMIT:
        why.append(f"grouped-nf4-gemm {rec.get('gnf4_commit')} is not the registered release commit {GNF4_COMMIT}")
    if rec.get("rows_record") != ROWS_RECORD:
        why.append(f"the installed route {rec.get('rows_record')} is not the registered one {ROWS_RECORD} (the bar reads "
                   "the plan the shipped route runs)")
    if (rec.get("head") or {}) != HEAD:
        why.append(f"the head {rec.get('head')} is not the registered {HEAD}")
    for k in ("poisoned", "arm_fault"):
        if rec.get(k):
            why.append(f"{k}: {rec[k]}")
    clk = rec.get("clocks") or {}
    if clk.get("error"):
        why.append(f"the clocks are unread ({clk['error']})")
    elif not shake:
        lim = [w for v in clk.values() if isinstance(v, dict) for w in v.get("enforced_limit_w") or [None]]
        if not lim or any(not _num(w) or w < MIN_POWER_W for w in lim):
            why.append(f"an enforced power limit {sorted({str(w) for w in lim})} W under {MIN_POWER_W} W")
    return why


def _cell(rec, M, why):
    shake = bool(rec.get("shakedown"))
    cell = (rec.get("cells") or {}).get(str(M)) or {}
    if cell.get("engaged") != engaged_want(M):
        why.append(f"M={M}: the engagement {cell.get('engaged')} is not I on K37 at M rows, K on K36 at one, D/D2 never")
    steps = cell.get("steps") or []
    if (steps != list(READ_STEPS)) if not shake else not (steps and steps == list(READ_STEPS)[:len(steps)]):
        why.append(f"M={M}: the reading's steps are not the registered ones")
    reps = cell.get("reps") or []
    if len(reps) != len(steps) or not steps:
        why.append(f"M={M}: {len(reps)} reps for {len(steps)} steps")
        return why
    for r, rep in enumerate(reps):
        rp = rep.get("replays_ms") or {}
        if rep.get("step") != steps[r] or rep.get("order") != arm_order(r):
            why.append(f"M={M} rep {r}: its step or order is not the registered Williams row")
            break
        if set(rp) != set(ARMS) or not all(_good(rp[a], ITERS) for a in ARMS):
            why.append(f"M={M} rep {r}: replays missing or malformed")
            break
        if rep.get("spacer_replays") != ITERS * len(ARMS):
            why.append(f"M={M} rep {r}: not one spacer replay before each timed replay")
            break
        if rep.get("d2_identical") is not True:
            why.append(f"M={M} rep {r}: D replayed different bits in a second capture")
            break
        if not isinstance(rep.get("i_rel"), (int, float)) or not math.isfinite(rep["i_rel"]):
            why.append(f"M={M} rep {r}: I against D unread")
            break
    clk = rec.get("clocks") or {}
    if not clk.get("error") and not all(isinstance((clk.get(clock_tag(M, a)) or {}).get("sm_mhz_median"),
                                                   (int, float)) for a in ARMS):
        why.append(f"M={M}: an arm has no clock samples")
    return why


def _adjust(s, num, den, clk, M):
    """The one-sided clock adjustment: the bound scales up by the numerator arm's SM clock over the denominator's when
    the numerator ran faster-clocked; it is never credited."""
    a, b = clk[clock_tag(M, num)]["sm_mhz_median"], clk[clock_tag(M, den)]["sm_mhz_median"]
    ratio = a / b
    s["clock_ratio"] = ratio
    s["ub_clock_adj"] = s["ub"] * max(1.0, ratio)
    s["labels"] = [f"clock-confounded: {num} SM {a:.0f} against {den}'s {b:.0f}"] if abs(ratio - 1) > CLOCK_TOL else []
    return s


def read_cell(rec, M):
    cell = rec["cells"][str(M)]
    ms = [{a: statistics.median(v) for a, v in rep["replays_ms"].items()} for rep in cell["reps"]]
    gbs = rec.get("copy_gbs")
    out = {"per_call_ms": {a: statistics.median(m[a] for m in ms) for a in ARMS},
           "self_pair": summarize([m["D"] / m["D2"] for m in ms]),
           "i_over_d": _adjust(summarize([m["I"] / m["D"] for m in ms]), "I", "D", rec["clocks"], M),
           "i_over_k": _adjust(summarize([m["I"] / m["K"] for m in ms]), "I", "K", rec["clocks"], M),
           "d_over_k": statistics.median(m["D"] / m["K"] for m in ms),
           "floor_int8_ms": (HEAD["N"] * HEAD["K"] + 4 * HEAD["N"]) / (gbs * 1e9) * 1e3 if _num(gbs) else None,
           "floor_bf16_ms": HEAD["N"] * HEAD["K"] * 2 / (gbs * 1e9) * 1e3 if _num(gbs) else None,
           "i_rel_max": max(rep["i_rel"] for rep in cell["reps"])}
    d, k = out["i_over_d"], out["i_over_k"]
    out["passes"] = d["ub"] <= BAR_D and d["ub_clock_adj"] <= BAR_D and k["ub"] <= BAR_K and k["ub_clock_adj"] <= BAR_K
    return out


def licensed_rows(passes):
    """The knob's licensed bound: the largest M in :data:`MS` such that every read M up to it PASSes, else None."""
    bound = None
    for M in MS:
        if not passes[M]:
            break
        bound = M
    return bound


def verdict(rec):
    shake = bool(rec.get("shakedown"))
    out = {"lane": LANE, "shakedown": shake}
    why = _global(rec, [])
    for M in MS:
        _cell(rec, M, why)
    if why:
        return dict(out, verdict="VOID", read=False, why=why)
    cells = {M: read_cell(rec, M) for M in MS}
    out["cells"] = {str(M): c for M, c in cells.items()}
    outside = [M for M, c in cells.items() if not SELF_BAND[0] <= c["self_pair"]["median"] <= SELF_BAND[1]]
    if shake:
        return dict(out, verdict="SHAKEDOWN", read=False, why=["a shakedown record is never a reading"]
                    + ([f"self-pair outside {SELF_BAND} at M = {outside}: reported"] if outside else []))
    if outside:
        return dict(out, verdict="VOID", read=False, why=[f"self-pair D/D2 median outside {SELF_BAND} at M = {outside}"])
    bad = [M for M, c in cells.items() if c["i_rel_max"] > TOL]
    if bad:
        return dict(out, verdict="TOLERANCE_FAIL", read=True,
                    why=[f"I against D beyond {TOL} at M = {bad}: a contract finding; no timing is ruled on"])
    bound = licensed_rows({M: c["passes"] for M, c in cells.items()})
    why = [f"M={M}: I/D {c['i_over_d']['median']:.4f} (ub {c['i_over_d']['ub']:.4f}, clock-adjusted "
           f"{c['i_over_d']['ub_clock_adj']:.4f}; bar {BAR_D}), I/K {c['i_over_k']['median']:.4f} (ub "
           f"{c['i_over_k']['ub']:.4f}, clock-adjusted {c['i_over_k']['ub_clock_adj']:.4f}; bar {BAR_K}): "
           f"{'PASS' if c['passes'] else 'MISS'}" for M, c in cells.items()]
    if bound is None:
        return dict(out, verdict="NO_GO", read=True, licensed_rows=None,
                    why=why + ["M = 2 misses: K2b is NO_GO at the shipped plan, and K2a is the explicit next decision"])
    return dict(out, verdict=f"GO_ROWS_{bound}", read=True, licensed_rows=bound,
                why=why + [f"every read M up to {bound} passes: E4B_INT8_HEAD_ROWS is licensed up to {bound}"])


# ---------------------------------------------------------------------------------------------------- self-test
def _fake(gains=None, noise=0.002, shake=False, n=len(READ_STEPS)):
    """A synthetic record: per M, I's time over D's (``gains``); K one row at about I's bytes."""
    gains = gains or {2: 0.54, 4: 0.54, 8: 0.55}
    rng = random.Random(8)
    rec = {"lane": LANE, "shakedown": shake, "device": "NVIDIA GeForce RTX 5090", "cc": [12, 0], "arms": list(ARMS),
           "spacer": SPACER, "iters": ITERS, "copy_gbs": 1500.0, "gnf4_commit": GNF4_COMMIT,
           "rows_record": json.loads(json.dumps(ROWS_RECORD)), "head": dict(HEAD), "cells": {}, "clocks": {"error": None}}
    for M in MS:
        reps = []
        for r in range(n):
            base = 0.405
            t = {"D": base, "D2": base, "I": base * gains[M], "K": 0.211}
            rp = {a: [t[a] * (1 + rng.gauss(0, noise)) for _ in range(ITERS)] for a in ARMS}
            reps.append({"step": r, "order": arm_order(r), "replays_ms": rp, "spacer_replays": ITERS * len(ARMS),
                         "d2_identical": True, "i_rel": 0.004})
        rec["cells"][str(M)] = {"steps": list(range(n)), "reps": reps, "engaged": engaged_want(M)}
        for a in ARMS:
            rec["clocks"][clock_tag(M, a)] = {"sm_mhz_median": 2707.0, "mem_mhz_median": 13801.0,
                                              "enforced_limit_w": [575.0]}
    return rec


def self_test():
    import copy
    n = 0

    def check(cond, msg):
        nonlocal n
        assert cond, msg
        n += 1
    check(williams(4) == [[0, 1, 3, 2], [1, 2, 0, 3], [2, 3, 1, 0], [3, 0, 2, 1]] and len(READ_STEPS) % 4 == 0, "williams")
    v = verdict(_fake())
    check(v["verdict"] == "GO_ROWS_8" and v["licensed_rows"] == 8 and v["read"], v["verdict"])
    # the bound: the largest M with every read M up to it passing
    check(licensed_rows({2: True, 4: True, 8: False}) == 4 and licensed_rows({2: True, 4: False, 8: True}) == 2
          and licensed_rows({2: False, 4: True, 8: True}) is None, "licensed_rows")
    v = verdict(_fake(gains={2: 0.54, 4: 0.54, 8: 0.80}))
    check(v["verdict"] == "GO_ROWS_4", "M = 8 over the D bar caps the knob at 4")
    v = verdict(_fake(gains={2: 0.54, 4: 0.90, 8: 0.54}))
    check(v["verdict"] == "GO_ROWS_2", "a miss at 4 caps at 2 even when 8 passes")
    v = verdict(_fake(gains={2: 0.80, 4: 0.54, 8: 0.54}))
    check(v["verdict"] == "NO_GO" and v["read"] and v["licensed_rows"] is None, "a miss at 2 is NO_GO")
    # the K bar binds on its own: I fast against D but slow against one row's bytes
    q = _fake(gains={2: 0.69, 4: 0.54, 8: 0.54})
    check(verdict(q)["verdict"] == "NO_GO", "I/K over 1.25 at M = 2 (0.69 x 0.405 / 0.211 = 1.32)")
    # the upper bound, never the median
    v = verdict(_fake(gains={2: 0.697, 4: 0.54, 8: 0.54}, noise=0.03))
    c2 = v["cells"]["2"]["i_over_d"]
    check(c2["median"] < BAR_D < c2["ub"] and v["verdict"] == "NO_GO", "median under, bound over")
    # the clock adjustment, one-sided: a faster-clocked I is never credited
    r = _fake(gains={2: 0.60, 4: 0.54, 8: 0.54}, noise=0.0005)
    r["cells"]["2"]["reps"] = [dict(rep, replays_ms=dict(rep["replays_ms"], K=[x * 1.21 / 1.0 for x in
                                                                              rep["replays_ms"]["K"]]))
                               for rep in r["cells"]["2"]["reps"]]
    check(verdict(r)["verdict"] == "GO_ROWS_8", "near-bar pass (I/K about 0.95)")
    r["clocks"][clock_tag(2, "I")]["sm_mhz_median"] = 2707.0 * 1.20
    check(verdict(r)["verdict"] == "NO_GO", "I faster-clocked than D by 20 %: the D bar's adjusted bound misses")
    # VOIDs: the whole record
    for mutate, msg in [(lambda q: q.update(device="NVIDIA RTX A4000"), "device"),
                        (lambda q: q.update(gnf4_commit="a83f6bb3a3a69aef8981fb81b6361bf62b06f6e7"), "package commit"),
                        (lambda q: q["rows_record"]["plan"].update({"8": [128, 128, 4]}), "one M off the shipped plan"),
                        (lambda q: q["rows_record"].update(plan=[32, 256, 4]), "the plan of M = 2 alone"),
                        (lambda q: q["rows_record"].update(rows_max=4), "the knob not at 8"),
                        (lambda q: q["head"].update(N=4096), "the head shape"),
                        (lambda q: q["clocks"][clock_tag(4, "D")].update(enforced_limit_w=[400.0]), "power"),
                        (lambda q: q["cells"]["8"]["reps"][2].update(d2_identical=False), "D2 bits"),
                        (lambda q: q["cells"]["2"]["reps"][1].update(order=["I", "D", "D2", "K"]), "order"),
                        (lambda q: q["cells"]["4"]["engaged"].update(I={}), "I never routed"),
                        (lambda q: q["cells"]["4"]["engaged"].update(D={"4": 1}), "D through the head"),
                        (lambda q: q["cells"]["2"]["reps"][3].update(spacer_replays=39), "spacer")]:
        q = copy.deepcopy(_fake())
        mutate(q)
        check(verdict(q)["verdict"] == "VOID", msg)
    q = _fake()
    for rep in q["cells"]["4"]["reps"]:
        rep["replays_ms"]["D2"] = [x * 1.03 for x in rep["replays_ms"]["D2"]]
    check(verdict(q)["verdict"] == "VOID", "self-pair band")
    q = _fake()
    q["cells"]["8"]["reps"][7]["i_rel"] = 0.05
    v = verdict(q)
    check(v["verdict"] == "TOLERANCE_FAIL" and v["read"] is True, "tolerance, final")
    q = _fake(shake=True, n=4)
    q["device"] = "NVIDIA RTX A4000"
    q["clocks"][clock_tag(2, "D")]["enforced_limit_w"] = [140.0]
    v = verdict(q)
    check(v["verdict"] == "SHAKEDOWN" and v["read"] is False, v["verdict"])
    print(f"p229_reduce self-test OK ({n} checks)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench")
    ap.add_argument("--out")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()
    if a.self_test:
        try:
            self_test()
        except AssertionError as e:
            print(f"p229_reduce self-test FAILED: {e}")
            return 1
        return 0
    try:
        with open(a.bench, encoding="utf-8") as f:
            rec = json.load(f)
        v = verdict(rec)
    except (OSError, ValueError) as e:            # a killed or never-written bench is a VOID record, never a crash
        v = {"lane": LANE, "verdict": "VOID", "read": False,
             "why": [f"the bench record is missing or unreadable ({type(e).__name__}: {str(e)[:120]})"]}
    with open(a.out, "w", encoding="utf-8", newline="\n") as f:
        json.dump(v, f, indent=1, sort_keys=True)
    print(f"P229_VERDICT {v['verdict']}")
    print("  " + "; ".join(v.get("why") or []))
    return 0


if __name__ == "__main__":
    sys.exit(main())
