# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""P228's reducer (bench/p228/PREREG-p228.md): K1's selection and microbench. At Qwen3-30B-A3B's dense shapes on one
RTX 5090, which cuBLASLt configuration should ``E4B_DENSE_LT_PIN`` pin for ``o_proj`` and the router at 2, 4 and 8 rows,
and does the pinned call beat the served default (``F.linear``, cuBLAS's own choice) at 2 and 4 rows?

Pure Python, no torch. The registered rules live here; the box imports them (``p228_bench.py``):
- **Block A, the SELECTION:** per (site, M), every candidate on cuBLASLt's own heuristic list (64 requested, INPLACE
  excluded) that cuBLASLt accepts, is bitwise equal run to run and is within the tolerance of ``F.linear``, timed on 8
  steps of 5 replays. :func:`select` names the pin (fastest, with a scheme-2 preference within :data:`TIE`).
- **Block B, the READING:** per (site, M), 60 reps (15 Williams squares of four arms): **D** (``F.linear`` at M, the
  served default: the CONTROL), **D2** (D captured again: the self-pair), **P** (the pinned call through e4b's own
  ``E4B_DENSE_LT_PIN`` path) and **T1** (``F.linear`` at one row, the one-row served path: reported).
- :func:`verdict`: per site PASS / MISS / TOLERANCE_FAIL / VOID; the selected pin table, the input to the code PR.

    python p228_reduce.py --bench bench_p228.json --out verdict_p228.json
    python p228_reduce.py --self-test
"""
from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import sys

LANE = "P228"
SITES = {"o_proj": {"N": 2048, "K": 4096, "copies": 48}, "router": {"N": 128, "K": 2048, "copies": 384}}
SITE_ORDER = ("o_proj", "router")
MS = (2, 4, 8)
RULED_MS = (2, 4)                            # the bar is read here; M = 8 is reported (its pin is still selected)
SELECT_STEPS, SELECT_ITERS = 8, 5            # block A: 8 input draws, 5 timed replays each
READ_STEPS = tuple(range(60))                # block B: 60 reps a cell, 15 whole Williams squares of four arms
ARMS = ["D", "D2", "P", "T1"]
SPACER = "S"                                 # D's calls with their own outputs, replayed untimed before every timed replay
ITERS = 10
BAR = 0.85                                   # P/D's upper bound (median + 2 SE), raw AND clock-adjusted, at M = 2 and 4
TOL = 1.5e-2                                 # max |P - F.linear| / max |F.linear| (the served routes' tolerance)
TIE = 1.02                                   # a scheme-2 (fp32 partials) candidate this close to the fastest is preferred
SELF_BAND = (0.99, 1.01)
CLOCK_TOL, MEM_CLOCK_TOL = 0.02, 0.01
MIN_POWER_W = 575
BOOT_B, BOOT_SEED = 2000, 228
REDUCTION_OK = (0, 2, 4)
CONFIG_FIELDS = ("algo_id", "tile", "split_k", "reduction_scheme", "cta_swizzling", "custom_option", "stages",
                 "inner_shape", "cluster_shape")
READ_VERDICTS = ("PASS", "MISS", "TOLERANCE_FAIL")


def _num(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x) and x > 0


def _good(xs, n):
    return isinstance(xs, list) and len(xs) == n and all(_num(x) for x in xs)


def select_ms(step_replays):
    """A candidate's selection time: the mean over the steps of each step's median replay (P222's)."""
    return statistics.fmean(statistics.median(x) for x in step_replays)


def eligible(c):
    cfg = c.get("config") or {}
    return (c.get("check_ok") is True and c.get("bitwise") is True and isinstance(c.get("rel"), (int, float))
            and c["rel"] <= TOL and set(cfg) == set(CONFIG_FIELDS) and cfg.get("reduction_scheme") in REDUCTION_OK
            and isinstance(c.get("step_ms"), list) and len(c["step_ms"]) == SELECT_STEPS
            and all(_good(x, SELECT_ITERS) for x in c["step_ms"]))


def select(cands):
    """The registered selection for one (site, M): among eligible candidates, the fastest by :func:`select_ms`; a
    scheme-2 candidate (fp32 partials) within :data:`TIE` of the fastest is preferred (the fastest such); ties go to
    the lower heuristic index. Returns ``(config, index)`` or ``(None, None)``."""
    ok = [c for c in cands if eligible(c)]
    if not ok:
        return None, None
    ok.sort(key=lambda c: (select_ms(c["step_ms"]), c["index"]))
    best = select_ms(ok[0]["step_ms"])
    two = [c for c in ok if c["config"]["reduction_scheme"] == 2 and select_ms(c["step_ms"]) <= TIE * best]
    pick = two[0] if two else ok[0]
    return dict(pick["config"]), pick["index"]


def selection_margin(cands):
    ts = sorted(select_ms(c["step_ms"]) for c in cands if eligible(c))
    return ts[1] / ts[0] if len(ts) > 1 else None


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


def clock_tag(site, M, arm):
    return f"{site}|{M}|{arm}"


def _global(rec, why):
    shake = bool(rec.get("shakedown"))
    if rec.get("lane") != LANE or rec.get("arms") != ARMS or rec.get("spacer") != SPACER or rec.get("iters") != ITERS:
        why.append("the lane, the arms, the spacer or the timed replays an arm are not the registered ones")
    if not shake and ("5090" not in str(rec.get("device", "")) or rec.get("cc") != [12, 0]):
        why.append(f"device {rec.get('device')} cc {rec.get('cc')}: the lane registers an RTX 5090 (sm_120)")
    if not isinstance((rec.get("key") or {}).get("cublaslt"), int):
        why.append("the loaded cuBLASLt version is unread")
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


def _cell(rec, site, M, why):
    shake = bool(rec.get("shakedown"))
    s = (rec.get("sites") or {}).get(site) or {}
    spec = SITES[site]
    if not shake and (s.get("N"), s.get("K"), s.get("copies")) != (spec["N"], spec["K"], spec["copies"]):
        why.append("the shape or the copies are not the registered ones")
    sel = (s.get("select") or {}).get(str(M)) or {}
    cfg, idx = select(sel.get("candidates") or [])
    if cfg is None:
        why.append(f"M={M}: the selection names no eligible candidate")
        return why
    if sel.get("chosen") != cfg:
        why.append(f"M={M}: block B timed the pin {sel.get('chosen')}, but the selection rule names {cfg}")
    cell = (s.get("cells") or {}).get(str(M)) or {}
    L = s.get("copies")
    eng = cell.get("engaged") or {}
    if eng.get("P") != {f"{site}|M{M}|eager|pinned": L} or eng.get("D") or eng.get("T1"):
        why.append(f"M={M}: the engagement is not P pinned on every copy and D/T1 never ({eng})")
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
        if not isinstance(rep.get("p_rel"), (int, float)) or not math.isfinite(rep["p_rel"]):
            why.append(f"M={M} rep {r}: P against D unread")
            break
    clk = rec.get("clocks") or {}
    if not clk.get("error") and not all(isinstance((clk.get(clock_tag(site, M, a)) or {}).get("sm_mhz_median"),
                                                   (int, float)) for a in ARMS):
        why.append(f"M={M}: an arm has no clock samples")
    return why


def read_cell(rec, site, M):
    s = rec["sites"][site]
    cell = s["cells"][str(M)]
    ms = [{a: statistics.median(v) / s["copies"] for a, v in rep["replays_ms"].items()} for rep in cell["reps"]]
    floor_ms = s["N"] * s["K"] * 2 / (rec["copy_gbs"] * 1e9) * 1e3 if _num(rec.get("copy_gbs")) else None
    out = {"per_call_ms": {a: statistics.median(m[a] for m in ms) for a in ARMS},
           "self_pair": summarize([m["D"] / m["D2"] for m in ms]),
           "p_over_d": summarize([m["P"] / m["D"] for m in ms]),
           "p_over_t1": statistics.median(m["P"] / m["T1"] for m in ms),
           "d_over_t1": statistics.median(m["D"] / m["T1"] for m in ms),
           "floor_ms": floor_ms, "p_rel_max": max(rep["p_rel"] for rep in cell["reps"])}
    clk = rec["clocks"]
    dc, pc = clk[clock_tag(site, M, "D")], clk[clock_tag(site, M, "P")]
    ratio = pc["sm_mhz_median"] / dc["sm_mhz_median"]
    p = out["p_over_d"]
    p["clock_p_over_d"] = ratio
    p["ub_clock_adj"] = p["ub"] * max(1.0, ratio)
    p["labels"] = [f"clock-confounded: SM {pc['sm_mhz_median']:.0f} against D's {dc['sm_mhz_median']:.0f}"] \
        if abs(ratio - 1) > CLOCK_TOL else []
    return out


def site_verdict(rec, site):
    shake = bool(rec.get("shakedown"))
    why = []
    for M in MS:
        _cell(rec, site, M, why)
    if why:
        return {"verdict": "VOID", "why": why, "read": False}
    cells = {M: read_cell(rec, site, M) for M in MS}
    out = {"cells": {str(M): c for M, c in cells.items()},
           "pins": {str(M): select(rec["sites"][site]["select"][str(M)]["candidates"])[0] for M in MS},
           "selection_margin": {str(M): selection_margin(rec["sites"][site]["select"][str(M)]["candidates"])
                                for M in MS}}
    outside = [M for M, c in cells.items() if not SELF_BAND[0] <= c["self_pair"]["median"] <= SELF_BAND[1]]
    if shake:
        return dict(out, verdict="SHAKEDOWN", read=False, why=["a shakedown record is never a reading"]
                    + ([f"self-pair outside {SELF_BAND} at M = {outside}: reported"] if outside else []))
    if outside:
        return dict(out, verdict="VOID", read=False, why=[f"self-pair D/D2 median outside {SELF_BAND} at M = {outside}"])
    bad = [M for M, c in cells.items() if c["p_rel_max"] > TOL]
    if bad:
        return dict(out, verdict="TOLERANCE_FAIL", read=True,
                    why=[f"P against D beyond {TOL} at M = {bad}: a contract finding; no timing is ruled on"])
    ok = all(cells[M]["p_over_d"]["ub"] <= BAR and cells[M]["p_over_d"]["ub_clock_adj"] <= BAR for M in RULED_MS)
    why = [f"M={M}: P/D {cells[M]['p_over_d']['median']:.4f}, upper bound {cells[M]['p_over_d']['ub']:.4f}, "
           f"clock-adjusted {cells[M]['p_over_d']['ub_clock_adj']:.4f}" for M in RULED_MS]
    return dict(out, verdict="PASS" if ok else "MISS", read=True,
                why=why + [f"{'both' if ok else 'not both'} <= {BAR} at M = {list(RULED_MS)}"])


def verdict(rec):
    why = _global(rec, [])
    out = {"lane": LANE, "shakedown": bool(rec.get("shakedown"))}
    if why:
        return dict(out, verdict="VOID", why=why, sites={})
    sites = {site: site_verdict(rec, site) for site in SITE_ORDER}
    out["sites"] = sites
    out["verdict"] = " ".join(f"{s}={sites[s]['verdict']}" for s in SITE_ORDER)
    out["key"] = rec["key"]
    out["pin_table"] = {f"{s}|{SITES[s]['N']}x{SITES[s]['K']}|M{M}": cfg for s in SITE_ORDER
                        for M, cfg in (sites[s].get("pins") or {}).items()}
    return out


# ---------------------------------------------------------------------------------------------------- self-test
def _cfg(i, scheme=4, split=9):
    return {"algo_id": 21, "tile": i, "split_k": split, "reduction_scheme": scheme, "cta_swizzling": 0,
            "custom_option": 0, "stages": 0, "inner_shape": 0, "cluster_shape": 0}


def _cands(times=(1.0, 0.9, 0.95), schemes=(4, 4, 2), rng=None):
    rng = rng or random.Random(5)
    return [{"index": i, "config": _cfg(i, scheme=sc), "check_ok": True, "bitwise": True, "rel": 0.005,
             "step_ms": [[t * (1 + rng.gauss(0, 0.001)) for _ in range(SELECT_ITERS)] for _ in range(SELECT_STEPS)]}
            for i, (t, sc) in enumerate(zip(times, schemes))]


def _fake(gains=None, noise=0.002, shake=False, n=len(READ_STEPS)):
    gains = gains or {"o_proj": {2: 0.6, 4: 0.6, 8: 0.6}, "router": {2: 0.5, 4: 0.5, 8: 1.0}}
    rng = random.Random(8)
    rec = {"lane": LANE, "shakedown": shake, "device": "NVIDIA GeForce RTX 5090", "cc": [12, 0], "arms": list(ARMS),
           "spacer": SPACER, "iters": ITERS, "copy_gbs": 1500.0, "key": {"cublaslt": 120804, "cc": [12, 0],
                                                                         "sm_count": 170},
           "sites": {}, "clocks": {"error": None}}
    for site in SITE_ORDER:
        spec = SITES[site]
        L = spec["copies"]
        s = {"N": spec["N"], "K": spec["K"], "copies": L, "select": {}, "cells": {}}
        for M in MS:
            cands = _cands(rng=rng)
            s["select"][str(M)] = {"candidates": cands, "chosen": select(cands)[0]}
            reps = []
            for r in range(n):
                base = 0.03 * L
                rp = {a: [base * (gains[site][M] if a == "P" else (0.9 if a == "T1" else 1.0)) * (1 + rng.gauss(0, noise))
                          for _ in range(ITERS)] for a in ARMS}
                reps.append({"step": r, "order": arm_order(r), "replays_ms": rp, "spacer_replays": ITERS * len(ARMS),
                             "d2_identical": True, "p_rel": 0.005})
            s["cells"][str(M)] = {"steps": list(range(n)), "reps": reps,
                                  "engaged": {"P": {f"{site}|M{M}|eager|pinned": L}, "D": {}, "T1": {}}}
            for a in ARMS:
                rec["clocks"][clock_tag(site, M, a)] = {"sm_mhz_median": 2707.0, "mem_mhz_median": 13801.0,
                                                         "enforced_limit_w": [575.0]}
        rec["sites"][site] = s
    return rec


def self_test():
    import copy
    n = 0

    def check(cond, msg):
        nonlocal n
        assert cond, msg
        n += 1
    # the selection: the fastest, a scheme-2 within TIE preferred, ineligible candidates never chosen
    cfg, idx = select(_cands(times=(1.0, 0.9, 0.95), schemes=(4, 4, 4)))
    check(idx == 1, f"the fastest: {idx}")
    cfg, idx = select(_cands(times=(1.0, 0.9, 0.915), schemes=(4, 4, 2)))
    check(idx == 2 and cfg["reduction_scheme"] == 2, "a scheme-2 within 1.02 is preferred")
    cfg, idx = select(_cands(times=(1.0, 0.9, 0.95), schemes=(4, 4, 2)))
    check(idx == 1, "a scheme-2 beyond 1.02 is not")
    c = _cands(times=(1.0, 0.9), schemes=(4, 4))
    c[1]["bitwise"] = False
    check(select(c)[1] == 0, "a non-deterministic candidate is never chosen")
    c = _cands(times=(1.0, 0.9), schemes=(4, 1))
    check(select(c)[1] == 0, "an INPLACE candidate is never chosen")
    c = _cands(times=(1.0, 0.9), schemes=(4, 4))
    c[1]["rel"] = 0.02
    check(select(c)[1] == 0, "a candidate beyond the tolerance is never chosen")
    check(select([])[0] is None, "no candidate")
    check(williams(4) == [[0, 1, 3, 2], [1, 2, 0, 3], [2, 3, 1, 0], [3, 0, 2, 1]] and len(READ_STEPS) % 4 == 0, "williams(4)")
    v = verdict(_fake())
    check(v["verdict"] == "o_proj=PASS router=PASS", v["verdict"])
    check(set(v["pin_table"]) == {f"{s}|{SITES[s]['N']}x{SITES[s]['K']}|M{M}" for s in SITE_ORDER for M in MS}, "pins")
    # the bar: the upper bound at BOTH ruled M; M = 8 never rules
    v = verdict(_fake(gains={"o_proj": {2: 0.6, 4: 0.9, 8: 0.6}, "router": {2: 0.5, 4: 0.5, 8: 2.0}}))
    check(v["sites"]["o_proj"]["verdict"] == "MISS" and v["sites"]["router"]["verdict"] == "PASS", v["verdict"])
    v = verdict(_fake(gains={"o_proj": {2: 0.847, 4: 0.847, 8: 0.6}, "router": {2: 0.5, 4: 0.5, 8: 1.0}}, noise=0.03))
    c2 = v["sites"]["o_proj"]["cells"]["2"]["p_over_d"]
    check(c2["median"] < BAR < c2["ub"] and v["sites"]["o_proj"]["verdict"] == "MISS", "median under, bound over")
    # the clock adjustment
    r = _fake(gains={"o_proj": {2: 0.83, 4: 0.83, 8: 0.6}, "router": {2: 0.5, 4: 0.5, 8: 1.0}}, noise=0.0005)
    check(verdict(r)["sites"]["o_proj"]["verdict"] == "PASS", "near-bar pass")
    for M in RULED_MS:
        r["clocks"][clock_tag("o_proj", M, "P")]["sm_mhz_median"] = 2707.0 * 1.05
    check(verdict(r)["sites"]["o_proj"]["verdict"] == "MISS", "a faster-clocked P is never credited")
    # VOIDs, global and per site
    for mutate, msg in [(lambda q: q.update(device="NVIDIA RTX A4000"), "device"),
                        (lambda q: q["key"].update(cublaslt=None), "version"),
                        (lambda q: q["clocks"][clock_tag("router", 2, "D")].update(enforced_limit_w=[400.0]), "power")]:
        q = copy.deepcopy(_fake())
        mutate(q)
        check(verdict(q)["verdict"] == "VOID", msg)
    for mutate, msg in [
        (lambda q: q["sites"]["o_proj"]["select"]["2"].update(chosen=_cfg(99)), "chosen not the rule's"),
        (lambda q: q["sites"]["o_proj"]["cells"]["4"]["engaged"].update(D={"o_proj|M4|eager|pinned": 1}), "D pinned"),
        (lambda q: q["sites"]["o_proj"]["cells"]["4"]["engaged"].update(P={}), "P not pinned"),
        (lambda q: q["sites"]["o_proj"]["cells"]["8"]["reps"][2].update(d2_identical=False), "D2 bits"),
        (lambda q: q["sites"]["o_proj"]["cells"]["2"]["reps"][1].update(order=["P", "D", "D2", "T1"]), "order"),
        (lambda q: q["sites"]["o_proj"].update(copies=47), "copies"),
    ]:
        q = copy.deepcopy(_fake())
        mutate(q)
        sv = verdict(q)["sites"]
        check(sv["o_proj"]["verdict"] == "VOID" and sv["router"]["verdict"] == "PASS", msg)
    q = _fake()
    for rep in q["sites"]["router"]["cells"]["4"]["reps"]:
        rep["replays_ms"]["D2"] = [x * 1.03 for x in rep["replays_ms"]["D2"]]
    check(verdict(q)["sites"]["router"]["verdict"] == "VOID", "self-pair band")
    q = _fake()
    q["sites"]["router"]["cells"]["2"]["reps"][7]["p_rel"] = 0.05
    sv = verdict(q)["sites"]["router"]
    check(sv["verdict"] == "TOLERANCE_FAIL" and sv["read"] is True, "tolerance, final")
    q = _fake(shake=True, n=4)
    q["device"] = "NVIDIA RTX A4000"
    v = verdict(q)
    check(v["verdict"] == "o_proj=SHAKEDOWN router=SHAKEDOWN" and not any(s["read"] for s in v["sites"].values()),
          v["verdict"])
    print(f"p228_reduce self-test OK ({n} checks)")


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
            print(f"p228_reduce self-test FAILED: {e}")
            return 1
        return 0
    try:
        with open(a.bench, encoding="utf-8") as f:
            rec = json.load(f)
        v = verdict(rec)
    except (OSError, ValueError) as e:            # a killed or never-written bench is a VOID record, never a crash
        v = {"lane": LANE, "verdict": "VOID", "sites": {},
             "why": [f"the bench record is missing or unreadable ({type(e).__name__}: {str(e)[:120]})"]}
    with open(a.out, "w", encoding="utf-8", newline="\n") as f:
        json.dump(v, f, indent=1, sort_keys=True)
    print(f"P228_VERDICT {v['verdict']}")
    for site, sv in (v.get("sites") or {}).items():
        print(f"  {site}: {sv['verdict']} -- {'; '.join(sv.get('why') or [])}")
    if not v.get("sites"):
        print("  " + "; ".join(v.get("why") or []))
    return 0


if __name__ == "__main__":
    sys.exit(main())
