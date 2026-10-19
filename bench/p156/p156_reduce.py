# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""P156's reducer (bench/p156/PREREG-p156.md): K39's decode (K25's ``lut="prmt"``) against today's select tree at
Qwen3-30B-A3B's W16 shapes, timed on S0's real router picks (P153's router replay), selection split from confirmation.

Pure Python, no torch. The registered rules live here, and the box imports them (``p156_bench.py``):
- **The question, fixed before data:** P, ``lut="prmt"`` at today's plan, against T, ``lut="tree"`` at today's plan.
  Only the decode differs. GO is read on P alone, from the CONFIRMATION, so P has no winner's curse.
- :func:`select`: the SWEEP rule, reported only. ``lut="prmt"`` over K25's 144-plan grid on the selection steps. Per
  projection, the plans that ran bit-identical to today's tree on the identity step are ranked by their mean selection
  time (ties by grid order). Today's plan is P and is never carried. The top three of each, paired by rank, are C1..C3:
  they price decode x plan for a later lane and never enter the verdict;
- :func:`rotation`: each confirmation rep's capture and replay order;
- :func:`verdict`: GO / NO_GO / VOID from the confirmation read. The reducer re-derives everything from the bench's raw
  replay times, including the plans the box carried, which must equal :func:`select` of the record or the read is VOID.

    python p156_reduce.py --bench bench_p156.json --out verdict_p156.json
    python p156_reduce.py --self-test
"""
from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import sys

RR_SHA256 = "3c59e72f947daa425e61f727401204099a1e782b902fb39687e04c46e338b14f"   # bench/p153/router-replay/rr.json.pt
S0_SHA256 = "3020dbce691bd98462b9c74b9fe903c1b124d7a6b95f22e8d964532479b282a1"   # its picks[:, 0:16], <i2 [48, 16, 128, 8]
S0_DISTINCT_TOTAL = 350868   # P153's record: S0's distinct experts summed over 48 layers x 128 steps
E, TOPK, TOKENS, LAYERS, STEPS = 128, 8, 16, 48, 128
SHAPES = {"gu": (1536, 2048), "dn": (2048, 768)}           # (N, K): gate_up [2 x 768, 2048], down [2048, 768]
TODAY = {"block_n": 32, "kc": 64, "warps": 4, "stages": 3}  # e4b's _K25_PLAN; dot_bf16 False is fixed
T_LUT, P_LUT = "tree", "prmt"                              # today's decode, and K39's (grouped-nf4-gemm#569)
GRID = [{"block_n": bn, "kc": kc, "warps": w, "stages": s}
        for bn in (16, 32, 64, 128) for kc in (64, 128, 256) for w in (2, 4, 8) for s in (1, 2, 3, 4)]
SELECT_STEPS = tuple(range(0, 64, 8))      # 8 steps of the first half
IDENTITY_STEP = 0                          # every swept plan's outputs against today's tree: all layers, both projections
CONFIRM_STEPS = tuple(range(64, 128))      # the 64 held-out steps, one paired rep each
CARRY = 3                                  # carried forward from each projection's sweep ranking (reported only)
ITERS = 10                                 # timed replays per graph
ORDER_SEED = 156                           # the sweep's plan order (one shuffle, both projections)
GO_BAR = 1.05                              # P's median ratio T/P minus 2 SE (of the median) must reach this
BOOT_B, BOOT_SEED = 2000, 156              # the median's SE: a bootstrap of 2,000 resamples from one seeded generator
SELF_BAND = (0.99, 1.01)                   # today's self-pair median, else the instrument is VOID
P152_BOTH_MS = 8.395073 + 3.35296          # P152's verdict: the removable time of both K25 lines, 136x48 + 136x64
P154_HEADROOM_MS = 6.8124                  # P154's verdict: that time minus the byte floor
BYTE_FLOOR_MS = P152_BOTH_MS - P154_HEADROOM_MS   # 4.94 ms
#: the issue/occupancy model's bands for P's median T/P (PREREG "Predictions"): M1 overlap with a fixed residual, M3
#: serial issue, M2 overlap with the residual hidden by occupancy 50 -> 67 %; the headline spans M1's top to M3's
MODELS = {"M1 overlap": (1.02, 1.11), "M3 serial": (1.15, 1.18), "M2 overlap + occupancy": (1.19, 1.28)}
PRED = {"fidelity_ms": (9.5, 14.0), "headline": (1.05, 1.18), "self_pair": 0.005, "shrink": (0.0, 0.03),
        "regs": {"T": 80, "P": 64}}


def plan_key(p):
    return (p["block_n"], p["kc"], p["warps"], p["stages"])


_GRID_INDEX = {plan_key(p): i for i, p in enumerate(GRID)}


def plan_order():
    """The sweep's timing order: GRID indices, one seeded shuffle (drift spreads over the grid, not by plan)."""
    order = list(range(len(GRID)))
    random.Random(ORDER_SEED).shuffle(order)
    return order


def rotation(arms, r):
    """Rep r's capture and replay order: the arms rotated by r, so each arm takes each position equally often."""
    k = r % len(arms)
    return list(arms[k:]) + list(arms[:k])


def confirm_arms(n_carried):
    return ["T", "T2", "P"] + [f"C{i + 1}" for i in range(n_carried)]


def _num(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x) and x > 0


def _good_replays(xs, n=ITERS):
    return isinstance(xs, list) and len(xs) == n and all(_num(x) for x in xs)


def select_ms(step_replays):
    """A plan's selection time: the mean over the selection steps of each step's median replay."""
    return statistics.fmean(statistics.median(x) for x in step_replays)


def _eligible(row, n_steps):
    sr = row.get("step_replays_ms")
    return (row.get("identical") is True and not row.get("error") and isinstance(sr, list) and len(sr) == n_steps
            and all(_good_replays(x) for x in sr))


def select(rows, n_steps=len(SELECT_STEPS)):
    """The registered sweep rule (module docstring). ``rows``: {"gu": [...], "dn": [...]}, one row per grid plan, each
    timed at ``lut="prmt"``. Today's plan is P, the registered question, so it is never carried."""
    ranked = {}
    for proj in ("gu", "dn"):
        ok = [r for r in rows[proj] if _eligible(r, n_steps) and r["plan"] != TODAY]
        ok.sort(key=lambda r: (select_ms(r["step_replays_ms"]), _GRID_INDEX[plan_key(r["plan"])]))
        ranked[proj] = [dict(r["plan"]) for r in ok]
    n = min(CARRY, len(ranked["gu"]), len(ranked["dn"]))
    return [{"gu": ranked["gu"][i], "dn": ranked["dn"][i]} for i in range(n)]


def rep_ms(rep):
    return {arm: statistics.median(v) for arm, v in rep["replays_ms"].items()}


def se_median(q, b=BOOT_B, seed=BOOT_SEED):
    """The standard error of the MEDIAN of ``q``, P155's rule: the larger of the normal-theory value,
    sqrt(pi / 2) x sd / sqrt(n), and the standard deviation of the medians of ``b`` resamples with replacement from one
    seeded generator drawn through ``random()`` alone (so any platform re-derives it). The bootstrap can only widen
    the bound, under heavy tails."""
    n = len(q)
    normal = math.sqrt(math.pi / 2) * statistics.stdev(q) / math.sqrt(n)
    rng = random.Random(seed)
    boot = statistics.stdev(statistics.median([q[int(rng.random() * n)] for _ in range(n)]) for _ in range(b))
    return max(normal, boot)


def summarize(q):
    n = len(q)
    med = statistics.median(q)
    se = se_median(q) if n > 1 else math.inf
    return {"n": n, "median": med, "se": se, "lb": med - 2 * se, "se_of_mean": statistics.stdev(q) / math.sqrt(n)
            if n > 1 else math.inf, "min": min(q), "max": max(q)}


def _structure(rec, why):
    """Every VOID that is a property of the record's shape, inputs and instrument; appends to ``why``."""
    shake = bool(rec.get("shakedown"))
    layers = rec.get("layers")
    if rec.get("rr_sha256") != RR_SHA256:
        why.append(f"router replay sha256 {rec.get('rr_sha256')} is not the pinned {RR_SHA256[:8]}")
    if rec.get("s0_sha256") != S0_SHA256:
        why.append(f"S0's picks hash {rec.get('s0_sha256')}, pinned {S0_SHA256[:8]}")
    if rec.get("grid") != GRID or rec.get("today") != TODAY:
        why.append("the grid or today's plan is not the registered one")
    if rec.get("luts") != {"T": T_LUT, "P": P_LUT, "sweep": P_LUT}:
        why.append(f"the decodes {rec.get('luts')} are not the registered T tree, P and sweep prmt")
    if rec.get("order") != plan_order():
        why.append("the sweep's plan order is not the registered shuffle")
    if not shake:
        if "5090" not in str(rec.get("device", "")) or rec.get("cc") != [12, 0]:
            why.append(f"device {rec.get('device')} cc {rec.get('cc')}: the lane registers an RTX 5090 (sm_120)")
        if layers != LAYERS:
            why.append(f"{layers} layers timed, registered {LAYERS}")
        if rec.get("select_steps") != list(SELECT_STEPS) or rec.get("confirm_steps") != list(CONFIRM_STEPS):
            why.append("the selection or confirmation steps are not the registered ones")
    else:                                       # a shakedown: fewer layers and steps, drawn from the registered ones
        if not isinstance(layers, int) or not 1 <= layers <= LAYERS:
            why.append(f"shakedown layers {layers}")
        ss, cs = rec.get("select_steps") or [], rec.get("confirm_steps") or []
        if not (ss and set(ss) <= set(SELECT_STEPS) and cs and cs == list(CONFIRM_STEPS[:len(cs)])):
            why.append("shakedown steps are not drawn from the registered ones")
    nreal = rec.get("n_real")
    if not (isinstance(nreal, list) and len(nreal) == STEPS and isinstance(layers, int)
            and all(isinstance(x, list) and len(x) == layers and all(isinstance(v, int) and TOPK <= v <= E for v in x)
                    for x in nreal)):
        why.append("n_real is not [128 steps][layers] of distinct-expert counts")
    elif layers == LAYERS and sum(map(sum, nreal)) != S0_DISTINCT_TOTAL:
        why.append(f"S0's distinct total {sum(map(sum, nreal))} is not P153's {S0_DISTINCT_TOTAL}")
    if rec.get("tile_faults"):
        why.append(f"tile table faults: {rec['tile_faults'][:3]}")
    sel = rec.get("select") or {}
    for proj in ("gu", "dn"):
        rows = sel.get(proj)
        if not isinstance(rows, list) or sorted(_GRID_INDEX.get(plan_key(r["plan"]), -1) for r in rows) != list(range(len(GRID))):
            why.append(f"{proj}: the sweep rows are not one per grid plan")
            continue
        today = [r for r in rows if r["plan"] == TODAY][0]
        # P that ran but moved a bit is a contract finding (verdict), never a VOID; VOID only when it failed to
        # compile, launch or replay, the instrument's failure
        if today.get("error") or (today.get("identical") is not False and not _eligible(today, len(rec.get("select_steps") or []))):
            why.append(f"{proj}: P (prmt at today's plan) did not compile, launch or replay cleanly in the sweep "
                       f"({today.get('error') or 'no clean replays'})")
    return why


def verdict(rec):
    why = _structure(rec, [])
    shake = bool(rec.get("shakedown"))
    out = {"lane": "P156", "shakedown": shake}
    if why:
        return dict(out, verdict="VOID", why=why)
    n_sel = len(rec["select_steps"])
    carried = select(rec["select"], n_sel)
    if rec.get("carried") != carried:
        return dict(out, verdict="VOID", why=["the carried plans are not the registered sweep of the recorded times"])
    arms = confirm_arms(len(carried))
    conf = rec.get("confirm") or {}
    reps = conf.get("reps") or []
    steps = rec["confirm_steps"]
    if conf.get("arms") != arms or len(reps) != len(steps) or not steps:
        return dict(out, verdict="VOID", why=[f"confirmation: arms {conf.get('arms')} / {len(reps)} reps, registered "
                                              f"{arms} / {len(steps)}"])
    for r, rep in enumerate(reps):
        if rep.get("step") != steps[r] or rep.get("rotation") != rotation(arms, r):
            return dict(out, verdict="VOID", why=[f"rep {r}: step {rep.get('step')} or its rotation is not as registered"])
        rp = rep.get("replays_ms") or {}
        if set(rp) != set(arms) or not all(_good_replays(rp[a]) for a in arms):
            return dict(out, verdict="VOID", why=[f"rep {r}: replays missing or malformed"])
        if set(rep.get("identical") or {}) != set(arms) - {"T"}:
            return dict(out, verdict="VOID", why=[f"rep {r}: identity not recorded for every arm"])
    ms = [rep_ms(rep) for rep in reps]
    if not all(rep["identical"]["T2"] is True for rep in reps):
        return dict(out, verdict="VOID", why=["today's tree replayed different bits in a second capture: not deterministic"])
    q = {a: [m["T"] / m[a] for m in ms] for a in arms if a != "T"}
    self_pair = summarize(q["T2"])
    out["self_pair"] = self_pair
    if not SELF_BAND[0] <= self_pair["median"] <= SELF_BAND[1]:
        return dict(out, verdict="VOID", why=[f"self-pair median {self_pair['median']:.4f} outside {SELF_BAND}"])
    # P moving a bit reads the same on every step: the identity step's (per projection) and the held-out steps'
    p_moved = ([IDENTITY_STEP] if any(next(r for r in rec["select"][pj] if r["plan"] == TODAY)["identical"] is False
                                      for pj in ("gu", "dn")) else [])
    p_moved += [rep["step"] for rep in reps if rep["identical"]["P"] is not True]
    p = dict(summarize(q["P"]), moved_bits_on_steps=p_moved[:8])
    out["P"] = p
    # the sweep's carried plans: REPORTED, never ruling; one that moved a bit is a contract finding, not timed in
    cands, findings = [], []
    if p_moved:
        findings.append({"arm": "P", "plan": dict(TODAY), "moved_bits_on_steps": p_moved[:8]})
    for i, c in enumerate(carried):
        arm = f"C{i + 1}"
        moved = [rep["step"] for rep in reps if rep["identical"][arm] is not True]
        if moved:
            findings.append({"arm": arm, "plan": c, "moved_bits_on_steps": moved[:8]})
            continue
        cands.append(dict(summarize(q[arm]), arm=arm, rank=i + 1, plan=c,
                          over_p=statistics.median(m["P"] / m[arm] for m in ms)))
    out["sweep_candidates"] = cands
    out["contract_findings"] = findings
    best = max(cands, key=lambda c: (c["median"], -c["rank"])) if cands else None
    out["sweep_best"] = best
    # REPORTED, never ruling
    t_ms = statistics.median(m["T"] for m in ms)
    out["today_ms_per_step"] = t_ms
    out["p_ms_per_step"] = statistics.median(m["P"] for m in ms)
    out["fidelity_vs_p152"] = t_ms / P152_BOTH_MS
    out["toward_the_floors"] = {"p_share_of_headroom": P152_BOTH_MS * (1 - 1 / p["median"]) / P154_HEADROOM_MS,
                                "p_ms_at_p152_scale": P152_BOTH_MS / p["median"], "byte_floor_ms": BYTE_FLOOR_MS}
    out["models_containing_p"] = [k for k, (lo, hi) in MODELS.items() if lo <= p["median"] <= hi]
    rows = rec["select"]
    sel_ms = {pj: {plan_key(r["plan"]): select_ms(r["step_replays_ms"]) for r in rows[pj] if _eligible(r, n_sel)}
              for pj in ("gu", "dn")}
    out["sweep"] = {pj: sorted(([list(k), v] for k, v in sel_ms[pj].items()), key=lambda kv: kv[1])[:10]
                    for pj in ("gu", "dn")}
    out["errors"] = {pj: [[list(plan_key(r["plan"])), r["error"]] for r in rows[pj] if r.get("error")] for pj in ("gu", "dn")}
    out["identity_failures"] = {pj: [list(plan_key(r["plan"])) for r in rows[pj] if r.get("identical") is False]
                                for pj in ("gu", "dn")}
    out["drift"] = rec.get("drift")
    out["compiled"] = rec.get("compiled")
    out["sweep_best_selection_over_p"] = None          # unread when P was not timed in the sweep (it moved a bit)
    if best and all(plan_key(TODAY) in sel_ms[pj] for pj in ("gu", "dn")):
        b = best["plan"]
        p_sel = sel_ms["gu"][plan_key(TODAY)] + sel_ms["dn"][plan_key(TODAY)]
        out["sweep_best_selection_over_p"] = p_sel / (sel_ms["gu"][plan_key(b["gu"])] + sel_ms["dn"][plan_key(b["dn"])])
    # a shakedown is never graded: its timings are the A2000's, and no A2000 timing grades or seeds anything
    out["predictions"] = {k: "UNREAD (a shakedown)" for k in _grade(out)} if shake else _grade(out)
    if shake:
        out["verdict"], out["why"] = "SHAKEDOWN", ["a shakedown record is never a reading"]
    elif p_moved:
        out["verdict"], out["why"] = "NO_GO", [f"P moved a bit on steps {p_moved[:8]}: K39's bitwise tier fails on "
                                               "sm_120, a contract finding against grouped-nf4-gemm; P's timing is not ruled on"]
    elif p["lb"] >= GO_BAR:
        out["verdict"], out["why"] = "GO", [f"P {p['median']:.4f}x, lower bound {p['lb']:.4f} >= {GO_BAR}"]
    else:
        out["verdict"], out["why"] = "NO_GO", [f"P {p['median']:.4f}x, lower bound {p['lb']:.4f} < {GO_BAR}"]
    return out


def _band(x, lo, hi):
    return "UNREAD" if x is None else ("HELD" if lo <= x <= hi else "MISSED")


def _grade(o):
    b = o.get("sweep_best")
    sel = o.get("identity_failures") or {}
    sop = o.get("sweep_best_selection_over_p")
    curse = sop - b["over_p"] if b and sop is not None else None
    return {
        "fidelity: today's both-GEMM step in 9.5-14.0 ms": _band(o["today_ms_per_step"], *PRED["fidelity_ms"]),
        "the headline: P's confirmed median T/P in 1.05-1.18": _band(o["P"]["median"], *PRED["headline"]),
        "contract: P and every swept plan keep today's bits": "HELD" if not (sel.get("gu") or sel.get("dn")
                                                                              or o["contract_findings"]) else "MISSED",
        "self-pair within 0.005 of 1": _band(abs(o["self_pair"]["median"] - 1), 0.0, PRED["self_pair"]),
        "the sweep's best: its selection over P exceeds its confirmation by 0-0.03": _band(curse, *PRED["shrink"]),
        "registers: T 80 and P 64 a thread, no spills (theoretical occupancy 50 % and 67 %)": _regs(o.get("compiled")),
    }


def _regs(c):
    """HELD when both projections' compiled variants read T's 80 and P's 64 registers, no spills; UNREAD when this
    Triton's cache did not expose them (reported metadata, never a fault)."""
    vs = [(arm, v) for arm in ("T", "P") for pj in ("gu", "dn") for v in (((c or {}).get(arm) or {}).get(pj) or [])]
    if {a for a, _ in vs} != {"T", "P"} or any(not isinstance(v.get("n_regs"), int)
                                               or not isinstance(v.get("n_spills"), int) for _, v in vs):
        return "UNREAD"
    return "HELD" if all(v["n_regs"] == PRED["regs"][a] and v["n_spills"] == 0 for a, v in vs) else "MISSED"


# ------------------------------------------------------------------------------------------------ self-test (mutants)
_BEST = {"block_n": 64, "kc": 64, "warps": 2, "stages": 4}


def _fake(p=1.12, best=1.14, noise=0.002, self_ratio=1.0, today_ms=11.5, layers=LAYERS, shakedown=False, **kw):
    rng = random.Random(7)
    sel = {}
    for proj, base in (("gu", 8.4 / p), ("dn", 3.4 / p)):                 # the sweep is prmt: P's time is the base
        rows = []
        for i, pl in enumerate(GRID):
            t = base * (1.0 if pl == TODAY else 1.05 + 0.01 * i)
            if pl == _BEST:
                t = base * p / (best + 0.02)                  # the sweep's best, a little better than it confirms
            rows.append({"plan": dict(pl), "identical": True, "error": None,
                         "step_replays_ms": [[t * (1 + rng.uniform(-1e-3, 1e-3)) for _ in range(ITERS)]
                                             for _ in SELECT_STEPS]})
        sel[proj] = rows
    carried = select(sel)
    arms = confirm_arms(len(carried))
    reps = []
    for r, s in enumerate(CONFIRM_STEPS):
        f = {"T": 1.0, "T2": 1 / self_ratio, "P": 1 / p, "C1": 1 / best, "C2": 1 / 1.02, "C3": 1 / 1.01}
        reps.append({"step": s, "rotation": rotation(arms, r),
                     "replays_ms": {a: [today_ms * f[a] * (1 + rng.gauss(0, noise)) for _ in range(ITERS)] for a in arms},
                     "identical": {a: True for a in arms if a != "T"}})
    cells = STEPS * layers                                   # S0's total spread over the cells, as integers
    total = S0_DISTINCT_TOTAL if layers == LAYERS else 57 * cells
    flat = [total // cells + (1 if i < total % cells else 0) for i in range(cells)]
    rec = {"device": "NVIDIA GeForce RTX 5090", "cc": [12, 0], "layers": layers, "shakedown": shakedown,
           "rr_sha256": RR_SHA256, "s0_sha256": S0_SHA256, "grid": [dict(pl) for pl in GRID], "today": dict(TODAY),
           "luts": {"T": T_LUT, "P": P_LUT, "sweep": P_LUT}, "order": plan_order(),
           "select_steps": list(SELECT_STEPS), "confirm_steps": list(CONFIRM_STEPS),
           "n_real": [flat[s * layers:(s + 1) * layers] for s in range(STEPS)], "tile_faults": [],
           "select": sel, "carried": carried, "confirm": {"arms": arms, "reps": reps}, "drift": {"gu": 1.0, "dn": 1.0}}
    rec.update(kw)
    return rec


def self_test():
    v = lambda rec: verdict(rec)["verdict"]                  # noqa: E731
    base = _fake()
    o = verdict(base)
    assert o["verdict"] == "GO" and o["sweep_best"]["arm"] == "C1", o["why"]
    assert all(g in ("HELD", "MISSED", "UNREAD") for g in o["predictions"].values()), o["predictions"]
    assert o["predictions"]["the headline: P's confirmed median T/P in 1.05-1.18"] == "HELD"
    assert o["models_containing_p"] == [] and verdict(_fake(p=1.08))["models_containing_p"] == ["M1 overlap"]
    assert all(c["plan"]["gu"] != TODAY and c["plan"]["dn"] != TODAY for c in o["sweep_candidates"])   # P never carried
    assert abs(o["sweep_best"]["over_p"] - 1.14 / 1.12) < 0.01                # the sweep's best, against P
    regs = "registers: T 80 and P 64 a thread, no spills (theoretical occupancy 50 % and 67 %)"
    assert o["predictions"][regs] == "UNREAD"                                # no metadata recorded
    good = {"T": {"gu": [{"n_regs": 80, "n_spills": 0}], "dn": [{"n_regs": 80, "n_spills": 0}]},
            "P": {"gu": [{"n_regs": 64, "n_spills": 0}], "dn": [{"n_regs": 64, "n_spills": 0}]}}
    assert verdict(_fake(compiled=good))["predictions"][regs] == "HELD"
    assert verdict(_fake(compiled=dict(good, P={"gu": [{"n_regs": 72, "n_spills": 0}], "dn": []})))[
        "predictions"][regs] == "MISSED"
    assert v(_fake(p=1.04)) == "NO_GO"                                       # a lift under the bar
    assert v(_fake(p=1.00)) == "NO_GO"                                       # flat
    o = verdict(_fake(p=1.00, best=1.30))
    assert o["verdict"] == "NO_GO" and o["sweep_best"]["median"] > 1.25     # the sweep never rules, however fast
    assert v(_fake(p=1.06, noise=0.08)) == "NO_GO"                           # median over the bar, lower bound under it
    assert v(_fake(p=1.06, noise=0.0005)) == "GO"                            # the same median, tight
    o = verdict(_fake(p=1.0575, noise=0.06))            # GO under the mean's SE, NO_GO under the median's
    pp = o["P"]
    assert o["verdict"] == "NO_GO" and pp["median"] - 2 * pp["se_of_mean"] >= GO_BAR > pp["lb"], pp
    gauss = random.Random(3)
    q = [gauss.gauss(1.0, 0.01) for _ in range(64)]
    assert se_median(q) >= math.sqrt(math.pi / 2) * statistics.stdev(q) / 8      # never below the normal-theory SE
    assert se_median(q[:63] + [50.0]) > 3 * math.sqrt(math.pi / 2) * statistics.stdev(q) / 8   # a wild rep widens it
    assert v(_fake(self_ratio=1.03)) == "VOID"                               # the self-pair is off: no instrument
    bad = _fake()
    bad["confirm"]["reps"][5]["identical"]["T2"] = False
    assert v(bad) == "VOID"                                                  # today's tree is not deterministic
    bad = _fake()
    bad["confirm"]["reps"][7]["identical"]["P"] = False
    o = verdict(bad)
    assert o["verdict"] == "NO_GO" and o["contract_findings"][0]["arm"] == "P"   # a bit moved: no timing ruling
    bad = _fake()
    bad["carried"] = list(reversed(bad["carried"]))
    assert v(bad) == "VOID"                                                  # the box carried something else
    bad = _fake()
    for r in bad["select"]["gu"]:
        if r["plan"] == TODAY:
            r["error"], r["identical"], r["step_replays_ms"] = "OutOfResources: shared memory", None, None
    assert v(bad) == "VOID"                                                  # P did not run in the sweep
    bad = _fake()
    for r in bad["select"]["dn"]:
        if r["plan"] == TODAY:                                               # the bench times only identical plans
            r["identical"], r["step_replays_ms"] = False, None
    o = verdict(bad)
    assert o["verdict"] == "NO_GO" and o["contract_findings"][0]["arm"] == "P"   # P moved a bit at step 0: a finding,
    assert o["P"]["moved_bits_on_steps"] == [IDENTITY_STEP]                      # the held-out rule, never a VOID
    assert o["sweep_best_selection_over_p"] is None and "UNREAD" in o["predictions"].values()
    bad = _fake()
    for r in bad["select"]["gu"]:
        if r["plan"] == TODAY:
            r["step_replays_ms"][0] = r["step_replays_ms"][0][:5]
    assert v(bad) == "VOID"                                                  # P identical but its replays malformed
    assert v(_fake(luts={"T": "tree", "P": "tree", "sweep": "prmt"})) == "VOID"   # P is not the prmt decode
    assert v(_fake(s0_sha256="0" * 64)) == "VOID"                            # not S0's picks
    assert v(_fake(rr_sha256="0" * 64)) == "VOID"                            # not the pinned replay
    assert v(_fake(device="NVIDIA RTX A2000 12GB", cc=[8, 6])) == "VOID"     # an A2000 is never a reading
    assert v(_fake(layers=40)) == "VOID"                                     # not every layer
    bad = _fake()
    bad["confirm"]["reps"].pop()
    assert v(bad) == "VOID"                                                  # a rep short
    bad = _fake()
    bad["confirm"]["reps"][3]["rotation"] = bad["confirm"]["reps"][4]["rotation"]
    assert v(bad) == "VOID"                                                  # the rotation is not as registered
    bad = _fake()
    for rep in bad["confirm"]["reps"][:2]:
        rep["identical"]["C1"] = False
    o = verdict(bad)
    assert o["verdict"] == "GO" and o["sweep_best"]["arm"] == "C2" and o["contract_findings"][0]["arm"] == "C1"
    bad = _fake()
    bad["grid"] = bad["grid"][:-1]
    assert v(bad) == "VOID"                                                  # the grid is not the registered one
    bad = _fake()
    bad["n_real"][0][0] += 1
    assert v(bad) == "VOID"                                                  # not P153's S0 total
    bad = _fake()
    bad["order"] = list(range(len(GRID)))
    assert v(bad) == "VOID"                                                  # not the registered shuffle
    bad = _fake()
    bad["confirm"]["reps"][9]["replays_ms"]["C2"] = bad["confirm"]["reps"][9]["replays_ms"]["C2"][:5]
    assert v(bad) == "VOID"                                                  # malformed replays
    bad = _fake(tile_faults=[{"step": 64, "layer": 3, "fault": "a real tile after a padding slot"}])
    assert v(bad) == "VOID"                                                  # the table is not a prefix
    sk = _fake(shakedown=True, layers=4, device="NVIDIA RTX A2000 12GB", cc=[8, 6])
    o = verdict(sk)
    assert o["verdict"] == "SHAKEDOWN"                                       # never a reading, every check run
    assert set(o["predictions"].values()) == {"UNREAD (a shakedown)"}        # and never graded
    bad = _fake()
    bad["select"]["dn"] = bad["select"]["dn"][:-1]
    assert v(bad) == "VOID"                                                  # a plan missing from the sweep
    print("p156_reduce self-test OK (39 cases)")


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
            print(f"p156_reduce self-test FAILED: {e}")
            return 1
        return 0
    if not (a.bench and a.out):
        ap.error("--bench and --out are required")
    with open(a.bench, encoding="utf-8") as f:
        rec = json.load(f)
    out = verdict(rec)
    with open(a.out, "w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(out, indent=1, sort_keys=True) + "\n")
    print(f"P156 {out['verdict']} " + json.dumps({"why": out["why"], "P": out.get("P"),
                                                  "sweep_best": out.get("sweep_best")}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
