# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""P155's reducer (bench/p155/PREREG-p155.md): K25's plan at Qwen3-30B-A3B's W16 shapes, timed on S0's real router
picks (P153's router replay), selection split from confirmation.

Pure Python, no torch. The registered rules live here, and the box imports them (``p155_bench.py``):
- :func:`select`: the SELECTION rule. Per projection, the plans that ran bit-identical to today's on the identity step,
  ranked by their mean selection-step time (ties by grid order); the top three of each, paired by rank into C1..C3;
- :func:`rotation`: each confirmation rep's capture and replay order;
- :func:`verdict`: GO / NO_GO / VOID, from the CONFIRMATION read only. The selection numbers never enter it.
The reducer re-derives everything from the bench's raw replay times (``step_replays_ms``, ``replays_ms``), including the
selection the box carried forward, which must equal :func:`select` of the record or the read is VOID.

    python p155_reduce.py --bench bench_p155.json --out verdict_p155.json
    python p155_reduce.py --self-test
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
TODAY = {"block_n": 32, "kc": 64, "warps": 4, "stages": 3}  # e4b's _K25_PLAN; lut "tree" and dot_bf16 False are fixed
GRID = [{"block_n": bn, "kc": kc, "warps": w, "stages": s}
        for bn in (16, 32, 64, 128) for kc in (64, 128, 256) for w in (2, 4, 8) for s in (1, 2, 3, 4)]
SELECT_STEPS = tuple(range(0, 64, 8))      # 8 steps of the first half
IDENTITY_STEP = 0                          # every plan's outputs against today's: all layers, both projections
CONFIRM_STEPS = tuple(range(64, 128))      # the 64 held-out steps, one paired rep each
CARRY = 3                                  # carried forward from each projection's ranking
ITERS = 10                                 # timed replays per graph
ORDER_SEED = 155                           # the selection's plan order (one shuffle, both projections)
GO_BAR = 1.15                              # the winner's median ratio minus 2 SE (of the median) must reach this
BOOT_B, BOOT_SEED = 2000, 155              # the median's SE: a bootstrap of 2,000 resamples from one seeded generator
SELF_BAND = (0.99, 1.01)                   # today's self-pair median, else the instrument is VOID
P152_BOTH_MS = 8.395073 + 3.35296          # P152's verdict: the removable time of both K25 lines, 136x48 + 136x64
P154_HEADROOM_MS = 6.8124                  # P154's verdict: that time minus the byte floor
PRED = {"fidelity_ms": (9.5, 14.0), "winner": (1.02, 1.15), "share_max": 0.25, "compaction": (1.00, 1.06),
        "self_pair": 0.005, "shrink": (0.0, 0.03), "regs": 80}


def plan_key(p):
    return (p["block_n"], p["kc"], p["warps"], p["stages"])


_GRID_INDEX = {plan_key(p): i for i, p in enumerate(GRID)}


def plan_order():
    """The selection's timing order: GRID indices, one seeded shuffle (drift spreads over the grid, not by plan)."""
    order = list(range(len(GRID)))
    random.Random(ORDER_SEED).shuffle(order)
    return order


def rotation(arms, r):
    """Rep r's capture and replay order: the arms rotated by r, so each arm takes each position equally often."""
    k = r % len(arms)
    return list(arms[k:]) + list(arms[:k])


def confirm_arms(n_carried):
    return ["T", "T2"] + [f"C{i + 1}" for i in range(n_carried)] + ["K"]


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
    """The registered selection rule (see the module docstring). ``rows``: {"gu": [...], "dn": [...]}, one row per plan."""
    ranked = {}
    for proj in ("gu", "dn"):
        ok = [r for r in rows[proj] if _eligible(r, n_steps)]
        ok.sort(key=lambda r: (select_ms(r["step_replays_ms"]), _GRID_INDEX[plan_key(r["plan"])]))
        ranked[proj] = [dict(r["plan"]) for r in ok]
    n = min(CARRY, len(ranked["gu"]), len(ranked["dn"]))
    return [{"gu": ranked["gu"][i], "dn": ranked["dn"][i]} for i in range(n)]


def rep_ms(rep):
    return {arm: statistics.median(v) for arm, v in rep["replays_ms"].items()}


def se_median(q, b=BOOT_B, seed=BOOT_SEED):
    """The standard error of the MEDIAN of ``q`` (loggetta's review: sd / sqrt(n) is the mean's, about 1.25x too small
    for the median). The larger of two: the normal-theory value, sqrt(pi / 2) x sd / sqrt(n), and the standard
    deviation of the medians of ``b`` resamples with replacement. The bootstrap alone is erratic at n = 64 (0.9-1.9x
    the mean's SE across seeds of the same normal data), so it can only widen the bound, under heavy tails. One seeded
    generator, drawn through ``random()`` alone, so the reducer re-derives the same number on any platform."""
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
    if rec.get("order") != plan_order():
        why.append("the selection's plan order is not the registered shuffle")
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
            why.append(f"{proj}: the selection rows are not one per grid plan")
            continue
        today = [r for r in rows if r["plan"] == TODAY][0]
        if not _eligible(today, len(rec.get("select_steps") or [])):
            why.append(f"{proj}: today's plan did not run cleanly ({today.get('error') or 'no clean replays'})")
    return why


def verdict(rec):
    why = _structure(rec, [])
    shake = bool(rec.get("shakedown"))
    out = {"lane": "P155", "shakedown": shake}
    if why:
        return dict(out, verdict="VOID", why=why)
    n_sel = len(rec["select_steps"])
    carried = select(rec["select"], n_sel)
    if rec.get("carried") != carried:
        return dict(out, verdict="VOID", why=["the carried plans are not the registered selection of the recorded times"])
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
        return dict(out, verdict="VOID", why=["today's plan replayed different bits in a second capture: not deterministic"])
    q = {a: [m["T"] / m[a] for m in ms] for a in arms if a != "T"}
    self_pair = summarize(q["T2"])
    out["self_pair"] = self_pair
    if not SELF_BAND[0] <= self_pair["median"] <= SELF_BAND[1]:
        return dict(out, verdict="VOID", why=[f"self-pair median {self_pair['median']:.4f} outside {SELF_BAND}"])
    # the carried plans: a plan that moved a bit on a held-out step is excluded (a contract finding), never timed in
    cands, findings = [], []
    for i, c in enumerate(carried):
        arm = f"C{i + 1}"
        moved = [rep["step"] for rep in reps if rep["identical"][arm] is not True]
        if moved:
            findings.append({"arm": arm, "plan": c, "moved_bits_on_steps": moved[:8]})
            continue
        cands.append(dict(summarize(q[arm]), arm=arm, rank=i + 1, plan=c))
    out["candidates"] = cands
    out["contract_findings"] = findings
    winner = max(cands, key=lambda c: (c["median"], -c["rank"])) if cands else None
    out["winner"] = winner
    # REPORTED, never ruling
    today_ms = statistics.median(m["T"] for m in ms)
    out["today_ms_per_step"] = today_ms
    out["fidelity_vs_p152"] = today_ms / P152_BOTH_MS
    k_moved = [rep["step"] for rep in reps if rep["identical"]["K"] is not True]
    out["compaction"] = dict(summarize(q["K"]), moved_bits_on_steps=k_moved[:8],
                             share_of_headroom=P152_BOTH_MS * (1 - 1 / statistics.median(q["K"])) / P154_HEADROOM_MS)
    rows = rec["select"]
    sel_ms = {p: {plan_key(r["plan"]): select_ms(r["step_replays_ms"]) for r in rows[p] if _eligible(r, n_sel)}
              for p in ("gu", "dn")}
    out["selection"] = {p: sorted(([list(k), v] for k, v in sel_ms[p].items()), key=lambda kv: kv[1])[:10]
                        for p in ("gu", "dn")}
    out["errors"] = {p: [[list(plan_key(r["plan"])), r["error"]] for r in rows[p] if r.get("error")] for p in ("gu", "dn")}
    out["identity_failures"] = {p: [list(plan_key(r["plan"])) for r in rows[p] if r.get("identical") is False]
                                for p in ("gu", "dn")}
    out["drift"] = rec.get("drift")
    out["today_compiled"] = rec.get("today_compiled")
    if winner:
        w = winner["plan"]
        t_sel = sel_ms["gu"][plan_key(TODAY)] + sel_ms["dn"][plan_key(TODAY)]
        sel_ratio = t_sel / (sel_ms["gu"][plan_key(w["gu"])] + sel_ms["dn"][plan_key(w["dn"])])
        out["winner_selection_ratio"] = sel_ratio
        out["winner_share_of_headroom"] = P152_BOTH_MS * (1 - 1 / winner["median"]) / P154_HEADROOM_MS
    # a shakedown is never graded: its timings are the A2000's, and no A2000 timing grades or seeds anything
    out["predictions"] = {k: "UNREAD (a shakedown)" for k in _grade(out)} if shake else _grade(out)
    if shake:
        out["verdict"], out["why"] = "SHAKEDOWN", ["a shakedown record is never a reading"]
    elif winner is None:
        out["verdict"], out["why"] = "NO_GO", ["no carried plan kept today's bits on the held-out steps"]
    elif winner["lb"] >= GO_BAR:
        out["verdict"], out["why"] = "GO", [f"{winner['arm']} {winner['median']:.4f}x, lower bound {winner['lb']:.4f} >= {GO_BAR}"]
    else:
        out["verdict"], out["why"] = "NO_GO", [f"best {winner['arm']} {winner['median']:.4f}x, lower bound "
                                               f"{winner['lb']:.4f} < {GO_BAR}"]
    return out


def _band(x, lo, hi):
    return "UNREAD" if x is None else ("HELD" if lo <= x <= hi else "MISSED")


def _grade(o):
    w = o.get("winner")
    sel = o.get("identity_failures") or {}
    return {
        "fidelity: today's both-GEMM step in 9.5-14.0 ms": _band(o["today_ms_per_step"], *PRED["fidelity_ms"]),
        "contract: no plan that ran moved a bit": "HELD" if not (sel.get("gu") or sel.get("dn") or o["contract_findings"])
        else "MISSED",
        "winner's confirmed median in 1.02-1.15 (NO_GO)": _band(w and w["median"], *PRED["winner"]),
        "a plan recovers at most 25 % of the 6.81 ms": _band(o.get("winner_share_of_headroom"), -1.0, PRED["share_max"]),
        "compaction's ceiling in 1.00-1.06": _band(o["compaction"]["median"], *PRED["compaction"]),
        "self-pair within 0.005 of 1": _band(abs(o["self_pair"]["median"] - 1), 0.0, PRED["self_pair"]),
        "winner's selection ratio exceeds its confirmation by 0-0.03": _band(
            w and o["winner_selection_ratio"] - w["median"], *PRED["shrink"]),
        "today's plan compiles to 80 registers, no spills (theoretical occupancy 50 %)": _regs(o.get("today_compiled")),
    }


def _regs(tc):
    """HELD when both projections' compiled variants of today's plan read 80 registers and no spills; UNREAD when this
    Triton's cache did not expose them (reported metadata, never a fault)."""
    vs = [v for p in ("gu", "dn") for v in ((tc or {}).get(p) or [])]
    if not vs or any(not isinstance(v.get("n_regs"), int) or not isinstance(v.get("n_spills"), int) for v in vs):
        return "UNREAD"
    return "HELD" if all(v["n_regs"] == PRED["regs"] and v["n_spills"] == 0 for v in vs) else "MISSED"


# ------------------------------------------------------------------------------------------------ self-test (mutants)
def _fake(win=1.20, noise=0.002, self_ratio=1.0, today_ms=11.5, layers=LAYERS, shakedown=False, **kw):
    rng = random.Random(7)
    sel = {}
    for proj, base in (("gu", 8.4), ("dn", 3.4)):
        rows = []
        for i, p in enumerate(GRID):
            t = base * (1.0 if p == TODAY else 1.05 + 0.01 * i)
            if p == {"block_n": 64, "kc": 64, "warps": 2, "stages": 4}:
                t = base / (win + 0.02)                      # the selection winner, a little better than it confirms
            rows.append({"plan": dict(p), "identical": True, "error": None,
                         "step_replays_ms": [[t * (1 + rng.uniform(-1e-3, 1e-3)) for _ in range(ITERS)]
                                             for _ in SELECT_STEPS]})
        sel[proj] = rows
    carried = select(sel)
    arms = confirm_arms(len(carried))
    reps = []
    for r, s in enumerate(CONFIRM_STEPS):
        f = {"T": 1.0, "T2": 1 / self_ratio, "C1": 1 / win, "C2": 1 / 1.02, "C3": 1 / 1.01, "K": 1 / 1.02}
        reps.append({"step": s, "rotation": rotation(arms, r),
                     "replays_ms": {a: [today_ms * f[a] * (1 + rng.gauss(0, noise)) for _ in range(ITERS)] for a in arms},
                     "identical": {a: True for a in arms if a != "T"}})
    cells = STEPS * layers                                   # S0's total spread over the cells, as integers
    total = S0_DISTINCT_TOTAL if layers == LAYERS else 57 * cells
    flat = [total // cells + (1 if i < total % cells else 0) for i in range(cells)]
    rec = {"device": "NVIDIA GeForce RTX 5090", "cc": [12, 0], "layers": layers, "shakedown": shakedown,
           "rr_sha256": RR_SHA256, "s0_sha256": S0_SHA256, "grid": [dict(p) for p in GRID], "today": dict(TODAY),
           "order": plan_order(), "select_steps": list(SELECT_STEPS), "confirm_steps": list(CONFIRM_STEPS),
           "n_real": [flat[s * layers:(s + 1) * layers] for s in range(STEPS)], "tile_faults": [],
           "select": sel, "carried": carried, "confirm": {"arms": arms, "reps": reps}, "drift": {"gu": 1.0, "dn": 1.0}}
    rec.update(kw)
    return rec


def self_test():
    v = lambda rec: verdict(rec)["verdict"]                  # noqa: E731
    base = _fake()
    o = verdict(base)
    assert o["verdict"] == "GO" and o["winner"]["arm"] == "C1", o["why"]
    assert all(g in ("HELD", "MISSED", "UNREAD") for g in o["predictions"].values()), o["predictions"]
    assert o["predictions"]["winner's confirmed median in 1.02-1.15 (NO_GO)"] == "MISSED"
    regs = "today's plan compiles to 80 registers, no spills (theoretical occupancy 50 %)"
    assert o["predictions"][regs] == "UNREAD"                                # no metadata recorded
    assert verdict(_fake(today_compiled={"gu": [{"n_regs": 80, "n_spills": 0}], "dn": [{"n_regs": 80, "n_spills": 0}]}))[
        "predictions"][regs] == "HELD"
    assert verdict(_fake(today_compiled={"gu": [{"n_regs": 96, "n_spills": 0}], "dn": [{"n_regs": 80, "n_spills": 0}]}))[
        "predictions"][regs] == "MISSED"
    assert v(_fake(win=1.10)) == "NO_GO"                                     # a lift under the bar
    assert v(_fake(win=1.00)) == "NO_GO"                                     # flat
    assert v(_fake(win=1.16, noise=0.08)) == "NO_GO"                         # median over the bar, lower bound under it
    assert v(_fake(win=1.16, noise=0.0005)) == "GO"                          # the same median, tight
    o = verdict(_fake(win=1.161, noise=0.08))           # GO under the mean's SE, NO_GO under the median's
    w = o["winner"]
    assert o["verdict"] == "NO_GO" and w["median"] - 2 * w["se_of_mean"] >= GO_BAR > w["lb"], w   # the median's SE decides
    gauss = random.Random(3)
    q = [gauss.gauss(1.0, 0.01) for _ in range(64)]
    assert se_median(q) >= math.sqrt(math.pi / 2) * statistics.stdev(q) / 8      # never below the normal-theory SE
    assert se_median(q[:63] + [50.0]) > 3 * math.sqrt(math.pi / 2) * statistics.stdev(q) / 8   # a wild rep widens it
    assert v(_fake(self_ratio=1.03)) == "VOID"                               # the self-pair is off: no instrument
    bad = _fake()
    bad["confirm"]["reps"][5]["identical"]["T2"] = False
    assert v(bad) == "VOID"                                                  # today's plan is not deterministic
    bad = _fake()
    bad["carried"] = list(reversed(bad["carried"]))
    assert v(bad) == "VOID"                                                  # the box carried something else
    bad = _fake()
    for r in bad["select"]["gu"]:
        if r["plan"] == TODAY:
            r["error"], r["identical"], r["step_replays_ms"] = "OutOfResources: shared memory", None, None
    assert v(bad) == "VOID"                                                  # today's plan did not run
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
    assert o["verdict"] == "NO_GO" and o["winner"]["arm"] == "C2" and o["contract_findings"][0]["arm"] == "C1"
    bad = _fake()
    bad["confirm"]["reps"][0]["identical"]["K"] = False
    o = verdict(bad)
    assert o["verdict"] == "GO" and o["compaction"]["moved_bits_on_steps"] == [CONFIRM_STEPS[0]]   # reported only
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
    assert v(bad) == "VOID"                                                  # the compact table is not a prefix
    sk = _fake(shakedown=True, layers=4, device="NVIDIA RTX A2000 12GB", cc=[8, 6])
    o = verdict(sk)
    assert o["verdict"] == "SHAKEDOWN"                                       # never a reading, every check run
    assert set(o["predictions"].values()) == {"UNREAD (a shakedown)"}        # and never graded
    bad = _fake()
    bad["select"]["dn"] = bad["select"]["dn"][:-1]
    assert v(bad) == "VOID"                                                  # a plan missing from the selection
    print("p155_reduce self-test OK (30 cases)")


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
            print(f"p155_reduce self-test FAILED: {e}")
            return 1
        return 0
    if not (a.bench and a.out):
        ap.error("--bench and --out are required")
    with open(a.bench, encoding="utf-8") as f:
        rec = json.load(f)
    out = verdict(rec)
    with open(a.out, "w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(out, indent=1, sort_keys=True) + "\n")
    print(f"P155 {out['verdict']} " + json.dumps({"why": out["why"], "winner": out.get("winner"),
                                                  "compaction": out.get("compaction", {}).get("median")}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
