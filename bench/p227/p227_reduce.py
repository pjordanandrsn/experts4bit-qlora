# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""P227's reducer (bench/p227/PREREG-p227.md): the serving roadmap's lever B in a microbench. At 2 to 8 decode
tokens, does e4b's per-row route (``E4B_NF4_SMALLM_BW=1``: one group per (token, slot) row, so grouped-nf4-gemm's K33
bandwidth GEMV) run the MoE expert call faster than the served route (the device tile table and K25), on two families,
each ruled on its own?

Pure Python, no torch. The registered rules live here, and the box imports them (``p227_bench.py``):
- **The families:** Qwen3-30B-A3B's NF4 expert shapes (128 experts, top-8, 48 layers, routing from P153's router
  replay, stream S1) and Mixtral-8x7B's (8 experts, top-2, 8 layers, P222's seeded block-B draw). The routing tables are
  pinned by sha256 here.
- **The arms**, per family and token count T in 2, 4, 8: **K** (the served route: device grouping, K25 at the family's
  served decode and plan; the CONTROL), **K2** (K captured again: the self-pair) and **B** (``E4B_NF4_SMALLM_BW=1``,
  resolved). 60 reps a cell (10 Williams squares of six orders), each arm's median of 10 timed replays, every timed
  replay right after an untimed replay of the spacer S (K's work).
- :func:`verdict`: per family PASS / MISS / TOLERANCE_FAIL / VOID, and B2's trigger. A family is never ruled on the
  other's numbers.

    python p227_reduce.py --bench bench_p227.json --out verdict_p227.json
    python p227_reduce.py --self-test
"""
from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import sys

LANE = "P227"
FAMILIES = {
    # (N, K) of gate_up and down; the K arm's decode and plan are each family's served default today
    "qwen3": {"E": 128, "topk": 8, "layers": 48, "shapes": [1536, 2048, 2048, 768],
              "k25": {"E4B_K25_LUT": "prmt", "E4B_K25_PLAN": "c1"}},
    "mixtral": {"E": 8, "topk": 2, "layers": 8, "shapes": [28672, 4096, 4096, 14336],
                "k25": {"E4B_K25_LUT": "tree", "E4B_K25_PLAN": "today"}},
}
FAMILY_ORDER = ("qwen3", "mixtral")
TS = (2, 4, 8)
RULED_TS = (2, 4)                            # the bar is read at these; T = 8 carries B2's trigger
STEPS = tuple(range(60))                     # 60 reps a cell: 10 whole Williams squares of six orders
ARMS = ["K", "K2", "B"]
SPACER = "S"                                 # K's calls with their own outputs, replayed untimed before every timed replay
ITERS = 10                                   # timed replays per arm per rep
BAR = 0.70                                   # B/K's upper bound (median + 2 SE), raw AND clock-adjusted, at T = 2 and 4
B2_TRIGGER = 1.25                            # B's median over the byte floor at T = 8 above this: B2 (a multi-row GEMV)
SELF_BAND = (0.99, 1.01)                     # K/K2's median, else the family's reading is VOID
TOL = 1.5e-2                                 # B against K: max |B - K| / max |K| per rep (the served route's tolerance)
BOOT_B, BOOT_SEED = 2000, 227
READ_VERDICTS = ("PASS", "MISS", "TOLERANCE_FAIL")   # a family read: final, never re-read; TOLERANCE_FAIL is deterministic
CLOCK_TOL, MEM_CLOCK_TOL = 0.02, 0.01        # B's median SM / memory clock off K's by more: labelled
MIN_POWER_W = 575                            # every NVML sample's enforced limit, else VOID
REPLAY_SHA256 = "3c59e72f947daa425e61f727401204099a1e782b902fb39687e04c46e338b14f"   # bench/p153/router-replay/rr.json.pt
QWEN3_ROWS0 = 16                             # P153's stream S1: rows 16..31; a cell of T tokens reads rows 16..16+T-1
MIXTRAL_SEED = 2221                          # P222's block-B draw (p222_reduce.picks): a cell reads its rows 0..T-1
# route_sha256 of each cell's table ([layer][step][row][k] int8 bytes, steps 0..59) and its distinct-expert total over
# (layer, step), read at registration from the replay and from the seed, before any box
ROUTE_SHA256 = {
    "qwen3": {2: "06babf485a2dc05aefbee86d7815e4eadac8895233f24017ec7b65a20aea9f0e",
              4: "ba658e969c7cfe330efdac6162965fb388506394a37021d304f2fa5b684131a0",
              8: "a0d1f7da755255c702656114f0d039fa6274f17e82286031f3968881113a54b8"},
    "mixtral": {2: "0b2080527f0de53764ee0b1c84b44c2aa31d78324d379cda2093dd684463c4a9",
                4: "0458b77bd94d909ff235ca425349a503972ff2497d2ac92f9b734a03f75c143e",
                8: "160a98e8cd2614a8a6ebbb0614e14a275c766ee481059223b2f15f69d53e0b56"},
}
DISTINCT_TOTAL = {"qwen3": {2: 43441, 4: 76171, 8: 113745}, "mixtral": {2: 1696, 4: 2640, 8: 3470}}


def expert_bytes(shapes):
    """One expert's NF4 bytes and fp32 absmax over both projections."""
    n1, k1, n2, k2 = shapes
    return n1 * k1 // 2 + n1 * (k1 // 64) * 4 + n2 * k2 // 2 + n2 * (k2 // 64) * 4


def mixtral_picks(steps=128, layers=8, tokens=16, seed=MIXTRAL_SEED):
    """P222's ``picks`` (bench/p222/p222_reduce.py), byte for byte: [layer][step][row] -> two distinct of eight."""
    rng = random.Random(seed)
    return [[[tuple(rng.sample(range(8), 2)) for _ in range(tokens)] for _ in range(steps)] for _ in range(layers)]


def route_sha256(tab):
    import hashlib
    return hashlib.sha256(bytes(e for layer in tab for step in layer for row in step for e in row)).hexdigest()


def williams(n):
    """Williams' design (P222's): for an odd n, the n orders and their mirrors, every ordered pair adjacent twice."""
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
    """P155's rule: the larger of sqrt(pi / 2) sd / sqrt(n) and a seeded bootstrap's SE of the median."""
    n = len(q)
    normal = math.sqrt(math.pi / 2) * statistics.stdev(q) / math.sqrt(n)
    rng = random.Random(seed)
    boot = statistics.stdev(statistics.median([q[int(rng.random() * n)] for _ in range(n)]) for _ in range(b))
    return max(normal, boot)


def summarize(q):
    med = statistics.median(q)
    se = se_median(q) if len(q) > 1 else math.inf
    return {"n": len(q), "median": med, "se": se, "ub": med + 2 * se, "lb": med - 2 * se, "min": min(q), "max": max(q)}


def _num(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x) and x > 0


def _good(xs, n=ITERS):
    return isinstance(xs, list) and len(xs) == n and all(_num(x) for x in xs)


def clock_tag(fam, T, arm):
    return f"{fam}|{T}|{arm}"


def _global(rec, why):
    shake = bool(rec.get("shakedown"))
    if rec.get("lane") != LANE or rec.get("arms") != ARMS or rec.get("spacer") != SPACER or rec.get("iters") != ITERS:
        why.append("the lane, the arms, the spacer or the timed replays an arm are not the registered ones")
    if not shake and ("5090" not in str(rec.get("device", "")) or rec.get("cc") != [12, 0]):
        why.append(f"device {rec.get('device')} cc {rec.get('cc')}: the lane registers an RTX 5090 (sm_120)")
    if not _num((rec.get("copy") or {}).get("gbs")):
        why.append("the copy rate (the floor's divisor) is unread")
    if not shake and rec.get("gnf4_gemv_bw_env") is not None:
        why.append(f"GNF4_GEMV_BW={rec['gnf4_gemv_bw_env']!r} in a reading: K33's gate is forced only in a shakedown")
    if rec.get("poisoned"):
        why.append(f"the CUDA context was poisoned: {rec['poisoned']}")
    if rec.get("arm_fault"):
        why.append(f"an arm could not be resolved, launched or captured: {rec['arm_fault']}")
    clk = rec.get("clocks") or {}
    if clk.get("error"):
        why.append(f"the clocks are unread ({clk['error']})")
    elif not shake:
        lim = [w for k, v in clk.items() if isinstance(v, dict) for w in v.get("enforced_limit_w") or [None]]
        if not lim or any(not _num(w) or w < MIN_POWER_W for w in lim):
            why.append(f"an enforced power limit {sorted({str(w) for w in lim})} W under {MIN_POWER_W} W "
                       "during a timed replay")
    return why


def _cell(rec, fam, T, why):
    """Every VOID that is a property of one (family, T) cell's record; appends to ``why``."""
    shake = bool(rec.get("shakedown"))
    f = (rec.get("families") or {}).get(fam) or {}
    spec = FAMILIES[fam]
    cell = (f.get("cells") or {}).get(str(T)) or {}
    if not cell:
        why.append(f"T={T}: no cell recorded")
        return why
    L = f.get("layers")
    if not shake and (L != spec["layers"] or f.get("shapes") != spec["shapes"] or f.get("k25") != spec["k25"]):
        why.append("the layers, the shapes or the K arm's decode and plan are not the registered ones")
    if not (f.get("resolved") or {}).get("on"):
        why.append("E4B_NF4_SMALLM_BW did not resolve on for this family: B is not the knob's route")
    steps = cell.get("steps") or []
    if (steps != list(STEPS)) if not shake else not (steps and steps == list(STEPS)[:len(steps)]):
        why.append(f"T={T}: the steps are not the registered ones")
    if not shake and cell.get("route_sha256") != ROUTE_SHA256[fam][T]:
        why.append(f"T={T}: the routing table does not hash to the pinned draw")
    dist = cell.get("distinct")
    if not (isinstance(dist, list) and len(dist) == len(steps) and isinstance(L, int)
            and all(isinstance(x, list) and len(x) == L and all(isinstance(v, int) and 1 <= v <= min(spec["E"], T * spec["topk"])
                                                                for v in x) for x in dist)):
        why.append(f"T={T}: the distinct-expert counts are not [steps][layers] of integers in range")
    elif not shake and sum(map(sum, dist)) != DISTINCT_TOTAL[fam][T]:
        why.append(f"T={T}: the distinct-expert total is not the pinned table's")
    eng = cell.get("engaged") or {}
    b, k = eng.get("B") or {}, eng.get("K") or {}
    if (b.get("bw_prmt32") != 2 * (L or 0) or any(b.get(x) for x in ("dotpad", "dotpad_splitk", "scalar", "scalar_splitk"))
            or b.get("route") != {"nf4_singleton|le256": L}):
        why.append(f"T={T}: B's eager pass did not run K33 on both projections of every layer, and only K33 ({b})")
    if any(k.get(x) for x in ("bw_prmt32", "bw_tree")) or k.get("route") != {"nf4_k25|le256": L}:
        why.append(f"T={T}: K's eager pass did not run K25 on every layer, and only K25 ({k})")
    lut, plan = spec["k25"]["E4B_K25_LUT"], spec["k25"]["E4B_K25_PLAN"]          # the REGISTERED decode and plan
    if k.get("k25_lut") != {f"{lut}|le256": L} or k.get("k25_plan") != {f"{plan}|le256": L}:
        why.append(f"T={T}: e4b's own tallies say K ran decode {k.get('k25_lut')} and plan {k.get('k25_plan')}, "
                   f"not {lut} and {plan} on every layer")
    if b.get("k25_lut") or b.get("k25_plan"):
        why.append(f"T={T}: B's eager pass reached K25 ({b.get('k25_lut')}, {b.get('k25_plan')})")
    reps = cell.get("reps") or []
    if len(reps) != len(steps) or not steps:
        why.append(f"T={T}: {len(reps)} reps for {len(steps)} steps")
        return why
    for r, rep in enumerate(reps):
        if rep.get("step") != steps[r] or rep.get("order") != arm_order(r):
            why.append(f"T={T} rep {r}: its step or its order is not the registered Williams row")
            break
        rp = rep.get("replays_ms") or {}
        if set(rp) != set(ARMS) or not all(_good(rp[a]) for a in ARMS):
            why.append(f"T={T} rep {r}: replays missing or malformed")
            break
        if rep.get("spacer_replays") != ITERS * len(ARMS):
            why.append(f"T={T} rep {r}: {rep.get('spacer_replays')} spacer replays, not one before each timed replay")
            break
        if rep.get("k2_identical") is not True:
            why.append(f"T={T} rep {r}: K replayed different bits in a second capture: not deterministic")
            break
        if not isinstance(rep.get("b_rel_diff"), (int, float)) or not math.isfinite(rep["b_rel_diff"]):
            why.append(f"T={T} rep {r}: B against K unread")
            break
    clk = rec.get("clocks") or {}
    if not clk.get("error") and not all(isinstance((clk.get(clock_tag(fam, T, a)) or {}).get("sm_mhz_median"),
                                                   (int, float)) for a in ARMS):
        why.append(f"T={T}: an arm has no clock samples")
    return why


def read_cell(rec, fam, T):
    f = rec["families"][fam]
    cell = f["cells"][str(T)]
    eb = expert_bytes(f["shapes"])
    gbs = rec["copy"]["gbs"]
    ms = [{a: statistics.median(v) for a, v in rep["replays_ms"].items()} for rep in cell["reps"]]
    floor = [sum(cell["distinct"][i]) * eb / (gbs * 1e9) * 1e3 for i in range(len(ms))]   # K33's floor: bytes / copy
    out = {"ms": {a: statistics.median(m[a] for m in ms) for a in ARMS},
           "self_pair": summarize([m["K"] / m["K2"] for m in ms]),
           "b_over_k": summarize([m["B"] / m["K"] for m in ms]),
           "b_over_floor": statistics.median(m["B"] / fl for m, fl in zip(ms, floor)),
           "b_over_floor_summary": summarize([m["B"] / fl for m, fl in zip(ms, floor)]),   # reported: the spread
           "k_over_floor": statistics.median(m["K"] / fl for m, fl in zip(ms, floor)),
           "floor_ms": statistics.median(floor),
           "distinct_per_layer": sum(map(sum, cell["distinct"])) / (len(ms) * f["layers"]),
           "b_rel_diff_max": max(rep["b_rel_diff"] for rep in cell["reps"])}
    clk = rec["clocks"]
    kc, bc = clk[clock_tag(fam, T, "K")], clk[clock_tag(fam, T, "B")]
    ratio = bc["sm_mhz_median"] / kc["sm_mhz_median"]
    s = out["b_over_k"]
    s["clock_b_over_k"] = ratio
    s["ub_clock_adj"] = s["ub"] * max(1.0, ratio)          # a faster-clocked B gets no credit: its ratio scaled up 1:1
    s["labels"] = [f"clock-confounded: SM {bc['sm_mhz_median']:.0f} MHz against K's {kc['sm_mhz_median']:.0f}"] \
        if abs(ratio - 1) > CLOCK_TOL else []
    if _num(bc.get("mem_mhz_median")) and _num(kc.get("mem_mhz_median")):
        mr = bc["mem_mhz_median"] / kc["mem_mhz_median"]
        s["mem_clock_b_over_k"] = mr
        if abs(mr - 1) > MEM_CLOCK_TOL:
            s["labels"].append(f"clock-confounded: memory {bc['mem_mhz_median']:.0f} MHz against K's {kc['mem_mhz_median']:.0f}")
    return out


def family_verdict(rec, fam):
    shake = bool(rec.get("shakedown"))
    why = []
    for T in TS:
        _cell(rec, fam, T, why)
    if why:
        return {"verdict": "VOID", "why": why}
    cells = {T: read_cell(rec, fam, T) for T in TS}
    out = {"cells": {str(T): c for T, c in cells.items()}}
    outside = [T for T, c in cells.items() if not SELF_BAND[0] <= c["self_pair"]["median"] <= SELF_BAND[1]]
    s8 = cells[8]["b_over_floor_summary"]
    out["b2_trigger"] = {"b_over_floor_t8": cells[8]["b_over_floor"], "threshold": B2_TRIGGER,
                         "fires": cells[8]["b_over_floor"] > B2_TRIGGER,
                         "reported_se": s8["se"], "reported_lb": s8["lb"], "reported_ub": s8["ub"]}   # never ruling
    if outside and not shake:
        return dict(out, verdict="VOID", why=[f"self-pair K/K2 median outside {SELF_BAND} at T = {outside}"],
                    b2_trigger=None)
    bad = [T for T, c in cells.items() if c["b_rel_diff_max"] > TOL]
    if shake:
        return dict(out, verdict="SHAKEDOWN", why=["a shakedown record is never a reading"]
                    + ([f"self-pair outside {SELF_BAND} at T = {outside}: reported"] if outside else []))
    if bad:
        return dict(out, verdict="TOLERANCE_FAIL", b2_trigger=None,
                    why=[f"B against K beyond {TOL} at T = {bad}: a contract finding; no timing is ruled on"])
    ok = all(cells[T]["b_over_k"]["ub"] <= BAR and cells[T]["b_over_k"]["ub_clock_adj"] <= BAR for T in RULED_TS)
    why = [f"T={T}: B/K {cells[T]['b_over_k']['median']:.4f}, upper bound {cells[T]['b_over_k']['ub']:.4f}, "
           f"clock-adjusted {cells[T]['b_over_k']['ub_clock_adj']:.4f}" for T in RULED_TS]
    return dict(out, verdict="PASS" if ok else "MISS",
                why=why + [f"{'both' if ok else 'not both'} <= {BAR} at T = {list(RULED_TS)}"])


def verdict(rec):
    why = _global(rec, [])
    out = {"lane": LANE, "shakedown": bool(rec.get("shakedown"))}
    if why:
        return dict(out, verdict="VOID", why=why, families={})
    fams = {fam: family_verdict(rec, fam) for fam in FAMILY_ORDER}
    for fv in fams.values():                      # READ: the family's outcome is final on this run (never re-read)
        fv["read"] = fv["verdict"] in READ_VERDICTS
    out["families"] = fams
    cs = [2 * 2 * 1024 ** 3 / x / 1e9 for x in rec["copy"].get("samples_s") or [] if _num(x)]
    out["copy_rate_gbs"] = {"median": rec["copy"]["gbs"], "min": min(cs) if cs else None,
                            "max": max(cs) if cs else None}                      # reported: the floor's divisor
    out["verdict"] = " ".join(f"{fam}={fams[fam]['verdict']}" for fam in FAMILY_ORDER)
    out["predictions"] = {} if rec.get("shakedown") else _grade(fams)
    return out


#: graded in the read, never ruling (PREREG, "Predictions"): the family verdicts and B2's trigger
PRED = {"mixtral": "PASS", "qwen3": "MISS", "b2_mixtral": True}


def _grade(fams):
    g = {}
    for fam in ("mixtral", "qwen3"):
        v = fams[fam]["verdict"]
        g[f"{fam}_verdict"] = "UNREAD" if v in ("VOID", "TOLERANCE_FAIL") else ("HELD" if v == PRED[fam] else "MISSED")
    b2 = (fams["mixtral"].get("b2_trigger") or {}).get("fires")
    g["b2_fires_on_mixtral"] = "UNREAD" if b2 is None else ("HELD" if b2 is PRED["b2_mixtral"] else "MISSED")
    return g


# ---------------------------------------------------------------------------------------------------- self-test
def _fake(gains=None, noise=0.002, k_ms=None, shake=False, n=len(STEPS)):
    """A synthetic record: per family, B runs at ``gains[fam][T]`` of K's time; seeded noise per replay."""
    gains = gains or {"qwen3": {2: 0.80, 4: 0.90, 8: 1.00}, "mixtral": {2: 0.40, 4: 0.50, 8: 0.80}}
    k_ms = k_ms or {"qwen3": {2: 2.3, 4: 3.4, 8: 5.4}, "mixtral": {2: 5.8, 4: 8.0, 8: 10.3}}
    rng = random.Random(9)
    rec = {"lane": LANE, "shakedown": shake, "device": "NVIDIA GeForce RTX 5090", "cc": [12, 0], "arms": list(ARMS),
           "spacer": SPACER, "iters": ITERS, "copy": {"gbs": 1500.0}, "families": {}, "clocks": {"error": None}}
    for fam in FAMILY_ORDER:
        spec = FAMILIES[fam]
        L = spec["layers"]
        cells = {}
        for T in TS:
            base, rem = divmod(DISTINCT_TOTAL[fam][T], len(STEPS) * L)      # the pinned total over the full table
            dist = [[base + (1 if (i * L + li) < rem else 0) for li in range(L)] for i in range(len(STEPS))][:n]
            reps = []
            for r in range(n):
                k = k_ms[fam][T]
                rp = {a: [k * (gains[fam][T] if a == "B" else 1.0) * (1 + rng.gauss(0, noise)) for _ in range(ITERS)]
                      for a in ARMS}
                reps.append({"step": r, "order": arm_order(r), "replays_ms": rp, "spacer_replays": ITERS * len(ARMS),
                             "k2_identical": True, "b_rel_diff": 0.006})
            cells[str(T)] = {"steps": list(range(n)), "route_sha256": ROUTE_SHA256[fam][T], "distinct": dist, "reps": reps,
                             "engaged": {"B": {"bw_prmt32": 2 * L, "route": {"nf4_singleton|le256": L}},
                                         "K": {"bw_prmt32": 0, "route": {"nf4_k25|le256": L},
                                               "k25_lut": {f"{spec['k25']['E4B_K25_LUT']}|le256": L},
                                               "k25_plan": {f"{spec['k25']['E4B_K25_PLAN']}|le256": L}}}}
            for a in ARMS:
                rec["clocks"][clock_tag(fam, T, a)] = {"sm_mhz_median": 2707.0, "mem_mhz_median": 13801.0,
                                                        "enforced_limit_w": [575.0]}
        rec["families"][fam] = {"layers": L, "shapes": list(spec["shapes"]), "k25": dict(spec["k25"]),
                                "resolved": {"on": True}, "cells": cells}
    return rec


def self_test():
    import copy
    n = 0

    def check(cond, msg):
        nonlocal n
        assert cond, msg
        n += 1

    # the pinned tables: Mixtral's re-derives here from P222's seed; Qwen3's is pinned from the replay (no torch here)
    mp = mixtral_picks()
    for T in TS:
        tab = [[[list(row) for row in mp[li][s][:T]] for s in STEPS] for li in range(8)]
        check(route_sha256(tab) == ROUTE_SHA256["mixtral"][T], f"mixtral T={T} table")
        check(sum(len({e for row in tab[li][s] for e in row}) for li in range(8) for s in STEPS)
              == DISTINCT_TOTAL["mixtral"][T], f"mixtral T={T} distinct")
    check(williams(3) == [[0, 1, 2], [1, 2, 0], [2, 0, 1], [2, 1, 0], [0, 2, 1], [1, 0, 2]], "williams(3)")
    check(len(STEPS) % 6 == 0 and arm_order(6) == arm_order(0), "whole squares")
    pairs = {}
    for r in range(6):
        o = arm_order(r)
        for a, b in zip(o, o[1:]):
            pairs[(a, b)] = pairs.get((a, b), 0) + 1
    check(set(pairs.values()) == {2} and len(pairs) == 6, "every ordered pair adjacent twice")
    check(expert_bytes(FAMILIES["qwen3"]["shapes"]) == 1536 * 1024 + 1536 * 32 * 4 + 2048 * 384 + 2048 * 12 * 4, "bytes")
    v = verdict(_fake())
    check(v["verdict"] == "qwen3=MISS mixtral=PASS", v["verdict"])
    check(v["families"]["mixtral"]["b2_trigger"]["fires"] in (True, False), "b2 read")
    check(v["predictions"] == {"mixtral_verdict": "HELD", "qwen3_verdict": "HELD",
                               "b2_fires_on_mixtral": "HELD" if v["families"]["mixtral"]["b2_trigger"]["fires"] else "MISSED"},
          v["predictions"])
    # the bar reads the UPPER bound at BOTH ruled T: a pass at T = 2 alone is a MISS
    v = verdict(_fake(gains={"qwen3": {2: 0.60, 4: 0.75, 8: 1.0}, "mixtral": {2: 0.40, 4: 0.50, 8: 0.8}}))
    check(v["families"]["qwen3"]["verdict"] == "MISS", "one T passing is not a PASS")
    v = verdict(_fake(gains={"qwen3": {2: 0.697, 4: 0.697, 8: 1.0}, "mixtral": {2: 0.40, 4: 0.50, 8: 0.8}}, noise=0.03))
    c2 = v["families"]["qwen3"]["cells"]["2"]["b_over_k"]
    check(c2["median"] < BAR < c2["ub"], f"the premise: median {c2['median']:.4f} under, ub {c2['ub']:.4f} over")
    check(v["families"]["qwen3"]["verdict"] == "MISS", "a median under the bar with an upper bound over it is a MISS")
    # T = 8 never rules the bar
    v = verdict(_fake(gains={"qwen3": {2: 0.6, 4: 0.6, 8: 1.5}, "mixtral": {2: 0.4, 4: 0.5, 8: 1.5}}))
    check(v["families"]["qwen3"]["verdict"] == "PASS", "T=8 does not rule the bar")
    # the clock adjustment: a faster-clocked B is never credited
    r = _fake(gains={"qwen3": {2: 0.68, 4: 0.68, 8: 1.0}, "mixtral": {2: 0.4, 4: 0.5, 8: 0.8}}, noise=0.0005)
    check(verdict(r)["families"]["qwen3"]["verdict"] == "PASS", "the near-bar pass")
    for T in RULED_TS:
        r["clocks"][clock_tag("qwen3", T, "B")]["sm_mhz_median"] = 2707.0 * 1.04
    vq = verdict(r)["families"]["qwen3"]
    check(vq["verdict"] == "MISS" and vq["cells"]["2"]["b_over_k"]["labels"], "clock-adjusted MISS, labelled")
    # each family is ruled on its own: a Qwen3 VOID leaves Mixtral's verdict standing
    r = _fake()
    r["families"]["qwen3"]["cells"]["4"]["reps"][3]["k2_identical"] = False
    v = verdict(r)
    check(v["families"]["qwen3"]["verdict"] == "VOID" and v["families"]["mixtral"]["verdict"] == "PASS", "per family")
    # VOIDs: structure
    for mutate, msg in [
        (lambda q: q.update(device="NVIDIA RTX A2000"), "device"),
        (lambda q: q["copy"].update(gbs=None), "copy"),
        (lambda q: q["clocks"].update(error="nvml"), "clocks"),
        (lambda q: q["clocks"][clock_tag("mixtral", 2, "K")].update(enforced_limit_w=[400.0]), "power"),
        (lambda q: q.update(gnf4_gemv_bw_env="1"), "a forced K33 gate in a reading"),
    ]:
        q = copy.deepcopy(_fake())
        mutate(q)
        check(verdict(q)["verdict"] == "VOID", msg)
    for mutate, msg in [
        (lambda q: q["families"]["qwen3"]["cells"]["2"].update(route_sha256="0" * 64), "route"),
        (lambda q: q["families"]["qwen3"]["cells"]["2"]["distinct"][0].__setitem__(0, 1), "distinct total"),
        (lambda q: q["families"]["qwen3"]["cells"]["4"]["reps"][1].update(order=["K", "K2", "B"][::-1]), "order"),
        (lambda q: q["families"]["qwen3"]["cells"]["4"]["reps"][1].update(spacer_replays=29), "spacer"),
        (lambda q: q["families"]["qwen3"]["cells"]["8"]["engaged"]["B"].update(dotpad=1), "B engagement"),
        (lambda q: q["families"]["qwen3"]["cells"]["8"]["engaged"]["B"].update(bw_prmt32=95), "B launches"),
        (lambda q: q["families"]["qwen3"]["cells"]["2"]["engaged"]["K"].update(bw_prmt32=1), "K engagement"),
        (lambda q: q["families"]["qwen3"].update(resolved={"on": False}), "resolved"),
        (lambda q: q["families"]["qwen3"]["cells"]["2"]["engaged"]["K"].update(k25_plan={"today|le256": 48}), "K plan"),
        (lambda q: q["families"]["qwen3"]["cells"]["4"]["engaged"]["K"].update(k25_lut={"tree|le256": 48}), "K lut"),
        (lambda q: q["families"]["qwen3"]["cells"]["8"]["engaged"]["B"].update(k25_plan={"c1|le256": 48}), "B on K25"),
        (lambda q: q["families"]["qwen3"].update(k25={"E4B_K25_LUT": "tree", "E4B_K25_PLAN": "today"}), "K plan"),
        (lambda q: q["families"]["qwen3"]["cells"]["2"]["reps"].pop(), "rep count"),
    ]:
        q = copy.deepcopy(_fake())
        mutate(q)
        fv = verdict(q)["families"]
        check(fv["qwen3"]["verdict"] == "VOID" and fv["mixtral"]["verdict"] != "VOID", msg)
    # the self-pair band rules a reading
    q = _fake()
    for rep in q["families"]["mixtral"]["cells"]["4"]["reps"]:
        rep["replays_ms"]["K2"] = [x * 1.03 for x in rep["replays_ms"]["K2"]]
    check(verdict(q)["families"]["mixtral"]["verdict"] == "VOID", "self-pair band")
    # B beyond the tolerance is a contract finding, no timing ruled
    q = _fake()
    q["families"]["mixtral"]["cells"]["2"]["reps"][5]["b_rel_diff"] = 0.05
    fv = verdict(q)["families"]["mixtral"]
    check(fv["verdict"] == "TOLERANCE_FAIL" and fv["b2_trigger"] is None and fv["read"] is True, "tolerance, read")
    # a TOLERANCE_FAIL beside a VOID: one family read, final; the other unread
    q["families"]["qwen3"]["cells"]["4"]["reps"][2]["k2_identical"] = False
    v = verdict(q)
    check((v["families"]["mixtral"]["read"], v["families"]["qwen3"]["read"]) == (True, False), "read flags")
    # B2's trigger: the median of B over the floor at T = 8, strictly above 1.25
    q = _fake()
    c = read_cell(q, "mixtral", 8)
    want = c["b_over_floor"] > B2_TRIGGER
    check(verdict(q)["families"]["mixtral"]["b2_trigger"]["fires"] is want, "b2 rule")
    # a shakedown is never a reading, and reads fewer steps and layers
    q = _fake(shake=True, n=6)
    for fam in FAMILY_ORDER:
        q["families"][fam]["layers"] = 2
        for T in TS:
            cell = q["families"][fam]["cells"][str(T)]
            cell["distinct"] = [row[:2] for row in cell["distinct"]]
            cell["engaged"]["B"]["bw_prmt32"] = 4
            cell["engaged"]["B"]["route"] = {"nf4_singleton|le256": 2}
            cell["engaged"]["K"]["route"] = {"nf4_k25|le256": 2}
            k25 = FAMILIES[fam]["k25"]
            cell["engaged"]["K"]["k25_lut"] = {f"{k25['E4B_K25_LUT']}|le256": 2}
            cell["engaged"]["K"]["k25_plan"] = {f"{k25['E4B_K25_PLAN']}|le256": 2}
    q["device"] = "NVIDIA RTX A4000"
    v = verdict(q)
    check(v["verdict"] == "qwen3=SHAKEDOWN mixtral=SHAKEDOWN" and v["predictions"] == {}, v["verdict"])
    # the reported spreads: B2's SE and the copy rate's
    v = verdict(_fake())
    b2 = v["families"]["mixtral"]["b2_trigger"]
    check(b2["reported_se"] > 0 and b2["reported_lb"] < b2["b_over_floor_t8"] < b2["reported_ub"], "b2 spread")
    q = _fake()
    q["copy"]["samples_s"] = [2 * 2 * 1024 ** 3 / (g * 1e9) for g in (1490.0, 1500.0, 1510.0)]
    c = verdict(q)["copy_rate_gbs"]
    check(abs(c["min"] - 1490.0) < 1e-6 and abs(c["max"] - 1510.0) < 1e-6, "copy spread")
    print(f"p227_reduce self-test OK ({n} checks)")


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
            print(f"p227_reduce self-test FAILED: {e}")
            return 1
        return 0
    try:
        with open(a.bench, encoding="utf-8") as f:
            rec = json.load(f)
    except (OSError, ValueError) as e:            # a killed or never-written bench is a VOID record, never a crash
        rec = None
        v = {"lane": LANE, "verdict": "VOID", "families": {},
             "why": [f"the bench record is missing or unreadable ({type(e).__name__}: {str(e)[:120]})"]}
    if rec is not None:
        v = verdict(rec)
    with open(a.out, "w", encoding="utf-8", newline="\n") as f:
        json.dump(v, f, indent=1, sort_keys=True)
    print(f"P227_VERDICT {v['verdict']}")
    for fam, fv in (v.get("families") or {}).items():
        print(f"  {fam}: {fv['verdict']} -- {'; '.join(fv.get('why') or [])}")
    if not v.get("families"):
        print("  " + "; ".join(v.get("why") or []))
    return 0


if __name__ == "__main__":
    sys.exit(main())
