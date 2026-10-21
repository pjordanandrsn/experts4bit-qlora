# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""P222's reducer (bench/p222/PREREG-p222.md): K25's decode x plan at Mixtral-8x7B's W16 shapes, where the served
default is still the select tree at today's plan (no read licenses ``mixtral``: e4b's K25 default families list only
``qwen3_moe``).

Pure Python, no torch. The registered rules live here, and the box imports them (``p222_bench.py``):
- **The routing:** seeded synthetic top-2 picks (no Mixtral router replay exists). :func:`picks` draws, for each
  (layer, step, row), two distinct experts of eight uniformly from ``random.Random(seed)``; block A (the selection) is
  seed 2220, block B (the reading) seed 2221. Both tables are pinned by sha256 here.
- **Block A, the SELECTION (reported, never ruling):** every plan of P156's 144-plan grid under ``lut="prmt"``, per
  projection, timed on 8 steps of A. :func:`select` names M, each projection's fastest plan. Today's plan and c1's are in
  the grid, so M may be either.
- **Block B, the READING:** five arms on 120 steps of B (12 Williams squares of 10, Amendment 1): T (the tree at today's plan, Mixtral's
  served default: the CONTROL), T2 (T again: the self-pair), P (prmt at today's plan), C1 (prmt at P160's c1 pair,
  e4b's ``E4B_K25_PLAN=c1``) and M (prmt at block A's selection). GO is read on M alone. Amendment 1 dropped TM (the tree at
  M's plan, reported only): p222-shake-1 found a selected plan whose tree variant cannot run (210,944 B of shared memory
  against the 101,376 B limit), and a reported arm never restricts what M may be.
- :func:`verdict`: GO / NO_GO / VOID, and on GO the served read's arm (:func:`served_arm`).

    python p222_reduce.py --bench bench_p222.json --out verdict_p222.json
    python p222_reduce.py --self-test
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import statistics
import sys

E, TOPK, TOKENS, LAYERS, STEPS = 8, 2, 16, 8, 128
SHAPES = {"gu": (28672, 4096), "dn": (4096, 14336)}         # (N, K): gate_up [2 x 14336, 4096], down [4096, 14336]
TODAY = {"block_n": 32, "kc": 64, "warps": 4, "stages": 3}  # e4b's _K25_PLAN; dot_bf16 False is fixed
C1 = {"gu": {"block_n": 64, "kc": 256, "warps": 4, "stages": 2},       # e4b's _K25_PLANS["c1"] (P160's carried pair,
      "dn": {"block_n": 128, "kc": 128, "warps": 8, "stages": 3}}      # P161's served read on Qwen3-30B-A3B)
T_LUT, P_LUT = "tree", "prmt"
GRID = [{"block_n": bn, "kc": kc, "warps": w, "stages": s}            # P156's grid, unchanged
        for bn in (16, 32, 64, 128) for kc in (64, 128, 256) for w in (2, 4, 8) for s in (1, 2, 3, 4)]
ORDER_SEED = 222                             # the sweep's plan order (one shuffle, both projections)
SEEDS = {"A": 2220, "B": 2221}               # the routing draws: A selects, B reads
ROUTE_SHA256 = {"A": "996d7f093545051860d601b7dbb606732bd28db308f47b7a3e9e2898e6fdf54c",
                "B": "5f368c0fdf03af0e8c6dd3419007e7ab9fffa5f9d9985a08eb0ba6863a6d0b75"}   # route_sha256(picks(seed)), [layer][step][row][k] int8
SELECT_STEPS = tuple(range(0, STEPS, 16))    # 8 steps of block A
SELECT_ITERS = 5                             # timed replays per (plan, projection, step) in the sweep
READ_STEPS = tuple(range(0, 120))            # 120 steps of block B: 12 whole Williams squares (10 orders each, Amendment 1)
ARMS = ["T", "T2", "P", "C1", "M"]
SPACER = "S"                                 # T's calls with their own outputs, replayed untimed before every timed replay
ITERS = 10                                   # timed replays per arm per rep
GO_BAR = 1.10                                # M's median T/M minus 2 SE, raw AND clock-adjusted, must reach this
TIE_BAR = 1.02                               # M's median over C1 below this: the served read takes C1's knob instead
BOOT_B, BOOT_SEED = 2000, 222
SELF_BAND = (0.99, 1.01)                     # T/T2's median, else the instrument is VOID
CLOCK_TOL, MEM_CLOCK_TOL = 0.02, 0.01        # an arm's median SM / memory clock off T's by more: labelled
MIN_POWER_W = 575                            # every NVML sample's enforced limit, else VOID
P145_K25_SHARE = 0.911                       # P145 (p145-5090-1): the K25 pair's share of Mixtral's W16 step, tree/today
PRED = {"t_over_p": (1.15, 1.35), "t_over_m": (1.30, 1.90), "self_pair": 0.005, "clock": CLOCK_TOL}


def plan_key(p):
    return (p["block_n"], p["kc"], p["warps"], p["stages"])


_GRID_INDEX = {plan_key(p): i for i, p in enumerate(GRID)}


def picks(seed, layers=LAYERS, steps=STEPS):
    """[layer][step][row] -> (e0, e1): two distinct experts of E, uniform, from one seeded generator, drawn in that
    loop order. Mixtral routes each token to its top-2 of 8; this draws the same shape with no router."""
    rng = random.Random(seed)
    return [[[tuple(rng.sample(range(E), TOPK)) for _ in range(TOKENS)] for _ in range(steps)] for _ in range(layers)]


def route_sha256(pk):
    """sha256 of the table's int8 bytes in [layer][step][row][k] order."""
    return hashlib.sha256(bytes(e for layer in pk for step in layer for row in step for e in row)).hexdigest()


def n_real(pk):
    """[step][layer] distinct experts across the 16 rows (one tile each: an expert gets at most 16 rows)."""
    return [[len({e for row in pk[li][s] for e in row}) for li in range(len(pk))] for s in range(len(pk[0]))]


def sweep_order():
    order = list(range(len(GRID)))
    random.Random(ORDER_SEED).shuffle(order)
    return order


def select_ms(step_replays):
    """A plan's selection time: the mean over the selection steps of each step's median replay (P156's)."""
    return statistics.fmean(statistics.median(x) for x in step_replays)


def _num(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x) and x > 0


def _good_replays(xs, n=ITERS):
    return isinstance(xs, list) and len(xs) == n and all(_num(x) for x in xs)


def _eligible(row, n_steps):
    sr = row.get("step_replays_ms")
    return (row.get("identical") is True and not row.get("error") and isinstance(sr, list) and len(sr) == n_steps
            and all(_good_replays(x, SELECT_ITERS) for x in sr))


def select(rows, n_steps=len(SELECT_STEPS)):
    """The registered selection: per projection, the eligible plan (bit-identical to the tree at today's plan, every
    replay read) with the lowest :func:`select_ms`, ties to the lower grid index. Today's and c1's plans compete."""
    m = {}
    for proj in ("gu", "dn"):
        ok = [r for r in rows.get(proj) or [] if _eligible(r, n_steps)]
        if not ok:
            return None
        ok.sort(key=lambda r: (select_ms(r["step_replays_ms"]), _GRID_INDEX[plan_key(r["plan"])]))
        m[proj] = dict(ok[0]["plan"])
    return m


def selection_margin(rows, n_steps=len(SELECT_STEPS)):
    """Per projection, the runner-up's selection time over the winner's (reported: a near-tie selection is visible)."""
    out = {}
    for proj in ("gu", "dn"):
        ts = sorted(select_ms(r["step_replays_ms"]) for r in rows.get(proj) or [] if _eligible(r, n_steps))
        out[proj] = ts[1] / ts[0] if len(ts) > 1 else None
    return out


def plans_for(m):
    """Each arm's (gate_up plan, down plan, decode). T/T2/P move only the decode; C1 and M move the plan under prmt.
    (Amendment 1 dropped TM, the tree at M's plan.)"""
    return {"T": (TODAY, TODAY, T_LUT), "T2": (TODAY, TODAY, T_LUT), "P": (TODAY, TODAY, P_LUT),
            "C1": (C1["gu"], C1["dn"], P_LUT), "M": (m["gu"], m["dn"], P_LUT)}


def served_arm(m, m_over_c1):
    """The served read's arm, named before data: M's plan when it is a new pair that beats C1 by TIE_BAR (a _K25_PLANS
    code PR first), else the existing knob it equals or nearly equals (no code change)."""
    if m == {"gu": TODAY, "dn": TODAY}:
        return "P"
    if m == C1 or m_over_c1 < TIE_BAR:
        return "C1"
    return "M"


def williams(n):
    """Williams' design: for an even n, n orders with every ordered pair adjacent exactly once (P157's); for an odd n,
    those n orders and their mirrors, 2n orders with every ordered pair adjacent exactly twice (Amendment 1: five arms)."""
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
    """Rep r's capture and replay order: row r mod 10 of Williams' design for five arms (Amendment 1). The spacer S
    precedes every timed replay."""
    w = williams(len(arms))
    return [arms[j] for j in w[r % len(w)]]


def se_median(q, b=BOOT_B, seed=BOOT_SEED):
    """The SE of the MEDIAN, P155's rule: the larger of sqrt(pi / 2) sd / sqrt(n) and a seeded bootstrap's."""
    n = len(q)
    normal = math.sqrt(math.pi / 2) * statistics.stdev(q) / math.sqrt(n)
    rng = random.Random(seed)
    boot = statistics.stdev(statistics.median([q[int(rng.random() * n)] for _ in range(n)]) for _ in range(b))
    return max(normal, boot)


def summarize(q):
    med = statistics.median(q)
    se = se_median(q) if len(q) > 1 else math.inf
    return {"n": len(q), "median": med, "se": se, "lb": med - 2 * se, "min": min(q), "max": max(q)}


def _structure(rec, why):
    """Every VOID that is a property of the record's shape, inputs and instrument; appends to ``why``."""
    shake = bool(rec.get("shakedown"))
    layers = rec.get("layers")
    if rec.get("route_sha256") != ROUTE_SHA256 and not shake:
        why.append("a block's routing table does not hash to the pinned seed draw")
    if rec.get("arm_fault"):
        why.append(f"an arm could not compile, launch or be captured (Amendment 1): {rec['arm_fault']}")
    if rec.get("arms") != ARMS or rec.get("spacer") != SPACER or rec.get("iters") != ITERS:
        why.append("the arms, the spacer or the timed replays an arm are not the registered ones")
    if not shake:
        if "5090" not in str(rec.get("device", "")) or rec.get("cc") != [12, 0]:
            why.append(f"device {rec.get('device')} cc {rec.get('cc')}: the lane registers an RTX 5090 (sm_120)")
        if layers != LAYERS:
            why.append(f"{layers} layers timed, registered {LAYERS}")
    elif not isinstance(layers, int) or not 1 <= layers <= LAYERS:
        why.append(f"shakedown layers {layers}")
    sel = rec.get("select") or {}
    order = rec.get("order")
    n_sel = len(rec.get("select_steps") or [])
    if not shake and (rec.get("select_steps") != list(SELECT_STEPS) or order != sweep_order()):
        why.append("the selection steps or the sweep order are not the registered ones")
    for proj in ("gu", "dn"):
        rows = sel.get(proj) or []
        want = [GRID[i] for i in order or []]
        if [r.get("plan") for r in rows] != want or (not shake and len(rows) != len(GRID)):
            why.append(f"select {proj}: the swept plans are not the grid in the registered order")
    m = select(sel, n_sel) if n_sel else None
    if m is None:
        why.append("the selection names no plan for a projection")
    elif rec.get("m") != m:
        why.append(f"block B timed M = {rec.get('m')}, but the selection rule names {m}")
    elif rec.get("plans") != {a: list(p) for a, p in plans_for(m).items()}:
        why.append("the arms' plans and decodes are not the registered ones for this M")
    blk = rec.get("read") or {}
    steps = blk.get("steps") or []
    if (steps != list(READ_STEPS)) if not shake else not (steps and steps == list(READ_STEPS)[:len(steps)]):
        why.append("block B's steps are not the registered ones")
    nr = blk.get("n_real")
    if not (isinstance(nr, list) and isinstance(layers, int) and len(nr) == STEPS
            and all(isinstance(x, list) and len(x) == layers and all(isinstance(v, int) and TOPK <= v <= E for v in x)
                    for x in nr)):
        why.append("block B's n_real is not [128 steps][layers] of distinct-expert counts")
    elif not shake and nr != n_real(picks(SEEDS["B"])):
        why.append("block B's distinct-expert counts are not the seed draw's")
    if blk.get("tile_faults"):
        why.append(f"tile table faults {blk['tile_faults'][:3]}")
    reps = blk.get("reps") or []
    if len(reps) != len(steps) or not steps:
        why.append(f"block B: {len(reps)} reps for {len(steps)} steps")
        return why
    for r, rep in enumerate(reps):
        if rep.get("step") != steps[r] or rep.get("order") != arm_order(r):
            why.append(f"rep {r}: its step or its order is not the registered Williams row")
            break
        rp = rep.get("replays_ms") or {}
        if set(rp) != set(ARMS) or not all(_good_replays(rp[a]) for a in ARMS):
            why.append(f"rep {r}: replays missing or malformed")
            break
        if rep.get("spacer_replays") != ITERS * len(ARMS):
            why.append(f"rep {r}: {rep.get('spacer_replays')} spacer replays, not one before each timed replay")
            break
        ident = rep.get("identical") or {}
        if set(ident) != set(ARMS) - {"T"}:
            why.append(f"rep {r}: identity not recorded for every arm")
            break
        if ident["T2"] is not True:
            why.append(f"rep {r}: T replayed different bits in a second capture: not deterministic")
            break
    clk = rec.get("clocks") or {}
    if clk.get("error") or not all(isinstance((clk.get(a) or {}).get("sm_mhz_median"), (int, float)) for a in ARMS):
        why.append(f"the clocks are unread ({clk.get('error') or 'an arm has no samples'}): the GO's clock bound needs them")
    elif not shake:
        lim = [w for a in ARMS for w in clk[a].get("enforced_limit_w") or [None]]
        if any(not _num(w) or w < MIN_POWER_W for w in lim):
            why.append(f"an enforced power limit {sorted({str(w) for w in lim})} W under {MIN_POWER_W} W during a timed replay")
    return why


def rep_ms(rep):
    return {arm: statistics.median(v) for arm, v in rep["replays_ms"].items()}


def read_block(blk, clocks):
    """Block B's paired ratios over T (each rep's median of 10 replays), with each arm's clocks against T's."""
    ms = [rep_ms(rep) for rep in blk["reps"]]
    out = {"self_pair": summarize([m["T"] / m["T2"] for m in ms]), "ms": {a: statistics.median(m[a] for m in ms)
                                                                         for a in ARMS}}
    base, mbase = clocks["T"]["sm_mhz_median"], clocks["T"].get("mem_mhz_median")
    for arm in ("P", "C1", "M"):
        clk, mclk = clocks[arm]["sm_mhz_median"], clocks[arm].get("mem_mhz_median")
        s = summarize([m["T"] / m[arm] for m in ms])
        s["clock_over_t"] = clk / base
        s["labels"] = [f"clock-confounded: SM {clk:.0f} MHz against T's {base:.0f}"] if abs(clk / base - 1) > CLOCK_TOL else []
        if _num(mclk) and _num(mbase):
            s["mem_clock_over_t"] = mclk / mbase
            if abs(mclk / mbase - 1) > MEM_CLOCK_TOL:
                s["labels"].append(f"clock-confounded: memory {mclk:.0f} MHz against T's {mbase:.0f}")
        else:
            s["mem_clock_over_t"] = None
            s["labels"].append("memory clock unread")
        s["lb_clock_adj"] = s["lb"] * min(1.0, base / clk)    # a faster-clocked arm's clock gain taken away at 1:1
        out[arm] = s
    out["M_over_C1"] = summarize([m["C1"] / m["M"] for m in ms])
    out["moved_bits"] = {arm: [rep["step"] for rep in blk["reps"] if rep["identical"][arm] is not True]
                         for arm in ("P", "C1", "M")}
    return out


def verdict(rec):
    why = _structure(rec, [])
    shake = bool(rec.get("shakedown"))
    out = {"lane": "P222", "shakedown": shake}
    if why:
        return dict(out, verdict="VOID", why=why)
    m = rec["m"]
    rd = read_block(rec["read"], rec["clocks"])
    out.update(m=m, read=rd, compiled=rec.get("compiled"))
    sp = rd["self_pair"]
    if not SELF_BAND[0] <= sp["median"] <= SELF_BAND[1]:
        return dict(out, verdict="VOID", why=[f"self-pair T/T2 median {sp['median']:.4f} outside {SELF_BAND}"])
    findings = [{"arm": a, "moved_bits_on_steps": rd["moved_bits"][a][:8]} for a in ("P", "C1", "M")
                if rd["moved_bits"][a]]
    out["contract_findings"] = findings
    ma = rd["M"]
    out["M"] = ma
    out["reported"] = {"P": rd["P"], "C1": rd["C1"], "M_over_C1": rd["M_over_C1"],
                       "m_is_today": m == {"gu": TODAY, "dn": TODAY}, "m_is_c1": m == C1,
                       "selection_margin": selection_margin(rec["select"], len(rec["select_steps"])),
                       "step_share_note": f"the K25 pair is {P145_K25_SHARE} of Mixtral's W16 step at tree/today (P145)"}
    out["predictions"] = {k: "UNREAD (a shakedown)" for k in _grade(out)} if shake else _grade(out)
    moved = [f for f in findings if f["arm"] in ("P", "C1", "M")]
    if shake:
        out["verdict"], out["why"] = "SHAKEDOWN", ["a shakedown record is never a reading"]
    elif moved:
        out["verdict"], out["why"] = "NO_GO", [f"{moved[0]['arm']} moved a bit (steps {moved[0]['moved_bits_on_steps']}): "
                                               "a contract finding against grouped-nf4-gemm; no timing is ruled on"]
    elif ma["lb"] >= GO_BAR and ma["lb_clock_adj"] >= GO_BAR:
        out["verdict"] = "GO"
        out["why"] = [f"M {ma['median']:.4f}x over T, lower bound {ma['lb']:.4f} and clock-adjusted "
                      f"{ma['lb_clock_adj']:.4f} >= {GO_BAR}"]
        out["served_arm"] = served_arm(m, rd["M_over_C1"]["median"])
    else:
        out["verdict"], out["why"] = "NO_GO", [f"M {ma['median']:.4f}x over T, lower bound {ma['lb']:.4f}, clock-adjusted "
                                               f"{ma['lb_clock_adj']:.4f}: not both >= {GO_BAR}"]
    return out


def _band(x, lo, hi):
    return "UNREAD" if x is None else ("HELD" if lo <= x <= hi else "MISSED")


def _grade(o):
    r = o["read"]
    return {
        "the decode: T/P in 1.15-1.35 (Qwen3's P156: 1.1755)": _band(r["P"]["median"], *PRED["t_over_p"]),
        "the headline: T/M in 1.30-1.90": _band(r["M"]["median"], *PRED["t_over_m"]),
        "M is not c1's pair (Qwen3's plan does not carry to Mixtral's shapes)": "MISSED" if o["m"] == C1 else "HELD",
        "self-pair within 0.005 of 1": _band(abs(r["self_pair"]["median"] - 1), 0, PRED["self_pair"]),
        "clocks: M's median SM clock within 2 % of T's": _band(abs(r["M"]["clock_over_t"] - 1), 0, PRED["clock"]),
        "contract: every arm keeps T's bits on every rep": "MISSED" if o["contract_findings"] else "HELD",
    }


# ------------------------------------------------------------------------------------------------ self-test (mutants)
def _sel_rows(best_gu=None, best_dn=None, iters=SELECT_ITERS, n_steps=len(SELECT_STEPS), order=None):
    """A synthetic sweep: every grid plan at 10 ms except the named winners at 5 ms."""
    rows = {}
    for proj, best in (("gu", best_gu or C1["gu"]), ("dn", best_dn or C1["dn"])):
        rows[proj] = [{"plan": dict(GRID[i]), "identical": True, "error": None,
                       "step_replays_ms": [[5.0 if GRID[i] == best else 10.0] * iters for _ in range(n_steps)]}
                      for i in (order or sweep_order())]
    return rows


def _fake(m_gain=1.6, p_gain=1.25, c1_gain=1.5, noise=0.002, self_ratio=1.0, t_ms=12.0, layers=LAYERS,
          shakedown=False, clk=None, mem=None, limit=575.0, best_gu=None, best_dn=None, **kw):
    rng = random.Random(7)
    sel = _sel_rows(best_gu, best_dn)
    m = select(sel)
    f = {"T": 1.0, "T2": 1 / self_ratio, "P": 1 / p_gain, "C1": 1 / c1_gain, "M": 1 / m_gain}
    reps = [{"step": s, "order": arm_order(r), "spacer_replays": ITERS * len(ARMS),
             "replays_ms": {a: [t_ms * f[a] * (1 + rng.gauss(0, noise)) for _ in range(ITERS)] for a in ARMS},
             "identical": {a: True for a in ARMS if a != "T"}} for r, s in enumerate(READ_STEPS)]
    nr = n_real(picks(SEEDS["B"], layers=layers))
    rec = {"device": "NVIDIA GeForce RTX 5090", "cc": [12, 0], "layers": layers, "shakedown": shakedown,
           "route_sha256": dict(ROUTE_SHA256), "arms": list(ARMS), "spacer": SPACER, "iters": ITERS,
           "select_steps": list(SELECT_STEPS), "order": sweep_order(), "select": sel, "m": m,
           "plans": {a: list(p) for a, p in plans_for(m).items()},
           "read": {"steps": list(READ_STEPS), "n_real": nr, "tile_faults": [], "reps": reps},
           "clocks": {"error": None, **{a: {"sm_mhz_median": (clk or {}).get(a, 2800.0),
                                            "mem_mhz_median": (mem or {}).get(a, 14001.0),
                                            "enforced_limit_w": [limit]} for a in ARMS}}}
    rec.update(kw)
    return rec


def self_test():
    v = lambda rec: verdict(rec)["verdict"]                  # noqa: E731
    pk = picks(SEEDS["B"])
    assert all(len(set(row)) == TOPK and all(0 <= e < E for e in row) for layer in pk for step in layer for row in step)
    assert route_sha256(pk) == ROUTE_SHA256["B"] and route_sha256(picks(SEEDS["A"])) == ROUTE_SHA256["A"]
    assert ROUTE_SHA256["A"] != ROUTE_SHA256["B"]
    assert sorted(sweep_order()) == list(range(len(GRID))) and len(GRID) == 144
    assert TODAY in GRID and C1["gu"] in GRID and C1["dn"] in GRID          # both knobs compete in the sweep
    # the selection's eligibility, tie-break and margin, on synthetic sweeps
    fast = {"block_n": 128, "kc": 256, "warps": 8, "stages": 2}
    sel = _sel_rows()
    row = next(r for r in sel["gu"] if r["plan"] == fast)
    row["step_replays_ms"] = [[1.0] * SELECT_ITERS for _ in SELECT_STEPS]
    row["identical"] = False
    assert select(sel)["gu"] == C1["gu"]                                     # a plan that moved a bit never wins
    row["identical"], row["error"] = True, "RuntimeError: out of shared memory"
    assert select(sel)["gu"] == C1["gu"]                                     # nor does a plan that errored
    row["error"] = None
    assert select(sel)["gu"] == fast                                         # once eligible, the fastest wins
    tie = _sel_rows()
    lo, hi = sorted((GRID.index(C1["gu"]), GRID.index(fast)))
    for r in tie["gu"]:
        if r["plan"] == fast:
            r["step_replays_ms"] = [[5.0] * SELECT_ITERS for _ in SELECT_STEPS]
    assert select(tie)["gu"] == GRID[lo] and GRID[hi] != GRID[lo]           # a tie goes to the lower grid index
    marg = _sel_rows()
    for r in marg["gu"]:
        if r["plan"] == fast:
            r["step_replays_ms"] = [[6.0] * SELECT_ITERS for _ in SELECT_STEPS]
    assert selection_margin(marg)["gu"] == 6.0 / 5.0                         # the runner-up over the winner, not the slowest
    o = verdict(_fake())
    assert o["verdict"] == "GO" and o["served_arm"] == "C1", o.get("why")  # M = c1 (the synthetic sweep's winner)
    assert o["reported"]["selection_margin"] == {"gu": 2.0, "dn": 2.0}      # the runner-up at 10 ms over the winner's 5
    assert all(g in ("HELD", "MISSED", "UNREAD") for g in o["predictions"].values()), o["predictions"]
    new_gu = {"block_n": 128, "kc": 256, "warps": 8, "stages": 2}
    o = verdict(_fake(best_gu=new_gu, m_gain=1.8, c1_gain=1.5))
    assert o["verdict"] == "GO" and o["m"]["gu"] == new_gu and o["served_arm"] == "M"      # a new pair: code PR first
    o = verdict(_fake(best_gu=new_gu, m_gain=1.52, c1_gain=1.50))
    assert o["verdict"] == "GO" and o["served_arm"] == "C1"                  # within TIE_BAR of c1: the knob arm
    o = verdict(_fake(best_gu=TODAY, best_dn=TODAY, m_gain=1.25, p_gain=1.25))
    assert o["verdict"] == "GO" and o["served_arm"] == "P"                   # M is today's plan: the decode alone
    assert v(_fake(m_gain=1.09)) == "NO_GO"                                  # under the bar
    assert v(_fake(m_gain=1.11, noise=0.15)) == "NO_GO"                      # median over the bar, bound under it
    assert v(_fake(m_gain=1.11, noise=0.0005)) == "GO"
    o = verdict(_fake(m_gain=1.13, noise=0.0005, clk={"M": 2912.0}))        # M 4 % faster-clocked: the bound adjusts
    assert o["verdict"] == "NO_GO" and o["M"]["lb"] >= GO_BAR > o["M"]["lb_clock_adj"], o["M"]
    o = verdict(_fake(m_gain=1.13, noise=0.0005, clk={"M": 2700.0}))        # a slower-clocked M is never credited
    assert o["verdict"] == "GO" and o["M"]["lb_clock_adj"] == o["M"]["lb"]
    assert any("memory 14200" in x for x in verdict(_fake(mem={"M": 14200.0}))["M"]["labels"])
    assert v(_fake(m_gain=1.0, c1_gain=1.6, p_gain=1.5)) == "NO_GO"          # only M rules: fast P and C1 are reported
    assert v(_fake(self_ratio=1.03)) == "VOID"
    for arm in ("P", "C1", "M"):
        bad = _fake()
        bad["read"]["reps"][7]["identical"][arm] = False
        o = verdict(bad)
        assert o["verdict"] == "NO_GO" and o["contract_findings"][0]["arm"] == arm, arm
    bad = _fake(arm_fault="M dn (128, 256, 8, 4) prmt: OutOfResources: shared memory")
    assert v(bad) == "VOID" and "Amendment 1" in verdict(bad)["why"][0]     # a recorded arm fault, never a crash
    bad = _fake()
    bad["read"]["reps"][5]["identical"]["T2"] = False
    assert v(bad) == "VOID"                                                  # T is not deterministic
    bad = _fake()
    bad["m"] = dict(C1, gu=TODAY)
    assert v(bad) == "VOID"                                                  # B timed a plan the selection did not name
    bad = _fake()
    row = next(r for r in bad["select"]["dn"] if r["plan"] != C1["dn"])
    row["step_replays_ms"] = [[0.1] * SELECT_ITERS for _ in SELECT_STEPS]  # the selection now names another plan
    assert v(bad) == "VOID"
    bad = _fake()
    bad["select"]["gu"] = bad["select"]["gu"][:-1]
    assert v(bad) == "VOID"                                                  # a plan missing from the sweep
    bad = _fake(order=list(range(len(GRID))))
    bad["select"] = _sel_rows(order=list(range(len(GRID))))
    assert v(bad) == "VOID"                                                  # swept, consistently, out of the shuffle
    bad = _fake()
    bad["plans"]["M"] = [dict(TODAY), dict(TODAY), T_LUT]
    assert v(bad) == "VOID"
    assert v(_fake(route_sha256={"A": ROUTE_SHA256["A"], "B": "0" * 64})) == "VOID"
    bad = _fake()
    bad["read"]["n_real"][0][0] = bad["read"]["n_real"][0][0] % E + 1 if bad["read"]["n_real"][0][0] < E else TOPK
    assert v(bad) == "VOID"                                                  # not the seed draw's counts
    assert v(_fake(device="NVIDIA RTX A2000 12GB", cc=[8, 6])) == "VOID"
    assert v(_fake(layers=4)) == "VOID"
    assert v(_fake(spacer=None)) == "VOID"
    bad = _fake()
    bad["read"]["reps"][3]["spacer_replays"] = 0
    assert v(bad) == "VOID"
    bad = _fake()
    bad["read"]["reps"][3]["order"] = bad["read"]["reps"][4]["order"]
    assert v(bad) == "VOID"
    bad = _fake()
    bad["read"]["reps"].pop()
    assert v(bad) == "VOID"
    bad = _fake()
    bad["read"]["reps"][9]["replays_ms"]["C1"] = bad["read"]["reps"][9]["replays_ms"]["C1"][:5]
    assert v(bad) == "VOID"
    bad = _fake()
    bad["read"]["tile_faults"] = [{"step": 3, "layer": 1, "fault": "x"}]
    assert v(bad) == "VOID"
    assert v(_fake(limit=480.0)) == "VOID"
    bad = _fake()
    bad["clocks"]["error"] = "OSError: libnvidia-ml.so.1"
    assert v(bad) == "VOID"
    sk = _fake(shakedown=True, layers=2, device="NVIDIA GeForce RTX 3090", cc=[8, 6], limit=350.0)
    sk["read"]["reps"] = sk["read"]["reps"][:6]
    sk["read"]["steps"] = sk["read"]["steps"][:6]
    o = verdict(sk)
    assert o["verdict"] == "SHAKEDOWN", o["why"]
    w = williams(len(ARMS))                                                  # five arms: 10 orders
    assert len(w) == 10 and all(sorted(row) == list(range(5)) for row in w)
    assert all(sorted(row[i] for row in w) == sorted(list(range(5)) * 2) for i in range(5))   # each position twice
    pairs = [(row[i], row[i + 1]) for row in w for i in range(4)]
    assert len(set(pairs)) == 20 and all(pairs.count(pr) == 2 for pr in set(pairs))           # each ordered pair twice
    w6 = williams(6)
    assert len({(row[i], row[i + 1]) for row in w6 for i in range(5)}) == 30                   # the even case, kept
    assert len(READ_STEPS) % len(w) == 0 and not set(SELECT_STEPS) - set(range(STEPS))
    print("p222_reduce self-test OK (50 checks)")



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
            print(f"p222_reduce self-test FAILED: {e}")
            return 1
        return 0
    if not (a.bench and a.out):
        ap.error("--bench and --out are required")
    with open(a.bench, encoding="utf-8") as f:
        rec = json.load(f)
    out = verdict(rec)
    with open(a.out, "w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(out, indent=1, sort_keys=True) + "\n")
    print(f"P222 {out['verdict']} " + json.dumps({"why": out["why"], "M": out.get("M"), "served_arm": out.get("served_arm")}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
