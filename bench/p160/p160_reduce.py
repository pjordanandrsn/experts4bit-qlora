# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""P160's reducer (bench/p160/PREREG-p160.md): decode x plan. K39's decode (K25's ``lut="prmt"``) at P156's carried tile
plans against prmt at today's plan, now qwen3_moe's served default (P158, e4b#1868), at Qwen3-30B-A3B's W16 shapes.

Pure Python, no torch. The registered rules live here, and the box imports them (``p160_bench.py``):
- **The question, fixed before data:** C1, P156's first carried plan pair (gate_up 64/256/4/2, down 128/128/8/3), at
  ``lut="prmt"``, against P, ``lut="prmt"`` at today's plan. Only the plan differs. GO is read on C1 alone, on the S1
  block. C2 and C3, T (the tree at today's plan) and the S0R block are reported, never ruling.
- **No selection here:** every plan is named before data (P156's record), so nothing in this lane carries a winner's
  curse; the read is P156's reported 1.90x (1.617 over P) re-asked under a registered rule, a spacer, a balanced order
  and per-arm clocks, on a stream P156 never timed.
- :func:`arm_order`: each rep's capture and replay order, a row of Williams' design over the six arms;
- :func:`verdict`: GO / NO_GO / VOID.

    python p160_reduce.py --bench bench_p160.json --out verdict_p160.json
    python p160_reduce.py --self-test
"""
from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import sys

RR_SHA256 = "3c59e72f947daa425e61f727401204099a1e782b902fb39687e04c46e338b14f"   # bench/p153/router-replay/rr.json.pt
E, TOPK, TOKENS, LAYERS, STEPS = 128, 8, 16, 48, 128
SHAPES = {"gu": (1536, 2048), "dn": (2048, 768)}           # (N, K): gate_up [2 x 768, 2048], down [2048, 768]
TODAY = {"block_n": 32, "kc": 64, "warps": 4, "stages": 3}  # e4b's _K25_PLAN; dot_bf16 False is fixed
T_LUT, P_LUT = "tree", "prmt"
#: P156's carried plans (bench/p156/receipts/p156-5090-1/bench_p156.json "carried"), in its rank order, never re-chosen
CARRIED = [
    {"gu": {"block_n": 64, "kc": 256, "warps": 4, "stages": 2}, "dn": {"block_n": 128, "kc": 128, "warps": 8, "stages": 3}},
    {"gu": {"block_n": 64, "kc": 256, "warps": 4, "stages": 3}, "dn": {"block_n": 32, "kc": 256, "warps": 4, "stages": 2}},
    {"gu": {"block_n": 64, "kc": 256, "warps": 8, "stages": 3}, "dn": {"block_n": 128, "kc": 128, "warps": 4, "stages": 2}},
]
ARMS = ["P", "P2", "C1", "C2", "C3", "T"]
#: each arm's (gate_up plan, down plan, decode); only the plan moves between P and C1, only the decode between P and T
PLANS = {"P": (TODAY, TODAY, P_LUT), "P2": (TODAY, TODAY, P_LUT),
         "C1": (CARRIED[0]["gu"], CARRIED[0]["dn"], P_LUT), "C2": (CARRIED[1]["gu"], CARRIED[1]["dn"], P_LUT),
         "C3": (CARRIED[2]["gu"], CARRIED[2]["dn"], P_LUT), "T": (TODAY, TODAY, T_LUT)}
SPACER = "S"                                 # P's calls with their own outputs, replayed untimed before every timed replay
#: the blocks, in run order. S1 is the READING: P153's second stream (rows 16-31), which no K25 microbench has timed.
#: S0R is REPORTED: P156's own held-out steps of S0 (rows 0-15), the reproduction of its 1.617.
BLOCKS = {"S1": {"rows": [16, 32], "steps": list(range(0, 126))},
          "S0R": {"rows": [0, 16], "steps": list(range(64, 124))}}
READING = "S1"
ROWS_SHA256 = {"S0": "3020dbce691bd98462b9c74b9fe903c1b124d7a6b95f22e8d964532479b282a1",   # picks[:, 0:16], <i2
               "S1": "df2194a45c36c756ec6a357c6f18d82693f6e2c4df8c2ec22c9456eb400eaedb"}   # picks[:, 16:32], <i2
DISTINCT_TOTAL = {"S0": 350868, "S1": 350529}   # P153's record: distinct experts summed over 48 layers x 128 steps
STREAM = {"S1": "S1", "S0R": "S0"}
ITERS = 10                                   # timed replays per arm per rep
GO_BAR = 1.05                                # C1's median P/C1 minus 2 SE, raw AND clock-adjusted, must reach this
BOOT_B, BOOT_SEED = 2000, 160
SELF_BAND = (0.99, 1.01)                     # P/P2's median, else the instrument is VOID
CLOCK_TOL = 0.02                             # an arm's median SM clock off P's by more than this is labelled
MIN_POWER_W = 575                            # every NVML sample's enforced limit, else VOID
P156_C1_OVER_P = 1.6167                      # P156's reported C1 over P on S0's steps 64-127 (RESULTS-p156.md)
PRED = {"headline": (1.50, 1.70), "repro": 0.03, "t_over_p": (1.15, 1.20), "self_pair": 0.005, "clock": CLOCK_TOL,
        "regs_c1": {"gu": 194, "dn": 127}}


def plan_key(p):
    return (p["block_n"], p["kc"], p["warps"], p["stages"])


def williams(n):
    """Williams' design for an even n: n orders, each a Latin row, every ordered pair of arms adjacent exactly once
    (P157's construction)."""
    first, lo, hi = [0], 1, n - 1
    while len(first) < n:
        first.append(lo)
        lo += 1
        if len(first) < n:
            first.append(hi)
            hi -= 1
    return [[(x + i) % n for x in first] for i in range(n)]


def arm_order(r, arms=ARMS):
    """Rep r's capture and replay order (r counted within its block): row r mod 6 of Williams' design. The spacer S is
    every timed replay's immediate predecessor, so no arm inherits a light or heavy neighbour's power and clock
    headroom; the design balances each arm's position in the rep and the arm timed before S."""
    return [arms[j] for j in williams(len(arms))[r % len(arms)]]


def _num(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x) and x > 0


def _good_replays(xs, n=ITERS):
    return isinstance(xs, list) and len(xs) == n and all(_num(x) for x in xs)


def se_median(q, b=BOOT_B, seed=BOOT_SEED):
    """The standard error of the MEDIAN, P155's rule: the larger of sqrt(pi / 2) x sd / sqrt(n) and the sd of the
    medians of ``b`` resamples drawn through one seeded generator's ``random()`` alone."""
    n = len(q)
    normal = math.sqrt(math.pi / 2) * statistics.stdev(q) / math.sqrt(n)
    rng = random.Random(seed)
    boot = statistics.stdev(statistics.median([q[int(rng.random() * n)] for _ in range(n)]) for _ in range(b))
    return max(normal, boot)


def summarize(q):
    n = len(q)
    med = statistics.median(q)
    se = se_median(q) if n > 1 else math.inf
    return {"n": n, "median": med, "se": se, "lb": med - 2 * se, "min": min(q), "max": max(q)}


def _structure(rec, why):
    """Every VOID that is a property of the record's shape, inputs and instrument; appends to ``why``."""
    shake = bool(rec.get("shakedown"))
    layers = rec.get("layers")
    if rec.get("rr_sha256") != RR_SHA256:
        why.append(f"router replay sha256 {rec.get('rr_sha256')} is not the pinned {RR_SHA256[:8]}")
    if rec.get("rows_sha256") != ROWS_SHA256:
        why.append("a stream's picks do not hash to the pinned S0 / S1")
    if rec.get("arms") != ARMS or rec.get("spacer") != SPACER or rec.get("plans") != {a: list(PLANS[a]) for a in ARMS}:
        why.append("the arms, their plans and decodes, or the spacer are not the registered ones")
    if rec.get("iters") != ITERS:
        why.append(f"{rec.get('iters')} timed replays an arm, registered {ITERS}")
    blocks = rec.get("blocks") or {}
    if list(blocks) != list(BLOCKS):
        why.append(f"blocks {list(blocks)}, registered {list(BLOCKS)} in that order")
        return why
    if not shake:
        if "5090" not in str(rec.get("device", "")) or rec.get("cc") != [12, 0]:
            why.append(f"device {rec.get('device')} cc {rec.get('cc')}: the lane registers an RTX 5090 (sm_120)")
        if layers != LAYERS:
            why.append(f"{layers} layers timed, registered {LAYERS}")
    elif not isinstance(layers, int) or not 1 <= layers <= LAYERS:
        why.append(f"shakedown layers {layers}")
    for b, spec in BLOCKS.items():
        blk = blocks[b]
        steps = blk.get("steps") or []
        if blk.get("rows") != spec["rows"]:
            why.append(f"{b}: rows {blk.get('rows')}, registered {spec['rows']}")
        if (steps != spec["steps"]) if not shake else not (steps and steps == spec["steps"][:len(steps)]):
            why.append(f"{b}: the steps are not the registered ones")
        nreal = blk.get("n_real")
        if not (isinstance(nreal, list) and len(nreal) == STEPS and isinstance(layers, int)
                and all(isinstance(x, list) and len(x) == layers and all(isinstance(v, int) and TOPK <= v <= E
                                                                         for v in x) for x in nreal)):
            why.append(f"{b}: n_real is not [128 steps][layers] of distinct-expert counts")
        elif layers == LAYERS and sum(map(sum, nreal)) != DISTINCT_TOTAL[STREAM[b]]:
            why.append(f"{b}: the distinct total {sum(map(sum, nreal))} is not P153's {DISTINCT_TOTAL[STREAM[b]]}")
        if blk.get("tile_faults"):
            why.append(f"{b}: tile table faults {blk['tile_faults'][:3]}")
        reps = blk.get("reps") or []
        if len(reps) != len(steps) or not steps:
            why.append(f"{b}: {len(reps)} reps for {len(steps)} steps")
            continue
        for r, rep in enumerate(reps):
            if rep.get("step") != steps[r] or rep.get("order") != arm_order(r):
                why.append(f"{b} rep {r}: its step or its order is not the registered Williams row")
                break
            rp = rep.get("replays_ms") or {}
            if set(rp) != set(ARMS) or not all(_good_replays(rp[a]) for a in ARMS):
                why.append(f"{b} rep {r}: replays missing or malformed")
                break
            if rep.get("spacer_replays") != ITERS * len(ARMS):
                why.append(f"{b} rep {r}: {rep.get('spacer_replays')} spacer replays, not one before each timed replay")
                break
            ident = rep.get("identical") or {}
            if set(ident) != set(ARMS) - {"T"} or rep.get("p2_is_p") not in (True, False):
                why.append(f"{b} rep {r}: identity not recorded for every arm")
                break
            if rep["p2_is_p"] is not True:
                why.append(f"{b} rep {r}: P replayed different bits in a second capture: not deterministic")
                break
    clk = rec.get("clocks") or {}
    if clk.get("error") or not all(isinstance(clk.get(b), dict) and all(
            isinstance((clk[b].get(a) or {}).get("sm_mhz_median"), (int, float)) for a in ARMS) for b in BLOCKS):
        why.append(f"the clocks are unread ({clk.get('error') or 'an arm has no samples'}): the GO's clock bound needs them")
    elif not shake:
        lim = [w for b in BLOCKS for a in ARMS for w in clk[b][a].get("enforced_limit_w") or [None]]
        if any(not _num(w) or w < MIN_POWER_W for w in lim):
            why.append(f"an enforced power limit {sorted({str(w) for w in lim})} W under {MIN_POWER_W} W during a timed replay")
    return why


def rep_ms(rep):
    return {arm: statistics.median(v) for arm, v in rep["replays_ms"].items()}


def block_read(blk, clocks):
    """One block's paired ratios over P (each rep's median of 10 replays), with each arm's clock against P's."""
    ms = [rep_ms(rep) for rep in blk["reps"]]
    out = {"self_pair": summarize([m["P"] / m["P2"] for m in ms]),
           "p_ms": statistics.median(m["P"] for m in ms), "ms": {a: statistics.median(m[a] for m in ms) for a in ARMS}}
    base = clocks["P"]["sm_mhz_median"]
    for arm in ("C1", "C2", "C3"):
        clk = clocks[arm]["sm_mhz_median"]
        s = summarize([m["P"] / m[arm] for m in ms])
        s["clock_over_p"] = clk / base
        s["labels"] = [f"clock-confounded: SM {clk:.0f} MHz against P's {base:.0f}"] if abs(clk / base - 1) > CLOCK_TOL else []
        # the conservative clock bound: a faster-clocked arm's whole clock gain taken away at 1:1 scaling
        s["lb_clock_adj"] = s["lb"] * min(1.0, base / clk)
        out[arm] = s
    out["T_over_P"] = summarize([m["T"] / m["P"] for m in ms])
    out["moved_bits"] = {arm: [rep["step"] for rep in blk["reps"] if rep["identical"][arm] is not True]
                         for arm in ("P", "P2", "C1", "C2", "C3")}
    return out


def verdict(rec):
    why = _structure(rec, [])
    shake = bool(rec.get("shakedown"))
    out = {"lane": "P160", "shakedown": shake}
    if why:
        return dict(out, verdict="VOID", why=why)
    reads = {b: block_read(rec["blocks"][b], rec["clocks"][b]) for b in BLOCKS}
    out["blocks"] = reads
    sp = reads[READING]["self_pair"]
    if not SELF_BAND[0] <= sp["median"] <= SELF_BAND[1]:
        return dict(out, verdict="VOID", why=[f"self-pair P/P2 median {sp['median']:.4f} outside {SELF_BAND}"])
    # a bit moved by P or C1 in EITHER block is a contract finding: no timing ruled on (it reads the same every rerun)
    findings = [{"arm": a, "block": b, "moved_bits_on_steps": reads[b]["moved_bits"][a][:8]}
                for b in BLOCKS for a in ("P", "P2", "C1", "C2", "C3") if reads[b]["moved_bits"][a]]
    out["contract_findings"] = findings
    c1 = reads[READING]["C1"]
    out["C1"] = c1
    out["reported"] = {"C2": reads[READING]["C2"], "C3": reads[READING]["C3"], "T_over_P": reads[READING]["T_over_P"],
                       "S0R_C1": reads["S0R"]["C1"], "p156_c1_over_p": P156_C1_OVER_P,
                       "c1_ms_at_p_scale": reads[READING]["ms"]["C1"]}
    out["compiled"] = rec.get("compiled")
    out["predictions"] = {k: "UNREAD (a shakedown)" for k in _grade(out)} if shake else _grade(out)
    c1_moved = [f for f in findings if f["arm"] in ("P", "C1")]
    if shake:
        out["verdict"], out["why"] = "SHAKEDOWN", ["a shakedown record is never a reading"]
    elif c1_moved:
        out["verdict"], out["why"] = "NO_GO", [f"{c1_moved[0]['arm']} moved a bit ({c1_moved[0]['block']} steps "
                                               f"{c1_moved[0]['moved_bits_on_steps']}): a contract finding against "
                                               "grouped-nf4-gemm; no timing is ruled on"]
    elif c1["lb"] >= GO_BAR and c1["lb_clock_adj"] >= GO_BAR:
        out["verdict"], out["why"] = "GO", [f"C1 {c1['median']:.4f}x over P, lower bound {c1['lb']:.4f} and clock-"
                                            f"adjusted {c1['lb_clock_adj']:.4f} >= {GO_BAR}"]
    else:
        out["verdict"], out["why"] = "NO_GO", [f"C1 {c1['median']:.4f}x over P, lower bound {c1['lb']:.4f}, clock-"
                                               f"adjusted {c1['lb_clock_adj']:.4f}: not both >= {GO_BAR}"]
    return out


def _band(x, lo, hi):
    return "UNREAD" if x is None else ("HELD" if lo <= x <= hi else "MISSED")


def _grade(o):
    b = o["blocks"]
    c1, c1r = b[READING]["C1"], b["S0R"]["C1"]
    return {
        "the headline: C1's median P/C1 on S1 in 1.50-1.70": _band(c1["median"], *PRED["headline"]),
        "reproduction: C1 on S0R within 0.03 of C1 on S1": _band(abs(c1r["median"] - c1["median"]), 0, PRED["repro"]),
        "the decode: T/P on S1 in 1.15-1.20 (P156: 1.1755)": _band(b[READING]["T_over_P"]["median"], *PRED["t_over_p"]),
        "self-pair within 0.005 of 1": _band(abs(b[READING]["self_pair"]["median"] - 1), 0, PRED["self_pair"]),
        "clocks: C1's median SM clock within 2 % of P's": _band(abs(c1["clock_over_p"] - 1), 0, PRED["clock"]),
        "contract: every arm keeps T's bits on every rep": "MISSED" if o["contract_findings"] else "HELD",
        "registers: C1 gate_up 194 and down 127, no spills": _regs(o.get("compiled")),
    }


def _regs(c):
    """HELD when C1's compiled variants read P156's sm_120 registers (gate_up 194, down 127) and no spills; UNREAD when
    this Triton's cache did not expose them (reported metadata, never a fault)."""
    vs = [(pj, v) for pj in ("gu", "dn") for v in (((c or {}).get("C1") or {}).get(pj) or [])]
    if {pj for pj, _ in vs} != {"gu", "dn"} or any(not isinstance(v.get("n_regs"), int)
                                                   or not isinstance(v.get("n_spills"), int) for _, v in vs):
        return "UNREAD"
    return "HELD" if all(v["n_regs"] == PRED["regs_c1"][pj] and v["n_spills"] == 0 for pj, v in vs) else "MISSED"


# ------------------------------------------------------------------------------------------------ self-test (mutants)
def _fake(c1=1.62, t=1 / 1.175, noise=0.002, self_ratio=1.0, p_ms=10.96, layers=LAYERS, shakedown=False,
          clk=None, limit=575.0, **kw):
    rng = random.Random(7)
    f = {"P": 1.0, "P2": 1 / self_ratio, "C1": 1 / c1, "C2": 1 / 1.55, "C3": 1 / 1.58, "T": 1 / t}
    blocks, clocks = {}, {"error": None, "n": 1}
    for b, spec in BLOCKS.items():
        cells = STEPS * layers
        total = DISTINCT_TOTAL[STREAM[b]] if layers == LAYERS else 57 * cells
        flat = [total // cells + (1 if i < total % cells else 0) for i in range(cells)]
        reps = [{"step": s, "order": arm_order(r), "spacer_replays": ITERS * len(ARMS), "p2_is_p": True,
                 "replays_ms": {a: [p_ms * f[a] * (1 + rng.gauss(0, noise)) for _ in range(ITERS)] for a in ARMS},
                 "identical": {a: True for a in ARMS if a != "T"}} for r, s in enumerate(spec["steps"])]
        blocks[b] = {"rows": list(spec["rows"]), "steps": list(spec["steps"]), "tile_faults": [],
                     "n_real": [flat[s * layers:(s + 1) * layers] for s in range(STEPS)], "reps": reps}
        clocks[b] = {a: {"sm_mhz_median": (clk or {}).get(a, 2800.0), "enforced_limit_w": [limit]} for a in ARMS}
    rec = {"device": "NVIDIA GeForce RTX 5090", "cc": [12, 0], "layers": layers, "shakedown": shakedown,
           "rr_sha256": RR_SHA256, "rows_sha256": dict(ROWS_SHA256), "arms": list(ARMS), "spacer": SPACER,
           "plans": {a: list(PLANS[a]) for a in ARMS}, "iters": ITERS, "blocks": blocks, "clocks": clocks}
    rec.update(kw)
    return rec


def self_test():
    v = lambda rec: verdict(rec)["verdict"]                  # noqa: E731
    o = verdict(_fake())
    assert o["verdict"] == "GO", o["why"]
    assert all(g in ("HELD", "MISSED", "UNREAD") for g in o["predictions"].values()), o["predictions"]
    assert o["predictions"]["the headline: C1's median P/C1 on S1 in 1.50-1.70"] == "HELD"
    assert abs(o["blocks"]["S1"]["T_over_P"]["median"] - 1.175) < 0.01
    assert v(_fake(c1=1.04)) == "NO_GO"                                      # under the bar
    assert v(_fake(c1=1.00)) == "NO_GO"                                      # flat
    assert v(_fake(c1=1.06, noise=0.20)) == "NO_GO"                          # median over the bar, bound under it
    assert v(_fake(c1=1.06, noise=0.0005)) == "GO"                           # the same median, tight
    o = verdict(_fake(c1=1.06, noise=0.12))             # GO under the mean's SE, NO_GO under the median's
    ms = [rep_ms(rep) for rep in _fake(c1=1.06, noise=0.12)["blocks"]["S1"]["reps"]]
    q = [m["P"] / m["C1"] for m in ms]
    assert o["verdict"] == "NO_GO" and o["C1"]["median"] - 2 * statistics.stdev(q) / math.sqrt(len(q)) >= GO_BAR > o["C1"]["lb"]
    # the clock bound: C1 at a 4 % higher clock, a 1.07 lift: raw bound GO, clock-adjusted under 1.05
    o = verdict(_fake(c1=1.07, noise=0.0005, clk={"C1": 2912.0}))
    assert o["verdict"] == "NO_GO" and o["C1"]["lb"] >= GO_BAR > o["C1"]["lb_clock_adj"], o["C1"]
    assert o["C1"]["labels"] and "clock-confounded" in o["C1"]["labels"][0]
    o = verdict(_fake(c1=1.07, noise=0.0005, clk={"C1": 2700.0}))           # a SLOWER-clocked C1 is never helped
    assert o["verdict"] == "GO" and o["C1"]["lb_clock_adj"] == o["C1"]["lb"]
    bad = _fake(c1=1.00)                                                     # a fast C2 never rules
    for rep in bad["blocks"]["S1"]["reps"]:
        rep["replays_ms"]["C2"] = [x / 1.5 for x in rep["replays_ms"]["P"]]
    o = verdict(bad)
    assert o["verdict"] == "NO_GO" and o["reported"]["C2"]["median"] > 1.4
    bad = _fake(c1=1.00)
    for rep in bad["blocks"]["S0R"]["reps"]:                                 # nor does the reproduction block
        rep["replays_ms"]["C1"] = [x / 1.6 for x in rep["replays_ms"]["P"]]
    assert v(bad) == "NO_GO"
    assert v(_fake(self_ratio=1.03)) == "VOID"                               # the self-pair is off: no instrument
    bad = _fake()
    bad["blocks"]["S1"]["reps"][5]["p2_is_p"] = False
    assert v(bad) == "VOID"                                                  # P is not deterministic
    for arm, block in (("C1", "S1"), ("C1", "S0R"), ("P", "S1")):
        bad = _fake()
        bad["blocks"][block]["reps"][7]["identical"][arm] = False
        o = verdict(bad)
        assert o["verdict"] == "NO_GO" and o["contract_findings"][0]["arm"] == arm, (arm, block)
    bad = _fake()
    bad["blocks"]["S1"]["reps"][7]["identical"]["C3"] = False                # a reported arm's finding never rules
    o = verdict(bad)
    assert o["verdict"] == "GO" and o["contract_findings"][0]["arm"] == "C3"
    assert o["predictions"]["contract: every arm keeps T's bits on every rep"] == "MISSED"
    assert v(_fake(rr_sha256="0" * 64)) == "VOID"
    assert v(_fake(rows_sha256={"S0": ROWS_SHA256["S0"], "S1": "0" * 64})) == "VOID"   # not S1's picks
    assert v(_fake(device="NVIDIA RTX A2000 12GB", cc=[8, 6])) == "VOID"     # an A2000 is never a reading
    assert v(_fake(layers=40)) == "VOID"
    bad = _fake()
    bad["plans"]["C1"] = [dict(TODAY), dict(TODAY), P_LUT]
    assert v(bad) == "VOID"                                                  # C1 is not P156's carried pair
    bad = _fake()
    bad["plans"]["T"] = [dict(TODAY), dict(TODAY), P_LUT]
    assert v(bad) == "VOID"                                                  # T is not the tree
    assert v(_fake(spacer=None)) == "VOID"                                   # no spacer
    bad = _fake()
    bad["blocks"]["S1"]["reps"][3]["spacer_replays"] = 0
    assert v(bad) == "VOID"                                                  # a rep without its spacer replays
    bad = _fake()
    bad["blocks"]["S1"]["reps"][3]["order"] = bad["blocks"]["S1"]["reps"][4]["order"]
    assert v(bad) == "VOID"                                                  # not the Williams row
    bad = _fake()
    bad["blocks"]["S1"]["reps"].pop()
    assert v(bad) == "VOID"                                                  # a rep short
    bad = _fake()
    bad["blocks"]["S1"]["reps"][9]["replays_ms"]["C2"] = bad["blocks"]["S1"]["reps"][9]["replays_ms"]["C2"][:5]
    assert v(bad) == "VOID"                                                  # malformed replays
    bad = _fake()
    bad["blocks"]["S1"]["n_real"][0][0] += 1
    assert v(bad) == "VOID"                                                  # not P153's S1 total
    bad = _fake()
    bad["blocks"] = {"S0R": bad["blocks"]["S0R"], "S1": bad["blocks"]["S1"]}
    assert v(bad) == "VOID"                                                  # the blocks out of their registered order
    bad = _fake()
    bad["blocks"]["S1"]["tile_faults"] = [{"step": 3, "layer": 1, "fault": "x"}]
    assert v(bad) == "VOID"                                                  # the table is not a prefix
    assert v(_fake(limit=480.0)) == "VOID"                                   # a timed replay under a 480 W limit
    bad = _fake()
    bad["clocks"]["error"] = "OSError: libnvidia-ml.so.1"
    assert v(bad) == "VOID"                                                  # no clocks: the clock bound is unread
    sk = _fake(shakedown=True, layers=4, device="NVIDIA RTX A2000 12GB", cc=[8, 6], limit=70.0)
    for b in BLOCKS:
        sk["blocks"][b]["reps"] = sk["blocks"][b]["reps"][:6]
        sk["blocks"][b]["steps"] = sk["blocks"][b]["steps"][:6]
    o = verdict(sk)
    assert o["verdict"] == "SHAKEDOWN", o["why"]                             # never a reading, every check run
    assert set(o["predictions"].values()) == {"UNREAD (a shakedown)"}
    w = williams(6)
    assert all(sorted(row) == list(range(6)) for row in w)
    assert all(sorted(row[i] for row in w) == list(range(6)) for i in range(6))            # each position once
    assert len({(row[i], row[i + 1]) for row in w for i in range(5)}) == 30                # each ordered pair once
    assert all(len(s["steps"]) % len(ARMS) == 0 for s in BLOCKS.values())                  # whole squares
    print("p160_reduce self-test OK (41 checks)")


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
            print(f"p160_reduce self-test FAILED: {e}")
            return 1
        return 0
    if not (a.bench and a.out):
        ap.error("--bench and --out are required")
    with open(a.bench, encoding="utf-8") as f:
        rec = json.load(f)
    out = verdict(rec)
    with open(a.out, "w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(out, indent=1, sort_keys=True) + "\n")
    print(f"P160 {out['verdict']} " + json.dumps({"why": out["why"], "C1": out.get("C1")}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
