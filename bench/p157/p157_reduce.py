# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""P157's reducer (bench/p157/PREREG-p157.md): K25's stages knocked out one at a time, at today's plan on Qwen3-30B-A3B's
W16 shapes, timed on S0's real router picks (P153's router replay).

Pure Python, no torch. The registered rules live here, and the box imports them (``p157_bench.py``):
- :func:`order`: each rep's capture and replay order (a Williams design);
- :func:`dose`: a ladder rung's DRAM dose per graph, from the slabs each call touches;
- :func:`band`: the registered bands of a 2-SE interval of r;
- :func:`verdict`: the read: the instrument (VOID), the per-arm gates and labels, the ceilings, A12's additivity,
  r_dram from the ARMS and its band, the ladder's out-of-sample check of that band, and the reported shape.
Everything is re-derived from the bench's raw replay times (``replays_ms``) and its recorded static and occupancy facts.

    python p157_reduce.py --bench bench_p157.json --out verdict_p157.json [--ncu ncu_p157.json]
    python p157_reduce.py --self-test
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
E, TOPK, TOKENS, LAYERS, STEPS = 128, 8, 16, 48, 128
SHAPES = {"gu": (1536, 2048), "dn": (2048, 768)}           # (N, K): gate_up [2 x 768, 2048], down [2048, 768]
TODAY = {"block_n": 32, "kc": 64, "warps": 4, "stages": 3}  # e4b's _K25_PLAN; lut "tree" and dot_bf16 False are fixed
CONFIRM_STEPS = tuple(range(58, 128))      # 70 router-replay steps, one paired rep each: 5 whole Williams squares of
                                           # the 14 arms (P157 selects nothing, so no step need be held out)
ITERS = 10                                 # timed replays per graph
SPACER = "S"                               # K25's calls with their own buffers, replayed untimed before every timed
                                           # replay (the same predecessor for every arm)
BOOT_B, BOOT_SEED = 2000, 157              # the median's SE: a bootstrap of 2,000 resamples from one seeded generator
SELF_BAND = (0.99, 1.01)                   # A0's self-pair (T2 / A0) median, else the instrument is VOID
CLOCK_TOL = 0.02                           # an arm's median SM clock off A0p's by more than this is clock-confounded
WIDE = 0.5                                 # a ladder rung whose r interval is wider than this is uninformative
NCU_DOSE_TOL, NCU_A3_TOL, NCU_LOW_BAR = 0.05, 0.02, 0.05
#: The arms, in the registered order (each rep rotates it). A0 is K25 as served, T2 is A0 again (the self-pair), the rest
#: are p157_arms.py's.
ARMS = ("A0", "T2", "A0p", "A1", "A2b", "A12", "A3", "AL", "AC", "AC4", "AC16", "ACX", "ACX0", "AE")
MASKS = {"AC": 0, "AC4": 3, "AC16": 15, "ACX": 127, "AL": 0}
LADDER = ("AC", "AC4", "AC16", "ACX")
CUBINS = {"ladder": LADDER + ("ACX0",), "loads": ("A3", "AL")}   # one cubin each: r's differences are data-only
COPY_TOL = 0.01                            # |ACX / A0p - 1| beyond this: r is read as "about the RESID copy"
TEST_RUNGS = ("AC4", "AC16")               # AC anchors onchip (g = 0); ACX (g = 1) is the copy check
PINNED = ("A1", "A2b", "A12", "A3", "AL", "AC", "AC4", "AC16", "ACX", "ACX0", "AE")   # occupancy held at A0's when it differs
PARITY = {"A1": "A0", "A2b": "A0", "A12": "A0", "A3": "A0"}                     # the arms whose loads must be A0's
#: What each ceiling prices; A2b's is never "staging" alone (A2a, which would have split them, is NOT BUILT).
CEILINGS = {"A1": "the select-tree decode", "A2b": "the MMA and its operand staging",
            "A12": "the decode and the MMA with its staging", "A3": "everything but the loads"}
BANDS = ("NOT_ADDITIVE", "THROUGHPUT_BOUND", "LATENCY_BOUND", "MIXED", "UNRESOLVED")
#: The registered predictions (PREREG, "Predictions"); each carries its HELD and MISSED consequences there.
PRED = {"a0_ms": (11.5, 13.5),               # P1: A0 per graph: P155 read 12.075 ms, P156's box 12.87 ms
        "copy": (0.995, 1.005),              # P2: A0p / A0 and ACX / A0p
        "r_bands": ("LATENCY_BOUND", "MIXED"),   # P3: r_dram's band
        "decode_ms": (1.9, 6.0),             # P4: A0p - A1: at least P156's measured saving from a cheaper decode
        "mma_ms": (0.0, 1.5),                # P5: A0p - A2b
        "mem_ms": (2.0, 5.5),                # P6: mem_dram = A3 - AL
        "onchip_ms": (5.0, 10.0)}            # P7: onchip = AC - AE; P8: the ladder agrees; P9: no rung is faster than
                                             # the rung below it (AC <= AC4 <= AC16 <= ACX within 2 SE)


def williams(n):
    """Williams' design for an even n: n orders, each a Latin row, every ordered pair of arms adjacent exactly once."""
    first, lo, hi = [0], 1, n - 1
    while len(first) < n:
        first.append(lo)
        lo += 1
        if len(first) < n:
            first.append(hi)
            hi -= 1
    return [[(x + i) % n for x in first] for i in range(n)]


def order(arms, r):
    """Rep r's capture and replay order: row r mod n of Williams' design over the arms (14 arms over 70 reps, 5 whole
    squares: each arm takes each position 5 times, and every ordered neighbour pair recurs 5 times). What each defence
    removes: the spacer S is every timed replay's IMMEDIATE predecessor, so no arm inherits a light or heavy
    neighbour's power and clock headroom; the design balances each arm's position in the rep (the clock and thermal
    drift across a rep) and its second-order neighbour (the arm timed before S)."""
    return [arms[j] for j in williams(len(arms))[r % len(arms)]]


def slab_bytes(proj):
    """One expert's slab in a call: its NF4 bytes and its fp32 absmax."""
    n, k = SHAPES[proj]
    return n * k // 2 + n * (k // 64) * 4


def dose(slabs, mask):
    """A rung's DRAM bytes per graph: sum over the graph's calls of (distinct masked experts) x (slab bytes). ``slabs``
    is the rep's record, {str(mask): {"gu": [per layer], "dn": [per layer]}}."""
    s = slabs[str(mask)]
    return sum(n * slab_bytes(p) for p in ("gu", "dn") for n in s[p])


def se_median(q, b=BOOT_B, seed=BOOT_SEED):
    """The standard error of the MEDIAN of ``q`` (P155's rule): the larger of the normal-theory value,
    sqrt(pi / 2) x sd / sqrt(n), and the standard deviation of the medians of ``b`` seeded resamples."""
    n = len(q)
    if n < 2:
        return math.inf
    normal = math.sqrt(math.pi / 2) * statistics.stdev(q) / math.sqrt(n)
    rng = random.Random(seed)
    boot = statistics.stdev(statistics.median([q[int(rng.random() * n)] for _ in range(n)]) for _ in range(b))
    return max(normal, boot)


def summarize(q):
    q = list(q)
    med = statistics.median(q)
    se = se_median(q)
    return {"n": len(q), "median": med, "se": se, "lo": med - 2 * se, "hi": med + 2 * se}


def band(lo, hi):
    """The registered bands of r's 2-SE interval [lo, hi], in order: entirely below 0 or above 1 is NOT_ADDITIVE;
    entirely at or below 0.25 THROUGHPUT_BOUND (the stages overlap); entirely at or above 0.75 LATENCY_BOUND (they
    serialize); inside [0.25, 0.75] MIXED; anything else UNRESOLVED."""
    if hi < 0 or lo > 1:
        return "NOT_ADDITIVE"
    if hi <= 0.25:
        return "THROUGHPUT_BOUND"
    if lo >= 0.75:
        return "LATENCY_BOUND"
    if lo >= 0.25 and hi <= 0.75:
        return "MIXED"
    return "UNRESOLVED"


def _num(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x) and x > 0


def _good(xs, n=ITERS):
    return isinstance(xs, list) and len(xs) == n and all(_num(x) for x in xs)


def rep_ms(rep):
    return {arm: statistics.median(v) for arm, v in rep["replays_ms"].items()}


def _structure(rec, why):
    """Every VOID that is a property of the record's shape, inputs and instrument; appends to ``why``."""
    shake = bool(rec.get("shakedown"))
    if rec.get("rr_sha256") != RR_SHA256:
        why.append(f"router replay sha256 {rec.get('rr_sha256')} is not the pinned {RR_SHA256[:8]}")
    if rec.get("s0_sha256") != S0_SHA256:
        why.append(f"S0's picks hash {rec.get('s0_sha256')}, pinned {S0_SHA256[:8]}")
    if rec.get("today") != TODAY:
        why.append("today's plan is not the registered one")
    if list(rec.get("arms") or []) != list(ARMS):
        why.append(f"arms {rec.get('arms')}, registered {list(ARMS)}")
    if rec.get("iters") != ITERS:
        why.append(f"{rec.get('iters')} replays per graph, registered {ITERS}")
    if rec.get("spacer") != SPACER:
        why.append(f"spacer {rec.get('spacer')}, registered {SPACER}")
    if rec.get("poisoned"):
        why.append(f"the CUDA context was poisoned: {rec['poisoned']}")
    cs = rec.get("confirm_steps") or []
    if not shake:
        if "5090" not in str(rec.get("device", "")) or rec.get("cc") != [12, 0]:
            why.append(f"device {rec.get('device')} cc {rec.get('cc')}: the lane registers an RTX 5090 (sm_120)")
        if rec.get("layers") != LAYERS:
            why.append(f"{rec.get('layers')} layers timed, registered {LAYERS}")
        if cs != list(CONFIRM_STEPS):
            why.append("the steps are not the registered held-out steps")
    elif not (cs and cs == list(CONFIRM_STEPS[:len(cs)]) and isinstance(rec.get("layers"), int)
              and 1 <= rec["layers"] <= LAYERS):
        why.append("a shakedown's steps and layers must be a prefix of the registered ones")
    reps = rec.get("reps") or []
    if len(reps) != len(cs):
        why.append(f"{len(reps)} reps for {len(cs)} steps")
    for i, rp in enumerate(reps):
        if rp.get("step") != (cs[i] if i < len(cs) else None):
            why.append(f"rep {i} times step {rp.get('step')}")
            break
        if rp.get("order") != order(ARMS, i):
            why.append(f"rep {i}'s order is not the registered design")
            break
        if set(rp.get("replays_ms", {})) != set(ARMS) or not all(_good(v) for v in rp["replays_ms"].values()):
            why.append(f"rep {i} lacks {ITERS} good replays of every arm")
            break
        idn = rp.get("identical") or {}
        for arm in ("T2", "A0p", "ACX0"):
            if idn.get(arm) is not True:
                why.append(f"rep {i}: {arm}'s outputs are not bit-identical to A0's")
        sl = rp.get("slabs") or {}
        if set(sl) != {str(m) for m in (0, 3, 15, 127)} or any(len(sl[m].get(p, [])) != rec.get("layers")
                                                                 for m in sl for p in ("gu", "dn")):
            why.append(f"rep {i} lacks the slab counts of every rung")
            break
        if sl["127"]["gu"] != sl["127"]["dn"] or (rec.get("n_real") and
                                                  sl["127"]["gu"] != rec["n_real"][CONFIRM_STEPS.index(rp["step"])]):
            why.append(f"rep {i}: the identity rung's slabs are not the routing's distinct experts")
            break
    st = rec.get("static") or {}
    for proj in ("gu", "dn"):
        a0, a0p = (st.get("A0") or {}).get(proj) or {}, (st.get("A0p") or {}).get(proj) or {}
        if not a0.get("sass_sha256") or a0.get("sass_sha256") != a0p.get("sass_sha256"):
            why.append(f"{proj}: the placebo A0p's SASS is not K25's")
        for name, arms in CUBINS.items():
            hs = {((st.get(r) or {}).get(proj) or {}).get("sass_sha256") for r in arms}
            if len(hs) != 1 or None in hs:
                why.append(f"{proj}: the {name} arms {arms} are not one cubin")


def _same_tool(a, b):
    return a.get("triton") == b.get("triton") and a.get("ptxas") == b.get("ptxas") and a.get("arch") == b.get("arch")


def gates(rec):
    """Per-arm static gates, re-derived from the recorded counts. Returns {arm: [failures]} (empty = passes)."""
    st, fails = rec.get("static") or {}, {}
    for arm in ARMS[2:]:
        fails[arm] = []
        for proj in ("gu", "dn"):
            r, a0 = (st.get(arm) or {}).get(proj), (st.get("A0") or {}).get(proj)
            if not r or not a0 or not _same_tool(r, a0):
                fails[arm].append(f"{proj}: no static record on A0's toolchain")
                continue
            if arm in PARITY:
                if r.get("unrolled_away"):               # ptxas unrolled the light loop: whole-function totals
                    want = {op: n * r["trip"] + a0["loads_outside_loop"].get(op, 0)
                            for op, n in a0["loop_loads"].items()}
                    for op, n in a0["loads_outside_loop"].items():
                        want.setdefault(op, n)
                    if r.get("loads_in_function") != want:
                        fails[arm].append(f"{proj}: function loads {r.get('loads_in_function')} are not A0's "
                                          f"loop x {r.get('trip')} plus A0's outside loads {want}")
                elif r.get("loads_per_weight") != a0.get("loads_per_weight"):
                    fails[arm].append(f"{proj}: loads per weight differ from A0's")
            if arm == "A2b" and not r.get("lane_instr_per_weight", math.inf) < a0.get("lane_instr_per_weight", 0):
                fails[arm].append(f"{proj}: A2b is not strictly cheaper than A0 per weight")
            if arm in CUBINS["ladder"] and r.get("loop_opcodes") != (st["A0p"].get(proj) or {}).get(
                    "loop_opcodes"):
                fails[arm].append(f"{proj}: the rung's loop opcodes are not A0p's (math parity)")
            if arm == "AL":
                a3 = (st.get("A3") or {}).get(proj) or {}
                key = "loads_in_function" if a3.get("unrolled_away") else "loop_opcodes"
                if r.get(key) != a3.get(key):
                    fails[arm].append(f"{proj}: AL's {key} are not A3's")
    return fails


def labels(rec):
    """Per-arm confounds, LABELLED beside the arm's numbers (never a reason to drop them): occupancy (the verified
    CTAs per SM, the carveout or the graph node's shared bytes differ from A0's) and clocks (median SM clock off A0p's
    by more than 2 %, either way)."""
    occ, out = rec.get("occupancy") or {}, {arm: [] for arm in ARMS[2:]}
    for arm in out:
        for proj in ("gu", "dn"):
            o, a0 = (occ.get(arm) or {}).get(proj) or {}, (occ.get("A0") or {}).get(proj) or {}
            if not o or not a0:
                out[arm].append(f"{proj}: occupancy unread")
                continue
            if o.get("ctas_per_sm") != a0.get("ctas_per_sm"):
                out[arm].append(f"{proj}: {o.get('ctas_per_sm')} CTAs/SM against A0's {a0.get('ctas_per_sm')}")
            if o.get("carveout") != a0.get("carveout"):
                out[arm].append(f"{proj}: carveout {o.get('carveout')} against A0's {a0.get('carveout')}")
            nodes = o.get("node_shared") or []
            if not nodes or any(n != o.get("launch_shared") for n in nodes):
                out[arm].append(f"{proj}: the graph nodes' sharedMemBytes {sorted(set(nodes))} are not the launch's "
                                f"{o.get('launch_shared')}")
    clk = rec.get("clocks") or {}
    base = (clk.get("A0p") or {}).get("sm_mhz_median")
    for arm in out:
        c = (clk.get(arm) or {}).get("sm_mhz_median")
        if not (_num(base) and _num(c)):
            out[arm].append("clock unread")
        elif abs(c / base - 1) > CLOCK_TOL:
            out[arm].append(f"clock-confounded: SM {c:.0f} MHz against A0p's {base:.0f}")
    return out


def _per_rep(ms, f):
    return [f(m) for m in ms]


def r_of(full, mem, onchip):
    lo_, hi_ = min(mem, onchip), max(mem, onchip)
    return (full - hi_) / lo_


def ladder(ms, reps, onchip, mem, scale=None):
    """Each rung's implied r per rep, from that rep's own arms and rung time (the arms' noise is carried):
    T_k - AE = max(onchip, g_k mem) + r min(onchip, g_k mem), with g_k = (f_k - f_AC) / (1 - f_AC) the rung's DRAM
    dose measured FROM AC's (AC anchors onchip; mem = A3 - AL prices the bytes above AL's, which is AC's dose). ``scale``
    {rung: c} carries ncu's correction of the estimated dose."""
    out = {}
    for k in LADDER:
        rs, gs = [], []
        for i, (m, rp) in enumerate(zip(ms, reps)):
            f = {r: dose(rp["slabs"], MASKS[r]) / dose(rp["slabs"], 127) * ((scale or {}).get(r, 1.0)) for r in LADDER}
            g = (f[k] - f["AC"]) / (1 - f["AC"])
            gs.append(g)
            lo_, hi_ = min(onchip[i], g * mem[i]), max(onchip[i], g * mem[i])
            rs.append((m[k] - m["AE"] - hi_) / lo_ if lo_ > 0 else math.nan)
        row = {"g_median": statistics.median(gs), "role": "anchor" if k == "AC" else "copy" if k == "ACX" else "test"}
        if all(math.isfinite(x) for x in rs):
            row.update(summarize(rs))
            row["width"] = row["hi"] - row["lo"]
            row["informative"] = row["role"] == "test" and row["width"] <= WIDE
            row["band"] = band(row["lo"], row["hi"])
        else:
            row.update({"informative": False, "band": "UNRESOLVED", "note": "zero dose: no r (the anchor)"})
        out[k] = row
    return out


def regime(b, lo):
    """A band as the ladder check reads it: NOT_ADDITIVE splits by side, since an interval entirely above 1 is serial
    or worse and one entirely below 0 is overlapped or better. A perfectly serial kernel has r = 1 exactly, on that
    edge, so without the split small noise alone would read it "disagree"."""
    if b == "NOT_ADDITIVE":
        return "ABOVE_1" if lo > 1 else "BELOW_0"
    return b


#: The regimes each arm call is consistent with (UNRESOLVED is consistent with every call).
COMPATIBLE = {"LATENCY_BOUND": {"LATENCY_BOUND", "ABOVE_1"}, "ABOVE_1": {"LATENCY_BOUND", "ABOVE_1"},
              "THROUGHPUT_BOUND": {"THROUGHPUT_BOUND", "BELOW_0"}, "BELOW_0": {"THROUGHPUT_BOUND", "BELOW_0"},
              "MIXED": {"MIXED"}}


def ladder_check(rows, call, call_lo):
    """The registered check: the arm call stands unless an informative test rung's r interval lies ENTIRELY in a band
    the call is not consistent with (:data:`COMPATIBLE`; UNRESOLVED never disagrees)."""
    if call not in BANDS[:4]:
        return "NOT_CHECKED (no arm call)"
    ok = COMPATIBLE[regime(call, call_lo)] | {"UNRESOLVED"}
    bad = [k for k in TEST_RUNGS if rows[k]["informative"] and regime(rows[k]["band"], rows[k]["lo"]) not in ok]
    return f"DISAGREE ({', '.join(bad)})" if bad else "AGREE"


def hinge(points):
    """The hinge T = max(a, b f) through the four rungs, least squares over the split (REPORTED only)."""
    pts = sorted(points)
    best = None
    for j in range(len(pts) + 1):
        flat, line = pts[:j], pts[j:]
        a = statistics.fmean(t for _, t in flat) if flat else 0.0
        b = sum(f * t for f, t in line) / sum(f * f for f, _ in line) if line else 0.0
        sse = sum((t - max(a, b * f)) ** 2 for f, t in pts)
        if best is None or sse < best[0]:
            best = (sse, a, b)
    _, a, b = best
    return a, b, (a / b if b > 0 else math.inf)


def verdict(rec, ncu=None):
    why = []
    _structure(rec, why)
    out = {"lane": "P157", "shakedown": bool(rec.get("shakedown")), "device": rec.get("device")}
    if why:
        out.update(state="VOID", why=why)
        return out
    reps = rec["reps"]
    ms = [rep_ms(rp) for rp in reps]
    arm_ms = {arm: summarize([m[arm] for m in ms]) for arm in ARMS}
    self_pair = summarize([m["T2"] / m["A0"] for m in ms])
    self_se = se_median([m["T2"] - m["A0"] for m in ms])
    if not SELF_BAND[0] <= self_pair["median"] <= SELF_BAND[1]:
        out.update(state="VOID", why=[f"self-pair median {self_pair['median']:.4f} outside {SELF_BAND}"])
        return out
    gate, lab = gates(rec), labels(rec)
    full = [m["ACX"] - m["AE"] for m in ms]                 # the ladder's cubin at A0's bytes (one cubin with AC)
    mem = [m["A3"] - m["AL"] for m in ms]
    onchip = [m["AC"] - m["AE"] for m in ms]
    ceilings = {}
    for arm, what in CEILINGS.items():
        d = summarize([m["A0p"] - m[arm] for m in ms])
        d["share"] = summarize([(m["A0p"] - m[arm]) / (m["A0p"] - m["AE"]) for m in ms])
        d.update(prices=what, valid=not gate[arm], labels=lab[arm], gate=gate[arm])
        ceilings[arm] = d
    add = summarize([m["A1"] + m["A2b"] - m["A0p"] - m["A12"] for m in ms])
    add["reading"] = ("ADDITIVE" if add["lo"] <= 0 <= add["hi"] else
                      "SUB_ADDITIVE (the stages overlap)" if add["hi"] < 0 else "SUPER_ADDITIVE")
    # ---- r_dram from the ARMS
    used = ("A3", "AL", "AC", "ACX")
    small = min(statistics.median(mem), statistics.median(onchip)) <= 2 * self_se
    neg = any(min(a, b) <= 0 for a, b in zip(mem, onchip))
    rr = summarize([r_of(f, a, b) for f, a, b in zip(full, mem, onchip)]) if not neg else {}
    r = {"full": summarize(full), "mem_dram": summarize(mem), "onchip": summarize(onchip), "self_pair_se_ms": self_se}
    if small or neg:
        r.update(band="UNRESOLVED", why="small denominator: min(mem_dram, onchip) within 2 self-pair SE of 0"
                 if small else "a rep's min(mem_dram, onchip) is not positive", called=False)
    else:
        r.update(rr, band=band(rr["lo"], rr["hi"]))
        blockers = [f"{a}: {x}" for a in used for x in gate[a]] + [f"{a}: {x}" for a in used for x in lab[a]]
        r["called"] = not blockers
        r["blockers"] = blockers
    # ---- the bytes axis: ncu where the box allowed it, else the router-replay ESTIMATE
    scale, axis = None, "ESTIMATE"
    if ncu and ncu.get("dram_read_bytes"):
        nb = ncu["dram_read_bytes"]
        est = ncu.get("estimate_bytes") or {}
        ok = all(_num(nb.get(k)) and _num(est.get(k)) for k in LADDER + ("A0p", "A3", "AL"))
        checks = {}
        if ok:
            # the weight bytes a rung moved: its DRAM reads less its cubin's non-weight reads (activations, tables,
            # codebook), which are its full-dose arm's reads less that arm's weight estimate
            base, base_l = nb["ACX"] - est["ACX"], nb["A3"] - est["A3"]
            for k in LADDER:
                checks[k] = (nb[k] - base) / est[k]
            checks["AL"] = (nb["AL"] - base_l) / est["AL"]
            checks["A3/A0p"] = nb["A3"] / nb["A0p"]
            checks["AC/A0p"], checks["AL/A0p"] = nb["AC"] / nb["A0p"], nb["AL"] / nb["A0p"]
        passed = ok and all(abs(checks[k] - 1) <= NCU_DOSE_TOL for k in LADDER + ("AL",)) and abs(
            checks["A3/A0p"] - 1) <= NCU_A3_TOL and checks["AC/A0p"] <= NCU_LOW_BAR and checks["AL/A0p"] <= NCU_LOW_BAR
        out["ncu"] = {"checks": checks, "passed": passed}
        if passed:
            scale, axis = {k: checks[k] for k in LADDER}, "NCU-SCALED"
        else:
            axis = "ESTIMATE (ncu disagrees: the ladder is reported, not used)"
    rows = ladder(ms, reps, onchip, mem, scale)
    usable = not axis.startswith("ESTIMATE (ncu disagrees")
    check = ladder_check(rows, r["band"], r.get("lo")) if r.get("called") and usable else "NOT_CHECKED"
    if r.get("called") and check.startswith("DISAGREE"):
        r["called"] = False
        r["why"] = "arms and ladder disagree"
    # ---- the shape, REPORTED only
    fs = [{k: dose(rp["slabs"], MASKS[k]) / dose(rp["slabs"], 127) for k in LADDER} for rp in reps]
    gb = [dose(rp["slabs"], 127) / 1e9 for rp in reps]
    incr = {}
    for a, b in zip(LADDER, LADDER[1:]):
        incr[f"{a}->{b}"] = summarize([(m[b] - m[a]) / ((f[b] - f[a]) * g) for m, f, g in zip(ms, fs, gb)])
    hs = [hinge([(f[k], m[k] - m["AE"]) for k in LADDER]) for m, f in zip(ms, fs)]
    finite = [h for h in hs if math.isfinite(h[2])]
    shape = {"increments_ms_per_gb": incr, "hinge_n": len(finite)}
    if len(finite) > 1:
        shape["hinge_knee_f"] = summarize([h[2] for h in finite])
        shape["hinge_slope_ms_per_gb"] = summarize([h[1] / g for h, g in zip(hs, gb) if math.isfinite(h[2])])
    copy_ = {name: summarize([m[a] / m[b] for m in ms]) for name, (a, b) in
             {"A0p/A0": ("A0p", "A0"), "ACX0/A0p": ("ACX0", "A0p"), "ACX/ACX0": ("ACX", "ACX0"),
              "ACX/A0p": ("ACX", "A0p")}.items()}
    gap = copy_["ACX/A0p"]["median"] - 1
    copy_["gap_within_tol"] = abs(gap) <= COPY_TOL
    if not copy_["gap_within_tol"]:
        r.setdefault("labels", []).append(f"about the RESID copy (copy gap {gap:+.2%})")
    rot = copy_["ACX/ACX0"]
    if not rot["lo"] <= 1 <= rot["hi"]:
        r.setdefault("labels", []).append(f"the rotation itself moves ACX by {rot['median'] - 1:+.2%} (ACX/ACX0)")
    out.update(state="SHAKEDOWN" if rec.get("shakedown") else "READ", arms_ms=arm_ms, self_pair=self_pair,
               copy=copy_, gates=gate, labels=lab, ceilings=ceilings, additivity_a12=add, r_dram=r,
               bytes_axis=axis, ladder=rows, ladder_check=check, shape=shape,
               dose_f=summarize([f["AC"] for f in fs]) | {"AC4": statistics.median(f["AC4"] for f in fs),
                                                          "AC16": statistics.median(f["AC16"] for f in fs)})
    out["predictions"] = grade(out) if not rec.get("shakedown") else "UNREAD (a shakedown)"
    return out


def _within(x, lohi):
    return lohi[0] <= x <= lohi[1]


def grade(v):
    """The registered predictions, HELD / MISSED / UNREAD (PREREG, "Predictions")."""
    a, c = v["arms_ms"], v["ceilings"]
    g = {"P1_a0_ms": _within(a["A0"]["median"], PRED["a0_ms"]),
         "P2_copy": _within(v["copy"]["A0p/A0"]["median"], PRED["copy"]) and v["copy"]["gap_within_tol"],
         "P3_r_band": (v["r_dram"]["band"] in PRED["r_bands"]) if v["r_dram"].get("called") else None,
         "P4_decode_ms": _within(c["A1"]["median"], PRED["decode_ms"]) if c["A1"]["valid"] else None,
         "P5_mma_ms": _within(c["A2b"]["median"], PRED["mma_ms"]) if c["A2b"]["valid"] else None,
         "P6_mem_ms": _within(v["r_dram"]["mem_dram"]["median"], PRED["mem_ms"]),
         "P7_onchip_ms": _within(v["r_dram"]["onchip"]["median"], PRED["onchip_ms"]),
         "P8_ladder": None if v["ladder_check"].startswith("NOT") else v["ladder_check"] == "AGREE",
         "P9_monotone": all(x["hi"] >= 0 for x in v["shape"]["increments_ms_per_gb"].values())}
    return {k: ("UNREAD" if x is None else "HELD" if x else "MISSED") for k, x in g.items()}


# ----------------------------------------------------------------------------------------------------- self-test --

def _fake(seed=0, n=70, layers=4, oc=7.0, dc=4.0, r=1.0, ae=0.4, noise=0.002, rung_r=None, rung_err=0.0,
          shake=False):
    """A synthetic record from the TRUE model, so each self-test case knows its answer and the anchor's algebra is
    tested, not assumed: on-chip time ``oc``, the full DRAM stream's cost ``dc``, overlap ``r``, and a dose f per arm:
    T(f) = AE + max(oc, f dc) + r min(oc, f dc). A3 = AE + 0.8 + dc and AL = AE + 0.8 + f_AC dc (loads only, an L2 leg of
    0.8 ms). ``rung_r`` overrides the test rungs' own r; ``rung_err`` adds a model error (ms) to them."""
    rng = random.Random(seed)
    rec = {"lane": "P157", "shakedown": shake, "device": "NVIDIA GeForce RTX 5090", "cc": [12, 0],
           "rr_sha256": RR_SHA256, "s0_sha256": S0_SHA256, "today": dict(TODAY), "arms": list(ARMS), "iters": ITERS,
           "spacer": SPACER, "layers": LAYERS, "confirm_steps": list(CONFIRM_STEPS[:n]), "reps": []}
    nreal = [[57 + (i + li) % 9 for li in range(LAYERS)] for i in range(len(CONFIRM_STEPS))]
    rec["n_real"] = nreal
    sh = "a" * 64
    rec["static"] = {}
    for arm in ARMS:
        if arm == "T2":
            continue
        rec["static"][arm] = {}
        for proj in ("gu", "dn"):
            rec["static"][arm][proj] = {"triton": "3.4.0", "ptxas": "12.8.93", "arch": "sm_120",
                                        "sass_sha256": sh if arm in ("A0", "A0p") else "b" * 64 if arm in
                                        CUBINS["ladder"] else "e" * 64 if arm in ("A3", "AL") else arm, "loads_per_weight": {"LDGSTS.E.BYPASS.128": 0.25},
                                        "lane_instr_per_weight": {"A2b": 23.8, "A0": 24.1}.get(arm, 10.0),
                                        "loop_opcodes": {"HMMA.1688.F32.TF32": 8} if arm in ("A0", "A0p")
                                        + CUBINS["ladder"] else {"LDGSTS": 3}, "loads_in_function": {"LDGSTS": 42},
                                        "loads_outside_loop": {"LDGSTS": 6}, "loop_loads": {"LDGSTS": 3}}
    rec["occupancy"] = {arm: {p: {"ctas_per_sm": 6, "carveout": -1, "launch_shared": 14592, "node_shared": [14592] * 4}
                              for p in ("gu", "dn")} for arm in ARMS if arm != "T2"}
    rec["clocks"] = {arm: {"sm_mhz_median": 2700.0} for arm in ARMS}
    for i in range(n):
        step = CONFIRM_STEPS[i]
        sl = {str(mk): {p: [min(nreal[i][li], mk + 1) if mk < 127 else nreal[i][li] for li in range(LAYERS)]
                        for p in ("gu", "dn")} for mk in (0, 3, 15)}
        sl["127"] = {p: list(nreal[i]) for p in ("gu", "dn")}
        f = {k: dose(sl, MASKS[k]) / dose(sl, 127) for k in LADDER}

        def t(fd, rr=r):
            return ae + max(oc, fd * dc) + rr * min(oc, fd * dc)
        m = {"A0": t(1), "T2": t(1), "A0p": t(1), "ACX0": t(1), "A1": t(1) - 2.5, "A2b": t(1) - 0.5, "A12": t(1) - 2.8,
             "A3": ae + 0.8 + dc, "AL": ae + 0.8 + f["AC"] * dc, "AE": ae}
        for k in LADDER:
            m[k] = t(f[k], rung_r if (rung_r is not None and k in TEST_RUNGS) else r)
            if k in TEST_RUNGS:
                m[k] += rung_err
        rp = {arm: [m[arm] * (1 + rng.gauss(0, noise)) for _ in range(ITERS)] for arm in ARMS}
        rec["reps"].append({"step": step, "order": order(ARMS, i), "replays_ms": rp, "slabs": sl,
                            "identical": {"T2": True, "A0p": True, "ACX0": True}})
    return rec


def self_test():
    cases = 0

    def expect(name, cond, v=None):
        nonlocal cases
        cases += 1
        if not cond:
            raise AssertionError(f"{name}: {json.dumps(v, default=str)[:1500] if v is not None else ''}")

    v = verdict(_fake(r=1.0))
    expect("serial arms read LATENCY_BOUND and agree", v["state"] == "READ" and v["r_dram"]["band"] == "LATENCY_BOUND"
           and v["r_dram"]["called"] and v["ladder_check"] == "AGREE", v.get("r_dram"))
    v = verdict(_fake(r=0.0))
    expect("overlapped arms read THROUGHPUT_BOUND and agree", v["r_dram"]["band"] == "THROUGHPUT_BOUND"
           and v["ladder_check"] == "AGREE", v["r_dram"])
    v = verdict(_fake(r=0.5))
    expect("half overlap reads MIXED", v["r_dram"]["band"] == "MIXED" and v["ladder_check"] == "AGREE", v["r_dram"])
    # (a) the true curve 5 SE off the line and 20 SE from the hinge, arms serial -> agree
    v = verdict(_fake(r=1.0, rung_r=0.93))
    expect("(a) a rung a little off the line still agrees", v["ladder_check"] == "AGREE", v["ladder"])
    # (b) a rung with no power -> uninformative
    v = verdict(_fake(r=1.0, noise=0.06))
    expect("(b) a noisy rung is uninformative", not v["ladder"]["AC4"]["informative"], v["ladder"]["AC4"])
    # (c) arms serial, rungs on the hinge -> disagree
    v = verdict(_fake(r=1.0, rung_r=0.0))
    expect("(c) arms serial, rungs on the hinge: disagree", v["ladder_check"].startswith("DISAGREE")
           and not v["r_dram"]["called"], v["ladder"])
    # (d) large model error relative to SE, both rungs in the right band -> agree
    v = verdict(_fake(r=1.0, rung_err=0.012, noise=0.0005))
    expect("(d) model error large against SE, right band: agree", v["ladder_check"] == "AGREE", v["ladder"])
    # (e) the anchor: with raw f_k (no anchor) a serial kernel would read AC4 near 1 - f_AC / f_AC4
    rec = _fake(r=1.0)
    rp = rec["reps"][0]
    f = {k: dose(rp["slabs"], MASKS[k]) / dose(rp["slabs"], 127) for k in LADDER}
    m = rep_ms(rp)
    raw = (m["AC4"] - m["AE"] - (m["AC"] - m["AE"])) / (f["AC4"] * (m["A3"] - m["AL"]))
    expect("(e) without the anchor a serial AC4 reads ~1 - f_AC/f_AC4", abs(raw - (1 - f["AC"] / f["AC4"])) < 0.05
           and raw < 0.8, raw)
    # (f) a hinge kernel above the knee: the anchor's bias, about f_AC D / onchip, must not cross a band edge
    v = verdict(_fake(oc=1.0, dc=6.0, r=0.0))
    expect("(f) a hinge above the knee still reads THROUGHPUT_BOUND and agrees", v["r_dram"]["band"] ==
           "THROUGHPUT_BOUND" and v["ladder_check"] == "AGREE" and 0.05 < v["r_dram"]["median"] < 0.15,
           {"r": v["r_dram"], "ladder": v["ladder"]})
    # VOIDs and gates
    bad = _fake()
    bad["reps"][3]["identical"]["A0p"] = False
    expect("A0p not bit-identical -> VOID", verdict(bad)["state"] == "VOID")
    bad = _fake()
    bad["reps"][7]["identical"]["ACX0"] = False
    expect("ACX0 not bit-identical -> VOID", verdict(bad)["state"] == "VOID")
    bad = _fake()
    for rp in bad["reps"]:
        rp["replays_ms"]["ACX"] = [x * 1.02 for x in rp["replays_ms"]["ACX"]]
    v = verdict(bad)
    expect("a 2 % copy gap labels r, never VOIDs", v["state"] == "READ" and any("RESID copy" in x for x in
                                                                               v["r_dram"].get("labels", []))
           and any("rotation itself" in x for x in v["r_dram"].get("labels", [])), v["r_dram"])
    full = statistics.median(rep_ms(rp)["ACX"] - rep_ms(rp)["AE"] for rp in bad["reps"])
    expect("full is ACX - AE (inside the ladder's cubin), not A0p - AE", abs(v["r_dram"]["full"]["median"] - full)
           < 1e-12, v["r_dram"]["full"])
    q = [math.sin(i) + (3.0 if i % 9 == 0 else 0.0) for i in range(70)]
    normal = math.sqrt(math.pi / 2) * statistics.stdev(q) / math.sqrt(len(q))
    expect("the SE is the median's, the larger of normal theory and the bootstrap",
           summarize(q)["se"] >= normal > statistics.stdev(q) / math.sqrt(len(q)), summarize(q))
    bad = _fake()
    bad["clocks"]["ACX"]["sm_mhz_median"] = 2600.0
    v = verdict(bad)
    expect("a label on ACX (full's arm) blocks the call", not v["r_dram"]["called"]
           and any(x.startswith("ACX:") for x in v["r_dram"]["blockers"]), v["r_dram"])
    bad = _fake()
    bad["static"]["AL"]["gu"]["sass_sha256"] = "f" * 64
    expect("A3 and AL not one cubin -> VOID", verdict(bad)["state"] == "VOID")
    bad = _fake()
    bad["static"]["A0p"]["gu"]["sass_sha256"] = "c" * 64
    expect("placebo SASS differs -> VOID", verdict(bad)["state"] == "VOID")
    bad = _fake()
    bad["static"]["AC16"]["dn"]["sass_sha256"] = "d" * 64
    expect("ladder not one cubin -> VOID", verdict(bad)["state"] == "VOID")
    bad = _fake()
    for rp in bad["reps"]:
        rp["replays_ms"]["T2"] = [x * 1.03 for x in rp["replays_ms"]["T2"]]
    expect("self-pair outside its band -> VOID", verdict(bad)["state"] == "VOID")
    bad = _fake()
    del bad["spacer"]
    expect("no spacer -> VOID", verdict(bad)["state"] == "VOID")
    bad = _fake()
    bad["rr_sha256"] = "0" * 64
    expect("wrong routing -> VOID", verdict(bad)["state"] == "VOID")
    bad = _fake()
    bad["reps"][2]["order"] = list(ARMS)
    expect("unregistered order -> VOID", verdict(bad)["state"] == "VOID")
    bad = _fake()
    bad["reps"][0]["slabs"]["127"]["gu"][0] += 1
    expect("identity rung not the routing -> VOID", verdict(bad)["state"] == "VOID")
    bad = _fake()
    bad["static"]["A3"]["gu"]["loads_per_weight"] = {"LDG.E.128": 0.25}
    v = verdict(bad)
    expect("A3 load parity fails -> r_dram not called", v["state"] == "READ" and not v["r_dram"]["called"]
           and not v["ceilings"]["A3"]["valid"], v["r_dram"])
    bad = _fake()
    bad["occupancy"]["AL"]["dn"]["node_shared"] = [14592, 0]
    v = verdict(bad)
    expect("a pad that missed the graph node labels AL and blocks the call", not v["r_dram"]["called"]
           and any("sharedMemBytes" in x for x in v["labels"]["AL"]), v["labels"]["AL"])
    bad = _fake()
    bad["clocks"]["A1"]["sm_mhz_median"] = 2600.0
    v = verdict(bad)
    expect("an arm 3.7 % off A0p's clock is labelled", any("clock-confounded" in x for x in v["labels"]["A1"])
           and v["r_dram"]["called"], v["labels"]["A1"])
    bad = _fake()
    bad["static"]["A2b"]["gu"]["lane_instr_per_weight"] = 24.5
    expect("A2b not cheaper -> its ceiling invalid", not verdict(bad)["ceilings"]["A2b"]["valid"])
    bad = _fake(oc=40.0, dc=0.008)            # every rep's mem_dram positive, its median inside 2 self-pair SE
    v = verdict(bad)
    expect("a near-zero mem_dram is a small denominator", v["r_dram"]["band"] == "UNRESOLVED"
           and not v["r_dram"]["called"] and v["r_dram"]["why"].startswith("small denominator"), v["r_dram"])
    v = verdict(_fake(dc=0.001))
    expect("a rep's non-positive denominator is UNRESOLVED", v["r_dram"]["band"] == "UNRESOLVED"
           and not v["r_dram"]["called"], v["r_dram"])
    rows = {"AC4": {"informative": True, "band": "NOT_ADDITIVE", "lo": 1.0015}, "AC16": {"informative": False}}
    expect("a serial rung just above 1 agrees with LATENCY_BOUND", ladder_check(rows, "LATENCY_BOUND", 0.9) == "AGREE")
    rows["AC4"].update(band="NOT_ADDITIVE", lo=-0.3)
    expect("a rung below 0 disagrees with LATENCY_BOUND", ladder_check(rows, "LATENCY_BOUND", 0.9).startswith(
        "DISAGREE"))
    v = verdict(_fake(r=1.4))
    expect("r above 1 reads NOT_ADDITIVE", v["r_dram"]["band"] == "NOT_ADDITIVE", v["r_dram"])
    w = williams(len(ARMS))
    pairs = [(row[j], row[j + 1]) for row in w for j in range(len(row) - 1)]
    expect("Williams: Latin rows and columns, each ordered pair once", all(sorted(row) == list(range(len(ARMS)))
                                                                         for row in w)
           and all(sorted(col) == list(range(len(ARMS))) for col in zip(*w)) and len(set(pairs)) == len(pairs))
    expect("bands", band(-0.3, -0.1) == "NOT_ADDITIVE" and band(-0.1, 0.2) == "THROUGHPUT_BOUND"
           and band(0.8, 1.0) == "LATENCY_BOUND" and band(0.3, 0.7) == "MIXED" and band(0.2, 0.8) == "UNRESOLVED")
    v = verdict(_fake(shake=True, n=8))
    expect("a shakedown is never graded", v["state"] == "SHAKEDOWN" and v["predictions"] == "UNREAD (a shakedown)", v)
    ncu = {"dram_read_bytes": {}, "estimate_bytes": {}}
    rec = _fake()
    for k in LADDER + ("A0p", "A3", "AL"):
        e = dose(rec["reps"][0]["slabs"], MASKS.get(k, 127))
        ncu["estimate_bytes"][k] = e
        ncu["dram_read_bytes"][k] = e * (1.02 if k == "AC4" else 1.0) + 3e6     # 3 MB of non-weight reads per arm
    v = verdict(rec, ncu)
    expect("ncu within 5 % scales the dose", v["bytes_axis"] == "NCU-SCALED" and v["ncu"]["passed"], v.get("ncu"))
    ncu["dram_read_bytes"]["AC16"] *= 1.2
    v = verdict(rec, ncu)
    expect("ncu off by 20 % leaves the ladder unused", v["bytes_axis"].startswith("ESTIMATE (ncu disagrees")
           and v["ladder_check"] == "NOT_CHECKED", v.get("ncu"))
    print(f"p157_reduce self-test OK ({cases} cases)")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench")
    ap.add_argument("--ncu")
    ap.add_argument("--out")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args(argv)
    if a.self_test:
        try:
            self_test()
        except AssertionError as e:
            print(f"p157_reduce self-test FAILED: {e}")
            return 1
        return 0
    if not (a.bench and a.out):
        ap.error("--bench and --out are required")
    rec = json.load(open(a.bench, encoding="utf-8"))
    ncu = json.load(open(a.ncu, encoding="utf-8")) if a.ncu else None
    v = verdict(rec, ncu)
    with open(a.out, "w", encoding="utf-8", newline="\n") as f:
        json.dump(v, f, indent=1, default=str)
    r = v.get("r_dram", {})
    print(f"P157_VERDICT {v['state']} r_dram={r.get('median', float('nan')):.3f} band={r.get('band')} "
          f"called={r.get('called')} ladder={v.get('ladder_check')} why={v.get('why')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
