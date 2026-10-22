# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""P226's reducer (bench/p226/PREREG-p226.md): the K25 plan at bucket 32. Qwen3-30B-A3B's decode at 32 rows sends
32 x top-8 = 256 routed rows through K25 (``nf4_smallm.gemm_nf4_grouped_smallm``, the decode ``lut="prmt"``), at the
plan ``c1`` (P160's first carried pair, qwen3_moe's served default since P161), which was chosen at 16 rows. Does any
other plan beat c1 at 32 rows?

Pure Python, no torch. The registered rules live here, and the box imports them (``p226_bench.py``):
- **The data:** the box's own capture (``p226_capture.py``): 32 distinct requests at bucket 32, 128 decode steps, its
  picks hashed BEFORE any timing. The bench records the hash it read; this reducer requires it to equal the capture's.
- **Selection (reported, never ruling):** P156's 144-plan grid on each projection separately, on steps 0-63 every 8th
  (8 steps), every plan timed at ``prmt``. Each projection is ranked by :func:`select_ms` (ties by grid index), c1's own
  plans INCLUDED. The ranked lists pair in order, (gate_up 1st, down 1st), (2nd, 2nd), ...; the pair equal to C1 is
  skipped, and the first CARRY pairs are carried as K1-K3 (:func:`select`). The reducer re-derives the carried pairs
  from the recorded sweep and VOIDs a record whose carried pairs differ.
- **Confirmation (the ruling):** steps 64-127, never touched by selection, one paired rep each, in a Williams row over six
  arms: C (c1: the CONTROL), C2 (c1 captured again: the self-pair, the same cubin), K1-K3 (the carried pairs), D (today's
  plan at ``prmt``, reported: c1's own gain at 256 rows). A spacer (C's calls, own outputs) is replayed untimed before
  every timed replay; NVML samples every arm's clocks and enforced power limit (P157 / P160's instrument).
- **The question, fixed before data:** GO when K1's median C/K1, minus 2 SE of the median, raw AND clock-adjusted, is at
  least GO_BAR (1.05). Otherwise NO_GO; c1 stays at 32 rows. A bit moved by any arm against C is a contract finding
  (NO_GO; no timing is ruled on). K2, K3, D, and the byte-floor headroom are reported, never ruling.

    python p226_reduce.py --bench bench_p226.json --capture capture.json --out verdict_p226.json
    python p226_reduce.py --self-test
"""
from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import sys

E, TOPK, TOKENS, LAYERS, STEPS = 128, 8, 32, 48, 128
SHAPES = {"gu": (1536, 2048), "dn": (2048, 768)}           # (N, K): gate_up [2 x 768, 2048], down [2048, 768]
TODAY = {"block_n": 32, "kc": 64, "warps": 4, "stages": 3}  # e4b's today plan; dot_bf16 False is fixed
#: c1, e4b's _K25_PLANS["c1"] (P160's C1, qwen3_moe's served default since P161): the CONTROL
C1 = {"gu": {"block_n": 64, "kc": 256, "warps": 4, "stages": 2}, "dn": {"block_n": 128, "kc": 128, "warps": 8, "stages": 3}}
LUT = "prmt"
GRID = [{"block_n": bn, "kc": kc, "warps": w, "stages": s}
        for bn in (16, 32, 64, 128) for kc in (64, 128, 256) for w in (2, 4, 8) for s in (1, 2, 3, 4)]   # P156's 144
SELECT_STEPS = tuple(range(0, 64, 8))        # 8 steps of the first half
CONFIRM_STEPS = tuple(range(64, 128))        # 64 held-out steps, one paired rep each
CARRY = 3
ITERS = 10                                   # timed replays per graph
ORDER_SEED = 226                             # the sweep's plan order (one shuffle, both projections)
ARMS = ["C", "C2", "K1", "K2", "K3", "D"]
SPACER = "S"                                 # C's calls with their own outputs, replayed untimed before every timed replay
GO_BAR = 1.05                                # K1's median C/K1 minus 2 SE, raw AND clock-adjusted, must reach this
BOOT_B, BOOT_SEED = 2000, 226
SELF_BAND = (0.99, 1.01)                     # C/C2's median, else the instrument is VOID
CLOCK_TOL, MEM_CLOCK_TOL = 0.02, 0.01        # an arm's median SM / memory clock off C's by more than this is labelled
MIN_POWER_W = 575                            # every NVML sample's enforced limit, else VOID
TILE_ROWS = 16                               # the served table's tile height: an expert with r rows takes ceil(r / 16)
COPY_GBPS = 1498.9                           # P141's copy rate, as P154's byte floor uses it
BYTES_PER_EXPERT = {"gu": 1769472, "dn": 884736}   # the NF4 store, 0.5625 B a weight (P154's record)


def plan_key(p):
    return (p["block_n"], p["kc"], p["warps"], p["stages"])


_GRID_INDEX = {plan_key(p): i for i, p in enumerate(GRID)}


def plan_order():
    """The sweep's timing order: GRID indices, one seeded shuffle (drift spreads over the grid, not by plan)."""
    order = list(range(len(GRID)))
    random.Random(ORDER_SEED).shuffle(order)
    return order


def plans_for(carried):
    """Each arm's (gate_up plan, down plan): C and C2 are c1, K1-K3 the carried pairs, D today's plan."""
    p = {"C": (C1["gu"], C1["dn"]), "C2": (C1["gu"], C1["dn"]), "D": (TODAY, TODAY)}
    for i, c in enumerate(carried):
        p[f"K{i + 1}"] = (c["gu"], c["dn"])
    return p


def williams(n):
    """Williams' design for an even n: n orders, each a Latin row, every ordered pair adjacent exactly once (P157's)."""
    first, lo, hi = [0], 1, n - 1
    while len(first) < n:
        first.append(lo)
        lo += 1
        if len(first) < n:
            first.append(hi)
            hi -= 1
    return [[(x + i) % n for x in first] for i in range(n)]


def arm_order(r, arms=ARMS):
    return [arms[j] for j in williams(len(arms))[r % len(arms)]]


def _num(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x) and x > 0


def _good_replays(xs, n=ITERS):
    return isinstance(xs, list) and len(xs) == n and all(_num(x) for x in xs)


def select_ms(step_replays):
    """A plan's selection time: the mean over the selection steps of each step's median replay (P156's)."""
    return statistics.fmean(statistics.median(x) for x in step_replays)


def _eligible(row, n_steps):
    sr = row.get("step_replays_ms")
    return (row.get("identical") is True and not row.get("error") and isinstance(sr, list) and len(sr) == n_steps
            and all(_good_replays(x) for x in sr))


def select(rows, n_steps=len(SELECT_STEPS)):
    """The registered carry rule (module docstring). ``rows``: {"gu": [...], "dn": [...]}, one row per grid plan."""
    ranked = {}
    for proj in ("gu", "dn"):
        ok = [r for r in rows[proj] if _eligible(r, n_steps)]
        ok.sort(key=lambda r: (select_ms(r["step_replays_ms"]), _GRID_INDEX[plan_key(r["plan"])]))
        ranked[proj] = [dict(r["plan"]) for r in ok]
    out = []
    for g, d in zip(ranked["gu"], ranked["dn"]):
        if (g, d) == (C1["gu"], C1["dn"]):
            continue
        out.append({"gu": g, "dn": d})
        if len(out) == CARRY:
            break
    return out


def tiles_for(counts):
    """The served table's tiles for one layer-step: one per TILE_ROWS rows of each expert's routed rows."""
    return sum(-(-c // TILE_ROWS) for c in counts if c > 0)


def floor_ms(unique_total, bytes_per_expert, gbps=COPY_GBPS):
    return unique_total * bytes_per_expert / (gbps * 1e9) * 1e3


def se_median(q, b=BOOT_B, seed=BOOT_SEED):
    """The SE of the MEDIAN (P155's rule): the larger of sqrt(pi / 2) x sd / sqrt(n) and the sd of ``b`` resampled
    medians drawn through one seeded generator's ``random()`` alone."""
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


def _structure(rec, cap, why):
    """Every VOID that is a property of the records' shape, inputs and instrument; appends to ``why``."""
    shake = bool(rec.get("shakedown"))
    layers = rec.get("layers")
    if not isinstance(cap, dict) or cap.get("faults") or not (cap.get("picks") or {}).get("sha256_le_i2"):
        why.append("the capture is missing, faulted or unhashed: there is no registered routing to time")
        return why
    if rec.get("picks_sha256") != cap["picks"]["sha256_le_i2"]:
        why.append("the bench timed picks whose hash is not the capture's")
    if (cap.get("batch"), cap["picks"].get("shape")) != (TOKENS, [LAYERS, TOKENS, STEPS, TOPK]):
        why.append(f"the capture is {cap.get('batch')} rows, picks {cap['picks'].get('shape')}; registered "
                   f"{TOKENS} rows, [{LAYERS}, {TOKENS}, {STEPS}, {TOPK}]")
    if rec.get("arms") != ARMS or rec.get("spacer") != SPACER or rec.get("iters") != ITERS or rec.get("lut") != LUT:
        why.append("the arms, the spacer, the decode or the replay count are not the registered ones")
    if not shake:
        if "5090" not in str(rec.get("device", "")) or rec.get("cc") != [12, 0]:
            why.append(f"device {rec.get('device')} cc {rec.get('cc')}: the lane registers an RTX 5090 (sm_120)")
        if layers != LAYERS:
            why.append(f"{layers} layers timed, registered {LAYERS}")
    elif not isinstance(layers, int) or not 1 <= layers <= LAYERS:
        why.append(f"shakedown layers {layers}")
    # ---- selection: the whole grid swept on the registered steps, and the carried pairs re-derived from it
    sel = rec.get("selection") or {}
    if list(sel.get("steps") or []) != list(SELECT_STEPS)[:len(sel.get("steps") or [])] or not sel.get("steps") or (
            not shake and list(sel["steps"]) != list(SELECT_STEPS)):
        why.append("the selection steps are not the registered ones")
    rows = sel.get("rows") or {}
    for proj in ("gu", "dn"):
        got = sorted(_GRID_INDEX.get(plan_key(r["plan"]), -1) for r in rows.get(proj) or [] if isinstance(r, dict))
        if got != list(range(len(GRID))):
            why.append(f"selection: {proj} does not cover the registered grid once each")
    if not why:
        derived = select(rows, len(sel["steps"]))
        if rec.get("carried") != derived:
            why.append(f"the carried pairs {rec.get('carried')} are not the registered rule's {derived}")
        if len(derived) != CARRY:
            why.append(f"{len(derived)} pairs carried, registered {CARRY}")
        elif rec.get("plans") != {a: [p for p in v] for a, v in plans_for(derived).items()}:
            why.append("the arms' plans are not C1, the carried pairs and today's")
    # ---- confirmation
    conf = rec.get("confirmation") or {}
    steps = conf.get("steps") or []
    if (steps != list(CONFIRM_STEPS)) if not shake else not (steps and steps == list(CONFIRM_STEPS)[:len(steps)]):
        why.append("the confirmation steps are not the registered ones")
    if set(steps) & set(sel.get("steps") or []):
        why.append("a confirmation step was also a selection step")
    if conf.get("tile_faults"):
        why.append(f"tile table faults {conf['tile_faults'][:3]}")
    reps = conf.get("reps") or []
    if len(reps) != len(steps) or not steps:
        why.append(f"{len(reps)} confirmation reps for {len(steps)} steps")
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
        if set(ident) != set(ARMS) - {"C"}:
            why.append(f"rep {r}: identity not recorded for every arm")
            break
        if ident["C2"] is not True:
            why.append(f"rep {r}: C replayed different bits in a second capture: not deterministic")
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


def confirm_read(conf, clocks):
    ms = [rep_ms(rep) for rep in conf["reps"]]
    out = {"self_pair": summarize([m["C"] / m["C2"] for m in ms]),
           "ms": {a: statistics.median(m[a] for m in ms) for a in ARMS}}
    base, mbase = clocks["C"]["sm_mhz_median"], clocks["C"].get("mem_mhz_median")
    for arm in ("K1", "K2", "K3"):
        clk, mclk = clocks[arm]["sm_mhz_median"], clocks[arm].get("mem_mhz_median")
        s = summarize([m["C"] / m[arm] for m in ms])
        s["clock_over_c"] = clk / base
        s["labels"] = [f"clock-confounded: SM {clk:.0f} MHz against C's {base:.0f}"] if abs(clk / base - 1) > CLOCK_TOL else []
        if not (_num(mclk) and _num(mbase)):
            s["mem_clock_over_c"] = None
            s["labels"].append("memory clock unread")
        else:
            s["mem_clock_over_c"] = mclk / mbase
            if abs(mclk / mbase - 1) > MEM_CLOCK_TOL:
                s["labels"].append(f"clock-confounded: memory {mclk:.0f} MHz against C's {mbase:.0f}")
        s["lb_clock_adj"] = s["lb"] * min(1.0, base / clk)      # a faster-clocked arm's clock gain taken away 1:1
        out[arm] = s
    out["D_over_C"] = summarize([m["D"] / m["C"] for m in ms])   # today's plan against c1 at 256 rows (reported)
    out["moved_bits"] = {arm: [rep["step"] for rep in conf["reps"] if rep["identical"][arm] is not True]
                         for arm in ("C2", "K1", "K2", "K3", "D")}
    return out


def headroom(cap, c_ms, layers):
    """Reported: C's both-projection time against the byte floor of the captured distinct experts (P154's method)."""
    per = cap.get("unique_per_step") or []
    if not per or not _num(c_ms):
        return None
    totals = [sum(st[:layers]) for st in per]
    u = statistics.fmean(totals)
    f = sum(floor_ms(u, BYTES_PER_EXPERT[p]) for p in ("gu", "dn"))
    return {"unique_per_step_mean": round(u, 2), "floor_ms": round(f, 4), "c_ms": round(c_ms, 4),
            "fraction_of_copy": round(f / c_ms, 4), "headroom_ms": round(c_ms - f, 4),
            "copy_gbps": COPY_GBPS, "box_copy_gbps": (cap.get("copy_rate") or {}).get("gbps")}


def verdict(rec, cap):
    why = _structure(rec, cap, [])
    shake = bool(rec.get("shakedown"))
    out = {"lane": "P226", "shakedown": shake}
    if why:
        return dict(out, verdict="VOID", why=why)
    read = confirm_read(rec["confirmation"], rec["clocks"])
    out["confirmation"] = read
    sp = read["self_pair"]
    if not SELF_BAND[0] <= sp["median"] <= SELF_BAND[1]:
        return dict(out, verdict="VOID", why=[f"self-pair C/C2 median {sp['median']:.4f} outside {SELF_BAND}"])
    findings = [{"arm": a, "moved_bits_on_steps": read["moved_bits"][a][:8]} for a in read["moved_bits"]
                if read["moved_bits"][a]]
    out["contract_findings"] = findings
    k1 = read["K1"]
    out["K1"] = dict(k1, plan=rec["carried"][0])
    out["reported"] = {"K2": read["K2"], "K3": read["K3"], "D_over_C": read["D_over_C"],
                       "headroom": headroom(cap, read["ms"]["C"], rec["layers"]),
                       "selection_ranks": rec.get("carried"), "compiled": rec.get("compiled")}
    if shake:
        out["verdict"], out["why"] = "SHAKEDOWN", ["a shakedown record is never a reading"]
    elif findings:
        f = findings[0]
        out["verdict"], out["why"] = "NO_GO", [f"{f['arm']} moved a bit against C (steps {f['moved_bits_on_steps']}): a "
                                               "contract finding against grouped-nf4-gemm; no timing is ruled on"]
    elif k1["lb"] >= GO_BAR and k1["lb_clock_adj"] >= GO_BAR:
        out["verdict"], out["why"] = "GO", [f"K1 {k1['median']:.4f}x over c1, lower bound {k1['lb']:.4f} and clock-"
                                            f"adjusted {k1['lb_clock_adj']:.4f} >= {GO_BAR}"]
    else:
        out["verdict"], out["why"] = "NO_GO", [f"K1 {k1['median']:.4f}x over c1, lower bound {k1['lb']:.4f}, clock-"
                                               f"adjusted {k1['lb_clock_adj']:.4f}: not both >= {GO_BAR}; c1 stays at 32 rows"]
    return out


# ------------------------------------------------------------------------------------------------ self-test (mutants)
def _sweep_rows(best_gu=None, best_dn=None, n_steps=len(SELECT_STEPS)):
    rows = {}
    for proj, best in (("gu", best_gu), ("dn", best_dn)):
        rows[proj] = []
        for i, p in enumerate(GRID):
            ms = 10.0 + i * 0.01 - (5.0 if best is not None and p == best else 0.0)
            rows[proj].append({"plan": dict(p), "identical": True, "step_replays_ms": [[ms] * ITERS] * n_steps})
    return rows


def _fake(k1=1.10, noise=0.002, self_ratio=1.0, c_ms=12.0, layers=LAYERS, shakedown=False, clk=None, limit=575.0,
          best_gu=None, best_dn=None, **kw):
    rng = random.Random(9)
    rows = _sweep_rows(best_gu, best_dn)
    carried = select(rows)
    f = {"C": 1.0, "C2": 1 / self_ratio, "K1": 1 / k1, "K2": 1 / 1.02, "K3": 1 / 1.01, "D": 1.3}
    steps = list(CONFIRM_STEPS)
    reps = []
    for r, s in enumerate(steps):
        reps.append({"step": s, "order": arm_order(r), "spacer_replays": ITERS * len(ARMS),
                     "replays_ms": {a: [c_ms * f[a] * (1 + rng.gauss(0, noise)) for _ in range(ITERS)] for a in ARMS},
                     "identical": {a: True for a in ARMS if a != "C"}})
    clocks = {"error": None, "n": 1}
    for a in ARMS:
        clocks[a] = {"sm_mhz_median": (clk or {}).get(a, 2800.0), "mem_mhz_median": 14001.0, "enforced_limit_w": [limit]}
    rec = {"lane": "P226", "shakedown": shakedown, "device": "NVIDIA GeForce RTX 5090", "cc": [12, 0], "layers": layers,
           "picks_sha256": "ab" * 32, "arms": list(ARMS), "spacer": SPACER, "iters": ITERS, "lut": LUT,
           "selection": {"steps": list(SELECT_STEPS), "rows": rows}, "carried": carried,
           "plans": {a: [p for p in v] for a, v in plans_for(carried).items()},
           "confirmation": {"steps": steps, "reps": reps, "tile_faults": []}, "clocks": clocks}
    cap = {"batch": TOKENS, "faults": [], "picks": {"sha256_le_i2": "ab" * 32, "shape": [LAYERS, TOKENS, STEPS, TOPK]},
           "unique_per_step": [[80] * LAYERS for _ in range(STEPS)], "copy_rate": {"gbps": 1500.0}}
    for k, v in kw.items():
        v(rec, cap)
    return rec, cap


def self_test():
    n = 0

    def ok(c, msg):
        nonlocal n
        assert c, msg
        n += 1
    ok(len(GRID) == 144 and C1["gu"] in GRID and C1["dn"] in GRID and TODAY in GRID, "P156's grid holds c1 and today")
    ok(TOKENS * TOPK == 256, "32 rows x top-8 is K25's 256-row bound")
    v = verdict(*_fake())
    ok(v["verdict"] == "GO" and v["K1"]["lb"] >= GO_BAR, f"a clear 1.10x is GO: {v.get('why')}")
    ok(verdict(*_fake(k1=1.03))["verdict"] == "NO_GO", "1.03x is NO_GO")
    ok(verdict(*_fake(self_ratio=1.03))["verdict"] == "VOID", "a self-pair off the band VOIDs")
    ok(verdict(*_fake(clk={"K1": 3000.0}, k1=1.06))["verdict"] == "NO_GO", "a faster clock is taken away: NO_GO")
    ok(verdict(*_fake(limit=400.0))["verdict"] == "VOID", "a capped power limit VOIDs")
    # the carry rule: c1's own pair is skipped, and the rest pair in rank order
    rows = _sweep_rows(best_gu=C1["gu"], best_dn=C1["dn"])
    car = select(rows)
    ok(len(car) == CARRY and all((c["gu"], c["dn"]) != (C1["gu"], C1["dn"]) for c in car), "c1's own pair is never carried")
    rows2 = _sweep_rows(best_gu=C1["gu"], best_dn=GRID[5])
    ok(select(rows2)[0] == {"gu": C1["gu"], "dn": GRID[5]}, "c1's gate_up with a better down is carried first")
    ok(verdict(*_fake(carried=lambda r, c: r.update(carried=list(reversed(r["carried"])))))["verdict"] == "VOID",
       "a carried list other than the rule's VOIDs")
    ok(verdict(*_fake(sha=lambda r, c: r.update(picks_sha256="cd" * 32)))["verdict"] == "VOID",
       "timing picks other than the capture's VOIDs")
    ok(verdict(*_fake(cf=lambda r, c: c.update(faults=["route: x"])))["verdict"] == "VOID", "a faulted capture VOIDs")
    ok(verdict(*_fake(ov=lambda r, c: r["confirmation"]["steps"].__setitem__(0, 0)))["verdict"] == "VOID",
       "a confirmation step reused from selection VOIDs")

    def moved(r, c):
        r["confirmation"]["reps"][3]["identical"]["K1"] = False
    o = verdict(*_fake(mv=moved))
    ok(o["verdict"] == "NO_GO" and o["contract_findings"], "a moved bit is a contract finding, not a timing")
    ok(tiles_for([17, 16, 1, 0, 32, 33]) == 2 + 1 + 1 + 2 + 3, "an expert with r rows takes ceil(r / 16) tiles")
    h = verdict(*_fake())["reported"]["headroom"]
    want = 80 * LAYERS * sum(BYTES_PER_EXPERT.values()) / (COPY_GBPS * 1e9) * 1e3
    ok(abs(h["floor_ms"] - want) < 1e-3 and h["headroom_ms"] < h["c_ms"], f"the byte floor is P154's method: {h}")
    ok(verdict(*_fake(shakedown=True, layers=4))["verdict"] == "SHAKEDOWN", "a shakedown is never a reading")
    print(f"p226_reduce self-test OK ({n} checks)")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--bench")
    ap.add_argument("--capture")
    ap.add_argument("--out")
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--check-args", action="store_true", help="apply the argument checks a real run applies, then stop")
    a = ap.parse_args(argv)
    if a.self_test:
        return self_test()
    if not (a.bench and a.capture and a.out):
        ap.error("--bench, --capture and --out (or --self-test)")
    if a.check_args:
        print("ARGS OK")
        return 0
    try:
        cap = json.load(open(a.capture, encoding="utf-8"))
    except (OSError, ValueError):
        cap = None
    out = verdict(json.load(open(a.bench, encoding="utf-8")), cap)
    with open(a.out, "w", encoding="utf-8", newline="\n") as f:
        json.dump(out, f, indent=1)
    print(f"P226 {out['verdict']} {json.dumps(out.get('why'))[:400]}")
    return 0 if out["verdict"] in ("GO", "NO_GO") else 27


if __name__ == "__main__":
    sys.exit(main())
