#!/usr/bin/env python3
"""Lane P131's reducer (bench/p131/PREREG-p131.md; e4b#846): K19 over a split tile table (``E4B_K19_PREFILL_BIG_M``,
e4b#1625 over grouped-nf4-gemm#542), read for speed on the served first-chunk prefill graph (Phase A) with its FUNCTION
gate deciding the quality branch: bitwise, or P117's teacher-forced instrument (Phase B) against R's own neutral
perturbations. Phase B's rule is P117's as P130 ran it (the functions below are P130's, byte for byte in substance).

Input: ``proc_<n>.json`` for every registered process (``p131_box.py``) and ``summary.txt`` (the premise line) in the
run directory. Every process captures the three arms (``ARMS``: the knob 0, 32 and 64) and times them interleaved.
Amendment 2: process 0 is the WARM-UP -- it runs Phase B and a speed pass whose timing is REPORTED ONLY (never in the span
guard, the gains, the m32-vs-m64 order or any bar); processes 1..N are the scored speed processes. Every integrity check
below (the records, engagement, capture order, refusals, the decode check, DETERMINISM and FUNCTION) reads every process,
0 included. ``--prove`` reads the proof's registered sizes.

**Lane VOID**, any of:
- a registered process record missing; another e4b or grouped-nf4-gemm commit; another model revision; a process off its
  registered Phase B role;
- a process whose build did not resolve the release's prefill defaults (``prefill_glue`` = mode 1 from the family
  allowlist: e4b#1615);
- the processes read different windows, or sizes other than the mode's;
- wrong speed engagement on a captured graph: ``T``, the capture's eager forwards (``CAPTURE_FORWARDS``), and per
  capture the deltas route ``int4_k19|gt256`` = forwards x layers and nothing else, K19's dispatch
  ``chained|lean|gt256`` (m0) or ``split|lean|gt256`` (m32, m64) = forwards x layers and nothing else, and the prefill
  folds = forwards x ``fold_table``; a capture order other than ``capture_order(n)`` (m0 m32 m64 rotated by n - 1, so
  over the processes every arm is captured in every position); the decode check missing; a speed record with other
  than ``rounds`` x windows timed
  replays per graph, or FUNCTION or digests over other than every speed window;
- the control's graph refused (an m0 capture).

**Per tall arm** (m32, m64), the first of these that applies:
- **REFUSED:** a capture of the arm refused in any process. A defect: the server's prefill graph would stand down.
- **DECODE_TOUCHED:** in any process, the 64-token eager forward under the arm (512 routed rows, a 64-row decode step's)
  shows a ``split`` table above 256 rows, or K19 calls above 256 rows other than m0's in the same process. A defect: the
  knob reaches decode. Which table m0 itself takes is ``E4B_INT4_WIDE_TILES``'s choice (on Qwen3-30B, 128 experts x 512
  rows is lane P122's ``auto`` bound, so the one-launch table), not this lane's.
- **HELD_DETERMINISM:** two processes digest a speed window's m0 replay differently.
- **NOISY:** the scored processes' m0 medians span more than ``NOISY_SPAN``.
- **Speed:** per scored process ``g = median(m0) / median(arm)``; over them, the two-sided 95 % t-interval of the mean
  of ``ln g`` (``T975``). FASTER iff its lower end is at least ``ln(MIN_GAIN)``.
- **Quality:** FUNCTION reads the arm's replay against m0's on every speed window of every process (logits and every
  layer's staged K and V). **BITWISE** iff ``torch.equal`` everywhere: the arm is the same arithmetic and needs no
  quality read. Otherwise the TOLERANCE branch: Phase B's subject (S32 or S64) by P117's rule: **QUALITY_VOID** on any
  integrity fault, **AT_PARITY** iff it passes the bar against the floor, **COST** otherwise.
- **LICENSED** iff FASTER and (BITWISE or AT_PARITY); else **HELD** naming ``NOT_FASTER`` and/or ``COST``, or
  ``QUALITY_VOID``.

**The default** (``DEFAULT_<arm>``): the licensed arm, or if both are licensed the faster by the paired per-process
interval of ``ln(median(m32) / median(m64))`` -- m64 iff its lower end is above 0, m32 iff its upper end is below 0, and
m32 (the shorter tall tile: the smaller register footprint) when the interval holds 0. ``DEFAULT_NONE`` when neither is.

Reported, never gated: every ratio and interval, the savings in ms, the predictions' HELD/MISSED, the eager medians,
Phase B's stats for every arm and whether each subject's log-probs were ``torch.equal`` to R's, the warm-up's speed
(``warmup``), and every process's GPU state at the start and end of its speed pass (``gpu_state``, Amendment 2).

    python p131_reduce.py --dir RUN_DIR --out verdict.json [--e4b-sha SHA] [--prove] [--gate-corpus SHA]
    python p131_reduce.py --self-test
"""
from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
from pathlib import Path

GNF4_SHA = "57f53e19fe65eeef4b7ed26f97d7bd8fdd87f0eb"   # grouped-nf4-gemm #542's merge commit
REVS = {"Qwen/Qwen3-30B-A3B": "ad44e777bcd18fa416d9da3bd8f70d33ebb85d39"}
MODES = {"reading": {"procs": 8, "windows": 64, "cont": 128, "prompt": 512, "speed_windows": 16, "rounds": 12, "warm": 2},
         "proof": {"procs": 4, "windows": 16, "cont": 32, "prompt": 512, "speed_windows": 4, "rounds": 3, "warm": 1}}
PHASE_B = {0: "all"}                 # Amendment 2: Phase B runs in the warm-up process 0
WARMUP = 0                           # its speed is reported only; the scored processes are 1..procs
CAPTURE_FORWARDS = 5                              # enable_prefill_graph: 2 warm-up forwards, the capture, 2 checks
DECODE_TOKENS = 64                                # 64 x top-8 = 512 routed rows: a 64-row decode step's
MIN_GAIN, NOISY_SPAN = 1.02, 1.02
T975 = {1: 12.7062, 2: 4.3027, 3: 3.1824, 4: 2.7764, 5: 2.5706, 6: 2.4469, 7: 2.3646}
TOL, SPREAD_X, SPREAD_MIN = 0.01, 2.0, 0.005    # P117's bar (P110's), unchanged
B16, B8 = (1, 2, 4, 8, 16), (1, 2, 4, 8)
BUCKETS = {"R": B16, "rep": B16, "half": B8, "chunk": B16, "rev": B16, "mutant_scale": B16, "S32": B16, "S64": B16}
FLOORS = ("half", "chunk", "rev")
OFF_ARMS = ("R", "rep") + FLOORS + ("mutant_scale",)
SUBJECTS = {"m32": "S32", "m64": "S64"}
ARMS = ("m0", "m32", "m64")
TALL = ("m32", "m64")


def capture_order(n: int) -> tuple:
    """Process ``n``'s registered capture order (1-based): m0 m32 m64 rotated by n - 1, as ``p131_box`` captures."""
    k = (n - 1) % len(ARMS)
    return ARMS[k:] + ARMS[:k]
GLUE_DEFAULT = {"mode": "1", "source": "default-allowlisted"}
# the f-band of the registration's cost model (f in [0.3, 0.8]): gain over m0, saving in ms, m64 over m32
PREDICT = {"m64_gain": (1.038, 1.159), "m32_gain": (1.033, 1.143), "m64_saving_ms": (1.2, 4.5),
           "m32_saving_ms": (1.1, 4.1), "m64_over_m32": (1.004, 1.013)}


def fold_table(layers: int) -> dict:
    return {"norm": layers + 1, "layer": layers, "attention": layers}


def k19_key(arm: str) -> str:
    return f"{'chained' if arm == 'm0' else 'split'}|lean|gt256"


def gt256(k19) -> dict:
    """K19's calls above 256 rows, by table and dispatch."""
    return {k: v for k, v in (k19 or {}).items() if k.endswith("|gt256") and v}


def decode_touched(arm_k19, m0_k19) -> bool:
    """Whether an arm's decode-check K19 record shows the knob reaching decode: a split table above 256 rows, or calls
    above 256 rows other than m0's in the same process (which table m0 takes is E4B_INT4_WIDE_TILES's choice)."""
    ka = gt256(arm_k19)
    return any(k.startswith("split|") for k in ka) or ka != gt256(m0_k19)


def decode_m0_bad(m0_k19, calls) -> bool:
    """Whether m0's own decode-check record is off: not ``calls`` lean K19 calls above 256 rows, or a split among them."""
    k0 = gt256(m0_k19)
    return any(k.startswith("split|") for k in k0) or sum(k0.values()) != calls or not all("|lean|" in k for k in k0)


def _bucket_for(n, buckets):
    return next(b for b in buckets if n <= b)


def expected_stats(arm, g, cont):
    """P117's: the runner's split into pieces of the largest bucket, each padded to its bucket, ``cont - 1`` steps."""
    bs = BUCKETS[arm]
    top, steps, out = bs[-1], cont - 1, {}
    left = g
    while left > 0:
        n = min(left, top)
        b = _bucket_for(n, bs)
        s = out.setdefault(str(b), {"pieces": 0, "rows": 0, "pad_rows": 0})
        s["pieces"] += steps
        s["rows"] += n * steps
        s["pad_rows"] += (b - n) * steps
        left -= n
    return out


def t_interval(xs):
    """(mean, low, high) of the two-sided 95 % t-interval of the mean of ``xs``."""
    n = len(xs)
    m = sum(xs) / n
    if n < 2:
        return m, float("-inf"), float("inf")
    h = T975[n - 1] * statistics.stdev(xs) / math.sqrt(n)
    return m, m - h, m + h


def stats(per_r, per_x):
    r = {x["window"]: x["nll"] for x in per_r}
    d = [x["nll"] - r[x["window"]] for x in per_x]
    n = len(d)
    if not n:
        return {"n": 0}
    mean = sum(d) / n
    sd = math.sqrt(sum((v - mean) ** 2 for v in d) / (n - 1)) if n > 1 else 0.0
    kls = [x["kl"] for x in per_x if "kl" in x]
    return {"n": n, "bias": mean, "spread": sum(abs(v) for v in d) / n, "se": sd / math.sqrt(n),
            "max_abs": max(abs(v) for v in d), "mean_kl": sum(kls) / len(kls) if kls else None,
            "argmax_agree": sum(x["argmax_agree"] for x in per_x) / n}


def floor_of(st, rep_identical):
    draws = list(FLOORS) + ([] if rep_identical else ["rep"])
    return {"draws": draws, "B_floor": max(abs(st[f]["bias"]) for f in draws), "S_floor": max(st[f]["spread"] for f in draws)}


def passes(s, fl):
    return s["bias"] <= fl["B_floor"] + TOL and s["spread"] <= SPREAD_X * max(fl["S_floor"], SPREAD_MIN)


def speed_faults(n, rec, size) -> list:
    """Lane-VOID engagement faults of process ``n``'s Phase A record (refusals and the decode check are read per arm)."""
    out = []
    sp, L = rec.get("speed") or {}, rec.get("layers")
    if sp.get("T") != size["prompt"] or sp.get("windows") != size["speed_windows"]:
        out.append(f"proc {n}: speed T {sp.get('T')} windows {sp.get('windows')}")
    if sp.get("capture_order") != list(capture_order(n)):
        out.append(f"proc {n}: capture order {sp.get('capture_order')}, registered {list(capture_order(n))}")
    graphs = sp.get("graphs") or {}
    if set(graphs) != set(ARMS):
        out.append(f"proc {n}: graphs {sorted(graphs)}")
        return out
    for g in ARMS:
        gr = graphs[g]
        if gr.get("status") == "refused":
            continue
        F = gr.get("forwards")
        seen = gr.get("seen") or {}
        if gr.get("status") != "on" or gr.get("T") != size["prompt"] or F != CAPTURE_FORWARDS:
            out.append(f"proc {n} {g}: status {gr.get('status')} T {gr.get('T')} forwards {F}")
            continue
        if seen.get("route") != {"int4_k19|gt256": F * L}:
            out.append(f"proc {n} {g}: route {seen.get('route')}, expected int4_k19|gt256 x {F * L}")
        if seen.get("k19") != {k19_key(g): F * L}:
            out.append(f"proc {n} {g}: K19 dispatch {seen.get('k19')}, expected {{{k19_key(g)}: {F * L}}}")
        want_folds = {k: F * v for k, v in fold_table(L).items()}
        if seen.get("folds") != want_folds:
            out.append(f"proc {n} {g}: prefill folds {seen.get('folds')}, expected {want_folds}")
    dc = sp.get("decode_check") or {}
    if dc.get("tokens") != DECODE_TOKENS or set(dc.get("seen") or {}) != set(ARMS):
        out.append(f"proc {n}: decode check {dc.get('tokens')} tokens over {sorted(dc.get('seen') or {})}")
    else:
        k0 = gt256(dc["seen"]["m0"].get("k19"))
        if decode_m0_bad(dc["seen"]["m0"].get("k19"), L):
            out.append(f"proc {n}: the decode check at m0 reads {k0} (want one lean K19 call a layer, no split)")
    if graphs["m0"].get("status") == "refused":
        return out
    have = [g for g in ARMS if graphs[g].get("status") != "refused"]
    nw = size["speed_windows"]
    if len(sp.get("digests_m0") or []) != nw:
        out.append(f"proc {n}: {len(sp.get('digests_m0') or [])} digests, expected {nw}")
    fn = sp.get("function") or {}
    for g in TALL:
        want_fn = nw if g in have else None
        if (fn.get(g) or {}).get("windows") != want_fn:
            out.append(f"proc {n}: FUNCTION {g} over {(fn.get(g) or {}).get('windows')} windows, expected {want_fn}")
    for g in have:
        k = len((sp.get("replay_ms") or {}).get(g) or [])
        if k != size["rounds"] * nw:
            out.append(f"proc {n} {g}: {k} timed replays, expected {size['rounds'] * nw}")
    return out


def quality_faults(q, layers, size) -> list:
    """Phase B integrity: P117's engagement per pass, the route, K19's dispatch above 256 rows (chained on R's arms,
    split on the subjects) and the prefill-fold deltas, the windows, the chunks."""
    C, P, nwin = size["cont"], size["prompt"], size["windows"]
    if not q:
        return ["Phase B record missing"]
    if "error" in q:
        return [f"Phase B raised: {q['error']}"]
    out = []
    for arm in OFF_ARMS + tuple(SUBJECTS.values()):
        if len((q.get("per_window") or {}).get(arm) or []) != nwin:
            out.append(f"arm {arm} scored {len((q.get('per_window') or {}).get(arm) or [])} windows, expected {nwin}")
        for k, e in enumerate((q.get("engagement") or {}).get(arm) or [None]):
            if e is None:
                out.append(f"arm {arm}: no engagement record")
                continue
            want = expected_stats(arm, nwin, C)
            pieces = sum(s["pieces"] for s in want.values()) // (C - 1)
            if e.get("decode_calls") != (C - 1) * layers * pieces:
                out.append(f"{arm} pass {k}: {e.get('decode_calls')} decode attention calls, expected "
                           f"{(C - 1) * layers * pieces}")
            if not (e.get("grouping_flags_in_pass") or {}).get("device_grouping"):
                out.append(f"{arm} pass {k}: device grouping off")
            st = e.get("graph_status") or {}
            if [st.get(str(b)) for b in BUCKETS[arm]] != ["eager: capture=False"] * len(BUCKETS[arm]):
                out.append(f"{arm} pass {k}: graph_status {st}")
            gs = e.get("graph_stats") or {}
            for b, w in want.items():
                got = gs.get(b, {})
                if (got.get("replays", 0), got.get("eager_steps", 0)) != (0, w["pieces"]) \
                        or got.get("rows") != w["rows"] or got.get("pad_rows") != w["pad_rows"]:
                    out.append(f"{arm} pass {k}: bucket {b} stats {got}, expected {w}")
            extra = [b for b, v in gs.items() if b not in want and (v.get("replays") or v.get("eager_steps"))]
            if extra:
                out.append(f"{arm} pass {k}: buckets {extra} ran, not in the registered split")
            chunks = -(-P // int(e.get("chunk") or P))
            calls = nwin * chunks * layers
            seen = e.get("seen") or {}
            route = (seen.get("route") or {}).get("int4_k19|gt256")
            if route != calls:
                out.append(f"{arm} pass {k}: int4_k19|gt256 x {route}, expected {calls}")
            gt = {kk: v for kk, v in (seen.get("k19") or {}).items() if kk.endswith("|gt256")}
            want_k19 = {("split|lean|gt256" if arm in SUBJECTS.values() else "chained|lean|gt256"): calls}
            if gt != want_k19:
                out.append(f"{arm} pass {k}: K19 dispatch above 256 rows {gt}, expected {want_k19}")
            want_folds = {k2: nwin * chunks * v for k2, v in fold_table(layers).items()}
            if (seen.get("folds") or {}) != want_folds:
                out.append(f"{arm} pass {k}: prefill folds {seen.get('folds')}, expected {want_folds}")
    if (q.get("chunk"), q.get("floor_chunk")) != (P, 256):
        out.append(f"chunks {q.get('chunk')}/{q.get('floor_chunk')}")
    return out


def _pred(value, lo_hi):
    lo, hi = lo_hi
    return {"value": value, "range": [lo, hi], "held": value is not None and lo <= value <= hi}


def corpus_report(recs: dict, gate_corpus: str | None) -> dict:
    """P130's Amendment 1, unchanged: the corpus commit the fetch gate resolved beside the one(s) the box read."""
    arrow = sorted({c for r in recs.values() for c in ((r.get("corpus") or {}).get("arrow") or [])})
    snaps = sorted({c for r in recs.values() for c in ((r.get("corpus") or {}).get("snapshots") or [])})
    same = None if not gate_corpus else (arrow == [gate_corpus] and snaps in ([], [gate_corpus]))
    return {"gate": gate_corpus or None, "box_arrow": arrow, "box_snapshots": snaps, "same": same}


def reduce(run: Path, e4b_sha: str, prove: bool = False, gate_corpus: str | None = None) -> dict:
    mode = "proof" if prove else "reading"
    size = MODES[mode]
    N = size["procs"]
    summ = (run / "summary.txt").read_text() if (run / "summary.txt").is_file() else ""
    out = {"lane": "P131", "mode": mode}
    procs = range(WARMUP, N + 1)        # Amendment 2: the warm-up 0, then the scored 1..N
    recs = {n: json.loads((run / f"proc_{n}.json").read_text()) for n in procs if (run / f"proc_{n}.json").is_file()}
    if "premise ok" not in summ or not recs:
        out.update(verdict="NO_READING", reasons=["premise did not hold" if "premise ok" not in summ else "no process record"])
        return out
    void = []
    missing = [n for n in procs if n not in recs]
    if missing:
        void.append(f"process records missing: {missing}")
    for n, r in recs.items():
        if r.get("e4b_sha") != e4b_sha:
            void.append(f"proc {n}: e4b {r.get('e4b_sha')} != {e4b_sha}")
        if r.get("gnf4_sha") != GNF4_SHA:
            void.append(f"proc {n}: grouped-nf4-gemm {r.get('gnf4_sha')} != {GNF4_SHA}")
        if REVS.get(r.get("model")) != r.get("revision"):
            void.append(f"proc {n}: model {r.get('model')}@{r.get('revision')} is not a registered revision")
        if r.get("proc") != n or r.get("phase_b") != PHASE_B.get(n, "none"):
            void.append(f"proc {n}: record proc {r.get('proc')} Phase B {r.get('phase_b')}, registered "
                        f"{PHASE_B.get(n, 'none')}")
        if r.get("prefill_glue") != GLUE_DEFAULT:
            void.append(f"proc {n}: prefill glue {r.get('prefill_glue')}, the release's default is {GLUE_DEFAULT}")
    if len({r.get("windows_digest") for r in recs.values()}) != 1:
        void.append("the processes read different windows")
    if void:
        out.update(verdict="VOID", reasons=void)
        return out
    L = recs[min(recs)]["layers"]
    for n, r in sorted(recs.items()):
        void += speed_faults(n, r, size)
    refused = {g: [] for g in TALL}
    for n, r in sorted(recs.items()):
        g = r["speed"]["graphs"]
        if g["m0"].get("status") == "refused":
            void.append(f"proc {n}: m0 refused ({g['m0'].get('why', '')[:160]})")
        for a in TALL:
            if g[a].get("status") == "refused":
                refused[a].append(f"proc {n}: {a} refused ({g[a].get('why', '')[:160]})")
    if void:
        out.update(verdict="VOID", reasons=void)
        return out
    out["refused"] = refused
    decode = {}
    for a in ARMS:
        decode[a] = []
        for n, r in sorted(recs.items()):
            seen = r["speed"]["decode_check"]["seen"]
            ka, k0 = gt256(seen[a].get("k19")), gt256(seen["m0"].get("k19"))
            if decode_touched(seen[a].get("k19"), seen["m0"].get("k19")):
                decode[a].append(f"proc {n}: {a} {ka} against m0 {k0}")
    timed_all = {n: r["speed"] for n, r in recs.items() if "m0" in (r["speed"].get("median_ms") or {})}
    first = timed_all[min(timed_all)]["digests_m0"]
    det_fail = [f"proc {n} differs from proc {min(timed_all)}" for n in sorted(timed_all)
                if timed_all[n]["digests_m0"] != first]
    timed = {n: sp for n, sp in timed_all.items() if n != WARMUP}     # Amendment 2: speed reads the scored processes only
    meds = [sp["median_ms"]["m0"] for sp in timed.values()]
    span = max(meds) / min(meds)
    noisy = [f"m0 medians span x{span:.4f} > x{NOISY_SPAN}"] if span > NOISY_SPAN else []
    out.update(determinism={"differ": det_fail}, span_m0=span, decode_check={"touched": decode})
    q = (recs.get(WARMUP) or {}).get("quality")
    qf = quality_faults(q, L, size)
    quality = {"faults": qf}
    if not qf:
        st = {arm: stats(q["per_window"]["R"], q["per_window"][arm]) for arm in OFF_ARMS[1:] + tuple(SUBJECTS.values())}
        fl = floor_of(st, q.get("rep_identical"))
        quality.update(stats=st, floor=fl, rep_identical=q.get("rep_identical"), subjects_bitwise=q.get("subjects_bitwise"),
                       bar={"tol": TOL, "spread_x": SPREAD_X, "spread_min": SPREAD_MIN, "bias_bar": fl["B_floor"] + TOL,
                            "spread_bar": SPREAD_X * max(fl["S_floor"], SPREAD_MIN)})
        if passes(st["mutant_scale"], fl):
            quality["faults"] = ["mutant_scale passes the bar: the gate cannot fail"]
    out["quality"] = quality
    common = "HELD_DETERMINISM" if det_fail else ("NOISY" if noisy else None)
    arms, reasons = {}, list(det_fail + noisy + quality["faults"])
    for a in TALL:
        both = {n: sp for n, sp in timed.items() if a in sp["median_ms"]}
        fn_all = {n: sp for n, sp in timed_all.items() if a in sp["median_ms"]}          # FUNCTION reads the warm-up too
        fn_differ = [f"proc {n} window {d['window']}" for n, sp in sorted(fn_all.items()) for d in sp["function"][a]["differ"]]
        rec = {"function": "BITWISE" if not fn_differ else "DIFFERS", "function_differ": fn_differ[:16]}
        if both:
            g = {n: sp["median_ms"]["m0"] / sp["median_ms"][a] for n, sp in sorted(both.items())}
            m, lo, hi = t_interval([math.log(v) for v in g.values()])
            rec.update(gains=g, mean=math.exp(m), ci95=[math.exp(lo), math.exp(hi)], min_gain=MIN_GAIN,
                       faster=math.exp(lo) >= MIN_GAIN,
                       saving_ms=sum(sp["median_ms"]["m0"] - sp["median_ms"][a] for sp in both.values()) / len(both))
        held = []
        if refused[a]:
            v = "REFUSED"
        elif decode[a]:
            v = "DECODE_TOUCHED"
        elif common:
            v = common
        else:
            if rec["function"] == "BITWISE":
                rec["quality"] = "BITWISE"
            elif quality["faults"]:
                rec["quality"] = "QUALITY_VOID"
            else:
                rec["quality"] = "AT_PARITY" if passes(quality["stats"][SUBJECTS[a]], quality["floor"]) else "COST"
            if rec["quality"] == "QUALITY_VOID":
                v = "QUALITY_VOID"
            else:
                if not rec.get("faster"):
                    held.append("NOT_FASTER")
                if rec["quality"] == "COST":
                    held.append("COST")
                v = "LICENSED" if not held else "HELD"
        rec.update(verdict=v, held_for=held)
        arms[a] = rec
        reasons += refused[a] + decode[a] + [f"{a} FUNCTION differs: {x}" for x in fn_differ[:4]]
        if "mean" in rec:
            reasons.append(f"{a} gain {rec['mean']:.4f} [{rec['ci95'][0]:.4f}, {rec['ci95'][1]:.4f}] vs x{MIN_GAIN}")
    out["arms"] = arms
    # the default: the licensed arm; both licensed -> the paired m32-vs-m64 interval decides, m32 on a tie
    both = {n: sp for n, sp in timed.items() if all(a in sp["median_ms"] for a in TALL)}
    order = {}
    if both:
        m, lo, hi = t_interval([math.log(sp["median_ms"]["m32"] / sp["median_ms"]["m64"]) for sp in both.values()])
        order = {"m64_over_m32": math.exp(m), "ci95": [math.exp(lo), math.exp(hi)],
                 "reading": "M64_FASTER" if lo > 0 else ("M32_FASTER" if hi < 0 else "INDISTINGUISHABLE")}
    out["order"] = order
    lic = [a for a in TALL if arms[a]["verdict"] == "LICENSED"]
    if len(lic) == 2:
        default = "m64" if order.get("reading") == "M64_FASTER" else "m32"
    else:
        default = lic[0] if lic else "NONE"
    out["default"] = default
    out.update(verdict=f"M32_{arms['m32']['verdict']} M64_{arms['m64']['verdict']} DEFAULT_{default}", reasons=reasons)
    sb = quality.get("subjects_bitwise") or {}
    out["predictions"] = {
        "m64_gain": _pred(arms["m64"].get("mean"), PREDICT["m64_gain"]),
        "m32_gain": _pred(arms["m32"].get("mean"), PREDICT["m32_gain"]),
        "m64_saving_ms": _pred(arms["m64"].get("saving_ms"), PREDICT["m64_saving_ms"]),
        "m32_saving_ms": _pred(arms["m32"].get("saving_ms"), PREDICT["m32_saving_ms"]),
        "m64_over_m32": _pred(order.get("m64_over_m32"), PREDICT["m64_over_m32"]),
        "function_bitwise": {"held": all(arms[a]["function"] == "BITWISE" for a in TALL)},
        "determinism": {"held": not det_fail},
        "decode_untouched": {"held": not any(decode.values())},
        "phase_b_subjects_bitwise": {"held": bool(sb) and all(sb.get(s) for s in SUBJECTS.values())},
        "verdict": {"value": out["verdict"], "held": out["verdict"] == "M32_LICENSED M64_LICENSED DEFAULT_m64"}}
    out["corpus"] = corpus_report(recs, gate_corpus)
    out["eager_median_ms"] = {n: sp.get("eager", {}).get("median_ms") for n, sp in sorted(timed.items())}
    w = timed_all.get(WARMUP) or {}
    wm = w.get("median_ms") or {}
    out["warmup"] = {"proc": WARMUP, "median_ms": wm, "eager_median_ms": (w.get("eager") or {}).get("median_ms"),
                     "gains_reported": {a: wm["m0"] / wm[a] for a in TALL if wm.get(a) and wm.get("m0")},
                     "m0_over_scored_median": (wm["m0"] / statistics.median(meds)) if wm.get("m0") and meds else None,
                     "excluded_from": ["span", "gains", "order", "bars", "predictions"]}
    out["gpu_state"] = {n: r.get("gpu_state") for n, r in sorted(recs.items())}
    return out


# ------------------------------------------------------------------------------------------------ self-test --

def _speed(size, L, *, ms, digest="d0", differ=None, refused=(), decode=None, n=1):
    nw, rounds = size["speed_windows"], size["rounds"]
    F = CAPTURE_FORWARDS
    graphs = {}
    for g in ARMS:
        if g in refused:
            graphs[g] = {"status": "refused", "why": "the 512-token prefill forward did not capture", "forwards": 3,
                         "seen": {}}
            continue
        graphs[g] = {"status": "on", "T": size["prompt"], "pool_mib": 3300, "forwards": F,
                     "seen": {"route": {"int4_k19|gt256": F * L}, "k19": {k19_key(g): F * L},
                              "folds": {k: F * v for k, v in fold_table(L).items()}}}
    # what Qwen3-30B's 512-row call takes by default: the wide one-launch table (P122's auto bound)
    dec = {g: {"route": {"int4_k19|gt256": L}, "k19": {"one_launch|lean|gt256": L, "one_launch|lean|le256": 3},
               "folds": {}} for g in ARMS}
    for g, k in (decode or {}).items():
        dec[g]["k19"] = k
    sp = {"T": size["prompt"], "windows": nw, "rounds": rounds, "graphs": graphs, "capture_order": list(capture_order(n)),
          "decode_check": {"tokens": DECODE_TOKENS, "seen": dec}}
    if "m0" in refused:
        return sp
    have = [g for g in ARMS if g not in refused]
    sp["function"] = {g: {"windows": nw, "differ": [{"window": w, "logits_equal": False, "kv_layers_differ": []}
                                                    for w in (differ or {}).get(g, ())]} for g in TALL if g in have}
    sp["digests_m0"] = [f"{digest}-{w}" for w in range(nw)]
    sp["replay_ms"] = {g: [ms[g]] * (rounds * nw) for g in have}
    sp["median_ms"] = {g: ms[g] for g in have}
    sp["eager"] = {"median_ms": {g: ms[g] * 1.4 for g in ARMS}}
    return sp


def _quality(size, L, d=None, bitwise=True):
    d = d or {}
    nwin, C, P = size["windows"], size["cont"], size["prompt"]
    arms = OFF_ARMS + tuple(SUBJECTS.values())
    per, eng = {}, {}
    for arm in arms:
        off = d.get(arm, 0.0 if arm in ("R", "rep", "S32", "S64") else (0.5 if arm == "mutant_scale" else 0.001))
        per[arm] = [{"window": w, "nll": 2.0 + 0.01 * (w % 5) + off, "argmax_agree": 1.0,
                     **({} if arm == "R" else {"kl": 1e-4})} for w in range(nwin)]
        want = expected_stats(arm, nwin, C)
        pieces = sum(s["pieces"] for s in want.values()) // (C - 1)
        chunk = 256 if arm == "chunk" else P
        calls = nwin * (-(-P // chunk)) * L
        eng[arm] = [{"decode_calls": (C - 1) * L * pieces,
                     "graph_status": {str(b): "eager: capture=False" for b in BUCKETS[arm]},
                     "graph_stats": {b: {"replays": 0, "eager_steps": s["pieces"], "rows": s["rows"], "pad_rows": s["pad_rows"]}
                                     for b, s in want.items()},
                     "grouping_flags_in_pass": {"device_grouping": True}, "chunk": chunk,
                     "seen": {"route": {"int4_k19|gt256": calls, "one_launch_lean|le256": 9},
                              "k19": {("split|lean|gt256" if arm in SUBJECTS.values() else "chained|lean|gt256"): calls,
                                      "one_launch|lean|le256": 9},
                              "folds": {k: calls // L * v for k, v in fold_table(L).items()}}}]
    return {"phase": "all", "windows": nwin, "prompt": P, "cont": C, "chunk": P, "floor_chunk": 256, "rep_identical": True,
            "subjects_bitwise": {"S32": bitwise, "S64": bitwise}, "per_window": per, "engagement": eng}


def _run_recs(mode="reading", e4b="a" * 40, L=4, m0=33.0, m32=30.9, m64=30.7, d=None, tweak=None, **kw):
    """Synthetic process records: every process at the given medians (with a small per-process jitter)."""
    size = MODES[mode]
    recs = {}
    for n in range(WARMUP, size["procs"] + 1):
        j = 1.0 + 0.0005 * ((n * 7) % 5 - 2)
        r = {"proc": n, "phase_b": PHASE_B.get(n, "none"), "model": "Qwen/Qwen3-30B-A3B",
             "revision": REVS["Qwen/Qwen3-30B-A3B"], "e4b_sha": e4b, "gnf4_sha": GNF4_SHA, "layers": L,
             "windows_digest": "w", "prefill_glue": dict(GLUE_DEFAULT),
             "corpus": {"arrow": ["b" * 40], "snapshots": ["b" * 40]},
             "speed": _speed(size, L, ms={"m0": m0 * j, "m32": m32 * j, "m64": m64 * j}, n=n, **kw)}
        if n in PHASE_B:
            r["quality"] = _quality(size, L, d)
        recs[n] = r
    if tweak:
        tweak(recs)
    return recs


def self_test() -> int:
    import tempfile
    cases = []

    def run(mode="reading", summary="premise ok\n", gate=None, **kw):
        with tempfile.TemporaryDirectory() as td:
            t = Path(td)
            (t / "summary.txt").write_text(summary)
            for n, r in _run_recs(mode, **kw).items():
                (t / f"proc_{n}.json").write_text(json.dumps(r))
            return reduce(t, "a" * 40, prove=(mode == "proof"), gate_corpus=gate)

    def drop(n):
        return lambda recs: recs.pop(n)

    o = run()
    cases.append(("both licensed bitwise, m64 faster", o["verdict"] == "M32_LICENSED M64_LICENSED DEFAULT_m64"
                  and o["arms"]["m64"]["quality"] == "BITWISE" and o["order"]["reading"] == "M64_FASTER"))
    cases.append(("predictions hold on the model's point", all(o["predictions"][k]["held"] for k in o["predictions"])))
    o = run(m64=30.9)
    cases.append(("a tie goes to m32", o["default"] == "m32" and o["order"]["reading"] == "INDISTINGUISHABLE"))
    o = run(m32=30.5, m64=30.9)
    cases.append(("m32 faster", o["default"] == "m32" and o["order"]["reading"] == "M32_FASTER"))
    o = run(m32=32.9)
    cases.append(("m32 not faster: only m64", o["verdict"] == "M32_HELD M64_LICENSED DEFAULT_m64"
                  and o["arms"]["m32"]["held_for"] == ["NOT_FASTER"]))
    o = run(m32=32.9, m64=32.8)
    cases.append(("neither faster", o["verdict"] == "M32_HELD M64_HELD DEFAULT_NONE"))
    o = run(differ={"m64": (0, 3)})
    cases.append(("m64 differs: the tolerance branch, at parity", o["arms"]["m64"]["function"] == "DIFFERS"
                  and o["arms"]["m64"]["quality"] == "AT_PARITY" and o["arms"]["m64"]["verdict"] == "LICENSED"))
    o = run(differ={"m64": (1,)}, d={"S64": 0.05})
    cases.append(("m64 differs and costs", o["arms"]["m64"]["verdict"] == "HELD" and o["arms"]["m64"]["held_for"] == ["COST"]
                  and o["default"] == "m32"))
    o = run(d={"S64": 0.05})
    cases.append(("bitwise ignores Phase B's subject", o["arms"]["m64"]["verdict"] == "LICENSED"
                  and o["arms"]["m64"]["quality"] == "BITWISE"))
    o = run(differ={"m32": (2,)}, d={"mutant_scale": 0.0})
    cases.append(("a mutant that passes voids the tolerance branch", o["arms"]["m32"]["verdict"] == "QUALITY_VOID"
                  and o["arms"]["m64"]["verdict"] == "LICENSED"))
    o = run(refused=("m64",))
    cases.append(("m64 refused", o["arms"]["m64"]["verdict"] == "REFUSED" and o["default"] == "m32"
                  and "mean" not in o["arms"]["m64"]))
    o = run(refused=("m0",))
    cases.append(("the control refused: VOID", o["verdict"] == "VOID" and any("m0 refused" in x for x in o["reasons"])))
    o = run(decode={"m32": {"split|lean|gt256": 4}})
    cases.append(("the knob reaches decode", o["arms"]["m32"]["verdict"] == "DECODE_TOUCHED"
                  and o["arms"]["m64"]["verdict"] == "LICENSED"))
    o = run(decode={"m64": {"chained|lean|gt256": 4}})
    cases.append(("a decode table other than m0's is touched", o["arms"]["m64"]["verdict"] == "DECODE_TOUCHED"
                  and o["arms"]["m32"]["verdict"] == "LICENSED"))
    o = run(decode={g: {"chained|lean|gt256": 4} for g in ARMS})
    cases.append(("whichever table auto picks, the same at every arm, is untouched", o["verdict"].startswith("M32_LICENSED")))
    o = run(decode={"m0": {"split|lean|gt256": 4}})
    cases.append(("a split at m0's decode check voids", o["verdict"] == "VOID"
                  and any("decode check at m0" in x for x in o["reasons"])))

    def det(recs):
        recs[5]["speed"]["digests_m0"][3] = "x"
    o = run(tweak=det)
    cases.append(("determinism", o["verdict"] == "M32_HELD_DETERMINISM M64_HELD_DETERMINISM DEFAULT_NONE"))

    def noisy(recs):
        for w in ("m0", "m32", "m64"):
            recs[6]["speed"]["median_ms"][w] *= 1.03
    o = run(tweak=noisy)
    cases.append(("noisy", o["verdict"] == "M32_NOISY M64_NOISY DEFAULT_NONE"))
    o = run(tweak=drop(4))
    cases.append(("a missing process voids", o["verdict"] == "VOID"))
    o = run(e4b="b" * 40)
    cases.append(("another e4b voids", o["verdict"] == "VOID"))

    def glue(recs):
        recs[2]["prefill_glue"] = {"mode": "0", "source": "explicit"}
    o = run(tweak=glue)
    cases.append(("the release's prefill default not in effect voids", o["verdict"] == "VOID"))

    def phase(recs):
        recs[2]["phase_b"] = "all"
    o = run(tweak=phase)
    cases.append(("Phase B off its process voids", o["verdict"] == "VOID"))

    def windows(recs):
        recs[3]["windows_digest"] = "v"
    o = run(tweak=windows)
    cases.append(("different windows void", o["verdict"] == "VOID"))

    def k19(recs):
        recs[2]["speed"]["graphs"]["m32"]["seen"]["k19"] = {"chained|lean|gt256": 20}
    o = run(tweak=k19)
    cases.append(("m32 captured on the chained table voids", o["verdict"] == "VOID"
                  and any("K19 dispatch" in x for x in o["reasons"])))

    def folds(recs):
        recs[1]["speed"]["graphs"]["m0"]["seen"]["folds"] = {}
    o = run(tweak=folds)
    cases.append(("prefill folds off voids", o["verdict"] == "VOID"))

    def replays(recs):
        recs[7]["speed"]["replay_ms"]["m64"] = recs[7]["speed"]["replay_ms"]["m64"][:-1]
    o = run(tweak=replays)
    cases.append(("a short replay list voids", o["verdict"] == "VOID"))

    def fnwin(recs):
        recs[8]["speed"]["function"]["m32"]["windows"] = 15
    o = run(tweak=fnwin)
    cases.append(("FUNCTION over fewer windows voids", o["verdict"] == "VOID"))

    def order(recs):
        recs[2]["speed"]["capture_order"] = list(ARMS)          # process 2 registers m32 m64 m0
    o = run(tweak=order)
    cases.append(("a capture order off the rotation voids", o["verdict"] == "VOID"
                  and any("capture order" in x for x in o["reasons"])))
    cases.append(("the rotation puts every arm in every position over the reading's processes",
                  all({capture_order(n).index(a) for n in range(1, MODES["reading"]["procs"] + 1)} == {0, 1, 2}
                      for a in ARMS)))

    def nodec(recs):
        recs[3]["speed"].pop("decode_check")
    o = run(tweak=nodec)
    cases.append(("a missing decode check voids", o["verdict"] == "VOID"))

    def qsplit(recs):
        e = recs[WARMUP]["quality"]["engagement"]["S64"][0]["seen"]["k19"]
        e["chained|lean|gt256"] = e.pop("split|lean|gt256")
    o = run(differ={"m64": (0,)}, tweak=qsplit)
    cases.append(("S64 passed on the chained table: QUALITY_VOID", o["arms"]["m64"]["verdict"] == "QUALITY_VOID"))

    def qerr(recs):
        recs[WARMUP]["quality"] = {"phase": "all", "error": "RuntimeError: boom"}
    o = run(tweak=qerr)
    cases.append(("Phase B raised; bitwise arms stand", o["arms"]["m64"]["verdict"] == "LICENSED"
                  and o["quality"]["faults"] == ["Phase B raised: RuntimeError: boom"]))
    o = run(summary="premise failed\n")
    cases.append(("no premise, no reading", o["verdict"] == "NO_READING"))
    o = run(mode="proof")
    cases.append(("the proof's sizes", o["mode"] == "proof" and o["verdict"].startswith("M32_LICENSED")))
    o = run(gate="b" * 40)
    cases.append(("corpus reported", o["corpus"]["same"] is True))
    o = run(m64=31.6)
    cases.append(("m64 inside its band; m64-over-m32 MISSED low", o["predictions"]["m64_gain"]["held"]
                  and not o["predictions"]["m64_over_m32"]["held"] and o["default"] == "m32"))
    cases.append(("t-interval", abs(t_interval([0.0, 0.0])[1]) < 1e-12 and t_interval([1.0])[1] == float("-inf")))

    # ---- Amendment 2: the warm-up process 0 (Phase B + a speed pass reported only) and the GPU state
    base = run()

    def warm_fast(recs):                       # the proof's process-1 effect, now in the warm-up: ~5 % faster everywhere
        for w in ARMS:
            recs[WARMUP]["speed"]["median_ms"][w] *= 0.95
    o = run(tweak=warm_fast)
    cases.append(("a fast warm-up is reported, never in the span guard", o["verdict"] == base["verdict"]
                  and o["span_m0"] == base["span_m0"] and o["warmup"]["m0_over_scored_median"] < 0.96))

    def warm_slow64(recs):
        recs[WARMUP]["speed"]["median_ms"]["m64"] *= 1.10
    o = run(tweak=warm_slow64)
    cases.append(("the warm-up never moves the gains or the order", o["arms"]["m64"]["mean"] == base["arms"]["m64"]["mean"]
                  and o["order"] == base["order"] and o["warmup"]["gains_reported"]["m64"] < 1.0))
    o = run(tweak=drop(WARMUP))
    cases.append(("a missing warm-up voids", o["verdict"] == "VOID" and any("[0]" in x for x in o["reasons"])))

    def pre_a2(recs):                          # the pre-amendment layout: Phase B in process 1
        recs[1]["phase_b"], recs[WARMUP]["phase_b"] = "all", "none"
        recs[1]["quality"] = recs[WARMUP].pop("quality")
    o = run(tweak=pre_a2)
    cases.append(("Phase B in process 1 (the pre-amendment layout) voids", o["verdict"] == "VOID"))

    def warm_fn(recs):
        recs[WARMUP]["speed"]["function"]["m64"]["differ"] = [{"window": 0, "logits_equal": False, "kv_layers_differ": []}]
    o = run(tweak=warm_fn)
    cases.append(("FUNCTION reads the warm-up", o["arms"]["m64"]["function"] == "DIFFERS"
                  and o["arms"]["m64"]["function_differ"] == ["proc 0 window 0"]))

    def warm_det(recs):
        recs[WARMUP]["speed"]["digests_m0"][3] = "x"
    o = run(tweak=warm_det)
    cases.append(("DETERMINISM reads the warm-up", o["verdict"] == "M32_HELD_DETERMINISM M64_HELD_DETERMINISM DEFAULT_NONE"))

    def gpu(recs):
        recs[WARMUP]["gpu_state"] = {"speed_start": {"clocks.sm": "2900"}, "speed_end": {"error": "nvidia-smi rc 9"}}
    o = run(tweak=gpu)
    cases.append(("the GPU state is reported for every process, never gated", o["verdict"] == base["verdict"]
                  and sorted(o["gpu_state"]) == list(range(WARMUP, MODES["reading"]["procs"] + 1))
                  and o["gpu_state"][WARMUP]["speed_start"]["clocks.sm"] == "2900"))
    bad = [name for name, ok in cases if not ok]
    for name, ok in cases:
        print(f"  {'ok ' if ok else 'BAD'} {name}")
    print(f"P131_REDUCE self-test: {len(cases) - len(bad)}/{len(cases)}" + (f" FAILED {bad}" if bad else ""))
    return 1 if bad else 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir")
    ap.add_argument("--out")
    ap.add_argument("--e4b-sha", default=os.environ.get("E4B_SHA", ""))
    ap.add_argument("--prove", action="store_true")
    ap.add_argument("--gate-corpus", default="")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args(argv)
    if a.self_test:
        return self_test()
    if not a.dir:
        ap.error("--dir RUN_DIR | --self-test")
    out = reduce(Path(a.dir), a.e4b_sha, prove=a.prove, gate_corpus=a.gate_corpus or None)
    if a.out:
        Path(a.out).write_text(json.dumps(out, indent=1, default=str))
    print(f"P131_VERDICT {out['verdict']}", flush=True)
    for r in out.get("reasons", [])[:24]:
        print(f"  {r}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
