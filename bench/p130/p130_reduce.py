#!/usr/bin/env python3
"""Lane P130's reducer (bench/p130/PREREG-p130.md; e4b#846): #1583's prefill knobs, P1 (``E4B_FUSE_PREFILL_GLUE``) and P2
(``E4B_PREFILL_LEAN_DISPATCH``), read for speed on the served first-chunk prefill graph (Phase A) and P1 for quality on
P117's teacher-forced instrument (Phase B). Phase B's rule is P117's (``bench/p117/p117_reduce.py``) by named
substitution: one subject, ``P1``, against R's own arithmetically neutral perturbations.

Input: ``proc_<n>.json`` for every registered process (``p130_box.py``) and ``summary.txt`` (the premise line) in the
run directory. ``--prove`` reads the proof's registered sizes.

Each process ``n`` runs with P1 at ``ORDER[n - 1]``; process 1 runs Phase B ``off`` and process 2 ``on``.

**Lane VOID**, any of:
- a registered process record missing; another e4b or grouped-nf4-gemm commit; another model revision;
- a process whose P1 setting (its record, and glue round 1's fold report) is not the registered one;
- the processes read different windows, or sizes other than the mode's;
- wrong speed engagement on a captured graph: ``T``, the capture's eager forwards (``CAPTURE_FORWARDS``), and per capture
  the deltas route ``int4_k19|gt256`` = forwards x layers and nothing else, K19's dispatch ``chained|gather|gt256``
  (``p2_off``) or ``chained|lean|gt256`` (``p2_on``) = forwards x layers and nothing else, and the prefill folds =
  forwards x ``fold_table`` with P1 on and none with it off; a speed record with other than ``rounds`` x windows timed
  replays per graph, or FUNCTION or digests over other than every speed window;
- the default's graph refused: a P1-unset process's ``p2_off`` capture.

**Per knob**, the first that applies:
- **REFUSED:** a capture with the knob on refused (P2: any ``p2_on``; P1: a P1-set process's ``p2_off``). A defect: the
  server's ``auto`` prefill graph would stand down under the knob.
- **FUNCTION_FAIL** (P2 only): on any speed window of any process, the ``p2_on`` replay is not ``torch.equal`` to the
  ``p2_off`` replay in the logits or any layer's staged K or V.
- **HELD_DETERMINISM:** two processes with the same P1 setting digest a speed window's ``p2_off`` replay differently. The
  cross-process premise of Phase B fails with it, so neither knob moves.
- **NOISY:** within either P1 setting, the processes' ``p2_off`` medians span more than ``NOISY_SPAN``.
- **P2:** per process ``r = median(p2_on) / median(p2_off)``; over the processes, the two-sided 95 % t-interval of the mean
  of ``ln r`` (``T975``). **LICENSED** iff its upper end is at most ``ln(P2_MARGIN)``; else **HELD_NOT_NO_SLOWER**.
- **P1:** per registered pair (``PAIRS``: a P1-unset process and the adjacent P1-set one) ``g = median_unset(p2_off) /
  median_set(p2_off)``; the two-sided 95 % t-interval of the mean of ``ln g``. FASTER iff its lower end is at least
  ``ln(P1_MIN_GAIN)``. Phase B: **QUALITY_VOID** on any integrity fault (P117's engagement per pass, plus the route and the
  prefill-fold deltas; a mutant that passes; R's saved log-probs failing their digests); else AT_PARITY iff P1 passes
  P117's bar against the floor, COST otherwise. **LICENSED** iff FASTER and AT_PARITY; else **HELD** naming
  ``NOT_FASTER`` and/or ``COST``.

Reported, never gated: every ratio and interval, the savings in ms against the glue map's ceilings, the predictions'
HELD/MISSED, the eager forward's medians, every Phase B arm's bias, spread, SE, max |d|, KL and argmax agreement.

    python p130_reduce.py --dir RUN_DIR --out verdict.json [--e4b-sha SHA] [--prove]
    python p130_reduce.py --self-test
"""
from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
from pathlib import Path

GNF4_SHA = "724ccc454f006c1a46836e434e997f31f293747f"          # grouped-nf4-gemm v0.45.0, e4b CI's pin
REVS = {"Qwen/Qwen3-30B-A3B": "ad44e777bcd18fa416d9da3bd8f70d33ebb85d39"}
MODES = {"reading": {"procs": 8, "windows": 64, "cont": 128, "prompt": 512, "speed_windows": 16, "rounds": 12, "warm": 2},
         "proof": {"procs": 4, "windows": 16, "cont": 32, "prompt": 512, "speed_windows": 4, "rounds": 3, "warm": 1}}
ORDER = (0, 1, 1, 0, 0, 1, 1, 0)                 # P1 per process, 1-based: ABBA, twice
PAIRS = ((1, 2), (4, 3), (5, 6), (8, 7))          # (P1 unset, P1 set), adjacent in time
PHASE_B = {1: "off", 2: "on"}
CAPTURE_FORWARDS = 5                              # enable_prefill_graph: 2 warm-up forwards, the capture, 2 checks
P2_MARGIN, P1_MIN_GAIN, NOISY_SPAN = 1.01, 1.03, 1.02
T975 = {1: 12.7062, 2: 4.3027, 3: 3.1824, 4: 2.7764, 5: 2.5706, 6: 2.4469, 7: 2.3646}
TOL, SPREAD_X, SPREAD_MIN = 0.01, 2.0, 0.005    # P117's bar (P110's), unchanged
B16, B8 = (1, 2, 4, 8, 16), (1, 2, 4, 8)
BUCKETS = {"R": B16, "rep": B16, "half": B8, "chunk": B16, "rev": B16, "mutant_scale": B16, "P1": B16}
FLOORS = ("half", "chunk", "rev")
OFF_ARMS = ("R", "rep") + FLOORS + ("mutant_scale",)
SUBJECT = "P1"
CEILING_MS = {"p1": 5.47, "p2": 1.34, "both": 6.80}         # bench/prefill-glue/DESIGN.md's glue map: device ms
PREDICT = {"p1_saving_ms": (2.5, 4.5), "p2_saving_ms": (0.5, 1.3), "both_saving_ms": (3.0, 5.5),
           "p1_bias": (-0.004, 0.004), "p1_spread_max": 0.016}
GRAPHS = ("p2_off", "p2_on")


def fold_table(layers: int) -> dict:
    return {"norm": layers + 1, "layer": layers, "attention": layers}


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
    """Lane-VOID engagement faults of process ``n``'s Phase A record (refusals are read per knob, not here)."""
    out = []
    sp, L, p1 = rec.get("speed") or {}, rec.get("layers"), rec.get("p1")
    if sp.get("T") != size["prompt"] or sp.get("windows") != size["speed_windows"]:
        out.append(f"proc {n}: speed T {sp.get('T')} windows {sp.get('windows')}")
    graphs = sp.get("graphs") or {}
    if set(graphs) != set(GRAPHS):
        out.append(f"proc {n}: graphs {sorted(graphs)}")
        return out
    for g in GRAPHS:
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
        want_k19 = {f"chained|{'lean' if g == 'p2_on' else 'gather'}|gt256": F * L}
        if seen.get("k19") != want_k19:
            out.append(f"proc {n} {g}: K19 dispatch {seen.get('k19')}, expected {want_k19}")
        want_folds = {k: F * v for k, v in fold_table(L).items()} if p1 else {}
        if seen.get("folds") != want_folds:
            out.append(f"proc {n} {g}: prefill folds {seen.get('folds')}, expected {want_folds}")
    if graphs["p2_off"].get("status") == "refused":
        return out
    have = [g for g in GRAPHS if graphs[g].get("status") != "refused"]
    nw = size["speed_windows"]
    if len(sp.get("digests_p2_off") or []) != nw:
        out.append(f"proc {n}: {len(sp.get('digests_p2_off') or [])} digests, expected {nw}")
    want_fn = nw if "p2_on" in have else None
    if (sp.get("function_p2") or {}).get("windows") != want_fn:
        out.append(f"proc {n}: FUNCTION over {(sp.get('function_p2') or {}).get('windows')} windows, expected {want_fn}")
    for g in have:
        k = len((sp.get("replay_ms") or {}).get(g) or [])
        if k != size["rounds"] * nw:
            out.append(f"proc {n} {g}: {k} timed replays, expected {size['rounds'] * nw}")
    return out


def quality_faults(off, on, layers, size) -> list:
    """Phase B integrity: P117's engagement per pass plus the route and prefill-fold deltas, windows, R's digests."""
    out = []
    C, P, nwin = size["cont"], size["prompt"], size["windows"]
    if not off or not on:
        return [f"Phase B record missing (off {bool(off)}, on {bool(on)})"]
    errors = [f"Phase B {q.get('phase')} raised: {q['error']}" for q in (off, on) if "error" in q]
    if errors:
        return errors
    for q, arms in ((off, OFF_ARMS), (on, (SUBJECT,))):
        for arm in arms:
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
                seen = e.get("seen") or {}
                route = (seen.get("route") or {}).get("int4_k19|gt256")
                if route != nwin * chunks * layers:
                    out.append(f"{arm} pass {k}: int4_k19|gt256 x {route}, expected {nwin * chunks * layers}")
                want_folds = {k2: nwin * v for k2, v in fold_table(layers).items()} if arm == SUBJECT else {}
                if (seen.get("folds") or {}) != want_folds:
                    out.append(f"{arm} pass {k}: prefill folds {seen.get('folds')}, expected {want_folds}")
    if not on.get("ref_ok"):
        out.append(f"R's saved log-probs failed their digests on load: {on.get('ref_bad')}")
    if (off.get("chunk"), off.get("floor_chunk"), on.get("chunk")) != (P, 256, P):
        out.append(f"chunks off {off.get('chunk')}/{off.get('floor_chunk')} on {on.get('chunk')}")
    return out


def _pred(value, lo_hi):
    lo, hi = lo_hi
    return {"value": value, "range": [lo, hi], "held": value is not None and lo <= value <= hi}


def corpus_report(recs: dict, gate_corpus: str | None) -> dict:
    """Amendment 1: the corpus commit the fetch gate resolved beside the one(s) the box read. Reported, never gated (the
    windows digest already gates that every process read the same windows)."""
    arrow = sorted({c for r in recs.values() for c in ((r.get("corpus") or {}).get("arrow") or [])})
    snaps = sorted({c for r in recs.values() for c in ((r.get("corpus") or {}).get("snapshots") or [])})
    same = None if not gate_corpus else (arrow == [gate_corpus] and snaps in ([], [gate_corpus]))
    return {"gate": gate_corpus or None, "box_arrow": arrow, "box_snapshots": snaps, "same": same}


def reduce(run: Path, e4b_sha: str, prove: bool = False, gate_corpus: str | None = None) -> dict:
    mode = "proof" if prove else "reading"
    size = MODES[mode]
    N = size["procs"]
    summ = (run / "summary.txt").read_text() if (run / "summary.txt").is_file() else ""
    out = {"lane": "P130", "mode": mode}
    recs = {n: json.loads((run / f"proc_{n}.json").read_text()) for n in range(1, N + 1) if (run / f"proc_{n}.json").is_file()}
    if "premise ok" not in summ or not recs:
        out.update(verdict="NO_READING", reasons=["premise did not hold" if "premise ok" not in summ else "no process record"])
        return out
    void = []
    missing = [n for n in range(1, N + 1) if n not in recs]
    if missing:
        void.append(f"process records missing: {missing}")
    for n, r in recs.items():
        if r.get("e4b_sha") != e4b_sha:
            void.append(f"proc {n}: e4b {r.get('e4b_sha')} != {e4b_sha}")
        if r.get("gnf4_sha") != GNF4_SHA:
            void.append(f"proc {n}: grouped-nf4-gemm {r.get('gnf4_sha')} != {GNF4_SHA}")
        if REVS.get(r.get("model")) != r.get("revision"):
            void.append(f"proc {n}: model {r.get('model')}@{r.get('revision')} is not a registered revision")
        flag = (((r.get("census") or {}).get("fusion_report") or {}).get("E4B_FUSE_T1_GLUE") or {}).get("prefill")
        if r.get("p1") != ORDER[n - 1] or r.get("proc") != n or flag != ("on" if ORDER[n - 1] else "off"):
            void.append(f"proc {n}: P1 {r.get('p1')} (fold report prefill={flag}), registered {ORDER[n - 1]}")
        if r.get("phase_b") != PHASE_B.get(n, "none"):
            void.append(f"proc {n}: Phase B {r.get('phase_b')}, registered {PHASE_B.get(n, 'none')}")
    if len({r.get("windows_digest") for r in recs.values()}) != 1:
        void.append("the processes read different windows")
    if void:
        out.update(verdict="VOID", reasons=void)
        return out
    for n, r in sorted(recs.items()):
        void += speed_faults(n, r, size)
    refused = {"p1": [], "p2": []}
    for n, r in sorted(recs.items()):
        g = r["speed"]["graphs"]
        if g["p2_off"].get("status") == "refused":
            (refused["p1"] if r["p1"] else void).append(
                f"proc {n}: p2_off refused ({g['p2_off'].get('why', '')[:160]})")
        if g["p2_on"].get("status") == "refused":
            refused["p2"].append(f"proc {n}: p2_on refused ({g['p2_on'].get('why', '')[:160]})")
    if void:
        out.update(verdict="VOID", reasons=void)
        return out
    out["refused"] = refused
    timed = {n: r["speed"] for n, r in recs.items() if "p2_off" in (r["speed"].get("median_ms") or {})}
    both = {n: sp for n, sp in timed.items() if "p2_on" in sp["median_ms"]}
    # FUNCTION (P2) over the processes that captured both graphs; DETERMINISM and NOISY over every timed p2_off
    fn_fail = [f"proc {n} window {d['window']}" for n, sp in sorted(both.items()) for d in sp["function_p2"]["differ"]]
    det_fail, spans = [], {}
    for p1 in (0, 1):
        group = [n for n in sorted(timed) if ORDER[n - 1] == p1]
        if len(group) > 1:
            first = timed[group[0]]["digests_p2_off"]
            det_fail += [f"P1={p1}: proc {n} differs from proc {group[0]}" for n in group[1:]
                         if timed[n]["digests_p2_off"] != first]
            meds = [timed[n]["median_ms"]["p2_off"] for n in group]
            spans[str(p1)] = max(meds) / min(meds)
    noisy = [f"P1={k}: p2_off medians span x{v:.4f} > x{NOISY_SPAN}" for k, v in spans.items() if v > NOISY_SPAN]
    out.update(function_p2={"differ": fn_fail[:16]}, determinism={"differ": det_fail}, spans=spans)
    # P2's interval
    p2 = {}
    if both:
        r = {n: sp["median_ms"]["p2_on"] / sp["median_ms"]["p2_off"] for n, sp in sorted(both.items())}
        m, lo, hi = t_interval([math.log(v) for v in r.values()])
        p2 = {"ratios": r, "mean": math.exp(m), "ci95": [math.exp(lo), math.exp(hi)], "margin": P2_MARGIN,
              "no_slower": math.exp(hi) <= P2_MARGIN,
              "saving_ms": sum(sp["median_ms"]["p2_off"] - sp["median_ms"]["p2_on"] for sp in both.values()) / len(both)}
    # P1's interval over the registered pairs
    p1s = {}
    pairs = [(a, b) for a, b in PAIRS[:N // 2] if a in timed and b in timed]
    if pairs:
        g = {f"{a}/{b}": timed[a]["median_ms"]["p2_off"] / timed[b]["median_ms"]["p2_off"] for a, b in pairs}
        m, lo, hi = t_interval([math.log(v) for v in g.values()])
        bp = [(a, b) for a, b in pairs if "p2_on" in timed[b]["median_ms"]]
        p1s = {"gains": g, "mean": math.exp(m), "ci95": [math.exp(lo), math.exp(hi)], "min_gain": P1_MIN_GAIN,
               "faster": math.exp(lo) >= P1_MIN_GAIN,
               "saving_ms": sum(timed[a]["median_ms"]["p2_off"] - timed[b]["median_ms"]["p2_off"] for a, b in pairs) / len(pairs),
               "both_saving_ms": (sum(timed[a]["median_ms"]["p2_off"] - timed[b]["median_ms"]["p2_on"] for a, b in bp) / len(bp)
                                  if bp else None)}
    out.update(p2_speed=p2, p1_speed=p1s)
    # Phase B (P117's rule, one subject)
    off, on = (recs.get(1) or {}).get("quality"), (recs.get(2) or {}).get("quality")
    L = recs[min(recs)]["layers"]
    qf = quality_faults(off, on, L, size)
    quality = {"faults": qf}
    if not qf:
        st = {arm: stats(off["per_window"]["R"], off["per_window"][arm]) for arm in OFF_ARMS if arm != "R"}
        st[SUBJECT] = stats(off["per_window"]["R"], on["per_window"][SUBJECT])
        fl = floor_of(st, off.get("rep_identical"))
        quality.update(stats=st, floor=fl, rep_identical=off.get("rep_identical"),
                       bar={"tol": TOL, "spread_x": SPREAD_X, "spread_min": SPREAD_MIN, "bias_bar": fl["B_floor"] + TOL,
                            "spread_bar": SPREAD_X * max(fl["S_floor"], SPREAD_MIN)})
        if passes(st["mutant_scale"], fl):
            quality["faults"] = ["mutant_scale passes the bar: the gate cannot fail"]
        else:
            quality["verdict"] = "AT_PARITY" if passes(st[SUBJECT], fl) else "COST"
    if quality["faults"]:
        quality["verdict"] = "QUALITY_VOID"
    out["quality"] = quality
    # per knob
    common = "HELD_DETERMINISM" if det_fail else ("NOISY" if noisy else None)
    if refused["p2"]:
        v2 = "REFUSED"
    elif fn_fail:
        v2 = "FUNCTION_FAIL"
    elif common:
        v2 = common
    else:
        v2 = "LICENSED" if p2.get("no_slower") else "HELD_NOT_NO_SLOWER"
    held1 = []
    if refused["p1"]:
        v1 = "REFUSED"
    elif common:
        v1 = common
    elif quality["verdict"] == "QUALITY_VOID":
        v1 = "QUALITY_VOID"
    else:
        if not p1s.get("faster"):
            held1.append("NOT_FASTER")
        if quality["verdict"] != "AT_PARITY":
            held1.append("COST")
        v1 = "LICENSED" if not held1 else "HELD"
    out["p1"] = {"verdict": v1, "held_for": held1}
    out["p2"] = {"verdict": v2}
    reasons = refused["p1"] + refused["p2"] + fn_fail[:8] + det_fail + noisy + quality["faults"]
    if p2:
        reasons.append(f"P2 ratio {p2['mean']:.4f} [{p2['ci95'][0]:.4f}, {p2['ci95'][1]:.4f}] vs margin x{P2_MARGIN}")
    if p1s:
        reasons.append(f"P1 gain {p1s['mean']:.4f} [{p1s['ci95'][0]:.4f}, {p1s['ci95'][1]:.4f}] vs x{P1_MIN_GAIN}")
    if quality.get("stats"):
        s, b = quality["stats"][SUBJECT], quality["bar"]
        reasons.append(f"P1 bias {s['bias']:+.5f} (bar {b['bias_bar']:.5f}), spread {s['spread']:.5f} "
                       f"(bar {b['spread_bar']:.5f})")
    out.update(verdict=f"P1_{v1} P2_{v2}", reasons=reasons)
    # the predictions, scored mechanically (reported, never gated)
    sq = (quality.get("stats") or {}).get(SUBJECT) or {}
    out["predictions"] = {
        "p1_saving_ms": _pred(p1s.get("saving_ms"), PREDICT["p1_saving_ms"]),
        "p2_saving_ms": _pred(p2.get("saving_ms"), PREDICT["p2_saving_ms"]),
        "both_saving_ms": _pred(p1s.get("both_saving_ms"), PREDICT["both_saving_ms"]),
        "p1_bias": _pred(sq.get("bias"), PREDICT["p1_bias"]),
        "p1_spread": _pred(sq.get("spread"), (0.0, PREDICT["p1_spread_max"])),
        "function_and_determinism": {"held": not fn_fail and not det_fail},
        "verdict": {"value": out["verdict"], "held": out["verdict"] == "P1_LICENSED P2_LICENSED"}}
    out["ceilings_ms"] = CEILING_MS
    out["corpus"] = corpus_report(recs, gate_corpus)
    out["eager_median_ms"] = {n: sp.get("eager", {}).get("median_ms") for n, sp in sorted(timed.items())}
    return out


# ------------------------------------------------------------------------------------------------ self-test --

def _speed(p1, size, L, *, off_ms, on_ms, digest="d0", differ=(), refused=()):
    nw, rounds = size["speed_windows"], size["rounds"]
    F = CAPTURE_FORWARDS
    graphs = {}
    for g in GRAPHS:
        if g in refused:
            graphs[g] = {"status": "refused", "why": "the 512-token prefill forward did not capture", "forwards": 3,
                         "seen": {}}
            continue
        graphs[g] = {"status": "on", "T": size["prompt"], "pool_mib": 3300, "forwards": F,
                     "seen": {"route": {"int4_k19|gt256": F * L},
                              "k19": {f"chained|{'lean' if g == 'p2_on' else 'gather'}|gt256": F * L},
                              "folds": {k: F * v for k, v in fold_table(L).items()} if p1 else {}}}
    sp = {"T": size["prompt"], "windows": nw, "rounds": rounds, "graphs": graphs}
    if "p2_off" in refused:
        return sp
    have = [g for g in GRAPHS if g not in refused]
    sp["function_p2"] = ({"windows": nw, "differ": [{"window": w, "logits_equal": False, "kv_layers_differ": []}
                                                    for w in differ]} if "p2_on" in have else None)
    sp["digests_p2_off"] = [f"{digest}-{w}" for w in range(nw)]
    ms = {"p2_off": off_ms, "p2_on": on_ms}
    sp["replay_ms"] = {g: [ms[g]] * (rounds * nw) for g in have}
    sp["median_ms"] = {g: ms[g] for g in have}
    sp["eager"] = {"median_ms": {"p2_off": off_ms * 1.4, "p2_on": on_ms * 1.4}}
    return sp


def _quality(phase, size, L, d=None):
    d = d or {}
    nwin, C, P = size["windows"], size["cont"], size["prompt"]
    arms = OFF_ARMS if phase == "off" else (SUBJECT,)
    per, eng = {}, {}
    for arm in arms:
        off = d.get(arm, 0.0 if arm in ("R", "rep") else 0.001)
        per[arm] = [{"window": w, "nll": 2.0 + 0.01 * (w % 5) + off, "argmax_agree": 1.0,
                     **({} if arm == "R" else {"kl": 1e-4})} for w in range(nwin)]
        want = expected_stats(arm, nwin, C)
        pieces = sum(s["pieces"] for s in want.values()) // (C - 1)
        chunk = 256 if arm == "chunk" else P
        eng[arm] = [{"decode_calls": (C - 1) * L * pieces,
                     "graph_status": {str(b): "eager: capture=False" for b in BUCKETS[arm]},
                     "graph_stats": {b: {"replays": 0, "eager_steps": s["pieces"], "rows": s["rows"], "pad_rows": s["pad_rows"]}
                                     for b, s in want.items()},
                     "grouping_flags_in_pass": {"device_grouping": True}, "chunk": chunk,
                     "seen": {"route": {"int4_k19|gt256": nwin * (-(-P // chunk)) * L, "one_launch_lean|le256": 9},
                              "folds": {k: nwin * v for k, v in fold_table(L).items()} if arm == SUBJECT else {}}}]
    q = {"phase": phase, "windows": nwin, "prompt": P, "cont": C, "chunk": P, "per_window": per, "engagement": eng}
    if phase == "off":
        q.update(floor_chunk=256, rep_identical=True)
    else:
        q.update(ref_ok=True, ref_bad=[])
    return q


def _run_recs(mode="reading", e4b="a" * 40, L=4, off_ms=40.0, p1_ms=36.5, p2_ratio=0.975, d=None, tweak=None):
    """Synthetic process records: P1-set processes at ``p1_ms``, P1-unset at ``off_ms``, P2 at ``p2_ratio``."""
    size = MODES[mode]
    recs = {}
    for n in range(1, size["procs"] + 1):
        p1 = ORDER[n - 1]
        base = p1_ms if p1 else off_ms
        jitter = 1.0 + 0.0005 * ((n * 7) % 5 - 2)
        r = {"proc": n, "p1": p1, "phase_b": PHASE_B.get(n, "none"), "model": "Qwen/Qwen3-30B-A3B",
             "revision": REVS["Qwen/Qwen3-30B-A3B"], "e4b_sha": e4b, "gnf4_sha": GNF4_SHA, "layers": L,
             "windows_digest": "w", "census": {"fusion_report": {"E4B_FUSE_T1_GLUE": {"prefill": "on" if p1 else "off"}}},
             "corpus": {"arrow": ["b" * 40], "snapshots": ["b" * 40]},
             "speed": _speed(p1, size, L, off_ms=base * jitter, on_ms=base * jitter * p2_ratio, digest=f"d{p1}")}
        if n in PHASE_B:
            r["quality"] = _quality(PHASE_B[n], size, L, d)
        recs[n] = r
    if tweak:
        tweak(recs)
    return recs


def self_test() -> int:
    import tempfile
    E = "a" * 40
    cases = []

    def run(recs, premise=True, prove=False, gate_corpus="b" * 40):
        with tempfile.TemporaryDirectory() as t:
            p = Path(t)
            (p / "summary.txt").write_text("premise ok\n" if premise else "premise failed\n")
            for n, r in (recs or {}).items():
                (p / f"proc_{n}.json").write_text(json.dumps(r))
            return reduce(p, E, prove, gate_corpus)

    big = {"mutant_scale": 0.8}

    def v(**kw):
        kw.setdefault("d", big)
        return run(_run_recs(**kw))

    def with_(fn, **kw):
        kw.setdefault("d", big)
        return run(_run_recs(tweak=fn, **kw))

    r = v()
    cases.append(("both licensed", r["verdict"] == "P1_LICENSED P2_LICENSED" and r["predictions"]["verdict"]["held"]))
    cases.append(("the corpus the box read is the gate's", r["corpus"] == {"gate": "b" * 40, "box_arrow": ["b" * 40],
                                                                       "box_snapshots": ["b" * 40], "same": True}))
    r = run(_run_recs(d=big), gate_corpus="c" * 40)
    cases.append(("a corpus drift is reported, not gated", r["corpus"]["same"] is False
                  and r["verdict"] == "P1_LICENSED P2_LICENSED"))
    cases.append(("the savings are scored", r["predictions"]["p1_saving_ms"]["held"]
                  and r["predictions"]["p2_saving_ms"]["held"] and r["predictions"]["both_saving_ms"]["held"]))
    cases.append(("no premise", run(_run_recs(d=big), premise=False)["verdict"] == "NO_READING"))
    cases.append(("no records", run({})["verdict"] == "NO_READING"))
    cases.append(("a process missing", with_(lambda R: R.pop(5))["verdict"] == "VOID"))
    cases.append(("wrong e4b", v(e4b="b" * 40)["verdict"] == "VOID"))

    def rev(R):
        R[3]["revision"] = "main"
    cases.append(("wrong revision", with_(rev)["verdict"] == "VOID"))

    def p1_wrong(R):
        R[3]["p1"] = 0
    cases.append(("a process off its registered P1", with_(p1_wrong)["verdict"] == "VOID"))

    def flag_wrong(R):
        R[6]["census"]["fusion_report"]["E4B_FUSE_T1_GLUE"]["prefill"] = "off"
    cases.append(("the fold report says P1 never patched", with_(flag_wrong)["verdict"] == "VOID"))

    def win(R):
        R[7]["windows_digest"] = "x"
    cases.append(("different windows", with_(win)["verdict"] == "VOID"))

    def fwd(R):
        R[2]["speed"]["graphs"]["p2_on"]["forwards"] = 4
    cases.append(("capture forwards", with_(fwd)["verdict"] == "VOID"))

    def route(R):
        R[4]["speed"]["graphs"]["p2_off"]["seen"]["route"]["int4_mtile|gt256"] = 1
    cases.append(("another route", with_(route)["verdict"] == "VOID"))

    def k19(R):
        R[5]["speed"]["graphs"]["p2_on"]["seen"]["k19"] = {"chained|gather|gt256": 5 * 4}
    cases.append(("P2 on but K19 gathered", with_(k19)["verdict"] == "VOID"))

    def folds_off(R):
        R[1]["speed"]["graphs"]["p2_off"]["seen"]["folds"] = {"norm": 25}
    cases.append(("prefill folds with P1 unset", with_(folds_off)["verdict"] == "VOID"))

    def folds_on(R):
        R[2]["speed"]["graphs"]["p2_off"]["seen"]["folds"]["attention"] -= 1
    cases.append(("prefill folds off the table", with_(folds_on)["verdict"] == "VOID"))

    def timed_n(R):
        R[8]["speed"]["replay_ms"]["p2_on"].pop()
    cases.append(("timed replays short", with_(timed_n)["verdict"] == "VOID"))

    def fn(R):
        R[6]["speed"]["function_p2"]["differ"].append({"window": 3, "logits_equal": True, "kv_layers_differ": [7]})
    r = with_(fn)
    cases.append(("P2 FUNCTION fails, P1 still read", r["p2"]["verdict"] == "FUNCTION_FAIL"
                  and r["p1"]["verdict"] == "LICENSED"))

    def det(R):
        R[4]["speed"]["digests_p2_off"][0] = "other"
    r = with_(det)
    cases.append(("determinism", r["verdict"] == "P1_HELD_DETERMINISM P2_HELD_DETERMINISM"))

    def noisy(R):
        R[8]["speed"]["median_ms"]["p2_off"] *= 1.05
    cases.append(("noisy", with_(noisy)["verdict"] == "P1_NOISY P2_NOISY"))
    r = v(p2_ratio=1.03)
    cases.append(("P2 slower", r["p2"]["verdict"] == "HELD_NOT_NO_SLOWER" and r["p1"]["verdict"] == "LICENSED"))

    def scatter(R):
        for n, f in zip(range(1, 9), (0.95, 1.04, 0.96, 1.05, 0.95, 1.04, 0.97, 1.03)):
            R[n]["speed"]["median_ms"]["p2_on"] = R[n]["speed"]["median_ms"]["p2_off"] * f
    r = with_(scatter)
    cases.append(("P2 interval too wide to show no slower", r["p2"]["verdict"] == "HELD_NOT_NO_SLOWER"
                  and r["p2_speed"]["mean"] < 1.0))
    r = v(p1_ms=39.4)
    cases.append(("P1 not faster", r["p1"]["verdict"] == "HELD" and r["p1"]["held_for"] == ["NOT_FASTER"]))
    r = v(d={**big, "P1": 0.05})
    cases.append(("P1 costs", r["p1"]["verdict"] == "HELD" and r["p1"]["held_for"] == ["COST"]
                  and r["p2"]["verdict"] == "LICENSED"))
    r = v(p1_ms=39.4, d={**big, "P1": 0.05})
    cases.append(("P1 neither", r["p1"]["held_for"] == ["NOT_FASTER", "COST"]))
    r = v(d={"mutant_scale": 0.002})
    cases.append(("mutant passes", r["p1"]["verdict"] == "QUALITY_VOID" and r["p2"]["verdict"] == "LICENSED"))

    def ref(R):
        R[2]["quality"].update(ref_ok=False, ref_bad=[3])
    cases.append(("R's log-probs corrupted", with_(ref)["p1"]["verdict"] == "QUALITY_VOID"))

    def calls(R):
        R[1]["quality"]["engagement"]["half"][0]["decode_calls"] -= 1
    cases.append(("Phase B calls", with_(calls)["p1"]["verdict"] == "QUALITY_VOID"))

    def qfolds(R):
        R[2]["quality"]["engagement"]["P1"][0]["seen"]["folds"] = {}
    cases.append(("P1 pass without its folds", with_(qfolds)["p1"]["verdict"] == "QUALITY_VOID"))

    def qerr(R):
        R[2]["quality"] = {"phase": "on", "error": "FileNotFoundError: digests.json"}
    r = with_(qerr)
    cases.append(("Phase B raised: P1 QUALITY_VOID, P2 still read", r["p1"]["verdict"] == "QUALITY_VOID"
                  and r["p2"]["verdict"] == "LICENSED"))

    def qroute(R):
        R[1]["quality"]["engagement"]["chunk"][0]["seen"]["route"]["int4_k19|gt256"] -= 4
    cases.append(("Phase B prefill route", with_(qroute)["p1"]["verdict"] == "QUALITY_VOID"))

    def p2_ref(R):
        for n in R:
            R[n]["speed"] = _speed(ORDER[n - 1], MODES["reading"], 4, off_ms=36.5 if ORDER[n - 1] else 40.0, on_ms=36.0,
                                   digest=f"d{ORDER[n - 1]}", refused=("p2_on",))
    r = with_(p2_ref)
    cases.append(("P2 refused everywhere, P1 still timed on every pair", r["p2"]["verdict"] == "REFUSED"
                  and r["p1"]["verdict"] == "LICENSED" and len(r["p1_speed"]["gains"]) == 4))

    def p1_ref(R):
        R[6]["speed"] = _speed(1, MODES["reading"], 4, off_ms=36.5, on_ms=36.0, refused=("p2_off", "p2_on"))
    r = with_(p1_ref)
    cases.append(("P1 refused under capture", r["p1"]["verdict"] == "REFUSED" and r["p2"]["verdict"] == "REFUSED"))

    def base_ref(R):
        R[4]["speed"] = _speed(0, MODES["reading"], 4, off_ms=40.0, on_ms=39.0, refused=("p2_off",))
    cases.append(("the default's graph refused", with_(base_ref)["verdict"] == "VOID"))
    r = v(d={**big, "half": 0.02, "P1": 0.025})
    cases.append(("the floor widens the bar", r["quality"]["verdict"] == "AT_PARITY"))

    def rep(R):
        q = R[1]["quality"]
        q["rep_identical"] = False
        q["per_window"]["rep"] = [dict(x, nll=x["nll"] + 0.004) for x in q["per_window"]["rep"]]
    r = with_(rep)
    cases.append(("rep joins the floor", "rep" in r["quality"]["floor"]["draws"] and r["p1"]["verdict"] == "LICENSED"))
    pr = run(_run_recs(mode="proof", d=big), prove=True)
    cases.append(("the proof's four processes", pr["verdict"] == "P1_LICENSED P2_LICENSED" and pr["mode"] == "proof"))
    cases.append(("a reading's records read as a proof", run(_run_recs(d=big), prove=True)["verdict"] == "VOID"))
    cases.append(("R's split", expected_stats("R", 64, 8) == {"16": {"pieces": 28, "rows": 448, "pad_rows": 0}}))
    cases.append(("half's split", expected_stats("half", 64, 8) == {"8": {"pieces": 56, "rows": 448, "pad_rows": 0}}))
    m, lo, hi = t_interval([0.0, 0.02, -0.02, 0.0])
    cases.append(("the t-interval", abs(m) < 1e-12 and abs(hi - 3.1824 * statistics.stdev([0.0, 0.02, -0.02, 0.0]) / 2) < 1e-12))
    bad = [n for n, ok in cases if not ok]
    print(f"p130_reduce self-test {'OK' if not bad else 'FAILED ' + str(bad)} ({len(cases)} cases)")
    return 0 if not bad else 1


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--self-test", action="store_true")
    p.add_argument("--dir")
    p.add_argument("--out")
    p.add_argument("--e4b-sha", default=os.environ.get("E4B_SHA", ""))
    p.add_argument("--prove", action="store_true")
    p.add_argument("--gate-corpus", default="", help="the fetch gate's corpus.main (Amendment 1; reported)")
    a = p.parse_args(argv)
    if a.self_test:
        return self_test()
    v = reduce(Path(a.dir), a.e4b_sha, a.prove, a.gate_corpus or None)
    json.dump(v, open(a.out, "w"), indent=1)
    print(f"P130_VERDICT {v['verdict']} {json.dumps(v['reasons'])}")
    for k in ("p2_speed", "p1_speed", "spans", "function_p2", "determinism", "corpus", "predictions"):
        if k in v:
            print(f"  {k}: {json.dumps(v[k])}")
    q = v.get("quality") or {}
    for arm, s in (q.get("stats") or {}).items():
        print(f"  {arm}: bias {s['bias']:+.5f} spread {s['spread']:.5f} se {s['se']:.5f} max {s['max_abs']:.5f} "
              f"kl {s['mean_kl']} agree {s['argmax_agree']:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
