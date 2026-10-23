#!/usr/bin/env python3
"""p232_reduce.py -- lane P232's registered rule (bench/p232/PREREG-p232.md; lever K, K2b): the served read of the int8
lm_head at 2, 4 and 8 rows (``E4B_INT8_HEAD_ROWS=8``, grouped-nf4-gemm K37's ``linear_i8``) on Qwen3-30B-A3B's default
server. P153's reducer (bench/p153/p153_reduce.py) by counted edits: its helpers, power floor, cap and census checks and
its NOISY / FUNCTION_FAIL rungs, at W2, W4 and W8 in place of W1 and W16; a per-T PASS (P153's quality bars at each
group T, and g_T >= 1.01) in place of its one FLIP; the TRIPWIRE (rung 6) added.

Records (``p232_box.py``): ``arm_{Ra,Ka,Kb,Rb,Rc,Kc,Kd,Rd}.json`` (ABBA twice, each through W2, W4 and W8 at five timed
reps, on the default graph server) and the warm-up ``arm_Rw.json`` (P131 Amendment 2's process 0: an R build and speed
pass before Ra, integrity-checked like every arm and VOID when missing, but reported only: it never enters a pair, a
self-pair, a rung's comparison or a bar); ``quality_R.json``, ``quality_K.json`` and ``quality_B16.json`` (Phase B's
instrument on the default server built eager, at group T = 2, 4 and 8 in turn, 24 wikitext windows x 128 positions,
the pass KV at the server's 4096 tokens). R = ``E4B_INT8_HEAD_ROWS`` unset; K = ``8``; both with ``E4B_INT8_HEAD_BW``
unset (on by default for qwen3_moe, ``default-allowlisted``). B16 = ``E4B_INT8_HEAD_BW=0``, quality only, reported.

THE RULING PAIR IS K37 AGAINST F.linear ON ONE GRID: both arms round the head in place to its per-row int8 grid; the
knob changes only the kernel at 2 to 8 rows. K's tokens are never held to R's (reported only).

NO REAL FIXTURE YET: no P232 box (shakedown or reading) has run, so every case below is synthetic, built to the box's
record shape (p232_box.py); the shakedown's record becomes the real fixture by an amendment before the reading.

The rule, first rung that applies:
  VOID           a record missing or not ok, the warm-up's included; the pins (e4b, grouped-nf4-gemm v0.50.0, torch,
                 the model), the 170-SM card, the census and modes, the split cap at its default with its plans, the
                 captured buckets (bucket T replayed in W_T), the dense attention GEMV off; the head's engagement (both
                 arms swap the one 151936 x 2048 head, default-allowlisted; K routes rows 2, 4 and 8 to K37 at plan
                 (32, 256, 4) and passes none of them; R passes them to F.linear and routes none; both route one row to
                 K36; in the quality phases the same at the eager decode's T; a zero count is a VOID); B16 swaps nothing;
                 the quality KV at 4096 tokens, 24 windows, 128 positions, group T as registered, one window set; every
                 timed block's power limit at least 575 W (unreadable is a VOID); the arms on different prompts or
                 lengths, or not at five reps; a void slope.
  NOISY          (a VOID, and a strike) any self-pair -- Rb/Ra, Kb/Ka, Rd/Rc, Kd/Kc -- at W2, W4 or W8 outside
                 [0.985, 1.015].
  FUNCTION_FAIL  an arm decodes different tokens from its own arm's first record (every R against Ra, every K against
                 Ka) on any row, workload or length; one arm's timed reps digest differently.
  per T          PASS when |mean NLL(K) - mean NLL(R)| <= 0.02 and argmax agreement >= 0.95 at group T, AND g_T (the
                 min over the FOUR K/R pairs at W_T) >= 1.01, compared unrounded; MISS otherwise, each failed bar named.
  verdict        FLIP_ROWS_8 / FLIP_ROWS_4 / FLIP_ROWS_2: the largest T with every T up to it passing (a MISS at 4
                 caps it at 2 even if 8 passes); NO_FLIP when T = 2 misses.
  TRIPWIRE       (reported, named, never changing the verdict) at any T: |NLL gap| > 0.005, argmax < 0.98, or the K
                 against R KL above max(3 x the R self-pair floor, 0.002), the floor being phase R's ``rep`` KL at that
                 T (a zero or absent floor falls to 0.002). Prints ``TRIPWIRE none`` or every condition that fired;
                 ``TRIPWIRE unread`` on a VOID (nothing was read).
A shakedown (``p232-shake-N``, every record ``shakedown``) reads ``SHAKEDOWN`` after integrity, never a reading: the
card, its plans and its power are not the registered ones and its reps are three; the rungs below are reported as
``would_read``.

Reported, never ruled: every pair and self-pair ratio, the geomean over the four pairs, the quality KL of K against R,
the bf16 comparison (K and R against B16, with P153's bars beside them), greedy token identity of K against R at W2,
W4 and W8 (rows identical out of T, first divergent step per row), per-row bucket stability from saved hidden states,
each arm's telemetry and the warm-up's rates.

    python p232_reduce.py --dir RUN_DIR --out verdict_p232.json --e4b-sha SHA
    python p232_reduce.py --self-test
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

TAGS = ("Ra", "Ka", "Kb", "Rb", "Rc", "Kc", "Kd", "Rd")      # ABBA twice
WARMUP = "Rw"      # P131 Amendment 2's warm-up process 0, before Ra: integrity-checked, reported, never scored
PAIRS = (("Ka", "Ra"), ("Kb", "Rb"), ("Kc", "Rc"), ("Kd", "Rd"))
SELF_PAIRS = (("Rb", "Ra"), ("Kb", "Ka"), ("Rd", "Rc"), ("Kd", "Kc"))
NOISY_BAND = (0.985, 1.015)
REPS = 5
SHAKE_REPS = 3                                            # the shakedown's reduced rep count (never a reading)
GROUPS = (2, 4, 8)                                        # T: the rows the knob routes, each read on its own
WORKLOADS = {"W2": 2, "W4": 4, "W8": 8}
BUCKETS = (1, 2, 4, 8, 16)
GNF4_SHA = "41ef086e6a3e2be5b158ff1bb38e0de67862a949"     # grouped-nf4-gemm v0.50.0, released: K36 and K37 as shipped
TORCH = "2.8.0+cu128"
MODEL, REV = "Qwen/Qwen3-30B-A3B", "ad44e777bcd18fa416d9da3bd8f70d33ebb85d39"
CENSUS = {"fuse_qkv_n": 48, "fuse_t1_glue_n": 193, "fuse_t1_glue_r2_n": [48, 48], "fuse_router_epilogue_n": 48}
FUSION_KNOBS = ("E4B_PAGED_FUSE_QKV", "E4B_FUSE_T1_GLUE", "E4B_FUSE_T1_GLUE_R2", "E4B_FUSE_ROUTER_EPI")
CAP = 16                                                  # every arm: the qwen3_moe default (e4b#1792)
HEAD_SHAPE = [151936, 2048]                               # Qwen3-30B-A3B's lm_head (N, K), untied
#: K's route record (engines/int8_head_bw.apply_int8_head_rows) at E4B_INT8_HEAD_ROWS=8: K37 at its shipped plan
ROWS_RECORD = {"rows_max": 8, "kernel": "dense_i8.linear_i8 (K37)", "plan": {str(m): [32, 256, 4] for m in range(2, 9)}}
K36, K37 = "K36 gemv_i8_bw", "K37 linear_i8"
MIN_POWER_W = 575                                         # the 5090 served floor (P149, P151, P153: the maintainer's)
# what the 5090's library plan gives each captured bucket at Qwen3's 4 KV heads and 4096-token slots, and the cap's
CAPTURED_PLANS = {1: (32, 16), 2: (32, 16), 4: (23, 16), 8: (11, 11), 16: (5, 5)}
Q_WINDOWS, Q_PROMPT, Q_CONT = 24, 512, 128                # 24 divides by 2, 4 and 8: every pass holds exactly T windows
PHASE_ARMS = {"R": ("R", "rep"), "K": ("ON",), "B16": ("ON",)}
QUALITY_ARMS = ("R", "K", "B16")
Q_BIAS, Q_ARGMAX = 0.02, 0.95                             # P153's bars, unchanged, at each T on its own
FLIP_G = 1.01
TRIP_BIAS, TRIP_ARGMAX, TRIP_KL_MULT, TRIP_KL_ABS = 0.005, 0.98, 3.0, 0.002
BW = ("bw_tree", "bw_prmt32", "bw_splitk")
SMS = 170                                                 # the RTX 5090: the plans above are its
SERVED_TOKENS, SERVED_BLOCKS = 4096, 256                  # the server's KV per sequence (16-token blocks)
VERDICTS_READ = ("FLIP_ROWS_8", "FLIP_ROWS_4", "FLIP_ROWS_2", "NO_FLIP", "FUNCTION_FAIL")


def _rate(r, w):
    return r["workloads"][w]["decode_tok_s"]


def _tok(r, w, n):
    return r["workloads"][w]["tokens"][str(n)]


def divergence(a: list, b: list) -> dict:
    """Rows identical out of the rows, and each row's first divergent step (None when identical): reported, never ruled."""
    first = []
    for x, y in zip(a, b):
        i = next((k for k, (p, q) in enumerate(zip(x, y)) if p != q), None)
        first.append(i if i is not None or len(x) == len(y) else min(len(x), len(y)))
    return {"rows_identical": sum(f is None for f in first), "rows": len(first), "first_divergent_step": first}


def gemv_faults(name, d) -> list:
    d = d or {}
    if d.get("bw_prmt32", 0) <= 0 or d.get("dotpad", 0) or d.get("dotpad_splitk", 0) or d.get("bw_tree", 0):
        return [f"{name}: dispatched {d}; the default is bw_prmt32 and no dot-pad at Qwen3's shapes"]
    return []


def record_faults(name, r, e4b_sha, arm, shake=False) -> list:
    out = []
    if r.get("e4b_sha") != e4b_sha:
        out.append(f"{name}: e4b {r.get('e4b_sha')} != {e4b_sha}")
    if r.get("gnf4_sha") != GNF4_SHA:
        out.append(f"{name}: grouped-nf4-gemm {r.get('gnf4_sha')} != {GNF4_SHA}")
    if r.get("torch") != TORCH:
        out.append(f"{name}: torch {r.get('torch')} != {TORCH}")
    if (r.get("model"), r.get("revision")) != (MODEL, REV):
        out.append(f"{name}: model {r.get('model')}@{r.get('revision')} is not {MODEL}@{REV}")
        return out
    if r.get("arm") != arm:
        out.append(f"{name}: the record says arm {r.get('arm')}")
    if not shake and r.get("sm_count") != SMS:
        out.append(f"{name}: {r.get('sm_count')} SMs, registered the RTX 5090's {SMS}")
    cen = {k: (r.get("fusions") or {}).get(k) for k in CENSUS}
    if cen != CENSUS:
        out.append(f"{name}: census {cen}, the default's {CENSUS}")
    modes, sources = r.get("fusion_modes") or {}, r.get("fusion_sources") or {}
    if any(str(modes.get(k)) != "auto" or sources.get(k) != "default-allowlisted" for k in FUSION_KNOBS):
        out.append(f"{name}: fusion modes {modes} / sources {sources}, the default is auto (default-allowlisted)")
    dg = r.get("dense_gemv_bw") or {}
    if dg.get("mode") != "0" or dg.get("modules"):
        out.append(f"{name}: the dense attention GEMV knob records {dg}; registered off in every arm")
    return out


def cap_faults(name, sc, want, shake=False) -> list:
    """The split cap at its qwen3_moe default (16, ``default-allowlisted``; e4b#1792) and its plan at every bucket in
    ``want`` (the speed build's captures, or the quality pass's T), none elsewhere. A shakedown's card plans its own
    splits: there the plans need only have run."""
    sc = sc or {}
    plans = {int(b): p for b, p in (sc.get("plans") or {}).items()}
    out = []
    if sc.get("cap") != CAP or sc.get("source") != "default-allowlisted":
        out.append(f"{name}: cap {sc.get('cap')} from {sc.get('source')}, registered {CAP} (default-allowlisted)")
    for b in want:
        p = plans.get(b) or {}
        plan, n = CAPTURED_PLANS[b]
        if not p.get("calls") or (not shake and (p.get("plan"), p.get("n_split")) != (plan, n)):
            out.append(f"{name}: B = {b} recorded {p}, registered plan {plan} -> {n} splits")
    if set(plans) - set(want):
        out.append(f"{name}: the cap met row counts {sorted(set(plans) - set(want))} outside {sorted(want)}")
    return out


def head_faults(name, r, arm, *, captured) -> list:
    """The head as built. R and K: the one lm_head swapped (:data:`HEAD_SHAPE`) by the qwen3_moe default
    (``E4B_INT8_HEAD_BW`` unset, ``default-allowlisted``); R's route record ``rows_max`` 1, K's :data:`ROWS_RECORD`
    (K37 at (32, 256, 4) at every M in 2..8). B16: nothing swapped, by the explicit ``0``. ``captured`` (the speed
    build): the graph captures' engagement -- both route one row to K36 and pass no one-row call; K routes rows 2, 4
    and 8 to K37 and passes none of them; R passes them to F.linear and routes none. A zero count is a VOID."""
    out = []
    d = r.get("int8_head_bw") or {}
    eng = d.get("engagement") or {}
    routed = eng.get("routed_modules_by_rows") or {}
    passed = eng.get("passed_modules_by_rows") or {}
    if arm == "B16":
        if (d.get("mode"), d.get("source")) != ("0", "explicit") or d.get("modules") or eng.get("modules") or routed \
                or d.get("rows") != {"rows_max": 1}:
            out.append(f"{name}: B16 must swap nothing (E4B_INT8_HEAD_BW=0, explicit); it records {d}")
        return out
    if (d.get("mode"), d.get("modules"), d.get("shape"), d.get("source")) != ("1", 1, HEAD_SHAPE, "default-allowlisted"):
        out.append(f"{name}: swapped {d.get('modules')} head(s) at {d.get('shape')} (mode {d.get('mode')}, source "
                   f"{d.get('source')}), registered the one lm_head at {HEAD_SHAPE} by the qwen3_moe default")
    want_rows = ROWS_RECORD if arm == "K" else {"rows_max": 1}
    if d.get("rows") != want_rows:
        out.append(f"{name}: the route record {d.get('rows')}, registered {want_rows}")
    if not captured:
        return out
    rows = [str(T) for T in GROUPS]
    if eng.get("modules") != 1 or routed.get("1") != 1 or "1" in passed:
        out.append(f"{name}: one-row calls routed {routed.get('1')}, passed {passed.get('1')} over {eng.get('modules')} "
                   "head(s): registered both arms routing one row to K36 (a zero count is a VOID)")
    if arm == "K":
        kern = eng.get("routed_kernel_by_rows") or {}
        if any(routed.get(t) != 1 or t in passed or kern.get(t) != K37 for t in rows) or kern.get("1") != K36:
            out.append(f"{name}: K routed {routed} ({kern}), passed {passed}: registered rows 2, 4 and 8 on K37, none "
                       "passed, one row on K36 (a zero count is a VOID)")
    elif any(t in routed or not passed.get(t) for t in rows):
        out.append(f"{name}: R routed {routed}, passed {passed}: registered rows 2, 4 and 8 passed to F.linear and "
                   "none routed (a zero count is a VOID)")
    return out


def quality_head_faults(name, p, arm, T) -> list:
    """The head at the eager decode's T (the calls this T's measurement added, read on the head itself): K routes T to
    K37 and passes no T-row call; R passes T to F.linear and routes none; B16 has no int8 head to call."""
    calls = p.get("head_calls") or {}
    routed, passed = calls.get("routed") or {}, calls.get("passed") or {}
    kern = ((p.get("int8_head_bw_measure") or {}).get("routed_kernel_by_rows") or {})
    t = str(T)
    if arm == "B16":
        return [f"{name} T={T}: B16's head was called as an int8 head: {calls}"] if routed or passed else []
    if arm == "K" and (not routed.get(t) or t in passed or kern.get(t) != K37):
        return [f"{name} T={T}: K routed {routed} ({kern}), passed {passed}: registered T rows on K37, none passed "
                "(a zero count is a VOID)"]
    if arm == "R" and (t in routed or not passed.get(t)):
        return [f"{name} T={T}: R routed {routed}, passed {passed}: registered T rows passed to F.linear, none routed "
                "(a zero count is a VOID)"]
    return []


def workload_faults(t, r, arm) -> list:
    """W_T replays bucket T's graph (captured at the build, where the head's engagement was read), and no call of the
    workload's own puts a T-row head call on the wrong route."""
    out = []
    for w, T in WORKLOADS.items():
        wl = r["workloads"].get(w) or {}
        gd = (wl.get("graph_delta") or {}).get(str(T)) or {}
        if not gd.get("replays"):
            out.append(f"{t} {w}: bucket {T} never replayed in its workload ({gd}); the capture is not what {w} ran")
        hc = wl.get("head_calls") or {}
        wrong = (hc.get("passed") or {}) if arm == "K" else (hc.get("routed") or {})
        if wrong.get(str(T)):
            out.append(f"{t} {w}: an eager {T}-row head call took the other arm's route ({hc})")
    return out


def power_faults(name, r) -> list:
    """Every timed block's power limit, read either side of it, at least :data:`MIN_POWER_W`; unreadable is a fault."""
    out = []
    for w in WORKLOADS:
        tel = (r["workloads"].get(w) or {}).get("telemetry") or {}
        if sorted(tel) != sorted(str(n) for n in (r.get("short"), r.get("long"))):
            out.append(f"{name} {w}: telemetry for blocks {sorted(tel)}, registered one per timed length")
        for n, t in tel.items():
            for side in ("before", "after"):
                pl = ((t or {}).get(side) or {}).get("power.limit")
                if pl is None or pl < MIN_POWER_W:
                    out.append(f"{name} {w} at {n} tokens: power.limit {pl} W {side} the block, the floor {MIN_POWER_W} W")
    return out


def telemetry_report(r) -> dict:
    """Reported, never ruled: what the card read beside one arm's timed blocks."""
    reads = [s for w in WORKLOADS for t in ((r["workloads"].get(w) or {}).get("telemetry") or {}).values()
             for s in ((t or {}).get("before") or {}, (t or {}).get("after") or {})]

    def vals(k):
        return [x[k] for x in reads if x.get(k) is not None]
    return {"power_limit_w": sorted(set(vals("power.limit"))), "clocks_max_sm": sorted(set(vals("clocks.max.sm"))),
            "temperature_max": max(vals("temperature.gpu"), default=None), "sm_clock_min": min(vals("clocks.sm"), default=None)}


def quality_faults(name, rec, arm, e4b_sha, shake=False) -> list:
    """One quality phase's process: the record's pins and build, then each T as registered."""
    out = record_faults(name, rec, e4b_sha, arm, shake)
    out += head_faults(name, rec, arm, captured=False)
    phases = rec.get("phases") or {}
    if sorted(phases) != sorted(str(T) for T in GROUPS):
        return out + [f"{name}: groups {sorted(phases)}, registered {list(GROUPS)}"]
    for T in GROUPS:
        p = phases[str(T)] or {}
        if p.get("group") != T or p.get("arms") != list(PHASE_ARMS[arm]):
            out.append(f"{name} T={T}: group {p.get('group')} arms {p.get('arms')}, registered {T} {list(PHASE_ARMS[arm])}")
        if (p.get("windows"), p.get("prompt"), p.get("cont")) != ({"wikitext": Q_WINDOWS}, Q_PROMPT, Q_CONT):
            out.append(f"{name} T={T}: windows {p.get('windows')} x {p.get('cont')} after {p.get('prompt')}, registered "
                       f"{Q_WINDOWS} wikitext windows x {Q_CONT} after {Q_PROMPT}")
        per = (p.get("per_window") or {}).get("wikitext") or {}
        for key in PHASE_ARMS[arm]:
            n = len(per.get(key) or [])
            want = T if key == "rep" else Q_WINDOWS          # measure_phase draws the floor on the first group only
            if n != want:
                out.append(f"{name} T={T}: {n} {key} windows, registered {want}")
            if key == "ON" and any(not isinstance(x.get("kl"), (int, float)) for x in per.get(key) or []):
                out.append(f"{name} T={T}: a {key} window carries no KL (a missing KL is unread, never 0)")
            if key == "ON" and any(not isinstance(x.get("argmax_agree"), (int, float)) for x in per.get(key) or []):
                out.append(f"{name} T={T}: a {key} window carries no argmax agreement")
            elif key == "ON":
                off = [x.get("window") for x in per.get(key) or [] if not integral_agree(x["argmax_agree"])]
                if off:
                    out.append(f"{name} T={T}: argmax agreement in window(s) {off[:4]} is not k/{Q_CONT} for an integer k: "
                               f"the window was not scored on the registered {Q_CONT} positions, so its counts are unread")
        out += cap_faults(f"{name} T={T}", p.get("attn_split_cap"), (T,), shake)
        qk = p.get("quality_kv") or {}
        served = (qk.get("max_tokens_per_seq"), qk.get("blocks_per_seq")) == ([SERVED_TOKENS], [SERVED_BLOCKS])
        if not shake:
            served = served and qk.get("plan_bT") == CAPTURED_PLANS[T][0]
        if not served or (qk.get("passes") or 0) < Q_WINDOWS // T or qk.get("plan_clash") or qk.get("split_cap") != CAP:
            out.append(f"{name} T={T}: the pass's KV {qk}: registered {SERVED_TOKENS} tokens ({SERVED_BLOCKS} blocks, plan "
                       f"{CAPTURED_PLANS[T][0]} at B = {T}), cap {CAP}, one KV per group of {T}")
        out += quality_head_faults(name, p, arm, T)
    return out


def _mean(xs):
    return sum(xs) / len(xs)


def integral_agree(frac: float) -> bool:
    """measure_phase's agreement is matched/positions; on the registered Q_CONT positions it is k/Q_CONT exactly."""
    k = frac * Q_CONT
    return abs(k - round(k)) < 1e-6


def agree_counts(windows: list) -> tuple:
    """Argmax agreement as exact pooled counts: each window's fraction is matched/Q_CONT (measure_phase), so the
    positions that agree are recovered as integers and the bars compare counts, never a float mean that can land a
    hair under 0.95."""
    return sum(round(x["argmax_agree"] * Q_CONT) for x in windows), len(windows) * Q_CONT


def agree_at_least(counts: tuple, bar: float) -> bool:
    """counts[0] / counts[1] >= bar, in integers (the bars are whole percents)."""
    return counts[0] * 100 >= round(bar * 100) * counts[1]


def compare(a: list, b: list) -> dict:
    """Arm ``a`` against arm ``b`` over the same windows (per-window means): the mean NLL gap a - b."""
    ref = {x["window"]: x["nll"] for x in b}
    return {"bias": _mean([x["nll"] - ref[x["window"]] for x in a]), "n": len(a)}


def quality_read(q: dict) -> dict:
    """Per T: K against R (the ruling numbers), the R self-pair floor (phase R's ``rep`` KL), and the reported bf16
    comparison. measure_phase scores an ON arm's argmax and KL against the saved R only: K against B16 is read as
    its NLL gap (exact over the same windows); its argmax agreement and KL have no instrument in the registered phases."""
    out = {}
    for T in GROUPS:
        t = str(T)
        wr = q["R"]["phases"][t]["per_window"]["wikitext"]
        rr, rep = wr["R"], wr.get("rep") or []
        kk = q["K"]["phases"][t]["per_window"]["wikitext"]["ON"]
        bb = q["B16"]["phases"][t]["per_window"]["wikitext"]["ON"]
        kl_rep = [x["kl"] for x in rep if x.get("kl") is not None]
        floor = _mean(kl_rep) if kl_rep else None
        ag = agree_counts(kk)
        out[t] = {"bias": compare(kk, rr)["bias"], "argmax_agree": ag[0] / ag[1], "argmax_counts": list(ag),
                  "kl": _mean([x["kl"] for x in kk]), "n": len(kk), "floor_kl": floor,
                  "rep_identical": q["R"]["phases"][t].get("rep_identical"),
                  "b16": {"R_vs_B16": {"bias": -compare(bb, rr)["bias"], "argmax_agree": _mean([x["argmax_agree"] for x in bb]),
                                       "kl": _mean([x["kl"] for x in bb])},
                          "K_vs_B16": {"bias": compare(kk, bb)["bias"], "argmax_agree": None, "kl": None,
                                       "unread": "argmax and KL: the registered phases score B16 against R only"},
                          "bars_p153": {"bias": Q_BIAS, "argmax": Q_ARGMAX}}}
    return out


def tripwire(qr: dict) -> list:
    """Rung 6: every condition that fired, at every T (empty: none)."""
    fired = []
    for T in GROUPS:
        x = qr[str(T)]
        if abs(x["bias"]) > TRIP_BIAS:
            fired.append(f"T={T}: |NLL gap| {abs(x['bias']):.6f} > {TRIP_BIAS}")
        if not agree_at_least(tuple(x["argmax_counts"]), TRIP_ARGMAX):
            fired.append(f"T={T}: argmax agreement {x['argmax_agree']:.6f} < {TRIP_ARGMAX}")
        bound = max(TRIP_KL_MULT * (x["floor_kl"] or 0.0), TRIP_KL_ABS)   # a zero or absent floor: the absolute bound
        if x["kl"] > bound:
            fired.append(f"T={T}: KL {x['kl']:.6f} > max({TRIP_KL_MULT:g} x floor {x['floor_kl']}, {TRIP_KL_ABS}) = {bound:.6f}")
    return fired


def bound_of(passes: dict):
    """The largest T such that every T up to it passes, or None (T = 2 misses)."""
    bound = None
    for T in GROUPS:
        if not passes[T]:
            break
        bound = T
    return bound


def reduce(arms: dict, quality: dict, e4b_sha: str) -> dict:
    out = {"lane": "P232", "verdict": None, "reasons": [], "tripwire": None,
           "bars": {"quality_bias": Q_BIAS, "quality_argmax": Q_ARGMAX, "q_windows": Q_WINDOWS, "flip_g": FLIP_G,
                    "noisy_band": list(NOISY_BAND), "reps": REPS, "min_power_w": MIN_POWER_W,
                    "tripwire": {"bias": TRIP_BIAS, "argmax": TRIP_ARGMAX, "kl_mult": TRIP_KL_MULT, "kl_abs": TRIP_KL_ABS}}}
    missing = [t for t in TAGS + (WARMUP,) if t not in arms or arms[t].get("status") != "ok"]
    missing += [f"quality_{a}" for a in QUALITY_ARMS if not quality.get(a) or quality[a].get("status") != "ok"]
    if missing:
        out.update(verdict="VOID", reasons=[f"record(s) missing or not ok: {missing}"], tripwire=["unread"])
        return out
    recs = [arms[t] for t in TAGS + (WARMUP,)] + [quality[a] for a in QUALITY_ARMS]
    flags = {bool(r.get("shakedown")) for r in recs}
    if len(flags) != 1:
        out.update(verdict="VOID", reasons=["the records mix shakedown and reading processes"], tripwire=["unread"])
        return out
    shake = flags.pop()
    out["shakedown"] = shake
    reps = SHAKE_REPS if shake else REPS
    void = []
    for t in TAGS + (WARMUP,):                                    # integrity reads every process, the warm-up's too
        r = arms[t]
        arm = t[:1]
        void += record_faults(t, r, e4b_sha, arm, shake)
        void += cap_faults(t, r.get("attn_split_cap"), BUCKETS, shake)
        void += head_faults(t, r, arm, captured=True)
        void += gemv_faults(f"{t} capture", r.get("dispatch_build"))
        g = r.get("graph_status") or {}
        if [g.get(str(b)) for b in BUCKETS] != ["graph"] * len(BUCKETS):
            void.append(f"{t}: graph_status {g}")
        if sorted(r.get("workloads") or {}) != sorted(WORKLOADS):
            void.append(f"{t}: workloads {sorted(r.get('workloads') or {})}, registered {sorted(WORKLOADS)}")
            continue
        void += workload_faults(t, r, arm)
        if not shake:
            void += power_faults(t, r)
        if any(_rate(r, w) is None for w in WORKLOADS):
            void.append(f"{t}: a decode slope is void")
        if r.get("reps") != reps:
            void.append(f"{t}: {r.get('reps')} timed reps, registered {reps}")
    if len({arms[t].get("prompts_sha256") for t in TAGS + (WARMUP,)}) != 1:
        void.append("the arms read different prompts")
    if len({(arms[t].get("short"), arms[t].get("long"), arms[t].get("reps")) for t in TAGS + (WARMUP,)}) != 1:
        void.append("the arms ran different lengths")
    for a in QUALITY_ARMS:
        void += quality_faults(f"quality_{a}", quality[a], a, e4b_sha, shake)
    if not void:
        shas = {json.dumps((quality[a]["phases"][str(T)].get("windows_sha256")), sort_keys=True)
                for a in QUALITY_ARMS for T in GROUPS}
        if len(shas) != 1 or "null" in shas:
            void.append(f"the quality phases read different (or unrecorded) windows: {sorted(shas)}")
    if void:
        out.update(verdict="VOID", reasons=void, tripwire=["unread"])
        return out
    # ---- quality, read once integrity holds: the per-T bars' numbers and the tripwire (rung 6, reported whatever rung
    # below decides)
    qr = quality_read(quality)
    out["quality"] = {t: {k: (round(v, 6) if isinstance(v, float) else v) for k, v in x.items()} for t, x in qr.items()}
    out["quality_unrounded"] = qr
    fired = tripwire(qr)
    out["tripwire"] = fired
    out["bucket_stability"] = {a: quality[a].get("bucket_stability") for a in QUALITY_ARMS}
    Ra = arms["Ra"]
    lens = (Ra["short"], Ra["long"])
    rates = {t: {w: _rate(arms[t], w) for w in WORKLOADS} for t in TAGS}
    pairs = {w: {f"{a}/{b}": rates[a][w] / rates[b][w] for a, b in PAIRS} for w in WORKLOADS}
    selfp = {w: {f"{a}/{b}": rates[a][w] / rates[b][w] for a, b in SELF_PAIRS} for w in WORKLOADS}
    cross = {w: {f"{a}/{b}": round(rates[a][w] / rates[b][w], 4) for a, b in (("Rc", "Ra"), ("Kc", "Ka"))}
             for w in WORKLOADS}
    g = {T: min(pairs[w].values()) for w, T in WORKLOADS.items()}       # the rule reads these, unrounded
    out["speed"] = {"decode_tok_s": rates, "g": {str(T): round(v, 4) for T, v in g.items()},     # rounded: the report's
                    "g_unrounded": {str(T): v for T, v in g.items()},
                    "pairs": {w: {k: round(v, 4) for k, v in p.items()} for w, p in pairs.items()},
                    "geomean": {w: round(math.prod(p.values()) ** (1 / len(p)), 4) for w, p in pairs.items()},
                    "self_pairs": {w: {k: round(v, 4) for k, v in p.items()} for w, p in selfp.items()},
                    "cross_block_self_pairs": cross, "telemetry": {t: telemetry_report(arms[t]) for t in TAGS + (WARMUP,)},
                    "warmup": {"decode_tok_s": {w: _rate(arms[WARMUP], w) for w in WORKLOADS},   # reported only
                               "over_Ra": {w: round(_rate(arms[WARMUP], w) / rates["Ra"][w], 4) for w in WORKLOADS}},
                    "k_vs_r_tokens": {f"{a}/{b} {w} {n}": divergence(_tok(arms[a], w, n), _tok(arms[b], w, n))
                                      for a, b in PAIRS for w in WORKLOADS for n in lens}}
    noisy = [f"{w} self-pair {k} = {v:.4f} outside [{NOISY_BAND[0]}, {NOISY_BAND[1]}]"
             for w in WORKLOADS for k, v in selfp[w].items() if not NOISY_BAND[0] <= v <= NOISY_BAND[1]]
    fn = []
    for t in TAGS:
        first = "Ra" if t.startswith("R") else "Ka"
        if t == first:
            continue
        for w in WORKLOADS:
            for n in lens:
                if _tok(arms[t], w, n) != _tok(arms[first], w, n):
                    fn.append(f"{t} != {first} on {w} at {n} tokens")
    for t in TAGS:
        for w in WORKLOADS:
            for n, ds in arms[t]["workloads"][w]["rep_digests"].items():
                if len(set(ds)) != 1:
                    fn.append(f"{t} {w} at {n} tokens: timed reps differ")
    per_t, passes = {}, {}
    for T in GROUPS:
        x = qr[str(T)]
        why = []
        if abs(x["bias"]) > Q_BIAS:
            why.append(f"|NLL gap| {abs(x['bias']):.6f} > {Q_BIAS}")
        if not agree_at_least(tuple(x["argmax_counts"]), Q_ARGMAX):
            why.append(f"argmax agreement {x['argmax_agree']:.6f} < {Q_ARGMAX}")
        if g[T] < FLIP_G:
            why.append(f"g_{T} {g[T]:.6f} (the min of four pairs at W{T}) < {FLIP_G}")
        passes[T] = not why
        per_t[str(T)] = {"result": "PASS" if not why else "MISS", "failed": why}
    out["per_T"] = per_t
    if noisy:
        rung = ("NOISY", noisy + ["NOISY is a VOID and a strike: no flip is read"])
    elif fn:
        rung = ("FUNCTION_FAIL", fn)
    else:
        bound = bound_of(passes)
        why = [f"T={T}: {per_t[str(T)]['result']}" + (f" ({'; '.join(per_t[str(T)]['failed'])})"
                                                       if per_t[str(T)]["failed"] else "") for T in GROUPS]
        if bound is None:
            rung = ("NO_FLIP", why + ["T = 2 misses: the knob stays opt-in, a reading, not a failure of K37"])
        else:
            rung = (f"FLIP_ROWS_{bound}", why + [f"every T up to {bound} passes: a separate PR may make "
                                                 f"E4B_INT8_HEAD_ROWS={bound} qwen3_moe's default, citing this read and P229"
                                                 + (" -- after the written look the TRIPWIRE asks for" if fired else "")])
    if shake:
        out.update(verdict="SHAKEDOWN", would_read=rung[0], reasons=["a shakedown record is never a reading"] + rung[1])
    else:
        out.update(verdict=rung[0], reasons=rung[1])
    return out


def tripwire_line(v: dict) -> str:
    t = v.get("tripwire")
    if t == ["unread"] or t is None:
        return "TRIPWIRE unread (VOID: nothing was read)"
    return "TRIPWIRE none" if not t else "TRIPWIRE " + "; ".join(t)


# ------------------------------------------------------------------ self-test --
E = "a" * 40


def _cap_record(want, *, plans=None):
    return {"cap": 16, "source": "default-allowlisted",
            "plans": plans if plans is not None else {str(b): {"calls": 48, "plan": CAPTURED_PLANS[b][0],
                                                               "n_split": CAPTURED_PLANS[b][1]} for b in want}}


def _dense_off():
    return {"mode": "0", "modules": 0}


def _head_record(arm, *, routed=None, passed=None, kern=None, modules=None, shape=None, captured=True, source=None,
                 rows=None):
    """The head's record as build_engine writes it (info["int8_head_bw"]): at a graph build the captures routed or
    passed buckets 1..16; an eager build (quality) called nothing yet."""
    if arm == "B16":
        return {"mode": "0", "modules": 0, "source": source or "explicit", "rows": rows or {"rows_max": 1},
                "engagement": {"modules": 0, "routed_modules_by_rows": {}, "passed_modules_by_rows": {}}}
    m = 1 if modules is None else modules
    if captured:
        rt = {"1": m, **({"2": m, "4": m, "8": m} if arm == "K" else {})}
        ps = {"16": m, **({} if arm == "K" else {"2": m, "4": m, "8": m})}
    else:
        rt, ps = {}, {}
    eng = {"modules": m, "routed_modules_by_rows": rt if routed is None else routed,
           "passed_modules_by_rows": ps if passed is None else passed}
    if arm == "K":
        eng["routed_kernel_by_rows"] = kern if kern is not None else {k: (K36 if k == "1" else K37)
                                                                      for k in eng["routed_modules_by_rows"]}
    return {"mode": "1", "modules": m, "shape": list(shape or HEAD_SHAPE), "int8_bytes": HEAD_SHAPE[0] * HEAD_SHAPE[1],
            "source": source or "default-allowlisted",
            "rows": rows if rows is not None else (dict(ROWS_RECORD) if arm == "K" else {"rows_max": 1}),
            "engagement": eng}


def _telem(pl=575.0):
    read = {"power.limit": pl, "clocks.max.sm": 3090.0, "temperature.gpu": 45.0, "clocks.sm": 2842.0, "power.draw": 120.0}
    return {"before": dict(read), "after": {**read, "temperature.gpu": 52.0}}


def _fake_arm(tag, rates, toks, *, e4b=E, census=None, graphs=None, tally=None, rep_ok=True, cap=None, sms=SMS,
              reps=REPS, head=None, dense=None, power=575.0, replays=None, wrong_call=None, shake=False, torch_v=TORCH,
              prompts="p", short=32, gnf4=GNF4_SHA):
    arm = tag[:1]
    w = {}
    for name, b in WORKLOADS.items():
        tk = toks[:b]
        hc = {"routed": {}, "passed": {}}
        if wrong_call == name:
            hc["passed" if arm == "K" else "routed"][str(b)] = 3
        w[name] = {"batch": b, "decode_tok_s": rates[name], "tokens": {str(short): tk, "160": tk},
                   "dispatch": {}, "head_calls": hc,
                   "graph_delta": {str(b): {"replays": (replays or {}).get(name, 6 * 128), "eager_steps": 0}},
                   "rep_digests": {str(short): ["x"] * (reps - 1) + ["x" if rep_ok else "y"], "160": ["z"] * reps},
                   "telemetry": {str(short): _telem(power if name == "W4" else 575.0), "160": _telem()}}
    return {"arm": arm, "tag": tag, "status": "ok", "shakedown": shake, "e4b_sha": e4b, "gnf4_sha": gnf4,
            "torch": torch_v, "model": MODEL, "revision": REV, "sm_count": sms,
            "graph_status": graphs or {str(b): "graph" for b in BUCKETS}, "fusions": census or dict(CENSUS),
            "fusion_modes": {k: "auto" for k in FUSION_KNOBS}, "fusion_sources": {k: "default-allowlisted" for k in FUSION_KNOBS},
            "attn_split_cap": cap if cap is not None else _cap_record(BUCKETS),
            "dense_gemv_bw": dense if dense is not None else _dense_off(),
            "int8_head_bw": head if head is not None else _head_record(arm),
            "prompts_sha256": prompts, "short": short, "long": 160, "reps": reps, "workloads": w,
            "dispatch_build": dict(tally or {"bw_prmt32": 192, "dotpad": 0})}


def _fake_phase(arm, T, *, bias=0.0, agree=1.0, kl=0.0, floor=0.0, n=Q_WINDOWS, group=None, cap=None,
                tokens=SERVED_TOKENS, calls=None, kern=None, rep=True, sha="w", cont=Q_CONT):
    keys = PHASE_ARMS[arm]
    per = {}
    for key in keys:
        cnt = T if key == "rep" else n
        per[key] = [{"window": i, "nll": 2.0 + 0.01 * i + (bias if key == "ON" else 0.0),
                     "argmax_agree": agree if key == "ON" else 1.0,
                     **({} if key == "R" else {"kl": kl if key == "ON" else floor})} for i in range(cnt)]
    if not rep and "rep" in per:
        per["rep"] = [{k: v for k, v in x.items() if k != "kl"} for x in per["rep"]]
    blocks = -(-tokens // 16)
    qk = {"passes": n // T + (1 if arm == "R" else 0), "asked": [656], "max_tokens_per_seq": [tokens],
          "blocks_per_seq": [blocks], "split_cap": CAP, "plan_clash": [],
          "plan_bT": CAPTURED_PLANS[T][0] if blocks >= 128 else 11}
    t = str(T)
    hc = calls if calls is not None else (
        {"routed": {t: 127 * n // T}, "passed": {"512": n}} if arm == "K"
        else {"routed": {}, "passed": {t: 127 * n // T, "512": n}} if arm == "R" else {"routed": {}, "passed": {}})
    meas = {"modules": 0 if arm == "B16" else 1, "routed_modules_by_rows": {k: 1 for k in hc["routed"]},
            "passed_modules_by_rows": {k: 1 for k in hc["passed"]}}
    if arm == "K":
        meas["routed_kernel_by_rows"] = kern if kern is not None else {k: K37 for k in hc["routed"]}
    return {"phase": "off" if arm == "R" else "on", "arms": list(keys), "windows": {"wikitext": n},
            "windows_sha256": {"wikitext": sha}, "group": group or T, "prompt": Q_PROMPT, "cont": cont,
            "per_window": {"wikitext": per}, "rep_identical": {"wikitext": True} if arm == "R" else {},
            "attn_split_cap": cap if cap is not None else _cap_record((T,)), "quality_kv": qk, "head_calls": hc,
            "int8_head_bw_measure": meas, "dispatch_measure": {}}


def _fake_quality(arm, *, per_t=None, head=None, e4b=E, shake=False, sms=SMS, drop_t=None, torch_v=TORCH):
    per_t = per_t or {}
    phases = {str(T): _fake_phase(arm, T, **per_t.get(T, {})) for T in GROUPS if T != drop_t}
    return {"mode": "quality", "arm": arm, "status": "ok", "shakedown": shake, "e4b_sha": e4b, "gnf4_sha": GNF4_SHA,
            "torch": torch_v, "model": MODEL, "revision": REV, "sm_count": sms, "fusions": dict(CENSUS),
            "fusion_modes": {k: "auto" for k in FUSION_KNOBS}, "fusion_sources": {k: "default-allowlisted" for k in FUSION_KNOBS},
            "dense_gemv_bw": _dense_off(), "int8_head_bw": head if head is not None else _head_record(arm, captured=False),
            "phases": phases, "bucket_stability": {"2": {"rows_identical": 4, "rows": 4, "max_abs_diff": 0.0}}}


# a plausible scale (decode tok/s at 2, 4 and 8 requests), exact: every pair 1.03, every self-pair 1.000
R_RATE = {"W2": 400.0, "W4": 700.0, "W8": 1100.0}
BASE = {t: (dict(R_RATE) if t.startswith("R") else {w: v * 1.03 for w, v in R_RATE.items()}) for t in TAGS + (WARMUP,)}


def _cases() -> list:
    toks = [[i, i + 1] for i in range(8)]
    other = [[i, i + 2] for i in range(8)]
    late = [[i, i + 1] if i else [i, i + 9] for i in range(8)]          # row 0 moves at step 1

    def arms(**over):
        out = {}
        for t, rates in BASE.items():
            kw = dict(over.get(t, {}))
            rr = {**rates, **kw.pop("rates", {})}
            out[t] = _fake_arm(t, rr, kw.pop("toks", toks), **kw)
        return out

    def qual(**over):
        return {a: _fake_quality(a, **over.get(a, {})) for a in QUALITY_ARMS}

    def scale(w, f, which="K"):
        return {t: {"rates": {w: BASE[t][w] * f / 1.03}} for t in TAGS if t.startswith(which)}

    def r(a=None, q=None):
        return reduce(arms() if a is None else a, qual() if q is None else q, E)

    def v(a=None, q=None):
        return r(a, q)["verdict"]

    def kq(per_t):
        return qual(K={"per_t": per_t})
    cases = [
        ("every pair 1.03, quality exact: FLIP_ROWS_8", v() == "FLIP_ROWS_8"),
        ("g_8 at 1.009 on every pair caps at 4", v(arms(**scale("W8", 1.009))) == "FLIP_ROWS_4"),
        ("g_T is the min of FOUR pairs: the second block at 1.005 at W8 caps at 4",
         (lambda x: x["verdict"] == "FLIP_ROWS_4" and x["speed"]["g"]["8"] == 1.005)(
             r(arms(Kc={"rates": {"W8": 1100.0 * 1.005}}, Kd={"rates": {"W8": 1100.0 * 1.005}})))),
        ("the first block passing is not enough", v(arms(Kc={"rates": {"W2": 400.0 * 1.0}}, Kd={"rates": {"W2": 400.0}}))
         == "NO_FLIP"),
        ("a MISS at 4 caps it at 2 even if 8 passes", v(arms(**scale("W4", 1.0))) == "FLIP_ROWS_2"),
        ("T = 2 missing on speed is NO_FLIP", v(arms(**scale("W2", 1.005))) == "NO_FLIP"),
        ("exactly 1.01 on every pair at every T passes",
         v(arms(**{t: {"rates": {w: BASE[t][w] / 1.03 * 1.01 for w in WORKLOADS}} for t in TAGS if t.startswith("K")}))
         == "FLIP_ROWS_8"),
        ("the bars read the UNROUNDED min: 1.00996 at W2 (rounds to 1.01) is NO_FLIP",
         (lambda x: x["verdict"] == "NO_FLIP" and x["speed"]["g"]["2"] == 1.01 and x["speed"]["g_unrounded"]["2"] < 1.01)(
             r(arms(**scale("W2", 1.00996))))),
        ("quality at T = 2 over the bias bar is NO_FLIP", v(q=kq({2: {"bias": 0.021}})) == "NO_FLIP"),
        ("quality at T = 4 over the bias bar caps at 2", v(q=kq({4: {"bias": -0.0201}})) == "FLIP_ROWS_2"),
        ("a bias just under 0.02 passes (and fires the tripwire)",
         (lambda x: x["verdict"] == "FLIP_ROWS_8" and x["tripwire"])(r(q=kq({8: {"bias": 0.0199999}})))),
        ("argmax 121/128 (0.9453) at T = 8 caps at 4", v(q=kq({8: {"agree": 121 / 128}})) == "FLIP_ROWS_4"),
        ("argmax 122/128 (0.9531) passes", v(q=kq({8: {"agree": 122 / 128}})) == "FLIP_ROWS_8"),
        ("a window scored on 127 positions (121/127) is a VOID, never rounded", v(q=kq({4: {"agree": 121 / 127}})) == "VOID"),
        ("the integrality check: 121/128 is integral, 121/127 is not",
         integral_agree(121 / 128) and not integral_agree(121 / 127)),
        ("pooled counts at the 0.95 edge: 2919/3072 passes, 2918 misses",
         agree_at_least((2919, 3072), Q_ARGMAX) and not agree_at_least((2918, 3072), Q_ARGMAX)),
        ("pooled counts at the 0.98 edge: 3011/3072 holds, 3010 fires",
         agree_at_least((3011, 3072), TRIP_ARGMAX) and not agree_at_least((3010, 3072), TRIP_ARGMAX)),
        ("a window with no KL on the scored arm is a VOID, never read as 0",
         (lambda q: (q["K"]["phases"]["2"]["per_window"]["wikitext"]["ON"][0].pop("kl"), v(q=q))[1])(kq({})) == "VOID"),
        ("the failed bar is named", any("argmax agreement" in y for y in r(q=kq({2: {"agree": 0.9}}))["reasons"])),
        ("a self-pair at 0.984 at W4 is NOISY",
         (lambda x: x["verdict"] == "NOISY" and any("W4 self-pair Kb/Ka" in y for y in x["reasons"]))(
             r(arms(Kb={"rates": {"W4": 700.0 * 1.03 * 0.984}})))),
        ("a self-pair at 1.016 at W2 is NOISY", v(arms(Rd={"rates": {"W2": 400.0 * 1.016}})) == "NOISY"),
        ("the second block's self-pair at W8 counts", v(arms(Kd={"rates": {"W8": 1100.0 * 1.03 * 1.02}})) == "NOISY"),
        ("NOISY outranks a passing g", v(arms(Rb={"rates": {"W2": 390.0}}, Kb={"rates": {"W2": 420.0}})) == "NOISY"),
        ("the band's edges pass (0.985 and 1.015 exactly)",
         v(arms(Ra={"rates": {"W2": 200.0}}, Rb={"rates": {"W2": 203.0}}, Ka={"rates": {"W2": 400.0}},
                Kb={"rates": {"W2": 394.0}})) == "FLIP_ROWS_8"),
        ("a cross-block drift is reported, never NOISY", v(arms(Rc={"rates": {"W2": 390.0}}, Rd={"rates": {"W2": 390.0}},
                                                               Kc={"rates": {"W2": 390.0 * 1.03}},
                                                               Kd={"rates": {"W2": 390.0 * 1.03}})) == "FLIP_ROWS_8"),
        ("K may decode other tokens than R (reported)",
         (lambda x: x["verdict"] == "FLIP_ROWS_8" and x["speed"]["k_vs_r_tokens"]["Ka/Ra W2 32"]["rows_identical"] == 1)(
             r(arms(**{t: {"toks": late} for t in TAGS if t.startswith("K")})))),
        ("a K arm differs from Ka: FUNCTION_FAIL", v(arms(Kd={"toks": other})) == "FUNCTION_FAIL"),
        ("an R arm differs from Ra: FUNCTION_FAIL", v(arms(Rc={"toks": other})) == "FUNCTION_FAIL"),
        ("timed reps differ: FUNCTION_FAIL", v(arms(Rd={"rep_ok": False})) == "FUNCTION_FAIL"),
        ("NOISY outranks FUNCTION_FAIL", v(arms(Kd={"toks": other, "rates": {"W8": 1200.0}})) == "NOISY"),
        ("FUNCTION_FAIL outranks a quality miss", v(arms(Kd={"toks": other}), kq({2: {"bias": 0.5}})) == "FUNCTION_FAIL"),
        # ---- rung 6
        ("TRIPWIRE none on the clean read", (lambda x: x["tripwire"] == [] and tripwire_line(x) == "TRIPWIRE none")(r())),
        ("|NLL gap| 0.0051 fires and the verdict holds",
         (lambda x: x["verdict"] == "FLIP_ROWS_8" and len(x["tripwire"]) == 1 and "T=4: |NLL gap|" in x["tripwire"][0]
          and "written look" in x["reasons"][-1])(r(q=kq({4: {"bias": -0.0051}})))),
        ("|NLL gap| just under 0.005 does not fire", r(q=kq({4: {"bias": 0.0049999}}))["tripwire"] == []),
        ("argmax 125/128 (0.9766) fires", any("argmax" in y for y in r(q=kq({2: {"agree": 125 / 128}}))["tripwire"])),
        ("argmax 126/128 (0.9844) does not fire", r(q=kq({2: {"agree": 126 / 128}}))["tripwire"] == []),
        ("a zero floor falls to 0.002: KL 0.0019 does not fire", r(q=kq({8: {"kl": 0.0019}}))["tripwire"] == []),
        ("a zero floor: KL 0.0021 fires", any("KL" in y for y in r(q=kq({8: {"kl": 0.0021}}))["tripwire"])),
        ("an absent floor falls to 0.002 (no rep KL)",
         r(q=qual(R={"per_t": {2: {"rep": False}}}, K={"per_t": {2: {"kl": 0.0019}}}))["tripwire"] == []
         and r(q=qual(R={"per_t": {2: {"rep": False}}}, K={"per_t": {2: {"kl": 0.0021}}}))["tripwire"] != []),
        ("a floor of 0.001: KL 0.0029 holds, 0.0031 fires",
         r(q=qual(R={"per_t": {4: {"floor": 0.001}}}, K={"per_t": {4: {"kl": 0.0029}}}))["tripwire"] == []
         and r(q=qual(R={"per_t": {4: {"floor": 0.001}}}, K={"per_t": {4: {"kl": 0.0031}}}))["tripwire"] != []),
        ("the floor is read per T: T = 4's floor does not lift T = 8's bound",
         r(q=qual(R={"per_t": {4: {"floor": 0.01}}}, K={"per_t": {8: {"kl": 0.0021}}}))["tripwire"] != []),
        ("every fired condition is listed, at every T",
         len(r(q=kq({2: {"bias": 0.006, "agree": 124 / 128, "kl": 0.01}, 8: {"bias": -0.006}}))["tripwire"]) == 4),
        ("the tripwire is reported on NO_FLIP too",
         (lambda x: x["verdict"] == "NO_FLIP" and x["tripwire"])(r(arms(**scale("W2", 1.0)), kq({2: {"bias": 0.006}})))),
        ("the tripwire never changes the verdict (FLIP_ROWS_8 with all three conditions fired)",
         v(q=kq({T: {"bias": 0.01, "agree": 123 / 128, "kl": 0.05} for T in GROUPS})) == "FLIP_ROWS_8"),
        ("TRIPWIRE prints each condition", tripwire_line(r(q=kq({2: {"bias": 0.006}}))).startswith("TRIPWIRE T=2: |NLL gap|")),
        ("a VOID's tripwire is unread, never none", tripwire_line(r(arms(Ka={"reps": 3}))) == "TRIPWIRE unread (VOID: nothing was read)"),
        # ---- VOID
        ("every process at three reps, not five, is VOID", v(arms(**{t: {"reps": 3} for t in TAGS + (WARMUP,)})) == "VOID"),
        ("one arm at three reps is VOID", v(arms(Ra={"reps": 3})) == "VOID"),
        ("a missing second-block arm is VOID", v({k: x for k, x in arms().items() if k != "Rd"}) == "VOID"),
        ("a missing warm-up process is VOID", v({k: x for k, x in arms().items() if k != WARMUP}) == "VOID"),
        ("a missing quality phase (B16) is VOID", v(q={k: x for k, x in qual().items() if k != "B16"}) == "VOID"),
        ("a quality phase missing T = 8 is VOID", v(q=qual(K={"drop_t": 8})) == "VOID"),
        ("the warm-up's integrity counts: another commit is VOID", v(arms(Rw={"e4b": "b" * 40})) == "VOID"),
        ("the warm-up's power counts", v(arms(Rw={"power": 450.0})) == "VOID"),
        ("a slow warm-up never reaches a pair or a self-pair",
         v(arms(Rw={"rates": {w: R_RATE[w] * 0.966 for w in WORKLOADS}})) == "FLIP_ROWS_8"),
        ("the warm-up is reported", (lambda x: x["speed"]["warmup"]["over_Ra"]["W2"] == 0.96)(
            r(arms(Rw={"rates": {"W2": 384.0}})))),
        ("K routed nothing at 4 rows (a zero count)",
         v(arms(Kb={"head": _head_record("K", routed={"1": 1, "2": 1, "8": 1})})) == "VOID"),
        ("K passed 8 rows to F.linear", v(arms(Kc={"head": _head_record("K", passed={"8": 1, "16": 1})})) == "VOID"),
        ("K's 4 rows on another kernel",
         v(arms(Ka={"head": _head_record("K", kern={"1": K36, "2": K37, "4": K36, "8": K37})})) == "VOID"),
        ("K's one row not on K36", v(arms(Kd={"head": _head_record("K", kern={"1": K37, "2": K37, "4": K37, "8": K37})})) == "VOID"),
        ("K's route record at another plan",
         v(arms(Ka={"head": _head_record("K", rows={**ROWS_RECORD, "plan": {**ROWS_RECORD["plan"], "8": [64, 256, 4]}})})) == "VOID"),
        ("K at rows_max 4", v(arms(Kb={"head": _head_record("K", rows={**ROWS_RECORD, "rows_max": 4})})) == "VOID"),
        ("R routed 2 rows", v(arms(Rb={"head": _head_record("R", routed={"1": 1, "2": 1})})) == "VOID"),
        ("R passed no 8-row call (a zero count)", v(arms(Rc={"head": _head_record("R", passed={"2": 1, "4": 1, "16": 1})})) == "VOID"),
        ("R's one row fell back to F.linear", v(arms(Ra={"head": _head_record("R", passed={"1": 1, "2": 1, "4": 1, "8": 1})})) == "VOID"),
        ("R routed no one-row call", v(arms(Rd={"head": _head_record("R", routed={})})) == "VOID"),
        ("the head swapped by an explicit 1, not the default", v(arms(Ka={"head": _head_record("K", source="explicit")})) == "VOID"),
        ("the head at another shape", v(arms(Ra={"head": _head_record("R", shape=[151936, 4096])})) == "VOID"),
        ("two heads swapped", v(arms(Kc={"head": _head_record("K", modules=2)})) == "VOID"),
        ("R with the knob (K's route record)", v(arms(Rc={"head": _head_record("R", rows=dict(ROWS_RECORD))})) == "VOID"),
        ("B16 swapped the head", v(q=qual(B16={"head": _head_record("R", captured=False)})) == "VOID"),
        ("B16 off by a default, not the explicit 0", v(q=qual(B16={"head": _head_record("B16", source="default-off: x")})) == "VOID"),
        ("B16's head was called as an int8 head", v(q=qual(B16={"per_t": {4: {"calls": {"routed": {"4": 9}, "passed": {}}}}})) == "VOID"),
        ("K's quality pass passed T rows to F.linear",
         v(q=kq({4: {"calls": {"routed": {"4": 381}, "passed": {"4": 2, "512": 24}}}})) == "VOID"),
        ("K's quality pass routed nothing at T (a zero count)",
         v(q=kq({8: {"calls": {"routed": {}, "passed": {"512": 24}}}})) == "VOID"),
        ("K's quality T on another kernel", v(q=kq({2: {"kern": {"2": K36}}})) == "VOID"),
        ("R's quality pass routed T rows", v(q=qual(R={"per_t": {2: {"calls": {"routed": {"2": 5}, "passed": {"2": 1500}}}}})) == "VOID"),
        ("R's quality pass passed nothing at T", v(q=qual(R={"per_t": {8: {"calls": {"routed": {}, "passed": {"512": 24}}}}})) == "VOID"),
        ("bucket T never replayed in W_T", v(arms(Kb={"replays": {"W4": 0}})) == "VOID"),
        ("an eager T-row call on the other arm's route in W_T", v(arms(Rc={"wrong_call": "W8"})) == "VOID"),
        ("a bucket left eager at the build", v(arms(Kd={"graphs": {**{str(b): "graph" for b in BUCKETS}, "8": "eager"}})) == "VOID"),
        ("a timed block under the 575 W floor", v(arms(Ka={"power": 450.0})) == "VOID"),
        ("an unreadable power limit", v(arms(Rb={"power": None})) == "VOID"),
        ("exactly 575 W passes", v(arms(**{t: {"power": 575.0} for t in TAGS})) == "FLIP_ROWS_8"),
        ("the telemetry is reported", r()["speed"]["telemetry"]["Ra"] == {
            "power_limit_w": [575.0], "clocks_max_sm": [3090.0], "temperature_max": 52.0, "sm_clock_min": 2842.0}),
        ("the arms on different prompts", v(arms(Kc={"prompts": "q"})) == "VOID"),
        ("the arms at different lengths", v(arms(Kc={"short": 40})) == "VOID"),
        ("a void slope", v(arms(Rb={"rates": {"W8": None}})) == "VOID"),
        ("a card that is not the 5090", v(arms(Kd={"sms": 188})) == "VOID"),
        ("another grouped-nf4-gemm", v(arms(Ka={"gnf4": "929a6906c12deba256a4ad4b18162681719f5cd1"})) == "VOID"),
        ("another torch", v(arms(Ka={"torch_v": "2.9.0+cu128"})) == "VOID"),
        ("another torch in a quality phase", v(q=qual(R={"torch_v": "2.9.0+cu128"})) == "VOID"),
        ("wrong commit", v(arms(Kc={"e4b": "b" * 40})) == "VOID"),
        ("wrong commit in a quality phase", v(q=qual(B16={"e4b": "b" * 40})) == "VOID"),
        ("the census off the default", v(arms(Ra={"census": {**CENSUS, "fuse_qkv_n": 0}})) == "VOID"),
        ("the GEMV on dot-pad at capture", v(arms(Rc={"tally": {"dotpad": 96}})) == "VOID"),
        ("the dense attention GEMV on", v(arms(Kd={"dense": {"mode": "1", "modules": 96}})) == "VOID"),
        ("the cap planned no bucket", v(arms(Kc={"cap": _cap_record(BUCKETS, plans={})})) == "VOID"),
        ("an explicit cap of 16 is not the default", v(arms(Ka={"cap": {**_cap_record(BUCKETS), "source": "explicit"}})) == "VOID"),
        ("the quality cap explicit", v(q=kq({2: {"cap": {**_cap_record((2,)), "source": "explicit"}}})) == "VOID"),
        ("the quality pass met another batch", v(q=kq({4: {"cap": _cap_record((4, 1))}})) == "VOID"),
        ("the quality KV at P115's 656 tokens", v(q=kq({8: {"tokens": 656}})) == "VOID"),
        ("the quality at 23 windows", v(q=qual(R={"per_t": {4: {"n": 23}}})) == "VOID"),
        ("the quality at 127 positions", v(q=kq({2: {"cont": 127}})) == "VOID"),
        ("a quality group other than T", v(q=kq({8: {"group": 4}})) == "VOID"),
        ("the phases on different windows", v(q=qual(B16={"per_t": {2: {"sha": "x"}}})) == "VOID"),
        ("the records mix a shakedown and a reading", v(arms(Kc={"shake": True})) == "VOID"),
        # ---- the shakedown
        ("a shakedown (three reps, another card, a capped board, its own plans) reads SHAKEDOWN, never a verdict",
         (lambda x: x["verdict"] == "SHAKEDOWN" and x["would_read"] == "FLIP_ROWS_8")(reduce(
             arms(**{t: {"shake": True, "reps": SHAKE_REPS, "sms": 128, "power": 350.0,
                         "cap": _cap_record(BUCKETS, plans={str(b): {"calls": 9, "plan": 7, "n_split": 7} for b in BUCKETS})}
                     for t in TAGS + (WARMUP,)}),
             {a: _fake_quality(a, shake=True, sms=128) for a in QUALITY_ARMS}, E))),
        ("a shakedown at five reps is VOID (its registered count is three)",
         v(arms(**{t: {"shake": True} for t in TAGS + (WARMUP,)}), {a: _fake_quality(a, shake=True) for a in QUALITY_ARMS})
         == "VOID"),
        ("a shakedown still reads the head's engagement",
         v(arms(**{t: {"shake": True, "reps": SHAKE_REPS, **({"head": _head_record("K", routed={"1": 1})} if t == "Kb" else {})}
                   for t in TAGS + (WARMUP,)}), {a: _fake_quality(a, shake=True) for a in QUALITY_ARMS}) == "VOID"),
        # ---- reported
        ("the geomean is over four pairs", r()["speed"]["geomean"]["W4"] == 1.03),
        ("the bf16 comparison is reported, never ruling",
         (lambda x: x["verdict"] == "FLIP_ROWS_8" and x["quality"]["2"]["b16"]["R_vs_B16"]["bias"] == -0.5
          and x["quality"]["2"]["b16"]["K_vs_B16"]["bias"] == -0.5)(
             r(q=qual(B16={"per_t": {T: {"bias": 0.5, "agree": 0.5, "kl": 0.5} for T in GROUPS}})))),
        ("the K/R KL is reported", r(q=kq({4: {"kl": 0.0012}}))["quality"]["4"]["kl"] == 0.0012),
        ("bucket stability is reported", r()["bucket_stability"]["K"]["2"]["rows"] == 4),
    ]
    return cases


def self_test() -> int:
    try:
        cases = _cases()
    except Exception as e:  # noqa: BLE001 -- a rule that crashes on a registered case fails its self-test
        print(f"p232_reduce self-test FAILED [{type(e).__name__}: {e}]")
        return 1
    bad = [n for n, ok in cases if not ok]
    print(f"p232_reduce self-test {'OK' if not bad else 'FAILED ' + str(bad)} ({len(cases)} cases)")
    return 0 if not bad else 1


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--self-test", action="store_true")
    p.add_argument("--dir")
    p.add_argument("--out")
    p.add_argument("--e4b-sha", default=os.environ.get("E4B_SHA", ""))
    a = p.parse_args(argv)
    if a.self_test:
        return self_test()

    def load(name):
        f = os.path.join(a.dir, name)
        return json.load(open(f)) if os.path.exists(f) else None
    arms = {t: r for t in TAGS + (WARMUP,) if (r := load(f"arm_{t}.json"))}
    quality = {q: r for q in QUALITY_ARMS if (r := load(f"quality_{q}.json"))}
    v = reduce(arms, quality, a.e4b_sha)
    json.dump(v, open(a.out, "w"), indent=1, default=str)
    print(f"P232_VERDICT {v['verdict']} {json.dumps(v['reasons'], default=str)[:1500]}")
    print(tripwire_line(v))
    return 0


if __name__ == "__main__":
    sys.exit(main())
