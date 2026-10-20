#!/usr/bin/env python3
"""p158_reduce.py -- lane P158's registered rule (bench/p158/PREREG-p158.md; e4b#1313): K39's served read, grouped-nf4-gemm
v0.50.0's K25 ``lut="prmt"`` as Qwen3-30B-A3B's batched-decode expert GEMM (``E4B_K25_LUT=prmt``, e4b#1862). P153's
reducer (bench/p153/p153_reduce.py) by counted edits: its records, integrity checks, power floor and cap checks; K25's
decode engagement in place of the head's; the bar on W16 and the guard on W1; a token-identity gate in place of SANE,
with its null first.

Records (``p158_box.py``): ``arm_{S0a,S1a,S1b,S0b,S0c,S1c,S1d,S0d}.json`` (ABBA twice, each through P109's W16 and W1 at
five timed reps, on the default graph server) and the warm-up process ``arm_S0w.json`` (P131 Amendment 2's process 0:
integrity-checked like every arm, VOID when missing, reported only). S0 = ``E4B_K25_LUT=tree`` (named), S1 = ``prmt``.

THE ARMS ARE BITWISE-IDENTICAL BY CONTRACT: K39's decode yields the tree's codebook values with the same tiling,
instruction shape and accumulation order (grouped-nf4-gemm's test_nf4_smallm_prmt_gpu.py, run as the premise). So the
gate is token IDENTITY, not a quality band, and its null comes first: S0 against S0 across processes.

The rule, first rung that applies:
  VOID           a record missing or not ok; P153's checks on every record (pins, card, census and modes, the bandwidth
                 GEMV, the captured buckets, bucket 1 replayed, the cap at its default with its plans at capture); the
                 dense attention GEMV off; the int8 head at its qwen3_moe default (P153's FLIP): on, default-allowlisted,
                 the one lm_head at 151936 x 2048, routing one-row calls with none falling back; K25's decode: ``info["k25_lut"]`` resolves to the arm's
                 decode (S0 tree, S1 prmt), and its engagement record after the build reads EXACTLY
                 ``{"<arm's lut>|le256": 960}`` (48 MoE layers x the four multi-row buckets 2, 4, 8, 16 x five forwards:
                 the residual licence's two probes, then two warm-ups and one capture; bucket 1 is T == 1 and stays on
                 the bandwidth GEMV), with no other decode, and after the runs the arm's decode only, its count equal
                 across every process; the bandwidth GEMV's build tally reads EXACTLY bw_prmt32 = 480 (48 layers x the
                 same five forwards at T == 1 x two GEMV calls); every timed block's power limit AND
                 enforced power limit, read either side of it, at least 575 W (unreadable is a VOID); the arms on
                 different prompts or lengths, or not at the registered five reps; a void slope.
                 THE NULL: any S0 process decodes different tokens from S0a (any row, workload or length), or any S0
                 arm's timed reps digest differently -- "instrument: tree not token-deterministic across processes".
  NOISY          (a VOID, and a strike) any self-pair -- S0b/S0a, S1b/S1a, S0d/S0c, S1d/S1c -- at W16 or W1 outside
                 [0.985, 1.015].
  FUNCTION_FAIL  (a CONTRACT finding against K39) with the null clean: an S1 process decodes different tokens from S0's
                 (any row, workload or length; each S1 against S0a), or an S1 arm's timed reps digest differently.
  FLIP           g16 = min over the FOUR S1/S0 pairs (S1a/S0a, S1b/S0b, S1c/S0c, S1d/S0d) at W16 >= 1.03 AND g1 (the
                 same at W1) >= 0.99, compared unrounded: a separate PR may make prmt qwen3_moe's K25 decode by default.
  NO_FLIP        otherwise (each failed bar named): the knob stays opt-in. A reading, not a failure of K39.

Labelled beside the verdict, never ruling: a pair is "clock-confounded" when its S1 arm's median sampled SM clock over
the timed W16 blocks differs from its S0 arm's by more than 2 %, either way (exact integer comparison); an arm whose sampler read nothing (or only
unreadable fields) is UNREAD, never 0, and labels nothing. Reported, never ruled: every pair and self-pair ratio (the
cross-block self-pairs too), the geomean over the four pairs, each arm's telemetry (power limits, enforced limits, max
SM clocks, the hottest and the slowest SM clock read beside its timed blocks) and its sampled clocks, power and event
reasons, and the warm-up's rates.

    python p158_reduce.py --dir RUN_DIR --out verdict_p158.json --e4b-sha SHA
    python p158_reduce.py --self-test
"""
from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys

TAGS = ("S0a", "S1a", "S1b", "S0b", "S0c", "S1c", "S1d", "S0d")      # ABBA twice
WARMUP = "S0w"     # P131 Amendment 2's warm-up process 0, before S0a: integrity-checked, reported, never scored
PAIRS = (("S1a", "S0a"), ("S1b", "S0b"), ("S1c", "S0c"), ("S1d", "S0d"))
SELF_PAIRS = (("S0b", "S0a"), ("S1b", "S1a"), ("S0d", "S0c"), ("S1d", "S1c"))
NOISY_BAND = (0.985, 1.015)
REPS = 5
WORKLOADS = ("W16", "W1")
BUCKETS = (1, 2, 4, 8, 16)
GNF4_SHA = "41ef086e6a3e2be5b158ff1bb38e0de67862a949"     # grouped-nf4-gemm v0.50.0 (gnf4#572): K39's lut="prmt"
MODEL, REV = "Qwen/Qwen3-30B-A3B", "ad44e777bcd18fa416d9da3bd8f70d33ebb85d39"
CENSUS = {"fuse_qkv_n": 48, "fuse_t1_glue_n": 193, "fuse_t1_glue_r2_n": [48, 48], "fuse_router_epilogue_n": 48}
FUSION_KNOBS = ("E4B_PAGED_FUSE_QKV", "E4B_FUSE_T1_GLUE", "E4B_FUSE_T1_GLUE_R2", "E4B_FUSE_ROUTER_EPI")
CAP = {"S0": 16, "S1": 16}                                # both arms: the qwen3_moe default (e4b#1792)
HEAD_SHAPE = [151936, 2048]                               # Qwen3-30B-A3B's lm_head: the int8 head ON by default (P153)
LUT = {"S0": "tree", "S1": "prmt"}
# K25's engagement at the build: 48 MoE layers x the multi-row buckets (T > 1 takes device grouping and, at up to 256
# (token, slot) rows, K25; bucket 1 is T == 1, the singleton bandwidth GEMV) x five forwards per row count: the residual
# licence (glue_r2.license_moe_residual, run before capture under GLUE_R2 auto) probes each layer twice (mod.mlp(h) and
# _moe_with_residual), then enable_decode_graphs runs two warm-ups and one capture. Counted where the route is chosen, so
# a replay never counts. The same five forwards at T == 1 give the bandwidth GEMV's build tally, two calls each.
K25_LAYERS, K25_BUCKETS = 48, (2, 4, 8, 16)
LICENCE_FORWARDS, WARMUPS, CAPTURES = 2, 2, 1
K25_FORWARDS_PER_BUCKET = LICENCE_FORWARDS + WARMUPS + CAPTURES                     # 5
K25_BUILD_CALLS = K25_LAYERS * len(K25_BUCKETS) * K25_FORWARDS_PER_BUCKET          # 960
GEMV_CALLS_PER_FORWARD = 2                                                          # gate_up and down at T == 1
BW_BUILD_CALLS = K25_LAYERS * K25_FORWARDS_PER_BUCKET * GEMV_CALLS_PER_FORWARD      # 480 (P153's build read 480)
MIN_POWER_W = 575                                         # the 5090 served floor (P149, P151, P153: the maintainer's)
# what the 5090's library plan gives each captured bucket at Qwen3's 4 KV heads and 4096-token slots, and the cap's
CAPTURED_PLANS = {1: (32, 16), 2: (32, 16), 4: (23, 16), 8: (11, 11), 16: (5, 5)}
FLIP_G16, NOREG_G1 = 1.03, 0.99
CLOCK_TOL_PCT = 2                                         # a pair's S1/S0 median sampled SM clock beyond 2 %: labelled
BW = ("bw_tree", "bw_prmt32", "bw_splitk")
SMS = 170                                                 # the RTX 5090: the plans above are its


def _rate(r, w):
    return r["workloads"][w]["decode_tok_s"]


def _tok(r, w, n):
    return r["workloads"][w]["tokens"][str(n)]


def divergence(a: list, b: list) -> dict:
    """Rows identical out of the rows, and each row's first divergent step (None when identical)."""
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


def w1_faults(t, r) -> list:
    """Phase D's (Amendment 4): W1 replays the bucket-1 graph captured at the build, so the engagement is the capture
    tally, bound to W1 by bucket 1's replays; an eager W1 step must pass on its own tally. The build's tally is a closed
    form here: exactly :data:`BW_BUILD_CALLS` bw_prmt32 calls."""
    gs1 = (r.get("graph_stats") or {}).get("1") or {}
    out = gemv_faults(f"{t} W1 capture", r.get("dispatch_build"))
    if (r.get("dispatch_build") or {}).get("bw_prmt32") != BW_BUILD_CALLS:
        out.append(f"{t}: bw_prmt32 at the build is {(r.get('dispatch_build') or {}).get('bw_prmt32')}, registered "
                   f"{BW_BUILD_CALLS} (48 layers x 5 forwards at T == 1 x 2 calls)")
    if not gs1.get("replays"):
        out.append(f"{t} W1: bucket 1 never replayed ({gs1}); the capture tally is not what W1 ran")
    if gs1.get("eager_steps") or any((r["workloads"]["W1"].get("dispatch") or {}).values()):
        out += gemv_faults(f"{t} W1 eager", r["workloads"]["W1"].get("dispatch"))
    return out


def record_faults(name, r, e4b_sha, arm) -> list:
    out = []
    if r.get("e4b_sha") != e4b_sha:
        out.append(f"{name}: e4b {r.get('e4b_sha')} != {e4b_sha}")
    if r.get("gnf4_sha") != GNF4_SHA:
        out.append(f"{name}: grouped-nf4-gemm {r.get('gnf4_sha')} != {GNF4_SHA}")
    if (r.get("model"), r.get("revision")) != (MODEL, REV):
        out.append(f"{name}: model {r.get('model')}@{r.get('revision')} is not {MODEL}@{REV}")
        return out
    if r.get("arm") != arm:
        out.append(f"{name}: the record says arm {r.get('arm')}")
    if r.get("sm_count") != SMS:
        out.append(f"{name}: {r.get('sm_count')} SMs, registered the RTX 5090's {SMS}")
    cen = {k: (r.get("fusions") or {}).get(k) for k in CENSUS}
    if cen != CENSUS:
        out.append(f"{name}: census {cen}, the default's {CENSUS}")
    modes, sources = r.get("fusion_modes") or {}, r.get("fusion_sources") or {}
    if any(str(modes.get(k)) != "auto" or sources.get(k) != "default-allowlisted" for k in FUSION_KNOBS):
        out.append(f"{name}: fusion modes {modes} / sources {sources}, the default is auto (default-allowlisted)")
    return out


def cap_faults(name, r, arm) -> list:
    """The split cap at its qwen3_moe default in BOTH arms (16, ``default-allowlisted``; e4b#1792), every bucket planned
    at capture."""
    sc = r.get("attn_split_cap") or {}
    plans = {int(b): p for b, p in (sc.get("plans") or {}).items()}
    out = []
    if sc.get("cap") != CAP[arm] or sc.get("source") != "default-allowlisted":
        out.append(f"{name}: cap {sc.get('cap')} from {sc.get('source')}, registered {CAP[arm]} (default-allowlisted)")
    for b, (plan, n) in CAPTURED_PLANS.items():
        p = plans.get(b) or {}
        if (p.get("plan"), p.get("n_split")) != (plan, n) or not p.get("calls"):
            out.append(f"{name}: B = {b} recorded {p}, registered plan {plan} -> {n} splits")
    if set(plans) - set(CAPTURED_PLANS):
        out.append(f"{name}: the cap met row counts {sorted(set(plans) - set(CAPTURED_PLANS))} outside {sorted(CAPTURED_PLANS)}")
    return out


def default_faults(name, r) -> list:
    """The other served levers stay at their defaults in both arms. The dense attention GEMV is off (P151's lever, no
    default). The int8 head is ON by qwen3_moe's default (P153's FLIP, e4b's INT8_HEAD_BW_DEFAULT_FAMILIES): the one
    lm_head swapped at :data:`HEAD_SHAPE` from ``default-allowlisted``, routing one-row calls with none falling back to
    ``F.linear`` at one row (P153's engagement check, carried)."""
    out = []
    dg = r.get("dense_gemv_bw") or {}
    if dg.get("mode") != "0" or dg.get("modules"):
        out.append(f"{name}: the dense attention GEMV records {dg}; registered off in both arms")
    ih = r.get("int8_head_bw") or {}
    eng = ih.get("engagement") or {}
    routed, passed = eng.get("routed_modules_by_rows") or {}, eng.get("passed_modules_by_rows") or {}
    if (ih.get("mode"), ih.get("modules"), ih.get("shape"), ih.get("source")) != ("1", 1, HEAD_SHAPE, "default-allowlisted"):
        out.append(f"{name}: the int8 head records {ih.get('mode')}/{ih.get('modules')}/{ih.get('shape')} from "
                   f"{ih.get('source')}; registered on by qwen3_moe's default at {HEAD_SHAPE}")
    if eng.get("modules") != 1 or routed != {"1": 1} or "1" in passed:
        out.append(f"{name}: the int8 head routed {routed}, passed {passed}; registered routing one-row calls with none "
                   "falling back to F.linear at one row")
    return out


def k25_faults(name, r, arm) -> list:
    """K25's decode. ``info["k25_lut"]`` resolves to the arm's decode at the build; its engagement record (one count per
    K25 call by decode, eager calls and graph captures, never a replay) reads exactly ``{"<lut>|le256":``
    :data:`K25_BUILD_CALLS` ``}`` after the build, starting from nothing, and holds only the arm's decode after the runs.
    A zero count, another decode or another count is a VOID."""
    out = []
    want = LUT[arm]
    k = r.get("k25_lut") or {}
    if (k.get("env"), k.get("lut"), k.get("why")) != (want, want, None):
        out.append(f"{name}: k25_lut resolved {k}, registered env {want!r} -> lut {want!r}")
    if r.get("k25_seen_before_build"):
        out.append(f"{name}: K25's record held {r.get('k25_seen_before_build')} before the build")
    key = f"{want}|le256"
    if r.get("k25_seen_build") != {key: K25_BUILD_CALLS}:
        out.append(f"{name}: K25's record after the build is {r.get('k25_seen_build')}, registered {{{key!r}: "
                   f"{K25_BUILD_CALLS}}} (48 layers x buckets 2, 4, 8, 16 x 5 forwards)")
    after = r.get("k25_seen_after_runs") or {}
    if set(after) != {key} or after.get(key, 0) < K25_BUILD_CALLS:
        out.append(f"{name}: K25's record after the runs is {after}, registered the arm's decode {key!r} only, at least "
                   f"the build's {K25_BUILD_CALLS}")
    return out


def power_faults(name, r) -> list:
    """Every timed block's power limit and enforced power limit, read either side of it, at least :data:`MIN_POWER_W`;
    unreadable is a fault."""
    out = []
    for w in WORKLOADS:
        tel = (r["workloads"].get(w) or {}).get("telemetry") or {}
        if sorted(tel) != sorted(str(n) for n in (r.get("short"), r.get("long"))):
            out.append(f"{name} {w}: telemetry for blocks {sorted(tel)}, registered one per timed length")
        for n, t in tel.items():
            for side in ("before", "after"):
                for f in ("power.limit", "enforced.power.limit"):
                    pl = ((t or {}).get(side) or {}).get(f)
                    if pl is None or pl < MIN_POWER_W:
                        out.append(f"{name} {w} at {n} tokens: {f} {pl} W {side} the block, the floor {MIN_POWER_W} W")
    return out


def _samples(r, w="W16"):
    return [s for blk in ((r["workloads"].get(w) or {}).get("clock_samples") or {}).values()
            for s in ((blk or {}).get("samples") or [])]


def clock_median(r, w="W16"):
    """The median sampled SM clock over an arm's timed ``w`` blocks, or None (UNREAD) when no sample read a clock."""
    xs = [s["clocks.sm"] for s in _samples(r, w) if isinstance(s.get("clocks.sm"), (int, float))]
    return statistics.median(xs) if xs else None


def clock_report(r) -> dict:
    out = {}
    for w in WORKLOADS:
        ss = _samples(r, w)

        def vals(k, ss=ss):
            return [s[k] for s in ss if isinstance(s.get(k), (int, float))]
        reasons = sorted({s["clocks_event_reasons.active"] for s in ss if isinstance(s.get("clocks_event_reasons.active"), int)})
        out[w] = {"n": len(ss), "sm_mhz_median": clock_median(r, w), "sm_mhz_min": min(vals("clocks.sm"), default=None),
                  "power_instant_w_median": statistics.median(vals("power.draw.instant")) if vals("power.draw.instant") else None,
                  "power_instant_w_max": max(vals("power.draw.instant"), default=None),
                  "enforced_limit_w": sorted(set(vals("enforced.power.limit"))), "event_reasons": reasons}
    return out


def clock_labels(arms) -> dict:
    """Per pair, beside the verdict: "clock-confounded" when S1's median sampled W16 SM clock is more than
    :data:`CLOCK_TOL_PCT` percent off S0's, either way (compared as ``100 |S1 - S0| > 2 S0``, exact on the
    whole-MHz samples); "clock unread" when either side has no clock sample. Never ruling."""
    out = {}
    for a, b in PAIRS:
        ca, cb = clock_median(arms[a]), clock_median(arms[b])
        if ca is None or cb is None:
            out[f"{a}/{b}"] = "clock unread"
        elif 100 * abs(ca - cb) > CLOCK_TOL_PCT * cb:
            out[f"{a}/{b}"] = f"clock-confounded: {a} {ca:.0f} MHz against {b} {cb:.0f} MHz ({ca / cb - 1:+.2%})"
        else:
            out[f"{a}/{b}"] = None
    return out


def telemetry_report(r) -> dict:
    """What the card read beside one arm's timed blocks."""
    reads = [s for w in WORKLOADS for t in ((r["workloads"].get(w) or {}).get("telemetry") or {}).values()
             for s in ((t or {}).get("before") or {}, (t or {}).get("after") or {})]

    def vals(k):
        return [x[k] for x in reads if x.get(k) is not None]
    return {"power_limit_w": sorted(set(vals("power.limit"))), "enforced_limit_w": sorted(set(vals("enforced.power.limit"))),
            "clocks_max_sm": sorted(set(vals("clocks.max.sm"))),
            "temperature_max": max(vals("temperature.gpu"), default=None), "sm_clock_min": min(vals("clocks.sm"), default=None)}


def _token_diffs(arms, tags, ref, lens) -> list:
    out = []
    for t in tags:
        for w in WORKLOADS:
            for n in lens:
                if _tok(arms[t], w, n) != _tok(arms[ref], w, n):
                    out.append(f"{t} != {ref} on {w} at {n} tokens")
    return out


def _rep_diffs(arms, tags) -> list:
    return [f"{t} {w} at {n} tokens: timed reps differ" for t in tags for w in WORKLOADS
            for n, ds in arms[t]["workloads"][w]["rep_digests"].items() if len(set(ds)) != 1]


def reduce(arms: dict, e4b_sha: str) -> dict:
    out = {"lane": "P158", "verdict": None, "reasons": [], "labels": {},
           "bars": {"flip_g16": FLIP_G16, "noreg_g1": NOREG_G1, "noisy_band": list(NOISY_BAND), "reps": REPS,
                    "min_power_w": MIN_POWER_W, "k25_build_calls": K25_BUILD_CALLS, "clock_tol_pct": CLOCK_TOL_PCT}}
    missing = [t for t in TAGS + (WARMUP,) if t not in arms or arms[t].get("status") != "ok"]
    if missing:
        out.update(verdict="VOID", reasons=[f"record(s) missing or not ok: {missing}"])
        return out
    void = []
    for t in TAGS + (WARMUP,):                                    # integrity reads every process, the warm-up's too
        r = arms[t]
        void += record_faults(t, r, e4b_sha, t[:2])
        void += cap_faults(t, r, t[:2])
        void += default_faults(t, r)
        void += k25_faults(t, r, t[:2])
        g = r.get("graph_status") or {}
        if [g.get(str(b)) for b in BUCKETS] != ["graph"] * len(BUCKETS):
            void.append(f"{t}: graph_status {g}")
        void += w1_faults(t, r)
        void += power_faults(t, r)
        if any(_rate(r, w) is None for w in WORKLOADS):
            void.append(f"{t}: a decode slope is void")
        if r.get("reps") != REPS:
            void.append(f"{t}: {r.get('reps')} timed reps, registered {REPS}")
    if len({arms[t].get("prompts_sha256") for t in TAGS + (WARMUP,)}) != 1:
        void.append("the arms read different prompts")
    if len({(arms[t].get("short"), arms[t].get("long"), arms[t].get("reps")) for t in TAGS + (WARMUP,)}) != 1:
        void.append("the arms ran different lengths")
    if not void:
        runs = {t: sum((arms[t].get("k25_seen_after_runs") or {}).values()) for t in TAGS + (WARMUP,)}
        if len(set(runs.values())) != 1:
            void.append(f"K25's call count after the runs differs across processes: {runs}")
    if void:
        out.update(verdict="VOID", reasons=void)
        return out
    S0a = arms["S0a"]
    lens = (S0a["short"], S0a["long"])
    # THE NULL first: the tree must be token-deterministic across processes (every S0, the warm-up's too, against S0a)
    null = _token_diffs(arms, ("S0b", "S0c", "S0d", WARMUP), "S0a", lens) + _rep_diffs(arms, ("S0a", "S0b", "S0c", "S0d", WARMUP))
    if null:
        out.update(verdict="VOID", reasons=["instrument: tree not token-deterministic across processes"] + null)
        return out
    rates = {t: {w: _rate(arms[t], w) for w in WORKLOADS} for t in TAGS}
    pairs = {w: {f"{a}/{b}": rates[a][w] / rates[b][w] for a, b in PAIRS} for w in WORKLOADS}
    selfp = {w: {f"{a}/{b}": rates[a][w] / rates[b][w] for a, b in SELF_PAIRS} for w in WORKLOADS}
    cross = {w: {f"{a}/{b}": round(rates[a][w] / rates[b][w], 4) for a, b in (("S0c", "S0a"), ("S1c", "S1a"))}
             for w in WORKLOADS}
    g16_min, g1_min = min(pairs["W16"].values()), min(pairs["W1"].values())   # the rule reads these, unrounded
    out["speed"] = {"decode_tok_s": rates, "g16": round(g16_min, 4), "g1": round(g1_min, 4),   # rounded: the report's
                    "g16_unrounded": g16_min, "g1_unrounded": g1_min,
                    "pairs": {w: {k: round(v, 4) for k, v in p.items()} for w, p in pairs.items()},
                    "geomean": {w: round(math.prod(p.values()) ** (1 / len(p)), 4) for w, p in pairs.items()},
                    "self_pairs": {w: {k: round(v, 4) for k, v in p.items()} for w, p in selfp.items()},
                    "cross_block_self_pairs": cross,
                    "telemetry": {t: telemetry_report(arms[t]) for t in TAGS + (WARMUP,)},
                    "clocks": {t: clock_report(arms[t]) for t in TAGS + (WARMUP,)},
                    "warmup": {"decode_tok_s": {w: _rate(arms[WARMUP], w) for w in WORKLOADS},   # reported only
                               "over_S0a": {w: round(_rate(arms[WARMUP], w) / rates["S0a"][w], 4) for w in WORKLOADS}},
                    "k25_seen_after_runs": {t: arms[t].get("k25_seen_after_runs") for t in TAGS + (WARMUP,)}}
    out["labels"] = {k: v for k, v in clock_labels(arms).items() if v}
    noisy = [f"{w} self-pair {k} = {v:.4f} outside [{NOISY_BAND[0]}, {NOISY_BAND[1]}]"
             for w in WORKLOADS for k, v in selfp[w].items() if not NOISY_BAND[0] <= v <= NOISY_BAND[1]]
    if noisy:
        out.update(verdict="NOISY", reasons=noisy + ["NOISY is a VOID and a strike: no flip is read"])
        return out
    # the contract, with the null clean: every S1 process against S0a, and S1's own timed reps
    fn = _token_diffs(arms, ("S1a", "S1b", "S1c", "S1d"), "S0a", lens) + _rep_diffs(arms, ("S1a", "S1b", "S1c", "S1d"))
    if fn:
        out["contract"] = {f"{a}/S0a {w} {n}": divergence(_tok(arms[a], w, n), _tok(arms["S0a"], w, n))
                           for a, _ in PAIRS for w in WORKLOADS for n in lens}
        out.update(verdict="FUNCTION_FAIL", reasons=["a CONTRACT finding against K39: prmt is not token-identical to "
                                                     "the tree in the served step"] + fn)
    elif g16_min >= FLIP_G16 and g1_min >= NOREG_G1:
        out.update(verdict="FLIP", reasons=[f"g16 {g16_min:.6f} (the min of four pairs) >= {FLIP_G16} and g1 {g1_min:.6f} "
                                            f">= {NOREG_G1}; every S1 token identical to S0's",
                                            "a separate PR may make E4B_K25_LUT=prmt qwen3_moe's K25 decode by default"])
    else:
        why = [f"g16 {g16_min:.6f} (the min of four pairs) < {FLIP_G16}"] if g16_min < FLIP_G16 else []
        why += [f"g1 {g1_min:.6f} < {NOREG_G1} (W1's no-regression line)"] if g1_min < NOREG_G1 else []
        out.update(verdict="NO_FLIP", reasons=why + ["the knob stays opt-in: a reading, not a failure of K39"])
    return out


# ------------------------------------------------------------------ self-test --
E = "a" * 40


def _cap_record(plans=None):
    return {"cap": 16, "source": "default-allowlisted",
            "plans": plans if plans is not None else {str(b): {"calls": 48, "plan": p, "n_split": n}
                                                     for b, (p, n) in CAPTURED_PLANS.items()}}


def _head_default(**over):
    rec = {"mode": "1", "modules": 1, "shape": list(HEAD_SHAPE), "source": "default-allowlisted",
           "engagement": {"modules": 1, "routed_modules_by_rows": {"1": 1},
                          "passed_modules_by_rows": {str(b): 1 for b in BUCKETS[1:]}}}
    rec.update(over)
    return rec


def _telem(pl=575.0, enforced=575.0):
    read = {"power.limit": pl, "enforced.power.limit": enforced, "clocks.max.sm": 3090.0, "temperature.gpu": 45.0,
            "clocks.sm": 2842.0, "power.draw": 120.0}
    return {"before": dict(read), "after": {**read, "temperature.gpu": 52.0}}


def _samples_block(mhz=2800.0, n=20):
    if mhz is None:
        return {"samples": [], "error": "sampler did not start"}
    return {"samples": [{"clocks.sm": mhz, "enforced.power.limit": 575.0, "power.draw.instant": 480.0,
                         "clocks_event_reasons.active": 0} for _ in range(n)], "error": None}


def _fake_arm(tag, r16, r1, toks, *, e4b=E, census=None, graphs=None, tally=None, replays=68, rep_ok=True, cap=None,
              toks16=None, sms=SMS, reps=REPS, lut=None, seen_build=None, seen_runs=None, seen_before=None, power=575.0,
              enforced=575.0, mhz=2800.0, head=None, dense=None):
    arm = tag[:2]
    d = tally or {"bw_prmt32": BW_BUILD_CALLS, "dotpad": 0}
    zero = {k: 0 for k in ("dotpad", "dotpad_splitk", "scalar", "scalar_splitk") + BW}
    key = f"{LUT[arm]}|le256"
    w = {}
    for name, rate in (("W16", r16), ("W1", r1)):
        b = 16 if name == "W16" else 1
        tk = (toks16 or toks)[:b] if name == "W16" else toks[:b]
        w[name] = {"batch": b, "decode_tok_s": rate, "tokens": {"32": tk, "160": tk}, "dispatch": dict(zero),
                   "rep_digests": {"32": ["x"] * (reps - 1) + ["x" if rep_ok else "y"], "160": ["z"] * reps},
                   "telemetry": {"32": _telem(power if name == "W1" else 575.0, enforced if name == "W16" else 575.0),
                                 "160": _telem()},
                   "clock_samples": {"32": _samples_block(mhz), "160": _samples_block(mhz)}}
    stats = {str(b): {"replays": replays if b == 1 else 44, "eager_steps": 0} for b in BUCKETS}
    return {"arm": arm, "tag": tag, "status": "ok", "e4b_sha": e4b, "gnf4_sha": GNF4_SHA, "model": MODEL, "revision": REV,
            "sm_count": sms, "graph_status": graphs or {str(b): "graph" for b in BUCKETS}, "fusions": census or dict(CENSUS),
            "fusion_modes": {k: "auto" for k in FUSION_KNOBS}, "fusion_sources": {k: "default-allowlisted" for k in FUSION_KNOBS},
            "attn_split_cap": cap if cap is not None else _cap_record(),
            "dense_gemv_bw": dense if dense is not None else {"mode": "0", "modules": 0},
            "int8_head_bw": head if head is not None else _head_default(),
            "k25_lut": lut if lut is not None else {"env": LUT[arm], "lut": LUT[arm], "why": None},
            "k25_seen_before_build": seen_before or {},
            "k25_seen_build": seen_build if seen_build is not None else {key: K25_BUILD_CALLS},
            "k25_seen_after_runs": seen_runs if seen_runs is not None else {key: K25_BUILD_CALLS + 96},
            "prompts_sha256": "p", "short": 32, "long": 160, "reps": reps, "workloads": w,
            "graph_stats": stats, "dispatch_build": dict(d)}


# P153's scale at W16 (~983 tok/s) and W1 (~264), exact: every W16 pair 1.10, every W1 pair 1.000, every self-pair 1.000
BASE = {t: ((983.0, 264.0) if t.startswith("S0") else (983.0 * 1.10, 264.0)) for t in TAGS + (WARMUP,)}


def _cases() -> list:
    toks = [[i, i + 1] for i in range(16)]
    other = [[i, i + 2] for i in range(16)]
    row7 = [[i, i + 1] if i != 7 else [i, i + 9] for i in range(16)]      # one W16 row differs at step 1

    def arms(**over):
        out = {}
        for t, (r16, r1) in BASE.items():
            kw = dict(over.get(t, {}))
            out[t] = _fake_arm(t, kw.pop("r16", r16), kw.pop("r1", r1), kw.pop("toks", toks), **kw)
        return out
    v = lambda a: reduce(a, E)["verdict"]  # noqa: E731
    r = lambda a: reduce(a, E)  # noqa: E731
    s1 = [t for t in TAGS if t.startswith("S1")]
    cases = [
        ("every W16 pair 1.10, W1 flat: FLIP", v(arms()) == "FLIP"),
        ("g16 is the min of FOUR pairs: the second block at 1.02 does not flip",
         (lambda x: x["verdict"] == "NO_FLIP" and x["speed"]["g16"] == 1.02)(
             r(arms(S1c={"r16": 983.0 * 1.02}, S1d={"r16": 983.0 * 1.02})))),
        ("the first block passing is not enough", v(arms(S1c={"r16": 983.0 * 1.025}, S1d={"r16": 983.0 * 1.025})) == "NO_FLIP"),
        ("exactly 1.03 on every pair flips", v(arms(**{t: {"r16": 983.0 * 1.03} for t in s1})) == "FLIP"),
        ("1.0299 on one pair does not", v(arms(S1a={"r16": 983.0 * 1.0299}, S1b={"r16": 983.0 * 1.0299})) == "NO_FLIP"),
        ("the bars read the UNROUNDED min: a g16 of 1.02996 (rounds to 1.03) is NO_FLIP",
         (lambda x: x["verdict"] == "NO_FLIP" and x["speed"]["g16"] == 1.03 and x["speed"]["g16_unrounded"] < 1.03)(
             r(arms(**{t: {"r16": 983.0 * 1.02996} for t in s1})))),
        ("the W1 guard: a g1 of 0.98996 (rounds to 0.99) is NO_FLIP, the guard named",
         (lambda x: x["verdict"] == "NO_FLIP" and x["speed"]["g1"] == 0.99 and any("g1" in y for y in x["reasons"]))(
             r(arms(**{t: {"r1": 264.0 * 0.98996} for t in s1})))),
        ("the W1 guard fires on one block (0.98; its self-pair stays 1)",
         (lambda x: x["verdict"] == "NO_FLIP" and any("g1" in y for y in x["reasons"]))(
             r(arms(S1c={"r1": 264.0 * 0.98}, S1d={"r1": 264.0 * 0.98})))),
        ("exactly 0.99 at W1 passes the guard", v(arms(**{t: {"r1": 264.0 * 0.99} for t in s1})) == "FLIP"),
        ("a W16 self-pair at 0.984 is NOISY", (lambda x: x["verdict"] == "NOISY" and any("W16 self-pair S0b/S0a" in y for y in x["reasons"]))(
            r(arms(S0b={"r16": 983.0 * 0.984})))),
        ("a W1 self-pair at 1.016 is NOISY too", v(arms(S1d={"r1": 264.0 * 1.016})) == "NOISY"),
        ("the band's edges pass (0.985 and 1.015 exactly)",
         v(arms(S0a={"r16": 1000.0}, S0b={"r16": 985.0}, S1a={"r16": 1100.0}, S1b={"r16": 1116.5})) == "FLIP"),
        ("NOISY outranks a passing g16", v(arms(S0b={"r16": 940.0})) == "NOISY"),
        # the identity gate and its null
        ("THE NULL: S0b's tokens differ from S0a's -> VOID (instrument), never FUNCTION_FAIL",
         (lambda x: x["verdict"] == "VOID" and x["reasons"][0] == "instrument: tree not token-deterministic across processes")(
             r(arms(S0b={"toks": other})))),
        ("the null reads W16 rows too", (lambda x: x["verdict"] == "VOID" and "tree not token-deterministic" in x["reasons"][0])(
            r(arms(S0d={"toks16": row7})))),
        ("the null reads the warm-up", (lambda x: x["verdict"] == "VOID" and "tree not token-deterministic" in x["reasons"][0])(
            r(arms(S0w={"toks": other})))),
        ("the null reads an S0 arm's own timed reps", (lambda x: x["verdict"] == "VOID" and "tree not" in x["reasons"][0])(
            r(arms(S0c={"rep_ok": False})))),
        ("the null outranks a contract break (both S0b and S1a differ): VOID",
         v(arms(S0b={"toks": other}, S1a={"toks": other})) == "VOID"),
        ("with the null clean, one S1 W16 row differs: FUNCTION_FAIL (a contract finding)",
         (lambda x: x["verdict"] == "FUNCTION_FAIL" and "CONTRACT" in x["reasons"][0]
          and x["contract"]["S1c/S0a W16 32"]["rows_identical"] == 15)(r(arms(S1c={"toks16": row7})))),
        ("every S1 differs at W1: FUNCTION_FAIL", v(arms(**{t: {"toks": other, "toks16": toks} for t in s1})) == "FUNCTION_FAIL"),
        ("an S1 arm's timed reps differ: FUNCTION_FAIL", v(arms(S1b={"rep_ok": False})) == "FUNCTION_FAIL"),
        ("FUNCTION_FAIL outranks a passing g16", v(arms(S1a={"toks16": row7})) == "FUNCTION_FAIL"),
        # K25's engagement
        ("S1 resolved the tree: VOID", v(arms(S1a={"lut": {"env": "prmt", "lut": "tree", "why": None}})) == "VOID"),
        ("S0 unset (env None) is not the named tree: VOID", v(arms(S0c={"lut": {"env": None, "lut": "tree", "why": None}})) == "VOID"),
        ("K25 off (lut None): VOID", v(arms(S0a={"lut": {"env": "tree", "lut": None, "why": "E4B_NF4_GROUPED_SMALLM=0"}})) == "VOID"),
        ("the build count is exact: 959 is VOID", v(arms(S1b={"seen_build": {"prmt|le256": 959}})) == "VOID"),
        ("the build count is exact: 576 (the licence's probes missed) is VOID",
         v(arms(S1b={"seen_build": {"prmt|le256": 576}})) == "VOID"),
        ("the GEMV's build tally is exact: 479 is VOID", v(arms(S0a={"tally": {"bw_prmt32": 479, "dotpad": 0}})) == "VOID"),
        ("the GEMV's build tally is exact: 288 (three forwards) is VOID",
         v(arms(S1c={"tally": {"bw_prmt32": 288, "dotpad": 0}})) == "VOID"),
        ("a zero count is VOID", v(arms(S1c={"seen_build": {}})) == "VOID"),
        ("S1 ran a tree call at the build: VOID", v(arms(S1a={"seen_build": {"prmt|le256": 960, "tree|le256": 48}})) == "VOID"),
        ("S1 ran a tree call in the runs, its total unchanged: VOID",
         v(arms(S1d={"seen_runs": {"prmt|le256": 1008, "tree|le256": 48}})) == "VOID"),
        ("S0 ran prmt: VOID", v(arms(S0b={"seen_build": {"prmt|le256": 960}, "seen_runs": {"prmt|le256": 1056}})) == "VOID"),
        ("a record held calls before the build: VOID", v(arms(S1a={"seen_before": {"prmt|le256": 1}})) == "VOID"),
        ("the run counts must match across processes", v(arms(S1c={"seen_runs": {"prmt|le256": 960 + 97}})) == "VOID"),
        ("the int8 head OFF in an arm (not the default server): VOID", v(arms(S1a={"head": {"mode": "0", "modules": 0}})) == "VOID"),
        ("the int8 head on by an explicit 1 (not the default): VOID", v(arms(S0b={"head": _head_default(source="explicit")})) == "VOID"),
        ("the int8 head fell back at one row: VOID", v(arms(S1c={"head": _head_default(engagement={
            "modules": 1, "routed_modules_by_rows": {"1": 1}, "passed_modules_by_rows": {"1": 1, "2": 1}})})) == "VOID"),
        ("the int8 head routed nothing: VOID", v(arms(S0d={"head": _head_default(engagement={
            "modules": 1, "routed_modules_by_rows": {}, "passed_modules_by_rows": {"2": 1}})})) == "VOID"),
        ("the dense attention GEMV on: VOID", v(arms(S0d={"dense": {"mode": "1", "modules": 96}})) == "VOID"),
        # power, clocks
        ("a timed block's power.limit under 575 W: VOID", v(arms(S1a={"power": 450.0})) == "VOID"),
        ("a timed block's ENFORCED limit at 480 W: VOID", v(arms(S1b={"enforced": 480.0})) == "VOID"),
        ("an unreadable enforced limit: VOID", v(arms(S0b={"enforced": None})) == "VOID"),
        ("exactly 575 W passes", v(arms(**{t: {"power": 575.0, "enforced": 575.0} for t in TAGS})) == "FLIP"),
        ("an S1 clock 3 % above S0's is labelled, not ruled",
         (lambda x: x["verdict"] == "FLIP" and "clock-confounded" in (x["labels"].get("S1a/S0a") or ""))(
             r(arms(S1a={"mhz": 2800.0 * 1.03})))),
        ("a clock 3 % BELOW is labelled too", "S1b/S0b" in r(arms(S1b={"mhz": 2800.0 * 0.97}))["labels"]),
        ("2 % exactly is not labelled (2856 against 2800 MHz)", "S1c/S0c" not in r(arms(S1c={"mhz": 2856.0}))["labels"]),
        ("2857 against 2800 MHz is", "S1c/S0c" in r(arms(S1c={"mhz": 2857.0}))["labels"]),
        ("an empty sampler is UNREAD, never 0 and never a VOID",
         (lambda x: x["verdict"] == "FLIP" and x["labels"].get("S1d/S0d") == "clock unread"
          and x["speed"]["clocks"]["S1d"]["W16"]["sm_mhz_median"] is None)(r(arms(S1d={"mhz": None})))),
        # structure, carried from P153
        ("one arm at three reps is VOID", v(arms(S0a={"reps": 3})) == "VOID"),
        ("every process at three reps, not five, is VOID", v(arms(**{t: {"reps": 3} for t in TAGS + (WARMUP,)})) == "VOID"),
        ("a missing second-block arm is VOID", reduce({k: x for k, x in arms().items() if k != "S0d"}, E)["verdict"] == "VOID"),
        ("a missing warm-up process is VOID", reduce({k: x for k, x in arms().items() if k != WARMUP}, E)["verdict"] == "VOID"),
        ("a slow warm-up never reaches a pair or a self-pair", v(arms(S0w={"r16": 983.0 * 0.966, "r1": 264.0 * 0.97})) == "FLIP"),
        ("the warm-up's integrity counts: another commit is VOID", v(arms(S0w={"e4b": "b" * 40})) == "VOID"),
        ("the warm-up is reported", (lambda x: x["speed"]["warmup"]["over_S0a"]["W16"] == 0.96)(r(arms(S0w={"r16": 983.0 * 0.96})))),
        ("an arm's cap planned no bucket", v(arms(S1c={"cap": _cap_record(plans={})})) == "VOID"),
        ("an explicit 16 is not the default", v(arms(S1a={"cap": {**_cap_record(), "source": "explicit"}})) == "VOID"),
        ("a card that is not the 5090", v(arms(S1d={"sms": 188})) == "VOID"),
        ("the GEMV on dot-pad", v(arms(S0c={"tally": {"dotpad": 96}})) == "VOID"),
        ("a bucket not captured", v(arms(S1a={"graphs": {**{str(b): "graph" for b in BUCKETS}, "4": "eager: x"}})) == "VOID"),
        ("wrong commit", v(arms(S1c={"e4b": "b" * 40})) == "VOID"),
        ("the geomean is over four pairs", round(r(arms())["speed"]["geomean"]["W16"], 2) == 1.10),
        ("the build counts are the closed forms", K25_BUILD_CALLS == 48 * 4 * 5 == 960 and BW_BUILD_CALLS == 48 * 5 * 2 == 480),
    ]
    return cases


def self_test() -> int:
    try:
        cases = _cases()
    except Exception as e:  # noqa: BLE001 -- a rule that crashes on a registered case fails its self-test
        print(f"p158_reduce self-test FAILED [{type(e).__name__}: {e}]")
        return 1
    bad = [n for n, ok in cases if not ok]
    print(f"p158_reduce self-test {'OK' if not bad else 'FAILED ' + str(bad)} ({len(cases)} cases)")
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
    v = reduce(arms, a.e4b_sha)
    json.dump(v, open(a.out, "w"), indent=1, default=str)
    print(f"P158_VERDICT {v['verdict']} {json.dumps(v['reasons'], default=str)[:1500]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
