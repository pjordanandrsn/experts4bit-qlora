#!/usr/bin/env python3
"""p121_reduce.py -- lane P121's registered rule (bench/p121/PREREG-p121.md; e4b#846), from the run directory.

Receipts: the speed phase's ``arm_K0a.json``, ``arm_K1a.json``, ``arm_K1b.json``, ``arm_K0b.json`` (ABBA, each
``p121_box.py --mode speed``) and the quality phase's ``quality_off.json`` (K0: R, its floor and the mutant) and
``quality_on.json`` (K1: ON against R). K0 is ``E4B_NF4_GROUPED_SMALLM=0`` (the NF4 M-tile); K1 is ``auto`` (K25 at
<= 256 routed rows, the shipped default). The continuity row's ``continuity_off.json`` (Cm) and ``continuity_on.json``
(Ct) are read beside the verdict and never change it.

The verdict is the first of these that applies:
  VOID          a record is missing or not ok; another e4b / grouped-nf4-gemm commit or model revision; the arms read
                different prompts or ran different lengths; a slope is void; or engagement or scope is wrong:
                - speed: a bucket not captured; ``max_seqs`` / buckets not 16 / 1-16; a fusion census not zero; the
                  graph captures took the wrong kernel (K1: K25 and no M-tile call at <= 256 rows; K0: no K25 and the
                  M-tile at <= 256 rows); K1's W1 tokens differ from K0's (T == 1 runs the same kernels in both arms);
                - quality: the group is not 16; a text short of its windows (``rep``: one group); the phases scored
                  different windows; a pass with the wrong decode attention calls, grouping flag or bucket statistics
                  (every decode step an eager padded step of its bucket, no replays); a pass that took the wrong kernel
                  (K0 passes: no K25, M-tile 2 x layers per decode step; ON: K25 2 x layers per decode step and no
                  M-tile call at <= 256 rows); ``mutant_scale`` passes the bar on either text (the gate cannot fail);
                  ON bit-equal to R in every window of both texts (nothing was read).
  NOISY         a self-pair (K0b/K0a or K1b/K1a, decode tok/s, either workload) outside [0.96, 1.04].
  FUNCTION_FAIL K0b's tokens differ from K0a's or K1b's from K1a's on any row of either workload at either length, or an
                arm's timed reps do not all digest the same. (K1 differing from K0 at W16 is expected and reported.)
  QUALITY_FAIL  on either text ON fails P110's bar against the floor (mean d_ON <= B_floor + 0.01 nats and
                mean |d_ON| <= 2 x max(S_floor, 0.005)). The perplexity move (exp of the mean NLL over each text's
                positions) is reported on both texts and gates nothing (the maintainer's ruling: license on P110's bar
                and g16 only).
  SLOWER        g16 = min(K1a/K0a, K1b/K0b) decode tok/s on W16 < 1.00.
  LICENSED      otherwise: K25's ``auto`` default is read on Qwen3-30B-A3B at the served W16 step.

The floor is ``half`` and ``chunk`` under R's arithmetic, plus ``rep`` if R did not repeat bit for bit (P110's B_floor
and S_floor).

**Memory (reported, no bar; the maintainer's addition):** each arm's and phase's peak allocated and reserved GiB after
the build and at its end, and K1 - K0 (the mean over the two pairs) for the speed arms.

**Continuity (reported, no bar):** P96's arms through the same instrument at one window a pass (T == 1) on 12 wikitext
windows: Cm (``0`` + T == 1 device grouping, the M-tile) scores R, Ct (``1``: K25 at T == 1) scores ON. Read as
"not run", "void" (with reasons: engagement, windows, commits) or "read" (bias, spread, perplexity move, argmax). Floats are summed with ``math.fsum``, so the verdict file is byte-identical on any Python.

    python p121_reduce.py --dir RUN_DIR --out verdict.json [--e4b-sha SHA]
    python p121_reduce.py --self-test
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import sys

TAGS = ("K0a", "K1a", "K1b", "K0b")
BUCKETS = (1, 2, 4, 8, 16)
TEXTS = ("wikitext", "c4val1")
GNF4_SHA = "56f90e3555b089cdc8fb508e42ea0d2c069ce8e5"   # grouped-nf4-gemm 0.43.0 main at registration (K25 since #429)
REVS = {"Qwen/Qwen3-30B-A3B": "ad44e777bcd18fa416d9da3bd8f70d33ebb85d39",
        "ibm-granite/granite-3.1-3b-a800m-instruct": "a02780686e08a03fe0d2679a293b5c74a90efa89"}
LAYERS = {"Qwen/Qwen3-30B-A3B": 48, "ibm-granite/granite-3.1-3b-a800m-instruct": 32}
WINDOWS = {"Qwen/Qwen3-30B-A3B": 48, "ibm-granite/granite-3.1-3b-a800m-instruct": 16}
GROUP = 16
OFF_ARMS = ("R", "rep", "half", "chunk", "mutant_scale")
FLOORS = ("half", "chunk")
CENSUS_KEYS = ("fuse_qkv_n", "fuse_t1_glue_n", "fuse_t1_glue_r2_n", "fuse_router_epilogue_n")
SELF_LO, SELF_HI = 0.96, 1.04
GAIN_MIN_W16 = 1.00
TOL, SPREAD_X, SPREAD_MIN = 0.01, 2.0, 0.005
K8_BUDGET, K8_GATED = 0.05, ()        # the perplexity move: reported on both texts, gated on neither
CONT_WINDOWS, CONT_GROUP = 12, 1
# (name, statistic, lo, hi): written before the data; evaluated beside the verdict, never gating it
PREDICTIONS = (("Q1", "g16", 1.10, 1.70), ("Q2", "w1_pairs", 0.98, 1.02), ("Q3", "self_pairs_all", 0.98, 1.02),
               ("Q4a", "on_bias_wikitext", -0.004, 0.004), ("Q4b", "on_bias_c4val1", -0.006, 0.006),
               ("Q4c", "on_spread_wikitext", 0.0, 0.015), ("Q4d", "k8_delta_wikitext", -0.03, 0.03),
               ("Q5", "mutant_bias_min", 0.3, 1e9))


def _fsum(xs) -> float:
    return math.fsum(float(x) for x in xs)


def _zero(v) -> bool:
    return all(x == 0 for x in v) if isinstance(v, (list, tuple)) else v == 0


def _bucket(n: int) -> int:
    return min(b for b in BUCKETS if b >= n)


def _groups(n: int, group: int) -> list:
    return [min(group, n - g0) for g0 in range(0, n, group)]


def _rate(r, w):
    return r["workloads"][w]["decode_tok_s"]


def _tok(r, w, n):
    return r["workloads"][w]["tokens"][str(n)]


def _agree(a, b):
    first = []
    for x, y in zip(a, b):
        i = next((k for k, (p, q) in enumerate(zip(x, y)) if p != q), None)
        first.append(i if i is not None or len(x) == len(y) else min(len(x), len(y)))
    return {"rows_identical": sum(f is None for f in first), "rows": len(first), "first_divergence": first}


def _stats(per, arm, ref):
    d = [x["nll"] - ref[x["window"]] for x in per.get(arm, [])]
    n = len(d)
    if not n:
        return {"n": 0}
    mean = _fsum(d) / n
    sd = math.sqrt(_fsum((v - mean) ** 2 for v in d) / (n - 1)) if n > 1 else 0.0
    kls = [x["kl"] for x in per[arm] if "kl" in x]
    return {"n": n, "bias": mean, "spread": _fsum(abs(v) for v in d) / n, "se": sd / math.sqrt(n),
            "max_abs": max(abs(v) for v in d), "mean_kl": _fsum(kls) / len(kls) if kls else None,
            "argmax_agree": _fsum(x["argmax_agree"] for x in per[arm]) / n}


def _passes(s, fl):
    return s["bias"] <= fl["B_floor"] + TOL and s["spread"] <= SPREAD_X * max(fl["S_floor"], SPREAD_MIN)


def _ppl(per_r):
    return math.exp(_fsum(x["nll"] for x in per_r) / len(per_r))


def _common_faults(name, r, e4b_sha) -> list:
    out = []
    if r.get("e4b_sha") != e4b_sha:
        out.append(f"{name}: e4b {r.get('e4b_sha')} != {e4b_sha}")
    if r.get("gnf4_sha") != GNF4_SHA:
        out.append(f"{name}: grouped-nf4-gemm {r.get('gnf4_sha')} != {GNF4_SHA}")
    if REVS.get(r.get("model")) != r.get("revision"):
        out.append(f"{name}: model {r.get('model')}@{r.get('revision')} is not a registered revision")
    f = r.get("fusions") or {}
    if not all(_zero(f.get(k, 1)) for k in CENSUS_KEYS):
        out.append(f"{name}: fusion census {f}, the default server has none")
    return out


def speed_faults(arms: dict, e4b_sha: str) -> list:
    void = []
    for t, r in arms.items():
        void += _common_faults(t, r, e4b_sha)
        g = r.get("graph_status") or {}
        if [g.get(str(b)) for b in BUCKETS] != ["graph"] * len(BUCKETS):
            void.append(f"{t}: graph_status {g}")
        if r.get("max_seqs") != 16 or list(r.get("buckets") or []) != list(BUCKETS):
            void.append(f"{t}: max_seqs {r.get('max_seqs')}, buckets {r.get('buckets')}")
        rb = r.get("route_build") or {}
        if t.startswith("K1") and not (rb.get("k25", 0) > 0 and rb.get("mtile_small", 0) == 0):
            void.append(f"{t}: the captures took {rb}; K1 registers K25 and no M-tile call at <= 256 rows")
        if t.startswith("K0") and not (rb.get("k25", 0) == 0 and rb.get("mtile_small", 0) > 0):
            void.append(f"{t}: the captures took {rb}; K0 registers the M-tile and no K25")
    if void:
        return void
    if len({r.get("prompts_sha256") for r in arms.values()}) != 1:
        void.append("the arms read different prompts")
    if len({(r.get("short"), r.get("long"), r.get("reps")) for r in arms.values()}) != 1:
        void.append("the arms ran different lengths")
    if len({(r.get("model"), bool(r.get("prove"))) for r in arms.values()}) != 1:
        void.append("the arms ran different subjects")
    if any(_rate(arms[t], w) is None for t in TAGS for w in ("W16", "W1")):
        void.append("a decode slope is void")
    if not void:
        a = arms["K0a"]
        for t in ("K1a", "K1b"):
            for n in (a["short"], a["long"]):
                if _tok(arms[t], "W1", n) != _tok(a, "W1", n):
                    void.append(f"{t}'s W1 tokens at {n} differ from K0a's: T == 1 should run the same kernels")
    return void


def quality_faults(off: dict, on: dict, e4b_sha: str) -> list:
    void = _common_faults("quality_off", off, e4b_sha) + _common_faults("quality_on", on, e4b_sha)
    if void:
        return void
    if (off.get("model"), off.get("prove")) != (on.get("model"), on.get("prove")):
        return [f"the phases ran different subjects: {off.get('model')} / {on.get('model')}"]
    L, want_n = LAYERS[off["model"]], WINDOWS[off["model"]]
    C, group = off.get("cont"), off.get("group")
    if group != GROUP:
        void.append(f"group {group}, registered {GROUP} (16 rows a decode step)")
    for k in ("cont", "group", "prompt", "chunk", "floor_chunk", "layers", "windows_sha256"):
        if off.get(k) != on.get(k):
            void.append(f"the phases differ in {k}: {off.get(k)} / {on.get(k)}")
    if off.get("layers") != L:
        void.append(f"{off.get('layers')} layers, the model registers {L}")
    if void:
        return void
    for name, rec, arms in (("quality_off", off, OFF_ARMS), ("quality_on", on, ("ON",))):
        for t in TEXTS:
            per = (rec.get("per_window") or {}).get(t, {})
            for arm in arms:
                n = len(per.get(arm, []))
                want = group if arm == "rep" else want_n
                if n != want:
                    void.append(f"{name} {t}: arm {arm} has {n} windows, expected {want}")
            sizes = _groups(want_n, group)
            for arm in arms:
                passes = (rec.get("engagement") or {}).get(t, {}).get(arm, [])
                arm_sizes = sizes[:1] if arm == "rep" else sizes
                if len(passes) != len(arm_sizes):
                    void.append(f"{name} {t} {arm}: {len(passes)} passes, expected {len(arm_sizes)}")
                    continue
                for k, (e, n) in enumerate(zip(passes, arm_sizes)):
                    void += [f"{name} {t} {arm} pass {k}: {m}" for m in _pass_faults(e, arm, n, C, L)]
    return void


def _pass_faults(e, arm, n, C, L) -> list:
    out = []
    halves = arm == "half"
    steps = (C - 1) * (2 if halves else 1)
    if e.get("decode_calls") != steps * L:
        out.append(f"{e.get('decode_calls')} decode attention calls, expected {steps * L}")
    if not (e.get("grouping_flags_in_pass") or {}).get("device_grouping"):
        out.append("device grouping off")
    st = e.get("graph_status") or {}
    if [st.get(str(b)) for b in BUCKETS] != ["eager: capture=False"] * len(BUCKETS):
        out.append(f"graph_status {st}")
    parts = [n // 2, n - n // 2] if halves else [n]
    want = {}
    for p in parts:
        b = _bucket(p)
        w = want.setdefault(str(b), {"eager_steps": 0, "pad_rows": 0})
        w["eager_steps"] += C - 1
        w["pad_rows"] += (b - p) * (C - 1)
    gs = e.get("graph_stats") or {}
    for b, w in want.items():
        got = gs.get(b, {})
        if (got.get("eager_steps"), got.get("pad_rows")) != (w["eager_steps"], w["pad_rows"]):
            out.append(f"bucket {b} stats {got}, expected {w}")
    if any(v.get("replays", 0) for v in gs.values()):
        out.append(f"a replay in a capture=False pass: {gs}")
    k = e.get("kernels") or {}
    calls = 2 * L * steps                     # gate_up and down, every MoE layer, every decode step
    if arm == "ON":
        if (k.get("k25"), k.get("mtile_small")) != (calls, 0):
            out.append(f"routes {k}, expected K25 {calls} and no M-tile call at <= 256 rows")
    elif (k.get("k25"), k.get("mtile_small")) != (0, calls):
        out.append(f"routes {k}, expected the M-tile {calls} at <= 256 rows and no K25")
    return out


def quality_read(off: dict, on: dict) -> dict:
    out = {}
    for t in TEXTS:
        ref = {x["window"]: x["nll"] for x in off["per_window"][t]["R"]}
        st = {a: _stats(off["per_window"][t], a, ref) for a in OFF_ARMS if a != "R"}
        st["ON"] = _stats(on["per_window"][t], "ON", ref)
        draws = list(FLOORS) + ([] if (off.get("rep_identical") or {}).get(t) else ["rep"])
        fl = {"draws": draws, "B_floor": max(abs(st[f]["bias"]) for f in draws),
              "S_floor": max(st[f]["spread"] for f in draws)}
        ppl_r, ppl_on = _ppl(off["per_window"][t]["R"]), _ppl(on["per_window"][t]["ON"])
        ident = all(x.get("kl", 1) == 0 and x["nll"] == ref[x["window"]] for x in on["per_window"][t]["ON"])
        out[t] = {"stats": st, "floor": fl, "bias_bar": fl["B_floor"] + TOL,
                  "spread_bar": SPREAD_X * max(fl["S_floor"], SPREAD_MIN), "mutant_passes": _passes(st["mutant_scale"], fl),
                  "on_passes": _passes(st["ON"], fl), "on_bit_equal": ident,
                  "k8": {"ppl_R": ppl_r, "ppl_ON": ppl_on, "delta": ppl_on - ppl_r, "gated": t in K8_GATED,
                         "passes": abs(ppl_on - ppl_r) <= K8_BUDGET}}
    return out


def continuity_read(coff, con, e4b_sha: str) -> dict:
    """P96's arms at T == 1 through Phase B's instrument: reported, never part of the verdict."""
    if not coff or not con or coff.get("status") != "ok" or con.get("status") != "ok":
        return {"status": "not run"}
    why = _common_faults("continuity_off", coff, e4b_sha) + _common_faults("continuity_on", con, e4b_sha)
    if (coff.get("arm"), con.get("arm")) != ("Cm", "Ct"):
        why.append(f"arms {coff.get('arm')} / {con.get('arm')}, registered Cm / Ct")
    for k in ("cont", "group", "prompt", "chunk", "layers", "windows_sha256", "model"):
        if coff.get(k) != con.get(k):
            why.append(f"the continuity phases differ in {k}")
    if coff.get("group") != CONT_GROUP or (coff.get("windows") or {}) != {"wikitext": CONT_WINDOWS}:
        why.append(f"group {coff.get('group')}, windows {coff.get('windows')}; registered 1 and 12 wikitext")
    if not why:
        L, C = LAYERS[coff["model"]], coff["cont"]
        for name, rec, arm in (("continuity_off", coff, "R"), ("continuity_on", con, "ON")):
            passes = (rec.get("engagement") or {}).get("wikitext", {}).get(arm, [])
            if len(passes) != CONT_WINDOWS:
                why.append(f"{name}: {len(passes)} passes, expected {CONT_WINDOWS}")
                continue
            for k, e in enumerate(passes):
                why += [f"{name} pass {k}: {m}" for m in _pass_faults(e, arm, 1, C, L)]
    if why:
        return {"status": "void", "reasons": why}
    ref = {x["window"]: x["nll"] for x in coff["per_window"]["wikitext"]["R"]}
    st = _stats(con["per_window"]["wikitext"], "ON", ref)
    ppl_r, ppl_on = _ppl(coff["per_window"]["wikitext"]["R"]), _ppl(con["per_window"]["wikitext"]["ON"])
    return {"status": "read", "stats": st, "ppl_R": ppl_r, "ppl_ON": ppl_on, "ppl_delta": ppl_on - ppl_r}


def _gib(b):
    return round(b / 2**30, 3) if isinstance(b, (int, float)) else None


def memory(arms: dict, recs: dict) -> dict:
    """Peak allocated / reserved GiB per arm and phase, and K1 - K0 over the speed pairs: reported, never gated."""
    out = {}
    for name, r in list((t, arms.get(t)) for t in TAGS) + list(recs.items()):
        if not r:
            continue
        p, pb = r.get("peak") or {}, r.get("peak_build") or {}
        out[name] = {"max_allocated_gib": _gib(p.get("max_allocated")), "max_reserved_gib": _gib(p.get("max_reserved")),
                     "build_max_allocated_gib": _gib(pb.get("max_allocated")),
                     "build_max_reserved_gib": _gib(pb.get("max_reserved"))}
    diffs = {}
    for k in ("max_allocated_gib", "max_reserved_gib"):
        v = [out.get(a, {}).get(k) for a in ("K1a", "K0a", "K1b", "K0b")]
        if all(x is not None for x in v):
            diffs[k] = round(math.fsum([v[0] - v[1], v[2] - v[3]]) / 2, 3)
    out["k1_minus_k0"] = diffs
    return out


def predictions(out: dict) -> dict:
    q = out.get("quality") or {}
    vals = {"g16": out.get("g16"), "w1_pairs": out.get("pair_ratios", {}).get("W1"),
            "self_pairs_all": list((out.get("self_pairs") or {}).values()) or None,
            "on_bias_wikitext": q.get("wikitext", {}).get("stats", {}).get("ON", {}).get("bias"),
            "on_bias_c4val1": q.get("c4val1", {}).get("stats", {}).get("ON", {}).get("bias"),
            "on_spread_wikitext": q.get("wikitext", {}).get("stats", {}).get("ON", {}).get("spread"),
            "k8_delta_wikitext": q.get("wikitext", {}).get("k8", {}).get("delta"),
            "mutant_bias_min": (min(q[t]["stats"]["mutant_scale"]["bias"] for t in TEXTS) if q else None)}
    res = {}
    for name, stat, lo, hi in PREDICTIONS:
        v = vals.get(stat)
        vs = v if isinstance(v, list) else [v]
        held = None if any(x is None for x in vs) else all(lo <= x <= hi for x in vs)
        res[name] = {"statistic": stat, "value": v, "band": [lo, hi],
                     "result": "UNREAD" if held is None else ("HELD" if held else "MISSED")}
    return res


def reduce(arms: dict, off: dict | None, on: dict | None, e4b_sha: str, coff=None, con=None) -> dict:
    out = {"lane": "P121", "verdict": None, "reasons": [],
           "bars": {"self_pair": [SELF_LO, SELF_HI], "gain_min_w16": GAIN_MIN_W16, "tol": TOL, "spread_x": SPREAD_X,
                    "spread_min": SPREAD_MIN, "k8_budget": K8_BUDGET, "k8_gated": list(K8_GATED)}}
    out["continuity"] = continuity_read(coff, con, e4b_sha)
    out["memory"] = memory(arms, {"quality_off": off, "quality_on": on, "continuity_off": coff, "continuity_on": con})
    missing = [t for t in TAGS if t not in arms or arms[t].get("status") != "ok"]
    missing += [n for n, r in (("quality_off", off), ("quality_on", on)) if not r or r.get("status") != "ok"]
    if missing:
        out.update(verdict="VOID", reasons=[f"record(s) missing or not ok: {missing}"])
        return out
    void = speed_faults(arms, e4b_sha) + quality_faults(off, on, e4b_sha)
    if void:
        out.update(verdict="VOID", reasons=void)
        return out
    q = quality_read(off, on)
    out["quality"] = q
    void = [f"{t}: mutant_scale passes the bar (the gate cannot fail)" for t in TEXTS if q[t]["mutant_passes"]]
    if all(q[t]["on_bit_equal"] for t in TEXTS):
        void.append("ON is bit-equal to R in every window of both texts: nothing was read")
    if void:
        out.update(verdict="VOID", reasons=void)
        return out
    K0a, K1a, K1b, K0b = (arms[t] for t in TAGS)
    short, long_ = K0a["short"], K0a["long"]
    rates = {t: {w: _rate(arms[t], w) for w in ("W16", "W1")} for t in TAGS}
    out["decode_tok_s"] = rates
    out["ms_per_step"] = {t: {w: arms[t]["workloads"][w].get("decode_ms_per_step") for w in ("W16", "W1")} for t in TAGS}
    selfp = {f"{a}/{b} {w}": rates[a][w] / rates[b][w] for a, b in (("K0b", "K0a"), ("K1b", "K1a")) for w in ("W16", "W1")}
    out["self_pairs"] = {k: round(v, 4) for k, v in selfp.items()}
    pairs = {w: [rates["K1a"][w] / rates["K0a"][w], rates["K1b"][w] / rates["K0b"][w]] for w in ("W16", "W1")}
    out["pair_ratios"] = {w: [round(x, 4) for x in v] for w, v in pairs.items()}
    out["g16"], out["g1"] = round(min(pairs["W16"]), 4), round(min(pairs["W1"]), 4)
    out["geomean"] = {w: round(math.sqrt(v[0] * v[1]), 4) for w, v in pairs.items()}
    out["k1_vs_k0_tokens"] = {f"{w}@{n}": _agree(_tok(K1a, w, n), _tok(K0a, w, n)) for w in ("W16", "W1") for n in (short, long_)}
    out["routes"] = {t: {"build": arms[t].get("route_build"), "W16": arms[t]["workloads"]["W16"].get("routes"),
                         "W1": arms[t]["workloads"]["W1"].get("routes")} for t in TAGS}
    fn = []
    for a, b in (("K0b", "K0a"), ("K1b", "K1a")):
        for w in ("W16", "W1"):
            for n in (short, long_):
                if _tok(arms[a], w, n) != _tok(arms[b], w, n):
                    rows = [i for i, (x, y) in enumerate(zip(_tok(arms[a], w, n), _tok(arms[b], w, n))) if x != y]
                    fn.append(f"{a} != {b} on {w} at {n} tokens, rows {rows}")
    for t in TAGS:
        for w in ("W16", "W1"):
            for n, ds in arms[t]["workloads"][w]["rep_digests"].items():
                if len(set(ds)) != 1:
                    fn.append(f"{t} {w} at {n} tokens: timed reps differ")
    noisy = [f"{k} = {v:.4f}" for k, v in selfp.items() if not SELF_LO <= v <= SELF_HI]
    qfail = []
    for t in TEXTS:
        s = q[t]["stats"]["ON"]
        if not q[t]["on_passes"]:
            qfail.append(f"{t}: ON bias {s['bias']:+.5f} (bar {q[t]['bias_bar']:.5f}), spread {s['spread']:.5f} "
                         f"(bar {q[t]['spread_bar']:.5f})")
        if q[t]["k8"]["gated"] and not q[t]["k8"]["passes"]:
            qfail.append(f"{t}: K8 perplexity {q[t]['k8']['ppl_R']:.5f} -> {q[t]['k8']['ppl_ON']:.5f} "
                         f"(|delta| > {K8_BUDGET})")
    out["predictions"] = predictions(out)
    if noisy:
        out.update(verdict="NOISY", reasons=noisy)
    elif fn:
        out.update(verdict="FUNCTION_FAIL", reasons=fn)
    elif qfail:
        out.update(verdict="QUALITY_FAIL", reasons=qfail)
    elif out["g16"] < GAIN_MIN_W16:
        out.update(verdict="SLOWER", reasons=[f"g16 = {out['g16']} (bar {GAIN_MIN_W16})"])
    else:
        out.update(verdict="LICENSED", reasons=[
            f"g16 = {out['g16']} (geomean {out['geomean']['W16']}); W1 {out['pair_ratios']['W1']}; "
            + "; ".join(f"{t} ON bias {q[t]['stats']['ON']['bias']:+.5f} (bar {q[t]['bias_bar']:.5f}), K8 "
                        f"{q[t]['k8']['delta']:+.4f}" for t in TEXTS)])
    return out


def load(run: str):
    arms = {}
    for t in TAGS:
        p = os.path.join(run, f"arm_{t}.json")
        if os.path.isfile(p):
            arms[t] = json.load(open(p))
    qs = []
    for n in ("quality_off.json", "quality_on.json", "continuity_off.json", "continuity_on.json"):
        p = os.path.join(run, n)
        qs.append(json.load(open(p)) if os.path.isfile(p) else None)
    return arms, qs[0], qs[1], qs[2], qs[3]


# ------------------------------------------------------------------ self-test --

E = "a" * 40
QWEN = "Qwen/Qwen3-30B-A3B"
GRAN = "ibm-granite/granite-3.1-3b-a800m-instruct"
ZERO_FUSIONS = {"fuse_qkv_n": 0, "fuse_t1_glue_n": 0, "fuse_t1_glue_r2_n": [0, 0], "fuse_router_epilogue_n": 0}


def _fake_arm(tag, rate16, rate1, toks16, toks1, *, model=QWEN, prove=False, e4b=E, graphs=None, route_build=None,
              digests=None, short=4, long_=8):
    k1 = tag.startswith("K1")
    rb = route_build or ({"k25": 1152, "mtile_small": 0, "mtile_large": 96} if k1
                         else {"k25": 0, "mtile_small": 1152, "mtile_large": 96})

    def wl(rate, toks):
        return {"decode_tok_s": rate, "decode_ms_per_step": round(1000 / rate, 3),
                "tokens": {str(short): [t[:short] for t in toks], str(long_): toks},
                "rep_digests": digests or {str(short): ["x"] * 3, str(long_): ["y"] * 3},
                "routes": {"k25": 0, "mtile_small": 0, "mtile_large": 0}}
    gib = 2**30
    peak = {"max_allocated": 22 * gib + (gib // 4 if k1 else 0), "max_reserved": 24 * gib, "allocated": 20 * gib, "reserved": 23 * gib}
    return {"tag": tag, "status": "ok", "e4b_sha": e4b, "gnf4_sha": GNF4_SHA, "model": model, "revision": REVS[model],
            "peak": peak, "peak_build": dict(peak),
            "prove": prove, "graph_status": graphs or {str(b): "graph" for b in BUCKETS}, "fusions": dict(ZERO_FUSIONS),
            "max_seqs": 16, "buckets": list(BUCKETS), "route_build": rb, "prompts_sha256": "p", "short": short,
            "long": long_, "reps": 3, "workloads": {"W16": wl(rate16, toks16), "W1": wl(rate1, toks1)}}


def _fake_pass(arm, n, C, L):
    halves = arm == "half"
    parts = [n // 2, n - n // 2] if halves else [n]
    gs = {str(b): {"replays": 0, "eager_steps": 0, "rows": 0, "pad_rows": 0} for b in BUCKETS}
    for p in parts:
        b = _bucket(p)
        gs[str(b)]["eager_steps"] += C - 1
        gs[str(b)]["pad_rows"] += (b - p) * (C - 1)
    steps = (C - 1) * (2 if halves else 1)
    calls = 2 * L * steps
    k = {"k25": calls, "mtile_small": 0, "mtile_large": 2 * L * n} if arm == "ON" else \
        {"k25": 0, "mtile_small": calls, "mtile_large": 2 * L * n}
    return {"decode_calls": steps * L, "grouping_flags_in_pass": {"device_grouping": True},
            "graph_status": {str(b): "eager: capture=False" for b in BUCKETS}, "graph_stats": gs, "kernels": k}


def _fake_quality(phase, *, model=QWEN, prove=False, e4b=E, bias=None, spread_noise=0.001, mutant=0.4,
                  rep_identical=True, group=GROUP, bit_equal=False, texts=TEXTS, n=None, arms=None):
    L, n = LAYERS[model], (n or WINDOWS[model])
    C = 128 if model == QWEN else 32
    bias = bias or {}
    arms = arms or (OFF_ARMS if phase == "off" else ("ON",))
    per, eng = {}, {}
    for t in texts:
        base = {w: 2.0 + 0.01 * w + (0.5 if t == "c4val1" else 0.0) for w in range(n)}
        per[t], eng[t] = {}, {}
        for arm in arms:
            off_by = {"R": 0.0, "rep": 0.0, "half": 0.0005, "chunk": -0.0004, "mutant_scale": mutant}.get(arm, bias.get(t, 0.0))
            noise = 0.0 if (arm == "R" or (arm == "ON" and bit_equal)) else spread_noise
            ws = range(group) if arm == "rep" else range(n)
            per[t][arm] = [{"window": w, "nll": base[w] + off_by + (noise if w % 2 else -noise), "argmax_agree": 1.0,
                            "kl": 0.0 if (arm == "ON" and bit_equal) else 0.001} for w in ws]
            sizes = _groups(n, group)[:1] if arm == "rep" else _groups(n, group)
            eng[t][arm] = [_fake_pass(arm, s, C, L) for s in sizes]
    return {"model": model, "revision": REVS[model], "e4b_sha": e4b, "gnf4_sha": GNF4_SHA, "prove": prove,
            "fusions": dict(ZERO_FUSIONS), "phase": phase, "arms": list(arms), "group": group, "prompt": 512, "cont": C,
            "chunk": 512, "floor_chunk": 256, "layers": L, "windows_sha256": {t: t for t in texts},
            "windows": {t: n for t in texts}, "rep_identical": {t: rep_identical for t in texts}, "per_window": per,
            "engagement": eng, "status": "ok", "arm": ("K0" if phase == "off" else "K1")}


def _fake_continuity(phase, bias=0.0, **kw):
    r = _fake_quality(phase, texts=("wikitext",), n=CONT_WINDOWS, group=CONT_GROUP, bias={"wikitext": bias},
                      arms=(("R",) if phase == "off" else ("ON",)), **kw)
    r["arm"] = "Cm" if phase == "off" else "Ct"
    return r


def self_test() -> int:
    base16 = [[r * 100 + i for i in range(8)] for r in range(16)]
    k25_16 = [list(r) for r in base16]
    k25_16[3][5] += 1                                                 # K1 differs from K0 at W16: expected, reported
    base1 = [[7, 8, 9, 10, 11, 12, 13, 14]]

    def arms(**over):
        a = {"K0a": _fake_arm("K0a", 300.0, 140.0, base16, base1), "K1a": _fake_arm("K1a", 390.0, 140.2, k25_16, base1),
             "K1b": _fake_arm("K1b", 391.0, 140.1, k25_16, base1), "K0b": _fake_arm("K0b", 301.0, 140.3, base16, base1)}
        a.update(over)
        return a

    def run(a=None, off=None, on=None):
        return reduce(a or arms(), off if off is not None else _fake_quality("off"),
                      on if on is not None else _fake_quality("on"), E)

    cases = []
    r = run()
    cases.append(("licensed", r["verdict"] == "LICENSED"))
    cases.append(("report", r["g16"] == round(min(390 / 300, 391 / 301), 4)
                  and r["k1_vs_k0_tokens"]["W16@8"]["rows_identical"] == 15 and r["k1_vs_k0_tokens"]["W1@8"]["rows_identical"] == 1))
    cases.append(("predictions held", all(v["result"] == "HELD" for v in r["predictions"].values())))
    cases.append(("missing arm", run(a={k: v for k, v in arms().items() if k != "K0b"})["verdict"] == "VOID"))
    cases.append(("missing quality", reduce(arms(), _fake_quality("off"), None, E)["verdict"] == "VOID"))
    cases.append(("wrong e4b", run(a=arms(K1a=_fake_arm("K1a", 390.0, 140.2, k25_16, base1, e4b="b" * 40)))["verdict"] == "VOID"))
    cases.append(("a bucket eager", run(a=arms(K1b=_fake_arm("K1b", 391.0, 140.1, k25_16, base1,
                                                              graphs={**{str(b): "graph" for b in BUCKETS}, "16": "eager: x"})))["verdict"] == "VOID"))
    cases.append(("K1 captured the M-tile", run(a=arms(K1a=_fake_arm("K1a", 390.0, 140.2, k25_16, base1,
                                                                      route_build={"k25": 0, "mtile_small": 1152})))["verdict"] == "VOID"))
    cases.append(("K0 captured K25", run(a=arms(K0b=_fake_arm("K0b", 301.0, 140.3, base16, base1,
                                                               route_build={"k25": 8, "mtile_small": 1144})))["verdict"] == "VOID"))
    a = arms()
    a["K0a"]["max_seqs"] = 64
    cases.append(("not 16 slots", run(a=a)["verdict"] == "VOID"))
    a = arms()
    a["K1a"]["fusions"]["fuse_t1_glue_n"] = 193
    cases.append(("a fusion engaged", run(a=a)["verdict"] == "VOID"))
    cases.append(("W1 differs between the arms", run(a=arms(K1b=_fake_arm("K1b", 391.0, 140.1, k25_16,
                                                                          [[7, 8, 9, 10, 11, 12, 13, 99]])))["verdict"] == "VOID"))
    on = _fake_quality("on")
    on["engagement"]["wikitext"]["ON"][1]["kernels"]["k25"] -= 2
    cases.append(("ON short of K25 calls", run(on=on)["verdict"] == "VOID"))
    on = _fake_quality("on")
    on["engagement"]["c4val1"]["ON"][0]["kernels"]["mtile_small"] = 96
    cases.append(("ON took the M-tile", run(on=on)["verdict"] == "VOID"))
    off = _fake_quality("off")
    off["engagement"]["c4val1"]["half"][0]["kernels"]["k25"] = 4
    cases.append(("OFF took K25", run(off=off)["verdict"] == "VOID"))
    off = _fake_quality("off")
    off["engagement"]["wikitext"]["R"][0]["graph_stats"]["16"]["eager_steps"] -= 1
    cases.append(("R bucket stats", run(off=off)["verdict"] == "VOID"))
    cases.append(("group 12", run(off=_fake_quality("off", group=12), on=_fake_quality("on", group=12))["verdict"] == "VOID"))
    off = _fake_quality("off")
    off["per_window"]["c4val1"]["R"].pop()
    cases.append(("a text short", run(off=off)["verdict"] == "VOID"))
    on = _fake_quality("on")
    on["windows_sha256"] = {"wikitext": "x", "c4val1": "c4val1"}
    cases.append(("phases scored different windows", run(on=on)["verdict"] == "VOID"))
    cases.append(("mutant passes", run(off=_fake_quality("off", mutant=0.004))["verdict"] == "VOID"))
    r = run(on=_fake_quality("on", bit_equal=True))
    cases.append(("ON bit-equal to R", r["verdict"] == "VOID" and "nothing was read" in " ".join(r["reasons"])))
    cases.append(("noisy", run(a=arms(K0b=_fake_arm("K0b", 280.0, 140.3, base16, base1)))["verdict"] == "NOISY"))
    moved = [list(x) for x in k25_16]
    moved[9][7] += 1
    cases.append(("K1 not deterministic", run(a=arms(K1b=_fake_arm("K1b", 391.0, 140.1, moved, base1)))["verdict"] == "FUNCTION_FAIL"))
    cases.append(("reps differ", run(a=arms(K0a=_fake_arm("K0a", 300.0, 140.0, base16, base1,
                                                           digests={"4": ["x"] * 3, "8": ["y", "z", "y"]})))["verdict"] == "FUNCTION_FAIL"))
    cases.append(("bias on c4val1", run(on=_fake_quality("on", bias={"c4val1": 0.02}))["verdict"] == "QUALITY_FAIL"))
    r = run(on=_fake_quality("on", bias={"wikitext": 0.006}))
    cases.append(("wikitext ppl move reported, not gated", r["verdict"] == "LICENSED"
                  and abs(r["quality"]["wikitext"]["k8"]["delta"]) > K8_BUDGET))
    r = run(on=_fake_quality("on", bias={"c4val1": 0.0055}))
    cases.append(("c4val1 ppl move reported, not gated", r["verdict"] == "LICENSED" and abs(r["quality"]["c4val1"]["k8"]["delta"]) > K8_BUDGET))
    r = reduce(arms(), _fake_quality("off"), _fake_quality("on"), E, _fake_continuity("off"), _fake_continuity("on", bias=0.002))
    c = r["continuity"]
    cases.append(("continuity read", r["verdict"] == "LICENSED" and c["status"] == "read" and abs(c["stats"]["bias"] - 0.002) < 1e-9))
    cases.append(("continuity not run", run()["continuity"] == {"status": "not run"}))
    m = run()["memory"]
    cases.append(("memory reported", m["K1a"]["max_allocated_gib"] == 22.25 and m["k1_minus_k0"] == {"max_allocated_gib": 0.25, "max_reserved_gib": 0.0}
                  and m["quality_off"]["max_allocated_gib"] is None))
    con = _fake_continuity("on")
    con["engagement"]["wikitext"]["ON"][3]["kernels"]["k25"] = 0
    r = reduce(arms(), _fake_quality("off"), _fake_quality("on"), E, _fake_continuity("off"), con)
    cases.append(("continuity void leaves the verdict", r["verdict"] == "LICENSED" and r["continuity"]["status"] == "void"))
    r = reduce(arms(), _fake_quality("off"), _fake_quality("on", bias={"c4val1": 0.02}), E, _fake_continuity("off"),
               _fake_continuity("on", bias=0.05))
    cases.append(("continuity never rescues or sinks", r["verdict"] == "QUALITY_FAIL" and r["continuity"]["status"] == "read"))
    r = run(a=arms(K1a=_fake_arm("K1a", 295.0, 140.2, k25_16, base1), K1b=_fake_arm("K1b", 296.0, 140.1, k25_16, base1)))
    cases.append(("slower at 16", r["verdict"] == "SLOWER" and r["predictions"]["Q1"]["result"] == "MISSED"))
    cases.append(("rep joins the floor", run(off=_fake_quality("off", rep_identical=False))["quality"]["wikitext"]["floor"]["draws"]
                  == ["half", "chunk", "rep"]))
    g = {t: _fake_arm(t, 300.0 if t.startswith("K0") else 350.0, 250.0, base16 if t.startswith("K0") else k25_16, base1,
                      model=GRAN, prove=True, route_build=({"k25": 768, "mtile_small": 0} if t.startswith("K1")
                                                           else {"k25": 0, "mtile_small": 768})) for t in TAGS}
    cases.append(("the Granite proof", reduce(g, _fake_quality("off", model=GRAN, prove=True),
                                              _fake_quality("on", model=GRAN, prove=True), E)["verdict"] == "LICENSED"))
    deep = copy.deepcopy(_fake_quality("on"))
    deep["status"] = "failed"
    cases.append(("a quality record not ok", run(on=deep)["verdict"] == "VOID"))
    bad = [n for n, ok in cases if not ok]
    print(f"p121_reduce self-test {'OK' if not bad else 'FAILED ' + str(bad)} ({len(cases)} cases)")
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
    arms, off, on, coff, con = load(a.dir)
    v = reduce(arms, off, on, a.e4b_sha, coff, con)
    s = json.dumps(v, indent=1, sort_keys=True, default=str)
    if a.out:
        with open(a.out, "w", newline="\n") as f:
            f.write(s + "\n")
    print(f"P121_VERDICT {v['verdict']} {json.dumps(v.get('reasons') or [])[:2000]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
