#!/usr/bin/env python3
"""p234_reduce.py -- lane P234's registered rule (bench/p234/PREREG-p234.md), lever B3's re-registration, and the
mechanical classification of every fetched box. P231's reducer (bench/p231/p231_reduce.py) by named edits: FUNCTION_FAIL
reads B1 against B0 on W1 only (bucket 1, where the switch takes no call); W16 keeps P231's 0.99 as a speed REGRESSION
bar, and its B1-vs-B0 tokens are reported, never ruled (its admission ramp runs the switched buckets, P231's reading).

Records: ``fit.json``; the speed arms ``arm_B0a``, ``arm_B1a``, ``arm_B1b``, ``arm_B0b`` (ABBA, each a fresh process);
``quality_off.json`` (B0: R, rep, half, chunk, mutant_scale) and ``quality_on.json`` (B1: ON), both at group 4. B0 leaves
``E4B_NF4_SMALLM_BW`` unset (the device tile table and K25 at 2..8 tokens); B1 sets it to ``1`` (singleton groups, K33).
Every rate is the decode-only slope. The verdict is the first of these that applies:

  VOID          a record is missing or not ok; another e4b / grouped-nf4-gemm commit, model revision, server size,
                prompts or lengths; a void slope; the fit failed; the mutant passes the quality bar; or engagement is
                wrong, each read fail-closed (a missing or empty record is a fault, never a pass):
                - every arm captured buckets 1, 2, 4, 8 and 16 as graphs, and records each bucket's route tally and
                  dispatch counts;
                - bucket 1 (both arms): ``nf4_singleton`` routes and K33 launches, no ``nf4_k25``;
                - bucket 16 (both arms): ``nf4_k25`` routes, no ``nf4_singleton`` and no K33 launch;
                - buckets 2, 4, 8: B1 ``nf4_singleton`` with K33 launches and no ``nf4_k25``; B0 ``nf4_k25`` with no
                  ``nf4_singleton`` and no K33 launch;
                - quality: group 4, the registered windows and passes per arm, P115's bucket statistics per pass, B0's
                  measurement with no K33 launch, B1's with K33 launches and ``nf4_singleton`` routes.
  FUNCTION_FAIL a self-pair (B0b/B0a, B1b/B1a) emits a different token on any row, workload or length; B1 differs
                from B0 at W1 (bucket 1, where the switch takes no call); or an arm's timed reps digest differently.
                B1 against B0 at W2, W4, W8 and W16 is reported (``token_diffs``), never a rung.
  NOISY         a self-pair's decode tok/s falls outside [0.985, 1.015] at W1, W2 or W4, or outside [0.96, 1.04] at
                W16 (W8 is reported only).
  QUALITY_FAIL  on either text ON misses P110's bar against the floor (bias_ON <= B_floor + 0.01; spread_ON <=
                2 x max(S_floor, 0.005)), or wikitext's K8 perplexity moves by more than 0.05.
  SLOWER        g2 or g4 = min(B1a/B0a, B1b/B0b) at W2 / W4 is below 1.03; g1 (the inert check) is below 0.99; or
                g16 is below 0.99 (the W16 regression bar).
  LICENSED      otherwise.

LICENSED, SLOWER, QUALITY_FAIL and FUNCTION_FAIL are answers (the runner exits 0); VOID and NOISY are not (27).

``--classify --work W`` reads a fetched box's work directory and fails closed (exit 2) unless it holds exactly one
``P234_EXIT_CODE.<nonce>`` and ``RUN_ID``. Before ``STARTED``: 13, 15, 16, 18 HOST_REFUSED; 41 FIT_REFUSED (no start, a
retry id; two consecutive are one strike); 9, 10, 28, 78, 130 or any other code PRE_INSTALL_HALT. After it: 0 an ANSWER;
25 (the premise, rows or served) and 31 (the server does not fit) HALT the lane until an amendment; any other code a
started box without an answer (a strike). NOT_RUN and refusals at create come from the launcher's receipt.

    python p234_reduce.py --dir W --out verdict.json --e4b-sha SHA
    python p234_reduce.py --classify --work W
    python p234_reduce.py --self-test
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import p115_reduce as pb  # noqa: E402  (Phase B's statistics, staged at its registered bytes)

TAGS = ("B0a", "B1a", "B1b", "B0b")
WORKLOADS = ("W1", "W2", "W4", "W8", "W16")
RULED, INERT, REGRESSION = ("W2", "W4"), ("W1",), ("W16",)
BATCH = {"W1": 1, "W2": 2, "W4": 4, "W8": 8, "W16": 16}           # each workload's rows (the box's WORKLOADS)
BUCKETS = ("1", "2", "4", "8", "16")
SWITCHED_BUCKETS = ("2", "4", "8")
SELF_SMALL, SELF_SMALL_W = (0.985, 1.015), ("W1", "W2", "W4")
SELF_WIDE = (0.96, 1.04)
GAIN_MIN, INERT_MIN, REGRESSION_MIN = 1.03, 0.99, 0.99
TOL, SPREAD_X, SPREAD_MIN, K8_BUDGET, K8_GATED = pb.TOL, pb.SPREAD_X, pb.SPREAD_MIN, pb.K8_BUDGET, pb.K8_GATED
TEXTS = pb.TEXTS
GNF4_SHA = "41ef086e6a3e2be5b158ff1bb38e0de67862a949"     # grouped-nf4-gemm v0.50.0
MIXTRAL = "mistralai/Mixtral-8x7B-Instruct-v0.1"
REVS = {MIXTRAL: "eba92302a2861cdc0098cc54bc9f17cb2c47eb61"}          # FAM Amendment 5's pin (P141's)
LAYERS, WINDOWS, GROUP = 32, 24, 4
SERVER_SIZE = (16, 768)
OFF_ARMS, FLOORS = ("R", "rep", "half", "chunk", "mutant_scale"), ("half", "chunk")
ANSWERS = ("LICENSED", "SLOWER", "QUALITY_FAIL", "FUNCTION_FAIL")
HOST_CODES = {13: "free disk", 15: "card, compute capability, SM count or VRAM", 16: "host RAM", 18: "CUDA unusable",
              19: "download below the floor (the probe)"}
FIT_CODES = {41: "the guard does not fit the deadline (a slow boot)"}
KIT_CODES = {9: "staging, install or hash pins", 10: "torch does not import (the image)",
             28: "the image's torch is not the pinned build", 78: "run id, environment or frozen knobs", 130: "a signal"}
HALT_CODES = {25: "the premise failed on the card (rows or served): the lane halts until amended",
              31: "the default server does not fit at 16 x 768: the lane halts until amended"}


#: grouped-nf4-gemm v0.50.0's decode-GEMV tally (``nf4_grouped.dispatch_counts()``): the box records every key, zeros kept,
#: so K33's absence is a count that ran. (v0.50.0 has no K25 counter: K25 is read positively from e4b's own route tally,
#: ``ROUTE_SEEN``'s ``nf4_k25``, counted at the branch that calls it.)
GNF4_DISPATCH_KEYS = ("dotpad", "dotpad_splitk", "scalar", "scalar_splitk", "bw_tree", "bw_prmt32", "bw_splitk")


def _routes(d):
    """e4b's route tally by route name (``ROUTE_SEEN`` keys are ``<route>|le256`` or ``<route>|gt256``); None when the
    record is not a dict (rule 7: absence is a fault, never an empty tally)."""
    if not isinstance(d, dict):
        return None
    out = {}
    for k, v in d.items():
        name = str(k).split("|", 1)[0]
        out[name] = out.get(name, 0) + v
    return out


def _bw_counted(d):
    """K33's launches from a dispatch record that names every gnf4 counter; None when the record is absent or partial."""
    if not isinstance(d, dict) or any(type(d.get(k)) is not int for k in GNF4_DISPATCH_KEYS):
        return None
    return sum(v for k, v in d.items() if str(k).startswith("bw_"))


def _rate(r, w):
    return ((r.get("workloads") or {}).get(w) or {}).get("decode_tok_s")


def _tok(r, w, n):
    return r["workloads"][w]["tokens"][str(n)]


def bucket_faults(tag: str, rec: dict) -> list:
    """Rule 7, fail-closed: each bucket's route tally and its complete gnf4 dispatch record must be present and show the
    registered route. A missing bucket, an empty route tally, or a dispatch record without every gnf4 counter is a
    fault: "no K33" and "no K25" are read from counts that ran, never from an absent record."""
    arm, out = tag[:2], []
    byb = rec.get("by_bucket")
    if not isinstance(byb, dict):
        return [f"{tag}: no per-bucket record"]
    for b in BUCKETS:
        e = byb.get(b)
        route = _routes(e.get("route")) if isinstance(e, dict) else None
        if not route:
            out.append(f"{tag}: bucket {b} has no route tally ({e!r})")
            continue
        bw = _bw_counted(e.get("dispatch"))
        if bw is None:
            out.append(f"{tag}: bucket {b} has no complete gnf4 dispatch record ({e.get('dispatch')!r})")
            continue
        single, k25 = route.get("nf4_singleton", 0), route.get("nf4_k25", 0)
        if b == "1" or (b in SWITCHED_BUCKETS and arm == "B1"):
            if not (single > 0 and bw > 0 and k25 == 0):
                out.append(f"{tag}: bucket {b} route {route}, K33 launches {bw} -- expected nf4_singleton and K33 only")
        else:
            if not (k25 > 0 and single == 0 and bw == 0):
                out.append(f"{tag}: bucket {b} route {route}, K33 launches {bw} -- expected nf4_k25 and no K33")
    return out


def speed_faults(arms: dict, e4b_sha: str) -> list:
    void = []
    for t in TAGS:
        r = arms.get(t)
        if not r or r.get("status") != "ok":
            void.append(f"{t}: missing or not ok")
            continue
        if r.get("e4b_sha") != e4b_sha:
            void.append(f"{t}: e4b {r.get('e4b_sha')} != {e4b_sha}")
        if r.get("gnf4_sha") != GNF4_SHA:
            void.append(f"{t}: grouped-nf4-gemm {r.get('gnf4_sha')} != {GNF4_SHA}")
        if REVS.get(r.get("model")) != r.get("revision"):
            void.append(f"{t}: model {r.get('model')}@{r.get('revision')} is not the registered pin")
        if (r.get("max_seqs"), r.get("max_tokens_per_seq")) != SERVER_SIZE:
            void.append(f"{t}: server {r.get('max_seqs')} x {r.get('max_tokens_per_seq')}, registered {SERVER_SIZE}")
        if "knob" not in r:
            void.append(f"{t}: no switch record (E4B_NF4_SMALLM_BW is recorded, unset as null)")
        elif (r["knob"] or "") != ("1" if t.startswith("B1") else ""):
            void.append(f"{t}: E4B_NF4_SMALLM_BW={r.get('knob')!r}")
        g = r.get("graph_status") or {}
        if [g.get(b) for b in BUCKETS] != ["graph"] * len(BUCKETS):
            void.append(f"{t}: graph_status {g}")
        void += bucket_faults(t, r)
        lens, reps = {str(r.get("short")), str(r.get("long"))}, r.get("reps")
        for w in WORKLOADS:
            wl = (r.get("workloads") or {}).get(w)
            if not isinstance(wl, dict):
                void.append(f"{t}: no {w} record")
                continue
            rd, tk = wl.get("rep_digests"), wl.get("tokens")
            if (not isinstance(reps, int) or reps <= 0 or not isinstance(rd, dict) or set(rd) != lens
                    or any(not isinstance(x, list) or len(x) != reps for x in rd.values())):
                void.append(f"{t} {w}: rep_digests {rd!r} do not name both lengths with {reps!r} reps each")
            if not isinstance(tk, dict) or set(tk) != lens:
                void.append(f"{t} {w}: tokens do not name both lengths ({sorted(tk) if isinstance(tk, dict) else tk!r})")
            elif any(not isinstance(x, list) or len(x) != BATCH[w] for x in tk.values()):
                void.append(f"{t} {w}: tokens are not {BATCH[w]} rows at each length")
            if not _rate(r, w) or _rate(r, w) <= 0:
                void.append(f"{t}: {w} decode-only slope is void")
    if void:
        return void
    for k in ("prompts_sha256", "short", "long", "reps", "buckets"):
        if len({json.dumps(arms[t].get(k)) for t in TAGS}) != 1:
            void.append(f"the arms differ in {k}")
    return void


def _pass_faults(e, arm, n, C) -> list:
    """P115's per-pass bucket statistics (``p115_reduce._pass_faults`` without the fused-kernel census): C - 1 eager
    decode steps per part, the part padded to its bucket; ``half`` decodes the group as two halves."""
    out = []
    halves = arm == "half"
    want_calls = (C - 1) * LAYERS * (2 if halves else 1)
    if e.get("decode_calls") != want_calls:
        out.append(f"{e.get('decode_calls')} decode attention calls, expected {want_calls}")
    st = e.get("graph_status") or {}
    if [st.get(b) for b in BUCKETS] != ["eager: capture=False"] * len(BUCKETS):
        out.append(f"graph_status {st}")
    parts = [n // 2, n - n // 2] if halves else [n]
    want = {}
    for p in parts:
        b = min(int(x) for x in BUCKETS if int(x) >= p)
        w = want.setdefault(str(b), {"eager_steps": 0, "pad_rows": 0})
        w["eager_steps"] += C - 1
        w["pad_rows"] += (b - p) * (C - 1)
    gs = e.get("graph_stats") or {}
    for b, w in want.items():
        got = gs.get(b, {})
        if (got.get("eager_steps"), got.get("pad_rows")) != (w["eager_steps"], w["pad_rows"]):
            out.append(f"bucket {b} stats {got}, expected {w}")
    if any(not isinstance(v, dict) or type(v.get("replays")) is not int for v in gs.values()):
        out.append(f"graph_stats without a replay count: {gs}")
    elif any(v["replays"] for v in gs.values()):
        out.append(f"a replay in a capture=False pass: {gs}")
    return out


def quality_faults(off: dict | None, on: dict | None, e4b_sha: str) -> list:
    void = []
    for name, rec in (("quality_off", off), ("quality_on", on)):
        if not rec or rec.get("status") != "ok":
            void.append(f"{name}: missing or not ok")
            continue
        if rec.get("e4b_sha") != e4b_sha:
            void.append(f"{name}: e4b {rec.get('e4b_sha')} != {e4b_sha}")
        if rec.get("gnf4_sha") != GNF4_SHA:
            void.append(f"{name}: grouped-nf4-gemm {rec.get('gnf4_sha')} != {GNF4_SHA}")
        if REVS.get(rec.get("model")) != rec.get("revision"):
            void.append(f"{name}: model {rec.get('model')}@{rec.get('revision')} is not the registered pin")
        if "knob" not in rec:
            void.append(f"{name}: no switch record (E4B_NF4_SMALLM_BW is recorded, unset as null)")
        if rec.get("group") != GROUP:
            void.append(f"{name}: group {rec.get('group')}, registered {GROUP}")
        if (rec.get("max_seqs"), rec.get("max_tokens_per_seq")) != SERVER_SIZE:
            void.append(f"{name}: server {rec.get('max_seqs')} x {rec.get('max_tokens_per_seq')}, registered {SERVER_SIZE}")
    if void:
        return void
    if (off.get("knob") or "") != "" or (on.get("knob") or "") != "1":
        void.append(f"the phases ran E4B_NF4_SMALLM_BW={off.get('knob')!r} / {on.get('knob')!r}, registered unset / '1'")
    for k in ("cont", "prompt", "chunk", "floor_chunk", "layers", "windows_sha256", "buckets"):
        if off.get(k) != on.get(k):
            void.append(f"the phases differ in {k}: {off.get(k)} / {on.get(k)}")
    if off.get("layers") != LAYERS:
        void.append(f"{off.get('layers')} layers, Mixtral registers {LAYERS}")
    off_bw, off_r = _bw_counted(off.get("dispatch_measure")), _routes(off.get("route_measure"))
    on_bw, on_r = _bw_counted(on.get("dispatch_measure")), _routes(on.get("route_measure"))
    if off_bw is None:
        void.append(f"quality_off: no complete gnf4 dispatch record ({off.get('dispatch_measure')!r})")
    elif off_bw != 0:
        void.append(f"quality_off: K33 launched in B0's measurement {off.get('dispatch_measure')}")
    if not off_r or off_r.get("nf4_k25", 0) <= 0 or off_r.get("nf4_singleton", 0) != 0:
        void.append(f"quality_off: route {off_r}, registered nf4_k25 and no nf4_singleton")
    if on_bw is None or on_bw <= 0:
        void.append(f"quality_on: B1's measurement shows no K33 ({on.get('dispatch_measure')!r})")
    if not on_r or on_r.get("nf4_singleton", 0) <= 0 or on_r.get("nf4_k25", 0) != 0:
        void.append(f"quality_on: route {on_r}, registered nf4_singleton and no nf4_k25")
    C = off.get("cont")
    sizes = [min(GROUP, WINDOWS - g0) for g0 in range(0, WINDOWS, GROUP)]
    for name, rec, arms in (("quality_off", off, OFF_ARMS), ("quality_on", on, ("ON",))):
        for t in TEXTS:
            per = (rec.get("per_window") or {}).get(t, {})
            eng = (rec.get("engagement") or {}).get(t, {})
            for arm in arms:
                want = GROUP if arm == "rep" else WINDOWS
                if len(per.get(arm, [])) != want:
                    void.append(f"{name} {t}: arm {arm} has {len(per.get(arm, []))} windows, expected {want}")
                passes = eng.get(arm, [])
                arm_sizes = sizes[:1] if arm == "rep" else sizes
                if len(passes) != len(arm_sizes):
                    void.append(f"{name} {t} {arm}: {len(passes)} passes, expected {len(arm_sizes)}")
                    continue
                for k, (e, n) in enumerate(zip(passes, arm_sizes)):
                    void += [f"{name} {t} {arm} pass {k}: {m}" for m in _pass_faults(e, arm, n, C)]
    return void


def quality_read(off: dict, on: dict) -> dict:
    out = {}
    for t in TEXTS:
        ref = {x["window"]: x["nll"] for x in off["per_window"][t]["R"]}
        st = {a: pb._stats(off["per_window"][t], a, ref) for a in OFF_ARMS if a != "R"}
        st["ON"] = pb._stats(on["per_window"][t], "ON", ref)
        draws = list(FLOORS) + ([] if (off.get("rep_identical") or {}).get(t) else ["rep"])
        fl = {"draws": draws, "B_floor": max(abs(st[f]["bias"]) for f in draws),
              "S_floor": max(st[f]["spread"] for f in draws)}
        ppl_r, ppl_on = pb._ppl(off["per_window"][t]["R"]), pb._ppl(on["per_window"][t]["ON"])
        out[t] = {"stats": st, "floor": fl, "bias_bar": fl["B_floor"] + TOL,
                  "spread_bar": SPREAD_X * max(fl["S_floor"], SPREAD_MIN),
                  "mutant_passes": pb._passes(st["mutant_scale"], fl), "on_passes": pb._passes(st["ON"], fl),
                  "k8": {"ppl_R": ppl_r, "ppl_ON": ppl_on, "delta": ppl_on - ppl_r, "gated": t in K8_GATED,
                         "passes": abs(ppl_on - ppl_r) <= K8_BUDGET}}
    return out


def reduce(arms: dict, off, on, e4b_sha: str, fit=None) -> dict:
    out = {"verdict": None, "reasons": [], "bars": {"self_small": list(SELF_SMALL), "self_wide": list(SELF_WIDE),
                                                    "gain": GAIN_MIN, "inert": INERT_MIN, "regression": REGRESSION_MIN}}
    void = []
    if not fit or fit.get("status") != "ok":
        void.append(f"the fit check failed or is missing: {(fit or {}).get('error')}")
    void += speed_faults(arms, e4b_sha) + quality_faults(off, on, e4b_sha)
    q = None
    if not void:
        q = quality_read(off, on)
        void += [f"the mutant passes the quality bar on {t}" for t in TEXTS if q[t]["mutant_passes"]]
    if void:
        out.update(verdict="VOID", reasons=void)
        return out
    out["quality"] = q
    rates = {t: {w: _rate(arms[t], w) for w in WORKLOADS} for t in TAGS}
    out["decode_tok_s"] = rates
    selfp = {f"{a}b/{a}a {w}": rates[f"{a}b"][w] / rates[f"{a}a"][w] for a in ("B0", "B1") for w in WORKLOADS}
    out["self_pairs"] = {k: round(v, 5) for k, v in selfp.items()}
    pairs = {w: [rates["B1a"][w] / rates["B0a"][w], rates["B1b"][w] / rates["B0b"][w]] for w in WORKLOADS}
    out["pair_ratios"] = {w: [round(x, 5) for x in v] for w, v in pairs.items()}
    out["g"] = {w: round(min(v), 5) for w, v in pairs.items()}
    B0a = arms["B0a"]
    fn = []
    for w in WORKLOADS:
        for n in (B0a["short"], B0a["long"]):
            for x, y in (("B0b", "B0a"), ("B1b", "B1a")) + ((("B1a", "B0a"), ("B1b", "B0b")) if w in INERT else ()):
                if _tok(arms[x], w, n) != _tok(arms[y], w, n):
                    rows = [i for i, (p, q_) in enumerate(zip(_tok(arms[x], w, n), _tok(arms[y], w, n))) if p != q_]
                    fn.append(f"{x} != {y} on {w} at {n} tokens, rows {rows}")
    # reported, never a rung: B1 against B0 where the switch acts or may act (W2, W4, W8, and W16's ramp), by row
    out["token_diffs"] = {f"{x}/{y} {w} {n}": [i for i, (p, q_) in enumerate(zip(_tok(arms[x], w, n), _tok(arms[y], w, n)))
                                             if p != q_]
                          for w in WORKLOADS if w not in INERT for n in (B0a["short"], B0a["long"])
                          for x, y in (("B1a", "B0a"), ("B1b", "B0b"))}
    for t in TAGS:
        for w in WORKLOADS:
            for n, ds in arms[t]["workloads"][w]["rep_digests"].items():
                if len(set(ds)) != 1:
                    fn.append(f"{t} {w} at {n} tokens: timed reps differ")
    noisy = []
    for k, v in selfp.items():
        w = k.split()[-1]
        lo, hi = SELF_SMALL if w in SELF_SMALL_W else (SELF_WIDE if w == "W16" else (0.0, float("inf")))
        if not lo <= v <= hi:
            noisy.append(f"{k} = {v:.5f} outside [{lo}, {hi}]")
    qfail = [f"{t}: ON bias {q[t]['stats']['ON']['bias']:.5f} / spread {q[t]['stats']['ON']['spread']:.5f} against "
             f"bars {q[t]['bias_bar']:.5f} / {q[t]['spread_bar']:.5f}" for t in TEXTS if not q[t]["on_passes"]]
    qfail += [f"{t}: K8 perplexity moved {q[t]['k8']['delta']:+.4f} (budget {K8_BUDGET})" for t in TEXTS
              if q[t]["k8"]["gated"] and not q[t]["k8"]["passes"]]
    slow = [f"{w}: g {out['g'][w]} < {GAIN_MIN}" for w in RULED if out["g"][w] < GAIN_MIN]
    slow += [f"inert {w}: g {out['g'][w]} < {INERT_MIN}" for w in INERT if out["g"][w] < INERT_MIN]
    slow += [f"the W16 regression bar: g {out['g'][w]} < {REGRESSION_MIN}" for w in REGRESSION
             if out["g"][w] < REGRESSION_MIN]
    txt = (f"g2 {out['g']['W2']}, g4 {out['g']['W4']}; inert W1 {out['g']['W1']}; W16 regression {out['g']['W16']}; "
           f"W8 {out['g']['W8']} (reported)")
    if fn:
        out.update(verdict="FUNCTION_FAIL", reasons=fn)
    elif noisy:
        out.update(verdict="NOISY", reasons=noisy)
    elif qfail:
        out.update(verdict="QUALITY_FAIL", reasons=qfail)
    elif slow:
        out.update(verdict="SLOWER", reasons=slow + [txt])
    else:
        out.update(verdict="LICENSED", reasons=[f"tokens as registered; quality within P110's bars at group 4; {txt}"])
    return out


# ---------------------------------------------------------------- classification --

def classify(rc: int, started: bool, run_id: str = "") -> dict:
    """The registered class of one fetched box: ``{"event", "started", "host_refused_cap", "halt", "why"}``. Amendment 2:
    a ``p234-shake-N`` box is an explore shakedown: SHAKE_OK at rc 0; SHAKE_HOST_REFUSED on a host or fit code (the
    host or its boot, not the kit: no halt, the next shake id); SHAKE_FAIL on any other code, which halts the lane until
    the kit is fixed. None of them counts as a started box, a strike or a reading's host refusal."""
    if run_id.startswith("p234-shake-"):
        base = {"started": False, "host_refused_cap": False}
        if rc == 0:
            return {**base, "event": "SHAKE_OK", "halt": False, "why": "an explore shakedown: never a reading, never counted"}
        if rc in HOST_CODES or rc in FIT_CODES:
            return {**base, "event": "SHAKE_HOST_REFUSED", "halt": False,
                    "why": f"exit {rc}: {HOST_CODES.get(rc) or FIT_CODES.get(rc)}; the host, not the kit: the next shake id"}
        return {**base, "event": "SHAKE_FAIL", "halt": True,
                "why": f"exit {rc}: {KIT_CODES.get(rc, 'a kit or harness fault')}; the lane halts until the kit is fixed"}
    if not started:
        if rc in HOST_CODES:
            return {"event": "HOST_REFUSED", "started": False, "host_refused_cap": True, "halt": False,
                    "why": HOST_CODES[rc]}
        if rc in FIT_CODES:
            return {"event": "FIT_REFUSED", "started": False, "host_refused_cap": False, "halt": False,
                    "why": FIT_CODES[rc] + "; two consecutive are one strike"}
        return {"event": "PRE_INSTALL_HALT", "started": False, "host_refused_cap": False, "halt": True,
                "why": KIT_CODES.get(rc, f"unexpected pre-install exit {rc}")}
    if rc == 0:
        return {"event": "ANSWER", "started": True, "host_refused_cap": False, "halt": False, "why": "the verdict"}
    if rc in HALT_CODES:
        return {"event": "HALT", "started": True, "host_refused_cap": False, "halt": True, "why": HALT_CODES[rc]}
    why = "VOID or NOISY (the verdict): a strike" if rc == 27 else f"exit {rc} after the start: a strike"
    return {"event": "NO_ANSWER", "started": True, "host_refused_cap": False, "halt": False, "why": why}


def read_box(work: str) -> dict:
    codes = sorted(glob.glob(os.path.join(work, "P234_EXIT_CODE.*")))
    if len(codes) != 1 or not os.path.isfile(os.path.join(work, "RUN_ID")):
        state = ("STARTED PRESENT: by the table this box is a box-started run without an answer unless the hand reading "
                 "shows otherwise" if os.path.exists(os.path.join(work, "STARTED")) else
                 "STARTED absent here (if the box started and nothing could be fetched, it still counts as started)")
        raise ValueError(f"{work!r} is not a P234 box work directory: it needs exactly one P234_EXIT_CODE.<nonce> "
                         f"(found {len(codes)}) and RUN_ID. {state}. The lane HALTS until the box's files are read by hand.")
    rc = int(open(codes[0], encoding="utf-8").read().strip())
    return {"run_id": open(os.path.join(work, "RUN_ID"), encoding="utf-8").read().strip(), "rc": rc,
            "started": os.path.exists(os.path.join(work, "STARTED"))}


# ------------------------------------------------------------------ self-test --

E = "a" * 40


def _disp(bw=0, drop=()):
    """A gnf4 dispatch record as the box writes it: every counter, zeros kept."""
    d = {k: 0 for k in GNF4_DISPATCH_KEYS}
    d["bw_prmt32"] = bw
    return {k: v for k, v in d.items() if k not in drop}


def _byb(arm, drop=None, override=None):
    def entry(b):
        if b == "1" or (b in SWITCHED_BUCKETS and arm == "B1"):
            return {"route": {"nf4_singleton|le256": 64}, "dispatch": _disp(128)}
        return {"route": {"nf4_k25|le256": 64}, "dispatch": _disp(0)}
    d = {b: entry(b) for b in BUCKETS if b != drop}
    d.update(override or {})
    return d


def _fake_arm(tag, rates, toks, *, e4b=E, byb=None, graphs=None, knob=None, digests=None, short=4, long_=8):
    arm = tag[:2]

    def wl(rate, b):
        return {"decode_tok_s": rate, "tokens": {str(short): [list(t[:short]) for t in toks[:b]],
                                                 str(long_): [list(t) for t in toks[:b]]},
                "rep_digests": digests or {str(short): ["x"] * 3, str(long_): ["y"] * 3}}
    return {"status": "ok", "e4b_sha": e4b, "gnf4_sha": GNF4_SHA, "model": MIXTRAL, "revision": REVS[MIXTRAL],
            "max_seqs": 16, "max_tokens_per_seq": 768, "knob": ("1" if arm == "B1" else None) if knob is None else knob,
            "graph_status": graphs or {b: "graph" for b in BUCKETS}, "by_bucket": _byb(arm) if byb is None else byb,
            "prompts_sha256": "p", "short": short, "long": long_, "reps": 3, "buckets": [1, 2, 4, 8, 16],
            "workloads": {w: wl(rates[w], BATCH[w]) for w in WORKLOADS}}


def _fake_pass(arm, n, C):
    halves = arm == "half"
    parts = [n // 2, n - n // 2] if halves else [n]
    gs = {}
    for p in parts:
        b = min(int(x) for x in BUCKETS if int(x) >= p)
        w = gs.setdefault(str(b), {"eager_steps": 0, "pad_rows": 0, "replays": 0})
        w["eager_steps"] += C - 1
        w["pad_rows"] += (b - p) * (C - 1)
    return {"decode_calls": (C - 1) * LAYERS * (2 if halves else 1),
            "graph_status": {b: "eager: capture=False" for b in BUCKETS}, "graph_stats": gs}


def _fake_quality(phase, *, on_bias=0.002, mutant=0.4, group=GROUP, bw=None, routes=None, drop_pass=False):
    C = 8
    arms = OFF_ARMS if phase == "off" else ("ON",)
    sizes = [min(GROUP, WINDOWS - g0) for g0 in range(0, WINDOWS, GROUP)]
    per, eng = {}, {}
    for t in TEXTS:
        per[t], eng[t] = {}, {}
        for arm in arms:
            n = GROUP if arm == "rep" else WINDOWS
            base = 2.0 + 0.01 * (t == "c4val1")
            d = {"R": 0.0, "rep": 0.0, "half": 0.001, "chunk": -0.001, "mutant_scale": mutant, "ON": on_bias}[arm]
            per[t][arm] = [{"window": i, "nll": base + 0.001 * i + d, "argmax_agree": 1.0} for i in range(n)]
            arm_sizes = sizes[:1] if arm == "rep" else sizes
            eng[t][arm] = [_fake_pass(arm, s, C) for s in arm_sizes][: (len(arm_sizes) - 1 if drop_pass else None)]
    return {"status": "ok", "e4b_sha": E, "gnf4_sha": GNF4_SHA, "model": MIXTRAL, "revision": REVS[MIXTRAL],
            "group": group, "max_seqs": 16, "max_tokens_per_seq": 768, "knob": "1" if phase == "on" else None,
            "cont": C, "prompt": 512, "chunk": 512, "floor_chunk": 256, "layers": LAYERS, "windows_sha256": {"w": "s"},
            "buckets": [1, 2, 4, 8, 16], "rep_identical": {t: True for t in TEXTS},
            "dispatch_measure": (_disp(999) if phase == "on" else _disp(0)) if bw is None else bw,
            "route_measure": ({"nf4_singleton|le256": 99, "nf4_mtile_captured|gt256": 6} if phase == "on" else
                              {"nf4_k25|le256": 99, "nf4_mtile_captured|gt256": 6}) if routes is None else routes,
            "per_window": per, "engagement": eng}


def _classify_cases() -> list:
    out = []
    with tempfile.TemporaryDirectory() as d:
        def box(rc, started, codes=1):
            w = os.path.join(d, f"w{rc}{int(started)}{codes}")
            os.makedirs(w)
            for i in range(codes):
                open(os.path.join(w, f"P234_EXIT_CODE.n{i}"), "w").write(f"{rc}\n")
            open(os.path.join(w, "RUN_ID"), "w").write("p234-5090-1\n")
            if started:
                open(os.path.join(w, "STARTED"), "w").close()
            return w
        b = read_box(box(13, False))
        out.append(("disk before the start is HOST_REFUSED", classify(b["rc"], b["started"])["event"] == "HOST_REFUSED"))
        out.append(("19 (the download probe) is HOST_REFUSED, never a strike", classify(19, False)["event"] == "HOST_REFUSED"
                    and not classify(19, False)["halt"]))
        out.append(("41 is FIT_REFUSED, never a halt", classify(41, False)["event"] == "FIT_REFUSED" and not classify(41, False)["halt"]))
        out.append(("a kit fault halts", classify(9, False)["event"] == "PRE_INSTALL_HALT" and classify(9, False)["halt"]))
        out.append(("the premise halts after the start", classify(25, True)["event"] == "HALT"))
        out.append(("the fit halts after the start", classify(31, True)["event"] == "HALT"))
        b = read_box(box(0, True))
        out.append(("0 after the start is an ANSWER", classify(b["rc"], b["started"])["event"] == "ANSWER"))
        out.append(("27 after the start is NO_ANSWER", classify(27, True)["event"] == "NO_ANSWER"))
        out.append(("9 after the start is NO_ANSWER (p234-5090-1's class)", classify(9, True, "p234-5090-1")["event"] == "NO_ANSWER"))
        sh0, sh9 = classify(0, True, "p234-shake-1"), classify(9, True, "p234-shake-2")
        out.append(("a shakedown is never counted (Amendment 2)", sh0["event"] == "SHAKE_OK" and not sh0["started"]
                    and not sh0["halt"] and sh9["event"] == "SHAKE_FAIL" and not sh9["started"] and sh9["halt"]))
        for c in sorted(HOST_CODES) + sorted(FIT_CODES):
            sh = classify(c, c != 41, "p234-shake-1")
            out.append((f"a shakedown's host/fit code {c} is SHAKE_HOST_REFUSED, never a halt",
                        sh["event"] == "SHAKE_HOST_REFUSED" and not sh["halt"] and not sh["started"]))
        for c in sorted(KIT_CODES) + [21, 22, 25, 130]:
            sh = classify(c, True, "p234-shake-1")
            out.append((f"a shakedown's code {c} is SHAKE_FAIL and halts", sh["event"] == "SHAKE_FAIL" and sh["halt"]))
        for name, w in (("two exit files", box(0, True, codes=2)), ("no exit file", box(0, True, codes=0))):
            try:
                read_box(w)
                out.append((f"fails closed: {name}", False))
            except ValueError:
                out.append((f"fails closed: {name}", True))
    return out


def self_test() -> int:
    base = [[r * 100 + i for i in range(8)] for r in range(16)]
    B0 = {"W1": 50.0, "W2": 60.0, "W4": 80.0, "W8": 100.0, "W16": 140.0}
    B1 = {"W1": 50.0, "W2": 70.0, "W4": 90.0, "W8": 98.0, "W16": 140.0}
    fit = {"status": "ok"}

    def arms(r0=None, r1=None, **over):
        a = {t: _fake_arm(t, (r1 or B1) if t.startswith("B1") else (r0 or B0), base) for t in TAGS}
        a.update(over)
        return a

    def v(a, off=None, on=None, f=fit):
        return reduce(a, off or _fake_quality("off"), on or _fake_quality("on"), E, fit=f)["verdict"]
    moved = [list(r) for r in base]
    moved[2][3] += 1                       # row 2: in W4, W8 and W16, never in W1 or W2
    moved0 = [list(r) for r in base]
    moved0[0][3] += 1                      # row 0: in every workload, W1 included
    cases = [
        ("licensed", v(arms()) == "LICENSED"),
        ("slower at W2", v(arms(r1={**B1, "W2": 60.0 * 1.02})) == "SLOWER"),
        ("slower at W4", v(arms(r1={**B1, "W4": 80.0 * 1.02})) == "SLOWER"),
        ("the W16 regression bar", v(arms(r1={**B1, "W16": 140.0 * 0.985})) == "SLOWER"),
        ("the inert W1 below 0.99", v(arms(r1={**B1, "W1": 50.0 * 0.985})) == "SLOWER"),
        ("W16 at 0.99 exactly passes", v(arms(r1={**B1, "W16": 140.0 * 0.99})) == "LICENSED"),
        ("W8 reported only", v(arms(r1={**B1, "W8": 50.0})) == "LICENSED"),
        ("quality fail", v(arms(), on=_fake_quality("on", on_bias=0.05)) == "QUALITY_FAIL"),
        ("the mutant passes: VOID", v(arms(), off=_fake_quality("off", mutant=0.001)) == "VOID"),
        ("noisy self-pair at W2", v(arms(B0b=_fake_arm("B0b", {**B0, "W2": 60.0 * 1.02}, base))) == "NOISY"),
        ("noisy at W16 (wide band)", v(arms(B1b=_fake_arm("B1b", {**B1, "W16": 140.0 * 1.05}, base))) == "NOISY"),
        ("W16 inside the wide band", v(arms(B1b=_fake_arm("B1b", {**B1, "W16": 140.0 * 1.03}, base))) == "LICENSED"),
        ("a self-pair token moves", v(arms(B1b=_fake_arm("B1b", B1, moved))) == "FUNCTION_FAIL"),
        ("B1 differs from B0 at W1", v(arms(B1a=_fake_arm("B1a", B1, moved0), B1b=_fake_arm("B1b", B1, moved0))) == "FUNCTION_FAIL"),
        ("B1 differs from B0 at W4 to W16 only: reported, not a rung",
         v(arms(B1a=_fake_arm("B1a", B1, moved), B1b=_fake_arm("B1b", B1, moved))) == "LICENSED"),
        ("function fail before noisy", v(arms(B1b=_fake_arm("B1b", {**B1, "W2": 70.0 * 1.05}, moved))) == "FUNCTION_FAIL"),
        ("reps differ", v(arms(B0a=_fake_arm("B0a", B0, base, digests={"4": ["x"] * 3, "8": ["y", "z", "y"]}))) == "FUNCTION_FAIL"),
        ("fit missing", v(arms(), f=None) == "VOID"),
        ("fit failed", v(arms(), f={"status": "error", "error": "OOM"}) == "VOID"),
        ("an arm missing", v({t: x for t, x in arms().items() if t != "B0b"}) == "VOID"),
        ("wrong e4b", v(arms(B1a=_fake_arm("B1a", B1, base, e4b="b" * 40))) == "VOID"),
        ("B0 with the knob", v(arms(B0a=_fake_arm("B0a", B0, base, knob="1"))) == "VOID"),
        ("a bucket eager", v(arms(B0a=_fake_arm("B0a", B0, base, graphs={**{b: "graph" for b in BUCKETS}, "4": "eager: x"}))) == "VOID"),
        ("bucket 16 missing", v(arms(B1a=_fake_arm("B1a", B1, base, byb=_byb("B1", drop="16")))) == "VOID"),
        ("bucket 4 empty route", v(arms(B0b=_fake_arm("B0b", B0, base, byb=_byb("B0", override={"4": {"route": {}, "dispatch": _disp(0)}})))) == "VOID"),
        ("no per-bucket record", v(arms(B0a=_fake_arm("B0a", B0, base, byb="x"))) == "VOID"),
        ("B1 bucket 2 on K25", v(arms(B1a=_fake_arm("B1a", B1, base, byb=_byb("B1", override={"2": {"route": {"nf4_k25|le256": 8}, "dispatch": _disp(0)}})))) == "VOID"),
        ("B0 bucket 8 with K33", v(arms(B0b=_fake_arm("B0b", B0, base, byb=_byb("B0", override={"8": {"route": {"nf4_k25|le256": 8}, "dispatch": _disp(2)}})))) == "VOID"),
        ("bucket 16 with K33", v(arms(B1b=_fake_arm("B1b", B1, base, byb=_byb("B1", override={"16": {"route": {"nf4_k25|le256": 8, "nf4_singleton|le256": 2}, "dispatch": _disp(2)}})))) == "VOID"),
        ("bucket 1 without K33", v(arms(B0a=_fake_arm("B0a", B0, base, byb=_byb("B0", override={"1": {"route": {"nf4_singleton|le256": 8}, "dispatch": _disp(0)}})))) == "VOID"),
        ("void slope", v(arms(B1a=_fake_arm("B1a", {**B1, "W4": None}, base))) == "VOID"),
        ("quality group 1", v(arms(), on=_fake_quality("on", group=1)) == "VOID"),
        ("quality off with K33", v(arms(), off=_fake_quality("off", bw=_disp(3))) == "VOID"),
        ("quality on without K33", v(arms(), on=_fake_quality("on", bw=_disp(0))) == "VOID"),
        ("a quality pass missing", v(arms(), off=_fake_quality("off", drop_pass=True)) == "VOID"),
        ("quality missing", reduce(arms(), None, _fake_quality("on"), E, fit=fit)["verdict"] == "VOID"),
    ]

    def arm_without(tag, path):            # one leaf deleted from an arm, the rest as LICENSED's
        a = arms()
        d = a[tag]
        for k in path[:-1]:
            d = d[k]
        del d[path[-1]]
        return a

    def q_without(phase, key):
        q = _fake_quality(phase)
        del q[key]
        return q

    def byb_disp(arm, b, disp):
        return _byb(arm, override={b: {"route": _byb(arm)[b]["route"], "dispatch": disp}})
    cases += [
        ("route keys fold by name", _routes({"nf4_k25|le256": 2, "nf4_k25|gt256": 1}) == {"nf4_k25": 3}
         and _routes(None) is None and _bw_counted(_disp(4, drop=("bw_splitk",))) is None and _bw_counted(_disp(4)) == 4),
        ("A: B0 bucket 2 dispatch missing",
         v(arms(B0a=_fake_arm("B0a", B0, base, byb=_byb("B0", override={"2": {"route": {"nf4_k25|le256": 8}}})))) == "VOID"),
        ("A: B0 bucket 4 dispatch None", v(arms(B0b=_fake_arm("B0b", B0, base, byb=byb_disp("B0", "4", None)))) == "VOID"),
        ("A: B0 bucket 8 dispatch {}", v(arms(B0a=_fake_arm("B0a", B0, base, byb=byb_disp("B0", "8", {})))) == "VOID"),
        ("A: bucket 16 dispatch partial",
         v(arms(B1a=_fake_arm("B1a", B1, base, byb=byb_disp("B1", "16", _disp(0, drop=("bw_tree",)))))) == "VOID"),
        ("A: bare route keys still read", v(arms(B0a=_fake_arm("B0a", B0, base, byb=_byb(
            "B0", override={"2": {"route": {"nf4_k25": 8}, "dispatch": _disp(0)}})))) == "LICENSED"),
        ("B: B0 arm knob missing", v(arm_without("B0a", ["knob"])) == "VOID"),
        ("B: B1 arm knob missing", v(arm_without("B1b", ["knob"])) == "VOID"),
        ("B: quality_off knob missing", v(arms(), off=q_without("off", "knob")) == "VOID"),
        ("C: rep_digests missing", v(arm_without("B0a", ["workloads", "W2", "rep_digests"])) == "VOID"),
        ("C: rep_digests empty", v(arms(B1a={**_fake_arm("B1a", B1, base), "workloads": {
            **_fake_arm("B1a", B1, base)["workloads"], "W2": {**_fake_arm("B1a", B1, base)["workloads"]["W2"], "rep_digests": {}}}})) == "VOID"),
        ("C: a length's digests missing", v(arms(B0b=_fake_arm("B0b", B0, base, digests={"4": ["x"] * 3}))) == "VOID"),
        ("C: too few reps", v(arms(B1b=_fake_arm("B1b", B1, base, digests={"4": ["x"] * 2, "8": ["y"] * 2}))) == "VOID"),
        ("C: tokens missing a length", v(arm_without("B1a", ["workloads", "W4", "tokens", "8"])) == "VOID"),
        ("C: a workload missing", v(arm_without("B0a", ["workloads", "W8"])) == "VOID"),
        ("D: quality_off dispatch missing", v(arms(), off=q_without("off", "dispatch_measure")) == "VOID"),
        ("D: quality_off route missing", v(arms(), off=q_without("off", "route_measure")) == "VOID"),
        ("D: quality_off on singleton", v(arms(), off=_fake_quality("off", routes={"nf4_singleton|le256": 99})) == "VOID"),
        ("D: quality_off K25 and singleton",
         v(arms(), off=_fake_quality("off", routes={"nf4_k25|le256": 99, "nf4_singleton|le256": 1})) == "VOID"),
        ("E: quality_on partly on K25",
         v(arms(), on=_fake_quality("on", routes={"nf4_singleton|le256": 99, "nf4_k25|le256": 50})) == "VOID"),
        ("E: quality_on route missing", v(arms(), on=q_without("on", "route_measure")) == "VOID"),
        ("E: quality_on dispatch partial", v(arms(), on=_fake_quality("on", bw=_disp(999, drop=("scalar",)))) == "VOID"),
    ]
    q = _fake_quality("off")
    del q["engagement"][TEXTS[0]]["R"][0]["graph_stats"]["4"]["replays"]
    cases.append(("a pass without a replay count", v(arms(), off=q) == "VOID"))
    q = _fake_quality("on")
    q["engagement"][TEXTS[1]]["ON"][2]["graph_stats"]["4"]["replays"] = None
    cases.append(("a replay count of None", v(arms(), on=q) == "VOID"))
    cases.append(("a dispatch counter of None", v(arms(B0a=_fake_arm("B0a", B0, base, byb=byb_disp(
        "B0", "2", {**_disp(0), "scalar": None})))) == "VOID"))
    a = arms()
    a["B0a"]["workloads"]["W4"]["tokens"]["8"].pop()
    cases.append(("a token row missing", v(a) == "VOID"))
    r = reduce(arms(), _fake_quality("off"), _fake_quality("on"), E, fit=fit)
    cases.append(("report", r["g"]["W2"] == round(70 / 60, 5) and r["g"]["W8"] == round(98 / 100, 5)
                  and set(r["quality"]) == set(TEXTS) and not any(r["token_diffs"].values())
                  and len(r["token_diffs"]) == 4 * 2 * 2 and not any(" W1 " in k for k in r["token_diffs"])))
    r = reduce(arms(B1a=_fake_arm("B1a", B1, moved), B1b=_fake_arm("B1b", B1, moved)), _fake_quality("off"),
               _fake_quality("on"), E, fit=fit)
    cases.append(("the W16 token diffs are reported by row", r["verdict"] == "LICENSED"
                  and r["token_diffs"]["B1a/B0a W16 8"] == [2] and r["token_diffs"]["B1b/B0b W4 8"] == [2]
                  and r["token_diffs"]["B1a/B0a W2 8"] == []))
    cases += _classify_cases()
    bad = [n for n, ok in cases if not ok]
    print(f"p234_reduce self-test {'OK' if not bad else 'FAILED ' + str(bad)} ({len(cases)} cases)")
    return 0 if not bad else 1


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--self-test", action="store_true")
    p.add_argument("--classify", action="store_true")
    p.add_argument("--work")
    p.add_argument("--dir")
    p.add_argument("--out")
    p.add_argument("--e4b-sha", default=os.environ.get("E4B_SHA", ""))
    p.add_argument("--check-args", action="store_true", help="parse and validate the arguments, then exit (tests)")
    a = p.parse_args(argv)
    if a.check_args:
        ok = bool(a.work) if a.classify else bool(a.dir and a.out and a.e4b_sha)
        print("ARGS OK" if ok else "ARGS BAD")
        return 0 if ok else 2
    if a.self_test:
        return self_test()
    if a.classify:
        try:
            box = read_box(a.work or "")
        except ValueError as e:
            print(f"P234_CLASSIFY_REFUSED {e}")
            return 2
        print("P234_CLASS " + json.dumps(dict(classify(box["rc"], box["started"], box["run_id"]), run_id=box["run_id"],
                                             rc=box["rc"])))
        return 0

    def load(name):
        f = os.path.join(a.dir, name)
        return json.load(open(f)) if os.path.exists(f) else None
    arms = {t: x for t in TAGS if (x := load(f"arm_{t}.json")) is not None}
    v = reduce(arms, load("quality_off.json"), load("quality_on.json"), a.e4b_sha, fit=load("fit.json"))
    json.dump(v, open(a.out, "w"), indent=1, default=str)
    print(f"P234_VERDICT {v['verdict']} {json.dumps(v['reasons'])[:3000]}")
    for k in ("g", "pair_ratios", "self_pairs", "decode_tok_s"):
        if k in v:
            print(f"  {k}: {json.dumps(v[k])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
