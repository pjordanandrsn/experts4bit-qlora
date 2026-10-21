#!/usr/bin/env python3
"""p223_reduce.py -- lane P223's registered rule (bench/p223/PREREG-p223.md, "The rule"), fixed before data. P149's box
Q+S reducer (bench/p149/p149_reduce.py) by substitution: the arms, no Q, no token-pinned blocks, no SANE phases.

Records (``p223_box.py``): ``arm_{D0a,I0a,I1a,D1a,D1b,I1b,I0b,D0b}.json`` (two ABBA blocks, P109's W16 and W1 at five
timed reps, every rep's tokens kept). Server S0 = D0 (the default) and I0 (``E4B_ATTN_SPLIT_CAP=1``,
``E4B_PAGED_BUCKETS=2,4,8,16``); server S1 = D1 (``E4B_SERVE_ATTN_INT8=1``) and I1 (int8 and both knobs).

The rule, in order:
  1. VOID       not an RTX 5090 at cc 12.0; an enforced power limit under 575 W; a record missing or not ok; P149's kit
                checks (pins, model, census and fusion modes, the server's shape, every bucket graph-captured and no eager
                decode step, a void slope, five reps, the arms on the same prompts and lengths, no decode-sized row on the
                int8 transient path); ENGAGEMENT, per process:
                  I arms  the split cap 1 (explicit) and the plan 1 at every bucket, the runner's buckets (2, 4, 8, 16),
                          zero bucket-1 replays, and every lone-row step in bucket 2 beside one pad row;
                  D arms  no cap (the library's plan), the buckets (1, 2, 4, 8, 16), bucket-1 replays present;
                  D1, I1  P149's int8 engagement counters nonzero (int8 projections built, every captured bucket's
                          capture routed through them); D0 and I0 build none.
  2. NOISY      (a VOID and a strike) any W1 self-pair -- D0b/D0a, I0b/I0a, D1b/D1a, I1b/I1a -- outside [0.985, 1.015].
  3. The schedule-invariance read rules the label: in every process of a server's I arm, each W16 row's 32-token stream
     equals the first 32 tokens of its own 160-token stream (timed rep i's 32 against timed rep i's 160, every rep), and
     the same for W1. The server is SCHEDULE_INVARIANT, or NOT_SCHEDULE_INVARIANT with the differing rows, steps and reps
     named. The D arms' same check is reported beside it. REPORTED apart, never ruling: each process's rep
     repeatability (the 160-token reps against each other, and the 32-token reps), so run-to-run nondeterminism shows
     as itself beside the schedule read.
  4. The price, reported, never a bar: p = tok/s(I) / tok/s(D) over the adjacent pairs (I0a/D0a and I0b/D0b for S0;
     I1a/D1a and I1b/D1b for S1); per server p1 = the min of its two W1 pairs and p16 the min of its two W16 pairs, each
     with its pairs and clocks; the four W16 self-pairs beside p16.
  5. The cross-W check, reported, never ruling: in every process, the W1 160-token stream against W16 row 0's, its first
     differing step or "equal".
The predictions are graded beside the read, never ruling.

    python p223_reduce.py --dir RUN_DIR --out verdict_p223.json --e4b-sha SHA
    python p223_reduce.py --gate RECORD --e4b-sha SHA       (one process's build and engagement, before the next)
    python p223_reduce.py --self-test
"""
from __future__ import annotations

import argparse
import json
import os
import sys

TAGS = ("D0a", "I0a", "I1a", "D1a", "D1b", "I1b", "I0b", "D0b")      # two ABBA blocks
SERVERS = {"S0": ("D0", "I0"), "S1": ("D1", "I1")}
PAIRS = {"S0": (("I0a", "D0a"), ("I0b", "D0b")), "S1": (("I1a", "D1a"), ("I1b", "D1b"))}
SELF_PAIRS = (("D0b", "D0a"), ("I0b", "I0a"), ("D1b", "D1a"), ("I1b", "I1a"))
NOISY_BAND = (0.985, 1.015)
REPS = 5
WORKLOADS = ("W16", "W1")
SHORT, LONG = 32, 160
ARM_BUCKETS = {"D": (1, 2, 4, 8, 16), "I": (2, 4, 8, 16)}
GNF4_SHA = "a83f6bb3a3a69aef8981fb81b6361bf62b06f6e7"    # v0.49.0: the CI pin of the kit's e4b commit (ships K37)
MODEL, REV = "mistralai/Mixtral-8x7B-Instruct-v0.1", "eba92302a2861cdc0098cc54bc9f17cb2c47eb61"
CENSUS = {"fuse_qkv_n": 0, "fuse_t1_glue_n": 65, "fuse_t1_glue_r2_n": [32, 32], "fuse_router_epilogue_n": 32}
FUSION_KNOBS = ("E4B_PAGED_FUSE_QKV", "E4B_FUSE_T1_GLUE", "E4B_FUSE_T1_GLUE_R2", "E4B_FUSE_ROUTER_EPI")
LAYERS, PROJ = 32, 128
CAPTURE_CALLS = 3          # int8 calls per projection per captured bucket: reported beside the nonzero rule (P149's form)
SERVER = {"max_seqs": 16, "max_tokens_per_seq": 768}
GPU_CLASS, CC, MIN_POWER_W = "5090", "12.0", 575.0
# the predictions (graded, never ruling)
P16_BAND, P1_BAND = (0.97, 1.00), (0.60, 0.95)


def _rate(r, w):
    return r["workloads"][w]["decode_tok_s"]


def _server(arm):
    return "S0" if arm[1] == "0" else "S1"


def first_diff(a: list, b: list):
    """The first step where two token lists differ (a length difference counts at the shorter's end), else None."""
    i = next((k for k, (p, q) in enumerate(zip(a, b)) if p != q), None)
    return i if i is not None or len(a) == len(b) else min(len(a), len(b))


# ------------------------------------------------------------------------------------------------- the checks --
def card_faults(name, r) -> list:
    out = []
    for key in ("card", "card_after"):
        c = r.get(key) or {}
        if GPU_CLASS not in str(c.get("name") or ""):
            out.append(f"{name}: {key} name {c.get('name')!r}, the lane registers the RTX {GPU_CLASS}")
        epl = c.get("enforced_power_limit")
        if not isinstance(epl, (int, float)) or epl < MIN_POWER_W:
            out.append(f"{name}: {key} enforced power limit {epl} W, the floor is {MIN_POWER_W:.0f} W")
    if str(r.get("cc")) != CC:
        out.append(f"{name}: compute capability {r.get('cc')}, registered {CC}")
    return out


def record_faults(name, r, e4b_sha, arm) -> list:
    out = []
    if r.get("e4b_sha") != e4b_sha:
        out.append(f"{name}: e4b {r.get('e4b_sha')} != {e4b_sha}")
    if r.get("gnf4_sha") != GNF4_SHA:
        out.append(f"{name}: grouped-nf4-gemm {r.get('gnf4_sha')} != {GNF4_SHA}")
    if (r.get("model"), r.get("revision")) != (MODEL, REV):
        return out + [f"{name}: model {r.get('model')}@{r.get('revision')} is not {MODEL}@{REV}"]
    if r.get("arm") != arm:
        out.append(f"{name}: the record says arm {r.get('arm')}")
    shape = {"max_seqs": r.get("max_seqs"), "max_tokens_per_seq": r.get("max_tokens_per_seq")}
    if shape != SERVER:
        out.append(f"{name}: server {shape}, registered {SERVER}")
    cen = {k: (r.get("fusions") or {}).get(k) for k in CENSUS}
    if cen != CENSUS:
        out.append(f"{name}: census {cen}, the default's {CENSUS}")
    modes = r.get("fusion_modes") or {}
    if any(str(modes.get(k)) not in ("auto", "1") for k in FUSION_KNOBS[1:]):
        out.append(f"{name}: fusion modes {modes}: the default server's folds resolve on for mixtral")
    return out + card_faults(name, r)


def _lone(calls: dict) -> dict:
    """``{bucket: calls}`` of the one-row decode calls in a ``DecodeCensus`` record (``"n:b"`` keys)."""
    out = {}
    for k, c in (calls or {}).items():
        n, b = k.split(":")
        if n == "1":
            out[b] = out.get(b, 0) + c
    return out


def engagement_faults(name, r, arm) -> list:
    """The knobs each arm binds, read from the KV's split record, the runner's buckets and replay counters, and the
    per-call decode census (PREREG "Engagement, recorded per process and binding")."""
    out = []
    inv = arm[0] == "I"
    want_b = list(ARM_BUCKETS[arm[0]])
    if r.get("runner_buckets") != want_b or r.get("buckets") != want_b:
        out.append(f"{name}: buckets {r.get('buckets')} / runner {r.get('runner_buckets')}, registered {want_b}")
    g = r.get("graph_status") or {}
    if [g.get(str(b)) for b in want_b] != ["graph"] * len(want_b) or set(g) - {str(b) for b in want_b}:
        out.append(f"{name}: graph_status {g}")
    sc = r.get("attn_split_cap") or {}
    plans = sc.get("plans") or {}
    if inv:
        if sc.get("cap") != 1 or sc.get("source") != "explicit":
            out.append(f"{name}: split cap {sc.get('cap')} ({sc.get('source')}), registered 1 (explicit)")
        if any(str(b) not in plans for b in want_b) or any((p or {}).get("n_split") != 1 for p in plans.values()):
            out.append(f"{name}: the split plan per bucket {plans}, registered 1 at every bucket {want_b}")
    elif sc.get("cap") is not None or sc.get("source") != "default-off" or plans:
        out.append(f"{name}: split cap {sc.get('cap')} ({sc.get('source')}, plans {plans}): the D arms keep the library's plan")
    gs = r.get("graph_stats") or {}
    if any((s or {}).get("eager_steps") for s in gs.values()):
        out.append(f"{name}: eager decode steps {gs}")
    rep1 = (gs.get("1") or {}).get("replays") or 0
    calls = {}
    for w in WORKLOADS:
        for c in ((r.get("workloads") or {}).get(w, {}).get("decode_calls") or {}).values():
            for k, v in c.items():
                calls[k] = calls.get(k, 0) + v
    lone = _lone(calls)
    window_pad = sum(((g or {}).get("2") or {}).get("pad_rows") or 0
                     for w in WORKLOADS for g in (((r.get("workloads") or {}).get(w) or {}).get("graph_stats") or {}).values())
    if inv:
        if rep1:
            out.append(f"{name}: {rep1} bucket-1 replays, registered zero")
        if not lone or set(lone) != {"2"}:
            out.append(f"{name}: the lone-row steps ran in buckets {lone}, registered bucket 2")
        elif lone["2"] != window_pad:
            out.append(f"{name}: {lone['2']} lone-row steps in bucket 2 but {window_pad} pad rows there over the same "
                       "windows: each lone row registers one")
    elif not rep1:
        out.append(f"{name}: no bucket-1 replay; the D arms run bucket 1")
    return out


def int8_faults(name, r, arm) -> list:
    """D1 and I1: P149's int8 engagement counters nonzero (projections built, every captured bucket's capture routed
    through them, no decode-sized row on the transient path); D0 and I0: none."""
    out = []
    a8 = r.get("attn_int8") or {}
    paths = r.get("capture_paths") or {}
    want_b = ARM_BUCKETS[arm[0]]
    if arm[1] == "0":
        if r.get("int8_modules") or a8.get("projections"):
            out.append(f"{name}: {arm} carries int8 projections ({r.get('int8_modules')}, {a8})")
        if any((p or {}).get("routed") or (p or {}).get("passed") for p in paths.values()):
            out.append(f"{name}: {arm}'s captures ran int8 calls {paths}")
        return out
    if not r.get("int8_modules") or not a8.get("projections"):
        out.append(f"{name}: {arm} built no int8 projections ({r.get('int8_modules')}, {a8})")
    for b in want_b:
        routed = ((paths.get(str(b)) or {}).get("routed") or {})
        if not sum(routed.values()):
            out.append(f"{name}: bucket {b}'s capture routed no int8 call ({paths.get(str(b))})")
    after = r.get("paths_after_reps") or {}
    if any(int(k) <= 16 for k in (after.get("passed") or {})):
        out.append(f"{name}: the transient path ran decode-sized rows {after.get('passed')}")
    return out


def int8_closed_form(r, arm) -> dict:
    """REPORTED: P149's capture closed form (3 calls a projection a captured bucket) against what each bucket ran."""
    if arm[1] == "0":
        return {}
    return {b: {"routed": (p or {}).get("routed"), "closed_form": {b: CAPTURE_CALLS * PROJ}}
            for b, p in (r.get("capture_paths") or {}).items()}


def speed_faults(t, r) -> list:
    out = []
    if any(_rate(r, w) is None for w in WORKLOADS):
        out.append(f"{t}: a decode slope is void")
    if r.get("reps") != REPS or (r.get("short"), r.get("long")) != (SHORT, LONG):
        out.append(f"{t}: {r.get('reps')} timed reps at {r.get('short')}/{r.get('long')}, registered {REPS} at {SHORT}/{LONG}")
    for w, b in (("W16", 16), ("W1", 1)):
        rt = (r.get("workloads") or {}).get(w, {}).get("rep_tokens") or {}
        for n in (SHORT, LONG):
            reps = rt.get(str(n)) or []
            if len(reps) != REPS or any(len(x) != b or any(len(s) != n for s in x) for x in reps):
                out.append(f"{t}: {w} at {n} tokens does not carry {REPS} reps of {b} rows of {n} tokens")
    return out


def process_faults(t, r, e4b_sha) -> list:
    arm = t[:2]
    return record_faults(t, r, e4b_sha, arm) + engagement_faults(t, r, arm) + int8_faults(t, r, arm) + speed_faults(t, r)


# ------------------------------------------------------------------------------------------------- the reads --
def invariance(r) -> dict:
    """Rule 3 on one process: per workload and row, timed rep i's 32-token stream against the first 32 tokens of timed rep
    i's 160-token stream, every rep. ``differing``: ``[{"workload", "row", "step", "reps"}]``, the first step where a rep's
    short stream departs from its long one, and the reps where it does."""
    diffs = []
    for w in WORKLOADS:
        rt = r["workloads"][w]["rep_tokens"]
        short, long_ = rt[str(SHORT)], rt[str(LONG)]
        for k in range(len(long_[0])):
            at = {i: first_diff(short[i][k], long_[i][k][:SHORT]) for i in range(len(short))}
            at = {i: x for i, x in at.items() if x is not None}
            if at:
                diffs.append({"workload": w, "row": k, "step": min(at.values()), "reps": sorted(at)})
    return {"invariant": not diffs, "differing": diffs,
            "w16_rows_differing": sorted({d["row"] for d in diffs if d["workload"] == "W16"})}


def repeatability(r) -> dict:
    """REPORTED, never ruling: per workload and length, each row's timed reps against its first (the 160-token reps
    against each other apart from the schedule read). ``differing``: ``[{"workload", "tokens", "row", "step", "reps"}]``."""
    diffs = []
    for w in WORKLOADS:
        for n in (str(SHORT), str(LONG)):
            reps = r["workloads"][w]["rep_tokens"][n]
            for k in range(len(reps[0])):
                at = {i: first_diff(reps[i][k], reps[0][k]) for i in range(1, len(reps))}
                at = {i: x for i, x in at.items() if x is not None}
                if at:
                    diffs.append({"workload": w, "tokens": int(n), "row": k, "step": min(at.values()), "reps": sorted(at)})
    return {"repeatable": not diffs, "differing": diffs}


def cross_w(r) -> dict:
    """Rule 5 on one process: W1's 160-token stream against W16 row 0's, rep by rep; the first differing step or equal."""
    w1, w16 = r["workloads"]["W1"]["rep_tokens"][str(LONG)], r["workloads"]["W16"]["rep_tokens"][str(LONG)]
    steps = [first_diff(a[0], b[0]) for a, b in zip(w1, w16)]
    steps = [x for x in steps if x is not None]
    return {"result": "equal" if not steps else "differs", "first_differing_step": min(steps) if steps else None}


def _clock(r, w):
    h = r["workloads"][w].get("heat") or {}
    return {n: [(h.get(n) or {}).get(e, {}).get("sm_mhz") for e in ("start", "end")] for n in (str(SHORT), str(LONG))}


def price(arms: dict) -> dict:
    """Rule 4, reported: per server, every I/D pair at W1 and W16, p1 and p16 (the min of each server's two pairs,
    unrounded beside the rounded), the clocks of each pair's processes, and the four W16 self-pairs."""
    rates = {t: {w: _rate(arms[t], w) for w in WORKLOADS} for t in TAGS}
    out = {"decode_tok_s": rates, "servers": {}}
    for s, prs in PAIRS.items():
        pairs = {w: {f"{a}/{b}": rates[a][w] / rates[b][w] for a, b in prs} for w in WORKLOADS}
        p1, p16 = min(pairs["W1"].values()), min(pairs["W16"].values())
        out["servers"][s] = {"p1": round(p1, 4), "p16": round(p16, 4), "p1_unrounded": p1, "p16_unrounded": p16,
                             "pairs": {w: {k: round(v, 4) for k, v in p.items()} for w, p in pairs.items()},
                             "clocks": {f"{a}/{b}": {w: {a: _clock(arms[a], w), b: _clock(arms[b], w)} for w in WORKLOADS}
                                        for a, b in prs}}
    out["self_pairs"] = {w: {f"{a}/{b}": round(rates[a][w] / rates[b][w], 4) for a, b in SELF_PAIRS} for w in WORKLOADS}
    return out


def grade(labels: dict, inv: dict, pr: dict, xw: dict) -> list:
    """The predictions table, each HIT or MISS; graded, never ruling."""
    sv = pr["servers"]
    d_arms = {a: any(inv[t]["w16_rows_differing"] for t in TAGS if t[:2] == a) for a in ("D0", "D1")}
    rows = [("W16, both servers: p16 in 0.97-1.00", all(P16_BAND[0] <= sv[s]["p16_unrounded"] <= P16_BAND[1] for s in sv),
             {s: sv[s]["p16"] for s in sv}),
            ("W1, both servers: p1 in 0.60-0.95", all(P1_BAND[0] <= sv[s]["p1_unrounded"] <= P1_BAND[1] for s in sv),
             {s: sv[s]["p1"] for s in sv}),
            ("Both servers SCHEDULE_INVARIANT", all(v == "SCHEDULE_INVARIANT" for v in labels.values()), labels),
            ("Cross-W under I: equal", all(xw[t]["result"] == "equal" for t in TAGS if t[0] == "I"),
             {t: xw[t]["result"] for t in TAGS if t[0] == "I"}),
            ("Both D arms NOT invariant (a W16 row differs, 32 vs 160)", all(d_arms.values()), d_arms)]
    return [{"prediction": p, "graded": "HIT" if ok else "MISS", "read": got} for p, ok, got in rows]


def reduce(arms: dict, e4b_sha: str) -> dict:
    out = {"lane": "P223", "verdict": None, "reasons": [],
           "bars": {"noisy_band": list(NOISY_BAND), "reps": REPS, "min_power_w": MIN_POWER_W, "cc": CC}}
    missing = [t for t in TAGS if t not in arms or arms[t].get("status") != "ok"]
    if missing:
        return {**out, "verdict": "VOID", "reasons": [f"record(s) missing or not ok: {missing}"]}
    void = []
    for t in TAGS:
        void += process_faults(t, arms[t], e4b_sha)
    if len({arms[t].get("prompts_sha256") for t in TAGS}) != 1:
        void.append("the arms read different prompts")
    if void:
        return {**out, "verdict": "VOID", "reasons": void}
    rates = {t: {w: _rate(arms[t], w) for w in WORKLOADS} for t in TAGS}
    noisy = [f"W1 self-pair {a}/{b} = {rates[a]['W1'] / rates[b]['W1']:.4f} outside [{NOISY_BAND[0]}, {NOISY_BAND[1]}]"
             for a, b in SELF_PAIRS if not NOISY_BAND[0] <= rates[a]["W1"] / rates[b]["W1"] <= NOISY_BAND[1]]
    if noisy:
        return {**out, "verdict": "NOISY", "reasons": noisy + ["NOISY is a VOID and a strike: nothing is read"]}
    inv = {t: invariance(arms[t]) for t in TAGS}
    labels, read = {}, {}
    for s, (d, i) in SERVERS.items():
        it = [t for t in TAGS if t[:2] == i]
        bad = {t: inv[t]["differing"] for t in it if not inv[t]["invariant"]}
        labels[s] = "NOT_SCHEDULE_INVARIANT" if bad else "SCHEDULE_INVARIANT"
        read[s] = {"label": labels[s], "i_arm": i, "differing": bad,
                   "d_arm_reported": {t: inv[t] for t in TAGS if t[:2] == d}}
    xw = {t: cross_w(arms[t]) for t in TAGS}
    rp = {t: repeatability(arms[t]) for t in TAGS}
    pr = price(arms)
    out.update(verdict="READ", labels=labels, schedule_invariance=read, price=pr, cross_w=xw, repeatability=rp,
               int8_closed_form={t: int8_closed_form(arms[t], t[:2]) for t in TAGS},
               library_split_plans={t: arms[t].get("library_split_plans") for t in TAGS},
               heat={t: {w: arms[t]["workloads"][w].get("heat") for w in WORKLOADS} for t in TAGS},
               predictions=grade(labels, inv, pr, xw))
    out["reasons"] = [f"{s}: {labels[s]}" + (f" (rows and steps: {read[s]['differing']})" if read[s]["differing"] else "")
                      for s in SERVERS]
    out["reasons"] += [f"{s}: p1 {pr['servers'][s]['p1']}, p16 {pr['servers'][s]['p16']} (reported, never a bar)"
                       for s in SERVERS]
    unrep = {t: v["differing"] for t, v in rp.items() if not v["repeatable"]}
    if unrep:
        out["reasons"].append(f"reported, never ruling: timed reps not repeatable {unrep}")
    return out


def gate_faults(r, tag: str, e4b_sha: str) -> list:
    """One process's own record, read by the runner before the next process (an instrument refusal, never a strike):
    the record checks, the engagement and the int8 counters of its arm."""
    if not r or r.get("status") != "ok":
        return [f"{tag}: the record is missing or not ok"]
    return process_faults(tag, r, e4b_sha)


def _json(path):
    return json.load(open(path)) if os.path.exists(path) else None


def load(d) -> dict:
    return {t: json.load(open(os.path.join(d, f"arm_{t}.json"))) for t in TAGS if os.path.exists(os.path.join(d, f"arm_{t}.json"))}


# ------------------------------------------------------------------------------------------------ self-test --
E = "e" * 40


def _streams(b, n, row_tok=None):
    return [[[(row_tok or {}).get(k, 1)] * n for k in range(b)] for _ in range(REPS)]


def _arm(tag, w1, w16, **kw):
    arm = tag[:2]
    inv, i8 = arm[0] == "I", arm[1] == "1"
    bk = list(ARM_BUCKETS[arm[0]])
    walls = lambda rate, b: {"32": [1.0] * 5, "160": [1.0 + b * 128 / rate] * 5}   # noqa: E731
    wl = {}
    lone = {"W16": 40, "W1": 6 * (SHORT + LONG)}                       # W1: every step a lone row, six passes a length
    for w, rate, b in (("W1", w1, 1), ("W16", w16, 16)):
        lb = "2" if inv else "1"
        wl[w] = {"batch": b, "walls": walls(rate, b), "rep_digests": {"32": ["d"] * 5, "160": ["d"] * 5},
                 "rep_tokens": {"32": _streams(b, 32), "160": _streams(b, 160)}, "decode_tok_s": rate, "heat": {},
                 "decode_calls": {"32": {f"1:{lb}": lone[w] // 2}, "160": {f"1:{lb}": lone[w] // 2}},
                 "graph_stats": {n: {"2": {"replays": 1, "eager_steps": 0, "rows": 0, "pad_rows": lone[w] // 2 if inv else 0}}
                                 for n in ("32", "160")}}
    gs = {str(b): {"replays": 100, "eager_steps": 0, "rows": 0, "pad_rows": 0} for b in bk}
    if inv:
        gs["2"]["pad_rows"] = sum(lone.values())
    plans = {str(b): {"calls": 3, "plan": 12, "n_split": 1} for b in bk} if inv else {}
    r = {"tag": tag, "arm": arm, "status": "ok", "e4b_sha": E, "gnf4_sha": GNF4_SHA, "model": MODEL, "revision": REV,
         "max_seqs": 16, "max_tokens_per_seq": 768, "buckets": bk, "runner_buckets": bk, "fusions": dict(CENSUS),
         "fusion_modes": {k: "auto" for k in FUSION_KNOBS}, "int8_modules": PROJ if i8 else 0,
         "attn_int8": {"projections": PROJ if i8 else 0}, "cc": CC,
         "card": {"name": "NVIDIA GeForce RTX 5090", "enforced_power_limit": 575.0},
         "card_after": {"name": "NVIDIA GeForce RTX 5090", "enforced_power_limit": 575.0},
         "capture_paths": {str(b): ({"routed": {str(b): CAPTURE_CALLS * PROJ}, "passed": {}} if i8 else
                                    {"routed": {}, "passed": {}}) for b in bk},
         "graph_status": {str(b): "graph" for b in bk}, "graph_stats": gs,
         "attn_split_cap": ({"cap": 1, "source": "explicit", "plans": plans} if inv else
                            {"cap": None, "source": "default-off", "plans": {}}),
         "reps": REPS, "short": SHORT, "long": LONG, "prompts_sha256": "p",
         "paths_after_reps": {"routed": {}, "passed": ({"512": 9} if i8 else {})}, "workloads": wl}
    r.update(kw)
    return r


def _fixture(i0_w1=0.8, i0_w16=0.99, i1_w1=0.8, i1_w16=0.99, d_differs=True):
    rate = {"D0": (1.0, 1.0), "I0": (i0_w1, i0_w16), "D1": (1.0, 1.0), "I1": (i1_w1, i1_w16)}
    arms = {t: _arm(t, 100.0 * rate[t[:2]][0], 1000.0 * rate[t[:2]][1]) for t in TAGS}
    if d_differs:                                     # P149's S0a: W16 row 15 parts from its 160-token run at step 24
        for t in TAGS:
            if t[0] == "D":
                arms[t]["workloads"]["W16"]["rep_tokens"]["32"] = _streams(16, 32, {15: 2})
    return arms


def _cases_engagement(cases):
    def void_when(desc, fn, tag):
        arms = _fixture()
        fn(arms[tag])
        v = reduce(arms, E)
        cases.append((desc, v["verdict"] == "VOID", v["reasons"][:3]))
    void_when("an I arm with a bucket-1 replay VOIDs", lambda r: r["graph_stats"].update({"1": {"replays": 3, "eager_steps": 0}}),
              "I1b")
    void_when("an I arm whose plan is 2 at bucket 16 VOIDs",
              lambda r: r["attn_split_cap"]["plans"]["16"].update(n_split=2), "I0a")
    void_when("an I arm without the cap VOIDs", lambda r: r.update(attn_split_cap={"cap": None, "source": "default-off",
                                                                                  "plans": {}}), "I0b")
    void_when("an I arm capped by a per-family default, not the knob, VOIDs",
              lambda r: r["attn_split_cap"].update(source="default-allowlisted"), "I1b")
    void_when("an I arm at the D buckets VOIDs", lambda r: r.update(runner_buckets=[1, 2, 4, 8, 16]), "I1a")
    void_when("a lone row in bucket 4 VOIDs", lambda r: r["workloads"]["W16"]["decode_calls"]["32"].update({"1:4": 1}), "I0a")
    void_when("lone rows that do not match bucket 2's pad rows over the same windows VOID",
              lambda r: r["workloads"]["W1"]["graph_stats"]["160"]["2"].update(pad_rows=575), "I1a")
    void_when("a D arm with no bucket-1 replay VOIDs", lambda r: r["graph_stats"]["1"].update(replays=0), "D0a")
    void_when("a D arm with a split cap VOIDs", lambda r: r.update(attn_split_cap={"cap": 1, "source": "explicit",
                                                                                   "plans": {"1": {"n_split": 1}}}), "D1b")
    void_when("D1 with zero int8 calls at a bucket's capture VOIDs",
              lambda r: r["capture_paths"].update({"4": {"routed": {}, "passed": {}}}), "D1a")
    void_when("I1 built without int8 projections VOIDs", lambda r: r.update(int8_modules=0), "I1b")
    void_when("D0 carrying int8 projections VOIDs", lambda r: r.update(int8_modules=4), "D0b")
    void_when("an eager decode step VOIDs", lambda r: r["graph_stats"]["8"].update(eager_steps=1), "D1a")
    void_when("a bucket not captured as a graph VOIDs", lambda r: r["graph_status"].update({"8": "eager: capture failed"}),
              "I0b")
    void_when("an enforced power limit of 550 W VOIDs", lambda r: r["card"].update(enforced_power_limit=550.0), "I0b")
    void_when("a power limit lowered during the process VOIDs", lambda r: r["card_after"].update(enforced_power_limit=450.0),
              "D0a")
    void_when("a card that is not a 5090 VOIDs", lambda r: r["card"].update(name="NVIDIA GeForce RTX 4090"), "D1b")
    void_when("compute capability 8.9 VOIDs", lambda r: r.update(cc="8.9"), "I1a")
    void_when("a census one fold short VOIDs", lambda r: r["fusions"].update(fuse_t1_glue_n=64), "I0a")
    void_when("four timed reps VOID", lambda r: r.update(reps=4), "D0b")
    void_when("a void slope VOIDs", lambda r: r["workloads"]["W1"].update(decode_tok_s=None), "I1b")


def self_test() -> int:
    cases = []
    v = reduce(_fixture(), E)
    cases.append(("clean records read both servers SCHEDULE_INVARIANT", v["verdict"] == "READ"
                  and v["labels"] == {"S0": "SCHEDULE_INVARIANT", "S1": "SCHEDULE_INVARIANT"}, v["reasons"]))
    cases.append(("the D arms' check is reported beside, row 15 named",
                  v["schedule_invariance"]["S0"]["d_arm_reported"]["D0a"]["w16_rows_differing"] == [15], None))
    cases.append(("p1 and p16 per server", (v["price"]["servers"]["S0"]["p1"], v["price"]["servers"]["S1"]["p16"])
                  == (0.8, 0.99), v["price"]["servers"]))
    cases.append(("every prediction HITs on the clean fixture", [p["graded"] for p in v["predictions"]] == ["HIT"] * 5,
                  v["predictions"]))
    arms = _fixture()
    arms["I1b"]["workloads"]["W16"]["rep_tokens"]["32"][2] = _streams(16, 32, {3: 9})[0]
    v = reduce(arms, E)
    cases.append(("one rep of one I1 row departing names S1 NOT_SCHEDULE_INVARIANT at its row and step",
                  v["labels"] == {"S0": "SCHEDULE_INVARIANT", "S1": "NOT_SCHEDULE_INVARIANT"}
                  and v["schedule_invariance"]["S1"]["differing"] == {"I1b": [{"workload": "W16", "row": 3, "step": 0,
                                                                               "reps": [2]}]}
                  and v["predictions"][2]["graded"] == "MISS", v["reasons"]))
    arms = _fixture()
    arms["I0a"]["workloads"]["W1"]["rep_tokens"]["160"][0][0] = [1] * 31 + [5] + [1] * 128
    v = reduce(arms, E)
    cases.append(("a W1 160-token stream departing at step 31 is NOT_SCHEDULE_INVARIANT at W1 row 0 step 31",
                  v["labels"]["S0"] == "NOT_SCHEDULE_INVARIANT"
                  and v["schedule_invariance"]["S0"]["differing"]["I0a"] == [{"workload": "W1", "row": 0, "step": 31,
                                                                              "reps": [0]}], None))
    cases.append(("the same departure shows in the cross-W check", v["cross_w"]["I0a"] == {"result": "differs",
                  "first_differing_step": 31}, v["cross_w"]["I0a"]))
    arms = _fixture()
    arms["I0a"]["workloads"]["W16"]["rep_tokens"]["160"][1][0] = [1] * 100 + [4] * 60
    v = reduce(arms, E)
    cases.append(("a W16 row-0 departure after step 32 leaves the label and shows in cross-W at step 100",
                  v["labels"]["S0"] == "SCHEDULE_INVARIANT" and v["cross_w"]["I0a"]["first_differing_step"] == 100
                  and v["repeatability"]["I0a"]["differing"] == [{"workload": "W16", "tokens": 160, "row": 0, "step": 100,
                                                                  "reps": [1]}], v["repeatability"]["I0a"]))
    arms = _fixture()
    arms["I0b"]["graph_stats"]["2"]["pad_rows"] += 7                 # padded bucket-2 replays at build, outside any window
    v = reduce(arms, E)
    cases.append(("bucket-2 pad rows from the build, outside every window, do not VOID", v["verdict"] == "READ", v["reasons"][:3]))
    arms = _fixture()
    arms["D0a"]["workloads"]["W16"]["rep_tokens"]["160"][3][5] = [1] * 40 + [6] * 120
    v = reduce(arms, E)
    cases.append(("a 160-token rep departing from its first rep at step 40 is reported as repeatability, never ruling",
                  v["verdict"] == "READ" and v["repeatability"]["D0a"]["differing"]
                  == [{"workload": "W16", "tokens": 160, "row": 5, "step": 40, "reps": [3]}]
                  and v["labels"]["S0"] == "SCHEDULE_INVARIANT" and v["repeatability"]["I0a"]["repeatable"], v["repeatability"]["D0a"]))
    arms = _fixture()
    arms["D0b"]["workloads"]["W1"]["decode_tok_s"] = 98.0
    v = reduce(arms, E)
    cases.append(("a W1 self-pair at 0.98 is NOISY", v["verdict"] == "NOISY", v["reasons"]))
    arms = _fixture()
    arms["I1b"]["workloads"]["W1"]["decode_tok_s"] = 81.3
    v = reduce(arms, E)
    cases.append(("a W1 self-pair at 1.016 is NOISY", v["verdict"] == "NOISY", v["reasons"]))
    arms = _fixture()
    arms["I0b"]["workloads"]["W1"]["decode_tok_s"] = 81.0
    arms["D0b"]["workloads"]["W1"]["decode_tok_s"] = 101.0
    v = reduce(arms, E)
    cases.append(("p1 is the min of the two adjacent pairs", v["verdict"] == "READ"
                  and abs(v["price"]["servers"]["S0"]["p1_unrounded"] - 0.8) < 1e-12
                  and v["price"]["servers"]["S0"]["pairs"]["W1"] == {"I0a/D0a": 0.8, "I0b/D0b": 0.802}, v["price"]["servers"]))
    v = reduce(_fixture(i1_w1=0.97, i1_w16=0.95), E)
    cases.append(("an S1 p1 of 0.97 and p16 of 0.95 MISS both price predictions",
                  [p["graded"] for p in v["predictions"][:2]] == ["MISS", "MISS"], v["predictions"][:2]))
    v = reduce(_fixture(d_differs=False), E)
    cases.append(("D arms matching their own 160-token runs MISS the last prediction",
                  v["predictions"][4]["graded"] == "MISS" and v["verdict"] == "READ", v["predictions"][4]))
    arms = _fixture()
    del arms["D1b"]
    cases.append(("a missing record VOIDs", reduce(arms, E)["verdict"] == "VOID", None))
    arms = _fixture()
    arms["I0b"]["prompts_sha256"] = "q"
    cases.append(("the arms on different prompts VOID", reduce(arms, E)["verdict"] == "VOID", None))
    _cases_engagement(cases)
    g = gate_faults(_fixture()["I0a"], "I0a", E)
    cases.append(("a clean I0a passes its own gate", g == [], g))
    cases.append(("an I record fails a D tag's gate", gate_faults(_fixture()["I0a"], "D0a", E) != [], None))
    cases.append(("a missing record fails the gate", gate_faults(None, "D0a", E) != [], None))
    cases.append(("first_diff: equal, a step, and a length", (first_diff([1, 2], [1, 2]), first_diff([1, 2], [1, 3]),
                  first_diff([1], [1, 2])) == (None, 1, 1), None))
    bad = [(c, extra) for c, ok, extra in cases if not ok]
    print(f"p223_reduce self-test: {len(cases) - len(bad)}/{len(cases)} OK" + (f"; FAILED {bad}" if bad else ""))
    return 1 if bad else 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir")
    ap.add_argument("--out")
    ap.add_argument("--e4b-sha")
    ap.add_argument("--gate", help="one process's record; exit 1 unless its build and engagement pass")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args(argv)
    if a.self_test:
        return self_test()
    if a.gate:
        tag = os.path.basename(a.gate)[len("arm_"):-len(".json")]
        f = gate_faults(_json(a.gate), tag, a.e4b_sha)
        print(f"P223_GATE {tag}: " + ("OK" if not f else "FAULT " + "; ".join(f)[:2000]), flush=True)
        return 1 if f else 0
    v = reduce(load(a.dir), a.e4b_sha)
    json.dump(v, open(a.out, "w"), indent=1, default=str)
    print(f"P223_VERDICT {v['verdict']}: " + "; ".join(str(x) for x in v["reasons"])[:2000], flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
