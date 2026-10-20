#!/usr/bin/env python3
"""p149_reduce.py -- lane P149 box Q+S's registered rule (bench/p149/PREREG-p149.md, "Box Q+S" and Amendment 4).

Records (``p149_box.py``): ``arm_{S0a,S1a,S1b,S0b,S0c,S1c,S1d,S0d}.json`` (ABBA twice, P109's W16 and W1 at five timed
reps; S0a and S1a also carry Q), ``tf_T1.json`` / ``tf_T2.json`` (Amendment 4's token-pinned W16 processes) and, reported
only, ``sane_S0.json`` / ``sane_S1.json``. S0 = the default server; S1 = ``E4B_SERVE_ATTN_INT8=1``.

``--q-only`` (after S1a, before any other process): Q's rule alone. A QUALITY_FAIL there stops the box.

The rule, first rung that applies (the verdict order; host refusals and box K's KERNEL_NO_GAIN are decided before):
  VOID           a record missing or not ok (after a QUALITY_FAIL only S0a and S1a exist, and that is not a VOID); pins,
                 model, card, census and fusion modes; the server's shape; every bucket captured and bucket 1 replayed;
                 ENGAGEMENT: S0 carries no int8 module; S1 swaps exactly 128 projections on 32 attention modules, R2's
                 rope fold patches all 32, and each captured bucket b ran its int8 calls at b rows on the kernel path and
                 none transiently; at Q, W1 replays bucket 1 and W16 bucket 16 with no eager step and no Python decode
                 call, and the transient path ran prefill rows only; Q's windows are P147's (sha256); the arms on
                 different prompts or lengths, or not at five reps; a void slope.
  NOISY          (a VOID, and a strike) any W1 self-pair -- S0b/S0a, S1b/S1a, S0d/S0c, S1d/S1c -- outside [0.985, 1.015].
  QUALITY_FAIL   Q at either gate: argmax agreement < 0.95, or d + 1.645 SE > ln(1 + 0.05 / ppl_S0) (P147's rule, S1
                 against S0 on this box): a PARK of the lever on this path; no speed term rules.
  FUNCTION_FAIL  an arm decodes different tokens from its own arm's first record; one arm's timed reps digest differently.
  FLIP           g1 = min over the four S1/S0 W1 pairs >= 1.03 AND g16 (the same at W16) >= 0.99, unrounded.
  NO_FLIP        otherwise, each failed bar named.

Amendment 4 never moves the verdict. Its label (ROUTING_CLEAR / ROUTING_UNRESOLVED / ROUTING_CONFOUNDED /
QUALIFIER_NOISY / QUALIFIER_VOID) decides only whether a FLIP's default-flip PR proceeds (CLEAR) or is held.

Reported, never ruled: every pair and self-pair, the geomean, heat per timed block, W16 token divergence, prefill-Q,
prefill TTFT, SANE, both streams' routing statistics.

    python p149_reduce.py --dir RUN_DIR --out verdict_p149.json --e4b-sha SHA [--q-only]
    python p149_reduce.py --self-test
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

TAGS = ("S0a", "S1a", "S1b", "S0b", "S0c", "S1c", "S1d", "S0d")      # ABBA twice
PAIRS = (("S1a", "S0a"), ("S1b", "S0b"), ("S1c", "S0c"), ("S1d", "S0d"))
SELF_PAIRS = (("S0b", "S0a"), ("S1b", "S1a"), ("S0d", "S0c"), ("S1d", "S1c"))
NOISY_BAND = (0.985, 1.015)
REPS = 5
WORKLOADS = ("W16", "W1")
BUCKETS = (1, 2, 4, 8, 16)
GNF4_SHA = "a83f6bb3a3a69aef8981fb81b6361bf62b06f6e7"    # v0.49.0: the release that ships K37
MODEL, REV = "mistralai/Mixtral-8x7B-Instruct-v0.1", "eba92302a2861cdc0098cc54bc9f17cb2c47eb61"
CENSUS = {"fuse_qkv_n": 0, "fuse_t1_glue_n": 65, "fuse_t1_glue_r2_n": [32, 32], "fuse_router_epilogue_n": 32}
FUSION_KNOBS = ("E4B_PAGED_FUSE_QKV", "E4B_FUSE_T1_GLUE", "E4B_FUSE_T1_GLUE_R2", "E4B_FUSE_ROUTER_EPI")
LAYERS, PROJ = 32, 128
SERVER = {"max_seqs": 16, "max_tokens_per_seq": 768, "buckets": list(BUCKETS)}
FLIP_G1, NOREG_G16 = 1.03, 0.99
# Q: P147's rule and gates (bench/p147/p147_reduce.py's gate(), S1 against S0)
ARGMAX_MIN, UPPER_Z, PPL_BUDGET = 0.95, 1.645, 0.05
Q_GATES = {"W1": (196, 1, "ae47744e4b30fd6ae427d51aab33f596e46336d257402a6e18e294b5f4a96ffe"),
           "W16": (123, 16, "4ae6aa040b7a92c54536c7a53359bdbd0983e4c47143f0eadae3145307177d2d")}
CONT = 128
# Amendment 4
T_TAGS, T_ORDER = ("T1", "T2"), {"T1": "ABBA", "T2": "BAAB"}


def _rate(r, w):
    return r["workloads"][w]["decode_tok_s"]


def _tok(r, w, n):
    return r["workloads"][w]["tokens"][str(n)]


def divergence(a: list, b: list) -> dict:
    """Rows identical out of the rows, and each row's first divergent step (None when identical): reported only."""
    first = []
    for x, y in zip(a, b):
        i = next((k for k, (p, q) in enumerate(zip(x, y)) if p != q), None)
        first.append(i if i is not None or len(x) == len(y) else min(len(x), len(y)))
    return {"rows_identical": sum(f is None for f in first), "rows": len(first), "first_divergent_step": first}


def ppl(nlls) -> float:
    return math.exp(sum(nlls) / len(nlls))


def gate(on: list, ref: list) -> dict:
    """P147's ``gate()``: S1's per-window records against S0's on the same windows."""
    d = [x["nll"] - y["nll"] for x, y in zip(on, ref)]
    n = len(d)
    mean = sum(d) / n
    sd = math.sqrt(sum((x - mean) ** 2 for x in d) / (n - 1)) if n > 1 else float("inf")
    se = sd / math.sqrt(n)
    agree = [sum(int(p == q) for p, q in zip(x["argmax"], y["argmax"])) / len(y["argmax"]) for x, y in zip(on, ref)]
    ppl_a = ppl([y["nll"] for y in ref])
    bound = math.log1p(PPL_BUDGET / ppl_a)
    upper = mean + UPPER_Z * se
    ok = (sum(agree) / n >= ARGMAX_MIN) and upper <= bound
    return {"windows": n, "mean_d": round(mean, 6), "sd": round(sd, 6), "se": round(se, 6), "upper": round(upper, 6),
            "bound": round(bound, 6), "ppl_S0": round(ppl_a, 4), "argmax_agreement": round(sum(agree) / n, 4),
            "pass": ok}


# ------------------------------------------------------------------------------------------------- the checks --
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
    shape = {"max_seqs": r.get("max_seqs"), "max_tokens_per_seq": r.get("max_tokens_per_seq"), "buckets": r.get("buckets")}
    if shape != SERVER:
        out.append(f"{name}: server {shape}, registered {SERVER}")
    cen = {k: (r.get("fusions") or {}).get(k) for k in CENSUS}
    if cen != CENSUS:
        out.append(f"{name}: census {cen}, the default's {CENSUS}")
    modes = r.get("fusion_modes") or {}
    if any(str(modes.get(k)) not in ("auto", "1") for k in FUSION_KNOBS[1:]):
        out.append(f"{name}: fusion modes {modes}: the default server's folds resolve on for mixtral")
    return out


def capture_faults(name, paths: dict, arm) -> list:
    """Each captured bucket b: S1's int8 calls all at b rows on the kernel path, none transient; S0: none at all."""
    out = []
    for b in BUCKETS:
        p = paths.get(str(b))
        if p is None:
            out.append(f"{name}: no capture record for bucket {b}")
            continue
        routed, passed = p.get("routed") or {}, p.get("passed") or {}
        if arm == "S0":
            if routed or passed:
                out.append(f"{name}: S0's bucket {b} ran int8 calls {p}")
        elif set(routed) != {str(b)} or passed:
            out.append(f"{name}: bucket {b}'s capture ran {p}; registered {b} rows on the kernel path only")
    return out


def int8_faults(name, r, arm) -> list:
    out = []
    a8 = r.get("attn_int8") or {}
    if arm == "S0":
        if r.get("int8_modules") or a8.get("projections"):
            out.append(f"{name}: S0 carries int8 projections ({r.get('int8_modules')}, {a8})")
    else:
        want = {"projections": PROJ, "attention_modules": LAYERS, "r2_rope_fold_attentions": LAYERS}
        got = {k: a8.get(k) for k in want}
        if got != want or r.get("int8_modules") != PROJ:
            out.append(f"{name}: the int8 census {got} ({r.get('int8_modules')} modules), registered {want}")
    out += capture_faults(name, r.get("capture_paths") or {}, arm)
    return out


def speed_faults(t, r) -> list:
    out = []
    g = r.get("graph_status") or {}
    if [g.get(str(b)) for b in BUCKETS] != ["graph"] * len(BUCKETS):
        out.append(f"{t}: graph_status {g}")
    gs1 = (r.get("graph_stats") or {}).get("1") or {}
    if not gs1.get("replays") or gs1.get("eager_steps"):
        out.append(f"{t}: bucket 1 {gs1}: W1 must replay the captured graph")
    if any(_rate(r, w) is None for w in WORKLOADS):
        out.append(f"{t}: a decode slope is void")
    if r.get("reps") != REPS:
        out.append(f"{t}: {r.get('reps')} timed reps, registered {REPS}")
    after = r.get("paths_after_reps") or {}
    if any(int(k) <= 16 for k in (after.get("passed") or {})):
        out.append(f"{t}: the transient path ran decode-sized rows {after.get('passed')}")
    return out


def q_faults(name, q, arm) -> list:
    """Q's engagement: the re-captured buckets as at the build, each gate on its bucket's graph, prefill rows only on
    the transient path, and P147's windows."""
    if not q:
        return [f"{name}: no Q record"]
    rc = q.get("recapture") or {}
    out = []
    if [(rc.get("graph_status") or {}).get(str(b)) for b in BUCKETS] != ["graph"] * len(BUCKETS):
        out.append(f"{name}: Q's re-capture {rc.get('graph_status')}")
    out += capture_faults(f"{name} Q", rc.get("capture_paths") or {}, arm)
    for g, (n, group, sha) in Q_GATES.items():
        r = (q.get("gates") or {}).get(g) or {}
        steps = -(-n // group) * CONT
        if (r.get("windows"), r.get("group"), r.get("sha256")) != (n, group, sha) or len(r.get("per_window") or []) != n:
            out.append(f"{name} Q {g}: {r.get('windows')} windows / group {r.get('group')} / {r.get('sha256')}")
        if r.get("replays") != steps or r.get("eager_steps"):
            out.append(f"{name} Q {g}: {r.get('replays')} replays and {r.get('eager_steps')} eager steps of bucket "
                       f"{r.get('bucket')}, registered {steps} replays")
        f = r.get("forcing") or {}
        if f.get("buf_mismatch") != 0 or f.get("ptr_mismatch") != 0 or f.get("steps") != steps:
            out.append(f"{name} Q {g}: the forcing did not reach the re-captured graph {f}")
        p = r.get("paths") or {}
        if (p.get("routed") or {}) or any(int(k) <= 16 for k in (p.get("passed") or {})):
            out.append(f"{name} Q {g}: Python int8 calls at decode {p}")
        if arm == "S1" and not (p.get("passed") or {}):
            out.append(f"{name} Q {g}: the prefill never took the transient path")
        if any(len(w.get("argmax") or []) != CONT for w in r.get("per_window") or []):
            out.append(f"{name} Q {g}: a window without {CONT} scored positions")
    return out


def q_rule(arms) -> dict:
    s0, s1 = arms["S0a"]["q"]["gates"], arms["S1a"]["q"]["gates"]
    out = {g: gate(s1[g]["per_window"], s0[g]["per_window"]) for g in Q_GATES}
    rep = {}
    for g in Q_GATES:
        p0 = (arms["S0a"]["q"].get("prefill_q") or {}).get(g)
        p1 = (arms["S1a"]["q"].get("prefill_q") or {}).get(g)
        if p0 and p1:
            rep[g] = gate(p1, p0)
    return {"gates": out, "pass": all(v["pass"] for v in out.values()), "prefill_q_reported": rep,
            "prefill_ttft_ms_511": {t: arms[t]["q"].get("prefill_ttft_ms_511", {}).get("median") for t in ("S0a", "S1a")}}


# -------------------------------------------------------------------------------------------- Amendment 4 --
def tf_faults(name, r) -> list:
    """The forcing's engagement in one token-pinned process."""
    out = []
    if not r or r.get("status") != "ok":
        return [f"{name}: missing or not ok"]
    blocks = r.get("blocks") or []
    if [b.get("stream") for b in blocks] != list(T_ORDER[name]):
        out.append(f"{name}: blocks {[b.get('stream') for b in blocks]}, registered {T_ORDER[name]}")
    same = bool(r.get("streams_identical"))
    for i, b in enumerate(blocks):
        c = b.get("counts") or {}
        if c.get("buf_mismatch") != 0 or not b.get("ptr_ok") or not b.get("fed_ok") or not c.get("steps"):
            out.append(f"{name} block {i} ({b.get('stream')}): the forcing did not reach the graph {c}")
        if b.get("stream") == "A" and c.get("argmax_mismatch") != 0:
            out.append(f"{name} block {i}: S0 did not reproduce its own stream ({c.get('argmax_mismatch')} steps)")
        if b.get("stream") == "B" and not same and not c.get("argmax_mismatch"):
            out.append(f"{name} block {i}: B forced nothing S0 would not have decoded")
        if b.get("decode_tok_s") is None or b.get("reps") != REPS:
            out.append(f"{name} block {i}: slope {b.get('decode_tok_s')} at {b.get('reps')} reps")
    return out


def qualifier(tf: dict, g16: float) -> dict:
    """Amendment 4's label, from T1 and T2 (``tf``: tag -> record) and the S rule's g16 (unrounded)."""
    faults = [f for t in T_TAGS for f in tf_faults(t, tf.get(t))]
    m = g16 - NOREG_G16
    if faults:
        return {"label": "QUALIFIER_VOID", "reasons": faults, "m": round(m, 6)}
    r, sp = {}, {}
    for t in T_TAGS:
        bl = tf[t]["blocks"]
        a = [b["decode_tok_s"] for b in bl if b["stream"] == "A"]
        b_ = [b["decode_tok_s"] for b in bl if b["stream"] == "B"]
        r[t] = sum(b_) / sum(a)
        sp[f"{t} A2/A1"], sp[f"{t} B2/B1"] = a[1] / a[0], b_[1] / b_[0]
    gr = sum(r.values()) / len(r)
    rho, e = gr - 1, max(abs(v - 1) for v in sp.values())
    rec = {"g16_routing": round(gr, 6), "r": {k: round(v, 6) for k, v in r.items()}, "rho": round(rho, 6),
           "e": round(e, 6), "m": round(m, 6), "self_pairs": {k: round(v, 6) for k, v in sp.items()},
           "streams_identical": bool(tf["T1"].get("streams_identical")),
           "routing": {t: tf[t].get("routing") for t in T_TAGS if tf[t].get("routing")}}
    if any(not NOISY_BAND[0] <= v <= NOISY_BAND[1] for v in sp.values()):
        rec["label"] = "QUALIFIER_NOISY"
    elif rec["streams_identical"] or rho + e <= m / 2:
        rec["label"] = "ROUTING_CLEAR"
    elif rho - e > m / 2:
        rec["label"] = "ROUTING_CONFOUNDED"
    else:
        rec["label"] = "ROUTING_UNRESOLVED"
    return rec


# ------------------------------------------------------------------------------------------------- the rule --
def reduce_q(arms: dict, e4b_sha: str) -> dict:
    """Q alone (``--q-only``, after S1a): GO / QUALITY_FAIL / VOID."""
    out = {"lane": "P149 box Q+S, Q", "verdict": None, "reasons": []}
    missing = [t for t in ("S0a", "S1a") if t not in arms or arms[t].get("status") != "ok"]
    if missing:
        return {**out, "verdict": "VOID", "reasons": [f"record(s) missing or not ok: {missing}"]}
    void = []
    for t in ("S0a", "S1a"):
        void += record_faults(t, arms[t], e4b_sha, t[:2]) + int8_faults(t, arms[t], t[:2]) + q_faults(t, arms[t].get("q"), t[:2])
    if void:
        return {**out, "verdict": "VOID", "reasons": void}
    q = q_rule(arms)
    out["q"] = q
    if not q["pass"]:
        return {**out, "verdict": "QUALITY_FAIL", "reasons": [f"Q {g}: {v}" for g, v in q["gates"].items() if not v["pass"]]
                + ["a PARK of the lever on this path; the box stops and no speed term rules"]}
    return {**out, "verdict": "GO", "reasons": [f"Q passes at both gates: {q['gates']}"]}


def reduce(arms: dict, tf: dict, e4b_sha: str) -> dict:
    out = {"lane": "P149 box Q+S", "verdict": None, "reasons": [],
           "bars": {"flip_g1": FLIP_G1, "noreg_g16": NOREG_G16, "noisy_band": list(NOISY_BAND), "reps": REPS,
                    "argmax_min": ARGMAX_MIN, "upper_z": UPPER_Z, "ppl_budget": PPL_BUDGET}}
    qv = reduce_q(arms, e4b_sha)
    out["q"] = qv.get("q")
    if qv["verdict"] == "QUALITY_FAIL" and not any(t in arms for t in TAGS[2:]):
        return {**out, "verdict": "QUALITY_FAIL", "reasons": qv["reasons"]}
    missing = [t for t in TAGS if t not in arms or arms[t].get("status") != "ok"]
    if missing:
        return {**out, "verdict": "VOID", "reasons": [f"record(s) missing or not ok: {missing}"]}
    void = list(qv["reasons"]) if qv["verdict"] == "VOID" else []
    for t in TAGS:
        r = arms[t]
        void += record_faults(t, r, e4b_sha, t[:2]) + int8_faults(t, r, t[:2]) + speed_faults(t, r)
    if len({arms[t].get("prompts_sha256") for t in TAGS}) != 1:
        void.append("the arms read different prompts")
    if len({(arms[t].get("short"), arms[t].get("long"), arms[t].get("reps")) for t in TAGS}) != 1:
        void.append("the arms ran different lengths")
    if void:
        return {**out, "verdict": "VOID", "reasons": void}
    S0a = arms["S0a"]
    lens = (S0a["short"], S0a["long"])
    rates = {t: {w: _rate(arms[t], w) for w in WORKLOADS} for t in TAGS}
    pairs = {w: {f"{a}/{b}": rates[a][w] / rates[b][w] for a, b in PAIRS} for w in WORKLOADS}
    selfp = {w: {f"{a}/{b}": rates[a][w] / rates[b][w] for a, b in SELF_PAIRS} for w in WORKLOADS}
    g1_min, g16_min = min(pairs["W1"].values()), min(pairs["W16"].values())     # the rule reads these, unrounded
    out["speed"] = {"decode_tok_s": rates, "g1": round(g1_min, 4), "g16": round(g16_min, 4),
                    "g1_unrounded": g1_min, "g16_unrounded": g16_min,
                    "pairs": {w: {k: round(v, 4) for k, v in p.items()} for w, p in pairs.items()},
                    "geomean": {w: round(math.prod(p.values()) ** (1 / len(p)), 4) for w, p in pairs.items()},
                    "self_pairs": {w: {k: round(v, 4) for k, v in p.items()} for w, p in selfp.items()},
                    "heat": {t: {w: arms[t]["workloads"][w].get("heat") for w in WORKLOADS} for t in TAGS},
                    "w16_s1_vs_s0": {f"{a}/{b} {n}": divergence(_tok(arms[a], "W16", n), _tok(arms[b], "W16", n))
                                     for a, b in PAIRS for n in lens}}
    out["routing_qualifier"] = qualifier(tf, g16_min)
    noisy = [f"W1 self-pair {k} = {v:.4f} outside [{NOISY_BAND[0]}, {NOISY_BAND[1]}]"
             for k, v in selfp["W1"].items() if not NOISY_BAND[0] <= v <= NOISY_BAND[1]]
    if noisy:
        return {**out, "verdict": "NOISY", "reasons": noisy + ["NOISY is a VOID and a strike: no flip is read"]}
    if qv["verdict"] == "QUALITY_FAIL":
        return {**out, "verdict": "QUALITY_FAIL", "reasons": qv["reasons"]}
    fn = []
    for t in TAGS:
        first = "S0a" if t.startswith("S0") else "S1a"
        for w in WORKLOADS:
            for n in lens:
                if t != first and _tok(arms[t], w, n) != _tok(arms[first], w, n):
                    fn.append(f"{t} != {first} on {w} at {n} tokens")
            for n, ds in arms[t]["workloads"][w]["rep_digests"].items():
                if len(set(ds)) != 1:
                    fn.append(f"{t} {w} at {n} tokens: timed reps differ")
    if fn:
        out.update(verdict="FUNCTION_FAIL", reasons=fn)
    elif g1_min >= FLIP_G1 and g16_min >= NOREG_G16:
        out.update(verdict="FLIP", reasons=[f"g1 {g1_min:.6f} (the min of four pairs) >= {FLIP_G1} and g16 "
                                            f"{g16_min:.6f} >= {NOREG_G16}"])
    else:
        why = [f"g1 {g1_min:.6f} (the min of four pairs) < {FLIP_G1}"] if g1_min < FLIP_G1 else []
        why += [f"g16 {g16_min:.6f} < {NOREG_G16} (W16's no-regression line)"] if g16_min < NOREG_G16 else []
        out.update(verdict="NO_FLIP", reasons=why + ["the lever stays opt-in"])
    lab = out["routing_qualifier"]["label"]
    out["default_flip_pr"] = (("proceeds" if lab == "ROUTING_CLEAR" else f"held ({lab})")
                              if out["verdict"] == "FLIP" else "n/a")
    return out


def load(d) -> tuple:
    arms = {t: json.load(open(os.path.join(d, f"arm_{t}.json"))) for t in TAGS if os.path.exists(os.path.join(d, f"arm_{t}.json"))}
    tf = {t: json.load(open(os.path.join(d, f"tf_{t}.json"))) for t in T_TAGS if os.path.exists(os.path.join(d, f"tf_{t}.json"))}
    return arms, tf


# ------------------------------------------------------------------------------------------------ self-test --
E = "e" * 40


def _paths(arm, bucket_rows=True):
    return {str(b): ({"routed": {}, "passed": {}} if arm == "S0" else
                     {"routed": {str(b): PROJ * 3}, "passed": {}}) for b in BUCKETS} if bucket_rows else {}


def _qrec(arm, nll=1.0, flip=0):
    gates = {}
    for g, (n, group, sha) in Q_GATES.items():
        per = [{"nll": nll + 0.001 * (i % 7) + (0.0002 if arm == "S1" else 0), "argmax": [i + (k < flip) for k in range(CONT)]}
               for i in range(n)]
        gates[g] = {"windows": n, "group": group, "sha256": sha, "bucket": group, "replays": -(-n // group) * CONT,
                    "eager_steps": 0, "paths": {"routed": {}, "passed": ({"511": n * PROJ} if arm == "S1" else {})},
                    "forcing": {"buf_mismatch": 0, "ptr_mismatch": 0, "steps": -(-n // group) * CONT}, "per_window": per}
    return {"recapture": {"graph_status": {str(b): "graph" for b in BUCKETS}, "capture_paths": _paths(arm)},
            "gates": gates, "prefill_ttft_ms_511": {"median": 40.0}}


def _arm(tag, w1, w16, *, toks=None, q=True, **kw):
    arm = tag[:2]
    walls = lambda rate, b: {"32": [1.0] * 5, "160": [1.0 + b * 128 / rate] * 5}   # noqa: E731
    wl = {}
    for w, rate, b in (("W1", w1, 1), ("W16", w16, 16)):
        wl[w] = {"batch": b, "walls": walls(rate, b), "rep_digests": {"32": ["d"] * 5, "160": ["d"] * 5},
                 "tokens": toks or {"32": [[1] * 32] * b, "160": [[1] * 160] * b}, "decode_tok_s": rate, "heat": {}}
    r = {"tag": tag, "arm": arm, "status": "ok", "e4b_sha": E, "gnf4_sha": GNF4_SHA, "model": MODEL, "revision": REV,
         "max_seqs": 16, "max_tokens_per_seq": 768, "buckets": list(BUCKETS), "fusions": dict(CENSUS),
         "fusion_modes": {k: "auto" for k in FUSION_KNOBS}, "int8_modules": PROJ if arm == "S1" else 0,
         "attn_int8": ({"projections": PROJ, "attention_modules": LAYERS, "r2_rope_fold_attentions": LAYERS}
                       if arm == "S1" else {"projections": 0}),
         "capture_paths": _paths(arm), "graph_status": {str(b): "graph" for b in BUCKETS},
         "graph_stats": {"1": {"replays": 1000, "eager_steps": 0}}, "reps": REPS, "short": 32, "long": 160,
         "prompts_sha256": "p", "paths_after_reps": {"routed": {}, "passed": ({"512": 9} if arm == "S1" else {})},
         "workloads": wl}
    if q and tag in ("S0a", "S1a"):
        r["q"] = _qrec(arm)
    r.update(kw)
    return r


def _tf(tag, a=(100.0, 100.0), b=(100.0, 100.0), same=False, **kw):
    order = T_ORDER[tag]
    vals = {"A": list(a), "B": list(b)}
    blocks = []
    for s in order:
        blocks.append({"stream": s, "decode_tok_s": vals[s].pop(0), "reps": REPS, "ptr_ok": True, "fed_ok": True,
                       "counts": {"buf_mismatch": 0, "argmax_mismatch": 0 if (s == "A" or same) else 7, "steps": 2544}})
    return {"tag": tag, "status": "ok", "blocks": blocks, "streams_identical": same, **kw}


def _fixture(s1_w1=1.05, s1_w16=1.0, **kw):
    arms = {t: _arm(t, 100.0 * (s1_w1 if t[:2] == "S1" else 1), 1000.0 * (s1_w16 if t[:2] == "S1" else 1)) for t in TAGS}
    return arms


def self_test() -> int:
    cases = []
    tf = {t: _tf(t) for t in T_TAGS}
    v = reduce(_fixture(), tf, E)
    cases.append(("a clean +5 % W1 with g16 1.0 FLIPs", v["verdict"] == "FLIP", v["reasons"]))
    cases.append(("routing-neutral streams CLEAR the default flip", v["routing_qualifier"]["label"] == "ROUTING_CLEAR"
                  and v["default_flip_pr"] == "proceeds", v.get("routing_qualifier")))
    v = reduce(_fixture(s1_w1=1.02), tf, E)
    cases.append(("+2 % W1 is NO_FLIP", v["verdict"] == "NO_FLIP" and v["default_flip_pr"] == "n/a", v["reasons"]))
    arms = _fixture()
    for t in ("S0c", "S0d"):
        arms[t]["workloads"]["W1"]["decode_tok_s"] = 103.0           # S1c/S0c = S1d/S0d = 1.019; every self-pair 1.0
    v = reduce(arms, tf, E)
    cases.append(("the min of the four pairs rules: two low pairs are NO_FLIP", v["verdict"] == "NO_FLIP", v["reasons"]))
    v = reduce(_fixture(s1_w16=0.98), tf, E)
    cases.append(("g16 0.98 is NO_FLIP", v["verdict"] == "NO_FLIP" and any("g16" in x for x in v["reasons"]), v["reasons"]))
    arms = _fixture()
    arms["S0b"]["workloads"]["W1"]["decode_tok_s"] = 98.0
    v = reduce(arms, tf, E)
    cases.append(("a W1 self-pair at 0.98 is NOISY", v["verdict"] == "NOISY", v["reasons"]))
    arms = _fixture()
    arms["S1a"]["q"]["gates"]["W16"]["per_window"] = [dict(w, argmax=[-1] * CONT) for w in arms["S1a"]["q"]["gates"]["W16"]["per_window"]]
    v = reduce(arms, tf, E)
    cases.append(("argmax agreement 0 at W16 is QUALITY_FAIL", v["verdict"] == "QUALITY_FAIL", v["reasons"]))
    only = {t: arms[t] for t in ("S0a", "S1a")}
    cases.append(("--q-only reads QUALITY_FAIL after S1a", reduce_q(only, E)["verdict"] == "QUALITY_FAIL", None))
    cases.append(("a QUALITY_FAIL with six arms unrun is not a VOID", reduce(only, {}, E)["verdict"] == "QUALITY_FAIL", None))
    cases.append(("--q-only on a clean pair is GO", reduce_q({t: _fixture()[t] for t in ("S0a", "S1a")}, E)["verdict"] == "GO",
                  None))
    arms = _fixture()
    for w in arms["S1a"]["q"]["gates"].values():
        for x in w["per_window"]:
            x["nll"] += 0.05
    cases.append(("an NLL shift of 0.05 nats fails the ppl bound", reduce(arms, tf, E)["verdict"] == "QUALITY_FAIL", None))
    arms = _fixture()
    arms["S1c"]["workloads"]["W16"]["tokens"] = {"32": [[2] * 32] * 16, "160": [[2] * 160] * 16}
    cases.append(("S1c decoding other tokens than S1a is FUNCTION_FAIL", reduce(arms, tf, E)["verdict"] == "FUNCTION_FAIL", None))
    arms = _fixture()
    arms["S1b"]["capture_paths"]["16"] = {"routed": {"16": 384}, "passed": {"16": 1}}
    cases.append(("a transient call at bucket 16's capture VOIDs", reduce(arms, tf, E)["verdict"] == "VOID", None))
    arms = _fixture()
    arms["S0c"]["int8_modules"] = 4
    cases.append(("S0 with int8 modules VOIDs", reduce(arms, tf, E)["verdict"] == "VOID", None))
    arms = _fixture()
    arms["S1d"]["attn_int8"]["r2_rope_fold_attentions"] = 31
    cases.append(("R2 on 31 of 32 attention modules VOIDs", reduce(arms, tf, E)["verdict"] == "VOID", None))
    arms = _fixture()
    arms["S1a"]["q"]["gates"]["W1"]["eager_steps"] = 3
    cases.append(("an eager Q step VOIDs", reduce(arms, tf, E)["verdict"] == "VOID", None))
    arms = _fixture()
    arms["S0a"]["q"]["gates"]["W16"]["forcing"]["buf_mismatch"] = 1
    cases.append(("a Q step whose graph buffer missed the forced token VOIDs", reduce(arms, tf, E)["verdict"] == "VOID", None))
    arms = _fixture()
    arms["S1a"]["q"]["gates"]["W1"]["sha256"] = "0"
    cases.append(("windows that are not P147's VOID", reduce(arms, tf, E)["verdict"] == "VOID", None))
    del arms["S1d"]
    arms["S1a"]["q"]["gates"]["W1"]["sha256"] = Q_GATES["W1"][2]
    cases.append(("a missing arm after Q passed VOIDs", reduce(arms, tf, E)["verdict"] == "VOID", None))
    # Amendment 4
    g16 = 1.0
    cl = qualifier({t: _tf(t, b=(101.0, 101.0)) for t in T_TAGS}, 1.03)
    cases.append(("rho 0.01 against m 0.04 at e 0 is CLEAR", cl["label"] == "ROUTING_CLEAR", cl))
    cf = qualifier({t: _tf(t, b=(102.0, 102.0)) for t in T_TAGS}, g16)
    cases.append(("rho 0.02 against m 0.01 is CONFOUNDED", cf["label"] == "ROUTING_CONFOUNDED", cf))
    un = qualifier({t: _tf(t, a=(100.0, 100.4), b=(100.6, 100.6)) for t in T_TAGS}, g16)
    cases.append(("rho near m/2 within e is UNRESOLVED", un["label"] == "ROUTING_UNRESOLVED", un))
    nz = qualifier({t: _tf(t, a=(100.0, 98.0)) for t in T_TAGS}, 1.03)
    cases.append(("a token-pinned self-pair at 0.98 is QUALIFIER_NOISY", nz["label"] == "QUALIFIER_NOISY", nz))
    neg = qualifier({t: _tf(t, b=(97.0, 97.0)) for t in T_TAGS}, 1.0)
    cases.append(("a negative rho (S1's tokens dearer) CLEARs", neg["label"] == "ROUTING_CLEAR", neg))
    thin = qualifier({t: _tf(t, a=(100.0, 100.3), b=(100.2, 100.2)) for t in T_TAGS}, 0.995)
    cases.append(("e > m/2 with rho >= 0 cannot CLEAR", thin["rho"] >= 0 and thin["label"] != "ROUTING_CLEAR", thin))
    dear = qualifier({t: _tf(t, a=(100.0, 100.3), b=(99.0, 99.0)) for t in T_TAGS}, 0.995)
    cases.append(("e > m/2 with rho < 0 can CLEAR", dear["label"] == "ROUTING_CLEAR", dear))
    bad = {t: _tf(t) for t in T_TAGS}
    bad["T1"]["blocks"][1]["counts"]["buf_mismatch"] = 3
    cases.append(("a forced token missing from the graph's buffer is QUALIFIER_VOID", qualifier(bad, 1.03)["label"] == "QUALIFIER_VOID", None))
    bad = {t: _tf(t) for t in T_TAGS}
    bad["T2"]["blocks"][0]["counts"]["argmax_mismatch"] = 0
    cases.append(("B forcing nothing new is QUALIFIER_VOID", qualifier(bad, 1.03)["label"] == "QUALIFIER_VOID", None))
    same = qualifier({t: _tf(t, same=True) for t in T_TAGS}, 1.0)
    cases.append(("identical streams CLEAR by construction", same["label"] == "ROUTING_CLEAR", same))
    v = reduce(_fixture(), {t: _tf(t, b=(102.0, 102.0)) for t in T_TAGS}, E)
    cases.append(("a FLIP stands while its default-flip PR is held", v["verdict"] == "FLIP"
                  and v["default_flip_pr"].startswith("held (ROUTING_CONFOUNDED"), v.get("default_flip_pr")))
    v = reduce(_fixture(), {}, E)
    cases.append(("no T records: the FLIP stands, the PR is held", v["verdict"] == "FLIP"
                  and v["default_flip_pr"] == "held (QUALIFIER_VOID)", v.get("default_flip_pr")))
    cases.append(("P147's gate: the bound is ln(1 + 0.05 / ppl)", abs(gate([{"nll": 1.0, "argmax": [1]}] * 3,
                  [{"nll": 1.0, "argmax": [1]}] * 3)["bound"] - round(math.log1p(0.05 / math.e), 6)) < 1e-9, None))
    bad = [(c, extra) for c, ok, extra in [(c[0], c[1], c[2]) for c in cases] if not ok]
    print(f"p149_reduce self-test: {len(cases) - len(bad)}/{len(cases)} OK" + (f"; FAILED {bad}" if bad else ""))
    return 1 if bad else 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir")
    ap.add_argument("--out")
    ap.add_argument("--e4b-sha")
    ap.add_argument("--q-only", action="store_true")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args(argv)
    if a.self_test:
        return self_test()
    arms, tf = load(a.dir)
    v = reduce_q(arms, a.e4b_sha) if a.q_only else reduce(arms, tf, a.e4b_sha)
    json.dump(v, open(a.out, "w"), indent=1, default=str)
    print(f"P149_VERDICT {v['verdict']}: " + "; ".join(str(x) for x in v["reasons"])[:2000], flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
