#!/usr/bin/env python3
"""sd2_reduce.py -- lane SD2 (e4b#1313): the proof's and the read's verdicts (bench/sd2/PREREG-sd2.md, Amendments 1-3).

`--prove prove.json --gpu-tests gpu_tests.json [--rule a2|a1]` -> `verdict_prove.json`. **PROVED** when every item
holds; otherwise **FAILED** with every failing item named. A mutant the addressing gate cannot see is named
GATE_TOO_WEAK:<m>, and the registered rule is that the gate is tightened before the read.

**The two addressing rules.**
- `a1` (Amendment 1, `sd2-prove-2`'s registered verdict): a (row, k) passes at row agreement AND continuation
  agreement >= 0.90; a mutant is caught when either falls below 0.90.
- `a2` (Amendment 2, the default): the LOGIT gate decides. A (row, k) passes when the largest per-row mean |Δ log p| of
  the oracle's token (verify against the sequential T == 1 oracle) is <= 0.25 nats; a mutant is caught when its largest
  exceeds it. Argmax agreement is reported, not gated: on chat it sits near 0.90 on the real build (C0 k=3
  continuations 0.906 on `sd2-prove-2`), where near-ties flip. The bound is CALIBRATED on `sd2-prove-2`: the real build
  reached at most 0.037 and every mutant had a row at 0.89 or more ((c), the RoPE shift, 1.18 on row 0), so 0.25 sits
  about 7x from each.

The other items (both rules):
- `census`: buckets 1, 2, 3, 4, 8 and 16 each "graph"; post-verify graphs 2, 3 and 4 each "graph"; three hooks;
- `capture`: every verify bucket's replay bitwise its padded eager step;
- `draft`: in-engine drafts against chain_at at >= 0.99 of drafted ids;
- `transition`: one drop counted, both requests at their lengths, no mirror mismatch, no state left;
- `gpu_tests`: the target's GPU tests exited 0 with none skipped for want of the card.

**The read** (Amendment 3). `--gate-v DIR` (after stage V, on the box): VOID when V0 fails (rule a2 on R1..R3 and
C1..C3, the six mutants, the draft, the transition, the GPU tests) or V1 is missing; E_ONLY when VERIFY_COST_REFUTES
fires without V_NOISY (the stop rule: E runs, Q is skipped); ALL otherwise. `--read DIR --expect-e4b --expect-gnf4
--expect-rev [--expect-w1]` -> `verdict_read.json`, the registered rule, first that applies: VOID, VERIFY_COST_REFUTES
(with the V_NOISY flag and the conflict clause), NOISY, QUALITY_FAIL, FASTER, NOT_FASTER. τ_SD1 and SD1's modelled S
are sd1-5090-2's (`bench/sd1/receipts/sd1-5090-2/verdict.json`); Q's bar is P110's, with P115's statistics.

`--self-test` runs every proof verdict under both rules, `sd2-prove-2`'s own numbers (FAILED [GATE_TOO_WEAK:c] under
a1, PROVED under a2), and every read verdict and VOID item.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import sys

BAR_ADDRESS, BAR_DRAFT = 0.90, 0.99
BOUND_DLOGP = 0.25          # Amendment 2: nats; calibrated on sd2-prove-2 (real <= 0.037, every mutant >= 0.89)
RULES = ("a2", "a1")
BUCKETS = ("1", "2", "3", "4", "8", "16")
POSTS = ("2", "3", "4")


def logit_max(x):
    """The largest per-row mean |Δ log p| of a (row, k) or mutant record, or None when it carries none."""
    v = [d for d in ((x or {}).get("mean_abs_dlogp") or []) if d is not None]
    return max(v) if v else None


def judge(prove: dict, gpu: dict, rule: str = "a2", rows=("R0", "C0"), mutant_keys=("a", "b", "c")) -> dict:
    """The proof's items; ``rows`` and ``mutant_keys`` default to the proof's, and the read's V0 passes its own."""
    if rule not in RULES:
        raise ValueError(f"rule {rule!r}: expected one of {RULES}")
    fails = []
    c = prove.get("census") or {}
    for b in BUCKETS:
        if (c.get("graph_status") or {}).get(b) != "graph":
            fails.append(f"census:bucket_{b}")
    for n in POSTS:
        if (c.get("post_graphs") or {}).get(n) != "graph":
            fails.append(f"census:post_{n}")
    if ((c.get("spec_build") or {}).get("hooks")) != 3:
        fails.append("census:hooks")
    cap = prove.get("capture_bitwise") or {}
    for n in POSTS:
        if cap.get(n) is not True:
            fails.append(f"capture:{n}")
    addr = prove.get("addressing") or {}
    want = [f"{r}_k{k}" for r in rows for k in (1, 2, 3)]
    for key in want:
        x = addr.get(key)
        if rule == "a1":
            ok = bool(x) and x.get("rows_agree", 0) >= BAR_ADDRESS and x.get("cont_agree", 0) >= BAR_ADDRESS
        else:
            lm = logit_max(x)
            ok = lm is not None and lm <= BOUND_DLOGP
        if not ok:
            fails.append(f"addressing:{key}")
    muts = prove.get("mutants") or {}
    for m in mutant_keys:
        x = muts.get(m)
        if not x:
            fails.append(f"mutant:{m}:missing")
        elif rule == "a1":
            if x.get("rows_agree", 1.0) >= BAR_ADDRESS and x.get("cont_agree", 1.0) >= BAR_ADDRESS:
                fails.append(f"GATE_TOO_WEAK:{m}")
        else:
            lm = logit_max(x)
            if lm is None:
                fails.append(f"mutant:{m}:missing")
            elif lm <= BOUND_DLOGP:
                fails.append(f"GATE_TOO_WEAK:{m}")
    d = prove.get("draft") or {}
    if d.get("of", 0) <= 0 or d.get("agree", 0) < BAR_DRAFT:
        fails.append("draft")
    t = prove.get("transition") or {}
    if not (t.get("dropped_batched") == 1 and t.get("spec_steps_alone", 0) > 0 and t.get("mirror_mismatches") == 0
            and t.get("state_left") == 0 and [t.get("len_a"), t.get("len_b")] == t.get("want")):
        fails.append("transition")
    if gpu.get("rc") != 0 or gpu.get("passed", 0) <= 0 or gpu.get("skipped_for_card", 1) != 0:
        fails.append("gpu_tests")
    fields = ("rows_agree", "cont_agree", "mean_abs_dlogp")
    return {"verdict": "PROVED" if not fails else "FAILED", "fails": fails, "rule": rule,
            "bound_dlogp": BOUND_DLOGP if rule == "a2" else None,
            "addressing": {k: dict({kk: addr[k].get(kk) for kk in fields}, logit_max=logit_max(addr[k]))
                           for k in want if k in addr},
            "mutants": {m: dict({kk: (muts.get(m) or {}).get(kk) for kk in fields}, logit_max=logit_max(muts.get(m)))
                        for m in mutant_keys},
            "draft": d, "transition": t}


def _good() -> tuple:
    prove = {"census": {"graph_status": {b: "graph" for b in BUCKETS}, "post_graphs": {n: "graph" for n in POSTS},
                        "spec_build": {"hooks": 3}},
             "capture_bitwise": {n: True for n in POSTS},
             "addressing": {f"{r}_k{k}": {"rows_agree": 0.98, "cont_agree": 0.97, "mean_abs_dlogp": [0.01] * (k + 1)}
                            for r in ("R0", "C0") for k in (1, 2, 3)},
             "mutants": {m: {"rows_agree": 0.3, "cont_agree": 0.3, "mean_abs_dlogp": [0.02, 2.0, 2.0, 2.0]}
                         for m in ("a", "b", "c")},
             "draft": {"agree": 0.995, "same": 764, "of": 768},
             "transition": {"spec_steps_alone": 4, "dropped_batched": 1, "len_a": 64, "len_b": 16, "want": [64, 16],
                            "mirror_mismatches": 0, "state_left": 0}}
    return prove, {"rc": 0, "passed": 40, "skipped_for_card": 0}


def _sd2_prove_2() -> dict:
    """sd2-prove-2's addressing and mutant records (store 72a69894), the calibration set."""
    p, _ = _good()
    p["addressing"] = {
        "R0_k1": {"rows_agree": 1.0, "cont_agree": 1.0, "mean_abs_dlogp": [0.002939, 0.002148]},
        "R0_k2": {"rows_agree": 1.0, "cont_agree": 1.0, "mean_abs_dlogp": [0.006527, 0.00338, 0.003902]},
        "R0_k3": {"rows_agree": 1.0, "cont_agree": 1.0, "mean_abs_dlogp": [0.005385, 0.002219, 0.000969, 0.002394]},
        "C0_k1": {"rows_agree": 1.0, "cont_agree": 1.0, "mean_abs_dlogp": [0.019688, 0.026519]},
        "C0_k2": {"rows_agree": 1.0, "cont_agree": 0.921875, "mean_abs_dlogp": [0.016608, 0.013777, 0.033104]},
        "C0_k3": {"rows_agree": 0.984375, "cont_agree": 0.90625, "mean_abs_dlogp": [0.01489, 0.007156, 0.006616, 0.036915]}}
    p["mutants"] = {
        "a": {"rows_agree": 0.6875, "cont_agree": 0.65625, "mean_abs_dlogp": [0.023935, 2.149183, 2.789426, 2.469237]},
        "b": {"rows_agree": 0.5625, "cont_agree": 0.96875, "mean_abs_dlogp": [0.889273, 5.01826, 5.342677, 0.01232]},
        "c": {"rows_agree": 0.9375, "cont_agree": 0.90625, "mean_abs_dlogp": [1.179762, 0.005705, 0.010378, 0.011125]}}
    p["draft"] = {"agree": 766 / 768, "same": 766, "of": 768}
    return p


def self_test() -> int:
    bad = []
    p, g = _good()
    for rule in RULES:
        if judge(p, g, rule)["verdict"] != "PROVED":
            bad.append(f"good under {rule} -> {judge(p, g, rule)['fails']}")

    def case(edit, want, rule="a2", gpu_edit=None):
        p2, g2 = copy.deepcopy(p), dict(g)
        edit(p2)
        if gpu_edit:
            gpu_edit(g2)
        got = judge(p2, g2, rule)
        if got["verdict"] != "FAILED" or want not in got["fails"]:
            bad.append(f"{rule} {want}: {got['fails']}")

    def proved(edit, rule="a2", why=""):
        p2 = copy.deepcopy(p)
        edit(p2)
        got = judge(p2, g, rule)
        if got["verdict"] != "PROVED":
            bad.append(f"{rule} should prove ({why}): {got['fails']}")

    for rule in RULES:
        case(lambda x: x["census"]["graph_status"].update({"3": "eager: X"}), "census:bucket_3", rule)
        case(lambda x: x["census"]["post_graphs"].pop("4"), "census:post_4", rule)
        case(lambda x: x["census"]["spec_build"].update({"hooks": 2}), "census:hooks", rule)
        case(lambda x: x["capture_bitwise"].update({"2": False}), "capture:2", rule)
        case(lambda x: x["addressing"].pop("C0_k1"), "addressing:C0_k1", rule)
        case(lambda x: x["mutants"].pop("b"), "mutant:b:missing", rule)
        case(lambda x: x["draft"].update({"agree": 0.98}), "draft", rule)
        case(lambda x: x["transition"].update({"dropped_batched": 0}), "transition", rule)
        case(lambda x: x["transition"].update({"len_a": 63}), "transition", rule)
        case(lambda x: x["transition"].update({"mirror_mismatches": 2}), "transition", rule)
        case(lambda x: None, "gpu_tests", rule, lambda gg: gg.update({"rc": 1}))
        case(lambda x: None, "gpu_tests", rule, lambda gg: gg.update({"skipped_for_card": 3}))
    # a1: the agreement gate
    case(lambda x: x["addressing"]["R0_k2"].update({"rows_agree": 0.89}), "addressing:R0_k2", "a1")
    case(lambda x: x["addressing"]["C0_k3"].update({"cont_agree": 0.5}), "addressing:C0_k3", "a1")
    case(lambda x: x["mutants"]["c"].update({"rows_agree": 0.95, "cont_agree": 0.95}), "GATE_TOO_WEAK:c", "a1")
    proved(lambda x: x["mutants"]["c"].update({"rows_agree": 0.97, "cont_agree": 0.40}), "a1", "caught by continuations")
    # a2: the logit gate decides; agreement is reported only
    case(lambda x: x["addressing"]["R0_k2"].update({"mean_abs_dlogp": [0.01, 0.30, 0.01]}), "addressing:R0_k2")
    case(lambda x: x["addressing"]["C0_k1"].update({"mean_abs_dlogp": []}), "addressing:C0_k1")
    case(lambda x: x["mutants"]["c"].update({"mean_abs_dlogp": [0.2, 0.01, 0.01, 0.01]}), "GATE_TOO_WEAK:c")
    case(lambda x: x["mutants"]["a"].update({"mean_abs_dlogp": [None]}), "mutant:a:missing")
    proved(lambda x: x["addressing"]["C0_k3"].update({"rows_agree": 0.85, "cont_agree": 0.80}), "a2",
           "agreement below 0.90 with the logits inside the bound")
    proved(lambda x: x["mutants"]["c"].update({"rows_agree": 0.99, "cont_agree": 0.99,
                                               "mean_abs_dlogp": [1.18, 0.0, 0.0, 0.0]}), "a2", "caught by logits")
    # the calibration set
    cal = _sd2_prove_2()
    v1, v2 = judge(cal, g, "a1"), judge(cal, g, "a2")
    if v1["verdict"] != "FAILED" or v1["fails"] != ["GATE_TOO_WEAK:c"]:
        bad.append(f"sd2-prove-2 under a1: {v1['fails']}")
    if v2["verdict"] != "PROVED":
        bad.append(f"sd2-prove-2 under a2: {v2['fails']}")
    try:
        judge(p, g, "a3")
        bad.append("an unknown rule was accepted")
    except ValueError:
        pass
    bad += read_self_test()
    if bad:
        print("sd2_reduce self-test FAILED:", bad)
        return 1
    print("sd2_reduce self-test OK (39 proof cases: both rules, and sd2-prove-2's calibration; 37 read cases: every verdict, every VOID item, the stage gate)")
    return 0


# ===================================================================================== the read (Amendment 3) ==

HEAD_SHA256 = "d2d6e2e63e09dc755053ae5c98cdececae3611ae5e202d4fa5411126dd3b1dfa"
WORKLOADS = ("R", "C-think", "C-nothink")
ARMS = ("OFF-a", "ON1-a", "ON2-a", "ON2-b", "ON1-b", "OFF-b")
#: sd1-5090-2's verdict.json, routes.eagle3.<w>.<k>: tau MEASURED, S_independent MODELLED (a test pins both)
TAU_SD1 = {"R": {1: 1.5353, 2: 1.8094, 3: 1.9479}, "C-think": {1: 1.7157, 2: 2.1912, 3: 2.4939},
           "C-nothink": {1: 1.6465, 2: 2.0339, 3: 2.2604}}
S_SD1 = {"R": {1: 1.0964, 2: 1.0163, 3: 0.9092}, "C-think": {1: 1.2253, 2: 1.2303, 3: 1.1635},
         "C-nothink": {1: 1.1758, 2: 1.1423, 3: 1.0550}}
READ_ROWS = ("R1", "R2", "R3", "C1", "C2", "C3")                      # V0's gate rows (Amendment 3)
READ_MUTANTS = ("a_R1", "a_C1", "b_R1", "b_C1", "c_R1", "c_C1")
E_BUCKETS = ("1", "2", "4", "8", "16")
SV_BAR, G_BAR, PAIR_LO, PAIR_HI, ANCHOR_TOL, TAU_DROP = 1.00, 1.05, 0.97, 1.03, 0.03, 0.20
TOL, SPREAD_X, SPREAD_MIN = 0.01, 2.0, 0.005                          # P110's bar
TEXTS = ("wikitext", "c4val1")


def _mean(xs):
    xs = list(xs)
    return math.fsum(xs) / len(xs) if xs else None


def v0_fails(v: dict, gpu: dict | None = None) -> list:
    """V0 by the proof's rule a2 on the read's rows (outside calibration). ``gpu`` None leaves out the GPU-test item
    (the box's own check before V1)."""
    rec = dict(v.get("v0") or {}, census=v.get("census") or {})
    fails = judge(rec, gpu or {}, "a2", rows=READ_ROWS, mutant_keys=READ_MUTANTS)["fails"]
    return [f for f in fails if gpu is not None or f != "gpu_tests"]


def verify_cost(v1: dict) -> dict:
    """S_V(k, w) = tau_SD1(k, w) x anchor / full(k), anchor the A/A's mean; V_NOISY when the two differ by > 3 %."""
    a, b = v1["anchor_a_ms"], v1["anchor_b_ms"]
    anchor = _mean([a, b])
    sv = {w: {k: TAU_SD1[w][k] * anchor / v1[f"k{k}"]["full_ms"] for k in (1, 2, 3)} for w in WORKLOADS}
    best = {w: max(sv[w].values()) for w in WORKLOADS}
    noisy = abs(b / a - 1.0) > ANCHOR_TOL
    return {"anchor_ms": anchor, "anchor_ratio": b / a, "v_noisy": noisy, "S_V": sv, "best_S_V": best,
            "refutes": all(best[w] < SV_BAR for w in WORKLOADS),
            "model_error": {w: {k: sv[w][k] / S_SD1[w][k] - 1.0 for k in (1, 2, 3)} for w in WORKLOADS},
            "terms": {f"k{k}": {t: v1[f"k{k}"].get(t) for t in ("full_ms", "verify_ms", "post_ms", "loop_ms")}
                      for k in (1, 2, 3)}}


def stage_gate(v: dict, gpu: dict) -> str:
    """The box's decision after stage V: VOID (V0 failed or V1 missing: stop, no timing read), E_ONLY
    (VERIFY_COST_REFUTES without V_NOISY: the stop rule, E runs and Q is skipped) or ALL."""
    if v0_fails(v, gpu) or not v.get("v1"):
        return "VOID"
    vc = verify_cost(v["v1"])
    return "E_ONLY" if vc["refutes"] and not vc["v_noisy"] else "ALL"


def _q_stats(per, arm, ref):
    """P115's ``_stats`` (bias, spread, max |d|, KL, argmax), with math.fsum (a test pins it to P115's)."""
    rows = per.get(arm) or []
    d = [x["nll"] - ref[x["window"]] for x in rows]
    if not d:
        return {"n": 0}
    kls = [x["kl"] for x in rows if "kl" in x]
    return {"n": len(d), "bias": _mean(d), "spread": _mean(abs(x) for x in d), "max_abs": max(abs(x) for x in d),
            "mean_kl": _mean(kls), "argmax_agree": _mean(x["argmax_agree"] for x in rows)}


def _q_passes(s, fl):
    return s.get("n", 0) > 0 and s["bias"] <= fl["B_floor"] + TOL and s["spread"] <= SPREAD_X * max(fl["S_floor"], SPREAD_MIN)


def quality(q: dict) -> dict:
    """Per text: the floor (chunk, plus rep unless R repeated bit for bit: P110's definitions), mutant_scale and ON1 /
    ON2 against it, and the w16 draw (reported)."""
    out = {}
    for t in TEXTS:
        ref = {x["window"]: x["nll"] for x in q["per_window"][t]["R"]}
        st = {a: _q_stats(q["per_window"][t], a, ref) for a in ("rep", "chunk", "mutant_scale")}
        for k in (1, 2):
            st[f"ON{k}"] = _q_stats(q["on"]["per_window"][f"ON{k}"], t, ref)
        st["w16"] = _q_stats(q["w16"]["per_window"][t], "R", ref)
        draws = ["chunk"] + ([] if (q.get("rep_identical") or {}).get(t) else ["rep"])
        fl = {"draws": draws, "B_floor": max(abs(st[f]["bias"]) for f in draws), "S_floor": max(st[f]["spread"] for f in draws)}
        out[t] = {"stats": st, "floor": fl, "bias_bar": fl["B_floor"] + TOL, "spread_bar": SPREAD_X * max(fl["S_floor"], SPREAD_MIN),
                  "mutant_passes": _q_passes(st["mutant_scale"], fl),
                  "passes": {f"ON{k}": _q_passes(st[f"ON{k}"], fl) for k in (1, 2)}}
    return out


def _first_diff(a, b):
    """Per row: None where identical, else the first step the two part."""
    out = []
    for x, y in zip(a, b):
        i = next((j for j, (p, r) in enumerate(zip(x, y)) if p != r), None)
        out.append(i if i is not None or len(x) == len(y) else min(len(x), len(y)))
    return out


def speed_bar(g: dict, pairs: dict, k: int) -> bool:
    """FASTER's speed bar at k: g_k >= 1.05 with both of the workload's pair ratios above 1, on at least two of three.
    (Inside NOISY's band the pair clause cannot bind once g >= 1.05; it is kept as registered and tested directly.)"""
    return sum(1 for w in WORKLOADS if g[(k, w)] >= G_BAR and min(pairs[(k, w)]) > 1.0) >= 2


def judge_read(gpu, v, e: dict, q, expect: dict, w1=None) -> dict:
    """The registered rule, first that applies: VOID, VERIFY_COST_REFUTES, NOISY, QUALITY_FAIL, FASTER, NOT_FASTER.
    ``e``: {arm: record}; ``q`` None when the box skipped Q; ``expect``: {"e4b", "gnf4", "rev"}; ``w1``: the tripwire's
    expected tokens (reported)."""
    void = []
    if not gpu:
        void.append("missing:gpu_tests")
    if not v:
        void.append("missing:V")
    void += [f"missing:E:{a}" for a in ARMS if not e.get(a)]
    for name, r in [("V", v), ("Q", q)] + [(f"E:{a}", e.get(a)) for a in ARMS]:
        if not r:
            continue
        if r.get("e4b_sha") != expect["e4b"] or r.get("gnf4_sha") != expect["gnf4"]:
            void.append(f"pins:{name}")
        if (r.get("config") or {}).get("revision") != expect["rev"]:
            void.append(f"revision:{name}")
        sb = r.get("spec_build") or (r.get("census") or {}).get("spec_build") or {}
        if sb.get("mode", "off") != "off" and sb.get("head_sha256") != HEAD_SHA256:
            void.append(f"head:{name}")
    vc = None
    if v:
        if (v.get("config") or {}).get("max_seqs") != 16:
            void.append("default:V:max_seqs")
        void += [f"V0:{f}" for f in v0_fails(v, gpu or {})]
        if not v.get("v1"):
            void.append("missing:V1")
        else:
            vc = verify_cost(v["v1"])
    stop_rule = bool(vc and vc["refutes"] and not vc["v_noisy"])
    rate, census = {}, {}
    for a in ARMS:
        r = e.get(a)
        if not r:
            continue
        k = {"OFF": 0, "ON1": 1, "ON2": 2}[a.split("-")[0]]
        if (r.get("config") or {}).get("max_seqs") != 16:
            void.append(f"default:{a}:max_seqs")
        for b in E_BUCKETS + (("3",) if k == 2 else ()):
            if (r.get("graph_status") or {}).get(b) != "graph":
                void.append(f"default:{a}:bucket_{b}")
        for w in WORKLOADS:
            x = (r.get("workloads") or {}).get(w)
            if not x or x.get("decode_tok_s") is None:
                void.append(f"missing:E:{a}:{w}")
                continue
            rate[(a, w)] = x["decode_tok_s"]
            if not k:
                continue
            c, gd = x.get("spec_census") or {}, x.get("graph_stats_delta") or {}
            census[(a, w)] = c
            if not c.get("steps"):
                void.append(f"never_speculated:{a}:{w}")
                continue
            if c.get("dropped_batched") or c.get("post_eager"):
                void.append(f"engagement:{a}:{w}")
            for n in range(2, k + 2):
                s = gd.get(str(n)) or {}
                if s.get("eager_steps", 1) != 0 or not s.get("replays"):
                    void.append(f"verify_eager:{a}:{w}:{n}")
            plain = sum((gd.get("1") or {}).get(f, 0) for f in ("replays", "eager_steps"))
            if plain > c.get("dropped_end", 0):
                void.append(f"plain_steps:{a}:{w}")
            if c.get("tau_live") is None or c["tau_live"] < TAU_SD1[w][k] - TAU_DROP:
                void.append(f"mechanism:{a}:{w}")
    qr = None
    if q:
        if q.get("windows_sha256") != (q.get("on") or {}).get("windows_sha256") \
                or q.get("windows_sha256") != (q.get("w16") or {}).get("windows_sha256"):
            void.append("Q:windows")
        try:
            qr = quality(q)
        except (KeyError, TypeError, ValueError) as ex:
            void.append(f"missing:Q:{type(ex).__name__}")
        if qr:
            n_w = (q.get("windows") or {})
            for t in TEXTS:
                if any(qr[t]["stats"][a].get("n") != n_w.get(t) for a in ("rep", "chunk", "mutant_scale", "ON1", "ON2")):
                    void.append(f"Q:windows:{t}")
                if qr[t]["mutant_passes"] and not stop_rule:
                    void.append(f"Q:mutant_scale_passes:{t}")
    elif not stop_rule:
        void.append("missing:Q")
    # ---- E's arithmetic (reported under every verdict that reaches it)
    g, pairs, self_pairs = {}, {}, {}
    if all((a, w) in rate for a in ARMS for w in WORKLOADS):
        for k in (1, 2):
            for w in WORKLOADS:
                p = (rate[(f"ON{k}-a", w)] / rate[("OFF-a", w)], rate[(f"ON{k}-b", w)] / rate[("OFF-b", w)])
                pairs[(k, w)], g[(k, w)] = p, _mean(p)
        for x in ("OFF", "ON1", "ON2"):
            for w in WORKLOADS:
                self_pairs[(x, w)] = rate[(f"{x}-b", w)] / rate[(f"{x}-a", w)]

    out = {"verdict": None, "void": void, "stop_rule": stop_rule, "verify_cost": vc, "quality": qr,
           "g": {f"k{k}": {w: {"g": g[(k, w)], "pairs": list(pairs[(k, w)])} for w in WORKLOADS} for k in (1, 2)} if g else None,
           "self_pairs": {f"{x}:{w}": r for (x, w), r in self_pairs.items()},
           "ms_per_token": {f"{a}:{w}": 1000.0 / r for (a, w), r in rate.items() if r},
           "tau_live": {f"{a}:{w}": {"tau_live": c.get("tau_live"), "tau_sd1": TAU_SD1[w][int(a[2])],
                                      "acceptance": c.get("acceptance")} for (a, w), c in census.items()},
           "memory": {a: {"allocated": (e.get(a) or {}).get("max_memory_allocated"),
                          "reserved": (e.get(a) or {}).get("max_memory_reserved")} for a in ARMS}}
    if all(e.get(a) for a in ARMS):
        out["identity"] = {f"{a}:{w}": _first_diff(e[a]["workloads"][w]["identity_tokens"],
                                                   e[f"OFF-{a[-1]}"]["workloads"][w]["identity_tokens"])
                           for a in ARMS if a.startswith("ON") for w in WORKLOADS if w in (e[a].get("workloads") or {})}
        if w1 is not None:
            out["tripwire_w1"] = {a: _first_diff([e[a]["workloads"]["R"]["identity_tokens"][0]], [w1])[0]
                                  for a in ("OFF-a", "OFF-b") if "R" in (e[a].get("workloads") or {})}
    if void:
        out["verdict"] = "VOID"
        return out
    if stop_rule:
        out["verdict"] = "VERIFY_COST_REFUTES"
        out["conflict"] = any(speed_bar(g, pairs, k) for k in (1, 2))
        return out
    noisy = [f"{x}:{w}" for (x, w), r in self_pairs.items() if not PAIR_LO <= r <= PAIR_HI]
    if noisy:
        out.update(verdict="NOISY", noisy=noisy)
        return out
    licensed = [k for k in (1, 2) if all(qr[t]["passes"][f"ON{k}"] for t in TEXTS)]
    out["q_passed"] = licensed
    if not licensed:
        out["verdict"] = "QUALITY_FAIL"
        return out
    fast = [k for k in licensed if speed_bar(g, pairs, k)]
    out["faster_at"] = fast
    out["verdict"] = "FASTER" if fast else "NOT_FASTER"
    return out


def _read_fixture():
    """A complete read whose verdict is FASTER at k = 1 (C-think and C-nothink clear 1.05; R does not)."""
    expect = {"e4b": "e" * 40, "gnf4": "f" * 40, "rev": "r" * 40}
    gpu = {"rc": 0, "passed": 123, "skipped_for_card": 0}
    base = {"e4b_sha": expect["e4b"], "gnf4_sha": expect["gnf4"]}
    spec_on = {"mode": "eagle3", "hooks": 3, "head_sha256": HEAD_SHA256}
    p, _ = _good()
    v0 = {"capture_bitwise": p["capture_bitwise"], "draft": p["draft"], "transition": p["transition"],
          "addressing": {f"{r}_k{k}": {"rows_agree": 1.0, "cont_agree": 1.0, "mean_abs_dlogp": [0.01] * (k + 1)}
                         for r in READ_ROWS for k in (1, 2, 3)},
          "mutants": {m: {"rows_agree": 0.5, "cont_agree": 0.5, "mean_abs_dlogp": [0.02, 2.0, 2.0, 2.0]} for m in READ_MUTANTS}}
    v = dict(base, config={"max_seqs": 16, "revision": expect["rev"]}, v0=v0,
             census=dict(p["census"], spec_build=spec_on),
             v1={"anchor_a_ms": 7.40, "anchor_b_ms": 7.45,
                 **{f"k{k}": {"full_ms": 8.0 + k, "verify_ms": 6.5 + k, "post_ms": 0.6, "loop_ms": 0.9} for k in (1, 2, 3)}})
    speed = {"OFF": {"R": 1.0, "C-think": 1.0, "C-nothink": 1.0}, "ON1": {"R": 1.02, "C-think": 1.12, "C-nothink": 1.08},
             "ON2": {"R": 0.95, "C-think": 1.10, "C-nothink": 1.01}}
    e = {}
    for a in ARMS:
        arm = a.split("-")[0]
        k = {"OFF": 0, "ON1": 1, "ON2": 2}[arm]
        gs = {b: "graph" for b in E_BUCKETS + (("3",) if k == 2 else ())}
        wl = {}
        for w in WORKLOADS:
            x = {"decode_tok_s": 135.0 * speed[arm][w] * (1.004 if a.endswith("b") else 1.0),
                 "identity_tokens": [[1, 2, 3, 4]] * 16,
                 "graph_stats_delta": {"1": {"replays": 3 if k else 9000, "eager_steps": 0},
                                       **{str(n): {"replays": 500, "eager_steps": 0} for n in range(2, k + 2)}}}
            if k:
                x["spec_census"] = {"prefills": 112, "steps": 4000, "dropped_batched": 0, "dropped_end": 3, "post_eager": 0,
                                    "tau_live": TAU_SD1[w][k] - 0.05, "acceptance": 0.5}
            wl[w] = x
        e[a] = dict(base, config={"max_seqs": 16, "revision": expect["rev"]}, graph_status=gs,
                    spec_build=spec_on if k else {"mode": "off"}, workloads=wl,
                    max_memory_allocated=30e9 + k * 1e9, max_memory_reserved=31e9 + k * 1e9)
    n = 48

    def per(off):
        return [{"window": i, "nll": 2.0 + 0.01 * (i % 7) + off, "argmax_agree": 0.99, "kl": 0.001} for i in range(n)]
    win = {t: "w" + t for t in TEXTS}
    q = dict(base, config={"max_seqs": 16, "revision": expect["rev"]}, windows={t: n for t in TEXTS}, windows_sha256=win,
             rep_identical={t: True for t in TEXTS},
             per_window={t: {"R": per(0.0), "rep": per(0.0), "chunk": per(0.004), "mutant_scale": per(0.5)} for t in TEXTS},
             on={"windows_sha256": dict(win), "per_window": {"ON1": {t: per(0.003) for t in TEXTS},
                                                            "ON2": {t: per(0.004) for t in TEXTS}}},
             w16={"windows_sha256": dict(win), "per_window": {t: {"R": per(0.002)} for t in TEXTS}})
    return gpu, v, e, q, expect


def read_self_test() -> list:
    bad = []
    gpu, v, e, q, ex = _read_fixture()

    def run(edit=None, w1=None):
        G, V, E, Q, X = copy.deepcopy((gpu, v, e, q, ex))
        if edit:
            edit(G, V, E, Q)
        Q = None if Q == {} else Q
        return judge_read(G, V, E, Q, X, w1)

    def want(verdict, edit=None, contains=None, why=""):
        got = run(edit)
        if got["verdict"] != verdict or (contains and not any(contains in f for f in got["void"])):
            bad.append(f"{why or verdict}: got {got['verdict']} {got['void'][:4]}")
        return got

    good = want("FASTER", why="the fixture")
    if good.get("faster_at") != [1] or good.get("q_passed") != [1, 2]:
        bad.append(f"fixture: faster_at {good.get('faster_at')} q_passed {good.get('q_passed')}")
    # VOID, item by item
    want("VOID", lambda G, V, E, Q: E.pop("ON2-b"), "missing:E:ON2-b")
    want("VOID", lambda G, V, E, Q: Q.clear(), "missing:Q", "Q missing without the stop rule")
    want("VOID", lambda G, V, E, Q: G.update(rc=1), "V0:gpu_tests")
    want("VOID", lambda G, V, E, Q: E["ON1-a"].update(e4b_sha="0" * 40), "pins:E:ON1-a")
    want("VOID", lambda G, V, E, Q: Q["config"].update(revision="x"), "revision:Q")
    want("VOID", lambda G, V, E, Q: E["ON2-a"]["spec_build"].update(head_sha256="0"), "head:E:ON2-a")
    want("VOID", lambda G, V, E, Q: E["OFF-b"]["config"].update(max_seqs=32), "default:OFF-b:max_seqs")
    want("VOID", lambda G, V, E, Q: E["ON2-a"]["graph_status"].pop("3"), "default:ON2-a:bucket_3")
    want("VOID", lambda G, V, E, Q: V["v0"]["addressing"]["C3_k2"].update(mean_abs_dlogp=[0.01, 0.3, 0.01]),
         "V0:addressing:C3_k2")
    want("VOID", lambda G, V, E, Q: V["v0"]["mutants"]["c_C1"].update(mean_abs_dlogp=[0.2, 0.01, 0.01, 0.01]),
         "V0:GATE_TOO_WEAK:c_C1", "a mutant inside the bound on a fresh row")
    want("VOID", lambda G, V, E, Q: V["v0"]["addressing"].pop("R2_k3"), "V0:addressing:R2_k3")
    want("VOID", lambda G, V, E, Q: V["v0"]["draft"].update(agree=0.98), "V0:draft")
    want("VOID", lambda G, V, E, Q: V.pop("v1"), "missing:V1")
    want("VOID", lambda G, V, E, Q: E["ON1-b"]["workloads"]["R"]["spec_census"].update(steps=0), "never_speculated:ON1-b:R")
    want("VOID", lambda G, V, E, Q: E["ON2-b"]["workloads"]["C-think"]["graph_stats_delta"]["3"].update(eager_steps=2),
         "verify_eager:ON2-b:C-think:3")
    want("VOID", lambda G, V, E, Q: E["ON1-a"]["workloads"]["C-nothink"]["spec_census"].update(post_eager=1),
         "engagement:ON1-a:C-nothink")
    want("VOID", lambda G, V, E, Q: E["ON1-a"]["workloads"]["R"]["graph_stats_delta"]["1"].update(replays=40),
         "plain_steps:ON1-a:R", "T == 1 steps beyond the end drops")
    want("VOID", lambda G, V, E, Q: E["ON2-a"]["workloads"]["C-think"]["spec_census"].update(tau_live=TAU_SD1["C-think"][2] - 0.21),
         "mechanism:ON2-a:C-think")
    want("VOID", lambda G, V, E, Q: E["OFF-a"]["workloads"]["R"].update(decode_tok_s=None), "missing:E:OFF-a:R")
    want("VOID", lambda G, V, E, Q: Q["per_window"]["c4val1"].update(mutant_scale=Q["per_window"]["c4val1"]["R"]),
         "Q:mutant_scale_passes:c4val1")
    want("VOID", lambda G, V, E, Q: Q["on"]["windows_sha256"].update(wikitext="other"), "Q:windows")
    want("VOID", lambda G, V, E, Q: Q["on"]["per_window"]["ON2"]["wikitext"].pop(), "Q:windows:wikitext")
    # VERIFY_COST_REFUTES, the noise flag and the conflict
    slow = {f"k{k}": {"full_ms": 20.0, "verify_ms": 18.0, "post_ms": 1.0, "loop_ms": 1.0} for k in (1, 2, 3)}
    r = want("VERIFY_COST_REFUTES", lambda G, V, E, Q: (V["v1"].update(slow), Q.clear()), why="refutes, Q skipped")
    if r.get("conflict") is not True:
        bad.append("the conflict: E meets the speed bar under VERIFY_COST_REFUTES")
    r = want("VERIFY_COST_REFUTES", lambda G, V, E, Q: (V["v1"].update(slow), Q.clear(),
                                                        [E[a]["workloads"][w].update(decode_tok_s=135.0) for a in ARMS for w in WORKLOADS]),
             why="refutes, no conflict")
    if r.get("conflict") is not False:
        bad.append("no conflict when E is flat")
    want("FASTER", lambda G, V, E, Q: V["v1"].update(slow, anchor_b_ms=7.40 * 1.04), why="V_NOISY: the rule cannot fire")
    want("VOID", lambda G, V, E, Q: (V["v1"].update(slow, anchor_b_ms=7.40 * 1.04), Q.clear()), "missing:Q",
         "V_NOISY needs Q")
    if stage_gate(v, gpu) != "ALL":
        bad.append("stage gate: ALL")
    v2 = copy.deepcopy(v)
    v2["v1"].update(slow)
    if stage_gate(v2, gpu) != "E_ONLY":
        bad.append("stage gate: E_ONLY")
    v2["v0"]["draft"]["agree"] = 0.5
    if stage_gate(v2, gpu) != "VOID":
        bad.append("stage gate: VOID")
    # NOISY, QUALITY_FAIL, NOT_FASTER, FASTER's edges
    want("NOISY", lambda G, V, E, Q: E["ON2-b"]["workloads"]["C-nothink"].update(decode_tok_s=135.0 * 1.01 * 1.04))
    want("QUALITY_FAIL", lambda G, V, E, Q: [Q["on"]["per_window"][f"ON{k}"].update(c4val1=[dict(x, nll=x["nll"] + 0.05)
                                                                                         for x in Q["per_window"]["c4val1"]["R"]])
                                             for k in (1, 2)])
    want("NOT_FASTER", lambda G, V, E, Q: Q["on"]["per_window"]["ON1"].update(wikitext=[dict(x, nll=x["nll"] + 0.05)
                                                                                          for x in Q["per_window"]["wikitext"]["R"]]),
         why="k = 1 fails Q on one text; k = 2 passes but is not fast enough")
    gg = {(1, "R"): 1.06, (1, "C-think"): 1.06, (1, "C-nothink"): 1.0}
    if speed_bar(gg, {(1, "R"): (1.12, 1.0), (1, "C-think"): (1.07, 1.05), (1, "C-nothink"): (1.0, 1.0)}, 1)             or not speed_bar(gg, {(1, "R"): (1.07, 1.05), (1, "C-think"): (1.07, 1.05), (1, "C-nothink"): (1.0, 1.0)}, 1):
        bad.append("speed bar: a pair at 1 must not count")
    want("FASTER", lambda G, V, E, Q: Q["per_window"]["wikitext"].update(rep=[dict(x, nll=x["nll"] + 0.03)
                                                                               for x in Q["per_window"]["wikitext"]["R"]]),
         why="rep joins the floor only when R did not repeat (it is bit-identical here)")
    r = run(w1=[1, 2, 3, 9])
    if r.get("tripwire_w1") != {"OFF-a": 3, "OFF-b": 3} or r["identity"]["ON1-a:R"] != [None] * 16:
        bad.append(f"tripwire / identity: {r.get('tripwire_w1')}")
    return bad


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--prove")
    ap.add_argument("--gpu-tests")
    ap.add_argument("--rule", choices=RULES, default="a2")
    ap.add_argument("--read", metavar="DIR", help="the read's records: gpu_tests.json, v.json, e_<arm>.json, q.json")
    ap.add_argument("--gate-v", metavar="DIR", help="after stage V: print SD2_GATE VOID | E_ONLY | ALL")
    ap.add_argument("--expect-e4b")
    ap.add_argument("--expect-gnf4")
    ap.add_argument("--expect-rev")
    ap.add_argument("--expect-w1")
    ap.add_argument("--out")
    a = ap.parse_args(argv)
    if a.self_test:
        return self_test()
    if a.gate_v:
        gate = stage_gate(json.load(open(os.path.join(a.gate_v, "v.json"))),
                          json.load(open(os.path.join(a.gate_v, "gpu_tests.json"))))
        print(f"SD2_GATE {gate}", flush=True)
        return 0
    if a.read:
        def rec(name):
            path = os.path.join(a.read, name)
            return json.load(open(path)) if os.path.exists(path) else None
        w1 = json.load(open(a.expect_w1))["tokens"] if a.expect_w1 else None
        out = judge_read(rec("gpu_tests.json"), rec("v.json"), {x: rec(f"e_{x}.json") for x in ARMS}, rec("q.json"),
                         {"e4b": a.expect_e4b, "gnf4": a.expect_gnf4, "rev": a.expect_rev}, w1)
        json.dump(out, open(a.out, "w"), indent=1, sort_keys=True, default=str)
        print("SD2_READ_VERDICT " + json.dumps({"verdict": out["verdict"], "void": out["void"][:12],
                                                "faster_at": out.get("faster_at"), "conflict": out.get("conflict")}),
              flush=True)
        return 0
    if a.prove:
        rec = judge(json.load(open(a.prove)), json.load(open(a.gpu_tests)), a.rule)
        json.dump(rec, open(a.out, "w"), indent=1, sort_keys=True)
        print("SD2_PROVE_VERDICT " + json.dumps({"verdict": rec["verdict"], "fails": rec["fails"], "rule": rec["rule"]}),
              flush=True)
        return 0
    ap.print_usage()
    return 2


if __name__ == "__main__":
    sys.exit(main())
