#!/usr/bin/env python3
# Copyright (c) 2026 Cerin Amroth LLC. MIT.
"""sc1b_read.py -- lane SC1b (#846), amendment A1: the evaluator for predictions Q1-Q5 and the read's tables, written and
merged before any box D data existed. It reads only what box D's reducer wrote (sc1b_arm_<engine>_b<B>.json and
sc1b_gap_e4b_<comparator>_b<B>.json) and decides each prediction HOLDS / REFUTED / UNREAD by the rules below; nothing here
re-reduces a trace.

Q1 (G1, B=1) and Q5 (G3, B=1): "the largest term of dP is dI_in". The largest term of dP is the term with the largest
    contribution in dP's direction, argmax_k sign(dP) x d_k over the identity's terms (the nine classes, I_in, idle_out);
    the remainder O is not a term. HOLDS iff that term is I_in. UNREAD if the gap is unread, or if idle_out is not
    nameable (G-inflate failed on either arm) and its contribution exceeds I_in's (the winner cannot be decided).
Q2 (G1): e4b runs >= 1.5x llama.cpp's in-graph kernels per B=1 step: the ratio of the two node-mode medians
    `kernels_in_graph`. UNREAD if either arm's node capture is VOID or carries NSYS_DIAGNOSTIC_ERRORS (counts need complete
    records; the class map does not enter).
Q3 (G1): d moe_expert < d I_in, signed, e4b minus llama.cpp. Read only when G1 is read (G-map holds for both arms: a
    CLASS_MAP_INCOMPLETE arm leaves the gap unread).
Q4 (G2, B=16): d idle_out carries >= 50 % of |dP|: same sign as dP and |d idle_out| >= 0.5 |dP|. UNREAD if the gap is
    unread or idle_out is not nameable. No noise clause: the prediction as registered has none (the noise is printed).

  sc1b_read.py DIR [--out RESULTS-sc1b.md] [--json verdicts.json]     DIR holds box D's sc1b_arm_*.json / sc1b_gap_*.json
  sc1b_read.py --self-test
"""
from __future__ import annotations

import argparse
import json
import os
import sys

TERMS = ("moe_expert", "moe_route", "attn", "dense_gemm", "norm_elem", "sample", "input_prep", "memcpy", "residual", "I_in",
         "idle_out")
GAPS = (("G1", "llamacpp", 1, 1.484), ("G2", "llamacpp", 16, 0.624), ("G3", "vllm", 1, 1.203), ("G3", "vllm", 16, 1.172),
        ("G4", "sglang", 1, 1.268), ("G4", "sglang", 16, 1.191))
Q2_RATIO = 1.5
Q4_SHARE = 0.5


def _load(d, name):
    p = os.path.join(d, name)
    if not os.path.isfile(p):
        return None
    with open(p) as f:
        return json.load(f)


def largest_term(g):
    """(term, contribution) with the largest contribution in dP's direction, or (None, None) when dP is 0."""
    dp = g["delta_P_ms"]
    if not dp:
        return None, None
    s = 1 if dp > 0 else -1
    k = max(TERMS, key=lambda t: s * g["delta"][t])
    return k, round(s * g["delta"][k], 6)


def q_largest_is_I_in(g):
    if not g or g.get("status") != "ok":
        return {"verdict": "UNREAD", "why": "the gap is unread" + (f": {g.get('why')}" if g else " (no gap record)")}
    k, c = largest_term(g)
    if k is None:
        return {"verdict": "UNREAD", "why": "dP is 0"}
    s = 1 if g["delta_P_ms"] > 0 else -1
    i_in = s * g["delta"]["I_in"]
    if not g["idle_out_nameable"] and s * g["delta"]["idle_out"] > i_in:
        return {"verdict": "UNREAD", "why": "idle_out is not nameable (G-inflate) and contributes more than I_in",
                "largest": k, "contribution_ms": c}
    return {"verdict": "HOLDS" if k == "I_in" else "REFUTED", "largest": k, "contribution_ms": c, "I_in_contribution_ms": round(i_in, 6),
            "delta_P_ms": g["delta_P_ms"]}


def q2(e, ll):
    for x, n in ((e, "e4b"), (ll, "llamacpp")):
        if not x or (x.get("node") or {}).get("status") != "ok":
            return {"verdict": "UNREAD", "why": f"{n}'s node capture is VOID or missing"}
        if "NSYS_DIAGNOSTIC_ERRORS" in x.get("labels", []):
            return {"verdict": "UNREAD", "why": f"{n} carries NSYS_DIAGNOSTIC_ERRORS"}
    ke, kl = e["node"]["kernels_in_graph"], ll["node"]["kernels_in_graph"]
    if not kl:
        return {"verdict": "UNREAD", "why": "llama.cpp shows no in-graph kernels"}
    r = round(ke / kl, 4)
    return {"verdict": "HOLDS" if r >= Q2_RATIO else "REFUTED", "ratio": r, "kernels_in_graph": [ke, kl]}


def q3(g):
    if not g or g.get("status") != "ok":
        return {"verdict": "UNREAD", "why": "G1 is unread" + (f": {g.get('why')}" if g else "")}
    de, di = g["delta"]["moe_expert"], g["delta"]["I_in"]
    return {"verdict": "HOLDS" if de < di else "REFUTED", "delta_moe_expert_ms": de, "delta_I_in_ms": di}


def q4(g):
    if not g or g.get("status") != "ok":
        return {"verdict": "UNREAD", "why": "G2 is unread" + (f": {g.get('why')}" if g else "")}
    if not g["idle_out_nameable"]:
        return {"verdict": "UNREAD", "why": "idle_out is not nameable (G-inflate failed on an arm)"}
    dp, di = g["delta_P_ms"], g["delta"]["idle_out"]
    ok = bool(dp) and (di > 0) == (dp > 0) and abs(di) >= Q4_SHARE * abs(dp)
    return {"verdict": "HOLDS" if ok else "REFUTED", "delta_idle_out_ms": di, "delta_P_ms": dp,
            "share": round(di / dp, 4) if dp else None, "noise_iqr_ms": g["noise_iqr"].get("idle_out")}


def read(d):
    arms = {(e, b): _load(d, f"sc1b_arm_{e}_b{b}.json") for e in ("e4b", "llamacpp", "vllm", "sglang") for b in (1, 16)}
    gaps = {(c, b): _load(d, f"sc1b_gap_e4b_{c}_b{b}.json") for _, c, b, _r in GAPS}
    v = {"Q1": q_largest_is_I_in(gaps[("llamacpp", 1)]), "Q2": q2(arms[("e4b", 1)], arms[("llamacpp", 1)]),
         "Q3": q3(gaps[("llamacpp", 1)]), "Q4": q4(gaps[("llamacpp", 16)]), "Q5": q_largest_is_I_in(gaps[("vllm", 1)])}
    return arms, gaps, v


def _f(x, n=3):
    return "--" if x is None else f"{x:.{n}f}"


def render(arms, gaps, v):
    out = ["# SC1b read: where each engine's decode step goes (box D)", "",
           "Verdicts by `bench/sc1b/sc1b_read.py` (amendment A1, merged before box D's data). Terms are per-step medians in ms.", "",
           "## Predictions", "", "| Q | verdict | detail |", "|---|---|---|"]
    for q, r in v.items():
        det = ", ".join(f"{k}={r[k]}" for k in r if k != "verdict")
        out.append(f"| {q} | {r['verdict']} | {det} |")
    out += ["", "## Arms", "", "| engine | B | status | labels | P | unprofiled | " + " | ".join(TERMS) + " | O |",
            "|" + "---|" * (len(TERMS) + 7)]
    for (e, b), a in arms.items():
        if not a:
            out.append(f"| {e} | {b} | missing | | | | " + " | ".join("" for _ in TERMS) + " | |")
            continue
        t = a.get("terms") or {}
        out.append(f"| {e} | {b} | {a.get('status')} | {' '.join(a.get('labels', []))} | {_f(a.get('P_ms'))} | {_f(a.get('unprofiled_ms'))} | "
                   + " | ".join(_f(t.get(k)) for k in TERMS) + f" | {_f(t.get('O'))} |")
    out += ["", "## Gaps (e4b minus the comparator; positive = e4b slower)", "",
            "| gap | comparator | B | status | reading | named | dP | dO | census ratio | SC1 ratio | top three |",
            "|---|---|---|---|---|---|---|---|---|---|---|"]
    for gid, c, b, r in GAPS:
        g = gaps.get((c, b))
        if not g:
            out.append(f"| {gid} | {c} | {b} | missing | | | | | | {r} | |")
            continue
        if g.get("status") != "ok":
            out.append(f"| {gid} | {c} | {b} | unread | {'; '.join(g.get('why', []))} | | | | | {r} | |")
            continue
        top = "; ".join(f"{k} {d:+.3f}" for k, d in g.get("top_three") or [])
        out.append(f"| {gid} | {c} | {b} | ok | {g['reading']} | {g.get('named_cause') or '--'} | {g['delta_P_ms']:+.3f} | "
                   f"{g['delta_O_ms']:+.3f} | {_f(g.get('census_ratio_unprofiled'))} | {g.get('sc1_ratio')} | {top} |")
    return "\n".join(out) + "\n"


def self_test():
    def gap(dp, d, nameable=True, status="ok"):
        full = {t: 0.0 for t in TERMS}
        full.update(d)
        return {"status": status, "delta_P_ms": dp, "delta": full, "idle_out_nameable": nameable, "noise_iqr": {"idle_out": 0.01}}
    checks = 0
    assert q_largest_is_I_in(gap(1.0, {"I_in": 0.6, "moe_expert": 0.3}))["verdict"] == "HOLDS"
    assert q_largest_is_I_in(gap(1.0, {"I_in": 0.3, "moe_expert": 0.6}))["verdict"] == "REFUTED"
    # an opposite-signed term is not "of dP", however large
    assert q_largest_is_I_in(gap(1.0, {"I_in": 0.4, "attn": -0.9}))["verdict"] == "HOLDS"
    assert q_largest_is_I_in(gap(1.0, {"I_in": 0.4, "idle_out": 0.5}, nameable=False))["verdict"] == "UNREAD"
    assert q_largest_is_I_in(gap(1.0, {"I_in": 0.6, "idle_out": 0.5}, nameable=False))["verdict"] == "HOLDS"
    assert q_largest_is_I_in({"status": "unread", "why": ["x"]})["verdict"] == "UNREAD"
    assert q_largest_is_I_in(None)["verdict"] == "UNREAD"
    checks += 7
    arm = lambda k, labels=(): {"node": {"status": "ok", "kernels_in_graph": k}, "labels": list(labels)}   # noqa: E731
    assert q2(arm(900), arm(500))["verdict"] == "HOLDS" and q2(arm(700), arm(500))["verdict"] == "REFUTED"
    assert q2(arm(900, ["NSYS_DIAGNOSTIC_ERRORS"]), arm(500))["verdict"] == "UNREAD"
    assert q2({"node": {"status": "void"}, "labels": []}, arm(500))["verdict"] == "UNREAD"
    checks += 3
    assert q3(gap(1.0, {"moe_expert": 0.1, "I_in": 0.5}))["verdict"] == "HOLDS"
    assert q3(gap(1.0, {"moe_expert": 0.5, "I_in": 0.1}))["verdict"] == "REFUTED"
    checks += 2
    assert q4(gap(-5.0, {"idle_out": -3.0}))["verdict"] == "HOLDS"
    assert q4(gap(-5.0, {"idle_out": -2.0}))["verdict"] == "REFUTED"
    assert q4(gap(-5.0, {"idle_out": 3.0}))["verdict"] == "REFUTED"
    assert q4(gap(-5.0, {"idle_out": -3.0}, nameable=False))["verdict"] == "UNREAD"
    checks += 4
    md = render({("e4b", 1): None}, {}, {"Q1": {"verdict": "UNREAD", "why": "x"}})
    assert "| Q1 | UNREAD |" in md
    checks += 1
    print(f"self-test OK ({checks} checks)")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dir", nargs="?")
    ap.add_argument("--out")
    ap.add_argument("--json")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args(argv)
    if a.self_test:
        return self_test()
    if not a.dir:
        ap.error("DIR is required")
    arms, gaps, v = read(a.dir)
    md = render(arms, gaps, v)
    if a.out:
        with open(a.out, "w") as f:
            f.write(md)
    if a.json:
        with open(a.json, "w") as f:
            json.dump({"verdicts": v}, f, indent=1)
    print("SC1B_READ " + json.dumps({q: r["verdict"] for q, r in v.items()}), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
