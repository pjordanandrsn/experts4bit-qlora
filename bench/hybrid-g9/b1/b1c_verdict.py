# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""PREREG-b1c verdict: the all-resident collapse cert. Bars hardcoded
from the prereg; --self-test both directions before any receipt."""

import argparse
import json
import sys
from pathlib import Path

AA_SPREAD_MAX = 7.5
HC_PASS = 15.0          # % C1-vs-C0 median step improvement
HC_PARTIAL = 8.0
HM_BRACKET_MAX = 0.60   # C1 moe_host <= 60% of C0's
GS_STEP_LO, GS_STEP_HI = 115.0, 165.0
GS_ATTN_MAX = 55.0


def _step(rep):
    return float(rep["decode_median_ms"]["step"])


def _tokens(rep):
    return rep["generated_tokens"]


def verdict(c0_1, c0_2, c1_1, c1_2, r1_1, r1_2, c0p, c1p, b16):
    out = {}
    d = b16["decode_median_ms"]
    gs = (GS_STEP_LO <= float(d["step"]) <= GS_STEP_HI
          and float(d["attention_host"]) <= GS_ATTN_MAX
          and bool(b16.get("collapse")))
    out["gs_b16_flag_on"] = {"pass": bool(gs),
                             "step_ms": float(d["step"]),
                             "collapse_recorded": bool(b16.get("collapse"))}
    if not gs:
        out["verdict"] = ("NO-VERDICT (B=16 sanity failed: the flag must "
                          "be inert on the subset placement and the run "
                          "must sit in the certified band with it ON)")
        return out

    arms = {}
    for name, (x, y) in (("C0", (c0_1, c0_2)), ("C1", (c1_1, c1_2)),
                         ("R1", (r1_1, r1_2))):
        m1, m2 = _step(x), _step(y)
        spread = abs(m1 - m2) / min(m1, m2) * 100.0
        arms[name] = {"step_ms": min(m1, m2), "aa_spread_pct": spread,
                      "aa_pass": spread < AA_SPREAD_MAX}
        if not arms[name]["aa_pass"]:
            out["arms"] = arms
            out["verdict"] = f"NO-VERDICT (G0 fail on arm {name})"
            return out
    out["arms"] = arms

    # every rep must carry a NON-EMPTY token record (the vacuous-gate
    # lesson: an identity gate that can compare {} == {} is not a gate)
    for name, rep in (("c0", c0_1), ("c1", c1_1), ("r1", r1_1)):
        if not sum(len(v) for v in _tokens(rep).values()):
            out["verdict"] = f"NO-VERDICT (empty token record in {name})"
            return out
    ident = {"c0_c1_bitwise": _tokens(c0_1) == _tokens(c1_1),
             "c1_r1_bitwise": _tokens(c1_1) == _tokens(r1_1)}
    out["identity"] = ident
    if not ident["c0_c1_bitwise"]:
        out["verdict"] = ("REFUTED (G1: the collapse changed decoded "
                          "tokens — the arithmetic-identity claim is "
                          "false; revert regardless of speed)")
        return out

    s0, s1 = arms["C0"]["step_ms"], arms["C1"]["step_ms"]
    sr = arms["R1"]["step_ms"]
    dpct = (s0 - s1) / s0 * 100.0
    m0 = float(c0p["decode_median_ms"]["moe_host"])
    m1_ = float(c1p["decode_median_ms"]["moe_host"])
    bracket_ratio = m1_ / m0 if m0 else 1.0
    gap_closed = (s0 - s1) / (s0 - sr) if s0 > sr else None
    out["h_c"] = {"c0_ms": s0, "c1_ms": s1, "improvement_pct": dpct,
                  "pass": dpct >= HC_PASS,
                  "partial": HC_PARTIAL <= dpct < HC_PASS}
    out["h_m"] = {"c0_moe_host_ms": m0, "c1_moe_host_ms": m1_,
                  "ratio": bracket_ratio,
                  "pass": bracket_ratio <= HM_BRACKET_MAX}
    out["reported"] = {"r1_ms": sr, "gap_closed_fraction": gap_closed,
                       "c1_vs_r1_residual_pct":
                           (s1 - sr) / sr * 100.0 if sr else None}

    hc, hm = out["h_c"], out["h_m"]["pass"]
    if hc["pass"] and hm:
        out["verdict"] = ("CERTIFIED (flip collapse_resident default ON "
                          "in this RESULTS PR — the #220 lesson)")
    elif hc["pass"]:
        out["verdict"] = ("CERTIFIED-WITH-OPEN-MECHANISM (wall win real; "
                          "the bracket attribution is mis-modeled — ship "
                          "and file the gap)")
    elif hc["partial"]:
        out["verdict"] = ("PARTIAL (ship only with zero regressions in "
                          "the B=16 sanity and every bracket; record "
                          "honestly)")
    else:
        out["verdict"] = ("REFUTED (revert; the ladder falls through to "
                          "the M=1 executor branch)")
    return out


def _rep(step, toks=None, attn=40.0, moe=None, collapse=False):
    d = {"step": step, "attention_host": attn, "dram_experts_host": 0.0}
    if moe is not None:
        d["moe_host"] = moe
    return {"decode_median_ms": d, "collapse": collapse,
            "generated_tokens": toks if toks is not None else {"0": [1, 2]}}


def self_test():
    tok = {"0": [5, 6, 7]}
    b16 = _rep(141.0, tok, attn=46.0, collapse=True)
    # certified: -25% wall, bracket 12/33
    v = verdict(_rep(66.0, tok), _rep(66.4, tok), _rep(49.5, tok),
                _rep(49.7, tok), _rep(49.4, tok), _rep(49.5, tok),
                _rep(88.0, tok, moe=33.0), _rep(64.0, tok, moe=12.8), b16)
    assert v["verdict"].startswith("CERTIFIED (flip"), v["verdict"]
    assert v["identity"]["c1_r1_bitwise"], v
    # wall win, bracket unmoved -> open mechanism
    v = verdict(_rep(66.0, tok), _rep(66.4, tok), _rep(49.5, tok),
                _rep(49.7, tok), _rep(49.4, tok), _rep(49.5, tok),
                _rep(88.0, tok, moe=33.0), _rep(64.0, tok, moe=30.0), b16)
    assert v["verdict"].startswith("CERTIFIED-WITH-OPEN"), v["verdict"]
    # partial band
    v = verdict(_rep(66.0, tok), _rep(66.4, tok), _rep(59.0, tok),
                _rep(59.2, tok), _rep(49.4, tok), _rep(49.5, tok),
                _rep(88.0, tok, moe=33.0), _rep(80.0, tok, moe=19.0), b16)
    assert v["verdict"].startswith("PARTIAL"), v["verdict"]
    # refuted on wall
    v = verdict(_rep(66.0, tok), _rep(66.4, tok), _rep(63.5, tok),
                _rep(63.6, tok), _rep(49.4, tok), _rep(49.5, tok),
                _rep(88.0, tok, moe=33.0), _rep(85.0, tok, moe=31.0), b16)
    assert v["verdict"].startswith("REFUTED (revert"), v["verdict"]
    # G1 breach dominates speed
    v = verdict(_rep(66.0, tok), _rep(66.4, tok),
                _rep(49.5, {"0": [9]}), _rep(49.7, {"0": [9]}),
                _rep(49.4, tok), _rep(49.5, tok),
                _rep(88.0, tok, moe=33.0), _rep(64.0, tok, moe=12.8), b16)
    assert v["verdict"].startswith("REFUTED (G1"), v["verdict"]
    # empty token record refuses (the vacuous-gate lesson)
    v = verdict(_rep(66.0, {}), _rep(66.4, {}), _rep(49.5, {}),
                _rep(49.7, {}), _rep(49.4, {}), _rep(49.5, {}),
                _rep(88.0, tok, moe=33.0), _rep(64.0, tok, moe=12.8), b16)
    assert v["verdict"].startswith("NO-VERDICT (empty token"), v["verdict"]
    # B=16 sanity: flag not recorded -> refuse
    b16_off = _rep(141.0, tok, attn=46.0, collapse=False)
    v = verdict(_rep(66.0, tok), _rep(66.4, tok), _rep(49.5, tok),
                _rep(49.7, tok), _rep(49.4, tok), _rep(49.5, tok),
                _rep(88.0, tok, moe=33.0), _rep(64.0, tok, moe=12.8),
                b16_off)
    assert v["verdict"].startswith("NO-VERDICT (B=16"), v["verdict"]
    # B=16 sanity: regression out of band -> refuse
    b16_bad = _rep(180.0, tok, attn=46.0, collapse=True)
    v = verdict(_rep(66.0, tok), _rep(66.4, tok), _rep(49.5, tok),
                _rep(49.7, tok), _rep(49.4, tok), _rep(49.5, tok),
                _rep(88.0, tok, moe=33.0), _rep(64.0, tok, moe=12.8),
                b16_bad)
    assert v["verdict"].startswith("NO-VERDICT (B=16"), v["verdict"]
    print("self-test OK: 8/8 branches exercised")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true")
    for f in ("c0-1", "c0-2", "c1-1", "c1-2", "r1-1", "r1-2",
              "c0p", "c1p", "b16"):
        ap.add_argument(f"--{f}")
    a = ap.parse_args()
    if a.self_test:
        self_test()
        sys.exit(0)
    load = lambda f: json.loads(Path(f).read_text())
    print(json.dumps(verdict(
        load(a.c0_1), load(a.c0_2), load(a.c1_1), load(a.c1_2),
        load(a.r1_1), load(a.r1_2), load(a.c0p), load(a.c1p),
        load(a.b16)), indent=2))
