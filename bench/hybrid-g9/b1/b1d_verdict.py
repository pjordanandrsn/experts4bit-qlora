# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""PREREG-b1d stage C verdict: the device-driven decode loop cert.
Bars hardcoded from the prereg; --self-test both directions first."""

import argparse
import json
import sys
from pathlib import Path

AA_SPREAD_MAX = 7.5
HG_PASS_MS = 20.0        # certified: graph step at or under this
HG_PARTIAL_MS = 30.0     # (20, 30] partial; above refuted
HD_OCC_TOL = 0.15        # graph kernel occupancy within 15% of eager's
GS_STEP_LO, GS_STEP_HI = 115.0, 165.0
GS_ATTN_MAX = 55.0


def _step(rep):
    return float(rep["step_ms_clean"])


def verdict(e1, e2, g1_, g2_, b16, eager_occ_ms, graph_occ_ms):
    out = {}
    d = b16["decode_median_ms"]
    gs = (GS_STEP_LO <= float(d["step"]) <= GS_STEP_HI
          and float(d["attention_host"]) <= GS_ATTN_MAX)
    out["gs_b16"] = {"pass": bool(gs), "step_ms": float(d["step"])}
    if not gs:
        out["verdict"] = "NO-VERDICT (B=16 certified point regressed)"
        return out

    arms = {}
    for name, (x, y) in (("eager", (e1, e2)), ("graph", (g1_, g2_))):
        m1, m2 = _step(x), _step(y)
        spread = abs(m1 - m2) / min(m1, m2) * 100.0
        arms[name] = {"step_ms": min(m1, m2), "aa_spread_pct": spread,
                      "aa_pass": spread < AA_SPREAD_MAX}
        if not arms[name]["aa_pass"]:
            out["arms"] = arms
            out["verdict"] = f"NO-VERDICT (G0 fail on arm {name})"
            return out
    out["arms"] = arms

    for name, rep in (("e1", e1), ("g1", g1_)):
        if not rep.get("tokens"):
            out["verdict"] = f"NO-VERDICT (empty token record in {name})"
            return out
    ident = e1["tokens"] == g1_["tokens"]
    out["identity"] = {"eager_graph_bitwise": bool(ident),
                       "n_tokens": len(e1["tokens"])}
    if not ident:
        out["verdict"] = ("REFUTED (G1: the graph loop decoded different "
                          "tokens; revert regardless of speed)")
        return out

    sg = arms["graph"]["step_ms"]
    occ_ratio = (graph_occ_ms / eager_occ_ms) if eager_occ_ms else 1.0
    hd = abs(occ_ratio - 1.0) <= HD_OCC_TOL
    out["h_g"] = {"graph_step_ms": sg, "eager_step_ms":
                  arms["eager"]["step_ms"],
                  "pass": sg <= HG_PASS_MS,
                  "partial": HG_PASS_MS < sg <= HG_PARTIAL_MS}
    out["h_d"] = {"eager_occ_ms": eager_occ_ms,
                  "graph_occ_ms": graph_occ_ms,
                  "ratio": occ_ratio, "pass": bool(hd)}
    out["reported"] = {"tok_s_single_stream": 1000.0 / sg,
                       "speedup_vs_eager":
                           arms["eager"]["step_ms"] / sg}

    if out["h_g"]["pass"] and hd:
        out["verdict"] = ("CERTIFIED (the graph loop becomes the B=1 "
                          "serving default behind the all-VRAM placement "
                          "gate; rung 3 registers against the new floor)")
    elif out["h_g"]["pass"]:
        out["verdict"] = ("CERTIFIED-WITH-OPEN-MECHANISM (wall bar met; "
                          "occupancy drifted beyond 15% -- ship and file "
                          "what the graph smuggled in or out)")
    elif out["h_g"]["partial"]:
        out["verdict"] = "PARTIAL (ship only with zero regressions)"
    else:
        out["verdict"] = ("REFUTED for the capture approach (revert; "
                          "rung 3 proceeds against the eager baseline)")
    return out


def _t(step, toks=None):
    return {"step_ms_clean": step, "b1d_timed": True,
            "tokens": toks if toks is not None else [1, 2, 3]}


def _b16(step=141.0, attn=46.0):
    return {"decode_median_ms": {"step": step, "attention_host": attn}}


def self_test():
    tok = [5, 6, 7, 8]
    # certified
    v = verdict(_t(43.0, tok), _t(43.4, tok), _t(13.9, tok), _t(14.0, tok),
                _b16(), 13.3, 13.4)
    assert v["verdict"].startswith("CERTIFIED ("), v
    # occupancy drift -> open mechanism
    v = verdict(_t(43.0, tok), _t(43.4, tok), _t(13.9, tok), _t(14.0, tok),
                _b16(), 13.3, 17.0)
    assert v["verdict"].startswith("CERTIFIED-WITH-OPEN"), v
    # partial band
    v = verdict(_t(43.0, tok), _t(43.4, tok), _t(24.0, tok), _t(24.2, tok),
                _b16(), 13.3, 13.4)
    assert v["verdict"].startswith("PARTIAL"), v
    # refuted on wall
    v = verdict(_t(43.0, tok), _t(43.4, tok), _t(33.0, tok), _t(33.2, tok),
                _b16(), 13.3, 13.4)
    assert v["verdict"].startswith("REFUTED for the capture"), v
    # G1 dominates speed
    v = verdict(_t(43.0, tok), _t(43.4, tok), _t(13.9, [9, 9, 9, 9]),
                _t(14.0, [9, 9, 9, 9]), _b16(), 13.3, 13.4)
    assert v["verdict"].startswith("REFUTED (G1"), v
    # empty tokens refused (the vacuous-gate lesson)
    v = verdict(_t(43.0, []), _t(43.4, []), _t(13.9, []), _t(14.0, []),
                _b16(), 13.3, 13.4)
    assert v["verdict"].startswith("NO-VERDICT (empty"), v
    # G0 arm spread
    v = verdict(_t(43.0, tok), _t(55.0, tok), _t(13.9, tok), _t(14.0, tok),
                _b16(), 13.3, 13.4)
    assert v["verdict"].startswith("NO-VERDICT (G0 fail on arm eager"), v
    # B=16 regression blocks
    v = verdict(_t(43.0, tok), _t(43.4, tok), _t(13.9, tok), _t(14.0, tok),
                _b16(step=190.0), 13.3, 13.4)
    assert v["verdict"].startswith("NO-VERDICT (B=16"), v
    print("self-test OK: 8/8 branches exercised")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true")
    for f in ("e1", "e2", "g1", "g2", "b16"):
        ap.add_argument(f"--{f}")
    ap.add_argument("--eager-occ-ms", type=float)
    ap.add_argument("--graph-occ-ms", type=float)
    a = ap.parse_args()
    if a.self_test:
        self_test()
        sys.exit(0)
    load = lambda f: json.loads(Path(f).read_text())
    print(json.dumps(verdict(load(a.e1), load(a.e2), load(a.g1),
                             load(a.g2), load(a.b16),
                             a.eager_occ_ms, a.graph_occ_ms), indent=2))
