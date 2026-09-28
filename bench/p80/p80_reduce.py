#!/usr/bin/env python3
"""Lane P80's reducer (bench/p80/PREREG-p80.md; e4b#511). Reads the five arm receipts step_decomp's P80 stage wrote
(A1, B1, B2, A2 = the memo's fixed order; P = the padded-eager oracle) and applies the registered decision rule. The rule's
quantities are computed HERE, from the receipts, so the verdict is not a reading of prose.

    python p80_reduce.py --dir <dir with p80_{A1,B1,B2,A2,P}.json> --out verdict.json
    python p80_reduce.py --self-test
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

BAND = 1.03                     # the p37 self-pair band the memo registered
ARMS = ("A1", "B1", "B2", "A2", "P")
MODES = {"A1": "eager", "A2": "eager", "B1": "graph", "B2": "graph", "P": "padded"}


def _first_divergence(x: dict, y: dict):
    for row in sorted(x, key=int):
        a, b = x[row], y.get(row, [])
        for i, (u, v) in enumerate(zip(a, b)):
            if u != v:
                return {"row": int(row), "token_index": i}
        if len(a) != len(b):
            return {"row": int(row), "token_index": min(len(a), len(b)), "length_differs": True}
    return None


def reduce(recs: dict) -> dict:
    """``recs`` maps arm -> receipt dict (or a dict with "status" when the arm did not produce one)."""
    v: dict = {"lane": "P80", "band": BAND, "reasons": [], "verdict": None}
    missing = [k for k in ARMS if not isinstance(recs.get(k), dict) or "timed_agg_tok_s" not in recs[k]]
    oom_b = [k for k in ("B1", "B2") if isinstance(recs.get(k), dict) and recs[k].get("oom")]
    if oom_b:
        v["verdict"] = "REFUTED"
        v["reasons"].append(f"candidate arm(s) {oom_b} OOM (the memo's failure criterion)")
        return v
    if missing:
        v["verdict"] = "VOID"
        v["reasons"].append(f"arm(s) without a receipt: {missing} (NOT_RUN/VOID, neither confirms nor refutes)")
        return v
    # engagement: each arm ran its declared path, over the registered trace, in both passes
    for k in ARMS:
        r = recs[k]
        if r.get("mode") != MODES[k]:
            v["verdict"] = "VOID"
            v["reasons"].append(f"{k} ran mode {r.get('mode')!r}, registered {MODES[k]!r}")
            return v
        if not r.get("warm_equals_timed_tokens"):
            v["reasons"].append(f"{k}: warm and timed passes decoded different tokens (non-determinism)")
    trace = recs["A1"]["trace"]
    if any(recs[k]["trace"] != trace for k in ARMS):
        v["verdict"] = "VOID"
        v["reasons"].append("arms disagree on the trace")
        return v
    for k in ("B1", "B2"):
        st = recs[k].get("graph_status") or {}
        stats = recs[k].get("graph_stats") or {}
        bad = {b: s for b, s in st.items() if s != "graph"}
        eager = sum(int(s.get("eager_steps", 0)) for s in stats.values())
        if bad or eager or not st:
            v["verdict"] = "REFUTED"
            v["reasons"].append(f"{k}: a bucket fell back to eager (status {bad or st}, eager_steps {eager}) -- "
                                "a required eager fallback refutes (memo)")
            return v
    if recs["P"].get("graph_status") and any(not str(s).startswith("eager") for s in recs["P"]["graph_status"].values()):
        v["verdict"] = "VOID"
        v["reasons"].append("P captured graphs; it must run the padded step eagerly")
        return v
    for k in ("A1", "A2"):
        if recs[k].get("graph_status") is not None:
            v["verdict"] = "VOID"
            v["reasons"].append(f"{k} carried graph state; the control must be the unpadded eager path")
            return v
    tok = {k: recs[k]["passes"][1]["tokens"] for k in ARMS}
    # identity: the candidate against its bitwise oracle, and each self-pair
    v["identity"] = {
        "B1_vs_P": _first_divergence(tok["B1"], tok["P"]),
        "B2_vs_P": _first_divergence(tok["B2"], tok["P"]),
        "B1_vs_B2": _first_divergence(tok["B1"], tok["B2"]),
        "A1_vs_A2": _first_divergence(tok["A1"], tok["A2"]),
        # reported, not decisive: unpadded eager runs other row counts through bf16 GEMMs (PREREG amendment of the memo)
        "B1_vs_A1": _first_divergence(tok["B1"], tok["A1"]),
    }
    idn = v["identity"]
    if idn["A1_vs_A2"] is not None:
        v["verdict"] = "VOID"
        v["reasons"].append(f"the control does not repeat itself: A1 != A2 at {idn['A1_vs_A2']}")
        return v
    if any(idn[k] is not None for k in ("B1_vs_P", "B2_vs_P", "B1_vs_B2")):
        v["verdict"] = "REFUTED"
        v["reasons"].append("the graph replay does not decode as its padded eager step: "
                            f"{ {k: idn[k] for k in ('B1_vs_P', 'B2_vs_P', 'B1_vs_B2') if idn[k] is not None} }")
        return v
    agg = {k: float(recs[k]["timed_agg_tok_s"]) for k in ARMS}
    v["agg_tok_s"] = agg
    r = {"A2/A1": agg["A2"] / agg["A1"], "B2/B1": agg["B2"] / agg["B1"],
         "B1/A1": agg["B1"] / agg["A1"], "B2/A2": agg["B2"] / agg["A2"],
         "P/A1": agg["P"] / agg["A1"], "B1/P": agg["B1"] / agg["P"]}
    v["ratios"] = r
    v["step_ms_by_active"] = {k: recs[k]["timed_mean_step_ms_by_active"] for k in ARMS}
    for sp in ("A2/A1", "B2/B1"):
        if not (1 / BAND <= r[sp] <= BAND):
            v["verdict"] = "VOID"
            v["reasons"].append(f"self-pair {sp} = {r[sp]:.4f} outside [{1 / BAND:.4f}, {BAND}] (drift)")
            return v
    if r["B1/A1"] > BAND and r["B2/A2"] > BAND:
        v["verdict"] = "CONFIRMED"
        v["reasons"].append(f"both pairings exceed {BAND}: B1/A1 {r['B1/A1']:.4f}, B2/A2 {r['B2/A2']:.4f}")
    else:
        v["verdict"] = "REFUTED"
        v["reasons"].append(f"a pairing at or below {BAND}: B1/A1 {r['B1/A1']:.4f}, B2/A2 {r['B2/A2']:.4f}")
    return v


def _load(d: Path) -> dict:
    out = {}
    for k in ARMS:
        f = d / f"p80_{k}.json"
        if f.is_file():
            try:
                out[k] = json.loads(f.read_text())
            except json.JSONDecodeError as e:
                out[k] = {"status": f"unreadable: {e}"}
    return out


def _synthetic(agg: dict, *, tokens_b=None, tokens_a=None, status_b="graph", eager_b=0):
    trace = [16] * 2 + [8] * 2
    base = {"0": [1, 2, 3], "1": [4, 5]}
    recs = {}
    for k in ARMS:
        mode = MODES[k]
        toks = (tokens_b if (k.startswith("B") and tokens_b) else tokens_a if (k.startswith("A") and tokens_a) else base)
        st = ({b: status_b for b in ("1", "2", "4")} if mode == "graph"
              else {b: "eager: capture=False" for b in ("1", "2", "4")} if mode == "padded" else None)
        stats = ({b: {"eager_steps": eager_b} for b in ("1", "2", "4")} if mode == "graph" else None)
        recs[k] = {"mode": mode, "trace": trace, "warm_equals_timed_tokens": True, "graph_status": st,
                   "graph_stats": stats, "timed_agg_tok_s": agg[k], "timed_mean_step_ms_by_active": {"16": 1.0},
                   "passes": [{"tokens": toks}, {"tokens": toks}]}
    return recs


def self_test() -> None:
    ok = {"A1": 100.0, "A2": 101.0, "B1": 150.0, "B2": 151.0, "P": 95.0}
    assert reduce(_synthetic(ok))["verdict"] == "CONFIRMED"
    slow = dict(ok, B1=102.0, B2=103.0)
    assert reduce(_synthetic(slow))["verdict"] == "REFUTED"
    drift = dict(ok, A2=110.0)
    assert reduce(_synthetic(drift))["verdict"] == "VOID"
    # the candidate must decode as its oracle: a B-only token change refutes even when faster
    diverge = _synthetic(ok, tokens_b={"0": [1, 2, 9], "1": [4, 5]})
    got = reduce(diverge)
    assert got["verdict"] == "REFUTED" and got["identity"]["B1_vs_P"] == {"row": 0, "token_index": 2}, got
    # a fallback refutes
    assert reduce(_synthetic(ok, status_b="eager: RuntimeError: x"))["verdict"] == "REFUTED"
    assert reduce(_synthetic(ok, eager_b=3))["verdict"] == "REFUTED"
    # an unpadded-eager difference alone is reported, not decisive (both A arms agree with each other)
    a_only = reduce(_synthetic(ok, tokens_a={"0": [1, 2, 7], "1": [4, 5]}))
    assert a_only["verdict"] == "CONFIRMED" and a_only["identity"]["B1_vs_A1"] is not None, a_only
    # a missing arm is VOID; a candidate OOM refutes
    m = _synthetic(ok)
    del m["P"]
    assert reduce(m)["verdict"] == "VOID"
    o = _synthetic(ok)
    o["B1"] = {"oom": True, "status": "error"}
    assert reduce(o)["verdict"] == "REFUTED"
    print("p80_reduce self-test OK (9 cases)")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir")
    ap.add_argument("--out")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()
    if a.self_test:
        self_test()
        return 0
    if not a.dir or not a.out:
        ap.error("--dir and --out are required")
    v = reduce(_load(Path(a.dir)))
    Path(a.out).write_text(json.dumps(v, indent=1))
    print(f"P80_VERDICT {v['verdict']} " + " | ".join(v["reasons"]))
    for k, x in (v.get("ratios") or {}).items():
        print(f"  {k} = {x:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
