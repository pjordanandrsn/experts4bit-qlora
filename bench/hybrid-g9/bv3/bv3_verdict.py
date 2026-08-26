# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""Verdict calculator for PREREG-bv3-graph-batch.

Report shape:
  eager_a/eager_b: BV2-style scheduler receipts at B
    ({"decode_median_ms": {"step": ms}, "generated_tokens": {row: [...]}}).
  graph_a/graph_b: _bv3_stage receipts
    ({"step_ms_clean", "batch", "tokens": {row: [...]}}).
  graph_b1: b1d graph receipt ({"step_ms_clean"}).

Bars verbatim from the prereg: PASS = graphed aggregate >= 1.5x the
fresh eager aggregate AND per-row identity (K6-B frame: equal length,
first divergence >= 32) AND graph_b1 within +/-5% of the certified
single-stream class; PARTIAL >= 1.2x; REFUTED below. REFUSE: A/A
spread (either pair) wider than half the PASS margin, identity
breach, degeneracy, missing pieces.
"""

import argparse
import json
import sys
from collections import Counter

PASS_RATIO = 1.5
PARTIAL_RATIO = 1.2
B1_CLASS_MS = 7.16          # current certified single-stream class
B1_TOL = 0.05
MIN_DIVERGE_STEP = 32
MIN_OVERLAP = 48          # aligned comparison must cover at least this
GRAM, MAX_REP = 8, 6
EPS = 1e-9


def _degenerate(t):
    if len(t) >= 64 and len(set(t)) < 30:
        return f"only {len(set(t))} distinct tokens"
    grams = Counter(tuple(t[i:i + GRAM]) for i in range(len(t) - GRAM + 1))
    worst = max(grams.values(), default=0)
    if worst > MAX_REP:
        return f"an {GRAM}-gram repeats {worst}x"
    return None


def _prefix(x, y):
    n = min(len(x), len(y))
    for i in range(n):
        if x[i] != y[i]:
            return i
    return n


def verdict(rep):
    for k in ("eager_a", "eager_b", "graph_a", "graph_b", "graph_b1"):
        if not rep.get(k):
            return ("REFUSE", f"missing {k!r}")
    ea, eb = rep["eager_a"], rep["eager_b"]
    ga, gb = rep["graph_a"], rep["graph_b"]
    e_ms = [ea["decode_median_ms"]["step"], eb["decode_median_ms"]["step"]]
    g_ms = [ga["step_ms_clean"], gb["step_ms_clean"]]
    B = ga["batch"]
    if gb.get("batch") != B:
        return ("REFUSE", "graph arms disagree on batch")
    e_base, e_spread = min(e_ms), abs(e_ms[0] - e_ms[1])
    g_base, g_spread = min(g_ms), abs(g_ms[0] - g_ms[1])
    ratio = e_base / g_base            # aggregate ratio == step ratio at same B
    # A/A: each pair's spread must not exceed half the PASS margin,
    # expressed in its own arm's step units
    margin_e = e_base * (1 - 1 / PASS_RATIO) / 2
    if e_spread > margin_e + EPS:
        return ("REFUSE", f"eager A/A spread {e_spread:.2f} ms wider "
                f"than half the PASS margin ({margin_e:.2f})")
    if g_spread > (g_base * (PASS_RATIO - 1) / 2) + EPS:
        return ("REFUSE", f"graph A/A spread {g_spread:.2f} ms wider "
                f"than half the PASS margin")
    b1 = rep["graph_b1"]["step_ms_clean"]
    lo, hi = B1_CLASS_MS * (1 - B1_TOL), B1_CLASS_MS * (1 + B1_TOL)
    if not (lo <= b1 <= hi):
        return ("REFUSE", f"B=1 sanity: {b1:.2f} ms outside "
                f"[{lo:.2f}, {hi:.2f}] -- the generalized loop broke "
                "the certified single-stream class")

    e_tok = ea["generated_tokens"]
    g_tok = ga["tokens"]
    if set(g_tok) - set(e_tok):
        return ("REFUSE", "graph rows without matching eager rows")
    idmsg = []
    for row in sorted(g_tok):
        # graph receipts carry the FULL stream (pre + warm + window);
        # both arms are the same prompts from decode step 0, so the
        # comparison is aligned by construction. Bugbot (e4b#261): a
        # short full-match must NOT pass -- require a real overlap, and
        # the divergence floor applies inside it.
        gt, et = g_tok[row], e_tok[row]
        why = _degenerate(gt)
        if why:
            return ("REFUSE", f"graph row {row} degenerate: {why}")
        n = min(len(gt), len(et))
        if n < MIN_OVERLAP:
            return ("REFUSE", f"row {row}: aligned overlap {n} < "
                    f"{MIN_OVERLAP} -- streams too short to certify "
                    f"(graph {len(gt)}, eager {len(et)})")
        p = _prefix(et, gt)
        if p < MIN_DIVERGE_STEP:
            return ("REFUSE", f"row {row} diverges at step {p} < "
                    f"{MIN_DIVERGE_STEP}")
        idmsg.append(f"{row}:{'id' if p == n else f'div@{p}'}")

    agg_e = 1000.0 / e_base * B
    agg_g = 1000.0 / g_base * B
    detail = (f"graph {g_base:.1f} vs eager {e_base:.1f} ms/step at "
              f"B={B} = {ratio:.2f}x ({agg_e:.1f} -> {agg_g:.1f} "
              f"agg tok/s); b1 sanity {b1:.2f} ms; identity "
              f"{','.join(idmsg)}")
    if ratio >= PASS_RATIO - EPS:
        return ("PASS", detail + " -- ships --b1d-loop graph honoring "
                "--batch")
    if ratio >= PARTIAL_RATIO - EPS:
        return ("PARTIAL", detail + " -- ships with disclosure")
    return ("REFUTED", detail + " -- the batch wall is not the host; "
            "record the decomposition")


# ---------------------------------------------------------------- self-test
def _mk(e=129.4, g=60.0, b1=7.16, n=64, div=None, B=4):
    base = list(range(200, 200 + n))

    def toks(shift=0, cut=None):
        t = [x + shift for x in base]
        if cut is not None:
            t[cut:] = [x + 90000 for x in t[cut:]]
        return t

    eager = {"decode_median_ms": {"step": e},
             "generated_tokens": {str(i): toks(i) for i in range(B)}}
    eager_b = {"decode_median_ms": {"step": e + 0.5},
               "generated_tokens": {str(i): toks(i) for i in range(B)}}
    graph = {"step_ms_clean": g, "batch": B,
             "tokens": {str(i): toks(i, cut=div) for i in range(B)}}
    graph_b = {"step_ms_clean": g + 0.2, "batch": B,
               "tokens": {str(i): toks(i, cut=div) for i in range(B)}}
    return {"eager_a": eager, "eager_b": eager_b,
            "graph_a": graph, "graph_b": graph_b,
            "graph_b1": {"step_ms_clean": b1}}


def _self_test():
    v = verdict
    out = v(_mk())                              # 129.4/60 = 2.16x
    assert out[0] == "PASS", out
    # boundary: exactly 1.5x passes; just under is PARTIAL
    assert v(_mk(g=129.4 / 1.5))[0] == "PASS"
    assert v(_mk(g=129.4 / 1.49))[0] == "PARTIAL"
    assert v(_mk(g=129.4 / 1.21))[0] == "PARTIAL"
    out = v(_mk(g=129.4 / 1.1))
    assert out[0] == "REFUTED" and "not the host" in out[1], out
    # b1 sanity two-sided
    for bad in (6.7, 7.6):
        out = v(_mk(b1=bad))
        assert out[0] == "REFUSE" and "sanity" in out[1], out
    # divergence: at 31 refuses, at 32 allowed
    out = v(_mk(div=31))
    assert out[0] == "REFUSE" and "diverges" in out[1], out
    assert v(_mk(div=32))[0] == "PASS"
    # short full-match refuses on overlap (Bugbot: a 10-token match
    # must not certify)
    out = v(_mk(n=10))
    assert out[0] == "REFUSE" and "overlap" in out[1], out
    out = v(_mk(n=47))
    assert out[0] == "REFUSE" and "overlap" in out[1], out
    assert v(_mk(n=48))[0] == "PASS"
    # eager A/A too wide refuses (margin = 129.4*(1-1/1.5)/2 = 21.6)
    r = _mk()
    r["eager_b"]["decode_median_ms"]["step"] = 129.4 + 22.0
    out = v(r)
    assert out[0] == "REFUSE" and "eager A/A" in out[1], out
    # degenerate graph stream refuses
    r = _mk()
    r["graph_a"]["tokens"]["0"] = [5, 6] * 32
    out = v(r)
    assert out[0] == "REFUSE" and "degenerate" in out[1], out
    # batch mismatch between graph arms refuses
    r = _mk()
    r["graph_b"]["batch"] = 8
    assert v(r)[0] == "REFUSE"
    # missing piece refuses
    r = _mk()
    del r["graph_b1"]
    assert v(r)[0] == "REFUSE"
    print("bv3_verdict self-test: OK")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("report", nargs="?")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()
    if a.self_test:
        _self_test()
        return
    rep = json.load(open(a.report))
    v, why = verdict(rep)
    print(f"BV3 VERDICT: {v}\n{why}")
    if v == "REFUSE":
        sys.exit(2)


if __name__ == "__main__":
    main()
