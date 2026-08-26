# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""Verdict calculator for PREREG-bv3b-parity.

Report shape:
  parity: the --grouping-parity receipt
    ({"parity": {layer: {max_abs_delta, max_abs_ref}}, "layers": N}).
  bv3: the BV3 report (receipts-bv3/bv3_report.json, verbatim).

Frame verbatim from the prereg: parity must hold on EVERY layer
(max|delta| <= max|ref| * 2^-7; any exceedance is the defect case and
REFUSES). Under a green frame the BV3 wall receipts re-adjudicate at
BV3's own bars with the kernel-swap identity frame: per-arm
determinism (graph_a == graph_b, eager_a == eager_b, every row),
no-crossing (every graph row's best-match eager row is itself),
degeneracy law on every stream. Token equality vs eager is NOT
required -- that is the point of the re-derivation.
"""

import argparse
import json
import sys
from collections import Counter

REL_BAR = 2.0 ** -7
PASS_RATIO = 1.5
PARTIAL_RATIO = 1.2
B1_CLASS_MS = 7.16
B1_TOL = 0.05
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
    par = rep.get("parity")
    bv3 = rep.get("bv3")
    if not par or not bv3:
        return ("REFUSE", "missing parity or bv3 receipts")
    cells = par.get("parity") or {}
    n_layers = par.get("layers")
    if not n_layers or len(cells) != n_layers:
        return ("REFUSE", f"parity covers {len(cells)} of "
                f"{n_layers!r} layers -- refusing a partial probe")
    worst_ratio, worst_layer = 0.0, None
    for li, c in cells.items():
        bar = c["max_abs_ref"] * REL_BAR
        if c["max_abs_delta"] > bar:
            return ("REFUSE", f"parity DEFECT: layer {li} max|delta| "
                    f"{c['max_abs_delta']:g} > max|ref|*2^-7 = {bar:g}"
                    " -- the lane is blocked on this, not on tokens")
        r = c["max_abs_delta"] / max(c["max_abs_ref"], 1e-30)
        if r >= worst_ratio:
            worst_ratio, worst_layer = r, li

    ea, eb = bv3["eager_a"], bv3["eager_b"]
    ga, gb = bv3["graph_a"], bv3["graph_b"]
    for name, x, y in (("graph", ga["tokens"], gb["tokens"]),
                       ("eager", ea["generated_tokens"],
                        eb["generated_tokens"])):
        for row in x:
            if x[row] != y.get(row):
                return ("REFUSE", f"{name} arms not internally "
                        f"deterministic on row {row}")
    e_tok, g_tok = ea["generated_tokens"], ga["tokens"]
    for row in g_tok:
        why = _degenerate(g_tok[row])
        if why:
            return ("REFUSE", f"graph row {row} degenerate: {why}")
        # crossing iff some OTHER row matches STRICTLY longer than the
        # own row -- max() with first-key tie-break falsely refused
        # legitimate ties (review, e4b#269)
        p_own = _prefix(g_tok[row], e_tok[row]) if row in e_tok else -1
        p_other = max((_prefix(g_tok[row], e_tok[k])
                       for k in e_tok if k != row), default=-1)
        if p_other > p_own:
            return ("REFUSE", f"crossing: graph row {row} matches "
                    f"another eager row longer ({p_other} > {p_own})")

    e_ms = [ea["decode_median_ms"]["step"], eb["decode_median_ms"]["step"]]
    g_ms = [ga["step_ms_clean"], gb["step_ms_clean"]]
    B = ga["batch"]
    e_base, g_base = min(e_ms), min(g_ms)
    e_spread = abs(e_ms[0] - e_ms[1])
    margin_e = e_base * (1 - 1 / PASS_RATIO) / 2
    if e_spread > margin_e + EPS:
        return ("REFUSE", f"eager A/A spread {e_spread:.2f} ms wider "
                "than half the PASS margin")
    if abs(g_ms[0] - g_ms[1]) > (g_base * (PASS_RATIO - 1) / 2) + EPS:
        return ("REFUSE", "graph A/A spread wider than half the PASS "
                "margin")
    b1 = bv3["graph_b1"]["step_ms_clean"]
    lo, hi = B1_CLASS_MS * (1 - B1_TOL), B1_CLASS_MS * (1 + B1_TOL)
    if not (lo <= b1 <= hi):
        return ("REFUSE", f"B=1 sanity {b1:.2f} ms outside "
                f"[{lo:.2f}, {hi:.2f}]")

    ratio = e_base / g_base
    detail = (f"parity worst ratio {worst_ratio:.3e} (layer "
              f"{worst_layer}) inside 2^-7={REL_BAR:.3e}; graph "
              f"{g_base:.1f} vs eager {e_base:.1f} ms at B={B} = "
              f"{ratio:.2f}x ({1000/e_base*B:.0f} -> "
              f"{1000/g_base*B:.0f} agg tok/s); b1 {b1:.2f} ms")
    if ratio >= PASS_RATIO - EPS:
        return ("PASS", detail + " -- the batched graph loop ships; "
                "divergence-vs-eager is certified reorder-class")
    if ratio >= PARTIAL_RATIO - EPS:
        return ("PARTIAL", detail + " -- ships with disclosure")
    return ("REFUTED", detail)


# ---------------------------------------------------------------- self-test
def _mk(worst=1e-4, layers=48, e=129.4, g=38.0, b1=7.16, div=None,
        cross=False, nondet=False):
    cells = {str(i): {"max_abs_delta": worst * 10.0,
                      "max_abs_ref": 10.0} for i in range(layers)}
    par = {"parity": cells, "layers": layers}
    base = list(range(300, 428))
    B = 4

    def toks(shift, cut=None):
        t = [x + shift for x in base]
        if cut is not None:
            t[cut:] = [x + 90000 for x in t[cut:]]
        return t

    et = {str(i): toks(i) for i in range(B)}
    gt = {str(i): toks(i, cut=div) for i in range(B)}
    if cross:
        gt["0"], gt["1"] = gt["1"], gt["0"]
    gtb = {k: (list(v) if not nondet or k != "2" else v[:-1] + [7])
           for k, v in gt.items()}
    bv3 = {"eager_a": {"decode_median_ms": {"step": e},
                       "generated_tokens": et},
           "eager_b": {"decode_median_ms": {"step": e + 0.4},
                       "generated_tokens": {k: list(v)
                                            for k, v in et.items()}},
           "graph_a": {"step_ms_clean": g, "batch": B, "tokens": gt},
           "graph_b": {"step_ms_clean": g + 0.1, "batch": B,
                       "tokens": gtb},
           "graph_b1": {"step_ms_clean": b1}}
    return {"parity": par, "bv3": bv3}


def _self_test():
    v = verdict
    # the receipts case: early divergence vs eager, parity green -> PASS
    out = v(_mk(div=3))
    assert out[0] == "PASS", out
    assert v(_mk())[0] == "PASS"
    # one layer outside the frame refuses as DEFECT
    r = _mk()
    r["parity"]["parity"]["17"]["max_abs_delta"] = 10.0 * REL_BAR * 1.01
    out = v(r)
    assert out[0] == "REFUSE" and "DEFECT" in out[1] and "17" in out[1], out
    # boundary: exactly at the bar passes
    r = _mk()
    r["parity"]["parity"]["17"]["max_abs_delta"] = 10.0 * REL_BAR
    assert v(r)[0] == "PASS"
    # partial probe refuses
    r = _mk()
    del r["parity"]["parity"]["5"]
    out = v(r)
    assert out[0] == "REFUSE" and "partial" in out[1], out
    # crossing refuses (rows swapped -> another row matches longer)
    out = v(_mk(cross=True))
    assert out[0] == "REFUSE" and "crossing" in out[1], out
    # equal-length ties with an earlier key must NOT refuse: make rows
    # 0 and 1 IDENTICAL streams in both arms (own-row tie) -- the old
    # first-key max() called this a crossing
    r = _mk()
    for arm in ("eager_a", "eager_b"):
        tk = r["bv3"][arm]["generated_tokens"]
        tk["1"] = list(tk["0"])
    for arm in ("graph_a", "graph_b"):
        tk = r["bv3"][arm]["tokens"]
        tk["1"] = list(tk["0"])
    out = v(r)
    assert out[0] == "PASS", out
    # non-deterministic graph arm refuses
    out = v(_mk(nondet=True))
    assert out[0] == "REFUSE" and "deterministic" in out[1], out
    # degenerate stream refuses
    r = _mk()
    r["bv3"]["graph_a"]["tokens"]["0"] = [1, 2] * 64
    r["bv3"]["graph_b"]["tokens"]["0"] = [1, 2] * 64
    out = v(r)
    assert out[0] == "REFUSE" and "degenerate" in out[1], out
    # tiers + b1 sanity
    assert v(_mk(g=129.4 / 1.5))[0] == "PASS"
    assert v(_mk(g=129.4 / 1.49))[0] == "PARTIAL"
    assert v(_mk(g=129.4 / 1.1))[0] == "REFUTED"
    for bad in (6.7, 7.6):
        assert v(_mk(b1=bad))[0] == "REFUSE"
    # missing pieces
    r = _mk()
    del r["bv3"]
    assert v(r)[0] == "REFUSE"
    print("bv3b_verdict self-test: OK")


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
    print(f"BV3B VERDICT: {v}\n{why}")
    if v == "REFUSE":
        sys.exit(2)


if __name__ == "__main__":
    main()
