# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""Calculator for PREREG-bv2-curve: refusal gates + the curve + the
preregistered system-425 interpretation line."""

import argparse
import json
import sys

SYS_BAR = 425.0
AA_PCT = 5.0
ANCHOR_MS = 7.39
ANCHOR_TOL = 0.03


def verdict(rep):
    ga = rep.get("graph_anchor_ms")
    if not ga:
        return ("REFUSE", "no graph anchor", None)
    if abs(ga - ANCHOR_MS) / ANCHOR_MS > ANCHOR_TOL:
        return ("REFUSE", f"graph anchor {ga:.2f} ms outside 3% of the "
                f"certified {ANCHOR_MS} class", None)
    curve = []
    ref0 = None
    for b in (1, 2, 4, 8, 16):
        arm = rep.get(f"b{b}")
        if not arm:
            return ("REFUSE", f"B={b} arm missing", None)
        sa, sb = arm.get("step_ms_a"), arm.get("step_ms_b")
        if not sa or not sb:
            return ("REFUSE", f"B={b}: missing A/A step times", None)
        spread = abs(sa - sb) / min(sa, sb) * 100
        if spread > AA_PCT:
            return ("REFUSE", f"B={b}: A/A spread {spread:.1f}% > "
                    f"{AA_PCT}%", None)
        if not arm.get("aa_tokens_identical"):
            return ("REFUSE", f"B={b}: A/A runs not token-identical",
                    None)
        r0 = arm.get("row0_tokens")
        if not r0 or len(set(r0)) < 30:
            return ("REFUSE", f"B={b}: row-0 trace missing or "
                    "degenerate", None)
        if ref0 is None:
            ref0 = r0
        elif r0 != ref0:
            return ("REFUSE", f"B={b}: row-0 continuation differs from "
                    "the B=1 reference -- batch composition changed "
                    "tokens", None)
        step = min(sa, sb)
        curve.append({"B": b, "step_ms": step,
                      "aggregate_tok_s": b * 1000.0 / step})
    best = max(curve, key=lambda c: c["aggregate_tok_s"])
    if best["aggregate_tok_s"] >= SYS_BAR:
        line = (f"SYSTEM-throughput 425 CERTIFIED at B={best['B']} "
                f"({best['aggregate_tok_s']:.1f} tok/s aggregate) -- a "
                "different claim than the REFUTED single-stream 425 "
                "(RESULTS-s3), and reported with that distinction")
    else:
        line = (f"system throughput tops at {best['aggregate_tok_s']:.1f}"
                f" tok/s (B={best['B']}) -- below the 425 line; the "
                "campaign reports the measured maximum and stops "
                "claiming")
    return ("CURVE", line, curve)


def _fab(steps, aa_ok=True, ident=True, anchor=7.39, degenerate=False):
    rep = {"graph_anchor_ms": anchor}
    r0 = list(range(60)) if not degenerate else [1, 2] * 30
    for b, s in steps.items():
        rep[f"b{b}"] = {"step_ms_a": s, "step_ms_b": s * (1.0 if aa_ok
                                                          else 1.10),
                        "aa_tokens_identical": ident,
                        "row0_tokens": list(r0)}
    return rep


def self_test():
    full = {1: 7.6, 2: 8.1, 4: 9.0, 8: 11.5, 16: 17.0}
    v, line, curve = verdict(_fab(full))
    assert v == "CURVE" and "CERTIFIED" in line, (v, line)  # 941 @ B=16
    low = {b: s * 4 for b, s in full.items()}
    v, line, _ = verdict(_fab(low))
    assert v == "CURVE" and "stops claiming" in line, (v, line)
    assert verdict(_fab(full, aa_ok=False))[0] == "REFUSE"
    assert verdict(_fab(full, ident=False))[0] == "REFUSE"
    assert verdict(_fab(full, anchor=8.2))[0] == "REFUSE"
    assert verdict(_fab(full, degenerate=True))[0] == "REFUSE"
    missing = _fab(full)
    del missing["b8"]
    assert verdict(missing)[0] == "REFUSE"
    # cross-B identity: perturb one arm's row-0 trace
    bad = _fab(full)
    bad["b4"]["row0_tokens"] = list(range(60))[::-1]
    assert verdict(bad)[0] == "REFUSE"
    print("self-test PASS (8 cases: certify line, stop-claiming line, "
          "and every refusal incl. cross-B identity)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("report", nargs="?")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()
    if a.self_test:
        self_test()
        return
    if not a.report:
        sys.exit("need a report (or --self-test)")
    v, line, curve = verdict(json.loads(open(a.report).read()))
    print(f"BV2: {v}\n  {line}")
    if curve:
        for c in curve:
            print(f"  B={c['B']:2d}: {c['step_ms']:6.2f} ms/step  "
                  f"{c['aggregate_tok_s']:6.1f} tok/s aggregate")
    if v == "REFUSE":
        sys.exit(2)


if __name__ == "__main__":
    main()
