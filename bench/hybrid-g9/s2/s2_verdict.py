# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""Verdict calculator for PREREG-s2lite Stage A."""

import argparse
import json
import sys

ACCEPT = 2.948           # tokens/step, RESULTS-s1-acceptance n1_k16
GO_X = 1.5
REFUTE_X = 1.2


def verdict(rep):
    gate = rep.get("gate") or {}
    if not gate.get("gate_pass"):
        return ("REFUSE", "bitwise gate did not pass: "
                f"rows {gate.get('rows_matching')}/{gate.get('rows_total')}"
                f", continuation_identical="
                f"{gate.get('continuation_identical')} -- no timing is "
                "read past a correctness failure")
    anchor = rep.get("anchor_step_ms")
    aa = rep.get("anchor_aa_spread_ms")
    if not anchor or anchor <= 0:
        return ("REFUSE", "no anchor")
    if aa is None or aa > anchor * (0.5 / 1.5) / 2:
        return ("REFUSE", f"anchor A/A {aa} too wide for the margins "
                "this map decides")
    vm = rep.get("verify_graph_ms")
    if not vm:
        return ("REFUSE", "no graphed verify time")
    if vm < 0.9 * anchor:
        return ("REFUSE", f"verify {vm:.2f} ms < 0.9x anchor "
                f"{anchor:.2f} -- a 17-row step cheaper than a 1-row "
                "step is instrument error")
    x = (ACCEPT / vm) / (1.0 / anchor)
    if x >= GO_X:
        return ("GO-BUILD", f"T_pred {x:.2f}x anchor (verify "
                f"{vm:.2f} ms, anchor {anchor:.2f} ms, acceptance "
                f"{ACCEPT}) >= {GO_X}x -- Stage B with PASS 1.8x")
    if x < REFUTE_X:
        return ("REFUTED-FOR-CELL", f"T_pred {x:.2f}x < {REFUTE_X}x: "
                "the graphed verify is too expensive for this cell")
    return ("INCONCLUSIVE", f"T_pred {x:.2f}x in [{REFUTE_X}, {GO_X}) "
            "-- Stage B proceeds with PASS bar 1.3x")


def _fab(vm, anchor=7.41, aa=0.01, gate=True, rows=17, cont=True):
    return {"gate": {"gate_pass": gate and cont,
                     "rows_matching": rows if gate else rows - 3,
                     "rows_total": 17,
                     "continuation_identical": cont},
            "anchor_step_ms": anchor, "anchor_aa_spread_ms": aa,
            "verify_graph_ms": vm}


def self_test():
    cases = [
        (_fab(9.0), "GO-BUILD"),                    # 2.43x
        (_fab(14.56), "GO-BUILD"),                  # 1.50x boundary
        (_fab(16.0), "INCONCLUSIVE"),               # 1.37x
        (_fab(18.2), "INCONCLUSIVE"),               # 1.20x boundary
        (_fab(19.0), "REFUTED-FOR-CELL"),           # 1.15x
        (_fab(9.0, gate=False), "REFUSE"),
        (_fab(9.0, cont=False), "REFUSE"),
        (_fab(6.0), "REFUSE"),                      # cheaper than anchor
        (_fab(9.0, aa=2.0), "REFUSE"),
        (dict(_fab(9.0), verify_graph_ms=None), "REFUSE"),
        (dict(_fab(9.0), anchor_step_ms=None), "REFUSE"),
    ]
    for rep, want in cases:
        got, why = verdict(rep)
        assert got == want, (got, want, why)
    print(f"self-test PASS ({len(cases)} cases: all three outcomes, "
          "both boundaries, and every refusal)")


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
    v, why = verdict(json.loads(open(a.report).read()))
    print(f"S2-LITE STAGE A VERDICT: {v}\n  {why}")
    if v == "REFUSE":
        sys.exit(2)


if __name__ == "__main__":
    main()
