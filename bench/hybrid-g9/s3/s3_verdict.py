# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""Verdict calculator for PREREG-s3-grouped-verify."""

import argparse
import json
import sys

# tokens/step from receipts-s2/s1_alpha_kext.json (same traces + same
# simulator as RESULTS-s1-acceptance; K>16 under-estimated)
ACCEPT = {16: 2.948, 32: 3.447, 64: 3.926}
GO_X = 1.5
REFUTE_X = 1.2


def verdict(rep):
    for g in ("grouping_parity", "numeric_parity"):
        gg = rep.get(g) or {}
        if not gg.get("pass"):
            return ("REFUSE", f"{g} gate did not pass -- timing an "
                    "unverified grouped path certifies nothing")
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
    vms = rep.get("verify_graph_ms")
    if isinstance(vms, (int, float)):          # legacy single-K shape
        vms = {"16": vms}
    if not vms:
        return ("REFUSE", "no graphed verify times")
    best = None
    for kstr, vm in vms.items():
        k = int(kstr)
        if k not in ACCEPT:
            return ("REFUSE", f"no committed acceptance for K={k}")
        if not vm:
            return ("REFUSE", f"empty verify time for K={k}")
        if vm < 0.9 * anchor:
            return ("REFUSE", f"verify({k}) {vm:.2f} ms < 0.9x anchor "
                    f"{anchor:.2f} -- a {k+1}-row step cheaper than a "
                    "1-row step is instrument error")
        x = (ACCEPT[k] / vm) / (1.0 / anchor)
        if best is None or x > best[1]:
            best = (k, x, vm)
    k, x, vm = best
    if x >= GO_X:
        return ("GO", f"K={k}: T_pred {x:.2f}x anchor (verify "
                f"{vm:.2f} ms, anchor {anchor:.2f} ms, acceptance "
                f"{ACCEPT[k]}) >= {GO_X}x -- Stage B on this cell, "
                "PASS 1.8x")
    if x < REFUTE_X:
        return ("REFUTED", f"best K={k}: T_pred {x:.2f}x < "
                f"{REFUTE_X}x: the graphed verify is too expensive at "
                "every measured K")
    return ("INCONCLUSIVE-LIMITED", f"best K={k}: T_pred {x:.2f}x in "
            f"[{REFUTE_X}, {GO_X}) -- Stage B on this cell with PASS "
            "bar 1.3x")


def _fab(vm, anchor=7.41, aa=0.01, gate=True, rows=17, cont=True,
         gp=True, np_=True):
    if isinstance(vm, (int, float)):
        vm = {"16": vm}
    return {"grouping_parity": {"pass": gp},
            "numeric_parity": {"pass": np_},
            "gate": {"gate_pass": gate and cont,
                     "rows_matching": rows if gate else rows - 3,
                     "rows_total": 17,
                     "continuation_identical": cont},
            "anchor_step_ms": anchor, "anchor_aa_spread_ms": aa,
            "verify_graph_ms": vm}


def self_test():
    cases = [
        (_fab(9.0), "GO"),                    # 2.43x
        (_fab(14.56), "GO"),                  # 1.50x boundary
        (_fab(16.0), "INCONCLUSIVE-LIMITED"),               # 1.37x
        (_fab(18.2), "INCONCLUSIVE-LIMITED"),               # 1.20x boundary
        (_fab(19.0), "REFUTED"),           # 1.15x
        (_fab(9.0, gate=False), "REFUSE"),
        (_fab(9.0, cont=False), "REFUSE"),
        (_fab(9.0, gp=False), "REFUSE"),
        (_fab(9.0, np_=False), "REFUSE"),
        (_fab(6.0), "REFUSE"),                      # cheaper than anchor
        (_fab(9.0, aa=2.0), "REFUSE"),
        (dict(_fab(9.0), verify_graph_ms=None), "REFUSE"),
        (dict(_fab(9.0), anchor_step_ms=None), "REFUSE"),
        # K-sweep: the best cell decides. K=64 at 9.0 ms is 3.23x even
        # while K=16 at 16.0 would be INCONCLUSIVE alone
        (_fab({"16": 16.0, "64": 9.0}), "GO"),
        # unknown K refuses (no committed acceptance)
        (_fab({"12": 9.0}), "REFUSE"),
        # any cell cheaper than anchor is instrument error even if
        # another cell would win
        (_fab({"16": 9.0, "64": 6.0}), "REFUSE"),
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
    print(f"S3 VERDICT: {v}\n  {why}")
    if v == "REFUSE":
        sys.exit(2)


if __name__ == "__main__":
    main()
