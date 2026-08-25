# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""Verdict calculator for PREREG-s1-acceptance. Two-sided bounds: GO
decides on the conservative (eager upper-bound) verify cost, REFUTED
decides on the optimistic (anchor lower-bound) cost. Self-test before
receipts."""

import argparse
import json
import sys

GO_X = 1.5
REFUTE_X = 1.1


def verdict(rep):
    alpha = rep.get("alpha") or {}
    if "refuse" in alpha:
        return ("REFUSE", f"simulator refused: {alpha['refuse']}")
    anchor = rep.get("anchor_step_ms")
    if not anchor or anchor <= 0:
        return ("REFUSE", "no anchor step time")
    verify = rep.get("verify_ms") or {}
    cells = alpha.get("cells") or {}
    if not cells:
        return ("REFUSE", "no acceptance cells")
    best = None
    for name, c in cells.items():
        if "refuse" in c:
            return ("REFUSE", f"cell {name}: {c['refuse']}")
        k = int(name.split("_k")[1])
        vm = verify.get(str(k)) or verify.get(k)
        if vm is None:
            return ("REFUSE", f"no verify_ms for K={k}")
        if vm < 0.9 * anchor:
            return ("REFUSE", f"verify_ms({k})={vm:.2f} < 0.9x anchor "
                    f"{anchor:.2f} -- a bigger step cannot be cheaper")
        tps = c["tokens_per_step"]
        cons = (tps / vm) / (1.0 / anchor)       # x over anchor tok/s
        opt = (tps / anchor) / (1.0 / anchor)    # = tps
        row = {"cell": name, "k": k, "tokens_per_step": tps,
               "verify_ms": vm, "x_conservative": cons,
               "x_optimistic": opt}
        if best is None or cons > best["x_conservative"]:
            best = row
    max_opt = max((c["tokens_per_step"] for c in cells.values()))
    if best["x_conservative"] >= GO_X:
        return ("GO", f"{best['cell']}: {best['x_conservative']:.2f}x "
                f"conservative (tokens/step {best['tokens_per_step']:.2f}"
                f", verify {best['verify_ms']:.2f} ms vs anchor "
                f"{anchor:.2f}) >= {GO_X}x -- register S2 with bars "
                "tied to this prediction")
    if max_opt < REFUTE_X:
        return ("REFUTED-FOR-DRAFTER",
                f"best OPTIMISTIC bound {max_opt:.2f}x < {REFUTE_X}x: "
                "prompt-lookup cannot pay even with a free verify step; "
                "a draft-model variant needs a fresh prereg")
    return ("MARGINAL", f"best conservative {best['x_conservative']:.2f}x "
            f"(cell {best['cell']}), best optimistic {max_opt:.2f}x -- "
            "register S2-lite for the single best cell only")


def _fab(tps_by_cell, verify, anchor=7.5, refuse_cell=None,
         sim_refuse=None):
    cells = {}
    for name, tps in tps_by_cell.items():
        c = {"windows": 4000, "tokens_per_step": tps, "mean_m": tps - 1}
        if refuse_cell == name:
            c["refuse"] = "3 windows < 3000"
        cells[name] = c
    a = {"cells": cells}
    if sim_refuse:
        a["refuse"] = sim_refuse
    return {"alpha": a, "verify_ms": verify, "anchor_step_ms": anchor}


def self_test():
    cases = [
        # 2.0 tokens/step against verify 1.2x anchor -> 1.67x cons -> GO
        (_fab({"n1_k4": 2.0}, {"4": 9.0}), "GO"),
        # exactly at the GO boundary: tps 1.8 / (9.0/7.5) = 1.5
        (_fab({"n1_k4": 1.8}, {"4": 9.0}), "GO"),
        # tps 1.05 -> optimistic 1.05 < 1.1 even free -> REFUTED
        (_fab({"n1_k4": 1.05}, {"4": 9.0}), "REFUTED-FOR-DRAFTER"),
        # tps 1.3, verify 1.5x anchor -> cons 0.87, opt 1.3 -> MARGINAL
        (_fab({"n1_k4": 1.3}, {"4": 11.25}), "MARGINAL"),
        # the best cell drives GO even when another is weak
        (_fab({"n1_k4": 1.05, "n1_k8": 2.4}, {"4": 9.0, "8": 10.0}),
         "GO"),
        # refusals
        (_fab({"n1_k4": 2.0}, {"4": 5.0}), "REFUSE"),   # cheaper than anchor
        (_fab({"n1_k4": 2.0}, {}), "REFUSE"),           # missing verify
        (_fab({"n1_k4": 2.0}, {"4": 9.0},
              refuse_cell="n1_k4"), "REFUSE"),
        (_fab({"n1_k4": 2.0}, {"4": 9.0},
              sim_refuse="only 3 traces"), "REFUSE"),
        (dict(_fab({"n1_k4": 2.0}, {"4": 9.0}),
              anchor_step_ms=None), "REFUSE"),
    ]
    for rep, want in cases:
        got, why = verdict(rep)
        assert got == want, (got, want, why)
    print(f"self-test PASS ({len(cases)} cases: GO incl. boundary, "
          "REFUTED on the optimistic side, MARGINAL between, and every "
          "refusal)")


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
    print(f"S1 VERDICT: {v}\n  {why}")
    if v == "REFUSE":
        sys.exit(2)


if __name__ == "__main__":
    main()
