#!/usr/bin/env python3
# Copyright (c) 2026 Cerin Amroth LLC. MIT.
"""p132_reduce.py -- lane P132's rule (#1313; bench/p132/PREREG-p132.md): where the 1 -> n row increment of SD2's verify goes.

Inputs in --dir:
  timing.json        p132_box.py --mode timing: SD2's V1 timed terms re-run in this process (sd2_box.v1_terms, unprofiled)
  canary.json        p132_box.py --mode canary: buckets 1-4 captured, one verify replay per n, the short census read
  census_n{1..4}.json bench/sc1b/sc1b_census.py arm on the profiled replays (n = 1: decode bucket 1; n = 2..4: verify)
  census_drv_n{N}_{graph,node}.json  p132_box.py --mode census records (the bracket, the build)
  gpu_tests.json     the target's GPU tests

First rung that applies:
  VOID   a record missing or not ok; the target, kernel package or model revision off the registration; a bucket of
         1..4 not captured; the canary failed; a census arm void or labelled CLASS_MAP_INCOMPLETE,
         CLASS_MAP_SEGMENT_BROKEN, NSYS_DIAGNOSTIC_ERRORS or CLOCK_MISMATCH; a census window not exactly the registered
         replays; the re-run anchor or a verify_ms(n) more than 5 % off sd2-5090-3's; the measured increment I_b(n)
         more than 10 % off the desk prediction's target I(n) (the maintainer's tolerance on the INCREMENT).
  NOISY  the re-run's A/A anchors differ by more than 3 % (SD2's ANCHOR_TOL).
  READ   otherwise, with two readings:
    H      share(n) = [dExp(n) - bytes(n) - route_c] / [I_b(n) - known_c(n)], dExp = the census's moe_expert + moe_route
           increment from n = 1 to n. SUPPORT if share >= 0.70 at every n = 2, 3, 4; REFUTE if share < 0.50 at every n;
           INCONCLUSIVE otherwise.
    BUILD  S(n) = dExp(n) - bytes(n), the savable part (the switch's overhead, not its byte cost). BUILD a 2-4-row expert
           kernel iff S(n) >= need(n) for some n; NO_BUILD otherwise (multi-token B = 1 levers close for this stack).
The registered terms (bytes, route_c, known_c, need, the target I) are the desk prediction's v2 numbers, printed by
bench/p132/desk_prediction.py (sha256 71229f08...) into bench/p132/desk_out.json; tests/test_p132.py holds them equal.

  p132_reduce.py --dir D --out verdict_p132.json
  p132_reduce.py --self-test
"""
import argparse
import json
import os
import sys

E4B_T = "88ae1cfd8fbe3477189f5a42d8448f395e974345"       # SD2's read target: the cost the desk predicted
GNF4_T = "724ccc454f006c1a46836e434e997f31f293747f"      # grouped-nf4-gemm v0.45.0
REV = "ad44e777bcd18fa416d9da3bd8f70d33ebb85d39"
NS = (2, 3, 4)
BUCKETS = (1, 2, 3, 4)
STEPS = 64
# sd2-5090-3 stage V1 (bench/sd2/receipts/sd2-5090-3/v.json) and the desk prediction v2 (bench/p132/desk_out.json)
ANCHOR_REG = 4.1825
VERIFY_REG = {2: 8.3016, 3: 8.7415, 4: 9.0799}
I_REG = {2: 4.1191, 3: 4.559, 4: 4.8974}
BYTES = {2: 0.7416997737407249, 3: 1.3423733929610906, 4: 1.8632262500767434}
ROUTE_C = 0.21695295919799806
KNOWN_C = {2: 1.3861183328073745, 3: 2.0211814853610734, 4: 2.5764238758100597}
NEED = {2: 2.358041666666667, 3: 1.6834983333333327, 4: 1.5577733333333335}
VERIFY_TOL, INCREMENT_TOL, ANCHOR_TOL = 0.05, 0.10, 0.03
SUPPORT_AT, REFUTE_BELOW = 0.70, 0.50
CLASSES = ("moe_expert", "moe_route", "attn", "dense_gemm", "norm_elem", "sample", "input_prep", "memcpy", "residual")
BLOCKING = ("CLASS_MAP_INCOMPLETE", "CLASS_MAP_SEGMENT_BROKEN", "NSYS_DIAGNOSTIC_ERRORS", "CLOCK_MISMATCH")


def _load(d, name):
    p = os.path.join(d, name)
    if not os.path.exists(p):
        return None
    with open(p) as f:
        return json.load(f)


def increments(timing):
    """I_b(n) = verify_ms(n) - the A/A anchors' mean, from the re-run (unprofiled) V1 terms; plus the anchor ratio."""
    a, b = timing["anchor_a_ms"], timing["anchor_b_ms"]
    anchor = (a + b) / 2.0
    ver = {n: timing[f"k{n - 1}"]["verify_ms"] for n in NS}
    return anchor, b / a, ver, {n: ver[n] - anchor for n in NS}


def h_label(shares):
    vals = [shares[n] for n in NS]
    if all(s >= SUPPORT_AT for s in vals):
        return "SUPPORT"
    if all(s < REFUTE_BELOW for s in vals):
        return "REFUTE"
    return "INCONCLUSIVE"


def judge(d):
    """The verdict for the run directory ``d`` (first rung that applies)."""
    void = []
    timing, canary, gpu = _load(d, "timing.json"), _load(d, "canary.json"), _load(d, "gpu_tests.json")
    census = {n: _load(d, f"census_n{n}.json") for n in (1,) + NS}
    drv = {(n, m): _load(d, f"census_drv_n{n}_{m}.json") for n in (1,) + NS for m in ("graph", "node")}
    if not gpu or gpu.get("rc") != 0 or gpu.get("failed"):
        void.append("gpu_tests")
    if not timing or timing.get("status") != "ok":
        void.append("timing missing or not ok")
    else:
        if (timing.get("e4b_sha"), timing.get("gnf4_sha")) != (E4B_T, GNF4_T):
            void.append(f"target off the registration: {timing.get('e4b_sha')}, {timing.get('gnf4_sha')}")
        if timing.get("revision") != REV:
            void.append(f"model revision {timing.get('revision')}")
        gs = timing.get("graph_status") or {}
        missing = [b for b in BUCKETS if gs.get(str(b)) != "graph"]
        if missing:
            void.append(f"buckets not captured: {missing}")
    if not canary or canary.get("status") != "ok":
        void.append("canary missing or failed")
    for n, c in census.items():
        if not c or c.get("status") != "ok":
            void.append(f"census n={n} missing or void")
            continue
        bad = [lab for lab in (c.get("labels") or []) if lab in BLOCKING]
        if bad:
            void.append(f"census n={n} labelled {bad}")
    for (n, m), r in drv.items():
        if not r or r.get("steps_bracketed") != STEPS:
            void.append(f"census driver n={n} {m}: not exactly {STEPS} replays bracketed")
    out = {"verdict": None, "void": void}
    if timing and timing.get("status") == "ok":
        anchor, aa, ver, inc = increments(timing)
        out.update({"anchor_ms": anchor, "anchor_ratio": aa, "verify_ms": ver, "I_b": inc})
        if abs(anchor / ANCHOR_REG - 1) > VERIFY_TOL:
            void.append(f"anchor {anchor:.4f} ms is >{VERIFY_TOL:.0%} off sd2-5090-3's {ANCHOR_REG}")
        for n in NS:
            if abs(ver[n] / VERIFY_REG[n] - 1) > VERIFY_TOL:
                void.append(f"verify_ms(n={n}) {ver[n]:.4f} is >{VERIFY_TOL:.0%} off {VERIFY_REG[n]}")
            if abs(inc[n] / I_REG[n] - 1) > INCREMENT_TOL:
                void.append(f"I_b(n={n}) {inc[n]:.4f} is >{INCREMENT_TOL:.0%} off the desk target {I_REG[n]}")
    if void:
        out["verdict"] = "VOID"
        return out
    t1 = census[1]["terms"]
    delta = {n: {c: census[n]["terms"].get(c, 0.0) - t1.get(c, 0.0) for c in CLASSES} for n in NS}
    dexp = {n: delta[n]["moe_expert"] + delta[n]["moe_route"] for n in NS}
    share = {n: (dexp[n] - BYTES[n] - ROUTE_C) / (out["I_b"][n] - KNOWN_C[n]) for n in NS}
    sav = {n: dexp[n] - BYTES[n] for n in NS}
    build = [n for n in NS if sav[n] >= NEED[n]]
    out.update({"delta_class": delta, "dExp": dexp, "share": share, "H": h_label(share), "S": sav,
                "need": NEED, "build_at": build, "BUILD": "BUILD" if build else "NO_BUILD",
                "unattributed": {n: out["I_b"][n] - sum(delta[n].values()) for n in NS}})
    out["verdict"] = "NOISY" if abs(out["anchor_ratio"] - 1) > ANCHOR_TOL else "READ"
    return out


# ---- self-test: synthetic run directories ------------------------------------------------------------------------
def _fake(d, *, anchor=(4.18, 4.185), verify=None, classes=None, steps=STEPS, labels=(), canary="ok", sha=E4B_T,
          buckets=BUCKETS, gpu_rc=0):
    verify = verify or {2: 8.30, 3: 8.74, 4: 9.08}
    timing = {"status": "ok", "e4b_sha": sha, "gnf4_sha": GNF4_T, "revision": REV, "anchor_a_ms": anchor[0],
              "anchor_b_ms": anchor[1], "graph_status": {str(b): "graph" for b in buckets}}
    for n in NS:
        timing[f"k{n - 1}"] = {"verify_ms": verify[n]}
    json.dump(timing, open(os.path.join(d, "timing.json"), "w"))
    json.dump({"status": canary}, open(os.path.join(d, "canary.json"), "w"))
    json.dump({"rc": gpu_rc, "failed": 0}, open(os.path.join(d, "gpu_tests.json"), "w"))
    base = {"moe_expert": 1.21, "moe_route": 0.51, "attn": 0.71, "dense_gemm": 1.81, "norm_elem": 0.37, "sample": 0.01,
            "input_prep": 0.01, "memcpy": 0.0, "residual": 0.0}
    classes = classes or {}
    for n in (1,) + NS:
        terms = dict(base)
        for c, v in (classes.get(n) or {}).items():
            terms[c] = v
        json.dump({"status": "ok", "labels": list(labels) if n == 2 else [], "terms": terms},
                  open(os.path.join(d, f"census_n{n}.json"), "w"))
        for m in ("graph", "node"):
            json.dump({"steps_bracketed": steps}, open(os.path.join(d, f"census_drv_n{n}_{m}.json"), "w"))


def _expert_heavy(scale=1.0):
    """Per n: the expert path carries the increment beyond the receipt-bounded rest (H's world)."""
    out = {}
    for n, inc in ((2, 4.12), (3, 4.56), (4, 4.90)):
        rest = (KNOWN_C[n] - BYTES[n] - ROUTE_C)
        out[n] = {"moe_expert": 1.21 + BYTES[n] + scale * (inc - KNOWN_C[n]), "moe_route": 0.51 + ROUTE_C,
                  "dense_gemm": 1.81 + rest}
    return out


def _dense_heavy():
    out = {}
    for n, inc in ((2, 4.12), (3, 4.56), (4, 4.90)):
        out[n] = {"moe_expert": 1.21 + BYTES[n], "moe_route": 0.51 + ROUTE_C, "dense_gemm": 1.81 + (inc - BYTES[n] - ROUTE_C)}
    return out


def self_test() -> int:
    import tempfile
    cases = []

    def run(name, want, **kw):
        with tempfile.TemporaryDirectory() as d:
            _fake(d, **kw)
            r = judge(d)
        got = {k: r.get(k) for k in want}
        cases.append((name, got == want, got))

    run("H's world reads SUPPORT and BUILD", {"verdict": "READ", "H": "SUPPORT", "BUILD": "BUILD"},
        classes=_expert_heavy())
    run("the dense world reads REFUTE and NO_BUILD", {"verdict": "READ", "H": "REFUTE", "BUILD": "NO_BUILD"},
        classes=_dense_heavy())
    run("a 60 % expert world is INCONCLUSIVE", {"verdict": "READ", "H": "INCONCLUSIVE"}, classes=_expert_heavy(0.6))
    run("noisy A/A anchors read NOISY", {"verdict": "NOISY"}, anchor=(4.10, 4.25), classes=_expert_heavy(),
        verify={2: 8.30, 3: 8.74, 4: 9.08})
    run("a verify off sd2-5090-3 by > 5 % is VOID", {"verdict": "VOID"}, verify={2: 8.90, 3: 8.74, 4: 9.08})
    # anchor +4.7 % and verify(2) -4.2 %, each inside 5 %, move I_b(2) by -13 %: the increment check alone voids it
    run("an increment off the desk target by > 10 % is VOID", {"verdict": "VOID"},
        anchor=(4.38, 4.38), verify={2: 7.95, 3: 8.74, 4: 9.08})
    run("a blocking census label is VOID", {"verdict": "VOID"}, labels=("CLASS_MAP_INCOMPLETE",))
    run("a short census window is VOID", {"verdict": "VOID"}, steps=STEPS - 1)
    run("an uncaptured bucket is VOID", {"verdict": "VOID"}, buckets=(1, 2, 4))
    run("a failed canary is VOID", {"verdict": "VOID"}, canary="failed")
    run("another target is VOID", {"verdict": "VOID"}, sha="0" * 40)
    run("failed GPU tests are VOID", {"verdict": "VOID"}, gpu_rc=1)
    # the build line is S(n) >= need(n) at some n, independent of H
    with tempfile.TemporaryDirectory() as d:
        _fake(d, classes=_expert_heavy())
        r = judge(d)
    cases.append(("S(n) is dExp - bytes", all(abs(r["S"][n] - (r["dExp"][n] - BYTES[n])) < 1e-12 for n in NS), r["S"]))
    cases.append(("need is the registered threshold", r["need"] == NEED, r["need"]))
    bad = [(n, g) for n, ok, g in cases if not ok]
    print(f"p132_reduce self-test {'OK' if not bad else 'FAILED ' + str(bad)} ({len(cases)} cases)")
    return 0 if not bad else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir")
    ap.add_argument("--out")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args(argv)
    if a.self_test:
        return self_test()
    r = judge(a.dir)
    json.dump(r, open(a.out, "w"), indent=1, sort_keys=True, default=str)
    print("P132_VERDICT " + json.dumps({k: r.get(k) for k in ("verdict", "H", "BUILD", "build_at", "void")}, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
