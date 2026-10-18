#!/usr/bin/env python3
# Copyright (c) 2026 Cerin Amroth LLC. MIT.
"""vsplit_reduce.py -- the verify-split rule (#1313; bench/vsplit/PREREG-vsplit.md): where SD2's 1 -> n row verify
increment goes, decided on ONE card from that card's own measurements.

Every number the rule compares comes from the box that ran it. P132 compared the box against sd2-5090-3's card, and
another RTX 5090 reads 13-15 % off at B = 1 (P116's matched pair), so P132 could only run on that card.

INPUTS (they decide the thresholds and the predictions):
  timing.json, timing_b.json  two unprofiled passes of SD2's V1 terms (p132_box.py --mode timing, sd2_box.v1_terms),
                              in separate processes: the A/A anchors, verify_ms(n), full_ms(k)
  census_n1.json              the decode bucket-1 census: its moe_expert share gives the B = 1 expert time
  TAU_SD1, G_BAR, WORKLOADS   SD2's acceptance and speed bar (bench/sd2/sd2_reduce.py, staged); card-independent
  RHO                         the distinct-expert ratio rho(n), from the four registered routing traces; card-independent
OUTCOME (tested; it never enters a threshold):
  census_n{2,3,4}.json        the verify replays' class SHARES (moe_expert + moe_route) at n = 2..4

Time comes from the unprofiled passes only. The Nsight census gives shares of its own profiled period (terms / P_ms);
a class's ms is its share x the unprofiled step of the same n (the anchor at n = 1, verify_ms at n >= 2).
  anchor     the four A/A anchors' mean (two per pass)
  I_b(n)     verify_ms(n) - anchor (both passes' mean)
  need(n)    the second-smallest over the workloads of full_ms(k) - TAU_SD1[w][k] x anchor / G_BAR, k = n - 1
  bytes(n)   (RHO[n] - 1) x share_moe_expert(1) x anchor        the extra distinct experts' byte cost
  dExp(n)    (share_moe_expert + share_moe_route)(n) x verify_ms(n) - (the same)(1) x anchor
  S(n)       dExp(n) - bytes(n)                                 the expert path's increment beyond its bytes

First rung that applies:
  VOID   a record missing or not ok; the target, kernel package or model revision off the registration; a bucket of
         1..4 not captured; the canary failed; a census arm void or labelled CLASS_MAP_INCOMPLETE,
         CLASS_MAP_SEGMENT_BROKEN, NSYS_DIAGNOSTIC_ERRORS or CLOCK_MISMATCH; a census window not exactly the registered
         replays; in either pass, anchor < verify_ms(2) < verify_ms(3) < verify_ms(4) does not hold. VOID is for record
         and instrument failures only.
  NOISY  in either pass, the A/A anchors differ by more than 3 %; or the two passes (the self-pair) differ by more
         than 3 % on the anchor or on any verify_ms(n).
  READ   otherwise, with two readings:
    H      share(n) = S(n) / [I_b(n) - bytes(n)], the part of the increment beyond its bytes that lies in the expert
           path. At an n where I_b(n) <= bytes(n) there is nothing beyond the bytes: H(n) = NO_EXCESS, and that n is left
           out of the label. Over the remaining n: SUPPORT if share >= 0.70 at every one; REFUTE if share < 0.50 at
           every one; INCONCLUSIVE otherwise. NO_EXCESS at every n gives H = NO_EXCESS (a reading, not a failure).
    BUILD  BUILD a 2-4-row expert kernel iff S(n) >= need(n) for some n; NO_BUILD otherwise (and always when H is
           NO_EXCESS: a kernel has nothing beyond the bytes to save).
REPORTING (never a rung): census-sum vs unprofiled (P_ms / the unprofiled step, the profiler's inflation); each class's
increment as a fraction of I_b(n) beside the registered prediction; the unattributed remainder.

  vsplit_reduce.py --dir D --out verdict_vsplit.json
  vsplit_reduce.py --self-test
"""
import argparse
import json
import math
import os
import sys

E4B_T = "88ae1cfd8fbe3477189f5a42d8448f395e974345"       # SD2's read target: the build the increment was measured on
GNF4_T = "724ccc454f006c1a46836e434e997f31f293747f"      # grouped-nf4-gemm v0.45.0
REV = "ad44e777bcd18fa416d9da3bd8f70d33ebb85d39"
NS = (2, 3, 4)
BUCKETS = (1, 2, 3, 4)
STEPS = 64
# rho(n) = E[distinct experts over n consecutive tokens] / 8, from the four registered routing traces
# (bench/p132/desk_prediction.py; bench/p132/desk_out.json "rho"); tests/test_vsplit.py holds them equal
RHO = {2: 1.6123626569634704, 3: 2.1082912071078432, 4: 2.538318086935167}
ANCHOR_TOL, SELF_PAIR_TOL = 0.03, 0.03
SUPPORT_AT, REFUTE_BELOW = 0.70, 0.50
CLASSES = ("moe_expert", "moe_route", "attn", "dense_gemm", "norm_elem", "sample", "input_prep", "memcpy", "residual")
EXPERT_PATH = ("moe_expert", "moe_route")
BLOCKING = ("CLASS_MAP_INCOMPLETE", "CLASS_MAP_SEGMENT_BROKEN", "NSYS_DIAGNOSTIC_ERRORS", "CLOCK_MISMATCH")


def _sd2():
    """SD2's acceptance and speed bar, from its reducer (staged beside this file on the box; bench/sd2 in a checkout)."""
    here = os.path.dirname(os.path.abspath(__file__))
    for d in (here, os.path.join(here, "..", "sd2")):
        if os.path.exists(os.path.join(d, "sd2_reduce.py")):
            sys.path.insert(0, d)
            break
    import sd2_reduce
    return sd2_reduce


def _load(d, name):
    p = os.path.join(d, name)
    if not os.path.exists(p):
        return None
    with open(p) as f:
        return json.load(f)


def timing_terms(passes):
    """anchor, verify_ms(n), full_ms(k) as the passes' mean; per pass: A/A ratio and the values (for the rungs)."""
    per = []
    for t in passes:
        a, b = t["anchor_a_ms"], t["anchor_b_ms"]
        per.append({"anchor": (a + b) / 2.0, "aa": b / a, "verify": {n: t[f"k{n - 1}"]["verify_ms"] for n in NS},
                    "full": {k: t[f"k{k}"]["full_ms"] for k in (1, 2, 3)}})
    m = len(per)
    anchor = math.fsum(p["anchor"] for p in per) / m
    verify = {n: math.fsum(p["verify"][n] for p in per) / m for n in NS}
    full = {k: math.fsum(p["full"][k] for p in per) / m for k in (1, 2, 3)}
    return anchor, verify, full, per


def need_ms(anchor, full, sd2):
    """need(n): the expert-path saving SD2's speed bar asks for, from this box's own full_ms(k) and anchor."""
    out = {}
    for n in NS:
        k = n - 1
        per_w = sorted(full[k] - sd2.TAU_SD1[w][k] * anchor / sd2.G_BAR for w in sd2.WORKLOADS)
        out[n] = per_w[1]                                 # the second-smallest: FASTER on 2 of 3 workloads
    return out


def shares(census):
    """Each class's share of the census's own profiled period (terms / P_ms)."""
    p = census["P_ms"]
    return {c: census["terms"].get(c, 0.0) / p for c in CLASSES}


def split(anchor, verify, sh):
    """The outcome, converted to ms with the unprofiled steps: per-class increments, dExp, bytes, S."""
    step = {1: anchor, **verify}
    ms = {n: {c: sh[n][c] * step[n] for c in CLASSES} for n in (1,) + NS}
    delta = {n: {c: ms[n][c] - ms[1][c] for c in CLASSES} for n in NS}
    e1 = ms[1]["moe_expert"]
    by = {n: (RHO[n] - 1.0) * e1 for n in NS}
    dexp = {n: math.fsum(delta[n][c] for c in EXPERT_PATH) for n in NS}
    return ms, delta, by, dexp


def h_label(share):
    """The label over the n that carry an excess (``share`` holds only those); NO_EXCESS when none does."""
    vals = list(share.values())
    if not vals:
        return "NO_EXCESS"
    if all(s >= SUPPORT_AT for s in vals):
        return "SUPPORT"
    if all(s < REFUTE_BELOW for s in vals):
        return "REFUTE"
    return "INCONCLUSIVE"


def judge(d):
    """The verdict for the run directory ``d`` (first rung that applies)."""
    void = []
    passes = [_load(d, "timing.json"), _load(d, "timing_b.json")]
    canary, gpu = _load(d, "canary.json"), _load(d, "gpu_tests.json")
    census = {n: _load(d, f"census_n{n}.json") for n in (1,) + NS}
    drv = {(n, m): _load(d, f"census_drv_n{n}_{m}.json") for n in (1,) + NS for m in ("graph", "node")}
    if not gpu or gpu.get("rc") != 0 or gpu.get("failed"):
        void.append("gpu_tests")
    for i, t in enumerate(passes):
        name = ("timing", "timing_b")[i]
        if not t or t.get("status") != "ok":
            void.append(f"{name} missing or not ok")
            continue
        if (t.get("e4b_sha"), t.get("gnf4_sha")) != (E4B_T, GNF4_T):
            void.append(f"{name}: target off the registration: {t.get('e4b_sha')}, {t.get('gnf4_sha')}")
        if t.get("revision") != REV:
            void.append(f"{name}: model revision {t.get('revision')}")
        gs = t.get("graph_status") or {}
        missing = [b for b in BUCKETS if gs.get(str(b)) != "graph"]
        if missing:
            void.append(f"{name}: buckets not captured: {missing}")
    if not canary or canary.get("status") != "ok":
        void.append("canary missing or failed")
    for n, c in census.items():
        if not c or c.get("status") != "ok" or not c.get("P_ms"):
            void.append(f"census n={n} missing or void")
            continue
        bad = [lab for lab in (c.get("labels") or []) if lab in BLOCKING]
        if bad:
            void.append(f"census n={n} labelled {bad}")
    for (n, m), r in drv.items():
        if not r or r.get("steps_bracketed") != STEPS:
            void.append(f"census driver n={n} {m}: not exactly {STEPS} replays bracketed")
    out = {"verdict": None, "void": void, "inputs": {}, "outcome": {}, "reporting": {}}
    timed = all(t and t.get("status") == "ok" for t in passes)
    if timed:
        anchor, verify, full, per = timing_terms(passes)
        for i, p in enumerate(per):
            seq = [p["anchor"]] + [p["verify"][n] for n in NS]
            if not all(x < y for x, y in zip(seq, seq[1:])):
                void.append(f"pass {i}: anchor < verify(2) < verify(3) < verify(4) does not hold: {seq}")
        out["inputs"].update({"anchor_ms": anchor, "verify_ms": verify, "full_ms": full, "I_b": {n: verify[n] - anchor
                              for n in NS}, "need": need_ms(anchor, full, _sd2()), "rho": RHO,
                              "passes": [{"anchor": p["anchor"], "aa": p["aa"], "verify": p["verify"]} for p in per]})
    if not void and timed:
        sh = {n: shares(census[n]) for n in (1,) + NS}
        ms, delta, by, dexp = split(anchor, verify, sh)
        out["inputs"]["share_n1"] = sh[1]
        out["inputs"]["bytes"] = by
    if void:
        out["verdict"] = "VOID"
        return out
    inc, need = out["inputs"]["I_b"], out["inputs"]["need"]
    sav = {n: dexp[n] - by[n] for n in NS}
    excess = [n for n in NS if inc[n] - by[n] > 0]
    share = {n: sav[n] / (inc[n] - by[n]) for n in excess}
    h = h_label(share)
    build = [] if h == "NO_EXCESS" else [n for n in NS if sav[n] >= need[n]]
    out["outcome"] = {"shares": {n: sh[n] for n in NS}, "delta_class": delta, "dExp": dexp, "S": sav, "share": share,
                      "H_by_n": {n: (share[n] if n in share else "NO_EXCESS") for n in NS}, "H": h,
                      "build_at": build, "BUILD": "BUILD" if build else "NO_BUILD"}
    out["reporting"] = {
        "census_over_unprofiled": {n: census[n]["P_ms"] / ({1: anchor, **verify}[n]) for n in (1,) + NS},
        "class_fraction_of_I_b": {n: {c: delta[n][c] / inc[n] for c in CLASSES} for n in NS},
        "unattributed_ms": {n: inc[n] - math.fsum(delta[n].values()) for n in NS},
    }
    per = out["inputs"]["passes"]
    noisy = [f"pass {i}: A/A {p['aa']:.4f}" for i, p in enumerate(per) if abs(p["aa"] - 1) > ANCHOR_TOL]
    a0, a1 = per[0], per[1]
    if abs(a1["anchor"] / a0["anchor"] - 1) > SELF_PAIR_TOL:
        noisy.append(f"self-pair anchor {a1['anchor'] / a0['anchor']:.4f}")
    noisy += [f"self-pair verify(n={n}) {a1['verify'][n] / a0['verify'][n]:.4f}" for n in NS
              if abs(a1["verify"][n] / a0["verify"][n] - 1) > SELF_PAIR_TOL]
    out["noisy"] = noisy
    out["verdict"] = "NOISY" if noisy else "READ"
    out["H"], out["BUILD"], out["build_at"] = out["outcome"]["H"], out["outcome"]["BUILD"], build
    return out


# ---- self-test: synthetic run directories ------------------------------------------------------------------------
_BASE = {"moe_expert": 1.21, "moe_route": 0.51, "attn": 0.71, "dense_gemm": 1.81, "norm_elem": 0.37, "sample": 0.01,
         "input_prep": 0.01, "memcpy": 0.0, "residual": 0.0}
_V1 = {"anchor": (4.1845, 4.1805), "verify": {2: 8.3016, 3: 8.7415, 4: 9.0799},
       "full": {1: 8.9166, 2: 9.7852, 3: 10.5617}}


def _timing(anchor, verify, full, sha=E4B_T, buckets=BUCKETS):
    t = {"status": "ok", "e4b_sha": sha, "gnf4_sha": GNF4_T, "revision": REV, "anchor_a_ms": anchor[0],
         "anchor_b_ms": anchor[1], "graph_status": {str(b): "graph" for b in buckets}}
    for n in NS:
        t[f"k{n - 1}"] = {"verify_ms": verify[n], "full_ms": full[n - 1]}
    return t


def _fake(d, *, terms=None, anchor=None, anchor_b=None, verify=None, verify_b=None, steps=STEPS, labels=(),
          canary="ok", sha=E4B_T, buckets=BUCKETS, gpu_rc=0):
    """A run directory. ``terms`` maps n -> per-class UNPROFILED ms summing to the step; the census records them, and
    its period, inflated x 1.2 (a profiler's uniform inflation), so only the shares carry over."""
    anchor, verify = anchor or _V1["anchor"], verify or _V1["verify"]
    json.dump(_timing(anchor, verify, _V1["full"], sha, buckets), open(os.path.join(d, "timing.json"), "w"))
    json.dump(_timing(anchor_b or anchor, verify_b or verify, _V1["full"], sha, buckets),
              open(os.path.join(d, "timing_b.json"), "w"))
    json.dump({"status": canary}, open(os.path.join(d, "canary.json"), "w"))
    json.dump({"rc": gpu_rc, "failed": 0}, open(os.path.join(d, "gpu_tests.json"), "w"))
    terms = terms or _world(0.9)
    for n in (1,) + NS:
        t = terms[n]
        json.dump({"status": "ok", "labels": list(labels) if n == 2 else [], "terms": {c: 1.2 * v for c, v in t.items()},
                   "P_ms": 1.2 * math.fsum(t.values())}, open(os.path.join(d, f"census_n{n}.json"), "w"))
        for m in ("graph", "node"):
            json.dump({"steps_bracketed": steps}, open(os.path.join(d, f"census_drv_n{n}_{m}.json"), "w"))


def _world(h, anchor=None, verify=None):
    """Unprofiled class ms per n, summing to the step, that put fraction ``h`` of the beyond-bytes increment in the
    expert path: at n = 1 the classes are _BASE scaled to the anchor; at n the expert path gains bytes(n) + h x
    (I_b - bytes) and dense_gemm the rest."""
    anchor = sum(anchor or _V1["anchor"]) / 2
    verify = verify or _V1["verify"]
    tot = math.fsum(_BASE.values())
    t1 = {c: v * anchor / tot for c, v in _BASE.items()}
    out = {1: t1}
    for n in NS:
        by = (RHO[n] - 1.0) * t1["moe_expert"]
        rest = verify[n] - anchor - by
        t = dict(t1)
        t["moe_expert"] += by + h * rest
        t["dense_gemm"] += (1 - h) * rest
        out[n] = t
    return out


def self_test() -> int:
    import tempfile
    cases = []
    sd2 = _sd2()

    def run(name, want, **kw):
        with tempfile.TemporaryDirectory() as d:
            _fake(d, **kw)
            r = judge(d)
        got = {k: r.get(k) for k in want}
        cases.append((name, got == want, got))
        return r

    r = run("an expert-path world reads SUPPORT and BUILD", {"verdict": "READ", "H": "SUPPORT", "BUILD": "BUILD"},
            terms=_world(0.9))
    cases.append(("share(n) is recovered from the shares", all(abs(r["outcome"]["share"][n] - 0.9) < 1e-9 for n in NS),
                  r["outcome"]["share"]))
    cases.append(("need(n) is the box's own formula", all(abs(r["inputs"]["need"][n] - need_ms(
        r["inputs"]["anchor_ms"], r["inputs"]["full_ms"], sd2)[n]) < 1e-12 for n in NS), r["inputs"]["need"]))
    run("a dense world reads REFUTE and NO_BUILD", {"verdict": "READ", "H": "REFUTE", "BUILD": "NO_BUILD"},
        terms=_world(0.2))
    run("a 60 % world is INCONCLUSIVE", {"verdict": "READ", "H": "INCONCLUSIVE"}, terms=_world(0.6))
    # the outcome flips the verdict with every input held fixed (the maintainer's anti-circularity condition)
    with tempfile.TemporaryDirectory() as d:
        _fake(d, terms=_world(0.9))
        hi = judge(d)
    with tempfile.TemporaryDirectory() as d:
        _fake(d, terms={1: _world(0.9)[1], **{n: _world(0.2)[n] for n in NS}})
        lo = judge(d)
    cases.append(("the n >= 2 shares alone flip H and BUILD", hi["inputs"]["need"] == lo["inputs"]["need"]
                  and hi["inputs"]["bytes"] == lo["inputs"]["bytes"] and (hi["H"], hi["BUILD"]) == ("SUPPORT", "BUILD")
                  and (lo["H"], lo["BUILD"]) == ("REFUTE", "NO_BUILD"), (hi["H"], hi["BUILD"], lo["H"], lo["BUILD"])))
    run("A/A anchors 4 % apart read NOISY", {"verdict": "NOISY"}, anchor=(4.10, 4.27))
    run("a self-pair 4 % apart reads NOISY", {"verdict": "NOISY"},
        verify_b={2: 8.3016 * 1.04, 3: 8.7415, 4: 9.0799})
    run("a non-monotone verify is VOID", {"verdict": "VOID"}, verify={2: 8.80, 3: 8.7415, 4: 9.0799})
    run("a blocking census label is VOID", {"verdict": "VOID"}, labels=("CLASS_MAP_INCOMPLETE",))
    run("a short census window is VOID", {"verdict": "VOID"}, steps=STEPS - 1)
    run("an uncaptured bucket is VOID", {"verdict": "VOID"}, buckets=(1, 2, 4))
    run("a failed canary is VOID", {"verdict": "VOID"}, canary="failed")
    run("another target is VOID", {"verdict": "VOID"}, sha="0" * 40)
    run("failed GPU tests are VOID", {"verdict": "VOID"}, gpu_rc=1)
    # an increment that is all bytes is the lane's answer (NO_EXCESS, NO_BUILD), never a VOID
    t1 = _world(0.9)[1]
    e1 = t1["moe_expert"]
    a_mid = sum(_V1["anchor"]) / 2
    small = {n: a_mid + 0.5 * (RHO[n] - 1.0) * e1 for n in NS}
    run("an all-bytes increment reads NO_EXCESS and NO_BUILD", {"verdict": "READ", "H": "NO_EXCESS", "BUILD": "NO_BUILD"},
        verify=small, terms=_world(0.9, verify=small))
    part = {2: small[2], 3: _V1["verify"][3], 4: _V1["verify"][4]}
    r = run("NO_EXCESS at n = 2 leaves H to n = 3, 4", {"verdict": "READ", "H": "SUPPORT"}, verify=part,
            terms=_world(0.9, verify=part))
    cases.append(("H_by_n marks the no-excess n", r["outcome"]["H_by_n"][2] == "NO_EXCESS"
                  and set(r["outcome"]["share"]) == {3, 4}, r["outcome"]["H_by_n"]))
    # a card 15 % faster everywhere reads the same verdict: nothing compares it with another card
    fast_a = tuple(x * 0.85 for x in _V1["anchor"])
    fast_v = {n: v * 0.85 for n, v in _V1["verify"].items()}
    run("a uniformly faster card reads the same", {"verdict": "READ", "H": "SUPPORT"}, anchor=fast_a, verify=fast_v,
        terms=_world(0.9, fast_a, fast_v))
    bad = [(n, g) for n, ok, g in cases if not ok]
    print(f"vsplit_reduce self-test {'OK' if not bad else 'FAILED ' + str(bad)} ({len(cases)} cases)")
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
    print("VSPLIT_VERDICT " + json.dumps({k: r.get(k) for k in ("verdict", "H", "BUILD", "build_at", "void", "noisy")},
                                         default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
