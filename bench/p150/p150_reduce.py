#!/usr/bin/env python3
"""p150_reduce.py -- lane P150's registered rule (bench/p150/PREREG-p150.md; e4b#1313): D2's QUALITY ruling.

P148 read D2, an int8 lm_head, GO in a microbench (a premise, never a gain); its served registration must rule on
quality first. The candidate is I8HEAD (the lm_head rounded to its per-output-row int8 grid; the attention projections,
experts, router, embeddings and norms bf16, untouched). It alone decides, against A, at BOTH of P125's ruled gates
(``t1`` 196 windows, ``k16`` 123), with the W8A16 split premise's rule unchanged (bench/i8dense, #1745):

    GO    iff, at t1 AND at k16: mean argmax agreement >= 0.95 AND the one-sided 95 % upper bound d + 1.645 SE <=
          P125's bound, ln(1 + 0.05 / ppl_A) over that gate's own windows (``p125_reduce.bound_of``);
    PARK  otherwise: the int8 head parks with the numbers.
    VOID  A or I8HEAD missing or not ok, an arm off its name, I8HEAD's round trip not exactly the lm_head (one module,
          no attention module), an arm's recorded windows (counts and wikitext stride) off ``WINDOWS``, or a ruled
          gate's windows off 196 / 123 or not the same windows in both arms, or an arm's held-out window hashes
          (``heldout``, written by p150_box.py) not the registered ones, or I8HEAD's reference log-probs (``ref_sha256``)
          not A's at a ruled gate.

REPORTED, never a rung: c4 (16 windows), P125's own gate outcome at each gate, and the round trip's error.

    python p150_reduce.py --dir /root/p150 --out verdict_p150.json
    python p150_reduce.py --self-test
"""
import argparse
import json
import math
import os
import statistics
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import p125_reduce as pr  # noqa: E402  (staged at P125's pinned bytes: bound_of, gate, _per, GATES, ARGMAX_MIN)

RULED, CANDIDATE = ("t1", "k16"), "I8HEAD"
COUNTS = {"t1": 196, "k16": 123, "c4": 16}
UPPER_Z = 1.645                                       # one-sided 95 %
WINDOWS = {"counts": COUNTS, "wikitext_stride": 1024}  # i8_box's rec["windows"]: every arm reads the same windows
#: p150_box.HELDOUT, restated (tests/test_p150.py holds the two equal): the held-out windows every ruled arm must read
HELDOUT = {"split": "validation", "stride": 1024,
           "sha256": {"t1": "5c444982538d1aaba2b4f47c00509c0604a72b3845f5cb98c7b30d07b81a76ec",
                      "k16": "7ab52a502ffc5292802db52e4325f025688deeed735a8c04ce33fba5789acded"}}
SCOPE = {"I8HEAD": {"modules": 1, "attention_modules": 0, "lm_head": True}}   # exactly the lm_head


def _load(d, name):
    try:
        return json.load(open(os.path.join(d, name)))
    except (OSError, ValueError):
        return None


def stats(nll_a, nll_x, argmax) -> dict:
    """One arm at one gate against A: mean d, its SE, argmax agreement, P125's bound and the one-sided upper bound."""
    n = len(nll_a)
    d = [x - a for a, x in zip(nll_a, nll_x)]
    md = statistics.mean(d)
    sd = statistics.stdev(d) if n > 1 else float("inf")
    se = sd / math.sqrt(n)
    bound, am = pr.bound_of(nll_a), statistics.mean(argmax)
    return {"windows": n, "d": round(md, 6), "sd": round(sd, 6), "se": round(se, 6), "upper": round(md + UPPER_Z * se, 6),
            "bound": round(bound, 6), "argmax": round(am, 5),
            "ppl_A": round(math.exp(statistics.mean(nll_a)), 4), "ppl_X": round(math.exp(statistics.mean(nll_x)), 4),
            "raw": {"d": md, "se": se, "bound": bound, "argmax": am}}     # the rule reads these, never the rounded fields


def gate_rule(s) -> tuple:
    """(passes, why): argmax >= 0.95 and d + UPPER_Z SE <= bound, compared UNROUNDED (``s["raw"]``; the review of
    #1807, P143 A3's lesson): the rounded fields are for reading only."""
    r = s["raw"]
    if r["argmax"] < pr.ARGMAX_MIN:
        return False, f"argmax {r['argmax']:.6f} < {pr.ARGMAX_MIN}"
    upper = r["d"] + UPPER_Z * r["se"]
    if upper > r["bound"]:
        return False, f"d + {UPPER_Z} SE = {upper:+.7f} > bound {r['bound']:.7f}"
    return True, f"argmax {r['argmax']:.6f}, d + {UPPER_Z} SE = {upper:+.7f} <= bound {r['bound']:.7f}"


def _arm_gate(qa, qx, g, n_want):
    ra, rx = pr._per(qa, g, "R"), pr._per(qx, g, "ON")
    if n_want is not None and (len(ra) != n_want or len(rx) != n_want):
        return None, f"{len(ra)} / {len(rx)} windows, not {n_want}"
    if [r["window"] for r in ra] != [r["window"] for r in rx]:
        return None, "the two arms read different windows"
    nll_a = [r["nll"] for r in ra]
    s = stats(nll_a, [r["nll"] for r in rx], [r["argmax_agree"] for r in rx])
    s["p125_gate"] = pr.gate(nll_a, [r["nll"] for r in rx], [r["argmax_agree"] for r in rx])["outcome"]
    return s, ""


def judge(d) -> dict:
    qa, qc = _load(d, "quality_A.json"), _load(d, f"quality_{CANDIDATE}.json")
    void = []
    for arm, rec in (("A", qa), (CANDIDATE, qc)):
        if not rec or rec.get("status") != "ok":
            void.append(f"quality_{arm} missing or not ok")
        elif rec.get("arm") != arm:
            void.append(f"quality_{arm} names arm {rec.get('arm')!r}")
        elif rec.get("windows") != WINDOWS:
            void.append(f"quality_{arm} read windows {rec.get('windows')}, not {WINDOWS}")
        elif rec.get("heldout") != HELDOUT:
            void.append(f"quality_{arm} read held-out windows {rec.get('heldout')}, not the registered {HELDOUT}")
    if qc and qc.get("status") == "ok":
        fq = qc.get("fakequant") or {}
        if {k: fq.get(k) for k in SCOPE[CANDIDATE]} != SCOPE[CANDIDATE]:
            void.append(f"{CANDIDATE}'s round trip is not exactly the lm_head with the attention untouched: "
                        f"{ {k: fq.get(k) for k in SCOPE[CANDIDATE]} }")
    if qa and qc and qa.get("status") == "ok" and qc.get("status") == "ok":     # belt and braces (the 07:03Z ruling)
        ra, rc = qa.get("ref_sha256") or {}, qc.get("ref_sha256") or {}
        if not all(ra.get(g) for g in RULED) or any(rc.get(g) != ra.get(g) for g in RULED):
            void.append(f"{CANDIDATE} did not score against A's reference log-probs: A {ra}, {CANDIDATE} {rc}")
    out = {"verdict": None, "void": void, "gates": {}, "reporting": {}}
    if void:
        out["verdict"] = "VOID"
        return out
    for g in RULED:
        try:
            s, why = _arm_gate(qa, qc, g, COUNTS[g])
        except (KeyError, TypeError):
            s, why = None, "per-window records missing"
        if s is None:
            void.append(f"gate {g}: {why}")
            continue
        s["passes"], s["why"] = gate_rule(s)
        out["gates"][g] = s
    if void:
        out["verdict"] = "VOID"
        return out
    out["verdict"] = "GO" if all(out["gates"][g]["passes"] for g in RULED) else "PARK"
    rep = out["reporting"]
    try:
        rep[f"{CANDIDATE}_c4"], _ = _arm_gate(qa, qc, "c4", None)
    except (KeyError, TypeError):
        rep[f"{CANDIDATE}_c4"] = None
    rep["fakequant"] = {k: (qc.get("fakequant") or {}).get(k) for k in ("scope", "modules", "attention_modules",
                                                                         "lm_head", "max_err_over_half_step",
                                                                         "rel_fro_err_max")}
    return out


def _fake(d, *, d_head=0.0005, sd=0.02, argmax_head=0.99, fail_gate=None, scope=None, windows=None, head=True,
          a_status="ok", stride=None, heldout=None, refs=None):
    """A and I8HEAD records: I8HEAD's nll is A's plus ``d_head`` and an alternating +-``sd`` (a deterministic mean shift
    with an SD of about ``sd``); ``fail_gate`` drops I8HEAD's argmax agreement to 0.94 at that gate only."""
    arms = ["A"] + ([CANDIDATE] if head else [])
    base = {}
    for arm in arms:
        gates = {}
        for g, (text, _group, _n, _nk) in pr.GATES.items():
            n = (windows or {}).get(g, COUNTS[g])
            pw = []
            for i in range(n):
                b = base.setdefault((g, i), 2.2 + 0.05 * math.sin(i))
                am = 1.0 if arm == "A" else (0.94 if g == fail_gate else argmax_head)
                nll = b if arm == "A" else b + d_head + (sd if i % 2 == 0 else -sd)
                pw.append({"window": i, "nll": nll, "argmax_agree": am})
            gates[g] = {"text": text, "windows": n, "per_window": {text: {("R" if arm == "A" else "ON"): pw}}}
        fq = None if arm == "A" else dict(SCOPE[CANDIDATE], scope="head", max_err_over_half_step=1.0, rel_fro_err_max=0.004)
        if arm == CANDIDATE and scope is not None:
            fq = dict(fq, **scope)
        rec = {"status": a_status if arm == "A" else "ok", "arm": arm, "gates": gates, "fakequant": fq,
               # the ruled counts as recorded, whatever the gates hold: the per-window count check is tested on its own
               "windows": {"counts": dict(COUNTS), "wikitext_stride": (stride or {}).get(arm, 1024)},
               "heldout": (heldout or {}).get(arm, HELDOUT),
               "ref_sha256": (refs or {}).get(arm, {"t1": "a" * 64, "k16": "b" * 64, "c4": "c" * 64})}
        json.dump(rec, open(os.path.join(d, f"quality_{arm}.json"), "w"))


def self_test() -> int:
    cases = []

    def run(name, want, **kw):
        with tempfile.TemporaryDirectory() as d:
            _fake(d, **kw)
            r = judge(d)
        cases.append((name, r["verdict"] == want, (r["verdict"], r["void"], {g: v.get("why") for g, v in r["gates"].items()})))
        return r

    r = run("a small head shift is GO", "GO", d_head=0.0005)
    cases.append(("c4 is reported, never ruled", "c4" not in r["gates"] and f"{CANDIDATE}_c4" in r["reporting"], None))
    run("argmax under 0.95 PARKs", "PARK", argmax_head=0.94)
    run("one gate failing PARKs: both gates must pass", "PARK", fail_gate="k16")
    # d itself sits under the bound (~0.0055) but its upper bound does not: the point-d mutant would say GO
    run("d under the bound with its upper bound over it PARKs", "PARK", d_head=0.0045, sd=0.02)
    run("d over the bound PARKs", "PARK", d_head=0.008)
    run("a head arm that also rounds attention is VOID", "VOID", scope={"attention_modules": 96, "modules": 97})
    run("an arm that does not round the lm_head is VOID", "VOID", scope={"lm_head": False})
    run("I8HEAD not run is VOID", "VOID", head=False)
    run("a short t1 is VOID", "VOID", windows={"t1": 108})
    run("a short k16 is VOID", "VOID", windows={"k16": 112})
    run("A not ok is VOID", "VOID", a_status="failed")
    run("an arm read at another wikitext stride is VOID", "VOID", stride={CANDIDATE: 2048})
    seen = {**HELDOUT, "split": "test", "sha256": {"t1": "0" * 64, "k16": "1" * 64}}
    run("I8HEAD read on windows that are not the registered held-out set is VOID", "VOID", heldout={CANDIDATE: seen})
    run("A read on windows that are not the registered held-out set is VOID", "VOID", heldout={"A": seen})
    run("a record without its held-out hashes is VOID", "VOID", heldout={CANDIDATE: None})
    run("I8HEAD scored against other reference log-probs is VOID", "VOID",
        refs={CANDIDATE: {"t1": "d" * 64, "k16": "b" * 64, "c4": "c" * 64}})
    run("A without its reference hashes is VOID", "VOID", refs={"A": None})

    def rounded(raw):
        return {"d": round(raw["d"], 6), "se": round(raw["se"], 6), "bound": round(raw["bound"], 6),
                "argmax": round(raw["argmax"], 5), "raw": raw}
    over = rounded({"d": 0.00401949, "se": 0.0009, "bound": 0.0054999, "argmax": 0.99})   # rounded: 0.0054995 <= 0.0055
    cases.append(("the bar reads the unrounded upper bound (over by 9e-8 PARKs)", gate_rule(over)[0] is False,
                  gate_rule(over)[1]))
    low = rounded({"d": 0.001, "se": 0.0003, "bound": 0.0055, "argmax": 0.949996})        # rounded: argmax 0.95
    cases.append(("the argmax gate reads the unrounded agreement (0.949996 PARKs)", gate_rule(low)[0] is False,
                  gate_rule(low)[1]))
    bad = [(n, d) for n, ok, d in cases if not ok]
    print(f"p150_reduce self-test {'OK' if not bad else 'FAILED ' + str(bad)} ({len(cases)} cases)")
    return 0 if not bad else 1


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--self-test", action="store_true")
    p.add_argument("--dir")
    p.add_argument("--out")
    a = p.parse_args(argv)
    if a.self_test:
        return self_test()
    v = judge(a.dir)
    json.dump(v, open(a.out, "w"), indent=1)
    print("P150_VERDICT " + json.dumps({"verdict": v["verdict"], "void": v["void"],
                                      **{g: {k: v["gates"][g][k] for k in ("passes", "argmax", "d", "se", "upper", "bound")}
                                         for g in v["gates"]}}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
