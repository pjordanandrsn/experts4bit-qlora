#!/usr/bin/env python3
"""i8_reduce.py -- the W8A16 SPLIT premise's rule (bench/i8dense/PREMISE-w8a16.md, "The split premise"; e4b#1313).

The candidate lever is I8ATTN (the attention projections int8, the head bf16). It alone decides, against A, at BOTH of
P125's ruled gates (``t1`` 196 windows, ``k16`` 123):

    GO    iff, at t1 AND at k16: mean argmax agreement >= 0.95 AND the one-sided 95 % upper bound d + 1.645 SE <=
          P125's bound, ln(1 + 0.05 / ppl_A) over that gate's own windows (``p125_reduce.bound_of``);
    PARK  otherwise: the W8A16 lever parks with the numbers (no third premise).
    VOID  A or I8ATTN missing or not ok, an arm off its name, I8ATTN's round trip not exactly the 96 attention modules
          (head untouched), an arm's recorded windows (counts and wikitext stride) off ``WINDOWS``, or a ruled gate's
          windows off 196 / 123 or not the same windows in both arms.

REPORTED, never a rung: I8HEAD (the head int8, the attention bf16) at every gate, which explains where the argmax flips
come from; c4 for both arms; P125's own gate outcome for each; each module's round-trip error.

    python i8_reduce.py --dir /root/i8dense --out verdict_i8.json
    python i8_reduce.py --self-test
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

RULED, CANDIDATE, REPORTED = ("t1", "k16"), "I8ATTN", "I8HEAD"
COUNTS = {"t1": 196, "k16": 123, "c4": 16}
UPPER_Z = 1.645                                       # one-sided 95 %
WINDOWS = {"counts": COUNTS, "wikitext_stride": 1024}  # i8_box's rec["windows"]: every arm reads the same windows
SCOPE = {"I8ATTN": {"modules": 96, "attention_modules": 96, "lm_head": False},
         "I8HEAD": {"modules": 1, "attention_modules": 0, "lm_head": True}}


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
    return {"windows": n, "d": round(md, 6), "sd": round(sd, 6), "se": round(se, 6), "upper": round(md + UPPER_Z * se, 6),
            "bound": round(pr.bound_of(nll_a), 6), "argmax": round(statistics.mean(argmax), 5),
            "ppl_A": round(math.exp(statistics.mean(nll_a)), 4), "ppl_X": round(math.exp(statistics.mean(nll_x)), 4)}


def gate_rule(s) -> tuple:
    """(passes, why): argmax >= 0.95 and d + UPPER_Z SE <= bound."""
    if s["argmax"] < pr.ARGMAX_MIN:
        return False, f"argmax {s['argmax']:.4f} < {pr.ARGMAX_MIN}"
    upper = s["d"] + UPPER_Z * s["se"]
    if upper > s["bound"]:
        return False, f"d + {UPPER_Z} SE = {upper:+.5f} > bound {s['bound']:.5f}"
    return True, f"argmax {s['argmax']:.4f}, d + {UPPER_Z} SE = {upper:+.5f} <= bound {s['bound']:.5f}"


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
    qa, qc, qh = _load(d, "quality_A.json"), _load(d, f"quality_{CANDIDATE}.json"), _load(d, f"quality_{REPORTED}.json")
    void = []
    for arm, rec in (("A", qa), (CANDIDATE, qc)):
        if not rec or rec.get("status") != "ok":
            void.append(f"quality_{arm} missing or not ok")
        elif rec.get("arm") != arm:
            void.append(f"quality_{arm} names arm {rec.get('arm')!r}")
        elif rec.get("windows") != WINDOWS:
            void.append(f"quality_{arm} read windows {rec.get('windows')}, not {WINDOWS}")
    if qc and qc.get("status") == "ok":
        fq = qc.get("fakequant") or {}
        if {k: fq.get(k) for k in SCOPE[CANDIDATE]} != SCOPE[CANDIDATE]:
            void.append(f"{CANDIDATE}'s round trip is not the 96 attention modules with the head untouched: "
                        f"{ {k: fq.get(k) for k in SCOPE[CANDIDATE]} }")
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
    if qh and qh.get("status") == "ok" and qh.get("arm") == REPORTED:
        fq = qh.get("fakequant") or {}
        rep[f"{REPORTED}_scope_ok"] = {k: fq.get(k) for k in SCOPE[REPORTED]} == SCOPE[REPORTED]
        rep[f"{REPORTED}_windows_ok"] = qh.get("windows") == WINDOWS
        for g in ("t1", "k16", "c4"):
            try:
                rep[f"{REPORTED}_{g}"], _ = _arm_gate(qa, qh, g, None)
            except (KeyError, TypeError):
                rep[f"{REPORTED}_{g}"] = None
    else:
        rep[REPORTED] = "not run or not ok (reported only: it rules nothing)"
    rep["fakequant"] = {k: (qc.get("fakequant") or {}).get(k) for k in ("scope", "modules", "max_err_over_half_step",
                                                                         "rel_fro_err_max")}
    return out


def _fake(d, *, d_attn=0.0005, sd=0.02, argmax_attn=0.99, d_head=0.004, argmax_head=0.96, scope_attn=None,
          windows=None, head=True, a_status="ok", stride=None):
    arms = ["A", CANDIDATE] + ([REPORTED] if head else [])
    base = {}
    for arm in arms:
        gates = {}
        for g, (text, _group, _n, _nk) in pr.GATES.items():
            n = (windows or {}).get(g, COUNTS[g])
            pw = []
            for i in range(n):
                b = base.setdefault((g, i), 2.2 + 0.05 * math.sin(i))
                shift = {"A": 0.0, CANDIDATE: d_attn, REPORTED: d_head}[arm]
                am = {"A": 1.0, CANDIDATE: argmax_attn, REPORTED: argmax_head}[arm]
                nll = b if arm == "A" else b + shift + (sd if i % 2 == 0 else -sd)    # deterministic: mean shift, SD ~ sd
                pw.append({"window": i, "nll": nll, "argmax_agree": am})
            gates[g] = {"text": text, "windows": n, "per_window": {text: {("R" if arm == "A" else "ON"): pw}}}
        fq = None if arm == "A" else dict(SCOPE[arm], scope={"I8ATTN": "attn", "I8HEAD": "head"}[arm],
                                           max_err_over_half_step=1.0, rel_fro_err_max=0.01)
        if arm == CANDIDATE and scope_attn is not None:
            fq = dict(fq, **scope_attn)
        rec = {"status": a_status if arm == "A" else "ok", "arm": arm, "gates": gates, "fakequant": fq,
               # the ruled counts as recorded, whatever the gates hold: the per-window count check is tested on its own
               "windows": {"counts": dict(COUNTS), "wikitext_stride": (stride or {}).get(arm, 1024)}}
        json.dump(rec, open(os.path.join(d, f"quality_{arm}.json"), "w"))


def self_test() -> int:
    cases = []

    def run(name, want, **kw):
        with tempfile.TemporaryDirectory() as d:
            _fake(d, **kw)
            r = judge(d)
        cases.append((name, r["verdict"] == want, (r["verdict"], r["void"], {g: v.get("why") for g, v in r["gates"].items()})))
        return r

    run("a small attention shift is GO", "GO", d_attn=0.0005)
    run("argmax under 0.95 PARKs", "PARK", argmax_attn=0.94)
    # d itself sits under the bound (~0.0053) but its upper bound does not: the point-d mutant would say GO
    run("d under the bound with its upper bound over it PARKs", "PARK", d_attn=0.0045, sd=0.02)
    run("d over the bound PARKs", "PARK", d_attn=0.008)
    run("the head failing rules nothing", "GO", d_attn=0.0005, d_head=0.02, argmax_head=0.90)
    r = run("the head not run is still a reading", "GO", head=False)
    cases.append(("the head's absence is reported", "not run" in str(r["reporting"].get(REPORTED)), None))
    run("a head-touching attention arm is VOID", "VOID", scope_attn={"lm_head": True, "modules": 97})
    run("a short t1 is VOID", "VOID", windows={"t1": 108})
    run("a short k16 is VOID", "VOID", windows={"k16": 112})
    run("A not ok is VOID", "VOID", a_status="failed")
    run("an arm read at another wikitext stride is VOID", "VOID", stride={CANDIDATE: 2048})
    r = run("the head at another stride rules nothing", "GO", stride={REPORTED: 2048})
    cases.append(("the head's stride is reported", r["reporting"].get(f"{REPORTED}_windows_ok") is False, None))
    bad = [(n, d) for n, ok, d in cases if not ok]
    print(f"i8_reduce self-test {'OK' if not bad else 'FAILED ' + str(bad)} ({len(cases)} cases)")
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
    print("I8_VERDICT " + json.dumps({"verdict": v["verdict"], "void": v["void"],
                                      **{g: {k: v["gates"][g][k] for k in ("passes", "argmax", "d", "se", "upper", "bound")}
                                         for g in v["gates"]}}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
