#!/usr/bin/env python3
"""i8_reduce.py -- the W8A16 quality premise's rule (bench/i8dense/PREMISE-w8a16.md; e4b#1313).

I8 against A at each of P125's two ruled gates (``t1``, ``k16``), by P125's own gate function (``p125_reduce.gate``,
staged at P125's pinned bytes): argmax agreement >= 0.95, mean d - 2 SE <= bound, SE <= bound / 3, mean d <= bound,
with bound = ln(1 + 0.05 / ppl_A) over that gate's own windows. ``c4`` is reported only.

  VOID   a record missing or not ok; an arm off its name; I8's round trip missing or not the 97 modules (48 fused
         q/k/v, 48 o, the head); a gate's window count off P125's (108, 112, 16) or the two arms' windows differing.
  GO     both t1 and k16 PASS: the kernel work may start (a premise, never a registered reading).
  PARK   either gate FAILs: no kernel; the lane parks with the number.
  UNDERPOWERED otherwise (neither failed, one is underpowered): reported with the windows it would have needed.
REPORTING: A's mean NLL and ppl per gate beside P125's prior (ppl_R 8.762 on Phase D's windows); each module's
round-trip error; c4's ppl for both arms.

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
import p125_reduce as pr  # noqa: E402  (staged at P125's pinned bytes: gate, _per, GATES)

RULED = ("t1", "k16")
MODULES = 97                       # 48 fused q/k/v + 48 o + the lm_head on Qwen3-30B-A3B


def _load(d, name):
    p = os.path.join(d, name)
    try:
        return json.load(open(p))
    except (OSError, ValueError):
        return None


def judge(d) -> dict:
    qa, qi = _load(d, "quality_A.json"), _load(d, "quality_I8.json")
    void = []
    for arm, rec in (("A", qa), ("I8", qi)):
        if not rec or rec.get("status") != "ok":
            void.append(f"quality_{arm} missing or not ok")
        elif rec.get("arm") != arm:
            void.append(f"quality_{arm} names arm {rec.get('arm')!r}")
    if qi and qi.get("status") == "ok":
        fq = qi.get("fakequant") or {}
        if fq.get("modules") != MODULES or fq.get("attention_modules") != MODULES - 1:
            void.append(f"I8's round trip covered {fq.get('modules')} modules, not {MODULES}")
    out = {"verdict": None, "void": void, "gates": {}, "reporting": {}}
    if void:
        out["verdict"] = "VOID"
        return out
    for g, (text, _group, n, _n_k) in pr.GATES.items():
        try:
            ra, rx = pr._per(qa, g, "R"), pr._per(qi, g, "ON")
        except (KeyError, TypeError):
            void.append(f"gate {g}: per-window records missing")
            continue
        if len(ra) != n or len(rx) != n or [r["window"] for r in ra] != [r["window"] for r in rx]:
            void.append(f"gate {g}: {len(ra)} / {len(rx)} windows, not {n} each over the same windows")
            continue
        nll_a = [r["nll"] for r in ra]
        res = pr.gate(nll_a, [r["nll"] for r in rx], [r["argmax_agree"] for r in rx])
        res["ppl_A"] = round(math.exp(statistics.mean(nll_a)), 4)
        res["ppl_I8"] = round(math.exp(statistics.mean(r["nll"] for r in rx)), 4)
        out["gates" if g in RULED else "reporting"][g] = res
    if void:
        out["verdict"] = "VOID"
        return out
    outcomes = [out["gates"][g]["outcome"] for g in RULED]
    out["verdict"] = "GO" if all(o == "PASS" for o in outcomes) else ("PARK" if "FAIL" in outcomes else "UNDERPOWERED")
    fq = qi["fakequant"]
    out["reporting"]["fakequant"] = {k: fq.get(k) for k in ("modules", "attention_modules", "max_err_over_half_step",
                                                            "rel_fro_err_max")}
    out["reporting"]["prior_ppl_R"] = pr.PRIOR["ppl_R"]
    return out


def _fake(d, *, shift=0.0, argmax=0.99, sd=0.004, modules=MODULES, windows=None, a_status="ok"):
    import random
    rnd = random.Random(7)
    recs = {}
    for arm in ("A", "I8"):
        gates = {}
        for g, (text, _group, n, _n_k) in pr.GATES.items():
            n = (windows or {}).get(g, n)
            pw = []
            for i in range(n):
                base = 2.17 + 0.05 * math.sin(i)
                nll = base if arm == "A" else base + shift + rnd.gauss(0.0, sd)
                pw.append({"window": i, "nll": nll, "argmax_agree": 1.0 if arm == "A" else argmax})
            gates[g] = {"text": text, "windows": n, "per_window": {text: {("R" if arm == "A" else "ON"): pw}}}
        recs[arm] = {"status": a_status if arm == "A" else "ok", "arm": arm, "gates": gates,
                     "fakequant": None if arm == "A" else {"modules": modules, "attention_modules": modules - 1,
                                                           "max_err_over_half_step": 1.0, "rel_fro_err_max": 0.004}}
    for arm, rec in recs.items():
        json.dump(rec, open(os.path.join(d, f"quality_{arm}.json"), "w"))


def self_test() -> int:
    cases = []

    def run(name, want, **kw):
        with tempfile.TemporaryDirectory() as d:
            _fake(d, **kw)
            r = judge(d)
        cases.append((name, r["verdict"] == want, (r["verdict"], r["void"], {g: v.get("why") for g, v in r["gates"].items()})))
        return r

    run("a tight round trip is GO", "GO", shift=0.0005, argmax=0.99)
    run("argmax under 0.95 PARKs", "PARK", shift=0.0, argmax=0.94)
    run("a shift past the bound PARKs", "PARK", shift=0.02, argmax=0.99)
    run("a noisy arm is UNDERPOWERED", "UNDERPOWERED", shift=0.0, argmax=0.99, sd=0.05)
    run("a round trip short of 97 modules is VOID", "VOID", modules=96)
    run("a short gate is VOID", "VOID", windows={"t1": 100})
    run("A not ok is VOID", "VOID", a_status="failed")
    r = run("c4 is reported, not ruled", "GO", shift=0.0005)
    cases.append(("c4 sits under reporting", "c4" in r["reporting"] and "c4" not in r["gates"], None))
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
                                      **{g: {k: v["gates"][g][k] for k in ("outcome", "argmax", "d", "se", "bound")}
                                         for g in v["gates"]}}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
