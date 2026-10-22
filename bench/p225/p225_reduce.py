#!/usr/bin/env python3
"""p225_reduce.py -- lane P225's verdict (bench/p225/PREREG-p225.md, "Outcomes"), fixed before data. It reuses P224's
reducer (``p224_reduce.verdict``, staged byte-identical) for G1-G5 and maps its result onto P225's named outcomes:

1. **The gate** (``p225_gate.py``'s record): any named failure is VOID with that reason, the first named failure
   leading. INCOMPLETE_TRAIN, ADAPTER_*, LEAK, PARTIAL_ENGAGEMENT and IDENTITY are all VOIDs, and all strikes.
2. **The LEARNED gate:** on the held-out ASSISTANT positions, over distinct sequences, mean(HF0 - REF) - 2 x SE must be
   at least MIN_REF_GAIN (0.05 nats); otherwise **NOT_LEARNED**, a strike.
3. **P224's verdict:** PASS and FAIL stand. A VOID whose only reason is "too small to judge" (the SE rung) is
   **INCONCLUSIVE**, a final answer. Any other VOID stays VOID with its reasons.

Exit codes: 0 for PASS, FAIL or INCONCLUSIVE (an answer); 27 for VOID or NOT_LEARNED (no answer), as P224's runner maps.

    python p225_reduce.py --dir RUN_DIR --pins pins.json --gates gates_p225.json --out verdict_p225.json
    python p225_reduce.py --self-test
"""
from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import p224_reduce as R  # noqa: E402  (staged beside this file, byte-identical to P224's)
import p225_canary as C  # noqa: E402

#: P224's reducer checks every arm's held-out count against its module constant; P225 registers its own set
R.N_HELDOUT = C.N_HELDOUT

ANSWERS = ("PASS", "FAIL", "INCONCLUSIVE")
SE_MARK = "too small to judge"


def learned(recs: dict) -> dict:
    """mean(HF0 - REF) on the assistant positions over distinct sequences, its paired SE, and the gate."""
    keys = recs["REF"].get("per_seq_ids_sha256") or []
    idx = R.distinct_index(keys)
    hf0, ref = recs["HF0"]["per_seq_loss_assistant"], recs["REF"]["per_seq_loss_assistant"]
    d = [hf0[i] - ref[i] for i in idx]
    mean = statistics.fmean(d) if d else math.nan
    se = statistics.stdev(d) / math.sqrt(len(d)) if len(d) > 1 else math.inf
    return {"gain": mean, "se": se, "lower": mean - 2 * se, "bar": R.MIN_REF_GAIN, "n": len(d),
            "ok": (mean - 2 * se) >= R.MIN_REF_GAIN}


def verdict(recs: dict, pins: dict, gates: dict) -> dict:
    out = {"lane": "P225"}
    fails = list((gates or {}).get("failures") or [])
    if not gates or (not gates.get("ok") and not fails):
        fails = ["GATES: the gate record is missing or carries no reasons"]
    if fails:
        return dict(out, verdict="VOID", reason=fails[0].split(":")[0], why=fails)
    try:
        lg = learned(recs)
    except (KeyError, TypeError, IndexError) as e:
        return dict(out, verdict="VOID", reason="RECORD", why=[f"RECORD: the arms' held-out records are unreadable ({e!r})"])
    out["learned"] = lg
    if not lg["ok"]:
        return dict(out, verdict="NOT_LEARNED", reason="NOT_LEARNED",
                    why=[f"NOT_LEARNED: REF improves on HF0 by {lg['gain']:.4f} nats on the assistant positions, lower "
                         f"bound {lg['lower']:.4f} < {lg['bar']}"])
    v = R.verdict(recs, pins)
    out["p224"] = {k: v.get(k) for k in ("verdict", "why", "effect", "margin", "se_pa_ref", "base_gap", "cost_band",
                                         "reported", "loss", "loss_assistant", "distinct")}
    if v["verdict"] in ("PASS", "FAIL"):
        return dict(out, verdict=v["verdict"], reason=v["verdict"], why=v["why"])
    why = v.get("why") or []
    if v["verdict"] == "VOID" and why and all(SE_MARK in w for w in why):
        return dict(out, verdict="INCONCLUSIVE", reason="INCONCLUSIVE", why=why)
    return dict(out, verdict="VOID", reason="P224_RULE", why=why)


# ---------------------------------------------------------------------------------------------------- the self-test

def self_test() -> int:
    import random
    n = 0

    def ok(c, msg):
        nonlocal n
        assert c, msg
        n += 1
    good = {"ok": True, "failures": []}
    pins = R.PINS
    o = verdict(R._fake(), pins, good)
    ok(o["verdict"] == "PASS" and o["learned"]["ok"], f"a passing record passes: {o.get('why')}")
    o = verdict(R._fake(), pins, {"ok": False, "failures": ["INCOMPLETE_TRAIN: 65 of 80", "ADAPTER_TARGETS: x"]})
    ok(o["verdict"] == "VOID" and o["reason"] == "INCOMPLETE_TRAIN", "the gate's first named failure leads")
    ok(verdict(R._fake(), pins, None)["verdict"] == "VOID", "no gate record is VOID")
    ok(verdict(R._fake(), pins, {"ok": False, "failures": []})["reason"] == "GATES", "a failed gate with no reason is VOID")
    hf0_like = R._fake(a=lambda r: r.__setitem__("REF", dict(r["HF0"])))
    o = verdict(hf0_like, pins, good)
    ok(o["verdict"] == "NOT_LEARNED", f"REF equal to HF0 is NOT_LEARNED: {o['verdict']}")
    rng = random.Random(7)
    noisy = R._fake(a=lambda r: r["PA"].update(per_seq_loss_assistant=[x + rng.gauss(0, 0.003 * math.sqrt(C.N_HELDOUT))
                                                                        for x in r["PA"]["per_seq_loss_assistant"]]))
    o = verdict(noisy, pins, good)
    ok(o["verdict"] == "INCONCLUSIVE" and all(SE_MARK in w for w in o["why"]), f"the SE rung alone is INCONCLUSIVE: {o}")
    unfaithful = R._fake(a=lambda r: r["P0"].update(per_seq_loss=[x + 0.03 for x in r["HF0"]["per_seq_loss"]]))
    o = verdict(unfaithful, pins, good)
    ok(o["verdict"] == "VOID" and o["reason"] == "P224_RULE", "an unfaithful base stays VOID, not INCONCLUSIVE")
    failing = R._fake(a=lambda r: r["PA"].update(canary=dict(r["PA"]["canary"], control_hits=3)))
    ok(verdict(failing, pins, good)["verdict"] == "FAIL", "a P224 FAIL stands")
    ok(verdict(R._fake(), dict(pins, e4b_sha="0" * 40), good)["verdict"] == "VOID", "a wrong pin stays VOID")
    ok(R.N_HELDOUT == C.N_HELDOUT and len(R._fake()["REF"]["per_seq_loss"]) == C.N_HELDOUT, "P225's held-out count")
    print(f"p225_reduce self-test OK ({n} checks)")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dir")
    ap.add_argument("--pins")
    ap.add_argument("--gates")
    ap.add_argument("--out")
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--check-args", action="store_true", help="apply the argument checks a real run applies, then stop")
    a = ap.parse_args(argv)
    if a.self_test:
        return self_test()
    if not (a.dir and a.pins and a.gates and a.out):
        ap.error("--dir, --pins, --gates and --out (or --self-test)")
    if a.check_args:
        print("ARGS OK")
        return 0
    pins = json.load(open(a.pins, encoding="utf-8"))
    try:
        gates = json.load(open(a.gates, encoding="utf-8"))
    except (OSError, ValueError):
        gates = None
    out = verdict(R.load(a.dir), pins, gates)
    with open(a.out, "w", encoding="utf-8", newline="\n") as f:
        json.dump(out, f, indent=1)
    print(f"P225 {out['verdict']} {json.dumps(out.get('why'))[:400]}")
    return 0 if out["verdict"] in ANSWERS else 27


if __name__ == "__main__":
    sys.exit(main())
