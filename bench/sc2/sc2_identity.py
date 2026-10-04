#!/usr/bin/env python3
# Copyright (c) 2026 Cerin Amroth LLC. MIT.
"""sc2_identity.py -- the value-identity gate for a serving lever (lane SC2b, #846): two driver runs of the SAME plan
(sc2_driver.py ``run`` outputs, e.g. the lever off and on) must stream identical text for every request.

It runs on a SERIAL plan on purpose. One request in flight means one batch composition, so a difference can only come
from the lever, not from which bucket a busy server happened to use. Under Poisson load, decode numerics can follow
the batch the scheduler formed, and a token difference there would not be the lever's.

IDENTICAL iff:
- the two runs carry the same prompt pool (``prompts_sha256``) and the same plan (arrival offsets, prompt indices,
  ``max_tokens``);
- every request is VALID in both;
- every request's streamed text is byte-equal.

Otherwise DIFFERS (with the first differing request and the position of the first differing character) or REFUSED
(the runs are not comparable).

  sc2_identity.py A.json B.json
  sc2_identity.py --self-test
"""
import argparse
import json
import sys


def compare(a: dict, b: dict) -> dict:
    if a.get("prompts_sha256") != b.get("prompts_sha256") or a.get("plan") != b.get("plan"):
        return {"verdict": "REFUSED", "why": "the two runs do not share the prompt pool and plan"}
    if a.get("mode") != "serial" or b.get("mode") != "serial":
        return {"verdict": "REFUSED", "why": "the gate reads serial runs only (one batch composition)"}
    ra, rb = a["requests"], b["requests"]
    if len(ra) != len(rb):
        return {"verdict": "REFUSED", "why": f"{len(ra)} vs {len(rb)} requests"}
    invalid = [i for i, (x, y) in enumerate(zip(ra, rb)) if not (x.get("valid") and y.get("valid"))]
    if invalid:
        return {"verdict": "REFUSED", "why": f"requests not VALID in both runs: {invalid[:5]}"}
    for i, (x, y) in enumerate(zip(ra, rb)):
        if x["text"] != y["text"]:
            at = next((k for k, (c, d) in enumerate(zip(x["text"], y["text"])) if c != d), min(len(x["text"]), len(y["text"])))
            return {"verdict": "DIFFERS", "request": i, "char": at, "n": len(ra),
                    "a": x["text"][max(0, at - 20):at + 20], "b": y["text"][max(0, at - 20):at + 20]}
    return {"verdict": "IDENTICAL", "n": len(ra), "chars": sum(len(x["text"]) for x in ra)}


def self_test() -> int:
    def run(texts, mode="serial", sha="s", plan=None):
        return {"prompts_sha256": sha, "mode": mode, "plan": plan or [[0.0, i, 8] for i in range(len(texts))],
                "requests": [{"valid": True, "text": t} for t in texts]}
    base = run(["alpha beta", "gamma", "delta epsilon"])
    cases = [compare(base, run(["alpha beta", "gamma", "delta epsilon"]))["verdict"] == "IDENTICAL",
             compare(base, run(["alpha beta", "gamma", "delta epsilom"])) == {
                 "verdict": "DIFFERS", "request": 2, "char": 12, "n": 3, "a": "delta epsilon", "b": "delta epsilom"},
             compare(base, run(["alpha beta", "gamma", "delta epsilon"], sha="t"))["verdict"] == "REFUSED",
             compare(run(["a"], mode="poisson"), run(["a"], mode="poisson"))["verdict"] == "REFUSED"]
    bad = run(["alpha beta", "gamma", "delta epsilon"])
    bad["requests"][1]["valid"] = False
    cases.append(compare(base, bad)["verdict"] == "REFUSED")
    print(f"sc2_identity self-test {'OK' if all(cases) else 'FAILED'} ({sum(cases)}/{len(cases)} cases)")
    return 0 if all(cases) else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("a", nargs="?")
    ap.add_argument("b", nargs="?")
    ap.add_argument("--self-test", action="store_true")
    x = ap.parse_args(argv)
    if x.self_test:
        return self_test()
    out = compare(json.load(open(x.a)), json.load(open(x.b)))
    print("SC2_IDENTITY " + json.dumps(out), flush=True)
    return 0 if out["verdict"] == "IDENTICAL" else 3


if __name__ == "__main__":
    sys.exit(main())
