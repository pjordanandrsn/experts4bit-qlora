#!/usr/bin/env python3
"""p147_reduce.py -- P147's verdict (bench/p147/PREREG-p147.md), fixed before data.

Inputs: the box's run record (``run.json``) and one record per arm (``arm_<ARM>.json``: A, A2 (the PREREG's A'),
I8ATTN, NF4ATTN, I8HEAD), plus P141's served FP8-KV paged reference (``quality_off_mixtral.json``, staged at its pinned bytes).

Order (the first that applies):
  1. VOID on the record: a shard off eba92302; a required arm missing (A, A2, I8ATTN, NF4ATTN); a set's window count,
     stride, group or sha256 off the registered one, or an arm's windows differing from A's; a candidate's round trip
     not exactly the 128 attention projections with the head untouched (I8HEAD: exactly the lm_head); a restored
     weight whose sha256 differs from A's.
  2. VOID on the instrument checks: (a) A against P141's R on the check set, |mean delta-NLL| <= 0.02 nats per text;
     (b) A2 against A, |delta-ppl| <= 0.001 per set (bit identity is reported, never ruled).
  3. Per candidate (I8ATTN, NF4ATTN), at BOTH gates W1 and W16: GO iff mean argmax agreement >= 0.95 AND
     d + 1.645 SE <= ln(1 + 0.05 / ppl_A); else PARK.
I8HEAD and K8 delta-ppl on the check set are reported only.

    python p147_reduce.py --dir RUN_DIR --ref quality_off_mixtral.json --out verdict_p147.json
    python p147_reduce.py --self-test
"""
import argparse
import json
import math
import os
import sys

REQUIRED = ("A", "A2", "I8ATTN", "NF4ATTN")
REPORTED = ("I8HEAD",)
CANDIDATES = ("I8ATTN", "NF4ATTN")
GATES = ("W1", "W16")
SETS = {   # set -> (windows, stride or None, group)
    "check_wikitext": (24, None, 1), "check_c4val1": (24, None, 1),
    "W1": (196, 1024, 1), "W16": (123, 1024, 16), "C4": (16, None, 16),
}
CHECK_TEXT = {"check_wikitext": "wikitext", "check_c4val1": "c4val1"}
CENSUS = {"I8ATTN": ("int8", "attn", 128), "NF4ATTN": ("nf4", "attn", 128), "I8HEAD": ("int8", "head", 1)}
ARGMAX_MIN = 0.95
UPPER_Z = 1.645                 # one-sided 95 % upper bound on the mean NLL shift (the split premise's rule)
PPL_BUDGET = 0.05               # K8's +0.05 ppl, in nats as ln(1 + 0.05 / ppl_A)
CHECK_A_TOL = 0.02              # check (a): |mean NLL(A) - mean NLL(P141's R)| per check text, in nats
CHECK_B_TOL = 0.001             # check (b): |ppl(A2) - ppl(A)| per set


def ppl(nlls) -> float:
    return math.exp(sum(nlls) / len(nlls))


def nlls(rec, s) -> list:
    return [w["nll"] for w in rec["sets"][s]["per_window"]]


def gate(arm, ref, s) -> dict:
    """The rule's numbers for one candidate at one gate, against A on the same windows."""
    a, b = ref["sets"][s]["per_window"], arm["sets"][s]["per_window"]
    d = [x["nll"] - y["nll"] for x, y in zip(b, a)]
    n = len(d)
    mean = sum(d) / n
    sd = math.sqrt(sum((x - mean) ** 2 for x in d) / (n - 1)) if n > 1 else float("inf")
    se = sd / math.sqrt(n)
    agree = [sum(int(p == q) for p, q in zip(x["argmax"], y["argmax"])) / len(y["argmax"]) for x, y in zip(b, a)]
    ppl_a = ppl([y["nll"] for y in a])
    bound = math.log1p(PPL_BUDGET / ppl_a)
    upper = mean + UPPER_Z * se
    ok = (sum(agree) / n >= ARGMAX_MIN) and upper <= bound
    return {"windows": n, "mean_d": round(mean, 6), "sd": round(sd, 6), "se": round(se, 6), "upper": round(upper, 6),
            "bound": round(bound, 6), "ppl_A": round(ppl_a, 4), "argmax_agreement": round(sum(agree) / n, 4),
            "pass": ok}


def record_faults(run, arms, ref) -> list:
    why = []
    if run.get("shards_ok") is not True:
        why.append("a checkpoint shard is off eba92302 (run.json shards_ok is not true)")
    for a in REQUIRED:
        if a not in arms:
            why.append(f"arm {a}: record missing")
    if "A" not in arms:
        return why
    want_sha = {s: ref["windows_sha256"][t] for s, t in CHECK_TEXT.items()}
    for name, rec in arms.items():
        for s, (n, stride, group) in SETS.items():
            got = (rec.get("sets") or {}).get(s)
            if not got:
                why.append(f"arm {name}: set {s} missing")
                continue
            if len(got.get("per_window") or []) != n or got.get("windows") != n or got.get("group") != group:
                why.append(f"arm {name}: set {s} is {got.get('windows')} windows / group {got.get('group')}, not {n} / {group}")
            if stride is not None and got.get("stride") != stride:
                why.append(f"arm {name}: set {s} stride {got.get('stride')} is not {stride}")
            if s in want_sha and got.get("sha256") != want_sha[s]:
                why.append(f"arm {name}: set {s} windows sha256 is not P141's")
            if got.get("sha256") != arms["A"]["sets"].get(s, {}).get("sha256"):
                why.append(f"arm {name}: set {s} windows differ from A's")
            if any(len(w.get("argmax") or []) != 128 for w in got.get("per_window") or []):
                why.append(f"arm {name}: set {s} has a window without 128 argmax positions")
        if name in CENSUS:
            fmt, scope, mods = CENSUS[name]
            c = rec.get("census") or {}
            if (c.get("format"), c.get("scope"), c.get("modules"), c.get("lm_head")) != (fmt, scope, mods, scope == "head"):
                why.append(f"arm {name}: round trip {c.get('format')}/{c.get('scope')}/{c.get('modules')} is not {fmt}/{scope}/{mods}")
        r = rec.get("restore")
        if name != "A" and (not r or r.get("mismatched") or r.get("restored") != 129):
            why.append(f"arm {name}: weights not restored to A's sha256 before it ({r})")
    return why


def checks(arms, ref) -> tuple:
    out, why = {"a": {}, "b": {}}, []
    for s, text in CHECK_TEXT.items():
        rn = [w["nll"] for w in ref["per_window"][text]["R"]]
        an = nlls(arms["A"], s)
        r, a = sum(rn) / len(rn), sum(an) / len(an)
        ok = abs(a - r) <= CHECK_A_TOL
        out["a"][text] = {"nll_A": round(a, 6), "nll_P141_R": round(r, 6), "delta_nll": round(a - r, 6),
                          "ppl_A": round(math.exp(a), 5), "ppl_P141_R": round(math.exp(r), 5), "pass": ok}
        if not ok:
            why.append(f"check (a) {text}: |NLL(A) {a:.6f} - NLL(P141 R) {r:.6f}| > {CHECK_A_TOL} nats")
    for s in SETS:
        a, b = arms["A"]["sets"][s]["per_window"], arms["A2"]["sets"][s]["per_window"]
        dp = ppl([w["nll"] for w in b]) - ppl([w["nll"] for w in a])
        ident = all(x["nll"] == y["nll"] and x["argmax"] == y["argmax"] for x, y in zip(a, b))
        ok = abs(dp) <= CHECK_B_TOL
        out["b"][s] = {"delta_ppl": round(dp, 7), "bit_identical": ident, "pass": ok}
        if not ok:
            why.append(f"check (b) {s}: |ppl(A') - ppl(A)| = {abs(dp):.6f} > {CHECK_B_TOL}")
    return out, why


def reduce(run, arms, ref) -> dict:
    v = {"lane": "P147", "verdict": None, "why": [], "candidates": {}, "reported": {}}
    why = record_faults(run, arms, ref)
    if why:
        v.update(verdict="VOID", why=why)
        return v
    v["checks"], why = checks(arms, ref)
    if why:
        v.update(verdict="VOID", why=why)
        return v
    for c in CANDIDATES:
        gates = {g: gate(arms[c], arms["A"], g) for g in GATES}
        v["candidates"][c] = {"gates": gates, "verdict": "GO" if all(x["pass"] for x in gates.values()) else "PARK"}
    for c in REPORTED:
        if c in arms:
            v["reported"][c] = {g: gate(arms[c], arms["A"], g) for g in GATES}
    k8 = {}
    for c in (c for c in CANDIDATES + REPORTED if c in arms):
        k8[c] = {CHECK_TEXT[s]: round(ppl(nlls(arms[c], s)) - ppl(nlls(arms["A"], s)), 5) for s in CHECK_TEXT}
    v["reported"]["k8_delta_ppl_check_set"] = k8
    v["verdict"] = "READ"
    return v


# ---- self-test (synthetic records; the mutants it must kill are in tests/test_p147.py) ----

def _synth(arm, shift=0.0, flips=0, noise=0.0, seed=0):
    import random
    rnd = random.Random(seed)
    sets = {}
    for s, (n, stride, group) in SETS.items():
        per = []
        for i in range(n):
            base = 1.2 + 0.1 * math.sin(i)
            jitter = (rnd.random() - 0.5) * 2 * noise
            am = list(range(128))
            for k in range(flips):
                am[k] = -1
            per.append({"nll": base + shift + jitter, "argmax": am})
        sets[s] = {"windows": n, "group": group, "per_window": per, "sha256": f"sha-{s}"}
        if stride:
            sets[s]["stride"] = stride
    rec = {"arm": arm, "sets": sets}
    if arm in CENSUS:
        fmt, scope, mods = CENSUS[arm]
        rec["census"] = {"format": fmt, "scope": scope, "modules": mods, "lm_head": scope == "head"}
    if arm != "A":
        rec["restore"] = {"restored": 129, "mismatched": []}
    return rec


def _ref():
    per = [{"nll": 1.2 + 0.1 * math.sin(i)} for i in range(24)]
    return {"windows_sha256": {"wikitext": "sha-check_wikitext", "c4val1": "sha-check_c4val1"},
            "per_window": {"wikitext": {"R": per}, "c4val1": {"R": per}}}


def self_test() -> int:
    run = {"shards_ok": True}
    base = {a: _synth(a) for a in ("A", "A2")}
    cases = []

    def verdict(arms, r=run):
        return reduce(r, arms, _ref())

    v = verdict({**base, "I8ATTN": _synth("I8ATTN", 0.001, 1, 0.002, 1), "NF4ATTN": _synth("NF4ATTN", 0.05, 20, 0.002, 2),
                 "I8HEAD": _synth("I8HEAD", 0.0005, 0, 0.001, 3)})
    cases.append(("a small shift with few flips is GO; a large one is PARK",
                  v["verdict"] == "READ" and v["candidates"]["I8ATTN"]["verdict"] == "GO"
                  and v["candidates"]["NF4ATTN"]["verdict"] == "PARK", v.get("candidates")))
    cases.append(("I8HEAD is reported, never ruled", "I8HEAD" in v["reported"] and "I8HEAD" not in v["candidates"], None))
    v = verdict({**base, "I8ATTN": _synth("I8ATTN", 0.0, 8, 0.0, 1), "NF4ATTN": _synth("NF4ATTN", 0.0, 0, 0.0, 2)})
    cases.append(("argmax agreement below 0.95 PARKs", v["candidates"]["I8ATTN"]["verdict"] == "PARK", None))
    # the point d is inside the bound but its one-sided upper bound is not: the rule must PARK (the point-d mutant GOes)
    bound = math.log1p(PPL_BUDGET / ppl([1.2 + 0.1 * math.sin(i) for i in range(196)]))
    v = verdict({**base, "I8ATTN": _synth("I8ATTN", 0.8 * bound, 0, 0.04, 5), "NF4ATTN": _synth("NF4ATTN", 0.0, 0, 0.0, 2)})
    g = v["candidates"]["I8ATTN"]["gates"]["W1"]
    cases.append(("a point d inside the bound with its upper bound outside PARKs",
                  g["mean_d"] <= g["bound"] < g["upper"] and v["candidates"]["I8ATTN"]["verdict"] == "PARK", g))
    arms = {**base, "I8ATTN": _synth("I8ATTN"), "NF4ATTN": _synth("NF4ATTN")}
    arms["I8ATTN"]["census"]["modules"] = 127
    cases.append(("a round trip off 128 modules VOIDs", verdict(arms)["verdict"] == "VOID", None))
    arms = {**base, "I8ATTN": _synth("I8ATTN"), "NF4ATTN": _synth("NF4ATTN")}
    arms["NF4ATTN"]["restore"]["mismatched"] = ["model.layers.3.self_attn.o_proj"]
    cases.append(("an unrestored weight VOIDs", verdict(arms)["verdict"] == "VOID", None))
    arms = {**base, "I8ATTN": _synth("I8ATTN"), "NF4ATTN": _synth("NF4ATTN")}
    arms["I8ATTN"]["sets"]["W16"]["per_window"] = arms["I8ATTN"]["sets"]["W16"]["per_window"][:100]
    cases.append(("a gate off its window count VOIDs", verdict(arms)["verdict"] == "VOID", None))
    arms = {**base, "I8ATTN": _synth("I8ATTN"), "NF4ATTN": _synth("NF4ATTN")}
    arms["A"] = _synth("A", 0.05)
    arms["A2"] = _synth("A2", 0.05)
    cases.append(("check (a) off P141's R VOIDs", verdict(arms)["verdict"] == "VOID", None))
    # 0.015 nats is inside (a)'s 0.02-nat bar but about 0.05 ppl at ppl 3.3: (a) is registered in nats, not ppl
    arms = {**base, "I8ATTN": _synth("I8ATTN"), "NF4ATTN": _synth("NF4ATTN")}
    arms["A"], arms["A2"] = _synth("A", 0.015), _synth("A2", 0.015)
    v = verdict(arms)
    cases.append(("check (a) is in nats: 0.015 nats off R passes", v["verdict"] == "READ"
                  and v["checks"]["a"]["wikitext"]["pass"] and abs(v["checks"]["a"]["wikitext"]["delta_nll"] - 0.015) < 1e-9,
                  v["checks"]["a"]))
    arms = {**base, "I8ATTN": _synth("I8ATTN"), "NF4ATTN": _synth("NF4ATTN")}
    arms["A2"] = _synth("A2", 0.002)
    cases.append(("check (b) off by more than 0.001 ppl VOIDs", verdict(arms)["verdict"] == "VOID", None))
    arms = {**base, "I8ATTN": _synth("I8ATTN"), "NF4ATTN": _synth("NF4ATTN")}
    arms["A2"] = _synth("A2", 0.0002)
    v = verdict(arms)
    cases.append(("a non-identical repeat inside 0.001 ppl passes, identity reported false",
                  v["verdict"] == "READ" and v["checks"]["b"]["W1"]["bit_identical"] is False, None))
    cases.append(("a missing candidate VOIDs", verdict({**base, "I8ATTN": _synth("I8ATTN")})["verdict"] == "VOID", None))
    cases.append(("a shard off the revision VOIDs", verdict({**base, "I8ATTN": _synth("I8ATTN"), "NF4ATTN": _synth("NF4ATTN")},
                                                           {"shards_ok": False})["verdict"] == "VOID", None))
    arms = {**base, "I8ATTN": _synth("I8ATTN"), "NF4ATTN": _synth("NF4ATTN")}
    arms["NF4ATTN"]["sets"]["W1"]["sha256"] = "other"
    cases.append(("an arm's windows differing from A's VOID", verdict(arms)["verdict"] == "VOID", None))
    bad = [(n, d) for n, ok, d in cases if not ok]
    print(f"p147_reduce self-test {'OK' if not bad else 'FAILED ' + str(bad)} ({len(cases)} cases)")
    return 0 if not bad else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dir")
    ap.add_argument("--ref")
    ap.add_argument("--out")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args(argv)
    if a.self_test:
        return self_test()
    run = json.load(open(os.path.join(a.dir, "run.json")))
    arms = {}
    for name in REQUIRED + REPORTED:
        p = os.path.join(a.dir, f"arm_{name}.json")
        if os.path.exists(p):
            arms[name] = json.load(open(p))
    v = reduce(run, arms, json.load(open(a.ref)))
    json.dump(v, open(a.out, "w"), indent=1)
    line = {c: x["verdict"] for c, x in v["candidates"].items()}
    print("P147 " + v["verdict"] + " " + json.dumps({"candidates": line, "why": v["why"][:3]}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
