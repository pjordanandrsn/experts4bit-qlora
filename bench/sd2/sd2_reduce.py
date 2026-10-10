#!/usr/bin/env python3
"""sd2_reduce.py -- lane SD2 (e4b#1313): the proof's verdict (bench/sd2/PREREG-sd2.md, Amendments 1 and 2).

`--prove prove.json --gpu-tests gpu_tests.json [--rule a2|a1]` -> `verdict_prove.json`. **PROVED** when every item
holds; otherwise **FAILED** with every failing item named. A mutant the addressing gate cannot see is named
GATE_TOO_WEAK:<m>, and the registered rule is that the gate is tightened before the read.

**The two addressing rules.**
- `a1` (Amendment 1, `sd2-prove-2`'s registered verdict): a (row, k) passes at row agreement AND continuation
  agreement >= 0.90; a mutant is caught when either falls below 0.90.
- `a2` (Amendment 2, the default): the LOGIT gate decides. A (row, k) passes when the largest per-row mean |Δ log p| of
  the oracle's token (verify against the sequential T == 1 oracle) is <= 0.25 nats; a mutant is caught when its largest
  exceeds it. Argmax agreement is reported, not gated: on chat it sits near 0.90 on the real build (C0 k=3
  continuations 0.906 on `sd2-prove-2`), where near-ties flip. The bound is CALIBRATED on `sd2-prove-2`: the real build
  reached at most 0.037 and every mutant had a row at 0.89 or more ((c), the RoPE shift, 1.18 on row 0), so 0.25 sits
  about 7x from each.

The other items (both rules):
- `census`: buckets 1, 2, 3, 4, 8 and 16 each "graph"; post-verify graphs 2, 3 and 4 each "graph"; three hooks;
- `capture`: every verify bucket's replay bitwise its padded eager step;
- `draft`: in-engine drafts against chain_at at >= 0.99 of drafted ids;
- `transition`: one drop counted, both requests at their lengths, no mirror mismatch, no state left;
- `gpu_tests`: the target's GPU tests exited 0 with none skipped for want of the card.

`--self-test` runs every verdict under both rules, and `sd2-prove-2`'s own numbers: FAILED [GATE_TOO_WEAK:c] under a1,
PROVED under a2.
"""
from __future__ import annotations

import argparse
import copy
import json
import sys

BAR_ADDRESS, BAR_DRAFT = 0.90, 0.99
BOUND_DLOGP = 0.25          # Amendment 2: nats; calibrated on sd2-prove-2 (real <= 0.037, every mutant >= 0.89)
RULES = ("a2", "a1")
BUCKETS = ("1", "2", "3", "4", "8", "16")
POSTS = ("2", "3", "4")


def logit_max(x):
    """The largest per-row mean |Δ log p| of a (row, k) or mutant record, or None when it carries none."""
    v = [d for d in ((x or {}).get("mean_abs_dlogp") or []) if d is not None]
    return max(v) if v else None


def judge(prove: dict, gpu: dict, rule: str = "a2") -> dict:
    if rule not in RULES:
        raise ValueError(f"rule {rule!r}: expected one of {RULES}")
    fails = []
    c = prove.get("census") or {}
    for b in BUCKETS:
        if (c.get("graph_status") or {}).get(b) != "graph":
            fails.append(f"census:bucket_{b}")
    for n in POSTS:
        if (c.get("post_graphs") or {}).get(n) != "graph":
            fails.append(f"census:post_{n}")
    if ((c.get("spec_build") or {}).get("hooks")) != 3:
        fails.append("census:hooks")
    cap = prove.get("capture_bitwise") or {}
    for n in POSTS:
        if cap.get(n) is not True:
            fails.append(f"capture:{n}")
    addr = prove.get("addressing") or {}
    want = [f"{r}_k{k}" for r in ("R0", "C0") for k in (1, 2, 3)]
    for key in want:
        x = addr.get(key)
        if rule == "a1":
            ok = bool(x) and x.get("rows_agree", 0) >= BAR_ADDRESS and x.get("cont_agree", 0) >= BAR_ADDRESS
        else:
            lm = logit_max(x)
            ok = lm is not None and lm <= BOUND_DLOGP
        if not ok:
            fails.append(f"addressing:{key}")
    muts = prove.get("mutants") or {}
    for m in ("a", "b", "c"):
        x = muts.get(m)
        if not x:
            fails.append(f"mutant:{m}:missing")
        elif rule == "a1":
            if x.get("rows_agree", 1.0) >= BAR_ADDRESS and x.get("cont_agree", 1.0) >= BAR_ADDRESS:
                fails.append(f"GATE_TOO_WEAK:{m}")
        else:
            lm = logit_max(x)
            if lm is None:
                fails.append(f"mutant:{m}:missing")
            elif lm <= BOUND_DLOGP:
                fails.append(f"GATE_TOO_WEAK:{m}")
    d = prove.get("draft") or {}
    if d.get("of", 0) <= 0 or d.get("agree", 0) < BAR_DRAFT:
        fails.append("draft")
    t = prove.get("transition") or {}
    if not (t.get("dropped_batched") == 1 and t.get("spec_steps_alone", 0) > 0 and t.get("mirror_mismatches") == 0
            and t.get("state_left") == 0 and [t.get("len_a"), t.get("len_b")] == t.get("want")):
        fails.append("transition")
    if gpu.get("rc") != 0 or gpu.get("passed", 0) <= 0 or gpu.get("skipped_for_card", 1) != 0:
        fails.append("gpu_tests")
    fields = ("rows_agree", "cont_agree", "mean_abs_dlogp")
    return {"verdict": "PROVED" if not fails else "FAILED", "fails": fails, "rule": rule,
            "bound_dlogp": BOUND_DLOGP if rule == "a2" else None,
            "addressing": {k: dict({kk: addr[k].get(kk) for kk in fields}, logit_max=logit_max(addr[k]))
                           for k in want if k in addr},
            "mutants": {m: dict({kk: (muts.get(m) or {}).get(kk) for kk in fields}, logit_max=logit_max(muts.get(m)))
                        for m in ("a", "b", "c")},
            "draft": d, "transition": t}


def _good() -> tuple:
    prove = {"census": {"graph_status": {b: "graph" for b in BUCKETS}, "post_graphs": {n: "graph" for n in POSTS},
                        "spec_build": {"hooks": 3}},
             "capture_bitwise": {n: True for n in POSTS},
             "addressing": {f"{r}_k{k}": {"rows_agree": 0.98, "cont_agree": 0.97, "mean_abs_dlogp": [0.01] * (k + 1)}
                            for r in ("R0", "C0") for k in (1, 2, 3)},
             "mutants": {m: {"rows_agree": 0.3, "cont_agree": 0.3, "mean_abs_dlogp": [0.02, 2.0, 2.0, 2.0]}
                         for m in ("a", "b", "c")},
             "draft": {"agree": 0.995, "same": 764, "of": 768},
             "transition": {"spec_steps_alone": 4, "dropped_batched": 1, "len_a": 64, "len_b": 16, "want": [64, 16],
                            "mirror_mismatches": 0, "state_left": 0}}
    return prove, {"rc": 0, "passed": 40, "skipped_for_card": 0}


def _sd2_prove_2() -> dict:
    """sd2-prove-2's addressing and mutant records (store 72a69894), the calibration set."""
    p, _ = _good()
    p["addressing"] = {
        "R0_k1": {"rows_agree": 1.0, "cont_agree": 1.0, "mean_abs_dlogp": [0.002939, 0.002148]},
        "R0_k2": {"rows_agree": 1.0, "cont_agree": 1.0, "mean_abs_dlogp": [0.006527, 0.00338, 0.003902]},
        "R0_k3": {"rows_agree": 1.0, "cont_agree": 1.0, "mean_abs_dlogp": [0.005385, 0.002219, 0.000969, 0.002394]},
        "C0_k1": {"rows_agree": 1.0, "cont_agree": 1.0, "mean_abs_dlogp": [0.019688, 0.026519]},
        "C0_k2": {"rows_agree": 1.0, "cont_agree": 0.921875, "mean_abs_dlogp": [0.016608, 0.013777, 0.033104]},
        "C0_k3": {"rows_agree": 0.984375, "cont_agree": 0.90625, "mean_abs_dlogp": [0.01489, 0.007156, 0.006616, 0.036915]}}
    p["mutants"] = {
        "a": {"rows_agree": 0.6875, "cont_agree": 0.65625, "mean_abs_dlogp": [0.023935, 2.149183, 2.789426, 2.469237]},
        "b": {"rows_agree": 0.5625, "cont_agree": 0.96875, "mean_abs_dlogp": [0.889273, 5.01826, 5.342677, 0.01232]},
        "c": {"rows_agree": 0.9375, "cont_agree": 0.90625, "mean_abs_dlogp": [1.179762, 0.005705, 0.010378, 0.011125]}}
    p["draft"] = {"agree": 766 / 768, "same": 766, "of": 768}
    return p


def self_test() -> int:
    bad = []
    p, g = _good()
    for rule in RULES:
        if judge(p, g, rule)["verdict"] != "PROVED":
            bad.append(f"good under {rule} -> {judge(p, g, rule)['fails']}")

    def case(edit, want, rule="a2", gpu_edit=None):
        p2, g2 = copy.deepcopy(p), dict(g)
        edit(p2)
        if gpu_edit:
            gpu_edit(g2)
        got = judge(p2, g2, rule)
        if got["verdict"] != "FAILED" or want not in got["fails"]:
            bad.append(f"{rule} {want}: {got['fails']}")

    def proved(edit, rule="a2", why=""):
        p2 = copy.deepcopy(p)
        edit(p2)
        got = judge(p2, g, rule)
        if got["verdict"] != "PROVED":
            bad.append(f"{rule} should prove ({why}): {got['fails']}")

    for rule in RULES:
        case(lambda x: x["census"]["graph_status"].update({"3": "eager: X"}), "census:bucket_3", rule)
        case(lambda x: x["census"]["post_graphs"].pop("4"), "census:post_4", rule)
        case(lambda x: x["census"]["spec_build"].update({"hooks": 2}), "census:hooks", rule)
        case(lambda x: x["capture_bitwise"].update({"2": False}), "capture:2", rule)
        case(lambda x: x["addressing"].pop("C0_k1"), "addressing:C0_k1", rule)
        case(lambda x: x["mutants"].pop("b"), "mutant:b:missing", rule)
        case(lambda x: x["draft"].update({"agree": 0.98}), "draft", rule)
        case(lambda x: x["transition"].update({"dropped_batched": 0}), "transition", rule)
        case(lambda x: x["transition"].update({"len_a": 63}), "transition", rule)
        case(lambda x: x["transition"].update({"mirror_mismatches": 2}), "transition", rule)
        case(lambda x: None, "gpu_tests", rule, lambda gg: gg.update({"rc": 1}))
        case(lambda x: None, "gpu_tests", rule, lambda gg: gg.update({"skipped_for_card": 3}))
    # a1: the agreement gate
    case(lambda x: x["addressing"]["R0_k2"].update({"rows_agree": 0.89}), "addressing:R0_k2", "a1")
    case(lambda x: x["addressing"]["C0_k3"].update({"cont_agree": 0.5}), "addressing:C0_k3", "a1")
    case(lambda x: x["mutants"]["c"].update({"rows_agree": 0.95, "cont_agree": 0.95}), "GATE_TOO_WEAK:c", "a1")
    proved(lambda x: x["mutants"]["c"].update({"rows_agree": 0.97, "cont_agree": 0.40}), "a1", "caught by continuations")
    # a2: the logit gate decides; agreement is reported only
    case(lambda x: x["addressing"]["R0_k2"].update({"mean_abs_dlogp": [0.01, 0.30, 0.01]}), "addressing:R0_k2")
    case(lambda x: x["addressing"]["C0_k1"].update({"mean_abs_dlogp": []}), "addressing:C0_k1")
    case(lambda x: x["mutants"]["c"].update({"mean_abs_dlogp": [0.2, 0.01, 0.01, 0.01]}), "GATE_TOO_WEAK:c")
    case(lambda x: x["mutants"]["a"].update({"mean_abs_dlogp": [None]}), "mutant:a:missing")
    proved(lambda x: x["addressing"]["C0_k3"].update({"rows_agree": 0.85, "cont_agree": 0.80}), "a2",
           "agreement below 0.90 with the logits inside the bound")
    proved(lambda x: x["mutants"]["c"].update({"rows_agree": 0.99, "cont_agree": 0.99,
                                               "mean_abs_dlogp": [1.18, 0.0, 0.0, 0.0]}), "a2", "caught by logits")
    # the calibration set
    cal = _sd2_prove_2()
    v1, v2 = judge(cal, g, "a1"), judge(cal, g, "a2")
    if v1["verdict"] != "FAILED" or v1["fails"] != ["GATE_TOO_WEAK:c"]:
        bad.append(f"sd2-prove-2 under a1: {v1['fails']}")
    if v2["verdict"] != "PROVED":
        bad.append(f"sd2-prove-2 under a2: {v2['fails']}")
    try:
        judge(p, g, "a3")
        bad.append("an unknown rule was accepted")
    except ValueError:
        pass
    if bad:
        print("sd2_reduce self-test FAILED:", bad)
        return 1
    print("sd2_reduce self-test OK (39 cases: both rules, and sd2-prove-2's calibration)")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--prove")
    ap.add_argument("--gpu-tests")
    ap.add_argument("--rule", choices=RULES, default="a2")
    ap.add_argument("--out")
    a = ap.parse_args(argv)
    if a.self_test:
        return self_test()
    if a.prove:
        rec = judge(json.load(open(a.prove)), json.load(open(a.gpu_tests)), a.rule)
        json.dump(rec, open(a.out, "w"), indent=1, sort_keys=True)
        print("SD2_PROVE_VERDICT " + json.dumps({"verdict": rec["verdict"], "fails": rec["fails"], "rule": rec["rule"]}),
              flush=True)
        return 0
    ap.print_usage()
    return 2


if __name__ == "__main__":
    sys.exit(main())
