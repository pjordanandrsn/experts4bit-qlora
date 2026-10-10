#!/usr/bin/env python3
"""sd2_reduce.py -- lane SD2 (e4b#1313): the proof's verdict (bench/sd2/PREREG-sd2.md, Amendment 1).

`--prove prove.json --gpu-tests gpu_tests.json` -> `verdict_prove.json`. **PROVED** when every item holds; otherwise
**FAILED** with every failing item named. A mutant that PASSES the 0.90 addressing bar is named GATE_TOO_WEAK:<m>: the
gate cannot see that mutant, and Amendment 1's rule is that the gate is tightened before the read.

The items:
- `census`: buckets 1, 2, 3, 4, 8 and 16 each "graph"; post-verify graphs 2, 3 and 4 each "graph"; three hooks;
- `capture`: every verify bucket's replay bitwise its padded eager step;
- `addressing`: every (row, k) at row agreement and continuation agreement >= 0.90;
- `mutants`: every mutant's row agreement < 0.90;
- `draft`: in-engine drafts against chain_at at >= 0.99 of drafted ids;
- `transition`: one drop counted, both requests at their lengths, no mirror mismatch, no state left;
- `gpu_tests`: the target's GPU tests exited 0 with none skipped for want of the card.

`--self-test` runs every verdict.
"""
from __future__ import annotations

import argparse
import copy
import json
import sys

BAR_ADDRESS, BAR_DRAFT = 0.90, 0.99
BUCKETS = ("1", "2", "3", "4", "8", "16")
POSTS = ("2", "3", "4")


def judge(prove: dict, gpu: dict) -> dict:
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
        if not x or x.get("rows_agree", 0) < BAR_ADDRESS or x.get("cont_agree", 0) < BAR_ADDRESS:
            fails.append(f"addressing:{key}")
    muts = prove.get("mutants") or {}
    for m in ("a", "b", "c"):
        x = muts.get(m)
        if not x:
            fails.append(f"mutant:{m}:missing")
        elif x.get("rows_agree", 1.0) >= BAR_ADDRESS:
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
    return {"verdict": "PROVED" if not fails else "FAILED", "fails": fails,
            "addressing": {k: {kk: addr[k].get(kk) for kk in ("rows_agree", "cont_agree", "mean_abs_dlogp")}
                           for k in want if k in addr},
            "mutants": {m: (muts.get(m) or {}).get("rows_agree") for m in ("a", "b", "c")},
            "draft": d, "transition": t}


def _good() -> tuple:
    prove = {"census": {"graph_status": {b: "graph" for b in BUCKETS}, "post_graphs": {n: "graph" for n in POSTS},
                        "spec_build": {"hooks": 3}},
             "capture_bitwise": {n: True for n in POSTS},
             "addressing": {f"{r}_k{k}": {"rows_agree": 0.98, "cont_agree": 0.97, "mean_abs_dlogp": [0.01] * (k + 1)}
                            for r in ("R0", "C0") for k in (1, 2, 3)},
             "mutants": {m: {"rows_agree": 0.3} for m in ("a", "b", "c")},
             "draft": {"agree": 0.995, "same": 764, "of": 768},
             "transition": {"spec_steps_alone": 4, "dropped_batched": 1, "len_a": 64, "len_b": 16, "want": [64, 16],
                            "mirror_mismatches": 0, "state_left": 0}}
    return prove, {"rc": 0, "passed": 40, "skipped_for_card": 0}


def self_test() -> int:
    bad = []
    p, g = _good()
    if judge(p, g)["verdict"] != "PROVED":
        bad.append(f"good -> {judge(p, g)}")

    def case(edit, want, gpu_edit=None):
        p2, g2 = copy.deepcopy(p), dict(g)
        edit(p2)
        if gpu_edit:
            gpu_edit(g2)
        got = judge(p2, g2)
        if got["verdict"] != "FAILED" or want not in got["fails"]:
            bad.append(f"{want}: {got['fails']}")

    case(lambda x: x["census"]["graph_status"].update({"3": "eager: X"}), "census:bucket_3")
    case(lambda x: x["census"]["post_graphs"].pop("4"), "census:post_4")
    case(lambda x: x["census"]["spec_build"].update({"hooks": 2}), "census:hooks")
    case(lambda x: x["capture_bitwise"].update({"2": False}), "capture:2")
    case(lambda x: x["addressing"]["R0_k2"].update({"rows_agree": 0.89}), "addressing:R0_k2")
    case(lambda x: x["addressing"]["C0_k3"].update({"cont_agree": 0.5}), "addressing:C0_k3")
    case(lambda x: x["addressing"].pop("C0_k1"), "addressing:C0_k1")
    case(lambda x: x["mutants"]["c"].update({"rows_agree": 0.95}), "GATE_TOO_WEAK:c")
    case(lambda x: x["mutants"].pop("b"), "mutant:b:missing")
    case(lambda x: x["draft"].update({"agree": 0.98}), "draft")
    case(lambda x: x["transition"].update({"dropped_batched": 0}), "transition")
    case(lambda x: x["transition"].update({"len_a": 63}), "transition")
    case(lambda x: x["transition"].update({"mirror_mismatches": 2}), "transition")
    case(lambda x: None, "gpu_tests", lambda gg: gg.update({"rc": 1}))
    case(lambda x: None, "gpu_tests", lambda gg: gg.update({"skipped_for_card": 3}))
    if bad:
        print("sd2_reduce self-test FAILED:", bad)
        return 1
    print("sd2_reduce self-test OK (16 cases)")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--prove")
    ap.add_argument("--gpu-tests")
    ap.add_argument("--out")
    a = ap.parse_args(argv)
    if a.self_test:
        return self_test()
    if a.prove:
        rec = judge(json.load(open(a.prove)), json.load(open(a.gpu_tests)))
        json.dump(rec, open(a.out, "w"), indent=1, sort_keys=True)
        print("SD2_PROVE_VERDICT " + json.dumps({"verdict": rec["verdict"], "fails": rec["fails"]}), flush=True)
        return 0
    ap.print_usage()
    return 2


if __name__ == "__main__":
    sys.exit(main())
