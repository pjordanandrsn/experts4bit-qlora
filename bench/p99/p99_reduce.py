#!/usr/bin/env python3
"""Lane P99's reducer (bench/p99/PREREG-p99.md; e4b#913): localise P98's replay fault from seven arm processes, each a
fresh ``serve_paged.build_engine`` running ``bench/p98/p98_box.py``'s workloads (via ``p99_box.py``, which logs a
``P99_STEP`` line before every decode call).

Arms: ``d0g`` P98's arm g repeated (Qwen3.6, buckets 1-16, K25 at its default); ``d1g`` / ``d1e`` OLMoE-1B-7B (no
linear-attention state) graphs / its padded-eager oracle; ``d2g`` / ``d2e`` Qwen3.6 with ``E4B_NF4_GROUPED_SMALLM=0``
(K25 off); ``d3g`` / ``d3e`` Qwen3.6 with the single bucket 16.

Each arm is ``ran`` (its record exists), ``faulted`` (no record, and its log holds ``device-side assert triggered``), or
``error`` (anything else). A faulted arm's last ``P99_STEP`` line names the call and row count it died on.

VOID when the fault does not reproduce in ``d0g`` (then a clean arm says nothing), when an oracle arm (``d1e``, ``d2e``,
``d3e``) did not run, when any arm ended in ``error``, or when a record is off the registered shape.

Otherwise LOCALISED, with three answers, each read from one graph arm:
- ``hybrid_state_necessary``: ``d1g`` did not fault (OLMoE, same stack, no linear state);
- ``k25_necessary``: ``d2g`` did not fault (Qwen3.6 with K25 off);
- ``multiple_buckets_necessary``: ``d3g`` did not fault (Qwen3.6 with bucket 16 alone).

A graph arm that ran must also decode exactly as its oracle (W16 and W1 tokens); one that does not is reported as a
SILENT MISMATCH, a finding of its own, beside the answers.

    python p99_reduce.py --dir RUN_DIR --out verdict.json
    python p99_reduce.py --self-test
"""
from __future__ import annotations

import argparse
import copy
import json
import re
import sys
from pathlib import Path

ARMS = ("d0g", "d1g", "d1e", "d2g", "d2e", "d3g", "d3e")
PAIRS = {"d1g": "d1e", "d2g": "d2e", "d3g": "d3e"}
BUCKETS = {"d0g": [1, 2, 4, 8, 16], "d1g": [1, 2, 4, 8, 16], "d1e": [1, 2, 4, 8, 16], "d2g": [1, 2, 4, 8, 16],
           "d2e": [1, 2, 4, 8, 16], "d3g": [16], "d3e": [16]}
MODEL = {"d1g": "allenai/OLMoE-1B-7B-0924-Instruct", "d1e": "allenai/OLMoE-1B-7B-0924-Instruct"}
QWEN = "Qwen/Qwen3.6-35B-A3B"
SHAPE = {"n": 16, "prompt": 256, "b1_new": 96, "warm_steps": 3}
FAULT = "device-side assert triggered"


def classify(rec, log):
    """``(state, last_step)``: ran / faulted / error, and the last (call, rows) a faulted arm logged."""
    if rec is not None:
        return "ran", None
    steps = re.findall(r"P99_STEP call=(\d+) rows=(\d+)", log or "")
    last = (int(steps[-1][0]), int(steps[-1][1])) if steps else None
    return ("faulted" if FAULT in (log or "") else "error"), last


def _bucket(rows, buckets):
    return min((b for b in buckets if rows <= b), default=None)


def reduce(recs, logs):
    why, arms = [], {}
    for a in ARMS:
        state, last = classify(recs.get(a), logs.get(a))
        arms[a] = {"state": state, "last_step": last,
                   "fault_bucket": _bucket(last[1], BUCKETS[a]) if (state == "faulted" and last) else None}
        rec = recs.get(a)
        if rec is not None:
            want = MODEL.get(a, QWEN)
            if rec.get("model") != want:
                why.append(f"{a}: model {rec.get('model')} != {want}")
            if rec.get("buckets") != BUCKETS[a]:
                why.append(f"{a}: buckets {rec.get('buckets')} != {BUCKETS[a]}")
            for k, v in SHAPE.items():
                if rec.get(k) != v:
                    why.append(f"{a}: {k} {rec.get(k)!r} != {v}")
            rh = rec.get("rehearsal", {})
            if rh.get("stand_in_attention") is not False or rh.get("placement") != "all-vram":
                why.append(f"{a}: rehearsal knobs {rh}")
    if arms["d0g"]["state"] != "faulted":
        why.append(f"the fault did not reproduce: d0g {arms['d0g']['state']}")
    for e in PAIRS.values():
        if arms[e]["state"] != "ran":
            why.append(f"oracle {e} {arms[e]['state']}")
    for a, v in arms.items():
        if v["state"] == "error":
            why.append(f"{a} ended in error (no record, no device-side assert in its log)")
    mismatch = []
    for g, e in PAIRS.items():
        if arms[g]["state"] == "ran" and arms[e]["state"] == "ran":
            for w in ("w16", "w1"):
                if (recs[g].get(w) or {}).get("tokens") != (recs[e].get(w) or {}).get("tokens"):
                    mismatch.append(f"{g} vs {e}: {w} tokens differ")
    answers = {"hybrid_state_necessary": arms["d1g"]["state"] == "ran",
               "k25_necessary": arms["d2g"]["state"] == "ran",
               "multiple_buckets_necessary": arms["d3g"]["state"] == "ran"}
    out = {"verdict": "VOID" if why else "LOCALISED", "why": why, "arms": arms, "silent_mismatch": mismatch}
    if not why:
        out["answers"] = answers
    return out


def _fixture():
    def rec(a):
        return {"model": MODEL.get(a, QWEN), "buckets": BUCKETS[a], **SHAPE,
                "rehearsal": {"stand_in_attention": False, "placement": "all-vram"},
                "w16": {"tokens": [[1, 2], [3]]}, "w1": {"tokens": [[4]]}}
    recs = {a: rec(a) for a in ARMS if a != "d0g"}
    logs = {a: "P99_STEP call=0 rows=2\n" for a in ARMS}
    logs["d0g"] = "P99_STEP call=0 rows=2\nP99_STEP call=1 rows=4\nCUDA error: device-side assert triggered\n"
    return recs, logs


def self_test():
    base_r, base_l = _fixture()
    cases = []

    def case(name, want_verdict, want_answers=None, edit=None, want_mismatch=None):
        r, lg = copy.deepcopy(base_r), copy.deepcopy(base_l)
        if edit:
            edit(r, lg)
        got = reduce(r, lg)
        ok = got["verdict"] == want_verdict
        if ok and want_answers is not None:
            ok = got.get("answers") == want_answers
        if ok and want_mismatch is not None:
            ok = bool(got["silent_mismatch"]) == want_mismatch
        if ok and want_verdict == "LOCALISED" and name == "fixture":
            ok = got["arms"]["d0g"]["last_step"] == (1, 4) and got["arms"]["d0g"]["fault_bucket"] == 4
        cases.append((name, ok, got))

    def fault(arm):
        def edit(r, lg):
            r.pop(arm, None)
            lg[arm] = "P99_STEP call=3 rows=9\nCUDA error: device-side assert triggered\n"
        return edit

    all_yes = {"hybrid_state_necessary": True, "k25_necessary": True, "multiple_buckets_necessary": True}
    case("fixture", "LOCALISED", all_yes)
    case("OLMoE faults too", "LOCALISED", dict(all_yes, hybrid_state_necessary=False), fault("d1g"))
    case("faults with K25 off", "LOCALISED", dict(all_yes, k25_necessary=False), fault("d2g"))
    case("faults with one bucket", "LOCALISED", dict(all_yes, multiple_buckets_necessary=False), fault("d3g"))
    case("fault not reproduced", "VOID", None, lambda r, lg: (r.__setitem__("d0g", copy.deepcopy(r["d1g"])),
                                                               r["d0g"].__setitem__("model", QWEN),
                                                               r["d0g"].__setitem__("buckets", BUCKETS["d0g"])))
    case("an oracle did not run", "VOID", None, fault("d2e"))
    case("an arm errored otherwise", "VOID", None, lambda r, lg: (r.pop("d3g"), lg.__setitem__("d3g", "OOM\n")))
    case("rehearsal knob", "VOID", None, lambda r, lg: r["d1e"]["rehearsal"].__setitem__("placement", "solver"))
    case("off-shape record", "VOID", None, lambda r, lg: r["d2g"].__setitem__("prompt", 64))
    case("wrong model in the OLMoE arm", "VOID", None, lambda r, lg: r["d1g"].__setitem__("model", QWEN))
    case("silent mismatch is reported", "LOCALISED", all_yes,
         lambda r, lg: r["d2g"]["w16"]["tokens"][0].__setitem__(0, 9), True)
    bad = [c for c in cases if not c[1]]
    for name, _ok, got in bad:
        print(f"SELF-TEST FAIL {name}: {json.dumps(got)[:400]}")
    if bad:
        return 1
    print(f"self-test OK ({len(cases)} cases)")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir")
    ap.add_argument("--out")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()
    if a.self_test:
        return self_test()
    d = Path(a.dir)
    recs = {arm: (json.loads((d / f"arm_{arm}.json").read_text()) if (d / f"arm_{arm}.json").is_file() else None)
            for arm in ARMS}
    logs = {arm: ((d / "logs" / f"arm_{arm}.log").read_text(errors="replace")
                  if (d / "logs" / f"arm_{arm}.log").is_file() else None) for arm in ARMS}
    v = reduce(recs, logs)
    Path(a.out).write_text(json.dumps(v, indent=1))
    print(f"P99 VERDICT {v['verdict']}")
    for w in v["why"]:
        print(f"  void: {w}")
    for arm, s in v["arms"].items():
        print(f"  {arm}: {s['state']} last_step {s['last_step']} fault_bucket {s['fault_bucket']}")
    if "answers" in v:
        print(f"  answers {v['answers']}")
    for m in v["silent_mismatch"]:
        print(f"  SILENT MISMATCH: {m}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
