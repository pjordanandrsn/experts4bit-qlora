#!/usr/bin/env python3
"""sc2j_quality.py -- lane SC2j's QUALITY verdict (bench/sc2/SC2j-PREREG.md, Amendment 1): P117's teacher-forced width
check, reduced by P117's own reducer and bar, at box Q's stack.

``p117_reduce.py`` (staged beside this file) holds two constants that name P117's registration rather than its method:
the grouped-nf4-gemm commit P117 ran (v0.42.0) and P117's window count per model (Qwen3 64, Granite 40). Box Q installs
grouped-nf4-gemm v0.50.0, and its proof smokes 8 Granite windows, so P117's reducer read VOID on every box Q record
(sc2j-r3). This wrapper sets exactly those two constants to box Q's registered values and calls ``p117_reduce.reduce``;
everything else, the bar (tol, spread), the floor draws, the engagement and FUNCTION checks and the model revisions, is
P117's, unchanged.

It fails closed. The box record must carry box Q's grouped-nf4-gemm commit, which the box exports to ``p117_box.py`` as
``GNF4_SHA``. ``--gnf4-sha``, the commit the box installed, must equal the registered ``GNF4_SHA`` below, and the record's
window count must be the registered count for its model. Any mismatch is VOID, through P117's own checks.

  --dir D --out FILE --e4b-sha SHA --gnf4-sha SHA
  --self-test
"""
import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [_HERE, os.path.join(_HERE, "..", "p117")]   # staged flat on the box; bench/p117 in the repo
import p117_reduce as pr  # noqa: E402

GNF4_SHA = "41ef086e6a3e2be5b158ff1bb38e0de67862a949"   # grouped-nf4-gemm v0.50.0, box Q's pin (bench/sc1/sc1_run.sh)
WINDOWS = {"Qwen/Qwen3-30B-A3B": 64,                       # the reading: P117's own count
           "ibm-granite/granite-3.1-3b-a800m-instruct": 8}  # the proof's smoke (SC2j-PREREG, "The QUALITY arm is smoked on 8 windows")
P117_GNF4_SHA, P117_WINDOWS = pr.GNF4_SHA, dict(pr.WINDOWS)


def at_box_q():
    """P117's reducer with box Q's two stack constants."""
    pr.GNF4_SHA, pr.WINDOWS = GNF4_SHA, dict(WINDOWS)


def reduce(run: Path, e4b_sha: str, gnf4_sha: str) -> dict:
    at_box_q()
    v = pr.reduce(run, e4b_sha)
    if gnf4_sha != GNF4_SHA:
        v = {**v, "verdict": "VOID", "reasons": [f"the box installed grouped-nf4-gemm {gnf4_sha!r}, not box Q's {GNF4_SHA}"]
             + list(v.get("reasons") or [])}
    v["lane"] = "SC2j QUALITY (P117's reducer and bar)"
    v["constants"] = {"gnf4_sha": GNF4_SHA, "windows": WINDOWS, "p117_registered": {"gnf4_sha": P117_GNF4_SHA,
                                                                                    "windows": P117_WINDOWS}}
    return v


def self_test() -> int:
    at_box_q()
    big = {"mutant_scale": 0.8}                        # P117's own self-test offset: the mutant sits far outside the bar
    cases = []

    def run(box, summary="premise ok\n", gnf4=GNF4_SHA, e4b="a" * 40):
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "summary.txt").write_text(summary)
            if box is not None:
                (Path(d) / "box.json").write_text(json.dumps(box))
            return reduce(Path(d), e4b, gnf4)["verdict"]

    gr = "ibm-granite/granite-3.1-3b-a800m-instruct"
    cases.append(("granite proof, 8 windows, box Q's gnf4", run(pr._box(model=gr, d=big)), "AT_PARITY"))
    cases.append(("qwen3 reading, 64 windows", run(pr._box(d=big)), "AT_PARITY"))
    b = pr._box(model=gr, d=big)
    b["gnf4_sha"] = None
    cases.append(("the box did not record GNF4_SHA", run(b), "VOID"))
    b = pr._box(model=gr, d=big)
    b["gnf4_sha"] = P117_GNF4_SHA
    cases.append(("P117's gnf4, not box Q's", run(b), "VOID"))
    cases.append(("the box installed another gnf4", run(pr._box(model=gr, d=big), gnf4="0" * 40), "VOID"))
    cases.append(("the installed gnf4 missing", run(pr._box(model=gr, d=big), gnf4=""), "VOID"))
    pr.WINDOWS = {**WINDOWS, gr: P117_WINDOWS[gr]}
    b40 = pr._box(model=gr, d=big)
    at_box_q()
    cases.append(("P117's 40 Granite windows, not the proof's 8", run(b40), "VOID"))
    cases.append(("e4b differs", run(pr._box(model=gr, d=big), e4b="b" * 40), "VOID"))
    cases.append(("premise failed", run(pr._box(model=gr, d=big), summary="premise failed\n"), "NO_READING"))
    cases.append(("no box record", run(None), "NO_READING"))
    cases.append(("a width cost", run(pr._box(model=gr, d={**big, "W64": 0.09})), "COST"))
    fails = 0
    for name, got, want in cases:
        fails += got != want
        print(f"SC2J_QUALITY self-test {'ok ' if got == want else 'BAD'} {name}: {got} (want {want})")
    print(f"SC2J_QUALITY self-test {'PASS' if not fails else 'FAIL'} ({len(cases) - fails}/{len(cases)})")
    return 1 if fails else 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--dir")
    ap.add_argument("--out")
    ap.add_argument("--e4b-sha", default=os.environ.get("E4B_SHA", ""))
    ap.add_argument("--gnf4-sha", default=os.environ.get("GNF4_SHA", ""))
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args(argv)
    if a.self_test:
        return self_test()
    if not a.dir or not a.out:
        ap.error("--dir and --out are required")
    v = reduce(Path(a.dir), a.e4b_sha, a.gnf4_sha)
    with open(a.out, "w") as f:
        json.dump(v, f, indent=1)
    print(f"SC2J_QUALITY_VERDICT {v['verdict']} {json.dumps(v.get('reasons') or [])}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
