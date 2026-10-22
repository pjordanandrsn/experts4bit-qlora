#!/usr/bin/env python3
# Copyright (c) 2026 Cerin Amroth LLC. MIT.
"""The cliff census's speed arms: P136's unprofiled default-server box (bench/p136/p136_box.py, staged at its pinned bytes)
with P109's workloads widened to W1, W2, W4 and W16, so each census batch has its own unprofiled step.

EXPLORE tier (bench/cliff/README.md): not a reading. P109's prompts hold 16 rows, so W2 and W4 take the first 2 and 4 rows,
as P136's census does for its batch.

    python cliff_box.py --prompts prompts.json --out arm_S1a.json --tag S1a [--short 32 --long 160 --reps 3]
    python cliff_box.py --self-test
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import p109_box  # noqa: E402  (staged at P109's registered bytes)
import p136_box  # noqa: E402  (staged at P136's pinned bytes)

P109_WORKLOADS = {"W16": 16, "W1": 1}
WORKLOADS = {"W16": 16, "W1": 1, "W2": 2, "W4": 4}


def self_test() -> int:
    cases = [
        ("P109's workloads at its staged bytes", p109_box.WORKLOADS == P109_WORKLOADS),
        ("P136's box self-test passes on them", p136_box.self_test() == 0),
        ("every widened batch fits P109's rows", all(b <= p109_box.ROWS for b in WORKLOADS.values())),
        ("W16 stays (P136's arm line reads it)", WORKLOADS.get("W16") == 16),
        ("W1 stays first among the small batches", list(WORKLOADS)[1] == "W1"),
    ]
    bad = [n for n, ok in cases if not ok]
    print(f"cliff_box self-test {'OK' if not bad else 'FAILED ' + str(bad)} ({len(cases)} cases)")
    return 0 if not bad else 1


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if "--self-test" in argv:
        return self_test()
    p109_box.WORKLOADS = dict(WORKLOADS)
    return p136_box.main(argv)


if __name__ == "__main__":
    sys.exit(main())
