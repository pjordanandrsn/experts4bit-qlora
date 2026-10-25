#!/usr/bin/env python3
"""p238_schedule.py -- P238's fixed process schedule (PREREG "Level 2" and "Level 1"): one process per (draw, point, arm) step load,
plus two op processes per draw, each with its global index. The runner and the reducer both read it, so the order is written once.

- Draw 1: op-e4b, op-unsloth, then the points ascending (2k, 4k, 8k, field), each point's arms E_F E_A U0 U1 UA.
- Draw 2: op-unsloth, op-e4b, then the points descending (field, 8k, 4k, 2k), each point's arms reversed (UA U1 U0 E_A E_F).
So each point's two draws form a palindrome per arm, and every row carries its global index (1..44) and its position in its point.

``python3 p238_schedule.py [--draw N]`` prints one line per process: index, kind, draw, point, arm, position.
"""
from __future__ import annotations

import sys

POINTS = ("2k", "4k", "8k", "field")
ARMS = ("E_F", "E_A", "U0", "U1", "UA")
FRAMEWORK = {"E_F": "e4b", "E_A": "e4b", "U0": "unsloth", "U1": "unsloth", "UA": "unsloth"}
OPS = {1: ("e4b", "unsloth"), 2: ("unsloth", "e4b")}


def schedule():
    """[{index, kind ("op" | "step"), draw, point, arm, framework, position}] in run order."""
    out = []
    for draw in (1, 2):
        for fw in OPS[draw]:
            out.append({"kind": "op", "draw": draw, "point": None, "arm": None, "framework": fw, "position": None})
        points = POINTS if draw == 1 else tuple(reversed(POINTS))
        arms = ARMS if draw == 1 else tuple(reversed(ARMS))
        for pt in points:
            for pos, arm in enumerate(arms):
                out.append({"kind": "step", "draw": draw, "point": pt, "arm": arm, "framework": FRAMEWORK[arm], "position": pos})
    for i, row in enumerate(out, 1):
        row["index"] = i
    return out


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    draw = int(argv[argv.index("--draw") + 1]) if "--draw" in argv else None
    for r in schedule():
        if draw is None or r["draw"] == draw:
            print(r["index"], r["kind"], r["draw"], r["point"] or "-", r["arm"] or r["framework"], r["position"] if r["position"] is not None else "-")
    return 0


if __name__ == "__main__":
    sys.exit(main())
