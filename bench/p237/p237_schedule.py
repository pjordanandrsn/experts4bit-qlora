#!/usr/bin/env python3
"""p237_schedule.py -- P237's fixed process order (bench/p237/PREREG-p237.md "Design"), one line per process:

    INDEX KIND BUDGET ARM DRAW

KIND is ``primer`` (a 10-step cache-priming run of the arm's own configuration, recorded, never ruled), ``draw`` (a timed draw, 60
steps) or ``probes`` (UT's tuning probes, expanded by the runner once UA's first draw at that budget has recorded N_auto: 8-step
runs at N_auto - 1 .. N_auto - 4, stopping at the first OOM). BUDGET is ``12g`` (--vram-cap-gib 12) or ``full`` (the whole card).
Per budget: the two primers (E, then UA), then the palindrome E UA [probes] UT UT UA E. The 12 GiB budget runs first.

``--shake`` prints the shakedown's own order (registered, never a reading): E and UA at 12 GiB for 24 steps each, so loop indices
20..22 (the profiled window) are inside the run, one probe slot, and UA at the full card (where ``auto`` may swap nothing).
"""
from __future__ import annotations

import sys

BUDGETS = ("12g", "full")
PALINDROME = (("draw", "E", 1), ("draw", "UA", 1), ("probes", "UT", 0), ("draw", "UT", 1), ("draw", "UT", 2), ("draw", "UA", 2),
              ("draw", "E", 2))
SHAKE = (("12g", "draw", "E", 1), ("12g", "draw", "UA", 1), ("12g", "probes", "UT", 0), ("full", "draw", "UA", 1))


def schedule(shake=False):
    rows = []
    if shake:
        for budget, kind, arm, draw in SHAKE:
            rows.append({"kind": kind, "budget": budget, "arm": arm, "draw": draw})
    else:
        for budget in BUDGETS:
            rows += [{"kind": "primer", "budget": budget, "arm": arm, "draw": 0} for arm in ("E", "UA")]
            rows += [{"kind": k, "budget": budget, "arm": arm, "draw": d} for k, arm, d in PALINDROME]
    for i, r in enumerate(rows, 1):
        r["index"] = i
    return rows


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if argv not in ([], ["--shake"]):
        print("usage: p237_schedule.py [--shake]", file=sys.stderr)
        return 2
    for r in schedule(shake=argv == ["--shake"]):
        print(f"{r['index']} {r['kind']} {r['budget']} {r['arm']} {r['draw']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
