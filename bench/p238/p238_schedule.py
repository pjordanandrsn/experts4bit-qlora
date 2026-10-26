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
# The box cap (PREREG "The split rule"): min(the rental policy's per-run override ceiling, which the launcher refuses above, and
# the hours the USD 15 run-worst buys at the box's rate after the downloads). The constants are pinned by the kit test.
POLICY_MAX_WALLCLOCK_H = 6                           # the policy's global wallclock cap: a run without an override
RUN_OVERRIDE_MAX_H = 18                              # the per-run override's ceiling (allowed only when the run's worst <= USD 15)
RUN_WORST_USD, USD_PER_H_MAX, DOWNLOAD_USD = 15.0, 1.05, 1.64
INSTALL_S, FETCH_S, OP_TIMING_S, RESERVE_S, HEADROOM_PCT = 2700, 2400, 1800, 900, 125   # the runner's guard fit (the kit test binds them)


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


def fit_s(n_proc, load_s):
    """The runner's guard fit for a box of ``n_proc`` processes at ``load_s`` s each: installs + the snapshot fetch + the processes
    + the op timing + the reserve, x 1.25 (integer, as the runner's shell arithmetic)."""
    return (INSTALL_S + FETCH_S + n_proc * load_s + OP_TIMING_S + RESERVE_S) * HEADROOM_PCT // 100


def box_cap_s(usd_per_h=USD_PER_H_MAX):
    return int(min(RUN_OVERRIDE_MAX_H, (RUN_WORST_USD - DOWNLOAD_USD) / usd_per_h) * 3600)


def needs_override(n_proc, load_s):
    """True when a box's guard fit exceeds the policy's global cap: that box runs only under the per-run override."""
    return fit_s(n_proc, load_s) > POLICY_MAX_WALLCLOCK_H * 3600


SPLITS = (1, 2, 4, 8)                                # K boxes: whole; by draw; each draw in 2 point groups; each (draw, point) alone
APPROVED_K_MAX = 2                                      # the lane cost approved for this registration: K = 4 or 8 needs a new approval


def blocks(k):
    """The K boxes of a split reading, box 1..K, each the sorted global indices it runs: whole (draw, point) blocks, never split by
    arm (every box holds all 5 arms of its points, so every ratio stays within one box). K = 1 is the whole schedule; otherwise each
    draw's points, in that draw's run order, are cut into K/2 contiguous groups, and a draw's two op processes ride in its first box."""
    s = schedule()
    if k == 1:
        return [[r["index"] for r in s]]
    if k not in SPLITS:
        raise ValueError(f"K {k} is not one of {SPLITS}")
    g, out = k // 2, []
    for d in (1, 2):
        order = list(POINTS if d == 1 else reversed(POINTS))
        size = len(order) // g
        for j in range(g):
            grp = set(order[j * size:(j + 1) * size])
            out.append([r["index"] for r in s if r["draw"] == d and ((r["kind"] == "op" and j == 0) or r["point"] in grp)])
    return out


def box_rows(spec):
    """``spec``: "all" (the whole schedule) or "K-i" (box i of a K-box split) -> its rows in run order; ValueError otherwise."""
    if spec == "all":
        return schedule()
    k, _, i = spec.partition("-")
    if not (k.isdigit() and i.isdigit()) or int(k) not in SPLITS[1:] or not 1 <= int(i) <= int(k):
        raise ValueError(f"box {spec!r}: 'all', or K-i with K in {SPLITS[1:]} and 1 <= i <= K")
    want = set(blocks(int(k))[int(i) - 1])
    return [r for r in schedule() if r["index"] in want]


def fewest_k(load_s, usd_per_h=USD_PER_H_MAX):
    """The fewest K whose largest box fits the box cap (K = 1 whenever the cap allows), else None."""
    cap = box_cap_s(usd_per_h)
    for k in SPLITS:
        if max(fit_s(len(b), load_s) for b in blocks(k)) <= cap:
            return k
    return None


def plan(load_s, usd_per_h=USD_PER_H_MAX):
    """The reading's box count, from the shakedown's measured per-load wall, decided before the reading launches: the fewest K
    that fits (fewest_k) when it is at most APPROVED_K_MAX; "NEEDS_APPROVAL" when only K = 4 or 8 fits (no reading launches: a new
    approval with the measured wall); None when no registered split fits (no reading launches)."""
    k = fewest_k(load_s, usd_per_h)
    if k is None:
        return None
    return k if k <= APPROVED_K_MAX else "NEEDS_APPROVAL"


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if "--plan" in argv:                                 # --plan LOAD_S [USD_PER_H] -> K, NEEDS_APPROVAL or NO_FIT
        rest = argv[argv.index("--plan") + 1:]
        k = plan(int(rest[0]), float(rest[1]) if len(rest) > 1 else USD_PER_H_MAX)
        print("NO_FIT" if k is None else k)
        return 0
    if "--box" in argv:
        rows = box_rows(argv[argv.index("--box") + 1])
    else:
        draw = int(argv[argv.index("--draw") + 1]) if "--draw" in argv else None
        rows = [r for r in schedule() if draw is None or r["draw"] == draw]
    for r in rows:
        print(r["index"], r["kind"], r["draw"], r["point"] or "-", r["arm"] or r["framework"], r["position"] if r["position"] is not None else "-")
    return 0


if __name__ == "__main__":
    sys.exit(main())
