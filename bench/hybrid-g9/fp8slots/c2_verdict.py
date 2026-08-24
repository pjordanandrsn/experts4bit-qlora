"""Verdict calculator for PREREG-c2.md — the in-engine controller A/B/A.

  python c2_verdict.py --score A1.json B.json A2.json

B1 (decision, deterministic counters): controller uniq_dram_total <=
0.88 x static (the two static arms must agree exactly).
Wall scoreability: the measured dram-bucket delta must exceed 3x the
static-pair spread, else WALL-UNSCOREABLE (B1 still stands).
B2 (wall, when scoreable): dram_ms_grand reduction >= 5%.
C2-CERTIFIED = B1 and B2; C2-DECISION-ONLY = B1 and unscoreable;
REFUTED otherwise.
"""
import json
import sys


def load(p):
    return json.load(open(p))


def score(a1p, bp, a2p):
    a1, b, a2 = load(a1p), load(bp), load(a2p)
    assert not a1["controller"] and not a2["controller"] and b["controller"]
    if a1["uniq_dram_total"] != a2["uniq_dram_total"]:
        print("DETERMINISM-BROKEN: static arms disagree on uniques "
              f"({a1['uniq_dram_total']} vs {a2['uniq_dram_total']})")
        print("VERDICT: VOID")
        return
    ua, ub = a1["uniq_dram_total"], b["uniq_dram_total"]
    red_u = 1 - ub / ua
    b1 = red_u >= 0.12
    ta = (a1["dram_ms_grand"] + a2["dram_ms_grand"]) / 2
    spread = abs(a1["dram_ms_grand"] - a2["dram_ms_grand"])
    delta = ta - b["dram_ms_grand"]
    red_t = delta / ta
    scoreable = delta > 3 * spread
    print("uniques: static %d | controller %d | reduction %.1f%% "
          "(bar >= 12%%) -> %s" % (ua, ub, red_u * 100,
                                   "B1-PASS" if b1 else "B1-FAIL"))
    print("dram ms: static %.0f/%.0f (spread %.0f) | controller %.0f | "
          "delta %.0f (%.1f%%)" % (a1["dram_ms_grand"], a2["dram_ms_grand"],
                                   spread, b["dram_ms_grand"], delta,
                                   red_t * 100))
    print("swaps %d | controller overhead %.0f ms | wall grand "
          "%.0f vs %.0f/%.0f ms (unscored)"
          % (b["swaps_total"], b["controller_ms"], b["wall_ms_grand"],
             a1["wall_ms_grand"], a2["wall_ms_grand"]))
    if not b1:
        print("VERDICT: REFUTED (decision value did not survive the "
              "engine)")
        return
    if not scoreable:
        print("wall: UNSCOREABLE (delta %.0f <= 3x spread %.0f)"
              % (delta, 3 * spread))
        print("VERDICT: C2-DECISION-ONLY")
        return
    b2 = red_t >= 0.05
    print("B2 wall reduction >= 5%%:", "PASS" if b2 else "FAIL")
    print("VERDICT:", "C2-CERTIFIED" if b2 else "REFUTED")


if __name__ == "__main__":
    score(sys.argv[2], sys.argv[3], sys.argv[4])
