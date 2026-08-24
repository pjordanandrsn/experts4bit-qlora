"""Verdict calculator for PREREG-g8close.md — closing the campaign's
open G8 B=16 balance gate (>= 0.80).

  python g8close_verdict.py --score A1.json C.json R.json A2.json

The metric is the g8 receipts' own: balance_ratio = min(gpu_ms_median,
dram_ms_median) / max(...), per-decode-step medians over the whole
serve. Arms: A1/A2 static mass-spread (the A/A pair), C = static
placement + SlotController, R = their-calibration placement (context
only, never scored). The gate CLOSES on the best serving arm among
{mean(A), C} at >= 0.80 with margin: |best - 0.80| must exceed 3x the
A/A balance spread, else UNDETERMINED. Balance is the gate's
definition, not the program's objective — walls are reported beside it.
"""
import json
import sys


def load(p):
    return json.load(open(p))


def bal(rep):
    return rep["balance_ratio"]


def score(a1p, cp, rp, a2p):
    a1, c, r, a2 = load(a1p), load(cp), load(rp), load(a2p)
    assert not a1["controller"] and not a2["controller"] and not r["controller"]
    assert c["controller"] and not c.get("controller_cp"), \
        "arm C must be the plain controller (cp=False, as registered)" 
    if a1["uniq_dram_total"] != a2["uniq_dram_total"]:
        print("DETERMINISM-BROKEN"); print("VERDICT: VOID"); return
    spread = abs(bal(a1) - bal(a2))
    a_mean = (bal(a1) + bal(a2)) / 2
    for name, rep in (("A1 static", a1), ("A2 static", a2),
                      ("C  ctrl", c), ("R  g8-calib", r)):
        print("%-12s balance %.3f  (gpu %.2f ms | dram %.2f ms | "
              "dram wall %.0f ms)" % (name, bal(rep),
                                      rep["gpu_ms_median"],
                                      rep["dram_ms_median"],
                                      rep["dram_ms_grand"]))
    print("A/A balance spread: %.3f" % spread)
    cands = {"static (mean A)": a_mean, "controller": bal(c)}
    best_name = max(cands, key=lambda k: cands[k])
    best = cands[best_name]
    print("best serving arm: %s at %.3f (gate >= 0.80)"
          % (best_name, best))
    if best >= 0.80 and (best - 0.80) > 3 * spread:
        print("VERDICT: G8-B16-CLOSED (by the %s arm)" % best_name)
    elif abs(best - 0.80) <= 3 * spread:
        print("VERDICT: UNDETERMINED (|%.3f - 0.80| <= 3x spread %.3f)"
              % (best, spread))
    else:
        print("VERDICT: NOT-CLOSED (best %.3f < 0.80 beyond noise)"
              % best)


if __name__ == "__main__":
    score(*sys.argv[2:6])
