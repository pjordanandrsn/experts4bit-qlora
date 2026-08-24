"""Verdict calculator for PREREG-productionization.md — four arms:
A1 (static), B (engine controller), C (engine controller + change-point),
A2 (static).

  python p_verdict.py --score A1.json B.json C.json A2.json

B1 parity: B's uniques reduction >= 12% AND within +/-3 points of the
C2 receipt (22.7%) — the engine-owned component reproduces the
driver-run value.
B2 boundary: over windows 1..9 (each starts at a content switch), the
CP arm's first-32-step uniques total <= 0.90 x the plain arm's, with
total reduction within 2 points of the plain arm's.
B3 wall (scoreable-only): if the dram-bucket delta of EACH controller
arm exceeds 3x the static-pair spread, both must show >= 5% reduction.
"""
import json
import sys

C2_ANCHOR = 0.227


def load(p):
    return json.load(open(p))


def first32(rep):
    return sum(w["uniq_first32"] for w in rep["windows"][1:]
               if w["uniq_first32"] is not None)


def score(a1p, bp, cp, a2p):
    a1, b, c, a2 = load(a1p), load(bp), load(cp), load(a2p)
    assert not a1["controller"] and not a2["controller"]
    assert b["controller"] and not b.get("controller_cp")
    assert c["controller"] and c.get("controller_cp")
    if a1["uniq_dram_total"] != a2["uniq_dram_total"]:
        print("DETERMINISM-BROKEN"); print("VERDICT: VOID"); return
    ua = a1["uniq_dram_total"]
    rb = 1 - b["uniq_dram_total"] / ua
    rc = 1 - c["uniq_dram_total"] / ua
    b1 = rb >= 0.12 and abs(rb - C2_ANCHOR) <= 0.03
    print("uniques: static %d | B %d (-%.1f%%) | C %d (-%.1f%%) | "
          "C2 anchor 22.7%%" % (ua, b["uniq_dram_total"], rb * 100,
                                c["uniq_dram_total"], rc * 100))
    print("B1 parity (>=12%% and |dB-22.7|<=3):",
          "PASS" if b1 else "FAIL")
    fb, fc = first32(b), first32(c)
    b2 = fc <= 0.90 * fb and rc >= rb - 0.02
    print("boundary first-32 uniques (w1..9): B %d | C %d (%.1f%% cut, "
          "bar >= 10%%); C total within 2pts of B: %s"
          % (fb, fc, 100 * (1 - fc / fb) if fb else 0,
             rc >= rb - 0.02))
    print("B2 boundary:", "PASS" if b2 else "FAIL")
    ta = (a1["dram_ms_grand"] + a2["dram_ms_grand"]) / 2
    spread = abs(a1["dram_ms_grand"] - a2["dram_ms_grand"])
    ok3 = True
    scoreable = True
    for name, arm in (("B", b), ("C", c)):
        delta = ta - arm["dram_ms_grand"]
        sc = delta > 3 * spread
        scoreable &= sc
        red = delta / ta
        print("wall %s: dram %.0f vs static %.0f (spread %.0f) -> "
              "-%.1f%% %s" % (name, arm["dram_ms_grand"], ta, spread,
                              red * 100,
                              "scoreable" if sc else "UNSCOREABLE"))
        if sc:
            ok3 &= red >= 0.05
    print("swaps B/C: %d/%d | cp_resets C: %d | controller_ms B/C: "
          "%.0f/%.0f" % (b["swaps_total"], c["swaps_total"],
                         c.get("cp_resets", 0), b["controller_ms"],
                         c["controller_ms"]))
    if not (b1 and b2):
        print("VERDICT: REFUTED")
        return
    if not scoreable:
        print("VERDICT: PROD-DECISION-ONLY (wall unscoreable)")
        return
    print("B3 wall (both arms >= 5%%):", "PASS" if ok3 else "FAIL")
    print("VERDICT:", "PROD-CERTIFIED" if ok3 else "REFUTED")


if __name__ == "__main__":
    score(*sys.argv[2:6])
