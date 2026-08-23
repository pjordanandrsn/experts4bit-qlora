"""Verdict calculator for PREREG-regime.md — the regime-split conversion
constants (input (b)). Pure arithmetic over one A/A-gated ladder sweep.

  python regime_verdict.py --score sweep.json

Model: per-step DRAM-tier time T = c_fix * C + c_u * U, with U =
uniques/step and C = layers-with-any-DRAM-work/step. Brackets between
adjacent ladder points; (c_fix, c_u) least-squares fitted on the ODD
brackets, scored on the EVEN brackets. The flat 58 us/unique null must
fail where it failed in the slot-value cycle, or the run is
UNINFORMATIVE.
"""
import json
import sys

US_FLAT = 58.0


def load(path):
    d = json.load(open(path))
    ladder = sorted(d["ladder"])
    pts = d["points"]

    def point(pn, v):
        return pts[f"p{pn}_v{v:g}"]
    return ladder, point


def dram_ms(p):
    return p["decode_median_ms"]["dram_experts_host"]


def uniq(p):
    return sum(pl["uniq_dram"] for pl in p["per_layer"]) \
        / max(1, p["per_layer"][0]["steps"])


def calls(p):
    return sum(pl["dram_steps"] for pl in p["per_layer"]) \
        / max(1, p["per_layer"][0]["steps"])


def gate(ladder, point):
    noises = []
    for v in ladder:
        a, b = dram_ms(point(1, v)), dram_ms(point(2, v))
        n = abs(a - b) / ((a + b) / 2)
        noises.append(n)
        print("gate v=%-5g dram %7.2f / %7.2f ms  noise %5.1f%%%s"
              % (v, a, b, n * 100, "  FAIL(>10%)" if n > 0.10 else ""))
    med = sorted(noises)[len(noises) // 2]
    ok = med <= 0.05 and max(noises) <= 0.10
    print("gate median %.1f%% (<=5%%), worst %.1f%% (<=10%%) -> %s"
          % (med * 100, max(noises) * 100,
             "ACCEPT" if ok else "REJECT -- destroy and re-hunt"))
    return ok


def score(path):
    ladder, point = load(path)
    if not gate(ladder, point):
        print("VERDICT: VOID (gate failed)")
        return
    for v in ladder:
        if abs(uniq(point(1, v)) - uniq(point(2, v))) > 1e-9 \
                or abs(calls(point(1, v)) - calls(point(2, v))) > 1e-9:
            print("DETERMINISM-BROKEN at v=%g" % v)
            print("VERDICT: VOID (counters differ across passes)")
            return
    br = []
    for lo, hi in zip(ladder, ladder[1:]):
        dT = [dram_ms(point(pn, lo)) - dram_ms(point(pn, hi))
              for pn in (1, 2)]
        dU = uniq(point(1, lo)) - uniq(point(1, hi))
        dC = calls(point(1, lo)) - calls(point(1, hi))
        br.append({"lo": lo, "hi": hi, "dT": sum(dT) / 2,
                   "spread": abs(dT[0] - dT[1]), "dU": dU, "dC": dC})
    fit = br[0::2]
    scored = br[1::2]
    sxx = sum(b["dC"] ** 2 for b in fit)
    sxy = sum(b["dC"] * b["dU"] for b in fit)
    syy = sum(b["dU"] ** 2 for b in fit)
    stx = sum(b["dC"] * b["dT"] * 1e3 for b in fit)
    sty = sum(b["dU"] * b["dT"] * 1e3 for b in fit)
    det = sxx * syy - sxy * sxy
    if abs(det) < 1e-9:
        print("VERDICT: VOID (degenerate fit -- dC and dU collinear on "
              "fit brackets)")
        return
    c_fix = (stx * syy - sty * sxy) / det
    c_u = (sty * sxx - stx * sxy) / det
    print("fitted: c_fix = %.1f us/layer-step, c_u = %.1f us/unique"
          % (c_fix, c_u))
    ok_n, null_miss = 0, 0
    for b in scored:
        pred = (c_fix * b["dC"] + c_u * b["dU"]) / 1e3
        allow = max(0.20 * abs(pred), 3 * b["spread"])
        good = abs(b["dT"] - pred) <= allow
        ok_n += good
        pn = US_FLAT * b["dU"] / 1e3
        # the null is held to the SAME allowance the model faces -- "must
        # fail where it failed before" means at the bars, not at 2x them
        nmiss = abs(b["dT"] - pn) > allow
        null_miss += nmiss
        print("scored %g->%g  dU %6.1f dC %5.2f | dT %6.2f  model %6.2f "
              "(allow %5.2f) %s | flat58 %6.2f %s"
              % (b["lo"], b["hi"], b["dU"], b["dC"], b["dT"], pred, allow,
                 "PASS" if good else "FAIL", pn,
                 "misses" if nmiss else "fits"))
    ss_res = ss_tot = 0.0
    mu = sum(b["dT"] for b in br) / len(br)
    for b in br:
        pred = (c_fix * b["dC"] + c_u * b["dU"]) / 1e3
        ss_res += (b["dT"] - pred) ** 2
        ss_tot += (b["dT"] - mu) ** 2
    print("R2 over all %d brackets: %.3f" % (len(br), 1 - ss_res / ss_tot))
    if null_miss < 2:
        print("VERDICT: UNINFORMATIVE (the flat-58 null survives the "
              "scored brackets -- no regime signal on this ladder)")
        return
    # only meaningful once regime signal exists: a noise-driven negative
    # coefficient in a flat world must not preempt the null check
    if c_u <= 0 or c_fix < 0:
        print("VERDICT: REFUTED (regime signal present but the fitted "
              "form is unphysical)")
        return
    print("VERDICT:", "REGIME-CERTIFIED" if ok_n >= 3 else "REFUTED",
          "(%d/4 scored brackets in bar)" % ok_n)


if __name__ == "__main__":
    score(sys.argv[2])
