"""Verdict calculator for PREREG-coroute.md — the within-step co-routing
model re-derivation. Pure arithmetic over receipts, frozen pre-box.

  python coroute_verdict.py --score W1.amort.json W2_sweep.json

W1 (calibration window): per-expert step-touch counts -> P_hat.
W2 (held-out window): the standard ladder sweep; bracket dU is scored
against three models:
  M1  direct transfer:  sum of P_hat over bracket experts (linearity,
      no independence assumption)              -- the registered model
  M2  one-parameter fit: P = 1-(1-p)^B_eff, B_eff fitted on W1
  M0  independence null: P = 1-(1-p)^B         -- must FAIL again
Uniques are deterministic counters per window (greedy decode), so there
is no box-noise gate; the bars carry binomial window-sampling error.
"""
import json
import math
import sys


def load_amort(path):
    return json.load(open(path))


def per_expert(cal):
    """(layer, e) -> (P_hat, p_marginal); plus steps."""
    S = cal["per_layer"][0]["steps"]
    B = cal["batch"]
    out = {}
    for li, pl in enumerate(cal["per_layer"]):
        touch, hist = pl["touch"], pl["hist"]
        for e in range(len(touch)):
            p_tok = hist[e] / max(1, S * B)          # P(a token draws e)
            out[(li, e)] = (touch[e] / max(1, S), min(1.0, p_tok))
    return out, S


def fit_beff(pe, B):
    best, best_err = 1.0, float("inf")
    cand = [1 + i * 0.05 for i in range(int((B - 1) / 0.05) + 1)]
    for beff in cand:
        err = 0.0
        for (ph, p) in pe.values():
            if p <= 0:
                continue
            err += (ph - (1 - (1 - p) ** beff)) ** 2
        if err < best_err:
            best, best_err = beff, err
    return best


def score(cal_path, sweep_path):
    cal = load_amort(cal_path)
    d = json.load(open(sweep_path))
    ladder = sorted(d["ladder"])
    pts = d["points"]
    B = cal["batch"]
    pe, S1 = per_expert(cal)
    beff = fit_beff(pe, B)
    print("calibration: %d steps, B=%d, fitted B_eff = %.2f" % (S1, B, beff))

    def point(pn, v):
        return pts[f"p{pn}_v{v:g}"]

    def uniq(p):
        return sum(pl["uniq_dram"] for pl in p["per_layer"]) \
            / max(1, p["per_layer"][0]["steps"])

    for v in ladder:
        u1, u2 = uniq(point(1, v)), uniq(point(2, v))
        if abs(u1 - u2) > 1e-9:
            print("DETERMINISM-BROKEN at v=%g: %.3f vs %.3f" % (v, u1, u2))
            print("VERDICT: VOID (uniques not deterministic on this stack)")
            return
        nv = sum(pl["uniq_nvme"] for p_ in (point(1, v), point(2, v))
                 for pl in p_["per_layer"])
        assert nv == 0, "NVMe tier touched"
    S2 = point(1, ladder[0])["per_layer"][0]["steps"]

    def vset(p):
        return set(map(tuple, p["manifest_vram"]))

    m1_ok, m2_ok, m0_fail = True, 0, 0
    n_br = 0
    for lo, hi in zip(ladder, ladder[1:]):
        n_br += 1
        added = vset(point(1, hi)) - vset(point(1, lo))
        du = uniq(point(1, lo)) - uniq(point(1, hi))
        p1 = sum(pe[k][0] for k in added)
        p2 = sum(1 - (1 - pe[k][1]) ** beff for k in added)
        p0 = sum(1 - (1 - pe[k][1]) ** B for k in added)
        var = sum(pe[k][0] * (1 - pe[k][0]) for k in added)
        sig = math.sqrt(max(var, 1e-12) * (1 / S1 + 1 / S2))
        a1 = max(0.10 * p1, 3 * sig)
        a2 = max(0.15 * p2, 3 * sig)
        a0 = max(0.15 * p0, 3 * sig)
        ok1 = abs(du - p1) <= a1
        ok2 = abs(du - p2) <= a2
        fail0 = abs(du - p0) > a0
        m1_ok &= ok1
        m2_ok += ok2
        m0_fail += fail0
        print("bracket %g->%g +%d | dU %7.1f | M1 %7.1f(±%5.1f)%s | "
              "M2 %7.1f(±%5.1f)%s | M0 %7.1f %s"
              % (lo, hi, len(added), du, p1, a1,
                 " PASS" if ok1 else " FAIL", p2, a2,
                 " PASS" if ok2 else " FAIL", p0,
                 "refuted" if fail0 else "NOT-refuted"))
    if m0_fail == 0:
        print("VERDICT: UNINFORMATIVE (the independence null passes on "
              "this window pair -- no co-routing signal to model)")
        return
    print("M2 (B_eff=%.2f): %d/%d brackets in bar" % (beff, m2_ok, n_br))
    print("VERDICT:", "M1-CERTIFIED" if m1_ok else "M1-REFUTED")


if __name__ == "__main__":
    score(sys.argv[2], sys.argv[3])
