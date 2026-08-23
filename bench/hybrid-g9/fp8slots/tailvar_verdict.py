"""Verdict calculator for PREREG-tailvar.md. Pure arithmetic, frozen
pre-box.

  python tailvar_verdict.py --score <windows_dir> <bracket_source_dir>

bracket_source_dir holds the committed co-routing sweep amort receipts
(p1_v7 / p1_v9 / p1_v10 / p1_v10.75 / p1_v11.5): bracket b_i = the
manifest-VRAM set difference between adjacent ladder points -- frozen
data, no on-box placement.

Per window w and bracket b: M_b(w) = sum over bracket experts of
touch_e / steps (the bracket's expected dU contribution). Windows are
split chronologically: w0-w4 fit, w5-w9 held out.

H1 structure: fit-set cv of the LAST (tail) bracket >= 2x the FIRST
(deepest); the argmax/argmin-cv brackets replicate on the held-out set.
H2 envelopes: |M - mean_fit| <= 2 cv_b mean_fit + 3 sigma_binom covers
>= 18/20 held-out cells.
H3 control: binomial-only envelopes cover < 70% of held-out cells,
else UNINFORMATIVE.
"""
import json
import math
import pathlib
import sys

LADDER = ["7", "9", "10", "10.75", "11.5"]


def brackets_from(src):
    sets = []
    for v in LADDER:
        d = json.load(open(pathlib.Path(src) / f"p1_v{v}.amort.json"))
        sets.append(set(map(tuple, d["manifest_vram"])))
    return [sets[i + 1] - sets[i] for i in range(len(sets) - 1)]


def window_masses(wdir, brackets):
    idx = json.load(open(pathlib.Path(wdir) / "windows.json"))
    out = []
    for tag in idx["windows"]:
        d = json.load(open(pathlib.Path(wdir) / f"{tag}.amort.json"))
        S = d["per_layer"][0]["steps"]
        nv = sum(pl["uniq_nvme"] for pl in d["per_layer"])
        assert nv == 0, f"{tag}: NVMe touched"
        touch = [pl["touch"] for pl in d["per_layer"]]
        row = []
        for b in brackets:
            m = sum(touch[l][e] for (l, e) in b) / S
            var = sum((touch[l][e] / S) * (1 - touch[l][e] / S)
                      for (l, e) in b) / S
            row.append((m, math.sqrt(max(var, 1e-12))))
        out.append(row)
    return out


def cv(xs):
    n = len(xs)
    mu = sum(xs) / n
    sd = math.sqrt(sum((x - mu) ** 2 for x in xs) / (n - 1))
    return sd / mu if mu > 0 else 0.0, mu


def score(wdir, src):
    br = brackets_from(src)
    ms = window_masses(wdir, br)
    n_fit = 5
    fit, held = ms[:n_fit], ms[n_fit:]
    print("bracket sizes:", [len(b) for b in br])
    cvs_fit, mus, cvs_held = [], [], []
    for bi in range(len(br)):
        c, mu = cv([w[bi][0] for w in fit])
        ch, _ = cv([w[bi][0] for w in held])
        cvs_fit.append(c); mus.append(mu); cvs_held.append(ch)
        print("b%d  fit mean %8.2f  cv %5.1f%%  | held cv %5.1f%%"
              % (bi + 1, mu, c * 100, ch * 100))
    h1 = (cvs_fit[-1] >= 2 * cvs_fit[0]
          and cvs_fit.index(max(cvs_fit)) == cvs_held.index(max(cvs_held))
          and cvs_fit.index(min(cvs_fit)) == cvs_held.index(min(cvs_held)))
    amax = cvs_fit.index(max(cvs_fit)) == cvs_held.index(max(cvs_held))
    amin = cvs_fit.index(min(cvs_fit)) == cvs_held.index(min(cvs_held))
    print("H1 structure: tail cv %.1f%% vs 2x deep %.1f%%; argmax "
          "replicates: %s; argmin replicates: %s -> %s"
          % (cvs_fit[-1] * 100, 2 * cvs_fit[0] * 100, amax, amin,
             "PASS" if h1 else "FAIL"))
    cov2, covb, cells = 0, 0, 0
    for w in held:
        for bi in range(len(br)):
            m, sig = w[bi]
            cells += 1
            if abs(m - mus[bi]) <= 2 * cvs_fit[bi] * mus[bi] + 3 * sig:
                cov2 += 1
            if abs(m - mus[bi]) <= 3 * sig:
                covb += 1
    h2 = cov2 >= 18
    h3 = covb < 0.70 * cells
    print("H2 envelopes: %d/%d held-out cells covered (bar >= 18) -> %s"
          % (cov2, cells, "PASS" if h2 else "FAIL"))
    print("H3 control: binomial-only covers %d/%d (must be < %d) -> %s"
          % (covb, cells, int(0.70 * cells),
             "informative" if h3 else "NOT informative"))
    if not h3:
        print("VERDICT: UNINFORMATIVE (windows not dispersed beyond "
              "sampling noise)")
        return
    print("deliverable u_b (2cv):", ["%.1f%%" % (2 * c * 100)
                                     for c in cvs_fit])
    print("VERDICT:", "ENVELOPES-CERTIFIED" if (h1 and h2) else "REFUTED")


if __name__ == "__main__":
    score(sys.argv[2], sys.argv[3])
