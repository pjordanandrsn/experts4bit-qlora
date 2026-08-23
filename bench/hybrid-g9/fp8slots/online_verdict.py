"""Verdict calculator for PREREG-online.md — online rate estimation vs
the static offline profile, at the controller horizon. Pure arithmetic,
frozen pre-box.

  python online_verdict.py --score <windows_dir> <bracket_source_dir>

Per window: U_b(t) = touched bracket-b experts at decode step t (from
the per-step series; all tiers — routing is placement-independent).
Online estimator (frozen): trailing mean of the previous 32 steps.
Target: mean of the next 32 steps. Offline baseline: the static
fit-set mean (windows w0–w4), evaluated on the same targets in the
held-out windows w5–w9. Evaluation points stride 8.

H1: online p90 relative error < offline p90 at ALL 4 brackets.
H2: max online p90 at the tail brackets (b3, b4) <= 28% (half the
    certified offline u_b4).
H3: max offline p90 at the tail brackets >= 30%, else UNINFORMATIVE.
Cross-window boundary errors are reported, never scored.
"""
import gzip
import json
import pathlib
import sys

LADDER = ["7", "9", "10", "10.75", "11.5"]
N_TRAIL, N_AHEAD, STRIDE, N_FIT = 32, 32, 8, 5


def brackets_from(src):
    sets = []
    for v in LADDER:
        d = json.load(open(pathlib.Path(src) / f"p1_v{v}.amort.json"))
        sets.append(set(map(tuple, d["manifest_vram"])))
    return [sets[i + 1] - sets[i] for i in range(len(sets) - 1)]


def load_series(wdir):
    idx = json.load(open(pathlib.Path(wdir) / "windows.json"))
    out = []
    for tag in idx["windows"]:
        am = json.load(open(pathlib.Path(wdir) / f"{tag}.amort.json"))
        assert sum(pl["uniq_nvme"] for pl in am["per_layer"]) == 0
        with gzip.open(pathlib.Path(wdir) / f"{tag}.series.json.gz",
                       "rt") as f:
            out.append(json.load(f)["per_layer_series"])
    return out


def u_series(per_layer_series, bracket):
    by_layer = {}
    for (l, e) in bracket:
        by_layer.setdefault(l, set()).add(e)
    T = len(per_layer_series[0])
    out = []
    for t in range(T):
        c = 0
        for l, es in by_layer.items():
            c += sum(1 for e in per_layer_series[l][t] if e in es)
        out.append(c)
    return out


def p90(xs):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(0.9 * len(xs)))]


def score(wdir, src):
    br = brackets_from(src)
    ser = load_series(wdir)
    U = [[u_series(w, b) for b in br] for w in ser]      # [win][bracket][t]
    static = [sum(sum(U[w][bi]) / len(U[w][bi]) for w in range(N_FIT))
              / N_FIT for bi in range(len(br))]
    on_p90, off_p90 = [], []
    for bi in range(len(br)):
        eo, ef = [], []
        for w in range(N_FIT, len(U)):
            s = U[w][bi]
            for t in range(N_TRAIL, len(s) - N_AHEAD + 1, STRIDE):
                tgt = sum(s[t:t + N_AHEAD]) / N_AHEAD
                if tgt <= 0:
                    continue
                pred = sum(s[t - N_TRAIL:t]) / N_TRAIL
                eo.append(abs(pred - tgt) / tgt)
                ef.append(abs(static[bi] - tgt) / tgt)
        on_p90.append(p90(eo))
        off_p90.append(p90(ef))
        print("b%d  static %8.2f | online p90 %5.1f%% | offline p90 %5.1f%% "
              "| n=%d" % (bi + 1, static[bi], on_p90[bi] * 100,
                          off_p90[bi] * 100, len(eo)))
    # boundary report (unscored): trailing-32 at end of w vs first-32 of w+1
    for bi in (2, 3):
        errs = []
        for w in range(len(U) - 1):
            s0, s1 = U[w][bi], U[w + 1][bi]
            tgt = sum(s1[:N_AHEAD]) / N_AHEAD
            if tgt > 0:
                errs.append(abs(sum(s0[-N_TRAIL:]) / N_TRAIL - tgt) / tgt)
        print("boundary b%d (unscored): median %5.1f%%, worst %5.1f%%"
              % (bi + 1, sorted(errs)[len(errs) // 2] * 100,
                 max(errs) * 100))
    h1 = all(o < f for o, f in zip(on_p90, off_p90))
    h2 = max(on_p90[2], on_p90[3]) <= 0.28
    h3 = max(off_p90[2], off_p90[3]) >= 0.30
    print("H1 online<offline all brackets:", "PASS" if h1 else "FAIL")
    print("H2 online tail p90 %.1f%% <= 28%%:" % (max(on_p90[2:]) * 100),
          "PASS" if h2 else "FAIL")
    print("H3 offline tail p90 %.1f%% >= 30%%:" % (max(off_p90[2:]) * 100),
          "yes" if h3 else "NO")
    if not h3:
        print("VERDICT: UNINFORMATIVE (static profile already adequate on "
              "these windows -- no non-stationarity to beat)")
        return
    print("VERDICT:", "ONLINE-CERTIFIED" if (h1 and h2) else "REFUTED")


if __name__ == "__main__":
    score(sys.argv[2], sys.argv[3])
