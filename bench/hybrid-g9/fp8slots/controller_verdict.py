"""Verdict calculator for PREREG-controller.md (C1: decision value of the
shadow slot controller). Pure, strictly-causal arithmetic over per-step
touched-expert series — no engine changes, no timing.

  python controller_verdict.py --score <fresh_windows_dir> <prior_windows_dir>

Prior (the deployed static profile) = pooled mean rates over the PRIOR
set (the committed design-set series). Static baseline = top-S by prior.
Controller (frozen): every EPOCH=8 steps re-estimate rates as
max(trailing-32 mean, 0.25 x prior) and perform only gain-gated swaps:
in/out pairs whose estimated rate gap exceeds THETA = SWAP_COST/HORIZON
= 4/32. Objective: per-step DRAM touch-mass (touched experts outside
the current VRAM set), S = 4045. The swap gate requires the estimated
gap to be both economically material (> THETA) and statistically real
(> 3 x the combined binomial sd of the two trailing estimates), so tied
margins cannot churn on sampling noise.

B1: swap-adjusted reduction (charging 2 uniques per swap) >= 10%.
B2: raw reduction >= 15%.
B3 (control): oracle-static reduction >= 8%, else UNINFORMATIVE.
"""
import collections
import gzip
import json
import pathlib
import sys

S_SLOTS = 4045
EPOCH, TRAIL, PRIOR_FLOOR = 8, 32, 0.25
SWAP_COST, HORIZON = 2.0, 32.0
THETA = 4.0 / 32.0


def load_windows(wdir):
    idx = json.load(open(pathlib.Path(wdir) / "windows.json"))
    out = []
    for tag in idx["windows"]:
        with gzip.open(pathlib.Path(wdir) / f"{tag}.series.json.gz",
                       "rt") as f:
            w = json.load(f)["per_layer_series"]
        L = len(w)
        E = 1 + max((e for l in w for st in l for e in st), default=0)
        out.append((w, L, E))
    L = out[0][1]
    E = max(x[2] for x in out)
    flat = []
    for w, _, _ in out:
        flat.append([[l * E + e for l in range(L) for e in w[l][t]]
                     for t in range(len(w[0]))])
    return flat, L * E


def pooled_prior(flat, NE):
    tot = [0] * NE
    n = 0
    for sw in flat:
        n += len(sw)
        for st in sw:
            for i in st:
                tot[i] += 1
    return [t / n for t in tot]


def run_controller(sw, prior, static_set, NE):
    cur = set(static_set)
    hist = collections.deque(maxlen=TRAIL)
    mass = 0
    swaps = 0
    for t, st in enumerate(sw):
        mass += sum(1 for i in st if i not in cur)
        hist.append(st)
        if (t + 1) % EPOCH == 0 and len(hist) >= 8:
            cnt = [0] * NE
            for h in hist:
                for i in h:
                    cnt[i] += 1
            est = [max(c / len(hist), PRIOR_FLOOR * p)
                   for c, p in zip(cnt, prior)]
            ins = sorted((i for i in range(NE) if i not in cur),
                         key=lambda i: -est[i])
            outs = sorted(cur, key=lambda i: est[i])
            n = len(hist)
            k = 0
            while k < len(ins) and k < len(outs):
                a, b = est[ins[k]], est[outs[k]]
                gap = a - b
                sd = ((a * (1 - a) + b * (1 - b)) / n) ** 0.5
                if gap <= max(THETA, 3 * sd):
                    break
                cur.discard(outs[k])
                cur.add(ins[k])
                swaps += 1
                k += 1
    return mass, swaps


def score(fresh_dir, prior_dir):
    prior_flat, NE = load_windows(prior_dir)
    fresh_flat, NE2 = load_windows(fresh_dir)
    NE = max(NE, NE2)
    prior = pooled_prior(prior_flat, NE)
    static_set = set(sorted(range(NE), key=lambda i: -prior[i])[:S_SLOTS])
    t_static = t_ctrl = t_swaps = 0
    t_orap = t_stap = 0        # split-half oracle: rank on even steps,
    for wi, sw in enumerate(fresh_flat):   # evaluate on odd -- selection
        st_m = sum(sum(1 for i in st if i not in static_set) for st in sw)
        cnt = [0] * NE                     # on sampling noise cannot
        for st in sw[0::2]:                # survive the split
            for i in st:
                cnt[i] += 1
        oset = set(sorted(range(NE), key=lambda i: -cnt[i])[:S_SLOTS])
        o_m = sum(sum(1 for i in st if i not in oset) for st in sw[1::2])
        s_m = sum(sum(1 for i in st if i not in static_set)
                  for st in sw[1::2])
        c_m, sw_n = run_controller(sw, prior, static_set, NE)
        print("w%-2d static %6d | ctrl %6d | oracle(odd) %5d vs %5d | "
              "swaps %4d" % (wi, st_m, c_m, o_m, s_m, sw_n))
        t_static += st_m
        t_ctrl += c_m
        t_orap += o_m
        t_stap += s_m
        t_swaps += sw_n
    raw = 1 - t_ctrl / t_static
    adj = 1 - (t_ctrl + SWAP_COST * t_swaps) / t_static
    ora = 1 - t_orap / t_stap
    print("totals: static %d | controller %d | swaps %d"
          % (t_static, t_ctrl, t_swaps))
    print("raw reduction %.1f%% | swap-adjusted %.1f%% | split-half "
          "oracle %.1f%% | oracle-gap captured %.0f%%"
          % (raw * 100, adj * 100, ora * 100,
             100 * adj / ora if ora > 0 else 0))
    if ora < 0.08:
        print("VERDICT: UNINFORMATIVE (split-half oracle gains only "
              "%.1f%% -- no non-stationarity for a controller to exploit)"
              % (ora * 100))
        return
    b1 = adj >= 0.10
    b2 = raw >= 0.15
    print("B1 swap-adjusted >= 10%%:", "PASS" if b1 else "FAIL")
    print("B2 raw >= 15%%:", "PASS" if b2 else "FAIL")
    print("VERDICT:", "CONTROLLER-CERTIFIED" if (b1 and b2) else "REFUTED")


if __name__ == "__main__":
    score(sys.argv[2], sys.argv[3])
