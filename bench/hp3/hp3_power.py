#!/usr/bin/env python3
"""bench/hp3/hp3_power.py -- HP3 Amendment 1's power table (ESTIMATES, never an HP3 number).

P(PARITY | true Delta = 0) for one pair, simulated through HP3's own verdict rule: hp3_reduce.paired_reading, which tests BEATS / BEHIND
before PARITY, with m = min(max(sigma_pooled, 0.005), 0.02) and sigma_pooled = hp1_reduce.pooled_sd over both frameworks' seed values.
Each framework's seed value is sigma * N(0, 1); the two are correlated rho over seeds.
    python3 bench/hp3/hp3_power.py [--reps 4000] [--seed 20261021]
"""
import argparse
import math
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import hp3_reduce as R  # noqa: E402

H1 = R.H1
CELLS = ((0.001, 0.0), (0.002, 0.0), (0.002, 0.5), (0.003, 0.0), (0.003, 0.5), (0.005, 0.0), (0.005, 0.82))
SEED_COUNTS = (3, 5, 8)


def p_verdicts(sigma, rho, n, reps, rng):
    """{verdict: share} over reps draws of n seeds per framework at true Delta = 0."""
    seeds = list(range(n))
    counts = {"BEATS": 0, "BEHIND": 0, "PARITY": 0, "INCONCLUSIVE": 0}
    for _ in range(reps):
        a, b = {}, {}
        for s in seeds:
            z, w = rng.gauss(0, 1), rng.gauss(0, 1)
            a[s], b[s] = sigma * z, sigma * (rho * z + math.sqrt(1 - rho * rho) * w)
        sp, _df, _ = H1.pooled_sd([("a", list(a.values())), ("b", list(b.values()))])
        m = min(max(sp, H1.MARGIN_FLOOR), H1.MARGIN_CAP)
        counts[R.paired_reading("a vs b", a, [], b, [], sp, m, seeds=seeds)["verdict"]] += 1
    return {k: v / reps for k, v in counts.items()}


def table(reps=4000, seed=20261021):
    rng = random.Random(seed)
    return {(s, r): {n: p_verdicts(s, r, n, reps, rng) for n in SEED_COUNTS} for s, r in CELLS}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--reps", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=20261021)
    a = ap.parse_args(argv)
    t = table(a.reps, a.seed)
    print(f"ESTIMATE, P(PARITY | true Delta = 0), registered verdict order; {a.reps} reps, seed {a.seed}")
    print("sigma  rho   " + "  ".join(f"n={n}" for n in SEED_COUNTS) + "   false-directional at n=" + "/".join(map(str, SEED_COUNTS)))
    for (s, r), row in t.items():
        par = "  ".join(f"{row[n]['PARITY']:.2f}" for n in SEED_COUNTS)
        fd = "/".join(f"{row[n]['BEATS'] + row[n]['BEHIND']:.3f}" for n in SEED_COUNTS)
        print(f"{s:.3f}  {r:.2f}  {par}   {fd}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
