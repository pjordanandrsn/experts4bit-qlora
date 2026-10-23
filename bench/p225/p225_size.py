#!/usr/bin/env python3
"""p225_size.py -- lane P225's held-out sizing RULE (bench/p225/PREREG-p225.md, "The held-out set"), from VARIANCE only.

Each run trains its own adapter, and the per-row SD of loss(PA) - loss(REF) on the assistant positions varies between
adapters. The verdict reads the ESTIMATED SE against SE_BAR, so the rule prices both sources of a miss:
- the inputs are the per-adapter SDs measured before the freeze (each from a named receipt; never an effect), and the
  largest per-row kurtosis among those same runs (a shape statistic; never an effect);
- log SD is treated as normal across adapters: mean m and sample SD v of the logs (n points); a NEW adapter's log SD
  is predictive t(n - 1) about m with scale v x sqrt(1 + 1/n);
- the run's own SD estimate adds relative noise sqrt((kurtosis - 1) / (4 N)) on the log scale (the delta-method SE of
  a sample SD; a bootstrap over rows on every input run agrees to 0.002);
- P(INCONCLUSIVE | N) = P(log SD_new + noise > log(SE_BAR x sqrt(N))), the two scales combined in quadrature;
- K is the SMALLEST whole number of topics (8 templates a topic, N = 8K) with P(INCONCLUSIVE) <= 0.10.
One formula, no extra factor. SD_90 (the 90 % predictive bound) is reported beside it as a check.

    python p225_size.py --sd 0.0481 0.0663 0.0470 ... --kurtosis 4.6
    python p225_size.py --self-test
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys

SE_BAR = 0.002               # 0.2 x FLOOR (0.01): P224's SE_FRACTION x FLOOR, unchanged
PRED = 0.90                  # the predictive probability the rule sizes for
TEMPLATES = 8                # 4 trigger + 4 control templates a topic (P224's canary)

#: one-sided t quantiles at 0.90 for df 1..10 (standard tables); the rule refuses df outside them
T90 = {1: 3.078, 2: 1.886, 3: 1.638, 4: 1.533, 5: 1.476, 6: 1.440, 7: 1.415, 8: 1.397, 9: 1.383, 10: 1.372}


def _t_cdf(x: float, df: int) -> float:
    """Student's t CDF by its regularized incomplete beta (a continued fraction; df >= 1)."""
    if x == 0:
        return 0.5
    z = df / (df + x * x)
    a, b = df / 2.0, 0.5
    ib = _betainc(a, b, z)
    return 1 - 0.5 * ib if x > 0 else 0.5 * ib


def _betainc(a: float, b: float, x: float) -> float:
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0
    lbeta = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log(1 - x)
    if x < (a + 1) / (a + b + 2):
        return math.exp(lbeta) * _cf(a, b, x) / a
    return 1 - math.exp(lbeta) * _cf(b, a, 1 - x) / b


def _cf(a: float, b: float, x: float) -> float:
    tiny, c, d = 1e-300, 1.0, 1 - (a + b) * x / (a + 1)
    d = 1 / (d if abs(d) > tiny else tiny)
    h = d
    for m in range(1, 300):
        for num in (m * (b - m) * x / ((a + 2 * m - 1) * (a + 2 * m)),
                    -(a + m) * (a + b + m) * x / ((a + 2 * m) * (a + 2 * m + 1))):
            d = 1 + num * d
            d = 1 / (d if abs(d) > tiny else tiny)
            c = 1 + num / c
            c = c if abs(c) > tiny else tiny
            h *= d * c
        if abs(d * c - 1) < 1e-14:
            break
    return h


def p_inconclusive(rows: int, m: float, v: float, n: int, kurtosis: float) -> float:
    scale = math.sqrt(v * v * (1 + 1 / n) + (kurtosis - 1) / (4 * rows))
    return 1 - _t_cdf((math.log(SE_BAR * math.sqrt(rows)) - m) / scale, n - 1)


def size(sds: list, kurtosis: float) -> dict:
    n = len(sds)
    if n < 2 or n - 1 not in T90 or any(not (s > 0) for s in sds):
        raise ValueError(f"the rule needs 2..11 positive SDs, got {sds}")
    if not (kurtosis >= 1):
        raise ValueError(f"kurtosis must be >= 1, got {kurtosis}")
    logs = [math.log(s) for s in sds]
    m, v = statistics.fmean(logs), statistics.stdev(logs)
    sd90 = math.exp(m + T90[n - 1] * v * math.sqrt(1 + 1 / n))
    k = 1
    while p_inconclusive(TEMPLATES * k, m, v, n, kurtosis) > 1 - PRED:
        k += 1
    rows = TEMPLATES * k
    return {"inputs": list(sds), "kurtosis": kurtosis, "n": n, "log_mean": m, "log_sd": v, "t90": T90[n - 1],
            "sd90": sd90, "rows_at_sd90": math.ceil((sd90 / SE_BAR) ** 2), "topics": k, "rows": rows,
            "p_inconclusive": p_inconclusive(rows, m, v, n, kurtosis),
            "p_inconclusive_one_topic_less": p_inconclusive(rows - TEMPLATES, m, v, n, kurtosis)}


def self_test() -> int:
    n = 0

    def ok(c, msg):
        nonlocal n
        assert c, msg
        n += 1
    r = size([0.0481, 0.0663, 0.0470], 4.6)
    ok(abs(r["sd90"] - 0.0807) < 0.002, f"three points give SD_90 about 0.081: {r['sd90']:.4f}")
    ok(r["p_inconclusive"] <= 0.10 < r["p_inconclusive_one_topic_less"], "K is the smallest with the prior <= 0.10")
    ok(abs(_t_cdf(1.886, 2) - 0.90) < 0.001 and abs(_t_cdf(1.533, 4) - 0.90) < 0.001, "the t CDF matches the table")
    ok(abs(_t_cdf(0.0, 3) - 0.5) < 1e-12 and abs(_t_cdf(-1.638, 3) - 0.10) < 0.001, "the t CDF is symmetric")
    five = size([0.0481, 0.0663, 0.0470, 0.0566, 0.0621], 4.6)
    ok(five["t90"] == 1.533 and (five["topics"], five["rows"]) == (162, 1296), f"the five points: {five['topics']}")
    ok(abs(five["sd90"] - 0.0717) < 0.0005 and five["rows_at_sd90"] == 1285, "SD_90 reproduces review's 0.0717")
    ok(size([0.0481, 0.0663, 0.0470, 0.0566, 0.0621], 1.0)["rows"] <= five["rows"], "estimate noise only adds rows")
    ok(size([0.0481, 0.0663, 0.0470, 0.0566, 0.0621], 9.0)["rows"] >= five["rows"], "heavier tails, more rows")
    for bad in ([0.05], [0.05, -0.01], [0.05] * 12):
        try:
            size(bad, 4.6)
            ok(False, f"{bad} must refuse")
        except ValueError:
            ok(True, "")
    print(f"p225_size self-test OK ({n} checks)")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--sd", type=float, nargs="+")
    ap.add_argument("--kurtosis", type=float, help="the largest per-row kurtosis among the input runs")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args(argv)
    if a.self_test:
        return self_test()
    if not (a.sd and a.kurtosis):
        ap.error("--sd SD [SD ...] --kurtosis K, or --self-test")
    print(json.dumps(size(a.sd, a.kurtosis), indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
