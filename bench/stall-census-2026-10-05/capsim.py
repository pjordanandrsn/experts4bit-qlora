#!/usr/bin/env python3
# Copyright (c) 2026 Cerin Amroth LLC. MIT.
"""capsim.py -- stall census (exploratory): a discrete-event model of serve_paged's scheduler on SC2's exact request
plans, to project what a cheaper prefill buys at the request level. Descriptive; it is the basis of SC2c's predictions,
not a measurement.

The scheduler as engines/scheduler.py runs it: FIFO admission into max_seqs slots; one 512-token prefill chunk per step,
before that step's decode; every decoding slot emits one token per step. Step costs are parameters:
  P   the prefill step (admission -> first token);
  R   host time a slot's FIRST decode adds to its step (_ensure_graph_ready's block claims);
  c0 + c1 x bucket   a decode step, bucket = the padded batch (1, 2, 4, 8, 16).
Plans are sc2_driver.plan() with SC2b's seeds (draw x 100 + rate, 120 requests, max_tokens U[64, 256]). SLO: TTFT <= 1 s
and TPOT <= 100 ms (SC2's); TTFT adds a 3 ms HTTP allowance.

  python capsim.py calibrate        # SC2b's four servers, against their measured attainment
  python capsim.py project          # the ceiling as P falls, R = 1 ms
"""

import random
import statistics
import sys

BUCKETS = (1, 2, 4, 8, 16)


def bucket(n):
    for b in BUCKETS:
        if n <= b:
            return b
    return 16 * ((n + 15) // 16)


def plan(mode, rate, n, seed, n_prompts=64, lo=64, hi=256):
    rng = random.Random(seed)
    out, t = [], 0.0
    for i in range(n):
        if mode == "poisson":
            t += rng.expovariate(rate)
        out.append((round(t, 6) if mode == "poisson" else 0.0, rng.randrange(n_prompts), rng.randint(lo, hi)))
    return out


def sim(reqs, P, R, c0, c1, max_seqs=16, http=0.003):
    reqs = [{"arr": a, "max": m, "out": 0} for a, _, m in reqs]
    n, i, t = len(reqs), 0, 0.0
    queue, active, done = [], [], 0
    while done < n:
        while i < n and reqs[i]["arr"] <= t:
            queue.append(reqs[i])
            i += 1
        while queue and len(active) < max_seqs:
            r = queue.pop(0)
            r["adm"] = t
            r["ph"] = "pf"
            active.append(r)
        pfs = [r for r in active if r["ph"] == "pf"]
        decs = [r for r in active if r["ph"] == "dec"]
        if not pfs and not decs:
            t = reqs[i]["arr"]
            continue
        step = 0.0
        if pfs:
            r = pfs[0]
            step += P
            r["first"] = t + step
            r["out"] = 1
            r["ph"] = "dec" if r["out"] < r["max"] else "end"
            if r["ph"] == "end":
                r["fin"] = t + step
        if decs:
            step += R * sum(1 for r in decs if r["out"] == 1)
            step += c0 + c1 * bucket(len(decs))
            for r in decs:
                r["out"] += 1
                if r["out"] >= r["max"]:
                    r["ph"] = "end"
                    r["fin"] = t + step
        t += step
        for r in [r for r in active if r["ph"] == "end"]:
            active.remove(r)
            done += 1
    ttft = [r["first"] - r["arr"] + http for r in reqs]
    tpot = [(r["fin"] - r["first"]) / (r["max"] - 1) for r in reqs]
    good = sum(1 for a, b in zip(ttft, tpot) if a <= 1.0 and b <= 0.1)
    return {
        "att": round(good / n, 3),
        "ttft_p50": round(statistics.median(ttft), 3),
        "tpot_p50": round(statistics.median(tpot), 4),
    }


def rows(P, R, c0, c1, draw, rates=(1, 2, 4, 8)):
    out = {"serial": None}
    # serial: one at a time
    s = plan("serial", 0, 24, draw)
    ttft = [P + 0.003 for _ in s]
    out["serial"] = {"ttft_p50": round(statistics.median(ttft), 3), "tpot_p50": round(c0 + c1 + R / 200, 4)}
    for r in rates:
        out[f"r{r}"] = sim(plan("poisson", r, 120, draw * 100 + r), P, R, c0, c1)
    return out


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "project"
    if what == "calibrate":
        # (P, R, c0 ms, c1 ms, draw): P = SC2b's admission->first-token p50, R = the bucket-controlled stall minus P,
        # c0 / c1 = sc2c_census.fit's bucket-controlled decode terms (bench/sc2/sc2c_census.py on sc2b-5090-1's traces)
        cases = {
            "ON_d1 (measured att 1.00/0.90/0.12/0.06)": [0.160, 0.055, 4.56, 0.357, 1],
            "ON_d2 (measured 1.00/1.00/0.29/0.06)": [0.170, 0.055, 4.86, 0.330, 2],
            "OFF_d1 (measured 1.00/0.57/0.10/0.04)": [0.221, 0.074, 4.65, 0.256, 1],
            "OFF_d2 (measured 0.98/0.83/0.19/0.03)": [0.215, 0.061, 4.97, 0.287, 2],
        }
        for name, (P, R, c0, c1, draw) in cases.items():
            rs = rows(P, R, c0 / 1e3, c1 / 1e3, draw)
            print(
                name,
                "| "
                + " | ".join(
                    f"{k}: att {v['att']} ttft50 {v['ttft_p50']} tpot50 {v['tpot_p50']}"
                    if "att" in v
                    else f"{k}: ttft50 {v['ttft_p50']}"
                    for k, v in rs.items()
                ),
            )
    else:
        for P in (0.165, 0.11, 0.08, 0.05, 0.03):
            R = 0.055 if P == 0.165 else 0.001
            row = []
            for r in (1, 2, 3, 4, 5, 6, 8):
                a = [sim(plan("poisson", r, 120, d * 100 + r), P, R, 4.7e-3, 0.34e-3)["att"] for d in (1, 2)]
                row.append(f"{r}:{min(a):.2f}")
            print(f"P={P:.3f} R={R}  min attainment over both draws by rate: " + " ".join(row))
