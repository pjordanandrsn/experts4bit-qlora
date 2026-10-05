#!/usr/bin/env python3
# Copyright (c) 2026 Cerin Amroth LLC. MIT.
"""sc2_trace.py -- lane SC2 (#846): what e4b's own request trace says about where its serving time goes. DESCRIPTIVE,
not a registered rule: it is the read's post-hoc analysis of ``sc2-5090-1``, kept as a tool so the numbers reproduce.

Input: ``serve_paged``'s ``E4B_PAGED_TRACE`` JSONL from a box E run (one row per request, in completion order of
submission: ``arrival``, ``admitted_at``, ``first_token_at``, ``finished_at``, ``out_len``, ``ttft``, ``queue_wait``,
``decode_s``). The rows follow box E's plan order -- 4 warm requests, then per draw a serial run of SERIAL_N requests and
one Poisson run of N requests per rate -- so the trace is cut into those workloads by count.

Per workload: p50 TTFT, p50 / max queue wait (waiting for one of the max_seqs slots), and the mean number of OTHER
requests' prefills that completed while a request was decoding. Over all requests: a least-squares fit

    decode_s = a x (out_len - 1) + b x (prefills completed during this request's decode)

``a`` is the per-token decode cost and ``b`` the stall each interleaved prefill costs every request decoding beside it.

  sc2_trace.py TRACE.jsonl [--serial-n 24 --n 120 --rates 1,2,4,8 --draws 2]
  sc2_trace.py TRACE.jsonl --plan warm:4,serial:24,r1:120,r2:120,r4:120,r8:120   (one server's own order, e.g. SC2b's)
  sc2_trace.py --self-test
"""
import argparse
import json
import random
import statistics
import sys


def segments(rows, serial_n=24, n=120, rates=(1, 2, 4, 8), draws=2, warm=4, plan=None):
    plan = plan or ([("warm", warm)] + [(f"{w}_d{k}", serial_n if w == "serial" else n)
                                        for k in range(1, draws + 1) for w in ["serial"] + [f"r{r}" for r in rates]])
    if sum(c for _, c in plan) != len(rows):
        raise SystemExit(f"REFUSED: the trace has {len(rows)} rows, the plan {sum(c for _, c in plan)}")
    out, i = [], 0
    for name, c in plan:
        out.append((name, rows[i:i + c]))
        i += c
    return out


def prefills_during(r, rows):
    return sum(1 for q in rows if q is not r and r["first_token_at"] < q["first_token_at"] <= r["finished_at"])


def fit(xs, ys):
    """Least squares through the origin on two regressors, by the normal equations."""
    s11 = sum(a * a for a, _ in xs)
    s12 = sum(a * b for a, b in xs)
    s22 = sum(b * b for _, b in xs)
    t1 = sum(a * y for (a, _), y in zip(xs, ys))
    t2 = sum(b * y for (_, b), y in zip(xs, ys))
    det = s11 * s22 - s12 * s12
    a, b = (t1 * s22 - t2 * s12) / det, (s11 * t2 - s12 * t1) / det
    pred = [a * x1 + b * x2 for x1, x2 in xs]
    mean = statistics.fmean(ys)
    r2 = 1 - sum((y - p) ** 2 for y, p in zip(ys, pred)) / sum((y - mean) ** 2 for y in ys)
    return a, b, r2


def analyse(rows, **kw):
    segs = segments(rows, **kw)
    per, xs, ys = {}, [], []
    for name, rs in segs:
        pf = [prefills_during(r, rs) for r in rs]
        per[name] = {"ttft_p50_s": round(statistics.median(r["ttft"] for r in rs), 4),
                     "queue_wait_p50_s": round(statistics.median(r["queue_wait"] for r in rs), 4),
                     "queue_wait_max_s": round(max(r["queue_wait"] for r in rs), 4),
                     "prefills_during_decode_mean": round(statistics.fmean(pf), 3),
                     "decode_s_mean": round(statistics.fmean(r["decode_s"] for r in rs), 4)}
        if name != "warm":
            xs += [(r["out_len"] - 1, p) for r, p in zip(rs, pf)]
            ys += [r["decode_s"] for r in rs]
    a, b, r2 = fit(xs, ys)
    return {"workloads": per, "fit": {"decode_ms_per_token": round(a * 1e3, 3), "stall_s_per_prefill": round(b, 4),
                                      "r2": round(r2, 4), "n": len(ys)}}


def self_test() -> int:
    rng = random.Random(0)
    a_true, b_true, rows, t = 0.006, 0.35, [], 0.0
    plan = [4] + [24, 30, 30, 30, 30] * 2
    for c in plan:
        for _ in range(c):
            rows.append({"arrival": t, "admitted_at": t, "first_token_at": t + 0.3, "out_len": rng.randint(64, 256),
                         "ttft": 0.3, "queue_wait": 0.0})
            t += rng.uniform(0.05, 0.6)
        t += 20.0                        # a gap longer than any decode window: no prefill is counted across workloads
    # each request gets an independent decode window; the prefills inside it are counted exactly as analyse() counts
    # them, and decode_s comes from the true model plus noise (no feedback from decode_s to the window)
    for r in rows:
        r["finished_at"] = r["first_token_at"] + rng.uniform(0.5, 6.0)
    for r in rows:
        n_pf = prefills_during(r, rows)
        r["decode_s"] = a_true * (r["out_len"] - 1) + b_true * n_pf + rng.gauss(0.0, 0.01)
    out = analyse(rows, serial_n=24, n=30)
    f = out["fit"]
    ok = [abs(f["decode_ms_per_token"] - 6.0) < 0.2, abs(f["stall_s_per_prefill"] - 0.35) < 0.02, f["r2"] > 0.99,
          set(out["workloads"]) == {"warm"} | {f"{w}_d{k}" for k in (1, 2) for w in ("serial", "r1", "r2", "r4", "r8")}]
    try:
        segments(rows[:-1], serial_n=24, n=30)
        ok.append(False)
    except SystemExit:
        ok.append(True)
    print(f"sc2_trace self-test {'OK' if all(ok) else 'FAILED'} ({sum(ok)}/{len(ok)} cases)")
    return 0 if all(ok) else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("trace", nargs="?")
    ap.add_argument("--serial-n", type=int, default=24)
    ap.add_argument("--n", type=int, default=120)
    ap.add_argument("--rates", default="1,2,4,8")
    ap.add_argument("--draws", type=int, default=2)
    ap.add_argument("--plan", help="name:count,... in the server's order (overrides the SC2 layout)")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args(argv)
    if a.self_test:
        return self_test()
    rows = [json.loads(line) for line in open(a.trace) if line.strip()]
    plan = [(x.split(":")[0], int(x.split(":")[1])) for x in a.plan.split(",")] if a.plan else None
    out = analyse(rows, serial_n=a.serial_n, n=a.n, rates=tuple(int(x) for x in a.rates.split(",")), draws=a.draws, plan=plan)
    for name, w in out["workloads"].items():
        print("SC2_TRACE " + json.dumps({"workload": name, **w}))
    print("SC2_TRACE_FIT " + json.dumps(out["fit"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
