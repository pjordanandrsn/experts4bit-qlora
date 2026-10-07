#!/usr/bin/env python3
# Copyright (c) 2026 Cerin Amroth LLC. MIT.
"""sc2e_basis.py -- lane SC2e's registered basis (bench/sc2/SC2e-PREREG.md; #846), reproduced from SC2c's committed
receipts. Descriptive: no rule reads it, it is how a reviewer re-derives every number in the registration's two tables.

``census``: SC2c's ON servers (``sc2c-5090-1``, the server SC2e's control repeats), cut into SC2c's workloads by the
request trace's plan order (warm 4, serial 24, then 120 per rate at 1, 2, 4 and 8 req/s). Per workload: the
decode-only step at bucket 16 (``step_ms`` p50 and its device time ``gpu.dec_issue - gpu.dec_prep`` p50), the prefill
steps' share of the workload's wall time, and the client's TPOT, TTFT and queue wait p50 from the request trace.

``model``: ``serve_capacity`` on SC2c's fitted ON costs (``capacity_check.json``), per SC2e arm and rate, seeds
draw x 100 + rate, 120 requests: attainment, the model's ceiling over 1-16 req/s, the decode step at full rows, and
the TPOT p50 and TTFT p99 at 8 req/s.

  python bench/sc2/sc2e_basis.py census bench/h2h-2026-10-02/sc2c/receipts/sc2c-5090-1/sc2
  python bench/sc2/sc2e_basis.py model bench/h2h-2026-10-02/sc2c/receipts/sc2c-5090-1/capacity_check.json
  python bench/sc2/sc2e_basis.py --self-test
"""
import argparse
import gzip
import json
import os
import statistics

PLAN = (("warm", 4), ("serial", 24), ("r1", 120), ("r2", 120), ("r4", 120), ("r8", 120))
RATES = (1, 2, 4, 8, 12, 16)
DRAWS = (1, 2)
N = 120
# arm -> (max_seqs, buckets): the registration's four servers
ARMS = {"s16": (16, (1, 2, 4, 8, 16)), "s32a": (32, (1, 2, 4, 8, 16, 32)), "s64c": (64, (1, 2, 4, 8, 16)),
        "s64a": (64, (1, 2, 4, 8, 16, 32, 64))}


def _rows(path):
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as f:
        return [json.loads(x) for x in f if x.strip()]


def _find(d, stem):
    for z in ("", ".gz"):
        p = os.path.join(d, stem + z)
        if os.path.exists(p):
            return p
    raise SystemExit(f"REFUSED: no {stem}[.gz] in {d}")


def _p50(xs):
    xs = [x for x in xs if x is not None]
    return round(statistics.median(xs), 3) if xs else None


def windows(reqs, plan=PLAN):
    """Cut a request trace into its workloads; each window spans its first arrival to its last finish."""
    if sum(c for _, c in plan) != len(reqs):
        raise SystemExit(f"REFUSED: the trace has {len(reqs)} rows, the plan {sum(c for _, c in plan)}")
    out, i = {}, 0
    for name, c in plan:
        rs = reqs[i:i + c]
        i += c
        out[name] = (min(r["arrival"] for r in rs), max(r["finished_at"] for r in rs), rs)
    return out


def census_window(steps, t0, t1, rs) -> dict:
    s = [x for x in steps if t0 <= x["t"] <= t1]
    dec16 = [x for x in s if x.get("bucket") == 16 and x.get("decode_rows") and not x.get("prefill_tokens")]
    pf = [x for x in s if x.get("prefill_tokens")]

    def dev(x):
        g = x.get("gpu") or {}
        return g["dec_issue"] - g["dec_prep"] if "dec_issue" in g and "dec_prep" in g else None
    wall = t1 - t0
    return {"wall_s": round(wall, 2), "decode16_n": len(dec16), "decode16_step_ms_p50": _p50(x["step_ms"] for x in dec16),
            "decode16_device_ms_p50": _p50(dev(x) for x in dec16),
            "prefill_share_of_wall": round(sum(x["step_ms"] for x in pf) / 1e3 / wall, 3) if wall > 0 else None,
            "tpot_ms_p50": _p50(r["decode_s"] / (r["out_len"] - 1) * 1e3 for r in rs if r["out_len"] > 1),
            "ttft_ms_p50": _p50(r["ttft"] * 1e3 for r in rs),
            "queue_wait_ms_p50": _p50(r["queue_wait"] * 1e3 for r in rs)}


def census(d) -> dict:
    out = {}
    for k in DRAWS:
        tag = f"e4b_on_d{k}"
        steps = _rows(_find(d, f"steps_{tag}.jsonl"))
        w = windows(_rows(_find(d, f"trace_{tag}.jsonl")))
        out[tag] = {name: census_window(steps, *w[name]) for name in ("serial", "r4", "r8")}
    return out


def model(capacity_check) -> dict:
    from experts4bit_qlora.serve_capacity import StepCosts, Workload, ceiling, simulate
    cc = json.load(open(capacity_check))
    out = {}
    for src in ("e4b_on_d1", "e4b_on_d2"):
        c = cc[src]["costs_ms"]
        for arm, (m, b) in ARMS.items():
            costs = StepCosts(prefill_step_s=c["prefill_step"] / 1e3, first_decode_s=c["first_decode"] / 1e3,
                              decode_base_s=c["decode_base"] / 1e3, decode_per_row_s=c["decode_per_row"] / 1e3,
                              buckets=b, source=src)
            r = ceiling(costs, RATES, max_seqs=m, n=N)
            at8 = [simulate(costs, Workload(n=N, rate=8, seed=d * 100 + 8), max_seqs=m).summary() for d in DRAWS]
            out[f"{arm}@{src}"] = {"ceiling": r["ceiling"], "attainment": {str(k): v for k, v in r["attainment"].items()},
                                   "decode_step_ms_full": round(costs.decode_step_s(m) * 1e3, 1),
                                   "r8_tpot_ms_p50": [round(x["tpot_p50_s"] * 1e3, 1) for x in at8],
                                   "r8_ttft_s_p99": [round(x["ttft_p99_s"], 3) for x in at8]}
    return out


def self_test() -> int:
    cases = []
    reqs = [{"arrival": float(i), "finished_at": float(i) + 0.5, "first_token_at": float(i) + 0.05, "out_len": 11,
             "decode_s": 0.45, "ttft": 0.05, "queue_wait": 0.001} for i in range(sum(c for _, c in PLAN))]
    w = windows(reqs)
    cases.append(("windows", w["serial"][0] == 4.0 and w["r8"][1] == len(reqs) - 1 + 0.5 and len(w["r1"][2]) == 120))
    steps = [{"t": 30.0 + j * 0.01, "step_ms": 9.0, "bucket": 16, "decode_rows": 16, "gpu": {"dec_prep": 0.5, "dec_issue": 8.5}}
             for j in range(50)] + [{"t": 31.0, "step_ms": 40.0, "prefill_tokens": 512}]
    c = census_window(steps, 30.0, 40.0, reqs[28:148])
    cases.append(("census", c["decode16_n"] == 50 and c["decode16_step_ms_p50"] == 9.0 and c["decode16_device_ms_p50"] == 8.0
                  and c["prefill_share_of_wall"] == 0.004 and c["tpot_ms_p50"] == 45.0 and c["ttft_ms_p50"] == 50.0))
    try:
        windows(reqs[:-1])
        cases.append(("refuses a short trace", False))
    except SystemExit:
        cases.append(("refuses a short trace", True))
    try:
        from experts4bit_qlora.serve_capacity import StepCosts  # noqa: F401
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "cc.json")
            costs = {"prefill_step": 39.48, "first_decode": 0.81, "decode_base": 4.129, "decode_per_row": 0.3476}
            json.dump({"e4b_on_d1": {"costs_ms": costs}, "e4b_on_d2": {"costs_ms": costs}}, open(p, "w"))
            m = model(p)
        cases.append(("model: today's server", m["s16@e4b_on_d1"]["ceiling"] == 4
                      and m["s16@e4b_on_d1"]["attainment"]["8"] == [0.7, 0.3417]))
        cases.append(("model: 64 slots reach 8", m["s64a@e4b_on_d1"]["ceiling"] == 8
                      and m["s64a@e4b_on_d1"]["decode_step_ms_full"] == 26.4 and m["s64c@e4b_on_d1"]["decode_step_ms_full"] == 38.8))
    except ImportError:
        print("SC2E_BASIS self-test: experts4bit_qlora not importable, model cases skipped")
    for name, ok in cases:
        print(f"  {'ok ' if ok else 'BAD'} {name}")
    bad = [n for n, ok in cases if not ok]
    print(f"SC2E_BASIS self-test: {len(cases) - len(bad)}/{len(cases)}" + (f" FAILED {bad}" if bad else ""))
    return 1 if bad else 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("what", nargs="?", choices=("census", "model"))
    ap.add_argument("path", nargs="?")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args(argv)
    if a.self_test:
        return self_test()
    if not a.what or not a.path:
        ap.error("census DIR | model CAPACITY_CHECK.json | --self-test")
    out = census(a.path) if a.what == "census" else model(a.path)
    for k, v in out.items():
        print(k, json.dumps(v))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
