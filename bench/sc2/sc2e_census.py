#!/usr/bin/env python3
# Copyright (c) 2026 Cerin Amroth LLC. MIT.
"""sc2e_census.py -- lane SC2e (#846): where a wider e4b serve_paged server spends its steps, from its own traces.
Descriptive: SC2e's rule (``sc2e_reduce.py``) reads P1 and P1b's step times from here and reports the rest with no bar.

Built on ``sc2c_census`` with the arm's own bucket list. One thing is new: a decode step wider than the largest
bucket runs as consecutive replays (``chunk_rows``), and the step trace's ``bucket`` names only the LAST piece. Such a
step is keyed ``<largest>x<pieces>`` (``16x4``: a 49-64-row step on the default list), never by its ``bucket``; a
one-piece step is keyed by its bucket. ``dec_pieces`` (e4b's step trace) counts the pieces.

Per server (``steps``):
- decode-only steps by key: n, step p50, device time p50 (``gpu.dec_issue - gpu.dec_prep``), host segments p50;
- prefill steps: n, step p50, the forward's device time p50 (``sc2c_census``), the decode rows riding them, and the
  direct stall (step minus the decode-only p50 of the same key);
- ``dec_pieces``: a histogram over decode steps;
- inter-step gaps under 20 ms (the engine loop while busy) as a share of busy time.

Per workload (``per_rate``, from the request trace): mean requests in the server (admitted to finished) and the queue
wait p50. ``fit`` is ``sc2c_census.fit`` with the time-weighted padded rows of the arm's buckets as the regressor.

  sc2e_census.py steps STEP_TRACE.jsonl --arm s64a [--require-wide]
  sc2e_census.py --self-test
"""
import argparse
import importlib.util
import json
import os
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ARM_BUCKETS = {"s16": (1, 2, 4, 8, 16), "s32a": (1, 2, 4, 8, 16, 32), "s64c": (1, 2, 4, 8, 16),
               "s64a": (1, 2, 4, 8, 16, 32, 64)}
ARM_SLOTS = {"s16": 16, "s32a": 32, "s64c": 64, "s64a": 64}
BUSY_GAP_S = 0.020


def _sib(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, f"{name}.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _med(xs):
    xs = [x for x in xs if x is not None]
    return round(statistics.median(xs), 4) if xs else None


def _bucket_for(n, buckets):
    for b in buckets:
        if n <= b:
            return b
    return buckets[-1]


def padded_rows(n: int, buckets) -> int:
    """Rows a decode step of ``n`` replays: consecutive pieces of the largest bucket, each padded to its bucket."""
    top, out = buckets[-1], 0
    while n > 0:
        c = min(n, top)
        out += _bucket_for(c, buckets)
        n -= c
    return out


def key(row, buckets) -> str:
    p = row.get("dec_pieces") or 1
    return f"{buckets[-1]}x{p}" if p > 1 else str(row.get("bucket"))


def _dev(row):
    g = row.get("gpu") or {}
    return (g["dec_issue"] - g["dec_prep"]) if "dec_issue" in g and "dec_prep" in g else None


def steps(rows: list, arm: str) -> dict:
    buckets = ARM_BUCKETS[arm]
    base = _sib("sc2c_census").steps(rows)
    dec = [r for r in rows if r.get("decode_rows") and not r.get("prefill_tokens")]
    by: dict = {}
    for r in dec:
        by.setdefault(key(r, buckets), []).append(r)

    def order(k):
        a, _, b = k.partition("x")
        return (int(a) if a.isdigit() else 1 << 30, int(b or 1))
    dec_out = {}
    for k in sorted(by, key=order):
        rs = by[k]
        segs = sorted({s for r in rs for s in r["seg"]})
        dec_out[k] = {"n": len(rs), "step_ms_p50": _med(r["step_ms"] for r in rs), "device_ms_p50": _med(_dev(r) for r in rs),
                      "rows_p50": _med(r["decode_rows"] for r in rs),
                      "host_seg_ms_p50": {s: _med(r["seg"].get(s) for r in rs) for s in segs}}
    pf = [r for r in rows if r.get("prefill_replays") or r.get("prefill_tokens")]
    direct = []
    for r in pf:
        b = dec_out.get(key(r, buckets), {}).get("step_ms_p50") if r.get("decode_rows") else 0.0
        if b is not None:
            direct.append(r["step_ms"] - b)
    pieces: dict = {}
    for r in (r for r in rows if r.get("decode_rows")):
        p = str(r.get("dec_pieces") or 1)
        pieces[p] = pieces.get(p, 0) + 1
    ts = [(r["t"], r["step_ms"]) for r in rows if "t" in r]
    # the engine loop's own time between consecutive steps while busy; a longer gap is the server idle between requests
    gaps = [g for g in (b[0] - a[0] - a[1] / 1e3 for a, b in zip(ts, ts[1:])) if g < BUSY_GAP_S]
    busy = sum(ms for _, ms in ts) / 1e3 + sum(gaps)
    return {"arm": arm, "buckets": list(buckets), "steps": len(rows), "prompts": base["prompts"],
            "decode_steps": dec_out, "dec_pieces": pieces,
            "prefill_steps": dict(base["prefill_steps"], riding_decode_rows_p50=_med(r.get("decode_rows", 0) for r in pf),
                                  direct_stall_ms_p50=_med(direct)),
            "busy_gap_share": round(sum(gaps) / busy, 4) if busy > 0 else None}


def wide_ok(s: dict) -> list:
    """Why a proof's trace lacks the arm's widest step (empty: present): the top bucket on s16, s32a and s64a, a
    four-piece step on s64c; plus the prefill forward's device time and >= 26 prompts, as SC2c's proof."""
    arm, why = s["arm"], []
    want = {"s16": "16", "s32a": "32", "s64c": "16x4", "s64a": "64"}[arm]
    if not s["decode_steps"].get(want, {}).get("n"):
        why.append(f"{arm}: no decode step keyed {want} (have {sorted(s['decode_steps'])})")
    if s["prompts"] < 26 or not s["prefill_steps"].get("forward_device_ms_p50"):
        why.append(f"{arm}: prompts {s['prompts']}, forward device time {s['prefill_steps'].get('forward_device_ms_p50')}")
    return why


def per_rate(trace: list, plan: list) -> dict:
    """Per workload of the request trace (cut by ``plan``): mean requests in the server and the queue wait p50."""
    seg = _sib("sc2c_census").segments(trace, plan)
    out = {}
    for name, rs in seg:
        t0 = min(r["arrival"] for r in rs)
        t1 = max(r["finished_at"] for r in rs)
        busy = sum(r["finished_at"] - r["admitted_at"] for r in rs)
        out[name] = {"mean_in_server": round(busy / (t1 - t0), 3) if t1 > t0 else None,
                     "queue_wait_ms_p50": _med(r["queue_wait"] * 1e3 for r in rs)}
    return out


def fit(trace: list, plan: list, arm: str) -> dict:
    """``sc2c_census.fit`` with the regressor at the arm's padded rows (a chained step counts every piece)."""
    cen = _sib("sc2c_census")
    buckets = ARM_BUCKETS[arm]
    cen.bucket = lambda n: padded_rows(n, buckets)
    return cen.fit(trace, plan)


def self_test() -> int:
    cases = []
    cases.append(("padded rows", [padded_rows(n, (1, 2, 4, 8, 16)) for n in (1, 3, 16, 17, 40, 64)] == [1, 4, 16, 17, 40, 64]
                  and padded_rows(33, (1, 2, 4, 8, 16, 32, 64)) == 64 and padded_rows(50, (1, 2, 4, 8, 16)) == 50))
    cases.append(("keys", key({"bucket": 2, "dec_pieces": 4}, (1, 2, 4, 8, 16)) == "16x4"
                  and key({"bucket": 64}, ARM_BUCKETS["s64a"]) == "64" and key({"bucket": 8, "dec_pieces": 1}, (1, 8)) == "8"))
    rows = []
    t = 0.0
    for i in range(40):                       # chained 64-row steps on the default list, then prefills riding them
        rows.append({"step": len(rows), "t": t, "step_ms": 38.0, "decode_rows": 60, "bucket": 16, "dec_pieces": 4,
                     "seg": {"dec_sync": 30.0}, "gpu": {"dec_prep": 1.0, "dec_issue": 37.0}})
        t += 0.040
    for i in range(30):
        rows.append({"step": len(rows), "t": t, "step_ms": 80.0, "prefill_tokens": 512, "prefill_replays": 1, "decode_rows": 60,
                     "bucket": 16, "dec_pieces": 4, "seg": {"pf_forward": 40.0}, "gpu": {"pf_prep": 1.0, "pf_forward": 40.0}})
        t += 0.082
    s = steps(rows, "s64c")
    cases.append(("chained steps keyed by pieces", s["decode_steps"].get("16x4", {}).get("n") == 40
                  and "16" not in s["decode_steps"] and s["dec_pieces"] == {"4": 70}
                  and s["decode_steps"]["16x4"]["device_ms_p50"] == 36.0))
    cases.append(("direct stall vs the same key", s["prefill_steps"]["direct_stall_ms_p50"] == 42.0
                  and s["prefill_steps"]["riding_decode_rows_p50"] == 60))
    cases.append(("gaps", s["busy_gap_share"] is not None and 0.0 < s["busy_gap_share"] < 0.05))
    cases.append(("wide present on s64c", wide_ok(s) == []))
    s2 = steps([dict(r, dec_pieces=1, bucket=32) for r in rows], "s64a")
    cases.append(("wide missing on s64a", any("keyed 64" in w for w in wide_ok(s2))))
    tr = [{"arrival": float(i), "admitted_at": float(i) + 0.1, "finished_at": float(i) + 2.1, "queue_wait": 0.1,
           "first_token_at": float(i) + 0.2, "out_len": 50} for i in range(8)]
    pr = per_rate(tr, [("warm", 4), ("r1", 4)])
    cases.append(("per rate", pr["r1"]["mean_in_server"] == round(8.0 / 5.1, 3) and pr["r1"]["queue_wait_ms_p50"] == 100.0))
    for name, ok in cases:
        print(f"  {'ok ' if ok else 'BAD'} {name}")
    bad = [n for n, ok in cases if not ok]
    print(f"SC2E_CENSUS self-test: {len(cases) - len(bad)}/{len(cases)}" + (f" FAILED {bad}" if bad else ""))
    return 1 if bad else 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("what", nargs="?", choices=("steps",))
    ap.add_argument("path", nargs="?")
    ap.add_argument("--arm", choices=sorted(ARM_BUCKETS))
    ap.add_argument("--require-wide", action="store_true")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args(argv)
    if a.self_test:
        return self_test()
    if not (a.what and a.path and a.arm):
        ap.error("steps STEP_TRACE.jsonl --arm ARM [--require-wide] | --self-test")
    rows = [json.loads(x) for x in open(a.path) if x.strip()]
    s = steps(rows, a.arm)
    why = wide_ok(s) if a.require_wide else []
    print("SC2E_STEPS " + json.dumps(dict(s, ok=not why, why=why)), flush=True)
    return 1 if why else 0


if __name__ == "__main__":
    sys.exit(main())
