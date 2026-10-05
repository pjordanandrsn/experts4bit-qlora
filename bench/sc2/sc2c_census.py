#!/usr/bin/env python3
# Copyright (c) 2026 Cerin Amroth LLC. MIT.
"""sc2c_census.py -- lane SC2c (#846): where e4b serve_paged's per-prefill stall goes, from the server's own traces.

Two instruments, both written by the server under test (no client clock is involved):

* ``E4B_PAGED_STEP_TRACE`` (one row per engine step, ``experts4bit_qlora/engines/step_trace.py``):
  - **prefill steps**, those that replayed the first-chunk graph: ``step_ms`` and its host segments; the forward's
    device time ``gpu.pf_forward - gpu.pf_prep`` (the GPU is idle at ``pf_prep``); and when the GPU finished the
    flush, ``gpu.pf_flush - gpu.pf_prep``;
  - **decode-only steps** by bucket: ``step_ms`` and the decode's device time ``gpu.dec_issue - gpu.dec_prep``;
  - **bookkeeping per request**: the host time of the prompt flush (``pf_flush``) and the first-decode block claims
    (``dec_ready``) summed over the run, and the slot resets inside ``plan`` and ``retire``, all divided by the
    prompts the run completed;
  - **the direct stall**: per prefill step, ``step_ms`` minus the median decode-only step of the same bucket (a
    prefill step's decode half is an ordinary decode), plus the run's mean ``dec_ready`` per prompt.
* ``E4B_PAGED_TRACE`` (one row per request): SC2's fit of each request's decode time, with the time-weighted decode
  bucket as a regressor, so a prefill's stall is not confounded with the batch growing as requests arrive:

      decode_s = a x steps + c x steps x mean_bucket + b x (other prompts completed during this decode)

  ``b`` is the stall per prefill (``sc2_trace``'s two-regressor fit is reported beside it).

Descriptive throughout: SC2c's registered rule (``sc2c_reduce.py``) reads ``stall_s_per_prefill`` from here for its
mechanism prediction, and reports the rest with no bar.

  sc2c_census.py steps STEP_TRACE.jsonl
  sc2c_census.py fit TRACE.jsonl --plan warm:4,serial:24,r1:120,r2:120,r4:120,r8:120
  sc2c_census.py --self-test
"""
import argparse
import json
import statistics
import sys

BUCKETS = (1, 2, 4, 8, 16)


def bucket(n: int) -> int:
    for b in BUCKETS:
        if n <= b:
            return b
    return BUCKETS[-1]


def _med(xs):
    xs = [x for x in xs if x is not None]
    return round(statistics.median(xs), 4) if xs else None


def _gpu(row, a, b):
    g = row.get("gpu") or {}
    return (g[b] - g[a]) if a in g and b in g else None


def steps(rows: list) -> dict:
    """The step-trace decomposition (see the module docstring)."""
    pf = [r for r in rows if r.get("prefill_replays")]
    dec = [r for r in rows if r.get("decode_rows") and not r.get("prefill_tokens")]
    by_b: dict = {}
    for r in dec:
        by_b.setdefault(r.get("bucket"), []).append(r)
    dec_ms = {b: _med(r["step_ms"] for r in rs) for b, rs in by_b.items()}
    segs = sorted({k for r in pf for k in r["seg"]})
    prompts = sum(r.get("prefill_replays", 0) for r in rows)
    sum_seg = {k: sum(r["seg"].get(k, 0.0) for r in rows) for k in ("pf_flush", "dec_ready", "plan", "retire")}
    direct = []
    for r in pf:
        base = dec_ms.get(r.get("bucket")) if r.get("decode_rows") else 0.0
        if base is not None:
            direct.append(r["step_ms"] - base)
    ready_per = sum_seg["dec_ready"] / prompts if prompts else None
    return {
        "steps": len(rows), "prompts": prompts,
        "prefill_steps": {"n": len(pf), "step_ms_p50": _med(r["step_ms"] for r in pf),
                          "seg_ms_p50": {k: _med(r["seg"].get(k) for r in pf) for k in segs},
                          "forward_device_ms_p50": _med(_gpu(r, "pf_prep", "pf_forward") for r in pf),
                          "flush_done_ms_p50": _med(_gpu(r, "pf_prep", "pf_flush") for r in pf)},
        "decode_steps": {str(b): {"n": len(rs), "step_ms_p50": dec_ms[b],
                                  "device_ms_p50": _med(_gpu(r, "dec_prep", "dec_issue") for r in rs)}
                         for b, rs in sorted(by_b.items(), key=lambda kv: (kv[0] is None, kv[0]))},
        "bookkeeping_ms_per_prompt": ({k: round(v / prompts, 4) for k, v in sum_seg.items()} if prompts else None),
        "direct_stall_ms_p50": (round(_med(direct) + ready_per, 4) if direct and ready_per is not None else None),
    }


# ------------------------------------------------------------------------------------------- per-request fit --

def segments(rows, plan):
    if sum(c for _, c in plan) != len(rows):
        raise SystemExit(f"REFUSED: the trace has {len(rows)} rows, the plan {sum(c for _, c in plan)}")
    out, i = [], 0
    for name, c in plan:
        out.append((name, rows[i:i + c]))
        i += c
    return out


def _solve(X, y):
    """Least squares through the origin (normal equations, Gauss-Jordan); returns (coefficients, R^2)."""
    k = len(X[0])
    M = [[sum(x[i] * x[j] for x in X) for j in range(k)] + [sum(x[i] * v for x, v in zip(X, y))] for i in range(k)]
    for c in range(k):
        p = max(range(c, k), key=lambda r: abs(M[r][c]))
        M[c], M[p] = M[p], M[c]
        for r in range(k):
            if r != c and M[c][c]:
                f = M[r][c] / M[c][c]
                M[r] = [a - f * b for a, b in zip(M[r], M[c])]
    coef = [M[i][k] / M[i][i] if M[i][i] else 0.0 for i in range(k)]
    pred = [sum(c * x for c, x in zip(coef, xx)) for xx in X]
    m = statistics.fmean(y)
    ss = sum((v - m) ** 2 for v in y)
    return coef, (1 - sum((v - p) ** 2 for v, p in zip(y, pred)) / ss) if ss else None


def fit(rows, plan) -> dict:
    """The stall per prefill, with and without the time-weighted decode bucket (see the module docstring)."""
    X2, X3, Y = [], [], []
    for name, rs in segments(rows, plan):
        if name == "warm":
            continue
        firsts = [q["first_token_at"] for q in rs]
        for r in rs:
            t0, t1 = r["first_token_at"], r["finished_at"]
            n_pf = sum(1 for q in rs if q is not r and t0 < q["first_token_at"] <= t1)
            cuts = sorted({t0, t1} | {t for t in firsts if t0 < t < t1} | {q["finished_at"] for q in rs if t0 < q["finished_at"] < t1})
            acc = 0.0
            for a, b in zip(cuts, cuts[1:]):
                mid = (a + b) / 2
                acc += bucket(sum(1 for q in rs if q["first_token_at"] <= mid < q["finished_at"])) * (b - a)
            dur = t1 - t0
            mb = acc / dur if dur > 0 else 1.0
            n = r["out_len"] - 1
            X2.append((n, n_pf))
            X3.append((n, n * mb, n_pf))
            Y.append(dur)
    if len(Y) < 4:
        return {"n": len(Y), "why": "too few requests to fit"}
    c2, r2 = _solve(X2, Y)
    c3, r3 = _solve(X3, Y)
    return {"n": len(Y),
            "two_regressor": {"decode_ms_per_token": round(c2[0] * 1e3, 4), "stall_s_per_prefill": round(c2[1], 4),
                              "r2": round(r2, 4) if r2 is not None else None},
            "bucket_controlled": {"decode_ms_per_token": round(c3[0] * 1e3, 4),
                                  "decode_ms_per_token_per_bucket_row": round(c3[1] * 1e3, 5),
                                  "stall_s_per_prefill": round(c3[2], 4), "r2": round(r3, 4) if r3 is not None else None}}


# ----------------------------------------------------------------------------------------------- self-test --

def self_test() -> int:
    import random
    ok = []
    # steps(): two prefill steps (one with a decode half at bucket 4), decode-only steps at buckets 4 and 8
    rows = [{"step": 0, "step_ms": 100.0, "prefill_replays": 1, "prefill_tokens": 512, "seg": {"plan": 1.0, "pf_flush": 60.0},
             "gpu": {"pf_prep": 2.0, "pf_forward": 42.0, "pf_flush": 64.0}},
            {"step": 1, "step_ms": 30.0, "decode_rows": 3, "bucket": 4, "seg": {"dec_ready": 20.0},
             "gpu": {"dec_prep": 1.0, "dec_issue": 6.0}},
            {"step": 2, "step_ms": 10.0, "decode_rows": 3, "bucket": 4, "seg": {}, "gpu": {"dec_prep": 1.0, "dec_issue": 6.0}},
            {"step": 3, "step_ms": 10.0, "decode_rows": 3, "bucket": 4, "seg": {}, "gpu": {"dec_prep": 1.0, "dec_issue": 6.0}},
            {"step": 4, "step_ms": 120.0, "prefill_replays": 1, "prefill_tokens": 512, "decode_rows": 3, "bucket": 4,
             "seg": {"pf_flush": 70.0}, "gpu": {"pf_prep": 2.0, "pf_forward": 44.0, "pf_flush": 74.0}},
            {"step": 5, "step_ms": 12.0, "decode_rows": 5, "bucket": 8, "seg": {"dec_ready": 20.0}, "gpu": {"dec_prep": 0.0, "dec_issue": 8.0}}]
    s = steps(rows)
    ok.append(s["prompts"] == 2 and s["prefill_steps"]["n"] == 2 and s["prefill_steps"]["forward_device_ms_p50"] == 41.0)
    ok.append(s["decode_steps"]["4"]["step_ms_p50"] == 10.0 and s["decode_steps"]["8"]["n"] == 1)
    ok.append(s["bookkeeping_ms_per_prompt"] == {"pf_flush": 65.0, "dec_ready": 20.0, "plan": 0.5, "retire": 0.0})
    # direct: step 0 (no decode half) 100 - 0, step 4 120 - 10 -> p50 105, + 40/2 ready per prompt = 125
    ok.append(s["direct_stall_ms_p50"] == 125.0)
    # fit(): synthetic requests whose decode time is exactly a*n + c*n*bucket + b*prefills; the bucket-controlled fit
    # recovers b, and the two-regressor fit absorbs the batch growth into its stall
    rng = random.Random(3)
    a, c, b = 0.0045, 0.0003, 0.12
    reqs, t = [], 0.0
    for _ in range(200):
        t += rng.expovariate(2.0)
        reqs.append({"arrival": t, "admitted_at": t, "first_token_at": t + 0.1, "out_len": rng.randint(64, 256)})
    # iterate to a fixed point: finish times depend on the overlap counts, which depend on finish times
    for r in reqs:
        r["finished_at"] = r["first_token_at"] + a * (r["out_len"] - 1)
    for _ in range(30):
        for r in reqs:
            t0, t1 = r["first_token_at"], r["finished_at"]
            n_pf = sum(1 for q in reqs if q is not r and t0 < q["first_token_at"] <= t1)
            cuts = sorted({t0, t1} | {q["first_token_at"] for q in reqs if t0 < q["first_token_at"] < t1}
                          | {q["finished_at"] for q in reqs if t0 < q["finished_at"] < t1})
            acc = sum(bucket(sum(1 for q in reqs if q["first_token_at"] <= (x + y) / 2 < q["finished_at"])) * (y - x)
                      for x, y in zip(cuts, cuts[1:]))
            mb = acc / (t1 - t0) if t1 > t0 else 1.0
            n = r["out_len"] - 1
            r["finished_at"] = t0 + a * n + c * n * mb + b * n_pf
    f = fit(reqs, [("serial", 0), ("r2", 200)])
    bc = f["bucket_controlled"]
    ok.append(abs(bc["stall_s_per_prefill"] - b) < 0.01 and bc["r2"] > 0.99)
    ok.append(f["two_regressor"]["stall_s_per_prefill"] > bc["stall_s_per_prefill"])
    try:
        fit(reqs[:-1], [("r2", 200)])
        ok.append(False)
    except SystemExit:
        ok.append(True)
    print(f"sc2c_census self-test {'OK' if all(ok) else 'FAILED'} ({sum(ok)}/{len(ok)} cases)")
    return 0 if all(ok) else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("what", nargs="?", choices=("steps", "fit"))
    ap.add_argument("path", nargs="?")
    ap.add_argument("--plan")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args(argv)
    if a.self_test:
        return self_test()
    rows = [json.loads(line) for line in open(a.path) if line.strip()]
    if a.what == "steps":
        print("SC2C_STEPS " + json.dumps(steps(rows)))
    else:
        plan = [(x.split(":")[0], int(x.split(":")[1])) for x in a.plan.split(",")]
        print("SC2C_FIT " + json.dumps(fit(rows, plan)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
