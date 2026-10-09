#!/usr/bin/env python3
# Copyright (c) 2026 Cerin Amroth LLC. MIT.
"""sc5_reduce.py -- lane SC5's reducer (#1478 item 4). It labels every cell, lists every losing cell, and VOIDs what did
not run as registered. A descriptive comparison: there is no single headline verdict.

**The record (``sc5.json``, written by the box):**
- ``frameworks``: the e4b side first, then the competitors.
- ``draws``: ``{draw: {memory: [block, ...]}}``. The blocks are in run order (ABBA); each is a dict with:
  - ``framework``, ``block`` (1 or 2, its occurrence in the draw), ``ready`` (bool), ``versions_ok`` (bool);
  - ``kv_tokens`` (the capacity the framework reported), ``kv_rounding`` (its block-size tolerance, in tokens),
    ``peak_mem_mib``;
  - ``cells``: ``{C: sc5_driver summary}``.
- ``quality``: ``{"reference_sha256", "reference_sha256_expected", "windows_sha256", "rows": {name: sc5_quality.compare()
  result or {"error": ...}}}``. The rows are ``<framework>`` (prefill-shaped), ``e4b_decode`` (reported) and ``floor``
  (bf16 chunked against full).

**VOID, per block** (its cells are never labelled; the reasons are listed):
- not ready, or a version off the lock;
- any invalid request in any of its cells, or a cell whose peak in flight is not its C;
- at the matched memory setting, ``|kv_tokens - MATCHED_KV_TOKENS| > kv_rounding``;
- a missing cell.

**Quality.** A reference whose sha256 differs from the expected one VOIDs every quality row. A row carrying ``error`` (a
parse refusal, such as a full-vocabulary V+1 slot, or an incomplete pass) is that row's VOID, and its framework's speed
rows stand.

**The noise bound and labels.**
- Per framework, cell and metric in one draw, the bound is ``|A1 / A2 - 1|`` from its two blocks. A framework whose own
  bound exceeds ``NOISE`` is NOISY there.
- e4b against a competitor X, per draw, pairs the blocks by occurrence: ``r_k = metric(e4b, block k) / metric(X, block
  k)``, for k = 1, 2.
  - **LEADS** when both ``r_k`` are better than 1 by more than ``NOISE``;
  - **TRAILS** when both are worse by more than it;
  - **WITHIN NOISE** otherwise, including when either framework is NOISY there.
- "Better" is lower for TTFT and TPOT, and higher for output tok/s.
- Over the draws, a cell's overall label is LEADS or TRAILS only when every readable draw says so; otherwise WITHIN NOISE.
- Every TRAILS cell is listed in ``losing_cells``.

The reducer takes ratios and never sums floats, and it writes sorted keys, so the output is byte-identical on any
Python.

    sc5_reduce.py --record sc5.json --out verdict.json
    sc5_reduce.py --self-test
"""
from __future__ import annotations

import argparse
import copy
import json
import sys

CONCURRENCIES = ("1", "16", "64")
MEMORY = ("default", "matched")
MATCHED_KV_TOKENS = 64 * 1024
NOISE = 0.05                      # placeholder: sized from SC2e's per-cell spread before any SC5 data
METRICS = {"ttft_p50_s": "lower", "ttft_p95_s": "lower", "tpot_p50_s": "lower", "tpot_p95_s": "lower",
           "output_tok_s": "higher"}


def block_faults(b: dict, memory: str) -> list:
    who = f"{b.get('framework')} block {b.get('block')}"
    out = []
    if not b.get("ready"):
        out.append(f"{who}: never ready")
    if not b.get("versions_ok"):
        out.append(f"{who}: an installed version is off the lock")
    cells = b.get("cells") or {}
    for c in CONCURRENCIES:
        s = cells.get(c)
        if s is None:
            out.append(f"{who}: cell C={c} missing")
            continue
        if s.get("invalid", 1) != 0:
            out.append(f"{who}: C={c} has {s.get('invalid')} invalid request(s) {s.get('errors')}")
        if s.get("peak_in_flight") != int(c):
            out.append(f"{who}: C={c} peaked at {s.get('peak_in_flight')} in flight")
    if memory == "matched":
        kv, tol = b.get("kv_tokens"), b.get("kv_rounding", 0)
        if kv is None or abs(int(kv) - MATCHED_KV_TOKENS) > int(tol):
            out.append(f"{who}: matched KV capacity {kv} tokens, registered {MATCHED_KV_TOKENS} +/- {tol}")
    return out


def _better(direction: str, r: float) -> int:
    """+1 when ratio r (e4b over X) is better for e4b by more than NOISE, -1 when worse by more, else 0."""
    if r is None:
        return 0
    if direction == "lower":
        return 1 if r < 1 - NOISE else (-1 if r > 1 + NOISE else 0)
    return 1 if r > 1 + NOISE else (-1 if r < 1 - NOISE else 0)


def _ratio(a, b):
    return None if a is None or b is None or b == 0 else a / b


def reduce_obj(rec: dict) -> dict:
    fws = rec["frameworks"]
    me, others = fws[0], fws[1:]
    void, by = [], {}
    for d, mems in rec["draws"].items():
        for m in MEMORY:
            blocks = mems.get(m) or []
            for b in blocks:
                f = block_faults(b, m)
                void.extend(f"draw {d} {m}: {x}" for x in f)
                if not f:
                    by.setdefault((d, m, b["framework"]), {})[b["block"]] = b
    noisy, labels, per_draw = [], {}, {}
    for m in MEMORY:
        for c in CONCURRENCIES:
            for metric, direction in METRICS.items():
                for x in others:
                    votes = []
                    for d in rec["draws"]:
                        a, bx = by.get((d, m, me), {}), by.get((d, m, x), {})
                        if set(a) != {1, 2} or set(bx) != {1, 2}:
                            continue                                             # a VOID block: this draw unreadable
                        vals = {f: [blk[k]["cells"][c].get(metric) for k in (1, 2)] for f, blk in ((me, a), (x, bx))}
                        noisy_here = []
                        for f, (v1, v2) in vals.items():
                            r = _ratio(v1, v2)
                            if r is None or abs(r - 1) > NOISE:
                                noisy_here.append(f)
                                noisy.append(f"draw {d} {m} C={c} {metric}: {f} blocks {v1} / {v2}")
                        rs = [_ratio(vals[me][k], vals[x][k]) for k in (0, 1)]
                        if noisy_here:
                            vote = "WITHIN NOISE"
                        else:
                            s = {_better(direction, r) for r in rs}
                            vote = "LEADS" if s == {1} else ("TRAILS" if s == {-1} else "WITHIN NOISE")
                        per_draw[f"{m}|{c}|{metric}|{x}|{d}"] = {"label": vote, "ratios": [None if r is None else round(r, 4)
                                                                                           for r in rs]}
                        votes.append(vote)
                    if votes:
                        labels[f"{m}|{c}|{metric}|{x}"] = votes[0] if len(set(votes)) == 1 else "WITHIN NOISE"
    q = rec.get("quality") or {}
    qrows, qvoid = {}, []
    if q.get("reference_sha256") != q.get("reference_sha256_expected"):
        qvoid.append("the reference's sha256 differs from the registered one: every quality row is VOID")
    else:
        for name, row in (q.get("rows") or {}).items():
            if "error" in row:
                qvoid.append(f"quality {name}: {row['error']}")
            else:
                qrows[name] = row
    losing = sorted(k for k, v in labels.items() if v == "TRAILS")
    counts = {v: sum(1 for x in labels.values() if x == v) for v in ("LEADS", "TRAILS", "WITHIN NOISE")}
    return {"lane": "SC5", "frameworks": fws, "noise_bound": NOISE, "matched_kv_tokens": MATCHED_KV_TOKENS,
            "void": void, "noisy": sorted(set(noisy)), "labels": labels, "per_draw": per_draw,
            "losing_cells": losing, "counts": counts, "quality": qrows, "quality_void": qvoid,
            "verdict": "READ" if labels else "NO_READING"}


# ------------------------------------------------------------------------------------------------ self-test --
def _cell(c, ttft=0.05, tpot=0.01, tok=1000.0):
    return {"invalid": 0, "errors": [], "peak_in_flight": int(c), "ttft_p50_s": ttft, "ttft_p95_s": ttft * 1.5,
            "tpot_p50_s": tpot, "tpot_p95_s": tpot * 1.2, "output_tok_s": tok}


def _block(f, k, kv=MATCHED_KV_TOKENS, tpot=0.01, tok=1000.0, **kw):
    b = {"framework": f, "block": k, "ready": True, "versions_ok": True, "kv_tokens": kv, "kv_rounding": 16,
         "peak_mem_mib": 28000.0, "cells": {c: _cell(c, tpot=tpot, tok=tok) for c in CONCURRENCIES}}
    b.update(kw)
    return b


def _record(e4b_tpot=(0.010, 0.010), vllm_tpot=(0.012, 0.012), sgl_tpot=(0.009, 0.009)):
    def draw():
        out = {}
        for m in MEMORY:
            order = [("e4b", 1), ("vllm", 1), ("sglang", 1), ("sglang", 2), ("vllm", 2), ("e4b", 2)]
            tp = {"e4b": e4b_tpot, "vllm": vllm_tpot, "sglang": sgl_tpot}
            out[m] = [_block(f, k, tpot=tp[f][k - 1]) for f, k in order]
        return out
    q = {"reference_sha256": "a" * 64, "reference_sha256_expected": "a" * 64, "windows_sha256": "b" * 64,
         "rows": {"e4b": {"nll_delta": 0.004, "argmax_agreement": 0.95, "label": "CLOSE"},
                  "vllm": {"nll_delta": 0.012, "argmax_agreement": 0.93, "label": "COMPARABLE"},
                  "sglang": {"nll_delta": 0.011, "argmax_agreement": 0.93, "label": "COMPARABLE"},
                  "e4b_decode": {"nll_delta": 0.006, "argmax_agreement": 0.94, "label": "CLOSE"},
                  "floor": {"nll_delta": 0.001, "argmax_agreement": 0.96, "label": "CLOSE"}}}
    return {"frameworks": ["e4b", "vllm", "sglang"], "draws": {"1": draw(), "2": draw()}, "quality": q}


def self_test() -> int:
    ok = []
    r = reduce_obj(_record())
    ok.append(r["verdict"] == "READ" and not r["void"] and not r["noisy"])
    ok.append(r["labels"]["default|16|tpot_p50_s|vllm"] == "LEADS")                    # 0.010 vs 0.012: e4b leads
    ok.append(r["labels"]["default|16|tpot_p50_s|sglang"] == "TRAILS")                  # 0.010 vs 0.009: 11 % behind
    ok.append("matched|64|tpot_p50_s|sglang" in r["losing_cells"])                     # every losing cell listed
    ok.append(r["labels"]["default|1|output_tok_s|vllm"] == "WITHIN NOISE")             # equal throughput
    ok.append(set(r["quality"]) == {"e4b", "vllm", "sglang", "e4b_decode", "floor"} and not r["quality_void"])
    rc = reduce_obj(_record(e4b_tpot=(0.0100, 0.0112)))                                 # e4b's own blocks 12 % apart
    ok.append(rc["labels"]["default|16|tpot_p50_s|vllm"] == "WITHIN NOISE" and bool(rc["noisy"]))
    rc = reduce_obj(_record(vllm_tpot=(0.0104, 0.0106)))       # vllm's own blocks 1.9 % apart; pairs 0.962 and 0.943
    ok.append(rc["labels"]["default|16|tpot_p50_s|vllm"] == "WITHIN NOISE" and not rc["noisy"]
              and rc["per_draw"]["default|16|tpot_p50_s|vllm|1"]["ratios"] == [0.9615, 0.9434])
    rec = _record()
    rec["draws"]["2"]["default"][1]["cells"]["16"]["tpot_p50_s"] = 0.010                # draw 2 disagrees for vllm
    rec["draws"]["2"]["default"][4]["cells"]["16"]["tpot_p50_s"] = 0.010
    ok.append(reduce_obj(rec)["labels"]["default|16|tpot_p50_s|vllm"] == "WITHIN NOISE")
    rec = _record()
    rec["draws"]["1"]["matched"][1]["kv_tokens"] = 80000                               # a capacity mismatch: VOID
    rv = reduce_obj(rec)
    ok.append(any("matched KV capacity 80000" in v for v in rv["void"])
              and rv["labels"]["matched|16|tpot_p50_s|vllm"] == "LEADS")                 # draw 2 still reads it
    rec = _record()
    rec["draws"]["1"]["default"][0]["cells"]["64"]["invalid"] = 2                      # an invalid request
    ok.append(any("2 invalid" in v for v in reduce_obj(rec)["void"]))
    rec = _record()
    rec["draws"]["1"]["default"][2]["cells"]["16"]["peak_in_flight"] = 12
    ok.append(any("peaked at 12" in v for v in reduce_obj(rec)["void"]))
    rec = _record()
    del rec["draws"]["2"]["matched"][3]["cells"]["1"]
    ok.append(any("cell C=1 missing" in v for v in reduce_obj(rec)["void"]))
    rec = _record()
    rec["draws"]["1"]["default"][5]["versions_ok"] = False
    ok.append(any("off the lock" in v for v in reduce_obj(rec)["void"]))
    rec = _record()
    rec["quality"]["rows"]["vllm"] = {"error": "slot 600 holds 151937 entries (V+1)"}  # a full-vocabulary response
    rq = reduce_obj(rec)
    ok.append("vllm" not in rq["quality"] and any("V+1" in v for v in rq["quality_void"])
              and rq["labels"]["default|16|tpot_p50_s|vllm"] == "LEADS")                 # its speed rows stand
    rec = _record()
    rec["quality"]["reference_sha256"] = "c" * 64
    rq = reduce_obj(rec)
    ok.append(rq["quality"] == {} and "sha256 differs" in rq["quality_void"][0])
    rec = _record()
    for d in rec["draws"].values():
        for m in MEMORY:
            for b in d[m]:
                b["ready"] = False
    ok.append(reduce_obj(rec)["verdict"] == "NO_READING")
    a = json.dumps(reduce_obj(_record()), sort_keys=True)
    ok.append(a == json.dumps(reduce_obj(copy.deepcopy(_record())), sort_keys=True))  # deterministic
    print(f"sc5_reduce self-test {'OK' if all(ok) else 'FAILED'} ({sum(ok)}/{len(ok)} cases)")
    return 0 if all(ok) else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--record")
    ap.add_argument("--out")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args(argv)
    if a.self_test:
        return self_test()
    if not (a.record and a.out):
        ap.error("--record --out, or --self-test")
    v = reduce_obj(json.load(open(a.record)))
    open(a.out, "w").write(json.dumps(v, indent=1, sort_keys=True) + "\n")
    print(f"SC5_VERDICT {v['verdict']} {json.dumps(v['counts'])} losing={len(v['losing_cells'])} void={len(v['void'])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
