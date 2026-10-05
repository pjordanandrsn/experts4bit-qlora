#!/usr/bin/env python3
# Copyright (c) 2026 Cerin Amroth LLC. MIT.
"""sc2g_reduce.py -- lane SC2g's registered rule (bench/sc2/SC2g-PREREG.md; #846): request-level serving of gpt-oss-20b,
from box G's driver files and e4b's request trace.

Files in DIR: ``<engine>_serial_d<D>.json`` and ``<engine>_r<R>_d<D>.json`` for ENGINES (sc2_driver.py ``run``), and
``trace_e4b_gptoss.jsonl`` (serve_paged's E4B_PAGED_TRACE, in box G's plan order: 4 warm, then per draw 24 serial and
120 per rate).

**Rows and ceilings** are SC2's rule unchanged (sc2_reduce.row / ceiling): UNREAD / INVALID / UNSTABLE / VALID; the
capacity ceiling is the largest rate VALID with attainment >= 0.95 in both draws.

**Every row carries its engine's ARITHMETIC** (ARITH below): the engines serve the same checkpoint's MXFP4 experts on
different activation precisions, and e4b's prefill is NF4. A ratio between two rows is labelled ARITH_MISMATCH when the
two arithmetics differ; it is reported, never voided, for it.

**Predictions** (each HOLDS / REFUTED / UNREAD; serial ones from VALID serial rows, means over the two draws):
  Q1  serial: e4b_gptoss's p50 TTFT >= 2 x vLLM's.
  Q2  serial: e4b_gptoss's p50 TPOT <= 1.5 x vLLM's.
  Q3  capacity: vLLM's ceiling >= 4 req/s and > e4b_gptoss's.
  Q4  the mechanism SC2 read post hoc, now registered: e4b's trace fits decode_s = a x (out_len - 1) + b x (prefills
      landing during the decode) with R^2 >= 0.9 and b >= 10 x a (each interleaved prefill stalls a running decode for
      at least ten of its own tokens).
  Q5  every row of every engine is VALID.
"""
import argparse
import importlib.util
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ENGINES = ("e4b_gptoss", "vllm", "sglang", "llamacpp")
RATES = (1, 2, 4, 8)
DRAWS = (1, 2)
# each label cites its source: e4b's from grouped-nf4-gemm v0.39.0 kernel/mxfp4_grouped.py (gemv_mxfp4_b32 :257 quantises the
# activations to int8 with per-32 fp32 scales, an exact int32 e2m1 dot; gemm_mxfp4_grouped_smallm / K21 :370 takes bf16
# activations, fp32 accumulation), the rest from finding_gptoss_mxfp4_serving_arithmetic_differs_per_engine (read at the pins)
ARITH = {"e4b_gptoss": "MXFP4 decode: T==1 GEMV on int8 activations (W4A8), K21 <= 256 rows on bf16 (W4A16); NF4 prefill (KEEP_NF4); sinks explicit-mask prefill",
         "vllm": "Marlin W4A16; TRITON_ATTN",
         "sglang": "SGLang default MXFP4 runner on sm_120 (recorded); triton attention",
         "llamacpp": "published GGUF (attention Q8_0); MMVQ W4A8 decode, MMQ W4A4 prefill"}
Q1_MIN, Q2_MAX, Q3_MIN, Q4_R2, Q4_RATIO = 2.0, 1.5, 4, 0.9, 10.0


def _sib(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, f"{name}.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _load(d, name):
    p = os.path.join(d, name)
    return json.load(open(p)) if os.path.exists(p) else None


def reduce(d: str) -> dict:
    red, tr = _sib("sc2_reduce"), _sib("sc2_trace")
    eng = {}
    for e in ENGINES:
        rows = {"serial": red.row([_load(d, f"{e}_serial_d{k}.json") for k in DRAWS], "serial")}
        for r in RATES:
            rows[r] = red.row([_load(d, f"{e}_r{r}_d{k}.json") for k in DRAWS], r)
        eng[e] = {"arith": ARITH[e], "rows": {str(k): v for k, v in rows.items()}, "ceiling": red.ceiling(rows)}

    def m(e, k):
        x = eng[e]["rows"]["serial"]
        return x["mean"].get(k) if x["status"] == "VALID" else None
    v = {}
    a, b = m("e4b_gptoss", "ttft_p50_s"), m("vllm", "ttft_p50_s")
    v["Q1"] = {"verdict": "UNREAD"} if None in (a, b) else {"verdict": "HOLDS" if a >= Q1_MIN * b else "REFUTED",
                                                              "ratio": round(a / b, 4), "label": "ARITH_MISMATCH"}
    a, b = m("e4b_gptoss", "tpot_p50_s"), m("vllm", "tpot_p50_s")
    v["Q2"] = {"verdict": "UNREAD"} if None in (a, b) else {"verdict": "HOLDS" if a <= Q2_MAX * b else "REFUTED",
                                                              "ratio": round(a / b, 4), "label": "ARITH_MISMATCH"}
    c = {e: eng[e]["ceiling"] for e in ENGINES}
    v["Q3"] = ({"verdict": "UNREAD", "ceilings": c} if None in (c["vllm"], c["e4b_gptoss"]) else
               {"verdict": "HOLDS" if c["vllm"] >= Q3_MIN and c["vllm"] > c["e4b_gptoss"] else "REFUTED", "ceilings": c})
    tp = os.path.join(d, "trace_e4b_gptoss.jsonl")
    if not os.path.exists(tp):
        v["Q4"] = {"verdict": "UNREAD", "why": "no e4b trace"}
    else:
        rows = [json.loads(line) for line in open(tp) if line.strip()]
        try:
            fit = tr.analyse(rows)["fit"]
            a_s, b_s = fit["decode_ms_per_token"] / 1e3, fit["stall_s_per_prefill"]
            ok = fit["r2"] >= Q4_R2 and b_s >= Q4_RATIO * a_s
            v["Q4"] = {"verdict": "HOLDS" if ok else "REFUTED", "fit": fit, "stall_over_token": round(b_s / a_s, 2) if a_s else None}
        except SystemExit as exc:          # the trace is not box G's full plan (the deadline cut an arm)
            v["Q4"] = {"verdict": "UNREAD", "why": str(exc)}
    statuses = {e: {k: x["status"] for k, x in eng[e]["rows"].items()} for e in ENGINES}
    allv = all(s == "VALID" for e in statuses.values() for s in e.values())
    unread = any(s == "UNREAD" for e in statuses.values() for s in e.values())
    bad = any(s in ("INVALID", "UNSTABLE") for e in statuses.values() for s in e.values())
    v["Q5"] = {"verdict": "HOLDS" if allv else ("UNREAD" if unread and not bad else "REFUTED"), "statuses": statuses}
    rep = {}
    for comp in ("vllm", "sglang", "llamacpp"):
        r = {"label": "ARITH_MISMATCH"}
        for k in ("ttft_p50_s", "tpot_p50_s"):
            x, y = m("e4b_gptoss", k), m(comp, k)
            r[f"serial_{k}_ratio"] = round(x / y, 4) if x and y else None
        for rt in RATES:
            x, y = eng["e4b_gptoss"]["rows"][str(rt)], eng[comp]["rows"][str(rt)]
            r[f"goodput_r{rt}_ratio"] = (round(x["mean"]["attainment"] / y["mean"]["attainment"], 4)
                                         if x["status"] == y["status"] == "VALID" and y["mean"]["attainment"] else None)
        rep[f"e4b_gptoss_vs_{comp}"] = r
    return {"engines": eng, "predictions": v, "report": rep}


# ------------------------------------------------------------------ self-test --

def _run(ttft, tpot, att, rate, invalid=0):
    return {"summary": {"valid": 100 - invalid, "invalid": invalid, "errors": ["x"] if invalid else [], "ttft_p50_s": ttft,
                        "ttft_p99_s": ttft * 3, "tpot_p50_s": tpot, "tpot_p99_s": tpot * 2, "e2el_p50_s": 1.0,
                        "output_tok_s": 1000.0, "goodput_rps": att * rate, "attainment": att, "achieved_rate": rate}}


def self_test() -> int:
    import random
    import tempfile
    cases = []
    prof = {"e4b_gptoss": (0.20, 0.0055, 1), "vllm": (0.04, 0.0040, 8), "sglang": (0.04, 0.0040, 8), "llamacpp": (0.06, 0.0035, 2)}

    def fill(d, over=None, trace_b=0.3):
        for e, (tt, tp, cap) in prof.items():
            for k in DRAWS:
                o = (over or {}).get(e)
                json.dump(o(k) if o else _run(tt, tp, 0, 0), open(os.path.join(d, f"{e}_serial_d{k}.json"), "w"))
                for r in RATES:
                    json.dump(_run(0.3, tp * 2, 1.0 if r <= cap else 0.3, r), open(os.path.join(d, f"{e}_r{r}_d{k}.json"), "w"))
        rng, rows, t = random.Random(1), [], 0.0      # a trace in box G's order, built from a known stall model
        for c in [4] + [24, 120, 120, 120, 120] * 2:
            seg = []
            for _ in range(c):
                seg.append({"first_token_at": t + 0.2, "out_len": rng.randint(64, 256), "ttft": 0.2, "queue_wait": 0.0,
                            "finished_at": t + 0.2 + rng.uniform(0.5, 6.0)})
                t += rng.uniform(0.05, 0.6)
            t += 20.0
            for r in seg:
                n = sum(1 for q in seg if q is not r and r["first_token_at"] < q["first_token_at"] <= r["finished_at"])
                r["decode_s"] = 0.006 * (r["out_len"] - 1) + trace_b * n + rng.gauss(0.0, 0.01)
            rows += seg
        with open(os.path.join(d, "trace_e4b_gptoss.jsonl"), "w") as f:
            f.writelines(json.dumps(r) + "\n" for r in rows)
    with tempfile.TemporaryDirectory() as d:
        fill(d)
        o = reduce(d)
        v = o["predictions"]
        cases.append(("all hold", all(v[q]["verdict"] == "HOLDS" for q in ("Q1", "Q2", "Q3", "Q4", "Q5"))
                      and {e: o["engines"][e]["ceiling"] for e in ENGINES} == {e: p[2] for e, p in prof.items()}))
        cases.append(("labels", v["Q1"]["label"] == "ARITH_MISMATCH" and o["engines"]["vllm"]["arith"].startswith("Marlin")))
    with tempfile.TemporaryDirectory() as d:                     # a stall too small to be prefill-bound
        fill(d, trace_b=0.02)
        cases.append(("not prefill-bound", reduce(d)["predictions"]["Q4"]["verdict"] == "REFUTED"))
    with tempfile.TemporaryDirectory() as d:                     # the deadline cut the trace short
        fill(d)
        p = os.path.join(d, "trace_e4b_gptoss.jsonl")
        lines = open(p).read().splitlines()[:-50]
        open(p, "w").write("\n".join(lines) + "\n")
        cases.append(("short trace", reduce(d)["predictions"]["Q4"]["verdict"] == "UNREAD"))
    with tempfile.TemporaryDirectory() as d:                     # e4b as fast as vLLM: Q1 refuted
        fill(d, over={"e4b_gptoss": lambda k: _run(0.05, 0.0055, 0, 0)})
        cases.append(("Q1 refuted", reduce(d)["predictions"]["Q1"]["verdict"] == "REFUTED"))
    bad = [n for n, ok in cases if not ok]
    print(f"sc2g_reduce self-test {'OK' if not bad else 'FAILED ' + str(bad)} ({len(cases)} cases)")
    return 0 if not bad else 1


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--self-test", action="store_true")
    p.add_argument("--dir")
    p.add_argument("--out")
    a = p.parse_args(argv)
    if a.self_test:
        return self_test()
    out = reduce(a.dir)
    json.dump(out, open(a.out, "w"), indent=1)
    for q, x in out["predictions"].items():
        print(f"SC2G_{q} {x['verdict']} " + json.dumps({k: w for k, w in x.items() if k != 'verdict'})[:300])
    for e, x in out["engines"].items():
        print(f"SC2G_ENGINE {e} ceiling={x['ceiling']} rows=" + json.dumps({k: y['status'] for k, y in x['rows'].items()}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
