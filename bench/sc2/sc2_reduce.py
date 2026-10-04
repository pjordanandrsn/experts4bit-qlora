#!/usr/bin/env python3
"""sc2_reduce.py -- lane SC2's registered rule (bench/sc2/SC2-PREREG.md; #846), from the driver's run files.

Files in DIR (sc2_driver.py ``run`` outputs): ``<engine>_serial_d<D>.json`` and ``<engine>_r<R>_d<D>.json`` for
engines ENGINES, draws 1..DRAWS, rates RATES.

**A row** is one engine at one workload (``serial``, or one rate). Its status is the first that applies:
  UNREAD    a draw's file is missing (the deadline, a skipped phase, a server that did not start);
  INVALID   any request in either draw is not VALID (an HTTP error, the wrong token count, the wrong finish);
  UNSTABLE  the draws disagree: serial -- p50 TTFT beyond 10 % or p50 TPOT beyond 5 %; a rate -- p50 TPOT beyond
            10 %, or SLO attainment beyond max(10 % of the larger, 0.05);
  VALID     otherwise.
**An engine's capacity ceiling** is the largest rate whose row is VALID with SLO attainment >= 0.95 in BOTH draws (the
driver's ``attainment``: the share of the run's requests VALID and within the SLO); 0 if no rate qualifies; UNREAD if
any rate's row is UNREAD or INVALID.

**Predictions** (each HOLDS / REFUTED / UNREAD from VALID rows only, means over the two draws):
  Q1  serial: e4b_int4's p50 TTFT >= 2 x vLLM's.
  Q2  serial: e4b_int4's p50 TPOT / vLLM's lies in [1.10, 1.35].
  Q3  serial: llama.cpp's p50 TPOT is the lowest of the four lane engines (not the labelled e4b_nf4 row).
  Q4  capacity: vLLM's ceiling >= e4b_int4's >= llama.cpp's.
  Q5  capacity: e4b_nf4's ceiling < e4b_int4's.
  Q6  every row of every engine is VALID.

Reported (no bar): every row's draw means; e4b_int4 against each comparator (serial TTFT and TPOT ratios, goodput
ratio per rate, = the attainment ratio at a common offered rate).
"""
import argparse
import json
import os
import statistics
import sys

ENGINES = ("e4b_int4", "vllm", "sglang", "llamacpp", "e4b_nf4")
RATES = (1, 2, 4, 8)
DRAWS = (1, 2)
SERIAL_TTFT_BAND, SERIAL_TPOT_BAND, RATE_TPOT_BAND, GOOD_REL, GOOD_ABS = 0.10, 0.05, 0.10, 0.10, 0.05
CAP_SHARE = 0.95


def _load(d, name):
    p = os.path.join(d, name)
    return json.load(open(p)) if os.path.exists(p) else None


def _rel(a, b):
    return abs(a - b) / max(abs(a), abs(b)) if max(abs(a), abs(b)) > 0 else 0.0


def row(runs: list, workload) -> dict:
    if any(r is None for r in runs):
        return {"status": "UNREAD", "why": "a draw is missing"}
    sums = [r["summary"] for r in runs]
    bad = [f"draw {i + 1}: {s['invalid']} invalid ({s.get('errors')})" for i, s in enumerate(sums) if s["invalid"]]
    out = {"draws": [{k: s.get(k) for k in ("valid", "ttft_p50_s", "ttft_p99_s", "tpot_p50_s", "tpot_p99_s", "e2el_p50_s",
                                             "output_tok_s", "attainment", "goodput_rps", "achieved_rate")} for s in sums]}
    if bad:
        return dict(out, status="INVALID", why=bad)
    mean = {k: statistics.fmean([s[k] for s in sums]) for k in ("ttft_p50_s", "tpot_p50_s", "output_tok_s", "attainment", "goodput_rps")
            if all(s.get(k) is not None for s in sums)}
    out["mean"] = {k: round(v, 6) for k, v in mean.items()}
    a, b = sums
    if workload == "serial":
        why = []
        if _rel(a["ttft_p50_s"], b["ttft_p50_s"]) > SERIAL_TTFT_BAND:
            why.append(f"p50 TTFT {a['ttft_p50_s']} vs {b['ttft_p50_s']}")
        if _rel(a["tpot_p50_s"], b["tpot_p50_s"]) > SERIAL_TPOT_BAND:
            why.append(f"p50 TPOT {a['tpot_p50_s']} vs {b['tpot_p50_s']}")
    else:
        why = []
        if _rel(a["tpot_p50_s"], b["tpot_p50_s"]) > RATE_TPOT_BAND:
            why.append(f"p50 TPOT {a['tpot_p50_s']} vs {b['tpot_p50_s']}")
        tol = max(GOOD_REL * max(a["attainment"], b["attainment"]), GOOD_ABS)
        if abs(a["attainment"] - b["attainment"]) > tol:
            why.append(f"attainment {a['attainment']} vs {b['attainment']} (tolerance {tol:.3f})")
        out["meets_capacity"] = all(s["attainment"] >= CAP_SHARE for s in sums)
    return dict(out, status="UNSTABLE" if why else "VALID", why=why)


def ceiling(rows: dict):
    rs = [rows[r] for r in RATES]
    if any(x["status"] in ("UNREAD", "INVALID") for x in rs):
        return None
    ok = [r for r in RATES if rows[r]["status"] == "VALID" and rows[r].get("meets_capacity")]
    return max(ok) if ok else 0


def reduce(d: str) -> dict:
    eng = {}
    for e in ENGINES:
        rows = {"serial": row([_load(d, f"{e}_serial_d{k}.json") for k in DRAWS], "serial")}
        for r in RATES:
            rows[r] = row([_load(d, f"{e}_r{r}_d{k}.json") for k in DRAWS], r)
        eng[e] = {"rows": {str(k): v for k, v in rows.items()}, "ceiling": ceiling(rows)}

    def m(e, k):
        x = eng[e]["rows"]["serial"]
        return x["mean"].get(k) if x["status"] == "VALID" else None
    v = {}
    a, b = m("e4b_int4", "ttft_p50_s"), m("vllm", "ttft_p50_s")
    v["Q1"] = {"verdict": "UNREAD"} if None in (a, b) else {"verdict": "HOLDS" if a >= 2 * b else "REFUTED", "ratio": round(a / b, 4)}
    a, b = m("e4b_int4", "tpot_p50_s"), m("vllm", "tpot_p50_s")
    v["Q2"] = {"verdict": "UNREAD"} if None in (a, b) else {"verdict": "HOLDS" if 1.10 <= a / b <= 1.35 else "REFUTED", "ratio": round(a / b, 4)}
    tp = {e: m(e, "tpot_p50_s") for e in ENGINES[:4]}             # the four lane engines; the labelled NF4 row is not in Q3
    v["Q3"] = ({"verdict": "UNREAD", "missing": [e for e, x in tp.items() if x is None]} if None in tp.values() else
               {"verdict": "HOLDS" if min(tp, key=tp.get) == "llamacpp" else "REFUTED", "lowest": min(tp, key=tp.get), "tpot_p50_s": tp})
    c = {e: eng[e]["ceiling"] for e in ENGINES}
    v["Q4"] = ({"verdict": "UNREAD"} if None in (c["vllm"], c["e4b_int4"], c["llamacpp"]) else
               {"verdict": "HOLDS" if c["vllm"] >= c["e4b_int4"] >= c["llamacpp"] else "REFUTED", "ceilings": c})
    v["Q5"] = ({"verdict": "UNREAD"} if None in (c["e4b_nf4"], c["e4b_int4"]) else
               {"verdict": "HOLDS" if c["e4b_nf4"] < c["e4b_int4"] else "REFUTED", "ceilings": c})
    statuses = {e: {k: x["status"] for k, x in eng[e]["rows"].items()} for e in ENGINES}
    allv = all(s == "VALID" for e in statuses.values() for s in e.values())
    unread = any(s == "UNREAD" for e in statuses.values() for s in e.values())
    v["Q6"] = {"verdict": "HOLDS" if allv else ("UNREAD" if unread and not any(
        s in ("INVALID", "UNSTABLE") for e in statuses.values() for s in e.values()) else "REFUTED"), "statuses": statuses}
    rep = {}
    for comp in ("vllm", "sglang", "llamacpp"):
        r = {}
        for k in ("ttft_p50_s", "tpot_p50_s"):
            x, y = m("e4b_int4", k), m(comp, k)
            r[f"serial_{k}_ratio"] = round(x / y, 4) if x and y else None
        for rt in RATES:
            x, y = eng["e4b_int4"]["rows"][str(rt)], eng[comp]["rows"][str(rt)]
            r[f"goodput_r{rt}_ratio"] = (round(x["mean"]["attainment"] / y["mean"]["attainment"], 4)
                                         if x["status"] == y["status"] == "VALID" and y["mean"]["attainment"] else None)
        rep[f"e4b_int4_vs_{comp}"] = r
    return {"engines": eng, "predictions": v, "report": rep}


# ------------------------------------------------------------------ self-test --

def _run(ttft, tpot, good, rate, invalid=0, tok_s=1000.0):
    return {"summary": {"valid": 100 - invalid, "invalid": invalid, "errors": ["x"] if invalid else [], "ttft_p50_s": ttft,
                        "ttft_p99_s": ttft * 3, "tpot_p50_s": tpot, "tpot_p99_s": tpot * 2, "e2el_p50_s": 1.0,
                        "output_tok_s": tok_s, "goodput_rps": good, "attainment": round(good / rate, 4) if rate else None,
                        "achieved_rate": rate}}


def _write(d, e, serial, rates, draws=DRAWS, skip=()):
    for k in draws:
        if (e, "serial", k) not in skip:
            json.dump(serial(k), open(os.path.join(d, f"{e}_serial_d{k}.json"), "w"))
        for r in RATES:
            if (e, r, k) not in skip:
                json.dump(rates(r, k), open(os.path.join(d, f"{e}_r{r}_d{k}.json"), "w"))


def self_test() -> int:
    import tempfile
    cases = []
    prof = {"e4b_int4": (0.15, 0.0045, 4), "vllm": (0.04, 0.0037, 8), "sglang": (0.04, 0.0037, 8),
            "llamacpp": (0.05, 0.0031, 4), "e4b_nf4": (0.2, 0.010, 2)}

    def fill(d, over=None, skip=()):
        for e, (tt, tp, cap) in prof.items():
            o = (over or {}).get(e, {})
            _write(d, e, lambda k, tt=tt, tp=tp, o=o: o.get("serial", lambda k: _run(tt, tp, 0, 0))(k),
                   lambda r, k, cap=cap, tp=tp, o=o: o.get("rate", lambda r, k: _run(0.3, tp * 2, r if r <= cap else r * 0.4, r))(r, k),
                   skip=skip)
    with tempfile.TemporaryDirectory() as d:
        fill(d)
        out = reduce(d)
        v = out["predictions"]
        cases.append(("all hold", all(v[q]["verdict"] == "HOLDS" for q in ("Q1", "Q2", "Q3", "Q4", "Q5", "Q6"))))
        cases.append(("ceilings", {e: out["engines"][e]["ceiling"] for e in ENGINES} == {e: p[2] for e, p in prof.items()}))
        cases.append(("report", out["report"]["e4b_int4_vs_vllm"]["serial_ttft_p50_s_ratio"] == round(0.15 / 0.04, 4)))
    with tempfile.TemporaryDirectory() as d:
        fill(d, skip={("sglang", 4, 2)})
        out = reduce(d)
        cases.append(("unread row", out["engines"]["sglang"]["rows"]["4"]["status"] == "UNREAD" and out["engines"]["sglang"]["ceiling"] is None
                      and out["predictions"]["Q6"]["verdict"] == "UNREAD" and out["predictions"]["Q4"]["verdict"] == "HOLDS"))
    with tempfile.TemporaryDirectory() as d:
        fill(d, over={"llamacpp": {"rate": lambda r, k: _run(0.3, 0.006, r, r, invalid=3 if r == 8 else 0)}})
        out = reduce(d)
        cases.append(("invalid row", out["engines"]["llamacpp"]["rows"]["8"]["status"] == "INVALID"
                      and out["predictions"]["Q6"]["verdict"] == "REFUTED" and out["predictions"]["Q4"]["verdict"] == "UNREAD"))
    with tempfile.TemporaryDirectory() as d:
        fill(d, over={"vllm": {"serial": lambda k: _run(0.04 if k == 1 else 0.06, 0.0037, 0, 0)}})
        out = reduce(d)
        cases.append(("unstable serial", out["engines"]["vllm"]["rows"]["serial"]["status"] == "UNSTABLE"
                      and out["predictions"]["Q1"]["verdict"] == "UNREAD"))
    with tempfile.TemporaryDirectory() as d:
        fill(d, over={"e4b_int4": {"serial": lambda k: _run(0.05, 0.0050, 0, 0)}})
        v = reduce(d)["predictions"]
        cases.append(("refuted", v["Q1"]["verdict"] == "REFUTED" and v["Q2"]["verdict"] == "REFUTED"))
    with tempfile.TemporaryDirectory() as d:                     # goodput draws within the absolute tolerance at r=1
        fill(d, over={"vllm": {"rate": lambda r, k: _run(0.3, 0.007, r - (0.04 if k == 2 else 0.0), r)}})
        out = reduce(d)
        cases.append(("goodput tolerance", out["engines"]["vllm"]["rows"]["1"]["status"] == "VALID"
                      and out["engines"]["vllm"]["ceiling"] == 8))
    with tempfile.TemporaryDirectory() as d:                     # the deadline dropped the labelled row: Q3 still reads
        fill(d, skip={("e4b_nf4", w, k) for w in ("serial",) + RATES for k in DRAWS})
        v = reduce(d)["predictions"]
        cases.append(("labelled row dropped", v["Q3"]["verdict"] == "HOLDS" and v["Q5"]["verdict"] == "UNREAD"
                      and v["Q6"]["verdict"] == "UNREAD"))
    bad = [n for n, ok in cases if not ok]
    print(f"sc2_reduce self-test {'OK' if not bad else 'FAILED ' + str(bad)} ({len(cases)} cases)")
    return 0 if not bad else 1


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--self-test", action="store_true")
    p.add_argument("--dir")
    p.add_argument("--out")
    a = p.parse_args(argv)
    if a.self_test:
        return self_test()
    out = reduce(a.dir)
    json.dump(out, open(a.out, "w"), indent=1)
    for q, r in out["predictions"].items():
        print(f"SC2_{q} {r['verdict']} {json.dumps({k: x for k, x in r.items() if k != 'verdict'})[:300]}")
    for e in ENGINES:
        en = out["engines"][e]
        print(f"SC2_ENGINE {e} ceiling={en['ceiling']} rows={json.dumps({k: x['status'] for k, x in en['rows'].items()})}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
