#!/usr/bin/env python3
# Copyright (c) 2026 Cerin Amroth LLC. MIT.
"""sc2f_basis.py -- lane SC2f's registered basis (bench/sc2/SC2f-PREREG.md; #846), reproduced from committed receipts.
Descriptive: no rule reads it. It is how a reviewer re-derives every predicted number in the registration.

``model``: ``serve_capacity`` on SC2e's s64a fitted costs (``capacity_check.json`` of ``sc2e-5090-1``: the server both
SC2f arms repeat), per draw, for two arms:
- ``off``: the costs as SC2e measured them (both P130 knobs off: SC2e predates them);
- ``on``: the same costs with the prefill step lowered by P130's measured saving, 5.18 ms per 512-token first-chunk
  graph replay (38.17 -> 32.99 ms, ``bench/p130/RESULTS-p130.md``). Nothing else moves: the model has no term for what
  P2 may take off a 64-row decode step.
At SC2f's seeds (serial = the draw; ladder = draw x 100 + rate; knee = 10000 x draw + 100 x rate + j), 120 requests a
rate: attainment, TTFT p50 and TPOT p50 per ladder rate, serial TTFT and TPOT p50, the model's ceiling on the ladder,
and the knee block's mean attainment (rates 12, 13, 14; the ladder seed and four more each).

``sensitivity``: the knee block under a whole-server level offset of +-3.6 % on every step (captured runners carry one,
P124 / P126), and with a smaller prefill saving. ``spread``: over seeds 0-59 on draw 1's costs, the share of 120-request
plans that reach 0.95 at 12-16 req/s, per arm: why one plan a draw cannot read the knee. ``levers``: attainment at
12-20 req/s for the levers after this one.

  python bench/sc2/sc2f_basis.py model bench/h2h-2026-10-02/sc2e/receipts/sc2e-5090-1/capacity_check.json
  python bench/sc2/sc2f_basis.py --self-test
"""
import argparse
import importlib.util
import json
import os
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
DRAWS = (1, 2)
LADDER = (4, 8, 12, 13, 14, 16)
KNEE = (12, 13, 14)
KNEE_J = (1, 2, 3, 4)
N, SERIAL_N = 120, 24
MAX_SEQS = 64
BUCKETS = (1, 2, 4, 8, 16, 32, 64)
P130_SAVING_MS = 5.18                     # 38.17 -> 32.99 ms per 512-token first-chunk replay (P130, read #1605)
OFFSET = 0.036                            # a captured runner's whole-life level offset (P124, P126)
SRC = "e4b_s64a_d{}"


def ladder_seed(draw, rate):
    return draw * 100 + rate


def knee_seeds(draw, rate):
    """The ladder's seed, then four more: every knee rate runs five plans a server."""
    return [ladder_seed(draw, rate)] + [10000 * draw + 100 * rate + j for j in KNEE_J]


def _cap():
    try:
        from experts4bit_qlora import serve_capacity
        return serve_capacity
    except ImportError:                   # a checkout without torch: the module itself imports nothing of the package
        p = os.path.join(HERE, "..", "..", "experts4bit_qlora", "serve_capacity.py")
        spec = importlib.util.spec_from_file_location("serve_capacity", p)
        m = importlib.util.module_from_spec(spec)
        sys.modules["serve_capacity"] = m
        spec.loader.exec_module(m)
        return m


def costs(c, *, saving_ms=0.0, scale=1.0, pf_scale=1.0, dec_scale=1.0, source=""):
    sc = _cap()
    return sc.StepCosts(prefill_step_s=(c["prefill_step"] - saving_ms) * pf_scale * scale / 1e3,
                        first_decode_s=c["first_decode"] / 1e3,
                        decode_base_s=c["decode_base"] * dec_scale * scale / 1e3,
                        decode_per_row_s=c["decode_per_row"] * dec_scale * scale / 1e3, buckets=BUCKETS, source=source)


def knee(sc_costs, draw) -> float:
    sc = _cap()
    a = [sc.simulate(sc_costs, sc.Workload(n=N, rate=r, seed=s), max_seqs=MAX_SEQS).attainment
         for r in KNEE for s in knee_seeds(draw, r)]
    return round(statistics.fmean(a), 4)


def arm(sc_costs, draw) -> dict:
    sc = _cap()
    rows = {}
    for r in LADDER:
        res = sc.simulate(sc_costs, sc.Workload(n=N, rate=r, seed=ladder_seed(draw, r)), max_seqs=MAX_SEQS)
        s = res.summary()
        rows[str(r)] = {"attainment": round(res.attainment, 4), "ttft_ms_p50": round(s["ttft_p50_s"] * 1e3, 1),
                        "tpot_ms_p50": round(s["tpot_p50_s"] * 1e3, 2)}
    ser = sc.simulate(sc_costs, sc.Workload(n=SERIAL_N, rate=None, seed=draw), max_seqs=MAX_SEQS).summary()
    ok = [r for r in LADDER if rows[str(r)]["attainment"] >= 0.95]
    return {"serial": {"ttft_ms_p50": round(ser["ttft_p50_s"] * 1e3, 1), "tpot_ms_p50": round(ser["tpot_p50_s"] * 1e3, 2)},
            "ladder": rows, "ceiling": max(ok) if ok else 0, "knee_mean": knee(sc_costs, draw)}


def model(capacity_check) -> dict:
    cc = json.load(open(capacity_check))
    out = {}
    for d in DRAWS:
        c = cc[SRC.format(d)]["costs_ms"]
        off, on = arm(costs(c, source="off"), d), arm(costs(c, saving_ms=P130_SAVING_MS, source="on"), d)
        ratio = {}
        for r in (8, 12):
            ratio[f"r{r}_tpot"] = round(on["ladder"][str(r)]["tpot_ms_p50"] / off["ladder"][str(r)]["tpot_ms_p50"], 3)
            ratio[f"r{r}_ttft"] = round(on["ladder"][str(r)]["ttft_ms_p50"] / off["ladder"][str(r)]["ttft_ms_p50"], 3)
        ratio["serial_ttft"] = round(on["serial"]["ttft_ms_p50"] / off["serial"]["ttft_ms_p50"], 3)
        ratio["serial_tpot"] = round(on["serial"]["tpot_ms_p50"] / off["serial"]["tpot_ms_p50"], 3)
        out[f"d{d}"] = {"costs_ms": c, "off": off, "on": on, "knee_gain": round(on["knee_mean"] - off["knee_mean"], 4),
                        "on_over_off": ratio}
    return out


def sensitivity(capacity_check) -> dict:
    cc = json.load(open(capacity_check))
    out = {}
    for d in DRAWS:
        c = cc[SRC.format(d)]["costs_ms"]
        rows = {"off": knee(costs(c), d), "on": knee(costs(c, saving_ms=P130_SAVING_MS), d)}
        for s in (1 + OFFSET, 1 - OFFSET):
            rows[f"off x{s:.3f}"] = knee(costs(c, scale=s), d)
            rows[f"on x{s:.3f}"] = knee(costs(c, saving_ms=P130_SAVING_MS, scale=s), d)
        for ms in (3.5, 2.0):
            rows[f"on -{ms} ms"] = knee(costs(c, saving_ms=ms), d)
        out[f"d{d}"] = rows
    return out


def spread(capacity_check, seeds=60) -> dict:
    sc = _cap()
    c = json.load(open(capacity_check))[SRC.format(1)]["costs_ms"]
    out = {}
    for name, kw in (("off", {}), ("on", {"saving_ms": P130_SAVING_MS})):
        k = costs(c, **kw)
        out[name] = {}
        for r in (12, 13, 14, 15, 16):
            a = [sc.simulate(k, sc.Workload(n=N, rate=r, seed=s), max_seqs=MAX_SEQS).attainment for s in range(seeds)]
            out[name][str(r)] = {"share_ge_0.95": round(sum(x >= 0.95 for x in a) / seeds, 2),
                                 "mean": round(statistics.fmean(a), 3)}
    return out


def levers(capacity_check) -> dict:
    sc = _cap()
    cc = json.load(open(capacity_check))
    lv = {"off": {}, "on": {"saving_ms": P130_SAVING_MS},
          "on + prefill x0.75": {"saving_ms": P130_SAVING_MS, "pf_scale": 0.75},
          "on + prefill x0.5": {"saving_ms": P130_SAVING_MS, "pf_scale": 0.5},
          "on + decode x0.8": {"saving_ms": P130_SAVING_MS, "dec_scale": 0.8}}
    out = {}
    for name, kw in lv.items():
        out[name] = {str(r): [round(sc.simulate(costs(cc[SRC.format(d)]["costs_ms"], **kw),
                                                sc.Workload(n=N, rate=r, seed=ladder_seed(d, r)),
                                                max_seqs=MAX_SEQS).attainment, 3) for d in DRAWS]
                     for r in (12, 14, 16, 20)}
    return out


def self_test() -> int:
    cases = []
    cases.append(("seeds", knee_seeds(1, 13) == [113, 11301, 11302, 11303, 11304] and knee_seeds(2, 12)[0] == 212
                  and len({s for d in DRAWS for r in KNEE for s in knee_seeds(d, r)}) == 30))
    try:
        _cap()
    except Exception as e:                # noqa: BLE001
        print(f"SC2F_BASIS self-test: serve_capacity not importable ({e}), model cases skipped")
    else:
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "cc.json")
            c1 = {"prefill_step": 40.9, "first_decode": 0.47, "decode_base": 4.664, "decode_per_row": 0.2376}
            c2 = {"prefill_step": 40.91, "first_decode": 0.49, "decode_base": 4.71, "decode_per_row": 0.2344}
            json.dump({SRC.format(1): {"costs_ms": c1}, SRC.format(2): {"costs_ms": c2}}, open(p, "w"))
            m = model(p)
            s = sensitivity(p)
        cases.append(("model: off reproduces SC2e's s64a at 12", m["d1"]["off"]["ladder"]["12"]["attainment"] == 1.0
                      and m["d2"]["off"]["ladder"]["12"]["attainment"] == 0.8917))
        cases.append(("model: the knee gain", m["d1"]["knee_gain"] == 0.0527 and m["d2"]["knee_gain"] == 0.0855))
        cases.append(("model: serial TTFT ratio", m["d1"]["on_over_off"]["serial_ttft"] == 0.882
                      and m["d1"]["on_over_off"]["serial_tpot"] == 1.0))
        cases.append(("model: ceilings", m["d1"]["off"]["ceiling"] == 12 and m["d1"]["on"]["ceiling"] == 14
                      and m["d2"]["off"]["ceiling"] == 8 and m["d2"]["on"]["ceiling"] == 12))
        cases.append(("sensitivity: an offset can erase the gain", s["d1"]["on x1.036"] < s["d1"]["off x0.964"]))
    for name, ok in cases:
        print(f"  {'ok ' if ok else 'BAD'} {name}")
    bad = [n for n, ok in cases if not ok]
    print(f"SC2F_BASIS self-test: {len(cases) - len(bad)}/{len(cases)}" + (f" FAILED {bad}" if bad else ""))
    return 1 if bad else 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("what", nargs="?", choices=("model", "sensitivity", "spread", "levers"))
    ap.add_argument("path", nargs="?")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args(argv)
    if a.self_test:
        return self_test()
    if not a.what or not a.path:
        ap.error("model|sensitivity|spread|levers CAPACITY_CHECK.json | --self-test")
    out = {"model": model, "sensitivity": sensitivity, "spread": spread, "levers": levers}[a.what](a.path)
    for k, v in out.items():
        print(k, json.dumps(v))
    return 0


if __name__ == "__main__":
    sys.exit(main())
