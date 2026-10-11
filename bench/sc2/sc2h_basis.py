#!/usr/bin/env python3
# Copyright (c) 2026 Cerin Amroth LLC. MIT.
"""sc2h_basis.py -- lane SC2h's predictions, before data (bench/sc2/SC2h-PREREG.md; #846).

``serve_capacity`` on SC2e's s64a costs (``capacity_check.json``) with P130's 5.18 ms prefill saving (SC2f's ``on``, this
lane's ``base``). The lookahead (``la``) is modelled as removing the exposed host gap from every decode step: the least
squares of ``gpu.dec_prep + step_ms - gpu.dec_issue`` on the bucket over SC2e's s64a decode-only steps, the same fit the
model's own step costs use (so the two subtract consistently). That is a ceiling: it assumes the lookahead hides all of
the gap at every bucket and adds no device time. Seeds are box O's: ladder draw x 100 + rate, knee 10,000 x draw +
100 x rate + j.

  sc2h_basis.py model      the knee blocks the rule can choose, serial and loaded TPOT ratios, the 0.95 crossing
  sc2h_basis.py --self-test
"""
import argparse
import gzip
import importlib.util
import json
import os
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
SC2E = os.path.join(REPO, "bench", "h2h-2026-10-02", "sc2e", "receipts", "sc2e-5090-1")
DRAWS = (1, 2)
BLOCKS = ((12, 13, 14), (13, 14, 15), (15, 16, 17), (16, 17, 18), (4, 8, 12))
KNEE_J = (1, 2, 3, 4)
DEC_SCALE = 0.85        # sensitivity: 0.54.0's decode device faster than SC2e's (P122, P124, P126 since)


def _sib(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, f"{name}.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def ols(xs, ys):
    mx, my = statistics.fmean(xs), statistics.fmean(ys)
    s = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sum((x - mx) ** 2 for x in xs)
    return my - s * mx, s


def host_fit(rows):
    dec = [r for r in rows if r.get("decode_rows") and not r.get("prefill_tokens") and r.get("bucket")]
    return ols([float(r["bucket"]) for r in dec], [r["gpu"]["dec_prep"] + r["step_ms"] - r["gpu"]["dec_issue"] for r in dec])


def _costs(sc, b, c, *, host=(0.0, 0.0), dec_scale=1.0):
    hb, hr = host
    base_ms, row_ms = c["decode_base"] - hb, c["decode_per_row"] - hr
    return sc.StepCosts(prefill_step_s=(c["prefill_step"] - b.P130_SAVING_MS) / 1e3, first_decode_s=c["first_decode"] / 1e3,
                        decode_base_s=base_ms * dec_scale / 1e3, decode_per_row_s=row_ms * dec_scale / 1e3,
                        buckets=b.BUCKETS)


def _knee(sc, b, k, draw, rates):
    seeds = [(r, draw * 100 + r) for r in rates] + [(r, 10000 * draw + 100 * r + j) for r in rates for j in KNEE_J]
    return round(statistics.fmean(sc.simulate(k, sc.Workload(n=b.N, rate=r, seed=s), max_seqs=b.MAX_SEQS).attainment
                                  for r, s in seeds), 4)


def model() -> dict:
    b = _sib("sc2f_basis")
    sc = b._cap()
    cc = json.load(open(os.path.join(SC2E, "capacity_check.json")))
    out = {"p130_saving_ms": b.P130_SAVING_MS, "dec_scale_sensitivity": DEC_SCALE}
    for d in DRAWS:
        c = cc[b.SRC.format(d)]["costs_ms"]
        with gzip.open(os.path.join(SC2E, "sc2", f"steps_e4b_s64a_d{d}.jsonl.gz"), "rt") as f:
            hf = host_fit([json.loads(x) for x in f if x.strip()])
        row = {"host_gap_fit": [round(v, 4) for v in hf]}
        for name, scale in (("sc2e_costs", 1.0), (f"decode_x{DEC_SCALE}", DEC_SCALE)):
            # under the scale, base's decode = (fit) x scale; la's = (fit - host) x scale + nothing: the host part does
            # not scale with a faster device, so la removes the same host gap from base's scaled step
            kb = _costs(sc, b, c, dec_scale=scale) if scale == 1.0 else sc.StepCosts(
                prefill_step_s=(c["prefill_step"] - b.P130_SAVING_MS) / 1e3, first_decode_s=c["first_decode"] / 1e3,
                decode_base_s=((c["decode_base"] - hf[0]) * scale + hf[0]) / 1e3,
                decode_per_row_s=((c["decode_per_row"] - hf[1]) * scale + hf[1]) / 1e3, buckets=b.BUCKETS)
            kl = _costs(sc, b, c, host=hf, dec_scale=scale)
            arms = {"base": b.arm(kb, d), "la": b.arm(kl, d)}
            blocks = {}
            for rates in BLOCKS:
                kb_, kl_ = _knee(sc, b, kb, d, rates), _knee(sc, b, kl, d, rates)
                blocks[",".join(map(str, rates))] = {"base": kb_, "la": kl_, "gain": round(kl_ - kb_, 4)}
            ratio = {"serial_tpot": round(arms["la"]["serial"]["tpot_ms_p50"] / arms["base"]["serial"]["tpot_ms_p50"], 3),
                     "serial_ttft": round(arms["la"]["serial"]["ttft_ms_p50"] / arms["base"]["serial"]["ttft_ms_p50"], 3)}
            for r in (8, 12):
                ratio[f"r{r}_tpot"] = round(arms["la"]["ladder"][str(r)]["tpot_ms_p50"] / arms["base"]["ladder"][str(r)]["tpot_ms_p50"], 3)
            row[name] = {"step_at_64_ms": {"base": round((kb.decode_base_s + 64 * kb.decode_per_row_s) * 1e3, 2),
                                           "la": round((kl.decode_base_s + 64 * kl.decode_per_row_s) * 1e3, 2)},
                         "ceilings": {a: arms[a]["ceiling"] for a in arms}, "knee_blocks": blocks, "la_over_base": ratio}
        out[f"d{d}"] = row
    return out


def self_test() -> int:
    ok = []
    a, s = ols([1.0, 2.0, 3.0], [2.0, 4.0, 6.0])
    ok.append(abs(a) < 1e-12 and abs(s - 2.0) < 1e-12)
    rows = [{"decode_rows": 1, "bucket": x, "step_ms": 4.0 + x, "gpu": {"dec_prep": 0.5, "dec_issue": 3.5 + x * 0.9}}
            for x in (1, 2, 4, 8)]
    hb, hr = host_fit(rows)
    ok.append(abs(hb - 1.0) < 1e-9 and abs(hr - 0.1) < 1e-9)
    print(f"SC2H_BASIS self-test: {sum(ok)}/{len(ok)}")
    return 0 if all(ok) else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("what", nargs="?", choices=("model",))
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args(argv)
    if a.self_test:
        return self_test()
    if a.what != "model":
        ap.error("model | --self-test")
    print(json.dumps(model(), indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
