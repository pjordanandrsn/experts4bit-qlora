#!/usr/bin/env python3
# Copyright (c) 2026 Cerin Amroth LLC. MIT.
"""sc2c_capacity_check.py -- SC2c's post-hoc capacity-model check (descriptive; bench/sc2/SC2c-PREREG.md "Post hoc").

For each arm and draw: ``serve_capacity.StepCosts.from_step_trace`` on that server's own step trace, ``simulate`` on
that server's own plans (seeds draw x 100 + rate, n 120, as box H drew them), beside the attainment the driver
measured. No rule reads it.

  python bench/h2h-2026-10-02/sc2c/sc2c_capacity_check.py bench/h2h-2026-10-02/sc2c/receipts/sc2c-5090-1/sc2 \
      --out bench/h2h-2026-10-02/sc2c/receipts/sc2c-5090-1/capacity_check.json

Step traces are read as ``steps_<tag>.jsonl`` or, as committed here, ``steps_<tag>.jsonl.gz``.
"""
import argparse
import gzip
import json
import os

from experts4bit_qlora.serve_capacity import StepCosts, Workload, simulate

RATES, DRAWS, ARMS, N = (1, 2, 4, 8), (1, 2), ("off", "on"), 120


def _rows(path: str) -> list:
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as f:
        return [json.loads(x) for x in f if x.strip()]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--out")
    a = ap.parse_args(argv)
    out = {}
    for arm in ARMS:
        for d in DRAWS:
            tag = f"e4b_{arm}_d{d}"
            sp = next((p for p in (os.path.join(a.dir, f"steps_{tag}.jsonl{z}") for z in ("", ".gz"))
                       if os.path.exists(p)), None)
            if sp is None:
                out[tag] = {"why": "no step trace"}
                continue
            try:
                costs = StepCosts.from_step_trace(_rows(sp), source=os.path.basename(sp))
            except ValueError as e:
                out[tag] = {"why": str(e)}
                continue
            row = {"costs_ms": {"prefill_step": round(costs.prefill_step_s * 1e3, 2),
                                "first_decode": round(costs.first_decode_s * 1e3, 2),
                                "decode_base": round(costs.decode_base_s * 1e3, 3),
                                "decode_per_row": round(costs.decode_per_row_s * 1e3, 4)},
                   "rates": {}}
            for r in RATES:
                f = os.path.join(a.dir, f"{tag}_r{r}.json")
                measured = json.load(open(f))["summary"]["attainment"] if os.path.exists(f) else None
                model = simulate(costs, Workload(n=N, rate=r, seed=d * 100 + r)).attainment
                row["rates"][str(r)] = {"model": round(model, 3), "measured": measured,
                                        "gap": (round(model - measured, 3) if measured is not None else None)}
            out[tag] = row
            print(tag, json.dumps(row))
    if a.out:
        with open(a.out, "w", newline="\n") as f:
            json.dump(out, f, indent=1)
            f.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
