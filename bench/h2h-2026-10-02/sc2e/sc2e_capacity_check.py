#!/usr/bin/env python3
# Copyright (c) 2026 Cerin Amroth LLC. MIT.
"""sc2e_capacity_check.py -- SC2e's post-hoc capacity-model check (descriptive; bench/sc2/SC2e-PREREG.md "Post hoc").

For each arm and draw: ``serve_capacity.StepCosts.from_step_trace`` on that server's own step trace, with the arm's
own bucket list, then ``simulate`` at the arm's ``max_seqs`` on that server's own plans (seeds draw x 100 + rate,
n 120, as box L drew them), beside the attainment the driver measured. No rule reads it.

One departure from SC2c's check: a decode step wider than the largest bucket runs as consecutive replays, and the step
trace's ``bucket`` names only the last piece. Such steps (``dec_pieces`` > 1, only on s64c) are left out of the fit;
the model chains pieces itself (``StepCosts.decode_step_s``). The excluded count is reported.

  python bench/h2h-2026-10-02/sc2e/sc2e_capacity_check.py bench/h2h-2026-10-02/sc2e/receipts/sc2e-5090-1/sc2 \
      --out bench/h2h-2026-10-02/sc2e/receipts/sc2e-5090-1/capacity_check.json
  python bench/h2h-2026-10-02/sc2e/sc2e_capacity_check.py --self-test

Step traces are read as ``steps_<tag>.jsonl`` or, as committed, ``steps_<tag>.jsonl.gz``.
"""
import argparse
import gzip
import json
import os
import sys

RATES, DRAWS, N = (1, 2, 4, 8, 12, 16), (1, 2), 120
ARMS = {"s16": (16, (1, 2, 4, 8, 16)), "s32a": (32, (1, 2, 4, 8, 16, 32)), "s64c": (64, (1, 2, 4, 8, 16)),
        "s64a": (64, (1, 2, 4, 8, 16, 32, 64))}


def _rows(path: str) -> list:
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as f:
        return [json.loads(x) for x in f if x.strip()]


def single_piece(rows: list) -> tuple:
    """The rows the fit may read, and how many chained decode-only steps were left out."""
    keep, dropped = [], 0
    for r in rows:
        if (r.get("dec_pieces") or 1) > 1 and r.get("decode_rows") and not r.get("prefill_tokens"):
            dropped += 1
            continue
        keep.append(r)
    return keep, dropped


def check(d: str) -> dict:
    from experts4bit_qlora.serve_capacity import StepCosts, Workload, simulate
    out = {}
    for arm, (m, buckets) in ARMS.items():
        for k in DRAWS:
            tag = f"e4b_{arm}_d{k}"
            sp = next((p for p in (os.path.join(d, f"steps_{tag}.jsonl{z}") for z in ("", ".gz")) if os.path.exists(p)),
                      None)
            if sp is None:
                out[tag] = {"why": "no step trace"}
                continue
            rows, dropped = single_piece(_rows(sp))
            try:
                costs = StepCosts.from_step_trace(rows, buckets=buckets, source=os.path.basename(sp))
            except ValueError as e:
                out[tag] = {"why": str(e)}
                continue
            row = {"max_seqs": m, "buckets": list(buckets), "chained_steps_left_out": dropped,
                   "costs_ms": {"prefill_step": round(costs.prefill_step_s * 1e3, 2),
                                "first_decode": round(costs.first_decode_s * 1e3, 2),
                                "decode_base": round(costs.decode_base_s * 1e3, 3),
                                "decode_per_row": round(costs.decode_per_row_s * 1e3, 4),
                                "decode_step_at_max_seqs": round(costs.decode_step_s(m) * 1e3, 2)},
                   "rates": {}}
            for r in RATES:
                f = os.path.join(d, f"{tag}_r{r}.json")
                measured = json.load(open(f))["summary"]["attainment"] if os.path.exists(f) else None
                model = simulate(costs, Workload(n=N, rate=r, seed=k * 100 + r), max_seqs=m).attainment
                row["rates"][str(r)] = {"model": round(model, 3), "measured": measured,
                                        "gap": (round(model - measured, 3) if measured is not None else None)}
            out[tag] = row
    return out


def self_test() -> int:
    cases = []
    rows = [{"decode_rows": 60, "bucket": 12, "dec_pieces": 4, "step_ms": 38.0, "seg": {}},
            {"decode_rows": 16, "bucket": 16, "dec_pieces": 1, "step_ms": 9.0, "seg": {}},
            {"decode_rows": 60, "bucket": 12, "dec_pieces": 4, "prefill_tokens": 512, "prefill_replays": 1,
             "step_ms": 80.0, "seg": {}},
            {"decode_rows": 8, "bucket": 8, "step_ms": 7.0, "seg": {}}]
    keep, dropped = single_piece(rows)
    cases.append(("chained decode-only steps leave the fit", dropped == 1 and len(keep) == 3
                  and all((r.get("dec_pieces") or 1) == 1 or r.get("prefill_tokens") for r in keep)))
    try:
        from experts4bit_qlora.serve_capacity import StepCosts
        tr = []
        for i in range(40):
            tr.append({"prefill_replays": 1, "prefill_tokens": 512, "step_ms": 40.0,
                       "seg": {"pf_forward": 39.0, "pf_flush": 0.5, "dec_ready": 0.8}})
            for b in (1, 2, 4, 8, 16):
                tr.append({"decode_rows": b, "bucket": b, "step_ms": 4.0 + 0.35 * b, "seg": {}})
            tr.append({"decode_rows": 60, "bucket": 12, "dec_pieces": 4, "step_ms": 99.0, "seg": {}})
        c = StepCosts.from_step_trace(single_piece(tr)[0], buckets=(1, 2, 4, 8, 16))
        cases.append(("the fit ignores the chained steps", abs(c.decode_per_row_s * 1e3 - 0.35) < 1e-9
                      and abs(c.decode_base_s * 1e3 - 4.0) < 1e-9 and abs(c.decode_step_s(64) * 1e3 - 4 * (4.0 + 5.6)) < 1e-9))
    except ImportError:
        print("SC2E_CAPACITY self-test: experts4bit_qlora not importable, the fit case skipped")
    for name, ok in cases:
        print(f"  {'ok ' if ok else 'BAD'} {name}")
    bad = [n for n, ok in cases if not ok]
    print(f"SC2E_CAPACITY self-test: {len(cases) - len(bad)}/{len(cases)}" + (f" FAILED {bad}" if bad else ""))
    return 1 if bad else 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("dir", nargs="?")
    ap.add_argument("--out")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args(argv)
    if a.self_test:
        return self_test()
    if not a.dir:
        ap.error("DIR | --self-test")
    out = check(a.dir)
    for tag, row in out.items():
        print(tag, json.dumps(row))
    if a.out:
        with open(a.out, "w", newline="\n") as f:
            json.dump(out, f, indent=1)
            f.write("\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
