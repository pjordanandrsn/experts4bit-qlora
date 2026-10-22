#!/usr/bin/env python3
"""a6_size.py -- Amendment 6's held-out sizing (bench/p224/PREREG-p224.md), from the variance only. Pure Python.

The input is the reading's per-row PA - REF loss differences on the assistant positions, CENTERED (their mean
subtracted), so the file carries the spread and no effect; every SE below is shift-invariant, so centering changes
no probability. ``a6_centered_diffs.json`` is that file (written by ``--center PA.json REF.json``).

Two levels, both reported (Amendment 6 registers the TWO-LEVEL one):
- one level: draw N rows with replacement from the 256 and count SE < 0.002 (it treats the 256 as the population);
- two levels: first resample the 256 into a population (the SD's own uncertainty), then draw N rows from that.
Each K has its own rng, ``random.Random(SEED + K)``, so any one row of the table reproduces alone.

    python a6_size.py                      # prints the table from a6_centered_diffs.json
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random

SEED, DRAWS, BAR = 20261022, 4000, 0.002
KS = (73, 76, 80, 83, 86, 90, 96)
HERE = os.path.dirname(os.path.abspath(__file__))


def _se(xs) -> float:
    n = len(xs)
    m = sum(xs) / n
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (n - 1) / n)


def p_pass(d, k: int, two_level: bool, draws: int = DRAWS) -> float:
    """P(SE < BAR) for a reading of ``8 * k`` rows."""
    rng, n, big = random.Random(SEED + k), len(d), 8 * k
    ok = 0
    for _ in range(draws):
        world = [d[rng.randrange(n)] for _ in range(n)] if two_level else d
        ok += _se([world[rng.randrange(n)] for _ in range(big)]) < BAR
    return ok / draws


def center(pa_path: str, ref_path: str) -> list:
    pa, ref = json.load(open(pa_path)), json.load(open(ref_path))
    assert pa["per_seq_ids_sha256"] == ref["per_seq_ids_sha256"] and len(set(ref["per_seq_ids_sha256"])) == 256
    d = [x - y for x, y in zip(pa["per_seq_loss_assistant"], ref["per_seq_loss_assistant"])]
    m = sum(d) / len(d)
    return [x - m for x in d]


def load() -> list:
    return json.load(open(os.path.join(HERE, "a6_centered_diffs.json")))["centered_pa_minus_ref_assistant"]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--center", nargs=2, metavar=("PA_JSON", "REF_JSON"))
    a = ap.parse_args(argv)
    if a.center:
        print(json.dumps({"centered_pa_minus_ref_assistant": center(*a.center)}))
        return 0
    d = load()
    sd = _se(d) * math.sqrt(len(d))
    print(f"rows {len(d)}  SD {sd:.5f}  point N {(sd / BAR) ** 2:.0f}")
    print("| topics K | rows N | P(SE < 0.002), two levels | one level |")
    for k in KS:
        print(f"| {k} | {8 * k} | {p_pass(d, k, True):.3f} | {p_pass(d, k, False):.3f} |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
