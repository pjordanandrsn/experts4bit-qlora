# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""D0s's rule (PREREG-d0s.md, "The rule"), applied to the box's arm records. CPU, standard library only.

    python bench/d0s/d0s_reduce.py --run RUN --d0 bench/d0/receipts/d0-p224s3 --ref-argmax DIR --out verdict.json

Per fixture f and merged cell X (``MERGED-Q``, ``MERGED-W``):

1. R = sum_rows(nll_BASE-ATTN-1 - nll_X) / sum_rows(nll_Q - nll_REF), the 90 % CI by a paired bootstrap over rows
   (2,000 resamples, ``random.Random(0)``); PASS iff the lower bound >= 0.95.
2. gap = [L(X) - L_D0(REF)] - [L(BASE-ATTN-1) - L_D0(Q)], every L pooled per token; PASS iff |gap| <= M_f, the
   registered absolute margin (0.05 x D0's improvement, checked here against the constants the registration fixed).
3. Canary, pooled exact counts: PASS iff 100 a_X >= 100 a_B - N (a_X: X's argmax == REF's; a_B: BASE-ATTN-1's == Q's).
4. Cost: each of TTFT, ITL and tokens/s at W1 and W16 within |m_X - m_B1| <= |m_B2 - m_B1| + 0.03 m_B1, and peak VRAM
   within 1 %.

A cell PASSES iff 1-4 all pass. A cell the writer refused is WRITER_REFUSED; an arm without a record is NOT_READ. Rows
or token counts that differ from D0's record make the fixture VOID.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random

#: the margins PREREG-d0s.md fixed, 0.05 x D0's registered improvement (never recomputed from the read)
MARGINS = {"nodq": 0.016432685048378637, "dq": 0.01682758935300882}
R_FLOOR, CANARY_POINTS, COST_FLOOR, VRAM_TOL = 0.95, 1, 0.03, 0.01
BOOT, SEED = 2000, 0
CELLS = ("MERGED-Q", "MERGED-W")
COST_METRICS = ("ttft_s", "itl_s", "tok_s")


def d0_record(d0_dir: str, fx: str) -> dict:
    """D0's per-row Q and REF NLL and token counts, and its registered improvement."""
    rows = {}
    with open(os.path.join(d0_dir, fx, "pass", "arms.csv"), newline="") as f:
        for r in csv.DictReader(f):
            if r["arm"] in ("Q", "REF"):
                rows.setdefault(int(r["row"]), {})[r["arm"]] = (float(r["nll"]), int(r["n"]))
    with open(os.path.join(d0_dir, fx, "summary.json")) as f:
        improvement = json.load(f)["step2"]["improvement"]
    return {"rows": [rows[i] for i in range(len(rows))], "improvement": improvement}


def pooled(nll, n):
    return sum(nll) / sum(n)


def percentile(xs, q):
    """The q-quantile of xs by linear interpolation (numpy's default)."""
    s = sorted(xs)
    pos = q * (len(s) - 1)
    lo = math.floor(pos)
    hi = min(lo + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (pos - lo)


def retention(num, den, *, boot=BOOT, seed=SEED):
    """Point R = sum(num)/sum(den) and its paired 90 % bootstrap CI over rows."""
    point = sum(num) / sum(den)
    rng = random.Random(seed)
    k = len(num)
    rs = []
    for _ in range(boot):
        idx = [rng.randrange(k) for _ in range(k)]
        d = sum(den[i] for i in idx)
        rs.append(sum(num[i] for i in idx) / d if d else float("nan"))
    return point, (percentile(rs, 0.05), percentile(rs, 0.95))


def cost_pass(x: dict, b1: dict, b2: dict) -> tuple:
    """(pass, rows): every metric at every width within the self-pair band, and peak VRAM within 1 %."""
    rows, ok = [], True
    for width in sorted(b1["cost"]):
        for m in COST_METRICS:
            mx, m1, m2 = x["cost"][width][m], b1["cost"][width][m], b2["cost"][width][m]
            band = abs(m2 - m1) + COST_FLOOR * abs(m1)
            good = abs(mx - m1) <= band
            ok &= good
            rows.append({"width": width, "metric": m, "x": mx, "b1": m1, "b2": m2, "band": band, "pass": good})
    v_ok = abs(x["peak_vram_bytes"] - b1["peak_vram_bytes"]) <= VRAM_TOL * b1["peak_vram_bytes"]
    rows.append({"metric": "peak_vram_bytes", "x": x["peak_vram_bytes"], "b1": b1["peak_vram_bytes"], "pass": v_ok})
    return ok and v_ok, rows


def reduce_fixture(fx: str, arms: dict, d0: dict, ref: dict, prepare: dict) -> dict:
    """The verdict for one fixture from its arm records (``{arm: record}``), D0's record and the REF argmax file."""
    out = {"fixture": fx, "cells": {}}
    n_rows = len(d0["rows"])
    margin = MARGINS[fx]
    if not math.isclose(0.05 * d0["improvement"], margin, rel_tol=0, abs_tol=1e-15):
        raise SystemExit(f"D0S VOID: 0.05 x D0's {fx} improvement is {0.05 * d0['improvement']!r}, not the registered "
                         f"{margin!r}")
    for arm in ("BASE-ATTN-1", "BASE-ATTN-2"):
        if arm not in arms:
            return {**out, "status": "NOT_READ", "why": f"no {arm} record"}
    n_ref = [r["REF"][1] for r in d0["rows"]]
    for arm, rec in arms.items():
        if [r["n"] for r in rec["rows"]] != n_ref:
            return {**out, "status": "VOID", "why": f"{arm}'s rows or token counts are not D0's"}
    if len(ref["rows"]) != n_rows or [r["n"] for r in ref["rows"]] != n_ref:
        return {**out, "status": "VOID", "why": "the REF argmax rows are not D0's"}
    nll = {arm: [r["nll"] for r in rec["rows"]] for arm, rec in arms.items()}
    q = [r["Q"][0] for r in d0["rows"]]
    rf = [r["REF"][0] for r in d0["rows"]]
    den = [a - b for a, b in zip(q, rf)]
    L = {arm: pooled(v, n_ref) for arm, v in nll.items()}
    L_q, L_ref = pooled(q, n_ref), pooled(rf, n_ref)
    N = sum(n_ref)
    a_b = sum(int(x == y) for rec_row, ref_row in zip(arms["BASE-ATTN-1"]["rows"], ref["rows"])
              for x, y in zip(rec_row["argmax"], ref_row["q_argmax"]))
    out.update(N=N, L={**L, "D0-Q": L_q, "D0-REF": L_ref}, improvement=d0["improvement"], margin=margin,
               attention_offset=L["BASE-ATTN-1"] - L_q, self_pair_nll_diff=sum(nll["BASE-ATTN-2"]) - sum(nll["BASE-ATTN-1"]),
               canary_base_agree=a_b)
    for cell in CELLS:
        cell_prep = ((prepare.get("fixtures") or {}).get(fx, {}).get("cells") or {}).get(cell, {})
        if cell_prep.get("status") == "WRITER_REFUSED":
            out["cells"][cell] = {"status": "WRITER_REFUSED", "why": cell_prep.get("why")}
            continue
        if cell not in arms:
            out["cells"][cell] = {"status": "NOT_READ", "why": f"no {cell} record"}
            continue
        num = [b - x for b, x in zip(nll["BASE-ATTN-1"], nll[cell])]
        r_point, (r_lo, r_hi) = retention(num, den)
        gap = (L[cell] - L_ref) - (L["BASE-ATTN-1"] - L_q)
        a_x = sum(int(x == y) for rec_row, ref_row in zip(arms[cell]["rows"], ref["rows"])
                  for x, y in zip(rec_row["argmax"], ref_row["ref_argmax"]))
        canary = 100 * a_x >= 100 * a_b - CANARY_POINTS * N
        c_ok, c_rows = cost_pass(arms[cell], arms["BASE-ATTN-1"], arms["BASE-ATTN-2"])
        gates = {"retention": r_lo >= R_FLOOR, "gap": abs(gap) <= margin, "canary": canary, "cost": c_ok}
        out["cells"][cell] = {
            "status": "PASS" if all(gates.values()) else "FAIL", "gates": gates,
            "R": r_point, "R_ci90": [r_lo, r_hi], "gap": gap, "canary_agree": a_x, "cost": c_rows,
            "writer_retention": (cell_prep.get("merge") or {}).get("retention_summary")}
    out["status"] = "READ"
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--run", required=True, help="the box run directory (prepare.json, results/)")
    ap.add_argument("--d0", required=True, help="D0's receipt directory (bench/d0/receipts/d0-p224s3)")
    ap.add_argument("--ref-argmax", required=True, help="the directory holding ref-argmax-<fixture>.json")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    with open(os.path.join(a.run, "prepare.json")) as f:
        prepare = json.load(f)
    verdict = {"scheme": "d0s-reduce/1", "fixtures": {}}
    for fx in ("nodq", "dq"):
        arms = {}
        for arm in ("BASE-ATTN-1", "BASE-ATTN-2") + CELLS:
            path = os.path.join(a.run, "results", f"{fx}-{arm}.json")
            if os.path.isfile(path):
                with open(path) as f:
                    arms[arm] = json.load(f)
        with open(os.path.join(a.ref_argmax, f"ref-argmax-{fx}.json")) as f:
            ref = json.load(f)
        if ref.get("status") != "OK":
            verdict["fixtures"][fx] = {"fixture": fx, "status": "VOID", "why": "the REF argmax is not OK"}
            continue
        verdict["fixtures"][fx] = reduce_fixture(fx, arms, d0_record(a.d0, fx), ref, prepare)
    with open(a.out, "w", encoding="utf-8") as f:
        f.write(json.dumps(verdict, indent=1, sort_keys=True) + "\n")
    for fx, v in verdict["fixtures"].items():
        cells = {c: (x["status"], round(x["R"], 4) if "R" in x else None) for c, x in (v.get("cells") or {}).items()}
        print(f"D0S {fx}: {v['status']} {cells}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
