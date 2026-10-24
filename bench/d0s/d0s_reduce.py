# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""D0s's rule (PREREG-d0s.md, "The rule"), applied to the box's arm records. CPU, standard library only.

    python bench/d0s/d0s_reduce.py --run RUN --d0 bench/d0/receipts/d0-p224s3 --ref-argmax DIR --out verdict.json

Per fixture f and merged cell X (``MERGED-Q``, ``MERGED-W``):

1. R = sum_rows(nll_BASE-ATTN-1 - nll_X) / sum_rows(nll_Q - nll_REF), the 90 % CI by a paired bootstrap over rows
   (2,000 resamples, ``random.Random(0)``); PASS iff the lower bound >= 0.95.
2. gap = [L(X) - L_D0(REF)] - [L(BASE-ATTN-1) - L_D0(Q)], every L pooled per token; PASS iff |gap| <= M_f, the
   registered absolute margin (0.05 x D0's improvement, checked here against the constants the registration fixed).
3. Canary, pooled exact counts: PASS iff 100 a_X >= 100 a_B - N (a_X: X's argmax == REF's; a_B: BASE-ATTN-1's == Q's).
4. Cost, slower-only: each of TTFT and ITL at W1 and W16 within m_X - m_B1 <= |m_B2 - m_B1| + 0.03 m_B1 (tokens/s
   mirrored: m_B1 - m_X), and peak VRAM within 1 % either way.
5. Engagement, before any gate: a record that is incomplete (rows or positions not D0's, an argmax list not one int per
   scored position, an NLL that is not a finite number, a cost record without all three metrics at W1 and W16, no peak
   VRAM) or served on another path (a BASE arm with a merged arena; a MERGED arm whose block's source is not its own,
   whose block or served adapter is not its fixture's, or whose expert base is not ``verified (merged)``; any arm whose
   placement, graph status or device differs from ``BASE-ATTN-1``'s) is VOID: the fixture's when it is a BASE arm or the
   REF argmax file, the cell's when it is a MERGED arm.

Gate 2 is two-sided on purpose (fidelity): a gap below -M (R above 1.05) FAILS as BEYOND_REF, "beyond REF: out of band,
an artifact to explain", never read as worse than REF; a gap above M FAILS as BELOW_REF. Gate 4 is one-sided (overhead):
a faster merged arm passes, and a speedup beyond 3 % is named as unexplained.

A cell PASSES iff 1-4 all pass and 5 holds. A cell the writer refused is WRITER_REFUSED; an arm without a record is
NOT_READ.
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
WIDTHS = ("W1", "W16")
#: metric -> True when a larger value is slower
COST_METRICS = {"ttft_s": True, "itl_s": True, "tok_s": False}
SOURCES = {"MERGED-Q": "arena", "MERGED-W": "checkpoint"}
MERGED_VERIFIED = "verified (merged)"
BEYOND_REF = "BEYOND_REF"
BELOW_REF = "BELOW_REF"
SIDE_TEXT = {BEYOND_REF: "beyond REF: out of band, an artifact to explain", BELOW_REF: "below REF: the retention is lost"}


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


def _number(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def _ints(xs, n) -> bool:
    return isinstance(xs, list) and len(xs) == n and all(isinstance(x, int) and not isinstance(x, bool) for x in xs)


def record_problems(rec, n_ref) -> list:
    """Gate 5's completeness half: why an arm record cannot be read (empty when it is whole)."""
    if not isinstance(rec, dict):
        return ["the record is not an object"]
    rows = rec.get("rows")
    if not isinstance(rows, list) or len(rows) != len(n_ref):
        return [f"{len(rows) if isinstance(rows, list) else 'no'} rows, D0 {len(n_ref)}"]
    out = []
    for i, (r, n) in enumerate(zip(rows, n_ref)):
        r = r if isinstance(r, dict) else {}
        if r.get("n") != n:
            out.append(f"row {i} scores {r.get('n')!r} positions, D0 {n}")
        if not _number(r.get("nll")):
            out.append(f"row {i} nll {r.get('nll')!r}")
        if not _ints(r.get("argmax"), n):
            out.append(f"row {i} argmax is not one int per scored position")
    cost = rec.get("cost") if isinstance(rec.get("cost"), dict) else {}
    for w in WIDTHS:
        for m in COST_METRICS:
            v = cost.get(w, {}).get(m) if isinstance(cost.get(w), dict) else None
            if not (_number(v) and v > 0):
                out.append(f"cost {w}.{m} {v!r}")
    if not (_number(rec.get("peak_vram_bytes")) and rec["peak_vram_bytes"] > 0):
        out.append(f"peak_vram_bytes {rec.get('peak_vram_bytes')!r}")
    return out


def ref_problems(ref, n_ref) -> list:
    rows = ref.get("rows") if isinstance(ref, dict) else None
    if not isinstance(rows, list) or len(rows) != len(n_ref):
        return ["the REF argmax rows are not D0's"]
    out = []
    for i, (r, n) in enumerate(zip(rows, n_ref)):
        r = r if isinstance(r, dict) else {}
        if r.get("n") != n or not _ints(r.get("ref_argmax"), n) or not _ints(r.get("q_argmax"), n):
            out.append(f"REF argmax row {i} is not one int per scored position ({n})")
    return out


def engagement_problems(arm: str, rec: dict, b1: dict, ids: dict) -> list:
    """Gate 5's path half: why this arm was not served as its arm (empty when it was)."""
    s = rec.get("served") if isinstance(rec.get("served"), dict) else {}
    out = []
    if arm.startswith("BASE-ATTN"):
        want = ids.get("attention_only")
        if s.get("merged_arena") is not None:
            out.append(f"{arm} served a merged arena")
    else:
        want = ids.get("adapter")
        block = s.get("merged_arena")
        if not isinstance(block, dict) or block.get("source") != SOURCES[arm]:
            out.append(f"{arm} served {'block source ' + repr(block.get('source')) if isinstance(block, dict) else 'no merged arena'}"
                       f", not {SOURCES[arm]!r}")
        elif want is None or block.get("adapter_id") != want:
            out.append(f"{arm}'s merged block was written for adapter {block.get('adapter_id')!r}, not {want!r}")
        status = ((s.get("fingerprints") or {}).get("experts") or {}).get("status")
        if status != MERGED_VERIFIED:
            out.append(f"{arm}'s expert base is {status!r}, not {MERGED_VERIFIED!r}")
    if want is None or s.get("adapter_id") != want:
        out.append(f"{arm} served adapter {s.get('adapter_id')!r}, not {want!r}")
    b1s = b1.get("served") if isinstance(b1.get("served"), dict) else {}
    for key in ("placement", "graph_status"):
        if key not in s or s.get(key) != b1s.get(key):
            out.append(f"{arm}'s {key} {s.get(key)!r} is not BASE-ATTN-1's {b1s.get(key)!r}")
    if rec.get("device") is None or rec.get("device") != b1.get("device"):
        out.append(f"{arm} ran on {rec.get('device')!r}, BASE-ATTN-1 on {b1.get('device')!r}")
    return out


def cost_pass(x: dict, b1: dict, b2: dict) -> tuple:
    """(pass, rows, unexplained speedups): every metric at both widths no slower than BASE-ATTN-1 by more than the
    self-pair's band floored at 3 %, and peak VRAM within 1 % either way."""
    rows, ok, fast = [], True, []
    for width in WIDTHS:
        for m, higher_is_slower in COST_METRICS.items():
            mx, m1, m2 = x["cost"][width][m], b1["cost"][width][m], b2["cost"][width][m]
            band = abs(m2 - m1) + COST_FLOOR * abs(m1)
            slower_by = (mx - m1) if higher_is_slower else (m1 - mx)
            good = slower_by <= band
            ok &= good
            if -slower_by > COST_FLOOR * abs(m1):
                fast.append(f"{width}.{m}")
            rows.append({"width": width, "metric": m, "x": mx, "b1": m1, "b2": m2, "band": band, "slower_by": slower_by,
                         "pass": good})
    v_ok = abs(x["peak_vram_bytes"] - b1["peak_vram_bytes"]) <= VRAM_TOL * b1["peak_vram_bytes"]
    rows.append({"metric": "peak_vram_bytes", "x": x["peak_vram_bytes"], "b1": b1["peak_vram_bytes"], "pass": v_ok})
    return ok and v_ok, rows, fast


def _void(out, why):
    return {**out, "status": "VOID", "why": why if isinstance(why, str) else "; ".join(why[:6])}


def reduce_fixture(fx: str, arms: dict, d0: dict, ref: dict, prepare: dict) -> dict:
    """The verdict for one fixture from its arm records (``{arm: record}``), D0's record, the REF argmax file and the
    box's prepare.json (the writer's cells and the derived adapters' ids)."""
    out = {"fixture": fx, "cells": {}}
    margin = MARGINS[fx]
    if not math.isclose(0.05 * d0["improvement"], margin, rel_tol=0, abs_tol=1e-15):
        raise SystemExit(f"D0S VOID: 0.05 x D0's {fx} improvement is {0.05 * d0['improvement']!r}, not the registered "
                         f"{margin!r}")
    for arm in ("BASE-ATTN-1", "BASE-ATTN-2"):
        if arm not in arms:
            return {**out, "status": "NOT_READ", "why": f"no {arm} record"}
    n_ref = [r["REF"][1] for r in d0["rows"]]
    bad = ref_problems(ref, n_ref)
    if bad:
        return _void(out, bad)
    fx_prep = ((prepare or {}).get("fixtures") or {}).get(fx) or {}
    attn = fx_prep.get("attention_only") or {}
    ids = {"attention_only": attn.get("adapter_id"), "adapter": attn.get("source_adapter_id")}
    b1 = arms["BASE-ATTN-1"]
    for arm in ("BASE-ATTN-1", "BASE-ATTN-2"):
        bad = record_problems(arms[arm], n_ref) or engagement_problems(arm, arms[arm], b1, ids)
        if bad:
            return _void(out, [f"{arm}: {b}" for b in bad])
    q = [r["Q"][0] for r in d0["rows"]]
    rf = [r["REF"][0] for r in d0["rows"]]
    den = [a - b for a, b in zip(q, rf)]
    nll = {arm: [r["nll"] for r in arms[arm]["rows"]] for arm in ("BASE-ATTN-1", "BASE-ATTN-2")}
    L = {arm: pooled(v, n_ref) for arm, v in nll.items()}
    L_q, L_ref = pooled(q, n_ref), pooled(rf, n_ref)
    N = sum(n_ref)
    a_b = sum(int(x == y) for rec_row, ref_row in zip(b1["rows"], ref["rows"])
              for x, y in zip(rec_row["argmax"], ref_row["q_argmax"]))
    out.update(N=N, improvement=d0["improvement"], margin=margin, attention_offset=L["BASE-ATTN-1"] - L_q,
               self_pair_nll_diff=sum(nll["BASE-ATTN-2"]) - sum(nll["BASE-ATTN-1"]), canary_base_agree=a_b)
    for cell in CELLS:
        cell_prep = (fx_prep.get("cells") or {}).get(cell, {})
        if cell_prep.get("status") == "WRITER_REFUSED":
            out["cells"][cell] = {"status": "WRITER_REFUSED", "why": cell_prep.get("why")}
            continue
        if cell not in arms:
            out["cells"][cell] = {"status": "NOT_READ", "why": f"no {cell} record"}
            continue
        x = arms[cell]
        bad = record_problems(x, n_ref) or engagement_problems(cell, x, b1, ids)
        if bad:
            out["cells"][cell] = _void({}, bad)
            continue
        x_nll = [r["nll"] for r in x["rows"]]
        L[cell] = pooled(x_nll, n_ref)
        num = [b - v for b, v in zip(nll["BASE-ATTN-1"], x_nll)]
        r_point, (r_lo, r_hi) = retention(num, den)
        gap = (L[cell] - L_ref) - (L["BASE-ATTN-1"] - L_q)
        side = BEYOND_REF if gap < -margin else BELOW_REF if gap > margin else None
        a_x = sum(int(v == y) for rec_row, ref_row in zip(x["rows"], ref["rows"])
                  for v, y in zip(rec_row["argmax"], ref_row["ref_argmax"]))
        canary = 100 * a_x >= 100 * a_b - CANARY_POINTS * N
        c_ok, c_rows, fast = cost_pass(x, b1, arms["BASE-ATTN-2"])
        gates = {"retention": r_lo >= R_FLOOR, "gap": side is None, "canary": canary, "cost": c_ok}
        out["cells"][cell] = {
            "status": "PASS" if all(gates.values()) else "FAIL", "gates": gates, "gap_side": side,
            "gap_side_text": SIDE_TEXT.get(side), "R": r_point, "R_ci90": [r_lo, r_hi], "gap": gap, "canary_agree": a_x,
            "cost": c_rows, "unexplained_speedup": fast,
            "writer_retention": (cell_prep.get("merge") or {}).get("retention_summary")}
    out["L"] = {**L, "D0-Q": L_q, "D0-REF": L_ref}
    out["status"] = "READ"
    return out


def _load(path):
    try:
        with open(path) as f:
            return json.load(f)
    except ValueError as e:
        return {"unreadable": f"{os.path.basename(path)}: {e}"}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--run", required=True, help="the box run directory (prepare.json, results/)")
    ap.add_argument("--d0", required=True, help="D0's receipt directory (bench/d0/receipts/d0-p224s3)")
    ap.add_argument("--ref-argmax", required=True, help="the directory holding ref-argmax-<fixture>.json")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    prepare = _load(os.path.join(a.run, "prepare.json"))
    verdict = {"scheme": "d0s-reduce/2", "fixtures": {}}
    for fx in ("nodq", "dq"):
        arms = {}
        for arm in ("BASE-ATTN-1", "BASE-ATTN-2") + CELLS:
            path = os.path.join(a.run, "results", f"{fx}-{arm}.json")
            if os.path.isfile(path):
                arms[arm] = _load(path)
        ref = _load(os.path.join(a.ref_argmax, f"ref-argmax-{fx}.json"))
        if ref.get("status") != "OK":
            verdict["fixtures"][fx] = {"fixture": fx, "status": "VOID", "why": "the REF argmax is not OK"}
            continue
        verdict["fixtures"][fx] = reduce_fixture(fx, arms, d0_record(a.d0, fx), ref, prepare)
    with open(a.out, "w", encoding="utf-8") as f:
        f.write(json.dumps(verdict, indent=1, sort_keys=True) + "\n")
    for fx, v in verdict["fixtures"].items():
        cells = {c: (x["status"], round(x["R"], 4) if "R" in x else None, x.get("gap_side") or x.get("why"))
                 for c, x in (v.get("cells") or {}).items()}
        print(f"D0S {fx}: {v['status']}{' (' + v['why'] + ')' if v.get('why') else ''} {cells}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
