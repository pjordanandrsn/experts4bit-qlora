#!/usr/bin/env python3
"""p60_reduce.py -- read P60 (bench/p60/P60-PREREG.md) from the run directory: replay.json + eids_b16.pt.json.

    python bench/p60/p60_reduce.py <run-dir> [--md out.md] [--json out.json]

  P0  instrument: the replay's served arm, `_gemv_int4_b32` kernel ms/step (profiler, median over profiled steps), within
      +-15 % of P57's census row 6.340 ms/step -- or NOTHING below is read
  P1  served - dedup (graph step ms, median over steps) >= 0.8 ms -> repeated rows are the cost (a grouped kernel's ceiling);
      refuted < 0.3
  P2  dedup / floor <= 1.15 (the kernel is near its byte roofline once each expert is read once); > 1.30 -> per-expert config
  P3  |sorted / served - 1| <= 0.03 (L2 already serves repeats; ordering rows by expert buys nothing)
  P4  mean distinct experts per layer per step within 58.7 +- 3 (P57 amendment 3's count; here teacher-forced)
stdlib only.
"""
from __future__ import annotations

import argparse
import json
import os

CENSUS_GEMV_MS = 6.340          # P57 B=16 control census, `_gemv_int4_b32` row (bench/p57/RESULTS-p57.md)
P57_DISTINCT = 58.67


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--md")
    ap.add_argument("--json")
    a = ap.parse_args()
    D = a.run_dir
    rp = os.path.join(D, "replay.json")
    ej = os.path.join(D, "eids_b16.pt.json")
    L = ["# Results -- P60: where the expert GEMV's B=16 headroom goes (real routing, replayed)", "",
         "Pre-registration: [`P60-PREREG.md`](P60-PREREG.md). Every number below is read from the run's receipts by `p60_reduce.py`.", ""]
    rep = {"verdicts": {}}
    if not os.path.exists(rp):
        L.append("`replay.json` MISSING -- nothing read.")
        print("\n".join(L))
        return
    r = json.load(open(rp))
    e = json.load(open(ej)) if os.path.exists(ej) else {}
    A = r["arms"]
    rep["replay"] = r
    rep["eids"] = {k: e.get(k) for k in ("mean_distinct", "min_layer_mean", "max_layer_mean", "steps", "layers", "eids_shape", "prompts_sha256")}
    L += [f"Box: {r['device']} ({r['sm_count']} SMs), torch {r['torch']}; copy bandwidth **{r['copy_bandwidth_gbs']:.0f} GB/s**; plans (sk) {r['plans']}; "
          f"{r['steps_replayed']} recorded steps replayed; eids {r['eids_shape']}.", "",
          "| arm | step ms (median) | mean | min | max | `_gemv_int4_b32` ms (profiler) | reduce ms |", "|---|---|---|---|---|---|---|"]
    for k in ("served", "sorted", "dedup"):
        x = A.get(k)
        if x:
            L.append(f"| {k} | {x['step_ms_median']:.3f} | {x['step_ms_mean']:.3f} | {x['step_ms_min']:.3f} | {x['step_ms_max']:.3f} | "
                     f"{x['gemv_kernel_ms_median'] if x['gemv_kernel_ms_median'] is None else round(x['gemv_kernel_ms_median'], 3)} | "
                     f"{x['reduce_kernel_ms_median'] if x['reduce_kernel_ms_median'] is None else round(x['reduce_kernel_ms_median'], 3)} |")
    f = A["floor"]
    L += [f"| floor (distinct bytes / measured bandwidth) | {f['step_ms_median']:.3f} | {f['step_ms_mean']:.3f} | | | | |",
          f"| floor at #564's 1528 GB/s | {f['at_1528_gbs_ms']:.3f} | | | | | |", "",
          f"Mean distinct experts per layer per step: **{f['mean_distinct']:.2f}** (recorder: {e.get('mean_distinct')}; layer means "
          f"{e.get('min_layer_mean')}..{e.get('max_layer_mean')}).", ""]
    v = rep["verdicts"]
    served_k = A["served"]["gemv_kernel_ms_median"]
    rel = served_k / CENSUS_GEMV_MS - 1 if served_k else None
    v["P0"] = "HOLDS" if rel is not None and abs(rel) <= 0.15 else "FAILED"
    v["P0_detail"] = f"served _gemv_int4_b32 {served_k} ms vs census {CENSUS_GEMV_MS} ({rel:+.1%})" if rel is not None else "no profiler row"
    if v["P0"] == "HOLDS":
        gap = A["served"]["step_ms_median"] - A["dedup"]["step_ms_median"]
        v["P1"] = "HOLDS" if gap >= 0.8 else ("REFUTED" if gap < 0.3 else "BETWEEN BANDS")
        v["P1_detail"] = f"served - dedup = {gap:.3f} ms/step (hold >= 0.8, refute < 0.3)"
        ratio = A["dedup"]["step_ms_median"] / f["step_ms_median"]
        v["P2"] = "HOLDS" if ratio <= 1.15 else ("REFUTED" if ratio > 1.30 else "BETWEEN BANDS")
        v["P2_detail"] = f"dedup / floor = {ratio:.3f} (hold <= 1.15, refute > 1.30)"
        s = A["sorted"]["step_ms_median"] / A["served"]["step_ms_median"] - 1
        v["P3"] = "HOLDS" if abs(s) <= 0.03 else "REFUTED"
        v["P3_detail"] = f"sorted / served - 1 = {s:+.2%} (hold within +-3 %)"
    md = f["mean_distinct"]
    v["P4"] = "HOLDS" if abs(md - P57_DISTINCT) <= 3 else "REFUTED"
    v["P4_detail"] = f"mean distinct {md:.2f} vs P57 {P57_DISTINCT} (+-3)"
    L += ["## Verdicts (pre-registered)", ""]
    for k in ("P0", "P1", "P2", "P3", "P4"):
        L.append(f"- **{k}: {v.get(k, 'NOT READ (P0 failed)')}** -- {v.get(k + '_detail', '')}")
    if v["P0"] != "HOLDS":
        dec = "P0 failed: the replay does not reproduce the served row -- nothing here licenses or refutes a kernel change."
    else:
        parts = []
        if v.get("P1") == "HOLDS":
            parts.append("P1 holds -> a grouped expert GEMV (one weight read per expert for all its rows) is licensed to BUILD (grouped-nf4-gemm lane K18, pre-registered before code)")
        if v.get("P3") == "REFUTED" and A["sorted"]["step_ms_median"] < A["served"]["step_ms_median"]:
            parts.append("P3 refuted in the favourable direction -> ordering rows by expert is a cheap lever: a consumer lane sorts rows once per layer and reads it in the step")
        if v.get("P2") == "REFUTED":
            parts.append("P2 refuted -> even one row per expert sits well above the floor: the per-expert kernel config is a lever of its own")
        dec = "; ".join(parts) or "No lever licensed: the headroom is not in repeated rows, row order or the per-expert config as measured here."
    L += ["", f"**Decision rule:** {dec}"]
    text = "\n".join(L)
    print(text)
    if a.md:
        open(a.md, "w").write(text + "\n")
    if a.json:
        json.dump(rep, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
