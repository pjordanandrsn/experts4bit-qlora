#!/usr/bin/env python3
"""bench/p54/p54_register_rows.py -- turn p54_reduce.py's JSON into claims-register rows (docs/claims.json).

    python bench/p54/p54_reduce.py <run-dir> --json rep.json
    python bench/p54/p54_register_rows.py rep.json --receipt-dir bench/p54/receipts --date 2026-09-21 [--merge docs/claims.json]

Rows (one per timed arm, medians of two draws, plus the distinct-expert read):
  e4b.serve.p54.qwen3.b16.control.5090.<date>   e4b.serve.p54.qwen3.b16.fqkv.5090.<date>
  e4b.serve.p54.qwen3.b1.control.5090.<date>    e4b.serve.p54.qwen3.b1.fqkv.5090.<date>
  e4b.serve.p54.qwen3.b16.distinct-experts.5090.<date>
Every row cites RESULTS-p54.md, P54-PREREG.md and the committed receipt files under --receipt-dir (the per-arm
step receipts and first-draw censuses copied from the fetched run, so `status: measured` is earned -- a receipt
public in this repo). The saving is stated as a same-box ratio; no cross-box number is quoted. `--merge` appends
the rows to an existing register (refusing to overwrite an id) and rewrites it with the same indentation.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

MODEL = "Qwen/Qwen3-30B-A3B"
HW = ("RTX 5090 (sm_120), one rented Vast.ai verified/secure host; the 5090 class carries ~8.5% inter-box "
      "dispersion, so only same-box ratios are quoted")
COND_COMMON = ("P42's census protocol (bench/p39/step_decomp.py): 512-token prompt, 128 generated, graph loop, 70 timed "
               "steps, fp8 paged KV, placement all-vram, --amort off; RTN int4 experts (E4B_SERVE_EXP_INT4=1) + "
               "uncalibrated int4 attention (E4B_SERVE_ATTN_INT4=1, 192 projections), K16 small-M route at its auto "
               "default; E4B_FUSE_T1_GLUE=1 E4B_FUSE_T1_GLUE_R2=1 E4B_FUSE_ROUTER_EPI=1; each timed arm drawn twice, "
               "interleaved (A B A B), median quoted; census (8 profiled replays after the timed window) on the first draw")


def rows(rep: dict, *, date: str, receipt_dir: str, e4b_sha: str, gnf4_sha: str) -> list[dict]:
    out = []
    ev_common = ["bench/p54/RESULTS-p54.md", "bench/p54/P54-PREREG.md"]
    for batch, tag in ((16, "b16"), (1, "b1")):
        pair = rep["pairs"][str(batch)] if str(batch) in rep["pairs"] else rep["pairs"][batch]
        ctl, fq = pair["control"], pair["fused"]
        for arm, d, flag in (("control", ctl, "--no-fuse-qkv"), ("fqkv", fq, "--fuse-qkv (Int4Linear.fuse, #651)")):
            if d["median"] is None:
                continue
            ms = d["median"]
            toks = batch * 1000.0 / ms
            arm_name = f"int4_{tag}" + ("" if arm == "control" else "_fqkv")
            ev = ev_common + [f"{receipt_dir}/e4b_b{batch}_{arm_name}.json", f"{receipt_dir}/e4b_b{batch}_{arm_name}_r2.json",
                              f"{receipt_dir}/logs/census_{arm_name}.txt"]
            row = {
                "id": f"e4b.serve.p54.qwen3.{tag}.{arm}.5090.{date}",
                "package": "experts4bit-qlora", "area": "serve",
                "claim": (f"Lane P54: {'batched (B=16, aggregate)' if batch == 16 else 'single-stream (B=1)'} paged decode of "
                          f"{MODEL}'s int4 serving stack on one RTX 5090 with q/k/v {'FUSED into one Int4Linear' if arm == 'fqkv' else 'as three projections'}: "
                          f"{ms:.3f} ms/step (median of two interleaved draws: {d['draws'][0]:.3f} / {d['draws'][1]:.3f}), "
                          f"{toks:.1f} tok/s on this box."),
                "value": round(ms, 3), "unit": "ms/step (median of two draws, same box)",
                "model": MODEL, "hardware": HW,
                "conditions": f"B={batch}, {flag}; {COND_COMMON}; e4b {e4b_sha[:12]}, grouped-nf4-gemm {gnf4_sha[:12]}",
                "measured_on": date, "status": "measured", "tier": "measured",
                "evidence": ev,
                "evidence_private": [f"receipts/experts4bit-qlora/{date}/p54-fqkv-1/ (receipt.json, teardown-proof.json, full fetched run)"],
                "notes": (f"Same-box pair: control {ctl['median']:.3f} vs fused {fq['median']:.3f} ms/step = saving "
                          f"{(ctl['median'] - fq['median']):.3f} ms ({(ctl['median'] - fq['median']) / ctl['median'] * 100:.1f} %); "
                          f"verdict per the registered band: {pair['verdict']}. Census (first draws): "
                          + (f"K16 small-M GEMM calls/step {pair.get('p3_calls')}; " if pair.get("p3_calls") else "")
                          + f"other families moved > 5 %: {pair.get('p3_others_moved_over_5pct') or 'none'}. "
                          "COMPARATOR: vs e4b's own control on the same box; no field engine on this row. No default changes from this row alone."),
            }
            out.append(row)
    ser = rep.get("series")
    if ser and ser.get("mean_distinct_over_layers") is not None:
        out.append({
            "id": f"e4b.serve.p54.qwen3.b16.distinct-experts.5090.{date}",
            "package": "experts4bit-qlora", "area": "serve",
            "claim": (f"Lane P54: at B=16 on the harness's natural prompts, {MODEL}'s router touches on average "
                      f"{ser['mean_distinct_over_layers']:.1f} distinct experts per layer per decode step "
                      f"(layer means {ser['min_layer_mean']:.1f}..{ser['max_layer_mean']:.1f} over {ser['layers']} layers; "
                      "uniform-random expectation 80.9 of 128) -- the count #564's expert-tier roofline depends on."),
            "value": round(ser["mean_distinct_over_layers"], 2), "unit": "distinct experts per layer per decode step (mean)",
            "model": MODEL, "hardware": HW,
            "conditions": f"B=16, --amort on --series-out (per-layer counters ON: this arm's step time is not quoted); {COND_COMMON}",
            "measured_on": date, "status": "measured", "tier": "measured",
            "evidence": ev_common + [f"{receipt_dir}/series_int4_b16.json.gz"],
            "evidence_private": [f"receipts/experts4bit-qlora/{date}/p54-fqkv-1/"],
            "notes": ("Read against #564: expert-tier byte roofline 5.336 ms at 64 distinct experts, 6.670 ms at 80, "
                      "against the measured _gemv_int4_b32 row of the same lane's census. A measurement with a soft prior "
                      "(<= 80 registered), not a claim about the kernel."),
        })
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("rep_json")
    ap.add_argument("--date", required=True)
    ap.add_argument("--receipt-dir", default="bench/p54/receipts")
    ap.add_argument("--e4b-sha", required=True)
    ap.add_argument("--gnf4-sha", required=True)
    ap.add_argument("--merge", help="docs/claims.json to append the rows to (refuses to overwrite an existing id)")
    a = ap.parse_args()
    rep = json.loads(Path(a.rep_json).read_text())
    new = rows(rep, date=a.date, receipt_dir=a.receipt_dir, e4b_sha=a.e4b_sha, gnf4_sha=a.gnf4_sha)
    if a.merge:
        reg = json.loads(Path(a.merge).read_text())
        have = {r["id"] for r in reg["claims"]}
        dup = [r["id"] for r in new if r["id"] in have]
        if dup:
            raise SystemExit(f"refusing to overwrite existing ids: {dup}")
        reg["claims"].extend(new)
        Path(a.merge).write_text(json.dumps(reg, indent=1, ensure_ascii=False) + "\n")
        print(f"appended {len(new)} rows to {a.merge}")
    else:
        print(json.dumps(new, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
