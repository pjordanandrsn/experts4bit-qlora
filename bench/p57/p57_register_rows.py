#!/usr/bin/env python3
"""bench/p57/p57_register_rows.py -- turn p57_reduce.py's JSON into claims-register rows (docs/claims.json).

    python bench/p57/p57_reduce.py <run-dir> --json rep.json
    python bench/p57/p57_register_rows.py rep.json --date 2026-09-22 --e4b-sha <sha> --gnf4-sha <sha> [--merge docs/claims.json]

Rows (one per timed arm, medians of two draws; same-box ratios only):
  e4b.serve.p57.qwen3.b1.control.5090.<date>        e4b.serve.p57.qwen3.b1.fr.5090.<date>
  e4b.serve.p57.qwen3.b16.control.5090.<date>       e4b.serve.p57.qwen3.b16.fr.5090.<date>
  e4b.serve.p57.qwen3.b16.nor2_control.5090.<date>  e4b.serve.p57.qwen3.b16.nor2_fqkv.5090.<date>
No P4 row: the distinct-expert arm produced no measurement (amendment 1). `--merge` appends, refusing to overwrite an id.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

MODEL = "Qwen/Qwen3-30B-A3B"
HW = ("RTX 5090 (sm_120), one rented Vast.ai verified/secure host (instance 52066091, AMD EPYC); the 5090 class carries "
      "~8.5% inter-box dispersion, so only same-box ratios are quoted")
COND_COMMON = ("P42's census protocol (bench/p39/step_decomp.py): 512-token prompt, 128 generated, graph loop, 127 (B=1) / 70 (B=16) "
               "timed steps, fp8 paged KV, placement all-vram, --amort off; RTN int4 experts (E4B_SERVE_EXP_INT4=1) + uncalibrated "
               "int4 attention (E4B_SERVE_ATTN_INT4=1, 192 projections), K16 small-M route at its auto default; E4B_FUSE_T1_GLUE=1 "
               "E4B_FUSE_ROUTER_EPI=1; each timed arm drawn twice, interleaved (A B A B), median quoted; census (8 profiled replays "
               "after the timed window) on the first draw")
PAIRS = {
    "b1_fr":    (1,  "b1",  "control", "fr",           "--fuse-qkv E4B_FUSE_T1_GLUE_R2=1 GNF4_GEMV_FUSED_REDUCE=0", "--fuse-qkv E4B_FUSE_T1_GLUE_R2=1 GNF4_GEMV_FUSED_REDUCE=1 (K17's fused split-K reduce)"),
    "b16_fr":   (16, "b16", "control", "fr",           "--no-fuse-qkv E4B_FUSE_T1_GLUE_R2=1 GNF4_GEMV_FUSED_REDUCE=0", "--no-fuse-qkv E4B_FUSE_T1_GLUE_R2=1 GNF4_GEMV_FUSED_REDUCE=1 (K17's fused split-K reduce)"),
    "b16_nor2": (16, "b16", "nor2_control", "nor2_fqkv", "--no-fuse-qkv E4B_FUSE_T1_GLUE_R2=0 (round-2 glue OFF)", "--fuse-qkv (Int4Linear.fuse, #651) E4B_FUSE_T1_GLUE_R2=0 (round-2 glue OFF)"),
}
WHAT = {
    "b1_fr": "K17's fused split-K reduce OFF vs ON at B=1 on the fused-q/k/v int4 stack",
    "b16_fr": "K17's fused split-K reduce OFF vs ON at B=16 on the unfused int4 stack",
    "b16_nor2": "P54's B=16 divergence split: unfused vs fused q/k/v with the round-2 glue OFF on both legs",
}


def rows(rep: dict, *, date: str, receipt_dir: str, e4b_sha: str, gnf4_sha: str) -> list[dict]:
    out = []
    ev_common = ["bench/p57/RESULTS-p57.md", "bench/p57/P57-PREREG.md"]
    for key, (batch, tag, ctl_name, fu_name, ctl_flags, fu_flags) in PAIRS.items():
        pair = rep["pairs"][key]
        ctl, fu = pair["control"], pair["fused"]
        arms = pair["arms"]  # [control_arm, fused_arm] receipt stems
        tokens = pair.get("tokens") or {}
        d1 = tokens.get("draw1_control_vs_fused") or {}
        tok_summary = (f"draw 1 {d1.get('identical')}/{d1.get('sequences')} sequences identical"
                       + ("" if d1.get("all_identical") else f", first divergences {d1.get('first_divergences')}"))
        for arm, d, flags, stem in ((ctl_name, ctl, ctl_flags, arms[0]), (fu_name, fu, fu_flags, arms[1])):
            if d["median"] is None:
                continue
            ms = d["median"]
            toks = batch * 1000.0 / ms
            ev = ev_common + [f"{receipt_dir}/e4b_b{batch}_{stem}.json", f"{receipt_dir}/e4b_b{batch}_{stem}_r2.json",
                              f"{receipt_dir}/logs/census_{stem}.txt"]
            out.append({
                "id": f"e4b.serve.p57.qwen3.{tag}.{arm}.5090.{date}",
                "package": "experts4bit-qlora", "area": "serve",
                "claim": (f"Lane P57 ({WHAT[key]}): {'batched (B=16, aggregate)' if batch == 16 else 'single-stream (B=1)'} paged decode of "
                          f"{MODEL}'s int4 serving stack on one RTX 5090, arm `{stem}` ({flags}): {ms:.3f} ms/step "
                          f"(median of two interleaved draws: {d['draws'][0]:.3f} / {d['draws'][1]:.3f}), {toks:.1f} tok/s on this box."),
                "value": round(ms, 3), "unit": "ms/step (median of two draws, same box)",
                "model": MODEL, "hardware": HW,
                "conditions": f"B={batch}, {flags}; {COND_COMMON}; e4b {e4b_sha[:12]}, grouped-nf4-gemm {gnf4_sha[:12]} (K17 merge)",
                "measured_on": date, "status": "measured", "tier": "measured",
                "evidence": ev,
                "evidence_private": [f"receipts/experts4bit-qlora/{date}/p57-5090-2/ (receipt.json, teardown-proof.json, full fetched run)"],
                "notes": (f"Same-box pair: control {ctl['median']:.3f} vs {fu_name} {fu['median']:.3f} ms/step = saving "
                          f"{(ctl['median'] - fu['median']):.3f} ms ({(ctl['median'] - fu['median']) / ctl['median'] * 100:.1f} %); "
                          f"verdict per the registered rule: {pair['verdict']}. Census (first draws): target family calls/step "
                          f"{pair.get('p3_calls')}; other families moved > 5 %: {pair.get('p3_others_moved_over_5pct') or 'none'}. "
                          f"Tokens fused-vs-control: {tok_summary}. "
                          "COMPARATOR: vs e4b's own control on the same box; no field engine on this row. No default changes from this row."),
            })
    return out


def distinct_row(d: dict, *, date: str, receipt_dir: str, e4b_sha: str, gnf4_sha: str) -> dict:
    """The P4 row from p57d's on-device count (P57 amendment 3)."""
    pl = d["per_layer"]
    means = [x["mean_distinct"] for x in pl]
    never = [sum(1 for f in x["touched_frac"] if f == 0) for x in pl]
    m, u = d["mean_distinct_over_layers"], d["uniform_random_expectation"]
    per_us = 5.336 / 64 * 1000
    return {
        "id": f"e4b.serve.p57.qwen3.b16.distinct-experts.5090.{date}",
        "package": "experts4bit-qlora", "area": "serve",
        "claim": (f"Lane P57 (amendment 3, run p57d-5090-1): at B=16 on the harness's wikitext-2 prompts, {MODEL}'s router touches on "
                  f"average {m:.1f} distinct experts per layer per decode step (layer means {min(means):.1f}..{max(means):.1f} over "
                  f"{d['layers']} layers, {d['steps']} decode steps counted on device; uniform-random expectation {u:.1f} of 128); "
                  f"{min(never)}-{max(never)} experts per layer were never touched. Against #564's expert-tier byte roofline "
                  f"({per_us:.1f} us per distinct expert per layer-step: 5.336 ms at 64, 6.670 at 80) the floor at {m:.1f} is "
                  f"{5.336 / 64 * m:.2f} ms/step, vs the same lane's measured _gemv_int4_b32 row of 6.34 ms."),
        "value": round(m, 2), "unit": "distinct experts per layer per decode step (mean over layers and steps)",
        "model": MODEL, "hardware": HW,
        "conditions": (f"B=16, --amort off, E4B_FUSE_ROUTER_EPI=0 (the router module is called), --no-fuse-qkv, --gen-tokens 128; the count is an "
                       f"on-device forward-hook accumulation on every MoE router (bench/p57/distinct_experts.py v2, capture-safe: every eager and "
                       f"graph-replayed decode call counted; prefill chunks excluded by row count); this arm's step time is not quoted; {COND_COMMON}; "
                       f"e4b {e4b_sha[:12]}, grouped-nf4-gemm {gnf4_sha[:12]}"),
        "measured_on": date, "status": "measured", "tier": "measured",
        "evidence": ["bench/p57/RESULTS-p57.md", "bench/p57/P57-PREREG.md", "bench/p57/distinct_experts.py",
                     f"{receipt_dir}/p57d/distinct_experts_b16.json", f"{receipt_dir}/p57d/summary.txt"],
        "evidence_private": [f"receipts/experts4bit-qlora/{date}/p57d-5090-1/"],
        "notes": ("Read against #564: a measurement with a soft prior (<= 80 registered; HOLDS). Two earlier attempts (p57b, p57c) counted 3 "
                  "warm-up steps because the first counter's torch.unique synchronised under CUDA-graph capture -- recorded in RESULTS-p57.md, "
                  "not registered. The per-expert touch fractions (touched_frac) carry the skew the mean hides."),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("rep_json")
    ap.add_argument("--date", required=True)
    ap.add_argument("--receipt-dir", default="bench/p57/receipts")
    ap.add_argument("--e4b-sha", required=True)
    ap.add_argument("--gnf4-sha", required=True)
    ap.add_argument("--merge", help="docs/claims.json to append the rows to (refuses to overwrite an existing id)")
    ap.add_argument("--distinct", help="p57d's distinct_experts_b16.json (amendment 3); adds the P4 row")
    a = ap.parse_args()
    rep = json.load(open(a.rep_json))
    new = rows(rep, date=a.date, receipt_dir=a.receipt_dir, e4b_sha=a.e4b_sha, gnf4_sha=a.gnf4_sha)
    if a.distinct:
        new.append(distinct_row(json.load(open(a.distinct)), date=a.date, receipt_dir=a.receipt_dir, e4b_sha=a.e4b_sha, gnf4_sha=a.gnf4_sha))
    if a.merge:
        path = Path(a.merge)
        reg = json.loads(path.read_text())
        claims = reg["claims"] if isinstance(reg, dict) else reg
        have = {r["id"] for r in claims}
        for r in new:
            if r["id"] in have:
                raise SystemExit(f"refusing to overwrite existing id {r['id']}")
            claims.append(r)
        path.write_text(json.dumps(reg, indent=2, ensure_ascii=False) + "\n")
        print(f"merged {len(new)} rows into {path}")
    else:
        print(json.dumps(new, indent=2))


if __name__ == "__main__":
    main()
