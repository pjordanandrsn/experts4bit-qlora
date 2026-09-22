#!/usr/bin/env python3
"""bench/p59/p59_register_rows.py -- claims-register rows for lane P59 from p59_reduce.py's JSON.

    python bench/p59/p59_register_rows.py <p59_rep.json> --date 2026-09-22 --e4b-sha <sha> --gnf4-sha <sha> --instance <id> [--merge docs/claims.json]

Rows: the headline `e4b.serve.p59.qwen3.b16.fqkv-kl.5090.<date>` (KL(int4 || int4_fqkv) at B=16 with the verdicts), and one
anchor row per NF4 pair (`...nf4-int4-kl...`, `...nf4-fqkv-kl...`). The determinism and prefill controls ride in the notes.
stdlib only. `--merge` appends, refusing to overwrite an id.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

MODEL = "Qwen/Qwen3-30B-A3B"
COND = ("B=16, sixteen distinct 512-token wikitext-2 rows (step_decomp._k8_window, the P54/P57/P58 prompts), 384 prefilled as one forward, 128 positions "
        "decoded one token per forward with a carried KV cache, ground-truth tokens fed back (teacher forcing); KL(P_ref||P_test) fp64 over the full "
        "vocabulary, token-weighted over 16 x 128 positions (bench/kl_fidelity.py primitives); arms built by bench/p44/serve_stack.build_served_model "
        "under the P42 hook: int4 = RTN int4 experts (E4B_SERVE_EXP_INT4=1) + uncalibrated int4 attention (E4B_SERVE_ATTN_INT4=1, 192 projections), "
        "K16 route auto, glue r1/r2 + router epilogue, GNF4_GEMV_FUSED_REDUCE unset; int4_fqkv = the same with qkv_fuse.fuse_qkv (48 modules, "
        "Int4Linear 192 -> 96); nf4 = NF4 experts + bf16 attention, no folds; int4_aa = int4 rebuilt in a fresh process")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("rep_json")
    ap.add_argument("--date", required=True)
    ap.add_argument("--e4b-sha", required=True)
    ap.add_argument("--gnf4-sha", required=True)
    ap.add_argument("--instance", required=True)
    ap.add_argument("--receipt-dir", default="bench/p59/receipts")
    ap.add_argument("--merge")
    a = ap.parse_args()
    rep = json.load(open(a.rep_json))
    P, V = rep["pairs"], rep["verdicts"]
    hw = f"RTX 5090 (sm_120), one rented Vast.ai verified/secure host (instance {a.instance}); same box, same session, same prompt rows for every arm"
    ev = ["bench/p59/RESULTS-p59.md", "bench/p59/P59-PREREG.md", "bench/p59/kl_b16.py", f"{a.receipt_dir}/p59_rep.json",
          f"{a.receipt_dir}/out/int4/census.json", f"{a.receipt_dir}/out/int4_fqkv/census.json"]
    priv = [f"receipts/experts4bit-qlora/{a.date}/p59-5090-1/ (receipt.json, teardown-proof.json, full fetched run; the fp32 logits stayed on the box)"]
    common = {"package": "experts4bit-qlora", "area": "serve", "model": MODEL, "hardware": hw, "measured_on": a.date, "status": "measured", "tier": "measured",
              "conditions": f"{COND}; e4b {a.e4b_sha[:12]}, grouped-nf4-gemm {a.gnf4_sha[:12]} (K17 merge)", "evidence_private": priv}
    p1 = P["int4 || int4_fqkv (P1)"]
    p3 = P.get("int4 || int4_aa (P3, determinism)") or {}
    p4 = P.get("int4 || int4_fqkv, PREFILL-last (P4)") or {}
    rows = [dict(common, **{
        "id": f"e4b.serve.p59.qwen3.b16.fqkv-kl.5090.{a.date}",
        "claim": (f"Lane P59: at B=16 the fused-q/k/v int4 attention store changes {MODEL}'s served output distribution by KL(unfused || fused) = "
                  f"{p1['kl_mean']:.5f} nats/token mean (median {p1['kl_median']:.5f}, p95 {p1['kl_p95']:.5f}, max {p1['kl_max_per_token']:.4f}) with top-1 "
                  f"agreement {p1['top1_agreement']:.4f} over {p1['n_tokens_scored']} teacher-forced decode positions; the same stack rebuilt in a fresh process "
                  f"reads KL {'exactly 0' if p3.get('exactly_zero') else p3.get('kl_mean')} (determinism control) and the shared prefill path reads "
                  f"{p4.get('kl_mean', float('nan')):.2e} (prefill control). Verdicts: P1 {V.get('P1')}, P2 {V.get('P2')}, P3 {V.get('P3')}, P4 {V.get('P4')}, census {V.get('census')}."),
        "value": round(p1["kl_mean"], 6), "unit": "nats/token, KL(int4 unfused || int4 fused q/k/v), B=16, teacher-forced full-vocab fp64, token-weighted",
        "evidence": ev,
        "notes": (f"{V.get('P1_detail')}. {V.get('P2_detail')}. Pre-registered bands: hold <= 0.01 nats & top-1 >= 0.97; refuted > 0.03 or < 0.95; the "
                  "shipped KL-from-checkpoint bar (<= 0.10 nats, top-1 >= 0.93) is the outer limit. The anchor is the NF4 control, not a bf16 reference "
                  "(Qwen3-30B-A3B bf16 is 61 GB). Decision rule outcome recorded in RESULTS-p59.md."),
    })]
    for key, tag, what in (("nf4 || int4", "nf4-int4-kl", "unfused int4 stack"), ("nf4 || int4_fqkv", "nf4-fqkv-kl", "fused-q/k/v int4 stack")):
        s = P.get(key)
        if not s:
            continue
        rows.append(dict(common, **{
            "id": f"e4b.serve.p59.qwen3.b16.{tag}.5090.{a.date}",
            "claim": (f"Lane P59 anchor: KL(NF4 control || {what}) at B=16 = {s['kl_mean']:.5f} nats/token mean (p95 {s['kl_p95']:.5f}), top-1 "
                      f"{s['top1_agreement']:.4f}, over {s['n_tokens_scored']} teacher-forced decode positions -- the distance of the int4 serving stack from "
                      "the register's NF4 control on the same rows; reported, not gated."),
            "value": round(s["kl_mean"], 6), "unit": f"nats/token, KL(NF4 || {what}), B=16, teacher-forced full-vocab fp64",
            "evidence": ev + [f"{a.receipt_dir}/out/nf4/census.json"],
            "notes": "An anchor for P2 (the fusion must not move this distance by more than 0.005). Not a quality licence: the NF4 control is itself a quantised stack.",
        }))
    if a.merge:
        path = Path(a.merge)
        reg = json.loads(path.read_text())
        have = {r["id"] for r in reg["claims"]}
        for r in rows:
            if r["id"] in have:
                raise SystemExit(f"refusing to overwrite existing id {r['id']}")
            reg["claims"].append(r)
        path.write_text(json.dumps(reg, indent=2, ensure_ascii=False) + "\n")
        print(f"merged {len(rows)} rows into {path}")
    else:
        print(json.dumps(rows, indent=1)[:1200])
        print(f"... {len(rows)} rows")


if __name__ == "__main__":
    main()
