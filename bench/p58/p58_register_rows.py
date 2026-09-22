#!/usr/bin/env python3
"""bench/p58/p58_register_rows.py -- claims-register rows for lane P58 from the fetched receipts (P37's row shape).

    python bench/p58/p58_register_rows.py <run-dir> --date 2026-09-22 --e4b-sha <sha> --gnf4-sha <sha> --instance <id> [--merge docs/claims.json]

Rows: one per VALID arm (`e4b.serve.h2h.p58.qwen3.5090.<date>.arm.<engine>.<b>.<arm>`), one headline per vLLM build
and batch (`e4b.serve.h2h.vllm-<version>.p58.qwen3.<b>.5090.<date>`: the ratio vLLM/e4b-int4 from the primary pair,
quoted only when p58_reduce.py's rules hold -- this script re-applies them: VALID arms, self-pairs inside 1.03x,
same prompts_sha), and one build-to-build row per batch when both builds are VALID. stdlib only; nothing licensed.
"""
from __future__ import annotations

import argparse
import glob
import importlib.util
import json
import os
from pathlib import Path

MODEL = "Qwen/Qwen3-30B-A3B"
GPTQ = "Qwen/Qwen3-30B-A3B-GPTQ-Int4 @ 9b534e4318b7ebc3c961a839f13eb18b1833f441"
COND_E4B = ("P54's int4 stack: RTN int4 experts (E4B_SERVE_EXP_INT4=1) + uncalibrated int4 attention (E4B_SERVE_ATTN_INT4=1, 192 "
            "projections), K16 small-M route auto, E4B_FUSE_T1_GLUE=1 E4B_FUSE_T1_GLUE_R2=1 E4B_FUSE_ROUTER_EPI=1, GNF4_GEMV_FUSED_REDUCE "
            "unset (two-launch default); --fuse-qkv at B=1 only; 512-token wikitext-2 prompts (step_decomp._k8_window), 128 generated, "
            "graph loop, 127 (B=1) / 70 (B=16) timed steps, fp8 paged KV, all-vram, --amort off")
COND_VLLM = ("offline LLM.generate, dtype auto (float16), max_model_len 2048, gpu_memory_utilization 0.90, prefix caching off, seed 0, "
             "default CUDA graphs (enforce_eager on the eager arm), kv auto; decode isolated by the slope method (32 -> 128 tokens, "
             "min-of-3), identical prompt token ids to the e4b arms (prompts_sha in every receipt)")


def load_reducer(run_dir: Path):
    src = run_dir / "p58_reduce.py"
    spec = importlib.util.spec_from_file_location("p58_reduce", src)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--date", required=True)
    ap.add_argument("--e4b-sha", required=True)
    ap.add_argument("--gnf4-sha", required=True)
    ap.add_argument("--instance", required=True)
    ap.add_argument("--receipt-dir", default="bench/p58/receipts")
    ap.add_argument("--merge")
    a = ap.parse_args()
    D = Path(a.run_dir)
    red = load_reducer(D)
    hw = f"RTX 5090 (sm_120), one rented Vast.ai verified/secure host (instance {a.instance}); same-box, same-session ratios only"
    ev_common = ["bench/p58/RESULTS-p58.md", "bench/p58/P58-PREREG.md"]
    priv = [f"receipts/experts4bit-qlora/{a.date}/p58-5090-1/ (receipt.json, teardown-proof.json, full fetched run)"]
    rows = []
    for B in (1, 16):
        pf = red.jload(D / f"prompts_b{B}.json")
        psha = pf.get("prompts_sha256")
        e4b, vl = {}, {vt: {} for vt in red.BUILDS}
        for p in sorted(glob.glob(str(D / f"e4b_b{B}_*.json"))):
            arm = os.path.basename(p)[len(f"e4b_b{B}_"):-5]
            d = red.jload(p)
            e4b[arm] = red.e4b_row(d, B, arm, str(D / "logs" / f"run_e4b_b{B}_{arm}.log"))
        for vt in red.BUILDS:
            for p in sorted(glob.glob(str(D / f"vllm_{vt}_b{B}_*.json"))):
                arm = os.path.basename(p)[len(f"vllm_{vt}_b{B}_"):-5]
                d = red.jload(p)
                tok, ms, st, why = red.vllm_row(d, arm, str(D / "logs" / f"run_vllm_{vt}_b{B}_{arm}.log"))
                if d.get("prompts_sha256") and psha and d.get("prompts_sha256") != psha:
                    st, why = "VOID", "prompts_sha differs"
                vl[vt][arm] = (tok, ms, st, why, d.get("vllm_version"))
        tag = f"b{B}"
        unit = "tok/s (decode; B=1 per stream, B=16 aggregate)"
        for arm, (tok, ms, st, why) in e4b.items():
            if not str(st).startswith("VALID"):
                continue
            rows.append({"id": f"e4b.serve.h2h.p58.qwen3.5090.{a.date}.arm.e4b.{tag}.{arm}", "package": "experts4bit-qlora", "area": "serve",
                         "claim": f"Lane P58 arm e4b/{arm} at B={B}: {tok:.1f} tok/s, {ms} ms/step (graph window) on this box.",
                         "value": round(tok, 1), "unit": unit, "model": MODEL, "hardware": hw,
                         "conditions": f"B={B}; {'NF4 control (no folds, no int4)' if arm.startswith('nf4') else COND_E4B}; e4b {a.e4b_sha[:12]}, grouped-nf4-gemm {a.gnf4_sha[:12]}",
                         "measured_on": a.date, "status": "measured", "tier": "measured",
                         "evidence": ev_common + [f"{a.receipt_dir}/e4b_b{B}_{arm}.json"], "evidence_private": priv,
                         "notes": "Per-arm row; the position is the headline ratio row. Same box, same session, identical prompt ids."})
        for vt in red.BUILDS:
            for arm, (tok, ms, st, why, ver) in vl[vt].items():
                if not str(st).startswith("VALID"):
                    continue
                rows.append({"id": f"e4b.serve.h2h.p58.qwen3.5090.{a.date}.arm.vllm-{ver}.{tag}.{arm}", "package": "experts4bit-qlora", "area": "serve",
                             "claim": f"Lane P58 arm vLLM {ver} ({vt} build)/{arm} at B={B}: {tok} tok/s, {ms} ms/step (slope) on this box, serving {GPTQ}.",
                             "value": tok, "unit": unit, "model": MODEL, "hardware": hw,
                             "conditions": f"B={B}; vLLM {ver}; {COND_VLLM}; {'enforce_eager=True' if arm == 'eager' else 'default CUDA graphs'}",
                             "measured_on": a.date, "status": "measured", "tier": "measured",
                             "evidence": ev_common + [f"{a.receipt_dir}/vllm_{vt}_b{B}_{arm}.json"], "evidence_private": priv,
                             "notes": f"Per-arm row ({st}). Quality quoted, never equated: GPTQ g128 vs e4b's RTN int4 -- different quantisers."})
        i4 = e4b.get("int4_r1")
        sp_i4 = red.pair(e4b.get("int4_r1", (None,))[0], e4b.get("int4_r2", (None,))[0])
        nf4 = e4b.get("nf4_r1")
        sp_nf4 = red.pair(e4b.get("nf4_r1", (None,))[0], e4b.get("nf4_r2", (None,))[0])
        for vt in red.BUILDS:
            v = vl[vt].get("graph_r1")
            sp_v = red.pair(vl[vt].get("graph_r1", (None,))[0], vl[vt].get("graph_r2", (None,))[0])
            ok = red.valid(v) and red.valid(i4) and all(x is not None and x <= red.SELF_PAIR for x in (sp_v, sp_i4, sp_nf4 or 1.0))
            if not ok:
                continue
            ratio = v[0] / i4[0]
            ver = v[4]
            rows.append({"id": f"e4b.serve.h2h.vllm-{ver}.p58.qwen3.{tag}.5090.{a.date}", "package": "experts4bit-qlora", "area": "serve",
                         "claim": (f"Lane P58 (pre-registered 2026-09-22): same box, same session, identical prompt token ids -- vLLM {ver} serving {GPTQ} "
                                   f"decodes at {v[0]:.1f} tok/s vs e4b's current int4 stack at {i4[0]:.1f} tok/s at B={B}: ratio {ratio:.3f} "
                                   f"({'vLLM ahead' if ratio > 1 else 'e4b ahead'}). e4b int4 / e4b NF4 control on this box: x{(i4[0] / nf4[0]):.3f}"
                                   + (f"; vLLM / e4b NF4: {v[0] / nf4[0]:.3f}" if red.valid(nf4) else "") + "."),
                         "value": round(ratio, 3), "unit": f"ratio vLLM {ver} / e4b int4 stack, decode tok/s at B={B} (>1 = vLLM ahead)",
                         "model": MODEL, "hardware": hw,
                         "conditions": f"B={B}; e4b: {COND_E4B}; vLLM {ver}: {COND_VLLM}; self-pairs e4b int4 {sp_i4:.4f}, e4b nf4 {sp_nf4 and round(sp_nf4, 4)}, vLLM {sp_v:.4f} (rule: inside 1.03x); prompts_sha {str(psha)[:16]}; e4b {a.e4b_sha[:12]}, grouped-nf4-gemm {a.gnf4_sha[:12]}",
                         "measured_on": a.date, "status": "measured", "tier": "measured",
                         "evidence": ev_common + [f"{a.receipt_dir}/vllm_{vt}_b{B}_graph_r1.json", f"{a.receipt_dir}/vllm_{vt}_b{B}_graph_r2.json",
                                                  f"{a.receipt_dir}/e4b_b{B}_int4_r1.json", f"{a.receipt_dir}/e4b_b{B}_int4_r2.json", f"{a.receipt_dir}/prompts_b{B}.json"],
                         "evidence_private": priv,
                         "notes": ("Same-box position; the vLLM number includes its serving loop and e4b's replay window has no scheduler, so the engine "
                                   "advantage is understated, not inflated. Quality quoted, never equated. e4b.serve.h2h.vllm-0.28.0.qwen3.5090.2026-09-05 "
                                   "stays as history (a different e4b stack)."),
                         "quoted_in": ["docs/SERVING-THROUGHPUT.md", "docs/STATUS.md"]})
        p_, s_ = vl["primary"].get("graph_r1"), vl["secondary"].get("graph_r1")
        sp_all = [red.pair(vl[vt].get("graph_r1", (None,))[0], vl[vt].get("graph_r2", (None,))[0]) for vt in red.BUILDS]
        if red.valid(p_) and red.valid(s_) and all(x is not None and x <= red.SELF_PAIR for x in sp_all):
            rows.append({"id": f"e4b.serve.h2h.p58.vllm-build-to-build.qwen3.{tag}.5090.{a.date}", "package": "experts4bit-qlora", "area": "serve",
                         "claim": f"Lane P58: vLLM {p_[4]} / vLLM {s_[4]} decode tok/s at B={B}, same box, same prompts, GPTQ-Int4 Marlin: {p_[0] / s_[0]:.3f} ({p_[0]} vs {s_[0]}).",
                         "value": round(p_[0] / s_[0], 3), "unit": f"ratio vLLM {p_[4]} / vLLM {s_[4]}, decode tok/s at B={B}",
                         "model": MODEL, "hardware": hw, "conditions": f"B={B}; both builds: {COND_VLLM}",
                         "measured_on": a.date, "status": "measured", "tier": "measured",
                         "evidence": ev_common + [f"{a.receipt_dir}/vllm_primary_b{B}_graph_r1.json", f"{a.receipt_dir}/vllm_secondary_b{B}_graph_r1.json"],
                         "evidence_private": priv, "notes": "Tells the next comparator lane whether pinning the vLLM version matters (P3 band +-5 %)."})
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
        print(json.dumps(rows, indent=1)[:1500])
        print(f"... {len(rows)} rows")


if __name__ == "__main__":
    main()
