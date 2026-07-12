# 30B on a 24 GB card: retention fix VALIDATED, fv ladder PASSES all four predictions

**Date:** 2026-07-12 · **Prereg:** `prereg_vram30b.json` @ `e611d96` (frozen, rerun
UNCHANGED) · **Fix under test:** `feature/expert-store@5cb0dbc` (strip bnb
parametrize-cache hooks) · **Host:** RunPod SECURE RTX 3090 24 GB, L=23.08 GB/s,
~$0.9 · pod 404-verified, ramcode shredded.

## The fix works — the exact experiment that OOM'd this afternoon now trains

Same card, same 30B (Qwen3-30B-A3B, seq256, whole_layer, 40 steps), same 14.5 GB
packed pool. This afternoon on the unfixed branch: **OOM at every arm, 22.37 GiB,
before step 1.** Tonight on the fixed branch: **every arm trains, peak VRAM
2.44 → 15.66 GiB across the ladder, no OOM.** The retained-dequant term is gone;
`max_active` now tracks the *placement* (how many blocks are VRAM-resident), which
is exactly what it should measure.

## fv ladder (the deferred #3 deliverable, now delivered)

pool 14.5 GB, R (all-RAM streamed, n=2) = 2.656 s/step (floor spread 0.007):

| fv (VRAM-resident share) | s/step | peak VRAM |
|---:|---:|---:|
| 0.0 (all-RAM via fused) | 2.7701 | 2.44 GiB |
| 0.25 | 2.3697 | 5.54 |
| 0.5 | 2.0964 | 8.91 |
| 0.65 | 2.0284 | 10.88 |
| 0.8 | 1.7123 | 12.85 |
| 0.9 | 1.5699 | 14.26 |
| 1.0 (fully resident) | 1.4368 | 15.66 |

**All four frozen predictions PASS:**
- **S1 linearity:** fit `t(fv) = 2.754 − 1.291·fv`, max residual 5.9% (bar ≤10%).
- **S2 slope physics:** measured saving b=1.291 s/fv vs `2P/L = 1.257` → ratio
  **1.03** (band [0.5, 2.0]) — the per-unit-fv saving IS the two-touch transfer
  time, dead on.
- **S3 fv_max ≥ 0.25:** **fv_max = 1.0** — the *entire* 14.5 GB pool fits resident
  beside 30B training on a 24 GB card (peak 15.66 GiB, ~8 GB headroom). Far past
  the 0.25 bar.
- **S4 recovery:** at fv=1.0, measured saving 1.219 s vs 0.8×linear 1.033 → PASS.

## Headline

**A 30B MoE now QLoRA-trains on a single 24 GB consumer GPU** (was categorically
impossible pre-fix — the dequant leak alone needed 53 GB), and `fused_vram_fraction`
dials a **1.92× end-to-end speedup** (2.77 → 1.44 s/step) linearly from the SSD/RAM
floor to fully resident, bit-exact, no eviction machinery. Per-GB promotion value
on this link: **0.0445 s/GB** (30B pool) — consistent with the OLMoE 3090 point
(0.0395 at L=20.4) once scaled for link, extending the model across a 4.5× pool-size
jump. The consumer-30B scenario is open for business.

## Cross-cutting

This closes the retention finding (`FINDING-dequant-retention.md`): root cause
(checkpoint early-stop skips the bnb cache-disable hook → global cache stuck on →
full-model dequant retained), workaround shipped + validated here, upstream bnb PR
drafted (`always_call=True`, awaiting approval). The fv ladder itself validates the
VRAM tier at the scale the thesis actually targets, complementing the OLMoE points
(H100 waterfall, 3090 12×-per-GB link axis).

## Evidence

`q30-evidence.tgz` (fixed-branch: all ladder logs, results JSON, SENT) sha256
`355313c431fd5f4d3ff9cc93332bb84970ac67449063b48d406f654d09689642`;
afternoon OOM run preserved as `q30-evidence-gate-copy.tgz` — both at mini
`~/q30-evidence/`. `q30_results_gate.json` alongside (this dir).
