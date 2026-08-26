# RESULTS — SV2: speculation REFUTED again; composition NOT refuted.

## 250 tok/s stays OPEN, through exactly one door.

Measured 2026-08-26 under PREREG-sv2-device-census. Receipts in
`receipts-sv2/` (instance 48735368: RTX 5090, gnf4 `1384feb` + e4b
`d5e530f`; destroyed, vast zero, by-id and list both checked).

```
SV2 CENSUS: OK  (knob 6.462 ms, A/A 0.08%, census reconciles at 1.00x)
  K=8:  verify 35.34 ms / accept 2.39  = 14.79 ms per accepted token -> short of 4.00
  K=16: verify 36.52 ms / accept 2.948 = 12.39 ms per accepted token -> short of 4.00
  K=32: verify 39.44 ms / accept 3.447 = 11.44 ms per accepted token -> short of 4.00
SV2 SPEC ROUTE: REFUTED (no cell clears; the S3 negative stands on the current stack)
```

## Every gate passed before anything was read

| gate | result |
|---|---|
| A/A (knob-ON graph pair) | 6.46 / 6.46 ms, spread 0.08%, token streams identical, 0 recompiles |
| anchor health | knob point 6.462 ms, 0.2% from the certified 6.476 (SV1 K6-B) |
| census reconciliation | kineto device total = 1.00× the measured step, coverage 100.0% |
| verify cells | all three present, all above the single-row floor |

## Deliverable 1 — the budget of the SHIPPED 6.46 ms step

First kernel census of the captured replay itself (every prior table
profiled the eager path). 8 untimed replays, CUDA activity only,
parsed by `f1/step_budget.py`:

| slice | ms/step | launches/step | share |
|---|---|---|---|
| `_gemv_nf4_dotpad` (MoE experts) | 2.469 | 96 | 38.2% |
| cuBLAS `gemvx` (attn proj + lm_head) | 1.787 | 145 | 27.7% |
| `_fp8_paged_decode_split` (attention) | 0.520 | 48 | 8.0% |
| elementwise (copies, index-select, casts) | 0.938 | 692 | 14.5% |
| router (topk + sort) | 0.311 | 96 | 4.8% |
| norm/residual triton fusions + append | ~0.43 | ~200 | 6.7% |

Call-count note: gemvx at 145/step = 3 projections/layer + lm_head;
one projection per layer lands below the row cut or in another cuBLAS
symbol. Slice-level attribution carries that caveat; the class-level
figure (matmul 4.26 ms, 65.7%) does not.

## Deliverable 2 — the 250 composition frame, floored by receipts

Per-step traffic from the box's own arena index and config
(`sv2_bytes.json`): experts 8×48 rows × 2.654 MB = **1.019 GB**; attn
bf16 + router **1.837 GB**; lm_head **0.622 GB**; fp8 KV at seen=517
0.5 MB. Total **3.479 GB/step** → streaming floor **2.22 ms** at the
box's measured triad (1569.4 GB/s), **2.61 ms** at a realistic 85%.

| treatment (prereg list) | measured | stream floor | addressable ceiling |
|---|---|---|---|
| MoE GEMV round 2 | 2.469 ms | 0.649 ms | **1.82 ms** (kernel runs at 3.8× its floor) |
| attn-proj GEMV block | 1.787 ms | 1.567 ms | 0.22 ms (already 1.14× — near bandwidth-bound) |
| fp8-COMPUTE attention | 0.520 ms | — (compute, not stream) | ≤ 0.52, realistically ~0.15–0.2 |
| elementwise/norm/router fusion residue | 1.57 ms | — | realistically ~0.7–0.8 (a halving) |

**Addressable-slice sum: ~2.5–2.7 ms at realistic per-treatment
fractions (85%-of-triad GEMVs, half the fusion residue, a third of
attention); absolute ceiling 4.13 ms.** The registered bar is
2.48 ms. The sum clears it — **the composition route is NOT
refuted** — with the honest structure stated plainly: about two
thirds of the pool sits in ONE slice. `_gemv_nf4_dotpad` streams
1.019 GB in 2.469 ms — 3.8× its bandwidth floor. Dot-pad's certified
11% was an instruction-overhead win on a kernel that is still
compute/latency-bound, not bandwidth-bound; the attn-side gemvx
proves 1.14× is reachable on this box. A round-2 MoE GEMV that
reaches even 1.5× its floor recovers ~1.5 ms alone.

## The registered adjudication

Two disjoint routes, refuted independently:

- **Speculation: REFUTED, second time, on the current stack.** Best
  cell 11.44 ms per accepted token vs the 4.00 bar (2.9× short).
  F2 + dot-pad + bitwise device grouping moved verify only
  36.4→36.5 ms at K=16 (within noise of S3's box). The verify step's
  cost is per-window fixed structure, exactly as S3 found; no
  current-stack lever touches it.
- **Composition: NOT refuted.** The pre-commitment ("250 is
  REFUTED-AS-COMPOSED only if BOTH routes fall short") therefore
  leaves **250 OPEN via composition only**. This is a frame result,
  not a demonstration: nothing here proves a 3.8×-floor kernel can
  reach 1.5×. It licenses the next lane and names its target.

**Next lane (registered by this result): the MoE GEMV round-2
kernel** — target ≥1.5 ms off the 2.469 ms slice (tensor-core
mapping past dot-pad's M-row waste), with the fusion residue
(~0.7 ms) as the second seam. Together they put ~4.2 ms/step ≈ 240
tok/s in honest reach, with fp8-COMPUTE attention the stretch to
250. A round-2 prereg must carry its own quality gates (bitwise or
mechanism-derived tolerance) before any wall number is cited.

## Receipts notes

- The arms compose-step SUMMARY PRINT crashed on a stale key AFTER
  `sv2_report.json` was written (`SV2-COMPOSE-FAIL` in
  `sv2_progress`); the verdict ran on the intact report on-box and
  reproduces bit-identically from the pulled receipts. Cosmetic;
  recorded.
- Draft cost in the speculation cells is 0.0 ms (prompt-lookup, the
  S3 basis), recorded in the calculator.
- The bandwidth floors use the box's OWN triad measurement from the
  provision gate, not a spec sheet.
