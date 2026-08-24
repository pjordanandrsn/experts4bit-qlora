# RESULTS — T1b REFUTED (compile route closed); the probe redraws the ladder

Two registered outcomes from one rental (EPYC 9755 + RTX 5090; receipts
in [receipts-t1b/](receipts-t1b/)). Cycle ≈ $1.00, box destroyed, zero
instances.

## T1b: REFUTED — the compile route is closed for good

The step marking cured T1's crash (no RuntimeError; B0 PASS,
continuations token-identical) and cudagraphs still lost: other
86.4 vs eager ~81.6 ms, step 145.8 vs 141.2. With ~100 graph breaks the
partitioned graphs' replay-and-copy overhead replaces the guard tax
rather than removing it. Per the prereg: no further compile attempts;
T1c (manual op-count reduction) remains the launch-path route — and the
probe below now aims it precisely.

## The attribution probe (go/no-go scored per PREREG-t2t3-probes)

`kernels.txt`, active window 12/12. Per decode step:

| what | device ms/step | calls/step |
|---|---|---|
| `_gemm_nf4_grouped` (VRAM expert GEMV) | **13.3** | 96 (2/layer) |
| dense mm/cutlass (projections) | ~4.9 | 241 |
| `_fp8_paged_decode_split` (attention) | **0.79** | 48 |
| copies (`copy_` + DtoD) | ~6.5 | ~5,000 + 3,216 |
| everything else (index/sum/mul/…) | ~5 | thousands |
| **total real kernel time** | **≈ 17.4** | — |

* **T2 verdict: CLOSED — NO TARGET.** The attention kernel is 0.79
  ms/step. The "103 ms device attention" in the bucket receipts was
  CUDA-event WALL around host-heavy code (launch gaps), not kernel
  time — the same class of illusion as the hostbill cycle's "device-
  bound" reading, one level deeper. Bucket brackets measure elapsed,
  profilers measure occupancy; only the second names a kernel target.
* **T3 opens (concentrated):** one kernel, `_gemm_nf4_grouped`, owns
  41.9% of all CUDA time at ~400 GB/s achieved vs the 5090's ~1.8
  TB/s — 3–4× kernel headroom, gnf4-side work.
* **T5 opens (the mountain, replacing T1c's guesswork):** the step is
  ~17 ms of GPU work inside ~140 ms of host dispatch. The spray is
  attributed: ~5,000 `copy_`, ~2,400 `.to`, ~880 `index_select`, and
  **174 `aten::nonzero` per step — each a device→host sync** (the
  per-layer tier-split algebra in the hybrid dispatch). The amort
  instrument itself contributes 48 `unique2` + 48 radix sorts per step
  (identical across arms, so past A/Bs stand; the PRODUCTION step
  without instrumentation is faster than every number we have quoted).

## SPEC-425 budget, re-frozen

Target 37.6 ms/step. Device floor after T3 ≈ 8–10 ms; the remaining
~28 ms of budget must come out of ~120 ms of host dispatch — T5 is
where 425 lives or dies. Ladder forward: **T5** (dispatch-algebra
diet: eliminate nonzero syncs via cached index tensors that change
only on placement swaps, batch small copies, prune `.to` churn),
**T3** (GEMV kernel), **T4** (overlap) — in that order.
