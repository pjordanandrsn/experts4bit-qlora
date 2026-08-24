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

`kernels.txt`, active window 12/12. Per decode step, PER-KERNEL times
(each a single row's self-CUDA — never summed across rows, because the
profiler's aten-op rows and kernel-event rows overlap in attribution):

| kernel / event | device ms/step | launches/step |
|---|---|---|
| `_gemm_nf4_grouped` (VRAM expert GEMV) | **13.3** | 96 (2/layer) |
| cutlass bf16 wmma (dense projections) | 2.2 | 240 |
| Memcpy DtoD | 2.5 | 3,216 |
| `_fp8_paged_decode_split` (attention) | **0.79** | 48 |

Total device occupancy lies between the ProfilerStep tree total
(17.4 ms/step) and the all-rows self-CUDA sum (~32 ms/step, which
double-counts aten-op rows against their kernel events); the honest
statement is a BOUND: **device work ≤ ~32 ms of the 141 ms step, so
host dispatch is ≥ ~110 ms** — and that bound is the finding. The
host-side spray, by call count per step: ~5,000 `copy_`, ~2,400
`.to`, ~880 `index_select`, ~540 `index`, and **174 `aten::nonzero` —
each a device→host sync** (the per-layer tier-split algebra). The
amort instrument adds 48 `unique2` + 48 radix sorts per step
(identical across arms, so every A/B stands; the un-instrumented
production step is faster than any number quoted in this campaign).

* **T2 verdict: CLOSED — NO TARGET.** The attention kernel is 0.79
  ms/step. The "103 ms device attention" bucket was CUDA-event WALL
  around host-heavy code (launch gaps), not kernel occupancy — the
  hostbill cycle's illusion one level deeper: brackets measure
  elapsed, profilers measure kernels; only the second names a target.
* **T3 opens (concentrated):** `_gemm_nf4_grouped` is the largest
  single kernel by an order of magnitude — 13.3 ms/step at ~400 GB/s
  achieved vs the 5090's ~1.8 TB/s, 3–4× kernel headroom, gnf4-side.
* **T5 opens (the mountain, replacing T1c's guesswork):** ≥ ~110 ms
  of host dispatch against ≤ ~32 ms of device work; the spray and its
  sync points are attributed above.

## SPEC-425 budget, re-frozen

Target 37.6 ms/step. Device floor after T3 ≈ 8–10 ms; the remaining
~28 ms of budget must come out of ~120 ms of host dispatch — T5 is
where 425 lives or dies. Ladder forward: **T5** (dispatch-algebra
diet: eliminate nonzero syncs via cached index tensors that change
only on placement swaps, batch small copies, prune `.to` churn),
**T3** (GEMV kernel), **T4** (overlap) — in that order.
