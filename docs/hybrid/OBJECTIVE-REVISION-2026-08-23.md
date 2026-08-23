# Objective revision — the phase-2 laws arrive from gnf4 (2026-08-23)

The gnf4 phase-2 gate program closed today, fully scored (gnf4 PRs
#203-#228; receipts in `bench/cold-engine/` there; every verdict from a
prereg frozen before its measurement). Three of its laws bind this
executor directly, and they revise the tier's objective. This document
registers the import, the revised objective, and the ordered falsifiable
predictions for the open gates — before any e4b measurement against them.

## The imported laws (receipts in gnf4)

* **L1 — one DRAM system (G1c, measured three times, last at G4').**
  CPU-tier GEMV reads, H2D fills, NVMe landings, and host memcpys share
  one memory system. Overlapping DRAM-crossing work with the DRAM-bound
  GEMV displaces its bandwidth ~1:1 plus an interleave penalty (κ ≈ 1.3
  measured; the G4' tribrid's overlap arm ran **18% slower than
  sequential** and above the serial sum of its alone-walls). Corollary:
  any "hide" measured under a light CPU load is an artifact — gnf4's
  E3b read hide = 1.0 at 132 GB/s load on a 212 GB/s system and it was
  true there and false at the tier's real 172 GB/s (RESULTS-p2-g1c).
* **L2 — the admission law (G2'').** Throttled residency admission
  takes `pairs / (frac · m)` steps, settle-inflated ×1.2-1.6 (measured
  across 16 traces). Any prefetch/promotion throttle trades per-step
  inflation against exactly this admission time.
* **L3 — the hit-mass crossover (G3'/G4').** Residency pays iff
  the CPU bytes it removes exceed the landing traffic it adds: measured
  0.56× no-cache steady wall on a heavy-reuse trace vs 1.44× on a
  thinner one at 0.7× capacity; 98% VRAM hit rate at steady state where
  it pays (RESULTS-p2-g3p, RESULTS-p2-g4p).

## The revised objective for this executor

> **Minimize bytes-through-DRAM per step.** Residency first (it deletes
> bytes); landings, fills, and prefetch H2D scheduled into **DRAM
> headroom** (GPU-bound phases, inter-step gaps, steps whose CPU work is
> not memory-bound); **never overlap DRAM-crossing work with the
> DRAM-bound GEMV for its own sake** — max(T_cpu, T_gpu, T_storage) was
> refuted as a wall model on shared-memory hardware; bytes-through-DRAM
> plus mixing is the wall.

## Registered predictions for the open gates, ordered

* **P1 (G8 B=16, the -14% wall): headroom-gating A/B.** The Phase-4
  prefetcher and KV/cold-landing copies currently overlap the CPU GEMV
  by design (ARCHITECTURE-NOTES: "the GPU overlaps... expert-streaming
  overlap discipline"). L1 predicts that overlap TAXES the B=16 CPU
  wall by ~(κ−1) × (overlapped DRAM-crossing bytes / B_dram). The A/B:
  identical serving step with copies gated out of the GEMV window
  (issued before it, after it, or on GPU-bound phases) vs the current
  always-overlap. Spoiler: a forced-overlap arm must show the tax, or
  the instrument is not measuring it. Counters for overlapped bytes
  already exist (prefetch/KV traffic); the prediction is quantitative
  before the run.
* **P2 (consistency, no box needed): thin routing on the crossover.**
  `offload_thin_uniq`'s measured gains re-derived against L3: thin
  calls are exactly the low-hit-mass side of the crossover; the model
  should reproduce the sign and rough size of the THIN8 deltas already
  in `b16close/RESULTS-b16.md`.
* **P3: any admission/prefetch throttle bounded by L2** — registered
  before tuning, so a deep throttle's convergence cost is priced,
  not discovered.
* **P4 (compatibility note): the B=16 kernel row-cost signature**
  (rows double → +80-178% at equal bytes) is call-structure, not DRAM —
  cache-blocking (b16close item 1) attacks a different term and remains
  the right kernel-side move alongside P1.

## Instrument laws 8-11 (appended to HANDOFF laws 1-7)

8. **Burst-clock gate for any GPU wall on rented hosts**: lazy-ramp
   hosts run decode-pattern launches at idle clocks (measured SM 180 MHz
   of 3,090; 162 µs vs 12-16 µs healthy). Probe = an exec-dominated
   workload (8.8 MB D2D copy) gap-vs-no-gap; pass iff
   gap ≤ max(3 × no-gap, no-gap + 45 µs); plus matmul20 vs the known
   healthy range. (Three screen iterations to get this right — the
   history and the wake-latency trap are in gnf4 PREREG-p2-g4p.)
9. **Blip-robust bars only** on shared cloud hosts: phase medians or
   per-step repeat medians — a 3× single-step bar over ~5 ms medians is
   a ~25 ms co-tenant-blip detector (gnf4 RESULTS-p2-g3p).
10. **Full warm-up inventory** (law 1 generalized): one untimed dry
    step exercising EVERY path — decode launch, fills, landings, CPU
    call — before any timed region; gnf4 lost two runs to residual
    one-time init landing in step 1.
11. **Calibrate at the operating point**: overlap/hide/bandwidth
    constants measured under a lighter load, a different thread count,
    or a different launch pattern than the tier's real one do not
    transfer (E3b's hide; the burst-clock collapse; both measured).

## Sequencing

P1 precedes further B=16 kernel work: it is the cheapest registered
−Δwall candidate (a scheduling change, no new kernels), its prediction
is quantitative from existing counters, and its spoiler is built in.
P2 costs one offline afternoon. P4 proceeds in gnf4 in parallel.
