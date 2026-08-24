# PREREG — B=1 single-stream decomposition (owner directive, 2026-08-24)

Registered before any measurement. A separate campaign from
T5b/SPEC-425 (which continues unaltered). **This cycle contains no
optimization work**: the deliverable is the scaling curve, the
three-arm causal decomposition, profiler attribution, and an explicit
next-branch decision. "Why is B=1 slow" becomes a measured boundary,
not a hunt.

## Question

Single-stream (B=1) decode of Qwen3-30B-A3B under the hybrid engine is
dramatically slower than resident specialized runtimes. How much of the
deficit comes from (a) heterogeneous residency, (b) hybrid-engine
orchestration, (c) ordinary framework/host dispatch, (d) the actual
M=1 kernels?

## Stage 1 — B scaling curve (one box, current main)

`step_decomp` at B ∈ {1, 2, 4, 8, 16}; same model, same arena (same
packed NF4 bytes), same context shape (prompt 512, gen 128, chunk
512), same placement policy (`--vram-gb 10 --dram-gb 60`, solver as
shipped), `--amort off` for every timed run. Full pinned command line
in `receipts-b1/CMDLINES.md`. Recorded per B: median ms/step,
aggregate tok/s (B·1000/step), per-stream tok/s (1000/step), and —
from a separate amort-ON + `--series-out` run at the same seed, valid
because greedy decode is deterministic and token identity between the
timed and stats runs is ASSERTED — routing statistics (uniques/tier
per layer, activation counts). The B=16 point must pass the GS
shape-gate (step ∈ [115, 165] ms, attn ≤ 55, dram ≤ 25) — it anchors
this box to the certified operating point. No cross-box absolute
comparisons anywhere in this campaign.

## Stage 2 — three B=1 arms, same packed NF4 base

- **H (hybrid, production)**: solver placement, production serving
  path — exactly the Stage-1 B=1 point.
- **R0 (all-VRAM hybrid)**: `--placement-override all-vram` moves
  every expert into the VRAM tier AFTER the solver runs, leaving the
  hybrid executor/wrapper machinery intact (tier splitting, cold-path
  checks, arena/view/pool constructed as usual). Isolates the cost of
  PHYSICAL heterogeneity while holding the executor constant. The rep
  must record `manifest_counts` = {vram: all, dram: 0, nvme: 0}.
- **R1 (direct resident)**: `--engine pipelined` —
  `enable_pipelined_residency` with every expert hot, the narrowest
  existing grouped-NF4 resident path (resident stacks, one
  address-dispatched gather, device-id GEMV; no hybrid tier, no CPU
  tier, no DRAM/NVMe bookkeeping, no per-layer placement decisions).
  Same paged-attention/KV stack. Known asymmetry, disclosed: its
  prefill (T>1) runs the saved reference forward; endpoints are
  decode-only. Mechanical risk, registered with fallback: if the
  pipelined engine cannot serve this operating point (e.g. prefill
  path refuses under the streaming loader), R1 falls back to
  `enable_hot_residency` on a materialized (non-arena) load with
  packed-byte equality asserted against the arena on sampled experts —
  the deviation would be disclosed in RESULTS.

Each arm: two timed runs (per-arm A/A), one profile run
(`--host-brackets --region-ops-out` + kernel table), H additionally
one amort-ON `--series-out` run for routing/locality.

## Instruments (this PR; reuse of step_decomp, no parallel bench stack)

- `--engine {hybrid,pipelined}` (R1), `--placement-override all-vram`
  (R0), `manifest_counts` + `engine` recorded in every rep.
- A `moe_block` region (each decoder layer's sparse-MoE block, i.e.
  router + experts) beside the existing `moe` (experts-only) region;
  `router_topk_host ≈ moe_block − moe` lands in the rep.
- Launch/sync/copy accounting: the sync-attr op table gains
  `cudaLaunchKernel`, `cudaMemcpyAsync`, `cudaStreamSynchronize`,
  `cudaDeviceSynchronize` rows.
- `b1_locality.py`: from the H series — per-layer token-to-token
  Jaccard and repeat probability of the routed set (the evidence any
  future prefetch/prediction must clear before being registered).
- `b1_verdict.py`: self-tested calculator computing the gaps and
  emitting the preregistered branch decision.

## Gates

- **G0**: per-arm A/A spread < 7.5%.
- **GS**: the B=16 anchor passes the certified band (box validity).
- **G1 (identity)**: R0 vs R1 tokens BIT-IDENTICAL (both all-GPU, same
  kernels). H tokens recorded against R0: divergence is EXPECTED (the
  documented CPU-vs-GPU cross-placement rounding law) — record
  agreement length and first-divergence index; a divergence is a
  disclosure, not a failure. The timed-vs-stats run token identity per
  arm is asserted (else the routing stats are not the timed run's).

## Preregistered interpretation (gap = relative step-time difference at B=1)

"≪" threshold: ≥ 25% of the slower side's step. With
H, R0, R1 the three median steps:

1. **H ≪ R0 dominant** (residency tax): the hybrid B=1 lane becomes a
   latency-hiding problem — placement for hit probability, earlier
   dispatch, GPU/CPU overlap, and prefetch ONLY if the locality trace
   licenses it.
2. **R0 ≪ R1 dominant** (abstraction tax): implement the all-resident
   collapse fast path — an all-VRAM placement must not execute tier
   splitting, CPU/NVMe bookkeeping, joins, or placement decisions in
   the token-critical path.
3. **H ≈ R0 ≈ R1, all slow** (M=1 executor/kernel problem): open the
   resident-B1 ladder — M=1/GEMV-specialized packed kernel,
   device-resident routing metadata, launch collapse; only then
   consider a narrow native decode executor.
4. **R1 fast, H slow**: two operating regimes, both recorded —
   competitive resident B=1 when the model fits; measured degradation
   as the model crosses VRAM. Not a failure.

Mixed outcomes compose (e.g. both gaps ≥ 25%: both branches open,
largest first). "Fast/slow" for case 3-vs-4 is anchored INTERNALLY:
R1 counts as "fast" if its host wall ≤ 2× its real kernel occupancy
(profiler self-CUDA), "slow" otherwise — no cross-runtime absolute
numbers enter the verdict.

## Native-code rule (registered)

No C++/native work is licensed by this cycle. If (and only if) the
FOLLOW-UP structural collapse on the evidence-selected branch still
leaves the resident direct path spending multiples of its CUDA
occupancy in host/framework work, the next registration may propose
the smallest native boundary that owns the per-token layer walk while
reusing existing kernels — never a full replacement runtime without
that measurement.

## Deliverables

- PR 1 (this): prereg + instruments + calculators, no measured claims.
- PR 2 (RESULTS): scaling curve, H/R0/R1 decomposition, attribution
  tables, locality trace, and the branch decision — no optimization.
