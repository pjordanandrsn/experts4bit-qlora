# PREREG — SV2: the graphed-step device census (the 250 frame)

Registered 2026-08-26, before measurement. Target under test
(Jordan; set after 275's refutation): **~250 tok/s single-stream** =
4.00 ms/step — a ~1.6× DEVICE-COMPUTE cut from the certified knob
point (6.48 ms). No orchestration lever can reach it (SV1 addendum:
the host is already evicted); the question is whether kernel-level
slices sum to ≥2.48 ms.

## Deliverables (instruments only; no speed bars)

1. **Graphed-step kernel census**: profile the SHIPPED captured
   replay (the b1d graph loop, knob ON) — kineto records kernels
   inside replays — and produce the per-kernel budget of the 6.48 ms
   step. The prior tables profiled the eager path; the graphed step
   is the shipped thing (score-the-shipped-thing).
2. **The 250 frame**: from that budget, the addressable-slice sum
   over the candidate treatments, each with its mechanism named:
   - attn-proj GEMV block (batched-head / persistent-kernel form),
   - MoE GEMV round 2 (tensor-core mapping beyond dot-pad's 15/16
     M-row waste),
   - fp8-COMPUTE attention (built, sm_89+ gated, quality never
     certified — the 5090 qualifies),
   - elementwise/norm/router fusion residue.
3. **S4 go/no-go**: recompute speculative-verify economics on the
   CERTIFIED batched-graph machinery (the S3 refutation predates F2,
   BV3, and bitwise device grouping). The cost sweep and the
   acceptance data must use MATCHED K (review, e4b#274): sweep
   K ∈ {8, 16, 32} rows and pair each cell with S1's acceptance AT
   THAT K (2.39 at K=8; 2.948 at K=16; 3.447 at K=32). Effective
   per-accepted-token cost at each K =
   (verify_step_ms(K) + draft_cost_ms(K)) / accept(K); S4 registers
   only if some cell's arithmetic clears 4.00 ms; the S3 negative
   stands for the old stack either way.

## Pre-commitment (two DISJOINT routes; units kept separate —
review, e4b#274)

250 has two candidate routes, refuted independently in their own
units:
- **Composition route**: the addressable-slice sum (milliseconds of
  device-compute cut) must reach ≥2.48 ms off the 6.48 ms knob
  point. Short of that, 250-by-composition is refuted.
- **Speculation route**: some S4 cell's effective per-accepted-token
  cost must clear ≤4.00 ms. No cell clearing, 250-by-speculation is
  refuted.
**250 itself is REFUTED-AS-COMPOSED only if BOTH routes fall
short** — and the RESULTS must say so rather than stretching either
side's estimates. The census cannot fail a bar — only REFUSE on its
own gates (profiler coverage, A/A, anchor health).

## Arms

One anchor-gated box: knob-ON graph loop ×2 (A/A) + the profiled
replay window + the K-row verify-step cost sweep (K ∈ {8, 16, 32}
rows through the batched graph path — matched to S1's acceptance
grid). Receipts under `bench/hybrid-g9/sv2/receipts-sv2/`.
