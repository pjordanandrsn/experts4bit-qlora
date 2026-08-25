# PREREG — S1: speculative decoding, acceptance before any executor

Registered 2026-08-25, before any measurement. Basis: the F1 census
arithmetic (merged) — charging every non-matmul cost to zero and the
nf4 GEMV to its loads floor leaves ~3.8 ms/step ≈ 262 tok/s, so the
step-cheapening lane cannot reach 425. Emitting >1 token/step is the
only registered route, and its worth is decided by two numbers this
cycle measures BEFORE any executor is built: the drafter's acceptance
and the verify step's cost.

## The drafter under test: prompt-lookup (n-gram), cost zero

At each position, find the longest suffix of the visible context
(n from 3 down to `n_min`) that re-occurs earlier; propose the K
tokens that followed its most recent earlier occurrence; no match
proposes nothing. No draft model, no VRAM, no extra launches — if
THIS drafter's arithmetic clears the bar, S2 is an executor build
with zero new model risk; if it refutes, the draft-model variant
needs its own prereg with this cycle as baseline.

Registered grid: n_min ∈ {1, 2, 3} × K ∈ {4, 8, 16}.

## Stage A instruments (local, self-tested)

- `s1_draft_sim.py`: replays recorded GREEDY traces (prompt + emitted
  tokens); at every step simulates the drafter and scores the longest
  exact prefix match m ∈ [0, K] against what the target actually
  emitted. Greedy target + exact-prefix scoring IS speculative
  acceptance for a greedy pipeline — no distribution math enters.
  Emits per-(n_min, K): window count, E[m], E[tokens/step] = E[m]+1,
  and the match distribution.
- `--verify-probe K` in step_decomp: after prefill, time R repeats of
  a (K+1)-row forward at ~512-token past (chunked-median, eager).
  Eager INCLUDES host gaps, so it is an UPPER bound on verify cost —
  and the decision map uses it only on the side where conservatism is
  sound.
- `s1_verdict.py`, self-tested both directions.

## Stage A arms (one box, shipped config: compile + fused append)

1. **Traces**: 4 runs × `--batch 8` × 128 greedy tokens at 4 disjoint
   prompt offsets → 32 distinct traces (`gen_capture` per row).
2. **Verify-cost curve**: `--verify-probe K` for K ∈ {4, 8, 16} plus
   the B=1 anchor step.
3. Simulator over the traces → α table; verdict composes.

## Registered decision map

For each (n_min, K): predicted throughput
`T_pred = E[tokens/step] / verify_ms(K)` with two bounds —
conservative uses the measured eager `verify_ms(K)` (upper-bound
cost), optimistic uses `verify_ms(K) := step_ms(1)` (a (K+1)-row step
cannot cost less than a 1-row step).

- **GO** — max over the grid of conservative `T_pred` ≥ **1.5×** the
  anchor throughput ⇒ register S2 (the executor), bars tied to this
  prediction. Sound because the cost side is an upper bound.
- **REFUTED-FOR-DRAFTER** — max over the grid of OPTIMISTIC `T_pred`
  < **1.1×** ⇒ prompt-lookup cannot pay even with a free verify step;
  the draft-model variant needs a fresh prereg. Sound because the
  cost side is a lower bound.
- **MARGINAL** — otherwise: register S2-lite (graph-compatible
  executor for the single best cell only) with its own bars.

## Refusals

- Trace degeneracy (the check-traces law): any trace with < 30
  distinct tokens in 128, or a repetition loop (any 8-gram occurring
  > 6 times), REFUSES that trace; fewer than 24 surviving traces
  refuses the cycle.
- Window floor: α computed from < 3000 windows per cell refuses.
- Verify-curve sanity: verify_ms(K) < 0.9 × step_ms(1) for any K
  refuses (a bigger step cannot be cheaper; that is instrument error).
- A/A on the anchor as in F1 (frame-relative, gain-bar family n/a —
  this cycle ships no treatment, only a prediction).

## What this cycle does NOT claim

No executor exists at the end of S1. The deliverable is the measured
acceptance table, the measured verify curve, and a registered
prediction with sound-sided bounds. S2's bars will be written against
the prediction, so an executor that lands under it fails S2, not S1.
