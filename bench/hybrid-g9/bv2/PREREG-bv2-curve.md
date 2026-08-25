# PREREG — BV2: the batch curve on the current shipped stack

Registered 2026-08-25, before any measurement. The B ∈ {1..16} curve
was last measured on the pre-collapse stack (b1, ~113–122 tok/s
aggregate at B=16). Everything shipped since applies at batch: the
compiled decoder-layer body (F1-B1), the fused KV append (F1-B2), the
collapse fast path, and the device-grouping machinery. BV2 re-measures
the curve on today's stack. This is a MEASUREMENT cycle — no
treatment, no bars to beat — so its calculator carries refusals and a
preregistered interpretation line only.

## Arms (one box, one session)

For B ∈ {1, 2, 4, 8, 16}: two runs each (A/A) through the production
scheduler path, standard window (prompt 512, 128 tokens, disjoint
per-row prompt offsets, offsets FIXED across arms so row 0's prompt is
identical everywhere). B=1 additionally runs the certified graph loop
(the 133.4 anchor) to disclose the eager-scheduler vs graph gap at
B=1; B>1 arms are eager-scheduler by construction (no B>1 graph
exists — building one would be a treatment, out of scope).

## Refusals (any one voids the affected arm or cycle)

1. **Determinism A/A**: the two runs at each B must be
   token-identical per row AND step-time spread < 5%.
2. **Cross-B identity**: row 0's prompt appears in every arm; its
   greedy continuation must be identical at every B and equal to the
   B=1 reference. A batch composition that changes tokens is a
   correctness failure, not a disclosure.
3. Anchor sanity: the B=1 graph arm within 3% of the certified
   7.39 ms class.
4. Trace degeneracy on row 0 (the check-traces law).

## Preregistered interpretation (fixed before numbers exist)

- The deliverable is the curve: per-B step time, aggregate tok/s
  (B × 1000/step), per-B decomposition as emitted by the harness.
- **The system-425 line**: single-stream 425 is REFUTED
  (RESULTS-s3). If aggregate throughput at any B ≤ 16 reaches
  **≥ 425 tok/s with every refusal clean**, the SYSTEM-throughput
  425 claim is certified — explicitly a different claim than the
  refuted single-stream one, and to be reported with that
  distinction in the same sentence, always.
- No bar moves after data; if the curve tops below 425, the campaign
  reports the measured maximum and stops claiming.

## Calculator

`bv2_verdict.py` (refusals + curve report + the preregistered
interpretation line), self-tested; receipts in `receipts-bv2/`.
