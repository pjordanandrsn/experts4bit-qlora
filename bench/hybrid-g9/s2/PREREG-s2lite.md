# PREREG — S2-lite: the graphed verify step for cell n1_k16

Registered 2026-08-25, before any Stage A measurement. Basis:
RESULTS-s1-acceptance (MARGINAL) — acceptance 2.948 tokens/step at
n_min=1, K=16 is established with receipts; the one number S1 refused
to fabricate is the cost of a GRAPHED verify step. S2-lite exists to
measure it, gate its correctness, and only then build the loop.

## Mechanism (no new triton kernel)

The paged decode read already consumes a per-row `(block_table, lens)`
pair. Verification maps **K+1 query rows to ONE slot** with a
staggered `lens_override` (`base+1 .. base+K+1`): row i attends over
the past plus draft tokens 0..i. Causality is enforced by lengths over
already-appended K/V. Rejected tokens are made unreadable by
`kv.rewind(slot, n)` — reads are governed entirely by lengths, so
rewind is a host-mirror update plus one `fill_` per layer, no data
movement. The MoE side of a K+1-row step routes through the M-tile
prefill path (compiles on the floor stack — K5).

## Stage A — gate, then cost (this cycle)

1. **Bitwise gate (eager)**: run K+1 sequential T=1 greedy steps plus
   4 continuation steps (the oracle); rewind; run ONE verify-mode step
   fed the oracle window as its draft. Every verified argmax must
   equal its sequential counterpart, AND the 4-step continuation after
   the verify world must equal the sequential world's. Any mismatch
   REFUSES the cycle — no timing is read.
2. **Graphed cost**: capture the verify step at fixed position (write
   addresses bake at capture — legal when every replay rewinds to the
   same position; the inter-replay rewind's ~48 `fill_` launches sit
   inside the timed span, on the conservative side). Report
   `verify_graph_ms` for K=16, plus the B=1 anchor with A/A.

## Registered decision map (Stage A)

Predicted executor throughput `T_pred = 2.948 / verify_graph_ms`
tokens/ms, compared as a multiple of the anchor's `1 / anchor_ms`:

- **GO-BUILD** — gate passes AND `T_pred ≥ 1.5×` anchor ⇒ Stage B
  (the executor: host drafter with an incremental n-gram index,
  device-addressed T=K+1 appends so the graph replays at moving
  positions, one host sync per step). Stage B bars are fixed NOW:
  PASS = measured end-to-end ≥ 1.8× anchor with greedy token identity
  exact over ≥ 512 tokens; PARTIAL ≥ 1.3×; REFUTED < 1.3×. The C++
  tripwire is registered with it: if the receipt shows inter-replay
  host cost ≥ 15% of the step, a C++/CUDA driver for the speculative
  loop alone becomes the follow-up registration.
- **REFUTED-FOR-CELL** — gate passes but `T_pred < 1.2×` ⇒ the graphed
  verify is too expensive for this cell's acceptance; the lane
  records the measured verify curve and closes (a draft-model variant
  would need BOTH a fresh acceptance table AND this verify cost).
- **INCONCLUSIVE** — 1.2× ≤ T_pred < 1.5× ⇒ Stage B proceeds only
  with the PARTIAL bar as its PASS bar (1.3×), everything else equal.
- **REFUSE** — gate failure, anchor A/A > half of (anchor × 0.5/1.5),
  or a verify_graph_ms < 0.9× anchor (a 17-row step cheaper than a
  1-row step is instrument error).

## Verdict calculator

`s2_verdict.py`, self-tested both directions; receipts in
`receipts-s2/stageA/`.
