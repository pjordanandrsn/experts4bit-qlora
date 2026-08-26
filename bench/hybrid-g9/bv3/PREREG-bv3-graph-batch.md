# PREREG — BV3: the B>1 CUDA-graph decode loop

Registered 2026-08-25, before any measurement. The lever BV2 named
and every prerequisite of which this campaign has since merged:
device-side expert grouping (capture-safe, S3), graph-mode appends
(F1-B2 + b1d), fused combine + fused QKV (F2). BV2's receipts:
eager B=1 costs 59.1 ms against the 7.35 ms graphed step — a ~52 ms
host tax per step — and the eager curve tops at 123.7 aggregate
tok/s (B=16, 129.4 ms/step), host-bound.

## Treatment

Generalize the certified b1d graph loop to B rows:

1. `Fp8PagedKV.graph_mode_init` accepts a slot LIST;
   `append_graph_bt1(layer, k[B,H,D], v[B,H,D])` appends one token to
   each of B slots — a per-slot loop of the certified fused
   single-slot append (2B extra launches/step, ≈32 at B=16, noise
   against the step) with each slot's own device table row and lens
   scalar; per-slot `seq_lens` publish stays the in-stream add it is
   today. No new kernels.
2. The harness's manual decode loop at `--batch B`: static
   `in_ids [B,1]` / `pos [B,1]`, `ctx.slots` = the B slots, greedy
   argmax per row copied back in place, one captured step replayed.
   MoE sees T=B rows → the merged device-grouping path (capture-safe
   by the S3 gates).

## Arms (anchor-gated box)

- `eager_b16_a/b`: the BV2 production scheduler at B=16 (the standing
  comparison arm, re-run fresh — BV2's numbers are registration
  basis, not the comparison).
- `graph_b16_a/b`: the captured loop at B=16, tokens per stream.
- `graph_b1`: sanity — must land in the current single-stream class
  (the b1d loop it generalizes).
- Identity: each graphed stream vs the same-slot eager stream, the
  K6-B frame (exact length, ≥32-step divergence gate, degeneracy
  law). The graphed loop drops rows that finish early? NO — fixed
  128 steps per row, no early exit (static shapes; EOS handling is
  serving polish, not this measurement).

## Bars (gain frame off the fresh eager B=16 median)

- **PASS** — graphed B=16 aggregate ≥1.5× the fresh eager B=16
  aggregate AND identity green AND `graph_b1` within ±5% of the
  current certified single-stream class. Ships as `--b1d-loop graph`
  honoring `--batch`.
- **PARTIAL** — ≥1.2×; ships with disclosure.
- **REFUTED** — <1.2×: the batch wall is not the host after all —
  record what the step decomposition says instead.
- REFUSE: A/A spread (either pair) wider than half the PASS margin,
  identity breach, degenerate streams, B=1 sanity outside ±5%.

## Calculator

`bv3_verdict.py`, self-tested before any receipt; receipts under
`bench/hybrid-g9/bv3/receipts-bv3/`.
