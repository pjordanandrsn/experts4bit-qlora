### P120 read (#846): `E4B_INT4_WIDE_TILES=1` makes Qwen3-30B-A3B's 64-row decode step 1.44× slower; it stays opt-in (bench and docs only)

- **The reading** (`p120-5090-1`, one RTX 5090, $0.878; lane $1.079): on SC2e's int4 stack, the captured 64-row step
  goes from 17.55 ms to 25.31 ms with the one-launch tile table above 256 routed rows. Every token is identical
  (16,704 positions), the mutant died, and the same-setting pairs agree within 0.09 %. Verdict SLOWER by the
  registered rule.
- **Why:** the single-program cumsum table costs 10.45 ms a step at 512 routed rows × 128 experts (218 µs a launch),
  against 1.43 ms for the chained builder's kernels. By subtraction the chained builder costs about 2.7 ms of the step,
  so a faster one-launch table remains a lever: a grouped-nf4-gemm launch-shape lane.
- **Predictions:** Q1 (the premise) and Q5 (noise) held; Q2–Q4 missed high.
- **Register row:** `e4b.serve.p120.wide-tiles.qwen3-int4.5090.2026-10-08`.
- **Files:** `bench/p120/RESULTS-p120.md`, `bench/p120/receipts/p120-5090-1/` (with `SHA256SUMS`).
