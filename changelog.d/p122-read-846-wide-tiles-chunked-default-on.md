### P122 read (#846): with grouped-nf4-gemm #519's chunked tile table, `E4B_INT4_WIDE_TILES=1` makes Qwen3-30B-A3B's 64-row decode step 4.3 % faster — DEFAULT_ON (bench and docs only)

- **The reading** (`p122-5090-1`, one RTX 5090, $0.837; lane $0.930): on SC2e's int4 stack the captured 64-row step goes
  from 17.53 ms to 16.78 ms (ON/OFF 0.9568, 0.9569). Every token is identical (16,704 positions), the mutant died, and
  the same-setting pairs agree within 0.1 %. The box is P120's, run at its registered bytes.
- **Why it turned:** the chunked table costs 2.51 ms a step at 512 routed rows × 128 experts, against 10.45 ms for P120's
  one-piece table. That is still 1.75× the chained builder's named kernels (Q2 missed high), so a multi-program table
  is the next lever.
- **The registered consequence:** a separate e4b PR makes the switch default to on for tables no larger than the one
  read, with `0` restoring the chained builder.
- **Register row:** `e4b.serve.p122.wide-tiles-chunked.qwen3-int4.5090.2026-10-08`.
- **Files:** `bench/p122/RESULTS-p122.md`, `bench/p122/receipts/p122-5090-1/` (with `SHA256SUMS`).
