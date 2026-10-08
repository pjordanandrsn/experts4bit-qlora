### P122 registered (#846): P120's speed read again on grouped-nf4-gemm #519's chunked cumsum tile table (bench and tests only)

- **Why.** P120 read `E4B_INT4_WIDE_TILES=1` SLOWER, 1.44× at 64 rows. The cause was the one-program cumsum table:
  10.45 ms a step at 128 experts. The chained builder it replaces still costs about 2.7 ms of the step. grouped-nf4-gemm
  #519 builds the table in 64-row chunks with a per-expert carry, giving the same integers.
- **The box is P120's**, run at its registered bytes: the eager profile with the knob 0 and 1 (now with the chunked
  table's time), four captured-graph ABBA arms of 256 timed steps with every token recorded, and the mutant.
- **The rule is P120's** (`bench/p122/p122_reduce.py`, 30 self-test cases): DEFAULT_ON iff both ratios are ≤ 0.98 with
  every token identical.
- **The premise on the card** is the decode-graph bucket tests plus #519's chunked cumsum tests compiled for the card.
- **Budget:** proof on Granite (guard 0.75 h); reading guard 1.5 h; lane ceiling $3.00.
