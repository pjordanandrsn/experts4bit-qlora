### P126 attempt 1 (#846): NOISY — no default moves; the rerun is Amendment 1's

`p126-5090-1` (one RTX 5090, $0.874) read **NOISY**: P = 4's two interleaved blocks disagree by 2.35 % (bound 1.5 %).
- **P = 8's blocks agree** at 0.8959 / 0.8974. The split table runs at 0.092× (P = 8) and 0.225× (P = 4) of the
  one-program table, which is 17.7 % of the eager 64-row step.
- **The cause** is one runner's level offset of about 3 % in the box's first block, lasting all 256 steps, so longer
  blocks would not remove it.
- At attempt 1, no default moved; attempt 2 later licensed P = 4 and #1476 enables `auto` within the measured shape
  bound. `bench/p126/RESULTS-p126.md`; receipts in `bench/p126/receipts/p126-5090-1/`.
