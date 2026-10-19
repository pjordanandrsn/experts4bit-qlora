### Added

- **P155's record** (#1313), stream (B)'s K25 plan sweep: `bench/p155/RESULTS-p155.md` and the public receipt copy under
  `bench/p155/receipts/`.
  - **The read** is NO_GO. The best confirmed plan pair is 1.019x today's on both W16 GEMMs (lower bound 1.018 against
    the 1.15 bar), over 64 held-out steps of S0's real router picks. The self-pair read 0.9999.
  - **The contract held on sm_120:** all 126 of 144 plans that ran kept today's bits. The other 18 per projection need
    more shared memory than the card has.
  - **Reported only:** compaction's ceiling is 1.003x, so it parks. Today's plan is 80 registers at 50 % theoretical
    occupancy, and plans that raise occupancy alone do not run faster.
  - **Total:** $0.274. **Default:** this record changes none. The decode cost (K39) is timed in its own lane.
