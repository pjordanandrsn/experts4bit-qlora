### Added

- **P156's record** (#1313), stream (B)'s decode lane: `bench/p156/RESULTS-p156.md` and the public receipt copy under
  `bench/p156/receipts/`.
  - **The read** is GO. K39's decode (K25's `lut="prmt"`) at today's plan is 1.1755x today's select tree on both W16
    GEMMs, with a lower bound of 1.1728 against the 1.05 bar, over 64 held-out steps of S0's real router picks. The
    self-pair read 0.9985, and P moved no bit. The median sits in the serial-issue model's band (1.15-1.18) that the
    static count predicted.
  - **Clocks:** the bench's end sampled a software power cap (2,272 MHz against 2,400). Giving that whole ratio to P,
    the lower bound is still 1.1103, so the GO does not depend on clocks.
  - **The contract held on sm_120:** all 144 prmt plans ran, every one bit-identical to the tree.
  - **Reported only:** the prmt sweep's best plan pair reads 1.90x the tree at today's plan. The same plans run slower
    under the tree, so it is the decode that lets them pay. It was selected with a 0.153 curse and needs its own read.
  - **Total:** $0.151. **Default:** this record changes none. The served read is its own registration.
