### Added

- P230 results (`bench/p230/RESULTS-p230.md`): grouped-nf4-gemm's programmatic dependent launch, as it ships (on,
  capped at 8 rows), read on the DEFAULT NF4 server for the first time: **KEEP**, at the bar's edge. With one request
  the decode step runs 1.00526x the switch off (min over two palindromic pairs; 3.707 -> 3.687 ms) with identical
  tokens on every row, and sixteen requests are untouched (the control, read PDL-free per bucket). The shipped default
  stands; register row `e4b.serve.p230.gnf4-pdl.qwen3-nf4.5090.2026-10-23`; receipts in
  `bench/p230/receipts/p230-5090-r1/`. Reported, not ruled on: the default build carries seven switched kernels and
  buckets 2 to 8 also carry PDL. The PREREG's sentence on pre-create refusals is corrected to bench/common rule 3.
