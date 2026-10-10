### Fixed

- DQ11's A2000 rehearsal requires the L reading's live CUDA train-prefetch
  schedule to issue forward and backward prefetches and be used during the
  actual forty updates. Before/after counter snapshots and positive deltas are
  preserved, including on refusal, and checked again by the rehearsal checker.
