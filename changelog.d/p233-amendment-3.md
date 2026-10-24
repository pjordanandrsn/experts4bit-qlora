### Changed

- P233 Amendment 3 (pre-data): every arm names `E4B_PAGED_HOT_ROWS=128`, so the NVMe tier holds at least as many pinned
  rows as one 512-token prefill can route (the tier's documented floor). The server's default of 64 is below that floor
  at the solver split, and the first shakedown stopped in phase S on it before any number was read. The box refuses an
  arm without the knob; the reducer VOIDs a phase whose recorded `hot_rows` is not 128; `staged-p233.sha256` is re-pinned.
  The read now speaks for solver placement with `E4B_PAGED_HOT_ROWS=128`.
