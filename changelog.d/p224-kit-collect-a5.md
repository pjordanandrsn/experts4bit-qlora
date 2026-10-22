### Fixed

- P224's box `collect()` carried a hard-coded three cost cells into `cost.json`'s blocks, so Amendment 5's graded
  `ttft_any_ms` and `itl_wall_ms` reached the reducer as unread. `p224-shake-4` recorded both cells and found the
  gap. `collect()` now takes its cells from the reducer's own `COST_METRICS`. A test runs that shake's six real slot
  records (sha256-pinned) through `collect()` and the reducer's G3: the graded W1 cells are present and inside the
  band. No rule changes.
