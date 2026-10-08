### P119 amendment 1 (#846): the census box records the KV pool's layer count as the int `kv_layers()` returns (bench and tests only)

- `p119-prove-1` (HARNESS_ERROR, $0.069) stopped in the box: it called `len()` on `kv_layers()`'s int. Fixed; the CPU
  test's stand-in pool now returns an int as the real one does, so the defect fails on CPU. `staged.sha256` re-pinned.
- No bracket, rule, prediction, guard or budget changes; before any reading.
