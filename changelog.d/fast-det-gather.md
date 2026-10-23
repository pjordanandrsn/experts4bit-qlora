### Added

- `E4B_TRAIN_DET_GATHER=1` (opt-in): the fused training forward's token gather (`engines/fast.py`) takes the
  deterministic backward the batched path already uses (`_GatherTokenRows`, #765) instead of `index_select`'s, whose
  `index_add_` adds each token's k gradient copies with CUDA float atomics, so dL/dx changed in its last bits call to
  call. The forward is unchanged bit for bit; the backward differs from the atomic one by float reordering only, not in
  expectation. Off by default until it is shown self-identical on a card; `FAST_TRAIN_STATS["det_gather"]` counts the
  forwards that took it.
