### Fixed

- The paged server at `E4B_PAGED_PLACEMENT=solver` now refuses at startup, by name, an `E4B_PAGED_HOT_ROWS` below the
  cold tier's hard floor, instead of failing mid-request on a long prompt ("request of N unique rows exceeds
  hot_rows"). The floor is `min(F x top_k, the most NVMe experts in any MoE layer)`, with
  `F = max(E4B_PAGED_CHUNK_TOKENS, max_seqs)` the most tokens one forward carries. It is checked right after the
  placement is solved and before the tier pins any row, and recorded in `/health` under `levers.hot_rows_floor`. The
  default `hot_rows` and `all-vram` are unchanged. Tests tie the floor to the real cold path on CPU: a forward of F
  tokens through `enable_hybrid_tier` passes at the floor and raises one row below it, and DRAM and VRAM experts never
  reach the tier.
