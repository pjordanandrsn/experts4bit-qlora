# Lever G3 as built (`E4B_PAGED_KV_POOL=shared`)

The design is `bench/kv-pool-g/DESIGN.md`, section G3; this file records how step G3 was built.

- `serve_paged.resolve_kv_pool` runs under `shared` only, before any weight is read, on the free memory
  `E4B_PAGED_MAX_SEQS=auto` reads. It calls `serve_recipe.size_shared_pool` with the `auto` widths, or with the
  explicit width alone.
- For each width, the rows are the most whose `estimate_serve_footprint` device total fits the free memory beside the
  prefill-graph reserve, the int8 head's copy and the 1.5 GiB margin. They are found by bisection, since the total does
  not decrease as rows are added, and capped at the claim-all size.
- A width qualifies when its rows hold the scratch rows plus, beyond them, one full sequence and
  `SHARED_POOL_MIN_TOKENS_PER_SLOT` (1,024) tokens a slot on average. That is the roadmap's floor, capped at the ceiling.
  The widest qualifying width wins.
- With no qualifying width, with the free memory unknown, or under the solver placement, the narrowest width keeps the
  claim-all pool.
- `Fp8PagedKV(pool_rows=...)` builds the pool. A claim-all runner refuses a pool smaller than its slots can reach
  (`PagedModelRunner._claim_all_last`), and the scheduler admits against `pool_rows - scratch_slots`.
- `/health`'s `engine.kv_pool_sizing` carries the inputs and the arithmetic: free bytes, margin, extra bytes, each
  candidate's rows, need and floor. It carries no throughput number; the shared pool's speed is its own reading.
- **The GPU smoke** (`bench/kv-pool-g/g3-smoke/`) runs at 64 slots on the default bucket list (`1,2,4,8,16`), so a
  64-row step runs as chained 16-row replays.
  - Claim-all against shared, token for token, with and without the lookahead, with claim-all run twice as the floor.
  - One shared run whose pool is sized from a cut free-memory reading, to `least + (claim_all - least) // 8` rows, so
    admission waits on its rows and every request must still finish.
