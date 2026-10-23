### The shared KV pool is sized from free memory (opt-in, `E4B_PAGED_KV_POOL=shared`)

- With `E4B_PAGED_KV_POOL=shared` (step G3 of `bench/kv-pool-g/DESIGN.md`) the server sizes the FP8 KV pool from the
  device's free memory before any weight is read, as `E4B_PAGED_MAX_SEQS=auto` prices a width. For each width the pool
  takes the most rows whose estimate fits, beside the prefill graph's reserve, the int8 head's copy and the 1.5 GiB
  margin, capped at the claim-all size (`serve_recipe.size_shared_pool`). A width qualifies when its rows hold the graph
  scratch slots plus one full sequence and 1,024 tokens a slot on average; under `auto` the widest such width wins, and
  an explicit width is kept. `E4B_PAGED_MAX_TOKENS_PER_SEQ` stays each slot's ceiling.
- The pool can therefore hold fewer rows than its slots could ever reach. Slots claim blocks as they write (G1) and the
  scheduler admits against the pool's rows less the scratch rows (G2), so a request waits for blocks rather than
  running the pool dry. `Fp8PagedKV(pool_rows=...)` builds such a pool, and a claim-all runner refuses one.
- `/health` reports `engine.kv_pool_sizing`: the pool's rows against the claim-all size, the blocks the scheduler admits
  against, what it holds reserved, how often its queue head waited, and the sizing's inputs and arithmetic.
- `claim-all` stays the default and is unchanged. With the free memory unknown, or under the solver placement, the
  shared pool keeps the claim-all size.
