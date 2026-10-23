### The paged server can claim KV blocks as slots write them (opt-in, `E4B_PAGED_KV_POOL=shared`)

- Under the decode graphs a slot used to claim every block it could ever reach (its whole `E4B_PAGED_MAX_TOKENS_PER_SEQ`)
  at its first graphed decode. With `E4B_PAGED_KV_POOL=shared` (step G1 of `bench/kv-pool-g/DESIGN.md`):
  - the prompt flush claims the prompt's blocks and the one its first decode writes;
  - before each step a slot claims through the block its write position lands in. The lookahead adds its in-flight
    steps, and a speculative verify adds k_eff, which the request's budget bounds.
- The claim is an async host-to-device copy into the same block-table tensor, queued on the engine stream ahead of the
  step's gather and replay, so the captured graphs read it. Graph capture, scratch and alias slots are unchanged.
- The scheduler admits by reserved blocks (G2) in this mode. The pool keeps its claim-all size, so the reservation does
  not bind yet; sizing the pool from free memory (G3) is separate and later. `/health` reports `kv_pool`.
- `claim-all` stays the default until a GPU smoke shows the contract: identical tokens on the same schedule.
- `tests/test_kv_pool_shared.py` covers, on the real pool and buckets with a stand-in graph:
  - every real row owns its write block at replay, under the synchronous path and the lookahead, with and without bulk
    KV;
  - identical streams to claim-all, and fewer rows held;
  - the free lists whole at idle;
  - the block-edge arithmetic and the budget-bounded speculative claim;
  - the server switch.

  With the per-step claim disabled, the replay check trips.
