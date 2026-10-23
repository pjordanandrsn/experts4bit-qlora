### The scheduler can admit by reserved KV blocks (opt-in, not yet wired to the server)

- `ContinuousScheduler(kv_blocks=N, block_tokens=16)`: a request is admitted only if a slot is free and the blocks already
  reserved plus its own need, `ceil((prompt_len + max_new_tokens) / block_tokens)`, fit `N`. The reservation is released
  when the slot is freed, by a finish or an abort. FIFO is kept: a head that does not fit waits, and nothing behind it
  is admitted first. A request that could never fit is refused when it is added.
- `stats()` reports `kv_blocks`, `kv_blocks_reserved` and `kv_blocks_waits` when the pool is set. `kv_blocks=None`, the
  default, admits exactly as before.
- This is step G2 of the shared-pool design (`bench/kv-pool-g/DESIGN.md`). The server does not pass `kv_blocks` yet;
  that comes with incremental block claims (G1) and stays opt-in until a GPU smoke.
- `tests/test_scheduler_kv_blocks.py` covers: no pool and a non-binding pool making the same runner calls; a binding
  pool keeping FIFO; the refusal; release on finish and abort; and the decode lookahead under a binding pool yielding
  the synchronous streams.
