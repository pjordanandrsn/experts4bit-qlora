### The paged server can space and shorten prefill while requests decode (opt-in)

- Two scheduler knobs reshape how prompt ingestion shares steps with running decodes. Both apply only while a decode
  row is live, and their defaults compose exactly the plans the scheduler made before.
  - `E4B_PAGED_PREFILL_EVERY=N` (`ContinuousScheduler(prefill_every=N)`, default 1): a step carries prefill only if the
    previous `N - 1` steps carried none. Running decodes take at most one prefill stall every N steps. The next chunk
    waits at most `N - 1` steps for its turn, so a given prompt waits about `N` steps per chunk owed ahead of it (its
    own, and those of prompts admitted before it). With nothing decoding, prefill runs every step as before.
  - `E4B_PAGED_LOADED_CHUNK_TOKENS=T` (`loaded_chunk_tokens`, default 0 = off): caps the chunk and the step's prefill
    budget at `T`, so one long chunk becomes several short stalls.
- `/health` reports both, and `stats()` counts `prefill_deferred` (steps that held pending prefill back) when
  `N > 1`.
- A first chunk shorter than the prefill graph's size runs eagerly, and the runner counts it as a `short_chunk`.
  Chunking a prompt differently can change its tokens, as any change of `E4B_PAGED_CHUNK_TOKENS` can. Neither knob is a
  default; a default needs a served read.
- `tests/test_scheduler_interleave.py` checks the defaults call for call against the previous planner (kept in the
  test verbatim), the spacing and the next chunk's `N - 1` bound under `prefill_every`, that an idle server never delays a prompt, the
  cap on both the chunk and the budget, refusals, and the environment parsing.
