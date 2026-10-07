### serve_paged: `E4B_PAGED_BUCKETS=auto` captures decode-graph buckets up to `max_seqs` (opt-in); `/health` reports the buckets asked for, each bucket's replays and the KV pool's size

- **Why.** SC2c's post hoc (`bench/h2h-2026-10-02/sc2c/README.md`) put `max_seqs` 16 → 64 first among the levers toward
  8 req/s under `serve_capacity`. Above 16 sequences the default bucket list stops at 16, so a wide decode step runs as
  consecutive 16-row replays with a host sync after each, and nothing reported it.
- **The change.**
  - `E4B_PAGED_BUCKETS=auto` (or `ServeSetup(buckets="auto")`) captures every power of two below `max_seqs`, then
    `max_seqs` itself (`serve_recipe.default_buckets`: 32 → `1,2,4,8,16,32`). Up to 16 sequences it equals the default
    list after its trim.
  - The default stays `1,2,4,8,16`, so every existing server captures what it did. An empty `E4B_PAGED_BUCKETS` now reads
    as the default, as the other knobs do; an unreadable value is refused by name.
  - The server logs when its widest step would run as consecutive replays, and when a bucket exceeds 64 rows (the T=1
    folds stop there).
  - `/health` gains `engine.buckets_requested`, `engine.graph_stats` (per bucket: replays, eager steps, rows, padding
    rows; `/stats` already had it) and `levers.kv.pool_mib` (`paged_kv_pool_bytes` on the pool's own geometry).
  - The step trace counts `dec_pieces`, the replays a decode step took; its `bucket` names only the last one.
  - `ServeSetup.decode_buckets` is what the server captures. The estimate sizes the scratch slots from it and says
    that graph pools for buckets above 16 rows are unmeasured.
- **Memory.** 103.5 MiB of FP8 KV per slot at 2,048 tokens on Qwen3-30B-A3B: 1.63 / 3.26 / 6.52 GiB at 16 / 32 / 64 slots.
- **Not changed.** No route, kernel or default. Decode rows above 16 take the paths prefill chunks take today
  (`Int4Linear`'s cached bf16 weight above 16 rows; the chained tile table above 256 routed rows). Speed is unread: lane
  SC2e (#846) registers it.
- **Tests** (the wide-step ones in a new `tests/test_serve_slots.py`; `tests/test_decode_graph_buckets.py` is pinned by
  lanes P109–P115). `default_buckets` against the trimmed default up to 16; `auto` round-trips through `ServeSetup.to_env` and
  the estimate prices its scratch slots; the env parser's cases and refusals; the startup log; `/health`'s new fields;
  `_kv_pool_mib` against a constructed pool; `dec_pieces` on a chained and a native 40-row step.
