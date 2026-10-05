### serve_paged: `/health` `kv_bookkeeping` counts bulk flushes written per layer (`flush_bulk_fallback`)

- `Fp8PagedKV.append_prompt` now returns whether its bulk path wrote the prompt. It still falls back to `append` per
  layer, with the same bytes, for a slot already holding tokens, prompts of different lengths or a demoted arena.
- `PagedModelRunner` counts those fallbacks as `flush_bulk_fallback`, and `/health`'s `kv_bookkeeping` block reports
  the count. Before this, `flush_bulk` counted the call, not what ran inside it.
- **Why now.** The `E4B_PAGED_BULK_KV` default (licensed by SC2c, #1166) waits on an engagement read on a hybrid model
  and on gpt-oss (#1131's review). That read should show the bulk path ran there, not that it was called.
- No behaviour change: same pool bytes, same paths. Two tests are added.
