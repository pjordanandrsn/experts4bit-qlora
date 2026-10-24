### The shared KV pool leaves the prefill graph room to engage

- `E4B_PAGED_KV_POOL=shared` sizes its pool to the most rows that fit. That consumed the slack claim-all's margins
  rely on, so at 64 slots on Qwen3-30B-A3B int4 (RTX 5090, 4,096 tokens a slot) only 724 MiB was free after the
  prefill graph's capture, and `auto` refused the graph: its runtime gate wants 745 MiB (its pool again plus the bulk
  KV flush).
- `size_shared_pool` now holds back the prefill graph's runtime headroom, computed by the same function the gate applies
  (`serve_recipe.prefill_graph_headroom_bytes`, with the flush bound from the same `append_prompt_peak_bytes`). On that
  configuration the pool gives back about 1,344 rows (~1.06 GiB). Claim-all sizing is unchanged.
