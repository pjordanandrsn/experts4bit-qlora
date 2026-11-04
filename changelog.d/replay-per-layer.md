### Routing-trace replay: per-layer budgets under three shipped residency rules

- `bench/train-routing-trace/replay.py --per-layer-k K[,K...]` replays a #1469 decode trace with K slots in each
  layer, one tier per layer, under three rules, each re-stated from the code it cites:
  - `lru_seeded`: SGLang v0.5.23 paged experts. Slots start as experts 0..K-1; the device rule decides a step that
    fits K, and the host wave rule a larger one.
  - `lru`: the vLLM expert pager's VRAM tier (e9264fda). It starts empty.
  - `lfu_protect`: e4b's ColdTier (grouped-nf4-gemm `kernel/nvme_residency.py`): LFU with an LRU tie-break; rows the
    request has already resolved are never evicted.
- It reports hits, misses and page-ins per layer and in total, with per-token rates. Page-ins can exceed misses:
  ColdTier's code protects only the rows a request has resolved so far, so a request can evict a row it has not yet
  reached.
- `tests/test_replay_per_layer.py` has 22 CPU tests on hand-computed traces, including the wave case and the
  protection case. A K below the router's top_k is refused. No hit rate is published here.
