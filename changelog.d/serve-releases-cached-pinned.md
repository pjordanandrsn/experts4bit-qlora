### `serve_paged.build_engine` releases the loader's cached pinned memory

- The loader stages through pinned host memory, and torch's caching host allocator keeps every freed pinned block for
  the life of the process.
- Lane SV4 (#1240, RTX 4090) read 1.1–1.3 GB of pinned memory beyond the priced cold-tier landing in every Qwen3-30B-A3B
  serve.
- Measured on an RTX A2000 host:
  - after `load_moe_4bit_streaming`: 1.09 GB reserved, 0 B allocated;
  - after the hybrid tier: 1.37 GB reserved, 4 MB allocated.
- `engines.host_heap.release_cached_pinned_memory()` calls `torch._C._host_emptyCache()` (private in torch 2.8–2.11,
  guarded). `build_engine` calls it before its heap trim and reports `pinned_cache_released`.
- Qwen3-30B-A3B's pinned memory after the build fell from 1,314 MiB to 14 MiB. Serving then adds only the cold-tier
  landing as it fills.
