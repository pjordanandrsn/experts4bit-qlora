### serve_paged: bulk KV bookkeeping is on by default (`E4B_PAGED_BULK_KV`, `0` restores the per-layer path)

- **The licence.** Lane SC2c (#1166) read DEFAULT_LICENSED on Qwen3-30B-A3B int4: the stall per prefill ON/OFF was 0.19,
  serial TTFT 3.78× / 3.74× faster, the capacity ceiling went 2 → 4, and the output was identical. SC2d (#1192) read the two
  engagement checks SC2c registered before the flip, FLIP_LICENSED: on gpt-oss-20b and on Qwen3.6-35B-A3B (hybrid),
  every bulk flush wrote in bulk (`flush_bulk_fallback` 0), and the streamed text was byte-equal OFF vs ON.
- **The change.** `PagedServeConfig.bulk_kv` defaults to `True`, and `E4B_PAGED_BULK_KV` unset or empty reads `1`;
  `0` keeps the per-layer path. The estimate (`estimate_serve_footprint`) now lists the bulk flush transient as not
  modelled (SC2d recorded 168 MiB on gpt-oss-20b).
- **Tests.** The env parser's unset/empty case, the dataclass default agreeing with it, and `/health`'s `requested`.
