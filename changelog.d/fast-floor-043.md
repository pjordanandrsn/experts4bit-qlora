### `[fast]` requires grouped-nf4-gemm 0.43.0

- The `fast` and `test` extras now require grouped-nf4-gemm >= 0.43.0 (was 0.30.0): the release that makes
  `GNF4_GEMV_BW=auto` its default. On GPUs with at least 160 SMs, single-request decode at Qwen3-30B-A3B's NF4 expert shapes
  takes the bandwidth-targeted GEMV, 1.24× as fast and unchanged at 16 requests (lane P116,
  `e4b.serve.p116.gemv-bw.qwen3.5090.2026-10-07`). This is P116's registered consequence (`bench/p116/RESULTS-p116.md:93-96`).
- `pyproject.toml`'s comment ladder, CI's `--requires` assertion, `docs/capabilities.json`, the solution pages and
  `docs/SERVING.md` state the new floor. `GNF4_GEMV_BW=0` restores dot-pad.
