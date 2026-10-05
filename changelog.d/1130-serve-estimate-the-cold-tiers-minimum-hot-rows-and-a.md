### Serve estimate: the cold tier's minimum `hot_rows`, and a refusal below it

- `serve_recipe.min_hot_rows(topology, setup)` is the fewest cold-tier rows a solver setup can serve with, by
  grouped-nf4-gemm's own ColdTier rule ("size hot_rows >= max routed experts per layer"): `top_k × max(chunk_tokens,
  max_seqs)`, at most `n_experts` and at most the NVMe rows.
  - Without a routing profile the solver fills layer by layer, so NVMe holds whole trailing layers.
  - The server's default of 64 is below that for Qwen3-30B-A3B (128 experts, top-8): a long prefill through an
    NVMe layer would be refused mid-request.
  - The default is far above it for Mixtral (8 experts). There, 64 rows of ~99 MB each in the pinned landing, the
    cold view and the setup tier crowd the DRAM tier out of the host budget.
- `estimate_serve_footprint` refuses a solver setup with rows on NVMe and `hot_rows` below the minimum, in words.
