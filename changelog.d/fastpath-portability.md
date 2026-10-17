### Added

- `bench/fastpath-portability/`: a records-and-code assessment of which Qwen3-30B-A3B serving fast paths carry to the
  other served families, classed per family (works / narrow gate / tune / port / unsafe). Three gates tie the speed to Qwen:
  K33's exact-shape auto gate in grouped-nf4-gemm, `serve_paged`'s fusion and prefill-glue family allow-lists, and the
  `Qwen3MoeAttention` class name for fused q/k/v. It proposes four paired readings and runs none. Documentation only.
