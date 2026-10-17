### Added

- `bench/async-reuse-20261017/`: an RTX A2000 correctness check of three paths that reuse device workspaces across asynchronous
  launches (grouped-nf4-gemm's int4 fused split-K reduce, its fp8 split-KV in-kernel combine, and e4b's expert offload), under
  NaN-poisoned workspaces and concurrent copy traffic. No reuse failure found: 0 of 8,400 and 0 of 3,600 launches mismatched,
  and 30 offloaded passes showed a gradient spread the same size as the resident model's own, with no NaN. It also records that
  offloaded layers keep the fp32 absmax while a resident build double-quantizes it, so an offloaded-versus-resident parity check
  must set `E4B_ABSMAX_DQ=0` on both sides. Documentation only.
