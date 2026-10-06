### Serve estimate: the DRAM tier's prefill on the GPU is priced

- Under the solver placement, a prefill chunk computes the DRAM tier's routed experts on the GPU
  (`hybrid._dram_on_gpu`, on by default as the runner's `gpu_only_prefill`). It uploads their NF4 bytes with absmax
  cast to bf16 and allocates the chunk's routed rows (bf16 in, two fp32 outs).
- The estimate did not price it. On ERNIE-4.5-21B-A3B (RTX A2000, 992 DRAM rows, no NVMe) the serving peak sat
  0.33 GiB over the estimate, and an allocator-history replay put 457.5 MiB live in `_dram_on_gpu` at the peak:
  64 experts × 5.98 MiB + 75 MiB, this exact arithmetic.
- New derived item "DRAM experts run on the GPU at prefill (one layer call)". It is priced by its excess over the
  cold rows' stack, since a layer call streams one or the other.
- `serve_recipe.dram_rows_per_layer` gives the most DRAM experts one layer holds under the solver's placement.
  `solver_tiers` and it now share one cached `solve_placement` call.
- ERNIE now reads 117 MiB over its peak, against 168 MiB of prefill-staging ceiling. OLMoE's three tiered receipts
  (128-token prompts) read 165–173 MiB over, the chunk-sized ceilings that short prompts do not reach.
