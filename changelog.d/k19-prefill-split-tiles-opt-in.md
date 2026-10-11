### `E4B_K19_PREFILL_BIG_M` (opt-in): K19's prefill rows over a split tile table (#846)

K19's prefill calls above 512 routed rows can run over grouped-nf4-gemm#542's split tile table: each expert's full 32-
or 64-row tiles in one launch, then its remainder in 16-row tiles. The default (`0`) keeps today's 16-row table.

- **Why.** A $0 shape census of Qwen3-30B-A3B on SC2's prompts found the rows of a 512-token chunk bimodal: one expert
  takes a median 398 of 4,096 routed rows. On the 16-row table that expert runs as ~25 tiles, each re-reading its weight
  slice. The split keeps the 16-row table's padding.
- **Scope.** Decode keeps the 16-row table: a 64-row step routes exactly 512 rows on a top-8 model. The knob composes
  with P130's lean dispatch (`gather_div`, `scatter`).
- **Reporting.** `prefill_routes.k19_prefill_big_m` reports the setting and `seen.moe_k19_dispatch` counts `split|...`
  calls. A kernel package without the split is refused while the knob is set.
- **Status.** Not licensed by any read. Its speed and a FUNCTION gate are the next registration's to read.
