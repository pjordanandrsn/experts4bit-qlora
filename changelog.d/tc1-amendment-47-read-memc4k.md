### Read: TC1 amendment 47 — on packed rows e4b's peak is 7.47 GB above Unsloth's; 6.1 GB of it is the padded LoRA delta padding every expert to the hottest (P112, P113 HELD; P114 FALSIFIED)

- `tc1-5090-100` ($1.05): the allocator census on packed 4,096-token rows. e4b's defaults 32.34 GB, with `E4B_ABSMAX_DQ=1` and the
  compact delta 29.54 GB, Unsloth 24.86 GB. Of the 7.47 GB gap, 1.35 GB is the fp32 absmax and 6.10 GB grouped-nf4-gemm's padded LoRA
  delta, about 11.6× the routed rows at the down projection. The compact delta leaves its own padded block and output (4.28 GB).
- Row `e4b.train.memory-census.packed-4k.qwen3.5090.2026-10-06`; STATUS says so. The next registration targets the padding.
