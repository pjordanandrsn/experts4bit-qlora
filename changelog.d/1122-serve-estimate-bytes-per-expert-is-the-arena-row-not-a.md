### Serve estimate: bytes per expert is the arena row, not a share of the stack

- `serve_recipe.bytes_per_expert(stack)` is the stack's growth from one expert to `n_experts`. It is exactly the arena
  row before alignment: packed 4-bit plus fp32 absmax, gate_up and down. The previous `slab // n_experts` smeared the
  stack's per-stack constants (the NF4 code table) across every row. On OLMoE that gave 3,538,945 bytes against the
  bake's 3,538,944, enough to round the aligned stride up a page. The solver's tier split and the hybrid tier's buffers
  use the exact figure now.
