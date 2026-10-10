### Fixed

- `describe_moe` reads a latent MoE's expert width from the stack itself (#1597). Nemotron-3 Super and Ultra feed
  their experts a 1024- / 2048-wide latent, and their stacks were sized with `hidden_size` (4096 / 8192). As a result
  `estimate_qlora_footprint` raised on Super and priced Ultra's adapters and activations on the wrong widths, and the
  int4 serve recipe split K on them. The real load was never affected. Every family whose stacks carry `hidden_size`
  is described byte-for-byte as before, checked against the released configs of every tested family.
