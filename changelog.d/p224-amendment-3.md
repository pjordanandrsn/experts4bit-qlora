### Changed

- P224 Amendment 3, before any data, after `p224-shake-1` found three kit faults: the arena is baked with
  grouped-nf4-gemm's `bake_nf4` from the HF snapshot (its index records the source commit and the quantizer the
  paged server checks; the relocation bake records neither), checked before any server process; the attention-only
  canary is registered as `reference` kernel with an fp32 expert absmax (an attention-only adapter never runs the fused
  path, so the double-quantized canary arm is dropped and F1 does not arise for it); the reduce gets its pins; the
  fixtures train on the fused kernel and run whatever the reduce outcome. Fits 6,030 s and 4,190 s; guards and worst
  cases unchanged.
