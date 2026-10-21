### Added

- `bench/hbd/` records the HBD read (one RTX 5090, $0.416; no parity claim). Across torch 2.8.0+cu128 and 2.12.1+cu130,
  each framework's exported base differs in exactly one of 533 slots of the 2-layer cut, layer 1's q_proj: the dequantized
  fingerprint's int16 sum differs by 554. Within either build, e4b's and Unsloth's dequantized bases are identical, and the
  transformers version and Unsloth's MoE backend do not move the base. HP2's base term is therefore consistent with a build
  effect; depth is untested.
