### Fixed

- DQ11 tensor byte hashes accept scalar losses and preserve the logical contiguous
  bytes of strided tensors. CPU regression coverage checks scalar loss/gradient
  neutrality, signed-zero/NaN payloads and stored versus canonical FP32 hashes.
  The proof recipe and scientific package/input locks are unchanged.
