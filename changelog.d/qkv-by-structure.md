### Added

- `bench/qkv-by-structure/`: a correctness check, on the A2000, of fusing q/k/v with the module selected by structure
  instead of the `Qwen3MoeAttention` class name, over every served family's committed fixture (no timing, random
  weights). Replacing the forward on a Qwen-like structure is unsafe: it crashes on Qwen3.6's gated q. A shared
  projection that keeps each family's own forward engages on 8 families plus Gemma-4's sliding layers. It is bitwise at
  one row on six, falls back cleanly in the negative cases, and replays bitwise under CUDA-graph capture. No speed or
  quality is read, and nothing in the package changes.
