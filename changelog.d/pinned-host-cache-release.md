### The streaming loader releases its cached pinned host blocks (about 2.2 GB on Qwen3-235B)

- safetensors reads each dense tensor opened with `device="cuda"` through a pinned host block. torch's caching host
  allocator rounds that block up to a power of two and keeps it after the copy. A load therefore left one cached block per
  size class behind, in host RAM, for the life of the process: 2.22 GB on Qwen3-235B and 0.69 GB on Mixtral-8x22B.
- `load_moe_4bit_streaming` now returns those free blocks to CUDA at the end of a CUDA load and logs the amount. The
  offloaded expert homes, which live tensors hold, are untouched.
- Measured on an RTX A2000 at 2 layers: 2,219,843,841 B and 688,013,313 B released, within 1 B of the prediction from
  the dense tensors' sizes. The step-0 loss was bitwise unchanged on both models. Receipts: `bench/pinned-host-cache/`.
