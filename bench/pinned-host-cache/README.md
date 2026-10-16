# The fixed pinned host term: safetensors' cached pinned reads

The host-memory slope reads (Q235 Amendment 5, `bench/q235/gate2/memslope.json`, and the Mixtral-8x22B read) found
`RssShmem` growing by exactly the estimator's pinned expert homes per layer, plus a model-dependent fixed term:
2.231 GB on Qwen3-235B and 0.700 GB on Mixtral-8x22B.

**Mechanism.** safetensors 0.8 reads each tensor opened with `device="cuda"` through a pinned host block from torch's
caching host allocator (`get_tensor_pinned_cuda`). The allocator rounds each block up to a power of two and keeps it
cached after the copy, and later reads of the same size class reuse it. The loader reads every non-expert tensor that
way, so a load leaves one cached block per size class of the dense tensors:

| model | size classes | predicted |
|---|---|---:|
| Qwen3-235B | 2 GiB (embed, head) + 64 MiB (q/o) + 4 MiB (k/v) + 1 MiB (router) + 8 KiB + 256 B | 2,219,843,840 B |
| Mixtral-8x22B | 512 MiB (embed, head) + 128 MiB (q/o) + 16 MiB (k/v) + 128 KiB (router) + 16 KiB | 688,013,312 B |

The prediction was posted before the measurement. An independent recount from the safetensors headers matched it to
the byte at 2 layers and at full depth.

**Measurement** (`pin_probe.py`, RTX A2000, 2-layer cuts, `results.json`). Loading uses the box's own `build()`, then
`torch._C._host_emptyCache()`, then the step-0 forward:

| model | released by `_host_emptyCache()` | vs predicted | outside the allocator after | step-0 loss vs the slope read |
|---|---:|---:|---:|---|
| Qwen3-235B | 2,219,843,841 B | +1 B | 10,485,760 B (10 MiB) | 12.84375, bitwise equal |
| Mixtral-8x22B | 688,013,313 B | +1 B | 10,485,760 B (10 MiB) | 8.880563735961914, bitwise equal |

The live pinned storages after the release are the expert homes alone (power-of-two total 3,623,878,656 B, equal to
what the allocator still owns). No live pinned tensor has the embedding's shape. Every shared mapping is
`/dev/zero (deleted)` (CUDA pinned memory). There is no `/dev/shm`, memfd or IPC mapping. The 10 MiB left outside
the allocator is the same on both models, so it is CUDA's, not the load's.

So the slope reads' fixed term is these cached blocks plus that 10 MiB: 2.2303 GB against 2.231, and 0.6985 GB against
0.700.

**Fix.** `load_moe_4bit_streaming` now ends with `_release_cached_pinned_blocks(device)`. On CUDA it synchronizes and
returns the allocator's free pinned blocks to CUDA. Blocks a live tensor holds (the homes) stay.
`torch.cuda.empty_cache()` does not do this.
