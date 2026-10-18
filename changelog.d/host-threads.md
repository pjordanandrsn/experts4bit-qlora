### Fixed

- `load_moe_4bit_streaming` caps torch's intra-op threads for the host-side expert fuse (offload on CUDA) at min(the pool, the CPU affinity, the container's cgroup CPU quota rounded up) and restores them after; the bytes are unchanged. A rented container seeing a 256-thread host under a ~31-CPU quota had loaded Qwen3-30B-A3B in ~2,000 s against 40-61 s elsewhere. Every load now logs the visible CPUs, the quota, torch's threads and `OMP_NUM_THREADS` (`experts4bit_qlora.util.container_cpu_quota` / `host_cpu_report`). Evidence: `bench/host-threads/`.
