# Host threads and a CPU quota: the slow offloaded load (machine 51612)

**What happened.** `p137-4090-4` and `p137-4090-7` loaded Qwen3-30B-A3B with expert offload in 1,957 s and 2,059 s on
Vast machine 51612 (AMD EPYC 7742, 256 CPUs visible). Across the receipt store, the same offloaded load took 40–61 s on
nine other hosts, 224- and 256-thread EPYCs among them. The same machine 51612 loaded the model resident in 35–58 s across
nine loads, and a resident load skips the host-side stack/cat. So on 51612 the reads were fast and the CPU-side fuse was
slow: its container held about 1/8 of the machine, consistent with a CFS quota of about 31 CPUs. P137's receipts recorded
neither the quota nor the thread count.

**The mechanism, checked at $0** (`omp_probe.py`, a diagnostic: no number here travels to another host). On a 6-core desktop host
(Intel Xeon W-1250, 6 cores and 12 threads), the loader's own `fuse_experts` assembled one Qwen3-30B-A3B-shaped layer: 128
experts, gate and up [768, 2048], down [2048, 768], bf16, about 1.2 GB. Medians of 4 runs:

| container | torch intra-op threads 12 | threads 2 |
|---|---|---|
| `docker run --cpus=2` (a CFS quota of 2 CPUs; 12 visible) | 24.16 s | 0.50 s |
| no quota | 7.68 s | 0.43 s |

A pool larger than the CPUs the process can actually run on makes this copy many times slower. Inside the quota the pool
spins through the quota and is throttled every period. The no-quota row is diagnostic only: that host runs a full
service stack on its 6 cores, so that slowdown is most likely contention, and it sets no rule. **The cap targets the quota
case.** With no quota it is min(pool, affinity), unchanged here. The new load log shows whether a real box ever hits a
no-quota slowdown; that would be a separate, measured fix. Inside
`--cpus=2`, `container_cpu_quota()` reads the cgroup v1 quota as 2.0 CPUs, so `host_thread_cap()` gives 2.

**The fix (this directory's PR).**
- `load_moe_4bit_streaming` logs the visible CPUs, the quota, torch's threads and `OMP_NUM_THREADS` at every load.
- For the staged (offload on CUDA) expert loop, the loader caps torch's intra-op pool at min(its size, the affinity, the
  quota rounded up), and restores it after.
- The stacked bytes are unchanged; only the speed moves.

The quantized-expert cache (#1750) is on hold: on a normal host it would save about a minute.
