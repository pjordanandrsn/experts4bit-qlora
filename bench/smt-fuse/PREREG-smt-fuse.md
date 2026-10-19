# SMT fuse measurement — does a 32-thread pool beat 16 for the loader's host-side expert fuse?

**Why.** `q235-pro6000-7` ran out of its box cap while fusing and quantizing experts: 2,088 s against about 661 s on
`-6`. Its host showed 32 visible CPUs, a 30.72-CPU quota and torch's default pool of 16, the physical cores on an SMT host.
#1752's cap only lowers that pool. Q235's simplification amendment needs a per-core rate, and needs to know whether the
second hardware thread per core helps this op. This is a **timing premise for that amendment, never a ruling**. If the
32/16 ratio's range includes 1, the pool stays at torch's default and only the cap's load term changes.

No A2000 timing seeds anything here: that box is correctness-only. The timing runs on one short rental of a
server-class SMT host.

## What runs (`smt_fuse_box.py`)

- **The slab:** one Qwen3-235B-A22B MoE layer's experts as the streaming loader builds them under expert offload, 128
  experts with gate/up [1536, 4096] and down [4096, 1536] in bf16 (4.83 GB), seeded.
- **One rep is the loader's own path, timed wall-clock from slab-in to nf4-out:**
  - `fuse`: `experts4bit_qlora.arch.moe_conventions.fuse_experts` on the host, the op torch's intra-op pool runs;
  - `h2d`: the fused stacks to the GPU;
  - `quant`: `Experts4bit.from_float(..., quant_type="nf4")` on the GPU.
  - The pool is set explicitly (`torch.set_num_threads`) per rep.
- **Order:** one untimed warm rep at 16, then the pools **16, 32, 32, 16** for **4 cycles**: 16 reps and 8 back-to-back
  (16, 32) pairs.
- **Recorded once:** the CPU model, physical and logical counts, threads per core, the affinity, the cgroup quota and
  torch's default pool.
- **Recorded per rep:** the pool torch reports, the 1-minute load average at its start, the segment walls, and the sha256
  of the fused bytes and of the nf4 bytes.

## What is reported

- **Per pool:** median seconds for each segment and in total, and both scaled to one layer (the amendment multiplies by
  94 layers).
- **Each pair's 32/16 ratio**, for the total and for the fuse alone, with the median and the **range across the 8
  pairs**. That range is the interval the decision reads. It is fixed here, before any data.
- **Bytes identical:** every rep's fused and nf4 bytes must agree, since the pool must be speed-only. If they differ, the
  probe exits 3 and the run is a finding, not a timing.

## The host and the run

- **Host:** a rented Vast.ai verified/secure single-GPU box on a **server-class SMT CPU** (AMD EPYC or Intel Xeon), any
  CUDA card. The box refuses (rc 16) unless it has 2 threads per core, at least 32 CPUs in the affinity and a quota of at
  least 16. It records the quota, so the 32-thread rep runs at the host's own quota, as Q235's did.
- **Pins:** e4b `E4B_PIN=6c5c27435c0e9c2f3d78e05ab102fd579434a7a1`, grouped-nf4-gemm 0.46.0
  `GNF4_SHA=6b7c2e7474d20c8843814dd46dad6cfc3d94b440`, torch 2.8.0+cu128 (the image's), bitsandbytes 0.50.2.
- **Phase cap, declared on the box** (`lane_record.phase_caps.probe = {"on_box": ...}`): min(time left − 600 s,
  1.5 × the 16 reps × the warm rep's own wall), checked between reps.
- **Run ids:** `smt-fuse-1` is the reading. `smt-fuse-2` and `-3` are spares for pre-create refusals, listings gone
  before the create, and NOT_RUN at the pre-flight. `smt-fuse-dry-<n>` are the $0 dry runs: correctness only, on 8
  experts, never a reading.
- **Cost:** guard 1.0 h at a $0.60/h cap, download at most $0.011/GB. **Worst $0.64** ($0.60 + about 3 GB of wheels).
  Expected under $0.20. Under the standing per-lane rule.
- **Launches** use the launcher current at launch, `--validate` passing (including the shared preflight's phase-cap and
  ceiling checks), the market rule's PIN, then CREATED or REFUSED, and the receipt committed within minutes of teardown.
