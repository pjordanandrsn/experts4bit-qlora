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
- **One rep is the loader's own path, every step of `-7`'s one timed block** ("fusing + quantizing ... offloaded to
  pinned CPU RAM"; the review pass named the two this first omitted), timed wall-clock from the first read to the
  offloaded layer:
  - `read`: each expert's three tensors from a safetensors file on the box's disk through
    `safe_open(..., device="cpu").get_tensor`, as the loader's host staging reads them. The page cache is warm after the
    first rep; the cold read of a real 474 GB checkpoint is a different term and is not measured here;
  - `fuse`: `experts4bit_qlora.arch.moe_conventions.fuse_experts` on the host, the op torch's intra-op pool runs;
  - `h2d`: the fused stacks to the GPU;
  - `quant`: `Experts4bit.from_float(..., quant_type="nf4")` on the GPU;
  - `offload`: `ExpertsLoRA` around it and `enable_expert_offload(pin=True)`, the pinned host home and the copy back. The
    cached pinned blocks are released after each rep, so every rep pays a fresh pin, as each real layer does.
  - The pool is set explicitly (`torch.set_num_threads`) per rep.
- **The whole block and its residual** (the maintainer's addition): each rep also records its whole-block wall, from the
  first read to the layer's memory freed (the real loader frees each layer's GPU copies and staging too), less the two
  integrity hashes, and the **residual**, that wall minus the five segments. A breakdown without a residual cannot tell
  whether an untimed step is what made `-7` 3.2x slower.
- **Order:** one untimed warm rep at each pool, then the pools **16, HIGH, HIGH, 16** for **4 cycles**: 16 reps and 8
  back-to-back (16, HIGH) pairs.
- **The high arm** (the review pass): HIGH = min(32, ceil(quota), affinity). The loader caps the expert loop's pool at
  ceil(quota) (#1752), so on a host with a 16–31 CPU quota (`-7`'s was 30.72) a 32-thread arm would be CFS-throttled and
  measure a pool no real load runs. HIGH is the pool a real load would run there if torch's default rose from physical
  to logical CPUs. A host where HIGH would be under 24 is refused. Each rep also records the cgroup's
  throttled-period share and throttled CPU time over the rep. **A rep throttled in more than 5 % of its CFS periods is
  flagged; its pair is reported but left out of the ratio's median and range.**
- **Recorded once:** the CPU model, physical and logical counts, threads per core, the affinity, the cgroup quota and
  torch's default pool.
- **Recorded per rep:** the pool torch reports, the 1-minute load average at its start, the segment walls, and the sha256
  of the fused bytes and of the nf4 bytes.

## What is reported

- **Per pool:** median seconds for each of the five segments, their sum, the whole block and the residual, each scaled
  to one layer (the amendment
  multiplies by 94 layers).
- **Each pair's HIGH/16 ratio**, for the total and for every segment, with the median and the **range across the 8
  pairs**. That range is the interval the decision reads. It is fixed here, before any data.
- **Bytes identical:** every rep's fused and nf4 bytes must agree, since the pool must be speed-only. If they differ, the
  probe exits 3 and the run is a finding, not a timing.

## The host and the run

- **Host:** a rented Vast.ai verified/secure single-GPU box on a **server-class SMT CPU** (AMD EPYC or Intel Xeon), any
  CUDA card. The box refuses (rc 16) unless it has 2 threads per core, at least 32 CPUs in the affinity and a quota of at
  least 24. A host whose high arm P would be under 24 is refused before the install (NOT_RUN, $0), since a 16-vs-under-24
  ratio is too weak to read. It records the quota and P.
- **Pins: Q235's own stack** (the maintainer's final pass: the premise must describe the loader Q235 runs). e4b
  `E4B_PIN=d65793e4a106c7db6ecfc219668a2d6881d207e5` (Q235's A15 source, which `q235-pro6000-7` ran), grouped-nf4-gemm 0.45.0
  `GNF4_SHA=724ccc454f006c1a46836e434e997f31f293747f` (`bench/q235/q235_drive.sh`), and **Q235's hash-locked wheel set**
  (`wheels.json`, sha256 `84f361ed…`, copied byte for byte from `d65793e4` with `q235_bootstrap.py` and
  `dq11_require_git.sh`): torch 2.11.0+cu128, transformers 5.19.0, bitsandbytes 0.50.2, in a python3.11 venv on the
  image Q235 uses. The fuse is a torch intra-op op, so torch is part of what is timed and is Q235's. Q235's pins; the load
  path is identical at `d65793e4` and main (families checked). The amendment does not move them.
- **Phase cap, declared on the box** (`lane_record.phase_caps.probe = {"on_box": ...}`): min(time left − 600 s,
  max(600 s, 1.5 × the sum of the 16 planned reps' walls, each from its own pool's warm rep on this box)), checked between
  reps. Both are the dry runs' corrections: a cap from the 16-thread warm rep alone tripped on the slower 32-thread reps,
  which are what this measures, and a pure 1.5 × cap on a sub-second slab tripped on a busy host's noise.
- **Run ids:** `smt-fuse-1` is the reading. `smt-fuse-2` and `smt-fuse-3` are spares for pre-create refusals, listings gone
  before the create, and NOT_RUN at the pre-flight. `smt-fuse-dry-<n>` are the $0 dry runs: correctness only, on 8
  experts, never a reading.
- **Cost:** guard 1.0 h at a $0.60/h cap, download at most $0.011/GB. **Worst $0.64** ($0.60 + about 3 GB of wheels).
  Expected under $0.20. Under the standing per-lane rule.
- **Launches** use the launcher current at launch, `--validate` passing (including the shared preflight's phase-cap and
  ceiling checks), the market rule's PIN, then CREATED or REFUSED, and the receipt committed within minutes of teardown.

## Amendment 1 (2026-10-19): the second hardware thread, pinned (`smt_sib_box.py`, ids `smt-sib-1..3`)

**Why.** `smt-fuse-2` (receipt store `1307dedc`, re-derived by the maintainer) read every HIGH/16 range across 1, and
the pool stays at torch's default. But its host had 128 physical cores under a 30.72-CPU quota, and nothing pinned the
threads, so the 31-thread arm could run one thread per physical core. It measured **the pool under a quota, not the
second hardware thread**, and must not be cited either way on that question (the maintainer's re-derivation says so).
The narrow answer is kept; this amendment asks the real question. It is still a **timing premise, never a ruling**.

**What changes.** Nothing in the registered `smt-fuse-*` reading or in `smt_fuse_box.py`. A new box module,
`smt_sib_box.py`, runs the **same block** (`smt_fuse_box.one_rep`, unchanged) in three pinned arms:

- **Pinning:** each rep runs in a **fresh process** whose CPU set is fixed with `sched_setaffinity` **before torch is
  imported**, so every intra-op worker inherits it (a running process's OpenMP workers keep the mask they were created
  with, which is why the pool cannot be re-pinned between reps in one process). Each process runs the block once untimed
  (allocator, CUDA context, bnb kernels), then **15 times timed; the rep is their per-segment median** (the review and
  final passes: one timed block per process cannot resolve the effect at `smt-fuse-2`'s spread, and 5 resolve only a
  fuse SIB/PHYS of about 0.65). The raw blocks are kept beside it; the
  rep's throttling is the worst of the fifteen, and its bytes count only if all fifteen agree.
- **The arms**, from the host's own sibling topology (`/sys/devices/system/cpu/cpuN/topology/thread_siblings_list`,
  cores whose two threads are both in the affinity, in CPU order):
  - **PHYS:** K physical cores, one hardware thread each, pool K;
  - **SIB:** the **same** K cores, both hardware threads, pool 2K;
  - **WIDE:** 2K physical cores, one hardware thread each, pool 2K.
- **K = min(16, floor(quota / 2.5))**, so the 2K-thread arms run at no more than 80 % of the quota and the cgroup does
  not bind before the cores do. On `smt-fuse-2`'s host (30.72) that is K = 12: 12 vs 24 threads.
- **Order:** one process per arm first (it sizes the cap), then **PHYS, SIB, WIDE, WIDE, SIB, PHYS** for **4 cycles**:
  24 reps and 8 triples, each arm once per triple.
- **Host noise, per rep:** the cgroup's throttled-period share (as `smt-fuse-2`), and from `/proc/stat` the busy share
  of the arm's CPUs and of its **companions** (the siblings of its cores that it does not use). A co-tenant on a PHYS or
  WIDE arm's companions turns it into a shared-core arm. **A triple is flagged when any rep in it is throttled in more
  than 5 % of its CFS periods, or a PHYS or WIDE rep's companions are more than 25 % busy. A flagged triple is reported
  but left out of the median and range.**

**What is read** (fixed here, before any data):

- **The reading:** each triple's **SIB/PHYS** ratio for **`fuse`**, the op the pool drives, with the median and the
  **range across the unflagged triples**:
  - range entirely **below 1**: the second hardware thread **helps** this op, by the median;
  - range entirely **above 1**: it **hurts**;
  - range **includes 1**: **no resolved effect**.
- **Resolution, stated before data, in ratio terms:** resampling `smt-fuse-2`'s own `fuse` reps (log spread 0.30 per
  block) into medians of 15 for each arm, a true **SIB/PHYS of 0.80, 0.75 or 0.70** (fuse time 20, 25 or 30 % lower)
  reads "below 1" in all 8 triples with probability 0.75, 0.91 and 0.97. **So a true fuse SIB/PHYS above about 0.80
  reads `includes_1` at least one time in four by construction, and a null reads "no measurable fuse effect larger than
  about 20 %: SIB/PHYS not resolved below about 0.80", never "no effect".** (Read as a speedup, 1/0.80 = 1.25x.) The
  same calculation on `block` (log spread 0.075) gives 0.77 at a SIB/PHYS of 0.95 and 0.90 at 0.93. At 5 timed blocks
  the fuse figure was about 0.65; that is why the count is 15. A within-process MISMATCH in any rep fails bytes
  identity. The min-max range rule is kept.
- **Time fit, tested** (`time_fit` in the self-test): every term is `smt-fuse-2`'s measured wall rounded up: setup
  240 s (209 measured), the slab 60 s, a block's wall 12.5 s (its probe phase, 217 s for the slab and 18 blocks, charged
  whole to the blocks, since each block also hashes 4.8 GB outside its timed span), a process start 30 s (not yet
  measured, taken high) and the 600 s reserve. 27 processes of 1 + 15 blocks need **7110 s**. That does not fit a
  1.0 h guard (nor did 5 blocks, at 3735 s), so the guard is **2.5 h**, which leaves 1890 s of margin, and the self-test asserts that
  at least 1800 s remain. The drive refuses an `smt-sib` launch whose deadline leaves less
  than that fit (read from `smt_sib_box.py --fit`), so the launch manifest's wallclock cannot drift from it.
- **Reported, never read:** SIB/PHYS on `block` (what the loader pays; offload dominates it and is flat with the pool),
  SIB/WIDE (what the second thread is worth against a real core; WIDE also spans more cache, so this ratio is
  confounded), every other segment's ratios, and the per-arm medians per layer.
- **VOID_BY_HOST:** fewer than **6** unflagged triples. The run is reported and a spare id reruns it.
- **Bytes identical:** every rep's fused and nf4 bytes must agree (exit 3 otherwise, a finding, not a timing).

**The host and the run.** As above (Q235's pins and wheel lock, the same install, the same on-box phase cap formula,
sized from each arm's whole process wall), plus a refusal before the install (rc 16, NOT_RUN, $0) unless K >= 8 and the
affinity holds at least 2K two-thread cores. The fetch no longer brings the box's venv back (`smt-fuse-2` fetched 6.8 GB
of it into the run directory, which made the receipt store read dirty for every lane until it was moved).

- **Run ids:** `smt-sib-1` is the reading; `smt-sib-2` and `smt-sib-3` are spares for pre-create refusals, listings gone
  before the create, NOT_RUN at the pre-flight and VOID_BY_HOST. `smt-sib-dry-<n>` are the $0 dry runs on 8 experts:
  wiring only, never a reading.
- **Cost:** guard **2.5 h** at a $0.60/h cap, download at most $0.011/GB. **Worst $1.53** ($1.50 + about 3 GB of
  wheels), expected about $0.90 (about 1.8 h at smt-fuse-2's $0.49/h). Under the standing per-lane rule. **Why the
  guard is 2.5 h: a block's wall is about 3x its timed `block_s`**, because `one_rep` hashes the 4.8 GB of fused and nf4
  bytes outside its timed span (the integrity check). The timed seconds are the reading; the wall is the cost.
- **The fit's terms are recorded on the box** for the next fit: each process's start (spawn to a pinned interpreter with
  torch and a CUDA context, the one term taken high here), its untimed block's wall and every timed block's whole wall;
  the summary carries their maxima and the wall/`block_s` ratio. Under the standing per-lane rule.

### Amendment 1a (2026-10-19): five more pre-create spares (`smt-sib-4..8`)

`smt-sib-1` was refused at the create (no_such_ask, $0, receipt store `b516bcbd`): the offer was taken between the read
and the create. The launcher walks the offers of ONE pinned machine within a run id, not several machines, so each
candidate machine still costs an id. **`smt-sib-4` to `smt-sib-8` are spares for refusals before any instance exists
(no_such_ask, a listing gone before the create) and NOT_RUN at the pre-flight, and for nothing else: never after the
probe's command has run, and never to repeat a reading.** One reading is still taken, on the first id that runs the
probe. Nothing else changes; the cap stays $1.56 per created box.
