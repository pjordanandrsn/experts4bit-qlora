# P159: what makes the loader's host-side expert fuse run 17x slower in some host windows?

**Status:** registration, with its kit: `p159_box.py` (the block, the arms, the plan), `p159_reduce.py` (the reads),
`p159_run.sh` and `p159_drive.sh` (box and controller), `staged-p159.sha256`, and `tests/test_p159.py`. No box runs
before one independent review and the maintainer's final pass; a $0 A2000 wiring run comes first.

## Why

- **The 17x state.** On `smt-sib-2` (the private receipt store; machine 11541, AMD EPYC 7B13, 2 sockets, 503 GB), the
  registered block's `fuse` segment ran at **6.4-11.2 s** a layer in 311 of 360 timed blocks and at **0.35-0.5 s** in
  49. The fast blocks form one wall-clock window: rep 11 from its block 6, all of reps 12 and 13, and rep 14 to its
  block 9. That is four consecutive processes, across all three of that run's arms (PHYS, SIB, WIDE), with the same
  slab and identical bytes. Then the host went slow again. `read` was a flat 0.01 s in both states.
- **The fast host.** `smt-fuse-2` (machine 10447, AMD EPYC 7V13) ran the same block fast in all 16 blocks (0.21-0.85 s,
  median 0.35 s), with the same fused and nf4 bytes.
- **Headroom does not fit the fuse state on its face.** At start, `smt-sib-2`'s host had 438 GB available of 503 (87 %)
  and `smt-fuse-2`'s had 264 of 1007 (26 %). The slow host had more headroom. These are single start snapshots, and
  neither kit recorded a cgroup memory limit.
- **Q235's offload.** Q235's own load (a $0 re-read of the `-6` and `-8` receipts) shows a different pattern. `offload`
  tracks free host RAM: 415 s at 4 % free (`-6`, 267 GB, load VmHWM 256 GB) against 114 s at 32 % free (`-8`, 540 GB,
  VmHWM 365 GB), while the checkpoint read and the host-side assemble were flat. **This registration treats the two as
  separate questions:** the fuse's 17x state (primary) and offload against headroom (secondary).

## The hypothesis, and why these arms

**Primary:** the slow state is the **fault cost of fresh anonymous memory**, and whether huge pages are **available**
switches it on and off. That availability is host-wide and moves with memory fragmentation over time, so it would
produce windows that span processes and arms.

The arithmetic behind it (not a measurement). One block's `fuse` (`fuse_experts` at e4b `d65793e4`: a `torch.cat` per
expert, then two `torch.stack`s) allocates **7.5 GiB of fresh anonymous memory**: 128 temporaries of 24 MiB (3 GiB,
all alive until the stack), the gate_up stack (3 GiB) and the down stack (1.5 GiB).
- At 4 KiB pages that is **1,966,080** first-touch faults; at 2 MiB pages, **3,840**.
- At a few microseconds a 4 KiB fault (fault, zero, map), the first is about 6-8 s and the second well under 1 s. That
  matches the two states.
- The slab is read through `safe_open(...).get_tensor`, which maps the file, so its first touch also lands in `fuse`.
  The page cache is warm (the slab was written on the box), and fault-around maps it in about 74,000 faults either way.
- **So the process's minor faults in `fuse` are predicted at about 2.04 M in a slow block and about 0.08 M in a fast
  one, about 26x.** It is not 512x, because the slab's file faults count in both.

Correlation cannot carry this alone, so two arms **manipulate the mechanism**, and both need no root:
- **N (no huge pages):** the process sets `prctl(PR_SET_THP_DISABLE, 1)` before torch is imported. If 4 KiB faults are
  the slow state's cost, **N reads SLOW even inside a fast window.**
- **P (pre-faulted):** the process runs with
  `GLIBC_TUNABLES=glibc.malloc.mmap_max=0:glibc.malloc.trim_threshold=<2^62>`. The allocator then serves every block
  from one heap it never returns, so the untimed warm block faults the arena once and the timed blocks reuse it. If
  faults are the cost, **P reads FAST even inside a slow window.**

N slow in a fast window, together with P fast in a slow window, identifies the mechanism. The counters corroborate the
call and do not carry it alone.

Two more arms:
- **R (slab residency):** the slab fully read into anonymous memory, untimed, at process start. R blocks fuse from it,
  with no file mapping. If the slab's first touch were the cost, R would read FAST in a slow window. Under the primary
  hypothesis R tracks M, since the 7.5 GiB of fresh output still faults.
- **B (headroom, secondary):** a balloon inside the box's own cgroup holds memory down to the **registered RAM floor**,
  so at the block's peak about 0.3 x the block's need stays free. The need is the cgroup's peak usage above its idle
  level (`memory.peak`), else the first M process's VmHWM plus the slab's 4.83 GB of page cache: the floor of
  1.3 x VmHWM, reached from above. It asks whether that floor keeps `offload` off Q235's
  slow path.

## What runs

- **The block** is the registered one (`smt_fuse_box.one_rep`): its read, fuse, h2d, quant and offload steps, in that
  order, on the same seeded 128-expert bf16 slab (4.83 GB, gate/up [1536, 4096], down [4096, 1536]).
  - `p159_box.py` carries the same calls plus counter reads at the segment boundaries. The counter reads sit outside
    every segment's timed span, so they land in the residual.
  - **Every block's fused and nf4 sha256 must equal the registered constants** that `smt-sib-2` and `smt-fuse-2`
    produced on two hosts: fused `099fa1099154a74f10f6f65af01d94af18d9b29949f210250599589f7c944d3d`, nf4
    `78784a7094e4d9fe3f43c9e74e4f9f734323aca515496497f12bd7694b5956b9`. A mismatch exits 3: a finding, not a timing.
  - The arms change only how memory is provided, never the bytes.
- **Processes.** Each process is fresh, runs one untimed warm block, then **15 timed blocks**. Five kinds of process:
  - **M:** 5 triples of (M, R, M'), rotated by one position per triple, so each arm takes each position. M and M' are
    identical (the null); R uses the resident slab.
  - **N:** 15 blocks with THP disabled for the process.
  - **P:** 15 blocks on the pre-faulted heap.
  - **B:** 15 M blocks with the balloon held. B also loads R's resident slab, so its footprint matches an M
    process's.
- **The pool:** every process is pinned to K physical cores, one hardware thread each, pool K, where K = min(16,
  physical cores, floor(quota / 1.25)). A host with K under 8 is refused before the install.
- **Order:** 4 cycles of M, X, M, Y, M, Z, then one closing M, so every non-M process sits between two M processes.
  (X, Y, Z) = (N, P, B), (P, B, N), (B, N, P), (N, P, B) by cycle. That makes 25 processes: 13 M, 4 N, 4 P, 4 B.
- **B runs only when it is safe:**
  - the cgroup's `memory.max` is finite;
  - the balloon fits in 96 GiB.

  Otherwise B's slots are not run and READ-4 is NOT_RUN. The balloon never pressures memory the box was not given.
- **The pins are Q235's stack, as the SMT fuse lane ran it:** e4b `d65793e4`, grouped-nf4-gemm 0.45.0 `724ccc45`, and
  the hash-locked wheel set `84f361ed…` (torch 2.11.0+cu128) in a python3.11 venv.

## Recorded

**Per block, at each segment boundary:**
- the process's minor and major faults (getrusage);
- after `fuse`, before the fused bytes are moved: `AnonHugePages` and `Anonymous` from `/proc/self/smaps_rollup`, i.e.
  the huge-page share of the fuse's own allocations;
- over `fuse` and over `offload`, `/proc/vmstat` deltas: thp_fault_alloc, thp_fault_fallback, thp_fault_fallback_charge,
  compact_stall, compact_fail, compact_success, pgfault, pgmajfault, pgscan_direct, pgsteal_direct, allocstall_*,
  numa_hint_faults and numa_pages_migrated.

**Per block, before it starts:**
- `/proc/buddyinfo`'s free blocks of order 9 and above, per node: the fragmentation the hypothesis names;
- MemAvailable, MemFree, Cached and AnonHugePages;
- memory PSI (some and full, total-microsecond deltas over the block);
- `/proc/stat` steal over the block;
- cgroup v2 `memory.current`, `memory.events` and `memory.stat` where readable.

**Once:** THP `enabled` and `defrag`, khugepaged `defrag`, `numa_balancing`, NUMA nodes, kernel, MemTotal, the cgroup
`memory.max`, and the `bench/common/host_facts.py` record beside the receipt.

**Engagement, a closed form each:**
- an N process must read `PR_GET_THP_DISABLE` back as set, and an N block's fuse must gain under 64 MiB of huge pages
  (`AnonHugePages` after fuse less before the block), or N is VOID;
- a P process must carry the registered `GLIBC_TUNABLES`, and a P timed block's fuse minor faults must be under 0.4 M
  (a fifth of the 4 KiB count), or P is VOID;
- an R block's `read` must touch no file (no slab mapping in the process during R blocks), or R is VOID.

An arm that fails its engagement check is VOID, and its side of every read is NOT_READ. On a dry run, the two byte
and fault bounds scale with the experts (x experts / 128).

## What is read (fixed here, before any data)

**Block state** comes from `fuse` seconds per layer:
- FAST at or below **1.5 s**;
- SLOW at or above **3.0 s**;
- MID in between (reported, never read).

Every block recorded so far sits at 0.21-0.85 s or 6.4-11.2 s.

**A window** is the state of the M and M' blocks in the last two triples of the M process before an N, P or B process,
and the first two triples of the M process after it. It is FAST if all 8 blocks are FAST, SLOW if all 8 are SLOW, and
MIXED otherwise. A process in a MIXED window is reported and not read.

**The null (M vs M'), the instrument:**
- **The flip base rate:** the share of triples whose M and M' differ in state. Reported.
- **VOID_INSTRUMENT** if, over triples where M and M' agree, the median |log(M'/M)| of `fuse` exceeds log 1.5.

**READ-1, the mechanism (primary):**
- s_NF is the share of SLOW blocks among N processes in FAST windows.
- f_PS is the share of FAST blocks among P processes in SLOW windows.
- Tested in this order; the first that holds is the read:
  1. **NOT_READ (state not observed):** no SLOW block among M and M'. Not a NO_GO: N's blocks still price the 4 KiB
     path on this host, as N over M, reported.
  2. **REFUTED_FAULTS:** f_PS <= 0.2. Pre-faulted memory stays slow in a slow window.
  3. **REFUTED_THP:** s_NF <= 0.2. No huge pages, and still fast in a fast window: THP is not the switch.
  4. **IDENTIFIED:** s_NF >= 0.8 AND f_PS >= 0.8, each over at least one process. The slow state is the fault cost of
     fresh anonymous memory without huge pages.
  5. **PARTIAL:** one side is in its band and the other has no window of its kind.
  6. **NOT_READ (no window):** neither an N process in a FAST window nor a P process in a SLOW window.
  7. **UNRESOLVED:** otherwise (a share between 0.2 and 0.8).

**READ-2, the counters (corroborating gate):** over M and M' blocks, with at least 12 SLOW and 12 FAST, all three must
hold:
- the median fuse minor faults of SLOW blocks are at least **5x** those of FAST blocks (predicted about 26x);
- the median `AnonHugePages` share after fuse is at most 0.1 in SLOW blocks and at least 0.5 in FAST blocks;
- the median `thp_fault_fallback` delta over `fuse` is larger in SLOW blocks.

**The call:**
- IDENTIFIED with READ-2 passing is **IDENTIFIED_CORROBORATED**.
- IDENTIFIED with READ-2 failing is **CONFLICT**: a finding, reported with both.
- IDENTIFIED with READ-2 unreadable (fewer than 12 SLOW or 12 FAST M and M' blocks) is **IDENTIFIED_UNCORROBORATED**.
- READ-2 alone never carries the call.

**READ-3, slab residency:** over triples where M and M' are both SLOW (at least 6), the share whose R is FAST.
- At least 0.8: the slab's first touch carries the state, against the primary hypothesis.
- At most 0.2: **not the slab's residency**.
- Otherwise UNRESOLVED. Fewer than 6 such triples is NOT_READ.

**READ-4, headroom at the registered floor (secondary, a separate question):** each B process's median `offload` over
the mean of its two bracketing M processes' M-block medians, over at least 3 B processes.
- All ratios above **1.25**: offload slows at the 1.3 x VmHWM floor.
- All ratios at or below 1.25: **the floor holds offload**.
- Otherwise UNRESOLVED.

`fuse` B/M is reported beside it.

**Reported, never read:**
- every arm's per-segment medians;
- N over M and P over M in each window kind;
- the windows' timeline, i.e. which hosts were in a slow window at all;
- every counter, by state and arm.

## Predictions (graded, never ruling)

| prediction | band |
|---|---|
| on machine 11541, both a SLOW and a FAST window occur among the 13 M processes | yes |
| fuse minor faults, SLOW blocks | 1.6-2.4 M (closed form 2.04 M) |
| fuse minor faults, FAST blocks | 0.05-0.4 M |
| AnonHugePages share after fuse: FAST / SLOW | >= 0.5 / <= 0.1 |
| N in a FAST window | SLOW (fuse >= 3.0 s) |
| P in every window | fuse <= 1.0 s |
| R against M in triples where M and M' agree | 0.8-1.25 |
| the M/M' flip base rate | <= 0.1 |
| READ-4 (if B runs) | the floor holds offload |
| THP `enabled` on the host | `always` (`madvise` would leave torch's allocations on 4 KiB pages in every window) |

## Floors, fit and guard

- **The host:**
  - the market rule's PIN, with **machine 11541 first** (the only host with a recorded slow window);
  - any CUDA card with at least 16 GB;
  - MemTotal at least 64 GiB;
  - MemAvailable at least 32 GiB at start. That is about 1.3x the predicted process VmHWM of about 21 GB: the 7.5 GiB
    fuse peak, R's 4.83 GB resident slab, a 4.83 GB pinned home, and the torch/CUDA context.

  These are refusals before the install.
- **The guard fit** follows `bench/common/README.md` ("Guard fit"). It is the canonical pre-install check, with
  `FIT_HEADROOM_PCT=125`. Every term is the max over hosts measured so far, plus 25 %:
  - the install and setup to the gate: 270 s, plus the slab, 75 s (`INSTALL_TO_GATE_S` = 345 s);
  - a process start: 3 s;
  - a block's whole wall: 26.5 s;
  - R's slab copy: 15 s per M process (not yet measured; taken high);
  - R's slab copy also in each B process, which loads it too;
  - the balloon: 120 s per B process (not yet measured; taken high);
  - the reserve: 600 s.

  Then 13 M processes x 442 s, plus 8 x 427 s (N, P), plus 4 x 562 s (B) give **NEED = 11,410 s**. So FIT = 12,355 s
  and the runtime check needs **15,443 s** (FIT x 1.25).
- **The static fit**, (boot 300 s + 12,355 s) x 1.25 = 15,819 s, sits within the **guard of 5.0 h** (18,000 s). The
  lane's own test asserts it.

## Run ids and cost

- **`p159-1` is the reading.**
  - `p159-2` is a second reading on a different machine. It is allowed only if `p159-1` reads NOT_READ (state not
    observed), and never to repeat a read state.
  - `p159-3` and `p159-4` are spares, for refusals before any instance exists and for NOT_RUN at the pre-flight only.
  - `p159-dry-<n>` are $0 wiring runs on 8 experts, never a reading. One A2000 wiring run (CPU-side counters, the N and
    P engagement checks) comes first, under a `gpu:a2000` claim with the baseline read.
- **Cost:** guard 5.0 h at a $0.60/h cap, download at most $0.011/GB:
  - **worst $3.03 per created box** ($3.00 plus about 3 GB of wheels);
  - **lane worst $12.13** over four ids ($3.033 x 4), inside the per-lane $15.
- **Launches** use the launcher current at launch, `--validate` passing, then CREATED or REFUSED, with the receipt
  committed within minutes of teardown.
