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
- **Corrected by running** (the $0 wiring run below): the allocator reuses part of the memory a block freed, so a
  4 KiB block faults 42-95 % of the fresh count (median 74 %), about 0.86-2.04 M at 128 experts. Even the low end is
  about 10x the fast block's 0.08 M.

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
  so at the block's peak about 0.3 x the block's need stays free. The need is the first M process's VmHWM plus the
  slab's 4.83 GB of page cache: the floor of 1.3 x VmHWM, reached from above. (`memory.peak` is reported, not used: it
  is the cgroup's lifetime high-water mark, the install's and the slab write's page cache included.) It asks whether
  that floor keeps `offload` off Q235's slow path, at this block's scale.

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
  - **B:** 15 M blocks with the balloon held.
  - **Every kind** (M, N, P and B) loads R's resident slab at process start, untimed, so every process has the same
    footprint, and the windows that M's blocks define apply to N's and P's.
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
- after `fuse`, before the fused bytes are moved: `AnonHugePages` and `Anonymous` from `/proc/self/smaps_rollup`
  (reported only: the wiring run showed the anonymous growth over the fuse is the two stacks alone, and none at all
  when the allocator reuses freed memory, so the share cannot be read; READ-2 reads the THP fault counters);
- over `fuse` and over `offload`, `/proc/vmstat` deltas: thp_fault_alloc, thp_fault_fallback, thp_fault_fallback_charge,
  compact_stall, compact_fail, compact_success, pgfault, pgmajfault, pgscan_direct, pgsteal_direct, allocstall_*,
  numa_hint_faults and numa_pages_migrated.

**Per block, before it starts:**
- `/proc/buddyinfo`'s free blocks of order 9 and above, per node: the fragmentation the hypothesis names;
- MemAvailable, MemFree, Cached and AnonHugePages;
- memory PSI (some and full, total-microsecond deltas over the block);
- `/proc/stat` steal over the block;
- cgroup v2 `memory.current`, `memory.events` and `memory.stat` where readable;
- `OMP_NUM_THREADS` and the process's CPU affinity (beside torch's pool, which the block already records), so a
  thread-count story and a fragmentation story read apart on one box's data.

**Once:** THP `enabled` and `defrag`, khugepaged `defrag`, `numa_balancing`, NUMA nodes, kernel, MemTotal, the cgroup
`memory.max`, `THP_MEM_ALLOC_ENABLE` (torch's switch for `MADV_HUGEPAGE` on its CPU allocations), and the
`bench/common/host_facts.py` record beside the receipt.

**Engagement, a closed form each:**
- an N process must read `PR_GET_THP_DISABLE` back as set, and every N block's fuse must gain under 64 MiB of huge
  pages (`AnonHugePages` after fuse less before the block), or N is VOID. A reading that is missing VOIDs N too: the
  check fails closed. The `PR_GET_THP_DISABLE` read-back is what carries N's engagement: when the allocator
  reuses freed memory the fuse gains no huge bytes whether or not its pages were huge, so the gain bound only
  catches a gross leak;
- a P process must carry the registered `GLIBC_TUNABLES`, and the **median** of its timed blocks' fuse minor faults
  must be under 0.4 M (a fifth of the 4 KiB count), or P is VOID. The median, so a block that grows the heap once does
  not void the arm; the maximum is reported;
- an R block's `read` must touch no file (no slab mapping in the process during R blocks), and the probe must see the
  mapping where it exists: every M process has an M or M' block that shows the slab mapped. Otherwise R is VOID.

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
- **The THP mode bounds what READ-1 can say.** Under THP `enabled` = `madvise` or `never`, or with THP absent (a kernel
  built without it, as on the lab A2000: no THP sysfs, no `thp_*` counters), with `THP_MEM_ALLOC_ENABLE` unset, torch's
  CPU allocations never get huge pages in any window (c10's `alloc_cpu` is `posix_memalign` with no `MADV_HUGEPAGE`). N
  is then the same as M, and READ-1 can read REFUTED_THP or a NOT_READ but never IDENTIFIED. A FAST window then has
  another cause, and P still prices the fault path.

**READ-2, the counters (corroborating gate):** over M and M' blocks, with at least 12 SLOW and 12 FAST, all three must
hold:
- the median fuse minor faults of SLOW blocks are at least **5x** those of FAST blocks (predicted about 26x);
- the median `thp_fault_alloc` delta over `fuse` is larger in FAST blocks;
- the median `thp_fault_fallback` delta over `fuse` is larger in SLOW blocks.

A kernel without THP counters (no `thp_*` in `/proc/vmstat`) reads READ-2 NOT_READ, never a FAIL.

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

**Its scope:** the block's VmHWM is about 21 GB, about 1/12 to 1/17 of Q235's 256-365 GB load, and B holds the
1.3 x floor, not Q235's 4 % free. So "the floor holds offload" speaks to the floor at this block's scale, and is
neither a reproduction of Q235's slow path nor a test of it. Many hosts expose `memory.max = max`, so NOT_RUN is a
likely outcome, not a surprise.

`fuse` B/M is reported beside it. Under a cgroup limit the balloon reclaims the slab's page cache first, so B's `read`
and `fuse` include disk reads. `offload` copies tensors already in memory, so READ-4 is not touched by it.

**Reported, never read:**
- every arm's per-segment medians;
- N over M and P over M in each window kind;
- the windows' timeline, i.e. which hosts were in a slow window at all;
- every counter, by state and arm.

## Predictions (graded, never ruling)

| prediction | band |
|---|---|
| on machine 11541, both a SLOW and a FAST window occur among the 13 M processes | yes |
| fuse minor faults, SLOW blocks | 0.8-2.4 M (closed form 2.04 M at fresh pages; the wiring run measured 42-95 % of it) |
| fuse minor faults, FAST blocks | 0.05-0.4 M |
| `thp_fault_alloc` over `fuse`, FAST / SLOW | at most 3,840 (every fuse page fresh; reuse lowers it, to roughly 1,600-3,650 at the wiring run's 42-95 %) / about 0 per block (host-wide counters) |
| N in a FAST window | SLOW (fuse >= 3.0 s) |
| P in every window | fuse <= 1.0 s |
| R against M in triples where M and M' agree | 0.8-1.25 |
| the M/M' flip base rate | <= 0.1 |
| READ-4 (if B runs) | the floor holds offload |
| THP `enabled` on the host | `always` (`madvise` would leave torch's allocations on 4 KiB pages in every window) |
| READ-1 | PARTIAL is likelier than IDENTIFIED: a FAST window held 49 of `smt-sib-2`'s 360 blocks (about 14 %), so an N process inside one is far from certain with 4 N processes. PARTIAL with P fast in a slow window (faults are the cost; the switch unread) is a good outcome, never a reason to re-run under `p159-2` |

## The $0 wiring run (`p159-dry-1`, 2026-10-20)

On the lab RTX A2000 (sm_86) under a `gpu:a2000` claim. The baseline read in the claim's own step showed the card idle
(1,526 MiB, 0 %, no compute process), and it read the same at release. The venv held the registered stack (torch
2.11.0+cu128, grouped-nf4-gemm 0.45.0, transformers 5.19.0, bitsandbytes 0.50.2) with e4b `d65793e4`'s tree. It ran 8
experts, all 25 processes, rc 0, bytes identical throughout: wiring only, never a reading. Its record is in
[`dryrun/`](dryrun/).
- **Confirmed by running:**
  - P's heap: 4,585 fuse faults a block against the slab-only closed form of 4,608, and no anonymous growth.
  - R: the slab is mapped in none of R's 65 blocks and in all 130 M and M' blocks (the positive control).
  - N: `prctl` reads back as set in all 4 processes.
  - B: NOT_RUN under no finite `memory.max` (the safe path).
  - The fit's process start: at most 2.1 s, under the registered 3 s.
- **Corrected by running:** the fault count's reuse (above), and READ-2's huge-page clause, from the `smaps` share to
  the THP fault counters. That kernel has no THP, so the huge-page path itself is read on the rented box.

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
  - the resident slab's copy: 15 s per process, in every kind (not yet measured; taken high);
  - the balloon: 120 s per B process (not yet measured; taken high);
  - the reserve: 600 s.

  Then 25 processes x 442 s (3 + 16 x 26.5 + 15), plus 4 x 120 s (B) give **NEED = 11,530 s**. So FIT = 12,475 s and
  the runtime check needs **15,593 s** (FIT x 1.25).
- **The static fit**, (boot 300 s + 12,475 s) x 1.25 = 15,969 s, sits within the **guard of 5.0 h** (18,000 s). The
  lane's own test asserts it.

## Run ids and cost

- **`p159-1` is the reading.**
  - `p159-2` is a second reading on a different machine. It is allowed only if `p159-1` reads NOT_READ (state not
    observed), and never to repeat a read state.
  - `p159-3` and `p159-4` are spares, for refusals before any instance exists and for NOT_RUN at the pre-flight only.
  - `p159-dry-<n>` are $0 wiring runs on 8 experts, never a reading. On a dry run B's balloon holds at most 1 GiB, so
    a wiring run never pressures a shared lab host. One A2000 wiring run (CPU-side counters, the N and
    P engagement checks) comes first, under a `gpu:a2000` claim with the baseline read.
- **Cost:** guard 5.0 h at a $0.60/h cap, download at most $0.011/GB:
  - **worst $3.03 per created box** ($3.00 plus about 3 GB of wheels);
  - **lane worst $12.13** over four ids ($3.033 x 4), inside the per-lane $15.
- **Launches** use the launcher current at launch, `--validate` passing, then CREATED or REFUSED, with the receipt
  committed within minutes of teardown.

## Amendment 2 (2026-10-20, before p159-2; the maintainer's ruling, recorded before any p159-2 data)

**Why.** `p159-1` ran on Vast machine 13450, under the host-order ruling, and read **VOID_BYTES**:

- the fused bytes equalled `FUSED_SHA` in all 315 blocks;
- the NF4 bytes differed in every block (315 distinct digests, none equal to `NF4_SHA`).

The same torch 2.11.0+cu128 and bitsandbytes 0.50.2 gave the registered NF4 bytes in every block on machines 11541 and
10447, and on the A2000 wiring run. The cause is not established. That host's THP mode was also `madvise`, which bounds
READ-1 (see "The THP mode bounds what READ-1 can say"). `p159-1`'s `result.json` is kept, copied unchanged from the
store, at `receipts/p159-1/result.json` (sha256 `d3c8b16b…`).

1. **Eligibility.** `p159-2`, on machine 11541, runs after `p159-1` reads NOT_READ **or any VOID_\***.
   - An instrument void is neither a reading nor a strike.
   - No new id: `p159-2` is the rerun. `p159-3` and `p159-4` keep their pre-create and pre-flight scope.
2. **The THP pre-flight refusal.** `p159_box.py --plan` refuses unless THP `enabled` is `always` or
   `THP_MEM_ALLOC_ENABLE` is set.
   - This is NOT_RUN at the pre-flight (rc 16), before the install, and not a strike.
   - The refusal record states the mode it saw (`absent` on a kernel without THP).
   - A dry run is exempt: it reads no state.
3. **The early bytes gate.**
   - After the first M block of the first M process, the fused digest must equal `FUSED_SHA` and the NF4 digest
     `NF4_SHA`.
   - Otherwise the box stops at once with VOID_BYTES (rc 3, `P159_BYTES_GATE` in the summary). The partial record is
     kept, and the reducer reads it.
   - That is about 10 minutes into the box, not after 25 processes.
   - On `p159-1`'s real `result.json` the gate fires on block 1 (the test asserts it).
   - Only the registered expert count is gated; a dry run's bytes are still checked for consistency at the end.
4. **Forensics.**
   - The box records the driver and VBIOS (`nvidia-smi --query-gpu=driver_version,vbios_version`) and an excerpt of
     `nvidia-smi -q -d ECC,PAGE_RETIREMENT,ROW_REMAPPER,PERFORMANCE`: ECC mode and error counters, retired and
     remapped memory, and clock-event reasons.
   - Each call is bounded at 30 s, so it fits inside the reserve.
   - It is evidence only, never a refusal.

**Unchanged:** the reads, the floors and the fit (the gate only ends a box earlier), the caps and the ids. The lane
worst stays $12.13; `p159-1` spent $1.163 of it.
