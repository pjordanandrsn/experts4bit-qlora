# P159's $0 wiring run (`p159-dry-1`, 2026-10-20): the closed forms confirmed and corrected by running

**What ran.**
- **The card and the claim:** the lab RTX A2000 (sm_86, 12 GB) under a `gpu:a2000` claim, 05:54Z to 06:07Z. The
  baseline read in the claim's own step showed the card idle: 1,526 MiB used, 0 %, no compute process. It read the
  same at release.
- **The stack:** the registered one, from the card's own venv: torch 2.11.0+cu128, grouped-nf4-gemm 0.45.0,
  transformers 5.19.0, bitsandbytes 0.50.2, safetensors 0.8.0, triton 3.6.0, with e4b `d65793e4`'s tree on the path.
- **The run:** `p159_run.sh` in its dry mode (`P159_DRY=1`, 8 experts), with the staged files at
  `staged-p159.sha256`. The guard-fit check printed `fit=12475s need=15593s`. All 25 processes ran, rc 0, with bytes
  identical throughout (the dry run checks identity within the run; the 128-expert constants are the box's).
- **Wiring only, never a reading.** At 8 experts the per-layer scaling puts nearly every block near the FAST/SLOW
  thresholds, so `verdict_p159.json`'s verdict line means nothing. That file is the box's, from the reducer before the
  fold below; under the folded reducer READ-2 reads NOT_READ here (no THP counters).

**The host:** Intel Xeon W-1250 (6 cores, 12 threads), 125.7 GiB. The kernel has **no THP**: no
`/sys/kernel/mm/transparent_hugepage`, and no `thp_*` keys in `/proc/vmstat`. So every block ran on 4 KiB pages, and
the huge-page path is the rented box's to read. There is no finite cgroup `memory.max`.

**Confirmed by running:**

| check | closed form / rule | measured |
|---|---|---|
| P's pre-faulted heap | slab-only fuse faults, about 4,608 a block (288 MiB / 64 KiB) | 4,585-4,586 in all 60 P blocks; anonymous growth 0 |
| R's residency and its positive control | no slab mapping in R; mapped in M and M' | 0 of 65 R blocks mapped; 130 of 130 M and M' mapped |
| N's engagement | `PR_GET_THP_DISABLE` reads back set | true in all 4 N processes |
| B's safety | runs only under a finite `memory.max` | NOT_RUN: no finite limit here |
| the fit's process start | 3 s registered | at most 2.1 s |

**Corrected by running** (folded into the PREREG, the reducer and the tests):
1. **Fault reuse.** 4 KiB first-touch faults in `fuse` are 42-95 % of the fresh-page count, median 74 %. The M arm's 65
   blocks read a median of 94,444 (53,738-120,556) against a closed form of 127,488 (122,880 anonymous + 4,608 slab) at
   8 experts. The allocator reuses part of the memory a block freed. The SLOW prediction band moves from 1.6-2.4 M to
   0.8-2.4 M; the 5x gate keeps about 10x room at the low end.
2. **The huge-page share cannot come from `smaps`.**
   - The anonymous growth over `fuse` is the two stacks alone: N's median is exactly 288 MiB, the 4.5 GiB of stacks
     at 8/128. The temporaries are freed by then, and the growth is less or nothing when the allocator reuses memory
     (P: 0).
   - So a fully huge-paged block would read at most 0.6 against the old 0.5 bar, and a reused one 0.
   - READ-2 now reads the THP fault counters: `thp_fault_alloc` larger in FAST blocks, and `thp_fault_fallback`
     larger in SLOW. A kernel without them reads NOT_READ.

**Files:** `result.json` (the box's record), `verdict_p159.json` (the box's reduce) and `summary.txt` (the runner's
summary). `SHA256SUMS` covers them and this README.
