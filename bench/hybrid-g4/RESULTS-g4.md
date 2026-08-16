# RESULTS — G4: speculative prefetch (route L+1 from L) at Qwen3-235B

Gate G4: **≥50% NVMe-stall reduction at 10–15% NVMe routing mass; free when
unused.** Same box, artifact, and methodology as the formal G3 run
(`../hybrid-g3/RESULTS-g3-formal.md`): Qwen3-235B-A22B, 127 GB NF4 arena,
Zen 5 EPYC 9655 + RTX 5090, greedy 64-token decode. Placement solved at
`dram_gb=25` → **11.03% of routing mass on NVMe** (in-band; sweep in
`g4_sweep.py` output picked the budget). `hot_rows=128`, all tier I/O
O_DIRECT, so no page-cache leakage between arms. Arms are cold-started
(each rebuilds the tier: enable → warm → timed decode → disable) and run
as a B1/C/B2 sandwich so arm order cannot leak residency and drift is
visible.

**Stall metric = demand misses in the timed window**: synchronous NVMe
fetches on the serving thread's critical path, counted by the tier itself
(`demand_misses`, split from speculative fetches — a single miss counter
conflates background warming with stalls, which run 1 demonstrated).

## Final measurement (`g4_dram25.json`)

| arm | tok/s | demand misses | demand waits | wall |
|---|---|---|---|---|
| B1 prefetch off | 1.169 | 5,526 | 0 | 54.8 s |
| **C prefetch on** | 1.115 | **2,596** | 2,750 | 57.4 s |
| B2 off, recheck | 1.184 | 5,543 | 0 | 54.0 s |

**Demand-stall reduction = 53.1% (5,534.5 → 2,596) — gate G4 PASSES**
(band ok, reduction ok). Residency byte-verify after arm C's timed decode:
16/16 sampled rows bit-identical to direct disk reads, residency maps a
consistent bijection. (The run-as-measured sampled while the prefetch
worker was still live; that race can only manufacture FALSE corruption —
a reassigned slot cannot match the old key's freshly-read reference — so
the clean verdict stands. The committed runner now quiesces the worker
before verifying, so the check no longer needs that argument.) Free-when-unused: pinned by `tests/test_hybrid_prefetch.py`
(zero submits with no NVMe mass) and by B-arm parity with pre-Phase-4
numbers (1.16–1.18 tok/s here vs 1.17 before any prefetch code existed).

## What it took — three defects, each measured before fixed

1. **Eviction race (crash).** The first 235B arm C died with
   `KeyError: '(layer 81, expert 73) not resident'`: nothing protected the
   demand path's ensure → row-reads window from a concurrent speculative
   ensure evicting in between. Fixed structurally in gnf4's ColdTier
   (demand-window protection; grouped-nf4-gemm PR #102), regression-pinned
   by an eviction-pressure stress test on both repos.
2. **Lock convoy (−21.5% "reduction", 6.6× slower; run 1).** ensure() held
   the tier lock across its O_DIRECT fills, so each speculative fire
   serialized every demand fetch behind up to ~340 MB of disk time — and
   the interim external transaction lock doubled it. Rework: plan under
   lock → fill outside it → publish per row as each read lands (reserved
   slots + pending-key events keep concurrent ensures correct; a demand
   caller colliding with an in-flight speculative fill waits ~one row-read,
   4.7 ms median, not a batch tail).
3. **BLAS thread-team thrash (still 6.3× slower; runs 2–3).** With the
   convoy gone the tier accounted for only ~28 s of arm C's 362 s wall. A
   6,726-sample in-process profile put **78.5% of wall on `torch.stack`**
   in `segment_tensor` — a CPU parallel region — while disk and the
   prefetch worker sat idle. The worker's 0.5-MFLOP numpy `@` dispatched
   to OpenBLAS, which woke a full spinning thread team per fire (~5.8K
   fires/decode) on a 96-core host; the teams oversubscribed every core
   against torch's OMP pool and the gnf4 persistent pool, and every
   parallel-region barrier on the main thread inherited the cost. Same
   failure family as the documented torch-thread-thrash trap. Fix: the
   predictor is a single-threaded ufunc broadcast+reduce; no BLAS call
   ever fires on the worker. Receipts: `g4_dram25_run1_lockconvoy_miss.json`,
   `g4_dram25_run2_instrumented_miss.json`, `g4_run3_sampled_frames.json`.

## Honest caveats

- **Wall-clock is neutral at this operating point** (C 57.4 s vs B mean
  54.4 s, +5%): halved stalls are absorbed by the 2,750 waits on
  in-flight speculative fills (11.4 s) plus reader-queue sharing. The
  gate is defined on stall reduction and passes on its own terms; turning
  the halved stalls into wall-clock wins wants a deeper prediction
  horizon (L+2 at decode's ~58 ms/layer vs ~15 ms/row fetch) and/or
  demand-priority I/O in the reader — Stage-2 levers, named not begun.
- At `hot_rows=128` the ~85-row/token NVMe working set thrashes by
  construction (~47% of B-arm decode time is demand fill). That is the
  stall-rich regime the gate asks for, not a recommended serving point;
  the placement solver would assign more DRAM on any real deployment.
- B1 vs B2 tails still differ by a few tokens (the batch=1 `index_add_`
  atomics nondeterminism filed with the G3 formal results — independent
  of prefetch; arm C's outputs verified byte-clean at the residency
  layer, and prefetch changes when bytes load, never their values).
- The 45.9→53.1% reduction across runs 2→4 at identical placement partly
  reflects speculative-fetch scheduling variance (spec fetched 3,935 rows
  in the final run vs 3,469 in run 2); the sandwich baselines moved <1%.

Receipts in this directory; runner (`g4_run.py`, arms/sampling/verify
flags) and solver sweep (`g4_sweep.py`) committed alongside.
