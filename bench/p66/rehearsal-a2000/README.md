# P66 rehearsal on the NAS RTX A2000 — NOT A READING

These receipts prove the census plumbing and the counting on real CUDA. **They are not P66's reading.**
- P66 registers an RTX 5090 on a rented box, with Qwen3-30B-A3B and gpt-oss-20b ([`../P66-PREREG.md`](../P66-PREREG.md)).
- This card is a shared, production-loaded RTX A2000 12 GB on a NAS, with other families.
- No number here is a result, and the time columns least of all.

## The box

| | |
|---|---|
| GPU | NVIDIA RTX A2000 12GB, sm_86, 26 SMs, 70 W, driver 575.64.05 |
| link | PCIe gen3 × 8 electrically: H2D 6.28 GB/s pinned at 64 MB (calibrate.py `--quick`) |
| VRAM | 256.6 GB/s triad |
| host | Intel Xeon W-1250 (6C/12T), 125 GB RAM, QNAP kernel 6.6.32 |
| image | `pytorch/pytorch:2.8.0-cuda12.8-cudnn9-devel` (torch 2.8.0+cu128, triton 3.4.0, Python 3.11.13) |
| packages | experts4bit-qlora from this branch's tree, grouped-nf4-gemm main, both `--no-deps` into a target dir: package code identical to main `94842a2` / gnf4 `66d41c8` for rounds 1–6, and to main `8020489` / gnf4 `f88df1e` (after the rebase) for round 7 |
| sharing | production services held ~1.4 GB VRAM; the lane kept under 8 GB; one job at a time under the shared lock |

Host unit costs measured on it (`box.json` → `probes`): aten launch 6.8 µs, Triton launch 16.7 µs, idle
`cudaStreamSynchronize` 5.6 µs, `.item()` round trip 7.8 µs. The 5090 box's own figures will differ (#108: host
cost varied ~5× across boxes); that is why the reading records them.

## The families (chosen to fit this card, not the registration's)

- **OLMoE-1B-7B-0924, NF4.** 16 MoE layers × 64 experts, top-8, 3,538,944 B per row-block.
  - Arena baked on the NAS by the lane's own `k8_bake.py` from `/share/models/OLMoE-1B-7B-0924`: 1024 rows,
    3.6 GB, in 32 s after a 343 s quantizing load.
  - Paths: `ref`, `pipe`, `hyb`.
- **gpt-oss-20b, layers 0–7, MXFP4.** 32 experts, top-4, 13,221,888 B per row.
  - Relocation-baked by `nvme_arena.bake` from `/share/models/gpt-oss-20b`: 256 rows, 3.38 GB, in 24 s.
  - Eight layers, so all-resident fits the lane's 8 GB.
  - Paths: `mref`, `mpin`, `mnvme`.

## Round 1 (`round1/`): the instrument as first written

Warm 2 and 4 tokens per pass. Per layer, from `round1/rows_*.json` read by `p66_reduce.py`:

| path | hot | launches | copies | syncs | windows | fixed? |
|---|---|---|---|---|---|---|
| `ref` (all-resident collapse) | 1 | 8 | 0 | 0 | uniform | — |
| `pipe` | 1, 0.5 | 18 | 2 | 0 | uniform, ctrl0/4/8 (cold fraction 0, 0.5, 1) | yes, spread 0 eager and graph |
| `pipe` | 0 | 15 | 2 | 0 | uniform | yes |
| `hyb` | 0.5 | 41 / 75 / 50 | 7 / 20 / 15 | 4 / 10 / 8 | ctrl0 / ctrl4 / ctrl8 | **no** |
| `mref`, `mpin` | 1, 0.5, 0 | 38 | 3 | 1 | uniform, ctrl0/2/4 | yes |
| `mnvme` | 1, 0.5, 0 | 37 | 6 | 4 | uniform, ctrl0/2/4 | yes |

- **Against the reference.** `pipe` adds +10 launches and +2 copies per layer (+7 and +2 at hot fraction 0), and
  0 syncs.
  - The `_fetch` region holds 8 launches and 2 copies (5 and 2 at hot fraction 0).
  - By name: +11 torch launches, and −1 Triton (the gather in, the fused `_swiglu_rows` and `_combine_rows` out).
- **MXFP4 NVMe against `mref`**, per layer: −1 launch, +3 copies, +3 syncs. The fetch holds 3 of the 4 syncs.
- **The have-skip simulator** equals the pipelined engine's own counters in 6 of 6 windows: 272, 0, 245, 483,
  511 and 0 cold rows.
- **Graph replay** submits 1 `cudaGraphLaunch` plus 2 staging copies per token. The device kernels are the eager
  ones plus those 2 copies.
- **Transfer.** Measured gather device time over cold_deadline's `gpu_us`: 1.01 (`pipe`, hot 0.5) and 1.08 (hot
  0). The hybrid tier's pageable H2D read 1.06–1.28.
- **Capture probes.**
  - `mnvme` refuses capture by design: "Mxfp4NvmeResidency cannot be captured in a CUDA graph: residency is a
    host-side disk read…".
  - The `hyb` probe named nothing ("operation failed due to a previous error during capture").

**What round 1 changed in the instrument.** These are the six items in the registration's disclosure.
1. The sync detector counted one warning per process raised by torch's own `set_sync_debug_mode`, not an engine.
   Now only warnings raised while tokens execute count.
2. Graph parity: the MXFP4 engines' padded-index guard is eager-only by design (one compare, one bool reduce and
   one D2H copy per layer, identified by name). Gate G4 now accounts for exactly those three rows.
3. The hybrid capture probe now names the first synchronizing op, under `set_sync_debug_mode("error")`.
4. A graph-against-eager output gate (G3) was added.
5. The all-hot tax splits into the added kernels' device time (0.59 ms/token here, 192 rows) and the shift in
   kernels both paths run. The shift was +1.70 ms/token captured, the same `_gemv_nf4_grouped` count running
   slower on the engine's store; +0.01 ms eager.
6. Time bands are graded only on the registered box.

## Round 2: withdrawn

It was queued to run the changed instrument and Level M. Round 3 ran the same instrument through the runner
itself, so round 2's waiter was withdrawn before it ever held the GPU.

## Round 3 (`round3/`): `p66_run.sh` itself, end to end

`p66_run.sh` ran with its documented `P66_REHEARSAL_*` overrides:
- the A2000 class and minimums;
- no install, gnf4 main as importable;
- local snapshots, with the OLMoE arena from round 1's bake;
- gpt-oss layers 0–7;
- warm 2 and 4 tokens per pass.

`summary.txt` stamps them. It finished **rc 0 with its success marker and `TP_DONE` under the nonce**: the
calibration blob, every Level L arm, the MXFP4 bake, all four Level M step_decomp arms, and the reduce.

**Counts.** Every per-layer count is round 1's, including the new RFC-matched arm (`pipe-0.83`: 53 of 64 hot = 17.19 %
cold on OLMoE, 18 launches / 2 copies / 0 syncs) and hot fraction 0.25 on every path (`pipe`, `hyb`, `mnvme`).

**Gates.** All five of round 1's gates hold on these receipts: nesting 39/39, simulator 14/14, graph check 21/21
(every replay **bitwise** equal to eager), graph parity 21/21, budget coverage 39/39. The new G6 (record
completeness) holds too, with 17 dropped device records in total; see finding 4 below.

**Transfer**, measured link-crossing device time over cold_deadline's `gpu_us` at the box's own 6.25 GB/s:

| path | windows | ratio |
|---|---|---|
| `pipe` | uniform and controlled, hot 0.83 / 0.5 / 0.25 / 0 | 0.983–1.023 |
| `mpin`, `mnvme` | hot-row re-copy taken off | 0.949–1.008 |
| `hyb` | pageable H2D | 1.046–1.284 |

**The all-hot tax** (`pipe-1.00 − ref`, captured):
- The 192 added device rows cost **0.49 ms/token** (2.6 µs each).
- The shared `_gemv_nf4_grouped` moved **+0.04 ms**. Round 1 read +1.70 ms for the same shift, so that number was
  this shared card's clock state, not layout. That is what finding 5 of the registration anticipated.

**At the RFC-matched cold fraction** (informational, this host): the fixed tax is 2.4 % of residency's cost
captured, but **20 %** eager (1.00 of 5.01 ms/token). Eager on a 6.4 µs-per-launch host is where the launch count
shows.

**Level M against Level L.** The served step's per-step API deltas equal the MoE token's per-token deltas **exactly**:
- `M_pipe-1.00 − M_ref` = +176 `cudaLaunchKernel`, +32 `cudaMemcpyAsync`, 0 `cudaStreamSynchronize`;
- `M_pipe-0.00 − M_ref` = +128, +32, 0.

Level L's `by_api` deltas are the same numbers. The served step adds no syncs under pipelined residency (8.33 per
step in both).

**Capture refusals**, now named:
- hybrid: `hot_residency.py:664 forward: hr = hot_row.nonzero(as_tuple=False)`;
- MXFP4 NVMe: `mxfp4_pipelined.py:430 _forward_decode: if bool((want >= self.E).any())`, before `_resolve_src`'s
  own by-design refusal.

**What round 3 found and changed** (each fixed before registration):
1. **The override stamp missed three knobs.** The name pattern had no digits, so `P66_REHEARSAL_GNF4_SRC`,
   `_NF4_FAMILY` and `_NF4_ARENA` were not written to `summary.txt`: a rehearsal could have under-declared itself.
   The pattern admits digits now, and `tests/test_p66_staged_pin.py` asserts it.
2. **`calibrate.py` leaves an 8 GiB NVMe test file** (`hybrid_calib_nvme.dat`) in the work dir, and the driver's
   receipt fetch would have pulled it back. The runner deletes it after calibrating, and the driver excludes
   `*.dat`.
3. **Level M's served hybrid arm prefills 128 tokens in one chunk.** The tier refuses when one forward routes more
   distinct cold experts than it holds. With the solver choosing VRAM on a 128-expert family, 64 rows is not
   safe, so that context arm now gets 256.
4. **The profiler drops device records.** It happened only on the paths that synchronize:
   - MXFP4 NVMe: exactly one per window, the window's first kernel (the padded-index guard);
   - hybrid: 1–3 of ~4,160 rows per window.
   The host count was identical. The sync-free paths lost none. Round 3's first read therefore called P4
   REFUTED on a lost record. The fixes:
   - eager fixedness is graded on the **host** counts, which are exact;
   - G6 bounds the loss at max(1, 0.2 %) of submissions, about 2.8× the worst seen, and records it;
   - G4 now compares graph device rows with the eager **host** submissions.
5. **step_decomp's kernels table is row-limited.** It prints 80 rows, so its time coverage is 99.9 % while its call
   counts drop the smallest rows: a delta of 148.7 calls per step against the true 192. P9 compares the
   sync-attr API counts, which are complete.

**Not explained here (context only):** the served hybrid arm (`M_hyb-0.50`, 512 of 1024 experts in VRAM by the
solver) shows 2 `nonzero` per layer. That means most layer-steps took no cold branch, where Level L's mixed layers
take 4. The solver's placement, with no routing profile, against real routing is the likely reason. It is not
checked.

## Round 4 (`round4/`): the round-3 fixes, same runner path

The tree was round 3's plus findings 1–5 (every staged file passed `sha256sum -c` on the box). Its result: rc 0,
the success marker and `TP_DONE`, all 14 overrides stamped (finding 1 fixed), and no `hybrid_calib_nvme.dat` left
(finding 2 fixed).
- **Gates.** All six hold (G6 is new since round 3).
- **Level M.** It agrees with Level L exactly again.
- **Transfer.** Pipelined 0.983–1.068.
- **The shared-GEMV shift under capture read +1.37 ms/token**, against +0.04 in round 3 and +1.70 in round 1. The
  same code and card give three different values, which is this shared card's state, not the engine. That is why
  P3 and P7 are graded on the registered box only.

Committed here: `summary.txt`, `box.json`, and `read-regenerated.md` (the current reducer over the round's rows,
which are not committed; they reproduced round 3's counts exactly).

## Round 5 (`round5/`): the proving mode and the time guard

After round 4, two changes: the compute rule's proving rental (`P66_MODE=prove`), and a fix to the runner's time
guard. `can_run` had been given each arm's alarm cap as its expected need, so on a 2 h guard a late arm with a
40-minute cap would be skipped while needing three minutes. The need and the cap are now separate numbers, and a
test holds need < cap for every arm.
- **(a) `P66_MODE=prove` through the real install path** (`round5/prove/`): no `SKIP_INSTALL`.
  - The runner found no `git` in the image and installed it.
  - It pip-installed experts4bit-qlora and grouped-nf4-gemm from GitHub at main `94842a2` / `66d41c8`. The
    branch is not pushed; its package code is identical to main.
  - The tripwire saw both packages in `site-packages`, the 4 GiB pin passed, and the runner wrote
    `P66_PROVED.<nonce>` and `P66_SUCCESS.<nonce>`.
  - **85 s end to end**, inside the proving rental's 10-minute guard. The timed shard fetch was skipped by design
    (local snapshot, no WAN pull of a NAS model). Its shell was checked with stubs; the live `hf_hub_download`
    is not rehearsed.
- **(b) `P66_MODE=full`** (`round5/full/`): rc 0, success, `TP_DONE`, **1271 s**. It shows two things the earlier
  rounds did not.
  1. **The profiler's device view loses records in captured windows too.** In three pipelined graph windows, one of
     the four tokens' input-staging `Memcpy DtoD` records was missing: 135 against 136 rows, a 0.25-per-token
     "spread" that read as P1 refuted and failed G4 in 3 of 21 windows. And `hyb-0.25/ctrl0` lost 16 of 3,072 eager
     device rows (0.52 %), over the 0.2 % G6 ceiling set from round 3. The host counts were exact in every window.
     This round's counts are therefore **not** read from the device view. Captured windows are counted from the
     graph's own node list (`CUDAGraph.debug_dump`, round 6). G6 now bounds the device view's loss at 2 %, the
     level where device TIME (P3, P7) stays reliable.
  2. **Host timing on the shared NAS is not a measurement.** This round's eager all-hot wall "tax" read 14.1
     ms/token, against 1.0 in round 3, and the hybrid tier's transfer ratios read 1.27–2.57, against 1.05–1.28.
     Other agents' jobs were running on the same host. Nothing here grades a time, and nothing should.

## The graph-node probe (`probe-graph-nodes/`: `dotprobe.py.txt` is the script as run, `dotprobe.log.txt` its output)

Before round 6 counted captured windows from the graph, a four-node test graph (aten add, a Triton kernel, a D2D
copy, aten mul) was dumped. The first attempt showed that **torch 2.8's `CUDAGraph.debug_dump()` writes nothing,
and raises nothing**, unless the graph is created with `keep_graph=True` and instantiated before the dump. The
second attempt, with that fix, read the graph two ways:
- the CUDA runtime (`cudaGraphGetNodes` / `cudaGraphNodeGetType` on `raw_cuda_graph()`, libcudart from
  `/usr/local/cuda/lib64`): KERNEL, KERNEL, MEMCPY, KERNEL;
- the dot dump: the same four.

Replay stayed correct. `tests/test_p66_reduce.py` carries that dump verbatim as the parser's fixture.

## Round 6 (`round6/`): captured windows counted from the graph

The tree before the rebase onto main 0.37.3; every staged file passed `sha256sum -c` on the box. Captured windows
are counted from the graph. Result: rc 0, success marker, `TP_DONE`, in **680 s**. `summary.txt` stamps `MODE full` and
the overrides.

- **Gates, all six hold**, on this tree:
  - G1 nesting: 39 / 39.
  - G2 simulator: 14 / 14.
  - G3 graph check: 21 / 21, every replay bitwise.
  - G4 graph parity: 21 / 21, now on the graph's nodes, **with the runtime and dot readers agreeing in every
    window**.
  - G5 budget coverage: 39 / 39.
  - G6 record completeness: 39 / 39, with 7 lost device records, the worst window at 0.065 %, all on the hybrid
    tier.
- **Captured counts from the graph**, per token, 16 layers:
  - `ref`: 128 kernels (8 per layer);
  - pipelined with `n_hot > 0`: 288 kernels + 32 copies = 320 (20 per layer); with `n_hot = 0`: 240 + 32 = 272
    (17 per layer);
  - MXFP4 (8 layers): 288 + 16 = 304, which is 41 eager submissions per layer less the 3-row eager-only guard.
  Each equals the eager host count, so capture removes no residency work: it moves the host submissions into one
  graph launch.
- **Verdicts on this box**:
  - P1, P2, P4, P5 and P6 hold: the counts are structural. The decision rule's P1 ∧ P2 branch fires.
  - P3 and P7 are recorded, not graded (not the registered box).
- **P9: Level M equals Level L exactly**, a third time: +176 launches / +32 copies / 0 syncs per step at
  `M_pipe-1.00`, and +128 / +32 / 0 at `M_pipe-0.00`.
- **Transfer**, measured over cold_deadline, at the box's 6.11 GB/s:
  - pipelined 0.964–1.093;
  - MXFP4 NVMe 0.932–0.986;
  - MXFP4 pinned 0.972–1.324;
  - hybrid 0.990–1.032.
- **Times, again not a measurement.** The shared-GEMV shift under capture read **−1.70** ms/token this round, the
  eager all-hot wall tax 11.8 ms/token, and the captured all-hot wall tax −0.64 ms/token.
  - Four rounds have now given the same GEMV shift as +1.70, +0.04, +1.37 and −1.70 on identical work.
  - The added kernels' own device time is the stable part: 0.59, 0.49, 0.56 and 0.48 ms/token.
  - P3's band is registered on the added kernels only, for this reason.

## Round 7 (`round7/`): the tree after the rebase onto main 0.37.3

This round's `staged.sha256` was the branch's, byte for byte, until round 8's changes. The one change since round 6 is `p66_run.sh`: a proof
now records the reading-only floors instead of enforcing them, P65 Amendment 1's rule.

- **(a) `P66_MODE=prove`, real install**, at the new main SHAs (experts4bit-qlora `8020489`, grouped-nf4-gemm
  `f88df1e`).
  - Only the card-class override is set, so the reading's floors applied on the A2000. Two genuinely failed and
    were **recorded, not enforced**:
    - `floor_would_refuse_reading rc=15 10531 MiB VRAM free`;
    - `rc=13 host RAM available 57 GB`.
  - Then the runner installed `git`, pip-installed both packages, passed the site-packages tripwire and the 16 GiB
    pin, and wrote `P66_PROVED`, in **73 s**.
- **(b) `P66_MODE=full`**: rc 0, success, `TP_DONE`, **484 s**.
  - All six gates hold, with the graph's two readers agreeing in every captured window.
  - P1, P2 and P4–P6 hold, and the decision rule fires P1 ∧ P2.
  - P9 agrees exactly.
  - Transfer over cold_deadline: pipelined 0.986–1.054, MXFP4 pinned 0.955–1.005, MXFP4 NVMe 0.950–1.007, hybrid
    1.013–1.038.
  - Added kernels' device time 0.52 ms/token (the stable figure); shared-GEMV shift +0.53 (the unstable one).
  - 6 lost device records, the worst window at 0.048 %.
- `round7/full/L/` keeps the eager and graph tables (step_budget's input), plain. The earlier rounds' `rows_*.json`
  are gzipped.

## Round 8 (`round8/`): the rental lessons, on the committed tree

Round 7's `staged.sha256` stopped being the branch's when three rental lessons from lanes P64 and P65 were applied.
Round 8 ran exactly the new pin.
- **The changes.**
  - The proof's guard is 0.23 h instead of 10 min, because launcher boot takes 3–5 min of a guard.
  - An egress probe measures the way the reading's fetch runs: eight parallel 50 MB ranges, since `snapshot_download`
    uses eight workers, in Python, since the image has no `curl`. A proof records it; a reading refuses below
    80 MB/s (rc 14).
  - G3 records the sha256 of the eager and the replayed outputs, so a bit statement across boxes can rest on hashes.
- **(a) `P66_MODE=prove`, real install** at main `0644620` / grouped-nf4-gemm `68a1250` (`round8/prove/`).
  - The reading's floors were in force on the A2000, and the two that genuinely fail were recorded, not enforced
    (`rc=15` VRAM free, `rc=13` host RAM).
  - The egress probe read **168.8 MB/s** over eight streams, recorded as `hf_cdn_mbps_8x`.
  - It then installed, passed the tripwire and the pin, and wrote `P66_PROVED`, in **90 s**.
- **(b) `P66_MODE=full`** (`round8/full/`, job `job8b.sh`): rc 0, success, `TP_DONE`, **733 s**.
  - A first attempt (`job.sh`, whose log is `job.log.txt`) also ended rc 0. Its full-mode receipts were lost with the
    container: the NAS's live `locked_run.sh` does not mount the runner directory. `job8b.sh` copies them out.
  - All six gates hold, and P1, P2 and P4–P6 hold; the decision rule fires P1 ∧ P2.
  - Every G3 record now carries `eager_sha256` and `graph_sha256`.
  - Transfer over cold_deadline: pipelined 0.966–1.017, MXFP4 pinned 1.006–1.071, MXFP4 NVMe 0.995–1.049, hybrid
    1.109–1.425.
  - Added kernels' device time 0.48 ms/token captured (the stable figure).
  - 13 lost device records in total.

## A download the lane did not list at first

Level M's `step_decomp.py` loads `Salesforce/wikitext` (`wikitext-2-raw-v1`) from the Hub for its prompts. Every
rehearsal that ran Level M fetched it, about 14 MB, into the job's HF cache on the NAS
(`/share/Container/gnf4-interp/p66/hfcache/datasets`). The M logs show it as the "unauthenticated requests"
warning. It is a dataset, not a NAS-held model, but the plan had not listed it. It is now in the prereg's
downloads, marked unpinned.

## Reproducing

The jobs ran under `/share/Container/gnf4-interp/p66/` behind the shared `a2000.lock`: `mkdir` to take it, retry
every 60 s, never break another's, `rmdir` when the container exits (`locked_run.sh`). Model directories were
mounted read-only from `/share/models/`. The reducer regenerates the reads:

```sh
python bench/p66/p66_reduce.py bench/p66/rehearsal-a2000/round7/full/L --level-m bench/p66/rehearsal-a2000/round7/full/M \
  --step-budget bench/hybrid-g9/f1/step_budget.py
# an earlier round: its rows are gzipped
mkdir -p /tmp/r6 && cp bench/p66/rehearsal-a2000/round6/L/* /tmp/r6/ && gunzip /tmp/r6/rows_*.json.gz
python bench/p66/p66_reduce.py /tmp/r6 --level-m bench/p66/rehearsal-a2000/round6/M
```

- Rounds 1–5 predate the graph-node count. The reducer falls back to their device rows for captured windows, so
  rounds 1 and 5 read P0-failed under the current gates: round 1 because it predates the graph check, round 5
  because of the lost records that motivated the node count. That is their historical record, kept as found.
- Only round 7 keeps the eager and graph tables (step_budget's input), so G5 reads 0 / 0 on the other rounds.
- Each round's `RESULTS-p66-generated.md` is the box-side read, written by the reducer as it stood during that run.
  Regenerating round 7 on another machine matched it except in the last digit of two float sums (summation order).
