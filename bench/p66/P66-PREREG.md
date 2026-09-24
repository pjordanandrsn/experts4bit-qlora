# P66 — what residency adds per token in launches and syncs, against the all-resident step, by cold fraction (#711)

Registered 2026-09-23, before the rental. Lane P66; the work item is experts4bit-qlora#711. The owner authorized
the work and a later rental (relayed by the dispatching session, 2026-09-23). The rental is a separate step; the
author of this registration does not run it.

## Question

vLLM's expert-granular residency RFC (vllm-project/vllm#57794) measured cold-expert cost at ≈ 5.1 ms/token
against a 1.15–2.15 ms/token naive estimate (1 × H100, 48 layers × 512 experts, top-10, 88 of 512 offloaded =
17.19 % cold). It names three unverified suspects: PCIe contention, "~48 extra kernel launches per decode step",
and hot-bank geometry. It measured only 0 and 88 offloaded, so a fixed per-step cost and a per-cold-expert cost
cannot be told apart in its data.

This repository's residency paths have the same shape: cold experts are read over UVA from pinned host memory
(the address-gather, which #105 found cannot overlap compute because it is a kernel).
- The pipelined engine's per-layer step is a fixed, id-independent sequence by design
  (`experts4bit_qlora/engines/pipelined.py:38-43`), so residency adds launches.
- #108 counted host launches and syncs once: eager, pipelined at `n_hot = E`, one box.
- Nothing has recorded how many launches residency adds against the all-resident step, confirmed they do not
  move with cold fraction, or put grouped-nf4-gemm's cold cost model (`kernel/cold_deadline.py:85-98`, bytes
  over the link plus bytes over VRAM) beside the measured transfer.
- That model's docstring excludes PCIe contention and kernel compute (`:29-34`).
- The pipelined ladder's one graded transfer prediction came in *faster* than its band: eager K=0 53.4 ms
  against [55.2, 117.5] (grouped-nf4-gemm `bench/homelab/RESULTS-pipelined-ladder.md:46`).

So, for each residency path, against its family's all-resident step on the same box:
1. How many CUDA launches, copies and host syncs per token does residency add?
2. Does that number move with cold fraction?
3. Does cold_deadline's bytes model predict the measured transfer?
4. What share of residency's cost at the RFC's cold fraction is the fixed part a two-point comparison books as
   cold-expert cost?

Out of scope (the issue's own line): concurrency and preemption. `serve.py` is batch-1 behind one worker, the
scheduler has no preemption (`experts4bit_qlora/engines/fp8_paged_kv.py:521`), and placement has no KV term.
PCIe *contention* between concurrent transfers is also not exercised: a batch-1 token issues its transfers one
layer at a time, so P7 tests the transfer in isolation and says nothing about contention.

## Instrument

### Level L — the MoE token (the registered reading)

[`p66_census.py`](p66_census.py), one process per path, each hot fraction an arm in that process.

- **The token.** A token is one decode step's expert work: every MoE layer of the real checkpoint's expert bytes,
  in order. Each layer is called exactly as the model calls it: `forward(hidden [1, H], top_k_index [1, k],
  top_k_weights [1, k])`. Attention, norms, router and LM head are left out on purpose. Residency does not touch
  them, so they cancel in every delta. Leaving them out is also what lets the routing, and so the cold
  fraction, be **set** rather than hoped for (#108's method).
- **Hot sets.** Per layer, the first `round(h·E)` of a seeded permutation (seed 66). Every path at hot fraction
  `h` holds the same experts.
- **Windows** (one routing condition each):
  - `uniform`: k distinct experts uniform over E. The realized cold fraction follows `h`.
  - `ctrlC`: exactly C of the k lanes come from the layer's cold set, lanes shuffled across slot positions. Run
    at every `h` strictly between 0 and 1: C ∈ {0, k/2, k}, so realized cold fraction 0, 0.5, 1.0 at a fixed
    hot set.
- **Passes per window.** Each pass takes fresh tokens: warm 4, then 8 each.
  - **eager, profiled.** torch.profiler (CPU + CUDA). CUDA API calls are counted only inside
    `e4b::p66.token` record_function regions, so the instrument's closing sync is outside by construction.
    They are attributed to the engine's own phase regions, which wrap `_fetch`, `_resolve_src`,
    `_invalidate` and `_cold_contrib` for the profiled passes only. The window's global API counts are
    recorded beside them, for gate G1.
  - **Device rows** (kernels, memcpy, memset: count and self device time) come from the same window's device
    view. The window is also written as a key-averages table that `bench/hybrid-g9/f1/step_budget.py` parses
    unchanged: the repository's census route, SV2's.
  - **sync-debug.** `torch.cuda.set_sync_debug_mode("warn")`, warnings aggregated by the innermost engine
    frame. Only warnings raised while tokens execute count.
  - **timing.** Unprofiled; CUDA events plus host wall and host submit time per token.
  - **graph** (where the engine captures). The whole L-layer token is captured as one CUDA graph (static
    id/weight buffers, side-stream warm) and replayed: profiled, timed, then checked against eager on the
    same token.
    - Captured with `CUDAGraph.enable_debug_mode()`, so `debug_dump()` prints the graph's own node list
      (`cudaGraphDebugDotPrint`). The node counts by type are the captured window's exact counts
      (`graph_nodes`). The profiler's view of a replay is used for device time only.
  - **traffic check** (pipelined only). A separate pass with the engine's own counters on, because they add
    launches.
- **Realized traffic.**
  - Cold lanes per token are read from the routing.
  - Cold rows actually copied are simulated for the pipelined engine from every token it executed. A cold lane
    whose slot already holds that expert copies nothing: `p66_reduce.simulate_pipelined_traffic`, the
    engine's rule. The simulation is checked against the engine's own counters (G2).
  - The MXFP4 engines count their own traffic, always.
  - The hybrid tier streams every routed cold expert.
- **Transfer.**
  - *Measured*: device time per token of the link-crossing work, meaning the `_gather_rows_*` kernels (UVA
    reads) plus `Memcpy HtoD` rows.
  - *Predicted*: `cold_deadline.gpu_us(rows, uniq, costs)` summed over layers, on exactly the rows copied, with
    the box's own calibration blob.
  - The blob comes from grouped-nf4-gemm `bench/calibrate.py` at the pinned cut, read by `Costs.from_blob`'s
    field names: `b_link` = pinned H2D at 64 MB, `b_vram` = device triad.
  - The MXFP4 engines also re-copy HOT rows device-to-device into their slots. That is outside the model by
    construction, and it is taken off the measure using the all-hot sibling's own gather time per hot row.
- **Host record** (#108: host cost varied ~5× across boxes).
  - CPU model, cores, cgroup CPU and memory limits, NUMA nodes, GPU, PCIe generation and width, power limit.
  - Unit host costs measured on the box: one aten launch, one Triton launch, one idle `cudaStreamSynchronize`,
    one `.item()` round trip.

[`p66_reduce.py`](p66_reduce.py) is pure Python. `tests/test_p66_reduce.py` drives every counting rule, gate and
verdict branch with synthetic event lists. `tests/test_p66_census_helpers.py` drives the profiler-tree walk,
routing and schedule on CPU.

### Level M — the served step (context, informational)

`step_decomp.py` from `bench/hybrid-g9/`, byte for byte, through [`p66_step.py`](p66_step.py). The shim only
swaps the pipelined arm's all-hot sets for a hot fraction's sets, the same seeded ones.
- Settings: B=1, prompt 128, 48 generated tokens, eager scheduler loop, its own torch-profile window (12 steps
  after 26).
- Outputs: device kernel calls per step (its kernels.txt through `step_budget.py`) and host API counts per step
  (its `--sync-attr-out`).
- Arms: `M_ref` (hybrid, `--placement-override all-vram`: the certified all-resident collapse), `M_pipe-1.00`,
  `M_pipe-0.00`, `M_hyb-0.50` (VRAM half, NVMe rest).
- The captured all-resident served step already has a census on this card class: SV2
  (`bench/hybrid-g9/sv2/RESULTS-sv2-device-census.md`). It is cited, not re-run.

## Paths, hot fractions, and how each is selected

| path | engine (entry point) | cold source | hot fractions | capture |
|---|---|---|---|---|
| `ref` (NF4 all-resident) | `enable_hybrid_tier`, every expert VRAM, `collapse_resident=True` (step_decomp's R0; SV2's step) | none | 1 | yes |
| `pipe` | `enable_pipelined_residency` on arena-materialized modules | pinned host arena, UVA gather | 1, 0.828125, 0.5, 0.25, 0 | yes |
| `hyb` | `enable_hybrid_tier`, hot in VRAM, the rest on NVMe (`cold_dest="gpu"`, 64-row pinned tier) | NVMe → pinned tier → pageable H2D | 0.5, 0.25 | refuses (the v0 dispatch syncs); probe names the op |
| `mref` (MXFP4 all-resident) | gnf4 `Mxfp4PipelinedGptOss`, every expert hot | none | 1 | yes |
| `mpin` | `Mxfp4PipelinedGptOss` | pinned all-E host arena, UVA gather | 0.5, 0 | yes |
| `mnvme` | gnf4 `Mxfp4NvmeResidency`, shared tier + slot store | NVMe → 64-row pinned tier, UVA gather | 1, 0.5, 0.25, 0 | refuses by design (`_resolve_src` raises) |

- `0.828125` = 106 of 128 hot = 17.19 % cold, exactly the RFC's 88 of 512.
- Hot fraction 0 is included on every path that has one.

## The families, and why these

- **Qwen3-30B-A3B** (`Qwen/Qwen3-30B-A3B` @ `ad44e777bcd18fa416d9da3bd8f70d33ebb85d39`, P60's pin) for the NF4
  paths.
  - It is the RFC's shape: 48 MoE layers, 128 experts, top-8, against its 48 layers and top-10.
  - Its NF4 experts (6144 rows × 2.53 MiB = 16.3 GB, `bench/hybrid-g9/box/RESULTS-box-run.md`) fit a 32 GB
    RTX 5090 **all-resident**, so the reference and the hot-fraction-1 arms exist on the same box as the cold
    ones. The NF4 model needs 19.84 GiB resident (`engines/pipelined.py`'s own note), so a 24 GB+ card.
  - It is the family SV2 censused on this card class, so the reference has a precedent.
  - The arena is k8_bake.py's (`bench/p39/`), P60's recipe.
- **gpt-oss-20b** (`openai/gpt-oss-20b` @ `6cee5e81ee83917806bbde320786a8fb61efebee`, tp4's pin) for the MXFP4
  paths.
  - It is the only natively-MXFP4 family that fits a 5090. Its 24 × 32 × 13.2 MB = 10.2 GB of expert rows
    (relocation-baked, bit-identical to the release) fit all-resident beside a pinned all-E arena.
  - e4b's model binding `enable_mxfp4_nvme_residency` refuses biased families by design and serves
    DeepSeek-V4 / Kimi-K3, neither of which fits the box. So the MXFP4 arms drive grouped-nf4-gemm's own
    gpt-oss engines directly, with the checkpoint's biases. This is the engine the binding wraps; it is a
    per-layer object either way.

## Box, cost, receipts

- **Box.** One RTX 5090 32 GB on Vast verified/secure, image `pytorch/pytorch:2.8.0-cuda12.8-cudnn9-devel`
  (Triton and gnf4_native need a C compiler).
  - In a reading, the runner refuses anything else before any install: the card class (15), VRAM free < 30000
    MiB (15), free disk on `/root` < 180 GB (13), host RAM available < 64 GiB or a cgroup limit below it (13), and
    a failed 16 GiB pin (13). A proof refuses only on the class, a dud box and its own disk need, and records
    the rest (below).
  - Why these minimums: the pipelined arm pins a 15.2 GiB arena beside 15.2 GiB of materialized experts.
    Disk holds two checkpoints (~61 + 14 GB), a 16.3 GB NF4 arena plus its transient 16 GB snapshot, and a
    10.2 GB MXFP4 arena. The launcher's offer filter already asks ≥ 320 GB disk and ≥ 98 GB RAM
    (adertha `compute/vast_provider.py`).
  - The earlier residency lanes ran on a 251 GB-RAM host for the hybrid tier's DRAM bus
    (`bench/hybrid-g9/box/`). This lane's hybrid arms use no DRAM tier, so they do not need that host class.
  - Power limit and PCIe link are recorded, not gated. Counts are structural, and the transfer ratio is
    normalized by the box's own measured link.
- **Downloads.**
  - Qwen3-30B-A3B bf16 ≈ 61 GB.
  - gpt-oss-20b ≈ 13.8 GB (`model*.safetensors` + JSON; not `original/` or `metal/`).
  - `Salesforce/wikitext` `wikitext-2-raw-v1`, ≈ 14 MB in the HF cache. Level M's `step_decomp.py` builds its
    prompts from it (`_k8_window`, the default `--ppl-source wikitext`) with no revision pin. The lane runs
    step_decomp unchanged, so the dataset stays unpinned. That is acceptable because Level M is context, not
    the reading, and only its API counts are compared (P9). The rehearsals fetched it the same way.
  - HF token staged from `~/.config/hf/token`, never on a command line.
- **Two rentals, in order.** The compute rule in force: a rental whose guard exceeds 1 h first needs a proving
  rental of at most $0.15 and 10 minutes. The reading needs more than an hour, so:
  1. **Proving rental: `P66_MODE=prove`, guard 10 min (0.17 h) at ≤ $0.66/h, so ≤ $0.11 per attempt.**
     - **What it does:** the card-class and dud-box checks, the pinned install (with `git` if the image lacks
       it), the tripwire, the 16 GiB pin, and then, if the guard still has time, one timed shard of the pinned
       Qwen3 fetch (`model-00001-of-00016.safetensors`, 240 s alarm). There is no calibration, no bake and no arm.
     - **What it records but does not enforce:** the reading's host floors (VRAM free, disk, host RAM, the pin).
       Each miss is written as `floor_would_refuse_reading rc=<code> …`. This is P65 Amendment 1's lesson
       (`bench/p65/P65-PREREG.md`): the proving box is not the reading's box, and P65 lost two proofs to
       reading-only floors that each missed by about 2 %.
     - **What still refuses a proof:** the card class (15), a dud box (10), and disk below 20 GB, the proof's own
       need (13). The reading enforces every floor exactly as registered.
     - **Passing:** rc 0 **and** `P66_PROVED.<nonce>`; the driver exits 26 on rc 0 without the marker. Its receipt
       is cited in the reading rental's launch.
     - **Refusals and budget:** a proof refused on class or dud is redrawn on another machine, its receipt
       excluding the machine. All attempts together cost ≤ $0.33 (at most three).
     - **The guard's risk, stated:** P65 measured 3–5 min of a 10-minute guard spent in launcher boot and
       pre-flight, and amended its proofs to 0.23 h (≤ $0.15). This proof is shaped to survive a 10-minute guard:
       - the install is its only long step; whole proofs ran in 85 s (round 5) and 73 s (round 7) through the real
         path on the NAS;
       - the timed fetch runs only if at least 30 s remain before the deadline's 120 s margin, and is capped at
         240 s;
       - if the install itself cannot finish in the time left, the proof fails and a redraw is spent.
       If the rule's owner accepts P65's 0.23 h precedent, that guard is safer at the same ≤ $0.15 ceiling.
     - Its measured fetch rate is recorded against the wallclock estimate below; it does not gate.
  2. **Reading rental: `P66_MODE=full`, guard 2.0 h at ≤ $0.66/h, so ≤ $1.32, expected ≈ $0.95.** Launched only
     after a proving receipt passes.
- **Wallclock of the reading, estimated ≈ 85 min.**
  - The A2000 rehearsals' stage timestamps, scaled to Qwen3 (48 layers × 128 experts against OLMoE's 16 × 64):
    install 5 min, calibration 3 (1.3–1.5 on the A2000).
  - Qwen3 fetch + bake ≈ 20: P60 fetched, baked and ran two arms in 24 min on this class.
  - Level L NF4 ≈ 15: 4–5.5 min on the A2000 at a third of the layers.
  - gpt-oss fetch + bake ≈ 5.
  - Level L MXFP4 ≈ 6: 1.5–3 min at 8 of 24 layers.
  - Level M ≈ 15: 3–16.5 min on the A2000; the first run pays Triton compiles.
  - reduce 1; margin ≈ 15.
  - Each arm is started only if its *expected* time plus a 10-minute teardown margin fits the deadline. Its alarm
    is a separate, larger cap.
  - Level M arms come last and are context, so a slow box skips them before any Level L arm.
- **Cost ceiling.** Proofs ≤ $0.33 plus the reading ≤ $1.32 = ≤ $1.65. Lane ceiling $2, hard stop $3. No second
  reading box on a disappointing result.
- **Runner.** `p66_drive.sh` (controller) + `p66_run.sh` (box), the P60/B374 pattern.
  - The driver stages the runner, the census, the reducer, the shim, and three tools reused unchanged
    (`bench/p39/k8_bake.py`, `bench/hybrid-g9/step_decomp.py`, `bench/hybrid-g9/f1/step_budget.py`). All are
    pinned by [`staged.sha256`](staged.sha256) and checked in CI by `tests/test_p66_staged_pin.py`.
  - The box installs e4b at the driver's own HEAD and gnf4 at `GNF4_SHA` from the manifest, plus a clone of the
    same sha for `bench/calibrate.py`. A tripwire asserts the installed modules resolve to site-packages.
  - The two sides bind with a fresh nonce: `TP_DONE.<nonce>`, `P66_EXIT_CODE.<nonce>`, `P66_SUCCESS.<nonce>`.
  - The runner has rehearsal-only overrides (`P66_REHEARSAL_*`: card class, host minimums, local snapshots, a
    prebaked arena, no install, fewer MXFP4 layers, fewer tokens). They let the NAS exercise the runner itself.
    The driver forwards none of them; `tests/test_p66_staged_pin.py` holds that. Any override in effect is
    stamped into `summary.txt` as `OVERRIDE …`, so a rehearsal's receipts cannot pass for a reading.
- **Exit codes.**
  - 9: stage, install or tripwire.
  - 10: dud box.
  - 11: download.
  - 12: bake.
  - 13: host-limited (disk, RAM or pin).
  - 15: wrong class.
  - 32: a Level L arm failed or was skipped. Receipts are kept and the success marker withheld.
  - 33: calibration.
  - 78: environment refused (including an unknown `P66_MODE`).
  - 130: interrupted.
  - Driver only: 26, a prove run that ended rc 0 without `P66_PROVED`.

  The verdicts are read from the rows, not from the exit code.
- **Receipts** go to `bench/p66/receipts/`, with the proving rental's under `bench/p66/receipts/prove/`.
  - Every `rows_*.json`, `box.json`, `calib.json`, the eager and graph tables, and the capture probes.
  - The Level M outputs, `summary.txt`, `versions.txt`, `forensics.txt`.
  - The box's `RESULTS-p66-generated.md`, the arena index and bake records, and the teardown proof.
  - The receipt and its ledger row are committed together.

## Gates (P0) — nothing below is read unless every one holds

- **G1 nesting.** In every eager window, region-scoped API counts equal the window's global counts for every
  class except sync. The only syncs outside token regions are the instrument's own, at most two
  `cudaDeviceSynchronize`. A gap means API events failed to nest, and the counts would be under-counts.
- **G2 simulator.** The pipelined have-skip simulation equals the engine's own `cold_pcie_bytes / row_bytes` in
  every pipelined window.
- **G3 graph correctness.** Every captured window's replay matches eager on the same token: `b_rel ≤ 1.5e-2`,
  the tolerance of `tests/test_pipelined_graphs.py` (bitwise is recorded and expected). A graph that baked a
  stale pointer replays happily while returning garbage (`engines/capture.py`). An engine that claims capture
  and fails to capture fails this gate.
- **G4 device parity.** Capture changes what the host submits, not what the device runs.
  - The captured token's work nodes (kernel + memcpy + memset, from `graph_nodes`) equal the eager token's *host*
    submissions. Any other node type fails.
  - Less the MXFP4 engines' eager-only padded-index guard. `bool((want >= self.E).any())` in `_forward_decode`
    is skipped under capture by design: one compare, one bool reduction and one D2H copy per layer.
  - Both sides are exact sources. The input-staging copies run outside the graph, so they are not nodes.
- **G5 budget coverage.** `step_budget.py` covers ≥ 90 % of each eager table's Self CUDA total. That is its own
  refusal rule.
- **G6 record completeness.** In eager, every host submission (launch, async or blocking copy, memset) yields
  exactly one device row.
  - A shortfall is lost profiler records. It is recorded per window, and fails above 2 % of submissions.
  - More device rows than submissions always fails: that is mis-scoped regions, not loss.
  - **No count is graded on the device view.** The rehearsals lost device records, never host records, in eager
    and in captured windows: up to 0.52 % of a window (disclosed below).
  - The device view carries device *time* only (P3, P7). There a lost record biases the reading by at most its
    own share.

## Predictions (written before the 5090 data)

The per-layer counts below come from reading each engine's enqueue sequence. The A2000 rehearsal (disclosed
below) confirmed them on another box.
- They are structural: the engines' Python issues the same calls whatever the GPU. They are graded on any box.
- The shared GEMV's launch plan can differ by SM count, but both sides of every delta call the same GEMV
  entry point the same number of times, so the deltas cannot move with it.
- Time bands (P3, P7) are registered for the 5090 only.

Per-token numbers are for Qwen3-30B-A3B (L = 48) and gpt-oss-20b (L = 24).

- **P1: pipelined is fixed.** Within every pipelined arm, the counts per token are identical in every window:
  `uniform` and `ctrl0/4/8`, realized cold fraction 0, 0.5 and 1.0. The spread is **exactly 0**.
  - Eager is graded on host launches, copies, syncs and memsets. Captured windows are graded on the graph's own
    node counts (work, kernel, memcpy, memset). Both are exact sources.
  - Launches are also identical across every arm with `n_hot > 0`.
  - This is the issue's hypothesis.
- **P2: the pipelined count against the all-resident step.** Per layer, per token.
  - With `n_hot > 0`: **+10 launches, +2 copies, +0 syncs** (Qwen3: **+480 launches, +96 copies per token**).
    Of these, the `_fetch` region holds 8 launches and 2 copies: two id-table lookups, the lane select, the
    segment-address lookup, the gather, and the three-launch row-index dispatch; the copies are `want_buf`
    and `have`. The other 2 launches are the engine's unfused epilogue and combine chain against the
    reference's fused `_swiglu_rows` / `_combine_rows`.
  - With `n_hot = 0`: **+7 launches, +2 copies, +0 syncs** (Qwen3: +336 / +96). The fetch is 5 launches and 2
    copies, because the row-index dispatch is skipped when nothing is hot.
- **P3: the fixed tax** (all hot, zero cold traffic, captured). The device time of the kernels residency adds is
  **0.5–1.5 ms/token** (Qwen3, 5090): 576 extra device rows at the 5090's 1.4–2.5 µs per small kernel, SV2's
  elementwise and index-select rates.
  - The device-time shift of the kernels both paths run equally often is reported beside it, never folded in.
    This is the same GEMV reading the engine's row-block store instead of the reference's stacks: the RFC's
    "hot-bank geometry".
  - Eager adds the host cost of the extra submissions, reported with the box's unit costs.
- **P4: MXFP4 NVMe.** Fixed across every window and hot fraction (spread 0).
  - **4 syncs per layer** (96 per token): the padded-index guard (1), plus 3 in the fetch (`want_buf.tolist()`,
    and the pageable H2D of the address table and of `have`).
  - Against `mref`, per layer: **−1 launch, +3 copies, +3 syncs**.
- **P5: MXFP4 pinned.** Identical to `mref` at every hot fraction: Δ = 0 launches, copies and syncs. Hot lanes
  are re-copied device-to-device either way.
- **P6: the hybrid tier moves with cold fraction.** It rides the v0 dispatch, so per-layer syncs depend on the
  layer's composition: **4** (no cold lane), **10** (mixed), **8** (no hot lane). Launches differ between the
  controlled windows (spread > 0).
- **P7: transfer.** On pipelined `uniform` windows at every `h < 1`, the measured gather device time over
  `cold_deadline.gpu_us` on the rows copied is:
  - **HOLDS** at [0.80, 1.25];
  - **REFUTED** above 1.50 or below 0.80;
  - **BETWEEN BANDS** otherwise.
  The hybrid tier's pageable H2D and the MXFP4 engines' ratios are recorded, not graded.
- **P8 (informational): the RFC reading.** At the RFC-matched 17.19 % cold, captured:
  - the fixed tax (`pipe-1.00 − ref`) over residency's whole cost (`pipe-0.83 − ref`), wall ms per token;
  - eager is reported beside it.
- **P9 (informational): Level M.** Compared by API name, for each pipelined hot fraction both levels ran:
  - the served step's per-step deltas against `M_ref` in `cudaLaunchKernel`, `cudaMemcpyAsync` and
    `cudaStreamSynchronize`, from step_decomp's own sync-attr counts, which are complete;
  - against Level L's per-token deltas for the same names.
  Residency touches only the MoE layers, so these are expected to agree **exactly**. The rehearsal did. The
  kernels table's call counts are *not* used for this: step_decomp prints 80 rows, and the smallest rows (where
  residency's kernels live) fall below the cut.

## Decision rule

- **¬P0**: nothing is read. The failing gate is filed and the run repeated once.
- **P1 ∧ P2**: register the measured claim. The pipelined engine adds a fixed, id-independent count per MoE
  layer, as above, eager and captured, whatever the cold fraction.
  - The cost model gets a **per-step fixed term**: P3's device time when captured, and submissions × the
    host's unit cost when eager.
  - It never gets a per-cold-expert launch term.
  - This is a grouped-nf4-gemm follow-up on `kernel/cold_deadline.py`, filed with the measured values. The
    docstring's exclusion list (`:29-34`) is then wrong about kernel compute only in the fixed part, and says
    so.
- **¬P1**: the pipelined enqueue sequence moved with routing. That is a defect against the design law it states
  (`pipelined.py:38-43`); it is filed with the moving kernel names before anything is claimed.
- **P1 ∧ ¬P2**: fixed, but not the registered count. The measured count is registered instead, with the
  difference attributed by name.
- **P4**: the NVMe engine's +3 syncs per layer are structural; the host needs the ids to issue disk reads. Two of
  the three are pageable H2D copies. A pinned-staging follow-up is licensed only if their measured host cost
  is ≥ 10 % of that token's wall.
- **P6**: the hybrid tier's cost depends on layer composition. `cold_dest="deadline"` prices only bytes, so its
  GPU side omits the mixed-layer dispatch term. This is recorded against gnf4's cold_deadline beside the P2
  follow-up.
- **P7 holds**: bytes over link plus bytes over VRAM predicts the variable part of residency. Any RFC-style gap
  in this engine is the fixed tax (P3/P8) or host launch cost, not transfer. **P7 refuted or between bands**:
  the transfer itself departs from bytes/link. The model needs a measured efficiency factor (UVA-read
  efficiency, #105 candidate 4) before any RFC comparison is made.
- **P8, the RFC comparison.**
  - Tax share ≥ 25 %: a two-point comparison would book the fixed tax as cold-expert cost. That is consistent
    with the RFC's launch suspect acting as a *fixed* term.
  - Tax share ≤ 10 %: launches cannot explain an RFC-size (2–4×) gap in this engine, so the gap would be
    link-side.
  - In between: stated as measured, no attribution.
  - Either way the reply to the RFC is a measurement of *this* engine, not of vLLM's two-bank forward. That
    forward adds ~1 launch per layer (48 per step); this engine adds 10.

Nothing here changes a default.

## What was run before this registration was final (disclosed: it changed the instrument)

This is not the reading. The reading is the RTX 5090 above.

**Box.** The NAS RTX A2000 12 GB (sm_86, 26 SMs, PCIe gen3 × 8, measured 6.28 GB/s H2D), host Xeon W-1250,
image `pytorch/pytorch:2.8.0-cuda12.8-cudnn9-devel`. It is shared with production services; VRAM stayed under
8 GB. Families: OLMoE-1B-7B NF4 (16 layers × 64 experts, top-8; arena baked on the NAS) and gpt-oss-20b
layers 0–7 MXFP4 (top-4). A2000 timings are clock-state dependent on a shared 70 W card and are never quoted
as a result. Receipts and a NOT-A-READING README are in [`rehearsal-a2000/`](rehearsal-a2000/).

**What it showed.** Every path ran end to end on real CUDA. Per layer:

| path | launches | copies | syncs | fixed across windows? |
|---|---|---|---|---|
| ref (NF4) | 8 | 0 | 0 | — |
| pipe, `n_hot > 0` | 18 | 2 | 0 | yes, spread 0 over cold fraction 0 / 0.5 / 1.0, eager and graph |
| pipe, `n_hot = 0` | 15 | 2 | 0 | yes |
| hyb h=0.5 | 41 / 75 / 50 (no-cold / mixed / no-hot) | 7 / 20 / 15 | 4 / 10 / 8 | **no**: it moves |
| mref, mpin (MXFP4) | 38 | 3 | 1 | yes |
| mnvme | 37 | 6 | 4 | yes |

- The have-skip simulator matched the engine's counters in 6 of 6 pipelined windows.
- Graph replay collapsed the host side to 1 graph launch plus 2 staging copies per token, with device kernels
  unchanged.
- cold_deadline's bytes model against the measured gather: 1.01 (h = 0.5) and 1.08 (h = 0). The hybrid tier's
  pageable H2D read 1.06–1.28.
- Host unit costs: aten launch 6.8 µs, Triton launch 16.7 µs, idle sync 5.6 µs.

**What it changed.** Each change was made before this registration merged.
1. **The sync detector counted the instrument.** One warning per process was raised at torch's own
   `set_sync_debug_mode` (`torch/cuda/__init__.py:1193`), not by an engine. Only warnings raised while tokens
   execute count now; the rest are recorded as `outside_tokens`.
2. **Graph parity was wrong for MXFP4.** The engine's padded-index guard runs only eagerly: one compare, one
   bool reduction and one D2H copy per layer, identified by name. G4 accounts for exactly those three rows on
   the MXFP4 paths and nothing else.
3. **A failed capture named nothing.** The hybrid probe read "operation failed due to a previous error during
   capture". The probe now first runs one layer under `set_sync_debug_mode("error")`, so the first
   synchronizing op and its engine frame are the receipt.
4. **Graph correctness was unchecked.** G3 was added: a stale-pointer graph would have produced counts from
   garbage.
5. **"The tax" was one number hiding two.** Graph-mode device time rose 2.29 ms/token for pipelined at h = 1
   against ref. Only 0.59 ms of it was the 192 added device rows. The other +1.70 ms was the *same*
   `_gemv_nf4_grouped` call count running slower on the engine's store; in eager the same shift was +0.01 ms.
   P3 now grades the added kernels only and reports the shared-kernel shift beside it, the RFC's third
   suspect. Whether that +16 % is layout or the A2000's clock state is exactly what one rehearsal box cannot
   tell.
6. **Time bands are graded only on the registered box**, because the A2000's eager and graph GEMV times
   disagreed by about 50 % on identical work (15.78 against 10.29 ms/token for the reference's GEMV).

**A third round ran `p66_run.sh` itself, end to end**, with its documented rehearsal overrides (stamped in its
`summary.txt`). It finished rc 0 with its success marker. A second round was withdrawn unrun: the third covered
it.
- Every per-layer count above reproduced, including the RFC-matched arm (`pipe-0.83`: 53 of 64 hot on OLMoE =
  17.19 % cold) and hot fraction 0.25 on every path.
- Gates:
  - G1: 39 / 39.
  - G2: 14 / 14.
  - G3: 21 / 21, every replay **bitwise** equal to eager.
  - G4: 21 / 21.
  - G5: 39 / 39.
  - G6: 39 / 39, with 17 lost device records.
- Transfer, measured over cold_deadline:
  - pipelined 0.983–1.023;
  - MXFP4 pinned and NVMe 0.949–1.008;
  - hybrid pageable H2D 1.046–1.284.
- The all-hot tax, captured: 0.49 ms/token of added kernels (192 rows, 2.6 µs each). The shared GEMV moved
  +0.04 ms, where round 1 read +1.70, so item 5's caution was warranted.
- P8 at 17.19 % cold on that host: the fixed tax was 2.4 % of residency's cost captured and 20 % eager.
- **P9 agreed exactly**: `M_pipe-1.00 − M_ref` = +176 launches, +32 copies, 0 syncs per step, and
  `M_pipe-0.00 − M_ref` = +128, +32, 0. These are Level L's per-token `by_api` deltas.

It changed six more things:

7. **The override stamp missed three knobs.** Their names carry digits (`…GNF4_SRC`, `…NF4_ARENA`), and the stamp
   pattern had none. A rehearsal could have under-declared itself; it cannot now, and a test holds it.
8. **`calibrate.py` leaves an 8 GiB test file** in the work dir, which the driver's fetch would have pulled back.
   It is deleted after calibrating and excluded from the fetch.
9. **The profiler drops device records.** It happened only on the synchronizing paths:
   - MXFP4 NVMe: one per window, the first kernel;
   - hybrid: 1–3 of ~4,160.
   The host counts were identical, yet the first read of round 3 called P4 refuted on a lost record. Eager
   fixedness is now graded on host counts, G6 bounds and records the loss, and G4 compares against host
   submissions.
10. **step_decomp's table is row-limited.** Its call counts miss residency's small kernels (148.7 against 192 per
    step) while its time coverage is 99.9 %. P9 uses the complete API counts.
11. **Level M's hybrid arm prefills 128 tokens in one chunk.** The tier's hard floor (one forward's distinct cold
    experts) makes 64 rows unsafe on a 128-expert family, so that context arm gets 256. Level L's arms are
    decode-only and keep 64.
12. **The stock devel image has no `git`**, and every install is a git URL. The runner installs git when it is
    missing. The proven Vast lanes did not need this, but the NAS image does.

**Not explained by the rehearsal (context only).** The served hybrid arm showed 2 `nonzero` per layer, so most of its
layer-steps took no cold branch, although the solver placed 512 of 1024 experts in VRAM. This is recorded, not
graded.

**Rounds 4–6.**
- **Round 4** ran round 3's fixes (the stamp, the 8 GiB file, the hybrid tier rows). The same gates held and P9
  agreed again.
- **Round 5** exercised two runner changes:
  - **13. The time guard was wrong.** `can_run` had been given each arm's alarm cap as its expected time. On a 2 h
    guard, a late arm with a 40-minute cap would have been skipped as out of time while needing three minutes.
    Expected time and cap are now separate numbers, and a test holds expected < cap for every arm.
  - **14. The proving mode**, for the compute rule. `P66_MODE=prove` ran the real install path on the NAS: no
    `git` in the image (installed by the runner), then both packages pip-installed from GitHub at main
    `94842a2` / `66d41c8`, whose package code is identical to this branch. The site-packages tripwire and the pin
    passed, and it wrote `P66_PROVED`, in **85 s**. The timed shard fetch was skipped by design (local snapshot,
    no WAN pull of a NAS model). The shell around it was checked with stubs; the live `hf_hub_download` is not
    rehearsed.

  Round 5's full run then showed:
  - **15. The profiler's device view drops records in captured windows too.** One input-staging copy of four
    tokens went missing in 3 of 21 graph windows, and 16 of 3,072 eager rows went missing in one hybrid window
    (0.52 %). The host counts were exact throughout. Counting a captured window from its device rows had read as
    P1 refuted, and G4 failed. **Captured windows are therefore counted from the graph itself**, through two
    readers that must agree: the CUDA runtime's node list (`cudaGraphGetNodes` / `cudaGraphNodeGetType` on
    `raw_cuda_graph()`), and torch's dot dump. G6 became a 2 % bound on the device view, which now carries time
    only.
  - **16. The NAS host is too loaded to time anything.** The eager all-hot wall tax read 14.1 ms/token, against
    1.0 in round 3, and the hybrid transfer ratios read 1.27–2.57, against 1.05–1.28. Other agents' jobs shared
    the host.
- **A probe before round 6** found that torch 2.8's `debug_dump()` writes **nothing, silently**, unless the graph
  is created with `keep_graph=True` and instantiated first. Without that, `capture_end` drops the graph. The same
  probe read a four-node test graph identically through both readers.
- **Round 6** ran the final census and reducer; its numbers are in `rehearsal-a2000/README.md`.
- **17. After a rebase onto main 0.37.3,** the proof adopted P65 Amendment 1's rule, recording reading-only floors
  instead of enforcing them. Round 7 rehearsed it with the reading's floors in force on the A2000, where they
  genuinely fail, then re-ran the full mode on the rebased tree. main's changes since the first cut touch only the
  int4-store GEMV path and a comment in `hot_residency.py`, and `int4_b32` in grouped-nf4-gemm; no path this lane
  measures. **Round 7 ran exactly the committed `staged.sha256`.** All six gates held, and P1, P2 and P4–P6 held.
  The proof recorded `rc=15` VRAM free and `rc=13` host RAM on the A2000 and still passed, in 73 s.

Amendments, dated, go below this line before any data is read.
