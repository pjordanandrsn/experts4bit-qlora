# P61 — what bounds the served expert GEMV at B=16: per-row work or per-expert bytes (registered 2026-09-22, before the run)

Owner directive (Jordan, 2026-09-22): *"proceed"*, answering my proposal after K18 to separate the served GEMV's per-row cost from its per-expert load cost before building another lever.

**Predecessors:**
- **P60** (`bench/p60/RESULTS-p60.md`): on Qwen3-30B-A3B's recorded B=16 routing on an RTX 5090, one row per distinct expert instead of one per routed row ran **0.92 ms/step** faster, and sorting rows by expert changed nothing (L2 already serves the repeats).
- **K18** (grouped-nf4-gemm `kernel/RESULTS-k18-grouped-expert-gemv.md`): a grouped GEMV that loads each expert's slice once for up to four rows, keeping every row's arithmetic, was bitwise exact and **1.49× slower**. That withdrew P60's reading of the gap as weight re-streaming (dated correction in RESULTS-p60, #688).

What is left unknown is what a repeated row actually costs.

## Question

On the served `gemv_int4_b32` at Qwen3-30B-A3B's expert shapes, how does the per-call time split into:
- a **per-row** term: each row's program re-reads its expert's slice (from L2 when the expert repeats) and does the int32 block dots;
- a **per-distinct-expert** term: that expert's bytes streamed from DRAM once?

And within the per-row term, is the L2 re-read or the arithmetic the cost?
- If the re-read, a load-sharing kernel is still the right idea and K18 lost only on its own overhead.
- If the arithmetic, sharing loads cannot buy it, and the lever is cheaper per-row arithmetic.

## Instrument (built and rehearsed before registration)

[`sweep_gemv.py`](sweep_gemv.py) uses the shipped gnf4 GEMV only: no model, no new kernel. Stores are synthetic int4-b32 weights for 128 experts **per layer**, 48 layers (~16 GB): each call reads its own layer's store, so no call is served from the previous layer's L2 lines, which P60's single shared store allowed.

- **Grid.**
  - R ∈ {16, 32, 64, 128} rows × D ∈ {1, 2, 4, …, 128} distinct experts (D | R, rows interleaved across experts), 26 cells per projection.
  - Each cell is one CUDA graph of 48 calls, 20 replays (median).
  - Both graph time per call and the `_gemv_int4_b32` kernel's own time per call are recorded (profiler over graph replays, the census's instrument).
- **Recorded.** P60's recorded ids (`bench/p60/receipts/eids_b16.int16.bin`, sha256 `c050961e…`, 128 steps × 48 layers), staged and verified on the box.
  - Arms: served (R = 128) and dedup (one row per distinct expert).
  - Each step's 96 calls go in one graph, on the per-layer stores **and** on one shared store (P60's replay).
- **Probes.** Device-to-device copy bandwidth of a 2 GiB buffer (the DRAM floor), and of two buffers each a quarter of the L2 (the L2-resident floor), read+write bytes per second. The L2 figure is a proxy, labelled as one; the P2 thresholds leave margin for it.

[`p61_reduce.py`](p61_reduce.py) fits, per projection, **t(R, D) = a + b·R + c·D** by least squares over the grid, on graph time and on kernel time. It then applies the rules below literally. `tests/test_p61_reduce.py` checks that the fit recovers known coefficients and that every verdict branch fires on the rows meant to trigger it.

**Rehearsal.** On the QNAP A2000, with 3 layers × 3 steps (correctness and plumbing only; A2000 timings are never quoted):
- the self-test passes on CUDA and under the interpreter;
- all 52 cells capture and profile;
- the reducer runs end to end, and its P0 gate correctly FAILS on the A2000's numbers, so the gate is live.

The rows record the split-K plan by R. On the A2000 (26 SMs) the plan changes with R. On the 5090 (170 SMs, above the 64-SM class the R term applies to) it should hold at sk 16 / 6 for every R. If it does not, the fit is read with that noted.

## Predictions (written before the data)

- **P0 — instrument (gate):** the shared-store served arm on the recorded routing is within **± 5 % of P60's 6.479 ms/step**. K18's served arm read 6.520 on another host. If P0 fails, nothing below is read.
- **P1 — the grid's model transfers to real routing:** b_graph (both projections) × the recorded rows dedup removes per step predicts the measured **per-layer-store** served − dedup gap within **± 25 %**. *Refuted by* more than ± 50 %: then skew or an R × D interaction that uniform cells do not have dominates, and P2 is read as informational only.
- **P2 — what a repeated row costs.** b₁ is the kernel's time per extra row when its one expert is L2-resident (the D = 1 column's slope). Compare it with the L2 re-read floor per row (the expert's slice bytes ÷ the L2-resident copy bandwidth), on both projections:
  - **≤ 1.3 → L2-BOUND:** the per-row cost is the weight re-read;
  - **≥ 2.0 → NOT-BYTES:** the per-row cost is the arithmetic, or issue, not bytes;
  - otherwise **MIXED**.

  **No side is predicted.** P60's gap works out to ≈ 0.26 µs per removed row summed over both projections. That is the same order as both an L2 re-read of a 1.77 MB / 0.88 MB slice and the slice's int32 dot, so either side is plausible.
- **P3 (informational):** c from the kernel fit ÷ the DRAM floor per expert (slice bytes ÷ DRAM copy bandwidth).
- **P4 (informational):** per-layer-store served − shared-store served. This is what P60's shared store could have given itself through cross-layer L2 reuse. The per-layer served kernel row sits beside P57's census 6.340 ms.

## Decision rule

- **¬P0** → nothing is read.
- **P0 ∧ P1 refuted** → the linear model does not transfer; no lever lane follows from this read.
- **P0 ∧ P1 (held, or between bands with that caveat)**, then by P2:
  - **L2-BOUND** → a low-overhead load-sharing design (no in-kernel tiling: tiles built once per layer and shared by both projections) is pre-registered with b as its bar.
  - **NOT-BYTES** → a cheaper per-row arithmetic path (a tensor-core int8 route for experts with ≥ 2 rows) is pre-registered with b as its bar.
  - **MIXED** → the larger measured share picks the next lane, pre-registered separately.

Nothing here changes a default.

## Budget and STOP rules

- **Rental:** one RTX 5090 (verified/secure), **≤ 1.0 h guard**, estimate **≤ $0.66**. Expected: install ~5 min, stores ~1, grid ~2, recorded ~5.
- **Ceilings:** lane ceiling $1, hard stop $2.
- **Exit codes:**
  - self-test fails → rc 21;
  - sweep fails → rc 32;
  - P0 fails → recorded, nothing read;
  - no second box on a disappointing result.
- **Receipts:** the receipt and its ledger row are committed together (backup, then post-commit diff).

Amendments, dated, go below this line before any data is read.

## Amendment 1 (2026-09-22, after run 1's P0 gate, before any P1–P4 read)

**Run 1: gate failed.** Run `p61-5090-1` (receipt in adertha-receipts `56c5170`, $0.0436) completed with rc 0 but **failed P0**:
- shared-store served read **7.253 ms/step** against P60's 6.479 (**+11.9 %**, band ± 5 %);
- the lighter dedup arm read 5.513, within 1.5 % of P60's 5.560;
- its card's `power.limit` was **450 W**. P60's was 600 W, and K18's was 575 W, where served read 6.520.

**What was seen:** the forensics and the runner's four summary lines (both stores' served and dedup step medians) only. The grid, the fit, and P1–P4 were **not computed**, and run 1 is never read beyond P0.

**Change:** `p61_run.sh` refuses a card whose `power.limit` is below **575 W** (rc 15), before any install, so the read box matches the P0 anchor's class. Nothing else changes.

**Re-run:** exactly one, `p61-5090-2`. This amends STOP-4 for this instrument failure only. If run 2 also fails P0, the lane stops and records that the served GEMV's replay depends on the host's power class.

