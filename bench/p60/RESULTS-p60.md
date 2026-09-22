# Results — P60: the expert GEMV's B=16 headroom is repeated rows — 0.92 ms/step, the ceiling a grouped kernel can recover; row order is worth nothing (RTX 5090, 2026-09-22)

Pre-registration: [`P60-PREREG.md`](P60-PREREG.md) (#685, merged 87c5096 before the run). Receipts: [`receipts/`](receipts/) — the recorded routing (`eids_b16.int16.bin`, raw little-endian int16 `[128 steps, 48 layers, 16 rows, 8]`, shape and sha256 in `eids_b16.int16.json`; converted from the box's `eids_b16.pt` so the receipt carries no pickle), its summary (`eids_b16.pt.json`), `replay.json`, the box's generated read, trimmed record/replay logs, `summary.txt`, `versions.txt`, `forensics.txt`, teardown proof. Every number below is read from them by [`p60_reduce.py`](p60_reduce.py).

Lane `p60-5090-1`: Vast instance 52104714, one RTX 5090 (170 SMs) on an AMD EPYC host; e4b **87c5096**, grouped-nf4-gemm **65cb104**, torch 2.8.0+cu128; rented 18:57Z, `TP_DONE` 19:21Z, destroyed 19:21:53Z (proven); **$0.23** against the $0.66 estimate. The served int4 stack's routing was recorded on the 16 wikitext rows every B=16 lane decoded (digest `f67e7e4d…`), 128-row prefill, 128 teacher-forced decode steps; the replay ran those 128 steps × 96 expert-GEMV calls, one CUDA graph per step, 20 replays each, on synthetic 128-expert weights at the real shapes, with gnf4's own plan (split-K 16 for `gate_up`, 6 for `down`). Copy bandwidth on the box: **1512 GB/s** (#564 used 1,528).

> **Correction (2026-09-22, after grouped-nf4-gemm lane K18).** This read attributed the 0.92 ms/step dedup gap to *re-streaming each expert's weight slice once per routed row*, called it "the ceiling a grouped expert GEMV can recover", and said P0/P2 suggest the kernel is load-bound. The number stands. The attribution does not.
>
> - The dedup arm drops three things at once: the repeated rows' loads, **their arithmetic**, and their programs.
> - P3 below already found that locality is not the constraint.
> - K18 built the grouped GEMV this read licensed. It loads each expert's slice once for up to four of its rows and keeps every row's arithmetic. It is bitwise exact and **1.49× slower** on these recorded ids: 9.689 against 6.520 ms/step on a second 5090 host, where this replay's served arm reads 6.520 and dedup 5.564 (grouped-nf4-gemm `kernel/RESULTS-k18-grouped-expert-gemv.md`, row `gnf4.kernel.k18-grouped-expert-gemv.5090.2026-09-22`).
>
> So what 0.92 ms measures is the cost of ~73 more rows per call (128 routed against 54.7 distinct), and nothing measured separates their loads from their arithmetic. That work is not optional: each routed row is a different token's activation. The original text below is kept as written.

## The read

| arm | step ms (median of 128) | mean | `_gemv_int4_b32` ms (profiler, 8 steps) | reduce ms |
|---|---|---|---|---|
| **served** (R = 128, routing order) | **6.479** | 6.475 | **6.155** | 0.303 |
| sorted (rows ordered by expert) | 6.444 | 6.462 | 6.359 | 0.316 |
| **dedup** (one row per distinct expert) | **5.560** | 5.564 | **5.120** | 0.171 |
| floor (distinct bytes / 1512 GB/s) | 4.598 | 4.606 | | |

Mean distinct experts per layer per step on these rows: **54.66** (layer means 48.8–70.2).

- **P0 — the instrument — HOLDS.** The replay's served kernel row reads **6.155 ms** against P57's census **6.340 ms** (−2.9 %, band ± 15 %). Synthetic weights plus recorded ids reproduce the served kernel's work, so the other arms are read.
- **P1 — repeated rows are the cost — HOLDS.** `served − dedup` = **0.919 ms/step** of graph time (band ≥ 0.8), **1.034 ms** on the kernel row, plus 0.132 ms of split-K reduce that the smaller R also saves. Each call streams its expert slice once per ROW (128) where only ~55 experts are distinct; the dedup arm reads each once. **That difference is the ceiling a grouped expert GEMV can recover** — about **8 % of the 11.6 ms B=16 step** — on loads; a grouped kernel still does 128 rows of arithmetic, so it reaches the dedup time only if the kernel is load-bound, which P0/P2 suggest it is.
- **P2 — one row per expert near roofline — BETWEEN BANDS.** `dedup / floor` = **1.209** on graph time (hold ≤ 1.15, refute > 1.30); on the kernel row alone 1.114. Not held, not refuted: there is a further ~0.5–1.0 ms between reading each expert once and the bandwidth floor, part of it the reduce and graph-node overhead the floor does not charge.
- **P3 — row order does not matter — HOLDS.** `sorted / served − 1` = **-0.55%** (band ± 3 %). As predicted: with split-K 16 each pass over the rows touches ~6 MB of distinct weights, far inside the 5090's L2, so repeats already hit cache whatever their order. Sorting rows is not a lever; it is L2→SM traffic and per-row work, not HBM misses, that the repeated rows cost.
- **P4 — routing reproduces — REFUTED (informational).** 54.66 distinct experts per layer per step against P57's 58.67 (± 3): teacher forcing on the wikitext rows routes narrower than the model's own greedy continuations. The replay reproduced the served kernel row (P0) with this routing, so the arms are read on it; a grouped kernel's benefit scales with the repeat ratio, so the serving-routing ceiling is a little lower than 0.92 ms (fewer repeats at 58.7 distinct).

## Decision rule, applied

**P0 ∧ P1 → a grouped expert GEMV is licensed to BUILD** — grouped-nf4-gemm lane **K18**, pre-registered before any code: one program per (expert, column block, split) processing every row routed to that expert (K16-style M-tiling over ~2–16 rows), so each expert's weight slice is read once per call. Its target is the dedup arm (≤ 5.56 ms/step on this replay), its gate is bitwise (or K6-frame) identity to the served rows, and its read replays **these recorded ids** (`receipts/eids_b16.int16.bin`) on the 5090 before any consumer change. **P3 → no row-order lane.** P2 between bands is recorded as the next question after K18 (the reduce and per-expert config at small R). Nothing here changes a default. Register row: `e4b.serve.p60.qwen3.b16.expert-gemv-repeat-cost.5090.2026-09-22`.

## Generated read (the box's `p60_reduce.py`, unedited)


Pre-registration: [`P60-PREREG.md`](P60-PREREG.md). Every number below is read from the run's receipts by `p60_reduce.py`.

Box: NVIDIA GeForce RTX 5090 (170 SMs), torch 2.8.0+cu128; copy bandwidth **1512 GB/s**; plans (sk) {'gate_up': 16, 'down': 6}; 128 recorded steps replayed; eids [128, 48, 16, 8].

| arm | step ms (median) | mean | min | max | `_gemv_int4_b32` ms (profiler) | reduce ms |
|---|---|---|---|---|---|---|
| served | 6.479 | 6.475 | 5.982 | 6.875 | 6.155 | 0.303 |
| sorted | 6.444 | 6.462 | 6.028 | 6.874 | 6.359 | 0.316 |
| dedup | 5.560 | 5.564 | 4.801 | 6.142 | 5.12 | 0.171 |
| floor (distinct bytes / measured bandwidth) | 4.598 | 4.606 | | | | |
| floor at #564's 1528 GB/s | 4.550 | | | | | |

Mean distinct experts per layer per step: **54.66** (recorder: 54.65966796875; layer means 48.765625..70.21875).

## Verdicts (pre-registered)

- **P0: HOLDS** -- served _gemv_int4_b32 6.154557875 ms vs census 6.34 (-2.9%)
- **P1: HOLDS** -- served - dedup = 0.919 ms/step (hold >= 0.8, refute < 0.3)
- **P2: BETWEEN BANDS** -- dedup / floor = 1.209 (hold <= 1.15, refute > 1.30)
- **P3: HOLDS** -- sorted / served - 1 = -0.55% (hold within +-3 %)
- **P4: REFUTED** -- mean distinct 54.66 vs P57 58.67 (+-3)

**Decision rule:** P1 holds -> a grouped expert GEMV (one weight read per expert for all its rows) is licensed to BUILD (grouped-nf4-gemm lane K18, pre-registered before code)
