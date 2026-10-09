# P124 — results: **DEFAULT_ON**. With grouped-nf4-gemm #522's 32- and 64-row tiles, `E4B_ATTN_INT4_WIDE=1` makes SC2e's 64- and 32-row decode steps about 3.4 % and 3.5 % faster on Qwen3-30B-A3B int4, within P110's quality bar (one RTX 5090, 2026-10-09)

Registration: `bench/p124/PREREG-p124.md` (#1415, merged `ee91522`), with Amendment 1 (#1421, merged `2ed9c11`, the launch
commit of the reading below). Issue: #846. Code under test:
- grouped-nf4-gemm #522 (`4ed26d96`, `block_m=`);
- e4b #1410 (`170b1532`, `E4B_ATTN_INT4_WIDE`).

**Verdict by `p124_reduce.py` on attempt 2 (`p124-5090-4`): `DEFAULT_ON`.** Every ON/OFF ratio is ≤ 0.98 at both
depths, the two blocks agree, and both subjects pass the quality bar. No VOID condition fired:
- the commits, revision and stack are the registered ones; the engine was built with the route enabled;
- every profile arm ran 8 eager steps of its bucket;
- every block ran in its registered order, and each of its two runners captured every bucket and replayed its bucket
  293 times with no eager step;
- engagement was exact:
  - one `_gemm_int4_b32_smallm` launch per projection (96) per profiled step with the route on, and none off;
  - the captures took the route at 32 and 64 rows with the route on, and the cached copy with it off;
  - every quality pass's decode calls were on its registered route;
  - ON64 did not score bit-equal to R;
- each setting emitted the same tokens in blocks a and b;
- G64on matched ON64 at 8,128 of 8,128 positions;
- both mutants failed the bar.

The premise held: the cuBLAS GEMMs the route replaces are 12.4 % of the 64-row eager step. The verdict re-derives byte
for byte on Python 3.9 and 3.13.

## Runs

| run | what | cost | outcome |
|---|---|---:|---|
| `p124-prove-1` | proof, Granite with int4 attention | $0.175 | PROVED |
| `p124-5090-1` | attempt 1 (ABBA arms) | $1.003 | **NOISY**: one arm's level shift (section below); led to Amendment 1 |
| `p124-prove-2` | proof under Amendment 1 | $0.170 | PROVED |
| `p124-5090-2` | attempt 2 | $0.096 | NOT_RUN: the launcher's pre-flight refused the host (HF CDN 4.2 MB/s < 20) |
| `p124-5090-3` | attempt 2 | $0.054 | NOT_RUN: the same (13.6 MB/s) |
| `p124-5090-4` | **attempt 2, the reading** | $0.787 | **DEFAULT_ON** |

The lane spent **$2.285**, against its $3.00 ceiling. The two NOT_RUN hosts were retried under the maintainer's standing
rule for pre-flight refusals, which measure nothing.

## The reading (`p124-5090-4`)

**Host:** one RTX 5090 (sm_120, driver 595.84) on an AMD EPYC 7B13 (256 threads), Vast instance 54950831.

**Cost:** $0.787. Teardown proven at 02:47:07Z.

**Stack:**
- e4b 0.50.0 at `2ed9c11`; grouped-nf4-gemm 0.43.0 at `4ed26d96`;
- torch 2.8.0+cu128, triton 3.4.0, transformers 5.17.0;
- `Qwen/Qwen3-30B-A3B` at `ad44e77`: SC2e's served stack (int4 experts on 48 layers, 96 int4 RTN attention
  projections after the q/k/v fusion, the folds), built with `E4B_ATTN_INT4_WIDE=1`;
- `E4B_INT4_WIDE_TILES` at its default (`auto`, #1404), so both settings build P122's chunked tile table;
- the box: load 161 s, arms 507 s.

**Served steps, interleaved** (Amendment 1). Each block has one captured runner per setting, both alive, decoding in
strict alternation. Each figure is `run_decode`'s synchronised wall, the median of 256 timed steps:

| depth | block | order | OFF | ON | ON/OFF | median of the per-pair ratios |
|---|---|---|---:|---:|---:|---:|
| 64 | a | OFF, ON | 16.376 ms | 15.800 ms | **0.9648** | 0.9714 |
| 64 | b | ON, OFF | 16.370 ms | 15.830 ms | **0.9670** | 0.9712 |
| 32 | a | OFF, ON | 11.815 ms | 11.361 ms | **0.9616** | 0.9621 |
| 32 | b | ON, OFF | 11.770 ms | 11.392 ms | **0.9679** | 0.9673 |

- **Saving per step:** the route saves 0.56 ms at 64 rows and 0.42 ms at 32.
- **The blocks agree:** within 0.23 % at 64 rows and 0.65 % at 32, against the 1.5 % bound. The same setting agrees
  across blocks within 0.38 %. Attempt 1's single-arm level shift did not recur.
- **The traced steps' replay device time** at 64 rows is 16.41 / 16.42 ms off and 16.00 / 16.01 ms on; at 32 rows,
  11.73 / 11.70 off and 11.31 / 11.35 on.
- **Tokens:** ON emits a different token from OFF at 16,035 of 18,752 positions at 64 rows. That is reported, not
  gated: the arithmetic changes, and quality is the gate.

**Eager twins** (`torch.profiler`, 8 steps; the route's mechanism, read from the class map frozen in P119):

| depth | step device ms, off → on | cuBLAS the route replaces | its `_gemm_int4_b32_smallm` time |
|---|---|---:|---:|
| 64 | 15.22 → 14.68 (−3.6 %) | 1.88 ms (12.4 % of the step) | 1.38 ms (**0.74** of it) |
| 32 | 10.94 → 10.46 (−4.4 %) | 1.36 ms (12.5 %) | 0.81 ms (0.59) |

The 64-row tile is the weaker of the two. It reads a quarter of the cuBLAS path's bytes and still costs 0.74 of its time
(Q2 missed high). At 32 rows the tile costs 0.59. A plan census for the 64-row tile (`block_n`, split-K, warps) is the
lever this read names.

**Quality** (P117's teacher-forced passes, 64 windows of 512 + 128; the same numbers as attempt 1, because the passes are
deterministic):

| arm | mean d (nats) | mean \|d\| | max \|d\| | mean KL | argmax agreement |
|---|---:|---:|---:|---:|---:|
| `half` (floor) | +0.0038 | 0.0158 | 0.070 | 0.0108 | 0.954 |
| `chunk` (floor) | +0.0012 | 0.0166 | 0.063 | 0.0117 | 0.954 |
| **`ON64`** | **+0.0011** | 0.0142 | 0.047 | 0.0106 | 0.957 |
| **`ON32`** | **+0.0014** | 0.0155 | 0.067 | 0.0106 | 0.960 |
| `mutant_scale` | +1.150 | 1.150 | 2.07 | 1.262 | 0.579 |
| `mutant_wide` | +0.542 | 0.542 | 1.87 | 0.573 | 0.718 |

- R repeats bit for bit.
- The bar: bias ≤ 0.0138, spread ≤ 0.0332. Both subjects pass and both mutants fail.

**Memory and the GPU:**
- **The route's workspace is 11.5 MiB** (Q9). It is built at enable, so it exists with the route on and off, and an
  interleaved block holds both settings at once. Q9 is therefore graded on the workspace itself, as the maintainer
  ruled; no ON − OFF difference could show it.
- **Peak allocated with two live runners:** 25.3–25.4 GB at 64 rows and 22.8 GB at 32.
- **Busy fraction:** 0.969–0.971 in every runner.
- **The GPU log** (`nvidia-smi` every 5 s) holds only 1–2 samples per block: a timed block lasts about 9 s. They read P1,
  SM clock 2,782–2,820 MHz, memory clock 13,801 MHz, and 61–70 °C.

**Predictions** (registered; evaluated by the reducer, none gates the verdict):

| # | statistic | read | band | result |
|---|---|---:|---|---|
| Q1 | replaced cuBLAS share | 0.124 | [0.08, 0.18] | HELD |
| Q2 | K16 / cuBLAS at 64 rows | 0.736 | [0.25, 0.65] | **MISSED** (high) |
| Q3 | 64-row ratios | 0.9648, 0.9670 | [0.89, 0.97] | HELD |
| Q4 | 32-row ratios | 0.9616, 0.9679 | [0.86, 0.97] | HELD |
| Q5 | busy fraction | ≥ 0.969 | ≥ 0.90 | HELD |
| Q6 | same setting across blocks | 0.38 % | ≤ 0.5 % | HELD |
| Q7 | subjects' bias | +0.0011, +0.0014 | ±0.003 | HELD |
| Q8 | mutants' bias | +1.15, +0.54 | ≥ +0.3 | HELD |
| Q9 | the route's workspace | 11.5 MiB | ±64 MiB | HELD |

**What it means:**
- **The route is faster at both depths, at no measured quality cost.** On SC2e's served stack it serves the attention
  projections of a 32- or 64-row decode step from the int4 grid instead of a cached bf16 copy. That makes the captured
  step 3.2–3.8 % faster, with quality inside P110's bar against the arithmetic's own floor.
- **The consequence registered for DEFAULT_ON** is a separate e4b PR. It gives `E4B_ATTN_INT4_WIDE` an `auto` default
  taking 17–64 rows, when the K16 route is on and the installed grouped-nf4-gemm takes `block_m=` (capability).
  `E4B_ATTN_INT4_WIDE=0` restores the cached copy, and the serve estimate prices the shared workspace. The claims row
  `e4b.serve.p124.attn-int4-wide.qwen3-int4.5090.2026-10-09` carries the 64-row ratio.
- **Not read here:** other models, cards, prompt lengths or tile plans. Serving attainment, for which the step ratios are
  inputs to `serve_capacity`.

## Attempt 1 (`p124-5090-1`): NOISY (decides nothing; kept as the amendment's evidence)

**Host:** one RTX 5090 (sm_120, driver 595.91.07) on an AMD EPYC 7763 (256 threads), Vast instance 54939679.

**Cost:** $1.003, $0.655 of it the 62.9 GB fetch. Teardown proven at 01:07:58Z.

**Stack:**
- e4b 0.50.0 at `ee91522`; grouped-nf4-gemm 0.43.0 at `4ed26d96`;
- torch 2.8.0+cu128, triton 3.4.0, transformers 5.17.0;
- `Qwen/Qwen3-30B-A3B` at `ad44e77`: SC2e's served stack (int4 experts on 48 layers, 96 int4 RTN attention
  projections after the q/k/v fusion, the folds), built with `E4B_ATTN_INT4_WIDE=1`;
- the box: load 192 s, arms 536 s.

**Verdict by `p124_reduce.py` (as registered at `ee91522`): `NOISY`.**
- **What fired:** at 64 rows the OFF pair is 3.12 % apart, against the 1.5 % bound.
- **Clean elsewhere:** the ON pair agrees within 0.33 %, and at 32 rows both pairs within 0.05 %.
- **No VOID condition fired:**
  - the commits, revision and stack are the registered ones;
  - every runner captured every bucket and replayed its bucket 293 times with no eager step;
  - engagement was exact: one `_gemm_int4_b32_smallm` launch per projection per profiled step with the route on and
    none off, and 96 cuBLAS launches removed per step;
  - each setting's two arms emitted the same tokens;
  - G64on matched ON64 at 8,128 of 8,128 positions;
  - both mutants failed the bar.
- **Re-derivation:** the verdict re-derives byte for byte on Python 3.9 and 3.13.

**Served steps** (`run_decode`'s synchronised wall: the median of 256 timed steps, and of each half):

| depth | arm | median | first half | second half | replay device ms (traced) |
|---|---|---:|---:|---:|---:|
| 64 | `OFF_a` | 16.412 | 15.974 | 16.608 | 16.434 |
| 64 | `ON_a` | 16.399 | 16.138 | 16.632 | 16.549 |
| 64 | `ON_b` | 16.344 | 16.060 | 16.610 | 16.546 |
| 64 | `OFF_b` | **16.924** | 16.466 | 17.129 | **16.960** |
| 32 | `OFF_a` | 11.811 | 11.592 | 11.925 | 11.739 |
| 32 | `ON_a` | 11.394 | 11.157 | 11.566 | 11.356 |
| 32 | `ON_b` | 11.397 | 11.146 | 11.572 | 11.358 |
| 32 | `OFF_b` | 11.805 | 11.597 | 11.929 | 11.741 |

**Why it read NOISY: a level shift of one arm.**
- `OFF_b` at 64 rows sits about 0.5 ms above `OFF_a` in both halves, and in its replay's device time.
- Every arm's second half is 0.35–0.65 ms slower than its first: the context grows by 293 tokens per window. That
  drift is the same in every arm.
- The GPU busy fraction is 0.97 in every arm, so `OFF_b`'s offset is on the device: its clock or power state during
  that arm. A longer arm would carry the same shift, so Amendment 1 interleaves the two settings step by step instead.

**Eager twins** (`torch.profiler`, 8 steps; the route's mechanism, read from the class map frozen in P119):

| depth | step device ms, off → on | cuBLAS the route replaces | its `_gemm_int4_b32_smallm` time |
|---|---|---:|---:|
| 64 | 15.23 → 14.69 (−3.6 %) | 1.89 ms (12.4 % of the step) | 1.40 ms (0.74 of it) |
| 32 | 10.95 → 10.47 (−4.4 %) | 1.37 ms (12.5 %) | 0.82 ms (0.60) |

**Quality** (P117's teacher-forced passes, 64 windows of 512 + 128):

| arm | mean d (nats) | mean \|d\| | mean KL | argmax agreement |
|---|---:|---:|---:|---:|
| `half` (floor) | +0.0038 | 0.0158 | 0.0108 | 0.954 |
| `chunk` (floor) | +0.0012 | 0.0166 | 0.0117 | 0.954 |
| **`ON64`** | **+0.0011** | 0.0142 | 0.0106 | 0.957 |
| **`ON32`** | **+0.0014** | 0.0155 | 0.0106 | 0.960 |
| `mutant_scale` | +1.150 | 1.150 | 1.262 | 0.579 |
| `mutant_wide` | +0.542 | 0.542 | 0.573 | 0.718 |

- R repeats bit for bit. The bar is bias ≤ 0.0138 and spread ≤ 0.0332.
- Both subjects pass it, and both mutants fail it.

**Memory and the GPU:**
- Peak allocated is 22.5–22.7 GB in every 64-row arm and 21.4 GB at 32 rows.
- The route's shared workspace is 11.5 MiB. ON − OFF reads 0 because the workspace is built at enable, in both
  settings.
- The busy fraction is 0.969–0.972 in every arm.

**Predictions (registered; attempt 1 decides none of them):**

| # | statistic | read | result |
|---|---|---:|---|
| Q1 | replaced cuBLAS share | 0.124 | HELD |
| Q2 | K16 / cuBLAS at 64 rows | 0.741 | MISSED (high) |
| Q3 | 64-row ratios | 0.9992, 0.9657 | MISSED |
| Q4 | 32-row ratios | 0.9647, 0.9654 | HELD |
| Q5 | busy fraction | ≥ 0.969 | HELD |
| Q6 | same-setting pairs | 3.1 % | MISSED |
| Q7 | subjects' bias | +0.0011, +0.0014 | HELD |
| Q8 | mutants' bias | +1.15, +0.54 | HELD |
| Q9 | memory (0 by construction; see the amendment) | 0 | HELD |

**What it means:** nothing yet. At 32 rows every arm agrees, and at 64 rows the clean pairs suggest a gain, but the
registered rule read NOISY. Attempt 2 interleaves the settings so that a shift in the GPU's state lands on both.

## Reproduce

```
python bench/p124/p124_reduce.py --dir bench/p124/receipts/p124-5090-4 --e4b-sha 2ed9c11f098e279dda8b506d75d9df1b2f1a9dd6
```

Attempt 1 is read by the reducer as merged at `ee91522` (`git show ee91522:bench/p124/p124_reduce.py`). It uses the ABBA
layout, which the amended reducer no longer reads:

```
python <that file> --dir bench/p124/receipts/p124-5090-1 --e4b-sha ee91522968f2560bb04c0eb2ede9dc7e8194f9f2
```

Receipts are checked by each directory's `SHA256SUMS`, and logs are committed past the `*.log` ignore.
