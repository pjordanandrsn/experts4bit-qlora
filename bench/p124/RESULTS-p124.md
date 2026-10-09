# P124 — results so far: attempt 1 read **NOISY**; Amendment 1 interleaves the served arms, and attempt 2 decides

Registration: `bench/p124/PREREG-p124.md` (#1415, reviewed by the maintainer, merged `ee91522`, the launch commit), with
its Amendment 1 section. Issue: #846. Code under test:
- grouped-nf4-gemm #522 (`4ed26d96`, `block_m=`);
- e4b #1410 (`170b1532`, `E4B_ATTN_INT4_WIDE`).

**Nothing below decides anything.** The registered rule read attempt 1 NOISY, so no default moves and no claim is made.
Attempt 2, under Amendment 1, decides.

## Proof (`p124-prove-1`)

**PROVED** on Granite-3.1-3B-A800M with int4 RTN attention ($0.175):
- the premise held on the card: 7, 25 and 11 tests passed, none skipped;
- engagement was exact over 128 projections;
- G64on matched ON64 at 1,240 of 1,240 positions;
- both mutants failed the bar.

The reducer read DEFAULT_ON_32 there. That is not a reading.

## Attempt 1 (`p124-5090-1`): NOISY

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
python bench/p124/p124_reduce.py --dir bench/p124/receipts/p124-5090-1 --e4b-sha ee91522968f2560bb04c0eb2ede9dc7e8194f9f2
```

Run it with `p124_reduce.py` as merged at `ee91522` (`git show ee91522:bench/p124/p124_reduce.py`). The amended
reducer reads attempt 2's block layout. Receipts are checked by `SHA256SUMS`, and logs are committed past the `*.log`
ignore.
