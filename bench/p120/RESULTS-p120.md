# P120 — results: **SLOWER**. Building the tile table above 256 routed rows in one launch (`E4B_INT4_WIDE_TILES=1`) makes SC2e's 64-row decode step 1.44× slower on Qwen3-30B-A3B int4, with identical tokens: the one-launch table costs 10.4 ms a step at 512 routed rows × 128 experts (one RTX 5090, 2026-10-08)

Registration: `bench/p120/PREREG-p120.md` (#1360, reviewed by the maintainer, merged `b38866c`, the launch commit).
Issue: #846. Code under test: grouped-nf4-gemm #515 (`3ce2ecd8`, the cumsum rank) and e4b #1357 (`d0ae2436`, the
switch).

Stack:
- e4b 0.50.0 at `b38866c`;
- grouped-nf4-gemm 0.43.0 at `3ce2ecd8`;
- torch 2.8.0+cu128, triton 3.4.0, transformers 5.17.0;
- `Qwen/Qwen3-30B-A3B` at `ad44e77`: SC2e's served stack (int4 experts on 48 layers, int4 RTN attention, the folds,
  fused q/k/v), bulk KV bookkeeping, device grouping; NF4 arena baked on the box.

**Verdict by `p120_reduce.py`: `SLOWER`** (ON/OFF 1.4428 and 1.4419, both > 1). No VOID condition fired:
- the commits, revision and stack are the registered ones;
- both profile arms ran 8 eager bucket-64 steps;
- every served arm captured every bucket and replayed bucket 64 exactly 261 times with no eager step;
- engagement: 48 chained builds per step with `0` became 48 one-launch tables and no radix sort with `1`;
- each setting's two arms emitted the same tokens;
- the mutant died.

## The reading (`p120-5090-1`)

**Host:** one RTX 5090 (sm_120, driver 595.71.05) on an AMD EPYC 7C13 (256 threads, 1,007 GiB RAM), Vast instance
54897838.

**Cost:** $0.878. Teardown proven at 19:32:51Z.

**Timeline:**
- install, tripwire and self-test from 19:14:59Z;
- premise passed at 19:22:19Z: 7 decode-graph tests, and 66 cumsum-rank tests compiled for sm_120 in 340 s;
- fetch until 19:26:32Z, bake until 19:27:35Z;
- the box: load 176 s, arms 142 s;
- lane complete 19:32:50Z.

**Served 64-row step** (`run_decode`'s synchronised wall: the bucket-64 graph replay and the step's host work; median
of 256 timed steps after 5 warm):

| arm | `E4B_INT4_WIDE_TILES` | median | min | max |
|---|---|---:|---:|---:|
| `OFF_a` | 0 | 17.553 ms | 16.462 | 19.092 |
| `ON_a` | 1 | 25.325 ms | 24.225 | 32.542 |
| `ON_b` | 1 | 25.302 ms | 24.204 | 25.950 |
| `OFF_b` | 0 | 17.548 ms | 16.468 | 18.414 |

- **ON/OFF:** 1.4428 (pair a) and 1.4419 (pair b); ON costs 7.76 ms more a step.
- **Noise:** the same-setting pairs agree within 0.03 % (OFF) and 0.09 % (ON), far inside the 1.5 % bound.
- **Tokens:** ON emits OFF's token at all 16,704 positions (261 steps × 64 rows). The mutant differs at 16,254, from
  step 0.

**Eager 64-row twin** (`torch.profiler`, 8 steps):

| | `0` (chained builder) | `1` (one launch) |
|---|---:|---:|
| step device time | 15.82 ms | 23.43 ms |
| the builder's frozen kernels | 1.43 ms (48 radix sorts) | 0 |
| `_tile_table_r1` | — | **10.45 ms** (48 launches, 218 µs each) |

**The kernel, not the graph, explains the verdict.** The eager step is already 7.62 ms slower, and the captured step
7.76 ms slower. The one-launch table at 512 routed rows and 128 experts runs one program over a [128, 512] hit matrix
and its cumsum.

On the proof's Granite (40 experts, so a [64, 512] tile; not a reading) the same launch took 40 µs and the step got
2.9 % faster. Doubling the tile multiplied the launch time by 5.4. That is consistent with a single program that no
longer fits its working set, a hypothesis this lane does not test.

**What a faster table could recover** (by subtraction; it assumes the rest of the step is unchanged, as the identical
route and tokens suggest):
- the chained builder costs 10.45 − 7.62 = **2.83 ms** of the eager step's device time (P119 grouped 2.87 ms by name);
- in the served step, 10.45 − 7.76 = **2.69 ms**, about 15 % of 17.55 ms.

## Against the predictions

| prediction (written before the data) | result |
|---|---|
| Q1: `0`'s builder share of the eager step in [0.06, 0.12] | **held** (0.0906) |
| Q2: `1`'s table time / `0`'s builder time in [0.5, 1.2] | **missed high** (7.29) |
| Q3: eager ON/OFF device time in [0.90, 0.98] | **missed high** (1.48) |
| Q4: both served ratios in [0.88, 0.97] | **missed high** (1.44, 1.44) |
| Q5: same-setting pairs within 0.5 % | **held** (0.09 %) |

Q2's registered consequence applies: the one-launch kernel limits the lever, and a launch-shape lane is next. Q3 and
Q4 missed on the slow side, a direction their registered consequences did not anticipate. Q3 settles Q4's question: the
eager step is as slow as the captured one, so the graph is not the cause.

## What it means (the registered consequences)

- **SLOWER:** `E4B_INT4_WIDE_TILES` stays opt-in, and `0` stays the default. On Qwen3-30B-A3B at 64 rows, `1` costs
  44 % a step.
- **The lever is real, the kernel is the limit.** The chained builder costs about 2.7–2.8 ms of a 17.5 ms 64-row step.
  A one-launch table near the pairwise kernel's cost at 256 rows (0.93 ms a step) would recover most of it.
- **Worth registering:** a grouped-nf4-gemm launch-shape lane for the cumsum table above 256 rows: more warps, or a
  multi-program table that splits the [E, R] hit matrix and its prefix. Correctness on the A2000; speed read as here.
  That is the next registration for this lever, not a conclusion of this one.

## What it took

| run | status | cost | note |
|---|---|---:|---|
| `p120-prove-1` | OK, PROVED | $0.201 | Granite: every arm, premise 7 + 66, tokens identical at 10,440 positions, the mutant dead; its timings are not a reading |
| `p120-5090-1` | **OK, SLOWER** | $0.878 | the reading |

**The lane cost $1.079**, inside its $3.00 ceiling.

**Receipts** are in `receipts/p120-5090-1/`, with `SHA256SUMS`:
- `box.json` (both profile tables, every arm's steps and tokens), `verdict.json`, `summary.txt`, `forensics.txt`,
  `versions.txt`, `bake.json`;
- the logs, including both premise logs, and the teardown proof.

These committed copies are the public record of the reading.

**Re-derive:**
```bash
python3 bench/p120/p120_reduce.py --dir bench/p120/receipts/p120-5090-1 --out /tmp/v.json --e4b-sha b38866c609d663cacbace283fb55b2060dffd958
```
The output equals the committed `verdict.json` byte for byte on Python 3.9 (macOS) and 3.13 (Windows, line endings
aside).
