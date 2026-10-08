# P122 — results: **DEFAULT_ON**. With grouped-nf4-gemm #519's chunked cumsum table, `E4B_INT4_WIDE_TILES=1` makes SC2e's 64-row decode step 4.3 % faster on Qwen3-30B-A3B int4, with identical tokens (one RTX 5090, 2026-10-08)

Registration: `bench/p122/PREREG-p122.md` (#1386, reviewed by the maintainer, merged `6357ecc`, the launch commit).
Issue: #846. Code under test: grouped-nf4-gemm #519 (`b155f1c`, the chunked table); the box is P120's at its
registered bytes.

Stack:
- e4b 0.50.0 at `6357ecc`;
- grouped-nf4-gemm 0.43.0 at `b155f1c`;
- torch 2.8.0+cu128, triton 3.4.0, transformers 5.17.0;
- `Qwen/Qwen3-30B-A3B` at `ad44e77`: SC2e's served stack (int4 experts on 48 layers, int4 RTN attention, the folds,
  fused q/k/v), bulk KV bookkeeping, device grouping; NF4 arena baked on the box.

**Verdict by `p122_reduce.py`: `DEFAULT_ON`** (ON/OFF 0.9568 and 0.9569, both ≤ 0.98). No VOID condition fired:
- the commits, revision and stack are the registered ones;
- both profile arms ran 8 eager bucket-64 steps;
- every served arm captured every bucket and replayed bucket 64 exactly 261 times with no eager step;
- engagement: 48 chained builds per step became 48 one-launch tables and no radix sort;
- each setting's two arms emitted the same tokens;
- the mutant died.

The premise held: the builder's frozen kernels are 9.1 % of the eager step.

## The reading (`p122-5090-1`)

**Host:** one RTX 5090 (sm_120, driver 595.71.05) on an AMD EPYC 7C13 (256 threads), Vast instance 54913207. Its GPU
UUID is the one P120's and P121's readings ran on (P121's forensics record its 450 W power limit). The cross-run table
comparison below is therefore on the same board, though it is still not a within-box ratio.

**Cost:** $0.837. Teardown proven at 21:32:15Z.

**Timeline:**
- premise from about 21:18Z: 7 decode-graph tests, and #519's 117 chunked tests compiled for sm_120;
- fetch until 21:25:51Z, bake until 21:26:54Z;
- the box: load 166 s, arms 132 s.

**Served 64-row step** (`run_decode`'s synchronised wall, median of 256 timed steps after 5 warm):

| arm | `E4B_INT4_WIDE_TILES` | median | min | max |
|---|---|---:|---:|---:|
| `OFF_a` | 0 | 17.526 ms | 16.279 | 18.762 |
| `ON_a` | 1 | 16.769 ms | 15.639 | 22.604 |
| `ON_b` | 1 | 16.786 ms | 15.826 | 18.862 |
| `OFF_b` | 0 | 17.541 ms | 16.397 | 18.602 |

- **ON/OFF:** 0.9568 (pair a) and 0.9569 (pair b). ON saves 0.76 ms a step.
- **Noise:** the same-setting pairs agree within 0.08 % (OFF) and 0.10 % (ON).
- **Tokens:** ON emits OFF's token at all 16,704 positions. The mutant differs at 16,254.

**Eager 64-row twin** (`torch.profiler`, 8 steps):

| | `0` (chained builder) | `1` (#519's chunked table) | P120's `1` (one-piece table) |
|---|---:|---:|---:|
| step device time | 15.82 ms | 15.42 ms | 23.43 ms |
| the builder's frozen kernels | 1.43 ms | 0 | 0 |
| `_tile_table_r1` | — | **2.51 ms** (52 µs a launch) | 10.45 ms (218 µs a launch) |

- **The chunked table is 4.2× faster than the one-piece table** at 512 routed rows and 128 experts: eight [128, 64]
  chunks in two passes, against one [128, 512] program.
- **It is still 1.75× the builder's frozen kernels**, so Q2 missed high. The step is faster anyway because the switch
  also removes the chained builder's generic glue, about 15 small kernels a layer whose names it shares with the
  route's own. The eager step saves 0.40 ms; the served step saves 0.76 ms, the extra coming from launches the graph no
  longer replays.

## Against the predictions

| prediction (written before the data) | result |
|---|---|
| Q1: `0`'s builder share of the eager step in [0.06, 0.12] | **held** (0.0906) |
| Q2: `1`'s table time / `0`'s builder time in [0.3, 1.0] | **missed high** (1.75) |
| Q3: eager ON/OFF device time in [0.88, 0.99] | **held** (0.975) |
| Q4: both served ratios in [0.86, 0.96] | **held** (0.9568, 0.9569) |
| Q5: same-setting pairs within 0.5 % | **held** (0.10 %) |

Q2's registered consequence for a high miss: chunking alone is not enough to beat the builder's own kernels, and a
multi-program table is the next lever. By this read's numbers, a table near the pairwise kernel's 0.93 ms at 256 rows
would save about 1.5 ms more a step.

## What it means (the registered consequence)

- **DEFAULT_ON:** P120's registered consequence applies. A separate e4b PR makes `E4B_INT4_WIDE_TILES` default to on
  for device-grouped calls whose one-launch table is no larger than the one read:
  - `next_pow2(E) × next_pow2(R) ≤ 128 × 512`;
  - and only when the installed grouped-nf4-gemm takes `rank=` and `rchunk=`; otherwise the chained builder, as today.
  - `E4B_INT4_WIDE_TILES=0` restores the chained builder.
  - Larger tables stay opt-in until read.
- **The register row** is `e4b.serve.p122.wide-tiles-chunked.qwen3-int4.5090.2026-10-08`. STATUS's sentence on the
  switch cites it beside P120's.
- **Next lever for this table:** a multi-program build (Q2).

## What it took

| run | status | cost | note |
|---|---|---:|---|
| `p122-prove-1` | OK, PROVED | $0.093 | Granite: every arm, premise 7 + 117, tokens identical at 10,440 positions; the chunked table 0.576 ms a step against P120's proof 1.29 (not a reading) |
| `p122-5090-1` | **OK, DEFAULT_ON** | $0.837 | the reading |

**The lane cost $0.930**, inside its $3.00 ceiling.

**Receipts** are in `receipts/p122-5090-1/`, with `SHA256SUMS`:
- `box.json` (both profile tables, every arm's steps and tokens), `verdict.json`, `summary.txt`, `forensics.txt`,
  `versions.txt`, `bake.json`;
- the logs (including both premise logs) and the teardown proof.

**Re-derive:**
```bash
python3 bench/p122/p122_reduce.py --dir bench/p122/receipts/p122-5090-1 --out /tmp/v.json --e4b-sha 6357ecc3d784a0f6aed1f0d33e1d53b8a72bb1ed
```
The output equals the committed `verdict.json` byte for byte on Python 3.9 (macOS) and 3.13 (Windows).
