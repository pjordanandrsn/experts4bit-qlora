# P120 — does the one-launch tile table above 256 routed rows (`E4B_INT4_WIDE_TILES=1`) make SC2e's 64-row decode step faster, with identical tokens? A within-box speed read on one RTX 5090 (registered 2026-10-08, before any run)

Issue: experts4bit-qlora#846 (the serving campaign; the owner's no-ask tier for a single run under $15). Lane number
claimed by `prereg/p120` (pushed 2026-10-08T18:15:55Z). Follows P119 (#1353). The code under test is
grouped-nf4-gemm #515 (merged `3ce2ecd8`: `build_group_tiles_fused(..., rank="cumsum")`) and e4b #1357 (merged
`d0ae2436`: `E4B_INT4_WIDE_TILES`, opt-in). Box, runner and driver are derived from P119's.

## Why this lane

- **P119** read SC2e's 64-row decode step at 15.64 ms of device time. Above 256 routed rows (64 rows × top-8) the tile
  table comes from the chained builder: argsort, scatter_add, two cumsums, searchsorted, aranges and a dozen small
  elementwise ops per MoE layer. Its uniquely named kernels alone are **1.39 ms a step (8.9 %)**. At 32 rows, the
  one-launch `_tile_table_r1` builds the same table in 0.93 ms.
- **#515** ranks rows by a cumsum along the existing hit matrix instead of the O(R²) pairwise compare, so one launch
  takes up to 1,024 rows. Its tables are the chained builder's integers: CPU interpreter tests pin all five tables
  bit for bit, and the premise re-runs them compiled on the card. **#1357** routes device-grouped calls of 257 to
  1,024 rows to it under `E4B_INT4_WIDE_TILES=1`.

**The question:** on SC2e's served stack, is the captured 64-row decode step faster with `1` than with `0`, by more
than the box's own noise, with every token identical?

## Instrument

**The model is SC2e's served stack**, built as P117 and P119 built it: `PagedServeConfig.from_env()` + `build_engine`
with `bench/sc1/sc1_run.sh`'s `SPEEDENV`, `ROUTEENV` and `E4B_PAGED_FUSE_QKV=1`, byte for byte. It is built eager with
one slot of 768 tokens; the arms build their own pools. The NF4 arena is baked on the box by P39's `k8_bake.py`.

**The windows:** 64 windows of 512 tokens from wikitext-2-raw test, as P117 takes them.

**The arms** (`bench/p120/p120_box.py`; each a fresh `Fp8PagedKV` and `PagedModelRunner`, device grouping on, bulk KV
bookkeeping; the knob is set before the arm starts and read on every call):

| arm | setting | what runs |
|---|---|---|
| profile `off`, `on` | `0`, `1` | P119's `decode_bracket` at 64 rows on buckets 1–64: 3 warm, then 8 padded eager steps (`capture=False`) under `torch.profiler`. The premise (`0`) and the engagement (`1`) |
| served `OFF_a`, `ON_a`, `ON_b`, `OFF_b` (ABBA) | `0`, `1`, `1`, `0` | every bucket 1–64 captured under the arm's setting, 64 windows prefilled, 5 warm decode steps, then **256 timed steps**. Each step is one bucket-64 replay; its time is `run_decode`'s synchronised wall (the replay and the step's host work). Every decode step's tokens are recorded |
| `MUTANT` | `1` | as `ON`, with grouped-nf4-gemm's builder wrapped to shift every live tile's expert id by one (`(grp + 1) % E`); the signature is kept, so e4b's `rank=` check passes |

**The frozen kernel names.** `BUILDER` covers the kernels only the chained builder launches at 64 rows: radix sort, the
accumulating scatter, the two cumsum scans and their inits, searchsorted and arange (P119's d64 table against d32's).
That sum is a lower bound on the builder's device time. `TABLE` is `_tile_table_r1`.

## The rule (`bench/p120/p120_reduce.py`, self-tested on 30 cases)

Every bound is a ratio inside the box. The first that fires is the verdict:

1. **VOID**, any of:
   - another e4b or grouped-nf4-gemm commit, or another model revision;
   - on the reading, a stack other than SC2e's;
   - rows outside 33–64 (the step must be one bucket-64 piece);
   - a profile arm other than 8 eager bucket-64 steps under its setting;
   - a served arm missing, a bucket not captured, or bucket 64 not replayed exactly 261 times with no eager step and no
     other bucket;
   - KV bookkeeping not bulk, or a step or token count short;
   - **not engaged**: with `1` a radix sort remains, or the one-launch tables per step differ from the chained builds
     `0` showed;
   - **nondeterministic**: a setting's two arms emit different tokens;
   - **the mutant survived**: its tokens equal `OFF_a`'s.
2. **PREMISE_ABSENT** (the reading only): with `0`, fewer chained builds per step than MoE layers, or the `BUILDER`
   kernels under **5 %** of the eager step's device time.
3. **TOKENS_DIFFER**: `ON_a` emits any token `OFF_a` does not, at any of the 261 × 64 positions.
4. **NOISY**: `OFF_b/OFF_a` or `ON_b/ON_a` (median step) is more than **1.5 %** from 1.
5. **DEFAULT_ON**: `ON_a/OFF_a` and `ON_b/OFF_b` are both **≤ 0.98**.
6. **SLOWER**: both > 1. **NO_GAIN**: otherwise.

Medians are taken by sorting and floats summed with `math.fsum`, so the verdict file is byte-identical on any Python.

## Predictions (written before any data; evaluated by the reducer, none gates the verdict)

| # | prediction | if it holds | if it misses |
|---|---|---|---|
| Q1 | `0`'s `BUILDER` share of the eager 64-row step in [0.06, 0.12] (P119: 0.089) | the lever is as P119 read it | low: the premise may fail. High: more to gain than predicted |
| Q2 | `1`'s `_tile_table_r1` time / `0`'s `BUILDER` time in [0.5, 1.2]. Basis: 0.93 ms at 256 rows (pairwise) on the same 65,536-element working set, over 1.39 | the cumsum rank costs what the pairwise one does at half the rows | high: the one-launch kernel limits the lever, and a launch-shape lane (warps, a split table) is next. Low: more headroom |
| Q3 | eager `ON/OFF` device time in [0.90, 0.98] | the saving is device time | the saving is launch gaps, which only the captured step shows |
| Q4 | both served ratios in [0.88, 0.97] | the gain is the builder's kernels and their in-graph launches, net of the one-launch table, over SC2e's 18.4 ms step | high: the chained builder costs less in a graph than eager (Q3 says whether). Low: the chained glue's launch gaps cost more than predicted, a lead for other chained glue |
| Q5 | both same-setting pairs agree within 0.5 % | the 2 % bar is four times the noise | the bar sits near the noise; longer arms next time |

## Consequence, registered now

- **DEFAULT_ON:** a separate e4b PR makes `1` the default for device-grouped calls whose one-launch table is no larger
  than the one read here: `next_pow2(E) × next_pow2(R) ≤ 128 × 512`, when the installed grouped-nf4-gemm takes `rank=`
  (otherwise the chained builder, as today). `E4B_INT4_WIDE_TILES=0` restores the chained builder; larger tables stay
  opt-in until read. The read PR adds a claims row.
- **NO_GAIN or SLOWER:** the knob stays opt-in. RESULTS names, from Q2 and Q3, whether the one-launch kernel or the
  graph explains it, and whether a launch-shape lane is worth registering.
- **TOKENS_DIFFER:** the knob stays opt-in and an issue is opened the same day. The premise proved the integers on the
  card, so the fault lies between the builder and the route; RESULTS names the first differing position.
- **NOISY:** no default; a rerun with longer arms is an amendment.
- **PREMISE_ABSENT:** no default; RESULTS names what replaced the chained builder's time on this stack.
- **VOID or NO_READING:** no consequence. A rerun is a new attempt, or an amendment if the cause is the harness.

## The proving rental

No local card runs the fp8 paged KV or the bucketed path. `tests/test_p120_box.py` covers the box on CPU with the
runner, the pool, the profiler and grouped-nf4-gemm's module stood in:
- each arm runs under its own setting, set before capture;
- one bucket-64 replay per step, warm then timed;
- every token is recorded;
- the mutant shifts only live tiles, keeps the signature, is restored, and moves the tokens;
- the reducer reads the record.

**The proof** runs the whole box end to end on `ibm-granite/granite-3.1-3b-a800m-instruct` @
`a02780686e08a03fe0d2679a293b5c74a90efa89` (its NF4 store, no int4 levers; 40 experts, top-8) with 40 rows. That is
still one bucket-64 piece, so 512 routed rows take the switch. It covers:
- the refusals, the install and the tripwire;
- the reducer self-test;
- both premise tests;
- fetch, bake, every arm and the reducer.

It is **PROVED** iff the lane exits 0 with a verdict other than VOID, NO_READING or TOKENS_DIFFER. The premise share
does not apply to the proof's model; its timings are not a reading.

## Budget and STOP rules

- **Proof:** one RTX 5090 (Vast verified/secure), guard 0.75 h at ≤ $0.75/h, estimate $0.5625.
- **Reading:** one RTX 5090, guard 1.5 h at ≤ $0.75/h, estimate $1.125. Expected about 40 minutes, most of it the
  61 GB fetch; the box itself about 8.
- **Lane ceiling:** $3.00, inside the owner's $15 no-ask tier.
- **STOP-1:** the refusals run before any install: dud box (10), card class (15), disk < 150 GB (13), host RAM < 60 GiB
  (16). The premise runs before the fetch (25): `tests/test_decode_graph_buckets.py`, 7 passed, none skipped; and
  grouped-nf4-gemm's `kernel/test_tile_table_cumsum_interp.py` at `3ce2ecd8` (staged as a byte copy), compiled for
  the card with `TRITON_INTERPRET=0`, 66 passed, none skipped.
- **STOP-2:** every time-left check is a per-mode variable that fits its own guard (enforced by
  `tests/test_p120_staged_pin.py`).
- **STOP-3:** a VOID is not retried inside the same launch.
- **STOP-4:** the driver refuses a dirty tree or a staged file that differs from `bench/p120/staged.sha256`.

## What this lane cannot say

- **Nothing about serving attainment.** No server or driver runs here. The step ratio is an input to `serve_capacity`,
  not a ceiling.
- **Nothing about other table sizes.** Bucket 64 always pads to 64 rows, so the read is at 512 routed rows with 128
  experts. Other expert counts and top-k values are not read, and neither are tables up to 1,024 rows.
- **Nothing about prefill.** Prefill chunks route more than 1,024 rows and keep the chained builder.
- **Nothing about quality beyond the token gate.** The tables are integer-identical by construction and by the
  premise; the gate checks the tokens.
- **Nothing absolute across boxes.** grouped-nf4-gemm is 0.43.0 here, not P119's 0.42.0, and the board differs, so
  only ratios within this box are read.

## Receipts

Fetched to the run directory's `p120/` and committed to `bench/p120/receipts/<run>/`:
- `box.json` (every arm's steps and tokens, both profile tables), `verdict.json`, `summary.txt`, `forensics.txt`,
  `versions.txt`, `bake.json`;
- the logs (including both premise logs), and the teardown proof;
- `SHA256SUMS`.

`RESULTS-p120.md` is written from those files.
