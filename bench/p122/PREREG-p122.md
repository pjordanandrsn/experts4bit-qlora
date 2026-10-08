# P122 — P120's question again on the chunked cumsum tile table: does `E4B_INT4_WIDE_TILES=1` make SC2e's 64-row decode step faster, with identical tokens, once grouped-nf4-gemm #519 builds the table in row chunks? One RTX 5090 (registered 2026-10-08, before any run)

Issue: experts4bit-qlora#846 (the serving campaign; the owner's no-ask tier for a single run under $15). Lane number
claimed by `prereg/p122` (pushed 2026-10-08T20:37:32Z). The maintainer asked for this read when approving
grouped-nf4-gemm #519 (bus, 20:36Z): P120's OFF and ON, plus the chunked table's time. Box, rule and runner are P120's.

## Why this lane

- **P120 read SLOWER** (#1374): ON/OFF 1.443 on the 64-row step. The kernel explained it, not the graph:
  `_tile_table_r1` cost 10.45 ms a step at 512 routed rows and 128 experts, 218 µs a launch, against 40 µs at 40
  experts. Doubling the [E, R] tile multiplied the time 5.4×.
- **By subtraction the lever is real.** The chained builder costs about 2.7–2.8 ms of the 17.55 ms step (P120's
  RESULTS).
- **#519 walks the rows in chunks.** `_tile_table_r1` gets `RCHUNK`: two passes over the rows with a per-expert carry,
  so no tile exceeds [EB, RCHUNK]. The wrapper chunks automatically once `next_pow2(E) × next_pow2(R)` exceeds 8,192:
  64-row chunks at 128 experts. The integers are the same. The interpreter tests cover 1–1,024 rows at 40, 128 and 256
  experts, and the same tests compiled on the A2000 pass (correctness only).

**The question:** with the chunked table, is the captured 64-row step faster with `1` than with `0`, by more than the
box's own noise, with every token identical?

## Instrument

**P120's, unchanged:** `bench/p120/p120_box.py` runs at P120's registered bytes, built on SC2e's int4 stack exactly as
P120 built it. It runs:
- the 64-row eager profile with the knob `0` and `1`: the premise and the engagement, and **the chunked table's device
  time** (`table_ms`, the `_tile_table_r1` kernels);
- four captured-graph arms, OFF_a, ON_a, ON_b and OFF_b, each with 5 warm and **256 timed steps**, every token recorded;
- the mutant (live tiles' expert ids shifted by one).

What changes is grouped-nf4-gemm, pinned at **`b155f1c`** (0.43.0 + #519, its merge commit), and the tripwire, which asserts `rchunk` and
`_cumsum_rchunk(128, 512) == 64`.

## The rule (`bench/p122/p122_reduce.py`: P120's, self-tested on 30 cases)

P120's rule, unchanged: every bound a ratio inside the box, and the first that fires is the verdict.

1. **VOID**: P120's conditions, including not engaged, nondeterministic, and the mutant surviving.
2. **PREMISE_ABSENT** (the reading only): fewer chained builds than MoE layers, or the builder's frozen kernels
   under **5 %** of the eager step.
3. **TOKENS_DIFFER**: ON emits any token OFF does not.
4. **NOISY**: a same-setting pair differs by more than **1.5 %**.
5. **DEFAULT_ON**: ON/OFF in the two ABBA pairs both **≤ 0.98**.
6. **SLOWER** if both are > 1; **NO_GAIN** otherwise.

## Predictions (written before any data; evaluated by the reducer, none gates the verdict)

| # | prediction | if it holds | if it misses |
|---|---|---|---|
| Q1 | `0`'s builder share of the eager step in [0.06, 0.12] (P120: 0.0906) | the lever is where P119 and P120 read it | the premise moved; RESULTS says what |
| Q2 | `1`'s table time / `0`'s builder time in [0.3, 1.0]. Basis: eight [128, 64] chunks in two passes; the pairwise table at 256 rows ([256, 256], 65,536 lanes) cost 0.93 ms a step, so about 0.4–1.4 ms over 1.43 | the chunks fit the program's registers | high: chunking is not enough; a multi-program table is next. Low: more headroom |
| Q3 | eager ON/OFF device time in [0.88, 0.99] | the eager step gains what the builder's kernels and glue cost, net of the table | as P120: the table, not the graph, decides |
| Q4 | both served ratios in [0.86, 0.96]. Basis: the chained builder's ~2.7 ms in the served step (P120, by subtraction), net of Q2's table, over 17.55 ms | the lever pays as the subtraction said | high: the table or its launch costs more in the graph than eager. Low: more than the subtraction |
| Q5 | same-setting pairs within 0.5 % (P120: 0.09 %) | the 2 % bar is four times the noise | longer arms next time |

## Consequence, registered now

- **DEFAULT_ON:** P120's registered consequence applies. A separate e4b PR makes `1` the default for device-grouped
  calls whose one-launch table is no larger than the one read (`next_pow2(E) × next_pow2(R) ≤ 128 × 512`), when the
  installed grouped-nf4-gemm takes `rank=` and `rchunk=`. `0` restores the chained builder. The read PR adds a claims row.
- **NO_GAIN or SLOWER:** the knob stays opt-in. RESULTS names from Q2 and Q3 whether a multi-program table is worth
  registering.
- **TOKENS_DIFFER:** the knob stays opt-in, and an issue is opened on grouped-nf4-gemm the same day.
- **NOISY, PREMISE_ABSENT, VOID or NO_READING:** no default change. A rerun is an amendment.

## The proving rental

P120's proof shape: Granite NF4 with 40 rows, still one bucket-64 piece, so 512 routed rows take the switch. It covers:
- the refusals, the install and the tripwire;
- the reducer self-test;
- both premise tests;
- fetch, bake, every arm and the reducer.

It is **PROVED** iff the lane exits 0 with a verdict other than VOID, NO_READING or TOKENS_DIFFER. Its timings are not
a reading.

## Budget and STOP rules

- **Proof:** one RTX 5090 (Vast verified/secure), guard 0.75 h at ≤ $0.75/h, estimate $0.5625.
- **Reading:** one RTX 5090, guard 1.5 h at ≤ $0.75/h, estimate $1.125. Expected about 25–40 minutes, mostly the
  61 GB fetch; the box about 6.
- **Lane ceiling:** $3.00, inside the owner's $15 no-ask tier.
- **STOP-1:** the refusals run before any install. The premise runs before the fetch (25):
  - `tests/test_decode_graph_buckets.py`: 7 passed, none skipped;
  - grouped-nf4-gemm's `kernel/test_tile_table_cumsum_chunked_interp.py` at #519's merge commit (staged as a byte copy),
    compiled for the card with `TRITON_INTERPRET=0`: 117 passed, none skipped.
- **STOP-2:** every time-left check fits its own guard (`tests/test_p122_staged_pin.py`).
- **STOP-3:** a VOID is not retried inside the same launch.
- **STOP-4:** the driver refuses a dirty tree or a staged file that differs from `bench/p122/staged.sha256`.

## What this lane cannot say

P120's limits apply:
- no serving attainment;
- only 512 routed rows at 128 experts;
- no prefill (it keeps the chained builder);
- nothing absolute across boxes.

The chunked table's time from P120's box is comparable with P120's 10.45 ms only as a cross-box observation, not a
within-box ratio.

## Receipts

Fetched to the run directory's `p122/` and committed to `bench/p122/receipts/<run>/`:
- `box.json`, `verdict.json`, `summary.txt`, `forensics.txt`, `versions.txt`, `bake.json`;
- the logs (force-added past the `*.log` ignore) and the teardown proof;
- `SHA256SUMS`.

`RESULTS-p122.md` is written from those files.
