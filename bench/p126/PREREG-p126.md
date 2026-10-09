# P126 — does splitting the one-launch cumsum tile table over P programs (`E4B_INT4_TILE_PROGRAMS=P`) make SC2e's 64-row decode step faster, with identical tokens? One RTX 5090 (registered 2026-10-09, before any run)

Issue: experts4bit-qlora#846 (the serving campaign; the owner's no-ask tier for a single run under $15). Lane number
claimed by `prereg/p126` (pushed 2026-10-09T04:15:33Z). Follows P122 (#1393) and P124 (#1415, #1421, #1427). The
maintainer's GO for this lever (bus, 2026-10-09T03:38Z) set its conditions:
- a fixed set of P chosen before any data: **P ∈ {1, 4, 8}**;
- P124 Amendment 1's interleaved blocks as written;
- a premise that pins a grouped-nf4-gemm commit carrying #524.

The code under test:
- grouped-nf4-gemm #524 (`build_group_tiles_fused(..., rank="cumsum", programs=P)`: `_tile_table_cumsum_mp`);
- e4b #1433 (`E4B_INT4_TILE_PROGRAMS`, opt-in).

grouped-nf4-gemm is pinned at **`<#524's merge commit>`**, after the 0.44.0 tag.

## Why this lane

- **The table is still one program.** P122 read the chunked cumsum table (#519) at **2.51 ms** of Qwen3-30B-A3B's
  64-row eager step: 52 µs a launch over 48 MoE layers, about 16 % of the step, and 1.75× the chained builder's named
  kernels. It walks 512 rows × 128 experts twice in `[128, 64]` chunks, once to count and once to rank and scatter.
  `E4B_INT4_WIDE_TILES=auto` (#1404) makes it the default table at 64 rows.
- **#524 splits it.** P programs each own a slice of the experts. Every program takes all the counts from one
  `tl.histogram` of the ids, then ranks and writes only its own experts' rows. The tables are the same integers at
  every P (the CPU interpreter and compiled tests). The kernel's `OWNERS` count shows every row written exactly once.
  #1433 passes `programs=P` under `E4B_INT4_TILE_PROGRAMS=P`.

**The question:** on SC2e's served stack, is the captured 64-row decode step faster with the table split over 4 or 8
programs than with one, by more than the box's own noise, with every token identical?

## Instrument

**The model is SC2e's served stack**, built as P117 to P124 built it:
- `PagedServeConfig.from_env()` + `build_engine` with `bench/sc1/sc1_run.sh`'s `SPEEDENV`, `ROUTEENV` and
  `E4B_PAGED_FUSE_QKV=1`, byte for byte;
- built eager with one slot of 768 tokens and no prefill graph;
- the NF4 arena baked on the box by P39's `k8_bake.py`;
- every other knob at its default: `E4B_INT4_WIDE_TILES` (`auto`: the cumsum table at 64 rows) and
  `E4B_ATTN_INT4_WIDE` (`auto`).

**The setting** is `E4B_INT4_TILE_PROGRAMS`. It is read on every call, and the box sets it before each runner's
captures or profile and unsets it after. The windows are 64 windows of 512 tokens of wikitext-2-raw test, as P117
takes them.

**The arms** (`bench/p126/p126_box.py`; each runner a fresh `Fp8PagedKV` and `PagedModelRunner`, device grouping on,
bulk KV bookkeeping):

| arm | setting | what runs |
|---|---|---|
| profile `1`, `4`, `8` | P | P119's `decode_bracket` at 64 rows on buckets 1–64: 3 warm, then 8 padded eager steps under `torch.profiler`. The premise (P = 1) and the mechanism (P = 4, 8) |
| served blocks, for each candidate in (4, 8): `a`, `b` | 1 and the candidate | one runner per setting, both alive, every bucket captured under its setting, the windows prefilled, then 5 warm, **256 timed steps** and 32 traced steps of each, in strict alternation step by step. Block `a` runs P = 1 first in every pair, block `b` the candidate first. Every decode step's tokens, the block's peak memory and its GPU log are recorded |
| `MUTANT` | 8 | P120's served arm with P120's `_Mutant` (every live tile's expert id shifted by one, the builder's signature kept, so the `programs=` capability still reads true). Its tokens must differ from P = 1's |

Two runners per block, not three, keep the second pool's memory at P124's measured peak (25.4 GB with two runners).

**Derived numbers:**
- **Premise and mechanism:** P = 1's `_tile_table_r1` time is the premise. P = 4's and P = 8's `_tile_table_cumsum_mp`
  times are the mechanism.
- **Block ratio:** for each block, `ratio = median(candidate) / median(P = 1)`.
- **Busy fraction:** each traced step's replay device time (`dec_issue − dec_prep`) over its wall.

## The rule (`bench/p126/p126_reduce.py`, self-tested on 35 cases)

Every bound is a ratio inside the box. The first that fires is the verdict:

1. **NO_READING:** no box record, or the premise tests failed.
2. **VOID**, any of:
   - another e4b or grouped-nf4-gemm commit, or another model revision;
   - on the reading, a stack other than SC2e's; a tile knob set outside the box;
   - rows outside 33–64 (the step must be one bucket-64 piece);
   - a profile arm other than 8 eager bucket-64 steps under its setting;
   - a block missing or out of its order, or a runner not captured under its setting;
   - a runner that did not replay bucket 64 exactly 293 times (5 + 256 + 32) with no eager step and no other bucket;
   - a step, token or traced-step count short; no peak memory;
   - **not engaged**: P = 1 not profiling one `_tile_table_r1` launch per MoE layer and no split table, or P > 1 not
     profiling one `_tile_table_cumsum_mp` launch per MoE layer and no one-program table, or any radix sort;
   - **nondeterministic**: a setting emits different tokens in two blocks;
   - **the mutant survived**: its tokens equal P = 1's.
3. **PREMISE_ABSENT** (the reading only): P = 1's one-program table is under **5 %** of the 64-row eager step's device
   time.
4. **TOKENS_DIFFER**: a candidate emits a token P = 1 does not, in any block, at any of the 293 × 64 positions.
5. **NOISY**: for a candidate, its two block ratios differ by more than **1.5 %**.
6. **DEFAULT_ON_4** or **DEFAULT_ON_8**: of the candidates whose two block ratios are both at most **0.98**, the one with
   the lower mean ratio (ties to the smaller P).
7. **SLOWER**: every ratio above 1. **NO_GAIN**: otherwise.

**Reported, never gated:** the median of each block's per-pair ratios, every runner's busy fraction, each block's peak
memory and GPU log, and the profile tables. Medians are taken by sorting and floats summed with `math.fsum`, so the
verdict file is byte-identical on any Python.

## Predictions (written before any data; evaluated by the reducer, none gates the verdict)

| # | prediction | if it holds | if it misses |
|---|---|---|---|
| Q1 | P = 1's table share of the eager 64-row step in [0.10, 0.22] (P122: 0.163) | the lever is as P122 read it | low: the premise may fail. High: more to gain |
| Q2 | P = 8's split table / P = 1's table in [0.10, 0.40]. Basis: an eighth of the rank work, plus a histogram and the launch | the split scales with P | high: the histogram, the per-program setup or the launch bound it; a different split is next |
| Q3 | P = 4's in [0.20, 0.55] | as Q2 | as Q2 |
| Q4 | both P = 8 block ratios in [0.86, 0.95]. Basis: about 1.9 ms saved of P124's ~15.8 ms served step | the saving is the table's | high: the captured step hides less than the eager twin shows |
| Q5 | both P = 4 block ratios in [0.88, 0.96] | as Q4 | as Q4 |
| Q6 | every runner's busy fraction ≥ 0.90 | the step is device-bound | host work bounds it; RESULTS says which |
| Q7 | each candidate's block ratios agree within 0.5 % | the interleave holds the noise | longer blocks next time |
| Q8 | the mutant differs from P = 1 at ≥ 50 % of its positions | the token gate can fail | the instrument has lost sensitivity |

## Consequence, registered now

- **DEFAULT_ON_4** or **DEFAULT_ON_8:** a separate e4b PR gives `E4B_INT4_TILE_PROGRAMS` an `auto` default. It takes
  the licensed P for the tables the lane read (`next_pow2(E) × next_pow2(R) ≤ 128 × 512`) when the installed
  grouped-nf4-gemm takes `programs=` (capability). `E4B_INT4_TILE_PROGRAMS=1` restores the one-program table. The read
  PR adds a claims row.
- **NO_GAIN or SLOWER:** the knob stays opt-in. RESULTS names from Q2 and Q3 whether the split or the step explains it.
- **TOKENS_DIFFER:** the knob stays opt-in and an issue is opened the same day. The tables are the same integers by
  test, so the fault lies between the builder and the route; RESULTS names the first differing position.
- **NOISY:** no default; a rerun with longer blocks is an amendment.
- **PREMISE_ABSENT:** no default; RESULTS names what replaced the table's time.
- **VOID or NO_READING:** no consequence. A rerun is a new attempt, or an amendment if the cause is the harness.

## The proving rental

No local card runs the fp8 paged KV or the bucketed path. `tests/test_p126_box.py` covers the box on CPU with the
runner, the pool, the profiler and the builder stood in:
- every arm runs under its own setting, and the box leaves the knob unset;
- each block alternates its two runners in lockstep, in its order;
- the mutant keeps the builder's signature and moves the tokens;
- the reducer reads the record.

**The proof** runs the whole box end to end on `ibm-granite/granite-3.1-3b-a800m-instruct` @
`a02780686e08a03fe0d2679a293b5c74a90efa89` (its NF4 store; no int4 levers) with 40 rows. That is 320 routed rows at 40
experts, so the cumsum table runs, still as one bucket-64 piece. It covers:
- the refusals, the install and the tripwire;
- the reducer self-test and the premise;
- fetch, bake, every arm and the reducer.

It is **PROVED** iff the lane exits 0 with a verdict other than VOID, NO_READING or TOKENS_DIFFER. The premise share
does not apply to the proof's model; its numbers are not a reading.

## Budget and STOP rules

- **Proof:** one RTX 5090 (Vast verified/secure), guard 0.75 h at ≤ $0.75/h, estimate $0.5625.
- **Reading:** one RTX 5090, guard 1.5 h at ≤ $0.75/h, estimate $1.125. Expected about 40 minutes, most of it the
  61 GB fetch; the box itself about 10.
- **Lane ceiling:** $3.00, inside the owner's $15 no-ask tier. A pre-flight NOT_RUN is retried under the maintainer's
  standing rule.
- **STOP-1:** the refusals run before any install: dud box (10), card class (15), disk < 150 GB (13), host RAM < 60 GiB
  (16). The premise runs before the fetch (25), each none skipped:
  - `tests/test_decode_graph_buckets.py`, 7 passed;
  - grouped-nf4-gemm's `kernel/test_tile_table_programs_interp.py` at the pinned commit (staged as a byte copy),
    compiled for the card with `TRITON_INTERPRET=0`, 202 passed.
- **STOP-2:** every time-left check is a per-mode variable that fits its own guard (enforced by
  `tests/test_p126_staged_pin.py`).
- **STOP-3:** a VOID is not retried inside the same launch.
- **STOP-4:** the driver refuses a dirty tree or a staged file that differs from `bench/p126/staged.sha256`.

## What this lane cannot say

- **Nothing about serving attainment.** No server or driver runs here.
- **Nothing about other table sizes.** Bucket 64 pads to 64 rows, so the read is at 512 routed rows with 128 experts.
- **Nothing about P outside {1, 4, 8}.**
- **Nothing absolute across boxes.** Only ratios inside this box are read.

## Receipts

Fetched to the run directory's `p126/` and committed to `bench/p126/receipts/<run>/`:
- `box.json`, `verdict.json`, `summary.txt`, `forensics.txt`, `versions.txt`, `bake.json`;
- the eight traced-step files;
- the logs (force-added past the `*.log` ignore) and the teardown proof;
- `SHA256SUMS`.

`RESULTS-p126.md` is written from those files.
