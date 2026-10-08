# P124 — does the int4 small-M GEMM for the attention projections above 16 rows (`E4B_ATTN_INT4_WIDE=1`) make SC2e's 64- and 32-row decode steps faster without costing teacher-forced quality? One RTX 5090 (registered 2026-10-08, before any run)

Issue: experts4bit-qlora#846 (the serving campaign; the owner's no-ask tier for a single run under $15). Lane number
claimed by `prereg/p124` (pushed 2026-10-08T23:43:41Z). Follows P119 (#1353), P120 (#1374) and P122 (#1393). The maintainer's GO for this lever
(bus, 2026-10-08T22:57Z) set its conditions: in-box ratios only, peak memory, the GPU busy fraction, and the quality
bar with a mutant. The code under test:
- grouped-nf4-gemm #522: `gemm_int4_b32_smallm(..., block_m=)`, a 32- or 64-row tile above 16 rows;
- e4b #1410: `E4B_ATTN_INT4_WIDE`, opt-in, which routes 17–64 rows to it through a workspace shared per stream.

The box is new (`bench/p124/p124_box.py`). It imports P119's decode bracket and P117's teacher-forced pass at their
registered bytes; runner and driver are derived from P122's.

## Why this lane

- **P119** read SC2e's 64-row decode step at 15.64 ms of device time. **2.05 ms** of it is `Int4Linear` above 16 rows,
  serving a cached bf16 copy of each attention projection with cuBLAS. On Qwen3-30B-A3B (fused q/k/v N 5,120 and o
  N 2,048, 48 layers) that copy is 1.81 GB a step: a 1.01 ms floor at 1.79 TB/s, read at 49 % of it. The int4 grid it
  is dequantised from is 510 MB (a 0.28 ms floor).
- **K16** already serves 2–16 rows from those int4 bytes (in-register dequant, bf16 MMA over 128-wide K chunks, a
  4-way split-K reduced in the launch), the default since its P5 read. #522 lets the same kernel take a 32- or 64-row
  tile, bit for bit unchanged at 16 rows or fewer. #1410 routes 17–64 rows to it under `E4B_ATTN_INT4_WIDE=1`.
- **The arithmetic changes.** The order of the bf16 products and their fp32 sums differs from cuBLAS's on the bf16
  copy, so tokens may part and token identity is no gate. P117's teacher-forced instrument and P110's bar decide
  quality.

**The question:** on SC2e's served stack, are the captured 64- and 32-row decode steps faster with the route on than
off, by more than the box's own noise, and is the teacher-forced NLL with the route on within P110's bar of the route
off?

## Instrument

**The model is SC2e's served stack**, built as P117, P119 and P120 built it:
- `PagedServeConfig.from_env()` + `build_engine` with `bench/sc1/sc1_run.sh`'s `SPEEDENV`, `ROUTEENV` and
  `E4B_PAGED_FUSE_QKV=1`, byte for byte, plus `E4B_ATTN_INT4_WIDE=1` (so the enable path's capability check runs on
  the card);
- built eager with one slot of 768 tokens and no prefill graph: the arms build their own pools;
- the NF4 arena baked on the box by P39's `k8_bake.py`;
- every other knob at its default, including `E4B_INT4_WIDE_TILES` (the 64-row step's tile table).

**The setting** is every `Int4Linear`'s `_wide` flag, which is what `E4B_ATTN_INT4_WIDE` sets at enable. Each arm sets
it before its captures, profile or pass (`Route` in the box). `Route` also counts each projection call that reaches
Python by route and row count: `gemv`, `k16` (2–16 rows), `wide` (17–64) or `bf16` (the cached copy). Eager steps,
prefills and graph captures reach Python; replays do not.

**The windows:** 64 windows of wikitext-2-raw test, 512 prompt tokens and 128 positions, as P117 takes them.

**The arms** (`bench/p124/p124_box.py`; each a fresh `Fp8PagedKV` and `PagedModelRunner`, device grouping on, bulk KV
bookkeeping):

| arm | route | what runs |
|---|---|---|
| profile `off64`, `on64`, `off32`, `on32` | off, on | P119's `decode_bracket` at 64 and at 32 rows on buckets 1–64: 3 warm, then 8 padded eager steps (`capture=False`) under `torch.profiler`. The premise and the mechanism |
| served `OFF_a`, `ON_a`, `ON_b`, `OFF_b` (ABBA) at 64 rows, then at 32 | off, on, on, off | every bucket 1–64 captured under the arm's setting, the windows prefilled, 5 warm decode steps, **256 timed steps** (each `run_decode`'s synchronised wall, P120's method), then 32 more with the runner's `StepTrace` attached. Every decode step's tokens and the arm's peak memory are recorded |
| quality `R`, `rep` | off | P117's `paged_pass` over the 64 windows on buckets 1–64: one 64-row piece a step, today's served arithmetic |
| quality `half`, `chunk` | off | the floor: buckets 1–32 (two 32-row pieces a step); prefill in 256-token pieces |
| quality `ON64`, `ON32` | on | **the subjects**: buckets 1–64 (one 64-row piece, the 64-row tile); buckets 1–32 (two 32-row pieces, the 32-row tile) |
| quality `mutant_scale` | on | `ON64` with P108's halved decode softmax scale: the bar must catch it |
| quality `mutant_wide` | on | `ON64` with the route reading each 32-block's scale from the next block: the bar must catch it, so the route is the one scored |
| quality `G64on` | on | `ON64` with the graphs **captured**: its emitted tokens must equal `ON64`'s at every position (FUNCTION) |

A served step's graph pads its rows to the bucket, so the route sees exactly 32 or 64 rows: the two depths are the two
tiles a served step can take.

**Derived numbers:**
- **The premise and the mechanism.** P119's frozen class map (`CLASSES`) puts cuBLAS in `dense_gemm` and K16 in
  `dense_int4`. The head's and the router's GEMMs are the same in both settings, so OFF's `dense_gemm` minus ON's is
  the cuBLAS time the route replaces. ON's `_gemm_int4_b32_smallm` time is what replaces it.
- **The busy fraction** of a traced step is the replay's device time (`dec_issue − dec_prep`, the runner's own CUDA
  events) over the step's wall. It says whether the step is device-bound, so whether a byte argument prices it.
- **Peak memory:** allocated and reserved per served arm, and the route's shared workspace
  (`int4_attn.wide_workspace_bytes()`).

## The rule (`bench/p124/p124_reduce.py`, self-tested on 38 cases)

Every speed bound is a ratio inside the box. The first that fires is the verdict:

1. **NO_READING:** no box record, or the premise tests failed.
2. **VOID**, any of:
   - another e4b or grouped-nf4-gemm commit, or another model revision;
   - on the reading, a stack other than SC2e's; the engine not built with `E4B_ATTN_INT4_WIDE=1`;
   - rows outside 33–64 (the 64-row step must be one bucket-64 piece);
   - a profile arm other than 8 eager steps of its bucket under its setting;
   - a served arm missing, a bucket not captured, or its bucket not replayed exactly 293 times (5 + 256 + 32) with no
     eager step and no other bucket;
   - KV bookkeeping not bulk; a step, token or traced-step count short; no peak memory;
   - a quality arm missing, short of windows, or off its registered buckets, grouping, decode attention calls
     ((128 − 1) × layers × pieces; G64on 0) or graph status;
   - **not engaged**, any of:
     - with `1`, a capture at 32 or 64 rows took the cached copy, or a projection missed its wide call;
     - with `0`, any wide call;
     - a profiled step without one `_gemm_int4_b32_smallm` launch per projection (`1`) or with any (`0`);
     - the route removing fewer cuBLAS launches per step than there are projections;
     - a quality pass's decode calls off its registered route by bucket;
     - ON64 scored bit-equal to R in every window;
   - **nondeterministic**: at either depth, a setting's two served arms emit different tokens;
   - **FUNCTION**: `G64on`'s tokens differ from `ON64`'s at any position;
   - **a mutant survived**: `mutant_scale` or `mutant_wide` passes the bar.
3. **PREMISE_ABSENT** (the reading only): the cuBLAS time the route replaces is under **5 %** of the 64-row eager step's
   device time.
4. **QUALITY_FAIL**: `ON64` or `ON32` fails P110's bar against the floor.
   - d_X(w) is X's mean continuation NLL minus R's. The floor draws are `half` and `chunk`, plus `rep` if R did not
     repeat bit for bit. B_floor is the largest |mean d_f|, S_floor the largest mean |d_f|.
   - `passes(X)`: mean d_X ≤ B_floor + 0.01 nats, and mean |d_X| ≤ 2 × max(S_floor, 0.005).
5. **NOISY**: at either depth, `OFF_b/OFF_a` or `ON_b/ON_a` (median step) is more than **1.5 %** from 1.
6. **DEFAULT_ON**: at both depths, `ON_a/OFF_a` and `ON_b/OFF_b` are at most **0.98**. **DEFAULT_ON_64** or
   **DEFAULT_ON_32**: at that depth only.
7. **SLOWER**: all four ratios > 1. **NO_GAIN**: otherwise.

**Reported, never gated:**
- every quality arm's bias, spread, SE, max |d|, mean KL against R and argmax agreement; R's repeatability; the
  function counts;
- the profile tables;
- every served arm's busy fraction, replay device time and peak memory, and ON − OFF memory;
- ON-against-OFF token agreement at both depths.

Medians are taken by sorting and floats summed with `math.fsum`, so the verdict file is byte-identical on any Python.

## Predictions (written before any data; evaluated by the reducer, none gates the verdict)

| # | prediction | if it holds | if it misses |
|---|---|---|---|
| Q1 | the replaced cuBLAS share of the 64-row eager step in [0.08, 0.18] (P119: 0.131) | the lever is as P119 read it | low: the premise may fail. High: more to gain |
| Q2 | ON's K16 time / the replaced cuBLAS time at 64 rows in [0.25, 0.65]. Basis: a quarter of the bytes; K16 is not at the byte floor | the tile reads the int4 bytes near K16's own efficiency | high: the 64-row tile is the limit (registers, split-K); a plan census is next. Low: more headroom than expected |
| Q3 | both 64-row served ratios in [0.89, 0.97]. Basis: about (1 − Q2's 0.45) × 2.05 ms saved on a ~16.8 ms served step | the saving is the replaced GEMMs, net of the tile | high: the graph hides less cuBLAS time than the eager twin shows (Q1 against the served delta says so). Low: launch effects beyond the GEMMs |
| Q4 | both 32-row served ratios in [0.86, 0.97]. Basis: the same bytes over P119's 10.99 ms 32-row step | as Q3, at 32 rows | as Q3 |
| Q5 | every served arm's busy fraction ≥ 0.90 | the steps are device-bound, so bytes price them | below: host work bounds the step and a byte argument overstates the gain; RESULTS says which arms |
| Q6 | every same-setting pair agrees within 0.5 % | the 1.5 % noise bound is three times the noise | the bar sits near the noise; longer arms next time |
| Q7 | ON64's and ON32's bias each in [−0.003, +0.003] nats | the route's bf16 order is as neutral as P117's wide buckets | the bar decides; a miss that still passes says the floor carries it |
| Q8 | both mutants' bias ≥ +0.3 nats | the gate can fail, and the route is the one scored | the instrument has lost sensitivity; RESULTS says so before any claim |
| Q9 | ON − OFF peak allocated memory at 64 rows within ±64 MiB | the shared workspace (about 7 MB a stream) is the route's only memory | a larger cost is named before any default |

## Consequence, registered now

- **DEFAULT_ON**, **DEFAULT_ON_64** or **DEFAULT_ON_32**: a separate e4b PR gives `E4B_ATTN_INT4_WIDE` an `auto`
  default. It takes the route for the licensed depths only (17–64 rows, 33–64, or 17–32), when the K16 route is on and
  the installed grouped-nf4-gemm takes `block_m=`, detected by capability. `0` restores the cached copy. The serve
  estimate prices the shared workspace, and the read PR adds a claims row.
- **QUALITY_FAIL:** the route stays opt-in. RESULTS names the subject, its bias and spread, and the bound it missed.
- **NO_GAIN or SLOWER:** the route stays opt-in. RESULTS names from Q2 and Q5 whether the tile or the host explains
  it, and whether a plan census (`block_n`, split-K) is worth registering.
- **NOISY:** no default; a rerun with longer arms is an amendment.
- **PREMISE_ABSENT:** no default; RESULTS names what replaced the cuBLAS time on this stack.
- **VOID or NO_READING:** no consequence. A rerun is a new attempt, or an amendment if the cause is the harness.

## The proving rental

No local card runs the fp8 paged KV or the bucketed path. `tests/test_p124_box.py` covers the box on CPU with the
runner, the pool, the profiler, P117's pass and grouped-nf4-gemm's kernels stood in, on real `Int4Linear` modules:
- each arm sets the route before it runs, and the modules are restored;
- the route counts are the ones the reducer registers;
- `mutant_wide` moves only the wide calls;
- the served arms replay one piece per step and record every token, the traced steps and the memory;
- the reducer reads the record.

**The proof** runs the whole box end to end on `ibm-granite/granite-3.1-3b-a800m-instruct` @
`a02780686e08a03fe0d2679a293b5c74a90efa89`. That is its NF4 expert store with int4 RTN attention
(`E4B_SERVE_ATTN_INT4=1`), so the route under test exists there, at 40 rows (still one bucket-64 piece) and 32
positions. It covers:
- the refusals, the install and the tripwire;
- the reducer self-test and the premise;
- fetch, bake, every arm and the reducer.

It is **PROVED** iff the lane exits 0 with a verdict other than VOID or NO_READING. The premise share does not apply to
the proof's model; its numbers are not a reading.

## Budget and STOP rules

- **Proof:** one RTX 5090 (Vast verified/secure), guard 0.75 h at ≤ $0.75/h, estimate $0.5625.
- **Reading:** one RTX 5090, guard 2.0 h at ≤ $0.75/h, estimate $1.50. Expected 50–75 minutes: the 61 GB fetch,
  the bake, the int4 repack at load (about 8 minutes), eight served arms and four profiles (about 10), nine quality
  passes (about 12).
- **Lane ceiling:** $3.00, inside the owner's $15 no-ask tier.
- **STOP-1:** the refusals run before any install: dud box (10), card class (15), disk < 150 GB (13), host RAM < 60 GiB
  (16). The premise runs before the fetch (25), on the card, each none skipped:
  - `tests/test_decode_graph_buckets.py`, 7 passed;
  - grouped-nf4-gemm's `kernel/test_int4_smallm_interp.py` at the pinned commit (staged as a byte copy), compiled with
    `TRITON_INTERPRET=0`, 25 passed;
  - `tests/test_int4_attn_wide.py`, 10 passed. Its two CUDA tests capture two projections of one width in one graph,
    sharing the route's workspace, and replay them to their eager bits.
- **STOP-2:** every time-left check is a per-mode variable that fits its own guard (enforced by
  `tests/test_p124_staged_pin.py`).
- **STOP-3:** a VOID is not retried inside the same launch.
- **STOP-4:** the driver refuses a dirty tree or a staged file that differs from `bench/p124/staged.sha256`.

## What this lane cannot say

- **Nothing about serving attainment.** No server or driver runs here. The step ratios are inputs to `serve_capacity`,
  not a ceiling.
- **Nothing about other models, cards, prompt lengths or row counts.** Only SC2e's Qwen3-30B-A3B int4 stack at 32 and
  64 rows is read. Eager steps of 17–31 or 33–63 rows (graphs off) take the same tiles padded, and are not timed.
- **Nothing about prefill.** Prefill chunks are above 64 rows and keep the cached copy.
- **Nothing about the tile's plan.** K16's plan (`block_n` 64, `KC` 128, split-K 4) is read as shipped; no other plan
  runs.
- **Nothing absolute across boxes.** Only ratios inside this box are read.

## Receipts

Fetched to the run directory's `p124/` and committed to `bench/p124/receipts/<run>/`:
- `box.json` (every arm's record), `verdict.json`, `summary.txt`, `forensics.txt`, `versions.txt`, `bake.json`;
- the eight traced-step files (`trace_d64_*.jsonl`, `trace_d32_*.jsonl`);
- the logs (force-added past the `*.log` ignore, including the three premise logs) and the teardown proof;
- `SHA256SUMS`.

`RESULTS-p124.md` is written from those files.
