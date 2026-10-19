# P153: D2's served read, grouped-nf4-gemm K36's int8 lm_head on Qwen3-30B-A3B (e4b#1313)

**Status:** registered 2026-10-19, before any data. Lane number reserved on the bus and acknowledged by the maintainer,
with the knob's design (b) approved: the head rounded in place to its int8 grid. One non-consumer review, then the
maintainer's final pass. $0 until the read.

## Why

- **The premise.** [P148](../p148/RESULTS-p148.md) read D2 GO in a CUDA-graph microbench on an RTX 5090. D2 is an int8
  copy of Qwen3-30B-A3B's lm_head (151936 x 2048, untied), rounded symmetrically per row with an fp32 scale, which halves
  the head's bytes. P148's prototype GEMV over it ran against cuBLAS on the bf16 head. Projected onto P142's in-situ head
  time, it saves **0.189546 ms** a step. That is a microbench premise, never a gain.
- **Quality first.** [P150](../p150/RESULTS-p150.md) ruled the grid GO at both of P125's gates, on held-out windows:
  - t1: upper bound 0.001225 against 0.005205;
  - k16: 0.001130 against 0.005659.
- **The kernel.** grouped-nf4-gemm K36 (`929a6906`) ships P148's kernel and quantizer verbatim as
  `dense_gemv.gemv_i8_bw` and `dense_gemv.quantize_rows_int8`. Its compiled contract on an RTX A2000 read 1.0059 x
  cuBLAS's error on the bf16 head at worst.
- **The knob.** e4b#1820 adds `E4B_INT8_HEAD_BW=1`. It rounds the head's bf16 weight IN PLACE to `bf16(q * s)`,
  exactly P150's I8HEAD round trip (a test checks it bitwise, with the quantizer run from P148's pinned bytes). One-row
  calls go to `gemv_i8_bw` over the same codes and scales. Every other call goes to `F.linear` on the rounded weight.
  - **Knob-on therefore changes multi-row outputs too.** Every row count computes the head P150 ruled on, so the
    served model never depends on the batch size. P150's GO at both gates covers that change.
  - The int8 codes and scales add about 0.31 GB after the server sized its slots; at this read's 16 fixed slots that is
    inside the card's headroom.
- **The A2000 smoke** (`$0`, correctness only): the real kernel and quantizer at `929a6906` and the knob's bytes, at
  Qwen3's head shape (untied). It read OK:
  - **The grid is P150's, bit for bit, on the device P150 used.** The knob's head equals `roundtrip_` applied on the GPU,
    as P150's box ran it (the engine built all-VRAM, then the round trip). A CPU round trip of the same weight differs in
    5,698 of 151,936 rows: CUDA's true division by a scalar multiplies by the reciprocal, so a row's scale can differ
    from the CPU's in its last bit. P150 ruled the GPU grid, and the served knob computes it.
  - Bucket 1's graph replay launches `_gemv_i8_kernel` only. Bucket 4's launches cuBLAS and no `_gemv_i8_kernel`.
  - Engagement: routed at one row, passed at four, never passed at one row (a non-contiguous one-row input included).
  - Values: bucket 1 within one bf16 output rounding of fp64 `(q * s) @ x`; bucket 4 on the rounded weight, unlike cuBLAS
    on the original.
- **The denominator** is the **unprofiled** one-request step: **3.913 ms**, from `pername-5090-1`'s own unprofiled
  speed arms. P142's profiled 4.380 ms is context only. The projection is **4.84 %** of 3.913 ms, so g1 would be about
  **1.051** if every microsecond the microbench saw survived serving.
  - Context, not a denominator: with the split cap at its qwen3_moe default, P143 A3's capped arms read 3.745 ms a step
    (at a 450 W limit), where 0.189546 ms would be 5.06 %.
- **What P151 adds** (context, not a term). P151, D1's served read of K35's dense GEMV, is P153's sibling: the same
  form, the same floor, a microbench premise of the same kind. It closed at NOISY (its first process ran slow), but its
  three clean pairs showed about +2.45 % W1, about **0.73** of P148's D1 projection. At that share D2's 4.84 % would read
  about 3.5 %, g1 about **1.036**: over the bar, but not by much.

## What runs (`p153_run.sh`)

P151's served read after its Amendment 1, with the head knob in place of the attention projections'.
- **Engine and arms:** the shipped default server: graphs on, `max_seqs` 16, 4096 tokens per sequence, the fusions and
  the bandwidth GEMV at their auto defaults, the attention split cap at its qwen3_moe default (16), and the dense
  attention GEMV off. The arms differ only in the knob: **S0** = `E4B_INT8_HEAD_BW=0` (named), **S1** = `1`. The box
  refuses an arm with the split cap, the dense attention GEMV, a fusion knob or a decode GEMV knob set.
- **A warm-up process first** (P131 Amendment 2's process 0; the maintainer's ruling on P151's NOISY, written in before
  any data): `S0w`, an S0 build and speed pass in a fresh process before S0a.
  - It is integrity-checked like every arm: pins, card, census, cap, engagement, power floor, prompts, lengths and reps.
    A missing or failed warm-up is a VOID.
  - Its speed is **reported only** (`verdict_p153.json` `speed.warmup`). It never enters a pair, a self-pair, a rung's
    comparison or a bar.
  - Why: `p151-5090-4`'s first process ran 3.3 % slow on a 600 W host with no throttle, which put a self-pair at 1.0345
    and made that read NOISY. P131's first process ran 4.7 % fast. The sign is not known in advance, so the first process
    is kept out of scoring.
- **Speed:** eight arms, ABBA twice (S0a S1a S1b S0b S0c S1c S1d S0d), each a fresh process through P109's W16 and W1 at
  **five timed reps**.
- **Telemetry:** power limit, max SM clock, temperature, SM clock and power draw, read by nvidia-smi immediately before
  and after every timed block, never during one.
- **SANE:** P143 A2's, unchanged, as P151 runs it: Phase B's instrument on the default server built eager, one window
  per pass (bucket 1, T == 1), 12 wikitext windows x 128 positions, every pass KV at the server's 4096 tokens with the
  server's cap. S0 scores R, S1 scores ON. It reads the served int8 head against the bf16 default inside A2's band; it
  reads P150's ruling and does not re-rule it.
- **The premise, on this card, before anything is fetched:** P136's 19 premise tests plus K36's compiled contract,
  grouped-nf4-gemm's `kernel/test_dense_gemv_i8.py` at `929a6906` byte for byte (9 tests; staged as
  `premise/test_dense_gemv_i8.py`), which runs K36's 1.05 x cuBLAS bound on the 5090 for the first time. It must read
  **28 passed**, or the box refuses with rc 25.
- **Order:** install and tripwire; reducer, box and quality self-tests; premise; fetch; NF4 arena bake; prompts; eight
  arms; two SANE phases; reduce.
- **Pins:**
  - e4b: the clean checkout the driver ships, which must contain #1820's merge.
  - grouped-nf4-gemm: `929a6906` by sha (0.47.1 plus `dense_gemv`'s bf16 and int8 GEMVs). The tripwire also checks
    K36's frozen plan and the sha256 of its `quantize_rows_int8` source (the grid P150 ruled on).
  - The rest: torch 2.8.0+cu128, triton 3.4.0, transformers 5.17.0; Qwen/Qwen3-30B-A3B @ `ad44e777`.

## The rule (`p153_reduce.py`, fixed before data)

The first rung that applies decides. P151's checks are carried; A3's are verbatim.

1. **VOID.** Any of:
   - a record is missing or not ok, the warm-up process's included;
   - any of A3's checks fails on any record: pins, the 170-SM card, census and modes, the GEMV, the captured buckets,
     bucket 1 replayed, the SANE pass KV at 4096 tokens, one window per pass, 12 windows;
   - the split cap is not at its default in both arms, with its plans at capture and in SANE;
   - the dense attention GEMV is not off in every record;
   - **the head's engagement** fails:
     - **S1** must swap the one lm_head, at 151936 x 2048, and route one-row calls (`routed_modules_by_rows ==
       {"1": 1}`).
     - **No one-row call may fall back** to `F.linear` (`"1"` absent from `passed_modules_by_rows`). A row that is not
       bf16 or not on the card would leave the head on cuBLAS in bucket 1's graph; a non-contiguous row is copied, never
       passed. Calls of 2+ rows always pass, to `F.linear` on the rounded weight.
     - At the speed build those calls are bucket 1's capture and warm-ups; `graph_status` and bucket 1's replays bind
       them to W1. In SANE they are the eager T == 1 decode.
     - **S0** must swap nothing, at the build or in SANE.
     - A zero count is a VOID.
   - **power:** any timed block's power limit, read before or after it, below **575 W** or unreadable;
   - the arms differ in prompts or lengths, or an arm is not at five reps;
   - a void slope.
2. **NOISY**, a VOID and a **strike**: any W1 self-pair (S0b/S0a, S1b/S1a, S0d/S0c, S1d/S1c) outside **[0.985, 1.015]**.
   A NOISY read reads no flip. The lane has no strikes, so a NOISY would be strike one.
3. **FUNCTION_FAIL:**
   - an arm decodes different tokens from its own arm's first record (every S0 against S0a, every S1 against S1a), on
     any row, workload or length;
   - an arm's timed reps digest differently.

   S1 against S0 is never held to identity: S1's head is on the int8 grid at every row count.
4. **QUALITY_FAIL:** SANE in P143 A2's band: |mean NLL(ON) - NLL(R)| > 0.02 nats, or argmax agreement < 0.95.
5. **FLIP:** g1 = **min over all four** S1/S0 pairs at W1 >= **1.03**, **and** g16 (the same at W16) >= **0.99**.
   - A separate PR may then make the knob qwen3_moe's default, citing this read and P150.
   - That PR waits for a grouped-nf4-gemm release carrying K36 to be e4b's pin; the release is the maintainer's.
6. **NO_FLIP:** otherwise, with each failed bar named. The knob stays opt-in; a reading, not a failure of K36.

**The bars read the unrounded minima** (A3's correction, carried).

**Reported, never ruling:**
- every pair and self-pair ratio, including the cross-block self-pairs;
- the geomean over the four pairs;
- the SANE KL;
- W1 token identity of S1 against S0;
- W16's token identity (rows identical out of 16, the first divergent step per row);
- each arm's telemetry: its power limits, its max SM clocks, the hottest temperature and the slowest SM clock read beside
  its timed blocks;
- the warm-up process's rates, and each over S0a's.

## Predictions (graded, never ruling)

- every W1 pair in **[1.025, 1.055]**;
- g16 in **[0.99, 1.01]**: the head at 2+ rows is cuBLAS on a weight of the same size and shape, so only the W16 ramp's
  one-row steps can move;
- every W1 self-pair within [0.99, 1.01] at five reps;
- SANE |bias| <= 0.005 and argmax >= 0.98 (P150's I8HEAD read d +0.000747 and argmax 0.989 at t1);
- S1's W1 tokens differ from S0's within 160 tokens;
- verdict **FLIP**.

## Tests

`tests/test_p153.py` checks:
- **the staged pins:** every piece shared with P143 A3's kit (11) is at A3's pins, and the premise file is K36's (its
  sha256 pinned);
- **the runner:** the power floor before the install, the premise at 28, K36's quantizer bytes and plan in the
  tripwire, the warm-up process before eight arms at five reps, only the head knob set per arm (the dense attention GEMV never set), and NOISY and
  VOID not readings;
- **the self-tests**, run from one directory as on the box: the reducer's 52 cases include P143 A2's real first block,
  which reads NOISY, and a warm-up as slow as `p151-5090-4`'s first arm, which must not; the box has 23;
- **verbatim carries:** A3's checks and rungs in the reducer, A2's `ServedKV` in the box;
- **reducer mutants**, each killed by the self-test: A3's ten (adapted, as P151 carried them), the head's engagement
  unchecked at the speed build and in SANE, a one-row fallback accepted, a zero count accepted, the head's shape
  unchecked, the dense attention GEMV allowed on, an unreadable power limit accepted, the floor at 450 W, and an
  explicit cap of 16 taken for the default, the warm-up's absence accepted, and the warm-up read into a self-pair;
- **box mutants**, each killed by the self-test: the split cap allowed in an arm, the dense attention GEMV allowed in an
  arm, the arms' values swapped, and an unreadable telemetry field read as zero;
- the driver's dry run, its run-id refusal, and its spares reaching `-6` and no further.

## The box

- **Run ids:** `p153-5090-1`, with `-2` and `-3` as pre-create spares only, and `-4` to `-6` as spares for refusals
  BEFORE THE INSTALL only (a host floor, rc 17 or 15; a pre-create refusal; `no_such_ask`). A box that starts and fails
  is still a strike. The driver refuses any other id.
- **Box terms** are P151's (P143 A3's):
  - card and price: one RTX 5090 (cc 12.0, >= 30 GB, 170 SMs), torch 2.8.0+cu128, <= $0.80/h including 170 GB;
  - download: <= $0.005/GB for 70 GB;
  - guard: 1.25 h;
  - worst: **$1.35**;
  - fallback tier: <= $0.85/h, only behind the exhausted-loop marker; worst **$1.41**.
  - The warm-up process adds one arm, about 3 minutes, inside the guard: `p151-5090-4` took 31 minutes end to end.
- **Power:** the launch screens on recorded power caps: only hosts with a recorded cap of at least 575 W are walked.
  The box still refuses before the install below 575 W (rc 17; 15 if unreadable). The launcher's pin skips caution
  machines.
- **Driver** (`p153_drive.sh`): P151's after its Amendment 1, on `bench/common/lane_liveness.sh`
  (`tests/test_lane_liveness.py` runs its launch handshake).
- **Approval:** the standing rule. One non-consumer review, then the maintainer's final pass.

## What a reading starts

- **FLIP:** the default PR, after the grouped-nf4-gemm release carrying K36.
- **NO_FLIP:** the knob stays opt-in.
