# P151: D1's served read, grouped-nf4-gemm K35's dense GEMV on Qwen3-30B-A3B's attention projections at one row (e4b#1313)

**Status:** registered 2026-10-19, before any data. Lane number reserved on the bus and acknowledged by the maintainer,
with the design approved on five terms (the knob PR, the pin, the power floor, the bar, SANE and engagement). One
non-consumer review, then the maintainer's final pass. $0 until the read.

## Why

- **The premise.** [P148](../p148/RESULTS-p148.md) read D1 GO in a CUDA-graph microbench on an RTX 5090: a Triton
  single-row GEMV against cuBLAS on Qwen3-30B-A3B's fused q/k/v (5120 x 2048) and o (2048 x 4096). Projected onto P142's
  in-situ times it saves **0.131137 ms** a step (q/k/v 0.059671 + o 0.071466). That is a microbench premise, never a
  gain.
- **The kernel.** grouped-nf4-gemm K35 (`9b824155`, `dense_gemv.gemv_bf16_bw`) ships P148's kernel body verbatim
  under a tolerance contract against cuBLAS. It routes nothing by itself.
- **The knob.** e4b#1810 (`b2e2c72b`) adds `E4B_DENSE_GEMV_BW=1`. It swaps every fused `qkv_proj` and `o_proj` for a
  `DenseGemvLinear` that shares the weight. A single contiguous bf16 CUDA row goes to `gemv_bf16_bw`; every other call
  goes to `F.linear` unchanged. Under CUDA graphs, bucket 1's capture records the kernel and buckets 2-16 record cuBLAS.
  The knob is off by default, and an RTX A2000 smoke passed on the same bytes (correctness only).
- **The denominator** is the **unprofiled** one-request step: **3.913 ms**, from `pername-5090-1`'s own unprofiled speed
  arms (W1 decode 3.9134 and 3.9198 ms a step). P142's profiled 4.380 ms is context only. The projection is
  **3.35 %** of 3.913 ms, so g1 would be about **1.035** if every microsecond the microbench saw survived serving.
  - Context, not a denominator: since e4b#1792 the split cap is qwen3_moe's default, and P143 A3's capped arms read 267.0
    tok/s at a 450 W limit (3.745 ms a step), where 0.131 ms would be 3.50 %.
- **The edge.** 1.03 sits just under the full projection, so **NO_FLIP is at least as likely as FLIP**. A NO_FLIP here
  is a reading, not a failure of K35.

## What runs (`p151_run.sh`)

P143 Amendment 3's served read, the binding form the split cap flipped under, with the knob in place of the cap.
- **Engine and arms:** the shipped default server: graphs on, `max_seqs` 16, 4096 tokens per sequence, the fusions and
  the bandwidth GEMV at their auto defaults, and the attention split cap at its qwen3_moe default (16). The arms differ
  only in the knob: **S0** = `E4B_DENSE_GEMV_BW=0` (named), **S1** = `1`. The box refuses an arm with the split cap, a
  fusion knob or a decode GEMV knob set.
- **Speed:** eight arms, ABBA twice (S0a S1a S1b S0b S0c S1c S1d S0d). Each is a fresh process through P109's W16 and W1
  at **five timed reps**.
- **Telemetry:** the card's power limit, max SM clock, temperature, SM clock and power draw. nvidia-smi reads them
  immediately before and after every timed block, never during one (a sampler would share the host with the timed
  loop). The runner adds a forensics read at the start.
- **SANE:** P143 A2's, unchanged. Phase B's instrument runs on the default server built eager, one window per pass
  (bucket 1, T == 1), over 12 wikitext windows x 128 positions. Every pass KV is built at the server's 4096 tokens with
  the server's cap. S0 scores R, S1 scores ON.
- **The premise, on this card, before anything is fetched:** P136's 19 premise tests plus K35's compiled contract,
  grouped-nf4-gemm's `kernel/test_dense_gemv.py` at `9b824155` byte for byte (12 tests; staged as
  `premise/test_dense_gemv.py`). Among them is the 1.05 x cuBLAS error bound at both served shapes, which K35 checked on
  an RTX A2000 and which runs here on the 5090. The premise must read **31 passed**, or the box refuses with rc 25.
- **Order:** install and tripwire; reducer, box and quality self-tests; premise; fetch; NF4 arena bake; prompts; eight
  arms; two SANE phases; reduce.
- **Pins:**
  - e4b: the clean checkout the driver ships, which must contain `b2e2c72b`.
  - grouped-nf4-gemm: `9b824155` by sha (0.47.1 plus `dense_gemv`).
  - The rest: torch 2.8.0+cu128, triton 3.4.0, transformers 5.17.0; Qwen/Qwen3-30B-A3B @ `ad44e777`.
  - The tripwire also checks K35's frozen plan, the knob's targets and its default, and the cap's qwen3_moe default.

## The rule (`p151_reduce.py`, fixed before data)

The first rung that applies decides. A3's checks are carried verbatim.

1. **VOID.** Any of:
   - a record is missing or not ok;
   - any of A3's checks fails on any record: pins, the 170-SM card, census and modes, the GEMV, the captured buckets,
     bucket 1 replayed, the SANE pass KV at 4096 tokens, one window per pass, 12 windows;
   - the split cap is not at its default in both arms: cap 16 from `default-allowlisted`, with its plans at capture and
     in SANE;
   - **the knob's engagement** fails:
     - **S1** must swap all 96 modules (48 fused `qkv_proj` at 5120 x 2048 and 48 `o_proj` at 2048 x 4096). Every one
       must route a one-row call (`routed_modules_by_rows == {"1": 96}`).
     - **No one-row call may fall back** to `F.linear` (`"1"` absent from `passed_modules_by_rows`). A row that is not
       bf16, contiguous and on the card passes to cuBLAS, and in bucket 1's capture that leaves the module on cuBLAS for
       every W1 replay. Calls of 2+ rows always pass, by construction.
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

   S1 against S0 is never held to identity: the kernel accumulates in another order than cuBLAS.
4. **QUALITY_FAIL:** SANE in P143 A2's band: |mean NLL(ON) - NLL(R)| > 0.02 nats, or argmax agreement < 0.95.
5. **FLIP:** g1 = **min over all four** S1/S0 pairs at W1 >= **1.03**, **and** g16 (the same at W16) >= **0.99**.
   - A separate PR may then make the knob qwen3_moe's default (per family, as the cap's), citing this read.
   - That PR waits for a grouped-nf4-gemm release carrying K35 to be e4b's pin; the release is the maintainer's.
6. **NO_FLIP:** otherwise, with each failed bar named. The knob stays opt-in.

**The bars read the unrounded minima** (A3's correction, carried): the rounded values are for the report only.

**Reported, never ruling:**
- every pair and self-pair ratio, including the cross-block self-pairs;
- the geomean over the four pairs;
- the SANE KL;
- W1 token identity of S1 against S0;
- W16's token identity (rows identical out of 16, the first divergent step per row);
- each arm's telemetry: its power limits, its max SM clocks, the hottest temperature and the slowest SM clock read beside
  its timed blocks.

## Predictions (graded, never ruling)

- every W1 pair in **[1.015, 1.040]**;
- g16 in **[0.99, 1.01]**: decode at 2+ rows routes nothing, so only the W16 ramp's one-row steps can move;
- every W1 self-pair within [0.99, 1.01] at five reps (A3 read 0.9991 to 0.9995);
- SANE |bias| <= 0.002 and argmax >= 0.99;
- S1's W1 tokens differ from S0's within 160 tokens;
- verdict **NO_FLIP**.

## Tests

`tests/test_p151.py` checks:
- **the staged pins:** every piece shared with P143 A3's kit (11) is at A3's pins, and the premise file is K35's (its
  sha256 pinned);
- **the runner:** the power floor before the install, the premise at 31, eight arms at five reps, only the knob set per
  arm, and NOISY and VOID not readings;
- **the self-tests**, run from one directory as on the box: the reducer's 47 cases include P143 A2's real first block,
  which reads NOISY; the box has 22;
- **verbatim carries:** A3's checks and rungs in the reducer, A2's `ServedKV` in the box;
- **seventeen reducer mutants**, each killed by the self-test:
  - the ten A3 registered, adapted;
  - the knob's engagement unchecked at the speed build;
  - the same in SANE;
  - a one-row fallback to `F.linear` accepted;
  - a zero count accepted;
  - an unreadable power limit accepted;
  - the floor at 450 W;
  - an explicit cap of 16 taken for the default;
- **three box mutants**, each killed by the self-test:
  - the split cap allowed in an arm;
  - the arms' values swapped;
  - an unreadable telemetry field read as zero;
- the driver's dry run and its run-id refusal.

## The box

- **Run ids:** `p151-5090-1`, with `-2` and `-3` as pre-create spares only. The driver refuses any other.
  - **Amendment 1 (2026-10-19):** `-4`, `-5` and `-6` are spares for refusals BEFORE THE INSTALL only: a host floor
    (rc 17 or 15), a pre-create refusal, or `no_such_ask`. A box that starts and fails is still a strike.
- **Box terms** are P143 A3's (the same kit and box class):
  - card and price: one RTX 5090 (cc 12.0, >= 30 GB, 170 SMs), torch 2.8.0+cu128, <= $0.80/h including 170 GB;
  - download: <= $0.005/GB for 70 GB;
  - guard: 1.25 h;
  - worst: **$1.35**;
  - fallback tier: <= $0.85/h, only behind the exhausted-loop marker; worst **$1.41**;
  - A3's receipt puts the run at $0.479.
- **Power:** the box refuses before the install when the power limit is below **575 W** (rc 17), or unreadable (rc 15),
  as P149's does. One floor for both 5090 served flips keeps the records comparable. The launcher's pin skips caution
  machines.
- **Driver** (`p151_drive.sh`): A3's by counted edits, on `bench/common/lane_liveness.sh` (`tests/test_lane_liveness.py`
  runs its launch handshake).
- **Approval:** the standing rule. One non-consumer review, then the maintainer's final pass.

## What a reading starts

- **FLIP:** the default PR, after the grouped-nf4-gemm release.
- **NO_FLIP:** the knob stays opt-in.
- **Either way,** D2 (the int8 lm_head, P150 GO on quality) gets its own kernel lane (K36), its own knob and its own
  served read.
