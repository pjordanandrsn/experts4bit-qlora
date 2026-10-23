# P232: the served read of the int8 lm_head at 2, 4 and 8 rows (`E4B_INT8_HEAD_ROWS`, lever K, K2b)

**Registered before any reading.** One RTX 5090 reads, on Qwen3-30B-A3B's default server, whether turning
`E4B_INT8_HEAD_ROWS=8` on keeps served quality and gains speed at 2, 4 and 8 concurrent requests. Its verdict decides
whether a separate PR may make the knob `qwen3_moe`'s default, and up to which row count. No default changes here.

**What is already off by default: the bf16 head.** Since P153 (`e4b.serve.p153.int8-head-bw.qwen3.5090.2026-10-19`),
`E4B_INT8_HEAD_BW` is on by default for `qwen3_moe`. It rounds the 151936 x 2048 head in place to its per-row int8 grid,
so **every row count already reads the int8 grid**. One row runs K36's GEMV. At 2 to 16 rows the default today runs
`F.linear` on the rounded bf16 weight. `E4B_INT8_HEAD_ROWS` changes only the kernel at 2 to 8 rows, to K37's
`dense_i8.linear_i8` over the same codes and scales. So the ruling pair here is **K37 against `F.linear` on the
identical grid**, never "int8 against bf16". The bf16 head is read too, reported only (below).

| the review's condition (2026-10-23) | where |
|---|---|
| the bar on the pair a default flip needs: the knob against unset on the default server | [The rule](#the-rule-p232_reducepy-fixed-before-data) |
| what the bar is on, and its tolerance, fixed before data and justified from K37's numerics | [The quality bar](#the-quality-bar-and-why) |
| the bf16 comparison reported with P153's bars, never ruling | [Reported](#reported-never-ruling) |
| a named tripwire, well inside the PASS bar, that holds a default PR for a written look | [The rule](#the-rule-p232_reducepy-fixed-before-data), rung 6 |
| a bar at every row count the knob would route (2, 4, 8), a default capped at what passes | [The rule](#the-rule-p232_reducepy-fixed-before-data) |

## Why

- **P229 licensed the kernel** (`e4b.kernel.p229.int8-head-rows.qwen3.5090.2026-10-23`, GO_ROWS_8). In a captured
  microbench of the head call, K37 ran in a median 0.51 of `F.linear`'s time at 2, 4 and 8 rows. It matched `F.linear`
  within the 1.5e-2 tolerance in every rep (largest relative difference 0.0078).
- **What P229 cannot say:** step time, served speed, or tokens. The head is one call in a decode step, and K37's outputs
  differ from `F.linear`'s within bf16's output rounding (DESIGN-K2.md section 4), so served tokens can change at 2 to 8
  requests. A default needs this read first (P229's sequencing, item 5).

## The arms

Both arms are the shipped default server (`PagedServeConfig.from_env()` + `build_engine(cfg)`), `qwen3_moe` defaults
throughout (`E4B_INT8_HEAD_BW` on, the split cap, the fusions, the bandwidth GEMV). They differ only in:

| arm | `E4B_INT8_HEAD_ROWS` | what 2 to 8 head rows run |
|---|---|---|
| **R** | unset | `F.linear` on the rounded grid (today) |
| **K** | `8` | K37 `linear_i8` at its shipped plan, (32, 256, 4) at every M from 2 to 8 |

A third, **B16**, runs only in the quality phases: `E4B_INT8_HEAD_BW=0`, the original bf16 head (`F.linear` on bf16 at
every row count), for the reported comparison.

## What runs (`p232_run.sh`, a code PR before the reading)

- **Speed:** P153's form at W2, W4 and W8. Eight arms ABBA twice (R_a K_a K_b R_b R_c K_c K_d R_d), each a fresh process
  through P109's `run_pass` with 2, 4 or 8 prompt rows added at once, so every decode step runs bucket T. Five timed reps;
  the rate is P109's slope (T x 128 / (the min wall at 160 tokens - the min wall at 32)). One unscored warm-up process
  first (P131 Amendment 2's). Power and clocks are read by nvidia-smi before and after every timed block, never during.
- **Quality:** three phases on the default server built eager (Phase B's instrument, `p115_quality.measure_phase`, as
  P153's SANE), each one process, each at **group** T = 2, 4 and 8 in turn: T windows decoded together per pass, so
  every decode step routes T rows through the head.
  - **24 wikitext windows x 128 positions** (24 divides by 2, 4 and 8, so every pass holds exactly T windows), every
    pass KV at the server's 4096 tokens with the server's cap.
  - Phase R scores R and saves its log-probs; phase K scores the knob against them; phase B16 scores the bf16 head
    against R's too.
- **The premise, on the card, before anything is fetched:** P229's premise (the real K37 and K36 kernels compiled on the
  card, 11 passed; the route's CPU suite, 45 passed), or rc 25.
- **Pins:** e4b at this registration's merge; grouped-nf4-gemm v0.50.0 `41ef086e` (K36 and K37 as released);
  torch 2.8.0+cu128, triton 3.4.0, transformers 5.17.0; Qwen/Qwen3-30B-A3B @ `ad44e777`.
- **Order:** download probe and guard fit (bench/common); install and tripwire; self-tests; premise; fetch; NF4 arena
  bake; prompts; the warm-up process and eight speed arms; the three quality phases; reduce.

## The quality bar, and why

**The bar is P153's, unchanged, at each T on its own** (DESIGN-K2.md section 4): at T = 2, 4 and 8 separately,
**|mean NLL(K) - mean NLL(R)| <= 0.02 nats** and **argmax agreement >= 0.95**, teacher-forced over the 24 x 128
positions.

**Why these, and why not token identity:**
- **Identity is not expected.** K37 computes `bf16(s * acc)`, with `acc = x . q` accumulated in fp32 by `tl.dot`.
  `F.linear` computes cuBLAS on `bf16(q * s)`. Both sit on the same grid (the codes are exact in bf16, |q| <= 127), but
  they differ in accumulation order and in where the scale is applied. P229 measured the gap on a random head: at most
  0.0078 of the largest output in magnitude, inside K37's 1.5e-2 contract. A logit gap of that size can flip a greedy
  token wherever the top two logits nearly tie, so greedy identity is reported, not gated.
- **A logprob bar reads what the gap does.** A per-logit difference at the output's rounding moves each position's
  log-softmax by about the same amount, and averages toward zero over 3,072 positions. So the mean NLL gap should sit far
  inside 0.02 nats. Argmax agreement fails only at near-ties, which a 0.95 floor tolerates.
- **The standard is the one the one-row head met.** P153 read its I8HEAD head against bf16 at bias +0.0022, argmax
  0.989, KL 0.0006 under these bars. K2b is held to exactly that standard, never a looser one. The gap here is smaller
  in kind than P153's (a kernel change on one grid, not a grid change), so these bars are generous for it, and the
  predictions say how much tighter the read is expected to land.

## The rule (`p232_reduce.py`, fixed before data)

The first rung that applies decides.

1. **VOID.** Any of:
   - a record missing or not ok, the warm-up process's included;
   - pins, the 170-SM card, the census and modes, the captured buckets (2, 4 and 8 replayed in their workloads), the
     quality KV at 4096 tokens, 24 windows, group T as registered;
   - **the head's engagement:** both arms swap the one lm_head (151936 x 2048) onto its grid (`E4B_INT8_HEAD_BW` on).
     **K** routes rows 2, 4 and 8 to K37 at plan (32, 256, 4) (`routed_kernel_by_rows`), none passed. **R** passes rows
     2, 4 and 8 to `F.linear` and routes none of them. Both route one row to K36. In the quality phases the same holds
     for the eager decode at T. A zero count is a VOID;
   - **B16** swaps nothing;
   - any timed block's power limit below **575 W**, or unreadable;
   - the arms differ in prompts or lengths, or an arm is not at five reps; a void slope.
2. **NOISY**, a VOID and a strike: any self-pair (R_b/R_a, K_b/K_a, R_d/R_c, K_d/K_c) at W2, W4 or W8 outside
   **[0.985, 1.015]**.
3. **FUNCTION_FAIL:** an arm decodes different tokens from its own arm's first record (every R against R_a, every K
   against K_a), on any row, workload or length; or an arm's timed reps digest differently. K against R is never held to
   identity.
4. **Per T in (2, 4, 8): PASS** when quality holds at T (the bars above) **and** g_T, the minimum over the four K/R pairs
   at W_T, is **>= 1.01**. MISS otherwise, with the failed bar named. The bars read the unrounded minima.
5. **The verdict:**
   - **FLIP_ROWS_8 / FLIP_ROWS_4 / FLIP_ROWS_2:** the largest T such that every T up to it PASSes. A separate PR may
     then make `E4B_INT8_HEAD_ROWS` `qwen3_moe`'s default at that value, citing this read and P229. A MISS at 4 caps it
     at 2 even if 8 passes.
   - **NO_FLIP:** T = 2 misses. The knob stays opt-in, a reading, not a failure of K37.
6. **TRIPWIRE, reported and named, never changing the verdict.** The PASS bar sits about 20x above the predicted gap, so
   a K37 defect several times larger than expected would still pass it. The wire fires at any T if **|mean NLL(K) -
   mean NLL(R)| > 0.005 nats**, or **argmax agreement < 0.98**, or **the K-against-R KL above 3x the R self-pair
   floor** (the KL of phase R's `rep` arm against R, measure_phase's floor draw). If it fires, the verdict still reads as
   the rungs above say, but **no default-flip PR may open until a short written look explains it**, reviewed as usual.
   The reducer prints `TRIPWIRE` with every condition that fired, or `TRIPWIRE none`.

## Reported, never ruling

- every pair and self-pair ratio, and the geomean over the four pairs, at each W;
- **the bf16 comparison (B16):** at each T, K against B16 and R against B16 (mean NLL gap, argmax agreement, KL), with
  P153's bars printed beside them. It shows the whole stack's quality; it can neither block nor license the flip;
- the quality KL of K against R at each T;
- greedy token identity of K against R at W2, W4 and W8: rows identical out of T, and the first divergent step per row;
- per-row bucket stability (a row's output at bucket T against bucket 2, from saved hidden states), as DESIGN-K2.md
  section 4 asks: a property, not a contract;
- each arm's telemetry (power limits, max SM clocks, the hottest temperature, the slowest SM clock beside its blocks);
- the warm-up process's rates.

## Predictions (graded in the read, never ruling)

| prediction | basis |
|---|---|
| quality at every T: \|NLL gap\| <= 0.001 nats and argmax >= 0.99 | a kernel change on one grid, 0.0078 relative at most in P229 |
| g_T in [1.02, 1.05] at every T | P229's 0.18 ms saved per head call, against a decode step of about 4-5 ms at 2-8 rows |
| every self-pair within [0.99, 1.01] | P153's at five reps |
| at W2, most rows identical to R's at 32 tokens, and divergence by 160 | near-ties occur over long generations |
| verdict **FLIP_ROWS_8** | |

## What this lane cannot say

- **No speed outside W2, W4 and W8:** W1 (K36 either way) and W16 (`F.linear` either way) are not touched by the knob.
- **No other model or head shape:** Qwen3-30B-A3B's 151936 x 2048 head only; other `qwen3_moe` checkpoints of the same
  head shape inherit a default unread, as P153's did.
- **No other card or release:** one RTX 5090 at 575 W, grouped-nf4-gemm v0.50.0.
- **No speculative decode:** the route stays refused with `E4B_PAGED_SPEC` until a read covers the pair.

## Budget and the fit

- **One RTX 5090 on Vast,** rented directly through pod-launch: a host whose latest recorded power cap is 575 W or more
  (or none), read again on the box (rc 17 below it); at most **$1.05/h** with a 150 GB disk.
- **The work, every term conservative:** install 600 s; fetch 1,500 s (57 GB at the probe's 40 MB/s floor); bake 400 s;
  the warm-up and eight speed arms, about 120 s each, 1,080 s; three quality phases at three group sizes, about 700 s
  each, 2,100 s; reserve 600 s. **FIT = 6,280 s, x 1.25 = 7,850 s.**
- **Guard 2.5 h** ((boot 300 + 6,280) x 1.25 = 8,225 s <= 9,000 s). **Worst case:** 2.5 h x $1.05 + 70 GB x $0.04 =
  **$5.43** a started box. A normal box is expected near one hour, about $1.
- **Ids:** `p232-5090-1` to `-3`, and `p232-shake-1` to `-3` for the shakedown (explore tier, any sm_86 or later card,
  `--shakedown`, a reduced rep count). The shakedown's record becomes the reducer's real fixture by an amendment before
  the reading. Two consecutive runs not read stop the lane; rc 17, 19 and 41 are not strikes, and a shakedown never is.

## Sequencing

1. This registration merges after one review and the maintainer's pass.
2. The kit (`p232_run.sh`, `p232_box.py`, `p232_reduce.py`, `p232_drive.sh`, with their tests, a `$0` dry run and the
   reducer's mutants) is a code PR on its own review.
3. The shakedown may run on the kit's head.
4. The reading runs at the kit's merge, after the P228 reading.
5. On FLIP_ROWS_N, a separate PR may set the default to N for `qwen3_moe`, citing this read, unless the TRIPWIRE fired:
   then that PR waits for the written look. On NO_FLIP the knob stays opt-in, and the reading is recorded as such.
