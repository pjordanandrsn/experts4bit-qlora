# P229: K2b's microbench, the int8 lm_head at 2, 4 and 8 rows on K37's released `linear_i8` (lever K)

**Registered before any reading.** The lane reads, on one RTX 5090, whether the int8 lm_head's 2..8-row route
(`E4B_INT8_HEAD_ROWS`, e4b#2024) reads near the int8 bytes' cost. The route runs grouped-nf4-gemm K37's released
`dense_i8.linear_i8` at its shipped plan, over the head's own codes and scales. Its verdict licenses the knob's upper
bound. It never licenses a default: a served read with P153's quality bars comes first.

| the review's condition (2026-10-23, e4b#2022 and e4b#2024) | where |
|---|---|
| the bar at EVERY row count the knob can route (2, 4 and 8), and the knob capped at what is read | [The rule](#the-rule-p229_reducepy-fixed-before-data) |
| the bar read on the plan the shipped route runs, recorded per row count | [Global VOIDs](#the-rule-p229_reducepy-fixed-before-data) |
| another plan passing is a finding for a grouped-nf4-gemm `PLANS_SMALL` PR, not a GO | [What this lane cannot say](#what-this-lane-cannot-say) |
| the fallback, fixed before data: a miss is NO_GO for K2b, and K2a becomes the explicit next decision | [The rule](#the-rule-p229_reducepy-fixed-before-data) |

## Why

The design note (`bench/dense-small-m/DESIGN-K2.md`, e4b#2022) gives the case. On `qwen3_moe` the lm_head is served on
its per-row int8 grid. One row runs K36 over the int8 codes, and every other row count runs `F.linear` on the bf16
rounding of the grid, which reads twice the bytes: about 622 MB a step against 311 MB on Qwen3-30B-A3B's 151936 x 2048
head. K2b routes 2..N rows to K37's `linear_i8` over the same codes.

**The question, per M in 2, 4 and 8:** does the route (I) run in at most **0.70** of the served path's time (D,
`F.linear` on the grid) and within **1.25x** of the head's one-row time (K, K36: the same int8 bytes)?

## The setup

- **The head:** one random bf16 weight at (151936, 2048), seed 229, scaled 0.02. e4b's own resolvers swap it:
  `apply_int8_head_bw(model, "1")` (rounded in place to its grid; K36 at one row), then
  `apply_int8_head_rows(model, 8)` (K37 at 2..8 rows).
  - The head is 311 MB of int8 and 622 MB of bf16. Each replay streams past the 5090's 96 MB L2 without copies.
- **The packages:** e4b at the registration's merge commit; grouped-nf4-gemm at the **v0.50.0 release commit**
  `41ef086e6a3e2be5b158ff1bb38e0de67862a949` (K36 and K37 as shipped), installed first without dependencies.

| arm | what | role |
|---|---|---|
| **D** | `F.linear(x, head.weight)` on the grid's bf16 rounding at M rows | the served path today: **the control** |
| D2 | D captured again | the self-pair |
| **I** | `head(x)`: the installed route at M rows (K37) | **the registered question** |
| **K** | `head(x[:1])`: the installed route at one row (K36) | the bytes' reference |

- **The reps:** 60 per M, 15 whole Williams squares of four arms.
  - Each rep copies its step's input (seed 22900 + M) into the static buffer the graphs read.
  - It replays in its Williams order: 10 timed replays per arm, each right after an untimed replay of the spacer S
    (D's work).
  - NVML samples clocks, power and the enforced limit, tagged `M|arm`.
  - Then D2 is compared with D bit for bit, and I with D.
- **The engagement:** one eager pass per arm before the reps, counted in the head's own `routed` record. I must add
  exactly one call at M rows, K exactly one at one row, and D and D2 none.

## The rule (`p229_reduce.py`, fixed before data)

- **Per rep:** each arm's median of its 10 replays. Ratios are the median over 60 reps, with P155's larger-of SE
  (seed 229). The upper bound is median + 2 SE.
- **The clock adjustment:** each upper bound is scaled up by the numerator arm's median SM clock over the
  denominator's when the numerator's is faster. I is never credited.
- **VOID, the whole record:**
  - not a 5090 at cc 12.0;
  - the arms, spacer or replays not as registered;
  - grouped-nf4-gemm not at the registered release commit;
  - **the installed route not `{rows_max: 8, kernel: K37, plan: (32, 256, 4) at EVERY M in 2..8}`**: the bar reads
    the plan the shipped route runs (`PLAN_SMALL_DEFAULT`);
  - the head not (151936, 2048);
  - a poisoned context or an arm fault;
  - unread clocks;
  - **any enforced power limit under 575 W**;
  - at any M: the engagement not as stated, a rep's order not its Williams row, a rep without its spacer replays,
    **D2 differing from D by a bit**, or I against D unread;
  - D/D2's median outside 0.99-1.01 at any M.
- **TOLERANCE_FAIL:** I against D beyond **max |I - D| / max |D| <= 1.5e-2** in any rep. It is final: read, never
  re-read, not a strike, and no timing is ruled on.
- **PASS at one M:** I/D's upper bound **and** its clock-adjusted bound **<= 0.70**, and I/K's upper bound **and** its
  clock-adjusted bound **<= 1.25**. MISS otherwise.
- **The verdict:**
  - **GO_ROWS_8 / GO_ROWS_4 / GO_ROWS_2:** the largest M in (2, 4, 8) such that every read M up to it PASSes. That is
    the knob's licensed bound. A miss at 4 caps it at 2 even if 8 passes.
  - **NO_GO:** M = 2 misses. K2b is NO_GO at the shipped plan, and K2a (the design note's section 8) becomes the
    explicit next decision, never a silent switch.
  - A VOID record is not a reading (rc 27).
- **Reported, never ruling:**
  - D/K at every M;
  - each arm's per-call time;
  - the int8 and bf16 byte floors (bytes over the measured copy rate);
  - the clocks, with "clock-confounded" where two arms' SM clocks differ by more than 2 %.

## Predictions (graded in the read, never ruling)

| prediction | basis | if HELD | if MISSED |
|---|---|---|---|
| **GO_ROWS_8** | an explore run (400 W, never quoted) put I near the int8 floor at every M | the knob is licensed to 8 | the 575 W card or the clocks differ from the exploration |
| I/K about 1.03-1.04 | the same bytes; K37's 16-row MMA tile costs little at a byte stream | | the multi-row tile costs more at 575 W |
| the self-pair within 0.005 of 1 | P222's and P228's | | a noisier card |

## Budget and the fit

- **One RTX 5090 on Vast**, rented directly.
  - A host with a recorded power cap of at least 575 W, read again on the box (rc 17 below it).
  - The pin and the expected and worst cost go on the bus before the create.
  - rent-watch is armed at the create (`--until`), and teardown is verified.
- **Worst case:** at most $1.05/h, **guard 1.0 h**, at most 10 GB at <= $0.04/GB: 1.0 x $1.05 + 10 x $0.04 = **$1.45**.
- **The fit, with at least 25 % headroom.** Arithmetic, every term conservative:
  - **The reading:** 3 cells x 60 reps x 80 replays (40 timed, 40 spacer) of up to 1 ms = 14.4 s, plus five captures a
    cell and K37's and K36's Triton compiles (up to 60 s each). About 120 s with the copy rate and the engagement.
  - **Total:** about 120 s; x2, raised to the floor: `NEED_BENCH` = **600 s**. `INSTALL_TO_GATE_S` = **900 s** (two
    installs and the two-part premise), `RESERVE` = 600 s.
  - **The runner's check:** `FIT` = 900 + 600 + 600 = 2100 s, and `FIT_NEED` = 2100 x 1.25 = 2625 s. Short of that is
    rc 41.
  - **The static fit:** (boot 300 s + 2100 s) x 1.25 = 3000 s <= 3600 s (1.0 h).
- **The premise, on the 5090 itself,** from staged copies in the repo's layout (`t/`; the CPU suite reads `bench/`
  files beside `tests/`):
  - `tests/test_int8_head_rows_interp.py` with `TRITON_INTERPRET=0`: the REAL kernels compiled on the card, its four
    CUDA-graph replays included: exactly **11 passed**;
  - `tests/test_int8_head_bw.py`, the route's CPU suite, whose wrapper runs the same file interpreted: exactly **45 passed**;
  - else rc 25.
- **The tripwire:** e4b and grouped-nf4-gemm at their commits; `dense_i8.plan(151936, 2048, M)` is (32, 256, 4) at
  every M in 2..8; `MAX_ROWS` is 8.
- **Run ids:** `p229-5090-1` to `-3`, and `p229-shake-1` to `-3` for the shakedown.
  - The shakedown is explore tier, may run this PR's head on any card at sm_86 or later, and passes
    `--shakedown --reps 4`.
  - Its record becomes the reducer's real fixture by an amendment before the reading.
  - Two consecutive runs that are not read stop the lane. rc 17 and rc 41 are not strikes, and a shakedown never is.

## Sequencing

1. This registration merges after one review and the maintainer's pass, and after e4b#2022 (the design note),
   e4b#2024 (its code) and the 0.57.0 tag.
2. The shakedown may run first.
3. The reading runs at the merge commit.
4. On GO_ROWS_N, the served read is registered next: W2, W4 and W8 (up to N) ABBA on the default Qwen3 server, the
   knob against unset, P153's quality bars per bucket. It comes before any default.
5. On NO_GO, K2a is put to the maintainer as a decision of its own.

## What this lane cannot say

- **No step time:** the head alone.
- **No other plan:** the bar reads the shipped plan only. Another plan's speed at this shape (an exploration saw one
  about 4-5 % faster) is a finding for a grouped-nf4-gemm `PLANS_SMALL` PR and a release, read again on that plan.
- **Scope:** one RTX 5090 at 575 W, one release of grouped-nf4-gemm, a random bf16 head and random inputs.
- **No quality:** I differs from D within the tolerance, and the tokens are a served read's to judge.
- **No speculative decode:** the route is refused with `E4B_PAGED_SPEC` until a read covers the pair.
