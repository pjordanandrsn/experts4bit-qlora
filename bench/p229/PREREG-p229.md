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
- **Worst case:** at most $1.05/h, **guard 1.0 h**, and a 10 GB download allowance at <= $0.04/GB:
  1.0 x $1.05 + 10 x $0.04 = **$1.45**.
  - The 10 GB is the download (two pip installs and their wheels), not the disk. The create asks for a 40 GB disk, and
    the runner refuses under 20 GB free (rc 13).
- **The fit, with at least 25 % headroom.** Arithmetic, every term conservative:
  - **The reading:** 3 cells x 60 reps x 80 replays (40 timed, 40 spacer) of up to 1 ms = 14.4 s, plus K37's and
    K36's Triton compiles (up to 60 s each: 120 s), plus five captures a cell, the copy rate, the head's swap and the
    engagement (up to 45 s together).
  - **Total:** about 180 s (14.4 + 120 + 45, rounded up); x2 = 360 s, raised to the 600 s floor:
    `NEED_BENCH` = **600 s**. `INSTALL_TO_GATE_S` = **900 s** (two
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
4. On GO_ROWS_N with N below 8, a code PR first lowers the route's `MAX_ROWS` to N, refusing larger values by name.
   The knob never routes more rows than were read, and the cap never moves by default.
   - On GO_ROWS_4 or GO_ROWS_2 that PR sets `MAX_ROWS` to 4 or 2. On NO_GO it refuses every value above 1. The default
     stays 1 in every case.
   - **Why the unread row counts (3, 5, 6, 7) are licensed with their neighbours:** K37's 2..16-row kernel takes M as
     a runtime argument inside one `BLOCK_M = 16` tile. Every M from 2 to 16 is the same compiled kernel at the same
     plan (the route record shows the plan per M), so the head's bytes and the tile's work do not change between read
     row counts.
5. On GO_ROWS_N, the served read is registered next: W2, W4 and W8 (up to N) ABBA on the default Qwen3 server, the
   knob against unset, P153's quality bars per bucket. It comes before any default.
6. On NO_GO, K2a is put to the maintainer as a decision of its own.

## What this lane cannot say

- **No step time:** the head alone.
- **No other plan:** the bar reads the shipped plan only. Another plan's speed at this shape (an exploration saw one
  about 4-5 % faster) is a finding for a grouped-nf4-gemm `PLANS_SMALL` PR and a release, read again on that plan.
- **Scope:** one RTX 5090 at 575 W, one release of grouped-nf4-gemm, a random bf16 head and random inputs.
- **No quality:** I differs from D within the tolerance, and the tokens are a served read's to judge.
- **No speculative decode:** the route is refused with `E4B_PAGED_SPEC` until a read covers the pair.

## Amendment 1 (2026-10-23, before any reading): the shakedown and the real fixture

- **`p229-shake-1`:** an RTX A4000 (sm_86), explore tier, about $0.01, teardown verified. It ran this kit at the
  registration's head (a5525a83; every staged file is unchanged since) end to end:
  - the installs and the tripwire: grouped-nf4-gemm 0.50.0 at the release commit, and `dense_i8.plan` at (32, 256, 4)
    for every M in 2..8;
  - the premise on the card: **11 passed** (the real kernels compiled, the graph replays included) and **45 passed**;
  - the bench: the route record `{rows_max: 8, K37, (32, 256, 4) at every M}`, the engagement exactly as registered,
    D2 bitwise D in every rep;
  - the reducer: `SHAKEDOWN`.
  - No times are quoted.
- **The real fixture:** its bench record is pinned at `bench/p229/fixtures/p229-shake-1/bench_p229.json` (sha256
  `584b1b61...`; no host identifiers).
  - `tests/test_p229.py` reduces it to SHAKEDOWN with every structural check passed.
  - Five departures from it each VOID the record.
  - A mutant without the "the route reports the shipped plan" check accepts one of them.

## Amendment 2 (2026-10-23, before any reading): a comment in the runner, re-pinned

- The repository's guard-fit check (`tests/test_lane_guard_fit.py`) requires each runner's fit refusal to carry the
  shared strike note. `p229_run.sh`'s rc 41 comment now reads "a refusal before the install, not a strike; two
  consecutive headroom refusals count as one strike". That is a comment only: no command, exit code, threshold or
  output changed.
- `staged-p229.sha256` is re-pinned for `p229_run.sh`. So Amendment 1's "every staged file is unchanged since" holds
  for every staged file except this one comment line. The real fixture is unchanged, and p229-shake-1's record still
  reduces as Amendment 1 states.

## Amendment 3 (2026-10-23, before any reading): the drive follows pod-launch's contract

- A reading launches through pod-launch, which runs `p229_drive.sh` without `E4B_SHA` (it proves the checkout HEAD is the
  manifest's `heads.e4b`) and keeps its own `receipt.json` and `instance.json` in the run directory. The drive now
  derives the launch commit from the clean checkout HEAD, refuses an `E4B_SHA` override and a dirty tree (P226's form),
  and fetches the box tree into `$RUN_DIR/p229/`, refusing a non-empty target.
- No command on the box, threshold, cost or output changes. The drive is not a staged file, so no pin moves; the
  shakedowns ran through a direct launcher that set `E4B_SHA`, which is why they did not see it.

## Amendment 4 (2026-10-23, before any reading): retry run ids for boxes that did not start

- The registration names three reading ids, `p229-5090-1` to `-3`. Every box needs its own id and receipt (the ledger
  holds one row per run, so an id is never reused), but a box that does not start uses an id without being read.
  `p229-5090-1` refused itself on the box before the install (rc 17: the board's power limit read 525 W against the
  575 W floor; not a strike, under the registered rc 17 rule). `p229-5090-2` was refused at create (the provider still
  listed the offer; nothing was created, $0), recorded under the row below. So `-1` to `-3` can run out before a box
  has started.
- This amendment adds the retry ids **`p229-5090-r1` to `p229-5090-r8`**, for ANY box that follows one that did not
  start, at the same tier and terms. A box that did not start is one of:
  - a refusal at create (nothing created);
  - a pre-flight teardown (the launcher tore a created instance down before the runner ran; P225's NOT_RUN);
  - the box's own refusal before the install (rc 17 and the like).
- The runner's and the drive's id cases accept them. This PREREG states no allowance for boxes that do not start, so N
  is 8, P225's.
- **The ids are labels, not budget.**
  - Started-box accounting is by outcome, never by id: no code parses an id's number.
  - The registered caps, the strike rules and the stop govern alone. Nothing is widened.
- **A box STARTS when its runner passes the refusals before the install (rc 17, rc 41, and the host refusals rc 10,
  13, 15, 16, 18 and 28) and begins the install**, whatever its outcome after that (read, HARNESS_ERROR, VOID, a
  mid-install or mid-run fault). One definition serves the cap and the stop. **Only started boxes count toward the cap
  and toward any stop; a box that did not start (a refusal at create, a pre-flight teardown, rc 17, a single rc 41)
  counts toward none of them; the registered rc 41 pair rule is unchanged.**
- **At most three boxes start.** The retry ids never add a started box; they only replace ids used by boxes that did
  not start. This keeps the registered budget: 3 x $1.45.
- **What the stop counts (line 125's text did not settle rc 17, so this settles it):**
  - "Two consecutive runs that are not read stop the lane" (Budget) counts started boxes only. A started box that is
    not read counts, including one that faults mid-install, so alternating faults cannot dodge the stop.
  - An rc 17 box never counts toward the stop, whatever label the launcher's receipt carries.
  - A single rc 41 is not a strike; two consecutive rc 41 count as one strike under the cross-lane rule (line 174),
    unchanged.
  - A refusal at create and a pre-flight teardown count toward nothing.
  - `p229-5090-1` (rc 17) counts 0.
  - Decided pre-data (no reading exists), after `-1`'s rc 17. Why: rc 17 is the host's power floor and rc 41 the guard
    fit, both refusals before any install; the stop exists to end a lane whose kit cannot produce a reading, not to let
    capped hosts end it.
- **A new outcome row, decided from the launcher's receipt and never classified:**

  | outcome | when | a reading? | a strike? |
  |---|---|---|---|
  | **refused at create** (Amendment 4) | the provider refused the create, so nothing was created ($0) | no | no; counts toward no cap, uses one id |

- If boxes that did not start ever use up `r8`, that is a launcher or market fault to fix, not a reason to add ids.
- **The slack, checked (a test, no kit change):** the box's deadline runs from the create, so a slow-ssh host must not
  halt the lane at the fit check. guard - FIT_NEED = 3600 - 2625 = 975 s, at least pod-launch's longest ssh wait
  (420 s) plus a 120 s staging margin. `tests/test_p229.py` asserts it.
- Pre-data: no reading exists. The runner's bytes change (the id case only), so its staged pin moves. A box launched
  before this amendment merges uses `-1` to `-3` from the earlier commit; a box launched after it launches from its
  merge commit.
