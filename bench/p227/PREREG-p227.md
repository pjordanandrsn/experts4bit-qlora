# P227: the per-row bandwidth GEMV against K25 at 2 to 8 decode tokens, on two families (serving roadmap lever B)

**Registered before any reading.** The lane times one thing at three batch sizes. On the NF4 store, it compares
e4b's MoE expert call at 2, 4 and 8 decode tokens through two routes:
- the route that serves those steps today: the device tile table, then K25;
- the per-row route that `E4B_NF4_SMALLM_BW=1` (e4b#2009) opens: one group per (token, slot) row, so the rows go to
  grouped-nf4-gemm's K33 bandwidth GEMV.

Two families are read, and **each is ruled on its own**.

| the maintainer's condition (2026-10-22) | where |
|---|---|
| both families registered, each ruled on its own; a Qwen3 miss is a finding, not a lane failure | [The rule](#the-rule-p227_reducepy-fixed-before-data) |
| the bar: per-row <= 0.7x K25 at T = 2 and 4 | [The rule](#the-rule-p227_reducepy-fixed-before-data) |
| B2 (a multi-row GEMV) only where per-row/floor at T = 8 > 1.25, per family | [The rule](#the-rule-p227_reducepy-fixed-before-data) |
| one RTX 5090, a 575 W floor, a pinned host; the A2000 stays off lane work | [Budget](#budget-and-the-fit) |
| review, then the maintainer's pass; rent the reading only after this registration and e4b#2009 merge | [Sequencing](#sequencing) |

## Why

A decode step of 2 to 8 tokens costs much more than a step of one ("the W2..W8 cliff", SD2 V1: a 2-row step at 8.30 ms
against 4.18 ms, `bench/sd2/RESULTS-sd2.md`, on e4b 0.53.0). Part of that is the expert GEMMs:
- **At one token:** every (token, slot) row is its own group, and grouped-nf4-gemm serves the call through K33. K33 is
  a GEMV tuned for streaming bandwidth (`e4b.serve.p116.gemv-bw.qwen3.5090.2026-10-07`).
- **Above one token:** on the graph server, `_collapsed_grouping` sends the call to the device tile table and K25, a
  small-M grouped GEMM with 16-row tiles.
- **What lever B asks:** whether K33, one row per program, beats K25 at 2 to 8 tokens.

**The trade, from the routing tables alone** (pinned below; arithmetic, not a measurement):
- **What the per-row route gives up:** rows that share an expert each read its weights, so its bytes scale with ROWS.
- **What the tile route gives up:** it reads each distinct expert once, but pads every expert's tile to 16 rows.
- **How much the rows share:**

| family | T | rows | distinct experts per layer (table mean) | rows / distinct |
|---|---|---|---|---|
| Qwen3-30B-A3B (P153's stream S1) | 2 | 16 | 15.08 | 1.06 |
| | 4 | 32 | 26.45 | 1.21 |
| | 8 | 64 | 39.49 | 1.62 |
| Mixtral-8x7B (P222's seeded draw) | 2 | 4 | 3.53 | 1.13 |
| | 4 | 8 | 5.50 | 1.45 |
| | 8 | 16 | 7.23 | 2.21 |

So the per-row route's extra bytes grow with T, and fastest on Mixtral. Whether its kernel's efficiency pays for them
is the question. An unregistered shakedown box ran the instrument once to check its wiring. Its numbers shaped this
design and are not quoted.

**The question, per family:** at T = 2 and at T = 4, does the per-row route run the expert call in at most 0.70 of the
served route's time, on routing the lane has pinned in advance?

## The arms

Per family and T, three arms, each one graph holding every layer's call:

| arm | route | role |
|---|---|---|
| **K** | device grouping: the tile table, then K25 at the family's served decode and plan | **the control** |
| K2 | K captured again | the self-pair (the same cubins) |
| **B** | `E4B_NF4_SMALLM_BW=1`: singleton groups, K33, one row per program | **the registered question** |

- **K is each family's served default today:**
  - Qwen3: K39's decode (`prmt`) and plan `c1` (P158, P161). An unset `qwen3_moe` resolves to both.
  - Mixtral: the tree and today's plan, since no read licenses `mixtral`.
  - The bench sets the same values explicitly, and the tripwire asserts that the defaults resolve to them.
- **B is the knob, resolved by e4b's own `resolve_nf4_smallm_bw`** at the family's shapes on the box's device.
  - The resolver refuses wherever K33 would not engage, so B can never be a dot-pad or scalar route in disguise.
  - Before the reps, one eager pass per arm records the engagement:
    - B: exactly 2 x layers `bw_prmt32` launches, and no dot-pad or scalar launch;
    - K: no K33 launch at all;
    - e4b's route tally: B is `nf4_singleton`, K is `nf4_k25`, one per layer;
    - e4b's own K25 tallies (`K25_LUT_SEEN`, `K25_PLAN_SEEN`): K ran the family's registered decode and plan on every
      layer, and B reached K25 on none. The reducer compares them with the registered values, not with the record's.
- **The call:** the collapse's own, `hot_residency._fused_over_stack`, with the token rows (`x_tokens`). That is
  gate_up, the act and down, with the glue each route builds. The ratio is the expert call's, not a step's.

## The routing: pinned before data

- **Qwen3:** P153's router replay, `bench/p153/router-replay/rr.json.pt` (sha256 `3c59e72f...`, staged).
  - All 48 layers, stream S1 (rows 16-31), steps 0-59.
  - A cell of T tokens reads rows 16..16+T-1, which is T concurrent requests' real routing.
- **Mixtral:** P222's seeded draw (`p222_reduce.picks`, seed 2221, re-derived in `p227_reduce.mixtral_picks`).
  - 8 layers, steps 0-59.
  - A cell of T tokens reads rows 0..T-1.
  - No Mixtral router replay exists, so this is uniform top-2 of 8.
- **The pins:** each cell's table (int8 bytes, `[layer][step][row][k]`) and its distinct-expert total are pinned in
  `p227_reduce.ROUTE_SHA256` and `DISTINCT_TOTAL`.
  - Qwen3's were read from the replay at registration.
  - Mixtral's re-derive in the reducer's self-test.

## The instrument (`p227_bench.py`)

P222's, with the arms above. Everything not named here is P222's.
- **The box:** one RTX 5090 (sm_120), on torch 2.8.0+cu128 and triton 3.4.0 (the image's).
  - e4b at this registration's merge commit, which carries e4b#2009.
  - **grouped-nf4-gemm v0.50.0** at `41ef086e` (P222's pin), with the same three kernel files pinned by sha256.
- **The stores:** random NF4 bytes and fp32 absmax in the served layout, every layer distinct.
  - Qwen3: 48 layers of 128 experts (about 16.3 GB).
  - Mixtral: 8 layers of 8 experts (about 6.3 GB).
  - Each family's graph streams far more than L2 per replay.
- **The copy rate:** the median of 20 copies of 2 GiB, measured on the box before the reps. The floor divides by it.
- **The reps:** 60 per (family, T), one routing step each, 10 whole Williams squares of the six orders of three arms.
  - Each rep captures the spacer S (K's work, with its own outputs, never timed), then each arm in the rep's order.
  - It replays the arms interleaved, 10 timed replays each, every timed replay right after an untimed replay of S.
  - An NVML thread samples SM and memory clocks, power and the **enforced power limit** every 5 ms while a timed
    replay runs, tagged by family, T and arm.
  - After the replays: K2 against K bit for bit, and B against K as max |B - K| / max |K|.

## The rule (`p227_reduce.py`, fixed before data)

- **Per rep:** each arm's median of its 10 timed replays. Every ratio is read as the median over the 60 reps, with
  P155's larger-of SE (seeded bootstrap, seed 227).
  - Upper bound: median + 2 SE.
  - Clock-adjusted upper bound: the upper bound times B's median SM clock over K's, when B's is the faster.
  - A faster-clocked B is never credited.
- **VOID, the whole record:**
  - not an RTX 5090 at cc 12.0;
  - the arms, spacer or replays not as registered;
  - an unread copy rate;
  - a poisoned context or an arm that cannot be resolved, launched or captured;
  - unread clocks;
  - **any sample's enforced power limit under 575 W**.
- **VOID, one family:**
  - its layers, shapes, or K's decode and plan not the registered ones;
  - the knob not resolved on;
  - a cell's steps, routing hash or distinct-expert total not the pinned ones;
  - the engagement not as stated above;
  - a rep's order not its Williams row, or a rep without its 30 spacer replays;
  - **K2 differs from K by a bit in any rep**;
  - K/K2's median outside 0.99-1.01 at any T.
- **What a family's VOID means:** that family is not read on that run. The other family's verdict stands. A run where
  no family is read (PASS, MISS or TOLERANCE_FAIL) is not a reading (the runner's rc 27, a strike).
  - A later run re-reads only the families not yet read. A family already read keeps its first verdict, and a later
    run's verdict for it is reported, never ruling.
  - Two consecutive runs in which a family is not read stop the lane for that family.
- **TOLERANCE_FAIL, one family:** B against K beyond 1.5e-2 in any rep. That is a contract finding against the
  route. No timing is ruled on, and B2's trigger is not read.
  - It is FINAL for that family: it counts as read, the family is not re-read, and it is not a strike. It is
    deterministic, so a second box would only repeat it (review, adopted).
- **PASS, one family:** at **both** T = 2 and T = 4, B/K's upper bound **and** its clock-adjusted upper bound are
  **<= 0.70**.
- **MISS, one family:** otherwise. T = 8 never rules the bar.
- **B2's trigger, per family:** at T = 8, the median over reps of B's time over the byte floor **> 1.25**.
  - The floor is the step's distinct experts' bytes (NF4 plus fp32 absmax, both projections, all layers) over the
    copy rate. That is K33's floor.
  - The trigger is read only on a PASS or a MISS.
- **Reported, never ruling:**
  - B/K at T = 8;
  - B2's spread: the SE and bounds of B over the floor at T = 8 (the trigger rules on the median alone);
  - the copy rate's median, minimum and maximum over its 20 samples;
  - each arm's milliseconds;
  - K's and B's time over the floor at every T;
  - the distinct experts per layer;
  - the max B/K difference;
  - the clocks, with the label "clock-confounded" where B's median SM clock is off K's by more than 2 %, or its median
    memory clock by more than 1 %.

## Predictions (graded in the read, never ruling)

| prediction | basis | if HELD | if MISSED |
|---|---|---|---|
| **Mixtral: PASS** | 16-row tiles holding 2 rows an expert on average at T = 4 waste most of K25's MMA, while K33 streams each row once | the per-row route opens Mixtral's 2-8 token regime (B3's served read) | K25's tile is cheaper at these shapes than the padding suggests |
| **Qwen3: MISS** | K25 at `c1` is already a tuned small-M plan at Qwen3's shapes, and the per-row route re-reads 6-21 % more bytes at T = 2-4 | lever B is small on Qwen3; the rest of its 2-token cost is elsewhere | the per-row route's efficiency outweighs the re-reads |
| **B2 fires on Mixtral** | 2.21 rows per distinct expert at T = 8: the per-row route reads about twice the floor's bytes | a multi-row GEMV is worth building, Mixtral first | K33's re-reads hit L2 more than the table suggests |
| the self-pair within 0.005 of 1 | P222's | | a noisier card; the band carries it |

## The shakedown

`p227-shake-1` (spare `-2`, `-3`) runs this lane's kit end to end on the cheapest NVIDIA card at sm_86 or later with
15 GB.
- **It runs** `--shakedown --layers-qwen3 4 --layers-mixtral 2 --reps 6`, with no power floor and the runner's own fit
  (`NEED_BENCH` 600 s).
- **It forces K33 on** (`GNF4_GEMV_BW=1`, set by the runner after the premise). A shakedown card is not a 5090, and
  K33's own gate engages only at 160 SMs or more, so the resolver would refuse there. That is the shakedown's only
  knob. A reading that carries it is VOID.
- **It is explore tier:** it may run this PR's head before the merge. It is never a reading, and its times seed
  nothing.
- **It finishes 0 only on the reducer's SHAKEDOWN**, which runs every structural check.
- **Its record becomes the reducer's real fixture:** sha256-pinned under `bench/p227/fixtures/`, with a test that
  reduces it. That is added by an amendment before the reading, and the synthetic self-test cases stay.
- **An instrument fault it finds** is fixed by an amendment before the reading.

## Budget and the fit

- **One RTX 5090 on Vast**, rented directly (the owner's direct-API rule, 2026-10-22).
  - Per-second billing.
  - A host with a recorded power cap of at least 575 W, read again on the box (rc 17 below it).
  - **The pin and the expected and worst cost go on the bus before the create.** rent-watch holds the deadline from
    the create, and teardown is verified by rent-watch.
- **Worst case:** declared at most $1.05/h, **guard 1.25 h**, at most 10 GB of downloads at <= $0.04/GB:
  1.25 x $1.05 + 10 x $0.04 = **$1.71**, under the standing rule. Nothing is fetched from a hub.
- **The fit, with at least 25 % headroom.** `NEED_BENCH` comes from conservative arithmetic: about 3 ms of host time per
  layer per call, and each replay at three times the cell's byte floor.
  - **Qwen3, per rep:** 12 calls (4 graphs x 3: two warm-ups and the capture) x 48 layers x 3 ms = 1.73 s, plus 80
    replays at three times the T = 8 floor (about 9 ms) = 0.72 s. That is about 2.5 s a rep, and 180 reps = 450 s.
  - **Mixtral, per rep:** 12 x 8 x 3 ms = 0.29 s, plus 80 replays at about 11 ms = 0.88 s. That is about 1.2 s, and
    180 reps = 216 s.
  - **Everything else:** compiles 120 s; stores and the copy rate 35 s.
  - **Total:** about 820 s; x1.5 = 1,230 s, rounded up: `NEED_BENCH` = **1500 s**.
  - The install, the tripwire, the self-tests and the premise: `INSTALL_TO_GATE_S` = 600 s. The reserve: `RESERVE` =
    600 s.
  - **The runner's check:** `FIT` = 600 + 1500 + 600 = 2700 s, and `FIT_NEED` = 2700 x 1.25 = 3375 s. Short of that is a
    refusal, rc 41.
  - **The static fit:** (boot 300 s + 2700 s) x 1.25 = 3750 s <= 4500 s (1.25 h). `tests/test_p227.py` asserts each
    figure.
- **The premise, on the card:** exactly **25 passed**, else rc 25. That is:
  - e4b#2009's GPU contract, `tests/test_nf4_smallm_bw_gpu.py`, both lean-glue settings, 20: each row at T = 2..8 bitwise
    its T == 1 row, K33 only, graph replay;
  - the served T == 1 GEMV route's tests, `tests/test_gemv_bw_served_gpu.py`, 5.
- **Run ids:** `p227-5090-1` to `-3`.
  - Two consecutive HARNESS_ERROR, VOID or NOT_RUN readings stop the lane.
  - A power-floor refusal (rc 17) or a fit refusal (rc 41) is not a strike.
  - A shakedown is never a strike.

## Sequencing

1. This registration merges after one independent review and the maintainer's pass. It merges after e4b#2009, whose
   commit it needs, and after the 0.57.0 tag.
2. The shakedown may run before that, on this PR's head (explore tier).
3. The reading runs at the merge commit: the pin and the costs on the bus, the create, rent-watch, the drive
   (`p227_drive.sh`), the teardown.

## Receipts

- **On the box,** `/root/p227`: `bench_p227.json`, `verdict_p227.json`, `summary.txt`, `versions.txt`, `forensics.txt`,
  `gpu_phases.csv`, `logs/`.
- **Fetched** to the controller and committed to the private receipt store.
- **Public:** a copy without host identifiers goes under `bench/p227/receipts/<run>/` with `SHA256SUMS`, and the
  results PR cites the claim id.
- **Re-derivation:** the maintainer re-derives the verdict from the store with this reducer before any quote.

## What follows

- **A family's PASS** opens B3 there: a served read of `E4B_NF4_SMALLM_BW=1` at 2 and 4 requests. It has its own
  registration with a quality bar, because the tokens differ from K25's. Mixtral first if it passes.
- **B2's trigger firing on a family** opens B2: a multi-row K33 (an `M_ROWS` constexpr over device-grouped rows,
  bitwise per row against M = 1). A design note comes first, and the family that fired first.
- **A family's MISS** sizes lever B there. It is a finding, not a failure.
- **Whatever the verdicts,** the 2-8 token step cost outside the expert call is the serving roadmap's census to locate,
  not this lane's.

## What this lane cannot say

- **No step time:** the expert call only, with no attention, router, head or glue outside `_fused_over_stack`.
- **Mixtral's routing is uniform** top-2 of 8, not its router. Qwen3's is real (P153's S1), with 16 rows of one stream.
- **Scope:** one RTX 5090 at a 575 W limit, random NF4 bytes, triton 3.4.0.
- **No quality:** B is a different function from K within the tolerance contract, and only a served read with a
  quality bar can speak to tokens.

## Amendment 1 (2026-10-22, before any reading): the shakedowns, the real fixture, TOLERANCE_FAIL final

- **Review 1** (four items, all taken):
  - K's decode and plan are read from e4b's own K25 tallies against the registered values.
  - A shakedown forces K33's gate (`GNF4_GEMV_BW=1`) after the premise, and a reading carrying it is VOID.
  - A run with no family read finishes 27.
  - A missing or unreadable bench record reduces to VOID.
  - Reported, never ruling: B2's spread and the copy rate's spread.
- **TOLERANCE_FAIL is final for its family** (above, "TOLERANCE_FAIL"), adopted on review.
- **`p227-shake-1`** (an RTX A4000, about $0.02; teardown verified) ran the kit at `afa06dda` end to end: the premise 25
  passed, the bench, the reducer, `qwen3=SHAKEDOWN mixtral=SHAKEDOWN`. Its bench predates the K25 tallies, so its
  record is not the fixture.
- **`p227-shake-2`** (the same card class, about $0.01; teardown verified) ran the kit at the amended head with the
  same outcome.
  - Its bench record is the reducer's real fixture: `bench/p227/fixtures/p227-shake-2/bench_p227.json`, sha256
    `35b6a65e...`.
  - `tests/test_p227.py` reduces it to `qwen3=SHAKEDOWN mixtral=SHAKEDOWN`, with every structural check passed.
  - Five departures from it each VOID that family alone.
  - A mutant restoring the old self-reported K check accepts one of them.
- **Reported from the shakedowns (never ruling, no times quoted):** on 2- and 4-layer graphs on the A4000, one family's
  K/K2 self-pair fell outside 0.99-1.01 at T = 4 in each run (Qwen3 in -1, Mixtral in -2). A shakedown reports it. On the
  5090 the reading's graphs hold 48 and 8 layers. If the band VOIDs a family there, that family is re-read under the
  per-family rule above.
