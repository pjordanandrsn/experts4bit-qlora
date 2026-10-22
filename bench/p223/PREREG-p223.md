# P223: the price of a schedule-invariant decode for Mixtral-8x7B on `serve_paged`

**Status:** registered before any data. It is a price read of an opt-in mode. It changes no default and no gate, and it
quotes no speed claim beyond its own record. It needs one independent review, then the maintainer's pass. The kit follows
in its own PR.

## Why

P149's served decode is not schedule-invariant. A request's tokens depend on what else shares its decode steps, and so on
`max_new_tokens` (P149 RESULTS, "The schedule dependence"). Three diagnostics on one RTX 5090 place the cause on
P149's one-layer cut of Mixtral-8x7B:
- **The split-KV count** (`p149mnt-5090-6`, `-7`): the fp8 paged decode attention chooses its split count from the
  graph bucket's row count. With one split at every bucket (`E4B_ATTN_SPLIT_CAP=1`), buckets 2-16 match bit for bit.
- **Not the NF4 expert route** (`p149b1-5090-5`): sending a one-row step's experts through K25 or the M-tile GEMM leaves
  the single remaining bucket-1 difference unchanged, to the bit.
- **Removed by not running bucket 1** (`p149b1-5090-8`): with one split and the buckets `2,4,8,16`, so that a lone row
  decodes in bucket 2 beside one pad row, every one of the 496 W16 positions of a 32-token run matches the same
  position of its 160-token run, bit for bit. Engagement was verified: zero bucket-1 replays, and all 229 lone-row steps
  ran in bucket 2. No single op is named.

So two existing knobs give schedule-invariant served logits on that cut. Schedule invariance means that at a fixed
batch width, a request's tokens do not depend on which bucket each step replays, and so not on `max_new_tokens`. Batch
invariance across widths is a stronger property, and this lane reports it (the cross-W check below) but does not rule
on it. Batch-invariant kernels are a known remedy for this class of nondeterminism (Thinking Machines, "Defeating
Nondeterminism in LLM Inference", 2025). What is open here is the price on this server, on the full model.

**The question:** what do the two knobs cost in served decode throughput at W1 and W16, for the bf16 attention server
and for the int8 attention server, and are both servers in fact schedule-invariant on the full model?

## The arms

| arm | server | invariance knobs | role |
|---|---|---|---|
| **D0** | the default (bf16 attention projections) | none | control for I0 |
| **I0** | the default | `E4B_ATTN_SPLIT_CAP=1`, `E4B_PAGED_BUCKETS=2,4,8,16` | the schedule-invariant mode, bf16 |
| **D1** | `E4B_SERVE_ATTN_INT8=1` (P149's S1) | none | control for I1 |
| **I1** | `E4B_SERVE_ATTN_INT8=1` | `E4B_ATTN_SPLIT_CAP=1`, `E4B_PAGED_BUCKETS=2,4,8,16` | the schedule-invariant mode, int8 |

D0 and D1 run today's bucket set (`auto`, which is `1,2,4,8,16` at 16 slots), so each I/D pair prices the knobs against
what is served today, the pad row included. Every other lever is unset in every arm.

**Engagement, recorded per process and binding (VOID otherwise):**
- **I arms:** the attention split plan is 1 at every bucket, the runner's buckets are `(2, 4, 8, 16)`, there are zero
  bucket-1 replays, and every lone-row step lands in bucket 2 with one pad row.
- **D arms:** the default split plan, the buckets `(1, 2, 4, 8, 16)`, and bucket-1 replays present.
- **D1 and I1:** P149's int8 engagement counters are nonzero.

## The read (P149's box Q+S kit, by substitution)

The kit is P149's box Q+S kit (`p149_box.py`, `p149_run.sh`, `p149_reduce.py`), with these substitutions only:
- **The arms** are those above, each with its knobs in the process environment.
- **No quality term:** the knobs change the decode's arithmetic only inside the kernels' own tolerance. This lane reads
  an opt-in mode and never licenses a default.
- **No token-pinned blocks** (P149 Amendment 4's T1/T2) and **no SANE phases**: the invariance read below replaces them.
- **The process order** is two ABBA blocks, eight fresh processes: D0a, I0a, I1a, D1a, D1b, I1b, I0b, D0b. Heat is
  recorded per timed block, as in P149.
- **The workloads** are P109's W16 and W1: 512-token wikitext prompts, SHORT 32 and LONG 160 new tokens, and five timed
  reps after one untimed warm pass. tok/s = B · (LONG − SHORT) / (min wall_LONG − min wall_SHORT).

## The rule (`p223_reduce.py`, fixed before data)

1. **VOID:** not an RTX 5090 at cc 12.0; an enforced power limit under 575 W; any engagement check above failing; a
   record missing.
2. **NOISY** (a VOID and a strike): any W1 self-pair (D0b/D0a, I0b/I0a, D1b/D1a, I1b/I1a) outside [0.985, 1.015].
3. **The schedule-invariance read** (it rules the label): in every process of an I arm, each W16 row's 32-token stream
   equals the first 32 tokens of its own 160-token stream, and the same holds for W1. Each server is then
   **SCHEDULE_INVARIANT** or **NOT_SCHEDULE_INVARIANT**. The latter is a finding, with the differing rows and steps
   named. The D arms' same check is reported beside it; P149's S0a differed in 1 of 16 rows, and its S1a in 2.
4. **The price** (reported, never a bar): p = tok/s(I) / tok/s(D) over the adjacent pairs, I0a/D0a and I0b/D0b for S0
   and I1a/D1a and I1b/D1b for S1. For each server, p1 is the minimum of its two W1 pairs and p16 the minimum of its two
   W16 pairs, each with its pairs and clocks. The cost of the knobs at W1 is reported for S0 and S1 separately. The four
   W16 self-pairs (D0b/D0a, I0b/I0a, D1b/D1a, I1b/I1a) are reported beside p16.
5. **The cross-W check** (reported, never ruling): P109's W1 prompt is W16's row 0, so in every process the W1 160-token
   stream is compared with W16 row 0's 160-token stream, giving the first differing step or "equal". Under I, a lone row
   runs in bucket 2 and the same row at W16 runs in bucket 16, so the grouped expert GEMMs see a different M. This is the
   batch-invariance probe the schedule read does not rule on.

## Predictions (graded in the read, never ruling)

| prediction | basis |
|---|---|
| **W16, both servers:** p16 in 0.97-1.00 | bucket 16 already runs 2 splits; only the tail steps leave bucket 16 |
| **W1, both servers:** p1 in 0.60-0.95 | every W1 step runs in bucket 2 beside a pad row, with one attention split; the pad row may also route experts the live row does not |
| **Both servers SCHEDULE_INVARIANT:** 0 of 16 W16 rows differ, 32 vs 160 | `p149b1-5090-8` on one layer |
| **Cross-W under I: equal** (moderate confidence) | an inference: `p149b1-5090-8` compared 32 vs 160 tokens at the SAME width, where a row's bucket-2 steps matched the same row's bucket-16 steps; across widths the co-batched rows differ too |
| **Both D arms NOT invariant:** at least one W16 row differs, 32 vs 160 | P149's S0a (1 row) and S1a (2 rows) |

## What follows (never automatic)

- **Both servers SCHEDULE_INVARIANT:** I0 and I1 are the instrument for P149's owed routing-matched read. A server's
  tokens no longer depend on the schedule there, so forced streams can cross lengths and arms, and the two arms run the same
  bucket set. That read pins `E4B_PAGED_BUCKETS` in both of its arms, so `dense_i8`'s one-row GEMV branch is outside the
  served step on both sides. It is its own registration.
- **p1 under 0.90 on either server:** the follow-up prices a cheaper invariant decode for one request, for example a
  fixed split count chosen by KV length rather than by row count, or a pad row that routes no experts. That is code, so
  it needs its own PR and read.
- **NOT_SCHEDULE_INVARIANT on a server:** the named rows and steps are the next attribution, as the three diagnostics
  were. A cross-W difference under I is reported as the batch-invariance gap, with its first step.

## Budget and sequencing

- **Host:** one RTX 5090 on Vast, under the market rule, at most $1.05/h including the disk, the 575 W floor checked on
  the box, and the launcher's stale-ask rules.
- **Pins:** e4b at the kit's merge, with grouped-nf4-gemm at that commit's CI pin; torch 2.8.0+cu128, triton 3.4.0,
  transformers 5.17.0; Mixtral at `eba92302` (P141's lock).
- **The fit** (P149's runner terms, 25% headroom):

  | term | s |
  |---|---|
  | boot | 300 |
  | install | 450 |
  | fetch | 3,000 |
  | bake | 1,500 |
  | 8 arm processes × 480 | 3,840 |
  | reduce | 120 |
  | reserve | 600 |
  | **total** | **9,810** |

  9,810 × 1.25 = 12,263 s, inside the **3.5 h guard** (12,600 s). The runner's own check, before the install, needs
  (9,810 − 300) × 1.25 = 11,888 s left. A test asserts the table against the guard.
- **Worst case:** 3.5 h × $1.05 + (93.4 GB checkpoint + 5 GB install) × $0.011/GB = **$4.76**, under the standing rule.
  The expected cost is set by the measured terms (P149's box Q+S read cost $1.574).
- **Run ids:** `p223-5090-1` is the reading. `-2` and `-3` are spares for pre-create refusals and host refusals, which
  are not strikes. Two box-started HARNESS_ERROR, VOID or NOISY reads stop the lane.
- **Sequencing:** this registration merges after one independent review and the maintainer's pass. Then the kit PR, by
  substitution, with its own review. Then the exec ACK with the kit hash and pins, the pin, and `--validate`. The
  diagnostics' records are in the private receipt store.

## Amendment 1 (before any data): the shakedown ids

**Why.** This registration names no shakedown. P149's shakedowns ran on the lab A2000, which is no longer used for lane
runs, and a full-model shakedown on a 5090 would repeat the reading's 93 GB fetch and its bake. P223 therefore
registers a shakedown on P149's one-layer cut of Mixtral (its Amendment 5 dry mode), on the reading's own card class.

**What it exercises, and what it does not.**
- It exercises everything P223 adds to P149's kit, end to end:
  - the eight processes in the registered order, with each arm's knobs, and the box self-test;
  - every engagement counter: the split plan per bucket, the runner's buckets, the bucket-1 replays, the lone rows
    against bucket 2's per-window pad rows, the int8 counters, and the decode census;
  - the invariance, repeatability and cross-W reads, and the reducer.
- It does NOT test scale. The full 93 GB fetch, the full bake and full-model memory and timing are P149's kit, already
  proven on full Mixtral by P149's own reads. A shakedown's tokens per second and its schedule-invariance line describe
  a one-layer model and mean nothing for the reading.

**The shakedown mode (`P223_SHAKE=1`).**
- **Run ids:** `p223-shake-1`, `-2` and `-3`, never `p223-5090-*`. The runner refuses a run whose `P223_SHAKE` does not
  match its id, either way, before any install (rc 78).
- **The model:** the fetch takes shards 1 and 19 of `eba92302` (9.1 GB) with the config and the tokenizer.
  `p149_slice.py` cuts them to one decoder layer, exactly as P149's Amendment 5 does.
- **The run:** everything after the cut is the kit's, with every check kept, at 2 timed reps (not 5) after the warm
  pass. The early-VOID read after each arm's first process reports and goes on, so one shakedown finds every fault.
- **The closed forms at one layer** (fixed here, before the kit). Every layer-dependent expectation in the rule takes
  P149's `census(L)` at L = 1, as P149's Amendment 5 dry mode does:
  - the fold census: `fuse_qkv_n` 0, `fuse_t1_glue_n` 2L + 1 = 3, `fuse_t1_glue_r2_n` [1, 1] and
    `fuse_router_epilogue_n` 1, where the reading's are 65, [32, 32] and 32;
  - the int8 attention projections: 4L = 4, where the reading has 128. D1 and I1 must still build them and route int8
    calls in every captured bucket. D0 and I0 must still carry none;
  - the reported int8 closed form: 3 × 4L = 12 calls per captured bucket;
  - the model: the slice, at the revision P149's dry mode records for it, not `eba92302`'s full checkpoint.

  Every other VOID check is unchanged: the pins, the card, the server's shape, the buckets and split plans, the
  replays, the lone rows, the census of decode calls, every bucket captured, no eager steps, and the prompts. The reps
  check takes the shake's own count: 2 timed reps per workload, each recording every row (5 in the reading).
- **The verdict is SHAKE, hard-coded, never a reading.** The reduce lists every instrument fault the rule would VOID on,
  across every process. Labels, prices and predictions are reported and never graded. A shake id can never write READ;
  a test proves it. rc 0 when there is no instrument fault, else 29.
- **An instrument run: never a reading and never a strike.** A fault is fixed and checked at USD 0 where possible, then
  the next shake id runs.
- **Host and cost:** one RTX 5090 at the reading's floors and rate cap, except a 60 GB disk, from main after the
  shakedown kit merges.
- **The shakedown table (s):**

  | term | s |
  |---|---|
  | boot | 300 |
  | install, self-tests and the compiled suites | 450 |
  | fetch (two shards, 9.1 GB) | 900 |
  | the cut and the bake | 600 |
  | 8 arm processes × 240 (2 timed reps) | 1,920 |
  | reduce | 120 |
  | reserve | 600 |
  | **total** | **4,890** |

  4,890 × 1.25 = 6,112 s, inside the **1.75 h shake guard** (6,300 s). The runner's own check needs (4,890 − 300) ×
  1.25 = 5,737 s left before the install. A test asserts the table against the guard.
- **Worst case:** 1.75 h × $1.05 + (9.1 GB + 5 GB install) × $0.011/GB = **$1.99**, under the standing rule.
- **Sequencing:** this amendment merges first, then the small kit PR, then the shakedown. Its self-test and every
  counter are read before the exec ACK request for `p223-5090-1`.

## Amendment 2 (2026-10-22, before any data): spare shake ids

**Why.** Two of Amendment 1's three shake ids were spent before the kit ran: `p223-shake-1` on a host refusal at the
launcher's pre-flight (the bandwidth probe's ssh was refused), and `p223-shake-2` on the box's power floor (the host
enforced 475 W against the 575 W floor, a refusal before any install). Neither is a strike. Each spends an id, as
Amendment 1 says.

**What changes.** `p223-shake-4`, `-5` and `-6` are registered as instrument runs under Amendment 1's terms exactly: the
same kit and shake mode, the same shakedown table and 1.75 h shake guard, the same worst case ($1.99), the same box
floors, the verdict SHAKE (never a reading, never a strike), and each spent the same way `-1` to `-3` are. The runner
and the driver accept the six shake ids and refuse any other. Nothing else changes, and no term of the reading moves.
