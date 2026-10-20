# P160: decode x plan — K39's decode at P156's carried tile plans against prmt at today's plan, at Qwen3-30B-A3B's W16 shapes

**Status:** registered before any data. A microbench PREMISE for stream (B) of e4b#1313. A GO opens two later pieces of
work, each its own PR: an opt-in per-projection K25 plan in e4b, and a served read with its own registration. This lane
changes no default, no gate and no speed claim, and it quotes no gain into the served step. The maintainer assigned the
lane on 2026-10-20 and set its conditions; each is mapped below.

| condition | where |
|---|---|
| P156's harness, microbench first | [The instrument](#the-instrument-p160_benchpy) |
| fresh held-out steps | [The data](#the-data-two-blocks) |
| the carried plans against prmt at today's plan, now the served default | [The arms](#the-arms) |
| P157's measurement lessons: a spacer graph, same-cubin differences, per-arm clocks and the enforced limit | [The instrument](#the-instrument-p160_benchpy), [The rule](#the-rule-p160_reducepy-fixed-before-data) |
| $0 until registered; a $0 A2000 shakedown before the box | [The shakedown](#before-the-box-the-0-a2000-shakedown) |
| a served read as its own lane, if this holds | [What follows](#what-follows) |

## Why

[P156](../p156/RESULTS-p156.md) read **GO** on K39's decode at today's plan: prmt ran both W16 expert GEMMs 1.1755x faster
than the select tree. [P158](../p158/RESULTS-p158.md) then read it in the served step (FLIP, g16 1.0925), and e4b#1868
made `prmt` qwen3_moe's K25 decode by default. So **P, prmt at today's plan, is now the served control.**

P156 also swept prmt over K25's 144-plan grid (reported, never ruling) and carried three plan pairs. Each confirmed far
above P on P156's 64 held-out steps:

| arm | plan (gate_up / down), block_n / kc / warps / stages | T/arm | over P (P/arm) |
|---|---|---|---|
| C1 | 64/256/4/2 / 128/128/8/3 | 1.9019 | **1.6167** (per rep 1.593-1.650) |
| C2 | 64/256/4/3 / 32/256/4/2 | 1.8313 | 1.5574 |
| C3 | 64/256/8/3 / 128/128/4/2 | 1.8652 | 1.5847 |

- **The gain is mostly gate_up's.** In P156's selection, prmt's gate_up read 8.04 ms a graph at today's plan and 3.98 ms
  at 64/256/4/2; down read 2.63 ms and 2.05 ms at 128/128/8/3.
- **It is the decode and the plan together.** Under the tree the same plans are slower than today's (P155's record:
  gate_up 64/256/4/2 at 14.23 ms against 8.32, 255 registers; down 128/128/8/3 at 5.29 ms against 3.37). K39's lighter
  decode (64 registers at today's plan against the tree's 80) is what lets the big tiles pay. On sm_120, P156 compiled
  C1's gate_up to 194 registers and its down to 127, no spills.
- **Why it needs its own read.** The figures above were reported, not registered. They were selected with a winner's
  curse of 0.153 (selection 1.770 against the confirmed 1.617; P156's prediction band was 0-0.03, MISSED). P156 also had
  none of P157's defences: no spacer, so each arm's predecessor never varied, and clocks only at the bench's start and
  end, while a lighter kernel can hold a higher boost.

**The question:** with the plan the only thing that moves, does C1 (prmt at P156's first carried plan pair) beat P (prmt
at today's plan) on a stream P156 never timed, under a registered rule with a spacer, a balanced order and per-arm clocks?

## The arms

Six arms, every plan named before data. The plans are P156's record's `carried`, in its rank order, never re-chosen here
(`p160_reduce.CARRIED`; `tests/test_p160.py` asserts it).

| arm | gate_up plan | down plan | decode | role |
|---|---|---|---|---|
| **P** | today's, 32/64/4/3 | today's | prmt | the served default: the control |
| **P2** | today's | today's | prmt | P captured again: the self-pair (the same cubin) |
| **C1** | 64/256/4/2 | 128/128/8/3 | prmt | **the registered question** |
| C2 | 64/256/4/3 | 32/256/4/2 | prmt | reported |
| C3 | 64/256/8/3 | 128/128/4/2 | prmt | reported |
| T | today's | today's | tree | reported: the decode alone, P156's GO re-read |

C1 against P moves only the plan; T against P moves only the decode. P/P2 is the only same-cubin difference, and it is
the instrument's check. **A plan is its own compile**, so C1 and P are different cubins by construction; their ratio
prices C1's whole compile (tile, pipeline depth, warps, registers, shared memory), which is exactly what a served change
would ship. No ablation inside a cubin is claimed.

## The data: two blocks

The routing is P153's router replay, `bench/p153/router-replay/rr.json.pt` (sha256 `3c59e72f...`): 48 layers x 32 rows x
128 decode steps x top-8, for two decoded streams of P109's 16 prompts. P156 timed only rows 0-15 (S0): selection on
steps 0, 8, ..., 56, confirmation on steps 64-127. Two blocks, in this order:

1. **S1, the READING:** rows 16-31, P153's second stream (`df2194a4...`, distinct total 350,529), on steps **0-125**: 126
   reps, 21 whole Williams squares. No K25 microbench has timed this stream.
2. **S0R, reported:** rows 0-15 (`3020dbce...`, distinct total 350,868), on P156's held-out steps **64-123**: 60 reps, 10
   squares. It re-reads P156's 1.617 under this lane's instrument, so a difference between the blocks separates the
   instrument from the routing. It never rules.

**How fresh S1 is, stated before data.** S1 decodes the same prompts as S0, with a different head (P153's int8 arm), so
its tokens diverge from S0's row by row. Of S1's 2,048 (row, step) cells, **43.3 %** carry picks equal to S0's at every
layer; four of the 16 rows are identical at every step, and **no step has all 16 rows identical**: at every step some
row's picks differ from S0's at some layer. The per-step distinct counts track S0's (S1/S0 0.999 over the replay).
The overlap is FRONT-LOADED, and it peaks at the step numbers P156 selected on (cells identical to S0's at
every layer, by S1 step):

| S1 steps | 0-15 | 16-31 | 32-47 | 48-63 | 64-79 | 80-95 | 96-111 | 112-127 |
|---|---|---|---|---|---|---|---|---|
| identical to S0 | 68.8 % | 59.8 % | 50.0 % | 44.5 % | 38.3 % | 32.0 % | 28.1 % | 25.0 % |

At P156's eight selection step numbers (0, 8, ..., 56) it is **57.0 %** (11, 11, 11, 9, 8, 8, 8 and 7 of 16 rows); over
the reading's steps without them, 42.7 %; over steps 64-125, **31.0 %**. So the reducer also REPORTS C1 on two splits of
the S1 block: steps 64-125 (62 reps, the least-overlapping half) and the block without those eight step numbers (118
reps). They never rule; they show whether the early overlap carried the headline.

So S1 is fresh for TIMING (no arm or plan was ever selected or timed on it), and partly fresh for ROUTING. A fully
independent stream (new prompts) needs its own router replay; it is not taken here. The plans were never selected on
any step either block uses: P156's selection steps (0, 8, ..., 56 of S0) appear in neither.

## The instrument (`p160_bench.py`)

P156's instrument, with the plan as the moving argument and P157's defences. Everything not named here is P156's.
- **The box:** one RTX 5090 (sm_120), torch 2.8.0+cu128 and triton 3.4.0 (the image's); e4b at this registration's
  merge commit; **grouped-nf4-gemm v0.50.0** at `41ef086e` (gnf4#572, the release carrying K39; P158's pin). The tripwire
  pins three kernel files, byte-identical to P156's pin: `nf4_smallm.py` `ee46969a...`, `int4_b32.py` `773eff5d...` and
  `nf4_grouped.py` `a7090915...`. It asserts that K39's decode is the kernel's LUT mode 4, that the served plan is today's,
  and that **an unset `E4B_K25_LUT` resolves qwen3_moe's decode to prmt by default** (`default-allowlisted`), so P is the
  served control.
- **The call, exactly as the served route makes it** (P156's): gate_up through the lean tile table's order with
  `gather_div=8`, down into the caller's order with `scatter=order`, `dot_bf16=False`. Only the plan or the decode moves.
- **The stores:** P156's, random NF4 bytes and fp32 absmax in the served layout for all 48 MoE layers (16.3 GB), so
  nothing replays from L2. Every arm reads the same bytes.
- **The routing, re-checked on the box before anything is timed:** the replay's sha256, each stream's slice hash, the
  shape and dtype, ids in [0, 128), and no row picking an expert twice; each step's real tiles a prefix of one tile per
  expert. The reducer checks each stream's distinct total against P153's.
- **Per arm, reported:** the registers, spills and shared memory Triton compiled each projection's plan to.

### The reps

On each step of a block, one paired rep:
- **The graphs:** the spacer **S** (P's calls with their own output buffers, never timed) and every arm's graph (both
  GEMMs of all 48 layers), captured in the rep's row of a **Williams design** over the six arms (r mod 6, r counted
  within the block): each block is whole squares, so each arm takes each position equally often and every ordered pair of
  arms is adjacent equally often.
- **The replays:** 10 timed replays per arm, interleaved arm by arm in the same order, **each right after an untimed
  replay of S**. S is every timed replay's immediate predecessor, so no arm inherits a light or heavy neighbour's power
  and clock headroom, and none finds its own outputs warm; the design balances position and the arm before S.
- **The clocks:** an NVML thread samples every 5 ms while a timed replay runs, tagged with the block and the arm: the SM
  and memory clocks, the averaged and instantaneous power, the clock-event reasons and the **enforced power limit**.
- **The bits:** after the replays, every arm's outputs are compared with T's, and P2's with P's, `torch.equal`.

The bench records raw replay times and the samples; `p160_reduce.py` derives everything from them.

## The rule (`p160_reduce.py`, fixed before data)

Every ratio is paired per rep (each arm's median of its 10 replays), read as the median over the block's reps with the
larger-of SE (P155's rule: the larger of sqrt(pi / 2) sd / sqrt(n) and a seeded 2,000-resample bootstrap of the median;
seed 160), and a lower bound of the median minus 2 SE.

- **VOID** (the instrument):
  - the replay's sha256 or a stream's hash is not the pinned one, or a stream's distinct total is not P153's;
  - the arms, their plans and decodes, the spacer, the blocks and their order, the steps or the 10 replays are not as
    registered; a rep's order is not its Williams row; a rep lacks 60 spacer replays;
  - not an RTX 5090 at cc 12.0, or fewer than 48 layers;
  - a tile-table fault;
  - **P2 differs from P by a bit on any rep** (P is not deterministic), or the S1 self-pair P/P2's median is outside
    0.99-1.01;
  - **the clocks are unread** (the GO's clock bound needs them), or **any sample's enforced power limit is under 575 W**.
- **NO_GO with a contract finding:** P or C1 moves a bit against T on any rep of either block. K39's bitwise tier would
  then fail on sm_120 at that plan, a finding against grouped-nf4-gemm, and no timing is ruled on. A deterministic
  mismatch reads the same on every rerun; it is not a VOID and not a strike. (C2 or C3 moving a bit is reported.)
- **GO:** on the S1 block, C1's median P/C1 minus 2 SE **>= 1.05**, AND the same bound **clock-adjusted >= 1.05**. The
  clock adjustment is one-sided and conservative: if C1's median SM clock over the block is above P's, the bound is
  multiplied by P's clock over C1's (the whole clock difference taken from C1 at 1:1 scaling, an upper bound for a
  kernel near the byte floor); a slower-clocked C1 is never credited. Each arm's median clock is its tagged samples'.
  - **The bar:** 1.05 is P156's, kept. K25's pair is about 10.4 ms of a 15.0 ms served step at prmt (P158's record), so a microbench
    1.05 is about a 1.034 step, the size of the flips P153 and P158 read.
- **NO_GO:** otherwise.
- **Reported, never ruling:** C2's and C3's P/C (with their clock bounds); T/P; the S0R block's every ratio; each arm's
  clocks and the label "clock-confounded" where an arm's median SM clock is off P's by more than 2 %, or its median
  memory clock by more than 1 % (GDDR7's is fixed; for a byte-bound pair it is the clock that matters), either way;
  C1 on S1's two splits (above); the
  milliseconds per arm; the compiled metadata; the contract findings; the walls.

## Predictions (graded in the read, never ruling)

| prediction | basis | if HELD | if MISSED |
|---|---|---|---|
| **the headline:** C1's median P/C1 on S1 is **1.50-1.70**; P(GO) about 90 % | P156's 1.617 (per rep 1.593-1.650) on S0 | P156's figure carries to an untimed stream under the defences | below: the curse, the predecessor or clocks carried part of it (the S0R block and the clocks say which); above: S1's routing favours big tiles more |
| **the overlap:** C1 on S1 steps 64-125 within 0.03 of C1 on all of S1 | a plan's speed follows the per-step distinct counts, not which rows repeat S0 | the early overlap did not carry the headline | routing-specific selection leaked into the reading; the split says how much |
| **reproduction:** C1 on S0R within 0.03 of C1 on S1 | the routing's statistics match (S1/S0 distinct 0.999) | the plan's gain is not stream-specific | the blocks differ by routing, or the instrument moved one block |
| **the decode:** T/P on S1 in **1.15-1.20** | P156's 1.1755 | P156's GO reproduces under the defences | the spacer or the clocks moved P156's own headline |
| **the self-pair** within 0.005 of 1 | P156's 0.9985 without a spacer | the spacer removed P156's small carry-over | a noisier card; the 2-SE guard carries it |
| **clocks:** C1's median SM clock within 2 % of P's | P158's served clocks differed by 0.25 % | the ratio is not a clock effect | labelled; the GO's one-sided bound already prices it |
| **contract:** every arm keeps T's bits on every rep | P156: all 144 prmt plans bit-identical on sm_120 | | a contract finding with the plan, block and steps |
| **registers:** C1 compiles to 194 (gate_up) and 127 (down), no spills | P156's sm_120 compile at the same toolchain | | re-based on the box's numbers |

## Before the box: the $0 A2000 shakedown

**Done on 2026-10-20, before this registration and before any rental**, under an exclusive `gpu:a2000` claim:
`p160_bench.py --shakedown --layers 4 --reps 6` on the lab's A2000 (sm_86), torch 2.8.0+cu128, triton 3.4.0, with
grouped-nf4-gemm's `kernel/` at `41ef086e` (its three files at the pinned hashes). Its record is
[`shakedown/shake_p160.json`](shakedown/shake_p160.json) (pinned by sha256 in `tests/test_p160.py`), its log
[`shakedown/shake.log`](shakedown/shake.log).
- **End to end:** the bench ran rc 0 in 23.4 s (6 reps a block, one Williams square each), and the reducer reads
  **SHAKEDOWN** with every structural check passing. A shakedown is never graded.
- **The contract on sm_86:** every arm kept T's bits, and P2 kept P's, on all 12 reps.
- **The defences ran:** 60 spacer replays a rep; 2,130 clock samples, every arm in both blocks sampled.
- **The compiled metadata on sm_86:** C1's gate_up 255 registers and 17,408 B of shared memory, its down 188 and 26,624
  B, no spills; C3's down spills 2 registers there. sm_120's counts are the box's (P156: C1 at 194 and 127).
- **What it cannot show:** sm_86 cannot see sm_120's speed. Its replay times are the A2000's, and seed nothing.
- **Changed after the run, outside the timed code:** the self-tests' printed counts and one added reducer self-test case
  (the median's SE against the mean's).

## Budget and sequencing

- **One RTX 5090 on Vast,** its own box: a short single read, so it takes per-second billing, not a whole-hour provider.
  Hosts with a recorded power cap of at least 575 W (`vast_min_power_limit_w: 575`, `vast_power_limit_unknown:
  exclude`), read within the last 3 days. Declared at $1.05/h, **guard 0.75 h**, at most 10 GB at <= $0.011/GB:
  worst case 0.75 x $1.05 + 10 x $0.011 = **$0.90**, under the standing rule. Disk 60 GB (the box's floor 20 GB free).
  Nothing is fetched from a hub.
- **The fit, e4b#1845's rule, with at least 25 % headroom:**
  - **Install, tripwire, self-tests and premise:** `INSTALL_TO_GATE_S` = 600 s (P156's).
  - **The bench:** `NEED_BENCH` = **600 s**, from P156's box walls. Its confirmation ran 384 graphs of the same shape
    (both GEMMs, 48 layers, 10 replays each) in **60.8 s** on an RTX 5090: 0.063 s of host time a graph beyond its
    replays at the arms' recorded medians. A P160 rep captures 7 graphs (S and six arms) and replays 60 timed and 60
    spacer replays (at P's time): 1.67 s, so 310.7 s over 186 reps. The compiles are sourced, not estimated: P156
    first-called all 288 prmt variants on the same card class and toolchain, median 0.25 s, slowest **2.65 s**, so
    this lane's 12 fresh variants (6 arms x 2 projections, P2 sharing P's) are taken at 12 x 2.65 = 31.8 s; P156's
    routing, stores and two tile passes add 3.6 s. About 346 s; x1.5 = 519 s, rounded up.
  - **The reserve:** `RESERVE` = 600 s. No Nsight step.
  - **The runner's check, before the install:** `FIT` = 600 + 600 + 600 = 1800 s, and `FIT_NEED` = 1800 x 1.25 = 2250 s.
    Anything short of that is a refusal, rc 41.
  - **The static fit,** (boot + fit) x 1.25 <= guard: boot 300 s + 1800 s = 2100 s, and 2100 x 1.25 = 2625 s <= 2700 s (0.75 h).
    `tests/test_p160.py` asserts it.
- **The premise, on the card:** K25's row-exact GPU tests (4, P156's) and K39's compiled contract (grouped-nf4-gemm
  v0.50.0's `kernel/test_nf4_smallm_prmt_gpu.py`, 23, P158's staged copy): exactly 27 passed, else rc 25.
- **Run ids** `p160-5090-1` to `-3`. Two consecutive HARNESS_ERROR, VOID or NOT_RUN runs stop the lane. P156's strike
  rules hold: a pre-flight host stall, a power-floor refusal (rc 17) or a fit refusal (rc 41) is not a strike (two
  consecutive fit refusals count as one); a time-left skip of the bench (rc 40) is.
- **Sequencing:** the rental waits for this registration's merge after review and the maintainer's pass, then the
  shared pre-flight, and `--validate` on the exact manifest.

## Receipts

Fetched to the run directory's `p160/` and published under `bench/p160/receipts/<run>/` by the 2026-10-19 citation rules:
`bench_p160.json` (every replay time, the clock summaries and each arm's compiled metadata), `verdict_p160.json`,
`summary.txt`, `forensics.txt`, `versions.txt`, `gpu_phases.csv` and the logs, with `SHA256SUMS`. The box writes no
instance id and no host name; the bench reads no GPU UUID. `RESULTS-p160.md` is written from those files.

## What follows

- **GO:**
  1. **An e4b PR:** the served route has one K25 plan for both projections (`_K25_PLAN`), so C1 needs a per-projection
     plan, tied to the prmt decode (the tree runs these plans slower, P155) and opt-in with no speed claim. Its tests
     are the plan-dispatch contract, not a timing.
  2. **A served read,** its own registration on P158's kit: C1's plans against today's served default, with its own
     bars. It quotes no gain from this lane.
- **NO_GO:** the carried plans park, and P156's reported 1.617 stays a reported figure with this read beside it. The
  S0R block, the clocks and T/P name what moved it.

## What this lane cannot say

- **Nothing about the served step.** It times the two K25 GEMMs alone.
- **Nothing about plans it did not carry,** or about decode x plan for other families, cards or row counts: one RTX
  5090, Qwen3-30B-A3B's shapes, 16 rows.
- **Nothing at another toolchain:** triton 3.4.0 (a triton bump is its own read, grouped-nf4-gemm#570).
- **Not fully independent routing:** S1 shares S0's prompts (above). Not the served NF4 picks: bf16 teacher-forced picks.
- **Nothing that depends on weight values:** the bytes are random, and outputs are compared decode against decode.

**Review:** one independent reviewer, then the maintainer's pass.
