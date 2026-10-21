# P222: K25's decode x plan at Mixtral-8x7B's W16 shapes, against Mixtral's served default (the tree at today's plan)

**Status:** registered before any data. A microbench PREMISE for e4b#1313, assigned by the maintainer on
2026-10-21 under the owner's standing rule for idle-time exploration. A GO opens a served read, its own registration. This
lane changes no default, no gate and no speed claim, and it quotes no gain into the served step. **The id:** P162-P221
appear in prose on main (TC1's prediction labels among them); P222 is the first id `git grep -w` finds nowhere.

| the maintainer's condition (2026-10-21) | where |
|---|---|
| a rented RTX 5090; the A2000 is off lane work (the owner's directive) | [Budget](#budget-and-sequencing), [The shakedown](#before-the-reading-the-paid-shakedown) |
| the GO threshold and the arms that decide it, before any box | [The rule](#the-rule-p222_reducepy-fixed-before-data) |
| the seeds | [The routing](#the-routing-seeded-two-blocks) |
| a sweep's plan chosen on one seed set and graded on another | [The routing](#the-routing-seeded-two-blocks), [The rule](#the-rule-p222_reducepy-fixed-before-data) |
| a plan other than `today` or `c1` needs a `_K25_PLANS` code PR before any served read | [What follows](#what-follows) |
| token gates compare identical schedules only (P149's bucket finding) | [What follows](#what-follows) |
| an exec ACK with the kit hash and the e4b and gnf4 pins | [Budget](#budget-and-sequencing) |
| the selection's margin over its runner-up, reported | [The rule](#the-rule-p222_reducepy-fixed-before-data) |

## Why

Mixtral-8x7B is the served family furthest from its floor at 16 requests, and nearly all of the gap is one kernel.
- **What serves it today:** at 16 rows the expert GEMMs run K25 (`nf4_smallm.gemm_nf4_grouped_smallm`) at the select
  tree and today's plan. e4b's per-family defaults (`K25_LUT_DEFAULT_FAMILIES`, `K25_PLAN_DEFAULT_FAMILIES`) list only
  `qwen3_moe`, so `mixtral` resolves to `tree` and `today`, with the reason recorded.
- **How much of the step it is:** P145's per-name record (`p145-5090-1`, `pername_b16.json`, RTX 5090, 16 slots, graphs
  on) has the K25 pair at 29.559 + 18.942 ms of a 53.224 ms step: **0.911** (`p149_preread.MOE_SHARE_W16`). That is
  0.924 ms a gate_up call and 0.592 ms a down call.
- **How far from the bytes (UNVERIFIED arithmetic, not a premise of this lane):** with all 8 experts read each step
  (below), NF4 at 0.5 B an element plus fp32 absmax per 64, that is about 0.57 TB/s for gate_up and 0.45 TB/s for
  down, against a ~1.5 TB/s copy rate. The registration does not need this figure, and nothing grades it.
- **What moved Qwen3's analogue:** K39's decode (`lut="prmt"`) read 1.18x on the K25 pair (P156) and 1.0925x served
  (P158); its plan `c1` read 1.66x over prmt at today's plan (P160) and 1.4654x served at 16 requests (P161). Mixtral's
  regime differs: 8 experts of ~4 rows each against 128 experts of ~1 row, and matrices of 28672 x 4096 and 4096 x 14336.
  So neither the decode's gain nor c1's carries by assumption, and c1's down grid falls from 1,280 programs at today's
  plan to 320 on 170 SMs. This lane sweeps the plan at Mixtral's shapes and grades the choice on fresh draws.
- **Nothing refuses Mixtral's shapes:** K25 needs K % 64 == 0 (4096 and 14336 both), masks N, and refuses `prmt` only
  on ROCm. `E4B_K25_LUT=prmt E4B_K25_PLAN=c1` serves Mixtral today with no code change.

**The question:** at Mixtral's W16 shapes, does K25 at the plan a sweep selects under K39's decode (M) beat the served
default (T, the tree at today's plan) by at least 1.10x on routing draws the selection never saw?

## The arms (block B, the reading)

| arm | gate_up plan | down plan | decode | role |
|---|---|---|---|---|
| **T** | today's, 32/64/4/3 | today's | tree | Mixtral's served default: **the control** |
| T2 | today's | today's | tree | T captured again: the self-pair (the same cubin) |
| P | today's | today's | prmt | the decode alone (`E4B_K25_LUT=prmt`, no code change), reported |
| C1 | 64/256/4/2 | 128/128/8/3 | prmt | e4b's `c1` (`E4B_K25_PLAN=c1`, no code change), reported |
| **M** | block A's selection | block A's selection | prmt | **the registered question** |
| TM | M's | M's | tree | M's plan without the decode, reported *(superseded by Amendment 1: dropped)* |

T against P moves only the decode; P against M moves only the plan; TM against M only the decode again. T/T2 is the *(superseded by Amendment 1)*
only same-cubin difference, and it is the instrument's check. **A plan is its own compile**, so ratios across plans price
the whole compile (tile, pipeline depth, warps, registers, shared memory), which is what a served change would ship.

## The routing: seeded, two blocks

No Mixtral router replay exists (P153's is Qwen3's). The routing is drawn in `p222_reduce.picks`: for each (layer, step,
row), two distinct experts of eight, uniform, from one `random.Random(seed)`, in [layer][step][row] order. **Block A
(the selection) is seed 2220; block B (the reading) is seed 2221.** 8 layers x 128 steps x 16 rows each, pinned by
sha256 of the int8 table (`ROUTE_SHA256`): A `996d7f09...`, B `5f368c0f...`.

- **How it compares with the served route:** with 16 rows at top-2 of 8, the expected distinct experts per layer is
  8 x (1 - (21/28)^16) = 7.920. B's draw reads **7.918** (944 of B's 1,024 (layer, step) cells touch all 8 experts, 76
  touch 7, 4 touch 6). Each expert takes at most 16 rows, so every layer is one tile per touched expert.
- **What it is not:** Mixtral's real router picks experts unevenly, which this draw does not model. At ~7.9 of 8
  experts touched, the weight stream per step is almost routing-independent, and the K25 programs differ only in rows
  per tile. The served read, with real prompts, is the routing test.
- **A selects, B reads:** the plan is chosen on A's steps 0, 16, ..., 112 (8 steps) and graded on B's steps 0-125 (126
  reps, 21 Williams squares). No draw of B is used by the selection. *(superseded by Amendment 1)*

## The instrument (`p222_bench.py`)

P160's instrument, with P156's sweep in front of it. Everything not named here is P160's.
- **The box:** one RTX 5090 (sm_120), torch 2.8.0+cu128 and triton 3.4.0 (the image's); e4b at this registration's
  merge commit; **grouped-nf4-gemm v0.50.0** at `41ef086e` (gnf4#572, P158's and P160's pin), with the same three kernel
  files pinned: `nf4_smallm.py` `ee46969a...`, `int4_b32.py` `773eff5d...` and `nf4_grouped.py` `a7090915...`. The
  tripwire asserts that K39's decode is the kernel's LUT mode 4, that an unset `E4B_K25_LUT` and `E4B_K25_PLAN` resolve
  `mixtral` to the tree (`default-off`) and to `today`, so T is the served control, and that e4b's `_K25_PLANS["c1"]`
  is the C1 arm's pair.
- **The call, exactly as the served route makes it:** gate_up through the lean tile table's order with `gather_div=2`
  (Mixtral's top-2), down into the caller's order with `scatter=order`, `dot_bf16=False`. Only the plan or the decode
  moves.
- **The stores:** random NF4 bytes and fp32 absmax in the served layout for **8 MoE layers**: gate_up [8, 28672, 2048]
  and down [8, 4096, 7168] bytes a layer: about 0.8 GB a layer with their absmax, 6.3 GB in all. One layer is far beyond L2, so nothing replays
  from cache. Every arm reads the same bytes.
- **Block A, the sweep:** per projection, every plan of P156's 144-plan grid under `prmt`, in one shuffled order (seed
  222). Each plan's outputs on A's first step are compared with the tree at today's plan, every layer, `torch.equal`
  (a plan that moves a bit is excluded and recorded). Then 5 timed replays of its one-projection graph on each of A's 8
  steps. A plan that cannot compile or launch is recorded and excluded.
- **Block B, the reading:** P160's reps: the spacer **S** (T's calls with their own outputs, never timed) and every arm's
  graph captured in the rep's Williams row (r mod 6); 10 timed replays per arm, interleaved, **each right after an *(superseded by Amendment 1)*
  untimed replay of S**; an NVML thread sampling SM and memory clocks, power and the **enforced power limit** every 5 ms
  while a timed replay runs; after the replays, every arm's outputs against T's, `torch.equal`.
- **Per arm, reported:** the registers, spills and shared memory each projection's plan compiled to.

## The rule (`p222_reduce.py`, fixed before data)

- **The selection (`select`):** per projection, among plans that kept the tree's bits and read every replay, the lowest
  mean over A's 8 steps of each step's median replay; ties to the lower grid index. M is that pair. Today's plan and
  c1's pair are in the grid, so M may be either. **Reported:** each projection's runner-up time over the winner's
  (`selection_margin`), so a near-tie selection is visible.
- **The reading:** every ratio is paired per rep (each arm's median of its 10 replays), read as the median over B's 126 *(superseded by Amendment 1)*
  reps with the larger-of SE (P155's rule: the larger of sqrt(pi / 2) sd / sqrt(n) and a seeded 2,000-resample bootstrap
  of the median; seed 222), and a lower bound of the median minus 2 SE.
- **VOID** (the instrument):
  - a block's routing table is not the pinned draw, or B's distinct-expert counts are not the draw's;
  - the sweep's plans or order, the selection steps, the arms, their plans and decodes, the spacer, B's steps or the 10
    replays are not as registered; **block B timed an M that `select` does not name from block A's record**;
  - a rep's order is not its Williams row, or a rep lacks 60 spacer replays; a tile-table fault;
  - not an RTX 5090 at cc 12.0, or fewer than 8 layers;
  - **T2 differs from T by a bit on any rep**, or T/T2's median is outside 0.99-1.01;
  - **the clocks are unread**, or **any sample's enforced power limit is under 575 W**.
- **NO_GO with a contract finding:** P, C1 or M moves a bit against T on any rep. K25's decodes and plans move no
  output bit by construction (`nf4_smallm.py`'s docstring), so this would be a finding against grouped-nf4-gemm, and no
  timing is ruled on. (TM moving a bit is reported.) *(superseded by Amendment 1)*
- **GO:** M's median T/M minus 2 SE **>= 1.10**, AND the same bound **clock-adjusted >= 1.10** (P160's one-sided
  adjustment: if M's median SM clock is above T's, the bound is multiplied by T's clock over M's; a slower-clocked M is
  never credited).
  - **The bar:** 1.10, not P160's 1.05, because the pair is 0.911 of Mixtral's W16 step (P145): a 1.10 microbench is
    about a 1.09 step at that share. P160's 1.05 against Qwen3's ~0.69 share was about a 1.034 step.
- **NO_GO:** otherwise.
- **On GO, the served read's arm (`served_arm`, named before data):** **P** if M is today's plan in both projections
  (the decode alone); **C1** if M is c1's pair, or if M's median over C1 on B is under **1.02** (a new pair that does not
  clearly beat the existing knob is not worth a code PR); otherwise **M**, which needs a `_K25_PLANS` code PR first.
- **Reported, never ruling:** P/T, C1/T and TM/T with their clock bounds; M over C1; each arm's clocks and the label *(superseded by Amendment 1)*
  "clock-confounded" where an arm's median SM clock is off T's by more than 2 %, or its median memory clock by more than
  1 %; the milliseconds per arm; the compiled metadata; the sweep's every plan; the selection margin; the walls.

## Predictions (graded in the read, never ruling)

| prediction | basis | if HELD | if MISSED |
|---|---|---|---|
| **the decode:** T/P in **1.15-1.35** | Qwen3's P156 1.1755; Mixtral's per-byte decode cost is a larger share at 4 rows a tile | the decode alone carries most of Qwen3's gain | below: Mixtral's pair is bound elsewhere (issue, occupancy); above: the tree costs more at these shapes |
| **the headline:** T/M in **1.30-1.90** | the pair's ~0.5 TB/s against ~1.5 copy | a plan closes much of the gap | below: the bytes are not the limit the arithmetic suggests; above: the sweep found more than the bandwidth gap allows (check the clocks) |
| **M is not c1's pair** | c1 was chosen at Qwen3's shapes; its down grid is 320 programs here | Mixtral needs its own plan (a code PR on GO) | c1 carries across families |
| **the self-pair** within 0.005 of 1 | P160's spacer | | a noisier card; the 2-SE guard carries it |
| **clocks:** M's median SM clock within 2 % of T's | P160 and P161 | the ratio is not a clock effect | labelled; the GO's one-sided bound already prices it |
| **contract:** every arm keeps T's bits on every rep | K25's construction; P156's 144 plans on sm_120 | | a contract finding |

## Before the reading: the paid shakedown

The $0 A2000 shakedown P160 ran is gone: the A2000 is off lane work since 2026-10-21 (the owner's directive). So after
this registration merges, and before the reading:
- **`p222-shake-1`** (spare `-2`): the same driver and runner on the **cheapest NVIDIA card at sm_86 or later with 15
  GB** that passes the usual screen, `--shakedown --layers 2 --reps 6 --grid 6 --select-steps 2` (today's plan swept *(superseded by Amendment 1)*
  first), no power floor. Declared at most $0.40/h, **guard 0.75 h**, at most 10 GB at <= $0.011/GB (the install; no
  model is fetched): worst case 0.75 x $0.40 + 10 x $0.011 = **$0.41**, under the standing rule. (Read-only search at registration, 2026-10-21: under the launcher's filters with a
  16 GB host-RAM floor, RTX A4000 listed none, L4 from $0.362/h, RTX 4090 from $0.442/h.) **Its own fit:** `NEED_BENCH` = 600 s for a shakedown id (12 plan compiles at P156's
  slowest 2.65 s = 31.8 s, 12 metadata compiles 31.8 s, and 2 layers' sweep and 6 reps a few seconds: about 70 s, *(superseded by Amendment 1)*
  floored at 600 s), so `FIT` = 600 + 600 + 600 = 1800 s, `FIT_NEED` = 2250 s, and (300 + 1800) x 1.25 = 2625 s <= 2700 s
  (0.75 h). It finishes 0 only on the reducer's
  **SHAKEDOWN**, which runs every structural check and is never graded.
- **What it shows:** the install, the tripwire, the premise, the sweep, the selection, the six arms, the spacer, the *(superseded by Amendment 1)*
  clocks and the receipts, end to end. **What it cannot show:** sm_120's speed. Its times seed nothing.
- An instrument fault it finds is fixed by an amendment before the reading.

## Budget and sequencing

- **One RTX 5090 on Vast,** per-second billing. Hosts with a recorded power cap of at least 575 W
  (`vast_min_power_limit_w: 575`, `vast_power_limit_unknown: exclude`), read within the last 3 days. Declared at
  $1.05/h, **guard 1.5 h**, at most 10 GB at <= $0.011/GB: worst case 1.5 x $1.05 + 10 x $0.011 = **$1.69**, under the
  standing rule. Disk 60 GB (the box's floor 20 GB free). Nothing is fetched from a hub.
- **The fit, e4b#1845's rule, with at least 25 % headroom.** `NEED_BENCH` is derived from records, every term
  conservative:
  - **The compiles:** 288 plan x projection first calls, each at P156's slowest (`p156-5090-1`): 288 x 2.65 = 763.2 s
    (P156's own sum was 98.8 s).
  - **The sweep's GPU time:** 144 plans x 8 steps x 7 executions (2 warm, 5 timed) x 8 layers at P145's per-call times
    (0.924 ms gate_up, 0.592 ms down, the tree at today's plan), scaled by P156's mean sweep slowdown over today's plan
    (1.155 gate_up, 1.739 down): 135.2 s.
  - **The sweep's host time:** 288 x 8 graphs at P160's per-graph host cost (0.063 s, from P156's confirmation): 145.2 s.
  - **The reading:** per rep, 7 graphs' host cost, 14 warm executions and 120 replays (60 timed, 60 spacer), every arm
    at T's time (12.13 ms a graph of 8 layers): 126 x 2.066 = 260.3 s. *(superseded by Amendment 1)*
  - **The compiled metadata:** 12 x 2.65 = 31.8 s. *(superseded by Amendment 1)*
  - So about 1,336 s; x1.5 = 2,003 s, rounded up: `NEED_BENCH` = **2100 s**. The step alarm is 2,700 s. *(superseded by Amendment 1)*
  - **The install, tripwire, self-tests and premise:** `INSTALL_TO_GATE_S` = 600 s (P160's). **The reserve:**
    `RESERVE` = 600 s.
  - **The runner's check, before the install:** `FIT` = 600 + 2100 + 600 = 3300 s, and `FIT_NEED` = 3300 x 1.25 = 4125 s.
    Anything short of that is a refusal, rc 41.
  - **The static fit,** (boot + fit) x 1.25 <= guard: boot 300 s + 3300 s = 3600 s, and 3600 x 1.25 = 4500 s <= 5400 s (1.5 h).
    The runner's check (4125 s) refuses only if the box takes more than 5400 - 4125 = 1275 s to reach it.
    `tests/test_p222.py` asserts each figure.
- **The premise, on the card:** K25's row-exact GPU tests (4, P156's) and K39's compiled contract (grouped-nf4-gemm
  v0.50.0's `kernel/test_nf4_smallm_prmt_gpu.py`, 23, P158's staged copy): exactly 27 passed, else rc 25. Both are P160's
  staged bytes.
- **Run ids** `p222-5090-1` to `-3`, and `p222-shake-1` / `-2` for the shakedown. Two consecutive HARNESS_ERROR, VOID or
  NOT_RUN readings stop the lane. P160's strike rules hold: a pre-flight host stall, a power-floor refusal (rc 17) or a
  fit refusal (rc 41) is not a strike; a time-left skip of the bench (rc 40) is. A shakedown is never a strike.
- **The host, for the served read that follows a GO:** this lane's machine is pinned first, so the served read does not
  carry a microbench-to-served card spread (a 2026-10-21 hypothesis about P149's and P161's carries). If it
  does not list, the served read names the cross-card carry and reports it, never attributes it.
- **Sequencing:** this registration's merge after one independent review and the maintainer's pass; then the shakedown;
  then the exec ACK on the bus (the kit hash `staged-p222.sha256`, the e4b launch commit and the gnf4 pin, compared with
  the reviewed ones), the pin before create, the shared pre-flight, and `--validate` on the exact manifest.

## Amendment 1 (2026-10-21, before any reading): TM is dropped, five arms, and an arm fault is recorded

The lines marked *(superseded by Amendment 1)* above stand as registered; this section replaces them.

**The evidence.** The paid shakedown `p222-shake-1` (an L4, sm_89, $0.022, receipt in the private receipt store) ran
the install, the tripwire, both self-tests, the premise (27 passed), the sweep and the selection cleanly. It then
crashed in block B while compiling **TM**, the tree at M's plan: M's down plan 128/256/8/4 under the tree needs
210,944 B of shared memory against the 101,376 B per-block limit. The RTX 5090 (sm_120) has a similar per-block limit,
so the reading would have crashed the same way.

**The ruling (the maintainer, before any reading), and the rules it sets:**
1. **TM is dropped.** M stays what the registration asks: the prmt plan the sweep selects, with no eligibility term
   for its tree variant. Restricting M to plans whose tree variant compiles would change what the GO reads and could
   exclude the best prmt plan, all to save an arm that never ruled. The decomposition keeps T/P (the decode at today's
   plan) and P/M (the plan under prmt); only the tree-at-M interaction goes. Reported, never ruling: P/T and C1/T with
   their clock bounds; a contract finding is read on P, C1 and M.
2. **Block B runs five arms** (T, T2, P, C1, M) in Williams' design for an odd count: the five cyclic orders and their
   mirrors, 10 orders a square (rep r takes row r mod 10), each arm in each position twice and each ordered pair
   adjacent twice. B's reading takes steps 0-119 of seed 2221: **120 reps, 12 whole squares**, read as the median over
   those 120 reps.
3. **An arm fault is recorded, never a crash.** A compile, launch or capture failure in the compiled-metadata pass or
   in block B is written to the bench record (`arm_fault`) and ends the bench (rc 34); the reducer VOIDs with the
   reason.
4. **The shakedown** runs `--shakedown --layers 2 --reps 10 --grid 6 --select-steps 2`: 10 reps, so every Williams
   order is tried. Its own fit is unchanged (12 plan compiles at P156's slowest 2.65 s = 31.8 s, 10 metadata compiles
   26.5 s, and 2 layers' sweep and 10 reps a few seconds: about 60 s, floored at 600 s). It exercises the five arms.
5. **The budget derivation** for the reading: per rep, 6 graphs' host cost, 12 warm executions and 100 replays (50
   timed, 50 spacer), every arm at T's time (12.13 ms a graph of 8 layers): 120 x 1.736 = 208.3 s. The compiled
   metadata: 10 x 2.65 = 26.5 s. So about 1,278 s; x1.5 = 1,918 s. `NEED_BENCH` stays **2100 s**, the registered and
   larger bound, so the fit, the guard and the worst case are as registered.

**Unchanged:** the GO rule and its bar, the seeds, block A and the selection, the served-arm rule, the card, the
budget and the shakedown's price. Next: `p222-shake-2` on the same terms, and the exec ACK only after it reads
SHAKEDOWN.

## Receipts

Fetched to the run directory's `p222/` and published under `bench/p222/receipts/<run>/` by the 2026-10-19 citation rules:
`bench_p222.json` (the sweep's every plan, every replay time, the clock summaries and each arm's compiled metadata),
`verdict_p222.json`, `summary.txt`, `forensics.txt`, `versions.txt`, `gpu_phases.csv` and the logs, with `SHA256SUMS`.
The box writes no instance id and no host name; the bench reads no GPU UUID. The record's `bench/p222/REDERIVE.md`
carries the public re-derivation line. `RESULTS-p222.md` is written from those files.

## What follows

- **GO:** a served read, its own registration on P158's and P161's kit, at Mixtral's real prompts and routing, against
  today's served default, with its own bars. It quotes no gain from this lane. Its arm is `served_arm`'s:
  - **P or C1:** no code change; the knob is set in the arm's environment.
  - **M:** first an e4b PR adding M's pair to `_K25_PLANS` (opt-in, no speed claim; its tests are the plan-dispatch
    contract), then the served read.
  - **Its token gate compares identical schedules only:** P149's follow-up found that the served decode's tokens can
    change with the graph bucket a step lands in (the fp8 split-KV combine plans its split count per bucket). The served
    read's identity check compares arms on the same prompts, lengths and arrival order.
- **NO_GO:** Mixtral's K25 stays at the tree and today's plan, and the sweep's record stays a reported table.

## What this lane cannot say

- **Nothing about the served step.** It times the two K25 GEMMs alone, on 8 layers.
- **Nothing about real routing:** the picks are seeded and uniform. Not Mixtral's router.
- **Nothing for other families, cards, toolchains or row counts:** one RTX 5090, Mixtral-8x7B's shapes, 16 rows,
  triton 3.4.0.
- **Nothing that depends on weight values:** the bytes are random, and outputs are compared decode against decode.

**Review:** one independent reviewer who is not a consumer of this lane, then the maintainer's pass.
