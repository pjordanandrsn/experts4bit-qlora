# P155: K25's plan at Qwen3-30B-A3B's W16 shapes, on S0's real router picks, selection split from confirmation

**Status:** registered before any data. A microbench PREMISE for stream (B) of e4b#1313. A GO opens a served FLIP read
as its own registration and quotes no gain from this lane. It changes no default, no gate and no speed claim. The
maintainer reserved the lane number on 2026-10-19 and set eight conditions, each mapped below.

| condition | where |
|---|---|
| 1. selection split from confirmation; GO >= 1.15x on the confirmation only; the rule and the number carried stated | [The rule](#the-rule-p155_reducepy-fixed-before-data) |
| 2. bitwise identity per plan, asserted; a plan that will not compile or launch on sm_120 recorded and excluded | [Identity](#the-phases) |
| 3. real routing, sha-pinned, not uniform | [The routing](#the-routing) (S0's picks, the maintainer's ruling) |
| 4. self-pairs ~1.000; pairs, not absolutes; capture order rotated | [Confirmation](#the-phases) |
| 5. a $0 A2000 shakedown before the box | [The shakedown](#before-the-box-the-0-a2000-shakedown) |
| 6. guard: the fit with >= 25 % headroom (e4b#1845's rule); 575 W, max age 3 days | [Budget](#budget-and-sequencing) |
| 7. predict how much of the 6.81 ms a plan alone recovers; the next lane if most remains | [Predictions](#predictions-graded-in-the-read-never-ruling) |
| 8. a microbench premise: GO opens the served read and quotes no gain | Status, and [What follows](#what-follows) |

## Why

[P154](../p154/RESULTS-p154.md) priced **6.81 ms** of headroom between the W16 grouped NF4 expert GEMMs (K25) and their
byte floor, 41.3 % of the unprofiled 16.497 ms step: they read their bytes at 0.42 of the copy rate. K25's plan (BLOCK_N,
KC, warps, stages) is its one knob with no quality cost, because the kernel's contract says the plan "moves no output
bit". e4b serves one plan, BLOCK_N 32, KC 64, 4 warps, 3 stages (`hot_residency._K25_PLAN`). That is K27's best of six
TF32 plans at Granite's and OLMoE's B=16 shapes, and it has never been swept at Qwen3's.

What a plan cannot change, and so the basis of the predictions:
- **The bytes.** At 16 tokens, each with 8 distinct experts, no expert gets more than 16 rows, so every routed expert
  has exactly one 16-row tile and streams its whole slab once.
- **The instructions per weight.** b1-decode's $0 static count (triton 3.4.0, SASS of K25's K loop at the served
  constexprs; to be published with grouped-nf4-gemm's K38 design note) is **34.5 lane instructions per weight on
  sm_120**: select tree 11.6, logic, shift and address 14.6, shared 2.2, global loads 1.6, the rest under 1 each.
  2,787 experts a step x 4.72 M weights x 34.5 is about 454 G lane instructions. At 170 SMs x 128 lanes x 2.4-2.8 GHz
  that is an issue estimate of **7.5-8.7 ms**: above the 4.94 ms byte floor, toward the recorded 11.75 ms. It is an
  estimate (a static count at one issue per scheduler cycle), not a timing.

A plan moves neither term, only how well the loads and the decode overlap. The empty tiles (about 78 of the 136 slots)
return before K25's K loop, so compacting the grid removes only their dispatches. This lane prices that as a reported
arm.

## The instrument (`p155_bench.py`)

- **The box:** one RTX 5090 (sm_120); torch 2.8.0+cu128 and triton 3.4.0 (the image's); e4b at this registration's merge
  commit; grouped-nf4-gemm **v0.48.0** at `ef2faf3816b30ecfd0e2a69cb27fb049a7d56a59`, e4b CI's pin. K25's two files are
  byte-identical to v0.47.1's, the build P152 timed. The tripwire pins them: `nf4_smallm.py` sha256 `96176de8...` and
  `int4_b32.py` `773eff5d...`. It also asserts that the served plan is today's plan and that K23's lean glue is on (the
  call form below).
- **The call, exactly as the served route makes it** (`hot_residency._fused_over_stack`, NF4 store, 16 rows):
  - gate_up reads the 16 token rows through the tile table's order: `k25(x[16, 2048], gu_p, gu_a, row0, rows, grp,
    order, gather_div=8, lut="tree", dot_bf16=False, **plan)`;
  - down reads the sorted rows and stores into the caller's order: `k25(h[128, 768], dn_p, dn_a, row0, rows, grp, None,
    scatter=order, ...)`;
  - both run over `build_group_tiles_fused(ids, 128, 16, lean=True, sorted_ids=True)`, the 136-slot table.
- **The stores:** random NF4 bytes and fp32 absmax in the served layout, distinct for all 48 MoE layers. That is 16.3 GB,
  the arena's size, so each step's graph streams about 7.4 GB and nothing replays from L2. K25's time does not depend on
  weight values: its decode has no data-dependent branch.
- **The grid:** BLOCK_N {16, 32, 64, 128} x KC {64, 128, 256} x warps {2, 4, 8} x stages {2, 3, 4}, **108 plans**, today's
  among them. The codebook decode (`tree`) and the precision (TF32 on fp32 weights, the served class) are fixed.

### The routing

S0's real picks, the maintainer's ruling: rows 0-15 of P153's router replay, `bench/p153/router-replay/rr.json.pt`
(merged in #1844).
- **The file:** sha256 `3c59e72f947daa425e61f727401204099a1e782b902fb39687e04c46e338b14f`. S0's slice, `picks[:, 0:16]`,
  is an int16 tensor of 48 layers x 16 rows x 128 steps x top-8, and its little-endian bytes hash to `3020dbce...`.
- **The checks, stated before data.** All of them pass at registration (`tests/test_p155.py`), and the box repeats them
  before anything is timed:
  - the shape and dtype;
  - every id in [0, 128);
  - each row's 8 picks distinct;
  - S0's distinct total, **350,868**, equal to P153's record.
- **What it holds:** a mean of **57.107** routed experts per layer-step (40 to 82), against P154's recorded 58.07 (42 to
  81). No expert gets more than 16 rows, so the real tiles are exactly the routed experts.
- **The caveat:** these are bf16 picks, teacher-forced on the served token stream (argmax fidelity 0.941), over P153's
  prompts. They are not the served NF4 picks.
- **The fallback was not needed.** A routing rebuilt from P154's recorded counts would have stood in had a check failed.
  None did, so it is not shipped.

### The phases

1. **IDENTITY** (condition 2). Every grid plan runs step 0 on all 48 layers, both projections, and its outputs are
   compared with today's plan's (`torch.equal`).
   - A plan that fails to compile or launch on sm_120 is recorded with its error and excluded. That is not a fault.
   - A plan that moves a bit is excluded and reported as a contract finding against gnf4's contract.
2. **SELECTION** (condition 1). Per projection, each plan that kept today's bits is timed as a graph of that projection's
   48 calls, on the 8 selection steps 0, 8, ..., 56: two warm calls, a capture, then 10 timed replays.
   - A plan's **selection time** is the mean over those steps of its median replay.
   - The plans run in one seeded shuffle (seed 155). Today's plan also opens and closes each projection's sweep, as a
     drift probe that is reported only.
3. **CONFIRMATION** (conditions 1 and 4), on the **64 held-out steps 64-127**, routing the selection never saw. Each step
   is one paired rep over these arms:
   - **T**, today's plan;
   - **T2**, today's plan captured again: the self-pair;
   - **C1-C3**, the carried candidates;
   - **K**, today's plan over the compacted table: the real-tile prefix `t_*[:n_real]`. That is K25's own code over the
     same integers, so its time is the ceiling of compaction. Reported only.

   Each arm's graph holds both GEMMs of all 48 layers, one MoE step's K25 work. Each rep rotates both the capture order
   and the replay order by one (`p155_reduce.rotation`), so every arm takes every position equally often. The 10 replays
   interleave arm by arm. After the replays, every arm's outputs are compared with T's, bit for bit.

The bench records raw replay times only. `p155_reduce.py` derives the selection and the verdict from them.

## The rule (`p155_reduce.py`, fixed before data)

- **SELECTION** (fixed before data):
  - per projection, the plans that kept today's bits are ranked by selection time, ties broken by grid order;
  - the top **3** of each projection go forward, paired by rank: Ci = (gate_up's rank i, down's rank i), i = 1..3;
  - **3 candidates are carried.** Today's plan can be one of them.
- **VOID:**
  - the replay's sha256 or S0's hash is not the pinned one;
  - the grid, today's plan, the steps or the shuffle are not as registered;
  - the card is not an RTX 5090 at sm_120, or fewer than 48 layers were timed;
  - S0's distinct total is not 350,868;
  - a tile-table fault (the real tiles are not a prefix of one tile per expert);
  - a plan is missing from the selection, or today's plan did not run cleanly;
  - **the carried plans differ from the selection rule applied to the recorded replay times**;
  - the confirmation's arms, its 64 reps, their steps, rotations or replays are not as registered;
  - T2 is not bit-identical to T in every rep (the instrument is not deterministic);
  - **the self-pair median is outside 0.99-1.01.**
- **Excluded, not VOID:** a carried plan that moves a bit on any held-out step. That is a contract finding.
- **GO:** the winner is the eligible candidate with the largest median of T/C over the 64 reps (ties go to the better
  rank). GO if its **median - 2 SE >= 1.15**, with SE = sd / sqrt(64) of its per-rep ratios. The bar covers gate_up and
  down together, one graph holding both GEMMs.

  GO is decided on the confirmation read only. No selection number enters it.
- **NO_GO:** otherwise, including when no candidate is eligible.
- **Reported, never ruling:**
  - the compaction arm: its median, SE, identity and share of the headroom;
  - today's microbench milliseconds a step, against P152's recorded 11.748 ms;
  - the winner's selection ratio beside its confirmed ratio, so the winner's curse is visible;
  - each projection's top ten;
  - the errors, the identity failures, the drift and the walls.

## Predictions (graded in the read, never ruling)

| prediction | if HELD | if MISSED |
|---|---|---|
| **Fidelity:** today's both-GEMM microbench step is 9.5-14.0 ms (P152's served census recorded 11.748 ms) | the microbench is the served regime | the regime differs. A GO still opens only the served read, and a NO_GO says "not at the microbench's regime" |
| **Contract:** no plan that runs moves a bit | gnf4's contract holds on sm_120 across the grid | a contract finding to gnf4, with the plans and steps |
| **The headline:** the winner's confirmed median is 1.00-1.12, so **NO_GO**. P(GO) about 20 % | a plan alone is not the lever at W16 | above: GO, and the served read registers |
| **Condition 7:** a plan alone recovers at most **25 %** of the 6.81 ms, at most 1.7 ms of P152's 11.748 ms | most of the gap stays; see What follows | a plan recovers more than the issue estimate allows. The estimate, or the regime, is wrong |
| **The compaction ceiling, K/T, is 1.00-1.06** | compaction (K38) parks, as b1-decode's note says | K38's persistent kernel registers its own microbench lane |
| **The self-pair** is within 0.005 of 1 | the instrument resolves the bar easily | a noisier card. The 2-SE guard carries it |
| **The winner's curse:** its selection ratio exceeds its confirmation by 0-0.03 | selection was honest, with a small curse | a large curse (above), or none at all (below) |

The basis of the headline: a plan changes neither the bytes nor the 34.5 instructions per weight. K27's TF32 sweep at
OLMoE's B=16 shapes, the closest regime (about 2 rows per expert), ranked today's plan first of six, with the runner-up
3 % behind. Against the issue estimate, at most about 1.35-1.57x (11.75 / 8.7 to 11.75 / 7.5) is possible in
principle, and only with perfect overlap.

## What follows

- **GO:** a served FLIP read registers as its own lane, with its own bars, on the per-projection K25 plans for
  qwen3_moe (P143's pattern: the sweep, then the served read). It quotes no gain from this lane.
- **NO_GO with most of the 6.81 ms left:** the maintainer's ruling names (2), tile compaction (K38, b1-decode's), as the
  next lane, and this run prices it with K. If K reads under 1.05, compaction parks. The record then names the decode
  cost as the candidate for the maintainer's decision: fewer instructions per weight, where b1-decode's count puts the
  select tree at 11.6, nibble extraction at about 7.5 and address arithmetic at about 4.

## Before the box: the $0 A2000 shakedown

Condition 5. `p155_bench.py --shakedown --layers 4 --select-steps 2 --confirm-reps 8` runs on the A2000 (sm_86) under an
exclusive `gpu:a2000` claim, in a throwaway pytorch 2.8.0 container with grouped-nf4-gemm at the pin. `p155_reduce.py`
then reads its record, which must come back **SHAKEDOWN** with no structural fault.
- **What it shows:** the harness end to end; bitwise identity across the grid on sm_86 for every plan that compiles
  there; the prefix the compaction arm relies on; the routing checks on the box's path.
- **Before the box:** its record is committed as a sha-pinned test fixture.
- **What it cannot show:** sm_86 cannot see sm_120's plan availability or speed. Which plans compile and every timing
  are sm_120's. No A2000 timing seeds any band here.

## Budget and sequencing

- **One RTX 5090:**
  - hosts with a recorded power cap of at least 575 W (`vast_min_power_limit_w: 575`, `vast_power_limit_unknown:
    exclude`), read within the last 3 days (`vast_power_limit_max_age_days: 3`);
  - declared at $1.05/h, **guard 1.0 h**, at most 10 GB at <= $0.011/GB. Worst case 1.0 x $1.05 + 10 x $0.011 =
    **$1.16**, under the standing rule;
  - disk 60 GB (the box's floor is 20 GB free). Nothing is fetched from a hub.
- **The fit, e4b#1845's rule, with at least 25 % headroom:**
  - **Install, self-tests and premise:** P152's and P154's boxes took about 96 s each. The allowance is 450 s for the
    install plus 150 s for the self-tests and the premise: `INSTALL_TO_GATE_S` = 600 s.
  - **The bench:** K27's whole 5090 bench, about 52 compiled K25 variants with their timing, took 27.5 s. P155 compiles
    216 variants, times 108 plans x 2 projections x 8 steps, then 64 reps x 6 graphs: about 6-10 min estimated.
    `NEED_BENCH` = 1200 s. Its alarm can run to 2400 s if the deadline allows.
  - **The reserve:** `RESERVE` = 600 s, for the reduce, the fetch-back and the teardown.
  - **The runner's check, before the install:** `FIT` = 600 + 1200 + 600 = 2400 s, and `FIT_NEED` = 2400 x 1.25 =
    3000 s. Anything short of that is a refusal, rc 41.
  - **The static fit:** boot 300 s + 2400 s = 2700 s, and 2700 x 1.25 = 3375 s <= 3600 s (1.0 h). `tests/test_p155.py`
    asserts it.
- **Run ids** `p155-5090-1` to `-3`. Two consecutive HARNESS_ERROR, VOID or NOT_RUN runs stop the lane.
  - **Not a strike:** a pre-flight host stall; a power-floor refusal (rc 17); a fit refusal (rc 41), except that two
    consecutive fit refusals count as one strike.
  - **A strike:** a time-left skip (STOP-2, rc 40).
- **Sequencing:** the rental waits for three things: this registration's merge, the maintainer's final pass, and the
  shakedown's committed record. Then the shared pre-flight, and `--validate` on the exact manifest.

## Receipts

Fetched to the run directory's `p155/` and published under `bench/p155/receipts/<run>/` by the 2026-10-19 citation rules:
`bench_p155.json` (every replay time), `verdict_p155.json`, `summary.txt`, `forensics.txt`, `versions.txt`,
`gpu_phases.csv` and the logs, with `SHA256SUMS`. The box writes no instance id and no host name.
`RESULTS-p155.md` is written from those files.

## What this lane cannot say

- **Nothing about the served step.** It times the two K25 GEMMs alone, with no attention, router or glue between them.
  A GO opens the served read, and that read measures the step.
- **Not the served NF4 picks:** bf16 teacher-forced picks on the served stream, over P153's prompts.
- **Nothing that depends on weight values:** the bytes are random, and outputs are compared only plan against plan.
- **Nothing about other cards, families or row counts:** one RTX 5090, Qwen3-30B-A3B's shapes, 16 rows.

**Review:** loggetta (it knows K25's plans best), tc-campaign if it is busy, then the maintainer's pass.
