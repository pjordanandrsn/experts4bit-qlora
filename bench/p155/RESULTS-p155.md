# P155 results: K25's plan is not the W16 lever. The best confirmed plan is 1.019x (NO_GO), and compaction is 1.003x (e4b#1313, stream (B))

**Status:** closed. **NO_GO** on the one run. The microbench is a premise, so nothing changes: no default, no gate and no
speed claim.

[P154](../p154/RESULTS-p154.md) priced 6.81 ms of headroom between the W16 grouped NF4 expert GEMMs (K25) and their byte
floor. [The registration](PREREG-p155.md) asked whether K25's plan, the one knob its contract says moves no output bit,
recovers it at Qwen3-30B-A3B's shapes on S0's real router picks:
- **The sweep:** 144 plans, called exactly as the served route calls K25.
- **Selection:** 8 steps.
- **Confirmation:** 64 held-out steps, in paired rotated reps.
- **GO:** median - 2 SE(median) >= 1.15 on both GEMMs.

| run | what | cost | verdict |
|---|---|---|---|
| [`p155-5090-1`](receipts/p155-5090-1/README.md) | identity, selection and confirmation over the grid; compaction and the compiled metadata reported | $0.274 | **NO_GO** |

That is $0.274 on one RTX 5090 launch, inside the registered worst case of $1.69. The $0 A2000 shakedown had passed first
([`shakedown/`](shakedown/shake_p155.json)).
- **The public receipt copy** is under [`receipts/`](receipts/), with a README and `SHA256SUMS`. The launcher's receipt
  and ledger row are in the private receipt store.
- **The re-derivation:** the maintainer re-derived `verdict_p155.json` byte for byte from the store with the registered
  reducer (self-test 30/30). It also re-derives byte for byte from the public `bench_p155.json` on another host.

## The read (`p155-5090-1`)

- **The box:** one RTX 5090, 575 W read on the box. e4b `4d3fd170`; grouped-nf4-gemm v0.48.0 `ef2faf38`, whose K25 files
  are v0.47.1's; triton 3.4.0. It is the same card as P152's and P154's reads (the same GPU UUID).
- **The instrument:** the self-pair read **0.9999** (SE 0.0001, range 0.9988-1.0019 over 64 reps), and the drift probe
  read 1.0001 for gate_up and 0.9992 for down. The bar was resolved with room to spare.
- **The contract on sm_120:**
  - 126 of the 144 plans ran in each projection, and **every one kept today's bits** on all 48 layers.
  - Every carried plan, and the compaction arm, kept them on all 64 held-out reps.
  - The other 18 plans per projection refused with `OutOfResources`, needing more than sm_120's 101,376 B of shared
    memory per block: BLOCK_N 128 with KC 256, and the 4-stage plans at 64 x 256 and 128 x 128. They are the same 18
    the A2000 refused.
- **The confirmation** (T/arm, median over 64 paired reps):

| arm | gate_up plan | down plan | median | SE(median) | lower bound |
|---|---|---|---|---|---|
| **C1** | 32/128/4/2 | 64/64/8/1 | **1.0188** | 0.0004 | **1.0180** |
| C2 | 32/128/4/1 | 32/64/4/1 | 1.0098 | 0.0005 | 1.0089 |
| C3 | 32/128/4/3 | 16/64/2/1 | 0.9954 | 0.0004 | 0.9946 |

  Plans read BLOCK_N/KC/warps/stages; today's is 32/64/4/3 in both.
- **Selection** (each plan's mean of 8 selection-step medians, ms per 48-call graph):

| projection | today's | best plan | best | ratio |
|---|---|---|---|---|
| gate_up | 8.3165 | 32/128/4/2 | 7.8511 | 1.059 |
| down | 3.3716 | 64/64/8/1 | 3.3309 | 1.012 |

  On both GEMMs together, the winner's selection ratio was 1.045 and its confirmation 1.019. That shrink of 0.026 is the
  winner's curse that the split was registered to remove.

## Reported, never ruling

- **Compaction.** Today's plan over the real-tile prefix (`t_*[:n_real]`, no empty programs) reads **1.0032**, which is
  0.6 % of the headroom. Compaction's ceiling is negligible, and K38 parks.
- **Fidelity.** Today's plan ran **12.075 ms** a step for both GEMMs over 48 layers. P152's served census recorded
  11.748 ms (1.028), so the microbench is the served regime.
- **Compiled metadata.** Today's plan is **80 registers, 0 spills**, 14,592 B shared, in both projections. That is 6 CTAs
  an SM, **50 % theoretical occupancy, register-limited**, as b1-decode's offline compile of the served specialization
  said.
- **Occupancy against time across the grid** (the theoretical occupancy implied by each plan's compiled registers and
  shared memory, at 64K registers, 48 warps and 100 KB of shared memory an SM):
  - Over all 126 plans, time ranks against occupancy with a Spearman correlation of -0.58 in both projections. That comes
    from the heavy plans: at 4-8 % theoretical occupancy, the median plan runs 1.6-5x slower than today's.
  - **Among the fast plans, occupancy does not decide.** gate_up's best plans are KC 128 at 33-42 % theoretical occupancy
    (90 and 80 registers). All six of its 67 % plans are slower than today's (T/plan 0.933-0.990); the best,
    32/64/4/1 (62 registers), runs at 8.404 ms against today's 8.317 at 50 %.
  - down's three fastest plans are 67 % (64/64/8/1 at 1.012, 32/64/4/1 at 1.006, 16/64/2/1 at 1.001), the best by
    1.2 %. Its other two 67 % plans (16/64/4/1 and 16/64/4/2) run about a quarter slower (T/plan 0.812 and 0.794).
  - **Occupancy is never varied alone here.** Every 67 % plan gets there with fewer stages or a narrower tile. gate_up's
    best 67 % plan is today's tile with the 3-stage pipeline cut to 1, so extra warps replace software pipelining as the
    latency hider rather than adding to it. And this is theoretical occupancy: achieved occupancy is unread.
- **Nsight Compute.** `ncu` was present, and the rented host refused the GPU counters (`ERR_NVGPUCTRPERM`). The probe
  recorded that and nothing was captured, so achieved occupancy and stall reasons are unread.
- **Walls:**
  - the bench took 895.6 s, against the 2450 s allowance: selection 782.0 s, confirmation 110.2 s;
  - compiling and first-calling the 252 variants took 169.2 s, against 145.3 s on the A2000 shakedown's host (1.16x).

## Predictions (graded, never ruling)

| prediction | read | grade |
|---|---|---|
| today's both-GEMM microbench step 9.5-14.0 ms | 12.075 ms | HELD |
| no plan that runs moves a bit | 126/126 per projection; every carried plan and K on all 64 reps | HELD |
| the winner's confirmed median 1.02-1.15 (NO_GO) | 1.0188 | **MISSED** (low) |
| a plan recovers at most 25 % of the 6.81 ms | 3.2 % (0.22 ms of P152's 11.748 ms) | HELD |
| compaction's ceiling 1.00-1.06 | 1.0032 | HELD |
| self-pair within 0.005 of 1 | 0.9999 | HELD |
| the winner's selection ratio exceeds its confirmation by 0-0.03 | 1.045 vs 1.019 (0.026) | HELD |
| today's plan 80 registers, no spills (50 % theoretical) | 80, 0 | HELD |
| Nsight: achieved occupancy <= 50 %, long-scoreboard the largest stall | counters refused | UNREAD |

The headline missed **low**: a plan is even less of a lever than predicted. That falls on the "no lever at all" side of
the band, under 1.02, which the PREREG named as the miss below.

## What it means for stream (B)

- **K25's plan and its grid are not where the W16 headroom lives.** The best confirmed plan is 1.9 % faster, recovering
  1.8 % of the two GEMMs' time (3.2 % of the 6.81 ms), and compaction 0.3 %.
- **No plan in this grid with higher theoretical occupancy was faster on gate_up; on down the best was 1.2 % faster.**
  Occupancy is not varied alone here (the 67 % plans also change stages or tile width), and achieved occupancy is
  unread, so this record does not say whether latency is the gap.
- **The candidate left is the decode cost:** fewer instructions per weight at the same plan. That is K39 (b1-decode's;
  24.1 to 17.0 lane instructions per weight under triton 3.4.0, bitwise against K25), and its own lane, **P156**, times it
  as registered. P155 could not move that variable.
- **Unnamed:** most of P154's 6.81 ms headroom over the byte floor (6.6 ms after the best plan; 6.05-7.07 ms over
  P154's floor range). Rented hosts refuse Nsight's counters, so naming it with stall reasons needs a host that allows
  them, or another instrument.

## What this lane cannot say

- **Nothing about the served step.** It times the two K25 GEMMs alone, with no attention, router or glue between them.
- **Not the served NF4 picks.** These are bf16 teacher-forced picks on the served stream (argmax fidelity 0.941), over
  P153's prompts.
- **Nothing about other triton versions.** The served stack pins 3.4.0, and a newer triton compiles K25 to about 50 % more
  decode instructions per weight. Any triton bump on the served path needs its own read.
- **Nothing about other cards, families or row counts:** one RTX 5090, Qwen3-30B-A3B's shapes, 16 rows.

## Reproduce

From a checkout of e4b at `4d3fd170`, with this lane's receipt copy:

```
python bench/p155/p155_reduce.py --self-test
python bench/p155/p155_reduce.py --bench bench/p155/receipts/p155-5090-1/bench_p155.json --out verdict_p155.json
```

The output equals `receipts/p155-5090-1/verdict_p155.json` byte for byte.
