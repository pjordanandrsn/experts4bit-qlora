# P156: K39's decode against today's select tree at Qwen3-30B-A3B's W16 shapes, on S0's real router picks

**Status:** registered before any data. A microbench PREMISE for stream (B) of e4b#1313. A GO opens a served FLIP read
as its own registration and quotes no gain from this lane. It changes no default, no gate and no speed claim. The
maintainer reserved the lane number on 2026-10-19, after K39 read bitwise, and set its conditions; each is mapped below.

| condition | where |
|---|---|
| 1. one RTX 5090, its own box (no riding P155's) | [Budget](#budget-and-sequencing) |
| 2. P155's discipline: selection split from confirmation on held-out steps of the router replay; paired, rotated reps | [The phases](#the-phases) |
| 3. GO on the median minus 2 SE (of the median) | [The rule](#the-rule-p156_reducepy-fixed-before-data) |
| 4. a guard with 25 % headroom, from measured compile and bench walls | [Budget](#budget-and-sequencing) |
| 5. a $0 A2000 shakedown before the box | [The shakedown](#before-the-box-the-0-a2000-shakedown) |
| 6. the TIME prediction against the issue/occupancy model, and whether K39 moves the recorded 11.75 ms toward the floors | [Predictions](#predictions-graded-in-the-read-never-ruling) |
| 7. a microbench premise: a served flip is its own later read | Status, and [What follows](#what-follows) |
| later rulings: 17.0 is a triton 3.4.0 number (grouped-nf4-gemm#570); P155's occupancy observation folded in, M2 the least likely | [Why](#why), [Predictions](#predictions-graded-in-the-read-never-ruling) |

## Why

[P155](../p155/PREREG-p155.md) read **NO_GO** on 2026-10-19. Across K25's 144-plan grid, its best carried plan confirmed
at **1.0188x** (lower bound 1.0180, bar 1.15), 3.2 % of P154's **6.81 ms** of W16 headroom. Compacting the empty tiles
read 1.003x. So neither K25's plan nor its empty tiles hold the headroom. Two floors frame what is left (P155's "Why"):
- **the bytes:** P154's byte floor, **4.94 ms** for both GEMMs a step;
- **the instructions:** b1-decode's static count of K25's K loop, compiled with the served launch's own specialization
  (`bench/sass_count`, grouped-nf4-gemm#567), is **24.1 lane instructions per weight on sm_120**. At 170 SMs x 128
  lanes x 2.4-2.8 GHz, that is an issue estimate of **5.2-6.1 ms**.

The recorded time is **11.75 ms** (P152's census; P155's microbench read 12.075 ms for the same pair). K25's plan could
not move the instructions per weight; the plan grid only moved occupancy and loads in flight, and P155 found that is not
the lever. **K39 moves the instructions per weight at the same plan.**

**K39** (grouped-nf4-gemm#569) is K25's `lut="prmt"`: the NF4 nibbles decode through PTX byte-permute lookups instead
of the select tree. It is opt-in and **bit-identical to `lut="tree"`**: 394 of 394 cases on the A2000 at the PR's head,
with real routing, edges, determinism and a CUDA-graph capture. At the served plan:

| toolchain | target | `lut="tree"` | `lut="prmt"` | prmt / tree |
|---|---|---|---|---|
| **triton 3.4.0, ptxas 12.8.93** (the served stack's, and this lane's) | sm_120 | 24.12 per weight, 80 registers | **17.00**, 64 registers | **0.70** |
| | sm_86 | 25.12 | 17.44 | 0.69 |
| triton 3.8.0, ptxas 12.9.86 (not served) | sm_120 | 36.19 | 18.38 | 0.51 |

- **The basis is a 3.4.0 number.** Under triton 3.8.0 the select tree's loop gains about 50 % in register moves
  (grouped-nf4-gemm#570, a candidate: a static count, its time unread), and prmt barely moves. This lane runs, and its
  predictions hold, at triton 3.4.0 only.
- **Occupancy:** 64 registers a thread let 8 CTAs (32 of 48 warps) share an SM, against 6 (24) at 80: **67 %** theoretical
  occupancy against **50 %**, register-limited. K39 also halves the plan's shared memory (6,400 B against 14,592 B on
  sm_86, from the compiled metadata).

**The question:** how far does the decode alone, P (`lut="prmt"`) against T (`lut="tree"`), both at today's plan, move
the recorded time toward max(bytes, issue)?

## The instrument (`p156_bench.py`)

P155's instrument, with the decode as the moving argument. Everything not named here is P155's, unchanged.
- **The box:** one RTX 5090 (sm_120); torch 2.8.0+cu128 and triton 3.4.0 (the image's); e4b at this registration's merge
  commit; **grouped-nf4-gemm main at 68bbefc4 (#569's merge, after v0.49.0; K39 unreleased)**,
  `68bbefc4f43aaa83870903ac2c8788bab3ab1b89`. No release carries K39 and none is cut for a box: the box installs that
  commit, whose package metadata still reads 0.49.0, so it never runs the 0.49.0 wheel. The tripwire pins three files:
  - `nf4_smallm.py` sha256 `ee46969a...` (K25 with K39's decode mode);
  - `int4_b32.py` `773eff5d...` (P155's);
  - `nf4_grouped.py` `a7090915...` (the source of K39's table words, `_bw_table_words()`).

  It also asserts that the served plan is today's plan with `lut="tree"`, that K23's lean glue is on, and that K39's
  decode is the kernel's LUT mode 4.
- **The call, exactly as the served route makes it** (P155's): gate_up through the tile table's order with
  `gather_div=8`, down into the caller's order with `scatter=order`, over the 136-slot lean table, `dot_bf16=False`.
  Only `lut` differs between T and P.
- **The stores:** P155's: random NF4 bytes and fp32 absmax in the served layout for all 48 MoE layers, 16.3 GB, so
  nothing replays from L2. Both decodes read the same bytes.
- **The routing:** P155's, re-checked on the box before anything is timed: rows 0-15 of P153's router replay,
  `bench/p153/router-replay/rr.json.pt` (sha256 `3c59e72f...`); S0's slice hashes to `3020dbce...`; S0's distinct
  total is 350,868.
- **Per decode, reported:** the registers, spills and shared memory Triton compiled T and P to at today's plan, read
  from its own cache.

### The phases

1. **IDENTITY.** Every grid plan runs step 0 at `lut="prmt"` on all 48 layers and both projections, and its outputs are
   compared with T's (`torch.equal`). A plan that fails to compile or launch is recorded with its error and excluded.
   A plan that moves a bit is excluded and reported as a contract finding. If P itself fails here, the read is VOID.
2. **THE SWEEP** (reported only). Per projection, each prmt plan that kept T's bits is timed as a graph of its 48 calls
   on the 8 selection steps 0, 8, ..., 56: two warm calls, a capture, then 10 timed replays, in one seeded shuffle (seed
   156). P opens and closes each projection's sweep as a drift probe. The top 3 of each projection, **today's plan
   excluded** (it is P), go forward, paired by rank as C1-C3. They price decode x plan for a later lane. They never
   enter the verdict.
3. **CONFIRMATION**, on the **64 held-out steps 64-127**. Each step is one paired rep over these arms:
   - **T**, the select tree at today's plan;
   - **T2**, T captured again: the self-pair;
   - **P**, K39 at today's plan: **the registered question, fixed before data**;
   - **C1-C3**, the sweep's carried prmt plans (reported).

   Each arm's graph holds both GEMMs of all 48 layers, one MoE step's K25 work. Each rep rotates both the capture order
   and the replay order by one, and the 10 replays interleave arm by arm. After the replays, every arm's outputs are
   compared with T's, bit for bit.

P is pre-declared, so it carries no winner's curse; the sweep's selection-to-confirmation split still governs C1-C3.
The bench records raw replay times only. `p156_reduce.py` derives the sweep and the verdict from them.

### Reported only: Nsight Compute

After the verdict is written, the runner captures T's and P's gate_up and down calls at today's plan on layer 0 of
step 64's routing (`p156_bench.py --ncu-target`; `--launch-skip 12 --launch-count 4`), with P155's sections and its
probe first. Refused counters (P155's host refused them) are recorded and the lane moves on. It never fails the lane.

## The rule (`p156_reduce.py`, fixed before data)

- **VOID:** P155's list, adapted:
  - the replay's sha256 or S0's hash is not the pinned one, or S0's distinct total is not 350,868;
  - the grid, today's plan, **the decodes (T tree, P and the sweep prmt)**, the steps or the shuffle are not as
    registered;
  - the card is not an RTX 5090 at sm_120, or fewer than 48 layers were timed;
  - a tile-table fault (the real tiles are not a prefix of one tile per expert);
  - a plan is missing from the sweep, or P did not run cleanly or bit-identically on the identity step;
  - **the carried plans differ from the sweep rule applied to the recorded replay times**;
  - the confirmation's arms, its 64 reps, their steps, rotations or replays are not as registered;
  - T2 is not bit-identical to T in every rep;
  - **the self-pair median is outside 0.99-1.01.**
- **NO_GO with a contract finding:** P moves a bit on any held-out step. K39's bitwise tier would then fail on sm_120,
  a finding against grouped-nf4-gemm, and no timing is ruled on.
- **GO:** P's **median of T/P over the 64 reps, minus 2 SE, >= 1.05**. SE is the standard error of the MEDIAN, P155's
  rule (`se_median`): the larger of sqrt(pi / 2) x sd / sqrt(64) and a 2,000-resample bootstrap from one generator
  seeded 156. The bar covers gate_up and down together, one graph holding both GEMMs.
  - **The bar:** K25's pair is 11.75 ms of a 16.5 ms step, so a microbench 1.05 is about a 1.035 step, the size of the
    served flips P143 and P153 read (1.10 would be about 1.069, and P155's 1.15 about 1.102).
  - GO is decided on P's confirmation alone. No sweep number enters it.
- **NO_GO:** otherwise.
- **Reported, never ruling:**
  - P's share of the 6.81 ms headroom, and its milliseconds at P152's scale against the 4.94 ms byte floor;
  - which of the models below contain P's median;
  - C1-C3: each one's T/C and its ratio to P, and the sweep's best selection-over-P beside its confirmation (the
    curse);
  - T's microbench milliseconds a step against P152's 11.748 ms;
  - each projection's sweep top ten, the errors, the identity failures, the drift, the compiled metadata and the walls.

## Predictions (graded in the read, never ruling)

The time model, from the two floors above and a 17.00 / 24.12 instruction ratio at the same clocks (2.4-2.8 GHz). The
issue estimate is 5.2-6.1 ms for T and **3.7-4.3 ms for P**, which puts P's issue under the 4.94 ms byte floor:

| model | T/P | why |
|---|---|---|
| **M1, overlap**: time = max(bytes, issue) + a fixed residual (11.75 - T's issue) | **1.02-1.11** | P's issue drops under the bytes, so the bytes bind and the residual stays |
| **M3, serial**: the issue adds to everything else | **1.15-1.18** | every saved instruction is saved time |
| **M2, overlap + occupancy**: M1, with the residual hidden in proportion to 50 -> 67 % occupancy | 1.19-1.28 | **the least likely** (below) |

**Why M2 is now the least likely.** P155 changed occupancy alone across its grid, and it did not pay:
- the 67 %, 62-register gate_up plan (32, 64, 4, 1) ran **8.404 ms**, against today's **8.317 ms** at 50 %;
- down's 67 % plans were the fastest, but by only 1.2 %;
- across the grid, time ranked against theoretical occupancy (Spearman about -0.58) only because the heavy,
  low-occupancy plans are slow; among the fast plans occupancy did not decide (serve-throughput's reading of P155).

So K39's 67 % is not expected to hide the residual on its own; what K39 moves that P155 could not is the instructions
per weight. M2 stays registered as the alternative a MISS above the headline would point to.

| prediction | if HELD | if MISSED |
|---|---|---|
| **Fidelity:** T's both-GEMM microbench step is 9.5-14.0 ms (P155 read 12.075 ms; P152's census 11.748) | the microbench is the served regime | the regime differs. A GO still opens only the served read |
| **The headline:** P's confirmed median T/P is **1.05-1.18**, from M1's upper part to M3; **P(GO) about 65 %** | the decode is a lever of the size the instruction count implies | below: the bytes and the residual bind (M1's lower part, or less): K39 parks with no speed claim. Above: M2's occupancy did hide the residual, against P155's evidence |
| **Contract:** P and every swept prmt plan that runs keep T's bits | K39's bitwise tier holds on sm_120 across the grid | a contract finding to grouped-nf4-gemm, with the plans and steps |
| **Registers:** T compiles to 80 registers a thread and P to 64, no spills (50 % and 67 % theoretical occupancy) | b1-decode's offline compile is the served one | the occupancy argument is re-based on the box's numbers |
| **The self-pair** is within 0.005 of 1 (P155: 0.9999) | the instrument resolves the bar easily | a noisier card. The 2-SE guard carries it |
| **The sweep's curse:** its best plan's selection-over-P exceeds its confirmed ratio to P by 0-0.03 (P155: 1.045 against 1.019, 0.027) | the sweep was honest, with a small curse | a large curse (above), or none at all (below) |

**Toward the floors** (reported): at M1's midpoint, 1.065, P would read about 11.03 ms at P152's scale, 10.5 % of the
6.81 ms of headroom; at M3's, 1.165, about 10.08 ms and 24.4 %. The headline's band, 1.05-1.18, is 8.2-26.3 % of it.

## What follows

- **GO:** a served FLIP read registers as its own lane, with its own bars, on `lut="prmt"` as qwen3_moe's K25 decode at
  W16 (P143's pattern: the microbench, then the served read). It quotes no gain from this lane. The
  per-plan question C1-C3 price stays a later lane's, if the sweep's best clears P by more than the curse.
- **NO_GO:** the decode is not the lever at W16 either. The record names what the Nsight capture (if the counters are
  available), the compiled metadata and the sweep point to, beside the access pattern and the latency the floors do not
  explain. K39 stays opt-in with no speed claim. The maintainer decides the next lane.

## Before the box: the $0 A2000 shakedown

Condition 5, **done on 2026-10-19 (22:43-22:47Z), before this registration and before any rental**, under an exclusive
`gpu:a2000` claim: `p156_bench.py --shakedown --layers 4 --select-steps 2 --confirm-reps 8` on the lab's A2000 (sm_86),
torch 2.8.0+cu128, triton 3.4.0. Its record is [`shakedown/shake_p156.json`](shakedown/shake_p156.json) (sha256
`b7fd7320...`, pinned in `tests/test_p156.py`).
- **What it ran:** grouped-nf4-gemm's `kernel/` at #569's head (193b9a9), the tree the merge commit carries byte for
  byte (`git diff 193b9a9 68bbefc4 -- kernel/` is empty), so it is the pin's kernel. This kit's reducer, byte for byte.
  This kit's bench, except its self-test, which gained four wiring checks after the run (the decode each arm and the
  sweep pass); the timed code is the same.
- **The harness, end to end:** both self-tests (the bench's 6 cases then, 7 now; the reducer's 36, with its mutants),
  the bench (rc 0 in 223 s), and the reducer, which reads **SHAKEDOWN** with every structural check passing. A
  shakedown is never graded.
- **The contract on sm_86:** **all 144 prmt plans compiled and ran in each projection, and every one kept T's bits** on
  all 4 layers; every carried plan and P kept them on all 8 held-out reps. P155's tree sweep lost 18 a projection to
  `OutOfResources` on the same card; K39's smaller shared memory (6,400 B against 14,592 B at today's plan) is the
  difference.
- **The compiled metadata:** T 71 registers and P 64, no spills, on sm_86. The registers prediction is graded on the box.
- **The walls:** compiling and first-calling the 288 prmt variants took 133.3 s of host CPU; each selection graph cost
  0.131 s of host time beyond its replays, and each confirmation graph 0.165 s (see Budget).
- **What it cannot show:** sm_86 cannot see sm_120's plan availability or speed. Its replay times are the A2000's and
  seed nothing: the predictions above were sent to the maintainer before the shakedown ran.

## Budget and sequencing

- **One RTX 5090**, its own box:
  - hosts with a recorded power cap of at least 575 W (`vast_min_power_limit_w: 575`, `vast_power_limit_unknown:
    exclude`), read within the last 3 days;
  - declared at $1.05/h, **guard 1.25 h**, at most 10 GB at <= $0.011/GB. Worst case 1.25 x $1.05 + 10 x $0.011 =
    **$1.42**, under the standing rule;
  - disk 60 GB (the box's floor is 20 GB free). Nothing is fetched from a hub.
- **The fit, e4b#1845's rule, with at least 25 % headroom:**
  - **Install, self-tests and premise:** P155's, `INSTALL_TO_GATE_S` = 600 s.
  - **The bench:** `NEED_BENCH` = **1600 s**, from two measurements, the larger taken, x1.5:
    - P155's box bench, the same shape (both projections over the 144-plan grid on 8 selection steps, then 64 reps of
      six arms), took **896 s** on an RTX 5090 with 126 plans running a projection: 169 s of compiles, 2,016 selection
      graphs and 384 confirmation graphs. Scaled to all 144 plans running, about 1,024 s;
    - this lane's shakedown walls through P155's formula (its 133.3 s of compiles; 2,336 selection graphs at 0.131 s
      of host plus 12 replays of P152's 11.748 ms pair; 384 confirmation graphs at 0.165 s likewise; 20 s of setup):
      about 907 s.

    So about 1,024 s; x1.5 = 1,536 s, rounded up. Its alarm can run to 3600 s if the deadline allows.
  - **Nsight Compute:** `NEED_NCU` = 300 s, P155's.
  - **The reserve:** `RESERVE` = 600 s.
  - **The runner's check, before the install:** `FIT` = 600 + 1600 + 300 + 600 = 3100 s, and `FIT_NEED` = 3100 x 1.25 =
    3875 s. Anything short of that is a refusal, rc 41.
  - **The static fit,** (boot + fit) x 1.25 <= guard (e4b#1855's form): boot 300 s + 3100 s = 3400 s, and
    3400 x 1.25 = 4250 s <= 4500 s (1.25 h). `tests/test_p156.py` asserts it.
- **Run ids** `p156-5090-1` to `-3`. Two consecutive HARNESS_ERROR, VOID or NOT_RUN runs stop the lane. P155's strike
  rules hold: a pre-flight host stall, a power-floor refusal (rc 17) or a fit refusal (rc 41) is not a strike (two
  consecutive fit refusals count as one); a time-left skip of the bench (STOP-2, rc 40) is.
- **Sequencing:** #569 merged (68bbefc4) and the shakedown's record is committed. The rental waits for this
  registration's merge after review and the maintainer's pass, then the shared pre-flight, and `--validate` on the exact
  manifest.

## Receipts

Fetched to the run directory's `p156/` and published under `bench/p156/receipts/<run>/` by the 2026-10-19 citation rules:
`bench_p156.json` (every replay time and both decodes' compiled metadata), `verdict_p156.json`, `ncu_p156.csv` when the
counters were available, `summary.txt`, `forensics.txt`, `versions.txt`, `gpu_phases.csv` and the logs, with
`SHA256SUMS`. The box writes no instance id and no host name. `RESULTS-p156.md` is written from those files.

## What this lane cannot say

- **Nothing about the served step.** It times the two K25 GEMMs alone. A GO opens the served read, and that read
  measures the step.
- **Nothing at another toolchain.** Its counts and its times are triton 3.4.0's; a triton bump is its own read
  (grouped-nf4-gemm#570).
- **Not the served NF4 picks:** bf16 teacher-forced picks on the served stream, over P153's prompts (P155's caveat).
- **Nothing that depends on weight values:** the bytes are random, and outputs are compared only decode against decode.
- **Nothing about other cards, families or row counts:** one RTX 5090, Qwen3-30B-A3B's shapes, 16 rows.

**Review:** loggetta (it reviewed K39 and the counter), serve-throughput (P155's harness) if it is free, then the
maintainer's pass.
