# P161: K25 at plan c1 in the served step, under K39's decode, as Qwen3-30B-A3B's batched-decode expert GEMM (e4b#1313)

**Status:** registered before any data. A served read of P160's microbench GO, by P158's served form. One non-consumer
review, then the maintainer's final pass. $0 until the read. e4b#1877 (the `E4B_K25_PLAN` knob) merged at `f0dab09a`
and is in no release as of 2026-10-20 (the latest is v0.56.0); the box installs e4b at this registration's merge, which carries the knob.

## Why

- **The premise.** [P160](../p160/RESULTS-p160.md) read plan c1 GO in a CUDA-graph microbench on an RTX 5090
  (`p160-5090-2`): the two K25 expert GEMMs of a W16 step on a router-replay stream ran **1.6598x** faster at c1 than at
  today's plan, both under K39's decode (`lut="prmt"`, qwen3_moe's served default since P158). That is the median of
  126 paired reps (SE 0.0015, lower bound 1.6568, clock-adjusted **1.6525**), with every output bit-identical.
  - That is a microbench premise, never a gain. It times the two GEMMs alone: 10.348 ms at today's plan, 6.231 ms at c1.
- **The plan.** c1 is P156's first carried pair: gate_up at block_n 64, kc 256, 4 warps, 2 stages; down at 128, 128, 8,
  3 (today's plan is 32, 64, 4, 3 for both). It changes tiling and staging only, never the codebook values or the
  reduction order along K, so K25 at c1 is bit-identical to today's plan by construction (BITWISE tier).
- **The knob.** e4b#1877 adds `E4B_K25_PLAN` = `today` (the default, also when unset) | `c1`. It is resolved once at the
  build (`info["k25_plan"]`), it refuses `c1` under any decode but `prmt` and where K25 does not run, and
  `hot_residency.K25_PLAN_SEEN` counts every K25 call by plan, beside `K25_LUT_SEEN` by decode (eager calls and graph
  captures, never a replay). Its served-route contract, `tests/test_k25_lut_gpu.py`, checks a B=16 step bit-equal under
  c1, prmt and the tree, eager and captured, and T == 1 bit-equal under c1.
- **The projection** (a prediction, never a bar). P158's S1 arms ran W16 at 1067.03 tok/s (the median of four), a
  **14.995 ms** step on today's default server; its S0 arms (the tree) at 976.07 tok/s, a 16.392 ms step.
  - **The K25 pair in that step:** P152's census put the tree's pair at 11.748 ms. P158's step saving (1.397 ms) puts
    prmt's served pair at **10.351 ms** (a served ratio of 1.1350). P160's microbench timed the same pair at 10.348 ms.
  - **At 1:1 carry**, c1 at P160's median takes 10.351 x (1 - 1 / 1.6598) = 4.115 ms off the step: g16 = 14.995 /
    10.880 = **1.378** (1.375 at the clock-adjusted bound).
  - **At P158's carry**, where the served saving was **0.797** of the 1:1 projection (1.397 ms of 1.754 ms): g16 =
    **1.280** (1.277 at the clock-adjusted bound).
- **The bar** is on W16, the batched decode where K25 runs. W1 is a no-regression guard: at T == 1 the expert path is
  the singleton bandwidth GEMV, not K25, so W1 should not move.

## What runs (`p161_run.sh`)

P158's served read by counted edits, with the plan knob in place of the decode's.
- **Engine and arms:** the shipped default server, as P158 ran it: graphs on, `max_seqs` 16, 4096 tokens per sequence,
  the fusions and the bandwidth GEMV at their auto defaults, the attention split cap at its qwen3_moe default (16), the
  int8 lm_head at its qwen3_moe default (**on**, P153's FLIP), the dense attention GEMV off, K25 at its default (auto),
  and **K25's decode at its qwen3_moe default (`prmt`, P158's FLIP), unset in both arms**. The arms differ only in the
  plan: **S0** = `E4B_K25_PLAN=today` (named, not unset), **S1** = `c1`. The box refuses an arm with `E4B_K25_LUT`,
  K25's mode, the int8 head, the split cap, the dense attention GEMV, a fusion knob or a decode GEMV knob set.
- **A warm-up process first** (`S0w`), as P158 ran it: integrity-checked like every arm, VOID when missing, its speed
  reported only, its tokens in the null.
- **Speed:** eight arms, ABBA twice (S0a S1a S1b S0b S0c S1c S1d S0d), each a fresh process through P109's W16 and W1 at
  **five timed reps**.
- **Telemetry:** P158's, unchanged: nvidia-smi before and after every timed block, and the 500 ms clock sampler during
  each, a process of its own.
- **No SANE phase.** The arms are bitwise-identical by contract, so the token-identity gate (below) replaces a quality
  read.
- **The plan's second witness, a shape spy.** Before the build, the box wraps `nf4_smallm.gemm_nf4_grouped_smallm` (with
  `functools.wraps`, since the route reads K25's signature to decide the lean glue) and counts every kernel call by the
  block shape and decode it receives (`"<block_n>/<kc>/<warps>/<stages>/<lut>"`), passing the call through unchanged.
  The plan record says which plan the route chose; the spy says what the kernel was handed. Python runs it on eager
  calls and graph captures only, never on a replay, like the records.
- **The premise, on this card, before anything is fetched:** P136's 19 premise tests (P158's files at P158's pins),
  K39's compiled contract (grouped-nf4-gemm's `kernel/test_nf4_smallm_prmt_gpu.py` at v0.50.0, 23 tests, P158's file)
  and e4b's served-route contract with c1's cases (`tests/test_k25_lut_gpu.py` frozen at `dd535cd6`, the first main
  with them, staged as `premise/test_k25_lut_gpu.py`, 20 tests). It must read **62 passed**, or the box refuses with
  rc 25.
- **Order:** P158's: the power floor and the guard fit, before the install; install and tripwire; reducer and box
  self-tests; premise; fetch; NF4 arena bake; prompts; nine processes, with the early gates after S0w and S1a; reduce.
- **Pins:**
  - e4b: the clean checkout the driver ships, which must contain e4b#1877's merge.
  - grouped-nf4-gemm: **v0.50.0** at `41ef086e` by sha, as P158. The tripwire checks K25's kernel file by sha256
    (`ee46969a`), the decode resolving to qwen3_moe's default (`env` unset, `prmt`, `default-allowlisted`), the plan
    table (`today` empty, `c1` the pair above), each plan resolving as named, `c1` refused under the tree, and the int8
    head's and the split cap's qwen3_moe defaults.
  - The rest: torch 2.8.0+cu128, triton 3.4.0, transformers 5.17.0; Qwen/Qwen3-30B-A3B @ `ad44e777`.

## The build's engagement counts, derived before data and confirmed on the code

P158's closed forms, carried: the residual licence's two probes, then two warm-ups and one capture, at row counts 2, 4,
8, 16 for K25 and 1 for the bandwidth GEMV:

      K25 calls:      48 MoE layers x 4 row counts (2, 4, 8, 16) x 5 forwards = 960
      kernel calls:   960 K25 calls x 2 (gate_up, down) = 1920
      bandwidth GEMV: 48 MoE layers x 1 row count  (1)            x 5 forwards x 2 calls (gate_up, down) = 480

P158's 5090 read confirmed both: K25's record read 960 and `bw_prmt32` 480 in every one of its nine processes, and
K25's record still read 960 after the runs (every timed step replays a graph, which no record counts).
- **Each witness, exactly, after the build:**
  - the decode record `{"prmt|le256": 960}` in both arms;
  - the plan record `{"<arm's plan>|le256": 960}`: `today` in S0, `c1` in S1;
  - the spy: S0 `{"32/64/4/3/prmt": 1920}` (both projections at today's shape); S1 `{"64/256/4/2/prmt": 960,
    "128/128/8/3/prmt": 960}` (gate_up and down each at its own c1 shape);
  - `bw_prmt32` = 480.
- **Two kernel calls per K25 call** is read from the code the box installs: the route calls K25 once per projection,
  gate_up with `_k25_gu_kw` and down with `_k25_dn_kw`.
- **Confirmed by running, as far as the lab's card can** ([`layercut/`](layercut/README.md), $0, the RTX A2000, before
  any rental): P158's 2-layer cut of Qwen3-30B-A3B, baked to NF4 as the box bakes it, built on today's default server
  once per plan, with the box's own spy (`p161_box.install_shape_spy`, imported, not copied). As in P158, the A2000
  confirms the **licence half** (decode graphs need sm_89+): 8 K25 calls per layer under the arm's plan and the default
  decode, and 16 kernel calls per layer at the arm's shapes.
- **Early gates on the box,** as P158's: after S0w (rc 43) and S1a (rc 44), the runner stops unless the decode and the
  plan resolved as registered and every build witness above reads exactly its closed form. Each is an instrument
  refusal, not a strike; the same gate firing on two consecutive runs counts as one strike. The reducer's VOID on the
  same counts stays as the backstop.
- After the runs each record holds the arm's key(s) only; the plan and decode records count the same calls, the spy
  twice that; and K25's total is the same in every process (the arms run the same prompts and lengths).

## The rule (`p161_reduce.py`, fixed before data)

The first rung that applies decides. P158's rule, carried, with the plan's witnesses beside the decode's.

1. **VOID.** Any of:
   - a record is missing or not ok, the warm-up process's included;
   - any of P153's checks fails on any record, carried as P158 carried them;
   - the dense attention GEMV is not off, or the int8 head is not at its qwen3_moe default in every record;
   - **K25's decode:** `info["k25_lut"]` is not `{"env": null, "lut": "prmt", "why": null, "source":
     "default-allowlisted"}`; the decode record after the build is not exactly `{"prmt|le256": 960}`, or held anything
     before it; after the runs it holds another key;
   - **K25's plan:** `info["k25_plan"]` is not `{"env": <arm's plan>, "plan": <arm's plan>, "why": null}`; the plan
     record after the build is not exactly `{"<arm's plan>|le256": 960}`, or held anything before it; after the runs it
     holds another plan;
   - **the spy:** after the build it is not exactly the arm's shapes above, or held anything before it; after the runs
     it holds another shape or fewer than the build's count at one;
   - **the witnesses disagree:** after the runs the plan record's total differs from the decode record's, or the spy's
     is not twice the plan's; or K25's total differs across the nine processes;
   - **the bandwidth GEMV's build tally** is not exactly `bw_prmt32` = 480;
   - **power:** any timed block's power limit or enforced power limit, read before or after it, below **575 W** or
     unreadable;
   - the arms differ in prompts or lengths, or an arm is not at five reps; a void slope;
   - **THE NULL** (checked before any S1 token is read): any S0 process (S0b, S0c, S0d and the warm-up) decodes
     different tokens from S0a on any row, workload or length, or any S0 arm's timed reps digest differently. That
     reads **VOID, "instrument: today's plan not token-deterministic across processes"**, never a finding against c1.
2. **NOISY**, a VOID and a **strike**: any self-pair (S0b/S0a, S1b/S1a, S0d/S0c, S1d/S1c) at **W16 or W1** outside
   **[0.985, 1.015]**. P158's self-pairs read 0.9994-1.0003 at W16 and 0.9998-1.0006 at W1. A NOISY read reads no flip.
3. **FUNCTION_FAIL**, a **contract finding against c1** (with the null clean): any S1 process decodes different tokens
   from S0a on any row, workload or length, or an S1 arm's timed reps digest differently. The verdict carries each S1
   arm's divergence from S0a (rows identical, first divergent step).
4. **FLIP:** g16 = **min over all four** S1/S0 pairs at W16 >= **1.03**, **and** g1 (the same at W1) >= **0.99**.
   - A separate PR may then make c1 qwen3_moe's K25 plan by default, citing this read and P160.
5. **NO_FLIP:** otherwise, with each failed bar named. The knob stays opt-in; a reading, not a failure of c1.

**The bars read the unrounded minima** (P143 A3's correction, carried).

**Labelled beside the verdict, never ruling:** P158's clock label, unchanged: a pair is **"clock-confounded"** when its
S1 arm's median sampled SM clock over the timed W16 blocks differs from its S0 arm's by more than 2 %, either way; an arm
with no clock sample reads **"clock unread"**, never 0. c1 compiles to more registers per thread than today's plan (P160:
194 and 127 against 64 on sm_120), and the label says so where the samples show a clock shift.

**Reported, never ruling:** P158's list (every pair and self-pair ratio, the cross-block self-pairs, the geomean over the
four pairs, each arm's telemetry and sampled clocks, K25's call totals after the runs per process, the warm-up's rates
and each over S0a's), plus the spy's counts after the runs per process.

## Predictions (graded, never ruling)

- g16 (the min of four W16 pairs) in **[1.20, 1.42]**: the projection is 1.378 at 1:1 carry and 1.280 at P158's carry
  (0.797); this read's own carry is the question it answers;
- g1 in **[0.99, 1.01]**: K25 is not on W1's T == 1 path;
- every W16 and W1 self-pair within [0.99, 1.01];
- every S1 token identical to S0's, at both workloads and both lengths;
- no pair clock-confounded;
- verdict **FLIP**.

## Budget and sequencing

- **One RTX 5090**, its own box, as P158:
  - hosts with a recorded power cap of at least 575 W, read within the last 3 days;
  - declared at <= $1.05/h, **guard 1.5 h**, at most **70 GB** downloaded at <= $0.011/GB (the 61 GB checkpoint is fetched
    on the box). Worst case 1.5 x $1.05 + 70 x $0.011 = **$2.35**, under the standing rule. P158's read cost $0.4190;
  - disk: 170 GB ordered; the runner refuses under 150 GB free.
- **The fit, e4b#1845's rule, with 25 % headroom on both sides,** P158's, unchanged: `INSTALL_TO_GATE_S` = 600 s (this
  premise adds 13 GPU tests to P158's 49), `NEED_BENCH` = **2,100 s** (the larger measured fetch-to-reduce wall at nine
  processes, p151-5090-4's 1,347 s, x1.5, rounded up), `RESERVE` = 600 s.
  - **The runner's check, before the install:** `FIT` = 600 + 2100 + 600 = 3300 s and `FIT_NEED` = 3300 x 1.25 =
    4125 s. Anything short of that is a refusal, rc 41.
  - **The static fit,** (boot + fit) x 1.25 <= guard: (300 + 3300) x 1.25 = 4500 s <= 5400 s (1.5 h). `tests/test_p161.py`
    asserts it.
- **Run ids** `p161-5090-1` to `-3`. Two consecutive HARNESS_ERROR, VOID, NOISY or NOT_RUN runs stop the lane. A
  pre-flight host stall, a power-floor refusal (rc 17), a fit refusal (rc 41) or an early-gate refusal (rc 43, 44) is
  not a strike (two consecutive refusals of the same kind count as one); a time-left skip of a step (STOP-2, rc 40) is.
- **Sequencing:** e4b#1877 merged (`f0dab09a`). The $0 layer-cut build on the A2000 confirms the licence half of the
  plan's witnesses, and its record joins this registration (`layercut/`). This registration then merges after its
  independent review and the maintainer's pass, and the box runs the shared pre-flight and `--validate` on the exact
  manifest.

## Tests

`tests/test_p161.py` checks:
- **the staged pins:** the nine pieces shared with P158's kit are at P158's pins but the route contract, which is
  `tests/test_k25_lut_gpu.py` frozen at `dd535cd6` (its sha256 pinned, its six functions, three of them c1's);
- **the runner:** the power floor and the guard fit before the install, the premise at 62, K39's kernel bytes, the
  decode's default, the plan table and c1's refusal under the tree in the tripwire, the warm-up process before eight arms
  at five reps, only `E4B_K25_PLAN` set per arm (`E4B_K25_LUT` never), no SANE phase, and NOISY and VOID not readings;
- **the guard fit** on both sides;
- **the build counts' derivation** in the code the box installs, with the spy's: today's shape and c1's pair as the
  route's tables state them, and one kernel call per projection;
- **the layer-cut record:** the licence half under each plan, with the default decode, the plan record and the spy;
- **the self-tests**, run from one directory as on the box: the reducer's 96 cases (the null before the contract, an
  exact 960, 1920 and 480, each witness's VOIDs, the witnesses tied, both early gates, the W1 guard, the unrounded bars,
  clock labels both ways and UNREAD, the enforced limit) and the box's 29 (the spy among them, on a stand-in kernel);
- **the mutants** each self-test must kill (40 against the reducer, 9 against the box);
- **the driver's run ids:** -1 to -3, nothing else.

## What this lane cannot say

- **Nothing about other families, cards, row counts, decodes or plans.** One RTX 5090, Qwen3-30B-A3B at W16 and W1, c1
  under prmt against today's plan under prmt. C2 and C3 (P160's 1.6100 and 1.6304) are not read; c1 under the tree is
  refused by the knob, never run.
- **Nothing about quality beyond identity.** The gate is token identity on P109's prompts; the contract tests are what
  hold the kernel bitwise in general.
- **Nothing exact about clocks.** The sampler reads at 500 ms; a pair's clock label is a confound named, not a
  correction applied.

**Review:** an independent pass (a seat that is not this lane's author and did not review P160's record), then the
maintainer's pass.
