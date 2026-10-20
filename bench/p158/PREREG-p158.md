# P158: K39's served read, the K25 decode `lut="prmt"` as Qwen3-30B-A3B's batched-decode expert GEMM (e4b#1313)

**Status:** registered before any data. A served read of P156's microbench GO, by P153's served form. One non-consumer
review, then the maintainer's final pass. $0 until the read. It registers after e4b#1862 (the `E4B_K25_LUT` knob)
merges; the box installs e4b at this registration's merge, which carries the knob.

## Why

- **The premise.** [P156](../p156/RESULTS-p156.md) read K39's decode GO in a CUDA-graph microbench on an RTX 5090:
  the two K25 expert GEMMs of a W16 step on S0's real router picks ran **1.1755x** faster under `lut="prmt"` than under
  today's select tree (the median of 64 paired reps, SE 0.0014, lower bound 1.1728), with every output bit-identical.
  - **Its clock caveat, bounded:** the box was sampled at 2,400 MHz at the start and 2,272 MHz under a power-cap event
    at the end, with no per-arm clocks. Giving the whole 1.0563 ratio to P under 1:1 clock scaling leaves a median of
    1.1128 and a lower bound of **1.1103**.
  - That is a microbench premise, never a gain. It times the two GEMMs alone.
- **The kernel.** grouped-nf4-gemm **v0.50.0** (`41ef086e`, gnf4#572) ships K39 as K25's opt-in `lut="prmt"`. Its three
  kernel files are byte-identical to P156's pin (`68bbefc4`), so the code this read installs is the code P156 timed.
  Its compiled contract, `kernel/test_nf4_smallm_prmt_gpu.py`, holds `prmt` bit-identical to `tree` at the served
  plan (BITWISE tier: the same codebook values, tiling, instruction shape and accumulation order).
- **The knob.** e4b#1862 adds `E4B_K25_LUT` = `tree` (the default, also when unset) | `prmt`. It is resolved once at
  the build (`info["k25_lut"]`), it refuses `prmt` without K39 in the installed kernel and on ROCm, and
  `hot_residency.K25_LUT_SEEN` counts every K25 call by decode (eager calls and graph captures, never a replay). Its
  served-route contract, `tests/test_k25_lut_gpu.py`, checks a B=16 step bit-equal under `prmt` and `tree`, eager and
  captured.
- **The projection** (a prediction, never a bar). P152's served census put the K25 pair at **11.748 ms** of a W16 step.
  P153's S0 arms ran W16 at 983.48 tok/s, a **16.269 ms** step on today's default server. If the microbench ratio
  carried 1:1 into the served step, g16 would be:
  - **1.121** at P156's median, 16.269 / (16.269 - 11.748 + 11.748 / 1.1755);
  - **1.077** at the clock-adjusted lower bound (1.1103).

  P156's own fidelity was 1.096 (its microbench T read 12.872 ms against the census's 11.748), so the share may carry
  over at more or less than 1:1.
- **The bar** is on W16, the batched decode where K25 runs. W1 is a no-regression guard: at T == 1 the expert path is
  the singleton bandwidth GEMV, not K25, so W1 should not move.

## What runs (`p158_run.sh`)

P153's served read by counted edits, with the K25 decode knob in place of the head's, no SANE phase and a clock sampler.
- **Engine and arms:** the shipped default server: graphs on, `max_seqs` 16, 4096 tokens per sequence, the fusions and
  the bandwidth GEMV at their auto defaults, the attention split cap at its qwen3_moe default (16), the int8 lm_head at
  its qwen3_moe default (**on**, P153's FLIP), the dense attention GEMV off, K25 at its default (auto). The arms differ
  only in the knob: **S0** = `E4B_K25_LUT=tree` (named, not unset), **S1** = `prmt`. The box refuses an arm with K25's
  mode, the int8 head, the split cap, the dense attention GEMV, a fusion knob or a decode GEMV knob set.
- **A warm-up process first** (P131 Amendment 2's process 0, as P153 ran it): `S0w`, an S0 build and speed pass in a
  fresh process before S0a. It is integrity-checked like every arm and VOIDs when missing. Its speed is reported only:
  it never enters a pair, a self-pair or a bar. Its tokens DO enter the null below.
- **Speed:** eight arms, ABBA twice (S0a S1a S1b S0b S0c S1c S1d S0d), each a fresh process through P109's W16 and W1 at
  **five timed reps**.
- **Telemetry:** P153's reads by nvidia-smi immediately before and after every timed block (power limit, **enforced
  power limit**, max SM clock, temperature, SM clock, power draw), plus a **clock sampler during each timed block**: an
  `nvidia-smi --query-gpu=clocks.sm,enforced.power.limit,power.draw.instant,clocks_event_reasons.active -lms 500`
  process of its own (never a thread of the timed process), its lines kept per block.
  - **The interval was measured before registration, at $0** (the lab A2000, a 48-call K25 graph replayed in ~2 s
    blocks, 12 interleaved pairs per setting). With no sampler, block over block read 1.0001 and 0.9978 (SE 0.0016 and
    0.0010). A 100 ms sampler read **1.0026 and 1.0025** (SE 0.0006 and 0.0008), a reproduced +0.25 % cost. 500 ms read
    0.9984 (SE 0.0007), inside the no-sampler spread. So the sampler runs at 500 ms: about 26 samples in a W16
    32-token block and about 48 in a 160-token one on the 5090.
- **No SANE phase.** The arms are bitwise-identical by contract, so the token-identity gate (below) replaces a quality
  read.
- **The premise, on this card, before anything is fetched:** P136's 19 premise tests (P153's files at P153's pins),
  K39's compiled contract (grouped-nf4-gemm's `kernel/test_nf4_smallm_prmt_gpu.py` at v0.50.0 byte for byte, staged as
  `premise/test_nf4_smallm_prmt_gpu.py`, 23 tests) and e4b's served-route contract (`tests/test_k25_lut_gpu.py`, 7
  tests). It must read **49 passed**, or the box refuses with rc 25. The two new files read 30 passed on the lab A2000
  at these pins (sm_86, torch 2.8.0+cu128, triton 3.4.0) before registration.
- **Order:** the power floor and the guard fit, before the install; install and tripwire; reducer and box self-tests;
  premise; fetch; NF4 arena bake; prompts; nine processes; reduce.
- **Pins:**
  - e4b: the clean checkout the driver ships, which must contain #1862's merge.
  - grouped-nf4-gemm: **v0.50.0** at `41ef086e` by sha. The tripwire checks K25's kernel file by sha256 (`ee46969a`,
    P156's pin), `"prmt"` among its decodes, the served plan at `lut="tree"`, the knob resolving `tree` and `prmt`, and
    the int8 head's qwen3_moe default.
  - The rest: torch 2.8.0+cu128, triton 3.4.0, transformers 5.17.0; Qwen/Qwen3-30B-A3B @ `ad44e777`.

## K25's engagement count, derived before data

At the build, `enable_decode_graphs` captures one graph per bucket (1, 2, 4, 8, 16) after **two eager warm-ups**
(`warmup: int = 2`), and `K25_LUT_SEEN` counts once per MoE-layer forward where the route is chosen. A bucket-1 step is
T == 1, which takes singleton groups and the bandwidth GEMV, never K25. Buckets 2 to 16 take device grouping at 2 x 8 to
16 x 8 (token, slot) rows, under K25's 256-row bound. So the record after the build reads exactly

    48 MoE layers x 4 buckets (2, 4, 8, 16) x 3 forwards (two warm-ups + one capture) = 576

under the arm's decode (`{"tree|le256": 576}` in S0, `{"prmt|le256": 576}` in S1), and nothing under the other.
`tests/test_p158.py` checks the three facts it rests on in the code the box installs (the warm-up default, the bucket set
and the build's call, T == 1's route). After the runs the record holds the arm's decode only, and its total is the same
in every process (the arms run the same prompts and lengths).

## The rule (`p158_reduce.py`, fixed before data)

The first rung that applies decides. P153's integrity checks are carried; the bars are moved to W16.

1. **VOID.** Any of:
   - a record is missing or not ok, the warm-up process's included;
   - any of P153's checks fails on any record: pins, the 170-SM card, census and modes, the bandwidth GEMV, the captured
     buckets, bucket 1 replayed, the split cap at its qwen3_moe default with its plans at capture;
   - the dense attention GEMV is not off, or the int8 head is not at its qwen3_moe default in every record (on,
     `default-allowlisted`, the one lm_head at 151936 x 2048, routing one-row calls with none falling back);
   - **K25's decode:** `info["k25_lut"]` is not `{"env": <arm's decode>, "lut": <arm's decode>, "why": null}`; the
     record after the build is not exactly `{"<arm's decode>|le256": 576}`, or held anything before it; after the runs
     it holds another decode, or its total differs across the nine processes;
   - **power:** any timed block's power limit or enforced power limit, read before or after it, below **575 W** or
     unreadable;
   - the arms differ in prompts or lengths, or an arm is not at five reps; a void slope;
   - **THE NULL** (checked before any S1 token is read): any S0 process (S0b, S0c, S0d and the warm-up) decodes
     different tokens from S0a on any row, workload or length, or any S0 arm's timed reps digest differently. That
     reads **VOID, "instrument: tree not token-deterministic across processes"**, never a finding against K39.
2. **NOISY**, a VOID and a **strike**: any self-pair (S0b/S0a, S1b/S1a, S0d/S0c, S1d/S1c) at **W16 or W1** outside
   **[0.985, 1.015]**. P153's self-pairs read 0.9958-1.0010 at W16 and 0.9993-1.0003 at W1. A NOISY read reads no flip.
3. **FUNCTION_FAIL**, a **contract finding against K39** (with the null clean): any S1 process decodes different tokens
   from S0a on any row, workload or length, or an S1 arm's timed reps digest differently. The verdict carries each S1
   arm's divergence from S0a (rows identical, first divergent step).
4. **FLIP:** g16 = **min over all four** S1/S0 pairs at W16 >= **1.03**, **and** g1 (the same at W1) >= **0.99**.
   - A separate PR may then make `prmt` qwen3_moe's K25 decode by default, citing this read and P156.
5. **NO_FLIP:** otherwise, with each failed bar named. The knob stays opt-in; a reading, not a failure of K39.

**The bars read the unrounded minima** (P143 A3's correction, carried).

**Labelled beside the verdict, never ruling:** a pair is **"clock-confounded"** when its S1 arm's median sampled SM
clock over the timed W16 blocks differs from its S0 arm's by more than 2 %, either way. The comparison is exact on the
whole-MHz samples (`100 |S1 - S0| > 2 S0`). An arm with no clock sample reads **"clock unread"**, never 0, and labels
nothing. On a power-capped board the lighter decode can hold a higher boost; the label says so where the samples show
it, and the verdict stands beside it.

**Reported, never ruling:**
- every pair and self-pair ratio, including the cross-block self-pairs;
- the geomean over the four pairs;
- each arm's telemetry (power limits, enforced limits, max SM clocks, the hottest temperature, the slowest SM clock read
  beside its timed blocks) and its sampled clocks: median and minimum SM clock, median and maximum instantaneous power,
  the enforced limits and the clock-event reasons seen, per workload;
- K25's call totals after the runs, per process;
- the warm-up process's rates, and each over S0a's.

## Predictions (graded, never ruling)

- g16 (the min of four W16 pairs) in **[1.06, 1.14]**: the projection is 1.121 at P156's median and 1.077 at its
  clock-adjusted lower bound, and P156's fidelity (1.096) leaves the carry-over uncertain either way;
- g1 in **[0.99, 1.01]**: K25 is not on W1's T == 1 path;
- every W16 and W1 self-pair within [0.99, 1.01];
- every S1 token identical to S0's, at both workloads and both lengths;
- no pair clock-confounded;
- verdict **FLIP**.

## Budget and sequencing

- **One RTX 5090**, its own box:
  - hosts with a recorded power cap of at least 575 W, read within the last 3 days;
  - declared at <= $1.05/h, **guard 1.5 h**, at most 10 GB at <= $0.011/GB. Worst case 1.5 x $1.05 + 10 x $0.011 =
    **$1.69**, under the standing rule;
  - disk 150 GB (a 61 GB checkpoint, its NF4 snapshot and arena).
- **The fit, e4b#1845's rule, with 25 % headroom on both sides,** from the two served reads of this form on the 5090
  (their own logs: p151-5090-4, p153-5090-1):

  | phase | p151-5090-4 | p153-5090-1 |
  |---|---|---|
  | install, tripwire, self-tests, premise | 102 s | 59 s |
  | fetch | 440 s | 287 s |
  | bake and prompts | 87 s | 75 s |
  | per process | about 91 s | about 75 s |
  | fetch to reduce, at nine processes | **1,347 s** | 1,039 s |

  - **Install to the gate:** `INSTALL_TO_GATE_S` = 600 s (P155's allowance; the larger measured is 102 s, and this
    premise adds 30 GPU tests, about a minute on the A2000).
  - **The work:** `NEED_BENCH` = **2,100 s**, the larger fetch-to-reduce wall (1,347 s) x1.5, rounded up.
  - **The reserve:** `RESERVE` = 600 s.
  - **The runner's check, before the install:** `FIT` = 600 + 2100 + 600 = 3300 s and `FIT_NEED` = 3300 x 1.25 =
    4125 s. Anything short of that is a refusal, rc 41.
  - **The static fit,** (boot + fit) x 1.25 <= guard: (300 + 3300) x 1.25 = 4500 s <= 5400 s (1.5 h). `tests/test_p158.py`
    asserts it. A 1.25 h guard would fit at exactly 4500 s with no room for a slow boot, so the guard is 1.5 h.
- **Run ids** `p158-5090-1` to `-3`. Two consecutive HARNESS_ERROR, VOID, NOISY or NOT_RUN runs stop the lane. A
  pre-flight host stall, a power-floor refusal (rc 17) or a fit refusal (rc 41) is not a strike (two consecutive fit
  refusals count as one); a time-left skip of a step (STOP-2, rc 40) is.
- **Sequencing:** e4b#1862 merges first (it is chained after e4b#1860). This registration then merges after its
  independent review and the maintainer's pass, and the box runs the shared pre-flight and `--validate` on the exact
  manifest.

## Tests

`tests/test_p158.py` checks:
- **the staged pins:** every piece shared with P153's kit (7) is at P153's pins, the premise file is K39's at
  grouped-nf4-gemm v0.50.0 (its sha256 pinned), and none of SANE's pieces is staged;
- **the runner:** the power floor and the guard fit before the install, the premise at 49, K39's kernel bytes and the int8
  head's default in the tripwire, the warm-up process before eight arms at five reps, only `E4B_K25_LUT` set per arm,
  no SANE phase, and NOISY and VOID not readings;
- **the guard fit** on both sides, from the measured walls;
- **the engagement count's derivation** in the code the box installs;
- **the self-tests**, run from one directory as on the box: the reducer's 62 cases (the null before the contract, the
  null reading the warm-up and S0's own reps, an exact 576, the W1 guard, the unrounded bars, clock labels both ways
  and UNREAD, the enforced limit) and the box's 24;
- **the mutants** each self-test must kill (26 against the reducer, 5 against the box);
- **the driver's run ids:** -1 to -3, nothing else.

## What this lane cannot say

- **Nothing about other families, cards, row counts or plans.** One RTX 5090, Qwen3-30B-A3B at W16 and W1, today's K25
  plan. Decode x plan (P156's reported C1-C3) is its own later read.
- **Nothing about quality beyond identity.** The gate is token identity on P109's prompts; the contract tests are what
  hold the kernel bitwise in general.
- **Nothing exact about clocks.** The sampler reads at 500 ms; a pair's clock label is a confound named, not a
  correction applied.

**Review:** an independent pass (a seat that is not this lane's author and did not review P156's record), then the
maintainer's pass.
