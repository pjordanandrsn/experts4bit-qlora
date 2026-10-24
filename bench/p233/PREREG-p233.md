# P233: does serving at the solver placement keep quality? (`E4B_PAGED_PLACEMENT=solver` against `all-vram`, MAP 2)

**Registered before any reading.** One RTX 5090 reads whether `serve_paged` at `E4B_PAGED_PLACEMENT=solver`, with
Qwen3-30B-A3B's experts split across all three tiers (VRAM, host DRAM, NVMe), keeps teacher-forced quality against the
same server at `all-vram`. Its verdict decides whether a doc or lane may propose serving at `solver` at all. **It says
nothing about speed, and `solver` stays non-default whatever it reads.**

| the review's condition (2026-10-23 and 2026-10-24) | where |
|---|---|
| a registered SOLVER-vs-ALL-VRAM quality read, SANE-style teacher-forced NLL and argmax agreement, the bar fixed before data, before any lane or doc proposes serving at `solver` | [The rule](#the-rule-p233_reducepy-fixed-before-data) |
| the instrument is P153's (P115 Phase B's `measure_phase`), called **unmodified**, eager at both placements | [The instrument](#the-instrument) |
| every arm loads the same checkpoint and tokens, and the record **proves each arm's tier placement** in bytes; otherwise VOID | [The rule](#the-rule-p233_reducepy-fixed-before-data), rung 1 |
| the VRAM+NVMe arm is reported only: what it attributes, and that it never rules | [The arms](#the-arms) |
| no speed, and `solver` stays non-default either way | [What this lane cannot say](#what-this-lane-cannot-say) |

## Why

- **`serve_paged` serves at `solver`, eagerly, through all three tiers** (bench/serve-hybrid/DESIGN-HC.md, sections 1-2).
  Every quality number so far was read at `all-vram`. At `solver` the MoE layers that are not all-resident take the
  baseline tiered forward instead of the all-resident collapse: VRAM rows on the GPU, DRAM rows on the host's CPU
  kernels, NVMe rows streamed to the GPU. The decode graphs are off there, and `E4B_PAGED_GRAPHS=1` is refused at startup
  (#2067).
- **Tokens across placements are not bitwise** (DESIGN-HC.md section 3). The DRAM tier dequantizes and accumulates in
  fp32 on the host, while the GPU path rounds its grouped-GEMM outputs and its SwiGLU epilogue to bf16. An explore box,
  never a reading, served both placements on one RTX 5090. At three tiers, 7 of 10 greedy prompts matched `all-vram`; the
  other three diverged at plausible near-tie tokens. At VRAM+NVMe, 10 of 10 matched. The expert-level gap was then
  re-derived on the CPU at Qwen3-30B-A3B's expert shape: the DRAM tier's output sits 4.8e-3 relative RMS from the GPU
  path's, about 2.7x one bf16 rounding, and it is the **more** accurate side (1.7e-3 from fp64 exact, against 4.5e-3 for
  the GPU path). That was ruled not a defect. It is also not a quality read: a token count on 10 prompts cannot bound
  what the gap does to log-probs, which is this lane's question.

## The arms

Every arm is the shipped server (`PagedServeConfig.from_env()` + `build_engine(cfg)`) on the same Qwen3-30B-A3B checkpoint,
NF4 arena and token windows, `qwen3_moe` defaults throughout, **eager**: `E4B_PAGED_GRAPHS=0` in every arm, so the only
difference between arms is where the experts sit.

| arm | `E4B_PAGED_PLACEMENT` | `E4B_PAGED_VRAM_GB` / `E4B_PAGED_DRAM_GB` | the expected split | role |
|---|---|---|---|---|
| **A** | `all-vram` | (unused) | every expert in VRAM; every MoE layer all-resident | **the reference**: its log-probs are saved and scored against |
| **S** | `solver` | `8.0` / `3.0` | all three tiers non-empty (the explore box's split: VRAM 3,236 / DRAM 1,213 / NVMe 1,695 of 6,144 experts) | **the ruling arm** |
| **C** | `solver` | `8.0` / `0` | VRAM and NVMe only, DRAM empty (the explore box's: 3,236 / 0 / 2,908) | **reported only** |

- **What C attributes, and why it never rules.** C keeps S's VRAM budget and moves S's DRAM experts onto NVMe. NVMe experts
  compute on the GPU path, like VRAM experts, so C differs from A only in taking the tiered forward rather than the
  collapse. S differs from C only in the DRAM tier's host arithmetic. Reading S against A beside C against A therefore
  splits S's gap into the tiered forward's part (C) and the DRAM tier's part (S beyond C). That split explains a result;
  it cannot change one. The question is S against A, so C is reported and never rules.
- **A's split** is all-resident by construction (`all-vram` moves every expert into the VRAM tier). Every MoE layer must
  take the collapse.

## The instrument

- **P115 Phase B's `measure_phase`, as P153 and P232 read it, called unmodified.** `bench/p115/p115_quality.py`'s own
  entry point refuses any placement but `all-vram`, so this lane's driver (`p233_box.py`, a code PR before the reading)
  builds each arm's server with the placement under test and then calls `measure_phase` itself.
  - **For arm A, the driver passes exactly the arguments `p115_quality`'s main passes,** and its test asserts so. A is
    therefore the instrument P153 read.
  - **The driver's test runs against `measure_phase`'s real record shape:** a committed P153 or P232 quality record, not
    hand-written fixtures.
- **Group T = 1 and T = 8, each read on its own.**
  - At T = 1 one window decodes per pass, as P153's SANE did.
  - At T = 8 eight windows decode together, so every decode step routes eight rows through each tier.
- **24 wikitext windows x 128 positions per T**, after a 512-token prompt (24 divides by 1 and 8). Every pass's KV is at
  the server's 4,096 tokens with the server's cap.
- **Three phases, each one process:**
  - **A:** saves its log-probs per (window, T), and its `rep` arm (A again on the first group) is the floor draw.
  - **S:** scores against A's saved log-probs.
  - **C:** scores against A's saved log-probs.
- **The premise, on the card, before anything is fetched:** the tier's own GPU tests and the CPU kernels it needs
  (`tests/test_hybrid_cold_dest.py`, `tests/test_serve_hybrid_compose.py`), at counts the kit names, or rc 25.
- **Pins:** e4b at the kit's merge; grouped-nf4-gemm v0.50.0 `41ef086e`; torch 2.8.0+cu128, triton 3.4.0, transformers
  5.17.0; Qwen/Qwen3-30B-A3B @ `ad44e777`.
- **Defaults are named per arm** (the e4b default at the launch commit, the same in every arm, and recorded):
  `E4B_INT8_HEAD_BW`, `E4B_INT8_HEAD_ROWS`, the split cap, the fusions. The reducer requires the same resolved values in
  every phase.

## The quality bar, and why

**The bar is P153's, at each T on its own:** **|mean NLL(S) - mean NLL(A)| <= 0.02 nats** and **argmax agreement >= 0.95**,
teacher-forced over the 24 x 128 positions.

- **Identity is not expected.** The DRAM tier's experts compute in fp32 on the host and the GPU path rounds through bf16.
  The gap is about 2.7x one bf16 rounding of an expert's output, on the quarter of the experts that sit in DRAM. A logit
  gap of that order flips a greedy token where the top two logits nearly tie, which is what the explore box saw. So
  tokens are reported, not gated.
- **A logprob bar reads what that gap does to the served distribution.** A per-expert difference at a few bf16 roundings,
  mixed by the router's weights and passed through 48 layers, should move each position's log-softmax far less than
  0.02 nats on average. Argmax agreement fails only at near-ties, which a 0.95 floor tolerates.
- **The standard is the one every served lever since P153 has met:** P153's int8 head (bias +0.0022, argmax 0.989, KL
  0.0006) and P232's 2-8 row route. The tiered path is held to that standard, not a looser one.

## The rule (`p233_reduce.py`, fixed before data)

The first rung that applies decides.

1. **VOID.** Any of:
   - a phase record missing or not ok;
   - the pins, the card (cc 12.0, 170 SMs), or the windows: 24 per T, 128 positions, one window set (the same sha) in
     every phase;
   - **the same model in every arm:** the checkpoint revision and the NF4 arena's sha differ between phases;
   - **the placement, proved per arm from the record** (the tier byte counts and expert counts as the solver resolved
     them, and the MoE-layer path counts from the engagement):
     - **A:** every expert in VRAM, nothing in DRAM or on NVMe, and every MoE layer on the collapse;
     - **S:** a non-zero byte count on each of VRAM, DRAM and NVMe, the tier budgets as registered, and at least one MoE
       layer on the tiered forward. Without that, "S against A" would not be measuring the solver;
     - **C:** DRAM empty, NVMe non-empty, and at least one layer tiered;
   - graphs on in any arm, or any decode bucket captured;
   - the defaults named above resolving differently between phases;
   - a missing per-window NLL, argmax or KL on a scored arm (never read as 0), or an agreement that is not k/128 for an
     integer k.
2. **Per T in (1, 8): SANE** when S against A holds both bars at T; **NOT_SANE** otherwise, with the failed bar named.
   The bars read the unrounded values, and argmax as exact pooled counts.
3. **The verdict:**
   - **SOLVER_SANE:** S is SANE at T = 1 and at T = 8. A doc or lane may then propose serving at `solver` as an opt-in
     placement, citing this read. It is not a default, and any speed claim needs a read of its own.
   - **SOLVER_NOT_SANE:** S fails at either T, with each failed bar named. No doc or lane proposes serving at `solver`
     until a later registration reads it SANE.
4. **TRIPWIRE, reported and named, never changing the verdict.** The PASS bar sits far above the predicted gap, so a
   defect several times larger than the DRAM tier's expected arithmetic would still pass it. The wire fires at any T if
   **|mean NLL(S) - mean NLL(A)| > 0.005 nats**, or **argmax agreement < 0.98**, or **the S-against-A KL above max(3 x
   the A self-pair floor, 0.002 nats per position)**. The floor is the KL of A's `rep` arm against A; a zero or absent
   floor falls to the absolute bound, never to 0. If it fires, the verdict still reads as above, but no doc or lane may
   propose serving at `solver` until a short written look explains it, reviewed as usual. The reducer prints `TRIPWIRE`
   with every condition that fired, or `TRIPWIRE none`.

## Reported, never ruling

- **C against A** at each T: mean NLL gap, argmax agreement and KL, with the bars printed beside them. Also S against C:
  the DRAM tier's own share.
- the KL of S against A at each T, and its per-window spread;
- greedy token identity at each T, from the teacher-forced argmax: positions where S's argmax differs from A's, per
  window;
- each arm's resolved split (expert and byte counts per tier), its layer-path counts and its build time;
- per phase, the host CPU model, its ISA flags (AVX2, AVX-512, AMX) and the CPU kernels' thread count;
- each phase's wall time (an explore figure, never a speed claim: the instrument's passes are not the server's decode
  loop).

## Predictions (graded in the read, never ruling)

| prediction | basis |
|---|---|
| S against A at every T: \|NLL gap\| <= 0.002 nats, argmax >= 0.98 | the DRAM gap is a few bf16 roundings on a quarter of the experts |
| S against A KL at most 0.001 at every T | below P153's int8 head (0.0006 there was a whole-head grid change) to within 2x |
| C against A at every T: \|NLL gap\| <= 0.0005, argmax >= 0.995 | the tiered forward on the GPU path only; 10 of 10 greedy prompts matched on the explore box |
| T = 1 and T = 8 within 2x of each other on every S statistic | the DRAM rows' arithmetic does not depend on how many rows share a step |
| verdict **SOLVER_SANE**, TRIPWIRE none | |

## What this lane cannot say

- **No speed.** Decode at `solver` is eager and runs DRAM rows on the host, and this lane reads quality only.
- **No other split.** One VRAM/DRAM budget (8.0 / 3.0 GiB) on one card. Other budgets move different experts between
  tiers; the arithmetic per tier is the same, but the mix is not read.
- **No other model, card or release.** Qwen3-30B-A3B at its pinned revision on the NF4 arena, one RTX 5090,
  grouped-nf4-gemm v0.50.0.
- **No other host CPU.** The DRAM tier's experts compute on the host's CPU kernels. Whether their fp32 reduction order
  is the same across instruction sets (AVX2, AVX-512, AMX) and thread chunking is not tested here, so a result on one
  host's CPU is not claimed for another's. The CPU model, its ISA flags and the kernels' thread count are recorded per
  phase, not screened.
- **No graphs, no prefill graph, no int4 experts, no `E4B_NF4_SMALLM_BW` at `solver`:** each stands down or refuses there
  (DESIGN-HC.md section 1), so none is read.
- **No default.** `solver` stays opt-in and non-default whatever this reads.

## Budget and the fit

- **One RTX 5090 on Vast,** rented through pod-launch, at most **$1.05/h** with a **150 GB** disk (the checkpoint, the
  NF4 arena and the NVMe tier's files). Host RAM at least **60 GiB** (the checkpoint's load plus the DRAM tier's 3 GiB).
  **No power floor:** this is a quality read, and the power limit is recorded, not ruled.
- **The work, every term conservative:** install 600 s; fetch 1,500 s (57 GB at the 40 MB/s floor); bake 400 s; phase A
  at two group sizes, about 600 s; phase S about 2,400 s and phase C about 2,400 s (eager, the tiered path's host and NVMe
  work at about 4x A's decode time, the explore box's per-prompt ratio); reserve 600 s. **FIT = 8,500 s, x 1.25 = 10,625
  s.**
- **Guard 3.25 h** ((boot 300 + 8,500) x 1.25 = 11,000 s <= 11,700 s). **Worst case:** 3.25 h x $1.05 + 70 GB x
  $0.04 = **$6.21** a started box. A normal box is expected near 1.5 hours, about $1.6. The shakedown reads the real phase
  times before the reading, and an amendment re-fits if they differ.
- **At most two reading boxes start.** A box STARTS when its runner passes the refusals before the install and begins
  the install, whatever its outcome after that. Two started boxes at most cost 2 x $6.2125 = **$12.43**.
- **A box that does not start** (refused at create, or refused before the install: the host floors, the power or card
  read, the download probe, the guard fit) costs at most **$0.30**: about 10 minutes of a box at $1.05/h for boot and the
  checks ($0.18), plus the probe's 2 GB at $0.04/GB ($0.08), rounded up. It counts toward no cap and no strike, but it uses an id. At most six such
  boxes are spent in this lane; the seventh stops it for a person.
- **The lane's worst case:** $12.43 + 6 x $0.30 = **$14.23**, inside the $15 rail. The shakedowns are explore tier,
  at its own caps, and are not counted here.
- **Ids:** `p233-5090-1` to `-3` for readings, plus `p233-5090-r1` to `-r6` for any box after one that did not start.
  Ids are labels, not budget: the two-started-box cap and the six non-start cap govern. `p233-shake-1` to `-3` are for the
  shakedown (explore tier, `--shakedown`: a reduced window count, any card that holds the arms). The shakedown's record
  becomes the reducer's real fixture by an amendment before the reading. Two consecutive started runs not read stop the
  lane, and a shakedown never counts.

## Sequencing

1. This registration merges after the independent review and the maintainer's pass.
2. The kit (`p233_run.sh`, `p233_box.py` around the unmodified `measure_phase`, `p233_reduce.py`, `p233_drive.sh`, with
   their tests, a `$0` dry run and the reducer's mutants) is a code PR on its own review.
3. The shakedown runs on the kit's head; its record becomes the real fixture by an amendment.
4. The reading runs at that amendment's merge, after an Approved on the launch commit.
5. On SOLVER_SANE with TRIPWIRE none, a doc or lane may propose serving at `solver` as an opt-in placement, citing this
   read. On SOLVER_NOT_SANE, or with the TRIPWIRE fired, it may not, until a written look or a later read.

## Amendment 1 (2026-10-24, before any data): the head knobs named explicitly in every arm

- **Why:** since e4b#2095 (merged after this registration), an UNSET `E4B_INT8_HEAD_ROWS` resolves to 8 on `qwen3_moe`
  at the 151936 x 2048 head, from lane P232's read (`e4b.serve.p232.int8-head-rows.qwen3.5090.2026-10-24`). A served
  Qwen3 registration launching at or after that commit names the knob explicitly rather than relying on an unset default
  whose meaning changed.
- **The change:** the Instrument's defaults bullet now reads as follows. Every arm (A, S and C) sets
  **`E4B_INT8_HEAD_BW=1`** and **`E4B_INT8_HEAD_ROWS=8`** explicitly; an unset knob is never relied on.
  - Each phase records the resolved values: `levers.int8_head_bw`, with its `rows` and `source`.
  - The reducer requires the same resolved values in every phase. A phase whose head is not the int8 head, or whose
    rows did not resolve to 8, is a VOID (rung 1).
  - The split cap and the fusions stay at the e4b default at the launch commit, the same in every arm, and recorded.
- **What does not change:** the arms, the instrument, the windows, the bar, the tripwire, the verdict, the budget and
  every cap. The head knobs are the same in every arm, so they cannot enter the S-against-A comparison. Naming them
  only fixes which head path all three arms share.
- Pre-data: no P233 box has run, and the kit is not yet written.

## Amendment 2 (2026-10-24, before any data): what the kit implements exactly

Written with the kit (`p233_run.sh`, `p233_box.py`, `p233_reduce.py`, `p233_drive.sh`), before any box. No bar, arm,
group size, window count or cap moves. Each item fixes something the registration left open.

- **Arm A is exactly `p115_quality`'s instrument.**
  - The box passes `measure_phase` the keywords `p115_quality.main()` passes, by name and in order, with main's
    defaults (prompt 512, continuation 128, chunk 512, floor chunk 256). It passes no `arms`, `ref_kw` or `stand_in`.
  - So A scores the instrument's full off-phase arms: R, rep, half, chunk and mutant_scale. Only R (the reference) and
    rep (the floor) are read by the rule; the rest are reported.
  - A test asserts the equality three ways: from main's source, against captured calls of main and of the box, and on
    `measure_phase`'s real record shape (P232's committed quality records).
- **The placement proof, read after the build from each MoE layer's tier state.** `build_engine` keeps no solver
  manifest.
  - **Experts per tier:** VRAM, DRAM and NVMe.
  - **Bytes per tier:** the expert count times the arena's bytes per expert. The resident stacks' own bytes are
    reported beside them, never ruled.
  - **Each layer's static path:** the collapse only when the layer is collapse-resident, not amortized and all-hot.
  - **Per-call path counts at every T,** from counters added for the phase and removed after it.
  - **VOID** when:
    - a layer is never called at some T, or its calls disagree with its static path;
    - A holds any expert outside VRAM or any layer off the collapse;
    - S has zero bytes on any tier, budgets other than 8.0 / 3.0, or no layer tiered;
    - C has a non-empty DRAM tier, an empty NVMe tier, budgets other than 8.0 / 0, or no layer tiered.
  - Whether the DRAM tier ran rows on the host CPU at a given T is reported, not ruled.
- **The same model in every phase:** the configured revision and the loaded model's commit hash, equal across phases.
  The arena is the full file's sha256 and its index's sha256, hashed inside each phase.
- **The head check (Amendment 1):** in every phase,
  - one int8 head module at [151936, 2048], mode 1, source `explicit`;
  - `rows_max` 8, source `explicit`.

  The head's call counts per T are recorded, not ruled.
- **The same resolved defaults:** the head, the split cap (cap and source) and the fusion modes and sources must be equal
  across phases, and the cap's and fusions' sources must be defaults. The census and the other lever records are kept.
- **No captures:** `config.graphs` is False, the build's graph status names no captured bucket, and every pass's
  engagement shows no graph and zero replays.
- **Missing values:**
  - A missing per-window NLL, argmax or KL on S or on C is a VOID, as is an R without an NLL in every window.
  - Window ids must run 0..W-1 once each, and `rep` must hold T entries.
- **The pass KV:** 4,096 tokens (256 blocks), the phase's own cap, at least W/T passes and no plan clash. The plan values
  are reported, not ruled.
- **The premise:** exactly **37 passed** from the tier's GPU tests (`tests/test_hybrid_cold_dest.py`) and **6 passed**
  from the compose tests (`tests/test_serve_hybrid_compose.py`), with no skips, or rc 25. `fastapi` joins the install,
  which the compose tests import.
- **Argmax positions (reported):** S's and C's positions that differ from A's are read from the instrument's own
  scoring calls, which are observed and never changed. S against C argmax agreement is computed from those lists;
  S-against-C KL is unread.
- **The CPU record:** the first processor's flags from `/proc/cpuinfo` (unread, never False, when absent), the CPU
  kernels' pool size, the CPU count, the affinity, torch's thread count and the NUMA node count.
- **Config pins:**
  - `E4B_PAGED_MAX_SEQS=16`;
  - A leaves the tier budgets unset;
  - S's and C's budget strings are exactly `8.0` / `3.0` and `8.0` / `0`;
  - the disk refuses below 140 GB free on the 150 GB rental.
- **Phases:** A, then S, then C, each its own process, with caps of 2,400 s for A and 4,800 s for S and C, all under the
  deadline. A failed phase stops the later ones, and the read is then a VOID.
- **The shakedown:** 8 windows (a shakedown at 24 is a VOID), any card at compute capability 8.6 or later with at least
  30,000 MiB.
- **Run ids:** `p233-5090-r1` to `-r6` are accepted by the runner and the drive exactly as the reading ids are.
- Pre-data: no P233 box has run.

## Amendment 3 (2026-10-24, before any data): the NVMe tier's pinned rows named in every arm

- **What the first shakedown showed:** the first shakedown box ran A to completion and then stopped in S's first
  prefill, before any S or C number existed: `nvme_residency` refused "request of 95 unique rows exceeds hot_rows=64".
  A shakedown is never a reading, and nothing was read.
- **Why it happens:** `E4B_PAGED_HOT_ROWS` was left unset, so the tier held the server's default, 64 pinned rows. The
  tier's contract (`experts4bit_qlora/engines/nvme_experts.py`, `enable_nvme_residency`) states a hard floor: `hot_rows`
  must be at least the number of unique cold experts one forward routes. For a prefill of T tokens that approaches
  min(T x k, E minus the hot experts). The instrument prefills in 512-token chunks with k = 8 and E = 128, so 64 is below
  the floor at the solver split. The same holds for the server itself at its own defaults (`E4B_PAGED_CHUNK_TOKENS=512`,
  `E4B_PAGED_HOT_ROWS=64`); that is a separate e4b change, not this registration's.
- **The change:** every arm (A, S and C) sets **`E4B_PAGED_HOT_ROWS=128`** (E), so the floor holds at any prefill, and the
  arms still differ only in placement.
  - The box refuses an arm whose knob is not exactly 128 (unset included).
  - Each phase records the tier's resolved `hot_rows` (`config.hot_rows`). The reducer VOIDs a phase whose recorded value
    is not the integer 128, or whose knob record is not "128" (rung 1, beside Amendment 1's head knobs).
  - The runner's tripwire asserts that the config reads `E4B_PAGED_HOT_ROWS` (default 64) and that `build_engine` hands
    `cfg.hot_rows` to the tier.
- **The cost:** the pinned rows take about 0.34 GB of host RAM (128 rows x 2,654,208 bytes, the arena's bytes per expert
  as the shakedown's placement record reads it), inside the 60 GiB host floor. The DRAM tier's 3 GiB budget is
  unchanged.
- **What it does not change:** no byte the forward computes. The same NF4 rows reach the same kernels; only how many cold
  rows the tier may stage at once moves. The arms, the instrument, the windows, the bar, the tripwire, the verdict, the
  budget and the ids are unchanged.
- **The scope:** the read speaks for **solver placement with `E4B_PAGED_HOT_ROWS=128`**. A doc or lane that cites it to
  propose solver serving names that value. A build-time floor check, if e4b adds one, is met: for this server the floor is
  min(max(chunk tokens, max sequences) x k, E) = min(512 x 8, 128) = 128.
- **The shakedown:** the next shakedown runs from this amendment's head before merge (a shakedown is never a reading); the
  reading launches only from the merged commit.
