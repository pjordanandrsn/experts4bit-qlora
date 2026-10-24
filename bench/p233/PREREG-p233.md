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
