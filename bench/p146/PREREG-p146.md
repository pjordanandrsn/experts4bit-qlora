# P146 — does LAYER_MAJOR=1 cut the CLI trainer's step time under expert offload on an RTX 4090?

**Question.** Layer-major accumulation is licensed three times as a library call: P137 2.365× (Qwen3-30B-A3B, RTX 4090),
P140 2.311× (Qwen3-30B-A3B, RTX 5090) and P144 2.118× (Mixtral-8x7B, RTX 4090). Each read ran the lane's own loop at
P137's recipe. The CLI trainer, `python -m experts4bit_qlora.train`, makes the same calls under
`OFFLOAD_EXPERTS=1 LAYER_MAJOR=1` (#1744). That was checked bit for bit against the library on a fixture, but its own
recipe was never timed. `docs/STATUS.md` says so: "its own recipe (AdamW, token-budget batches) is unread". This lane
times the CLI as a user runs it.

**Kit.** P137's kit at its Amendment 3, by named substitutions (`bench/p146/`). The arm runner changes: each arm is one
fresh process that runs `train.py`'s own `main()`, and the steps, batches, optimizer and evaluation are `train.py`'s.
STRICT terms are P137's wherever they apply. Where the CLI's shape changes a term, the change is named below.

## Seen data, disclosed (sets no bar)

| run | model, card | S | host-link fit |
|---|---|---|---|
| `p137-4090-7` | Qwen3-30B-A3B, RTX 4090 | 2.365 | T 1.459 s, C 1.743 s |
| `p140-5090-1` | Qwen3-30B-A3B, RTX 5090 | 2.311 | T 1.139 s, C 1.461 s |
| `p144-4090-1` | Mixtral-8x7B, RTX 4090 | 2.118 | T 2.004 s, C 3.369 s |

These set the scale of the prediction below and nothing else. No A2000 timing seeds or bands anything; the A2000 runs
the $0 dry run for correctness only.

## The pins

- e4b `E4B_PIN=4917506a22d960a6586bd5cefe9a0deca662fc46`: main with #1744 (`FAST_TRAIN` in `train.py`, checkpointing
  enabled before `enable_fast_train`) and #1752 (the host thread cap).
- grouped-nf4-gemm 0.46.0 `GNF4_SHA=6b7c2e7474d20c8843814dd46dad6cfc3d94b440`.
- torch 2.8.0+cu128 (the image's), transformers 5.18.0, bitsandbytes 0.50.2, `huggingface_hub>=0.23,<2`.
- `Qwen/Qwen3-30B-A3B` @ `ad44e777bcd18fa416d9da3bd8f70d33ebb85d39`, P137's.
- **The data:** `tatsu-lab/alpaca` at revision `dce01c9b08f87459cf36a430d809084718273017`, its one train parquet
  (`data/train-00000-of-00001-a09b74b3ef9c3b56.parquet`), staged by the driver from the controller as `alpaca.parquet`
  with `DATA_SHA=06391b656a06fd3fb9d213160ef2398796c3b7f3dc75ef1e3ced30d461517073`. That is P137 Amendment 1 carried
  over: nothing is downloaded on the box. `train.py` calls `load_dataset("tatsu-lab/alpaca", split=...)` twice, for
  `train[:2000]` and `train[2000:2064]`. The box serves both calls from the staged parquet with the same split strings,
  and records them. The parquet is kept outside the repository, since the dataset's licence is not the repository's.

## The arms and the run

- **The subject:** `train.py`'s `main()` at its own defaults:
  - SEQ 192, `TOKEN_BUDGET` 2048 (length-bucketed micro-batches of at most 2048 padded tokens), `GRAD_ACCUM` 4;
  - r 8 / α 16, attention LoRA on, expert LoRA on, router frozen;
  - torch AdamW at lr 2e-4 with cosine warm-up;
  - `N_TRAIN` 2000, `EVAL_EVERY` 50, `SEED` 0;
  - the chunked loss at its `auto` default.
- **The lane sets only these:** `STEPS=34` (2 warm + 32 timed, P137 Amendment 3's n); `DO_GEN=0` (the sample
  generations are not training); `TR1_CENSUS=1`, `train.py`'s own per-step clock; `E4B_OFFLOAD_STATS=1`, P137's staging
  counter; `MODEL` (the fetched snapshot); and `OUT`.
- **Arms.** Only `LAYER_MAJOR` differs:
  - **A0:** `OFFLOAD_EXPERTS=1 FAST_TRAIN=1`, the stock accumulation loop on the fused path;
  - **L1:** `OFFLOAD_EXPERTS=1 LAYER_MAJOR=1`, which implies `FAST_TRAIN=1`.
- **Order:** ABBA, A0a L1a L1b A0b. **Each arm is one fresh process**, as a user runs the CLI: it loads the model, runs
  the BEFORE evaluation, the 34 steps and the AFTER evaluation, then saves its adapter.
  - This undoes P137 Amendment 2 (one process, one load) on purpose: a user's process is the subject.
  - Amendment 2 existed because a fresh load cost about 38 min on a contended host. #1752's cap made that load 64.8 s
    on the same host class (P144).
- **What the box adds** (`p146_box.py`). Nothing in `train.py` changes. Before calling `main()`, the arm process wraps:
  - `load_dataset`, to serve the staged parquet;
  - the clock's `step_end`, to add each step's staging count and peak allocated memory to its row and reset both;
  - `enable_fast_train`, to record the patched and offloaded layer counts and the load time;
  - `torch.optim.AdamW`, to hash the adapters it is built over;
  - `train.eval_loss`, to record the BEFORE and AFTER held-out losses.
- **The step wall** is `train.py`'s own. It runs from before `zero_grad` to after `sched.step`, with one CUDA sync at
  the end of the step. It includes the batch assembly, the gradient clip and the optimizer.
- **Card:** the RTX 4090 class, compute capability 8.9. Any other card is refused (rc 15).
- **Host floors** (P137's): 120 GB of disk and 64 GB of host RAM.
- **Premise:** at the pin, `tests/test_layer_major.py`, `test_offload_hold.py`, `test_offload.py` and
  `test_train_fast_train.py` (the CLI's own) give exactly **57 passed**, none skipped. That count was read on an A2000
  at `4917506a`.

## The rule (`p146_reduce.py`, 43 self-test cases)

**VOID, STRICT**, if any of these holds:
- an arm record is missing (a process that crashed, OOMed at the budget floor or timed out writes none);
- off-pin or mixed stacks, or a dry record in a reading (or the reverse);
- the fused path not patched, or the experts not offloaded, on exactly n = 48 layers;
- e4b's reentrant checkpoint not on all 48 layers. This is the order #1744 fixed: a later
  `gradient_checkpointing_enable` would put Hugging Face's checkpoint back;
- an arm's switches or recipe not the registered CLI recipe;
- **`train.py`'s OOM back-off halved the token budget** in any arm, so the arms would carry different batches;
- the alpaca rows requested other than as `train[:2000]` and `train[2000:2064]`;
- **the arms did not start from the same adapters**: the sha256 at the optimizer's construction differs. This replaces
  Amendment 2's snapshot check: each fresh process seeds and builds its own adapters;
- **the BEFORE held-out losses are not bit-identical** across the four processes;
- step 0's loss is not bit-identical across arms. The stock loop sums each micro-batch's loss / 4 in order; layer-major
  divides the sum by 4. A power-of-two scale commutes with rounding, so equal per-micro-batch losses give equal step
  losses bit for bit;
- **step 0 not cold:**
  - A0a and A0b exactly M × (2n − 1) + 1 = 381;
  - L1a equal to L1b and above 2n = 96. Each fresh L1 process runs the drift guard's one-time stock forwards in its
    first step, as Amendment 2's L1a did;
- steady stagings not exactly 380 (A0) and 95 (L1) per step;
- an arm with other than 32 timed steps;
- L1's peak above A0's by more than **4.25 GiB**;
- a self-pair median ratio outside **[0.97, 1.03]**. Across fresh processes this reads process-to-process variance;
  it is never re-banded.

**Otherwise:** LICENSED iff S = min(A0a/L1a, A0b/L1b) ≥ **1.10**, and NO_GAIN if not.

**REPORTING:** the AFTER held-out |L1 − A0| ≤ **0.01**, and S inside the predicted band **[1.5, 2.6]**.

**The memory term at the CLI's shape.** P137's 4.25 GiB term is (M − 1) × [checkpointed layer inputs + saved fp32
log-probs]. A CLI micro-batch carries at most 2048 padded tokens:
- checkpoints: 48 × 2048 × 2048 × 2 B = 0.375 GiB;
- log-probs: under the `auto` chunked loss, a forward whose fp32 logits would reach 1 GiB is chunked, so they stay
  below 1 GiB;
- so the worst case is 3 × (0.375 + 1.0) = 4.13 GiB, inside the term.

## The prediction, set a priori from the host link

**Model** (P140's). A0 = C + 4T and L1 = C + T, so S = (C + 4T) / (C + T).
- **T, the copy:** Qwen3-30B-A3B stages 32.3 GB per pass (2n − 1 = 95 stagings, P140's figure). The pass is the same
  under the CLI. At r = 17–25 GB/s on 4090 hosts (P137's fit gave 22.1), **T = 1.29–1.90 s**.
- **C, the non-overlapped compute:** this is where the CLI differs.
  - A CLI micro-batch packs up to 2048 padded tokens of rows of at most 192 tokens. P137's carried two alpaca rows.
  - A fused-MoE step's cost is largely fixed per active expert (`train.py`'s own note on `TOKEN_BUDGET`), so C grows
    more slowly than the token count.
  - Attention LoRA, the gradient clip and the batch assembly add to C. The stock loop's per-micro-batch `loss.item()`
    syncs add to A0 only.
  - **C = 1.7–6 s.** Its width is the main uncertainty.

**Corners:**

| C (s) | T (s) | S |
|---|---|---|
| 1.7 | 1.90 | 2.58 |
| 1.7 | 1.29 | 2.29 |
| 6 | 1.90 | 1.72 |
| 6 | 1.29 | 1.53 |

**Predicted band: S in [1.5, 2.6], point about 2.0** (REPORTING). The box records `lscpu`, the container's CPU quota
and the thread cap (the load log), and the pinned host-to-device ceiling, so a miss can be read against C and r. None of
them re-bands.

## Predictions (written before any data)

- **Q1.** Every arm fits in 24 GB at `TOKEN_BUDGET` 2048 without the OOM back-off.
- **Q2.** Every arm starts from the same adapters, and the BEFORE held-out and step 0 are bit-identical across the four
  processes. Steady stagings are exactly 380 and 95.
- **Q3.** S is in [1.5, 2.6].
- **Q4.** L1's peak is within the 4.25 GiB term of A0's.
- **Q5.** The AFTER held-out |Δ| is within 0.01 (REPORTING).
- **Q6.** The self-pairs hold across fresh processes. This is the lane's main risk: no lane has read a cross-process
  self-pair under Amendment 3's n.
- **Q7.** LICENSED, about 80 %.

## Consequence, registered now

- **LICENSED:** a claim row for the CLI trainer (`OFFLOAD_EXPERTS=1 LAYER_MAJOR=1` against `OFFLOAD_EXPERTS=1
  FAST_TRAIN=1`) on Qwen3-30B-A3B, the RTX 4090 class, at `train.py`'s default recipe. The STATUS sentence "its own
  recipe ... is unread" is replaced by the number. No default changes; the claim's scope is the CLI only.
- **NO_GAIN:** the STATUS sentence records the CLI number and the host-link fit; the library licences stand as they are.
- **VOID:** under P137's stop rules. Two consecutive box-started HARNESS_ERROR or VOID results stop the lane until a
  simplification amendment lands. A self-pair VOID (Q6) would be answered by a design amendment, not a rerun.

## Runs, cost and the guards

- **Run ids:** `p146-4090-1` is the reading. `p146-4090-2` to `p146-4090-4` are spares under the flicker rule: an id
  refused before the create, or NOT_RUN at the pre-flight, is consumed and is not an attempt. `p146-dry-<n>` are the $0
  dry runs.
- **Spare ids** (added after `-1` to `-4` were spent at $0.001 in total, with no box-started run): `p146-4090-5` to
  `p146-4090-8`, for pre-create refusals, listings gone before the create, and NOT_RUN at the pre-flight only. No term changes.
- **Guards:** guard 3.0 h.
  - The fetch check is 3,300 s: 61.1 GB at the launcher's 20 MB/s floor is about 3,050 s.
  - The arm check is 6,000 s: four processes of at most 1,500 s each. Each is a load (about 65 s on a capped host),
    two 64-row evaluations under offload, 34 steps and an adapter save.
  - Fetch check, arm check and a 600 s margin come to 9,900 s, inside the guard with setup.
- **Expected box time** is about 45 min: a 61 GB fetch (about 10 min) and four processes of about 6–8 min.
- **Cost:**
  - Rate cap $0.75/h for the GPU and 200 GB, download at most $0.011/GB.
  - **Worst $2.92**: $0.75 × 3.0 h, plus 61.1 GB × $0.011.
  - Expected about $0.6–1.2.
  - Under the lane's standing authority, at most $15 per run, after this registration merges.
- **Before any box:** a $0 dry run (`p146-dry-1`) of this kit on the A2000 with a tiny Qwen3-MoE fixture
  (`p146_tiny.py`), for correctness only.
- **Launches** use the launcher current at launch, `--validate` passing, the market rule (up to three stable candidate
  machines, ranked, the worst cost at the dearest), and rsync before any hand abort. The receipt is committed within
  minutes of teardown.

P137's history (Amendment 1: staged data; Amendment 2: one process; Amendment 3: 32 timed steps and both warm-ups) is
inherited where named above: see `bench/p137/PREREG-p137.md`.

## Amendment A1 (after `p146-4090-5`, VOID, strike 1): a host-quiet guard

**What happened.** `p146-4090-5` ran on an RTX 4090 for $0.489. The box was machine 10384, an EPYC host with 192 CPUs
under a 23.04-CPU container quota and a pinned H2D ceiling of 20.28 GB/s.
- It VOIDed on the self-pairs alone: A0b/A0a 1.0457 and L1b/L1a 1.0967, against [0.97, 1.03] (Q6, the registered main
  risk).
- Every other STRICT rung held across the four fresh processes: the adapter sha, the BEFORE held-out and the step-0 loss
  were bit-identical; stagings were 381 / 380 and 336 / 95; the reentrant checkpoint ran on 48/48; the budget stayed
  2048; each arm ran 32 timed steps.
- The arms that moved were the later two. A0a and L1a ran 20:59–21:19Z, tight, and their step walls tracked each other
  step for step (correlation 0.95: the batches are the same in every arm). From about 21:19Z, L1b and A0b were slower and
  noisy, and that correlation vanished (0.24 and −0.05). The host changed under the run; nothing a process carries did.

**Seen, disclosed, never cited** (a VOID): medians 12.533 / 8.266 / 9.065 / 13.106 s, pairs 1.516 and 1.446. On the
first pair the host-link fit gives T 1.42 s, inside the predicted 1.29–1.90, and C 6.84 s, above the predicted
1.7–6 s. The band [1.5, 2.6] stays as registered (REPORTING, never re-banded).

**The change (design terms, fixed before any data under them).** A guard that separates "the host changed under us"
(**VOID_BY_HOST: no reading, not a strike**, and the run id becomes a spare) from "the arms are unstable" (VOID, a
strike). The self-pair band is unchanged and still decides on a quiet host.
- **CPU, per arm, over its timed window** (from the end of warm step 2 to the end of the last step):
  - **throttling:** the share of the box cgroup's CFS enforcement periods in which it was throttled, Δ`nr_throttled` /
    Δ`nr_periods` (`cpu.stat`, v2 or v1; 0 when no period elapsed, i.e. no quota is enforced). A `cpu.stat` that
    reads but has no period counters (cgroup v2 with the cpu controller not delegated, so no CFS quota exists) is
    recorded as `not_enforced`, counted as 0, and never refused; only a wholly unreadable `cpu.stat` refuses. The review pass
    corrected a first form: the throttled time (`throttled_usec`) is summed by the kernel over every runqueue the group
    ran on, so per wall second it reads "CPUs throttled on average" and grows with the CPU count. That figure is kept
    as REPORTING (`throttled_cpu_s_per_wall_s`);
  - **steal:** `/proc/stat` steal as a fraction of all CPU time on the aggregate `cpu` line.
  - **VOID_BY_HOST if either exceeds 5 %.** The argument rests on the band, not on any box: the self-pair band allows
    3 %, so a host that takes more than 5 % of a window can by itself move a median past the band. A quiet box shows
    neither signal. `p146-4090-5` recorded neither, so no seen value sets the threshold.
- **CPU pressure (PSI), a guard where it is readable** (the maintainer's ruling: neighbour run-queue contention, which
  is what `-5` looked like, shows as neither throttling nor steal on bare metal):
  - the "some" total of one file fixed once for all four arms (the cgroup v2 `cpu.pressure`, else
    `/proc/pressure/cpu`), as a share of each arm's timed window;
  - **VOID_BY_HOST if |A0b − A0a| > 5 points or |L1b − L1a| > 5 points.** The self-pair form asks "did the host change
    between the replicate arms". A four-arm spread could VOID on the lever itself, since A0 and L1 do different host
    work by design, so their own baselines may differ;
  - unreadable at the pre-flight: recorded as `psi: absent`, and the box runs on with the other signals (a host booted
    with `psi=0` is not excluded). Recorded in some arms but not all, or from different files, is VOID_BY_HOST.
- **Link:** one pinned host-to-device probe (`engines.offload.report_offload_environment`, 256 MB × 20) in its own
  process before the first arm and after every arm. That is five probes: `before_A0a`, then `after_A0a`, `after_L1a`,
  `after_L1b`, `after_A0b`.
  - **VOID_BY_HOST if any probe is more than 15 % below the first.** 15 % is A11's two-sample agreement tolerance for
    that probe. T is bound by the host link, so the link is the quantity that moves S.
- **A guard signal that was not recorded** (an unreadable `cpu.stat` or `/proc/stat`, a missing or failed probe) is
  also VOID_BY_HOST: the host cannot show it was quiet.
- **Pre-flight (the maintainer's pre-pass):** right after the box starts, before the install and the fetch, the box reads
  `cpu.stat` (with its period counters) and `/proc/stat` once, names the PSI source (or `absent`), and runs one
  pinned H2D probe in plain torch. If `cpu.stat`, `/proc/stat` or the probe is unreadable it refuses
  (rc 29, `BOX_REFUSED host_guard=unreadable`): NOT_RUN at the pre-flight, at minimal cost and not a strike, so no
  box is spent on a certain VOID_BY_HOST.
- **Order of the rule:** the structural STRICT rungs first (a broken arm is a strike on any host), then the guard,
  then the self-pair band, then S.
- **Pin preference** (not a floor): prefer offers whose CPU allotment is the whole host or close to it (`-5` had a
  23-CPU quota on a 192-CPU host). Each PIN post records the allotment against the host's cores.

Unchanged: S ≥ 1.10, the self-pair band [0.97, 1.03], the band [1.5, 2.6], every rung above, the 3.0 h guard, worst
$2.92 per box.

**Order.** This amendment's review, a $0 dry run on the A2000 that shows both signals recorded, then one box on a spare
(`p146-4090-6` or `-7`). A second box-started VOID or harness error stops P146 until a simplification lands; a
VOID_BY_HOST is neither.
