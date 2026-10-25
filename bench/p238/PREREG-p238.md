# P238: grouped-nf4-gemm 0.51.0's fused NF4 training GEMM against Unsloth's NF4-fused grouped GEMM (unsloth-zoo#1935), on Qwen3.6-35B-A3B's experts, at op level and per training step, on one RTX 5090 (a head-to-head at the released pins; descriptive; registered before any run)

Lane number claimed by `p238/registration`. A head-to-head the project was asked for: both libraries now fuse the NF4 decode into
the grouped expert GEMM of QLoRA training, and no reviewed harness has raced the two. It is **descriptive**: it reports ratios
beside their draw bands, it licenses nothing, and no default moves on it.

## The question

When both libraries train LoRA adapters on the routed experts of Qwen3.6-35B-A3B over the SAME NF4 bytes, how long do their expert
GEMMs take, how long does a training step take, how much memory and energy do they use, and do the losses agree? It is asked at op
level (one layer's expert block, as training calls it) and per optimizer step (the whole model), each with and without each
library's fused path.

## The arms

| arm | library | the knob, set explicitly | what it selects |
|---|---|---|---|
| **E_F** | experts4bit-qlora 0.58.1 + grouped-nf4-gemm 0.51.0 | `GNF4_TRAIN_GEMM=fused` | the fused NF4 grouped GEMM on every expert call (`nf4_route.route_for`, nf4_route.py:55-135) |
| **E_A** | the same | `GNF4_TRAIN_GEMM=auto` | the library's default: on sm_120 a call with 1 to 16 present expert groups takes `dense` (dequant + `torch.mm`), more take `fused` |
| **U0** | unsloth 2026.10.8 + unsloth_zoo 2026.10.8 | `UNSLOTH_MOE_NF4_FUSED=0` | the dequant path: the stack is dequantized, then the generic Triton grouped GEMM |
| **U1** | the same | `UNSLOTH_MOE_NF4_FUSED=1` | `maybe_nf4_grouped_mm` (moe_nf4_grouped.py:521) on every frozen base GEMM, forward and dX |
| **UA** | the same | `UNSLOTH_MOE_NF4_FUSED=auto` | as `1` while the call's rows are at most `UNSLOTH_MOE_NF4_AUTO_MAX_ROWS` (128) x E, else the dequant path (moe_nf4_grouped.py:69-81) |

- **#1935** is unslothai/unsloth-zoo#1935, merged as `4fafdb26`. Its `moe_nf4_grouped.py` is byte-equal in unsloth_zoo 2026.10.8,
  and moe_utils.py carries its four hooks (1183, 1185, 5322, 5521). The 128-row crossover was measured on an L4 in bf16 (its
  docstring, line 21): context only, never a prediction here.
- **On this model E_A is expected to equal E_F almost everywhere.** At 256 experts, top-8 and >= 64 rows per expert, nearly every
  call presents more than 16 groups. The arm is kept because it is the library's default, and the per-route counts are reported per
  draw so a reader sees how often `dense` ran.
- **UA at the 4,096-token point sits exactly AT its cap:** 4,096 x 8 = 32,768 rows = 128 x 256, and the cap is inclusive (`<=`), so
  UA fuses there and not at 8,192 tokens.
- **Every other knob of both libraries is set explicitly** to the value the comparison is about, and recorded:
  `UNSLOTH_MOE_NF4_AUTO_MAX_ROWS=128`, `UNSLOTH_MOE_NF4_AUTOTUNE=0` (fixed tiles, line 316), `UNSLOTH_MOE_GROUPED_TRITON=auto`.
  Every other `UNSLOTH_*` key the three zoo modules read (found by scanning their source, `check_env` in
  `bench/common/unsloth_moe_counter.py`) is unset, and every grouped-nf4-gemm and e4b knob is at its default except
  `GNF4_TRAIN_GEMM`.

## What is trained: LoRA on the routed experts, rank 8, on both sides

**Why not attention-only LoRA.** Released e4b fuses the frozen expert GEMM in training **only when the experts carry LoRA**:
- `enable_fast_train` patches `ExpertsLoRA` wrappers only (`experts4bit_qlora/engines/fast.py:925-958`);
- the inference patch on a bare `ExpertsNbit` falls back to the reference forward whenever grad is enabled and the hidden states
  require grad (`fast.py:213-223`), which holds at every expert layer once an adapter sits upstream.

With attention-only LoRA, e4b's E_F and E_A would both time the reference dequant path, while Unsloth's #1935 hook sits on the base
GEMM whether or not the experts carry adapters (moe_utils.py:5303-5322; `_base_is_recomputable`, 943-965, needs only a frozen
`Params4bit`). That asymmetry is stated in the results as an e4b property, and a fused frozen-base training forward for LoRA-less
experts is recorded as a candidate e4b feature (unscheduled, the project's own issue tracker).

**The adapter set, identical on both sides** (the matched set of TC2, at rank 8):
- the routed experts' `gate_up_proj` and `down_proj` in all 40 layers, and attention `q_proj, k_proj, v_proj, o_proj` (UT4) where
  the layer has them;
- rank **8**, alpha **16** (scaling alpha / r = 2), dropout 0, fp32 adapters, `--lora-init matched:3407` (TC1 T20: every A drawn
  per slot from one seeded stream, every B zero).

**Rank 8 is a registered deviation from TC2's rank 16.** TC2's matched set at rank 16 (926,187,520 fp32 adapters) ran e4b's fused
path out of memory resident on 32 GB at micro-batch 2 (`tc1-5090-27`, TC2 amendment 4). Rank 8 halves the adapters (about 463 M).
A point that runs out of memory at rank 8 is reported OOM (TC1's verdict), never shrunk.

**The two parameterizations are the same adapter, from source:**

| | e4b 0.58.1 (`experts4bit_qlora/lora.py`) | Unsloth via PEFT 0.21.2 `ParamWrapper` (`peft/tuners/lora/layer.py`) |
|---|---|---|
| per expert or shared | per expert: `gate_up_lora_A [E, r, hidden]`, `gate_up_lora_B [E, gate_up_out, r]`, `down_lora_A [E, r, inter]`, `down_lora_B [E, hidden, r]` (514-517) | per expert: `lora_A = Linear(in, r * E)`, `lora_B = Linear(r * E, out)` (2455-2456); `get_delta_weight` reads A as `[E, r, in]` (2549-2552), so expert e owns its own rank-r pair |
| scaling | `alpha / r` (506) | `lora_alpha / r` (2462; rsLoRA off) |
| dropout | none in the expert delta | 0, and a non-zero value is refused for a `ParamWrapper` (2363-2366, 2444-2447) |
| init before the matched draw | A normal(std 1/r), B zero (519-521) | A kaiming-uniform, B zero; both overwritten by `matched:3407` |

- Per layer the trainable count is E x r x (in + out) on both sides: gate_up 256 x 8 x (2048 + 1024) = **6,291,456**, down
  256 x 8 x (512 + 2048) = **5,242,880**.
- Unsloth reads the PEFT factors back per expert for its grouped GEMM (`_extract_lora_from_wrapper`, moe_utils.py:2806-2865, and the
  Qwen extractor qwen3_moe.py:50-79, registered on `Qwen3_5MoeExperts` at qwen3_5_moe.py:46-47).
- **The shakedown ASSERTS** equal trainable-parameter counts per projection and per layer, equal adapter shapes per expert, the same
  scaling, and the same matched A bytes per slot on both sides. A difference STOPS the lane, and it is brought to the maintainer. It
  is never "close enough".
- **PEFT is pinned** (0.21.2, the version e4b's own TC1 venv pins) because Unsloth leaves it open (`peft>=0.18.0`, unsloth 2026.10.8
  METADATA), and its `ParamWrapper` defines Unsloth's adapter layout.

## The environment and pins

- **One venv for both frameworks**, as TC1 amendment 80: venv-unsloth, torch 2.12.1+cu130 from `unsloth[cu130-torch2121]`.
- **All from PyPI, release-proven:** `experts4bit-qlora` **0.58.1**, `grouped-nf4-gemm` **0.51.0**, `unsloth` **2026.10.8**,
  `unsloth_zoo` **2026.10.8**, `peft` **0.21.2**, `bitsandbytes` **0.50.2**, and `transformers` at the version pip resolves under
  unsloth's cap (`<= 5.17.0`), recorded and proven like the rest.
  - `bench/tc1/release_proof.py` (sha256 `aff3fc80db8d35bbf94a9aaeed125c9d424cfe93bf16eb0e6868b6f9d03f13e0`, loggetta's frozen
    helper) proves every one against PyPI's published wheel: the RECORD compare on disk, the import origin, a fresh bytecode prefix.
    Every install passes `pip --no-compile`. A refusal ends the box (rc 9).
  - The commits the wheels were built from, as code constants: e4b `605d21e67958fa94d5e73fc3a95ffcb5c8eae02d` (v0.58.1),
    grouped-nf4-gemm `30adb1bfa76f95f9ce4a4c29734b8f36de907eca` (v0.51.0).
- **The model:** `Qwen/Qwen3.6-35B-A3B` at revision `995ad96eacd98c81ed38be0c5b274b04031597b0` (qwen3_5_moe: hidden 2048,
  moe_intermediate 512, 256 routed experts top-8, one shared expert of 512, 40 hybrid layers). ONE snapshot download per box,
  shared by both frameworks.
  - e4b builds the native text model (`qwen3_5_moe_text`) and drops the vision tower. Unsloth trains through its vision path
    (`FastVisionModel`, `finetune_vision_layers=False`) with the tower present and frozen. That asymmetry is reported in every peak
    VRAM row, and never corrected for.

## The premise: the same NF4 bytes

- **On CPU, at $0, before this page** (`bench/p238/premise_bytes.py`, committed with it): layer 0's two expert stacks, quantized by
  Unsloth's loader path (bitsandbytes `Params4bit(..., blocksize=64, compress_statistics=True, quant_type="nf4")`, as
  `_make_expert_params4bit` builds it) and by e4b's (`ExpertsNbit._quantize_stack` + `absmax_dq._compress_one`), are equal in
  packed bytes, the fp32 absmax, the nested state (quantized absmax, state2 absmax, offset, state2 code) and the NF4 code.
  **SAME_BYTES** at that scope only.
- **On the box, BINDING on every load (all 40, and the op process):** each load hashes, per layer and projection, its packed
  bytes, its nested absmax state and its code in one canonical element order (`p238_bytes.py`). The reducer requires the 40 loads'
  tables equal, and equal to the op process's table. A mismatch on any load STOPS the box before the next load (rc 27), and its
  first differing layer is reported.

## Level 1: op level (one layer's expert block, in one process)

`bench/p238/p238_op.py`. Layer 0's real stacks (gate_up [256, 1024, 2048], down [256, 2048, 512]), quantized by each library's own
loader path and held to the premise. One process imports both libraries.

**PRIMARY: the expert block as training calls it,** rank-8 expert LoRA included, forward + backward (dX and the LoRA dA, dB):

| arm | the call |
|---|---|
| E_F, E_A | `nf4_qlora.fused_grouped_lora` as e4b's patched `ExpertsLoRA` training forward calls it (`fast.py` `fused_experts_train_forward`), at the arm's `GNF4_TRAIN_GEMM` |
| U0, U1, UA | `moe_utils.forward_native_grouped_mm`'s expert GEMMs as the patched `Qwen3_5MoeExperts.forward` calls them (the base GEMM through `_nf4_fused_base_mm` or `_base_grouped_mm`, plus the separated LoRA through the generic grouped GEMM), at the arm's `UNSLOTH_MOE_NF4_FUSED` |

**SECONDARY, labelled the kernel-only comparison and never in the headline:** the frozen base GEMM alone, forward + dX, through the
same entry points with no adapter.

- **Same counters as training:** each op call must fire the same counters the step-level record fires for that arm
  (dispatch decisions, NF4 launches, `DGRAD_STATS`). An op entry point that differs from the training call VOIDs that op row.
- **The sweep:** rows per expert 32 / 64 / 128 / 256 / 512, each under uniform routing and under a seeded skew (Zipf s = 1.1 over the
  256 experts, seed 238, with at least one expert at 0 rows). The same expert-sorted rows and offsets go to every arm.
- **Capture mode: ONE across all arms.** If every arm can run inside a captured CUDA graph (static inputs kept alive for the graph's
  life), all are timed captured; if any cannot, ALL are timed eager, and the other mode is reported as secondary only. The mode
  is decided by a capture probe on every arm before any timing and recorded.
- **Timing:** 10 untimed warm-up calls per arm, then **20 reps**. Each rep visits the five arms in a palindrome (E_F E_A U0 U1 UA UA
  U1 U0 E_A E_F), odd reps reversed. A visit is 50 calls between two CUDA events; its time is the interval over 50. Wall per call
  from `perf_counter` around the same visits.
- **Numerical parity:** one fp32 reference serves both libraries: bitsandbytes' `dequantize_4bit` of the premise-held bytes in
  fp32, with the LoRA term in fp32. Each arm's forward output, dX and LoRA gradients are within **max|err| <= 2e-2 x max|ref|**
  (P236's bf16 bound). Bit identity between arms and against each arm's own second call is reported as information.
- **Energy:** the NVML total-energy counter (`nvmlDeviceGetTotalEnergyConsumption`, mJ) delta over a window of **>= 10 s** of
  back-to-back calls per arm, J per call = delta / calls, at the primary table's uniform-routing cells. Cross-check:
  `power.draw.instant` sampled every 0.25 s over the same windows, read only with >= 40 samples. If the counter is unsupported on
  the card, the sampled method alone, stated. Energy is DESCRIPTIVE.

## Level 2: per training step (40 loads)

TC1's harness, `bench/tc1/tc1_arm.py`, through the generic observe-only hook (`TC1_ARM_HOOK`, its own reviewed PR):
- `post_load(args, model)` installs `UnslothMoeCounter.patch_loaded(experts)` on the loaded expert modules (the zoo's compile-cache
  copy of moe_utils can appear during the load) and the grouped-nf4-gemm route wrapper;
- `step_edge(i)` snapshots at every optimizer-step boundary and once after the last step;
- `record()` writes P238's engagement records into the receipt.

The hook observes only: every wrapper it installs returns the wrapped function's output unchanged (a mutant per wrapper), and its
module's sha256 is in the box's staged pin. No other tc1_arm edit is made by this lane.

**The step points** (packed TC1 Alpaca rows, `--pack 1`, seq 2048, no padding inside a shape; `--accum 1`, so one optimizer step is
one micro-batch):

| point | micro-batch | tokens per step | rows per expert |
|---|---|---:|---:|
| 2k | 1 x 2048 | 2,048 | 64 |
| 4k | 2 x 2048 | 4,096 | 128 (UA's cap, inclusive) |
| 8k | 4 x 2048 | 8,192 | 256 |
| field | TC1's field micro-batch: 2 x <= 2048, dynamically padded (`--pack 0`, micro-batch 2) | as the batch | as the batch |

32 and 512 rows are op level only: 1,024 tokens is too small a training step to time, and 16,384 tokens risks OOM.

**ONE LOAD PER POINT:** 5 arms x 4 points x 2 draws = **40 loads**, each a fresh process. The schedule is fixed:
- **draw 1** runs the points ascending (2k, 4k, 8k, field), each point's arms in the order E_F E_A U0 U1 UA;
- **draw 2** runs the points descending (field, 8k, 4k, 2k), each point's arms reversed (UA U1 U0 E_A E_F), so each point's two draws
  form an ABBA palindrome per arm.
- Every row records its global load index (1..40) and its order position within its point.

**Each load:** 60 optimizer steps, `adamw_8bit`, lr 2e-4 constant, each framework's gradient checkpointing as TC1's field recipe
sets it (`--grad-ckpt unsloth`), bf16 autocast, the same packed batches in the same order for every arm at a point. Profiled steps 21..23 (counting from 1,
`--profile-steps 3 --profile-warm 20`), as amendment 80.

**What is measured per load:**
- **wall s/step:** the median over steps 11..60 (the profiled steps sit inside it, a known overhead a median absorbs, as amendment 80);
- **device time, kernels only,** of the three profiled steps, by **amendment 81's at-source method** (the `device_ms_method` marker
  in the profile). ONE method; there is no fallback. **The reading launches only after amendment 81 merges.** The shakedown may run
  before it, since it is not read;
- **peak VRAM:** `max_memory_allocated` over the process (each load is fresh), with Unsloth's frozen vision tower reported beside it;
- **energy:** the NVML energy-counter delta over steps 11..60 (one window, >= 10 s at every point), J per step and J per token;
  cross-check as at op level;
- **loss:** step-0 loss, every step's training loss, held-out loss after step 60 (`--eval-n 48`).

## Engagement (every arm, every load; each fails closed)

- **E_F:** `FAST_TRAIN_STATS["patched"]` == 40 and `skipped` empty; `DGRAD_STATS["kernel"]` == the expected dX calls (2 per
  expert-module forward under grad); `ROUTE_STATS` dense, grouped_mm and decoded all 0; the profiled `_gemm_nf4_grouped` kernel count
  == the expected forward GEMMs in the profiled steps.
- **E_A:** the same patch and dgrad checks; a module-attribute wrapper on `nf4_route.train_gemm_route` logs (n_groups, route) per
  call; every route == `route_for`'s answer for its own n_groups; `ROUTE_STATS` dense == the logged dense calls; the per-route counts
  are reported.
- **All Unsloth arms** (`bench/common/unsloth_moe_counter.py`, installed by `patch_loaded` at post_load):
  - `check_env` on the source-scanned keys against the arm's registered pins; `check_library`; `check_dispatch` with
    `base_must_be_triton=True` (every point is inside every class limit on sm_120);
  - the counter records each expert type's forward and the module object that serves it; a served object that decides no GEMM, a
    GEMM decided through an object that serves no forward, or an expert left on a forward without the predicate VOIDs.
- **U0:** `check_unfused` (no fused launch, no fused call), and F15 v2 (`bench/tc1/f15v2.py`) at amendment 80's Triton-path bars.
- **U1:** `check_fused` with expected launches = 2 x F + 2 x G, where F is the expert-module forwards (under checkpointing, the
  no-grad forward and its recompute) and G the forwards under grad (each adds a dX per projection); the wrapper's fused calls ==
  the in-op `launches` (`check_equal`).
- **UA:** `check_auto` plus `check_fused(snap, 2 x fused_calls_expected(snap, "auto") ...)` by the same arithmetic over the calls the
  library's own predicate admitted. At 8k (above the cap) it reads like U0, by design.
- **The shakedown proves every one of these on a real record.** If any arm's binding counter does not fire on a real step there:
  STOP and report. No substitution.

## Validity, quality and the rule (`bench/p238/p238_reduce.py`, self-tested)

**VOID a row** (an op cell, or one load) when any of these holds; **READ** otherwise. Each check fails closed: a missing record is a
fault.
- the record is missing, a dry run, or not `ok`; another library version, torch, PEFT, bitsandbytes or card (an RTX 5090, cc 12.0,
  >= 160 SMs); a release proof missing or refused;
- the premise fails on that load (the box stops);
- an engagement check above fails;
- the adapter-match assertions fail (the box stops);
- **the card floor on every load:** `bench/common/card_clock.py`'s 20 s burn before any install (busy SM median >= 2700 MHz; no hard
  reason on any sample; SwPowerCap clear on >= 95 % of busy samples; rc 18 otherwise), and each load's own clock log at the same bars;
- op level: a parity miss, or a timing sample missing, non-positive or not finite;
- step level: fewer than 60 steps, a step that is not one micro-batch of the point's token count (the step contract, by step count
  and token count), or a profile without amendment 81's `device_ms_method`.

**Quality (registered bars):**
- **Within a framework** (only the expert GEMM path changes): step-0 loss within **1e-3**; held-out after step 60 within **0.01**
  (U1 and UA against U0; E_A against E_F).
- **Across frameworks** (attention and norm kernels differ too): step-0 within **0.01**, TC1's registered cross-framework bar
  (`bench/tc1/TC1-PREREG.md`:63); held-out within **0.05** (TC1's COMPARABLE).
- A miss is reported beside the row's speed ratio as QUALITY_FAIL, never hidden. A step-level speed ratio is quoted only between two
  READ rows whose quality holds.

**Stability:** the two draws of a (point, arm) within 5 % (e4b) or 10 % (Unsloth) on wall and device time, else the pair is flagged
UNSTABLE and kept out of the headline (TC1's rule).

**Reported, never a bar:** per row the ratio medians and draw bands; per load the order position, global index, load wall and the
Unsloth Mega-cache line; the E_A route counts; op-level bit identity; the energy cross-check.

## Readings

- **The headline:** per step point, E_F against U1 (fused against fused) and E_A against UA (each library's auto), in wall s/step
  and device time, with peak VRAM and J/step beside them, losses and wins together.
- Also: U1 against U0 (Unsloth's own fused gain), E_F against U0, E_A against E_F (expected near 1, stated in advance).
- The op tables (primary, then the labelled secondary) per rows-per-expert and routing.
- **An OOM is a result.** If 8k OOMs on both sides, the step read covers 2k, 4k and field. If it OOMs on one side only, that is a
  READ memory difference, reported as such.

## Predictions

Written before any P238 data. Expectations, not the sizing basis.

| # | prediction |
|---|---|
| P1 | at op level, U1 is faster than U0 in device time at <= 128 rows per expert, and the gap shrinks above |
| P2 | E_A is within 2 % of E_F at every step point and op cell (auto takes `fused` on nearly every call here) |
| P3 | the premise holds on all 40 loads (SAME_BYTES) |
| P4 | no arm VOIDs on engagement |
| P5 | the adapter-match assertions hold (the source says the parameterizations are identical) |

The cross-framework direction is not predicted, only reported.

## Consequence, registered now

**READ:** `RESULTS-p238.md` and a head-to-head row "grouped-nf4-gemm fused vs unsloth-zoo NF4-fused (QLoRA training, Qwen3.6
experts)" in `docs/REPRODUCE.md`, with the wins and the losses together and the e4b property above stated plainly. No default
moves, and no claim beyond the registered points and cells. **VOID** rows are reported as VOID with their reasons; the one rerun
path is below.

**Scope.** Expert-LoRA QLoRA training at rank 8 on one model, one card, two libraries at their released versions. Nothing about
attention-only training (except the stated e4b property), other ranks, other models, other cards, or serving.

## Run ids, order and the stop

- **The shakedown first:** `p238-shake-1` (retry ids `p238-shake-r1`, `p238-shake-r2`). It runs the preflight (both frameworks load
  Qwen3.6 text-only at the pins and train it), the premise on every layer, the adapter-match assertions, every engagement counter on
  a real record, one op cell and one load per arm at the 2k point. It is never read. It measures the per-load wall, from which the
  reading's guard is fit. If either framework cannot train the model: STOP and report.
- **Then the reading:** `p238-5090-1`, its one rerun `p238-5090-2`; retry ids `p238-5090-r1` to `r3` for a box that does not start.
  - **The split rule:** the reading's fit is 40 x the shakedown's measured per-load wall, plus the op process, plus a 25 % headroom.
    If that exceeds the 8 h guard, or its worst cost exceeds USD 15, the reading is SPLIT across two boxes **by draw, never by arm**
    (`p238-5090-1a` draw 1 with the op process, `p238-5090-1b` draw 2), the card floor checked on each box. The split is decided
    from the shakedown's receipt and stated before the reading launches.
- A box is started at its `STARTED` marker (after the refusals and the pre-install fit). Two strikes end the lane. The runner's exit
  codes follow P236's table (13, 15, 16, 18 host refusals before `STARTED`; 41 fit; 9, 10, 28, 78, 130 halts; 27 VOID).

## Budget

- **One RTX 5090 per box** (Vast verified/secure, cc 12.0, >= 160 SMs, VRAM >= 30,000 MiB; host RAM and disk at TC1 amendment 80's
  Qwen3.6 floors), at most **USD 1.05/h** (the program's 5090 policy rate on 2026-10-23), downloads at most USD 0.02/GB (about 72 GB of model plus
  about 10 GB of installs per box).
- **Shakedown:** about 1.5 h; worst 2 h x USD 1.05 + 82 GB x USD 0.02 = **USD 3.74**.
- **Reading:** estimated 40 loads x about 8 minutes + the op process (about 1 h) = about 6.3 h; worst at the 8 h guard
  8 x USD 1.05 + USD 1.64 = **USD 10.04** on one box, or about USD 6 per box when split. Fit from the shakedown, as above.
- **Retries:** a box that does not start costs at most its boot slack, about USD 0.25.
- **Lane worst:** USD 3.74 + 2 x USD 10.04 (a reading and its rerun) + 3 x USD 0.25 = **USD 24.57** over four runs; every run is under
  USD 15. Expected: about USD 11.

## Launch

- **One exact launch commit per box:** this registration's merge (or a later amendment's), through pod-launch with `--validate`
  passing there, the host pinned, rent-watch armed; then the maintainer's Approved naming that commit and the kit's hash.
- **The reading launches only after** amendment 81 (device time at source) and the tc1_arm hook PR have merged.
- **STOP-4:** the driver refuses a dirty tree, an `E4B_SHA` override, an unregistered run id or a staged file that differs from
  `staged-p238.sha256`; its fetch passes `rsync --no-links`.

## What was seen before this page (stated, not hidden)

- No GPU number from either fused path on this model or card.
- The CPU premise (above), on one layer, from the pinned checkpoint's shards (sha256-checked against the hub's etags).
- From source only: the arms' dispatch, the counters' increment sites, the adapter parameterizations, and the e4b property above.
- Locally, on CPU stand-ins at the real module paths: the shared counter's 35 tests (`tests/test_unsloth_moe_counter.py`).

## What this lane cannot say

- Nothing about attention-only training speed (e4b does not fuse there; stated as a property, not timed).
- Nothing about other ranks, other models, other cards, serving, or either library's routes not taken here.
- Nothing about the 128-row crossover on other cards: it is measured here on one RTX 5090 only.

## Receipts

Fetched to the run directory's `p238/` and committed to `bench/p238/receipts/<run>/`: the op record, every load's tc1_arm receipt
and profile, `verdict_p238.json`, `summary.txt`, `versions.txt`, `logs/`, with `SHA256SUMS`. The card's identity
(`forensics.txt`, `outer.log`) is kept only in the private receipt store and named in `STORE-ONLY.sha256`.

## The kit (reviewed in this PR before the maintainer's pass)

`bench/p238/`: `premise_bytes.py` (the CPU premise, as run), `p238_bytes.py` (the per-load premise table), `p238_op.py` (level 1),
`p238_hook.py` (the observe-only `TC1_ARM_HOOK` module), `p238_run.sh` (the box), `p238_drive.sh` (the driver), `p238_reduce.py`
(the rule, with its self-test), `staged-p238.sha256`; `tests/test_p238.py` (the pins, the schedule, the reducer's cases, a dry run of
the driver for every registered id, and the hook's output-equality mutants).
