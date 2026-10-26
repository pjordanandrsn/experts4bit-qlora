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
  `UNSLOTH_MOE_NF4_AUTO_MAX_ROWS=128`, `UNSLOTH_MOE_NF4_AUTOTUNE=0` (fixed tiles, line 316), `UNSLOTH_MOE_GROUPED_TRITON=auto`,
  and `UNSLOTH_MOE_FUSED_ROUTER=0` (its shipped default: opt-in, `moe_router_triton.py:69`; it switches the router inside the timed
  step), and `UNSLOTH_ENABLE_LOGGING=1` (TC1's run scripts set it so tc1_arm records the loader's MoE banner lines; it changes no
  computation, and it is the same on all three Unsloth arms). tc1_arm itself sets no other `UNSLOTH_*` key for these arms (no
  `--unsloth-moe-backend`, no speed tilt: `apply_unsloth_knobs`). No other `UNSLOTH_*` key may be SET in an Unsloth arm's environment: `check_env` (`bench/common/unsloth_moe_counter.py`)
  refuses any set key that is not registered at its value, and holds the registered names to a scan of every source file of the
  installed unsloth and unsloth_zoo packages (the per-module counts are in the receipt). The env dump proves each pin.
- **Gradient checkpointing, explicit on both sides:** the Unsloth arms pass `--grad-ckpt unsloth` (the shipped default, made explicit:
  each layer's forward runs once without grad and is replayed under grad in backward); e4b's arms checkpoint HF non-reentrant
  (`gradient_checkpointing_enable(use_reentrant=False)`, `bench/tc1/tc1_arm.py:1996`), where both passes run under grad and one is
  backwarded. The engagement bars below are built BY STRUCTURE, so they hold under either.
- **e4b's frozen absmax is double-quantized: a PIN of the shipped default, not a deviation.** At v0.58.1, nested absmax is
  `enable_fast_train`'s and the CLI trainer's default for resident training (`absmax_dq.py:1-2`, `44-46`), and it is the form
  Unsloth's loader builds (`bnb_4bit_use_double_quant=True`) and the premise compares. The same docstring keeps TC1's switch explicit,
  so both e4b arms set it both ways: `E4B_ABSMAX_DQ=1` in the environment (`enable_fast_train` then REQUIRES it and raises on a
  refusal, `engines/fast.py:878-880`, `1037-1042`) AND `--absmax-dq 1` on tc1_arm, which compresses before the LoRA wrap and
  refuses an arm that compressed nothing (`bench/tc1/tc1_arm.py:1984-1994`).
  - **BINDING per e4b load, else VOID:** the receipt's `absmax_dq.modules` == 40; every expert stack reads compressed
    (`is_absmax_compressed`, 40 of 40); `FAST_TRAIN_STATS["absmax_dq"]` has `required` true and `kept_fp32` null. (Its `compressed`
    count is 0 by design here, since tc1_arm compressed first and the second call is idempotent, `fast.py:1063-1071`.) The premise
    table's nested fields agree, since it fails closed on an fp32 absmax. A reducer mutant: an e4b load without the compression
    VOIDs on both checks.
- Every other grouped-nf4-gemm and e4b knob is at its default except `GNF4_TRAIN_GEMM`.
- **Unsloth's frozen absmax is double-quantized too: its loader default, kept.** tc1_arm's Unsloth arms default to
  `--unsloth-double-quant off` (TC1's P2-1 regime), which would hold fp32 absmax against e4b's nested one and fail the premise on
  every load. So all three Unsloth arms pass `--unsloth-double-quant default` (tc1_arm passes nothing and keeps the loader's
  default, `bench/tc1/tc1_arm.py:376-386`).
  - **BINDING per Unsloth load, else VOID:** tc1_arm's `unsloth_double_quant.loaded_nested` (read off the loaded expert `Params4bit`,
    `loaded_expert_double_quant`) is true, and the premise table's nested fields agree. A reducer mutant: an Unsloth load with fp32
    absmax VOIDs on both checks.
- **Unsloth's loader is named, not discovered:** `--unsloth-loader FastModel` (qwen3_5_moe is a vision model to Unsloth). tc1_arm's
  FastLanguageModel-to-FastModel fallback is then unreachable, and `loader_used` must read `FastModel`. The vision tower stays
  frozen. The UT4 targets name the language model's attention projections, and the trainable-count assertion (equal per projection
  and per layer on both sides) fails if any adapter lands in the tower.

## What is trained: LoRA on the routed experts, rank 8, on both sides

**Why not attention-only LoRA.** Released e4b fuses the frozen expert GEMM in training **only when the experts carry LoRA**:
- `enable_fast_train` patches `ExpertsLoRA` wrappers only (`experts4bit_qlora/engines/fast.py:909-962`);
- the inference patch on a bare `ExpertsNbit` falls back to the reference forward whenever grad is enabled and the hidden states
  require grad (`fast.py:214-228`), which holds at every expert layer once an adapter sits upstream.

With attention-only LoRA, e4b's E_F and E_A would both time the reference dequant path, while Unsloth's #1935 hook sits on the base
GEMM whether or not the experts carry adapters (moe_utils.py:5303-5322; `_base_is_recomputable`, 943-965, needs only a frozen
`Params4bit`). That asymmetry is stated in the results as an e4b property, and a fused frozen-base training forward for LoRA-less
experts is recorded as a candidate e4b feature (unscheduled; e4b#2136).

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
  scaling, and the same matched A bytes per slot on both sides.
- **The layout check with a non-zero B.** PEFT stores B rank-major (`get_delta_weight` reads it as `[out, r, E]`, 2554), and
  Unsloth re-reads it for its grouped GEMM through its own layout resolver. B starts at zero, so neither training nor the A bytes can
  show a mismatch. The check sets every expert's B on layer 0 to the same fixed seeded per-expert values, each written into its
  library's own storage in that library's layout (a view of PEFT's flat B, read as `[out, r, E]`; e4b's `[E, out, r]`). It then
  asserts that each arm's applied forward output, dX and LoRA gradients on a seeded routed input match the ONE fp32 reference
  (which applies `scaling * B[e] @ A[e]` per expert), within the op tolerance. A library that reads B in another layout misses it
  (a tested mutant). B is then restored to zero and checked exactly zero.
  - **It runs in the op processes,** which load the model through the same tc1_arm path as the step loads (Level 1). It changes B
    and runs forwards, so it cannot run inside a step load, whose hook only observes. The step loads' adapter census (counts,
    shapes, scaling and the A bytes per (layer, projection)) is compared with the op processes' by the reducer.
- A difference in any of these STOPS the lane, and it is brought to the maintainer. It is never "close enough".
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
- **On the box, BINDING on every load (the 40 step loads and the 4 op processes):** each load hashes, per layer and projection, its packed
  bytes, its nested absmax state and its code in one canonical element order (`p238_bytes.py`). The reducer requires the 40 loads'
  tables equal, and equal to every op process's table. A mismatch on any load STOPS the box before the next load (rc 27), and its
  first differing layer is reported.

## Level 1: op level (one layer's expert block, on the model training loads)

`bench/p238/p238_op.py`. The op call is each library's **layer-0 experts module of the model as training loads it**: the op process
loads the model through tc1_arm's own load path (`load_e4b` / `load_unsloth`, imported from `bench/tc1/tc1_arm.py`, with the same
flags as that framework's step-level arms, then the same matched init and enable steps), so every patch, wrapper, adapter and
quantized byte is the one the step level trains. The premise table of every op process is held equal to the 40 step loads'.

- **Two op processes per draw, one per framework** (a 35B model each; two in one process do not fit). Each runs all of its
  framework's arms, switching the arm's knob between visits; both knobs are read per call (`nf4_route.py:125`; moe_nf4_grouped.py
  `nf4_fused_mode`, 56-64; moe_utils.py:1185). Draw 1 runs op-e4b then op-unsloth, draw 2 op-unsloth then op-e4b. Within-framework
  ratios come from inside one process; cross-framework ratios come from across processes, both draws.
- A switch of `UNSLOTH_MOE_NF4_FUSED` is checked to take effect on the very next call (the counter's per-call mode read), and a
  shape that enters zoo's `FAILED_SHAPES` in any arm VOIDs that process's rows at that shape (the cache persists across arms).

**PRIMARY: the expert block as training calls it,** rank-8 expert LoRA included, forward + backward (dX and the LoRA dA, dB): the
loaded layer-0 experts module's `forward(hidden_states, top_k_index, top_k_weights)` (e4b's `ExpertsLoRA` patched by
`enable_fast_train`; Unsloth's patched `Qwen3_5MoeExperts.forward` with its PEFT-wrapped stacks), then `backward` of a fixed
seeded upstream gradient, at the arm's knob.

**SECONDARY, labelled the kernel-only comparison and never in the headline:** the frozen base GEMM alone, forward + dX, on the same
loaded layer-0 bytes, through each library's base-GEMM entry point: grouped-nf4-gemm's fused grouped GEMM and its dgrad kernel as
`fused_grouped_lora` calls them (E_F / E_A), `maybe_nf4_grouped_mm` (U1 / UA), and zoo's `_base_grouped_mm` recompute path (U0).
The secondary rows are exempt from the same-counters rule below (they are not a training call, by label), and their entry points
are cited by source line in the kit.

- **Same counters as training:** each op call must fire the same counters the step-level record fires for that arm
  (dispatch decisions, NF4 launches, `DGRAD_STATS`). An op entry point that differs from the training call VOIDs that op row.
- **The sweep:** rows per expert 32 / 64 / 128 / 256 / 512, each under uniform routing and under a seeded skew (Zipf s = 1.1 over the
  256 experts, seed 238, with at least one expert at 0 rows). The same expert-sorted rows and offsets go to every arm.
- **Capture mode: ONE across all arms.** If every arm can run inside a captured CUDA graph (static inputs kept alive for the graph's
  life), all are timed captured; if any cannot, ALL are timed eager, and the other mode is reported as secondary only. The mode
  is decided by a capture probe on every arm before any timing and recorded.
- **Timing:** 10 untimed warm-up calls per arm, then **20 reps**. Each rep visits the process's arms in a palindrome (op-e4b:
  E_F E_A E_A E_F; op-unsloth: U0 U1 UA UA U1 U0), odd reps reversed. A visit is 50 calls between two CUDA events; its time is the
  interval over 50. Wall per call from `perf_counter` around the same visits.
- **Numerical parity:** one fp32 reference serves both libraries: bitsandbytes' `dequantize_4bit` of the premise-held bytes in
  fp32, with the LoRA term in fp32 from the loaded (matched) adapters, computed in each op process from its own loaded bytes (equal
  across processes by the premise). Each arm's forward output, dX and LoRA gradients are within **max|err| <= 2e-2 x max|ref|**
  (P236's bf16 bound). Bit identity between arms and against each arm's own second call is reported as information.
- **Energy:** the NVML total-energy counter (`nvmlDeviceGetTotalEnergyConsumption`, mJ) delta over a window of **>= 10 s** of
  back-to-back calls per arm, J per call = delta / calls, at the primary table's uniform-routing cells. Cross-check:
  `power.draw.instant` sampled every 0.25 s over the same windows, read only with >= 40 samples. If the counter is unsupported on
  the card, the sampled method alone, stated. Energy is DESCRIPTIVE.

## Level 2: per training step (40 loads)

TC1's harness, `bench/tc1/tc1_arm.py`, through the generic observe-only hook (`TC1_ARM_HOOK`, its own reviewed PR):
- `post_load(args, model)` installs `UnslothMoeCounter.patch_loaded(experts)` on the loaded expert modules (the zoo's compile-cache
  copy of moe_utils can appear during the load) and the grouped-nf4-gemm route wrapper;
- `step_edge(i)` snapshots at every optimizer-step boundary and once after the last step. Edge 0 precedes the first train-mode
  forward (the counter's structural count refuses a training pass outside a marked step), and the edges are 0..60, closing steps
  0..59;
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
sets it (above), bf16 autocast, the same packed batches in the same order for every arm at a point. Profiled steps 21..23
(counting from 1, `--profile-steps 3 --profile-warm 20`), as amendment 80.

**What is measured per load:**
- **wall s/step:** the median over steps 11..60 (the profiled steps sit inside it, a known overhead a median absorbs, as amendment 80);
- **device time, kernels only,** of the three profiled steps, by **amendment 81's at-source method** (the `device_ms_method` marker
  in the profile). ONE method; there is no fallback. **The reading launches only after amendment 81 merges.** The shakedown may run
  before it, since it is not read;
- **peak VRAM:** `max_memory_allocated` over the process (each load is fresh), with Unsloth's frozen vision tower reported beside it;
- **energy:** the NVML energy-counter delta over two spans that leave out the profile window, edges 10 to 20 and 23 to 59, i.e.
  J = (E20 - E10) + (E59 - E23) over 46 steps (steps 11-20 and 24-59; >= 10 s at every point), J per step and J per token (tokens
  of the same 46 steps), computed by the reducer. The hook reads the counter at every step edge (observation only). Left out:
  tc1_arm starts the profiler after edge 20 and stops it (and writes the profile) before edge 23, outside the step clock, and steps
  21-23 run under CUPTI tracing, a cost the two frameworks pay differently; the excluded pairs come from the row's own
  `profile_warm` / `profile_steps`, which must equal 20 / 3, else UNREAD. No eval sits in a span: the runner passes
  `--eval-every 60`, so the only eval is the registered held-out after step 60, which runs after edge 59 (and before the closing
  edge 60). A missing edge or counter reads UNREAD, never 0, and so does an included edge pair whose wall exceeds tc1_arm's step
  time (which excludes evals) by more than max(0.2 s, 0.2 x the step): non-step work in a span can make the energy UNREAD, never
  move it;
  cross-check as at op level, over the same two spans;
- **loss:** step-0 loss, every step's training loss, held-out loss after step 60 (`--eval-n 48`).

## Engagement (every arm, every load; each fails closed)

**One rule for both frameworks: the backward BY STRUCTURE.** Each optimizer step backwards every expert module once per micro-batch
(here one), so the expected dX launches are 2 projections x 40 layers x 60 steps = **4,800** per load, counting only the backward
passes whose forward was admitted to the fused path (the auto arms). Forward launches count ALL admitted forwards, recompute and
evals included. Never "forwards under grad": under HF non-reentrant checkpointing both passes run under grad and one is backwarded,
and tc1_arm's eval runs with grad on and backwards none. The per-call under-grad split is recorded beside the structural count as
information only (it includes the grad-on eval forwards, so it never binds).

- **E_F:** `FAST_TRAIN_STATS["patched"]` == 40 and `skipped` empty; `DGRAD_STATS["kernel"]` == 4,800 over the run;
  `ROUTE_STATS` dense, grouped_mm and decoded all 0; the logged routes all `fused`; the profiled `_gemm_nf4_grouped` count == all
  admitted forward GEMMs in the three profiled steps (2 passes x 2 projections x 40 layers x 3 steps, the recompute included).
- **E_A:** the same patch check; `DGRAD_STATS["kernel"]` + `ROUTE_STATS["dense_dgrad"]` == 4,800; a module-attribute wrapper on
  `nf4_route.train_gemm_route` logs (n_groups, route) per call; every route == `route_for`'s answer for its own n_groups;
  `ROUTE_STATS` dense_fwd == the logged dense calls; the per-route counts are reported.
- **All Unsloth arms** (`bench/common/unsloth_moe_counter.py`; the hook installs it at post_import, calls `patch_loaded` with the
  loaded experts at post_load, and `mark_step` at every step edge):
  - `check_env` against the arm's registered pins (above); `check_library`; `check_dispatch(snap, structural,
    base_must_be_triton=True)` (every point is inside every class limit on sm_120), where `structural` is the counter's
    `structural(1, 60)`: each module's training passes inside a step must agree on every decision, else VOID;
  - bnb 4-bit experts are served by the PACKAGE moe_utils (`forward_moe_backend` hands them to `moe_utils_bnb4bit`, whose relative
    import reaches the package, moe_utils.py:2065-2075 and moe_utils_bnb4bit.py:367-376); the counter decides that per module with
    the library's own `_moe_uses_bnb4bit_expert_weights`. A served object that decides no GEMM, a GEMM decided through an object
    that serves no forward, or an expert left on a forward without the predicate VOIDs.
- **U0:** `check_unfused` (no fused launch, no fused call), and F15 v2 (`bench/tc1/f15v2.py`) at amendment 80's Triton-path bars.
- **U1:** `check_fused(snap, expected_launches(snap, structural))`: the in-op `launches` (forward and the separate dX op,
  moe_nf4_grouped.py:384) equals every fused call plus the structural dX (one per fused projection per backward). Also every
  `maybe_nf4_grouped_mm` call fused (`nf4_declined` 0), and the wrapper's fused calls equal the library's eager `calls`
  (`check_library`). The lane's own forward count is independent of `nf4_maybe_calls`: a decline before the call
  (`_base_is_recomputable` false, moe_utils.py:1185) never reaches the wrapper.
- **UA:** `check_auto` (every call the library's own predicate admitted fused, every other declined) plus the same
  `check_fused(snap, expected_launches(snap, structural))`, where the structural dX counts only backwards whose forward was
  admitted. At 8k (above the cap) it reads like U0, by design.
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
UNSTABLE and kept out of the headline (TC1's rule). The reducer reports, per point and pair, the quality bars per draw (a miss reads
QUALITY_FAIL with the ratio beside it, not quoted; UA against U0 carries the within-framework bars without a registered ratio), the
arms that are UNSTABLE, whether the pair is headline-eligible, and each draw's own wall ratio.

**The box's verdict** (the merged K boxes' for a split) is READ only when every op row READs and every step row READs or is a
reproduced OOM; each VOID row is reported with its faults, and a VOID row is never dropped to rescue the rest (the runner exits 27).

**Reported, never a bar:** per row the ratio medians and draw bands; per load the order position, global index, load wall and the
Unsloth Mega-cache line; the E_A route counts; op-level bit identity; the energy cross-check.

## Readings

- **The headline:** per step point, E_F against U1 (fused against fused) and E_A against UA (each library's auto), in wall s/step
  and device time, with peak VRAM and J/step beside them, losses and wins together.
- Also: U1 against U0 (Unsloth's own fused gain), E_F against U0, E_A against E_F (expected near 1, stated in advance).
- The op tables (primary, then the labelled secondary) per rows-per-expert and routing.
- **An OOM is a result.** If 8k OOMs on both sides, the step read covers 2k, 4k and field. If it OOMs on one side only, that is a
  READ memory difference, reported as such. The reducer reads a step row that is tc1_arm's OOM stub (status `oom`, exit 5, with
  its phase) on the registered box (release proof, `env_box.txt`, card) as **OOM**, a READ outcome kept out of every speed
  ratio. An OOM must reproduce: both draws of that (point, arm) OOM, else both draws are VOID (an unstable memory reading). A
  harness error, a hook fault (exit 10) or any other status stays VOID, and an op process has no OOM outcome. The box is READ when
  every op row READs and every step row READs or is a reproduced OOM. Per point and pair, the reducer reports RATIO (both arms
  READ), BOTH_OOM (the point leaves that pair's step read), OOM_DIFFERENCE (one arm's draws all OOM, the other's all READ, with
  the fitting arm's peak VRAM) or VOID.

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
  a real record, one op cell per framework (each in its own op process) and one load per arm at the 2k point. It is never read.
  It measures the per-load wall, from which the reading's guard is fit. If either framework cannot train the model: STOP and report.
- **Then the reading:** `p238-5090-1`, its one rerun `p238-5090-2`; retry ids `p238-5090-r1` to `r3` for a box that does not start.
  - **The box cap** is min(the rental policy's per-run override ceiling, 18 h, which the launcher refuses above; the hours the
    USD 15 run-worst buys at the box's rate after its downloads, 12.7 h at USD 1.05/h). `p238_schedule.py` holds the policy's
    global cap (`POLICY_MAX_WALLCLOCK_H`, 6 h) and the override ceiling (`RUN_OVERRIDE_MAX_H`, 18 h), both pinned by the kit test.
    A box whose guard exceeds the 6 h global cap runs only under the per-run override, which the policy allows only when the
    box's worst is at most USD 15.
  - **The split rule:** a box's fit is its process count x the shakedown's measured per-load wall, plus installs, the snapshot
    fetch, the op timing and the reserve, x 1.25 (the runner's guard fit). The reading runs in the fewest K of 1, 2, 4 or 8
    boxes whose largest box fits the box cap (K = 1 whenever one box fits), decided from the shakedown's receipt
    (`p238_schedule.py --plan <its measured wall>`) and stated before the reading launches. **The lane ceiling is K <= 2**, the
    lane cost approved for this registration: if only K = 4 or 8 fits, the plan reads `NEEDS_APPROVAL` and no reading launches (the measured
    wall goes to a new approval; at the lane bound K = 4 is about USD 128 and K = 8 about USD 247); if no K fits, it
    reads `NO_FIT` and no reading launches. The K boxes are whole (draw, point) blocks, **by draw and point group, never by arm**
    (`p238_schedule.blocks`): K = 2 is draw 1 and draw 2; K = 4 cuts each draw's points, in that draw's run order, into two
    groups of two; K = 8 runs each (draw, point) alone. Every box holds all 5 arms of its points, so every ratio is within one
    box; a draw's two op processes ride in its first box. Box i of K runs as `p238-5090-<R>-<K>-<i>` (R = 1 the reading, 2 the
    rerun), the card floor checked on each box.
  - **The machine:** every box of a split runs on box 1's machine (pinned to it), so a split differs from one box only by the
    re-rents and the stability band stays within one host. Only if that machine cannot be rented within 24 h of the previous
    box's receipt may a later box run on another machine of the registered class, each box passing its own card floor; that
    reading is labelled `cross_host`: its 5 % / 10 % stability band is reported as a CROSS-HOST band (conservative: it can only
    flag more pairs UNSTABLE), its ratios stay within one host (each box holds both arms of its points), and each draw's own
    ratio is reported beside the headline.
  - **The reduction:** a box of a split reduces its own faults only (its release proof, `env_box.txt`, card, clocks and rows;
    `--mode part`) and exits 0 with PART_OK, or 27 on a fault; a PART_OK is never a strike. The reading is the controller's
    merge of the K fetched boxes (`--work <box 1> ... <box K>`): each is exactly one of the K blocks, together they cover the
    44-process schedule exactly once (no gap, no index twice), and the pair checks, the OOM reproduce rule and the readings run
    on the merged rows. Each box's machine comes from the controller's store-only rental record beside it and only sets
    `cross_host` (a box without one VOIDs the merge); the verdict and any public text carry `cross_host` yes / no and never a
    machine, instance or GPU id.
- A box is started at its `STARTED` marker (after the refusals and the pre-install fit). Two strikes end the lane. The runner's exit
  codes follow P236's table (13, 15, 16, 18 host refusals before `STARTED`; 41 fit; 9, 10, 78, 130 halts; 27 VOID). 78 includes an
  ambient knob: any `UNSLOTH_*`, `E4B_*`, `GNF4_*` or `NF4_QLORA_*` key already set on the box, other than the five Unsloth pins the
  runner itself sets and `E4B_SHA` (the launch commit the driver hands in), is refused before `STARTED` (both frameworks alike).
  The reducer also reads the box's `env_box.txt` (those prefixes and `PYTORCH_CUDA_ALLOC_CONF`, written before the first process):
  its key set must EQUAL the five Unsloth pins, `PYTORCH_CUDA_ALLOC_CONF` and `E4B_SHA` (the per-arm `GNF4_TRAIN_GEMM`,
  `E4B_ABSMAX_DQ` and `UNSLOTH_MOE_NF4_FUSED` are set only on each arm's own command line); anything else, or no file, VOIDs.
  The refusal, the dump and the reducer's set come from one list (the reducer's `ENV_PREFIXES` / `ENV_EXACT` / `ENV_BOX`, bound
  to the runner's text by the kit test). Any other `PYTORCH_*` key is information: the runner records the image's version env
  (`PYTORCH_`, `TORCH_`, `CUDA_`, `NV_`, `NVIDIA_`) in `env_image.txt`, and it never VOIDs a row (it reaches both frameworks on
  the box alike, and the image is pinned). The
  DRIVER passes each registered box code through unchanged, and ALWAYS the launcher's host codes 13, 14, 17 and 18 (it reads
  only the driver's own code and excludes a machine on those). The driver's own codes are 23 (no readable exit file fetched),
  24 (the box said 0 without this run's `P238_SUCCESS` marker: never a success) and 26 (any other code).
- **Host refusals and the pin walk.** The launcher excludes a machine on 13, 14, 17 and 18 only (its host-refusal set). 15 (card,
  compute capability, SM count or VRAM) and 16 (host RAM) end the box WITHOUT excluding the machine, so P238's pin walk excludes it
  itself: the driver records the machine of every 15 / 16 refusal and the screen skips it for the rest of the lane (a card that
  failed those once does not change by itself). Each such refusal is a retry id at the non-start cost (its boot slack, about
  USD 0.25 at USD 1.05/h), never a strike.

## Budget

- **One RTX 5090 per box** (Vast verified/secure, cc 12.0, >= 160 SMs, VRAM >= 30,000 MiB; host RAM and disk at TC1 amendment 80's
  Qwen3.6 floors), at most **USD 1.05/h** (the program's 5090 policy rate on 2026-10-23), downloads at most USD 0.02/GB (about 72 GB of model plus
  about 10 GB of installs per box).
- **Shakedown:** about 1.5 h; its guard is its own fit (7 processes at the registered default 900 s: 4.9 h, under the policy's
  6 h default); worst 4.9 h x USD 1.05 + 82 GB x USD 0.02 = **USD 6.78**.
- **Per box:** every box's guard is at most the box cap, so every box's worst (its guard x USD 1.05 + USD 1.64 of downloads) is at
  most **USD 15**. That is the per-box rule; a split READING is K boxes and may exceed USD 15 in total.
- **Reading, as a function of K** (estimated 44 loads x about 8 minutes + the op timing, about 6.4 h of work): its worst is the
  sum of its K boxes' worsts, at most K x USD 15. At 480 s a load, K = 1: one box of 10.0 h, **USD 12.18**. At 900 s, K = 2: two
  boxes of 9.6 h, USD 11.70 each, **USD 23.40**. At 1,500 s only K = 4 fits (the largest box 9.0 h), at 3,000 s only K = 8:
  both read `NEEDS_APPROVAL` (the lane ceiling is K <= 2). Fit from the shakedown, as above.
- **Retries:** a box that does not start costs at most its boot slack, about USD 0.25.
- **Lane worst:** the shakedown + 2 readings (a reading and its rerun) + 3 retries = USD 6.78 + 2 x (the reading's worst) +
  USD 0.75: **USD 31.89** at K = 1 (480 s), **USD 54.33** at K = 2 (900 s), at most USD 6.78 + 2 x K x USD 15 + USD 0.75 in
  general: USD 67.53 at the K <= 2 ceiling (USD 127.53 at K = 4 and USD 247.53 at K = 8, neither registered without a new
  approval). Expected at K = 1: about USD 20.
- **The daily rail** (USD 100/day across the program) is the one a split can bind: the K boxes run one after another on one
  machine and can span two calendar days, and they share each day with every other lane's boxes. Every P238 box is disclosed
  at create and at teardown, as every rental is.

## Launch

- **One exact launch commit per box:** this registration's merge (or a later amendment's), through pod-launch with `--validate`
  passing there, the host pinned, rent-watch armed; then the maintainer's Approved naming that commit and the kit's hash.
- **The reading launches only after** amendment 81 (device time at source) and the tc1_arm hook PR have merged.
- **A box over 6 h launches only once the per-run override is in the rental policy,** with `wallclock_override: true` in its
  launch manifest (the launcher's `--wallclock-override`; the policy then checks its guard against the 18 h ceiling and its
  estimate against USD 15). Until then only a box whose guard fits the 6 h global cap runs (the shakedown does: 4.9 h, with no
  override).
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

## Amendment 1 (2026-10-26): the card floor re-pinned at TC1 amendment 83

This supersedes the validity bullet "the card floor on every load" above, whose text is unchanged. Nothing else changes: the arms,
points, schedule, readings, predictions, budget and run ids stand. No P238 load has run, so no reading moves.

- **The pin.** `bench/common/card_clock.py` is re-pinned at TC1 amendment 83's merge (sha256
  `14494c6d045f5840457fb26477c83435f17ecfdf35438a0cfcbd703d6547eba8`). `staged-p238.sha256` moves for it, `p238_reduce.py` and
  `p238_run.sh`; every staged file is again pinned at its bytes in the tree.
- **The burn before any install** runs in the `clock` mode, named in its call line (`--mode clock`): the busy SM median >= 2700 MHz;
  no slowdown bit on any sample (HwSlowdown 0x8, SwThermalSlowdown 0x20, HwThermalSlowdown 0x40, HwPowerBrakeSlowdown 0x80,
  BoardLimit 0x200); Reliability 0x400 on fewer than 95 % of busy samples; SwPowerCap 0x4 recorded and never refusing. rc 18
  otherwise, as before.
- **Each load's own clock log** is judged by `card_clock.passes(summary, 2700, mode="clock")` itself. `p238_reduce.check_card` holds
  no copy of the rule. The row's `clock` carries card_clock's summary fields under their own names (`busy_samples`, `sm_median`,
  `busy_reliability_samples`, `any_sample_bits_seen`, and `busy_swpowercap_fraction`, recorded only). A summary without the
  Reliability count reads that clause UNREAD and the load is VOID: fail closed. `passes()` takes `mode` keyword-only with no
  default, so a caller that names no rule fails loudly.
- **Why.** The first two shakedowns refused at this floor before any install. p238-shake-1's card burned at a busy median of
  2,493.5 MHz and refuses under both rules. p238-shake-r1's card, a stock 575 W card at 2,827 MHz, was refused by amendment 80's
  burn bars on SwPowerCap (98.6 % of busy samples) and Reliability on a sample. TC1 amendment 83's scan of 737 training clock logs
  found SwPowerCap on more than 5 % of busy samples in 117 of 422 healthy-clock RTX 5090 logs, and Reliability at most 81 % on a
  healthy clock against 100 % on every sub-floor log.
- **Tests.** `tests/test_p238.py` judges real clock logs from the TC1 record through `clock_facts` and `check_card` (the healthy
  boundary card and a clean card pass; the power-capped card refuses on its clock alone; amendment 79's capped card on its clock
  and persistent Reliability), and checks that the verdict is `card_clock.passes()` in the named mode. The reducer's self-test adds
  the clock clauses (persistent Reliability, a slowdown bit, a summary without the Reliability count, too few busy samples) and the
  two cases that must stay clean (SwPowerCap on every busy sample, Reliability at 94.5 %).
- **The shakedown.** p238-shake-r2 re-burns under this rule on the re-cut kit; shake-r1's stored burn summary predates amendment 83
  and is not re-read as a pass.
