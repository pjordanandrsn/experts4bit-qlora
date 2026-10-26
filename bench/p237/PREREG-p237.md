# P237: e4b's CPU expert offload with layer-major against Unsloth's offload_layers, same RTX 4090, two VRAM budgets

DRAFT (not yet registered). Owner-requested; registered before any box. Every rule below is fixed before data.

## The question

On one rented RTX 4090, training Qwen3-30B-A3B on TC1's field recipe, how does e4b's CPU expert offload with layer-major
accumulation compare against Unsloth's `offload_layers`, in held-out quality, step time, setup time, host RAM and
host-to-device transfer, at a 12 GiB VRAM budget and at the full card?

**The two frameworks offload different things, and the verdict says so.**
- **e4b** (`load_moe_4bit_streaming(offload=True)`) moves every layer's frozen 4-bit **experts** to pinned host RAM as
  the layer is built. It keeps one layer's experts on the GPU at a time. This is a fixed policy, not a budget-sized plan.
- **Unsloth 2026.10.8** (`offload_layers`, also called block swap) moves the frozen weights of **whole decoder layers**,
  all 128 experts each, to pinned host RAM, and streams them back layer by layer. With `"auto"` it sizes how many layers
  to swap from the VRAM budget.
- Each arm runs its framework's offload mode at the budget. The comparison is not a kernel-for-kernel one.

**12 GiB is an artificial budget on a 24 GiB RTX 4090, not an RTX A2000.** Every result line at that budget says so. The
other budget is the **full card**: a 4090 reports about 23.99 GiB, so its "24 GiB" budget is the per-process fraction
1.0, with no artificial limit.

## Arms and budgets

| arm | framework | offload |
|---|---|---|
| **E** | experts4bit-qlora 0.58.1 | CPU expert offload (`--offload 1`) plus layer-major accumulation |
| **UA** | Unsloth 2026.10.8 | `offload_layers="auto"` |
| **UT** | Unsloth 2026.10.8 | `offload_layers=N_t`, the registered tuning rule below |

Each arm runs at both budgets: **12 GiB** (artificial) and the **full card**.

**The tuning rule (UT), fixed before data.** It applies only where UA's `auto` chooses N_auto > 0.
1. After UA's first draw records N_auto, run 8-step probes at N = N_auto − 1, N_auto − 2, down to N_auto − 4.
2. Stop at the first OOM.
3. N_t is the smallest N that completes all 8 steps with `max_memory_reserved` at most the cap minus 512 MiB.
4. If no probe passes step 3 (the first one OOMs, or every probe runs over the margin), N_t = N_auto: the record reports
   "tuned equals auto", and UT's draws still run.
5. A probe that faults (any status but ok or OOM, or no per-step reserved record) stops the probes, and UT does not run at
   that budget: a fault is not an outcome the rule ranks.
6. Probes are recorded and never timed or read. Each probe's reserved peak is the maximum of its per-step
   `max_memory_reserved` (`TC1_STEP_MEM=1`, `step_mem[].max_reserved`). The reducer re-derives N_t from the probe receipts
   (`p237_host.tune`, the runner's own function), and a UT draw run at any other N is VOID.

**Where auto needs no offload.** At the full card, `auto` may choose N_auto = 0, because the 4-bit model fits. E still
offloads every layer, by policy. tc1_arm then writes UA's row as `refused` (3): the offload did not engage
(`offload_engaged`, e4b#2146). That row is read as N_auto = 0 only when BOTH hold: its reason names an empty or absent block
swap, AND the arm's log carries one of Unsloth's own not-engaged lines (unsloth 2026.10.8, matched by prefix): `Unsloth:
offload_layers = 'auto' swaps nothing` (`models/_utils.py:9703-9705`) or `Unsloth: offload_layers = 'auto' loads every
layer onto the GPU` (`models/loader_utils.py:1045`, `models/_utils.py:9015`). A `refused` UA row without such a line is a
refusal, never an N_auto reading. If it reads
N_auto = 0, the verdict says in words: "Unsloth needed no offload at the full card; E's offload was not required to fit."
No tuning probe runs there, since UT would have nothing to tune, and no UA timing exists at that budget. Amendment
78's resident 24 GiB e4b fit (its own claim id) is quoted beside this cap as context only. It is never in a ratio and
never a P237 measurement.

## The recipe (TC1's field recipe, unchanged)

- **Model and dtype:** `Qwen/Qwen3-30B-A3B` @ `ad44e777bcd18fa416d9da3bd8f70d33ebb85d39`, bf16 compute, gradient
  checkpointing on.
- **Shape:** seq 2048, micro-batch 2, gradient accumulation 4, 60 optimizer steps.
- **Adapters:** LoRA r 16, alpha 16, dropout 0, on every attention projection and every routed expert. The adapters are
  fp32 (`--adapter-dtype fp32`), with the matched init `matched:3407`. They are identical on both frameworks.
- **Optimizer:** `adamw_8bit`, LR 2e-4, weight decay 0.001, 5 warm-up steps, linear schedule, seed 3407.
- **Data:**
  - `unsloth/alpaca-cleaned` @ `0fe581eb…`, shuffled with `Random(3407)`; rows 0-1199 are train, the next 48 held out.
  - The row order is TC1's fixed `train[((i*accum+j)*M+k) % len]`, identical on both frameworks.
  - The token sha is recorded per arm and must match across arms.
- **Held-out set:** the first 8 of the 48 held-out rows, scored at step 0 and every 20 steps (`eval_loss`, one row at a
  time, `labels=input_ids`).

## The harness

TC1's reviewed harness is extended, and nothing in it is replaced.
- **The TC1 harness extension, MERGED as e4b#2146 (`8e92ac66`, `tc1_arm.py` sha256 `434c77c1…`).** This lane uses it as merged,
  and edits nothing in `tc1_arm.py`.
  - **The hook, OBSERVE ONLY:** `TC1_ARM_HOOK=p237_arm:make` with `TC1_ARM_HOOK_SHA256` read from this lane's staged pin. The
    factory takes no arguments.
    - The phases are `post_import(args)`; `post_load(args, model)`, after every harness setup and before the first forward;
      `step_edge(i)`, before optimizer step i and once after the last (i = N); and `record()`, into the receipt's `arm_hook`.
    - **No phase launches device work.** Counters, NVML and host reads only. For the inner profiled steps `step_edge` runs
      while the profiler records, and amendment 81's `device_ms` counts kernels at the source. Each phase runs inside
      `record_function("tc1_arm.arm_hook.<phase>")` and its seconds go into `arm_hook.seconds`.
    - Any exception a phase raises, an OOM included, is `harness_error` (exit 10) naming `arm_hook.<phase>`, never the arm's
      `refused` or `oom`.
  - **The registered options, recorded as given AND as resolved:**
    - `--vram-cap-gib 12` at the 12 GiB budget (none at the full card): the fraction before the load, the getter read back,
      `vram_cap` in the receipt. A torch without the getter refuses.
    - `--unsloth-offload-layers auto|N` (UA, UT): `from_pretrained(offload_layers, max_memory={0: "12GiB"})`, and the swapped
      layers off `model._unsloth_block_swap.indices`.
      - At the 12 GiB budget it needs the cap.
      - At the full card the option runs WITHOUT `--vram-cap-gib`, which tc1_arm refuses (`max_memory` comes from the
        cap). So at the full card UA passes `--vram-cap-gib` set to the card's own total (fraction 1.0, read back), and
        `max_memory` is that total. The cap passed is the exact double of total / 2^30, so the fraction reads back
        as exactly 1.0. `max_memory` is that cap printed to 6 significant digits (`tc1_arm.py`'s `{cap:g}GiB`), within
        0.0001 GiB of the total.
      - `refused` (3) when the offload did not engage as given: no swap object, an empty swap, or an explicit N resolving to
        another count.
    - `--layer-major 1` (E): the step body is `layer_major_step`. Its drift guard runs once before the window, on step 0's
      micro-batches, with its gradients discarded. Its allocator peak goes into `layer_major.guard.peak_vram_gb` and is reset
      before the window, so `peak_vram_gb` is the steps' max on every arm.
    - `step_contract` (optimizer steps, accum, micro-batch, the tokens-per-step sha): this lane's reducer asserts it is equal
      across every arm of a budget, else that budget is VOID.
- **This lane's observing hook, `bench/p237/p237_arm.py`:**
  - `post_load` installs the shared counter on the forwards read off the loaded model (`patch_loaded`, which picks each
    module's serving object by the library's own predicates; the bnb 4-bit experts are served by the package
    `moe_utils`), and the gnf4 route wrapper on `nf4_route.train_gemm_route`;
  - `step_edge(i)` calls the counter's `mark_step(i)` and snapshots at the timed window's and the profiled window's
    edges. Only training passes count, by `mod.training`, so the held-out evals are excluded;
  - **Counts are PER EDGE, never a total minus something.** Under `--layer-major 1` the drift guard's M+1 stock forwards and
    its layer-major forward run in train mode BEFORE `step_edge(0)`, so a forward counted outside any step is recorded as
    such (edge `None`) and never enters a step's count;
  - `record` writes the counter's `structural(micro_batches, n_steps)` (the structural backward counts, the training
    passes' agreement, and the under-grad split as information), the block-swap engagement checks and the
    transfer-byte counters. A missing or faulted structure fails closed.
  - It configures nothing: the cap, the offload and layer-major come only from the registered options above.
- **Runner and reducer:** this lane's own, in the P235 form, which is a compact lane runner made by named edits of a parent
  lane's runner. Copying TC1's 3,674-line runner and 13,099-line reducer is not that form.
  - `bench/p237/p237_run.sh` is P236's runner (`bench/p236/p236_run.sh`) by named edits for the shape: the nonce, the
    refusals, the guard fit, STARTED and finish.
  - Its environment is amendment 80's recipe from `bench/tc1/tc1_run.sh` (the `qwen3pos80` family): `venv-unsloth`, e4b
    and grouped-nf4-gemm from PyPI, the release proof and the card clock.
  - Its arms are amendment 80's two (`fused_attn4_m_fd`, `ckpt_unsloth_m_fv`), flag for flag, plus the extension's
    options. A test parses every argv the runner builds with `tc1_arm.py`'s own `build_parser()`.
  - `p237_reduce.py` is this lane's own. It imports the shared counter's checks and F15 v2's `counts()`, never
    re-typed. `p237_host.py` holds the runner's decisions (CPU class, the full-card cap, N_auto, the probe outcomes, N_t)
    as functions the tests and the reducer call.
- **Shared helpers:** imported from amendment 80, never copied.
  - `bench/common/card_clock.py`: the loaded clock probe.
  - `bench/tc1/f15v2.py`: Unsloth's expert-GEMM engagement.
  - `release_proof.py`: provenance.
- **Shared with P238:** `bench/common/unsloth_moe_counter.py`, one module counting Unsloth's generic grouped calls and
  `NF4_FUSED_STATS`, with mutants for both paths, the fused path's bypass of the generic counter included. It lands as
  its OWN small PR (ruled on the bus, 2026-10-25), written by one lane and reviewed by another before the maintainer's pass. P237
  and P238 both import it from main. P237 imports it unchanged and adds only its own arm wiring; its sm_89 per-call
  dispatch bar (below) comes from the module's per-call record (ruled 2026-10-25).
- **Order, as it landed:** amendment 80, the shared counter (e4b#2132, `42b6f2b9`), amendment 81 (e4b#2144, `1f43d364`),
  the harness extension (e4b#2146, `8e92ac66`), and amendment 82 (e4b#2148, a dataset refusal is lane code 21). This
  registration and its kit are cut on their bytes. This lane's runner and staged pin follow `tc1_arm.py` at `434c77c1…`.

## The VRAM budget (the same mechanism for both frameworks)

- **The cap:** before any load, `torch.cuda.set_per_process_memory_fraction(cap / total)`. Unsloth also gets
  `max_memory={0: cap}`.
- **e4b:** its training offload path reads neither free nor total memory, so the fraction is a pure allocator ceiling.
- **Unsloth:** its `auto` planner reads the fraction through `torch.cuda.get_per_process_memory_fraction`
  (`usable_cuda_bytes`, `_free_device_bytes`).
- **The getter check:** the box asserts the getter exists before loading. It does at torch v2.12.1 and v2.8.0, but if it
  were missing, Unsloth would silently plan for the whole card. A missing getter is a refusal.
- **Per draw, the record holds:**
  - the fraction read back, and `max_memory_reserved`, which must be at most the cap;
  - nvidia-smi's peak `memory.used`, since the CUDA context and Triton caches sit outside the fraction.

## Engagement (positive, binding, from the counters' real semantics)

A draw whose engagement check fails is **VOID**.

**E: the offload.** It is checked with `E4B_OFFLOAD_STATS=1`, which counts only real host-to-device stagings (`stage()`
returns early if a layer is already staged).
- Layer-major stages each layer's experts once in the forward and once in the backward, and there is one resident slot.
  So a step stages **2n = 96** on Qwen3 (n = 48), less one when layer 0 is already resident at the step's start (the
  previous step's backward ended on it): **95** in the steady state. P137 measured exactly these on a 4090 (96 cold,
  95 after).
- The bar is per step and exact: step i's train-mode stagings == 2n − r_i, where r_i = 1 when the hook's edge-i snapshot
  shows layer 0's handle resident, else 0. The step after an eval (loop indices 20 and 40) starts cold, because an
  eval's stock forward evicts after each layer, so it reads 96, as does any step after the guard if the guard leaves
  no layer resident. Eval stagings are counted apart, by `model.training`, and never enter a step.
- Any other count, or any step with no stagings, VOIDs the draw. So does any staging tagged other than a whole-layer
  tag: `offload.py` tags whole-layer copies `sync`, `cold_miss` or `prefetch`; `routed` and `speculative` stage rows.

**E: the layer-major path.** The record proves `layer_major_step` ran on every optimizer step, not the stock loop.
- The count of `layer_major_step` calls must equal the step count.
- The stock loop would read M × (2n − 1) = 380 stagings, so the staging count also separates the two paths.

**UA and UT: the block swap.** All of these must hold:
- `model._unsloth_block_swap` exists, and its `indices` hold exactly the arm's N layers.
- At every edge, the swap's own residency agrees with the tensors: every frozen parameter of each block it reports
  evicted has `numel() == 0` (a forward that skipped the swap hook could not compute on a 0-element tensor), and no
  resident block holds a 0-element parameter. At most `depth` blocks are resident at an edge (the recorded
  `prefetch_depth`, 2 by default). The backward's last hook re-arms the first `depth` blocks for the next forward
  (unsloth_zoo 2026.10.8 `block_swap.py`, `_bwd` at i = 0 calls `_arm(forward=True)`), so "every swapped layer empty
  between steps" would be false on a healthy run. That was the draft's wording before the source was read; it is
  corrected here, before data.
- Per optimizer step (train mode, per interval), every swapped block has **at least M evictions and at least M − 1 real
  fetches** through the wrapped `_Block.evict` / `_Block.prefetch`, counted only on a residency change (both return at
  once otherwise). Each training micro-batch needs the block resident for its forward, and its backward releases it.
  The first fetch of a step may land in the previous interval (the re-arm above).
- The full-card UA at N = 0 has no swap, and records that.

**UA and UT: the expert GEMMs** (the TC1-79 lesson: no torch-level counter that a backend can bypass).
- **The dispatch on this card (sm_89) is MIXED, decided per call.** In unsloth_zoo 2026.10.8, every grouped GEMM's
  forward picks Triton or `torch._grouped_mm` through `moe_utils._triton_grouped_mm_wanted` (line 572). That function
  uses the class `_triton_grouped_mm_kind` gives the call (lora if min(K, N) ≤ 32, else many at E ≥ 64) and the sm_89 row
  of `_TRITON_GROUPED_MM_AUTO_ROWS` (lines 503-508: lora -1, many 32, dw 0).
  - **LoRA GEMMs (rank 16):** always Triton, and their dW always goes through `generic_grouped_wgrad` (lines 611-616).
  - **Frozen base GEMMs (many):** Triton only at M ≤ 32 × 128 = 4,096 routed rows; above that, `torch._grouped_mm`. A
    full micro-batch (2 × 2048 tokens × top-8 = 32,768 rows) takes torch. Under TC1's dynamic padding, a micro-batch of
    ≤ 512 padded tokens (≤ 4,096 routed rows) flips back to Triton, so the split varies per call.
  - The 9E / 2E figures of amendments 79-80 are sm_120 (RTX 5090) facts and do NOT transfer to this card (ruled 2026-10-25,
    option A).
- **The binding bar: per call, per class, from the library's OWN predicate**, via the shared module
  (`bench/common/unsloth_moe_counter.py`, e4b#2132). It wraps the predicate in every module object that can run the
  forward: the package `moe_utils` and the globals of the function `get_forward_moe_backend()` returns (a compile-cache
  copy has its own globals). It records which object served, then logs (class, M, E, decision) per call. It runs
  unprofiled. All of these must hold, exactly, else VOID:
  - every LoRA-class call DECIDES Triton (the predicate returns False silently on a dtype, a tensor subclass or a dim);
  - the counted `generic_grouped_mm` / `generic_grouped_wgrad` calls equal the counts the per-call decisions imply: the
    Triton-decided forwards, the backward gx of each Triton-forwarded call (training's grad-disabled backward branch,
    asserted), and the LoRA dW;
  - `GENERIC_CALLS["fallback"]` delta == 0 (a Triton-decided call that ran eager);
  - `check_library` passes (wrapper counts == the library's own `GENERIC_CALLS`).
  The resolved per-class row limits are recorded in each draw's snapshot. Every pin is set before the zoo import,
  because the policy cache is keyed on the environment at first call.
- **The cross-check:** on one profiled window per Unsloth draw, the per-call decisions in the window must equal the
  profile's kernel counts, read by amendment 80's F15 v2 `counts()`: Triton grouped-GEMM kernels == the Triton-decided
  calls, and `aten::_grouped_mm` == the torch-decided calls, exactly. F15 v2's own `engagement()` refuses any mixed
  dispatch, so it is not this lane's check; only its counting is used. This bar is REGISTERED here, before data. The
  shakedown's real 4090 record TESTS it (gates are tested on a real record) and never derives it. Disagreement VOIDs
  the draw. The window is **loop indices 21..23, the same for e4b and Unsloth**:
  `--profile-warm 21 --profile-steps 3`, steps 22..24 counting from 1. It is amendment 80's window (`POS80_PROFILE_WARM=20`)
  moved one step later, by ruling before data. Loop index 20 is the step right after the eval at 20: E stages it cold (96,
  see Engagement) and Unsloth re-arms after an eval-mode forward. Starting at 21 keeps that step out of the profile. The
  window also sits past amendment 79's transient, where a cold Unsloth draw compiled and saved its Mega-cache bundle while
  a warm one loaded it (memcpy/fill 342 against 109 ms), and it changes nothing either framework does. The choice was made
  before any P237 data. Its 3 profiled steps sit inside the timing median (steps 11..60 counting from 1, TC1's
  `step_ms[10:]`), a known overhead on 3 of 50 steps.
  Each draw's Mega-cache lines are reported beside its row, never ruled.
- **The timed steps stay unprofiled.**

**Every knob an arm depends on is set explicitly, both frameworks'.** Each arm's env dump proves no ambient value
leaked in.
- **The box environment, one list** (`p237_reduce.ENV_PREFIXES` / `ENV_EXACT` / `ENV_BOX`). A test binds the runner and
  the reducer to it, with mutants that widen or narrow each side. The list is used in three places:
  - **The runner's ambient refusal, before STARTED, exit 78.** It refuses any `UNSLOTH_*`, `E4B_*`, `GNF4_*`,
    `NF4_QLORA_*` or `TC1_*` key already in the box's environment, except the keys the runner sets itself and
    `E4B_SHA`, the driver's launch commit. An ambient `E4B_TRAIN_DETERMINISTIC` would move e4b's step time, as an
    ambient Unsloth knob moves Unsloth's.
  - **`env_box.txt`**, dumped after the exports.
  - **The reducer's registered set**, which `env_box.txt` must EQUAL at the registered values; else the draws VOID.
    This set holds the six `UNSLOTH_*` pins, `GNF4_TRAIN_GEMM=auto`, `PYTORCH_CUDA_ALLOC_CONF` and `E4B_SHA`.
  - The per-process keys (`E4B_OFFLOAD_STATS`, `TC1_STEP_MEM`, `TC1_ARM_HOOK`, `TC1_ARM_HOOK_SHA256`) are set on each
    process's own line, after the dump, so the registered set holds no `TC1_*` key.
  - Any other `PYTORCH_*` key, and the image's own environment, goes to `env_image.txt` as information, never a fault.
- **UA and UT:** `UNSLOTH_MOE_GROUPED_TRITON=auto`, the shipped default, set explicitly (ruled 2026-10-25).
  - Every `UNSLOTH_*` environment read in ANY module of the pinned unsloth_zoo and unsloth wheels must be unset or at
    its registered value. The shared module's `check_env` enforces this by a SOURCE SCAN of every module in both wheels
    (ruled 2026-10-25), not a hand-kept module list or a name prefix. It records the scanned key count per module.
  - Unset keys cost nothing. A SET key that is not registered VOIDs the draw, so an ambient
    `UNSLOTH_DISABLE_MOE_TRITON=1` VOIDs it.
  - The registered set is exactly the scanned keys this lane's arms set, each with its value: the pins above, plus the
    keys TC1's runner sets. That includes `UNSLOTH_ENABLE_LOGGING=1` (`tc1_run.sh:924`), which `moe_utils` reads. It is a
    logging switch, kept because the Mega-cache lines this lane records come from it.
  - The kit derives the set from the arm's own `env_set` record and its runner line, never by hand. A registered key
    the library never reads VOIDs, as `check_env` rules.
- **UA and UT:** `UNSLOTH_MOE_NF4_FUSED=0`. That path (unsloth-zoo#1935, in 2026.10.8) bypasses the generic grouped calls
  the binding counter wraps. So every Unsloth draw must record `NF4_FUSED_STATS` with `launches == 0` and `calls == 0`
  (the shared module's `check_unfused`); any other value VOIDs the draw. The same arms also pin
  `UNSLOTH_MOE_NF4_AUTO_MAX_ROWS=128` and `UNSLOTH_MOE_NF4_AUTOTUNE=0`, the shipped defaults (ruled 2026-10-25). They are
  inert while the fused path is off, but are set so no ambient value can reach a timed arm.
- **UA and UT:** `UNSLOTH_MOE_FUSED_ROUTER=0`, the shipped default. It is opt-in, `"1"` only, and re-read on every
  eager router call (unsloth_zoo 2026.10.8 `moe_router_triton.py:28,69`; used in `qwen3_moe.py:123`). So it would switch
  this model's router inside the timed step. The whole-wheel scan covers it, and the env dump proves the pin.
- **UA and UT:** `--grad-ckpt unsloth`, TC1's default (`tc1_arm.py:6508`, `use_gradient_checkpointing="unsloth"` at
  2160), set explicitly and recorded as the receipt's `ckpt_mode`. It is Unsloth's no-grad-first-then-replay checkpointer.
  - The binding generic-GEMM bar counts the backward GEMMs by STRUCTURE: expert layers × micro-batches × GEMM class. It
    never counts graph-building or under-grad forwards, since under HF non-reentrant checkpointing BOTH passes build a
    graph and only one is backwarded.
  - Decisions are shape-determined, so each layer's two passes must decide the same Triton/torch path and class. That
    decision is the one counted, and a disagreement VOIDs the draw. The under-grad split is kept as a cross-check.
  - So the bar holds under either checkpointer. Under HF non-reentrant checkpointing both passes run under grad, yet
    each backward runs once (measured in review, 2026-10-25).
  - Any `ckpt_mode` other than `unsloth` VOIDs the draw.
- **E:** `GNF4_TRAIN_GEMM=auto`, e4b's shipped default, set explicitly. The lane compares each framework's offload
  as shipped, and Unsloth's `UNSLOTH_MOE_NF4_FUSED=0` is its default too (ruled 2026-10-25). On an RTX 4090 (sm_89),
  grouped-nf4-gemm 0.51.0's `auto` routes each call by its present-group count (`nf4_route.route_for`):
  - `fused` above `DENSE_AUTO_MAX_GROUPS` = 16;
  - `dense` for 1 to 16;
  - never `grouped_mm` (sm_90 only) or `decoded` (opt-in).

  The per-call decision is `nf4_qlora.py:248`, `train_gemm_route(a_cat.device, len(sizes))`, where `sizes` lists the
  present groups. It imports the function at call time, so a module-attribute wrapper on
  `nf4_route.train_gemm_route` sees every call and logs `(n_groups, route)`.

  The check, per E draw, fails closed:
  - every logged call's route equals `route_for((8, 9), requested="auto", n_groups=n)`;
  - `ROUTE_STATS["dense_fwd"]` equals the logged dense calls;
  - the grouped_mm (`fwd`, `dgrad`) and `decoded_*` counts are 0.

  Anything else VOIDs the draw. The per-route call counts (fused, dense) are reported beside the row, so readers see
  how often each route ran. At this recipe (2 × 2048 tokens × top-8 over 128 experts) almost every call has well over
  16 present groups, so the dense count is expected near 0. That expectation is reported, never ruled. No
  fused-pinned arm is registered: the question is the shipped defaults.

**Fail closed.** A missing counter, record or field is VOID, never a pass. The engagement checks are proven on a real
$0 record before the kit: e4b's on P137's stored record, and Unsloth's on the real stored pos79 records plus the
shakedown. Each check also has a mutant that removes the engaged path and must VOID.

## Measures (recorded per draw)

- **Held-out quality:** held-out loss at step 60, in nats per token.
- **Step time:**
  - wall seconds per optimizer step: the median of steps 11 and on, timed from the first micro-batch to after the
    optimizer and scheduler steps with a CUDA sync;
  - device time per step, kernels only, by ONE method fixed here: amendment 81's at-source `device_ms` from
    `tc1_arm.py`'s `summarize_profile`, which excludes rows where torch's `is_user_annotation` is true. Every
    profile must carry `device_ms_method == "kernels_only_at_source"` (81's field). A profile without it, or with
    another value, VOIDs the draw's device time. Amendment 80's by-name method is NOT computed on P237's profiles:
    it subtracts the annotation row, so on a post-81 profile it would subtract that row a second time (80's
    function refuses such a profile). The F15 v2 cross-check reads kernel-name counts, not `device_ms`, so it is
    unaffected.
**Every measure below is computed by the reducer per draw** (`p237_reduce.measures`), never assembled by hand. A measure
whose input is missing reads UNREAD, never 0, and a test removes each input in turn. A measure not computed there is not
registered.
- **Setup time:** load, offload placement, and warm-up, each separately. Together they partition the time before the first
  timed step: `load_s + placement + warm-up == prologue_s + step 0`.
  - load: `load_s`;
  - placement: `prologue_s − load_s −` E's guard seconds (eval0 and the harness setup included);
  - warm-up: step 0 plus E's guard seconds.
  The warm draws give the setup figures. Each primer gives the first-run (cold cache) setup, reported beside them.
- **Peak host RAM:** VmHWM, `ru_maxrss` and cgroup `memory.peak` (TC1's `host_ram_report`).
- **Peak VRAM:** `peak_vram_gb`, the training steps' allocator max on every arm. E's drift-guard call (its first
  `layer_major_step`) holds every micro-batch's graph at once, so its peak is recorded apart as
  `layer_major.guard.peak_vram_gb` and reported beside E's row, never in it.
- **Host-to-device transfer bytes:** software counts of the bytes each framework enqueues, not PCIe hardware counters.
  - The figure is the median over the timed steps of the train-mode bytes per step.
  - e4b: the hook's `record_copy` bytes, also totalled per staging tag.
  - Unsloth: the hook's real fetches, summing each fetched block's `nbytes()`.
  - Cross-check: `nvidia-smi dmon -s ut` PCIe rx/tx, as the median MB/s over the process, reported beside them, never
    ruling.
- **nvidia-smi's peak `memory.used`** per process (MiB), beside `peak_vram_gb`.
- **The card:**
  - the busy SM-clock median and the reason bits seen;
  - the PCIe generation and width (their maxima, from `nvidia-smi`, in `box.json`).
  A pinned host-to-device ceiling probe was drafted here and is struck before data, by ruling: no decision rule reads it.

## The host floors (refusals before install, never strikes)

- **Loaded clock:** `card_clock.py`'s loaded bf16 matmul burn before install.
  - The host is refused unless, at the burn (amendment 80's rule, with the 4090's own floor):
    - the busy SM-clock median is at least **2,500 MHz**;
    - Reliability (0x400), HW slowdown (0x8), SW and HW thermal (0x20, 0x40), HW power brake (0x80) and 0x200 are absent
      on EVERY busy sample;
    - SwPowerCap (0x4) is absent on at least 95 % of busy samples. A healthy card can touch it briefly under a
      full-power burn, and refusing that would cost a retry id for nothing.
  - Calibration: across 12 logged 4090 arms in the private receipt store, busy medians ran 2,520-2,745 MHz, and 2,500
    clears all 12. 2,520 MHz is the 4090's rated boost clock. **Limit:** n = 12 arms is a small calibration set. A
    capped card still refuses: HP3's ran about 1,492 MHz with reason bits 0x0, which a bit-only screen would pass.
  - Per draw, a busy median under 2,500 MHz, or the bits present on more than 5 % of busy samples, VOIDs that draw.
- **Host CPU class:** a fast class only: AMD EPYC 9xx5 (Zen 5), Ryzen 7000 or 9000 desktop, or Intel Core 13th/14th
  gen or Ultra desktop. EPYC Zen 2-4 and Xeon hosts are refused. B=1 offload steps are host-bound, so the CPU class is
  named on every result line.
- **Host RAM, disk and download:** at least 128 GiB host RAM (e4b's offload read 113.5 GB peak host RAM at this recipe on
  a 4090), at least 200 GB disk, and a download probe of at least 40 MB/s.
- **Exit codes against the launcher.** The driver hands the box's code to the launcher through
  `bench/common/drive_tail.sh`'s `drive_finish`. The kit lists EVERY exit code its driver and runner can return against
  the launcher's host-refusal set (13, 14, 17, 18), which are the only codes that exclude a host. A host code appears only
  on a line stating a host-floor reason. A dataset refusal is a lane code, never 13 (TC1 amendment 82's rule), and a test
  reads the runner's own text to hold that. For each code it states:
  - whether it is a host fault (it excludes the machine) or a harness fault (25, probe_error: never excludes);
  - how the pin walk proceeds: retry on the same host, or move to the next;
  - what the attempt costs.
  A code outside the table refuses the launch at review time.

  **The table** (`p237_run.sh`'s codes, passed through by `p237_drive.sh` via `drive_finish`; a test holds the driver's list
  equal to the runner's codes):

  | code | where | fault | the launcher | the walk | what it costs |
  |---|---|---|---|---|---|
  | 0 | a READ, or a finished shakedown | none | - | done | the box |
  | 13 | disk under 200 GB (`BOX_REFUSED disk=`) | host | excludes the machine | next host | the boot, before STARTED |
  | 14 | download MEASURED under 40 MB/s (`BOX_REFUSED egress:`) | host | excludes the machine | next host | the boot |
  | 18 | CUDA unusable, driver under 580, CPU class, card clock (`BOX_REFUSED host floor:`) | host | excludes the machine | next host | the boot |
  | 15 | card class, cc, SMs or VRAM | host fact the screen should have refused | does not exclude | stop: the pin screen is wrong | the boot |
  | 16 | host RAM under 128 GiB | as 15 | does not exclude | stop | the boot |
  | 41 | the guard fit | harness (the deadline) | does not exclude | stop: the fit needs the maintainer | the boot |
  | 9 | staging, install, pins, release proof, a probe that measured nothing, snapshot, prepare | harness | does not exclude | stop | up to the box's wall |
  | 10 | torch does not import | image | does not exclude | stop | the boot |
  | 29 | the dataset (amendment 82: never 13, and never 21, which is the driver's own start failure) | harness | does not exclude | stop | up to the install |
  | 27 | the reducer ruled VOID, or wrote nothing | the reading | does not exclude | stop; a retry id only by ruling | the box |
  | 78 | run id, mode, frozen knobs, an ambient `UNSLOTH_*` | harness | does not exclude | stop | the boot |
  | 130 | a signal | - | does not exclude | stop | up to the box's wall |
  | 20, 21, 22 | the driver's own staging, start or fetch failure | harness | does not exclude | stop | the boot, or the box |
  | 23, 24, 26 | `drive_finish`: no exit file, 0 without this nonce's success marker, an unregistered code | harness | does not exclude | stop | the box |

  "Next host" spends the retry id the launch walks on (the run ids are `p237-4090-1`, `-r1`, `-r2`; the shakedown's
  `p237-shake-1`, `-r1`, `-r2`). "Stop" ends the walk for the maintainer. The runner never writes 17.
  No box code is one the driver exits on its own (`p237_drive.sh`, `bench/common/drive_tail.sh`), except 0 and 78, which
  mean the same on both sides. A test holds this, with a mutant that puts 21 back (amendment 82's rule).

## Design

- **One host.** Every arm and both budgets run on one box, in one session.
- **Cache priming:** ONE cache-priming warm-up per framework per budget runs before that budget's palindrome, so every
  timed draw is warm. A primer is a 10-step run of the arm's own configuration (E for e4b, UA for Unsloth). It is
  recorded and never ruled. Its SETUP time is reported as the **first-run (cold cache) setup** beside the warm setup
  figures, because a user's first run is cold. Only the warm draws enter the verdict.
- **Order, a palindrome per budget:** primers, then `E UA UT UT UA E` at 12 GiB, then the same at the full card. UT's
  probes run after UA's first draw, before the first UT draw.
- **Cache state, every run:** recorded beside each row (primers, probes and draws).
  - Unsloth's Mega-cache lines, from the process log.
  - The cache files before and after each process: Triton's (`/root/.triton/cache`), inductor's
    (`/tmp/torchinductor_root`) and Unsloth's compiled cache. No new file means a warm hit.
- **Draws:** two per arm per budget. Step 0 and the drift guard are excluded from the timing window and recorded.

## The verdict rule (mechanical, per budget)

1. **VOID:** any draw failing engagement, the cap, the clock or the token sha. A VOID arm enters no reading.
2. **Quality** (TC1's own bar for this recipe, TC1-PREREG.md's verdict column):
   - **COMPARABLE** when |mean held-out at step 60 (U*) − mean held-out at step 60 (E)| ≤ **0.05 nats**, each side the
     mean of its two draws; **QUALITY_FAIL** otherwise.
   - Each comparison is also reported against **0.01**, amendment 69's same-stack bar, so this lane reads beside it.
   - Layer-major's gradients are not bitwise equal to the stock loop's, so a small held-out move on E is expected and
     recorded.
3. **Speed:** per comparison U*/E, the ratio of median wall seconds per step (the verdict) and of device time (reported
   beside it). Each is the median over draws, with the interval over the cross-draw ratios.
   - Offload is host-bound by nature, so device time alone would hide the cost a user pays.
   - A ratio is quoted only when both arms are VALID and COMPARABLE, and neither is UNSTABLE (draws more than 5 % apart).
4. **Also reported per arm:** setup time, peak host RAM, transfer bytes per step, and peak VRAM.
5. **A public verdict either way,** losses beside wins, with the budget label and the host CPU class on every line.

## Predictions (falsifiable, graded in the read)

- **P-a:** at 12 GiB, E's steady stagings read exactly 95/step and its wall step is 2.5-5 s (P137 read 3.2 on a 4090).
- **P-b:** at 12 GiB, UA swaps N_auto ≥ 20 of 48 layers and moves more host-to-device bytes per step than E: whole layers
  of all 128 experts against E's experts once per step.
- **P-c:** at 12 GiB, UA's wall step is slower than E's (ratio UA/E > 1.0), and UT is faster than UA.
- **P-d:** at the full card, UA chooses N_auto = 0.
- **P-e:** every COMPARABLE comparison also falls within 0.01.

## What this lane cannot say

- **Nothing about an RTX A2000 or any 12 GB card.** 12 GiB is an artificial budget on a 4090, which has the 4090's
  bandwidth, clocks and PCIe.
- **Nothing about other models, recipes, ranks or sequence lengths.** One model and one recipe.
- **No kernel-for-kernel reading.** The two arms offload different things by design.
- **Software byte counts are not PCIe hardware counters.**
- **One host:** the ratio is a property of this card, host CPU class, torch and recipe.

## Sequencing

1. The $0 code read (done) and the rulings on arms, cap, tuning, quality and design.
2. Amendment 80 merged, with its `card_clock.py`, `f15v2.py` and release-proof wiring.
3. The shared counter (`bench/common/unsloth_moe_counter.py`, e4b#2132) merged.
4. Amendment 81 (e4b#2144) merged, with the at-source `device_ms` and its `device_ms_method` field.
5. The TC1 harness extension (e4b#2146) merged; amendment 82 (e4b#2148) follows it.
6. This registration and its kit, cut on top of the hook's merge. Then one independent review and the maintainer's
   pass.
7. A $0 CPU check of the new code, then a short GPU shakedown (registered, never a reading).
8. The driver dry run, the $0 validate, and the maintainer's Approved.
9. The box, the maintainer's re-derivation, and the results PR.

## Runs and budget

- **Box:** one RTX 4090 at ≤ $0.60/h, guard 6.0 h, download ≤ 100 GB at ≤ $0.011/GB. The worst case is
  6.0 × 0.60 + 100 × 0.011 = **$4.70**.
- **Shakedown:** one short box, at most **$1.50**, registered and never a reading (`p237_schedule.py --shake`).
  - E and UA at 12 GiB, 24 steps each, so the profiled window (loop indices 21..23) is inside the run and every
    engagement check runs on a real record.
  - UT's probes at 12 GiB.
  - UA at the full card, the N_auto = 0 path.
  - Its measured per-process wall becomes `P237_LOAD_S_REGISTERED`, the guard fit's per-process term for the reading,
    registered by amendment before the reading box. The runner takes it only as whole seconds of at least 300, else 78.
- **The fetch carries receipts and logs only** (the store's rules). `p237_drive.sh`'s exclude list follows TC1's (the
  venv, the caches, the adapters, `tp4_alpaca`'s downloaded source text), adds every tensor extension, and caps any one
  file at 10 MiB. The driver lists what it leaves on the box, and a test runs the list over a stub box tree.
- **Lane worst:** one reading box and one shakedown, plus one retry id for a box that did not start, at most **$8**,
  inside the $15 lane cap.

Amendments, dated, go below this line before any data is read.
