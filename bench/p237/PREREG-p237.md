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
4. If the first probe OOMs, N_t = N_auto: the record reports "tuned equals auto", and UT's draws still run.
5. Probes are recorded and never timed or read.

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
        `max_memory` is that total.
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
- **Runner and reducer:** this lane's own, `bench/p237/p237_run.sh` and `p237_reduce.py`, derived from
  `bench/tc1/tc1_run.sh` and `tc1_reduce.py` by named edits. Each changed line is one of the named edits.
- **Shared helpers:** imported from amendment 80, never copied.
  - `bench/common/card_clock.py`: the loaded clock probe.
  - `bench/tc1/f15v2.py`: Unsloth's expert-GEMM engagement.
  - `release_proof.py`: provenance.
- **Shared with P238:** `bench/common/unsloth_moe_counter.py`, one module counting Unsloth's generic grouped calls and
  `NF4_FUSED_STATS`, with mutants for both paths, the fused path's bypass of the generic counter included. It lands as
  its OWN small PR (ruled 09:12Z): b1-decode writes it, cerin-maintainer reviews it, then the maintainer's pass. P237
  and P238 both import it from main. P237 imports it unchanged and adds only its own arm wiring; its sm_89 per-call
  dispatch bar (below) comes from the module's per-call record (ruled 09:34Z).
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
- Layer-major stages each layer's experts once per optimizer step: **exactly 2n − 1 = 95** on Qwen3 (n = 48) on every
  timed step, and 96 on step 0. P137 measured exactly these values on a 4090.
- Any other steady count VOIDs the draw.

**E: the layer-major path.** The record proves `layer_major_step` ran on every optimizer step, not the stock loop.
- The count of `layer_major_step` calls must equal the step count.
- The stock loop would read M × (2n − 1) = 380 stagings, so the staging count also separates the two paths.

**UA and UT: the block swap.** All of these must hold:
- `model._unsloth_block_swap` exists, and its `indices` hold exactly the arm's N layers.
- Between steps, every frozen parameter of those layers has `numel() == 0`. A forward that skipped the swap hook could
  not compute on a 0-element tensor.
- The wrapped `_Block.prefetch` counts at least one real fetch per swapped layer per forward.
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
  - The 9E / 2E figures of amendments 79-80 are sm_120 (RTX 5090) facts and do NOT transfer to this card (ruled 09:34Z,
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
  the draw. The window is **loop indices 20..22, the same for e4b and Unsloth**:
  `--profile-warm 20 --profile-steps 3`, steps 21..23 counting from 1, amendment 80's choice (`POS80_PROFILE_WARM=20`).
  It sits past amendment 79's transient, where a cold Unsloth draw compiled and saved its Mega-cache bundle while
  a warm one loaded it (memcpy/fill 342 against 109 ms), and it changes nothing either framework does. The choice was made
  before any P237 data. Its 3 profiled steps sit inside the timing median (steps 11..60 counting from 1, TC1's
  `step_ms[10:]`), a known overhead on 3 of 50 steps.
  Each draw's Mega-cache lines are reported beside its row, never ruled.
- **The timed steps stay unprofiled.**

**Every knob an arm depends on is set explicitly, both frameworks'.** Each arm's env dump proves no ambient value
leaked in.
- **UA and UT:** `UNSLOTH_MOE_GROUPED_TRITON=auto`, the shipped default, set explicitly (ruled 09:34Z).
  - Every `UNSLOTH_*` environment read in ANY module of the pinned unsloth_zoo and unsloth wheels must be unset or at
    its registered value. The shared module's `check_env` enforces this by a SOURCE SCAN of every module in both wheels
    (ruled 14:43Z), not a hand-kept module list or a name prefix. It records the scanned key count per module.
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
  `UNSLOTH_MOE_NF4_AUTO_MAX_ROWS=128` and `UNSLOTH_MOE_NF4_AUTOTUNE=0`, the shipped defaults (ruled 09:26Z). They are
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
    each backward runs once (measured by review-desk, 14:48Z).
  - Any `ckpt_mode` other than `unsloth` VOIDs the draw.
- **E:** `GNF4_TRAIN_GEMM=auto`, e4b's shipped default, set explicitly. The lane compares each framework's offload
  as shipped, and Unsloth's `UNSLOTH_MOE_NF4_FUSED=0` is its default too (ruled 09:11Z). On an RTX 4090 (sm_89),
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
- **Setup time:** load, offload placement (time to the first trainable step), and warm-up (step 0, including E's drift
  guard), each separately. The warm draws give the setup figures; each primer gives the first-run (cold cache) setup,
  reported beside them.
- **Peak host RAM:** VmHWM, `ru_maxrss` and cgroup `memory.peak` (TC1's `host_ram_report`).
- **Peak VRAM:** `peak_vram_gb`, the training steps' allocator max on every arm. E's drift-guard call (its first
  `layer_major_step`) holds every micro-batch's graph at once, so its peak is recorded apart as
  `layer_major.guard.peak_vram_gb` and reported beside E's row, never in it.
- **Host-to-device transfer bytes:** software counts of the bytes each framework enqueues, not PCIe hardware counters.
  - e4b: `E4B_OFFLOAD_STATS` bytes per policy.
  - Unsloth: the wrapped `_Block.prefetch`, summing each fetched block's `sizes`.
  - Cross-check: `nvidia-smi dmon -s ut` PCIe rx/tx, reported beside them, never ruling.
- **The card:**
  - the busy SM-clock median and the reason bits on every busy sample;
  - the PCIe generation and width;
  - a pinned host-to-device ceiling probe.

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

## Design

- **One host.** Every arm and both budgets run on one box, in one session.
- **Cache priming:** ONE cache-priming warm-up per framework per budget runs before that budget's palindrome, so every
  timed draw is warm. A primer is a 10-step run of the arm's own configuration (E for e4b, UA for Unsloth). It is
  recorded and never ruled. Its SETUP time is reported as the **first-run (cold cache) setup** beside the warm setup
  figures, because a user's first run is cold. Only the warm draws enter the verdict.
- **Order, a palindrome per budget:** primers, then `E UA UT UT UA E` at 12 GiB, then the same at the full card. UT's
  probes run after UA's first draw, before the first UT draw.
- **Cache state, every run:** Unsloth's Mega-cache lines, and e4b's Triton and inductor cache hit or miss, are recorded
  beside each row (primers, probes and draws).
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
- **Shakedown:** one short box, at most **$1.50**.
- **Lane worst:** one reading box and one shakedown, plus one retry id for a box that did not start, at most **$8**,
  inside the $15 lane cap.

Amendments, dated, go below this line before any data is read.
