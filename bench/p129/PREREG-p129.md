# P129 — one fused q/k/v projection for e4b's training attention: NF4 base and fp32 LoRA (DRAFT, registered before any run)

Issue: experts4bit-qlora#835. The lane number was claimed by `prereg/p129` (2026-10-09). It follows P128 (#1453), whose census found
training attention the largest launch bucket. The registration is staged as P128's was:
- **Phase 1** builds the fused projection and is gated on an RTX A2000 by counts only.
- **Phase 2** is a rented A/B, registered by amendment once Phase 1 passes. No box runs before then.

## The question

At TC1's field recipe, one training step of a two-layer Qwen3-MoE at Qwen3-30B-A3B's layer dimensions made 799 kernel launches (P128's
census). Attention made 230 of them: 90 in the forward, 90 in the checkpoint recompute and 50 in the backward. Per layer pass the
forward ran:
- 16 `mm`;
- 8 dtype casts (the fp32 LoRA in and out);
- 8 bitsandbytes dequantizes (the nested absmax, then the NF4 weight);
- 8 adds.

That is about 10 launches a projection, and q, k and v are about 30 of the 45. The three projections read the same input, so most of that
work is per-projection repetition.

**Does one fused q/k/v projection (NF4 base and fp32 LoRA) remove that repetition in training, exactly where it can be exact, and does
that shorten the step on a host-bound box?**

## The subject

An opt-in knob, `E4B_TRAIN_FUSE_QKV=1` (off by default), applied after `add_attention_lora` and `enable_fast_train`. It replaces each
eligible attention module's `q_proj`, `k_proj` and `v_proj` (each a `LoRALinear` around a bitsandbytes NF4 `Linear4bit`) with one
fused module, and the module's forward with serving's fused forward (`engines/qkv_fuse.py`'s `_fused_forward`, reused, not copied).
That forward looks up the modeling module's rotary at call time, so e4b's fused training rope and RMSNorm patches stay in force.

**The fused base.**
- The three projections' packed NF4 rows are concatenated along N. The blocks are 64 elements along a row of K = 2048, so each block
  stays within one row and the packed bytes are the same bytes.
- The nested (double-quantized) absmax cannot be concatenated, because each projection's nested state carries its own offset. The
  fused module therefore holds the fp32 absmax, expanded once from each projection's nested state by bitsandbytes' own dequantize.
- So the fused dequantize computes, bit for bit, the three dequantized weights stacked. It is one launch where today's path takes two
  per projection, and costs about 0.5 MB more per layer.

**The fused LoRA.**
- One cast of the input to the adapters' dtype and one `mm` against the three A matrices concatenated, `[3r, K]`.
- Three B `mm`s writing the three output slices, and one cast back.
- The trainable parameters stay the original nine q/k/v adapters (`torch.cat` and slicing route their gradients), so optimizer
  state and saved adapters are unchanged.

**Refusals.** Anything below keeps today's path, and the module is counted as refused with its reason:
- the three adapters differ in rank, alpha, scaling or dtype;
- any adapter carries dropout or other per-projection semantics;
- a base is not NF4 with the same blocksize, quant type and compute dtype;
- K is not a multiple of the blocksize (so a block would span rows);
- any projection carries a bias;
- the attention class is not one whose fused forward serving already reviews (`Qwen3MoeAttention`);
- a rotary that is not rotate-half.

**Untouched.** Serving: `fuse_qkv` and its folds are not called or changed. With the knob unset, training runs today's ops; tests assert
that bit for bit (the same op sequence, values and gradients as the commit before).

## Phase 1 — the build and its gate (RTX A2000, counts and correctness only)

The model is P128's two-layer random Qwen3-MoE at real layer dimensions, set up as TC1's e4b arm, rows of 270 tokens, the same
seeds.

**Exactness, stated before the read:**
- **The dequantize:** the fused module's dequantized weight `torch.equal` the three projections' dequantized weights stacked, every
  layer.
- **The projections and the loss:** concatenating changes the GEMM shapes, so cuBLAS may pick another algorithm or reduction order.
  Each q/k/v output, the loss and every gradient (the nine adapters included) must be within TC1's rounding bar of today's path: each
  tensor within `2**-6` of its largest entry for bf16 and `2**-16` for fp32. Bitwise is reported, not claimed.

**Counts, per training step, against today's path on the same model:**
- **Kernel launches:** today 799.
  - **Prediction:** about 80 fewer, `[60, 100]`: per layer pass about 18 fewer (q/k/v from about 30 to about 12), twice a
    layer (forward and recompute), plus 6–10 in the backward, over two layers.
  - **Gate:** at least 50 fewer (≥ 6 %).
- **Python calls** (cProfile, autograd on the calling thread): today 16,241.
  - **Prediction:** 4–12 % fewer (six module calls a layer pass become one fused call; the forward is serving's, not HF's).
  - **Gate:** at least 3 % fewer.

**Phase 1's rule:**
- **PASS:** exactness and both count gates hold.
- **NO_GAIN:** exact, but a count gate missed.
- **FAIL:** an exactness bar missed. P129 then stops; the read names the tensor and its distance.

## Phase 2 — the rented A/B (by amendment after a PASS; the frame is P128's, carried over)

- **The box:** TC1's field recipe over 60 load-gated steps in venv-unsloth (torch 2.12), e4b at its defaults. The shipped arm (bf16
  adapters) and the matched arm (fp32 adapters), knob 0 against 1, two draws a side in ABBA order, every arm profiled.
- **The recount gate:** the box recounts launches and Python calls with the knob on and off on its own card. It is VOID unless each
  reproduces at least 0.8 of Phase 1's relative cut.
- **The premise gate:** VOID for speed unless the matched knob-0 arm's GPU busy share is at most 0.85.
- **Two ratios:** the wall ratio judges the remedy; the device-time ratio bounds its cost. They are reported separately.
- **Quality:** step-0 held-out within 0.0005 per draw pair and held-out at N within 0.005, each arm.
- **The rungs:** VOID / NOISY (draws 5 % apart) / QUALITY_FAIL / NO_GAIN (wall above 0.98 on either arm) / GAIN (both at most 0.98,
  stays opt-in) / DEFAULT_ON (GAIN, plus a second host at GAIN or, GPU-bound there, at most 1.01).
- **Predictions:** set at the amendment from Phase 1's counts, priced at P128's per-launch and per-op CPU self times.

## Budget

- **Phase 1:** no rental.
- **Phase 2:** one RTX 5090 box (about $1.5–2), plus at most one replication. The lane stays under $15.

## Phase 1 read, as registered (RTX A2000, 2026-10-09): counts PASS, exactness FAIL on the end-to-end clause

The prototype is `E4B_TRAIN_FUSE_QKV=1`, the module above. It was read on the registered model, rows and seeds against today's path.
Both layers fused, none refused.

**Counts (gate PASS; both predictions missed high):**

| per training step | today | fused | change | gate | prediction |
|---|---|---|---|---|---|
| kernel launches | 769 | 659 | −110 (−14.3 %) | ≥ 50 fewer: **HELD** | `[60, 100]`: missed high |
| Python calls | 16,240 | 14,266 | −12.2 % | ≥ 3 %: **HELD** | 4–12 %: missed high |

**Exactness:**
- **The dequantize: bitwise, as claimed.** `torch.equal` on both layers.
- **The fused projection in isolation, reported:** the same input and the same upstream gradient through the fused module and through the
  three `LoRALinear` modules. Every tensor is within TC1's rounding bar, none bit for bit:

  | tensor | relative difference | bar |
  |---|---|---|
  | q/k/v outputs (bf16) | 6.7e-3 | `2**-6` |
  | input gradient (bf16) | 1.06e-2 | `2**-6` |
  | the six adapter gradients (fp32) | about 4e-7 | `2**-16` |

- **The end-to-end clause: FAIL.**
  - The loss read 8.682032 against 8.682492, a relative difference of 5.3e-5 against a `2**-16` bar.
  - All 24 gradients differ. The worst is layer 0's `gate_up_lora_B`, at 0.18 of its largest entry.

**Cause.** The clause applied an op-level rounding bar to a whole bf16 MoE training step. A reorder-class change moves the projections by a
rounding step. Attention and the router amplify that: a flipped top-k choice moves the expert gradients by whole contributions. So no
change of this class can meet that clause in bf16 MoE training. The condition in review named "the projections and the loss", and this page
widened it to every gradient. The condition was itself ambiguous about the end-to-end loss. By the rule this read is **FAIL**, and the bar
is not replaced on these records. Amendment 1 re-asks the end-to-end question in its own units, against a neutral floor, on fresh seeds.

## Amendment 1 (2026-10-09T14:24Z, after the Phase 1 read, before any new run): the end-to-end question against a neutral floor

The question Phase 1's end-to-end clause should have asked: **does the fused projection move a short training run more than the eager path
moves under changes that are equally valid?** A single-seed comparison of one MoE step cannot say; a floor over seeds can.

**The instrument.** P128's two-layer random Qwen3-MoE at Qwen3-30B-A3B's layer dimensions, set up as TC1's e4b arm at e4b's defaults
(fp32 LoRA, the reentrant checkpoint, `NF4_QLORA_SINGLE_LADDER=auto`). Per seed `s`:
- 30 optimizer steps of bitsandbytes' AdamW8bit, lr 2e-4 with 5 linear warm-up steps, weight decay 0.001;
- each step one batch of 2 rows × 270 random tokens from a generator seeded by `s`;
- the adapters initialised from `s` (`lora_B` drawn N(0, 0.02), so the attention adapters are live from step 0);
- a fixed held-out set of 4 such batches from `s + 1000`, evaluated at steps 0 and 30.

**The seeds (fresh; none was read before):** 211, 223, 227, 229, 233.

**Divergence of a run V from the baseline run B with the same seed:**
- `D_traj` = max over steps 1..30 of |training loss of V − of B|;
- `D_held` = |held-out loss of V at step 30 − of B|.

**The neutral floor.** The eager path under three changes that leave the arithmetic's meaning unchanged and move only its reduction order:
- **F1** `NF4_QLORA_SINGLE_LADDER=0` (the expert LoRA block unladdered);
- **F2** `NF4_QLORA_PAD_BUCKETS=1` (the LoRA delta bucketed);
- **F3** each 2-row batch as two 1-row micro-batches with gradients accumulated. Both rows carry 270 tokens, so the mean loss is the
  same quantity.

The floor of each measure is its **worst draw**: the maximum over F1–F3 × the five seeds (15 draws).

**The fused path:** `E4B_TRAIN_FUSE_QKV=1` against B, the same five seeds (5 draws).

**The end-to-end gate** (registered now):
- **HELD** iff the fused path's maximum `D_traj` is at most the floor's worst `D_traj`, **and** its maximum `D_held` at most the floor's
  worst `D_held`, **and** (the backstop, TC1's held-out bar) every fused `D_held` is at most 0.005.
- Per-step gradients (the first step's relative differences) are reported, not gated.

**The counts gate is unchanged** (≥ 50 fewer launches, ≥ 3 % fewer Python calls), re-read on the same build.

**The rule:**
- **PASS:** the counts gate and the end-to-end gate both HELD.
- **NO_GAIN:** the counts gate missed.
- **FAIL:** the end-to-end gate missed. The read then names the measure and the draw.

**Prediction:** the end-to-end gate HELD (about 70 %). The fused projection moves each run by a rounding step, the same class as F1–F3.
The floor's own magnitude is not predicted; it is read.

**Budget:** the RTX A2000 only: 20 runs of 30 steps. Phase 2's speed A/B is still registered by its own amendment after a PASS.

## Amendment 1 read (RTX A2000, 2026-10-09): PASS

e4b at `590962a` (`p129-train-fuse-qkv`: the module above, committed before the read), grouped-nf4-gemm main at `e21a712`, torch 2.11.
25 runs, none failed: the baseline, F1, F2, F3 and the fused path, on each of the five registered seeds.

**The end-to-end gate: HELD.**

| measure | fused, worst of 5 | floor, worst of 15 | backstop |
|---|---|---|---|
| `D_traj` (max step loss difference) | 0.00311 | 0.00361 (F3, seed 233) | |
| `D_held` (held-out at step 30) | 0.00064 | 0.00078 (F3, seed 233) | every fused draw ≤ 0.005: held |

The fused draws by seed (`D_traj` / `D_held`):
- 211: 0.00228 / 0.00050
- 223: 0.00311 / 0.00048
- 227: 0.00237 / 0.00041
- 229: 0.00236 / 0.00041
- 233: 0.00234 / 0.00064

The floor's draws lie in 0.00171–0.00361 (`D_traj`) and 0.00002–0.00078 (`D_held`). The fused path moves a 30-step run by
less than the eager path's own equally valid variants do. The prediction (HELD, about 70 %) is met. This instrument's runs barely
learn (random tokens on random weights: the baseline's training loss went from 8.759 to 8.754), so it measures numerical divergence, which
is what it was registered for.

**The counts gate, re-read on this build: HELD.** grouped-nf4-gemm's newer main adds 16 launches to both paths:
- kernel launches 785 → 675 (−110, −14.0 %);
- Python calls 16,440 → 14,466 (−12.0 %);
- the dequantize still `torch.equal`.

**By the rule, PASS.** The next step is Phase 2's speed A/B, registered by its own amendment with P128's frame: a recount gate, a
host-bound premise gate, wall and device ratios separately, and the TC1 held-out bars.

**Records and reducer.**
- `records/a1/`: the 25 runs, the counts, the isolated projection, the first step's gradients and the environment.
- `p129_reduce.py`: re-derives this verdict from them (`python bench/p129/p129_reduce.py`; `--selftest` runs 6 hand-built cases), into
  `RESULTS-p129-a1.md`.
- `instrument/`: the scripts that made the records, with the model builder.

The first step's gradients, reported as registered: the worst relative difference from the baseline, per seed, was
- 0.009–0.010 for F1;
- 0.009–0.065 for F2;
- 0.128–0.149 for F3;
- 0.110–0.222 for the fused path.

The fused path's step-1 differences are the size of the micro-batch split's, the floor's own largest. A per-step gradient is not a
measure any reorder-class change can pass, which is why Amendment 1 does not gate it.

## Amendment 2 (2026-10-09T15:03Z, after Amendment 1's PASS, before any box): Phase 2, the speed A/B at TC1's field recipe

**Why.** Amendment 1 passed: launches −14.0 %, Python calls −12.0 %, and the end-to-end read inside the neutral floor. Phase 2's frame
(above) set the shape. This amendment fixes the token, the validity, the numbers, the host and the predictions. Random tokens barely
learn, so Phase 2 also carries TC1's held-out bars on the field recipe, where learning shows.

**The build.**
- **e4b:** main at launch. It carries the module, merged as opt-in code at `f539896` (`engines/train_qkv_fuse.py`, `E4B_TRAIN_FUSE_QKV`,
  off by default), and this amendment's harness.
- **grouped-nf4-gemm:** v0.44.0 at `d1f64ba`. Its single-block ladder is `auto` by default, so the matched arm takes it.
- **Pins:** the manifest pins both SHAs.

**The box** (token `qwen3fqkv`, `bench/tc1/tc1_run.sh`):
- **Recipe:** TC1's field recipe (Alpaca rows, micro-batch 2 × accumulation 4, the reentrant checkpoint on all 48 layers), 60
  load-gated steps, venv-unsloth (torch 2.12), e4b at its defaults otherwise.
- **Sides:** `E4B_TRAIN_FUSE_QKV=0` (`q0`) against `=1` (`q1`), on the shipped arm (bf16 adapters, native init) and the matched arm (fp32
  adapters, matched init).
- **Draws:** two a side in ABBA order: shipped `q0`, shipped `q1`, matched `q0`, matched `q1`, matched `q1` d2, matched `q0` d2, shipped
  `q1` d2, shipped `q0` d2.
- **Profiling:** every arm, with `--phase-peaks 1` and TC1's profile steps.
- **New receipt fields** (`bench/tc1/tc1_arm.py`):
  - a `train_qkv` record: the knob, whether e4b has the module, modules fused and refused, fused calls;
  - `profile.launches_per_step`: the four launch API calls Phase 1 counted, per profiled step;
  - `weights_commit`: the commit e4b's weights loaded from (the config's `_commit_hash`).
- **The weight pin, fixed here:** e4b's arm called `load_moe_4bit_streaming` without `revision=`, although the tokenizer and every other
  framework's snapshot lookup passed the registered one. The arm now passes `--revision` to the loader, which pins the config and the
  snapshot and refuses a full-sha mismatch. The reducer (`weights_commit_why`, every family) VOIDs a draw whose `weights_commit` is
  not the pinned revision. Receipts from before the field carry none and keep their verdicts.

**Validity** (`fqkv_why`, on top of TC1's):
- torch 2.12;
- e4b's field defaults: the double-quantized absmax, the reentrant checkpoint on all 48 layers, `E4B_CKPT_OFFLOAD` unset;
- `q1`: the knob at `1`, all 48 attention modules fused, none refused, fused calls recorded;
- `q0`: the knob at `0`, none fused, on the same build (the module present);
- `weights_commit` recorded, and the pinned revision;
- a profile with its launch count.

**The gates and the verdict** (`score_fqkv`; two VALID draws a side, medians; the first rung that applies):
- **Recount** (rows `R_m`, `R_shipped`): launches per profiled step must fall from `q0` to `q1` by at least 0.8 of Phase 1's 14.0 %
  (11.2 %) on each arm. VOID otherwise.
  - *Narrowed from the frame:* the frame had the box recount Python calls as well, but it does not. TC1's harness has no call counter, and
    adding a counting step would change what the arms run. The call count belongs to the code path, which Phase 1 counted on the same
    code (−12.0 %). Launches can move with the card and the recipe (kernel choices, the checkpoint mode, the adapter dtype), so the box
    recounts those.
- **Premise** (row `PREMISE`): the matched `q0`'s `busy_t` (device ms per profiled step over the timed s/step, median of its draws) must
  be at most 0.85. VOID for speed otherwise.
- **Wall** (rows `W_m`, `W_shipped`): s/step `q1 / q0` per arm, from two stable draws a side (TC1's 5 % rule; NOISY otherwise).
- **Device** (row `DEVICE`): device ms per profiled step `q1 / q0` and the training-phase peak, per arm. Reported, not gated.
- **Quality** (row `QUALITY`): on each arm, step-0 held-out within 0.0005 per draw pair, and held-out at N within 0.005.
- **The rungs:** VOID / NOISY / QUALITY_FAIL / NO_GAIN (wall above 0.98 on either arm) / GAIN (both at most 0.98).
- **DEFAULT_ON** is read across two boxes on different hosts: one at GAIN, and the second either at GAIN or GPU-bound (`PREMISE`
  FALSIFIED). A GPU-bound second host counts only with both `R` rows and `QUALITY` HELD and both wall ratios at most 1.01.

**Predictions.** They are priced from the field recipe's own receipts (TC1 amendments 70–72, the same token shape and e4b defaults):
- **Launches:** `q0` makes about 79,600 a step (matched, the ladder engaged) and 75,800 (shipped).
  - Phase 1 removed about 55 a layer per micro-batch (110 over two layers, one micro-batch with its recompute). That is about 10,600 a
    step here (48 layers × 4 micro-batches).
  - The cut is **12–15 %** on each arm (about 13.3 % matched, 13.9 % shipped): both `R` rows HELD.
- **Wall, on a host-bound host:** `q1 / q0` in **[0.86, 0.95]** on each arm, about 0.91.
  - On amendments 70–71's host-bound hosts the field recipe stepped about 3.0 s on 76–80k launches, about 38 µs of wall per launch.
  - The removed launches are priced between P128's per-op CPU self times (about 20 µs) and that average.
- **Device:** `q1 / q0` in [0.97, 1.02] on each arm. One wider dequantize and product replaces three, and one LoRA-A product replaces
  three; the arithmetic is the same.
- **Peak:** within 0.1 GB. The packed rows replace the bases' own. The expanded fp32 absmax (31 MB over 48 layers) replaces the
  nested 8 MB, adding about 24 MB.
- **Quality:** HELD, as Amendment 1's floor read predicts.
- **The rung, on a host-bound host:** GAIN, about 70 %. NOISY under host load is the main alternative.

**Decision rules.**
- **GAIN:** the knob stays opt-in until DEFAULT_ON's second host reads. That box is the lane's one replication.
- **DEFAULT_ON:** e4b makes the fused projection its default in its own PR (unset means on; `0` keeps today's path).
- **NO_GAIN with the recount HELD on a host-bound host:** the launch cut does not buy wall time at this recipe, and the module stays
  opt-in. The read says where the host time went, from the profile's CPU self time by family, `q0` against `q1`.
- **QUALITY_FAIL:** nothing turns on. The read finds out why before anything else.
- **VOID on the recount:** the field recipe engages the module differently from Phase 1's instrument. The read finds out how.
- **VOID on the premise** (a GPU-bound host): the box's rows stand as the GPU-bound reading DEFAULT_ON asks for. The lane's one
  replication then seeks a host-bound host. Two GPU-bound hosts in a row leave the speed question open, with no further re-run on host
  grounds (TC1 amendment 72's rule for loaded hosts, applied to GPU-bound ones).

**The host.** One RTX 5090 at the policy rate, 4 h guard, TC1's 98 GB host floor.
- The launcher's machine ranking is off (`ADERTHA_PREFER_MACHINES=0`). It ranks fast host CPUs first, and those run the field recipe
  GPU-bound: amendment 72's machine 18967 had the matched arm busy 0.884 of its step.
- The box avoids 18967 and TC1's standing exclusions.

**Budget.** Eight profiled 60-step e4b arms come to about $1–3 a box with the download (amendments 71 and 72 billed $2.75 and $0.80). Two
boxes at most keep it under $6, and the lane stays under $15.

**The reducer** (`bench/tc1/tc1_reduce.py`): the `qwen3fqkv` family, `fqkv_why`, `weights_commit_why`, `score_fqkv` and its render
section.

Self-test case 126 covers the weight pin:
- a draw loaded from another commit is VOID, in `qwen3fqkv` and in an older family;
- a `qwen3fqkv` draw without the record is VOID;
- an older family's receipt without it keeps its verdict.

Self-test case 125 reads GAIN on the host-bound fixture, and checks the other branches:
- a −5 % recount is VOID;
- a GPU-bound `q0` is VOID;
- walls of 0.99 read NO_GAIN;
- a 0.01 held-out shift reads QUALITY_FAIL;
- a `q1` draw with 47 fused modules is VOID;
- a `q0` draw on a build without the module is VOID.

## Phase 2 read (`tc1-5090-146`, 2026-10-09): QUALITY_FAIL, as registered

**The box.**
- **Host:** machine 152440 (AMD EPYC 7K62, RTX 5090) at $0.659/h; $1.874. All eight arms are VALID.
- **e4b `be7a88d`:** Amendment 2's harness at its merge, plus the fetch guards (#1471). Its package code differs from Amendment 2's
  pin `fc99614` only in `experts4bit_qlora/recipe.py` (#1467, a footprint-estimate constant that no measured path calls), as the
  manifest recorded.
- **grouped-nf4-gemm:** v0.44.0 at `d1f64ba`.
- **Reduction:** main's `tc1_reduce.py` at `d14bcb1` (`RESULTS-p129-a2.md`).

**Load.** The load gate voided six draws and ran them again. Three first draws stand at their third attempt above the 6.0 gate:
- shipped `q0` at load 12.21;
- shipped `q1` at 8.96;
- matched `q0` at 6.63.

Every second draw ran at its first attempt, under the gate (3.71–4.43). The wall rows rest partly on the three loaded draws. Each side's
two draws still agree within the 5 % rule.

| row | verdict | reading | prediction |
|---|---|---|---|
| `R_m` | HELD | launches per step 79,581 → 68,517, −13.9 % (gate 11.2 %) | 12–15 %: held |
| `R_shipped` | HELD | 75,833 → 64,761, −14.6 % | held |
| `PREMISE` | HELD | matched `q0` busy_t 0.523 (≤ 0.85) | |
| `W_m` | HELD | `q1 / q0` 0.896 [0.882, 0.909] | [0.86, 0.95]: held |
| `W_shipped` | HELD | 0.879 [0.868, 0.890] | held |
| `DEVICE` | reported | 0.987 matched, 0.985 shipped; peak +0.005 / +0.023 GB | [0.97, 1.02]: held |
| `QUALITY` | **FALSIFIED** | step 0: −0.01233 on both arms, in every draw (bar 0.0005); at N: −0.00077 matched, −0.00153 shipped (bar 0.005) | HELD: falsified |
| `FQKV` | **QUALITY_FAIL** | | GAIN: falsified |

**By the rule, nothing turns on, and the read finds out why.**

**Why: GEMM-shape rounding, amplified through 48 layers.**
- **Where and how:** measured on an RTX A2000 (`instrument/p129_step0.py`; records in `records/a2/`) with the real Qwen3-30B-A3B at the
  pin, under expert offload, on the box's eight held-out rows. The local copy was checked file for file against the revision's
  sha256s.
- **Build:** e4b `be7a88d`, grouped-nf4-gemm `d1f64ba`, transformers 5.18.0, bitsandbytes 0.50.2.
- **What differs:** the adapters are at init (`lora_B` zero), so only the base projections differ.

| variant | the q/k/v base projections | step-0 held-out | vs stock | mean per-row \|row − D1\| |
|---|---|---|---|---|
| A | stock: three dequantizes and matmuls | 1.94705 | | 0.0132 |
| D1 | the three matmuls in fp32, rounded to bf16 | 1.95649 | +0.0094 | 0 |
| D2 | stock, q's matmul split in two along N | 1.96983 | +0.0228 | 0.0255 |
| C | one matmul over the three dequantized weights concatenated, in the stock forward | 1.97018 | +0.0231 | 0.0170 |
| C2 | the stock forward fed the fused module's q/k/v | 1.97018 | +0.0231 | 0.0170 |
| B | the fused module and serving's fused forward | 1.97018 | +0.0231 | 0.0170 |

- **No semantic difference.** B, C2 and C agree row for row. On one held-out row, C2 and B are bitwise equal in every recorded tensor of
  all 48 layers: q and k before and after rope, v, the attention output, the projection output and the layer output. Serving's forward
  does what the stock forward does with the same q, k and v.
- **The difference is the GEMM shape.** Against the stock path, the first difference is layer 0's k; layer 0's q is bitwise equal. k
  differs by 2.8e-3 relative and v by 2.5e-3, under one bf16 step. The difference grows with depth:
  - the attention output: 0.4 % at layer 0, 5.5 % at layer 11, 9.8 % at layer 35;
  - the residual stream: 0.6 % at layer 0, 1.2 % at layer 35, 8.0 % at the last layer.
- **The floor is as large as the shift.** Splitting q's matmul in two (D2), a change of the same class, moves the step-0 loss as much
  as fusion does (+0.0228 against +0.0231). The fp32 reference lies between the stock and the fused paths.
- **The sign follows the card.** On the A2000 the fused path's step-0 loss is 0.0231 above the stock path's. On the box's RTX 5090 it
  was 0.0123 below.

TC1's step-0 bar (0.0005) presumes the initial computation is unchanged. That holds for changes confined to the adapters, whose
`lora_B` is zero at init. A change to the base projections' GEMM shape cannot meet it on this model, whatever its quality. This is a
registration error of the same kind as Phase 1's: an exactness expectation applied end to end. It does not change this read.

Held-out at N held on both arms. The last training step's loss lies in 0.8153–0.8206 across all eight draws. Nothing turns on. A
re-measure needs an amendment that sets the step-0 clause against a floor measured on the box, before any new box.

## Amendment 3 (2026-10-09T21:20Z, after the Phase 2 read, before any new box): the step-0 clause against a floor measured on the box

**Why.** `tc1-5090-146` read QUALITY_FAIL on TC1's step-0 bar (0.0005), and it stays QUALITY_FAIL. The investigation showed that no
change to the base projections' GEMM shape can meet that bar on this model:
- the fused path is the stock forward with one GEMM shape;
- a neutral split of q's matmul moves the step-0 loss as much;
- the sign follows the card.

This amendment registers a step-0 clause that a rounding-class change can pass only by staying inside the rounding family. It does not
re-read box 146. A fresh box reads it.

**The box** (token `qwen3fqkv3`): Amendment 2's box unchanged, with the same arms, order, steps, load gate, profiling, host rule (ranking
off, machine 18967 avoided) and policy rate. Each q0 arm also runs the step-0 floor (`TC1_QKV_FLOOR=1`), in the same process before any
training step. The build is e4b main carrying the harness (`tc1_arm.qkv_floor_rows`, `tc1_reduce.py`'s `qwen3fqkv3`) and this page,
with grouped-nf4-gemm v0.44.0 `d1f64ba`. The manifest pins both.

**The floor.** The step-0 held-out per row on the q0 arm under registered bf16 schedules of the same q/k/v base projections:
- **A:** the stock path (its step-0 rows);
- **A0:** the stock arithmetic through the floor's hook, a self-check that must equal A row for row;
- **D1:** the three matmuls in fp32, rounded to bf16 (the reference);
- **D2:** q's matmul split in two along N;
- **D3:** k and v as one matmul;
- **D4:** q's matmul split in four.

The fused path B is the q1 arm's step-0 rows. Only each projection's 4-bit base is swapped, so the adapters add their delta the stock
way.

**The clause.** For each arm, e_x is the mean over the held-out rows of |x − D1|. Step 0 passes when e_B ≤ max(e_A, e_D2, e_D3, e_D4):
the fused path is no farther from fp32 than the worst registered schedule of the same projections. There is no margin. It is an
envelope test, so the schedules set the bar. The per-row absolute error also removes the sign difference between cards.

**Validity** (`fqkv3_why`): Amendment 2's predicates. On each q0 arm, the floor record must:
- carry every mode for every row;
- have A0 equal to the stock step-0 rows;
- show D3 sharing k's matmul on every call.

A missing or broken floor VOIDs the draw, and an incomplete floor makes the rung VOID.

**Everything else as Amendment 2:**
- held-out at N within 0.005 on each arm;
- the recount (11.2 %);
- the premise (matched `q0` busy_t ≤ 0.85);
- the wall and device ratios;
- the rungs VOID / NOISY / QUALITY_FAIL / NO_GAIN / GAIN;
- DEFAULT_ON needs a second host.

**Calibration, not a reading** (the numbers this rule and the old clause give on data already in hand):

| data | e_A | e_D2 | e_D3 | e_D4 | e_B | max(e_A, e_D2, e_D3, e_D4) | this rule | the old clause (\|B − A\| ≤ 0.0005) |
|---|---|---|---|---|---|---|---|---|
| RTX A2000, the real model at the pin, box 146's eight rows (`records/a2/`; D3 and D4 not run there) | 0.0132 | 0.0255 | — | — | 0.0170 | ≥ 0.0255 | PASS | FAIL (0.0231) |
| RTX A2000, the two-layer random model, `lora_B` zero (`records/a3/`) | 0.0022 | 0.0017 | 0.0020 | 0.0017 | 0.0013 | 0.0022 | PASS | FAIL (0.00077) |
| the same, `lora_B` non-zero | 0.0019 | 0.0025 | 0.0017 | 0.0018 | 0.0016 | 0.0025 | PASS | PASS (0.00037) |

**Predictions** for the new box:
- **The step-0 ratio:** e_B / max(e_A, e_D2, e_D3, e_D4) on each arm in [0.35, 1.10], about 0.65.
  - If the fused path is one more member of the rounding family, the chance that it lies beyond the worst of four others is about one
    in five.
  - So step 0 passes on an arm with about 80 % probability.
  - Both arms share the base projections at init (`lora_B` is zero on both), so the two arms' ratios should be close.
- **Held-out at N:** within 0.005 on each arm.
- **The other rows:** as Amendment 2 predicted, which box 146 met.
- **The rung:** GAIN on a host-bound host, about 70 %.

**Decision rules** (Amendment 2's):
- **GAIN:** the knob stays opt-in until DEFAULT_ON's second host reads, and that box is the lane's one replication.
- **QUALITY_FAIL under this clause:** the fused path lies outside the rounding family on the box's card. Nothing turns on, and the read
  says on which rows.

**Budget.** One RTX 5090 at the policy rate: about $2–3 with the download. The floor adds five held-out passes of eight rows to each
q0 arm before training, seconds per pass. The lane has spent $2.085 so far and stays under $15.

## Amendment 3 read (`tc1-5090-147`, 2026-10-10): GAIN

**The box.**
- **Host:** machine 27708 (AMD EPYC 7B13, RTX 5090) at $0.689/h; $2.759. All eight arms are VALID.
- **Build:** e4b `51e5ae1` (Amendment 3 and its harness on main), grouped-nf4-gemm v0.44.0 `d1f64ba`.
- **Floor:** both q0 arms carry the full record. Every mode covers every row, A0 equals the stock rows, D3 shared on 384 of 384 calls
  (48 layers × 8 rows), and 48 attention modules were hooked.
- **Reduction:** main's `tc1_reduce.py` at `19bfbe1` (`RESULTS-p129-a3.md`).

**Load.** The host was heavily shared. The load gate voided 14 draws and ran them again, and every attempt that stands was above the
6.0 gate (loads 12.3–32.0). The last draw (shipped `q0` d2, load 29.4) had its re-runs skipped by the host-limited deadline. Its first
attempt stands, as registered. The wall rows were read under that contention. They agree with box 146's on a different host (0.896 and
0.879 there).

| row | verdict | reading | prediction |
|---|---|---|---|
| `R_m` | HELD | launches per step 79,576 → 68,513, −13.9 % | 12–15 %: held |
| `R_shipped` | HELD | 75,833 → 64,761, −14.6 % | held |
| `PREMISE` | HELD | matched `q0` busy_t 0.464 | |
| `W_m` | HELD | `q1 / q0` 0.895 [0.890, 0.901] | [0.86, 0.95]: held |
| `W_shipped` | HELD | 0.868 [0.862, 0.875] | held |
| `DEVICE` | reported | 0.987 on each arm; peak +0.005 / +0.023 GB | [0.97, 1.02]: held |
| `QUALITY` | HELD | step 0, both arms: e_B 0.02313 against max(e_A 0.03138, e_D2 0.03138, e_D3 0.03523, e_D4 0.03905) = 0.03905; at N −0.00001 matched, +0.00063 shipped | HELD: held |
| `FQKV` | **GAIN** | | GAIN about 70 %: held |

- **The step-0 ratio:** e_B / max is 0.59 on each arm, against a prediction of [0.35, 1.10], about 0.65.
- **Identical across arms and hosts:** the two arms read the same step 0, because `lora_B` is zero on both. The step-0 held-out values
  equal box 146's on another host (1.95917 knob off, 1.94684 fused).
- **D2 was not a distinct schedule on this card:** splitting q's matmul in two reproduced the stock rows exactly (e_D2 = e_A). The
  envelope's maximum came from D4.

**What the envelope measures** (report-only, RTX A2000, the real model at the pin, the box-146 rows; `instrument/p129_routing.py`,
`records/a3/routing_step0.json`):
- **Routing flips:** the fused path changes which experts the router picks. At layer 0, 3.4 % of tokens get a different top-8 set from
  the stock path's. That rises to 14–22 % in the deepest layers, about 15 % on average, and the q split does the same.
- **The causal check:** with every layer's router output replaced by the stock path's, the fused path's step-0 held-out is 1.94455,
  against the stock path's 1.94705 and the unpinned fused path's 1.97018. Pinning the routing removes 111 % of the mean shift. Per row,
  the mean |fused − stock| falls from 0.0241 to 0.0038.

The step-0 shift is the router's discrete top-8 choice amplifying sub-ulp q/k/v rounding. That is what the fp32-anchored envelope bounds.

**Decision** (Amendment 2's rules): GAIN. The knob stays opt-in until DEFAULT_ON's second host reads, and that box is the lane's one
replication. Box 146 cannot serve as the second host: it carries no floor, so it cannot be read under this clause.

## Amendment 4 (2026-10-10T04:40Z, after `tc1-5090-148`, before its rerun): the DEFAULT_ON second-host box on current main

**Why.** `tc1-5090-148`, DEFAULT_ON's second-host box at `51e5ae1`, measured nothing:
- About three minutes into the run, the container restarted: its first process started after the lane had begun, and no lane process
  remained.
- The driver at that commit counted the lane with a `pgrep` that matched its own ssh shell, so it never declared the lane dead. The box
  waited 2.3 h until the driver was stopped; it ended as a harness error at $1.809.
- Main has since replaced that check (#1512: lane identity and host uptime; #1517: a host reboot recorded as a host fault).

The replication therefore moves to a build that carries that driver. Its kernel side stays box 147's.

**The build.**
- **e4b:** main at launch, read from git, carrying this amendment. The manifest pins it.
- **grouped-nf4-gemm:** v0.44.0 `d1f64ba`, as box 147.
- **The package-diff audit against `51e5ae1`, box 147's build.** The manifest generator lists every `experts4bit_qlora` file changed
  between `51e5ae1` and the pin. It refuses the launch if any changed file is not in the list below, each read as off the Qwen3-30B-A3B
  training path:

| file | the change (as of main `f088be1`) | why it is off the measured path |
|---|---|---|
| `__init__.py` | the version, 0.52.0 | metadata |
| `arch/moe_conventions.py` | Qwen3.5-MoE's native composite text root | other families only |
| `arch/moe_plan.py` | the composite text-root scope | runs only for a convention with a text checkpoint prefix; Qwen3-MoE has none |
| `engines/chunked_lm_loss.py` | the chunk workspace's bytes per logit, 10 → 12 | priced only by `recipe`'s estimate; the runtime chunking decision does not read it |
| `engines/pipelined.py` | gpt-oss routing keyword names | gpt-oss only |
| `loader.py` | one module-path entry for `qwen3_5_moe_text` | other families only |
| `recipe.py` | memory-estimate terms | estimates; no measured path calls them |

The manifest's preregistration text records the audit's `git diff --stat` and this reason.

**The host.** Avoid machine 27708, box 147's, under the replication rule. Avoid machine 46990, which restarted box 148's container. Also
avoid 18967 and TC1's standing lists, with the launcher's ranking off, at the $0.85/h policy rate.

**Everything else as Amendment 3:** the token `qwen3fqkv3`, the arms, the floor, the clauses, the predictions and the decision rules.
DEFAULT_ON needs this box at GAIN, or GPU-bound with the recount and quality held and both wall ratios at most 1.01. If every standing
attempt again sits above the load gate, the read says so and does not call DEFAULT_ON on contention alone.

**Budget.** The lane has spent $6.653 ($0.211 + $1.874 + $2.759 + $1.809 for boxes 143, 146, 147 and 148), with $8.347 left under
the $15 cap. This box is about $2–3.

## Box log

- **`tc1-5090-142`** (2026-10-09, $0): refused before any instance existed. The cheapest eligible RTX 5090 billed $0.93/h with storage,
  above the $0.85/h policy rate.
- **`tc1-5090-143`** (2026-10-09, $0.211, machine 19317): a harness failure, not a reading; no arm ran and nothing is reduced.
  - The model fetch failed in venv-e4b's hub client (`process() takes no keyword arguments`): a brotli older than 1.2 under
    huggingface_hub 2.x's httpx2.
  - Every arm was written as a `not_run` stub, and the box ended OK.
  - TC1's harness now installs `brotli>=1.2.0` in venv-e4b, probes the fetch before building the other venvs, and ends a box that
    staged no model with rc 15.
  - The box reruns on the main commit that carries that fix.
- **`tc1-5090-144`** (2026-10-09, $0): refused before any instance existed. The receipts store's head was behind its upstream for a
  moment.
- **`tc1-5090-145`** (2026-10-09, $0): refused. The offer seen at $0.82/h was taken before the launcher searched, and the cheapest
  left billed $0.93/h.
- A $0.95/h ceiling for box 1 was set after those refusals and then found not to apply: the launcher's policy fixes RTX 5090s at
  $0.85/h whatever a manifest declares. Box 1 ran inside the policy rate.
- **`tc1-5090-146`** (2026-10-09, $1.874, machine 152440 at $0.659/h): the Phase 2 read above.
- **`tc1-5090-147`** (2026-10-10, $2.759, machine 27708 at $0.689/h): the Amendment 3 read above.
- **`tc1-5090-148`** (2026-10-10, $1.809, machine 46990): DEFAULT_ON's second-host box, a harness failure with nothing measured. The
  container restarted about three minutes in. The driver at `51e5ae1` did not see the lane die, so the box was stopped by hand after
  2.3 h. Amendment 4 reruns it.
