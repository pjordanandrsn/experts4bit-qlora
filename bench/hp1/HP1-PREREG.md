# HP1 — held-out parity: e4b as shipped vs Unsloth vs axolotl on Qwen3-30B-A3B, one evaluator, three seeds per arm (DRAFT, not registered)

Status: DRAFT. The bar was RULED by the maintainer on 2026-10-17T22:47Z (bus 20261017T224744Z-81856, on packet sha256
b49f6328): approved with five changes, all folded in below. Scoring may proceed once this file is registered and the shakedowns pass.
The later binding rulings are folded in too:
- equal order_sha VOIDs a mismatched pair (bus 20261017T233104Z-7528);
- the shared base is a checked premise; the box-side shakedown VOIDs cleanly; the run ids, host rule and spend go to the maintainer
  as an envelope for ACK (all bus 20261018T002045Z-34273).

Lane: heldout-parity (owner direction 2026-10-17). Issue: experts4bit-qlora#835 (TC campaign). The harness pieces are:
- the TC1 harness `bench/tc1/tc1_arm.py`. Its `--order-seed` and `--hp1-export` / `--hp1-export-base` hooks land in e4b #1709
  (Q235 Amendment 10 re-seals it, and R2 re-cites once).
- the `qwen3hp1` token in `bench/tc1/tc1_run.sh`;
- this directory: `hp1_lib.py`, the evaluator `hp1_eval.py`, the reducer `hp1_reduce.py`, and the shakedown cut `hp1_trunc.py`.

## Why this lane

TC1b (`bench/h2h-2026-10-02/tc1/RESULTS-tc1b-vs-tc1.md`, receipt `tc1-5090-15`) settled the matched question. At 200 steps of
the field recipe, e4b with PEFT's init and fp32 adapters and Unsloth with the same init and precision were
EQUIVALENT-AT-EVERY-EVAL (step 200: 0.7687 vs 0.7688, paired Δ +0.0001 ± 0.0016, 16 rows). The same table shows e4b AS
SHIPPED ending +0.0258 ± 0.0069 above the matched arm (0.7945), with a much lower train loss (0.6063 vs 0.7379), i.e. it
overfits. The axolotl rows (steady16/18, scattermoe, native init) read 0.7704–0.7705 against e4b-matched 0.7673 and
e4b-shipped 0.7937–0.7948. TC1b's decision rule P2 asked for the shipped defaults to become an open item. Nothing was filed.

The shipped arm differs from the matched arm in two knobs only (`experts4bit_qlora/lora.py`):
- **init:** LoRA A ~ N(0, 1/r) (std 0.0625 at r 16) against PEFT's kaiming-uniform(a=√5) (std = 1/√(3·fan_in), about 0.0128
  at fan_in 2048). That is about 5x larger on the attention slots.
- **adapter dtype:** bf16 expert adapters (attention adapters are already fp32) against fp32 everywhere.

Every result so far is single-seed and reads 8–16 held-out rows. Single-seed MoE parity is a lottery, so none of them can
carry a parity or a "beats" claim.

## Claims under test

- **K (knob):** which of the two knobs, or both, accounts for the shipped plateau (2×2 ablation on e4b, ≥3 seeds per cell).
- **P (position):** e4b in the default the K rule selects reaches held-out parity with, or beats, the stronger of Unsloth and
  axolotl at their pinned versions and native settings, under one evaluator and over three seeds.

## Fixture (TC1b's field recipe, unchanged except where marked NEW)

- Model `Qwen/Qwen3-30B-A3B` @ `ad44e777bcd18fa416d9da3bd8f70d33ebb85d39`. One RTX 5090 (32 GB) for every scored arm,
  and the same box draw for all of them.
- Data: TC1's tokens (alpaca template, `unsloth/alpaca-cleaned` at tp4_alpaca.py's pinned sha, registered split
  1,200 train / 48 held-out). The train-only sha is asserted per run.
- seq 2048, micro-batch 2 × accum 4, N = 200, adamw_8bit, lr 2e-4, wd 0.001, linear schedule with 5 warm-up steps.
  r 16 / α 16 on q/k/v/o plus every expert's gate/up/down (trainable 642,514,944 asserted on every arm).
- 4-bit base, named per framework: e4b = e4b NF4 experts + NF4 attention (`--attn-4bit 1`); Unsloth = bnb-4bit expert stacks
  (Params4bit) + bnb Linear4bit attention; axolotl = `load_in_4bit`, `quantize_moe_experts: true`, double-quant off. All are
  NF4 blocksize 64. TC1's frozen-base probe read e4b and Unsloth as SAME-BYTES-CLASS at layer 0. HP1 re-records it per arm.
- **NEW, `--order-seed S`:** the 1,200 train rows are permuted with `random.Random(S)` before the fixed (i·accum+j) layout.
  Today `--seed` moves only the init, so a three-seed spread would measure init noise alone. With the order seed, seed s
  moves both the init and the data order, and every arm at seed s sees the same order (paired). Default off = today's
  fixed order, byte-identical.
- Seeds: SELECTION seeds 1–3 = {3407, 3408, 3409} (the K block only); CONFIRMATION seeds 4–6 = {3410, 3411, 3412} (the P
  block only, never seen by the selection). For each arm and seed, `--seed s --order-seed s`, plus `matched:s` where the arm's
  init is kaiming (below). `--order-seed` lands as its own e4b PR (ruling item 4) before this registration.

## Arms (24 scored runs at most), in this order

**K block (selection), e4b only, 2×2 × seeds 1–3 = 12 runs.** All four are e4b 0.4x at one pinned SHA (GitHub main at registration),
fused path, attention 4-bit:

| cell | init | expert-adapter dtype | flags |
|---|---|---|---|
| K00 = shipped | N(0, 1/r) | bf16 | `--lora-init native --adapter-dtype native` |
| K10 | kaiming (PEFT's distribution) | bf16 | `--lora-init matched:s --adapter-dtype native` |
| K01 | N(0, 1/r) | fp32 | `--lora-init native --adapter-dtype fp32` |
| K11 = TC1b "matched" | kaiming | fp32 | `--lora-init matched:s --adapter-dtype fp32` |

**P block (confirmation), seeds 4–6, run after the K readout:** the K-selected cell E (3 runs), K00 as shipped (3 runs; dropped
when E = K00, so 9–12 runs in all), and both competitors at their native settings, as their users get them (3 runs each):
- `unsloth/ckpt_unsloth_native`: Unsloth 2026.9.14 + unsloth_zoo 2026.9.9, `venv-unsloth` (torch 2.12.1+cu130,
  `UNSLOTH_MOE_BACKEND=grouped_mm`, TC1's comparator configuration). Native init (`random_state=s`), native fp32 adapters,
  `use_gradient_checkpointing="unsloth"`.
- `axolotl/ckpt_axolotl_best`: axolotl 0.20.0 via its own ModelLoader, `--axolotl-best 1` (KernelsPlugin, scattermoe,
  moe_bnb_fast). Native PEFT init (seeded s), native adapters. This is axolotl's documented MoE configuration and the
  stronger axolotl row in steady16/18. Beating axolotl means beating its best.

Every run shares TC1's loop (`run_arm`): the same optimizer call, tokens, order and schedule. A framework's own trainer
(`axolotl train`, SFTTrainer) is NOT run: its batching and packing would change tokens per step. Stated, not hidden.

## The one evaluator (`bench/hp1/hp1_eval.py`, NEW)

Every arm's FINAL adapter (step 200: e4b `adapter_<tag>.pt`, PEFT `adapter_model.safetensors`) is scored by the same code.
No framework's eval loop is used.
1. **Canonical adapter export.** Each adapter's A/B tensors are mapped to canonical slots (layer, kind ∈ {q,k,v,o,gate,up,down},
   expert), fp32, [r, in] / [out, r], using the harness's T20 structural slot mapping (the code that already applies
   `matched:s` across e4b's fused layout and PEFT's per-expert layout). An unmapped or duplicate slot refuses the arm.
   Scaling α/r is applied once, from the arm's receipt.
2. **One frozen base.** Qwen3-30B-A3B at the pin, quantized once to NF4 (bitsandbytes 0.50.2 `quantize_4bit`, blocksize 64,
   no double quant). The bytes are hashed and the hash goes in the receipt. The same base object serves every arm.
3. **One forward.** HF transformers 5.18.0 `Qwen3MoeForCausalLM`, eager attention. Each NF4 weight is dequantized by
   bitsandbytes' own dequantize to bf16, and the LoRA delta (x·Aᵀ·Bᵀ·α/r) is added in fp32 to the projection output
   before the SwiGLU. bf16 autocast, as in training. No fused kernel of any framework is on this path.
4. **Rows and masking.** The 48 registered held-out rows (TC1b read 16), the alpaca template, labels = input ids (TC1's
   eval_loss convention; the identical token count per row is asserted per arm). The score per arm is the mean over rows
   of the per-row token-mean CE, plus the token-weighted mean, REPORTED beside it.
5. **Base identity (STRICT; ruling item 3).** Each training run hashes the base it TRAINED on: every frozen slot (layer, kind,
   expert) dequantized the framework's own way to bf16, sha256 per slot in canonical order, folded into one `train_base_digest`
   (extending T21's layer-0 probe to every layer). The evaluator computes the same digest over its own base. An arm whose
   digest differs (quant scheme, block size, double quant) is ALSO scored on its own training base by the same scoring
   function (`hp1_eval.score_rows`), run inside that arm's process at step 200 on its own model. Both numbers are reported,
   and the verdict uses the own-base number for that arm. A framework is never scored on a base it did not train against.
   One export serves several frameworks ONLY when their re-derived digests match. The shared base is a checked premise, re-derived
   per arm on the box, never assumed (maintainer, bus 20261018T002045Z-34273). On the A2000 cut, e4b and Unsloth-t28 trained on
   the bitwise same dequantized base (digest 5e7564ea, NF4/64 with nested statistics), so there one export served both.
6. **Self-checks (STRICT).** (a) A zero-adapter row reproduces the base held-out within 1e-4 of each arm's in-loop step-0
   loss class. (b) The e4b and PEFT adapters of K11-seed-3407 and the Unsloth arm at step 0 (all B = 0) score identically.
   (c) A byte-flipped adapter changes the score (control).
7. **Cross-check (REPORTING, never a verdict).** Each arm's own in-loop step-200 held-out on the same 48 rows, and its
   difference from the evaluator's number.

## Readings and decision rules (RULED 2026-10-17T22:47Z; fixed before any scored number)

Notation: y(arm, s) = the evaluator's step-200 held-out loss for arm and seed s. **Lower held-out loss is better**: every
one-sided test points that way. σ_seed = the pooled within-arm SD of y across seeds, over every arm's scored runs (K and P
blocks together). It is the lane's seed-to-seed unit.

**K readout (selection, seeds 1–3; 2×2 on e4b, df_error = 8).** Two-way ANOVA on y over (init, dtype), with the interaction
term. Reported: the init main effect, the dtype main effect and the init × dtype interaction, each ± SE with its p.
- A knob is CAUSAL iff its main effect (native − kaiming for init; bf16 − fp32 for dtype) is > 0, i.e. the shipped
  setting gives the higher loss, at one-sided p < 0.05 (t, df 8). Otherwise it is NOT-SHOWN.
- **Without interaction** (interaction p ≥ 0.05), the selected cell E is the cheapest cell whose mean is within 1 σ_seed of
  the best cell's mean. The cheapness order is K00 < K10 < K01 < K11: bf16 adapters save memory and time, and init is free.
- **With interaction** (p < 0.05), the main effects are reported but do not decide. E is chosen on CELL MEANS: the best
  cell, then the cheapest cell within 1 σ_seed of it.
- σ_seed for this rule is the K block's own pooled SD (df 8). The final σ_seed (all arms) is used for P.
- The selection is mechanical: the reducer applies this rule on the box between the blocks, with no human choice. If
  E = K00, the shipped default stands and the plateau is reported as seed noise.
- E's setting is FILED as an e4b default change (a default change is RED). HP1 changes no default.

**P readout (confirmation, FRESH seeds 4–6; never the selection seeds, so no winner's curse).** C* = the competitor with the
lower mean on seeds 4–6. Paired by seed: Δ_s = y(E, s) − y(C*, s). Δ̄ = their mean, SE = sd(Δ_s)/√3. CI = the 90% two-sided
t-interval Δ̄ ± 2.920 · SE (df 2). Margin m = min(max(σ_seed, 0.005), 0.02).
- **BEATS** iff CI upper < 0 AND |Δ̄| > σ_seed.
- **BEHIND** iff CI lower > 0 AND |Δ̄| > σ_seed.
- **PARITY** iff the whole CI lies within [−m, +m] (equivalence by TOST, not a point estimate).
- Otherwise **INCONCLUSIVE**, with Δ̄, the CI, m and σ_seed stated.

Stated up front: with three seeds the CI half-width is 2.92 · SE. INCONCLUSIVE is a likely and honest outcome, and it is
reported as such, never rounded to PARITY. The same reading is reported for K00 (as shipped today) vs C* on seeds 4–6, and
for E vs each competitor separately.

**Validity (STRICT, per run).** TC1's predicates: the tokens/train sha, the trainable count, C1, the frozen-base probe
class, matched-init completeness on kaiming e4b cells, the K-cell dtype census (bf16 expert adapters on K00/K10, fp32 on
K01/K11), and the order-seed permutation identical across arms at each s: every run carries a non-null `order` whose
`order_seed` equals its seed, and every arm compared at seed s has the same `order_sha`. A mismatch VOIDs that pair; the reducer
reads K_INCOMPLETE / P_INCOMPLETE and names the arm (maintainer, bus 20261017T233104Z-7528: required, not optional). A VOID run is redrawn once on the same box.
A second VOID voids the cell, and the readout states it.

## Shakedown ($0, before any paid minute; correctness only, never a number; exception RULED approved, item 5)

On the seat A2000 (driver 575, CUDA 12.9), `TC1_LOCAL_BOX=1`:
1. `--order-seed` selftest. Off is byte-identical to today's tokens and layout. On permutes the train rows, and two
   frameworks at the same s get the same order sha.
2. Every e4b K cell, 3 steps, resident on a 2-layer truncation of the pinned checkpoint (real weights, `hp1_trunc.py`): the
   flags take, the dtype census matches the cell, the canonical adapter and the base export write, and the training-base digest
   proves the evaluator's dequantizer bitwise on layer 0. (Resident on the cut is the box's own path; the full model fits the
   12 GB card only under offload, which no scored arm uses.)
3. Unsloth 2026.9.14 in `venv-unsloth-t28` (torch 2.8.0+cu128) on a 2-layer truncation of the pinned checkpoint (real
   weights; config num_hidden_layers 2): it loads, trains 3 steps and saves a PEFT adapter.
4. `hp1_eval.py` on the 2-layer truncation: all three self-checks, and canonical export of an e4b and a PEFT adapter.

**Cannot be shaken on the seat:** `venv-unsloth` (cu130) and `venv-axolotl` (torch 2.14 cu130) need driver ≥ 580. Both
have VALID paid-box precedents in this harness at these pins (tc1-5090-15, steady16/18). The box therefore runs a box-side
shakedown first: the 2-layer truncation through the cu130 Unsloth arm and the axolotl arm, 3 steps each, plus the
evaluator. It runs before any scored arm (~10 min, inside the cap), and a failure ends the draw before scoring.

**Result, 2026-10-18 (A2000, branch heldout-parity/hp1-prereg on order-seed 6c7a3166): PASS on run 2.**
- **Run 1 was VOIDED by the evaluator's STRICT zero-adapter check (+0.437 nats on every arm).** `hp1_base_slots` bound every
  expert slot to the last layer's dequantizer (a late-binding closure), so layer 0 exported layer 1's experts. The digest could
  not see it, because every framework went through the same walker. Fixed; a regression test fails without the fix.
- **Run 2: e4b K00/K10/K01/K11 and Unsloth 2026.9.14 (venv-unsloth-t28), 3 steps each.**
  - Every arm was CELL OK, with C1 bit-exact and the dtype census each cell requires.
  - Matched init was complete on K10/K11, and every arm had one order_sha (25d50439).
  - Evaluator `--require-all-scored`: rc 0, with all five arms SCORED.
  - Zero adapter vs in-loop step 0: −0.0018 (e4b) and −0.0023 (Unsloth).
  - Evaluator vs in-loop step 3: within 0.0047 on every arm, so the canonical export round-trips both layouts.
  - The perturbed-adapter control moved the score.
  - The digest equalled each training digest.
- The driver's `qwen3hp1` flow was dry-run with stubbed harness calls: the full path, a failed box-side shakedown (the draw ends
  unscored) and an empty K selection (the P block is skipped).

## Budget and stop rules

**Envelope (for the maintainer's ACK; nothing rents before it):**
- **Run id:** `hp1-5090-1`. Token `qwen3hp1`, staged from main after this registration merges (with `bench/hp1/*` via
  TC1_EXTRA_STAGE).
- **Host rule:** one RTX 5090 32 GB on RunPod secure or Vast verified-secure. Driver ≥ 580 (the cu130 venvs), host RAM ≥ 98 GB,
  free disk ≥ 300 GB. The disk holds the 57 GB checkpoint, about 20 GB of venvs, the base exports (about 17 GB each; one when
  the digests match), about 2.6 GB per canonical adapter × 27, and receipts.
- **Spend:** cap $15 (the owner's), guard 10 h, teardown by explicit pod id with a heartbeat watch. The live list is posted
  after, and the spend is disclosed on the bus.
- **Two strikes:** two consecutive HARNESS_ERROR / VOID / NOT_RUN draws stop the lane.

The re-estimate below adds the evaluator's eager forward (a per-expert dequant + matmul on every forward), measured on no 30B
yet; the A2000 cut gives no 30B timing.

One RTX 5090, RunPod secure (the 5090 row, $1.10/h) or Vast verified-secure. Estimate:
- e4b: up to 18 runs (12 K, 3 E, 3 K00) × ~13 min (TC1b: 3.16 s/step plus prologue), ≈ 3.9 h
- Unsloth: 3 × ~17 min, ≈ 0.85 h
- axolotl: 3 × ~19 min, ≈ 0.95 h
- venvs, model fetch and the box-side shakedown: ≈ 0.75 h
- evaluator: up to 24 arms × ~4 min plus two base loads, ≈ 1.7 h (not yet measured at 30B)

That totals ≈ 8.2 h ≈ $9.00. Cap $15 (owner, 2026-10-17). Guard 10 h ($11 worst case), so a full redraw does not fit inside
the cap; the two-strikes rule applies instead. Teardown by
explicit pod id, with a heartbeat watch. The live list is posted after. Receipts go into the store under #225's tensor
rule. Two strikes (two consecutive HARNESS_ERROR / VOID / NOT_RUN draws) stops the lane and sends a simplification
proposal to the maintainer. The K block runs first, so a deadline cannot eat the knob readout. The reducer then applies the K rule, and the P block runs second.

## Predictions (scored HELD / FALSIFIED / UNTESTED)

- **H1:** the init knob is CAUSAL (native N(0,1/r) is about 5x PEFT's scale, which matches the low-train/high-held-out signature).
- **H2:** the dtype knob is NOT-SHOWN.
- **H3:** the K-selected cell is K10 (kaiming, bf16): the gap closes at no memory cost.
- **H4:** E vs C* reads PARITY or INCONCLUSIVE. BEATS is not predicted: TC1b's matched pair differed by 1e-4.
- **H5:** K00 vs C* on seeds 4–6 reads BEHIND or INCONCLUSIVE.

## Amendments

(none yet)
