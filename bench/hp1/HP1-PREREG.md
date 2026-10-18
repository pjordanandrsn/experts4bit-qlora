# HP1 — held-out parity: e4b as shipped vs Unsloth vs axolotl on Qwen3-30B-A3B, one evaluator, three seeds per arm

Status: registered when this file merges, before any box. The maintainer ruled the bar on 2026-10-17 (22:47Z) and approved it with
five changes, all folded in below. The later binding rulings are folded in too:
- an unequal order_sha VOIDs a pair (2026-10-17);
- the shared base is a checked premise;
- the box-side shakedown ends the draw cleanly;
- the run id, host rule and spend go to the maintainer as an envelope for ACK;
- every STRICT failure VOIDs, fail-closed, and a TC1-invalid run is never scored;
- every verdict uses evaluator numbers only;
- 6(b) is the round trip;
- an early base check ends the draw on a systematic miss, and base-level VOIDs are never redrawn;
- the headline needs both competitors (Amendment 3: Unsloth's cell; a failed axolotl is dropped);
- the redraw rule is as implemented (2026-10-18).

Scoring starts only after this file merges and the box-side shakedown passes.

Lane HP1 (the owner's direction, 2026-10-17); issue experts4bit-qlora#835. The pieces:
- the TC1 harness `bench/tc1/tc1_arm.py`. Its `--order-seed` and `--hp1-export` / `--hp1-export-base` hooks landed in e4b #1709,
  and Q235 Amendment 10 re-seals it.
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
  init is kaiming (below). `--order-seed` landed in e4b #1709.

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

Every arm's FINAL (step-200) adapter is scored by the same code. No framework's eval loop is used.
1. **Canonical adapter export (in the training process, `--hp1-export`).** The harness's own `lora_slots` maps every LoRA slot
   to (layer, kind ∈ {q, k, v, o, gate_up, down}, expert): e4b's fused ExpertsLoRA / LoRALinear, PEFT's ParamWrapper / Linear.
   Each slot is written in fp32 as A [r, in] / B [out, r], with the scaling its module applies. An unmapped or duplicate slot,
   shapes that do not chain, or mixed scalings within one stack, refuse the export.
2. **The bases: what each run trained on, as stored.** Each framework's first run (`--hp1-export-base 1`) writes its frozen base AS
   STORED: packed NF4 per slot, the absmax as the framework's dequant multiplies by it (nested statistics un-nested), the code,
   the blocksize, the dtype, and the bf16 non-quantized slots.
   - Every run records its training-base digest: each frozen slot dequantized the framework's own way, fingerprinted, and folded
     with sha256. On the first layer it also proves that the evaluator's stored-slot dequant is bitwise the framework's.
   - The evaluator re-derives each export's digest and scores an arm only on the export whose digest equals that arm's training
     digest. One export serves several frameworks only when their digests match. The shared base is a checked premise, per arm,
     never assumed.
   - On the A2000 cut, e4b and Unsloth-t28 trained on the bitwise same dequantized base (NF4/64 with nested statistics).
   - An arm whose training base differs from another framework's is scored by the evaluator on its OWN exported base, with the
     same STRICT checks on that base.
   - An arm that no export claims has no evaluator number. It is a run-level VOID, and its redraw re-exports its framework's base.
     If its own base is still not exported, or fails a STRICT check, the arm is unusable and its cell incomplete
     (K_INCOMPLETE / P_INCOMPLETE). An in-loop number never stands in for it: a mixed pair would leave the in-loop-vs-evaluator
     offset (up to the 0.02 round-trip bound, the margin's cap) uncancelled.
3. **One forward.** It uses HF transformers 5.18.0's `Qwen3MoeForCausalLM` skeleton with eager attention. Each layer's experts
   and q/k/v/o are replaced by `hp1_lib`'s eager modules:
   - every weight is dequantized from the stored slot by bitsandbytes' `dequantize_4bit` to bf16, and the projection runs in bf16;
   - the LoRA delta (x·Aᵀ·Bᵀ·scaling) is added in fp32 before the SwiGLU, and the routing weights accumulate in fp32;
   - bf16 autocast, as in training. No fused kernel of any framework is on this path.
4. **Rows.** The 48 registered held-out rows (TC1b read 16), with TC1's `eval_loss` convention (labels = input ids). y is the mean
   over rows of each row's token-mean CE; the token-weighted mean is REPORTED beside it.
5. **STRICT checks.** Each check VOIDs and fails closed:
   - (a) **Base digest.** An export's re-derived digest must equal its training run's. Otherwise the base is VOID, and so is every
     arm that trained on it.
   - (b) **Zero adapter.** With no adapter, the evaluator's score is within **0.005** (TC1's SAME-BYTES-CLASS bound) of every
     matched arm's in-loop step-0 held-out loss. Otherwise the base and its arms are VOID. An arm with no step-0 read is VOID.
   - (c) **Round trip ("6(b)").** For each arm, the evaluator's step-200 score is within **0.02** (TC1b's fixed band) of the arm's
     own in-loop step-200 score. This proves its layout exported correctly. Otherwise the arm is VOID. *Changed from the earlier
     draft:* 6(b) was a step-0 identity between e4b's and PEFT's adapters. With every B = 0 at step 0 it is vacuous, so the
     maintainer ruled the round trip in its place (2026-10-18). It is a gross-error detector: an export or layout bug lands near
     +0.4 nats, not 0.02.
   - (d) **Control.** A perturbed adapter moves the score. Otherwise the base and its arms are VOID.
   - (e) **Adapter file.** The file's sha256 equals the receipt's.
6. **TC1 validity, read by the evaluator before any scoring.** A TC1-invalid run is VOID, never scored. The run must have:
   - receipt status `ok`, and C1 bit-exact;
   - the tokens sha;
   - trainable = 642,514,944 (asserted on every arm, e4b included);
   - matched-init completeness on kaiming cells;
   - the dtype census: bf16 expert adapters on the native e4b cells (K00, K10), fp32 only on fp32 cells;
   - an `order` carrying its own seed;
   - an HP1 export with the stored-slot proof;
   - in-loop reads at step 0 and step 200.
7. **Cross-check (REPORTING).** For every scored arm, the evaluator's score minus its in-loop step-200 score.

**Every verdict uses evaluator numbers only.** An in-loop number never becomes a y; it appears only in the STRICT checks above and
in REPORTING. A shared evaluator offset therefore cancels in every paired difference.

**Registered constants are enforced, fail-closed.**
- The reducer refuses an evaluator output that read other than 48 held-out rows.
- A run whose receipt says other than N = 200 steps gives no y, so its cell is incomplete.
- A forwarded controller knob can never score silently. The seeds are fixed by the tag rules.

**Each VOID carries a level.** A run-level VOID (TC1-invalid, the adapter file, the round trip, or no export claiming the arm) can
be redrawn. A base-level VOID (the base's digest, its zero-adapter check, or its control failed) is a systematic miss, and a redraw
cannot fix it.

The evaluator writes its JSON after every base, so a deadline mid-pass keeps what was scored.

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
Checked in this order; the first that holds is the verdict (with σ_seed below 0.005, BEATS can also satisfy PARITY, and
BEATS is read):
- **BEATS** iff CI upper < 0 AND |Δ̄| > σ_seed.
- **BEHIND** iff CI lower > 0 AND |Δ̄| > σ_seed.
- **PARITY** iff the whole CI lies within [−m, +m] (equivalence by TOST, not a point estimate).
- Otherwise **INCONCLUSIVE**, with Δ̄, the CI, m and σ_seed stated.

Stated up front: with three seeds the CI half-width is 2.92 · SE. INCONCLUSIVE is a likely and honest outcome, and it is
reported as such, never rounded to PARITY. The same reading is reported for K00 (as shipped today) vs C* on seeds 4–6, and
for E vs each competitor separately.

**The headline needs both competitors.** C* is defined over both competitors' complete confirmation cells. If either competitor's
cell is incomplete after its redraw, the headline reads **P_INCOMPLETE**. The complete competitor's reading is reported beside
it, labelled per-competitor, never as the headline. C* is chosen on the same seeds it is compared on. That leans E vs C* slightly
toward BEHIND, and the per-competitor readings show both.

**Validity (STRICT, per run).** TC1's predicates: the tokens/train sha, the trainable count, C1, the frozen-base probe
class, matched-init completeness on kaiming e4b cells, the K-cell dtype census (bf16 expert adapters on K00/K10, fp32 on
K01/K11), and the order-seed permutation identical across arms at each s: every run carries a non-null `order` whose
`order_seed` equals its seed, and every arm compared at seed s has the same `order_sha`. A mismatch VOIDs that pair; the reducer
reads K_INCOMPLETE / P_INCOMPLETE and names the arm.

**Early base check (as implemented in the `qwen3hp1` token).** The first K run exports the e4b base. The evaluator checks that
base and run at once: the digest, the zero-adapter check against that run's step 0, and its validity. This happens before the
other eleven K runs are paid for.
- A run-level VOID there gets its one redraw.
- A base that is not OK after that ENDS THE DRAW (EARLY_BASE_CHECK_FAILED). A systematic miss then costs about one run, not the K
  block.

**Redraw (as implemented).**
- After each block's evaluator pass, every RUN-level VOID is redrawn ONCE on the same box, with the same invocation plus its
  framework's base export. Its first receipt is kept as `hp1/void/<name>.a1`. Then the block is evaluated again.
- A base-level VOID is never redrawn.
- A second VOID voids that cell, and the readout states it.
- An incomplete K block (any K cell without three usable runs) skips the P block (K_INCOMPLETE).

## Shakedown ($0, before any paid minute; correctness only, never a number; the exception is approved by the maintainer)

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

**Result (A2000, 2026-10-18): PASS on run 3, at this registration's code.**
- **Run 1 was VOIDED by the evaluator's STRICT zero-adapter check (+0.437 nats on every arm).** `hp1_base_slots` bound every
  expert slot to the last layer's dequantizer (a late-binding closure), so layer 0 exported layer 1's experts. The digest could
  not see it, because every framework went through the same walker. It was fixed in #1709, with a regression test.
- **Run 2 passed, before the review changes.**
- **Run 3: e4b K00/K10/K01/K11 and Unsloth 2026.9.14 (venv-unsloth-t28), 3 steps each.** It was trained with `bench/tc1/tc1_arm.py`
  sha256 `9f43b45b267d3b4fbc0855fd26e8b1c15de7e92832344fe42c74ecba5e2a4939` (as merged in #1709) and `bench/hp1/hp1_lib.py`
  `d951d1b21e555da99009dfbee3881821557a78aa97256a4bb2ecdee3d323e1c8`. Both are byte-identical to this registration.
  - The evaluator pass first ran out of device memory on the duplicate base, because the first base's model was still held.
    The fix: `evaluate()` drops each base's scorer and releases the device before the next base, with a regression test. The
    final evaluator, `bench/hp1/hp1_eval.py` `ffbda48a6c671547de3e5d67b12948bace6f7372112c48c507c907d346caf3da` (this
    registration's), re-ran over run 3's arms and passed:
    - `--require-all-scored` rc 0, with all five arms SCORED;
    - zero adapter −0.0018 (e4b) and −0.0023 (Unsloth);
    - round trip −0.0011 to −0.0039, against the 0.02 bound;
    - the perturbed-adapter control moved the score;
    - one base digest, 5e7564ea, whose duplicate (Unsloth's export) was verified and UNUSED.
- The driver's `qwen3hp1` flow was dry-run with stubbed harness calls. The paths covered:
  - the full path;
  - a failed box-side shakedown (the draw ends unscored);
  - the early base check: OK, base-level VOID (the draw ends after one run), and run-level VOID (redrawn, then OK);
  - an empty K selection (the P block is skipped);
  - a run-level VOID K run (redrawn once, then re-evaluated);
  - the P pass's time gate.

  These are committed as `tests/test_hp1_driver.py`. It extracts the HP1 functions from `tc1_run.sh` and runs them with stubbed
  harness calls; a mutation of the early check is caught.

## Budget and stop rules

**Envelope (for the maintainer's ACK; nothing rents before it):**
- **Run id:** `hp1-5090-1`. Token `qwen3hp1`, staged from main after this registration merges (with `bench/hp1/*` via
  TC1_EXTRA_STAGE).
- **Host rule:** one RTX 5090 32 GB on RunPod secure or Vast verified-secure. Driver ≥ 580 (the cu130 venvs), host RAM ≥ 98 GB,
  free disk ≥ 300 GB. The disk holds the 57 GB checkpoint, about 20 GB of venvs, the base exports (about 17 GB each; one when
  the digests match), about 2.6 GB per canonical adapter × 27, and receipts.
- **Spend:** cap $15 (the owner's), guard 10 h, teardown by explicit pod id with a heartbeat watch. The worst case per provider
  at the guard:
  - RunPod secure: 10 h × $1.10/h = **$11.00** (no download billing);
  - Vast verified-secure: 10 h × $0.85/h + about 72 GB of downloads (the 57 GB checkpoint plus wheels) × $0.011/GB ≈ $9.29.
  The run's worst case is the larger, **$11.00**, inside the $15 cap. The live list is posted
  after, and the spend is disclosed on the bus.
- **Two strikes:** two consecutive HARNESS_ERROR / VOID / NOT_RUN draws stop the lane.

The re-estimate below adds the evaluator's eager forward (a per-expert dequant + matmul on every forward), measured on no 30B
yet; the A2000 cut gives no 30B timing.

One RTX 5090, RunPod secure (the 5090 row, $1.10/h) or Vast verified-secure. Estimate:
- e4b: up to 18 runs (12 K, 3 E, 3 K00) × ~13 min (TC1b: 3.16 s/step plus prologue), ≈ 3.9 h
- Unsloth: 3 × ~17 min, ≈ 0.85 h
- axolotl: 3 × ~19 min, ≈ 0.95 h
- venvs, model fetch and the box-side shakedown: ≈ 0.75 h
- evaluator: up to 24 arms × ~4 min plus two base loads, ≈ 1.7 h (not yet measured at 30B). Each pass is gated on the last pass's
  measured seconds per arm × its own arm count × 1.3 + 10 min (the K pass is sized from the early pass × 12), and redraws are gated
  on their own time-left check.

That totals ≈ 8.2 h ≈ $9.00. Cap $15 (owner, 2026-10-17). Guard 10 h ($11 worst case), so a full redraw does not fit inside
the cap; the two-strikes rule applies instead. Teardown by
explicit pod id, with a heartbeat watch. The live list is posted after. Receipts go into the receipt store. Two strikes (two consecutive HARNESS_ERROR / VOID / NOT_RUN draws) stops the lane and sends a simplification
proposal to the maintainer. The K block runs first, so a deadline cannot eat the knob readout. The reducer then applies the K rule, and the P block runs second.

## Predictions (scored HELD / FALSIFIED / UNTESTED)

- **H1:** the init knob is CAUSAL (native N(0,1/r) is about 5x PEFT's scale, which matches the low-train/high-held-out signature).
- **H2:** the dtype knob is NOT-SHOWN.
- **H3:** the K-selected cell is K10 (kaiming, bf16): the gap closes at no memory cost.
- **H4:** E vs C* reads PARITY or INCONCLUSIVE. BEATS is not predicted: TC1b's matched pair differed by 1e-4.
- **H5:** K00 vs C* on seeds 4–6 reads BEHIND or INCONCLUSIVE.

## Amendments

### Amendment 1 (2026-10-18, before any box): the bulk files stay on the box, and their identity comes back

The base exports (about 17 GB each), the canonical adapters (about 2.6 GB each) and the shakedown cut stay on the box. The driver's
fetch excludes them (e4b #1720), and they are gone when the box is torn down. The record pins them anyway:
- every receipt carries its adapter's sha256 and its training-base digest;
- the box writes `hp1/bulk_manifest.json` before the fetch, with the path, size and sha256 of every excluded file;
- the manifest comes back with the receipts.

The evaluator's checks therefore rest on the receipt digests plus this manifest. No rule, threshold or registered reading changes.

**Note to Amendment 1 (2026-10-18, after `hp1k-5090-8`; no rule, threshold or registered reading changes).** The shakedown's own
canonical adapters (`hp1/shake/canonical_*.safetensors`, about 107 MB each at the 2-layer cut) now come back with the receipts, so a
shakedown evaluator VOID can be re-scored off the box (#1767). Their receipts already pin their sha256s, and the bulk manifest stops
listing them. The K and P blocks' canonical adapters, every base export and the shakedown cut stay on the box as above.

### Amendment 2 (2026-10-18, before any box; takes effect only if the maintainer picks the split path): HP1 on two boxes of at most 6 h each

**Why.** The compute policy caps a rental at 6 h of wallclock. The one-box draw estimates about 7.5 h of work, so it needs either a
higher cap or this split. The cut falls on the line the registration already draws between selection and confirmation.

**Box 1, `hp1k-5090-1` (selection; token `qwen3hp1k`).**
- Steps: the cu130 box-side shakedown, the early base check after the first K run, the twelve K runs, the K evaluator pass (with the
  one-redraw rule), and the mechanical selection.
- It carries out `eval_K.json`, `select.json` and its `bulk_manifest.json` (Amendment 1), and prints their sha256s to its summary.
  Box 2's execution request pins all three.
- About 4.0 h of estimated work.

**Box 2, `hp1p-5090-1` (confirmation; token `qwen3hp1p`).**
- Box 2 REFUSES unless the staged `eval_K.json`, `select.json` and box 1's `bulk_manifest.json` match the pinned sha256s
  (`TC1_HP1_K_EVAL_SHA256`, `TC1_HP1_K_SELECT_SHA256`, `TC1_HP1_K_MANIFEST_SHA256`), and unless the selection re-derives from the
  staged `eval_K.json` (checked against box 1's manifest).
- Then come its OWN box-side shakedown, an early base check on its first E run (which exports its e4b base), the P block, and the
  P evaluator pass (with the one-redraw rule). The verdict is read over box 1's `eval_K.json` plus box 2's `eval_P.json`. The
  reducer checks every scored adapter and base against the union of both boxes' manifests.
- About 4.5 h of estimated work.

**Each box.**
- The same card class (one RTX 5090).
- The receipts record the driver and machine id.
- One registered redraw id each (`-2`), and the two-strikes rule per box.
- Pre-create spares (maintainer, 2026-10-18): box 1 `hp1k-5090-6`, `-7`, `-8`, `-9`, `-10`, `-11`. A refusal before any instance exists consumes an id and is not a strike; a box-started failure counts as the rulings say.

**The limitation, stated.** K and P run on different boxes of the same class.
- Every verdict pairs only within box 2: E vs C* on the same box, at the same seeds.
- A box-to-box shift can move σ_seed only through the pooled K groups, never a paired difference.
- σ_seed is REPORTED per block (per box) beside the pooled value the verdict uses (`sigma_seed_by_block`).

**Spend.** Each box's worst at its 6 h guard is 6 × $0.85 + about 72 GB × $0.011 ≈ $5.89 on Vast, or 6 × $1.10 = $6.60 on RunPod.
Both boxes together come to $11.78 / $13.20, under the owner's $15. The disk is about 140 GB needed on box 1 (vast_disk_gb 250)
and about 175 GB on box 2 (vast_disk_gb 300).

### Amendment 3 (2026-10-18, before any scored run): any further axolotl failure drops axolotl; the headline then reads E vs Unsloth

**Why.** Box 1 drew twice, and both draws ended at the box-side shakedown before any scored run:
- `hp1k-5090-1`: a tokens-file fam label (fixed in #1735).
- `hp1k-5090-3`: the axolotl shakedown arm, offline, refused at load. axolotl's KernelsPlugin (`--axolotl-best 1`) fetches its hub kernels by version. `arm_once` let the Hub online only for TC1's tag prefix `axolotl/ckpt_axolotl_best*`, and HP1's axolotl tags don't match it. Fixed in the same change as this amendment: an axolotl arm carrying `--axolotl-best 1` is online whatever its tag.

The maintainer's ruling (strike 2) continues the lane in a simplified form. One more axolotl failure should not spend another box.

**The rule.**
- Any axolotl failure from here on drops axolotl from HP1: a box-side shakedown arm, or a scored P arm with no evaluator y on some confirmation seed.
- The headline then reads E vs Unsloth (C* = Unsloth), and the readout says axolotl was dropped and why (`axolotl_dropped`).
- The same applies to a data-order problem that disappears once axolotl's runs are left out: axolotl is dropped, and the reading is not refused.
- "The headline needs both competitors" (P readout) becomes: **the headline needs Unsloth's complete confirmation cell**. With both complete, C* stays the stronger one, as registered.

**On the boxes.**
- Box 1: an axolotl shakedown failure is recorded (`HP1 AXOLOTL_DROPPED`) and does not end the draw. The K block is e4b only. e4b and Unsloth still gate the shakedown, and the evaluator reads only their receipts and bases.
- Box 2: `TC1_HP1_DROP_AXOLOTL=1` (set by the controller when box 1's summary records the drop) runs no axolotl arm anywhere. Without it, box 2's own shakedown applies the same rule as box 1.

**Unchanged.** Every threshold, the K rule, σ, m, the verdict bands, the evaluator, both boxes' seeds and the spend cap. A box 2 redraw does not fit inside the cap and goes back to the maintainer.

### Amendment 4 (2026-10-18, before any scored y and before the next box): the 48-layer zero-adapter tolerance, from a calibration that can fail

**Why.** `hp1k-5090-9` cleared the shakedown and trained `hp1_k00_s3407` on the full 48 layers (in-loop held-out 1.914 -> 0.870 over 200
steps). Then the early base check VOIDed the e4b base:
- The base's digest matched its training run (12,723 slots).
- The STRICT zero-adapter check read evaluator 1.921029 against in-loop step 0 1.914030: +0.006999 against the 0.005 bound.
- That bound was set on the 2-layer cut, where the same check reads -0.0023 (e4b) and +0.0010 (Unsloth), and this registration says
  the evaluator was "measured on no 30B yet".
- It is an instrument constant failing at depth, not a training fault.
- A constant picked after seeing 0.37% would be tuned to pass. So the tolerance comes from a calibration whose rules are fixed here,
  and which can close HP1.

**The calibration box (`qwen3hp1cal`, run id `hp1cal-5090-1`, pre-create spare `-2`).**
- One RTX 5090 under the registered host rule, worst $0.90 (0.75 h × $0.85 + 65 GB × $0.004/GB, the per-GB rate capped in its
  manifest). Same source as box 1.
- One e4b K00 run (seed 3407, 1 optimizer step), its base exported as the boxes build it.
- It runs on 48 CALIBRATION rows: held-out rows 49–96 of the same unpacked prepare and template. The eval set is a prefix, so rows
  1–48 are HP1's own held-out rows and 49–96 are disjoint from them.
- The one evaluator then reads the per-row zero-adapter losses, and the per-row gaps against the run's in-loop step-0 rows
  (`zero_adapter_row_gaps`). Nothing on this box is a scored number.

**The rules (`bench/hp1/hp1_calib.py`).**
1. **Shape (fail-closed).** The per-row gaps must look like numeric drift. If any row's |gap| > 5 × the median |gap|, the two paths
   differ (template, truncation or masking), and HP1 CLOSES (`CLOSE_SHAPE`).
2. **Tolerance.** `zero_adapter_tol` at 48 layers = 2 × the mean |gap| over the 48 calibration rows, rounded up to 1e-6.
3. **Pre-box ceiling.** HP1's registered parity margin is m = min(max(σ_seed, 0.005), 0.02), which depends on σ_seed and is unknown
   before data. Before any box: if tol > 0.01 (half the 0.02 cap, which no σ_seed could rescue), HP1 CLOSES (`CLOSE_CEILING`). The 0.005
   floor is not a pre-box ceiling: that would decide the result before the measurement.
4. **Verdict-stage ceiling.** At the verdict, the reducer CLOSES the headline (`CLOSED_TOL`) if the largest tolerance any input used is
   > m/2 for the realized m.
5. **Per-row output.** `hp1_eval.py` writes the per-row zero-adapter losses and gaps from now on, so a future VOID can be diagnosed
   from the receipt. `hp1k-5090-9`'s cannot be.

**Then.** The calibration verdict and `calib.json` (sha256 printed as `HP1_CARRY`) are posted, and the maintainer re-derives them.
- Box 1 runs as `hp1k-5090-10` (the last box 1 under any rule; `-11` is a pre-create spare only), from the same source.
- Its command pins `TC1_HP1_ZERO_TOL` to the derived tol, used by every full-depth evaluator pass (early, K, P).
- The 2-layer shakedown keeps 0.005: it was calibrated there.
- Box 2 carries the same pin.
- If rule 1 or 3 fires, HP1 closes. Its only datum is then K00's single in-loop curve (1.914 -> 0.870, 200 steps, 48 layers), reported
  as such.

**Dropped from the ruling's draft.** An A2000 2-layer parity step. It existed to validate an A2000 calibration, and the calibration
runs on the box class itself. The A2000 cannot run the boxes' resident step-0 path at 48 layers, and `tc1_arm` refuses an HP1 export
under offload.

**Spend.** The maintainer raised HP1's envelope to $16.50: billed $3.256 + calibration ≤ $1 + box 1 $6.09 + box 2 $6.09 = $16.44.
Every other threshold, rule and registered reading is unchanged.
