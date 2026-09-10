# TP4 — SAME-BOX TRAINING HEAD-TO-HEAD: e4b (GitHub `main`) vs Unsloth vs plain HF+PEFT+bnb, EVERY SUPPORTED MoE FAMILY, THE FIELD'S RECIPE, THREE BOXES IN PARALLEL (pre-registered 2026-09-10, before any box is rented)

**Owner directive (Jordan, 2026-09-10, chat, verbatim):** *"set up and run a same box training comparison between my packages (get the latest from gh) and the rest of the field to we have concrete numbers on the most heavily benchmarked examples that are run in the exact same way on the same box. true 1:1 on every supported model. run multiple boxes in parallel to save time."*

Lineage: tp1 (P36, internal parity, six families) → P38 (Qwen3 vs Unsloth) → tp2 (P40, six families vs Unsloth, one box, tp2's fixture) → **tp4**. tp3 (`bench/tp3/tp3_arm.py`) is the harness lineage (tp2 + the structural attention census, e4b#434/#435). tp4 changes three things at once and says so: (1) **e4b comes from GitHub `main`**, not the shipped PyPI cut; (2) **the fixture is the field's most-run recipe** — the Unsloth notebooks' Alpaca recipe — not tp2's clinical seq-512 fixture; (3) **a third framework**, the plain Hugging Face stack (transformers + bitsandbytes 4-bit + PEFT), the thing every other trainer (LLaMA-Factory, Axolotl's default path) reduces to. Because the fixture changed, **one pair is repeated at tp2's fixture** (the Qwen3 anchor, below) so this lane's boxes are tied to the two earlier lanes' numbers.

## Claim under test

For each supported MoE family, on the identical QLoRA problem run the identical way on one machine: (a) does each framework train it in the 4-bit regime at all (support: OK / REFUSED / OOM / INSTALL_FAILED / LOAD_FAULT / HARNESS_ERROR / ALARM / NOT_RUN — a refusal is a row, never coerced), (b) is quality comparable at N steps (held-out loss), (c) what does each cost (s/step, peak VRAM, tok/s, J/step, adapter size). The headline per family is the pair of **s/step ratios other/e4b** (Unsloth/e4b and HF/e4b; > 1 = e4b faster) from the primary triple, each quoted only when both arms are VALID and the quality reading is COMPARABLE; otherwise the row states why. Every ratio is within one box; **no number from this lane is ever divided into a number from another box**.

## Families ("every supported model")

`SUPPORTED_ARCHITECTURES` on e4b `main` (2026-09-10) names nine loader families: olmoe, qwen3_moe, qwen3_5_moe, gpt_oss, gemma4, gemma4_text, granitemoe, kimi_k3, deepseek_v4. `gemma4_text` is a config shape of gemma4, not a release (the gemma4 row covers it). Mixtral is a convention-route family every earlier training lane ran; it stays. Revisions are the Hugging Face `main` commits on 2026-09-10 (tp1/tp2's where they exist; verified unchanged); the fetch is UNPINNED and the run PROVES the staged snapshot equals the pin (e4b#404; P38 amendment 2), else the family is aborted as `load_fault`, never coerced.

| fam | checkpoint @ revision | n_layers (registered) | notes |
|---|---|---|---|
| granite | `ibm-granite/granite-3.1-3b-a800m-instruct` @ `a02780686e08a03fe0d2679a293b5c74a90efa89` | 32 | tp2: Unsloth attached LoRA to attention only → VOID by trainable count; recorded again, not corrected |
| olmoe | `allenai/OLMoE-1B-7B-0924-Instruct` @ `7f1c97f440f06ce36705e4f2b843edb5925f4498` | 16 | tp2: Unsloth crashed at engage (`IndexError … got 2`); a row |
| gptoss | `openai/gpt-oss-20b` @ `6cee5e81ee83917806bbde320786a8fb61efebee` | 24 | e4b experts built bare (no ExpertsLoRA) → fused REFUSED by citation + probe; `attn_only` secondary row (tp1/tp2's form) |
| qwen3 | `Qwen/Qwen3-30B-A3B` @ `ad44e777bcd18fa416d9da3bd8f70d33ebb85d39` | 48 | the most heavily benchmarked MoE; also the ANCHOR family |
| qwen3_5 | `Qwen/Qwen3.6-35B-A3B` @ `995ad96eacd98c81ed38be0c5b274b04031597b0` | 40 | NEW family (qwen3_5_moe; 256 experts + a SHARED dense expert; 30 linear-attention + 10 full-attention layers; multimodal config) — first training row anywhere |
| gemma4 | `google/gemma-4-26B-A4B-it` @ `4d7ae4984b7db7de8f8457170b3f1a419ee76d52` | 30 | tp2's e4b arms VOIDed on the 4·L attention count; #435's structural census (115) is what this lane uses |
| mixtral | `mistralai/Mixtral-8x7B-Instruct-v0.1` @ `eba92302a2861cdc0098cc54bc9f17cb2c47eb61` | 32 | e4b arms with EXPERT OFFLOAD (the 32 GB card; tp1/tp2's setting); Unsloth resident (its only mode) |
| deepseek_v4 | `deepseek-ai/DeepSeek-V4-Flash` @ `60d8d70770c6…` | 43 | **NOT RUN by registration**: 159.6 GB of safetensors (per-expert MXFP4) exceeds the registered class (32 GB VRAM + 98 GB ordered host RAM) even with e4b's expert offload; e4b's NVMe tier is not a registered training path; no comparator loads it. Four `not_run` rows, written on box A |
| kimi_k3 | `moonshotai/Kimi-K3` @ `f831ab668142…` | — | **NOT RUN by registration**: 1,560.9 GB; no single box of any rentable class. Four `not_run` rows, written on box A |

## Fixture — the field's recipe, identical for every framework per family

The Unsloth notebooks' Alpaca recipe, verbatim from `unslothai/notebooks` (`nb/Qwen3_(14B)-Alpaca.ipynb`, read 2026-09-10): `max_seq_length = 2048`, `load_in_4bit = True`; `get_peft_model(r = 16, target_modules = [q,k,v,o,gate,up,down], lora_alpha = 16, lora_dropout = 0, bias = "none", use_gradient_checkpointing = "unsloth", random_state = 3407)`; `SFTConfig(per_device_train_batch_size = 2, gradient_accumulation_steps = 4, warmup_steps = 5, max_steps = 60, learning_rate = 2e-4, optim = "adamw_8bit", weight_decay = 0.001, lr_scheduler_type = "linear", seed = 3407)`; `alpaca_prompt` = the three-slot Instruction/Input/Response template with EOS appended (`formatting_prompts_func`); dataset `unsloth/alpaca-cleaned`. Registered here as:

- **Text**: `unsloth/alpaca-cleaned` @ dataset revision `0fe581eb78617869b6e17dd195a3fe4045b3723d`, file `alpaca_data_cleaned.json` sha256 `bd844b8247a0f543804b6ce0882b0aaec4bbf5e8d66167df6213a0f1e4fe878b` (51,760 rows) → `tp4_alpaca.py` shuffles once with seed 3407 and takes **train 1,200 / held-out 48** rows → `ds_alpaca.json` sha256 **`5324987afa4042556953026289e8dbdbe8a936b32832ed9e603b9192b706a2fb`** (asserted on the box before anything runs). Tokenised **once per family** with that family's tokenizer at the pinned revision, the notebook's template, EOS appended, truncation at 2048 → `tokens_<fam>.json` (sha in every receipt).
- **Steps**: N = 60 optimizer steps; each step = 4 micro-batches × 2 rows (right-padded to the longest row, attention mask, labels −100 on pads — the same tensors to every framework), rows in fixed order; eval every 20 steps on the 48 held-out rows (batch 1).
- **Adapter**: r 16, α 16, dropout 0, bias none; **placement = every attention projection found by STRUCTURE (q/k/o present, v optional) + every routed expert** — e4b `add_attention_lora` + `ExpertsLoRA`; Unsloth `get_peft_model(target_modules=…)` + its "Enabling LoRA on MoE parameters" path; HF PEFT `target_modules` (by structure) + `target_parameters` (every 3-D expert stack; one A/B per expert, PEFT ≥ 0.17). Adapters fp32 in every arm (cast recorded). **Trainable-parameter counts asserted equal across frameworks per family**; a mismatch is recorded and VOIDs the row (tp2's rule). For **qwen3_5** (a shared dense expert e4b does not adapt) Unsloth's `target_modules` is attention-only (`q,k,v,o`) so both frameworks adapt the same set — pre-registered exception, recorded in the receipt (`unsloth_targets`).
- **Optimizer**: `bitsandbytes.optim.AdamW8bit(lr 2e-4, weight_decay 0.001)` — the SAME call in every arm; linear schedule with 5 warmup steps in transformers' formula (lr factor `step/5` then `(60-step)/55`; step 0 trains at lr 0 as the notebooks do); seed 3407; bf16 compute, no autocast context (tp1/P38's setting); gradient checkpointing on (e4b/HF: HF non-reentrant; Unsloth: `"unsloth"`).
- **Anchor fixture** (Qwen3 only, box A): **tp2/P38 as run** — clinical text (`ds_manifest.json` sha `76fb9036…`), seq 512, batch 1 × accum 1, r 8 / α 16, lr 1e-4, `torch.optim.AdamW` (wd 0.01, constant), seed 0, N 60. Byte-for-byte tp2's arms (`fused_attn4`, `ckpt_unsloth`), tagged `_p38`.

## Environments (per box; every version in `versions.txt` and every receipt)

- **e4b**: GitHub **`main`** — experts4bit-qlora at the commit the launch manifest pins as `heads.e4b` (the commit carrying this document, so the box installs exactly the tree the launcher verified) + grouped-nf4-gemm `main` @ **`d9fd170d83305fbe3f9b5ae2e661d45307c0a2c5`**; the box asserts both `direct_url.json` commit ids. transformers **5.17.0**, bitsandbytes **0.50.2**, peft **0.20.0** (the latest releases on 2026-09-10; e4b's own suite passed on this transformers on CPU: 1335 passed / 218 skipped), the image's torch 2.8.0+cu128 / triton 3.4.0 (`venv-e4b`, system site packages).
- **HF (plain stack)**: `venv-e4b`'s transformers / bitsandbytes / peft / torch — the same interpreter as the e4b arm, so the only difference between those two arms is the code path.
- **Unsloth**: **latest PyPI at launch** (`unsloth[cu128-torch280]` + `unsloth_zoo`; 2026.9.4 / 2026.9.3 on 2026-09-10), its own venv, **no transformers/bnb/peft pins from us** (P38 amendment 1 — it pins transformers ≤ 5.5.0 itself, recorded); torchao removed once if the tripwire fails on `ScalingType` (P38 amendment 2).
- Image `pytorch/pytorch:2.8.0-cuda12.8-cudnn9-devel`; `HF_HUB_DISABLE_XET=1`; HF pulls authenticated (token staged by the driver, never in a command line; every registered checkpoint is ungated).

## Arms per family (each one process, one JSON `<fam>_<fw>_<tag>.json`, one alarm), in this order

1. `e4b/fused_attn4` — PRIMARY e4b: `enable_fast_train(dgrad=True)` + attention 4-bit (structural census); gpt-oss: skipped as REFUSED with tp1/tp2 cited, `e4b/attn_only` runs instead (secondary row; probes and refreshes the stubs on this box).
2. `unsloth/ckpt_unsloth` — PRIMARY Unsloth: `FastLanguageModel` (P38's loader); if it refuses with a message naming FastModel / vision / multimodal, ONE recorded retry with `FastModel` (receipt `loader_used`, `loader_fallback_reason`) — pre-registered, not an amendment.
3. `hf/hf_peft` — PRIMARY HF: `AutoModelForCausalLM` + `BitsAndBytesConfig(load_in_4bit, nf4, bf16 compute, no double quant)` + PEFT as above. **What ends up 4-bit is a RECORDED fact, never a VOID**: transformers' bnb quantizer converts `nn.Linear` only, so the expert stacks are expected to stay bf16 and the row carries its regime label (`Params4bit_expert_stacks`, `Linear4bit` counts).
4. `e4b/reference_attn4` — the e4b reference path (dense expert loop), the internal parity control (tp1's rule: |Δ final| ≤ 0.05 and median step |Δ| ≤ 0.05 → PASS; informational). Runs LAST so the deadline guard drops it before a primary arm.
5. **Secondary pair `_mb1`** (micro-batch 1 × accum 8 — the same tokens per step, half the activation memory), run for e4b and Unsloth (and HF if HF OOMed) **only when a primary arm on that family OOMed**; quoted separately as SECONDARY, never mixed with the primary ratios.
6. **Anchor pair** (`qwen3`, box A only): `e4b/fused_attn4_p38` + `unsloth/ckpt_unsloth_p38` at the anchor fixture.

Alarms per arm are deadline-derived (the launcher's `E4B_RENT_DEADLINE_EPOCH` minus a 15-min fetch margin, capped by the arm's registered ceiling: granite 1800/1800/1800/2400 s, olmoe 2400/2400/2400/3000, gpt-oss 3600/2400/2400, qwen3 3600/3600/1800/5400, qwen3_5 3600/3600/1800/5400, gemma4 3600/3600/1800/5400, mixtral 5400/2400/1800/6000, anchor 1800 each — fused/unsloth/hf/reference). An arm that cannot fit before the deadline is a `not_run` stub `host-limited deadline`.

## Validity rules (VOID never enters a ratio)

Let L = registered n_layers, A = accum (4; 8 for `_mb1`; 1 for `_p38`). **e4b fused**: `n_patched == L`, `kernel_calls_per_step_min ≥ 2·L·A`, `n_attn4 == structural census expected_count` (tp3 T10: Gemma-4 115, four-projection families 4·L); C1 frozen bytes bit-exact. **Unsloth**: `Params4bit_expert_stacks ≥ 2·L`, `experts_forward_calls_per_step_min ≥ L·A`, `n_bnb4bit_unwrapped ≥ L`, the "Enabling LoRA on MoE parameters" banner; C1. **HF**: `experts_forward_calls_per_step_min ≥ L·A`, PEFT adapted ≥ 1 expert parameter; C1. **All**: step count == N, same tokens sha, same trainable count as the family's e4b arm of the same recipe.

## Verdict and reading per family (pre-registered)

- **Support**: the vocabulary above; every attempt a row (a stub for every non-run, with the reason).
- **Positions** (both arms VALID): s/step ratio other/e4b (median of steps 11..N), peak-VRAM Δ, J/step ratio, tok/s; the other arm's regime label beside it. **Quality reading**: held-out |Δ| at N=60 ≤ 0.05 nats reads COMPARABLE (a reading threshold, not a gate; P38/P40's); otherwise the ratio carries the FLAGGED mark and is not a clean position.
- **e4b internal parity** per family (fused vs reference): tp1's rule, informational.
- **Cross-lane anchor**: the Qwen3 `_p38` pair's ratio vs tp2's **1.457** (2026-09-06, RTX 5090) and P38's **1.413** (2026-09-05, RTX 5090): within ±10 % of tp2 AGREES; beyond is a finding about box/stack variance (or about e4b `main` vs the shipped 0.35.1 tp2 used), stated, never averaged away.
- **Field position rule** ([[feedback_three_axis_throughput_quoting]]): a ratio against e4b's own reference is labelled internal; every quoted position names the comparator, its weight precision (the regime label) and its footprint in the same sentence.

## Pre-registered predictions (falsifiable; competitor estimates biased UP per the standing correction)

- **P1** granite & olmoe: all three frameworks train at the field recipe; Unsloth/e4b in [0.9, 1.4] (host-bound at ≤ 1B active); HF trains with bf16 experts (regime recorded).
- **P2** qwen3 (field recipe): e4b fused and Unsloth train, plain HF OOMs on the 32 GB card (60 GB of bf16 experts); Unsloth/e4b in [1.1, 1.8].
- **P3** qwen3 anchor pair: within ±10 % of tp2's 1.457.
- **P4** qwen3_5: e4b and Unsloth both load and engage (Unsloth possibly via the FastModel fallback), trainable counts match with attention-only Unsloth targets; Unsloth/e4b in [1.0, 1.8]; HF OOMs.
- **P5** gemma4: e4b fused trains (census 115), Unsloth trains; Unsloth/e4b in [1.0, 1.6]; HF OOMs.
- **P6** mixtral: Unsloth resident OOMs at the field recipe AND at `_mb1` on 32 GB; e4b (offload) trains both; HF OOMs at load → no Mixtral ratio on this class (the footprint rows are the result).
- **P7** gptoss: e4b fused REFUSED (bare experts), `attn_only` OK; Unsloth refuses or has no 4-bit MoE path (tp2: REFUSED); HF refuses (the checkpoint's MXFP4 quantization config vs bnb) or OOMs.
- **P8** quality COMPARABLE on every quoted pair.
- **P9** e4b internal parity PASS on every family where both e4b arms run.

## Out of scope

Time-to-target; longer curves; a bf16-LoRA (non-4-bit) Unsloth arm; Axolotl (no 4-bit MoE expert path in a release, CLI-only recipe — a follow-up lane if wanted); any change to tp1/tp2 verdicts, gates, thresholds or licences; serving. **Nothing in this lane licenses a claim**; the receipts feed `training_support` per family × path exactly as tp2's did.

## Boxes, cost, stop rules

Three **RTX 5090** boxes (Vast verified, on-demand; the launcher's pre-flight: ≥ 320 GB disk, ≥ 98 GB host RAM, ≥ 40 MB/s), launched **in parallel** from the mini through `adertha-agents/tools/pod-launch.sh` → `adertha.compute.rent` (adertha-agents#101: three live controllers, each reserving its estimate; #102: a live peer's in-progress receipt directory and `reservations.jsonl` are not dirt). Rate ceiling **$0.85/h** each (23 qualifying offers at $0.61–0.94 on 2026-09-10). Wall-clock guards and estimates: **box A 3.5 h ($2.98)**, **box B 5 h ($4.25)**, **box C 5 h ($4.25)** — each in the (2, 20] band (`one-of:CTO,CSO`; the owner's authorization is the top of every band), total ≈ $11.5 against the $35/run cap, $50 CTO daily ceiling and $100 global budget. The box is refused unless `nvidia-smi` names the registered class. Failure classes write stubs in the vocabulary; **STOP rules**: an arm that cannot fit before the deadline is skipped `host-limited`; a second consecutive pre-flight refusal on the same box role ends that box's attempts for the day (a row and a defect, not a retry loop); no relaunch changes the fixture. Every relaunch of this registered lane after a fixed defect is routine (owner's seventh directive) and is logged as an amendment below.

## Amendments (dated entries, before the data they touch)

### Amendment 1 (2026-09-10 20:55Z, before any box-C data exists): box C re-draws under a $0.69/h ceiling with this account's proven-unattachable machines excluded

Box C's first attempt (`tp4-c`, committed receipt `receipts/experts4bit-qlora/2026-09-10/tp4-c/receipt.json`, $0.0294,
teardown proven) was **refused by the pre-flight at 39.0 MB/s against the registered 40 MB/s floor** on machine
145701 — the cheapest qualifying offer, and the lowest-advertised link (657 Mbit/s) of the 21–23 offers that clear
the disk and RAM minimums. Box C carries the lane's two heaviest downloads (Gemma-4 51.6 GB, Mixtral 93.4 GB), so
the floor refused the host whose link would have consumed a large share of the 5 h guard fetching. **The floor is
not touched.**

Two facts force the re-draw to be re-aimed rather than simply repeated: the offer search is `dph_total` ascending,
so an unchanged relaunch buys machine 145701 again; and the launcher's machine-exclusion class cannot express this
refusal (`rent.py`: *"bandwidth or `loading` failures are not this class"* — it is bound to the ssh-readiness shape
P39 needed). So the re-draw:

- **`usd_per_hour` 0.85 → 0.69** for box C only, which excludes the 39.0 MB/s host by price. The estimate falls
  from $4.25 to **$3.45** (0.69 × 5 h), still in the (2, 20] band. If no qualifying offer sits under the ceiling the
  launcher refuses **before creating anything** (`VastRefused`, $0), which is the outcome to prefer over another
  rental at a known-slow host.
- **The four committed ssh-readiness refusals this account already owns are passed as
  `exclude_vast_machine_receipts`** — `p39-box1` and `p39-box1-2` (machine 59164, the current cheapest offer at
  $0.538, which failed to attach twice after 33 attempts each), `p39-box1b-2` (136897) and `p39-box1b` (147546).
  This is the mechanism used exactly as designed, on receipts that already satisfy its class.
- Run id **`tp4-c-2`** (the ledger holds a row for `tp4-c` and refuses a repeat).
- Everything else is unchanged: same families, same fixture, same guard, same pins.

The registered stop rule still binds: a **second** consecutive pre-flight refusal on box C ends that box's attempts
for the day, and Gemma-4 and Mixtral are then reported as not measured rather than retried into a loop.

### Amendment 2 (2026-09-10 21:30Z, before any large-family data exists): N = 20 and a cheaper eval instrument for the large families -- the alarms this lane inherited cannot hold 60 steps at this fixture

**Measured, not predicted.** Box B's first Qwen3-30B-A3B arm ran 24 minutes past its load line without reaching
step 10. The box says why: `nvidia-smi pmon` shows **sm 99-100 % with memory-bandwidth 0 %** at 107 W and 2985 MHz
with one CPU core saturated -- real GPU work in a long stream of tiny kernels, which is what a 128-expert
48-layer MoE at batch 2 looks like. Against tp2's measured **4.108 s/step** for the same model and arm at seq 512 /
batch 1 / accum 1, this lane's fixture puts **32x the tokens through a step** (2 rows x 2048 x 4 accum = 16,384 vs
512), so a step costs ~130 s and 60 of them cost **~2.2 h before any eval** -- against a registered per-arm alarm
of 3600 s. The arm cannot finish, and an alarmed arm yields a stub, not a measurement.

**What was wrong.** The fixture came from the Unsloth notebook (correctly -- it is the field's recipe) while the
per-arm alarms and the eval instrument came from tp2, whose fixture was 32x smaller per step. Two registered
quantities that only make sense together were taken from different places. The alarms were checked against the
wall clock, never against the fixture they had to cover.

**The amendment, for the large families only** (`qwen3`, `qwen3_5`, `gemma4`, `mixtral` -- those whose per-step cost
at this fixture exceeds ~60 s; the small families granite/olmoe/gpt-oss are unaffected and box A's rows stand):

- **N = 20 optimizer steps** (was 60). The measured quantity is `s_per_step_median_11plus`, so N = 20 leaves **10
  measured steps** after the 10 warm-up steps the median already discards. Thinner than 50 and stated as such;
  every other fixture term -- seq 2048, micro-batch 2 x accum 4, r 16 / alpha 16, lr 2e-4, AdamW-8bit, linear
  warmup 5, seed 3407, the Alpaca text -- is **unchanged**, because those are the field's recipe and N is not.
- **The eval instrument shrinks: 8 held-out rows, at step 0 and step N only** (was 48 rows every 20 steps). The
  eval is *our* quality reading, not part of the notebook's recipe -- the notebook holds nothing out at all. At
  ~7 s per batch-1 forward at seq 2048 the old instrument cost ~22 min per arm, more than the training it was
  measuring. The reading stays **paired**: every arm of a family evaluates the identical 8 rows, and the
  registered threshold (held-out |delta| <= 0.05 nats reads COMPARABLE) applies to that pair. A reading on 8 rows
  is noisier in absolute terms and is quoted only as the paired comparison it is.
- **Warm-up is unchanged at 10 steps**, so `s_per_step_median_11plus` keeps its meaning across every family and
  across tp2.

**Box B is restarted in place**, keeping its already-fetched 61 GB Qwen3 snapshot: the box-side script already reads
`TP4_STEPS`, `TP4_EVAL_N` and `TP4_EVAL_EVERY` from the environment, so no code changes and no new rental. The
restart re-uses the run's nonce, and the first attempt produced **no receipts** -- so no data is discarded and this
amendment still precedes every large-family measurement. Box C's launch (`tp4-c-2`, still queued for a controller
slot) carries `TP4_STEPS=20`; the drive script forwards that variable already.

**The Qwen3 anchor pair moves from box A to box B, and its own fixture is untouched.** Box A's remaining budget
cannot reach it: at 21:35Z it had OLMoE and gpt-oss still to run plus ~2.6 h of guard, and the anchor needs a fresh
61 GB Qwen3 download it would have to make from scratch. Box B **already holds that snapshot** (57 GB on disk,
verified), so the pair costs it only two short arms -- the anchor runs at tp2/P38's fixture (clinical text, seq 512,
batch 1 x accum 1, r 8, lr 1e-4, torch AdamW, seed 0, **N = 60**, 48 held-out rows), none of which amendment 2
changes, and it is scheduled FIRST on box B so the deadline cannot eat the one arm-pair that ties this lane to the
earlier two. Box B therefore runs `qwen3anchor qwen3 qwen3_5`, and its clinical dataset is built and sha-verified
against `ds_manifest.json` exactly as box A's was. This is a scheduling change, not a fixture change; box A's own
`qwen3anchor` slot becomes a `not_run` row naming this relocation.

**What this costs the comparison, said plainly:** the large families are measured over 10 steps rather than 50, so
their s/step medians carry more variance than box A's, and their quality readings rest on 8 rows rather than 48.
Both are recorded per row. The alternative -- 60 steps at this fixture -- is ~2.2 h per arm, i.e. ~6.5 h for one
family's three primary arms, which no 5 h box can hold and which would have bought stubs instead of numbers.

Related defect filed while this lane was in flight, not fixed under it: **experts4bit-qlora#542** (the HF arm's
expert-parameter list is selected by the name substring `experts`, so GraniteMoe's `input_linear`/`output_linear`
stacks are never adapted and box A's Granite HF row will VOID with that reason).
