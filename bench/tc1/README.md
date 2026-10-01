# TC1 — Qwen3-30B-A3B on one RTX 5090: e4b vs Unsloth vs HF+PEFT vs axolotl at matched work, matched init and matched adapter precision

Pre-registration: `TC1-PREREG.md` (written by the PI; the harness refuses a real run that does not name it with `--prereg`).
Lineage: `bench/tp4/` (its arm driver, run script, reducer and CI test are the base; the tp4 tree is a committed receipt and is never edited).

| file | role |
|---|---|
| `tc1_arm.py` | per-arm driver, FOUR frameworks (e4b, Unsloth, HF+PEFT+bnb, axolotl 0.20.0) through ONE `run_arm` (copy of `bench/tp4/tp4_arm.py` @ 10ce711d + T19 `--adapter-dtype fp32\|native` on EVERY framework, T20 `--lora-init matched:<seed>` with the structural slot map and the `matched_init` receipt block, T21 `frozen_base_probe` with its byte-flip control, T22 `TC1_*` names; phase 2: `--unsloth-moe-backend` / `--unsloth-speed-tilt` / `--unsloth-double-quant` with backend engagement counters, `--framework axolotl` through axolotl's own `ModelLoader`, `--axolotl-best`; phase 3: per-step kernel LoRA-path counters, the name-free `matched_init_sha`, C1 over every frozen tensor with a real-storage control, torch-level grouped_mm counters, `arm_facts`, dynamo snapshots, per-row held-out losses, `--hf-double-quant`, `--hf-experts-implementation`; `--selftest` on CPU prints each control's FAILING case) |
| `tc1_run.sh` | BOX side: installs venv-e4b (transformers 5.18.0 / bitsandbytes 0.50.2 / peft 0.21.2), venv-unsloth-t28 (unsloth[cu128-torch280] on the image's torch 2.8), venv-unsloth (unsloth[cu130-torch2121], + e4b/gnf4 for the t212 row) and venv-axolotl (uv, CPython 3.12, torch cu130) -- every cu130 venv behind the driver >= 580 gate, refused rows below it; fetches the family with the pin proof, tokenises once, then `TC1_FAMILIES=qwen3` runs the JUDGED family in order (e4b fused_m, Unsloth m, e4b reference_m third, the two `_d2` draws, HF m, axolotl m, the two profiled arms with dmon, the `_mb1` pair on an OOM) and `TC1_FAMILIES=qwen3native` runs the labelled rows on their own box (its own e4b fused_m, Unsloth best / t28 / triton, e4b shipped / nodgrad / t212, axolotl best, hf mb1 t214); every arm under OMP_NUM_THREADS = the physical core count; reduces |
| `tc1_drive.sh` | CONTROLLER side (the launcher's `--command`): stages `tc1_run.sh`, `tc1_arm.py`, `tc1_reduce.py` and `bench/tp4/tp4_alpaca.py` (referenced, not copied), starts under a nonce, heartbeats, fetches receipts; forwards EVERY `TC1_*` knob the box script and the arm read (asserted by `tests/test_tc1_arm.py`) |
| `tc1_reduce.py` | the support / validity / VERDICT / draws / position / equivalence / frozen-base / P1–P10 (+ P1b) table (copy of `tp4_reduce.py` + R1–R9: two-draw stability at 5 %, the position's cross-draw interval, matched-set predicates incl. the name-free sha, the step-0 SAME-BYTES-CLASS / NEAR / VOID bands, equivalence vs `e4b/fused_attn4_m` with the band set by the fused-vs-reference control and the draw-noise floor, per-row paired held-out stats, Unsloth backend / torch grouped_mm engagement, the `qwen3native` family; `--selftest` on hand-built receipts prints each predicate's FAILING case, stdlib only) |
| `../tp4/tp4_alpaca.py` | the pinned Alpaca subset builder, used as is |
| `../../tests/test_tc1_arm.py` | CI: the selftest end to end, the `--prereg` and `--lora-init` refusals, the slot map on the REAL peft (skipped when peft is absent), the probe on genuine bnb storage, the run script's arm order and flags, the driver's forwarded-knob list |
| `../../tests/test_tc1_reduce.py` | CI: the reducer's selftest (≥ 16 cases incl. one VOID per predicate, QUALITY_FAIL, UNSTABLE, DIVERGENT) |

Receipts: `<fam>_<fw>_<tag>.json`, one per attempt; a stub carries `written_by: tc1_run.sh`. Every OK receipt carries, beside
tp4's fields, `adapter_dtype`, `lora_init`, `matched_init {seed, n_slots_set, n_slots_expected, unmapped, complete, ...}` and
`frozen_base_probe {slots: {gate_up, down, q_proj: {sha, regime, n_bytes, method, control_detects_flip}}, control_detects_flip}`.

Things measured while building this (on CPU, 2026-10-01), stated so the registration can rely on them or correct them:

- e4b's per-expert NF4/64 bytes (`Experts4bit.from_float`) equal bitsandbytes' whole-stack `quantize_4bit` sliced at expert 0
  (packed bytes, absmax and the bf16 dequant all equal) — so the SAME-BYTES reading between e4b and a bnb-held stack is attainable.
- e4b's `quantize_attention_projections_4bit` builds `Params4bit(..., quant_type="nf4")` with bnb's default `compress_statistics=True`:
  the e4b attention slot's regime reads `nf4/64+dq` (double-quant ON), which is not what the draft's P10 assumes for e4b's side.
- At the pinned kernel tag (`v0.34.0` → commit `846b512b…`, also CI's pin; the draft's `bdcd6ad3…` is the annotated tag object),
  `lora_delta_grouped` casts the activations to `A.dtype` on all three paths and `fused_grouped_lora` adds `delta.to(out.dtype)`:
  fp32 adapters on the fused path are supported by source reading (not run here: no GPU).

The axolotl arms and the Unsloth native-best / t28 / triton rows landed in phase 2 (facts cited from `axolotl-arm-spec.md` and
`UPSTREAM-NOTES`); phase 3 moved the labelled rows to the `qwen3native` token and added the controls the design review asked for.
The grouped-nf4-gemm pin is the v0.34.0 COMMIT `846b512b905468c08f5748943d08769b572affa2`; the HF arm's double-quant defaults ON to
match e4b's attention Params4bit (Unsloth and axolotl keep it OFF; the probe labels every slot's regime).

## Lane TC1b — the `qwen3curve` family token (TC1B-PREREG.md, the PI's; drafted in `TC1b-PREREG-draft.md`)

The same files, extended (no parallel copy of the arm driver); the token runs on its own RTX 5090 box with `TC1_FAMILIES=qwen3curve`
(the box then defaults `PREREG` to `tc1/TC1B-PREREG.md` unless `TC1_PREREG` says otherwise). `tc1_run.sh`'s `tc1_curve_family` runs, in
THIS order, one process per arm: `e4b/fused_attn4_m_200` · `unsloth/ckpt_unsloth_m_200` (the matched pair: `--adapter-dtype fp32
--lora-init matched:$MATCHED_SEED`, venv-unsloth + `--unsloth-moe-backend grouped_mm`; recipe `curve` = N 200, eval every 40 on the first 16
held-out rows, the linear schedule decaying to step 200 with 5 warm-up steps — `tc1_arm.py`'s `_lam(step, N=a.steps)`; alarms 4,800 / 9,000 s)
· `e4b/fused_attn4_shipped_200` (`--adapter-dtype native --lora-init native`, recipe `curve`, 4,800 s) · the anchor pair at tp2/P38's fixture
(recipe `anchor` = tp4_run.sh's `A_*` literals, clinical text built and sha-verified by `n9_datasets.py` + `ds_manifest.json` exactly as
`tp4_run.sh` did, tokenised with the `clinical` template at seq 512 and 8 held-out rows — what tp4's anchor RAN, TP4-PREREG amendment 3;
`e4b/fused_attn4_p38` with native precision and init, `unsloth/ckpt_unsloth_p38` with tp4's fp32 cast, native init, the loader's double-quant,
tp4's seven targets and `FastLanguageModel`, EXCEPT that it runs in venv-unsloth (cu130 torch 2.12.1, grouped_mm) — recorded by `--note` in its
receipt — and `unsloth/ckpt_unsloth_p38_t28` in venv-unsloth-t28 with the loader-default backend, which IS tp4's arm byte-for-byte; 1,800 s
each) · the tokens-per-step pair `e4b/fused_attn4_m_t1` + `unsloth/ckpt_unsloth_m_t1` (recipe `t1`: micro-batch 1 × accum 1, N 20, 8 rows at 0
and N; 3,600 s) · the rank pair `e4b/fused_attn4_m_r64` + `unsloth/ckpt_unsloth_m_r64` (recipe `r64`: r 64 / alpha 64; 3,600 s). Every
sub-fixture names its OWN e4b arm for `--expect-trainable` (`expect_of`), as tp4's anchor did.

Knobs (all forwarded by `tc1_drive.sh`, asserted by `tests/test_tc1_arm.py`): `TC1_CURVE_STEPS` (200), `TC1_CURVE_EVAL_EVERY` (40),
`TC1_CURVE_EVAL_N` (16), `TC1_T1_MB` (1), `TC1_T1_ACCUM` (1), `TC1_R64_R` (64), `TC1_R64_ALPHA` (64). The anchor fixture is literals, not knobs.
`tc1_drive.sh` stages `bench/flagship-matrix/drivers/n9_datasets.py` and `bench/flagship-matrix/ds_manifest.json` beside the TC1 pieces
(referenced, never copied). `tc1_arm.py` gains `--note` (verbatim into every receipt and stub).

**Tokens.** The tokens file's sha256 covers `{train, eval[:eval_n]}` (`prepare`), so the curve family's file, tokenised with 16 held-out rows,
cannot carry TC1's `qwen3` sha (8 rows) although its TRAIN rows are byte-identical; `summary.txt`'s `TOKENS` line prints a train-only sha
(sha256 of `json.dumps(train, separators=(",", ":"))`) beside the file sha for the assertion the registration should make. The t1 / r64 arms
take the first 8 of the 16 rows (a prefix), paired with TC1's 8.

**Reducer** (`tc1_reduce.py`, R10): validity per sub-fixture against its own e4b arm (tokens sha, trainable count, step-0 band, name-free
matched sha, N per arm; the registered counts 642,514,944 on arms 1-3 / t1 and 321,257,472 on the anchor arms; r64 asserted equal between its
two arms only), then (a) the curve table (paired mean ± SE over the 16 rows per eval for arms 1-3, paired |Δ| 2 vs 1 and 3 vs 1) and the curve
reading — EQUIVALENT-AT-EVERY-EVAL iff every paired |Δ| ≤ 0.02 (the DRAFT's band; the registration may tie it to the TC1 floor — said in the
output), else DIVERGENT with the first divergent step and the sign at 200; (b) the plateau test (REPRODUCES-P38 iff ≥ +0.01 at 200 and ≤ 0 at
40); (c) time to the matched pair's step-200 held-out + 0.02 per arm; (d) 11..200 s/step medians beside TC1's 11..20 medians when `--tc1-dir`
names a TC1 receipt dir (TRAVELS within 10 %; the in-receipt 11..20 window is printed as a same-draw check); (e) the anchor ratio vs tp2 1.457 /
P38 1.413 ± 10 %, the t28 variant separately (tp4's own anchor row is NOT in this tree: `TP4_ANCHOR_RATIO = None`, P4 says so and reads
against tp2); (f) the t1 and r64 pairs as SCALING POINTS under the matched set's predicates, listed, never a position. P1–P4 of the draft
scored HELD / FALSIFIED / UNTESTED. The selftest adds 10 cases (44 in all): a DIVERGENT curve, a REPRODUCES-P38 plateau and both refutations,
a failing anchor, a VOID r64 pair (trainable count, the LoRA loop, the manual grouped-mm fallback), the registered-count VOIDs, TRAVELS /
DOES-NOT-TRAVEL, a target never reached, and the curve token rendered alone (no TC1 P1–P10 table) and beside TC1's tokens.
