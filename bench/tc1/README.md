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
