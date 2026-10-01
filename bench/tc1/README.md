# TC1 — Qwen3-30B-A3B on one RTX 5090: e4b vs Unsloth vs HF+PEFT vs axolotl at matched work, matched init and matched adapter precision

Pre-registration: `TC1-PREREG.md` (written by the PI; the harness refuses a real run that does not name it with `--prereg`).
Lineage: `bench/tp4/` (its arm driver, run script, reducer and CI test are the base; the tp4 tree is a committed receipt and is never edited).

| file | role |
|---|---|
| `tc1_arm.py` | per-arm driver, three frameworks through ONE `run_arm` (copy of `bench/tp4/tp4_arm.py` @ 10ce711d + T19 `--adapter-dtype fp32\|native` on EVERY framework, T20 `--lora-init matched:<seed>` with the structural slot map and the `matched_init` receipt block, T21 `frozen_base_probe` with its byte-flip control, T22 `TC1_*` names; `--selftest` on CPU) |
| `tc1_run.sh` | BOX side: installs venv-e4b (transformers 5.18.0 / bitsandbytes 0.50.2 / peft 0.21.2) and venv-unsloth (the registered unsloth 2026.9.14 + unsloth_zoo 2026.9.9), fetches the family with the pin proof, tokenises once, runs TC1-PREREG's arms IN ITS REGISTERED ORDER (`draw2` = the second draw of an arm, a fresh process, tag `_d2`; `todo_arm` = a `not_run` row for the axolotl and Unsloth native-best arms this cut does not implement), the `_mb1` secondary pair on an OOM, reduces |
| `tc1_drive.sh` | CONTROLLER side (the launcher's `--command`): stages `tc1_run.sh`, `tc1_arm.py`, `tc1_reduce.py` and `bench/tp4/tp4_alpaca.py` (referenced, not copied), starts under a nonce, heartbeats, fetches receipts; forwards EVERY `TC1_*` knob the box script and the arm read (asserted by `tests/test_tc1_arm.py`) |
| `tc1_reduce.py` | the support / validity / VERDICT / draws / position / equivalence / frozen-base / P1–P10 table (copy of `tp4_reduce.py` + R1–R7; `--selftest` on hand-built receipts, stdlib only) |
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

Not in this cut (TODO hooks in `tc1_run.sh`, `not_run` rows with the reason): `--framework axolotl` (arms 6 / 6b) and the Unsloth
native-best knobs (arm 7); both wait for `UPSTREAM-NOTES.md`.
