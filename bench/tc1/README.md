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

## Lane TC3 — the `qwen3frontier` / `qwen3frontier12` family tokens (TC3-PREREG.md, the PI's; drafted in `TC3-PREREG-draft.md`)

The same files, extended; nothing copied. `tc1_run.sh`'s `tc1_frontier_family` (24 GB box: `TC1_FAMILIES=qwen3frontier TC1_GPU_CLASS=4090`, the class
check accepts "NVIDIA GeForce RTX 4090", box class `RTX 4090`) runs, in THIS order, one process per arm, every arm `--adapter-dtype fp32 --lora-init matched:$MATCHED_SEED`:
`e4b/fused_attn4_m` (resident; expected OOM, 1200 s) · `e4b/fused_attn4_m_offload` (the `arm` wrapper's OFFLOAD positional = 1 → `--offload 1`, 3600 s) · `e4b/fused_attn4_m_mb1`
(resident, recipe `mb1`, 3600 s) · `unsloth/ckpt_unsloth_m` (venv-unsloth grouped_mm; expected OOM, 3600 s) · `unsloth/ckpt_unsloth_m_mb1` (3600 s) · `hf/hf_peft_m` (expected OOM at
load, 1800 s) · `hf/hf_peft_m_offload` (`--hf-offload 1`: `device_map="auto"` under `max_memory = {0: "<GPU GiB − 2>GiB", "cpu": "<host RAM GiB>GiB"}` from `hf_max_memory`, with
`llm_int8_enable_fp32_cpu_offload=True` — the switch transformers' own error names for a 4-bit device_map with CPU entries, `--hf-offload-fp32-cpu 0` turns it off — and
`hf_device_map` summarised (`device_map_summary`: entries and expert entries per device) on the receipt; a `NotImplementedError` / `RuntimeError` inside the loop is a `refused` row
with the exception text; 3600 s) · `axolotl/ckpt_axolotl_m` (2700 s) · `axolotl/ckpt_axolotl_m_layeroffload` (`--axolotl-layer-offload 1`: `layer_offloading: true` in the config
dict, schema `config.py:653`, and the `LayerOffloadManager` + `_LayerOffloadContext` the trainer mixin would run, driven by `run_arm` around every micro-batch; 3600 s) ·
`axolotl/ckpt_axolotl_m_zero3` (`--axolotl-zero3 1`: a `refused` row — the DeepSpeed engine is created by transformers' Trainer inside `axolotl.train.train`, not by `ModelLoader`;
the config it would have used and the installed deepspeed version are on the row; `axolotl[deepspeed]` is installed as a separate non-fatal step only under this token; 3600 s) ·
`e4b/reference_attn4_m_offload` (the parity / equivalence anchor under offload, 5400 s). Every row records `peak_vram_gb`, `host_ram_high_water_gb` (max over the arm of the process
peak RSS — `/proc/self/status` VmHWM and `ru_maxrss` — and the cgroup `memory.peak` when it rose during the arm; the parts under `host_ram`), `host_ram_total_gb` and `memory_lever`.

`tc1_frontier12_family` (the owned 12 GB RTX A2000, `TC1_FAMILIES=qwen3frontier12 TC1_GPU_CLASS="RTX A2000"`, box class the string itself): `e4b/fused_attn4_m_offload` ·
`e4b/fused_attn4_m_offload_d2` (`draw2`) · `e4b/reference_attn4_m_offload` · `e4b/fused_attn4_m` (resident; expected OOM) · `unsloth/ckpt_unsloth_m_mb1` (`UNS_VENV=t28`,
`--unsloth-moe-backend default`: that host's driver is 575, so every cu130 venv is refused by the gate — recorded in the receipt's `env.torch` and `versions.txt`) · `hf/hf_peft_m_mb1`
· `axolotl/ckpt_axolotl_m` (a `refused` row naming the driver). Alarms: the 24 GB token's classes (a choice; the draft names none for this box).

**TC3 hand run on the owned box** (`TC1_LOCAL_BOX=1`: no launcher, no nonce / instance id / deadline from rent.py — deadline = now + 6 h; the checkpoint from
`TC1_LOCAL_SNAPSHOT`, a directory at the registered revision, linked into a private HF cache under `TC1_LOCAL_OUT/hf-cache` so every loader resolves the pin offline; the pin proof
compares the directory's `config.json` sha256 — and the safetensors count/bytes — with the Hub's at the pin when the network answers, else records `pin_proof: offline`, never
fetches; the venvs, tokens and receipts under `TC1_LOCAL_OUT`; `TC1_LOCAL_PYTHON` = the interpreter the venvs are made from — if it is itself a venv its torch is invisible to
`--system-site-packages`, so venv-e4b installs the same `torch==<version>` from the matching `/whl/<cuNNN>` index; `tc1_drive.sh` never forwards `TC1_LOCAL_*`). From the
`gpu-dev` container, with the worktree's `bench/tc1/*` and `bench/tp4/tp4_alpaca.py`, `bench/flagship-matrix/drivers/n9_datasets.py`, `bench/flagship-matrix/ds_manifest.json`
copied into `$OUT` (the script refuses to start without them):
```
OUT=/workspace/tc1-frontier12; mkdir -p $OUT; cd $OUT
TC1_LOCAL_BOX=1 TC1_LOCAL_OUT=$OUT TC1_LOCAL_SNAPSHOT=/models/Qwen3-30B-A3B TC1_LOCAL_PYTHON=/workspace/venv-k3rel/bin/python \
TC1_BOX=A TC1_FAMILIES=qwen3frontier12 TC1_GPU_CLASS="RTX A2000" \
E4B_SHA=<the 40-char e4b commit the receipts should cite> GNF4_SHA=846b512b905468c08f5748943d08769b572affa2 \
bash tc1_run.sh > outer.log 2>&1
```
(`TC1_LOCAL_PYTHON`'s path is the k3rel venv's interpreter on that box — verify it with `ls` before the run; `/models/Qwen3-30B-A3B` must hold `config.json` and the
`*.safetensors` of revision `ad44e777…`. `TC1_MIN_DISK_GB` defaults to 40 under a hand run. The reducer runs at the end as on a rental; `pin_proof_qwen3frontier12.json`
sits beside the receipts and every receipt's `note` carries the proof.)

**Reducer** (`tc1_reduce.py`, R11): the frontier tokens have their OWN anchor, `e4b/fused_attn4_m_offload` (the resident e4b arm is expected to OOM), and their own control,
`e4b/reference_attn4_m_offload`; validity = TC1's predicates against that anchor. Readings: (a) the FIT TABLE — per framework every arm with its lever, VERDICT, peak VRAM,
host-RAM high-water, s/step, J/step, regime, and a fit line (`FITS on this box via …` / `NO ARM COMPLETED on this box: …`); (b) the in-box equivalence of every matched arm against
the anchor under TC1's R4 bands (EQUIVALENT ≤ 0.02 on both the median per-step |Δ train| and |Δ held-out at N|, COMPARABLE ≤ 0.05) with tp1's parity on the fused/reference offload
pair, and, with `--tc1-dir <TC1 receipt dir>`, the matched trajectory of the offload anchor against TC1's resident `e4b/fused_attn4_m` — the same tokens sha, name-free
`matched_init_sha`, fp32-only adapters and N asserted first, then median per-step |Δ| ≤ 0.02 reads EQUIVALENT-TO-RESIDENT, else DIVERGENT-FROM-RESIDENT; (c) no cross-box ratio
anywhere — P1's ratios are within the 24 GB box, P4 prints the two s/step measurements with the 2× factor applied; (d) P1–P4 of the draft scored HELD / FALSIFIED / UNTESTED
(P1's four clauses each named in the evidence; P2 is about frameworks — an e4b resident arm that also completed is noted, never a refutation; P3 per token). The selftest adds
six cases (50 in all): the OOM-only Unsloth column, the HF offload `refused` row, the axolotl ZeRO-3 `refused` row, a VOID offload equivalence (plus a DIVERGENT in-box pair and a
missing anchor), P1's legs within the box, the resident comparison with `--tc1-dir` (EQUIVALENT / DIVERGENT / N-A on a tokens mismatch), the 12 GB token's P2 legs, and both
tokens end to end through files. Facts read while building this: axolotl's `layer_offloading` is a trainer mixin over two trainer-free classes; its ZeRO-3 engine is the Trainer's;
transformers 5.18's bnb-4bit quantizer refuses CPU entries in a device_map without `llm_int8_enable_fp32_cpu_offload`; axolotl 0.20.0 pins `torch<=2.14.0` (not the 2.14.1 the TC1
prose assumed) and `packaging==26.0`, which uv's first-index strategy cannot satisfy with the cu130 extra index (both TC1 boxes) — `UPSTREAM-NOTES.md` "TC3 addendum". Nothing in
TC3 ran on a GPU; axolotl and DeepSpeed were read, not run.

## Lane TC1b — reducer notes

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

## Lane TC2 — the `tc2small` and `tc2big` family tokens (TC2-PREREG.md, the PI's; drafted in `campaign-2026-10-01/notes/TC2-PREREG-draft.md`)

The same files, extended (no parallel copy of anything). Two tokens, one box each; the box defaults `PREREG` to `tc1/TC2-PREREG.md` for both unless
`TC1_PREREG` says otherwise. `TC1_BOX=B` is admitted by `tc1_run.sh` and `tc1_drive.sh` and defaults `TC1_FAMILIES` to `tc2big`; box A keeps `qwen3` and
takes `TC1_FAMILIES=tc2small`.

**`tc2small`** (`tc2_small_box` → `tc2_small_family`, recipe `small`: N 60, 48 held-out rows every 20 steps — tp4's registration for the small families;
`TC1_SMALL_STEPS` / `TC1_SMALL_EVAL_N` / `TC1_SMALL_EVAL_EVERY`, forwarded by `tc1_drive.sh`; the function shadows `STEPS` / `EVAL_N` / `EVAL_EVERY` with
`local`, so its stubs carry the same N as its receipts and the box's reduce call passes no `--steps` for this token): granite
(`ibm-granite/granite-3.1-3b-a800m-instruct` @ `a0278068…`, 32 layers), olmoe (`allenai/OLMoE-1B-7B-0924-Instruct` @ `7f1c97f4…`, 16), gptoss
(`openai/gpt-oss-20b` @ `6cee5e81…`, 24). MODE normal, in THIS order: `e4b/fused_attn4_m` · `hf/hf_peft_m` · `e4b/reference_attn4_m` · `e4b/fused_attn4_m_d2`
· `hf/hf_peft_m_d2` · `unsloth/ckpt_unsloth_m` (venv-unsloth grouped_mm, the seven) · `unsloth/ckpt_unsloth_m_experts` (granite only:
`q_proj,k_proj,v_proj,o_proj,input_linear,output_linear`, tp4 amendment 4's second arm) · `hf/hf_peft_m_t214` (`HF_VENV=t214`,
`--hf-experts-implementation grouped_mm`) · `axolotl/ckpt_axolotl_m` · `axolotl/ckpt_axolotl_best` · `e4b/fused_attn4_shipped`. MODE gptoss (tp4's):
`e4b/fused_attn4_m` and `e4b/reference_attn4_m` are REFUSED stubs written first (tp1/tp2 cited; tp4's bias rule), refreshed by `e4b/attn_only_m`'s own probes;
then `e4b/attn_only_m` ×2 (`--attn-4bit 0`) · `unsloth/ckpt_unsloth_m` (`--unsloth-load-in-4bit 1`) · `unsloth/ckpt_unsloth_mxfp4` ×2 (`--unsloth-load-in-4bit 0`:
the 16-bit load that keeps the MXFP4 experts packed; grouped_mm) · `hf/hf_peft_m` · `axolotl/ckpt_axolotl_m`. Alarms = tp4's per-family ceilings
(fetch/e4b/unsloth/hf/reference: granite 1800/1800/1800/1800/2400, olmoe 2400/2400/2400/2400/3000, gptoss 3000/3600/2400/2400/3600), axolotl = hf + 900.

**`tc2big`** (`tc2_big_box` → `tc2_big_family`, the field recipe: N 20, 8 rows at 0 and N): qwen3_5 (`Qwen/Qwen3.6-35B-A3B` @ `995ad96e…`, 40 layers; Unsloth
targets UT4 = `q_proj,k_proj,v_proj,o_proj` as tp4, plus `unsloth/ckpt_unsloth_m_experts` with UT4 + `--unsloth-target-parameters
mlp.experts.gate_up_proj,mlp.experts.down_proj`) and mixtral (`mistralai/Mixtral-8x7B-Instruct-v0.1` @ `eba92302…`, 32; e4b arms `--offload 1`, the others
resident; the seven). In THIS order: `e4b/fused_attn4_m` · `unsloth/ckpt_unsloth_m` · `e4b/fused_attn4_m_d2` · `unsloth/ckpt_unsloth_m_d2` ·
`unsloth/ckpt_unsloth_m_experts` (qwen3_5) · `hf/hf_peft_m` · `axolotl/ckpt_axolotl_m` · `axolotl/ckpt_axolotl_best` · `e4b/fused_attn4_shipped` ·
`e4b/reference_attn4_m` LAST · the `_mb1` pair on an OOM as `tc1_family`. Alarms qwen3_5 6000/3600/3600/1800/5400, mixtral 7200/5400/2400/1800/6000, axolotl = hf + 900.

**Arm driver** (`tc1_arm.py`, T23–T27): `--unsloth-load-in-4bit 0|1` (receipt `unsloth_load_in_4bit`; the double-quant kwarg is not passed on a 16-bit load);
`--unsloth-target-parameters a,b` (passed as PEFT `target_parameters` only when `get_peft_model`'s signature names it, read at runtime, else the arm REFUSES;
receipt `unsloth_target_parameters`); the census gains `expert_param_classes` (the class of every frozen expert parameter — `Mxfp4ExpertParam` on the packed
load) and `Params4bit_expert_linears` (per-expert Linear4bit under an experts container: gpt-oss's bnb-4bit class); `unsloth_zoo.mxfp4_gemm.Mxfp4GroupedMM.apply`
is counted per step (`unsloth_packed_calls_per_step_min/_max`; an absent name lands in `unsloth_backend_absent`); an attn_only arm's tag suffix names the
stubs it refreshes (`attn_only_m` → `fused_attn4_m` / `reference_attn4_m`, a `_d2` draw the same stubs) and its matched init / name-free sha cover the trainable
(attention) slots only; the HF arm passes `experts_implementation` only if the installed transformers accepts the kwarg (a TypeError naming it reloads without,
`accepted=False`) and records what dispatched (`hf_experts_dispatch`: config + torch grouped_mm calls per step); the probe decodes a self-decoding packed
parameter (`<class>-packed/dequantize()`, its control flipping one byte of the real storage in place and restoring it). Every name here that comes from
unsloth / unsloth_zoo source is read from `UPSTREAM-NOTES.md`, not executed on this machine (UNVERIFIED by execution; the selftest exercises the harness's
own paths on stand-ins).

**Reducer** (`tc1_reduce.py`, R11): the five families with their registered n_layers (32/16/24/40/32), attention census (granite 128, olmoe 64, mixtral 128;
gpt-oss REFUSED on the bias rule; qwen3_5 unregistered → the receipt's own structural census governs) and pins; TC1's readings per family, plus the HF
position quoted as `HF (bf16 experts) / e4b` (or 4-bit) with its regime; an Unsloth arm whose trainable count differs from e4b's VOID with the reason
`attention-only: …` when it adapted no expert parameter; gpt-oss (`attn_only_m` as the anchor) prints a `NO COMMON ADAPTER SET` line — both s/step values,
both peaks, both trainable counts — in place of every position, and applies no trainable / sha / quality predicate against e4b to the other frameworks' arms
there; `ckpt_unsloth_mxfp4` is VALID only with ≥ 2L packed expert parameters of a recorded class, `moe_backend_selected == grouped_mm` and the MXFP4 grouped
GEMM counted ≥ L·A per step; gpt-oss's bnb-4bit Unsloth arm reads the per-expert Linear4bit regime (≥ 2L Params4bit under its experts, either MoE banner);
mixtral's `FOOTPRINT` line (e4b under offload vs Unsloth resident: peak VRAM, s/step) leads its block; a t214 arm whose dispatch did not reach grouped_mm is
recorded on its row, never VOID. P1–P7 of the draft in their own table (`## TC2 predictions P1–P7`); P4's e4b leg reads against tp4's 6.3344 s/step
(`RESULTS-tp4-p46cut.md`), P5 against tp2's 0.361 (= Unsloth/e4b, the lane's convention) and 29.16 / 3.22 GB. The selftest adds 10 cases (53 in all).
