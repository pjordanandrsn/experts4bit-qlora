# TC1 — Qwen3-30B-A3B, one RTX 5090 per box: e4b (main) vs Unsloth 2026.9.14 vs HF 5.18/PEFT 0.21.2 vs axolotl 0.20.0 at matched work, matched init, matched adapter precision and matched 4-bit regime, with the frameworks' native rows on a second box (registered 2026-10-01, before any box is rented)

Issue: experts4bit-qlora#835. Lineage: tp4 (`bench/tp4/TP4-PREREG.md`; amendment 6 = the axolotl arm, registered 2026-09-19, never run) and its re-run on grouped-nf4-gemm 0.32.1 (`bench/tp4/RESULTS-tp4-p46cut.md`, 2026-09-19: Unsloth/e4b 4.490 on this family). Upstream facts: `bench/tc1/UPSTREAM-NOTES.md` (read from source, not assumed). Design review: `bench/tc1/DESIGN-REVIEW.md` (an independent adversarial pass; every HIGH finding is closed below, each MED either closed or carried as a stated limit).

## Why this lane, and what it corrects

1. **The comparator ran its slowest path in every earlier position.** Unsloth's MoE dispatcher prefers `torch._grouped_mm`, which on torch 2.8 runs only on sm_90; on the RTX 5090 / torch-2.8 image P38, tp2 and tp4 all recorded `moe_backend native_torch` (a compiler-disabled per-expert Python loop). Its own code names the fast route ("For max throughput use the grouped_mm backend") and its installer names torch 2.12.x+cu130 for Blackwell. The standing 4.490 compares e4b's fused kernel with that loop.
2. **Adapter precision was not matched in tp4, although its registration said so.** `tp4_arm.py` casts the adapters of non-e4b arms to fp32 and leaves e4b's as built: the committed 4.490 receipt records 192 e4b expert-adapter tensors in bf16 against Unsloth's 576 in fp32. bf16 adapters are cheaper per step and lose small updates to rounding.
3. **Adapter init was not matched.** e4b draws LoRA A from N(0, 1/r); PEFT (and Unsloth through it) from kaiming-uniform with bound 1/sqrt(fan_in): 3-5x the std at r 16. The quality readings on record (e4b lower at N=20 on every family; Unsloth lower at 200 steps in P38) are consistent with an init-scale effect and prove nothing about kernels.
4. **The 4-bit regime was not matched on double-quant** (e4b/HF off; Unsloth's and axolotl's defaults on), and the engagement predicate that protects e4b's number (`kernel_calls_per_step_min`) was satisfied by the loop fallback P46 caught.

## Claim under test

On one RTX 5090, Qwen3-30B-A3B at the pinned revision, the field recipe, the same tokens, the same 642,514,944 trainable parameters (attention + every routed expert, r 16, alpha 16), fp32 adapters, one per-slot LoRA A init, NF4 blocksize 64 on every quantised module with double-quant matched where the frameworks allow it (experts: off in every 4-bit-expert arm, as e4b's; attention: e4b's `quantize_attention_projections_4bit` leaves bitsandbytes' double-quant ON, so the HF arm's `BitsAndBytesConfig` sets it on too — HF quantises attention only — while Unsloth's and axolotl's single config governs experts and attention together and is set off for the experts' sake; the per-module regime is recorded in every receipt), the same bnb AdamW8bit call and schedule, N = 20:
(a) which frameworks train it in the 4-bit MoE regime at all (a refusal, an OOM or an install failure is a row);
(b) whether the loss trajectory is the SAME FUNCTION across frameworks (equivalence bands tied to the in-draw floor);
(c) what each costs (s/step median over steps 11..20, two interleaved draws, quoted as a point and the cross-draw interval; tok/s; peak VRAM; J/step);
(d) on a second box, what each framework's native configuration does on the same problem, labelled, beside the matched rows.

## Fixture (tp4's field recipe, byte-identical where tp4 defined it)

`unsloth/alpaca-cleaned` @ `0fe581eb…` -> `tp4_alpaca.py` (seed 3407, 1,200 train / 48 held-out; `ds_alpaca.json` sha `5324987a…`); tokenised once with the family's tokenizer at the pinned revision, the notebooks' Alpaca template, EOS appended, truncation 2048; N = 20; each step = 4 micro-batches x 2 rows, right-padded to the longest row, labels -100 on pads, rows in fixed order (the padded-length distribution is recorded per arm); eval at step 0 and step N on the first 8 held-out rows, **per-row losses recorded**; r 16 / alpha 16 / dropout 0 / bias none on every attention projection found by structure + every routed expert; bnb AdamW8bit(lr 2e-4, wd 0.001), linear schedule with 5 warm-up steps (steps 0-1 at lr 0 as the notebooks run it); seed 3407; bf16 compute, no autocast; gradient checkpointing on (e4b/HF/axolotl: HF non-reentrant; Unsloth: "unsloth", reentrant with layer-input offload — its own mode, recorded).

Model: `Qwen/Qwen3-30B-A3B` @ `ad44e777bcd18fa416d9da3bd8f70d33ebb85d39` (48 layers, 128 experts, top-8), unpinned fetch with the proof that the staged snapshot equals the pin (e4b#404), else LOAD_FAULT rows.

## Environments (every version in `versions.txt` and every receipt)

- e4b: GitHub `main` at the launch commit + grouped-nf4-gemm at the **v0.34.0 release commit `846b512b905468c08f5748943d08769b572affa2`** (the commit the annotated tag `v0.34.0` points at; `bdcd6ad3…` is the tag object itself); transformers 5.18.0, bitsandbytes 0.50.2, peft 0.21.2 in `venv-e4b` on the image's torch 2.8.0+cu128 / triton 3.4.0 (`pytorch/pytorch:2.8.0-cuda12.8-cudnn9-devel`).
- HF: `venv-e4b`'s interpreter — the only difference from the e4b arm is the code path. On torch 2.8 transformers' default `grouped_mm` experts implementation is Hopper-only, so the per-expert Python fallback runs; recorded, and an `hf_best` row on torch 2.14 exists on the second box if any HF row trains.
- Unsloth, two venvs: `venv-unsloth` = `unsloth[cu130-torch2121]==2026.9.14` + `unsloth_zoo==2026.9.9` (torch 2.12.1+cu130, the newest torch its wheel admits; its own pins, transformers <= 5.5.0, recorded) with `UNSLOTH_MOE_BACKEND=grouped_mm` — the comparator; `venv-unsloth-t28` = `unsloth[cu128-torch280]` on the image's torch 2.8 — the field-image configuration every earlier position was measured against, a labelled row. cu130 wheels need driver >= 580 (29 of 30 verified 5090 hosts sampled 2026-10-01 have it); a host below records every cu130 arm as UNSUPPORTED-on-this-host with the driver, no relaunch. Zoo commit recorded; zoo main's #1529 ("never float32") is not in the release — stated.
- axolotl: `axolotl==0.20.0` in `venv-axolotl` built with `uv` (CPython 3.12; torch 2.14.1+cu130 — torch >= 2.13 ships only as cu130 Linux wheels, same driver gate; transformers 5.17.0 / peft 0.21.0 / trl 1.13.0 at its pins), alarm 2,700 s; INSTALL_FAILED rows with the pip log tail, never silent absence.

## Arms

Matched set, box A (`TC1_FAMILIES=qwen3`), in THIS order (the equivalence anchor third so no deadline can eat it):
1. `e4b/fused_attn4_m` — `enable_fast_train(dgrad=True)` (an opt-in, stated: e4b's default `dgrad=False` is a labelled row on box B) + attention 4-bit (structural census), expert adapters cast to fp32, matched init.
2. `unsloth/ckpt_unsloth_m` — `venv-unsloth`, `UNSLOTH_MOE_BACKEND=grouped_mm`, `FastLanguageModel` (FastModel fallback as tp4), `load_in_4bit=True` with `bnb_4bit_use_double_quant=False` where its loader accepts it (the loaded quant state is read back and recorded either way), targets = the notebooks' seven (auto-expanded to the expert stacks), matched init, fp32 adapters, `use_gradient_checkpointing="unsloth"`.
3. `e4b/reference_attn4_m` — the per-expert reference path, fp32 adapters, matched init: the internal parity control and the e4b-side floor for the equivalence bands.
4. `e4b/fused_attn4_m_d2`, 5. `unsloth/ckpt_unsloth_m_d2` — second draws, interleaved.
6. `hf/hf_peft_m` — transformers + BitsAndBytesConfig(nf4, bf16 compute, double-quant ON to match e4b's attention projections; tp4's arm ran it off) + PEFT target_modules (structure) + target_parameters (every 3-D floating expert stack, structure), matched init; its regime (bf16 expert stacks, weight-side PEFT delta) RECORDED, never a VOID.
7. `axolotl/ckpt_axolotl_m` — axolotl 0.20.0's model stack for `load_in_4bit: true, adapter: qlora, quantize_moe_experts: true, bnb_4bit_use_double_quant: false` + `lora_target_parameters` by structure, under this harness's loop, matched init; TP4 amendment 6's predicates.
8. `e4b/fused_attn4_m_prof`, 9. `unsloth/ckpt_unsloth_prof` — 3 warm + 3 profiled steps each (torch.profiler, P45's instrument) with `nvidia-smi dmon` beside: device-busy fraction, device events per step, top kernels (the Unsloth profile must list `_grouped_mm` kernels).
Secondary `_mb1` pair (micro-batch 1 x accum 8) for any framework whose primary matched arm OOMed, as tp4.

Native and labelled rows, box B (`TC1_FAMILIES=qwen3native`), every ratio within this box:
- `e4b/fused_attn4_m` (one draw, the box's own anchor), `e4b/fused_attn4_shipped` (bf16 expert adapters, N(0, 1/r) init: as the loader builds it), `e4b/fused_attn4_m_nodgrad` (`dgrad=False`, e4b's default), `e4b/fused_attn4_m_t212` (e4b + grouped-nf4-gemm installed into `venv-unsloth`'s torch 2.12.1+cu130 — the torch a Blackwell user installs; an install or tripwire failure is a row);
- `unsloth/ckpt_unsloth_best` (grouped_mm + `UNSLOTH_MOE_RECOMPUTE=0` + `UNSLOTH_MOE_GC_REPLAY_PIN=1`, native init: "native kernels under the harness loop" — its trainer loop and packing are NOT measured, stated), `unsloth/ckpt_unsloth_t28` (the field-image configuration), `unsloth/ckpt_unsloth_triton` (`UNSLOTH_MOE_BACKEND=unsloth_triton`; what its dispatcher runs with NF4 experts is what the counters say) (a `use_gradient_checkpointing=True` Unsloth row is not in this cut; P38 measured the two modes 0.2 % apart at seq 512);
- `axolotl/ckpt_axolotl_best` (`plugins: [KernelsPlugin]`, `expert_backend: scattermoe`, `moe_bnb_fast: true`: ScatterMoE keeps the bnb-4-bit experts packed and fuses the LoRA factors — the closest field analogue of e4b's fused path; a plugin that cannot be driven outside `axolotl train` is a refused row with the reason);
- `hf/hf_peft_m_mb1_t214` (the HF arm in `venv-axolotl`'s torch 2.14 with `experts_implementation="grouped_mm"`, only if HF's primary OOMed on box A).
Alarms (deadline-derived, capped): e4b fused 3600, Unsloth 3600, reference 5400, HF 1800, axolotl 2700 (+ install), profiled arms 2400, every labelled row 3600. An arm that cannot fit before the deadline is a `not_run` stub `host-limited`; box B's rows are dropped from the END of its list.

## Validity (VOID never enters a ratio or an equivalence reading)

tp4's rules (L = 48, A = 4) plus: **e4b fused** `n_patched == L`, `kernel_calls_per_step_min >= 2·L·A` AND `lora_path_loop == 0 on every step` (the grouped-LoRA delta on the padded path; the loop share is recorded) and `n_attn4 == structural census`; **Unsloth** `Params4bit_expert_stacks >= 2·L`, `experts_forward_calls_per_step_min >= L·A`, `n_bnb4bit_unwrapped >= L`, the MoE-LoRA banner, and for a grouped_mm arm `torch._grouped_mm` calls per step >= 6·L·A with zero manual-fallback calls; **HF / axolotl** `experts_forward_calls_per_step_min >= L·A` and >= 1 expert parameter adapted by structure (axolotl: `quantized_moe_experts_n >= 2·L` for its 4-bit regime label); **all**: C1 frozen bytes bit-exact over every frozen expert parameter whatever its dtype, with a positive control that flips one byte of a copy of a hashed tensor and runs the same hasher; step count == N; same tokens sha; same trainable count as `e4b/fused_attn4_m`; **matched set**: `adapter_dtypes_after` all fp32, `matched_init.complete`, one identical name-free `matched_init_sha` across every OK arm of the set, double-quant recorded off (or recorded on and the arm labelled, never silently), step-0 held-out within **0.01** of `e4b/fused_attn4_m` reads SAME-BYTES-CLASS, 0.01-0.05 reads NEAR (recorded, still VALID), > 0.05 VOIDs the arm (with B = 0 and lr 0 at step 0 this reads the frozen bytes, not the init), and no dynamo recompile inside steps 11..N (else UNSTABLE).

## Verdict column (mechanical, per attempt)

VALID — OK, validity holds, quality COMPARABLE · VOID — OK, a validity predicate fails · QUALITY_FAIL — validity holds, held-out |delta| at N > 0.05 nats against `e4b/fused_attn4_m` · OOM · UNSUPPORTED — REFUSED / INSTALL_FAILED / LOAD_FAULT · non-readings: HARNESS_ERROR, ALARM, NOT_RUN; UNSTABLE is a flag on a VALID row (draws apart by > 5 %, or recompiles inside the window) that keeps it out of a position.

## Readings (pre-registered)

- **Position** (per comparator, box A): s/step ratio other/e4b as the point (medians over both draws) AND the interval (min, max) over the four cross-draw ratios; quoted only when both arms are VALID and not UNSTABLE and the quality reading is COMPARABLE; peak VRAM delta, J/step ratio, tok/s beside it; the comparator's regime label (which modules are 4-bit, double-quant, backend, delta path, checkpointing mode, torch) in the same sentence.
- **Equivalence** (matched set, pairwise against `e4b/fused_attn4_m`; the fused-vs-reference pair is the e4b-side control): |delta step-0| <= 0.01; **EQUIVALENT** iff median per-step |delta train loss| and |delta held-out at N| are both <= max(0.005, 3 x the in-draw fused-vs-reference |delta| of the same quantity) — if the reference did not run, EQUIVALENT is unreadable and only COMPARABLE (<= 0.05 on both) or DIVERGENT remain; an equivalence narrower than the draw noise floor (max |held-out(d1) - held-out(d2)| within an arm at N) is reported as "inside the draw noise". Also reported: |delta train loss at step 2| (the first init-sensitive step), paired held-out mean +- SE over the 8 rows and the count of rows favouring each arm.
- **Frozen base across frameworks**: dequantised bytes of layer 0 expert 0 (gate_up, down) and layer 0 q_proj, sha per arm: SAME-BYTES / DIFFERENT / N-A (regime differs), with each arm's own flip control shown to detect.
- **Native rows** (box B): every ratio vs that box's `e4b/fused_attn4_m`, labelled with the configuration; `e4b_t212` within 5 % of the torch-2.8 arm closes the torch confound for the equivalence reading, else torch enters the P3 bisection.
- **Where the step goes** (arms 8-9): device-busy fraction, events per step, top kernels; the ratio is stated as a property of (card, host CPU model, torch, fixture) on one host.
- e4b internal parity (fused_m vs reference_m), tp1's band, informational.

## Predictions (falsifiable; competitor estimates biased up)

- P1 matched position Unsloth(grouped_mm, torch 2.12.1)/e4b in **[0.8, 3.0]**: the standing 4.490 was against the loop; the grouped GEMM over dequantised NF4 with the LoRA delta as two more grouped GEMMs should be several times faster than it, and fp32 adapters cost e4b's delta too. Below 1.0: e4b is slower at matched work on this card and the public position is withdrawn. The 4.490 row is superseded by this lane's row whatever the value.
- P1b `ckpt_unsloth_t28` reads within 15 % of tp4's 29.05 s/step **only if** its recorded backend is `native_torch`; if its dispatcher engages the Triton path on this image, the row is a new measurement, not continuity.
- P2 each arm's two draws agree within 5 %; refuted -> UNSTABLE, no position.
- P3 the matched set is EQUIVALENT (e4b fused, e4b reference, Unsloth). Refuted -> no position from this lane; TC4 bisects adapter precision, init, checkpointing mode, attention implementation and torch version (the `e4b_t212` row is the first cut).
- P4 `e4b/fused_attn4_shipped` is 1.05-1.3x faster per step than `e4b/fused_attn4_m` on box B and its held-out loss at N=20 is lower by >= 0.01 nats (init scale); refuted if the matched arm is as good or better.
- P5 HF+PEFT OOMs at the primary recipe and at `_mb1` (bf16 expert stacks on 32 GB).
- P6 axolotl matched arm: INSTALL_FAILED, UNSUPPORTED or OOM; if it trains in the 4-bit regime, axolotl/e4b in [1.5, 6]. `ckpt_axolotl_best` (ScatterMoE): if it trains, in [0.7, 2.5] — the one competitor path that, like e4b's, never dequantises the whole stack.
- P7 `ckpt_unsloth_best` at most 1.3x faster than `ckpt_unsloth_m` at a higher peak VRAM.
- P8 the profiled grouped_mm Unsloth arm is not host-bound (device busy >= 0.5); the profiled e4b arm reads >= 0.5 too (P46 fixed the loop; P45 had read 0.108 before it).
- P9 e4b internal parity PASS; P10 C1 bit-exact in every OK arm with the real-storage control detecting; expert slot hashes SAME-BYTES between e4b and Unsloth (NF4/64, double-quant off on both — attainable: on CPU e4b's per-expert bytes equal bitsandbytes' whole-stack slice); the attention slot SAME-BYTES between e4b and HF (both NF4/64 with double-quant) and DIFFERENT-by-regime for Unsloth/axolotl (their double-quant is off) — recorded either way.

## Decision rules

- P1 and P3 hold -> the training position for this family moves to this lane's matched row (point + interval, fp32 adapters, matched init, matched regime, Unsloth on grouped_mm), superseding `e4b.train.h2h.unsloth.qwen3.5090.2026-09-19`; that row stays measured with the precision/backend note.
- P3 refuted -> no position; TC4 (causal) before any further comparison on this family.
- P1 below 1.0 -> the README/STATUS position is withdrawn pending the cause; nothing is retuned.
- Native rows are reported beside matched rows, never instead of them; TC1c (the H100 matched pair, where `torch._grouped_mm` is native even on torch 2.8) is registered next whatever TC1 reads — the card class is the counterexample the 5090 cannot see; Gemma-4 (e4b's own parity floor, P67) and Mixtral under offload are carried beside any Qwen3 position in the write-up.

## Budget and stop rules

Two RTX 5090 draws on Vast verified-secure, rate ceiling $0.69/h, guard 4.5 h each, estimate $3.11 each (< $15). Pre-flight: >= 320 GB disk, >= 98 GB host RAM, >= 40 MB/s. STOP: not a 5090 (refused); a second consecutive pre-flight refusal ends the day's attempts; an arm past its alarm is a row; no relaunch changes the fixture; a cu130 driver gate failure is rows, not a relaunch.

## Out of scope (stated, not hidden)

Unsloth's and axolotl's own trainer loops with packing/padding-free (a different token stream; a follow-up row if wanted); serving; Gemma-4 and Mixtral (TC2); larger rank and the tokens-per-step scaling (TC1b); other card classes (TC1c); DeepSeek-V4 / Kimi-K3; any change to a gate, threshold, licence or registered claim.

## Harness

`bench/tc1/` (`tc1_arm.py`, `tc1_run.sh`, `tc1_drive.sh`, `tc1_reduce.py`; CI tests `tests/test_tc1_arm.py`, `tests/test_tc1_reduce.py`), a copy of tp4's with the additions this document names; `README.md` there lists them and the facts measured while building it (the attention double-quant, the kernel pin, the SAME-BYTES attainability). Every control has a failing case the selftests run. The on-box disk floor stays tp4's 200 GB (the launcher's pre-flight orders 320 GB).

## Amendments

### Amendment 1 (2026-10-01T23:02Z, before any arm has produced a row): the driver floor is a lane refusal and a redraw, not four hours of refused rows

**What the first draws showed.** `tc1-5090-1` / `-2` (22:52Z) were refused by `tc1_drive.sh` before any work: the launch
command lacked `TC1_BOX=A` (a launch defect, not a harness one; $0.0201 + $0.0120). `tc1-5090-3` (22:55Z, the matched
box) drew machine 37958 at driver 575.57, the host P86 had already met below the cu130 floor; the run script's gate
correctly marked `cu130_ok=0` and then continued into the fetch, which under the registration would have produced
refused rows for the comparator (`ckpt_unsloth_m`), its second draw and both axolotl arms -- a box with no position. It
was stopped by the operator at 22:58Z ($0.0297, rc 143, teardown proven). `tc1-5090-4` (the native box) failed its
pre-flight on download bandwidth (19.7 MB/s on machine 147733, $0.0154, teardown proven after a provider 429).

**What was wrong.** The registration's stop rule read "a cu130 driver gate failure is rows, not a relaunch". That is the
right rule for an arm, and the wrong rule for a floor every judged arm depends on: the launcher already classifies a
lane's registered host floor (RAM, driver, CPU vendor) as refusal code 18 and excludes that machine on the next draw.

**The amendment.** `tc1_run.sh` refuses with code 18 at the driver gate, before any install or fetch, when the host's
driver is below 580; the receipt names the machine; the next draw passes it as a lane-refusal exclusion. Nothing in the
fixture, the arms, the predicates or the bands changes. The four receipts above stay in the ledger as what they are.
The registered stop rule now reads: a driver below 580 is a code-18 refusal and one redraw with the machine excluded;
a second refusal in the same day ends the lane's attempts for the day.

### Amendment 2 (2026-10-01T23:47Z, before any arm has produced a row): the cu130 Unsloth venv needs PyTorch's cu130 index

**What the first good-host draws showed.** `tc1-5090-7` (the native box, driver 595.91) and `tc1-5090-12` (the matched box,
driver 580.126) both reached the venv builds; on `-7` the `venv-unsloth` pip ran 30 minutes at 99 % CPU with no network
socket and a 25 MB venv: the `unsloth[cu130-torch2121]` extra pins `torch==2.12.1+cu130` and `torchvision` `+cu130`, which
exist only on PyTorch's cu130 index, so pip's resolver backtracked through every candidate it had until its 2,700 s alarm
would have made the comparator INSTALL_FAILED on both boxes. Both were stopped by the operator at 23:46Z
(rc 143; ≈ $0.30 and ≈ $0.10, teardown proven).

**The amendment.** `tc1_run.sh` passes `--extra-index-url https://download.pytorch.org/whl/cu130` on the `venv-unsloth`
install and on the e4b/grouped-nf4-gemm install into it (the `t212` row), exactly as the axolotl venv already did. The
box-side venv builds are otherwise unchanged; the registered torch for the comparator stays 2.12.1+cu130; nothing in the
fixture, the arms, the predicates or the bands moves.

### Amendment 3 (2026-10-02T03:33Z, after the three TC1 boxes produced their rows; before any TC2 / TC1c box and before the TC1 read): the axolotl venv never installed, so the axolotl rows are re-asked on their own box

**What the boxes showed.** `tc1-5090-16` (the matched box) and `tc1-5090-14` (the native box) both reached the axolotl venv build and
both failed it in the same way (`logs/pip_axolotl.log`): uv resolved `axolotl==0.20.0` against PyTorch's cu130 index first and
reported "there is no version of packaging==26.0 ... `packaging` was found on https://download.pytorch.org/whl/cu130, but not at the
requested version" -- uv's default first-index strategy takes every package the extra index carries from that index alone, and the
cu130 index carries an old `packaging`. The arm rows read INSTALL_FAILED -> UNSUPPORTED on both boxes, and the reducer scored P6 HELD
("axolotl UNSUPPORTED") on them. That HELD rests on this harness's install line, not on axolotl: **P6 is re-scored UNTESTED on
`tc1-5090-14` and `tc1-5090-16`**, and the read quotes no axolotl row from either box. Reproduced off-box before this amendment with
uv 0.12.22 resolving for x86_64-manylinux_2_28 / CPython 3.12: the registered line fails with the same message; the same line with
`--index-strategy unsafe-best-match` resolves axolotl 0.20.0, torch 2.14.0+cu130, torchao 0.18.0+cu130, triton 3.8.0, transformers
5.17.0, peft 0.21.0, bitsandbytes 0.50.2, packaging 26.0, xformers 0.0.35 (axolotl's own pins; the cu130 wheels win on version).

Two smaller things the same boxes showed: the reducer read the profiled arms' kernel-table sidecars (`<arm>_prof_profile.json`) as
receipts and printed them as HARNESS_ERROR rows (`e4b/fused_attn4_m_prof_profile`, `unsloth/ckpt_unsloth_prof_profile` on `-16`);
and the native box's `hf/hf_peft_m_mb1_t214` row read NOT_RUN because its gate ("the judged family's `hf_peft_m` OOMed on THIS box")
cannot be read on a box that never runs `hf_peft_m` -- the gate's fact is on the matched box (`-16`: OOM at micro-batch 2 and at
micro-batch 1), so the row was never asked.

**The amendment.**
1. `tc1_run.sh` passes `--index-strategy unsafe-best-match` on the axolotl uv install (the only change to that venv's recipe; the pins,
   the python, the torch route are the registered ones).
2. `tc1_reduce.py` skips `*_profile.json` sidecars when it loads a receipt directory (they are kept in the receipts, as registered; they
   are not rows).
3. A new family token `qwen3axolotl` (`tc1_axolotl_family`) re-asks the axolotl rows on one more RTX 5090 draw, each a position within
   that box against its own `e4b/fused_attn4_m`, in this order: `e4b/fused_attn4_m`, `axolotl/ckpt_axolotl_m` (arm 7's flags), `e4b/fused_attn4_m_d2`,
   `axolotl/ckpt_axolotl_m_d2`, `axolotl/ckpt_axolotl_best` (the scattermoe native-best row), `hf/hf_peft_m_mb1_t214` (gated on the matched
   box's registered OOM record above, so it runs unconditionally here). Two draws of the matched pair so the axolotl matched position carries the
   same cross-draw interval and 5 % stability rule as the Unsloth one; the token builds no Unsloth venv (it runs no Unsloth arm). The reducer
   registers the token (`EXPECTED["qwen3axolotl"]`, the axolotl second draw) and scores P6 from that box when it ran (the judged box's axolotl
   row otherwise, as before). Alarms: fetch 5400, e4b 3600, HF 1800, axolotl 2700 (TC1's). One draw, ceiling $0.69/h, 3 h guard, estimate $2.07.
4. The pin for the new box is this amendment's merge on `main`: e4b 0.38.1, whose `experts4bit/` tree is byte-identical to the
   TC1 boxes' 079a422 (`git diff --stat 079a422 e36968f -- experts4bit/` is empty; the read repeats the check at the merge), with
   grouped-nf4-gemm unchanged at 846b512 (`tc1_drive.sh`'s pin). TC2 and TC1c launch from the same merge for the same reason.

**What does not move.** The fixture, the arms already measured, the predicates, the bands, the Unsloth / HF / e4b rows of `-14`,
`-15` and `-16` (they stand as measured), the decision rules. P6's wording is unchanged; only which box answers it.

### Amendment 4 (2026-10-02T17:38Z, a correction, before any re-run): the axolotl rows were this harness's no-autocast loop, not axolotl

**What was published.** The TC1 read (`e4b.train.h2h.unsloth.qwen3.5090.2026-10-02.axolotl-unsupported`) and the TC3 read said axolotl 0.20.0
at its pins does not train Qwen3-30B-A3B: five attempts on three cards died in the first forward inside transformers' `Qwen3MoeTopKRouter`
(`F.linear`: bf16 activations against an fp32 router weight), and the cause was left open.

**What the source says** (axolotl 0.20.0, transformers 5.17.0, accelerate 1.15.0, read 2026-10-02). axolotl's `ModelLoader` upcasts every
module whose name ends in `.gate` to fp32 on purpose (`loaders/model.py:611-616` -> `_convert_embedding_modules_dtype`, `:1449-1451`,
`before_kbit_train_or_finetune=True`) and the later cast back to bf16 (`:636-642`) leaves `.gate` in fp32. In axolotl's own run that is
safe: its trainer sets `TrainingArguments.bf16`, accelerate's `mixed_precision="bf16"` wraps `model.forward` in `torch.autocast`
(`accelerator.py:1824-1835`), and `F.linear` casts the fp32 weight to bf16 on every call. This harness uses axolotl's loader but drives the
forward itself with no autocast, as registered for every arm ("bf16 compute, no autocast"), so the fp32 router met bf16 activations. The
receipts agree: every failing axolotl arm records `router_gate ... "dtype": "torch.float32"`, the HF and e4b arms bf16. Granite trained
because its router is named `block_sparse_moe.router`. **The failure is the harness's, not axolotl's**, and the published row is retired.

**The fix.** `load_axolotl` casts the frozen fp32 `.gate` weights to bf16 once, after `ModelLoader.load()`
(`axolotl_router_recast`; trainable and quantised gates untouched). That is exactly what autocast computes per call, so the router's
logits and input gradient equal those of axolotl's own trainer, and the "no autocast" rule stays uniform across arms. The receipt
records the count and whether each cast was an exact bf16 round trip (informational; autocast rounds the same way). A recorded
difference that remains, labelled and not a matched-work predicate: axolotl auto-enables its LoRA kernels, which keep fp32 adapter
weights but compute the attention adapters in bf16, where the HF arm computes them in fp32.

**The scattermoe arm.** Its KernelsPlugin fetches `kernels-community` kernels by version at load, and every arm ran with the Hub
offline, so it refused at load on every family. From this amendment only `axolotl/ckpt_axolotl_best` runs with `HF_HUB_OFFLINE=0`, as an
axolotl user's run does; the model revision stays pinned by sha, and the arm records the kernel commits it fetched (`hub_kernels_cached`).

**The re-run.** TC1 amendment 3's `qwen3axolotl` token on one RTX 5090 from this amendment's merge: e4b `fused_attn4_m` x2, axolotl
`ckpt_axolotl_m` x2, the scattermoe native-best, the HF torch-2.14 `grouped_mm` mb1 row. P6 is re-scored on it as registered: axolotl plain
UNSUPPORTED/OOM or axolotl/e4b in [1.5, 6]; scattermoe in [0.7, 2.5] if it trains. A new failure is a row with its own cause. Budget: one
RTX 5090, ceiling $0.69/h, 4.5 h, under $3.20; the standing no-ask tier.

**Not fixed here, recorded.** On Qwen3.6-35B-A3B (`tc1-5090-27`) the axolotl arm refused with `NoMatchingPeftModuleError`: axolotl's
loader read the checkpoint as multimodal (`is_multimodal: true`) and built the vision-language class, whose language layers sit under a
different module path from the `AutoModelForCausalLM` skeleton on which this harness enumerates target names. That refusal is also the
harness's; building the skeleton from the class axolotl loads is the fix, not made in this amendment.

### Amendment 5 (2026-10-02T19:55Z, after the axolotl re-run read, before any box): native-best against native-best on one card

**Why.** The axolotl re-run (`tc1-5090-30`) found the campaign's strongest counterexample to e4b's speed lead: axolotl's scattermoe
native-best stepped at 0.929 x e4b's MATCHED fused path on the RTX 5090 (one draw, axolotl's own init). That compares axolotl's best
configuration with e4b's matched one, and e4b's own native configuration (as shipped: bf16 expert adapters, its own init) was measured
only on another host (0.671 x its matched step, `.shipped-vs-matched`). Before any statement that one framework trains this family fastest
as its users run it, the three native-best configurations run on one box, two interleaved draws each.

**The token** `qwen3nativebest` (TC1_BOX=A): e4b `fused_attn4_shipped`, axolotl `ckpt_axolotl_best` (scattermoe, Hub reachable for its
kernels as amendment 4 allows), Unsloth `ckpt_unsloth_best` (grouped_mm, speed tilt, native init), interleaved, two draws each; then e4b
`fused_attn4_m` once as the box's matched anchor for the validity predicates. The field recipe and tokens as TC1. Not matched work: the
adapter precision and init are each framework's own, and every row says so; held-out losses are reported beside the times, never
judged EQUIVALENT across different inits.

**Prediction P13** (registered before the box; read off the reducer's per-arm medians and draw-stability lines): e4b as shipped is the
fastest native configuration of the three -- axolotl native-best / e4b shipped > 1.0 AND Unsloth native-best / e4b shipped > 1.0, each
over stable pairs (draws within 5 %). Either ratio below 1.0 on stable pairs FALSIFIES it, and that framework's native path is the
fastest of the three on this card, said so. An unstable pair leaves that half UNTESTED.

**Decision rules.** P13 holds -> a labelled native-best row registers e4b shipped as the fastest native configuration on the 5090 for this
family, beside (never replacing) the matched positions; refuted -> the faster framework's native row is the finding, and no "fastest"
statement is made for e4b on this card. Nothing here moves a matched position. Budget: one RTX 5090, ceiling $0.69/h, 4.5 h, under
$3.20; the standing no-ask tier.

### Amendment 6 (2026-10-02T21:27Z, after `tc1-5090-33`'s first five arms, before any re-run): the scattermoe arm's second draw reaches the Hub too

**What happened.** Amendment 4 let `axolotl/ckpt_axolotl_best` reach the Hub for its kernels by matching that tag exactly. Amendment 5's
second draw of the same arm is tagged `ckpt_axolotl_best_d2`, so it ran with the Hub offline and refused at load (`Version 2 of
'kernels-community/rotary' is not available in the local cache`). A harness defect: the arm is the same configuration, and its first
draw on the same box trained (5.128 s/step). With one draw, P13's axolotl half is UNTESTED by its own rule.

**The amendment.** The Hub rule covers both draws (`ckpt_axolotl_best` and `ckpt_axolotl_best_d2`). The axolotl half of P13 is re-asked
on one box with the amendment-5 token minus the Unsloth arms and the matched anchor (`TC1_SKIP`): e4b `fused_attn4_shipped` x2 and axolotl
`ckpt_axolotl_best` x2, interleaved as registered. P13's Unsloth half is scored on `tc1-5090-33`, which holds both Unsloth draws. Budget:
one RTX 5090, $0.69/h, 4.5 h, under $3.20; the standing no-ask tier.

### Amendment 7 (2026-10-02T21:45Z, after `tc1-5090-33`'s full read, before any re-run): the whole native-best box again, with clocks recorded

**What happened.** `tc1-5090-33` ran all seven arms of amendment 5's token. Amendment 5 registered the token in the box script but not in
the reducer, which therefore had no `n_layers` for it and read the matched anchor VOID. With the token registered (below), every arm that
trained is VALID. P13 still reads UNTESTED on both halves by its own rule:

- **The axolotl half.** The second scattermoe draw refused with the Hub offline (amendment 6).
- **The Unsloth half.** e4b shipped's two draws differ by 6.1 % (4.306 and 4.578 s/step) against the 5 % rule. Unsloth's pair agreed
  within 0.5 %.

The second e4b draw ran 5.5-8.5 % slower on every one of steps 11-20, on byte-identical tokens, at a lower mean power (255.7 W against
266.8 W). Unsloth's two draws on the same box differed per step in both directions. The box recorded no clock, temperature or host load,
so it cannot say whether the card or the host slowed.

**The amendment.**

1. The re-run is amendment 5's whole token: all seven arms in the same order. Amendment 6's reduced set (no Unsloth arms, no anchor) is
   withdrawn, because P13's Unsloth half is now as untested as its axolotl half, and one box answers both.
2. P13 is scored on the re-run box alone. `tc1-5090-33`'s rows are reported as that box's reading and are never pooled with the re-run's
   draws.
3. Each second, the box's sampler also records the SM and memory clocks, the GPU temperature, the active clock-event reasons, the host
   load average and the host's aggregate CPU counters (`gpuclk_<arm>.txt`, beside the unchanged `vram_<arm>.txt`). These are descriptive.
   No predicate reads them, and they cannot void an arm.
4. The reducer registers the token: `n_layers` 48, attention census 192, the seven arms in box order, a second draw for each native arm,
   and P13 scored mechanically per half. A half is HELD when its point ratio over two stable draws a side is above 1.0, FALSIFIED at or
   below 1.0, and UNTESTED otherwise. P13 is FALSIFIED when either half is, and HELD when both are.
5. If e4b shipped's pair is unstable again, P13 stays UNTESTED. The instability, with its telemetry, is then the reported finding, and
   no third box runs under this amendment.

Budget: one RTX 5090, $0.69/h ceiling, 4.5 h cap, under $3.20 (`tc1-5090-33` cost $0.51); the standing no-ask tier.

### Amendment 8 (2026-10-02T23:23Z, after `tc1-5090-34`'s read, before any box): e4b shipped against axolotl's scattermoe over 200 steps

**What `tc1-5090-34` showed.** All seven arms of amendment 5's token are VALID. P13 is scored there and is final:

- **The Unsloth half HELD.** Unsloth native-best / e4b shipped is 1.794 [1.790, 1.797]. e4b's draws are 3.277 and 3.279 s/step, and
  Unsloth's are 5.869 and 5.889.
- **The axolotl half is UNTESTED.** The scattermoe draws ran 4.574 and 4.107 s/step, 10.8 % apart.
- **P13 is UNTESTED.** No third box runs for it.

The scattermoe arm's spread has a visible shape. In both draws, steps 1, 3 and 6 took 216, 61-65 and 76-81 s, and later spikes fall on
the same steps in both draws. Dynamo recorded no recompile and no graph break, so the cost sits outside torch.compile. On steps 14-16,
the arm ran 2.98-3.46 s against e4b shipped's 3.02-3.20 s on the same tokens, a ratio of 0.94-0.99 when each side takes its faster draw.
This is consistent with a per-process kernel warm-up keyed on batch shape, such as Triton autotuning in the scattermoe kernels. That
cause is not measured. A 20-step window cannot separate such a warm-up from the steady step, and a user's run is longer than 20 steps.

**The token** `qwen3nativebest200` (TC1_BOX=A) runs these arms in order:

1. e4b `fused_attn4_shipped_200`;
2. axolotl `ckpt_axolotl_best_200`;
3. the second draw of each;
4. e4b `fused_attn4_m_200` as the box's matched anchor.

Every arm uses TC1b's curve recipe: the field recipe for 200 steps, with held-out loss every 40 steps on 16 rows. Each framework keeps
its own init and adapter precision, as in amendment 5. The Hub rule becomes a prefix (`ckpt_axolotl_best*`), so every scattermoe tag
reaches the Hub. The box installs no Unsloth venv, because it has no Unsloth arm.

**The reducer** registers the token with `n_layers` 48, census 192 and the 200-step matched arm as its anchor. That anchor serves the
validity predicates, the trainable count and the 0.05-nat quality gate. TC1b measured e4b shipped 0.026 above matched at step 200.

**Prediction P14** (a new question; P13 is not re-scored): over steps 101..200, axolotl scattermoe native-best / e4b shipped lies in
[0.90, 1.10]. Each side needs two VALID draws whose late-window medians agree within 5 %. The ratio is taken over the per-side medians,
and its interval over the four cross-draw ratios. Outside the band P14 is FALSIFIED. An unstable, missing or non-VALID side leaves it
UNTESTED.

**The ordering reading.** If the whole interval lies above 1.0, e4b shipped steps faster at steady state. If it lies below 1.0, axolotl's
scattermoe does. Otherwise there is no ordering at this resolution. The 11..200 medians are reported beside the late window.

**Decision rules.** Whatever P14 reads, no "fastest" statement is made for e4b, because P13 is UNTESTED. The ordering reading becomes a
labelled native-best row for steady state over 200 steps on this card, beside and never instead of the matched positions. If axolotl is
faster, that is the finding, and the claims register and the solution page say so.

Budget: one RTX 5090, $0.69/h ceiling, 4.5 h cap, under $3.20 (about 2 h of box time estimated); the standing no-ask tier.

### Amendment 9 (2026-10-03T05:31Z, after the P14 read, before any box): P14 on a second host

**Why.** P14 was read on one host, `tc1-5090-35` (Vast machine 96642, Intel Xeon E5-2698 v4). TC1's within-box ratios have moved
between hosts before. On `tc1-5090-34`'s AMD EPYC 9334, the scattermoe arm's warm steps 14-16 ran at 0.94-0.99 of e4b shipped, nearer
parity than `tc1-5090-35`'s 0.911. The steady-state ordering is the campaign's main counterexample to e4b, so it is asked once more on
a different host before the register treats it as more than one host's reading.

**The box.** The `qwen3nativebest200` token, unchanged: the same five arms and order, the 200-step curve recipe, and the same pins. The
e4b head is `7c31b88`, the commit `tc1-5090-35` ran, so the harness and the package are byte-identical to that box. A box that lands
on machine 96642 is a repeat of the same host, not a second host. That leaves P15 UNTESTED and allows one relaunch under this amendment.

**Prediction P15**, read off the box's own P14 line, with no new reducer code: the second host replicates P14's ordering. That needs
three things on two stable pairs:

1. the late-window ratio, axolotl scattermoe / e4b shipped over steps 101..200, lies in [0.90, 1.10];
2. its whole cross-draw interval lies below 1.0;
3. the box ran on a machine other than 96642.

P15 is FALSIFIED if the ratio leaves the band, or if the interval reaches or crosses 1.0. An unstable or missing pair, or a repeat of
the same machine, leaves P15 UNTESTED.

**Decision rules.**

- **HELD.** The `.native-steady-state` row adds the second host's reading, and its statement covers two hosts.
- **FALSIFIED.** The row, STATUS and the solution page say the steady-state ordering between axolotl's scattermoe and e4b as shipped
  is host-dependent, and give both hosts' readings. No single-host ordering is stated as general.

In every case, the two boxes' numbers are never divided into each other.

Budget: one RTX 5090, $0.69/h ceiling, 4.5 h cap, under $3.20 (`tc1-5090-35` cost $1.76); the standing no-ask tier.

### Amendment 10 (2026-10-03T08:42Z, before any box): e4b's grouping and index-copy syncs, A/B on one card (#945)

**Why.** TC1's profiled matched arm on an RTX 5090 was 48 % device-busy, and the profile spent 1.27 s of each 7.28 s profiled step in
`cudaStreamSynchronize` while the GPU still had work queued. The A2000 accounting in #945 traces every one of the step's ~4,400 syncs
to the MoE layer: 13 per layer pass. Five come from e4b's grouping in the forward; five forward and three backward come from
grouped-nf4-gemm's pageable index copies. #946 makes e4b's grouping read its counts back once, and grouped-nf4-gemm#438 adds an opt-in
pinned ring that removes the copies' syncs. Both are value-identical. Together they take a layer pass from 13 syncs to 1. Whether that
moves the 5090 training step is the question.

**The token** `qwen3syncab` (TC1_BOX=A): e4b against itself on one card, at the field recipe for N = 20.

- **Legacy side:** `E4B_GROUPING=legacy GNF4_PINNED_RING=0`.
- **New side:** `E4B_GROUPING=single GNF4_PINNED_RING=1`.
- **Arms:** the shipped arm (`fused_attn4_shipped_*`) and the matched arm (`fused_attn4_m_*`), two draws each, in ABBA order so neither
  side always runs first.
- **Pins:** e4b at a head carrying #946, and grouped-nf4-gemm at the merge of #438 rather than 846b512. Both sides run the same code; only
  the two variables differ.
- **Engagement record:** each arm records the path it ran (`sync_ab`: grouping mode, ring switch, ring staged count).
- **Validity:** the reducer voids an arm whose record contradicts its tag. A legacy arm must show legacy grouping, the ring off and
  nothing staged. A new arm must show single grouping, the ring on and staged > 0. The matched legacy arm is the box's anchor.

**Predictions** (registered before the box): over two VALID draws a side, each side's draws within 5 %, sync1 / legacy s/step (medians
of steps 11..20) lies in [0.75, 0.95], at least 5 % faster and at most 25 %. P16 is the shipped arm and P17 the matched arm. Outside
the band is FALSIFIED; an unstable, missing or non-VALID side is UNTESTED. The interval over the four cross-draw ratios is reported
beside each. The band comes from the profile: the sync wait was about 17 % of a profiled step, and an unprofiled step has less host
work, so the gain is expected to be smaller.

**Decision rules.**

- **Both ratios below 0.95 on stable pairs** (HELD, or FALSIFIED below 0.75): the ring becomes grouped-nf4-gemm's default outside
  capture in its next release, and #945 records the step gain.
- **Either ratio at or above 0.95:** the ring stays opt-in, and #945 records that removing the syncs did not move the 5090 step, so the
  sync wait was not that step's bottleneck.
- **Either way:** held-out losses at N are reported per side. They are expected to agree within the draws' own noise, because both
  changes are value-identical and the fused path's run-to-run nondeterminism (atomics) is the only difference.

Budget: one RTX 5090, $0.69/h ceiling, 2.5 h cap (eight e4b arms of about 5 minutes plus setup), under $1.75; the standing no-ask tier.

### Amendment 11 (2026-10-03T10:02Z, after amendment 10's read, before any box): the steady-state comparison again, with e4b's syncs removed

**Why.** P14 and P15 found axolotl's scattermoe native-best stepping at 0.911 and 0.901 of e4b as shipped over steps 101..200, on
two hosts. Amendment 10 then measured e4b's own fused step at 0.866 (shipped) of its previous form, from removing 12 of 13 host
syncs per MoE layer pass. That changes e4b, not axolotl, so the steady-state ordering may have changed too. Only a new box can
say: no number from one box is divided by another's.

**The box.** The `qwen3nativebest200` token, unchanged (amendment 8's arms, order and 200-step recipe), with:

- e4b at a head carrying #946, whose single-read grouping is the default;
- grouped-nf4-gemm at the merge of grouped-nf4-gemm#439, whose pinned ring is the default outside capture.

No environment variables are set: this measures e4b as a user installs it after both changes.

**Prediction P18** (registered before the box), read off the box's own P14 line with no new reducer code: the late-window ratio,
axolotl scattermoe / e4b shipped over steps 101..200, lies in [0.97, 1.15] on two stable pairs.

- **The reasoning.** P14 and P15 read about 0.90-0.91 against the old e4b, and amendment 10's 13 % gain moves that toward about
  1.05; host variation widens the band.
- **The ordering reading** is reported beside, as in amendment 8: an interval wholly above 1.0 means e4b shipped is faster at steady
  state; wholly below 1.0, axolotl; otherwise no ordering.
- **FALSIFIED** outside the band. **UNTESTED** on an unstable or missing pair.

**Decision rules.** The result is a new register row for the post-#945 code. `.native-steady-state` stays as measured, labelled as
the code before #945. If the interval lies above 1.0, STATUS says e4b as shipped now steps faster than axolotl's scattermoe at
steady state on that host. If it lies below 1.0, the steady-state finding stands for the new code too. If it spans 1.0, STATUS
says the two are at parity within the draw noise. Either way, e4b finishing a 200-step run first and the held-out comparison are
reported as before.

Budget: one RTX 5090, $0.69/h ceiling, 4.5 h cap, under $3.20; the standing no-ask tier.

### Amendment 12 (2026-10-03T10:34Z, before any box): where e4b's step goes after #945, profiled (P19)

**Why.** Amendment 10 removed 12 of 13 host syncs per MoE layer pass, and e4b's step got 13-15 % faster. TC1's only profile of e4b's
fused step (`tc1-5090-16`, matched arm) predates that change: 48 % device-busy, about 141,000 device events and 753,000 CPU ops per
step. The next change to e4b depends on what bounds the step now: kernel-launch volume, the one remaining sync, or the GPU itself.

**The token** `qwen3prof945` (TC1_BOX=A) runs TC1's profile instrument on three arms in this order: 3 warm and 3 profiled steps, with
`nvidia-smi dmon` beside each.

1. e4b shipped with the single-read grouping and pinned ring;
2. e4b matched with the same;
3. e4b matched with the legacy grouping and pageable copies, as the before-picture on the same host.

grouped-nf4-gemm is pinned at the merge of #439. Each arm's `sync_ab` record names its path. That record now carries the ring's
actual state, because #439 made it the default, plus the environment value beside it. The engagement predicate voids an arm whose
record contradicts its tag.

**Readings.**

- **Descriptive:** per arm, the device busy fraction, device events and CPU ops per step, and CPU self time by family.
- **Prediction P19:** the matched arm's device busy fraction on the new path is at least the legacy arm's plus 0.05. FALSIFIED below
  that; UNTESTED if either arm is not VALID or lacks a profile.

**Decision rule.** The next engineering target for #945 follows from the profile.

- **Device events and CPU ops per step still near the before-picture's, with the busy fraction rising but under 0.75:** the step is
  launch-bound, and the next work is launch volume (fusion in the attention and LoRA paths, then graph capture).
- **Busy fraction at or above 0.75:** the GPU is the bound, and the next work is the kernels.

Budget: one RTX 5090, $0.69/h ceiling, 1.5 h cap, under $1.05; the standing no-ask tier.

### Amendment 13 (2026-10-03T11:31Z, before any box): grouped-nf4-gemm's padded LoRA delta trimmed, A/B on one 5090 (P20, P21)

**Why.** Amendment 12's profile (`tc1-5090-41`) read P19 HELD. On the matched arm the device busy fraction rose from 0.595 to 0.740, while device events
(136,969 against 140,809) and CPU ops (750,409 against 753,087) per step stayed near the before-picture. Under that amendment's decision rule
this is the launch-bound branch, and the next work is launch volume in the attention and LoRA paths. The profile also named the first target:

- **Shipped arm** (busy 0.830): 94 ms of its 2.05 s of device time per step went to one scalar-times-tensor kernel, 1,152 calls. That kernel is
  grouped-nf4-gemm's `lora_delta_grouped` multiplying the padded `[G, widest, N]` expert delta by `scaling`, in the forward, the checkpoint
  recompute and the backward. At the field recipe (r16, alpha 16) the scaling is 1.0.
- **Matched arm:** its fp32 counterpart made 5,008 calls and took 228.7 ms. That is the shipped arm's 3,856 fp32 calls plus the same 1,152.

grouped-nf4-gemm#440 trims the padded delta with four changes, each exact:

1. a flat row index;
2. a gather whose backward is a plain scatter (`_GatherRows`; autograd's own is an atomic `index_add_`);
3. no zero fill or slice copy;
4. `scaling` on the gathered rows, skipped at exactly 1.

Forward and gradients are `torch.equal` to the previous body. On an RTX A2000 (a 2-layer Qwen3-MoE at this layer shape, ABBA), launches per step
went 1,424 → 1,344, device time 228.5 → 216.3 ms, and wall time 241.0 → 227.1 ms. `NF4_QLORA_LEAN_DELTA=0` restores the previous body.

**The token** `qwen3leanab` (TC1_BOX=A) runs e4b against itself on the shipped arm and the matched arm, two draws each in ABBA order:

- `*_lean0`: `NF4_QLORA_LEAN_DELTA=0`, the previous body;
- `*_lean1`: `NF4_QLORA_LEAN_DELTA=1`, the trimmed body.

Both sides run the post-#945 sync path (single-read grouping and the pinned ring). The LoRA path selector stays on `auto`, so the measurement
is the shipped behavior. grouped-nf4-gemm is pinned at the merge of #440, e4b at this amendment's merge. Each arm's `lean_ab` record names
three things: the body actually in force, the environment value, and the process's per-path delta call counts. The engagement predicate voids
an arm on any of these:

- its record contradicts its tag;
- the padded path never served it;
- its `sync_ab` record is not the post-#945 path.

**Predictions.**

- **P20** (shipped arm) and **P21** (matched arm): lean1 / lean0 s/step lies in **[0.90, 0.99]**. The figure is the median over two VALID draws
  a side, each side's draws within 5 %; the interval over the four cross-draw ratios is reported beside it. FALSIFIED outside the band;
  UNTESTED if a side is missing, not VALID or unstable.
- **Basis.** The removed multiply alone is 4.6 % of the shipped arm's device time. The rest of the trim removes index sorts, fills and copies.
  The A2000, which is device-bound, measured 0.943 of wall. On the 5090 the step is partly host-bound, so the launch cut counts as well as
  the device time.

**Decision rule.** The trimmed body is already gnf4's default, because it is exact; this box decides whether it stays.

- **Both HELD:** it stays, and a register row records the step-time effect.
- **A stable ratio in (0.99, 1.01] on an arm:** it stays, because it is exact and launches less; the register says no measurable step effect
  on that arm.
- **A stable ratio above 1.01 on either arm:** gnf4's default reverts (`NF4_QLORA_LEAN_DELTA` defaults to 0) until the cause is found.
- **Below 0.90:** FALSIFIED as over-predicted. The body stays, and the read names what else moved.

Budget: one RTX 5090, $0.69/h ceiling, 1.5 h cap, under $1.05; the standing no-ask tier.

### Amendment 14 (2026-10-03T12:14Z, before any box): grouped-nf4-gemm's prefill M-tile height from the group sizes, A/B on one 5090 (P22, P23)

**Why.** Amendment 12's profile (`tc1-5090-41`) puts the grouped GEMM forward at 811 ms of the shipped arm's 2.05 s of device time.
At this fixture's 1,521 real tokens a step, that is under a tenth of either of the card's rooflines. One cause is in its launch rule:
`gemm_4bit_grouped` uses one M-tile height for a whole launch, keyed on the largest group, so a single hot expert puts every group on
128-row tiles. Qwen3's real router does this here. On a 4-layer slice of the checkpoint, trained through e4b's fused step on these
token rows on an RTX A2000, the median group was 35 rows, while the largest group in a call had a median of 155 rows (90th percentile
377).

grouped-nf4-gemm#441 adds `GNF4_PREFILL_TILE_RULE=cost`, which picks the height that minimises `tiles x (96 + BLOCK_M)` over the actual
sizes. Outputs are bit-identical across rules. On the A2000:

- replaying 64 recorded calls took 711.5 ms under `max` and 545.5 ms under `cost`, against 538.5 ms at the best height;
- the 4-layer training step, in ABBA order, took 1.619 / 1.589 s under `max` and 1.484 / 1.493 s under `cost`.

The default stays `max` until this box reads.

**The token** `qwen3tileab` (TC1_BOX=A) runs e4b against itself on the shipped arm and the matched arm, two draws each in ABBA order:

- `*_tilemax`: `GNF4_PREFILL_TILE_RULE=max`;
- `*_tilecost`: `GNF4_PREFILL_TILE_RULE=cost`.

Both sides run the trimmed LoRA delta (#440) on the post-#945 sync path. grouped-nf4-gemm is pinned at the merge of #441, and e4b at this
amendment's merge. Each arm's `tile_ab` record names the rule in force, the environment values and the tile heights it launched
(`PREFILL_BM_STATS`). The engagement predicate voids an arm on any of these:

- its record contradicts its tag;
- it is a cost arm that launched no tile shorter than 128;
- it is not on the trimmed delta and the post-#945 sync path.

**Predictions.** Each figure is the median over two VALID draws a side, each side's draws within 5 %; the interval over the four cross-draw
ratios is reported beside it.

- **P22** (shipped arm): tilecost / tilemax s/step lies in **[0.85, 0.97]**.
- **P23** (matched arm): tilecost / tilemax s/step lies in **[0.88, 0.98]**.

Each is FALSIFIED outside its band, and UNTESTED if a side is missing, not VALID or unstable.

**Basis.** A 23 % cut of the 811 ms forward would be 0.92 of a fully device-bound 2.2 s shipped step. The matched step is longer
(3.1 s), and the forward is the same size within it, so its ratio sits higher. The A2000 measured 0.926 on its 4-layer step.

**Decision rule.**

- **Both stable ratios at or below 0.99:** gnf4's default becomes `cost` (D 96).
- **A stable ratio above 1.01 on either arm:** the default stays `max`.
- **Otherwise:** the rule stays opt-in, and the register says no measurable step effect on that arm.

The prediction verdicts are read beside the rule, not instead of it.

Budget: one RTX 5090, $0.69/h ceiling, 1.5 h cap, under $1.05; the standing no-ask tier.

### Amendment 15 (2026-10-03T12:43Z, before any box): e4b's fused training RMSNorm against the Hugging Face composite, A/B on one 5090 (P24, P25, P26)

**Why.** Amendment 12's profile (`tc1-5090-41`) is launch-bound on the matched arm. After the LoRA delta (#440, amendment 13), the
largest launch item left is the four Hugging Face RMSNorms per layer. Each is a composite of about 8 kernels forward and 10
backward, and gradient checkpointing runs the forward twice. Together that is roughly 20,000 of the shipped step's 134,000 device
events and 17 % of its CPU op time.

#961 adds `E4B_FUSED_RMSNORM=1`: a one-launch Triton forward and a dx-only backward for frozen norms, mirroring the composite's
casts. It is **not bit-identical**, because the row reductions run in another order: on an RTX A2000 about 1 element in 100,000
differs, by one bf16 ulp. On the A2000 (2-layer Qwen3-MoE, ABBA) launches per step went 1,344 → 1,109 and wall 227.6 → 216.4 ms.

**The token** `qwen3rmsab` (TC1_BOX=A) runs e4b against itself on the shipped arm and the matched arm, two draws each in ABBA order:

- `*_rms0`: `E4B_FUSED_RMSNORM=0`, the composite;
- `*_rms1`: `E4B_FUSED_RMSNORM=1`.

Both sides run the trimmed LoRA delta on the post-#945 sync path, with gnf4's default tile rule in force at the pinned gnf4. e4b is
pinned at a main commit carrying #961 and this amendment. Each arm's `rms_ab` record names the environment value and the patched and
called counts. The engagement predicate voids an rms1 arm that did not request, patch and call the fusion, and an rms0 arm that
patched anything.

**Predictions.** Speed figures are the median over two VALID draws a side, each side's draws within 5 %.

- **P24** (shipped arm): rms1 / rms0 s/step lies in **[0.85, 0.97]**.
- **P25** (matched arm): rms1 / rms0 s/step lies in **[0.88, 0.98]**.
- **P26** (quality): on each arm the two sides' mean held-out loss at N agrees within **0.01**. On `tc1-5090-42` the two draws of one
  configuration differed by up to 0.0047.

Each is FALSIFIED outside its band, and UNTESTED if a side is missing, not VALID or unstable.

**Decision rule.**

- **P26 HELD and both stable speed ratios at or below 0.99:** `E4B_FUSED_RMSNORM` becomes on by default.
- **P26 FALSIFIED:** the fusion stays opt-in whatever the speed, and the read names the held-out gap.
- **Otherwise:** it stays opt-in, and the register says no measurable step effect on that arm.

Budget: one RTX 5090, $0.69/h ceiling, 1.5 h cap, under $1.05; the standing no-ask tier.

### Amendment 16 (2026-10-03T13:50Z, before any box): the steady-state comparison again, with e4b after amendments 10–15 (P27)

**Why.** P14 and P15 found axolotl's scattermoe native-best stepping at 0.911 and 0.901 of e4b as shipped over steps 101..200, on
two hosts. Amendment 11 re-asked after #945, but its box (`tc1-5090-40`) read UNTESTED on an unstable host. Since then e4b's own
step has moved four more times, each measured against itself on one 5090:

- #945's host syncs: 0.866 (shipped);
- #440's trimmed LoRA delta: 0.939;
- #441/#442's tile rule: 0.924;
- #965's fused rotary: exact, fewer launches, unmeasured;
- #961's fused RMSNorm, per amendment 15's decision.

None of these changes axolotl. The ordering may have flipped, and only one box can say: no number from one box is divided by
another's.

**The box.** The `qwen3nativebest200` token, unchanged (amendment 8's arms, order and 200-step recipe). e4b is pinned at the main
commit that carries amendment 15's decision, and grouped-nf4-gemm at the main commit carrying #442. No environment variables are
set: this measures e4b as a user installs it then. The launcher's machine ranking (adertha-agents#137, deployed) orders the offers.

**Prediction P27** (registered before the box), read off the box's own P14 line with no new reducer code: the late-window ratio,
axolotl scattermoe / e4b shipped over steps 101..200, lies in **[1.05, 1.50]** on two stable pairs.

- **The reasoning.** About 0.905 against the old e4b, divided by the product of e4b's measured factors (0.866 × 0.939 × 0.924 ≈ 0.751,
  lower still if the RMSNorm fusion lands), gives about 1.2–1.3. Those factors are 11..20-step medians from other hosts, so the
  band is wide.
- **The ordering reading** is reported beside, as in amendment 8: an interval wholly above 1.0 means e4b shipped is faster at steady
  state; wholly below 1.0, axolotl; otherwise no ordering.
- **FALSIFIED** outside the band. **UNTESTED** on an unstable or missing pair.

**Decision rules.** A stable reading becomes a new register row, the successor of `.native-steady-state` for the post-amendment-15
code; the earlier row stays as measured, labelled as the code before #945.

- **Interval above 1.0:** STATUS says e4b as shipped now steps faster than axolotl's scattermoe at steady state on that host.
- **Interval below 1.0:** the old finding stands for the new code.
- **Interval spanning 1.0:** STATUS says parity within the draw noise.

The 200-step totals and the held-out comparison are reported as before. An UNTESTED box is re-run once, on another host.

Budget: one RTX 5090, $0.69/h ceiling, 4.5 h cap, under $3.20; the standing no-ask tier.

### Amendment 17 (2026-10-03T16:35Z, after amendment 16's read, before any box): P27's ordering on a second host (P28)

**Why.** Amendment 16's box (`tc1-5090-46`, Vast machine 142284, AMD EPYC 7663) read P27 HELD: axolotl scattermoe / e4b shipped over
steps 101..200 at **1.238** [1.231, 1.246], e4b faster at steady state. That reverses P14 and P15 for the new code, but on one host.
The campaign's rule is counterexamples before a broad claim, and the 5090 step is host-launch-bound (the same e4b arm has measured
3.2–5.9 s on different hosts). So the ordering is re-asked on a second host, as amendment 9 did for P14.

**The box.** Byte-identical to `tc1-5090-46`: the `qwen3nativebest200` token, e4b pinned at `e5859e9`, grouped-nf4-gemm at
`00929a4`, no environment set. The launcher's machine preference is switched off for this launch (`ADERTHA_PREFER_MACHINES=0`):
with it on, the offer order puts machine 142284 first again.

**Prediction P28** (registered before the box), read off the box's own P14 line: on a machine other than 142284, the late-window
ratio axolotl scattermoe / e4b shipped over steps 101..200 lies in **[1.05, 1.50]**, with its whole cross-draw interval above 1.0.
That replicates the ordering.

- A box on machine 142284 is a repeat: P28 UNTESTED.
- FALSIFIED outside the band, or with an interval that reaches 1.0.
- UNTESTED on an unstable or missing pair.

**Decision rules.**

- **HELD:** the amendment-16 register row records the replication on a second host, and STATUS drops "one host".
- **FALSIFIED:** the row and STATUS say the ordering is host-dependent and name both hosts.
- **UNTESTED:** the row stays one-host, and the box is not re-run inside this amendment.

Budget: one RTX 5090, $0.69/h ceiling, 4.5 h cap, under $3.20; the standing no-ask tier.

### Amendment 18 (2026-10-03T17:12Z, before any box): P27's ordering on a second host, with the machine excluded by evidence (P29)

**Why.** Amendment 17's box (`tc1-5090-47`) landed on Vast machine 142284 again, the same machine as `tc1-5090-46`. With the
launcher's machine preference off, that machine was still the cheapest RTX 5090 offer. By amendment 17's rule the box is a repeat
and P28 is UNTESTED, and the rule forbids re-running inside that amendment. adertha-agents#139 adds the missing mechanism: a launch
can cite a COMPLETED receipt whose machine it must not reuse (`--avoid-vast-machine-receipt`; manifest field
`avoid_vast_machine_receipts`). The exclusion and its reason are recorded in the new receipt.

**The box.** Byte-identical to `tc1-5090-46`: the `qwen3nativebest200` token, e4b pinned at `e5859e9`, grouped-nf4-gemm at
`00929a4`, no environment set. The manifest cites `receipts/experts4bit-qlora/2026-10-03/tc1-5090-46/receipt.json` in
`avoid_vast_machine_receipts`, so machine 142284 is out of the offer search. The launcher's adertha head carries #139.

**Prediction P29** (registered before the box), read off the box's own P14 line: the late-window ratio axolotl scattermoe / e4b
shipped over steps 101..200 lies in **[1.05, 1.50]**, with its whole cross-draw interval above 1.0. That replicates P27's ordering
on a second host.

- FALSIFIED outside the band, or with an interval that reaches 1.0.
- UNTESTED on an unstable or missing pair.
- UNTESTED if the receipt does not show machine 142284 excluded and a different machine bought.

**Decision rules.**

- **HELD:** the amendment-16 register row records the replication on a second host, and STATUS drops "one host".
- **FALSIFIED:** the row and STATUS say the ordering is host-dependent and name both hosts.
- **UNTESTED:** the row stays one-host.

`tc1-5090-47`'s same-host repeat is reported beside as a repeatability reading, never as a second host.

Budget: one RTX 5090, $0.69/h ceiling, 4.5 h cap, under $3.20; the standing no-ask tier.

### Amendment 19 (2026-10-03T22:52Z, before any box): the matched-work positions again, with e4b after amendments 10–15 (P30, P31, P32)

**Why.** TC1's headline matched-work positions were measured on e4b before #945:

- Unsloth (`grouped_mm`) / e4b **1.437** [1.434, 1.440] (`tc1-5090-16`), register `e4b.train.h2h.unsloth.qwen3.5090.2026-10-02`;
- axolotl / e4b **1.416** [1.398, 1.435] (amendment 4), register `e4b.train.h2h.axolotl.qwen3.5090.2026-10-02`.

Since then e4b's matched arm (fp32 adapters, matched init) has stepped faster four times, each measured against itself on one 5090:

- #945's host syncs: 0.847;
- #440's trimmed LoRA delta: 0.911;
- #442's tile rule: 0.968;
- #975's fused RMSNorm: 0.959.

The fused rotary (#965) is exact and unmeasured. Nothing on the other frameworks' side changed. The amendment-16 and -18 boxes
have already shown the native steady-state ordering against axolotl flipping. The matched positions are the campaign's headline,
and only new boxes can re-read them: no number from one box is divided by another's.

**The boxes.** The tokens are unchanged:

- box A runs `qwen3`, TC1's matched set: e4b fused, Unsloth `grouped_mm` and e4b reference, two draws where registered, then HF,
  axolotl and the profiled pair;
- box B runs `qwen3axolotl`, amendment 3's axolotl rows re-asked on their own box.

Both pin e4b at the main commit carrying this amendment, which has every default from amendments 10–15, and grouped-nf4-gemm at the
main commit carrying #442. No environment variables are set.

**Predictions** (registered before the boxes), read off each box's own lines with no new reducer code:

- **P30** (box A): the MATCHED POSITION unsloth/e4b lies in **[1.6, 2.8]**. The point estimate is 1.437 divided by the product of e4b's
  matched factors (≈ 0.716), about 2.0; host variation widens the band.
- **P31** (box A): the box's P3 line is HELD. The matched set (e4b reference, Unsloth) is EQUIVALENT to e4b fused, so the fused
  RMSNorm's near-exact numerics keep e4b inside the equivalence bands.
- **P32** (box B): the MATCHED POSITION axolotl/e4b lies in **[1.6, 2.8]** (1.416 / 0.716 ≈ 1.98).

Each is FALSIFIED outside its band, and UNTESTED where the box quotes no position: an unstable pair, a non-VALID arm or a missing
receipt.

**Decision rules.**

- **A HELD or FALSIFIED reading** becomes a new register row for the post-amendment-15 code, and STATUS quotes it as the current
  position. The 2026-10-02 rows stay active, labelled as the code before #945, as in amendments 11 and 16.
- **P31 FALSIFIED** blocks any position from box A being quoted until the equivalence gap is explained.

Budget: two RTX 5090 boxes, $0.69/h ceiling each, 4.5 h cap each, under $4.20 together; the standing no-ask tier, a single run
under $15.

### Amendment 20 (2026-10-04T02:36Z, before any box): grouped-nf4-gemm's per-pass host reuse, A/B on one 5090 (P33, P34)

**Why.** e4b's fused training step is host-bound on these boxes. In amendment 19's read, the same code's matched arm stepped at 2.147 s
on a Ryzen 9 9950X3D host and 3.973 s on an EPYC 7C13 host, both on an RTX 5090. A profile of the step on an RTX A2000 (4-layer slice, TC1's token rows, every current default) found a MoE
layer pass re-doing host work it had already done:

- the gate_up and down GEMMs each upload `expert_ids`;
- the two LoRA deltas each upload `(rows, ids)` and rebuild the same flat padded-row index (about ten launches);
- the backward's dgrad uploads `expert_ids` twice more: 8 transfers per layer forward + backward, 4 of them repeats;
- the adapters' advanced-index gather backward radix-sorts its indices (about ten launches per adapter).

grouped-nf4-gemm#444 adds `GNF4_HOST_REUSE=1`, which turns on three changes:

- a content-, device- and stream-keyed memo of those uploads, never used under capture;
- a one-entry memo of the delta's device plan;
- a scatter backward for the adapter gather when the ids are distinct.

The forward output and every adapter gradient are `torch.equal` with the flag off and on. Under deterministic mode the input gradient
is too; outside it, the input gradient already varies from run to run with the flag off.

On one `ExpertsLoRA` layer at Qwen3-30B-A3B's shape on the A2000 (three interleaved off/on pairs of 300 repetitions):

- the forward's host median went from 3.193 / 3.150 / 2.947 ms to 2.784 / 2.822 / 2.906 ms;
- the backward's went from 3.003 / 2.871 / 2.720 ms to 2.401 / 2.422 / 2.508 ms.

The default stays off until this box reads.

**The token** `qwen3reuseab` (TC1_BOX=A) runs e4b against itself on the shipped arm and the matched arm, two draws each in ABBA order:

- `*_reuse0`: `GNF4_HOST_REUSE=0`;
- `*_reuse1`: `GNF4_HOST_REUSE=1`.

Every other setting is the current default: the post-#945 sync path, the trimmed LoRA delta, the cost tile rule and the fused RMSNorm.
grouped-nf4-gemm is pinned at the merge of #444, and e4b at this amendment's merge. Each arm's `reuse_ab` record names the flag in
force, its environment value and the process's hit counts (`HOST_REUSE_STATS`). The engagement predicate voids an arm on any of these:

- its record contradicts its tag;
- it is a reuse1 arm whose upload memo or plan memo never hit;
- it is a reuse0 arm on which anything hit;
- it is not on the trimmed delta and the post-#945 sync path.

**Predictions.** Each figure is the median over two VALID draws a side, each side's draws within 5 %; the interval over the four cross-draw
ratios is reported beside it.

- **P33** (shipped arm): reuse1 / reuse0 s/step lies in **[0.92, 0.99]**.
- **P34** (matched arm): reuse1 / reuse0 s/step lies in **[0.93, 0.99]**.

Each is FALSIFIED outside its band, and UNTESTED if a side is missing, not VALID or unstable.

**Basis.**

- On the A2000's host, the layer bench saves about 0.26 ms per forward pass and 0.42 ms per backward pass. Over 48 layers, with each
  forward run twice under checkpointing, that is about 45 ms per micro-batch, or about 180 ms per 4-micro-batch step.
- Against a fully host-bound 2.1–4.0 s step, that is 0.92–0.96 (more saved on a slower host, but a longer step). A step that is partly device-bound passes on less, which is why the
  upper edge sits at 0.99.
- The matched step is longer and carries the same per-pass saving, so its band sits slightly higher.

**Decision rule.**

- **Both stable ratios at or below 0.99:** gnf4's default becomes on.
- **A stable ratio above 1.01 on either arm:** the default stays off.
- **Otherwise:** the flag stays opt-in, and the register says no measurable step effect on that arm.

The prediction verdicts are read beside the rule, not instead of it.

Budget: one RTX 5090, $0.69/h ceiling, 1.5 h cap, under $1.05; the standing no-ask tier.

### Amendment 21 (2026-10-04T03:11Z, before any box): keeping MoE activations instead of recomputing them, A/B on one 5090 (P35, P36, P37)

**Why.** Hugging Face checkpoints each decoder layer whole, so e4b's backward re-runs every MoE forward. On a 4-layer Qwen3-30B-A3B
slice on an RTX A2000 (these token rows, mb2 x accum 4, every current default), the recomputed MoE forwards were 232 of 917 ms of device
time, and 79 of 505 ms of the step's host compute.

e4b#1007 adds `E4B_MOE_KEEP_LAYERS=n`. In the last n decoder layers, only `self_attn` is checkpointed, and the MoE activations are kept
from forward to backward. grouped-nf4-gemm#445 adds `NF4_QLORA_COMPACT_DELTA=1`, which makes the padded LoRA delta save its input rather
than its padded block. Together they cut one MoE layer's saved activations from 229 MB to about 55 MB at 380 tokens with fp32
adapters. Every trainable gradient is `torch.equal` with the policy on and off (deterministic mode).

On the slice, keeping 4 of 4 layers took the step from 1.2545 / 1.2689 s to 0.9697 / 0.978 s, and 2 of 4 to 1.1349 / 1.1383 s. Each
kept layer costs about 113 MB of peak at the slice's largest micro-batch (1,036 padded tokens). This box's largest is 1,110, so about
121 MB per layer. The 5090 arms peak at 25.5 GB (shipped) and 27.2 GB (matched) today, so the kept counts are sized to the headroom:

- **32 layers on the shipped arm:** about +3.9 GB, to about 29.4 GB.
- **16 on the matched arm:** about +1.9 GB, to about 29.2 GB.

**The token** `qwen3keepab` (TC1_BOX=A) runs e4b against itself, two draws each in ABBA order:

- `*_keep0`: the defaults (whole-layer checkpointing);
- `fused_attn4_shipped_keep1`: `E4B_MOE_KEEP_LAYERS=32 NF4_QLORA_COMPACT_DELTA=1`;
- `fused_attn4_m_keep1`: `E4B_MOE_KEEP_LAYERS=16 NF4_QLORA_COMPACT_DELTA=1`.

Every other setting is the default at the pinned commits. grouped-nf4-gemm is pinned at the merge of #445, and e4b at this amendment's
merge (which carries #1007). Each arm's `keep_ab` record names the layers kept and the compact delta's state. The engagement predicate
voids an arm on any of these:

- it is a keep1 arm that did not keep exactly its arm's count with the compact delta on;
- it is a keep0 arm that kept any layer;
- it is not on the trimmed delta and the post-#945 sync path.

**Predictions.** Each figure is the median over two VALID draws a side, each side's draws within 5 %; the interval over the four cross-draw
ratios is reported beside it.

- **P35** (shipped arm, 32 of 48 kept): keep1 / keep0 s/step lies in **[0.80, 0.95]**.
- **P36** (matched arm, 16 of 48 kept): keep1 / keep0 s/step lies in **[0.86, 0.98]**.
- **P37:** on each arm, the keep1 side's median peak is **at most 31.0 GB**, and the two sides' mean held-out at N agree **within
  0.005**. Gradients are identical by construction; the draws differ only by the step's existing nondeterminism.

P35 and P36 are FALSIFIED outside their bands. P37 is FALSIFIED if either keep1 side OOMs or misses either bound. Each is UNTESTED if
a side is missing, not VALID or unstable.

**Basis.**

- Keeping every layer on the slice gave 0.773, and keeping half gave 0.902: about 0.227 per fraction of layers kept on a device-bound
  card.
- On the host-bound 5090 the avoided host work is the recompute's ~16 % of host compute, plus device time where the card is busy.
- Keeping 2/3 of the layers predicts about 0.85–0.90 on the shipped arm, and 1/3 about 0.92–0.95 on the matched arm. The bands widen
  both ways for the host mix.

**Decision rule.** This is a memory-for-time trade, so no default changes on this box. The register records each arm's ratio and peak
as the trade at that count. A HELD P37 with both ratios at or below 0.99 makes `E4B_MOE_KEEP_LAYERS` the recommended setting for cards
with that headroom (README and STATUS). Any other reading leaves it documented as opt-in, with the numbers.

Budget: one RTX 5090, $0.69/h GPU ceiling (the offer's price; 320 GB of disk is billed on top, as on every TC1 box), 1.5 h cap, under
$1.30 with disk; the standing no-ask tier.

### Amendment 22 (2026-10-04T16:15Z, before any box): grouped-nf4-gemm's dense route against its fused kernels, A/B on one RTX 5090, Qwen3-30B-A3B and Mixtral-8x7B (P38–P40)

**Why.** TC2 amendment 7's box D (`tc1-5090-55`) read e4b's reference loop on Mixtral-8x7B at 4.69 s/step against the fused kernels'
5.52 s on the same box. The reference loop dequantizes each expert and runs a dense GEMM. An RTX A2000 replay of the expert GEMMs then
showed grouped-nf4-gemm's fused 4-bit kernels far from efficient at large groups:

| shapes | forward, dense/fused | dgrad, dense/fused |
|---|---|---|
| Mixtral, 1,024 rows per expert | 0.29–0.30 | 0.13–0.14 |
| Qwen3-30B-A3B, 256 rows per expert | 0.57–0.58 | 0.22 |

The fused forward was ahead only at Qwen3's down projection with 64 rows. grouped-nf4-gemm#459 adds `GNF4_TRAIN_GEMM=dense`, an
opt-in route that dequantizes one present expert at a time and multiplies with `torch.mm` on any CUDA card. This amendment reads
it on the full training step.

**The box.** One RTX 5090 with two A/B families, each a matched arm (fp32 adapters, matched init) in two draws a side, in ABBA order.
Side 0 sets `GNF4_TRAIN_GEMM=fused`; side 1 sets `GNF4_TRAIN_GEMM=dense`. Everything else is the default.

- `qwen3denseab`: Qwen3-30B-A3B, TC1's qwen3 token's tokens and recipe.
- `mixtraldenseab`: Mixtral-8x7B-Instruct at TC2's field recipe, resident, with `E4B_ABSMAX_DQ=1` on both sides. That leaves room
  for the dense route's one-expert transient (235 MB at gate_up); box D read 29.03 GB with it.

grouped-nf4-gemm is pinned at #459's merge. Engagement is read off each arm's `route_ab` record: side 1 counts `dense_fwd` and
`dense_dgrad` calls, side 0 counts none.

**Predictions** (registered before the box):

- **P38** (Mixtral): dense/fused s/step lies in **[0.55, 0.90]**, both sides stable. The expert GEMMs dominate this step, and the
  reference loop already reads 0.85.
- **P39** (Qwen3-30B-A3B): dense/fused s/step lies in **[0.85, 1.15]**. This is a wide band on purpose: the device time falls, but
  the loop adds one launch per present expert per projection (up to 128) to a 5090 step that is launch-bound.
- **P40:** on each family, |mean held-out at N, dense − fused| ≤ **0.01**. Both routes share the dequant bytes and differ only in the
  GEMM's accumulation order.

Each is FALSIFIED outside its band, and UNTESTED where a side is unstable or not engaged.

**Decision rules.**

- **P38 and P40 HELD:** grouped-nf4-gemm's `auto` takes the dense route on cards other than sm_90 for calls with at most 16 present
  groups (Mixtral's case, not Qwen3-30B-A3B's). That is grouped-nf4-gemm's own PR, citing this box.
- **P39 HELD with the ratio at or below 0.95:** `auto` takes dense for every group count on non-sm_90 cards.
- **Otherwise:** fused stays for that family.
- No position against another framework is read here. A position follows in its own box once `auto` changes.

**Budget.** One RTX 5090 at the policy rate ($0.85/h), 3 h guard, a 192 GB host floor (Mixtral's checkpoint). About $2.50 with the
download; the standing no-ask tier; the campaign's daily cap is $100, counted 9 am to 9 am Central.

### Amendment 23 (2026-10-04T18:56Z, before any box): where e4b's resident training memory goes, against Unsloth's — a memory census on one RTX 5090 (P41–P43)

**Why.** Every resident matched pair this week put e4b's peak above Unsloth's: Qwen3-30B-A3B 27.21 against 24.27 GB (5090), Qwen3.6-35B-A3B
over 32 GB against 30.47 GB, OLMoE 7.67 against 5.97 GB, and Mixtral 31.07 against 29.12 GB before the absmax was double-quantized. The
fp32 expert absmax explains part of it (`E4B_ABSMAX_DQ=1` brought Mixtral level), and on Qwen3.6 the bf16 non-routed projections explain
another part. The rest is unattributed, and it is what stands between e4b and a resident Qwen3-30B-A3B on a 24 GB card (e4b's micro-batch-1
peak 26.06 GB) and Qwen3.6 at micro-batch 2 on 32 GB.

**The box.** One RTX 5090, token `qwen3memcensus`, Qwen3-30B-A3B, the matched set at micro-batch 1 × accum 8 (the same tokens per step), one
draw per arm, with the harness's memory census on (`--mem-census 1`):

- e4b `fused_attn4_m_mb1` (defaults: fp32 absmax);
- e4b `fused_attn4_m_mb1_dq` (`E4B_ABSMAX_DQ=1`);
- Unsloth `ckpt_unsloth_m_mb1`.

The census records PyTorch's allocator history over the run. At the moment of peak allocated memory it reports:

- the live allocations, grouped by the first frame in e4b, grouped-nf4-gemm, Unsloth, bitsandbytes or the optimizer;
- the static bytes by class: frozen expert weights, absmax, other frozen weights, adapters, gradients, optimizer state.

No speed is read: the census slows the step. Positions stay with the boxes that read them.

**Predictions** (registered before the box):

- **P41** (the instrument): the census finds e4b's fp32 expert absmax within 2 % of the analytic 1.81 GB (29.0 B expert parameters /
  64 × 4 bytes), and the `_dq` arm's within 2 % of that over 3.94 (#1040's measured ratio).
- **P42:** the census attributes at least 90 % of e4b's peak and of Unsloth's to named groups (sites or static classes); anything else is
  reported as unattributed.
- **P43:** after the absmax, e4b's excess over Unsloth at micro-batch 1 lies in **[0.5, 2.5] GB**, and the read names its largest group.

**Decision rules.** This is a measurement, not a position. The read names the excess's largest groups, and each fix that follows is its own
registration with its own A/B.

**Budget.** One RTX 5090 at the policy rate, 2 h guard, about $1 with the download; the standing no-ask tier; the campaign's daily cap is
$100, counted 9 am to 9 am Central.
### Amendment 24 (2026-10-04T19:31Z, before any box): the RTX 5090's fp32 `bmm` host cost and the torch 2.12 environment, on one RTX 5090 (P44–P49)

**Why.** The TC1 profile instrument puts the largest single e4b host row on both 5090 profiles in `aten::bmm`, the padded LoRA delta's
two batched products and their backward (grouped-nf4-gemm `kernel/nf4_qlora.py:_lora_delta_padded`), on the matched arm's fp32 adapters:

| profile | arm | host self per `bmm` call | `bmm` host per step | device per call |
|---|---|---|---|---|
| `tc1-5090-49` (0.42.0) | matched, fp32 | 324 µs | 998 ms | 118 µs |
| `tc1-5090-41` | matched, fp32 | 197 µs | 607 ms | 122 µs |
| `tc1-5090-41` | shipped, bf16 | 15 µs | 47 ms | — |
| `tc1c-h100-15` (0.45.0) | matched, fp32 | 30 µs | 93 ms | 102 µs |

The same 5090 profiles read the launch APIs at 2.5–6 µs a call, so the time is not a blocked launch: it sits in the `bmm` call's own
host code. On `tc1-5090-41` the matched arm's step is 1,085 ms longer than the shipped arm's, and `bmm` alone accounts for 560 ms of
that. An RTX A2000 (sm_86, torch 2.8.0+cu128) replay of the same shapes reads 70–100 µs per fp32 call against 45–60 µs for bf16, with
no gap between new and repeated shapes, so the effect is not general.

TC1's native box (`tc1-5090-14`, 2026-10-02) also ran e4b's matched arm in two environments on one card: 9.336 s/step on the field
image's torch 2.8.0+cu128, and 8.069 s/step in the torch 2.12.1+cu130 venv, 0.864×. That reading was one draw each, on e4b 0.38.0,
and was never attributed. Every 5090 position of record runs e4b on torch 2.8.0+cu128 and Unsloth on torch 2.12.1+cu130.

**Hypothesis.** cuBLAS 12.8 on sm_120 spends a per-shape host search on fp32 batched GEMMs. The padded delta's shape (present groups,
widest group) changes with the routing on every call, so the search is paid on nearly every call. bf16 and the H100 take paths whose
search is cheap or cached.

**The box** (token `qwen3bmmab`). One RTX 5090.

1. **Replay, no model.** `bench/tc1/bmm_bench.py` replays the padded delta's two `bmm` and their backward at the shapes of the 96
   forward calls in `routecalls-qwen3.json` (r 16). It runs once under each environment: venv-e4b (torch 2.8.0+cu128) and venv-unsloth
   (torch 2.12.1+cu130). For each of fp32, bf16, fp32 with `preferred_blas_library("cublaslt")`, and fp32 with TF32, in a fresh process,
   it reads the host time of each forward `bmm` with the queue drained first, so a launch cannot block. The shapes are taken:
   - as recorded (first exposure),
   - again (repeats),
   - as new shapes the process has not used (cold),
   - at one fixed shape,
   - with the widest group rounded up to a multiple of 32.

   The cublasLt and TF32 rows are descriptive. TF32 changes the numerics, so it is not a candidate for the matched arm.
2. **Training A/B** on TC1's qwen3 token's tokens and recipe, ABBA order, two draws a side:
   - the matched arm (fp32 adapters, matched init): `fused_attn4_m_tv0` (venv-e4b) vs `fused_attn4_m_tv1` (venv-unsloth with e4b and
     grouped-nf4-gemm at the box's pins, as TC1's `t212` row);
   - the shipped arm (native bf16 adapters): `fused_attn4_shipped_tv0` vs `fused_attn4_shipped_tv1`.

   The two venvs also differ in transformers (5.18.0 vs 5.5.0) and triton (3.4.0 vs 3.7.1), so this half reads the environment as a
   whole. The replay is what isolates cuBLAS. Engagement: each arm's receipt records the torch its tag names (`env.torch` 2.8.* for
   `_tv0`, 2.12.* for `_tv1`) and the adapter dtype its arm names.

**Predictions** (registered before the box):

- **P44** (the anomaly outside training): under torch 2.8.0+cu128, fp32's median host time per forward `bmm` at the recorded shapes
  is ≥ **100 µs** and ≥ **3×** bf16's.
- **P45** (per shape): under torch 2.8.0+cu128, fp32's median on cold shapes is ≥ **2×** its median on repeated shapes.
- **P46** (the environment): torch 2.12.1+cu130's fp32 cold median is ≤ **0.5×** torch 2.8.0+cu128's.
- **P47** (the matched arm): `_tv1` / `_tv0` s/step lies in **[0.70, 0.95]**, both sides stable (two draws within 5 %).
- **P48** (where the gain lands): the shipped arm's `_tv1` / `_tv0` ratio is at least the matched arm's ratio + **0.05**.
- **P49:** on each arm, |mean held-out at N, `_tv1` − `_tv0`| ≤ **0.01**.

P44–P46 are FALSIFIED outside their bounds and UNTESTED where the replay did not run in that environment. P47–P49 are FALSIFIED
outside their bounds, and UNTESTED where a side is unstable, not VALID or not engaged.

**Decision rules.**

- **P44 FALSIFIED:** the profile's `bmm` time is a load effect, not a property of the call. No grouped-nf4-gemm change follows from this box.
- **P44 and P45 HELD:** grouped-nf4-gemm gets an opt-in that rounds the padded delta's widest group up to a bucket, so the shapes
  recur. It is read in its own A/B box (the replay's bucket rows size it). Padding rows are zero and sliced away, but a different
  shape can select a different kernel, so the change is reorder-class, not bit-identical.
- **P46 HELD:** the docs say that fp32 adapters on an sm_120 card want torch ≥ 2.12 (cu130).
- **P47 and P48 HELD:** the 5090 positions of record carry an environment asymmetry in Unsloth's favour. STATUS names it with this
  box's ratio. A position with both frameworks on torch 2.12.1+cu130 follows in its own box. No position against another framework is
  read here.

**Budget.** One RTX 5090 at the policy rate ($0.85/h), 3 h guard, TC1's 98 GB host floor. Qwen3-30B-A3B's download plus eight arms
and the replay come to about $1.50; this is in the standing no-ask tier.

### Amendment 25 (2026-10-04T21:48Z, after amendment 24's read, before any box): the matched position with both frameworks on one stack, on one RTX 5090 (P50–P52)

**Why.** Every 5090 position on Qwen3-30B-A3B has run e4b and Unsloth in different environments:

- e4b in the field image's: torch 2.8.0+cu128, transformers 5.18.0, triton 3.4.0;
- Unsloth in its own venv: torch 2.12.1+cu130, transformers 5.5.0, triton 3.7.1, because its grouped_mm path needs it.

Amendment 24 (`tc1-5090-66`) read e4b's matched arm at **0.882×** its field-image step in Unsloth's venv (P47 HELD). Its replay showed the
cause is not the padded LoRA delta's `bmm`. e4b requires only `torch>=2.2` and `transformers>=5.0`, so Unsloth's stack is an e4b
environment too. The comparison with the fewest confounds runs both frameworks on that one stack.

**The box** (token `qwen3samestack`). One RTX 5090, TC1's qwen3 tokens and field recipe, the matched set (fp32 adapters, matched init).
In this order:

1. e4b `fused_attn4_m` in venv-unsloth (e4b and grouped-nf4-gemm installed there at the box's pins, as TC1's `t212` row);
2. Unsloth `ckpt_unsloth_m` (grouped_mm, the notebooks' seven targets, as TC1);
3. e4b `reference_attn4_m` in venv-unsloth;
4. e4b `fused_attn4_m_t28` in venv-e4b, then its second draw;
5. Unsloth's second draw;
6. e4b `fused_attn4_m`'s second draw.

HF, axolotl and the profiled arms are not run. Engagement: each e4b receipt records the torch its tag names (`env.torch` 2.12.* for
`fused_attn4_m`, `fused_attn4_m_d2` and `reference_attn4_m`; 2.8.* for `_t28`).

**Predictions** (registered before the box):

- **P50:** with both frameworks on torch 2.12.1+cu130 / transformers 5.5.0, Unsloth/e4b lies in **[1.9, 2.9]**, both pairs stable
  (two draws within 5 %). The basis: amendment 19's 1.997 with e4b on the field image, divided by P47's 0.882, is about 2.26. Hosts
  move both steps.
- **P51:** on this second host, e4b's matched arm in venv-unsloth over venv-e4b lies in **[0.80, 0.95]**, both sides stable. This
  replicates P47.
- **P52:** on one stack the matched set holds. e4b's reference and Unsloth each read EQUIVALENT or INSIDE-DRAW-NOISE against e4b's
  fused arm, and e4b's parity PASSES.

Each is FALSIFIED outside its band, and UNTESTED where a side is missing, unstable or not engaged.

**Decision rules.**

- **P50 and P52 HELD:** this box's ratio becomes the Qwen3-30B-A3B 5090 position to quote (`….qwen3.5090.<date>.same-stack`).
  Amendment 19's 1.997 stays as the reading with e4b in the field image's environment, and STATUS names both environments.
- **P50 FALSIFIED:** the ratio is recorded as a labelled row; the quoted position does not change.
- **P51 HELD:** the environment gain replicates across hosts. Splitting it between torch, transformers and triton is its own registration.

**Budget.** One RTX 5090 at the policy rate ($0.85/h), 3 h guard, TC1's 98 GB host floor. Qwen3-30B-A3B's download plus seven arms come to
about $1.50; this is in the standing no-ask tier.

### Amendment 26 (2026-10-04T22:17Z, before any box): prebound Triton launches, A/B on one RTX 5090 (P53–P55)

**Why.** The H100 profile (`tc1c-h100-15`) put e4b's step at 2.8 s of host time against 1.5 s of device time. Each launch of the
fused RMSNorm and rotary kernels, and of grouped-nf4-gemm's training GEMMs, spends most of its host time in Triton's per-call binding
and in host work that repeats for one grouping.

experts4bit-qlora#1078 (`E4B_TRITON_PREBIND=1`) and grouped-nf4-gemm#468 (`GNF4_TRITON_PREBIND=1`) are opt-in, and both cover Triton
3.4 and 3.6. Each specialization's first launch goes through Triton; later launches call the same compiled kernel's launcher
directly. grouped-nf4-gemm also reuses one grouping's upload, plan and M-tile by value. Outputs are bit-identical (`torch.equal`, and
the same compiled-kernel object). On an RTX A2000 host, the host µs per call fell, flag off → on:

- RMSNorm forward: 152–160 → 117–121;
- the fused GEMM forward: 633–654 → 433–452;
- its dgrad: 448 → 318.

That arithmetic gives an estimated 130–200 ms per step on a 5090 host. It is an estimate; this box measures it.

**The box** (token `qwen3prebindab`). One RTX 5090, venv-e4b (torch 2.8.0+cu128, triton 3.4, a version the prebound path covers),
TC1's qwen3 tokens and field recipe. The shipped and the matched arm, each `_pb0` (both flags 0) against `_pb1` (both flags 1), two
draws a side in ABBA order. Every other setting is the default.

Engagement is read off each receipt's `prebind_ab` record:

- a `_pb1` arm requested both flags, and each side counted prebound launches;
- a `_pb0` arm requested neither and counted none.

**Predictions** (registered before the box):

- **P53** (shipped): `_pb1` / `_pb0` s/step lies in **[0.90, 0.98]**, both sides stable (two draws within 5 %).
- **P54** (matched): `_pb1` / `_pb0` lies in **[0.92, 0.99]**. The fp32-adapter step is longer, so the same saving is a smaller share.
- **P55:** on each arm, |mean held-out at N, `_pb1` − `_pb0`| ≤ **0.005**. The compiled kernels are the same, so only run-to-run
  nondeterminism separates the sides.

Each is FALSIFIED outside its band, and UNTESTED where a side is unstable, not VALID or not engaged.

**Decision rules.**

- **P53, P54 and P55 HELD:** both flags default on, for the Triton versions the prebound path covers; other versions keep Triton's own
  launch. That is one PR in each repository, citing this box.
- **Either ratio above 1.01:** both stay off.
- **Otherwise:** they stay opt-in.
- No position against another framework is read here.

**Budget.** One RTX 5090 at the policy rate ($0.85/h), 3 h guard, TC1's 98 GB host floor; no Unsloth venvs are built. Qwen3-30B-A3B's
download plus eight arms come to about $1.30; this is in the standing no-ask tier.

### Amendment 27 (2026-10-04T22:48Z, after amendment 25's first box): one re-draw of the same-stack box, on another machine

Amendment 25's box (`tc1-5090-67`, Intel Core Ultra 9 285K) read:

- **P52 HELD:** Unsloth and e4b's reference are EQUIVALENT to e4b's fused arm on one stack, and parity PASSES.
- **P50 UNTESTED:** Unsloth's two draws were 18.2 % apart (4.913 / 5.899 s/step).
- **P51 UNTESTED:** e4b's field-image draws were 8.4 % apart (2.305 / 2.507).

e4b's same-stack draws were stable (2.220 / 2.188). The SM clock was steady at 2,437–2,445 MHz on e4b's arms and 2,865 MHz on Unsloth's,
and the host load average stayed near 1.2, so the logs do not show the cause.

**This amendment** re-draws the same token (`qwen3samestack`) once on a different machine. The launcher avoids box 1's machine. The
predictions, bands and decision rules are amendment 25's, unchanged, and the re-draw is read on its own, not pooled with box 1.

**Its reading is final.** If a side is unstable again, P50 and P51 stay UNTESTED, and no further re-draw is made under amendment 25.
Box 1's descriptive medians, Unsloth/e4b about 2.45 and the environment about 0.92, are not readings and are not quoted.

**Budget.** One RTX 5090 at the policy rate, 3 h guard, about $1; this is in the standing no-ask tier.

### Amendment 28 (2026-10-05T00:05Z, after amendment 23's read, before any box): the double-quantized expert absmax as a default, A/B on Qwen3-30B-A3B and Mixtral-8x7B (P56–P58)

**Why.** The memory census (amendment 23, `tc1-5090-65`) found every static class the same in e4b and Unsloth except the expert absmax.
e4b keeps it in fp32 by default: 1.812 GB on Qwen3-30B-A3B. `E4B_ABSMAX_DQ=1` (#1040) stores it in 0.460 GB, as Unsloth does, and
brings e4b's peak to 0.43 GB above Unsloth's.

The switch stayed opt-in after TC2 amendment 7 only because that box's Mixtral position fell outside P18's band. Its own cost was never
read on its own: no A/B of e4b with the switch on against off on one box has run.

**The boxes.** Two RTX 5090s, e4b against itself on the matched arm (fp32 adapters, matched init), resident. `_dq0` sets
`E4B_ABSMAX_DQ=0` and `_dq1` sets `E4B_ABSMAX_DQ=1`, two draws a side in ABBA order. Every other setting is the default.

- `qwen3dqab`: Qwen3-30B-A3B, TC1's tokens and field recipe.
- `mixtraldqab`: Mixtral-8x7B-Instruct at TC2's pin and field recipe.

Engagement: each receipt records the absmax its tag names. A `_dq1` arm has `absmax_dq` true on every MoE layer, and a `_dq0` arm false.

**Predictions** (registered before the boxes):

- **P56** (Qwen3-30B-A3B): `_dq1` / `_dq0` s/step lies in **[0.97, 1.03]** with both sides stable, and the peak falls by
  **[1.25, 1.45] GB** (the census: 1.35).
- **P57** (Mixtral-8x7B): `_dq1` / `_dq0` lies in **[0.97, 1.03]**, both sides stable, and the peak falls by **[1.9, 2.3] GB**.
  The arithmetic is 45.1 B expert parameters / 64 × 4 bytes × (1 − 1/3.94), about 2.10 GB; across boxes, TC2 amendments 6 and 7 read 2.04.
- **P58:** on each family, |mean held-out at N, `_dq1` − `_dq0`| ≤ **0.005**. The double-quantized absmax is not bit-identical, and TC2
  amendment 7 read 0.0017 on Mixtral.

Each is FALSIFIED outside its band, and UNTESTED where a side is unstable, not VALID or not engaged.

**Decision rules.**

- **P56, P57 and P58 HELD:** `E4B_ABSMAX_DQ` defaults on for the resident fused path. Engines that refuse it keep the fp32 absmax. That
  is e4b's own PR, citing these boxes.
- **Either ratio above 1.03:** the switch stays opt-in, and its cost is quoted.
- **Otherwise:** it stays opt-in.
- No position against another framework is read here.

**Budget.** Two RTX 5090s at the policy rate ($0.85/h), 3 h guards. TC1's 98 GB host floor for Qwen3 and 192 GB for Mixtral
(TC2's), about $1.30 and $2.50 with the downloads; this is in the standing no-ask tier.

### Amendment 29 (2026-10-05T00:15Z, after amendment 27's re-draw): the same-stack speed pair over 60 steps (P50, P51 re-asked)

**Why.** Amendment 25's box and its one registered re-draw each lost one speed pair to instability, a different pair each time:

| box | host | unstable pair |
|---|---|---|
| `tc1-5090-67` | Core Ultra 9 285K | Unsloth 18.2 % apart; e4b's field-image arm 8.4 % |
| `tc1-5090-68` | EPYC 7663 | e4b's same-stack arm 7.8 % |

The matched set held on both boxes (P52 HELD twice). By amendment 27's rule P50 and P51 stay UNTESTED under amendment 25. TC1's 20-step
runs take their median over 10 steps (11..20), so a host hiccup of a few steps moves a draw by more than 5 %. The window is what failed;
no prediction was read.

**This amendment** re-asks P50 and P51 with their bands unchanged ([1.9, 2.9] and [0.80, 0.95]), on a longer window:

- `TC1_STEPS=60`, so each arm's median is over steps 11..60 (50 steps);
- everything else as amendment 25's box, except that e4b's reference arm is not run. P52 held on both boxes, and the reference loop at 60
  steps would take about 30 minutes.

One box, on a machine neither earlier box used. Its reading is final, and no re-draw follows under this amendment. If P50 and P52 have
both HELD (P52 on amendment 25's boxes), amendment 25's decision rule applies to this box's ratio.

**Budget.** One RTX 5090 at the policy rate ($0.85/h), 3 h guard, TC1's 98 GB host floor; about $1.80 with the download. This is in the
standing no-ask tier.

### Amendment 30 (2026-10-05T00:52Z, after amendment 26's read): the shipped arm's prebind pair over 60 steps (P53, P55 re-asked)

**Why.** Amendment 26's box (`tc1-5090-69`, Intel Core Ultra 9 285K, Vast machine 151350) read P54 HELD: the matched arm stepped 0.973×
with the prebound launches. P53 went UNTESTED because the shipped arm's flags-off draws were 8.3 % apart (1.926 / 1.772 s/step). The
shipped arm's flags-on draws were stable (1.972 / 1.970) and slower than both, so the unread pair points against P53. The 20-step window
takes its median over 10 steps. Two of the last three boxes on that machine lost a pair to instability.

**This amendment** re-asks P53 (shipped `_pb1` / `_pb0` in **[0.90, 0.98]**) and P55's shipped half (|mean held-out at N, `_pb1` −
`_pb0`| ≤ **0.005**), bands unchanged, on a longer window:

- `TC1_STEPS=60`, so each median covers steps 11..60;
- the shipped arm only, `_pb0` / `_pb1`, two draws a side in ABBA order, venv-e4b, every other setting as amendment 26's box;
- on a machine other than 151350.

**Its reading is final.** The decision rule is amendment 26's: P53, P54 (HELD on amendment 26's box) and P55 HELD turn both flags on by
default for the Triton versions they cover. Otherwise they stay opt-in, and a FALSIFIED P53 is quoted as the prebound launches' cost on
the shipped arm.

**Budget.** One RTX 5090 at the policy rate ($0.85/h), 3 h guard, TC1's 98 GB host floor; about $1.20 with the download. This is in the
standing no-ask tier.

### Amendment 31 (2026-10-05T01:00Z, after amendment 28's read): Qwen3-30B-A3B's absmax pair over 60 steps (P56, P58 re-asked)

**Why.** Amendment 28 read P57 HELD on Mixtral (`tc1-5090-71`): the double-quantized absmax costs 2.3 % of the step for 2.04 GB.
Qwen3-30B-A3B's box (`tc1-5090-70`, Vast machine 45511) lost both speed pairs to instability (5.6 % and 23.8 % apart), and its step times
followed the host's load average (6.6 to 19.4). Its peak drop (1.33 GB) and held-out move (+0.0026) were inside their bands, but P56 is
one prediction and stays UNTESTED.

**This amendment** re-asks P56 (`_dq1` / `_dq0` in **[0.97, 1.03]** and the peak lower by **[1.25, 1.45] GB**) and P58's Qwen3 half
(|Δ held-out| ≤ **0.005**), bands unchanged:

- `TC1_STEPS=60`, so each median covers steps 11..60;
- the token `qwen3dqab` otherwise as amendment 28's box;
- on a machine other than 45511, 138786 (amendment 27's, load 30–37) and 151350.

**Its reading is final.** The decision rule is amendment 28's: P56, P57 (HELD) and P58 HELD make `E4B_ABSMAX_DQ` the default for the
resident fused path. Otherwise it stays opt-in.

**Budget.** One RTX 5090 at the policy rate ($0.85/h), 3 h guard, TC1's 98 GB host floor; about $1.50. This is in the standing no-ask tier.

### Amendment 32 (2026-10-05T02:08Z, after an unregistered RTX A2000 decomposition): one variable, Triton 3.4 against 3.7.1, on one RTX 5090 (P59–P61)

**Why.** Amendment 24 read e4b's matched arm at **0.882×** in Unsloth's venv (torch 2.12.1, transformers 5.5.0, triton 3.7.1) against
the field image's (torch 2.8.0, transformers 5.18.0, triton 3.4.0). Its replay ruled out the padded LoRA delta's `bmm`.

An unregistered diagnostic on the owned RTX A2000 then split the three. It used a 4-layer real-router slice of Qwen3-30B-A3B, TC1's
tokens and recipe, `tc1_arm.py` unchanged, and three to six interleaved draws per environment:

- torch 2.8 with triton 3.7.1 forced in reads 0.931× torch 2.8 with triton 3.4;
- torch 2.12 at triton 3.7.1 adds nothing (1.011);
- transformers 5.5 against 5.18 adds nothing (1.005).

The device profile puts the gain in grouped-nf4-gemm's kernels: `_dgrad_nf4_grouped` about −40 % and `_gemm_nf4_grouped` about −15 % of
device time under triton 3.7. The A2000 runs that slice device-bound and the 5090 step is host-bound, so the share may not transfer. This
box reads the one variable on the 5090.

**The box** (token `qwen3tritonab`). One RTX 5090, venv-e4b (torch 2.8.0+cu128), TC1's qwen3 tokens and field recipe, `TC1_STEPS=60`
(50-step medians). Each of the matched and the shipped arm runs `_tr0` (venv-e4b's own triton 3.4) against `_tr1` (triton 3.7.1
installed alone into the box's work dir and put first on the arm's `PYTHONPATH`), two draws a side in ABBA order. The prebound launches
are off on both sides (`E4B_TRITON_PREBIND=0 GNF4_TRITON_PREBIND=0`): they cover triton 3.4 / 3.6, and on `_tr0` alone they would be a
second variable. The box avoids machines 45511, 138786 and 151350.

Engagement: each receipt's `env.triton` reads 3.4.* on `_tr0` and 3.7.* on `_tr1`, and its `prebind_ab` shows both flags off. A failed
triton 3.7.1 install leaves every `_tr1` arm an `install_failed` row.

**Predictions** (registered before the box):

- **P59** (matched): `_tr1` / `_tr0` s/step lies in **[0.82, 0.95]**, both sides stable. Most of amendment 24's 0.882 is Triton.
- **P60** (shipped): `_tr1` / `_tr0` lies in **[0.85, 0.98]**. The bf16-adapter step is shorter, so the same kernel gain is a different
  share.
- **P61:** on each arm, |mean held-out at N, `_tr1` − `_tr0`| ≤ **0.005**. On the A2000 the step-0 held-out was identical to 5 decimals
  across the two Tritons.

Each is FALSIFIED outside its band, and UNTESTED where a side is unstable, not VALID or not engaged.

**Decision rules.**

- **P59 and P61 HELD:** the 5090's environment gain is Triton's.
  - e4b's docs say that triton ≥ 3.7 (torch ≥ 2.12, or the override this box uses) runs grouped-nf4-gemm's training kernels faster.
  - grouped-nf4-gemm investigates why triton 3.4's code for those kernels is slower: its own registration.
  - The prebound launches gain triton 3.7 coverage, which is grouped-nf4-gemm's and e4b's own PRs.
- **P59 FALSIFIED high (above 0.95):** the A2000 decomposition does not transfer to the host-bound 5090 step, and the 0.882 stays
  unattributed between torch and transformers.
- No position against another framework is read here.

**Budget.** One RTX 5090 at the policy rate ($0.85/h), 3 h guard, TC1's 98 GB host floor; no Unsloth venvs are built. About $1.50 with
the download; this is in the standing no-ask tier.

### Amendment 33 (2026-10-05T02:19Z, after amendment 29's box): load-gated draws, and the same-stack pair re-asked under them (P50, P51)

**Why.** The same-stack speed pair has now gone unread on three boxes, each with one or two unstable pairs:

| box | host | unstable pairs |
|---|---|---|
| `tc1-5090-67` | Core Ultra 9 285K | Unsloth 18 %; e4b's field-image arm 8 % |
| `tc1-5090-68` | EPYC 7663 | e4b's same-stack arm 8 % |
| `tc1-5090-72` | EPYC 7C13, machine 45511, 60-step runs | e4b's same-stack arm 13 %; field-image arm 6 % |

The samplers TC1 amendment 7 put beside every arm show why, on all but the 285K. The step is host-bound, and on these multi-tenant hosts
the host's load average follows the slow draws:

- `tc1-5090-72`: e4b's same-stack draws ran at a median load1 of 22.9 and 11.0;
- `tc1-5090-70`: 6.6 / 7.3 / 12.1 / 19.4 across its four draws;
- `tc1-5090-66`'s shipped pair: 4.0 against 19.0;
- `tc1-5090-68`: 30–37 throughout.

Pairs read stable at a median load1 of about 5 or below. Machine 151350, the 285K, lost pairs at a load near 1, a different and unexplained
cause, so it is avoided.

**The instrument.** `tc1_run.sh` now runs each arm through a gate.

- With `TC1_LOAD_GATE` set, an OK arm whose median host load1 over its own run exceeds the gate is set aside to `loadvoid/` and run again,
  at most `TC1_LOAD_RETRIES` more times while the deadline allows.
- The last attempt stands whatever its load. Every attempt writes a `LOADGATE` line, and the reducer prints them.
- Unset, nothing changes, so every box registered before this amendment keeps its instrument.

**The box.** The token `qwen3samestack` as amendment 29 ran it:

- `TC1_STEPS=60`, no reference arm;
- `TC1_LOAD_GATE=6.0` and `TC1_LOAD_RETRIES=2`;
- on a machine other than 45511, 138786 and 151350.

P50 (Unsloth/e4b on one stack in **[1.9, 2.9]**) and P51 (e4b venv-unsloth / venv-e4b in **[0.80, 0.95]**) are re-asked with their bands
unchanged, read as before: two stable VALID draws a side.

**Its reading is final.** No further re-draw follows under amendments 25, 27, 29 or 33.

**Budget.** One RTX 5090 at the policy rate ($0.85/h), 4 h guard (a voided draw adds up to one arm's time), TC1's 98 GB host floor; about
$2.50 with the download and up to two re-runs. This is in the standing no-ask tier.

### Amendment 34 (2026-10-05T03:26Z, after amendment 32's read): amendment 24's environment gain, split between transformers and torch on the 5090 (P62–P65)

**Why.** Amendment 24 read e4b's matched arm at 0.882× in Unsloth's environment (torch 2.12.1, transformers 5.5.0, triton 3.7.1) against
the field image's (torch 2.8.0, transformers 5.18.0, triton 3.4). An unregistered RTX A2000 decomposition put that gain in triton 3.7.1's
device code. Amendment 32 (`tc1-5090-74`) then read triton 3.7.1 alone at 0.992× on the 5090's matched arm (P59 FALSIFIED) and 0.971×
on its shipped arm (P60 HELD). By that amendment's rule, the gain stays unattributed between torch and transformers on the host-bound
5090 step.

The A2000 profile points at host work in both:

- torch 2.12 removed about 2,300 host events per 4-layer step, mostly in attention with a padded mask;
- transformers 5.5 removed about 560, in the router, where 5.18 casts the routing weights to bf16 and e4b casts them back.

On a 48-layer host-bound step those could matter.

**The box** (token `qwen3envsplit`). One RTX 5090, TC1's qwen3 tokens and field recipe, `TC1_STEPS=60`, load-gated draws
(`TC1_LOAD_GATE=6.0`, `TC1_LOAD_RETRIES=2`), avoiding machines 45511, 138786 and 151350. The matched arm in three environments, two
draws each in ABC CBA order:

| side | venv | torch | transformers | triton |
|---|---|---|---|---|
| `_e0` | venv-e4b | 2.8.0 | 5.18.0 | 3.4 |
| `_e1` | venv-e4b-tf55, built on the box exactly as venv-e4b but with transformers 5.5.0 | 2.8.0 | 5.5.0 | 3.4 |
| `_e2` | venv-unsloth + e4b | 2.12.1 | 5.5.0 | 3.7.1 |

The prebound launches are off on every side; they cover triton 3.4 and 3.6 only, so they would be a fourth variable. Engagement: each
receipt records the torch and transformers its side names, and both prebind flags off. A failed venv-e4b-tf55 build leaves `_e1` arms
`install_failed`.

**Predictions** (registered before the box):

- **P62** (transformers alone): `_e1` / `_e0` lies in **[0.90, 1.00]**.
- **P63** (torch 2.12 + triton 3.7, at transformers 5.5): `_e2` / `_e1` lies in **[0.85, 0.99]**.
- **P64** (the whole environment, amendment 24 re-read under the gate): `_e2` / `_e0` lies in **[0.80, 0.95]**.
- **P65:** |mean held-out at N| of `_e1` − `_e0` and of `_e2` − `_e0` are each ≤ **0.005**.

Each speed prediction needs two stable VALID draws per side. Each is FALSIFIED outside its band and UNTESTED where a side is unstable, not
VALID or not engaged.

**Decision rules.**

- **P62 HELD with the ratio at or below 0.97:** e4b finds what transformers 5.18 adds to its step, router casts first, and works around
  it in its own PR with its own A/B.
- **P63 HELD:** e4b's docs say that torch ≥ 2.12 runs its host-bound training step faster on an RTX 5090.
- No position against another framework is read here; amendment 33's box is the same-stack position.

**Budget.** One RTX 5090 at the policy rate ($0.85/h), 4 h guard, TC1's 98 GB host floor, with venv-unsloth built for `_e2`. About $2
with the download and the gate's possible re-runs; this is in the standing no-ask tier.


### Amendment 35 (2026-10-05T04:25Z, after amendment 33's read, before any box): amendment 26's prebound-launch A/B under triton 3.7.1 (P66–P68)

**Why.** Amendments 26 and 30 read the prebound Triton launches on an RTX 5090 under triton 3.4: the matched arm at 0.973× and the shipped
arm at 0.980× of the flags off. By amendment 26's rule both flags became the default for the Triton versions the prebound path covers.
experts4bit-qlora#1108 and grouped-nf4-gemm#471 added triton 3.7. Their evidence is from an RTX A2000 host: outputs bit-identical, and the
host µs per call lower with the flags on (one launch 25–51 → 19–39; grouped-nf4-gemm's fused forward 439–483 → 334–377). No training step
was timed. Triton 3.7.1's own launch is cheaper than 3.4's, so the saving per launch is smaller than amendment 26's A2000 figures.
Amendment 33's same-stack position ran e4b under triton 3.7.1 before #1108, so with the flags unset. Every e4b run in that environment now
takes the prebound path.

**The box** (token `qwen3prebind37`). One RTX 5090, TC1's qwen3 tokens and field recipe, `TC1_STEPS=60`, load-gated draws
(`TC1_LOAD_GATE=6.0`, `TC1_LOAD_RETRIES=2`), avoiding machines 45511, 138786 and 151350.

- Amendment 26's eight arms in its ABBA order: the shipped arm and the matched arm, each `_pb0` (both flags 0) against `_pb1` (both
  flags 1), two draws a side.
- Every arm in venv-unsloth with e4b and grouped-nf4-gemm at the box's pins (TC1's t212 install: torch 2.12.1+cu130, transformers 5.5.0,
  triton 3.7.1). Both pins must include #1108 and #471.
- Engagement: amendment 26's `prebind_ab` predicate, and each receipt records torch 2.12.* and triton 3.7.*.

**Predictions** (registered before the box):

- **P66** (shipped): `_pb1` / `_pb0` s/step lies in **[0.97, 1.00]**.
- **P67** (matched): `_pb1` / `_pb0` lies in **[0.97, 1.00]**.
- **P68:** on each arm, |mean held-out at N, `_pb1` − `_pb0`| ≤ **0.005**.

The basis: under triton 3.4 the 5090 read 0.980 (shipped) and 0.973 (matched). On the A2000 the 3.7 saving per call is a third
(RMSNorm) to a half (the fused GEMM) of the 3.4 one, so each ratio is expected near 0.985–0.995, and neither side should be slower. Each speed prediction needs two stable VALID
draws a side. Each is FALSIFIED outside its band and UNTESTED where a side is unstable, not VALID or not engaged.

**Decision rules.**

- **P66, P67 and P68 HELD:** triton 3.7 stays in the prebound path's supported versions, and the register gains a row for the default on
  triton 3.7.
- **Either ratio above 1.01, or P68 FALSIFIED:** triton 3.7 comes out of the supported versions, one PR in each repository, citing this
  box. Runs on 3.7 then keep Triton's own launch.
- **Otherwise** (a ratio in (1.00, 1.01], below 0.97, or UNTESTED): triton 3.7 stays covered and no speed is claimed for it.
- No position against another framework is read here.

**Budget.** One RTX 5090 at the policy rate ($0.85/h), 4 h guard, TC1's 98 GB host floor, venv-unsloth built for the t212 install.
About $2 with the download and the gate's possible re-runs; this is in the standing no-ask tier.

### Amendment 36 (2026-10-05T04:52Z, after amendment 33's read, before any box): grouped-nf4-gemm's compact padded LoRA delta, A/B for peak and speed on one stack (P69–P72)

**Why.** On one stack Unsloth peaks 3.22 GB below e4b (amendment 33: 24.27 vs 27.49 GB). Amendment 23's census put all of that gap in
two places:

- the fp32 expert absmax, 1.35 GB (double-quantizing it is the CLI trainer's default since #1105; the harness keeps it explicit);
- transients, led by grouped-nf4-gemm's padded LoRA delta: its zero-padded input block in the adapters' fp32 (0.62 GB, two live at the
  peak) and the batched products' outputs (0.45 GB, three live).

grouped-nf4-gemm#445 ships `NF4_QLORA_COMPACT_DELTA=1`, opt-in "until a within-box A/B decides the default". The delta becomes one
autograd node that saves its input instead of its padded block, and rebuilds the block in backward. Values and every gradient are
`torch.equal`. On an RTX A2000 (one layer, 380 tokens) it cut the memory saved per layer from 229 to 55 MB with fp32 adapters, and
added about 0.7 ms (+3.5 %) to a layer's backward device time. Under whole-layer checkpointing, one layer at a time holds those saved
blocks, so the expected effect is a lower backward peak, not a lower static footprint. No 5090 step has measured it at TC1's defaults.

**The box** (token `qwen3compactab`). One RTX 5090, TC1's qwen3 tokens and field recipe, `TC1_STEPS=60`, load-gated draws
(`TC1_LOAD_GATE=6.0`, `TC1_LOAD_RETRIES=2`), avoiding machines 45511, 138786 and 151350.

- The shipped and the matched arm, each `_cd0` (`NF4_QLORA_COMPACT_DELTA=0`, the default) against `_cd1` (`=1`), two draws a side in
  amendment 26's ABBA order. Every other setting is the default.
- Every arm in venv-unsloth with e4b and grouped-nf4-gemm at the box's pins (TC1's t212 install: torch 2.12.1+cu130, transformers 5.5.0,
  triton 3.7.1), the stack of the quoted position.
- Engagement, read off each receipt: `keep_ab.gnf4_compact_delta` is the side's flag, no layer kept its MoE activations, the padded LoRA
  path ran (`lean_ab.lora_path_calls.padded` > 0), and `env.torch` is 2.12.*.

**Predictions** (registered before the box):

- **P69** (matched, memory): the median peak falls by **[0.3, 2.0] GB** (`_cd0` − `_cd1`). The basis: the census's live padded block,
  0.62 GB at micro-batch 1, no longer held from forward to backward; the rebuilt block is transient.
- **P70** (matched, speed): `_cd1` / `_cd0` s/step lies in **[0.97, 1.02]**.
- **P71** (shipped, speed): `_cd1` / `_cd0` lies in **[0.97, 1.02]**. The rebuild adds device time; the one node replaces about ten
  autograd nodes per projection, so on this host-bound step the two may cancel.
- **P72:** on each arm, |mean held-out at N, `_cd1` − `_cd0`| ≤ **0.005**.

Each prediction needs two stable VALID draws a side. Each is FALSIFIED outside its band and UNTESTED where a side is unstable, not VALID
or not engaged.

**Decision rules.**

- **P69–P72 HELD:** `NF4_QLORA_COMPACT_DELTA` becomes grouped-nf4-gemm's default, in one PR citing this box. The position against
  Unsloth is not restated from this box; a later box reads peaks side by side.
- **P70 or P71 above 1.02:** it stays opt-in.
- **P69 below 0.3 GB:** it stays opt-in. The next memory lever is the padded block's size under a hot expert, not what it saves.
- **Otherwise** (a ratio below 0.97, P69 above 2.0 GB, or UNTESTED): it stays opt-in pending its own registration.
- No position against another framework is read here.

**Budget.** One RTX 5090 at the policy rate ($0.85/h), 4 h guard, TC1's 98 GB host floor, venv-unsloth built for the t212 install.
About $2 with the download and the gate's possible re-runs; this is in the standing no-ask tier.

### Amendment 37 (2026-10-05T06:50Z, after amendment 36's read, before any box): the compact padded LoRA delta again, with its backward releasing each intermediate at its last use, on another host (P73–P76)

**Why.** Amendment 36 (`tc1-5090-80`) found grouped-nf4-gemm's compact delta faster than registered (0.969 matched, 0.970 shipped) and
heavier, not lighter: the matched peak rose 0.229 GB. Its backward held every intermediate until it returned, so the padded output
gradient was still live while the input block and its gradient were rebuilt. grouped-nf4-gemm#473 releases each intermediate at its last
use; no operation, operand layout, dtype or call order changes, and every output and gradient stays `torch.equal`. On an RTX A2000, at
Qwen3-30B-A3B's shapes with a skewed router (one `lora_delta_grouped` call, fp32 and bf16 adapters, 380 and 1,100 tokens), its backward
peak went from 24–55 % above the autograd path's to 2–30 % below it in all eight cells, with time per call unchanged. That is a per-call
reading; the training step's peak is this box's question. A speed default also needs a second host: amendment 36's box was the only one.

**The box** (token `qwen3compactab2`). Amendment 36's box unchanged (the shipped and the matched arm, `_cd0` vs `_cd1`, two draws a side
in ABBA order, every arm in venv-unsloth, 60 steps, load-gated draws, `TC1_LOAD_GATE=6.0`, `TC1_LOAD_RETRIES=2`), with grouped-nf4-gemm
pinned at or after #473's merge, on a machine other than 145701 (amendment 36's) and the three avoided before (45511, 138786, 151350).
Engagement is amendment 36's.

**Predictions** (registered before the box):

- **P73** (matched, memory): the median peak changes by at most +0.05 GB and falls by at most 0.50 GB (`_cd0` − `_cd1` in
  **[−0.05, 0.50] GB**).
- **P74** (matched, speed): `_cd1` / `_cd0` lies in **[0.95, 0.99]**.
- **P75** (shipped, speed): `_cd1` / `_cd0` lies in **[0.95, 0.99]**.
- **P76:** on each arm, |mean held-out at N, `_cd1` − `_cd0`| ≤ **0.005**.

The basis: #473 changes no operation, so amendment 36's speed (0.969 / 0.970) should carry to another host within a few points, and the
A2000's per-call backward peaks now sit at or below the autograd path's. Each prediction needs two stable VALID draws a side. Each is
FALSIFIED outside its band and UNTESTED where a side is unstable, not VALID or not engaged.

**Decision rules.**

- **P73–P76 HELD:** `NF4_QLORA_COMPACT_DELTA` becomes grouped-nf4-gemm's default, in one PR citing amendments 36 and 37.
- **P73 FALSIFIED above (the peak rises more than 0.05 GB):** it stays opt-in; the release-early change did not reach the step's peak.
- **P74 or P75 above 0.99:** it stays opt-in; the speed did not replicate.
- **Otherwise** (a ratio below 0.95, a drop above 0.50 GB, or UNTESTED): it stays opt-in pending its own registration.
- No position against another framework is read here.

**Budget.** One RTX 5090 at the policy rate ($0.85/h), 4 h guard, TC1's 98 GB host floor, venv-unsloth built for the t212 install.
About $2 with the download and the gate's possible re-runs; this is in the standing no-ask tier.

### Amendment 38 (2026-10-05T08:17Z, after amendment 37's read, before any box): the compact padded LoRA delta's default decision, a third host and a second family (P77–P83)

**Why.** Two boxes have now read grouped-nf4-gemm's compact delta on Qwen3-30B-A3B in venv-unsloth:

| box | host | matched | shipped | matched peak |
|---|---|---|---|---|
| `tc1-5090-80` (amendment 36, before #473) | EPYC 7B13 | 0.969 | 0.970 | +0.229 GB |
| `tc1-5090-83` (amendment 37, with #473) | EPYC 7702P | 0.967 | 0.948 | −0.288 GB |

Amendment 37's rule kept the flag opt-in because the shipped ratio fell below a two-sided band, on the side the decision wanted. A default in
grouped-nf4-gemm changes every family's step, and no family but Qwen3-30B-A3B has been read with the flag. Mixtral-8x7B takes the padded
LoRA path too (TC2 amendment 8's box: 11,264 padded calls), and at default settings it peaks at 31.07 GB of the card's 32.

**The box** (tokens `qwen3compactab3` and `mixtralcompactab`, in that order). One RTX 5090, every arm in venv-unsloth with e4b and
grouped-nf4-gemm at the box's pins (grouped-nf4-gemm at or after #473), 60 steps, load-gated draws (`TC1_LOAD_GATE=6.0`,
`TC1_LOAD_RETRIES=2`), a 192 GB host floor, on a machine other than 145701 and 45379 (amendments 36 and 37's) and 151350, 45511 and 138786.

- `qwen3compactab3`: amendment 36's box unchanged, the shipped and the matched arm, `_cd0` vs `_cd1`, two draws a side in ABBA order.
- `mixtralcompactab`: Mixtral-8x7B-Instruct at TC2's pin and field recipe, resident at default settings (the dense route), the matched arm
  only (the shipped arms skipped), `_cd0` vs `_cd1`, two draws a side.

Engagement is amendment 36's (the resolved flag, no kept layers, the padded route, torch 2.12).

**Predictions** (registered before the box), one-sided: the decision needs the flag no slower and no heavier, not a size of gain.

- **P77** (Qwen3, matched): `_cd1` / `_cd0` ≤ **0.99**.
- **P78** (Qwen3, shipped): `_cd1` / `_cd0` ≤ **0.99**.
- **P79** (Qwen3, matched peak): `_cd1` − `_cd0` ≤ **+0.05 GB**.
- **P80** (Mixtral, matched): `_cd1` / `_cd0` ≤ **1.01**. Mixtral routes 2 of 8 experts, so its padded blocks are small and the saving may be
  too; level is allowed.
- **P81** (Mixtral, matched peak): `_cd1` − `_cd0` ≤ **+0.05 GB**.
- **P82, P83:** on each arm of each family, |mean held-out at N, `_cd1` − `_cd0`| ≤ **0.005**.

Each needs two stable VALID draws a side, and is FALSIFIED on the wrong side of its bound and UNTESTED where a side is unstable, not VALID
or not engaged.

**Decision rules.**

- **P77–P83 HELD:** `NF4_QLORA_COMPACT_DELTA` becomes grouped-nf4-gemm's default, in one PR citing amendments 36, 37 and 38 (`=0` keeps
  the autograd path).
- **Any of them FALSIFIED:** it stays opt-in, and the read names which family and which side.
- **Any UNTESTED, none FALSIFIED:** it stays opt-in, and no further box is registered for this flag under these rules.
- No position against another framework is read here.

**Budget.** One RTX 5090 at the policy rate ($0.85/h), 4 h guard, a 192 GB host floor, venv-unsloth built for the t212 install, Qwen3 and
Mixtral downloaded. About $3 with the downloads and the gate's possible re-runs; this is in the standing no-ask tier.

### Amendment 39 (2026-10-05T09:39Z, after TC2 amendment 9's read, before any box): the packed 4,096-token regime with both frameworks on one stack (P84, P85, P86)

**Why.** Every TC1 position reads the field recipe, whose Alpaca rows are short. The receipts record about 1,000–1,400 real tokens per
optimizer step against the recipe's nominal 16,384 (seq 2,048 × micro-batch 2 × accum 4). There the step is host-launch-bound: the
5090's positions, the environment gain and every host lever were read in that regime. Packed training at a full sequence length is common
and puts about 12× the real tokens through each step. The device then does most of the work, and e4b's lead may shrink or reverse:

- e4b's thinnest lead is on the H100 (1.061), where its step is the most device-bound;
- the 2026-10-02 profiles (`…h100.2026-10-02.dispatch-profile`) put Unsloth's 5090 device time per step at 2.31 s against e4b's 2.96 s
  (9.868 × 0.234 against 6.169 × 0.480), e4b's lead then coming from host time;
- e4b's loss is Hugging Face's, which materialises the full-vocabulary logits and upcasts them to fp32. At 4,096 tokens that is about 1.2 GB
  of bf16 logits plus 2.5 GB each for the fp32 copy and its gradient. Amendment 23's census puts e4b's fixed memory at about 24.5 GB.

**The instrument** (bench/tc1, TC1 harness PR for this amendment):

- `TC1_PACK=1`: the field recipe's Alpaca examples, same template, tokenizer and order, extended past the 1,200 + 48 registered rows in
  `tp4_alpaca.py`'s own shuffled order (the prefix is checked byte-for-byte), joined with EOS and cut into rows of exactly `TC1_SEQ`
  tokens; labels are the inputs, no padding, full causal attention across example boundaries. Held-out rows are packed the same way from a
  disjoint pool. Unset, the field recipe's tokens file is byte-identical to before.
- `TC1_FREE_OUTPUTS=1`: each micro-batch's model output is released once its loss is read. Unset, the previous micro-batch's output, its
  logits included, stays referenced through the next forward and the optimizer step, inside every arm's measured peak. Every framework,
  every arm.
- The packed family is read against its own fixture: a receipt that is not packed, not at seq 4,096, not micro-batch 1 × accum 4, not
  16,384 real and 0 padded tokens on every step, or not `free_outputs` is VOID.

**The box** (token `qwen3samestack4k`). TC1 amendment 25's same-stack family as amendment 33 ran it, on the packed rows:

- `TC1_SEQ=4096 TC1_MB=1 TC1_ACCUM=4 TC1_PACK=1 TC1_FREE_OUTPUTS=1`, `TC1_STEPS=30`, held-out at steps 0 and 30 only
  (`TC1_EVAL_EVERY=30`, 8 rows of 4,096 tokens), e4b's reference arm not run;
- e4b's matched arm in venv-unsloth (two draws), Unsloth 2026.9.14 grouped_mm (two draws), e4b's matched arm in venv-e4b (`_t28`, two
  draws), every e4b arm resident at default settings;
- load-gated draws (`TC1_LOAD_GATE=6.0`, `TC1_LOAD_RETRIES=2`), avoiding machines 151350, 45511 and 138786.

**Predictions** (registered before the box):

- **P84:** with both frameworks on one stack, Unsloth/e4b lies in **[0.80, 1.60]**, both pairs stable. The band is wide on purpose: the
  field recipe's 2.352 is host time, and the 2026-10-02 device times point the other way.
- **P85:** e4b's matched arm in venv-unsloth over venv-e4b lies in **[0.80, 1.00]**, both sides stable. A device-bound step should gain less
  from torch 2.12's host savings than the field recipe's 0.900.
- **P86:** every e4b arm that runs completes resident (VALID). The estimate sits close to the card, so this is the prediction most at risk.

Each is FALSIFIED outside its band (P86: an e4b arm that OOMs) and UNTESTED where a side is missing, unstable or not engaged.

**Decision rules.**

- **P84 read with both pairs stable, whichever side it favours:** the ratio is recorded as Qwen3-30B-A3B's packed 4,096-token position
  (`e4b.train.h2h.unsloth.qwen3.5090.<date>.packed-4k`), beside the field recipe's 2.352, and STATUS names both regimes. A reading below
  1.00 is said as an e4b loss in that regime.
- **P86 FALSIFIED (e4b OOMs while Unsloth trains):** that is recorded as an e4b loss in that regime, with the peak where it failed. A
  chunked loss for e4b is then its own registration, and so is a 2,048-token packed box.
- **P85** says how much of the torch 2.12 gain survives a device-bound step; it moves no default.

**Budget.** One RTX 5090 at the policy rate ($0.85/h), 4 h guard, TC1's 98 GB host floor, venv-unsloth built for e4b's same-stack arms and
Unsloth. Six arms of 30 steps at an estimated 15–40 s a step: about $3 with the download; this is in the standing no-ask tier.

### Amendment 40 (2026-10-05T11:38Z, after amendment 39's read, before any box): the packed 4,096-token regime again, e4b with its chunked LM loss (P87, P88, P89)

**Why.** Amendment 39 (`tc1-5090-86`) found every e4b arm out of memory at step 1 on packed 4,096-token rows, allocating 2.32 GiB: the
fp32 copy of the full-vocabulary logits in Hugging Face's causal-LM loss. Unsloth trained the same rows at 24.86 GB. experts4bit-qlora#1142
adds an opt-in chunked LM loss (`E4B_CHUNKED_LM_LOSS`, applied by `enable_fast_train`): the loss is computed over token chunks under
non-reentrant checkpointing, so no chunk's logits are kept and the full logits are never materialised. Its loss matches the stock loss to
fp32 rounding, and its gradients sit inside run-to-run noise. On an RTX A2000 a 4-layer slice of the same checkpoint hit the stock path's
exact 2.32 GiB failure at 4,096 tokens and trained through it chunked, at 7.11 GiB.

**The box** (token `qwen3samestack4kce`). Amendment 39's box unchanged — the same packed rows (`TC1_PACK=1 TC1_SEQ=4096 TC1_MB=1
TC1_ACCUM=4 TC1_FREE_OUTPUTS=1`), amendment 25's same-stack arms, e4b's reference not run — with:

- `TC1_E4B_ENV="E4B_CHUNKED_LM_LOSS=1"` on every e4b arm (512-token chunks), e4b's other settings at their defaults;
- `TC1_STEPS=40`, held-out at 0 and 40 only (`TC1_EVAL_EVERY=40`): amendment 39's Unsloth draws were 7.7 % apart over 20 median steps on a
  busy host;
- load-gated draws, avoiding machines 151350, 45511 and 138786;
- grouped-nf4-gemm at main, e4b at this amendment's merge (after #1142).

Engagement: amendment 39's packed fixture on every arm, and on every e4b arm the receipt's `chunked_lm_loss` record (new in `tc1_arm.py`:
the variable set, e4b has the loss, training forwards went through it, none fell back at run time).

**Predictions** (registered before the box):

- **P87:** with both frameworks on one stack, Unsloth/e4b lies in **[0.80, 1.60]**, both pairs stable (amendment 39's P84, unread there).
- **P88:** e4b's matched arm in venv-unsloth over venv-e4b lies in **[0.80, 1.00]**, both sides stable (amendment 39's P85).
- **P89:** every e4b arm that runs completes resident (VALID). The chunked loss removes the 2.32 GiB allocation that failed and the bf16
  logits beside it, against amendment 39's 29.5 GiB in use at the failure.

Each is FALSIFIED outside its band (P89: an e4b arm that OOMs) and UNTESTED where a side is missing, unstable or not engaged.

**Decision rules.**

- **P87 read with both pairs stable and P89 HELD, whichever side it favours:** the ratio is recorded as Qwen3-30B-A3B's packed 4,096-token
  position, labelled "e4b with `E4B_CHUNKED_LM_LOSS=1`, opt-in" (`e4b.train.h2h.unsloth.qwen3.5090.<date>.packed-4k-chunked`), beside
  amendment 39's out-of-memory row. A reading below 1.00 is said as an e4b loss in that regime.
- **P89 HELD:** whether the chunked loss becomes e4b's default is its own registration, with its cost read on the field recipe.
- **P89 FALSIFIED:** the chunked loss alone does not fit the regime on 32 GB; the next memory lever (the absmax double-quantized, the
  compact delta) is its own registration.

**Budget.** One RTX 5090 at the policy rate ($0.85/h), 4 h guard, TC1's 98 GB host floor, venv-unsloth built for e4b's same-stack arms and
Unsloth. Six arms of 40 steps at an estimated 12–20 s a step: about $3 with the download; this is in the standing no-ask tier.

### Amendment 41 (2026-10-05T13:16Z, after amendment 39's read, with amendment 40's box running): e4b's chunked LM loss at the field recipe — its default decision (P90–P93)

**Why.** Amendment 39 found e4b at its defaults out of memory on packed 4,096-token rows, on the fp32 copy of the full-vocabulary logits.
experts4bit-qlora#1142 adds `E4B_CHUNKED_LM_LOSS` (opt-in), and amendment 40's box reads whether it fits that regime. A default must also
cost the common case nothing it should not: the field recipe's short rows. On an RTX A2000 4-layer slice, where the LM head is a large
share of the step, the chunked loss cost 2–6 %; on the 48-layer step the head is a far smaller share. Each chunk's logits are recomputed in
backward, and finding the supervised tokens costs one host sync per training forward.

**The box** (token `qwen3chunkab`). One RTX 5090, TC1's qwen3 tokens and field recipe, 60 steps, load-gated draws (`TC1_LOAD_GATE=6.0`,
`TC1_LOAD_RETRIES=2`), avoiding machines 151350, 45511 and 138786:

- the shipped and the matched arm, each `_ce0` (`E4B_CHUNKED_LM_LOSS=0`, the default) against `_ce1` (`=1`, 512-token chunks), two draws a
  side in ABBA order;
- every arm in venv-unsloth with e4b and grouped-nf4-gemm at the box's pins (TC1's t212 install), every other setting at its default.

Engagement: each receipt's `chunked_lm_loss` record (amendment 40): `_ce1` arms chunked their training forwards with no run-time fallback,
`_ce0` arms chunked none; `env.torch` 2.12.*.

**Predictions** (registered before the box), one-sided, as amendment 38 learned: the decision needs the switch no slower and no heavier.

- **P90** (matched): `_ce1` / `_ce0` ≤ **1.01**.
- **P91** (shipped): `_ce1` / `_ce0` ≤ **1.01**.
- **P92** (matched peak): `_ce1` − `_ce0` ≤ **+0.05 GB**.
- **P93:** on each arm, |mean held-out at N, `_ce1` − `_ce0`| ≤ **0.005**.

Each needs two stable VALID draws a side, and is FALSIFIED on the wrong side of its bound and UNTESTED where a side is unstable, not VALID
or not engaged.

**Decision rules.**

- **P90–P93 HELD, and amendment 40's P89 HELD:** `E4B_CHUNKED_LM_LOSS` becomes e4b's default in `enable_fast_train` (512-token chunks;
  `=0` keeps the stock loss), in one PR citing amendments 39, 40 and 41.
- **P90–P93 HELD but P89 not HELD:** it stays opt-in; the chunked loss does not by itself buy the regime it is for.
- **Any of P90–P93 FALSIFIED:** it stays opt-in, and the read names which arm and which side.
- **Any UNTESTED, none FALSIFIED:** it stays opt-in pending a re-ask.
- No position against another framework is read here.

**Budget.** One RTX 5090 at the policy rate ($0.85/h), 4 h guard, TC1's 98 GB host floor, venv-unsloth built for the t212 install.
About $2 with the download; this is in the standing no-ask tier.

### Amendment 42 (2026-10-05T13:57Z, after amendment 38's read, before any box): the same-stack position on a second host, on the current code (P94, P95)

**Why.** The Qwen3-30B-A3B position to quote, Unsloth/e4b **2.352** (amendment 33, `tc1-5090-76`), is one box on one host model, an AMD
EPYC 7B13. Two later readings say the host can move e4b's step a lot:

- amendment 38's box (EPYC 9655) stepped e4b's matched arm in 2.17 s, against 3.4–3.9 s on EPYC 7B13 and 7702P hosts, with the same code
  and card;
- TC2 amendment 9 traced most of Mixtral's 0.836 to the host: Unsloth's Mixtral step was 3.0 s on a Core Ultra 9 285K and 3.7 s on an
  EPYC 7B13, while e4b's moved little.

Positions are within-box readings of their host, but a quoted number should survive a second host. The code also moved since amendment
33: the prebound launches now cover triton 3.7 (#1108), so both of e4b's sides take them.

**The box** (token `qwen3samestackh2`). Amendment 33's box unchanged — amendment 25's same-stack family, `TC1_STEPS=60`, no reference arm,
load-gated draws (`TC1_LOAD_GATE=6.0`, `TC1_LOAD_RETRIES=2`) — on the current code (e4b at this amendment's merge, grouped-nf4-gemm at
main), on a machine other than 145701 (amendment 33's) and 151350, 45511 and 138786. Nothing else is set.

**Predictions** (registered before the box):

- **P94:** Unsloth/e4b on one stack lies in **[1.9, 2.9]**, both pairs stable (amendment 25's P50 band).
- **P95:** e4b's matched arm in venv-unsloth over venv-e4b lies in **[0.80, 0.95]**, both sides stable (P51's band).

Each is FALSIFIED outside its band and UNTESTED where a side is missing, unstable or not engaged.

**Decision rules.**

- **P94 HELD:** 2.352 stays the position to quote; this box's reading and host are recorded beside it, and STATUS says it held on two
  hosts.
- **P94 FALSIFIED:** STATUS quotes the two hosts' readings as the position's range, each with its host, and the 2.352 row's note says it
  depends on the host. Which of the host and the code moved it is not read from this box.
- **P95** replicates the environment gain once more; it moves no default.

**Budget.** One RTX 5090 at the policy rate ($0.85/h), 4 h guard, TC1's 98 GB host floor, venv-unsloth built. About $2 with the download;
this is in the standing no-ask tier.

### Amendment 43 (2026-10-05T15:34Z, after amendment 40's read, before any box): the packed 4,096-token regime with e4b's chunked loss again, the LoRA loop read as a recorded route (P96, P97, P98)

**Why.** Amendment 40's box (`tc1-5090-91`) showed the chunked LM loss doing its job: every e4b arm trained the packed rows to step 40
at a 32.4–32.6 GB peak, where amendment 39's had OOMed at step 1. But every e4b arm read VOID under TC1's rule that a fused arm's
per-expert LoRA loop never runs. On those rows grouped-nf4-gemm's `auto` route took the loop on every step for about 1.5 % of its delta
calls (at most 2.9 %): the calls whose padded block would exceed its 2 GiB limit (`NF4_QLORA_PAD_BYTES_LIMIT`), which a hot expert's block
does at 4,096 tokens. That is e4b's default route doing what it was built to do; the rule was written for the field recipe, where the loop
never runs, to catch a path that silently fell back.

**The change of instrument** (this family only): a fused e4b arm's per-expert loop is a recorded route, not a VOID, while its share of a
step's delta calls stays at or below **5 %** on every step; above that the arm is VOID as before. The share is printed for every e4b arm.
Every other family keeps the rule, and amendment 40's box reads as it did.

**These bands are not blind.** Read as if VALID, amendment 40's box said Unsloth/e4b 1.43 and the environment 0.892 on an EPYC 7K62. The
bands below were set with those numbers in view; this box is their replication on another host, not a test of a prediction made without
them.

**The box** (token `qwen3samestack4kce2`). Amendment 40's box unchanged — packed rows of exactly 4,096 real tokens (`TC1_PACK=1
TC1_SEQ=4096 TC1_MB=1 TC1_ACCUM=4 TC1_FREE_OUTPUTS=1`), `TC1_E4B_ENV="E4B_CHUNKED_LM_LOSS=1"`, 40 steps, held-out at 0 and 40, amendment
25's same-stack arms, e4b's reference not run, load-gated draws — on a machine other than 152440 (amendment 40's), 151350, 45511, 138786,
40093 and 57910.

**Predictions** (registered before the box):

- **P96:** with both frameworks on one stack, Unsloth/e4b lies in **[1.25, 1.65]**, both pairs stable.
- **P97:** e4b's matched arm in venv-unsloth over venv-e4b lies in **[0.84, 0.95]**, both sides stable.
- **P98:** every e4b arm that runs completes resident and VALID under this family's rule.

Each is FALSIFIED outside its band (P98: an e4b arm that OOMs) and UNTESTED where a side is missing, unstable or not engaged.

**Decision rules.**

- **P96 read with both pairs stable and P98 HELD, whichever side it favours:** the ratio is recorded as Qwen3-30B-A3B's packed 4,096-token
  position, labelled "e4b with `E4B_CHUNKED_LM_LOSS=1`, opt-in; grouped-nf4-gemm's loop route at the recorded share"
  (`e4b.train.h2h.unsloth.qwen3.5090.<date>.packed-4k-chunked`), beside amendment 39's out-of-memory row.
- **Amendment 41's default decision reads this box's P98 in place of amendment 40's P89**, which read UNTESTED on the loop rule, not on
  fit. With P98 HELD and amendment 41's P90–P93 HELD, `E4B_CHUNKED_LM_LOSS` becomes e4b's default in `enable_fast_train`.
- **P98 FALSIFIED:** the chunked loss alone does not fit the regime on this host; the read says where it failed.

**Budget.** One RTX 5090 at the policy rate ($0.85/h), 4 h guard, TC1's 98 GB host floor, venv-unsloth built. About $1.6 with the download;
this is in the standing no-ask tier.

### Amendment 44 (2026-10-05T17:12Z, after amendment 41's read, with amendment 43's box running and unread): e4b's chunked LM loss as `auto` — its default decision (P99–P103)

**Why.** The chunked LM loss is what lets e4b train packed 4,096-token Qwen3 rows (amendments 39, 40). At the field recipe it costs the
shipped arm 1.049 of its step (amendment 41, P91 FALSIFIED), so `=1` stays opt-in. experts4bit-qlora#1178 adds `E4B_CHUNKED_LM_LOSS=auto`:
a training forward chunks only when its stock fp32 logits (positions × vocabulary × 4 bytes, read from the labels' shape) would reach
1 GiB, and a smaller one runs the stock forward untouched (`small_calls`). The gate was set from shapes, not timings. TC1's field
micro-batches are 0.29 GiB at the median and 0.64 GiB at the largest over a 60-step run, and the two longest of its 1,200 rows padded
together are 0.86 GiB. One packed 4,096-token row is 2.32 GiB. So on the field recipe `auto` should be the stock path, and on packed rows
it is `=1`'s path exactly: every packed forward is 4,096 positions over the gate, and #1178's tests pin both sides of it. This box reads
the field side. Amendment 43's box, running `=1` on packed rows, reads the packed side (its P98).

**The box** (token `qwen3chunkauto`). Amendment 41's box with `auto` in place of `1`. One RTX 5090, TC1's qwen3 tokens and field recipe,
60 steps, load-gated draws (`TC1_LOAD_GATE=6.0`, `TC1_LOAD_RETRIES=2`), avoiding machine 145701 (amendment 41's host, where every attempt
ran above the gate) and machines 151350, 45511 and 138786:

- the shipped and the matched arm, each `_ca0` (`E4B_CHUNKED_LM_LOSS=0`, the default) against `_ca1` (`=auto`), two draws a side in ABBA
  order;
- every arm in venv-unsloth with e4b (at a main that has #1178) and grouped-nf4-gemm at the box's pins (TC1's t212 install), every other
  setting at its default.

Engagement: each receipt's `chunked_lm_loss` record. `_ca1` arms have `env` `auto`, a patched model, the `small_calls` counter and no
run-time fallback; `_ca0` arms have nothing patched and no chunked or gated forward; `env.torch` 2.12.*. Whether the gate fired is P99's
to score, not validity's.

**Predictions** (registered before the box, and before amendment 43's box is read), one-sided:

- **P99** (the gate, structural): on every `_ca1` arm, `chunked_calls` 0 and `small_calls` 240 (every training forward of 60 steps × 4).
  FALSIFIED iff a VALID `_ca1` arm chunked a forward (`chunked_calls` > 0: the gate fired). HELD iff every `_ca1` arm is VALID and reads
  exactly 0 / 240. An arm that chunked nothing but gated another count leaves P99 UNTESTED: that is an extra or missing training forward,
  a question about the instrument or the trainer, and the read reports the count.
  (This split was made at review, 2026-10-05T18:44Z, before any box: the first text and the reducer disagreed on an arm that chunked nothing but gated a count other than 240.)
- **P100** (shipped): `_ca1` / `_ca0` ≤ **1.02**.
- **P101** (matched): `_ca1` / `_ca0` ≤ **1.02**.
- **P102** (matched peak): `_ca1` − `_ca0` ≤ **+0.05 GB**.
- **P103:** on each arm, |mean held-out at N, `_ca1` − `_ca0`| ≤ **0.005**.

Under P99 the `_ca1` side runs the stock forward plus one gate check per forward, so P100 and P101 guard against a cost nobody expects.
Their bound, 1.02 rather than amendment 41's 1.01, sits above the default side's draw spread amendment 41 read on the same arms (1.4 % shipped,
0.8 % matched; its chunked side's were 4.8 % and 5.4 % on a loaded host), because this is close to an A/A reading. P100–P103 each need two stable VALID draws a side, and are FALSIFIED on the wrong
side of their bound and UNTESTED where a side is unstable, not VALID or not engaged.

**Decision rules.**

- **P99–P103 HELD, and amendment 43's P98 HELD:** `auto` becomes e4b's default in `enable_fast_train` and the CLI trainer
  (`E4B_CHUNKED_LM_LOSS` unset means `auto`; `0` keeps the stock loss everywhere; `1` or a chunk size chunks every training forward), in
  one PR citing amendments 39, 40, 41, 43 and 44. This rule replaces amendments 41 and 43's rule for `=1`, which amendment 41's read already
  closed (P91 FALSIFIED).
- **P99–P103 HELD but P98 not HELD:** `auto` stays opt-in; the chunked loss does not by itself buy the regime it is for.
- **P99 FALSIFIED:** the gate fired on the field recipe, which the shapes say it cannot. `auto` stays opt-in and the read finds the forward.
- **Any of P100–P103 FALSIFIED:** `auto` stays opt-in, and the read names which arm and which side.
- **Any UNTESTED, none FALSIFIED:** `auto` stays opt-in pending a re-ask.
- No position against another framework is read here.

**Budget.** One RTX 5090 at the policy rate ($0.85/h), 4 h guard, TC1's 98 GB host floor, venv-unsloth built for the t212 install. About
$2.5 with the download; this is in the standing no-ask tier.

### Amendment 45 (2026-10-05T19:17Z, after TC1c amendment 9's read, before any box): OMP_NUM_THREADS at the host's physical cores against the container's CPU allotment (P104, P105, P106)

**Why.** Since phase 3 (F19) every TC1 arm has run with `OMP_NUM_THREADS` set to the host's physical cores, and a rented container can be
held to far fewer. TC1c amendment 9's RunPod H100 pod was allotted 18 vCPUs and ran 72 threads. e4b's draws there came 18 % and 29 %
apart while Unsloth's held. Vast lists a 5090 rental's share of its host in `cpu_cores_effective` (for example 24 of 96 cores on machine
152440), and the busiest host this campaign used, 145701, shows 256 CPUs to a one-GPU container that ran 128 threads. If the allotment is
enforced as a CFS quota, idle OpenMP workers spin into it and the main thread is throttled. e4b's step is the host-bound one, so that would
cost e4b most. It would also be one cause of TC1 amendment 38's unread lead: e4b's step was 37–44 % shorter on a host whose container saw
48 CPUs than on the 256-CPU host. No box has recorded the allotment (#1196 adds it), so this registration is also the probe.

**The box** (token `qwen3ompab`). One RTX 5090, TC1's qwen3 tokens and field recipe, 60 steps, load-gated draws (`TC1_LOAD_GATE=6.0`,
`TC1_LOAD_RETRIES=2`):

- e4b's matched arm and Unsloth's matched arm, both in venv-unsloth (amendment 25's same-stack pair), each `_om0` (`OMP_NUM_THREADS` = the
  host's physical cores, as every box so far) against `_om1` (`OMP_NUM_THREADS` = the container's allotment: the cgroup v2 `cpu.max` quota,
  else the v1 CFS quota, in CPUs rounded up, capped by the affinity count), two draws a side in ABBA order;
- the lane computes the allotment before any install. A host whose allotment is not below its physical cores has no contrast. It refuses
  with rc 18 (a host floor), and the launcher's lane-refusal class names the machine for the next draw.

Engagement: each receipt's `arm_facts` records `omp_num_threads`, and torch's intra-op pool matches it. Both frameworks run on torch 2.12.*.
The scorer requires every `_om1` receipt of a framework to run fewer threads than every `_om0` receipt, else UNTESTED for want of contrast.

**Predictions** (registered before the box), one-sided:

- **P104** (e4b): `_om1` / `_om0` ≤ **0.97**, at least 3 % faster at its allotment.
- **P105** (Unsloth): `_om1` / `_om0` ≤ **1.01**, no slower at its allotment.
- **P106:** on each framework, |mean held-out at N, `_om1` − `_om0`| ≤ **0.005**.

Each needs two stable VALID draws a side. Each is FALSIFIED on the wrong side of its bound, and UNTESTED where a side is unstable, not VALID,
not engaged or without contrast.

**Decision rules.**

- **P104, P105 and P106 HELD:** later TC1 and TC1c boxes run every arm, of every framework, at the container's allotment. The registration
  that does so cites this read. STATUS says that every position quoted before it ran at the host's physical cores. A separate registration
  asks whether e4b should size torch's pool to the container by default, and later registrations re-read the quoted positions under the new
  policy. Until they do, the old positions stay quoted.
- **P104 FALSIFIED:** the thread count is not e4b's lever on that host. The harness keeps the physical cores. TC1c amendment 9's instability
  stays unexplained.
- **P104 HELD, P105 FALSIFIED:** the harness keeps the physical cores (a policy must hold for both frameworks), and the read reports both.
- **Any UNTESTED, none FALSIFIED:** a re-ask is allowed.
- **Three draws in a row refused for no contrast** (each on a new machine): the read records that those Vast containers carried no CPU quota
  below their physical cores, and this amendment closes without a box.
- No position against another framework is read here.

**Budget.** One RTX 5090 at the policy rate ($0.85/h), 4 h guard, TC1's 98 GB host floor, venv-unsloth built. About $2.5 with the download;
a refused draw costs about a cent. This is in the standing no-ask tier.
### Amendment 46 (2026-10-05T20:32Z, after RD1's read, before any box): grouped-nf4-gemm's decoded route against its fused kernels, A/B on one RTX 5090, OLMoE and Qwen3-30B-A3B (P107–P111)

**Why.** Lane RD1 read grouped-nf4-gemm's frozen-expert GEMM routes per call on one RTX 5090
(`bench/moegen/rd1/RESULTS-rd1.md`, experts4bit-qlora#1201).
- Its `decoded_cap` arm is `dequant_groups` for a chunk of groups, then one Triton grouped bf16 GEMM launch, with the decode transient capped at
  256 MiB. It was at most 0.85 of the best shipped route at seq 512 and 2048 on 4 of the 7 many-group families: DECODED HELD.
- Across RD1's many-group cells it beat the fused kernels wherever a call had at least 48 rows per present group (0.37–0.98 × v1), and lost
  at 37 or fewer (1.05–2.24 × v1).
- grouped-nf4-gemm#487 ships the route opt-in, as `GNF4_TRAIN_GEMM=decoded`, with `auto` untouched.
- RD1's decision names this box as the route's licence: a per-call win is not a step-time claim.

**The box's first step: the sm_120 gate.**
- **What it runs.** Before any arm, `tc1_decoded_gate` runs grouped-nf4-gemm's compiled tests for the route on the box's card. They run from
  a checkout at `GNF4_SHA` (the installed package's commit) in venv-e4b, the arms' venv:
  - `kernel/test_nf4_route.py -k "decoded or cap_bounds or dequant_groups"`: RD1's fp32-reference gate (the route's error at most 2× the
    dense route's on every call), capped == uncapped bit for bit, the cap's bound on the decode transient, and `dequant_groups`
    bit-equal to `dequant_ref`;
  - `kernel/test_nf4_route_decision.py`: `auto` never answers `decoded`.
- **What counts as passing.** Each run must also pass every test `DECGATE_REQUIRED` names for it. A grouped-nf4-gemm without the route has
  none of those tests, and `-k` alone would then select only the dequant tests and pass. That was rehearsed at gnf4 `054be19`, from before #487.
- **A failure refuses the box before any timing.** That covers a failed or missing test and a gate that cannot run: `BOX_REFUSED
  decoded-gate`, exit 19.
  - 19 is not one of the host-limited codes adertha admits as machine evidence (13, 14, 17, 18), because a kernel defect names no machine.
  - `decgate.json` is the record.
- **Rehearsed on the seat's RTX A2000** (correctness only, no timing):
  - #487's head passed (24 + 38 tests);
  - `054be19` was refused for missing tests;
  - a SHA that cannot be checked out was refused.

**The box.** One RTX 5090, venv-e4b (torch 2.8.0+cu128, triton 3.4.0, the software RD1 measured), `TC1_STEPS=60`.
- **Draws:** load-gated (`TC1_LOAD_GATE=6.0`, `TC1_LOAD_RETRIES=2`), avoiding machines 145701, 151350, 45511 and 138786.
- **Pins:** grouped-nf4-gemm at #487's merge; e4b at a main that carries this amendment.
- **Two families:** each runs the fused kernels (`_dec0`, `GNF4_TRAIN_GEMM=fused`) against the decoded route (`_dec1`, `=decoded`,
  `GNF4_DECODED_MAX_BYTES` at its 256 MiB default).
  - Arm: the matched arm (fp32 adapters, matched init), two draws a side in ABBA order, every arm resident, every other setting at its default.
  - Side 0 is what `auto` runs on these calls today, which all have more than 16 present groups.
  - **`olmoedecab`:** OLMoE-1B-7B-0924-Instruct at TC2's pin, through `tc1_prepare` (TC1's field recipe). It has 64 experts, top-8: a
    field micro-batch's median of about 512 positions puts about 64 rows on each expert, above RD1's line.
  - **`qwen3decab`:** Qwen3-30B-A3B, TC1's qwen3 tokens. It has 128 experts, top-8: about 32 rows per expert at the median, below the line.

Engagement is read from each arm's `route_ab` record:
- `_dec1` resolves `decoded` and counts decoded forward and dgrad calls, with no dense forward;
- `_dec0` resolves `fused` and counts no decoded forward (the counter is present, as at #487) and no dense forward;
- every arm's `env.torch` is 2.8.*.

**Predictions** (registered before the box):
- **P107** (the gate, structural): the sm_120 gate passes.
  - FALSIFIED iff it ran and a test failed. The box then ends and P108–P111 are UNTESTED.
  - HELD iff every run exited 0 with no failure, error or skip, and every required test passed.
  - UNTESTED otherwise.
- **P108** (OLMoE): dec1/dec0 s/step ≤ **0.95**, on the median AND on every one of the four cross-draw ratios.
  - P108's HELD moves a default, so it is bounded at the upper end of the instrument's own interval: a median at or below 0.95 with any
    cross-draw ratio above it reads FALSIFIED (this bound was set at review, 2026-10-05, before any box).
  - The expectation is about 0.85: RD1 read 0.79 (skewed) and 0.73 (uniform) × v1 per call at seq 512, and the expert GEMMs (forward,
    recompute, dgrad) are a large part of this step but not all of it.
- **P109** (Qwen3-30B-A3B): dec1/dec0 lies in **[0.97, 1.25]**.
  - Below the line no win of 3 % is expected. Nor should the loss exceed RD1's per-call reading at 512 rows (1.05 skewed, 1.35 uniform)
    on an expert-dominated step.
- **P110:** on each family, |mean held-out at N, dec1 − dec0| ≤ **0.01**.
  - Both routes decode the same bytes; they differ only in the GEMM's accumulation order (matched-set EQUIVALENT).
- **P111:** on each family, matched peak dec1 − dec0 ≤ **+0.30 GB**.
  - The route's decode transient is at most 256 MiB a call; RD1 read 180–327 MiB above the inputs on the many-group shapes.

P108–P111 each need two stable VALID draws a side. They are FALSIFIED outside their bound, and UNTESTED where a side is unstable, not VALID
or not engaged.

**Decision rules.**
- **P107, P108, P110 and P111 HELD:** grouped-nf4-gemm's `auto` takes the decoded route on compute capability (12, 0), the card measured
  (RTX 5090), for a call with more than `DENSE_AUTO_MAX_GROUPS` present groups and at least 48 rows per present group (RD1's per-call line).
  - That is grouped-nf4-gemm's own PR, citing RD1 and this box.
  - Other capabilities, other 12.x parts included, keep today's route until they are measured; so do calls below the line.
    (The scope was narrowed from 12.x to (12, 0) at review, 2026-10-05, before any box.)
  - If P109 is FALSIFIED below 0.97 (Qwen3-30B-A3B gains below the line too), the same PR still lands at 48, which never moves a call that
    lost per call, and a lower line is registered as a re-ask.
  - If P109 is FALSIFIED above 1.25, the read names it; it does not change this rule, which keeps those calls fused.
- **P108 FALSIFIED:** `auto` is unchanged and the route stays opt-in.
- **P110 or P111 FALSIFIED:** `auto` is unchanged, and the read names the family.
- **P107 FALSIFIED:** the box was refused before any arm. The route is fixed in grouped-nf4-gemm before any re-draw.
- **Any UNTESTED, none FALSIFIED:** `auto` is unchanged pending a re-ask.
- No position against another framework is read here.

**Budget.** One RTX 5090 at the policy rate ($0.85/h), a 4 h guard, and TC1's 98 GB host floor.
- No Unsloth venv is built for these tokens.
- About $3 with both downloads (OLMoE about 14 GB, Qwen3-30B-A3B about 61 GB). This is in the standing no-ask tier.

### Amendment 47 (2026-10-06T00:25Z, after amendments 43, 44 and 45's reads, before any box): where e4b's packed-row memory goes, against Unsloth's — the census on packed 4,096-token rows (P112, P113, P114)

**Why.** On packed 4,096-token rows e4b with its chunked LM loss peaked at 32.47 GB against Unsloth's 24.86 (amendment 43), about 1.1 GB
under the RTX 5090's capacity. `auto` (amendment 44, #1203) makes those rows e4b's default path, so that headroom is now the margin on every
long-row run. Part of the 7.6 GB gap is known: TC1's e4b arms run the library default, the fp32 expert absmax, which amendment 23 measured
1.35 GB above Unsloth's double-quantized one. The rest scales with tokens per micro-batch, because amendment 23's field-recipe transient gap
was 0.43 GB. Amendment 23's largest e4b transients were grouped-nf4-gemm's padded LoRA delta, a zero-padded fp32 input block and its
products. Those blocks grow with the hottest expert's rows, and at 4,096 tokens they reach the delta's 2 GiB pad limit (amendments 40 and
43 saw the loop that limit triggers). grouped-nf4-gemm's opt-in compact delta (`NF4_QLORA_COMPACT_DELTA=1`) was built to shrink that
transient. It took 0.29 GB off the field recipe's matched peak (amendment 37), where the block is small.

**The box** (token `qwen3memc4k`). One RTX 5090, amendment 39's packed rows (`TC1_PACK=1`, seq 4,096, micro-batch 1 × accum 4,
`TC1_FREE_OUTPUTS=1`), the matched set, 20 steps, one draw per arm in this order, each with the harness's memory census on
(`--mem-census 1`), every arm in venv-unsloth:

- e4b `fused_attn4_m_p4`: the library's defaults (fp32 absmax, the padded delta, the chunked LM loss as `auto`);
- e4b `fused_attn4_m_p4_lev`: `E4B_ABSMAX_DQ=1` + `NF4_QLORA_COMPACT_DELTA=1`, e4b's two memory levers;
- Unsloth `ckpt_unsloth_m_p4`: TC1's qwen3 Unsloth arm (grouped_mm).

Validity: amendment 39's packed-row predicates. On e4b, the levers its tag names (`absmax_dq`, grouped-nf4-gemm's `gnf4_compact_delta`), and
chunked training forwards with no fallback. The per-expert loop is a recorded route up to 5 % of a step's delta calls (amendment 43). The
census slows the step, so no speed is read.

**Predictions** (registered before the box):

- **P112** (the instrument): the census attributes at least 90 % of each arm's peak to named groups, as amendment 23's P42 did.
- **P113:** e4b's defaults peak **5.0 to 10.0 GB** above Unsloth's (peak allocated). The band was set with amendment 43's 7.6 GB, a
  `peak_vram_gb` reading, in view.
- **P114:** with both levers, e4b's peak is at most **3.0 GB** above Unsloth's: the absmax and the padded delta are most of the gap.

**Decision rules.** This is a measurement, not a position. The read names each excess's largest class and groups at the peak.

- **P114 HELD:** the levers close most of the packed gap. A registered A/B reads the compact delta's speed on packed rows, the regime
  where it matters, and asks for a size-gated default the way amendment 44 did for the chunked loss.
- **P114 FALSIFIED:** the read names what is left, and the next registration targets it.
- Positions stay with the boxes that read them.

**Budget.** One RTX 5090 at the policy rate ($0.85/h), 3 h guard, TC1's 98 GB host floor, venv-unsloth built. About $1.5 with the download;
this is in the standing no-ask tier.

### Amendment 48 (2026-10-06T03:19Z, after amendment 47's read, before any box): grouped-nf4-gemm's bucketed LoRA-delta padding on packed rows (P115–P118)

**Why.** Amendment 47's census put 6.10 GB of e4b's packed-row excess over Unsloth in grouped-nf4-gemm's padded LoRA delta. The delta pads
every expert's rows to the hottest expert's count: about 380,000 padded rows against 32,768 routed rows at the down projection. The compact
delta left the same-width block in place. grouped-nf4-gemm#490 adds `NF4_QLORA_PAD_BUCKETS=1` (opt-in). It sorts the groups by rows, cuts
buckets in which the widest group has at most twice the narrowest's rows, and pads each bucket only to its own widest, so the padded rows are
at most twice the real rows. Values are equal to rounding (the `bmm` shapes differ). On an RTX A2000, on the delta alone at the down
projection's shape (fp32 adapters, 32,768 Zipf-skewed rows), it took the delta's forward-and-backward peak from 8.77 to 1.06 GiB. That is
an engineering check on #490, not a TC1 reading.

**The box** (token `qwen3padbk`). One RTX 5090, amendment 39's packed rows (`TC1_PACK=1`, seq 4,096, micro-batch 1 × accum 4,
`TC1_FREE_OUTPUTS=1`), 40 load-gated steps (`TC1_LOAD_GATE=6.0`, `TC1_LOAD_RETRIES=2`), every arm in venv-unsloth at e4b's defaults (the
chunked LM loss as `auto` chunks these rows), grouped-nf4-gemm at a main that has #490:

- the shipped and the matched arm, each `_pk0` (`NF4_QLORA_PAD_BUCKETS=0`, one padded block) against `_pk1` (`=1`, buckets), two draws a side
  in ABBA order.

Engagement: grouped-nf4-gemm's per-path counters on each receipt (`lean_ab.lora_path_calls`). `_pk1` arms make bucketed calls and no
single-block padded call; `_pk0` arms the reverse. The chunked loss serves the packed rows on every arm, and `env.torch` is 2.12.*. The
per-expert loop is a recorded route up to 5 % of a step's delta calls (amendment 43). #490 leaves `auto`'s route rule sizing the single
block, so the same calls take the loop on both sides.

**Predictions** (registered before the box), one-sided:

- **P115** (matched peak): `_pk0` − `_pk1` ≥ **3.0 GB**. Buckets take at least 3 GB off the fp32 arm's peak.
- **P116** (matched): `_pk1` / `_pk0` ≤ **1.02**.
- **P117** (shipped): `_pk1` / `_pk0` ≤ **1.02**.
- **P118:** on each arm, |mean held-out at N, `_pk1` − `_pk0`| ≤ **0.005**.

Each needs two stable VALID draws a side. It is FALSIFIED on the wrong side of its bound and UNTESTED where a side is unstable, not VALID or
not engaged.

**Decision rules.**

- **P115–P118 HELD:** buckets buy the packed regime without a speed or quality cost there. Before they can be a default, the field recipe's
  short rows must be shown not to pay for the extra launches (a few more `bmm`s per projection on a host-bound step). A size-gated `auto`
  (bucket only where the single block would be large) and its field-recipe A/B are the next registration, as amendment 44 did for the
  chunked loss.
- **P115 FALSIFIED:** buckets do not buy the packed regime's memory on a full step. They stay opt-in, and the read says where the peak went.
- **P116 or P117 FALSIFIED:** they cost speed on packed rows. They stay opt-in, and the read names the arm.
- **Any UNTESTED, none FALSIFIED:** a re-ask is allowed.
- No position against another framework is read here.

**Budget.** One RTX 5090 at the policy rate ($0.85/h), 4 h guard, TC1's 98 GB host floor, venv-unsloth built. About $3 with the download;
this is in the standing no-ask tier.

**The re-ask** (2026-10-06T05:14Z, after the first box's read, before it runs). The first box, `tc1-5090-101`, read every `_pk1` arm VOID. The
arm's `lora_loop_share` divided the loop's calls by the sum of loop, padded and grouped_mm calls, and grouped-nf4-gemm#490's `padded_bucketed`
calls were not in that sum, so the loop read 1.000 where it served 1.5 % of calls. `tc1_arm.py` now sums every `lora_path_*` counter. The
re-ask is the same box (token, arms, order, steps, gate, predictions and decision rules unchanged) on the fixed arm, on a host other than
130223.

### Amendment 49 (2026-10-06T07:21Z, after amendment 48's re-ask read, before any box): grouped-nf4-gemm's bucketed padding at the field recipe (P119–P122)

**Why.** Amendment 48's re-ask (#1249) read `NF4_QLORA_PAD_BUCKETS=1` on packed 4,096-token rows with all four predictions HELD. The matched
arm stepped 0.893 of the single padded block's time with its peak 4.29 GB lower, and the shipped arm stepped 0.933. Its decision rule left
one question before a default: whether TC1's field recipe pays for the extra `bmm` launches buckets add (a few per projection), on a step
that is host-bound at about 1,000–1,400 real tokens. That rule anticipated a size-gated `auto` and its field-recipe A/B. This registration
asks the simpler question first: whether buckets cost the field recipe anything at all, applied everywhere. If they do not, a gate buys
nothing, and the default needs none. If they do, the gate is registered next as amendment 48 anticipated.

**The box** (token `qwen3fieldbk`). Amendment 41's box design with buckets in place of the chunked loss. One RTX 5090, TC1's qwen3 tokens and
field recipe (seq 2,048, micro-batch 2 × accum 4), 60 load-gated steps (`TC1_LOAD_GATE=6.0`, `TC1_LOAD_RETRIES=2`), every arm in
venv-unsloth at e4b's defaults, grouped-nf4-gemm at a main that has #490, avoiding machine 145701:

- the shipped and the matched arm, each `_fb0` (`NF4_QLORA_PAD_BUCKETS=0`) against `_fb1` (`=1`), two draws a side in ABBA order.

Engagement: grouped-nf4-gemm's per-path counters on each receipt. `_fb1` arms make bucketed calls and no single-block padded call; `_fb0`
arms the reverse. `env.torch` is 2.12.*. TC1's no-loop rule applies as written: the field recipe never takes the loop.

**Predictions** (registered before the box), one-sided:

- **P119** (matched): `_fb1` / `_fb0` ≤ **1.01**.
- **P120** (shipped): `_fb1` / `_fb0` ≤ **1.01**.
- **P121** (matched peak): `_fb1` − `_fb0` ≤ **+0.05 GB**.
- **P122:** on each arm, |mean held-out at N, `_fb1` − `_fb0`| ≤ **0.005**.

Each needs two stable VALID draws a side, is FALSIFIED on the wrong side of its bound, and is UNTESTED where a side is unstable, not VALID
or not engaged.

**Decision rules.**

- **P119–P122 HELD, with amendment 48's re-ask HELD:** bucketed padding becomes grouped-nf4-gemm's default (`NF4_QLORA_PAD_BUCKETS` unset
  means buckets; `0` keeps the single block), in one grouped-nf4-gemm PR citing amendments 47, 48 and 49. e4b picks it up through its
  grouped-nf4-gemm floor.
- **P119 or P120 FALSIFIED:** buckets cost the field recipe. They stay opt-in, and a size-gated `auto` (buckets only where the single
  block would be large) and its field-recipe A/B are registered next, as amendment 48 anticipated.
- **P121 or P122 FALSIFIED:** they stay opt-in, and the read names the arm.
- **Any UNTESTED, none FALSIFIED:** they stay opt-in pending a re-ask.
- No position against another framework is read here.

**Budget.** One RTX 5090 at the policy rate ($0.85/h), 4 h guard, TC1's 98 GB host floor, venv-unsloth built. About $2.5 with the download;
this is in the standing no-ask tier.

**The re-ask** (2026-10-06T08:48Z, after the first box's read, before it runs). The first box, `tc1-5090-103`, read P119-P122 UNTESTED: the single
block's draws were 12-21 % apart. The re-ask is the same box (token, arms, order, steps, gate, predictions and decision rules unchanged) on
another host, with one instrument added on every arm, both sides alike: `TC1_PAD_CENSUS=1`. It records each grouped-LoRA delta call's single
padded block (rows and bytes, per projection) from host facts, at a few microseconds of host arithmetic per call. Those sizes are what a
size-gated `auto` would be placed between, if P119 or P120 falls on the slow side.

### Amendment 50 (2026-10-06T10:45Z, after amendment 49's re-ask read, before any box): bucketed padding as `auto` — its default decision (P123–P125)

**Why.** Amendment 48's re-ask read grouped-nf4-gemm's bucketed padding on packed 4,096-token rows: 0.893 of the single block's step on
the matched arm, 4.29 GB off its peak, 0.933 on the shipped arm, all HELD. Amendment 49 and its re-ask could not settle its speed at TC1's
field recipe: the single block's own draws came 8–21 % apart on two hosts. The re-ask's census (49,152 delta calls per arm) measured every
field call at 9,040 routed rows at most (median 3,968). Routed rows are a call's tokens × top-k, a property of the batch's shape, not of the
router: TC1's largest field micro-batch is 1,130 positions (× 8 = 9,040). Every packed call carries 4,096 × 8 = 32,768.
grouped-nf4-gemm#491 adds `NF4_QLORA_PAD_BUCKETS=auto`, which buckets a call only when it carries at least 16,384 routed rows. Under it the
field recipe runs the single block op for op (#491's tests pin a call under the gate as the single block bit for bit), and packed rows run
the buckets amendment 48 measured. This box reads the field side's structure.

**The box** (token `qwen3fieldauto`). Amendment 49's box with `auto` in place of `1`. One RTX 5090, TC1's qwen3 tokens and field recipe, 60
load-gated steps, every arm in venv-unsloth at e4b's defaults, grouped-nf4-gemm at a main that has #491, avoiding machines 145701, 130223 and
55583:

- the shipped and the matched arm, each `_fa0` (`NF4_QLORA_PAD_BUCKETS=0`) against `_fa1` (`=auto`), two draws a side in ABBA order.

Engagement: each receipt records the bucket mode grouped-nf4-gemm resolved (`gnf4_pad_buckets_mode`, `auto` on `_fa1` with its row gate,
`0` on `_fa0`; the arm adds this record) and the per-path counters. The single-block path serves the delta on every arm; `env.torch` 2.12.*.

**Predictions** (registered before the box). They read structure and values from the VALID receipts, with no draw-stability requirement:

- **P123** (the gate): every `_fa1` arm makes 0 bucketed calls and as many single-block calls as its arm's `_fa0` receipt. FALSIFIED iff a
  VALID `_fa1` arm bucketed a call; HELD iff all four are VALID and do neither.
- **P124:** on each arm, |mean held-out at N, `_fa1` − `_fa0`| ≤ **0.005**.
- **P125** (matched peak): the median of `_fa1`'s two peaks − the median of `_fa0`'s ≤ **+0.05 GB**.

**Speed is reported, not scored.** Under P123 both sides run the same delta ops; `_fa1` adds one integer comparison per call. Amendment 49's
two boxes showed this baseline's own draws 8–21 % apart at the field recipe, so a speed band here would read the baseline's noise, not the
gate. The read reports both sides' medians.

**Decision rules.**

- **P123–P125 HELD, with amendment 48's re-ask HELD:** `auto` becomes grouped-nf4-gemm's default (`NF4_QLORA_PAD_BUCKETS` unset means
  `auto`; `0` keeps the single block everywhere; `1` buckets every call), in one grouped-nf4-gemm PR citing amendments 47, 48, 49 and 50.
- **P123 FALSIFIED:** the gate fired at the field recipe, which the shapes say it cannot. `auto` stays opt-in, and the read finds the call.
- **P124 or P125 FALSIFIED:** `auto` stays opt-in, and the read names the arm.
- **Any UNTESTED, none FALSIFIED:** `auto` stays opt-in pending a re-ask.

**What this box can and cannot show** (maintainer, 2026-10-06T10:48Z, before the box).
- The shapes and #491's tests make P123–P125 close to certain, so these are an **integration check**, not a test of an
  uncertain hypothesis. What it can catch: `auto` not reaching grouped-nf4-gemm in the real training loop (hence the
  resolved-mode record), a field call larger than the census saw, or a peak or held-out change the shapes don't predict.
- The default flip changes behaviour only on calls of **≥ 16,384 routed rows**. The evidence for those calls is
  amendment 48's: one model (Qwen3-30B-A3B, top-8), packed 4,096-token rows, one RTX 5090. A recipe of another model
  that crosses the gate flips on grouped-nf4-gemm's correctness tests alone. For example, gpt-oss at 4,096 tokens ×
  top-4 is exactly 16,384, and the gate's `>=` fires there.
- So the default-flip PR carries #1250's conditions: the changelog states that evidence scope; the bucketed path's
  correctness tests cover every expert geometry grouped-nf4-gemm ships for; and the flip lands in a release, with
  `NF4_QLORA_PAD_BUCKETS=0` documented as the way back.
- **Order.** This registration merges after the amendment 49 re-ask read (#1258) and grouped-nf4-gemm#491 are on main;
  it rests on both.

**Budget.** One RTX 5090 at the policy rate ($0.85/h), 4 h guard, TC1's 98 GB host floor, venv-unsloth built. About $1.5 with the download;
this is in the standing no-ask tier.

### Amendment 51 (2026-10-06T12:29Z, after amendment 50's read, before any box): the packed 4,096-token position on one stack at e4b's defaults (P126–P129)

**Why.** On packed 4,096-token rows the register holds two Qwen3-30B-A3B readings. Amendment 39's: e4b at its then-defaults ran out of
memory at step 1, an e4b loss. Amendment 43's: a labelled 1.278 with `E4B_CHUNKED_LM_LOSS=1` set by hand. Since then two defaults changed by
registered rule. The chunked LM loss runs as `auto` (amendment 44, #1203), and on these rows every forward chunks. grouped-nf4-gemm's
bucketed padding runs as `auto` (amendment 50, grouped-nf4-gemm#492), and on these rows every delta call buckets. Amendment 48 read buckets
at 0.893 of the single block's step and 4.29 GB lighter. Nothing now needs to be set for e4b to train these rows. This box reads the
position at e4b's defaults.

**The box** (token `qwen3samestack4kd`). Amendment 43's box with nothing set (no `TC1_E4B_ENV`). One RTX 5090, amendment 39's packed rows
(`TC1_PACK=1`, seq 4,096, micro-batch 1 × accum 4, `TC1_FREE_OUTPUTS=1`), 40 load-gated steps, TC1 amendment 25's same-stack arms: e4b's
matched arm in venv-unsloth and in venv-e4b (`_t28`), Unsloth's matched arm, two draws each, e4b's reference skipped, e4b and
grouped-nf4-gemm at mains that have #1203 and #492, avoiding machines 145701, 130223 and 55583.

Engagement on every e4b arm: both defaults served the rows by themselves. The chunked loss ran with `E4B_CHUNKED_LM_LOSS` unset (chunked
forwards, no fallback). Bucketed padding ran with `NF4_QLORA_PAD_BUCKETS` unset, grouped-nf4-gemm resolved `auto`, and every padded call
was bucketed. The per-expert loop is a recorded route up to 5 % of a step's calls (amendment 43).

**Predictions** (registered before the box). The bands were set with amendment 43's 1.278 and 0.915 and amendment 48's 0.893 and 28.23 GB
in view; the registration says so.

- **P126:** Unsloth/e4b on one stack in **[1.25, 1.80]**.
- **P127:** e4b's environment ratio (venv-unsloth / venv-e4b) in **[0.84, 0.98]**.
- **P128:** every e4b arm that ran completed resident (no OOM).
- **P129:** e4b's matched arm (venv-unsloth) peaks at most **29.5 GB** (the median of its two draws).

P126 and P127 need two stable VALID draws a side, as amendment 25 reads them.

**Decision rules.**

- **P126 read with both pairs stable and P128 HELD, whichever side it favours:** the ratio is recorded as Qwen3-30B-A3B's packed
  4,096-token position at e4b's defaults (`e4b.train.h2h.unsloth.qwen3.5090.<date>.packed-4k-defaults`). It supersedes amendment 39's
  out-of-memory row as the default-settings reading. That row stays as the record of the code before #1203 and #492. Amendment 43's labelled
  row stays as the reading with the chunked loss set by hand and no buckets.
- **P128 FALSIFIED:** e4b at its defaults does not fit these rows on this host. The read says where, and amendment 39's row stays the
  default-settings reading.
- **P129** is a memory reading beside the position; it moves no default.

**Which defaults, exactly** (maintainer, 2026-10-06T12:31Z, before the box).
- The box measures e4b at its defaults with grouped-nf4-gemm at a **main** carrying #492. Until grouped-nf4-gemm releases #492 and
  experts4bit-qlora's floor moves to that release, a user installing e4b does not get bucketed padding by default.
- So the row names the grouped-nf4-gemm commit (or release) the box ran, and its claim says "with grouped-nf4-gemm <version>".
- STATUS presents it as the default-settings position only once e4b's floor carries the release with #492. Until then it reads "at
  e4b's defaults with grouped-nf4-gemm main <sha>, unreleased".
- The box launches after grouped-nf4-gemm#492 merges. Your waiter checks that.

**Budget.** One RTX 5090 at the policy rate ($0.85/h), 4 h guard, TC1's 98 GB host floor, venv-unsloth built. About $2 with the download;
this is in the standing no-ask tier.

### Amendment 52 (2026-10-06T14:27Z, after amendment 51's read, before any box): bucketed padding on packed rows in the field image's torch 2.8 (P130–P133)

**Why.** Amendment 51 read e4b at its defaults on packed rows at 10.06 s/step in venv-unsloth (torch 2.12.1, triton 3.7.1), and at 13.62
in the field image's venv-e4b (torch 2.8.0, triton 3.4). That is an environment ratio of 0.739; P127 was FALSIFIED against [0.84, 0.98].
Amendment 43 read the same ratio at 0.915 with the chunked loss and no buckets. Between the two boxes e4b in torch 2.12 got faster and e4b
in torch 2.8 slower. They ran on different hosts, so that comparison is a lead, not a reading. Bucketed padding (grouped-nf4-gemm#490,
`auto` by default since #492) is what changed. Amendments 48 and 50 read it in torch 2.12 only, and the default applies to every torch.

**The box** (token `qwen3padbk28`). Amendment 48's box in venv-e4b. One RTX 5090, packed 4,096-token rows, 40 load-gated steps, e4b's
defaults otherwise (the chunked loss `auto`), grouped-nf4-gemm at the box's pin, avoiding machines 145701, 130223 and 55583:

- the shipped and the matched arm, each `_k0` (`NF4_QLORA_PAD_BUCKETS=0`) against `_k1` (`=1`), two draws a side in ABBA order, every arm in
  venv-e4b.

Engagement: as amendment 48, with `env.torch` 2.8.*.

**Predictions** (registered before the box), one-sided, as amendment 48 registered them:

- **P130** (matched): `_k1` / `_k0` ≤ **1.02**.
- **P131** (shipped): `_k1` / `_k0` ≤ **1.02**.
- **P132** (matched peak): `_k0` − `_k1` ≥ **3.0 GB**.
- **P133:** on each arm, |mean held-out at N, `_k1` − `_k0`| ≤ **0.005**.

**Decision rules.**

- **P130–P133 HELD:** buckets cost torch 2.8 nothing on packed rows. P127's gap lies elsewhere, and the read says where to look next.
- **P130 or P131 FALSIFIED:** buckets cost torch 2.8 on packed rows. grouped-nf4-gemm's `auto` default then needs a gate on that
  environment or a fix. The read names the arm, and the next registration is the gate or the fix. Until that registration reads, the
  read itself adds one line to e4b's `docs/STATUS.md` and grouped-nf4-gemm's `docs/STATUS.md`: packed-row training under torch 2.8
  should set `NF4_QLORA_PAD_BUCKETS=0`, with the measured ratio. That line is docs only, and the default does not change on it.
- **P132 or P133 FALSIFIED:** the read names it; the torch-2.12 default is not touched by a torch-2.8 reading alone. A P133 miss is
  a numerics question, not a speed one, so the next step is a $0 correctness check of the bucketed delta under torch 2.8 / triton 3.4
  on the A2000, before any rented box.
- **Any UNTESTED, none FALSIFIED:** a re-ask is allowed.

**Budget.** One RTX 5090 at the policy rate ($0.85/h), 4 h guard, TC1's 98 GB host floor. About $1.5 with the download; this is in the
standing no-ask tier.

### Amendment 53 (2026-10-06T17:47Z, after amendment 52's read, before any box): where torch 2.8's extra time goes at e4b's defaults on packed rows (P134–P136)

**Why.** Amendment 51 read e4b at its defaults on packed rows at 0.739 of its torch-2.12 step in the field image's torch 2.8 (P127
FALSIFIED). Amendment 52 then showed that bucketed padding is not that cost. In torch 2.8 the buckets step 0.983 (matched) and 0.939
(shipped) of the single block's step. Its GPU traces, reported and not scored, point elsewhere. Over each arm's training window the
median nvidia-smi utilisation was:

- matched arm: 97 % on the single block, 87 % with buckets;
- shipped arm: 97 % and 96 %;
- amendment 48's torch-2.12 arms: 96–98 % with buckets, on both arms and both hosts;
- amendment 51's host, e4b at its defaults: 97–98 % in torch 2.12, 75–76 % in torch 2.8.

A diagnostic count on an RTX A2000 (sm_86; it counts, it does not time) found:

- the bucketed delta issuing the same kernels under torch 2.8 and torch 2.11: 115 per forward and backward of both projections, against
  the single block's 66;
- no added synchronisation and no cudaMalloc / cudaFree;
- the same counts with fp32 and bf16 adapters.

So the idle time is not an extra launch, sync or allocation that sm_86 shows. This box asks the 5090 where torch 2.8's added time goes.

**The box** (token `qwen3prof28`). One RTX 5090, packed 4,096-token rows, 40 load-gated steps. The matched arm (fp32 adapters, matched
init) with e4b's defaults otherwise, avoiding machines 145701, 130223 and 55583. Three sides, two draws each, in A B C C B A order:

- `q212`: venv-unsloth (torch 2.12), buckets `auto`;
- `q28`: venv-e4b (torch 2.8), buckets `auto`;
- `q28k0`: venv-e4b, `NF4_QLORA_PAD_BUCKETS=0` (the single block).

Every arm carries amendment 12's profile instrument: three warm steps, then steps 3–5 under torch.profiler with CPU and CUDA activity.
The timed s/step stays the median of steps 11..40, so the profiled steps sit outside it. There is no per-micro-batch timing, since its
syncs would drain the queue the profile reads.

Engagement (`prof28_why`): the torch the side names. `q212` / `q28` must have `NF4_QLORA_PAD_BUCKETS` unset, resolved `auto`, and every
padded call bucketed. `q28k0` must have it set to 0, resolved 0, with single-block calls and none bucketed. The chunked LM loss must serve
the packed rows with its variable unset. The per-expert loop is a recorded route (≤ 5 %). The profile summary is not part of validity: an
arm without one is VALID, and its P135 / P136 reading is UNTESTED.

**Predictions** (registered before the box):

- **P134:** s/step `q212` / `q28` ≤ **0.92**. The environment ratio at the defaults; amendment 43 read 0.915 without buckets, amendment
  51 0.739 with them.
- **P135:** D = timed `q28` − timed `q212`, the per-step time torch 2.8 adds: the median s/step of steps 11..40 in ms, as medians of two
  draws. The share of D that is not device time, 1 − (device `q28` − device `q212`) / D, is ≥ **0.5**. D ≤ 0 leaves nothing to attribute
  (UNTESTED). Device time is the profiler's device self time per profiled step, summed over streams.
- **P136:** the device busy fraction of `q28` ≤ `q28k0`'s − **0.03**. The fraction is device ms per profiled step over the TIMED ms per
  step.
- **Why the timed step, not the profiled wall** (maintainer review, before any box). torch.profiler adds host overhead to every op it
  records, and that overhead differs between torch versions and grows with the launch count. The bucketed delta issues 115 kernels to
  the single block's 66 on the A2000 count. A profiled wall would therefore push P135 and P136 toward HELD from the instrument alone.
  The profile's device time comes from CUPTI kernel durations, which the profiler does not stretch. The profiled wall and the busy
  fraction against it are reported beside each reading, never scored.

Reported, not scored:

- per arm, the CPU self ms per profiled step by op family, and the device ms by family;
- the family deltas `q28` − `q212` and `q28` − `q28k0`, largest increase first (`prof28_family_deltas`);
- the nvidia-smi utilisation medians.

**Decision rules.**

- **P135 HELD:** the added time is host-side. The read names the CPU op family whose self time grows most from `q212` to `q28`, and the
  next registration targets it. If that is the batched matmul's host path (`aten::bmm` and its launches), the candidate is a
  shape-stable bucket ladder in grouped-nf4-gemm, read on packed rows in both torches. The ladder rounds bucket widths up to a fixed set,
  so repeated calls see repeated shapes.
- **P135 FALSIFIED:** the added time is device time. The read names the device family that grows most, and the next registration targets
  it. Candidates are triton 3.4's code for grouped-nf4-gemm's kernels against triton 3.7's, and the fp32 batched GEMMs of cuBLAS in cu128
  against cu130.
- **P134 FALSIFIED:** on this host, torch 2.8 at the defaults is no further behind than amendment 43's ratio without buckets. Amendment
  51's 0.739 is then host-dependent, and STATUS says so beside it. P135 and P136 are still read.
- **P136 FALSIFIED:** under the profiler the buckets do not lower torch 2.8's device busy share. Amendment 52's nvidia-smi reading stays
  an observation, and the read says which instrument disagrees.
- No default changes on this box; it is a diagnostic. **Any UNTESTED, none FALSIFIED:** a re-ask is allowed.

**Budget.** One RTX 5090 at the policy rate ($0.85/h), 4 h guard, TC1's 98 GB host floor. Six arms: about $2 with the download.

*Note (2026-10-06, before amendment 53's box reads):* the RTX A2000 count in amendment 53's *Why* is committed with its script, as a
diagnostic that licenses nothing: [`../h2h-2026-10-02/tc1/a2000-bucket-counts/`](../h2h-2026-10-02/tc1/a2000-bucket-counts/README.md)
(`bench/tc1/bucket_count.py`). Its millisecond fields are not readings. One correction to the *Why*: fp32 and bf16 adapters give close
counts, not the same ones. With buckets, fp32 runs 115.3 kernels / 113.3 launch calls per iteration and bf16 112.5–113.7 / 111.7. The
torch-version comparison stands: identical with fp32 adapters, and equal launch calls with bf16. Nothing registered changes.

### Amendment 54 (2026-10-06T20:04Z, after amendment 53's read, before any box): two remedies for torch 2.8's host time at e4b's defaults (P137–P140)

**Why.** Amendment 53 read e4b's matched arm at its defaults on packed rows at 0.790 of its torch-2.12 speed in torch 2.8. 59.7 % of the
added 2.63 s per step is not device time, and the CPU family that grew most is the batched matmul. `aten::bmm` makes the same ~26,750
calls a step in both torches, about 23,750 of them added by the buckets. Profiled, it takes about 268 µs of self time per call
in torch 2.8 against 86 µs in torch 2.12. Two earlier readings bear on why:

- **Amendment 24** (RTX 5090, a replay with no model): an fp32 `bmm` costs 119 µs of host time on a shape the process has not used,
  against 38 µs on a repeated one, under torch 2.8 and 2.12 alike. The router gives nearly every bucket a new shape: the bucket's group
  count and width both move from call to call.
- **A diagnostic count on an RTX A2000** (cuBLAS 12.8, sm_86, counts only;
  [`../h2h-2026-10-02/tc1/a2000-cublas-api/`](../h2h-2026-10-02/tc1/a2000-cublas-api/README.md)): every fp32 `bmm` asks cuBLASLt's
  heuristic for an algorithm, gets 0 back, and falls back. A bf16 call gets 21. That is the same for repeated and new shapes.

Torch 2.12 ships cuBLAS 13 (cu130), torch 2.8 cuBLAS 12.8 (cu128). In training, torch 2.12's 86 µs is below the replay's 119 µs for a new
shape, so torch 2.12 is not paying the new-shape cost on every call, and torch 2.8 pays more than it. (268 µs is profiled CPU self time,
which also counts waits on a full launch queue, so it is an upper bound on host work and this comparison is a lead.) One reading of that: cuBLASLt's
heuristics cache (8,192 entries by default) holds the step's shapes under cuBLAS 13 and thrashes under 12.8. This box puts two remedies
against the defaults, one per reading, without settling the mechanism first:

- `c1`: `CUBLASLT_HEURISTICS_CACHE_CAPACITY=262144`, a bigger cache, no code change;
- `c2`: grouped-nf4-gemm's bucket ladder (`NF4_QLORA_PAD_BUCKETS_LADDER=1`, grouped-nf4-gemm#498). It rounds each bucket's width and
  group count up to quarter-octave rungs, so shapes repeat: over 40 Zipf(1) routings of 4,096 tokens, 227 distinct bucket shapes
  become 41.

**The box** (token `qwen3ladder28`). One RTX 5090, packed 4,096-token rows, 40 load-gated steps. The matched arm (fp32 adapters, matched
init) in venv-e4b (torch 2.8), e4b's defaults otherwise, avoiding machines 145701, 130223 and 55583. Three sides, two draws each, in
A B C C B A order:

- `c0`: the defaults;
- `c1`: the cuBLASLt cache raised to 262,144 entries;
- `c2`: the ladder.

Every arm carries amendment 12's profile instrument (steps 3–5, outside the timed steps 11..40), so the read can report `aten::bmm`'s
host time per call on each side.

Engagement (`ladder28_why`): torch 2.8; buckets `auto` with every padded call bucketed; the chunked loss serving the rows unset; and the
one change the side names, and no other. `c0` sets neither, `c1` only the cache, `c2` only the ladder, resolved on, by a
grouped-nf4-gemm that has it. The receipt records both settings (`lean_ab`).

**Predictions** (registered before the box), one-sided:

- **P137:** s/step `c2` / `c0` ≤ **0.95**.
- **P138:** s/step `c1` / `c0` ≤ **0.97**.
- **P139:** on `c1` and `c2`, |mean held-out at N − `c0`'s| ≤ **0.005**.
- **P140:** `c2`'s matched peak ≤ `c0`'s + **1.5 GB** (medians of two draws). The ladder pads at most 1.56× the buckets' rows.

Reported, not scored: each side's `aten::bmm` self time per profiled call and the CPU family deltas against `c0`, and the nvidia-smi
utilisation medians.

**Decision rules.**

- **P137, P139 and P140 HELD:** the ladder recovers torch 2.8's host time. The next registration is its default decision, read in torch
  2.12 and on the shipped arm, on packed rows and at the field recipe.
- **P138 HELD with P139:** the bigger cache recovers it with no code change. e4b's STATUS and grouped-nf4-gemm's STATUS each gain one line
  for torch 2.8 users, with the measured ratio, and the next registration asks whether e4b should set it for them.
- **P137 and P138 both FALSIFIED:** shape novelty is not the in-training cost. The read reports both sides' `aten::bmm` time per call,
  and the next candidate is fewer, wider buckets: fewer calls rather than repeated shapes.
- **P140 FALSIFIED with P137 HELD:** the ladder's speed costs memory. The read reports both, and its default question waits for the
  memory to be read on the shipped arm.
- **P139 FALSIFIED on `c2`** (maintainer review, before any box): the ladder pads with zero groups and should not move training at all, so
  a held-out shift is a defect signal, not a trade-off. Before any default question, a $0 correctness check of the laddered delta against
  the unladdered one comes first: values and gradients, under torch 2.8 / triton 3.4 on the RTX A2000.
- No default changes on this box. **Any UNTESTED, none FALSIFIED:** a re-ask is allowed.

**Budget.** One RTX 5090 at the policy rate ($0.85/h), 4 h guard, TC1's 98 GB host floor. Six arms: about $2 with the download.

### Amendment 55 (2026-10-06T22:11Z, after amendment 54's read, before any box): amendment 47's packed-row memory census at the current defaults (P141–P143)

**Why.** At e4b's defaults on packed 4,096-token rows the matched arm peaks at 28.23 GB against Unsloth's 24.86 (amendment 51, a
`peak_vram_gb` reading): 3.37 GB more. Amendment 47's census read the gap before bucketed padding, at +7.47 GB of peak allocated. It
found 1.35 GB in the fp32 expert absmax and 6.10 GB in transients, almost all grouped-nf4-gemm's padded LoRA delta (the input block and
its products). Buckets took 4.29 GB off the matched peak (amendment 48), and became grouped-nf4-gemm's default (#492). What is left of
the gap has not been attributed. Neither lever amendment 47 read can be reused as it stood:

- the compact delta is a single-block body, and with buckets on, the buckets win, so it no longer applies;
- the double-quantized expert absmax (`E4B_ABSMAX_DQ=1`) is the trainer's default (`python -m experts4bit_qlora.train`) but not the
  library's, which TC1's arms run.

This box names the remaining excess before any lever is built for it.

**The box** (token `qwen3memc4kb`). Amendment 47's box at the current code. One RTX 5090, packed rows (`TC1_PACK=1`, seq 4,096,
micro-batch 1 × accum 4, `TC1_FREE_OUTPUTS=1`), the matched set, 20 steps. One draw per arm, in this order, each with the memory census
on (`--mem-census 1`), every arm in venv-unsloth:

- e4b `fused_attn4_m_p4d`: the library's defaults. That is the fp32 absmax, bucketed padding `auto`, and the chunked LM loss `auto`.
- e4b `fused_attn4_m_p4d_dq`: the same with `E4B_ABSMAX_DQ=1`.
- Unsloth `ckpt_unsloth_m_p4d`: TC1's qwen3 Unsloth arm (grouped_mm).

Validity (`memc4kb_why`) applies amendment 39's packed-row predicates and amendment 47's census and torch checks. On e4b it also
requires:

- the absmax the tag names;
- the delta not compact;
- every padded call bucketed under `auto` with its variable unset;
- the chunked loss serving the rows unset.

The per-expert loop is a recorded route (≤ 5 %). No speed is read.

**Predictions** (registered before the box):

- **P141** (the instrument): the census attributes at least 90 % of each arm's peak to named groups.
- **P142:** e4b's defaults peak **2.0 to 5.0 GB** above Unsloth's (peak allocated). The band was set with amendment 51's 3.37 GB, a
  `peak_vram_gb` reading, in view.
- **P143:** with the double-quantized absmax, e4b's peak is at most **2.5 GB** above Unsloth's.

**Decision rules.** This is a measurement, not a position. The read names the excess's largest class at the peak and e4b's largest
live groups.

- **If the bucketed delta's blocks are the largest remaining excess**, the next registration is a compact bucketed delta in
  grouped-nf4-gemm. That delta would save each bucket's input rather than its padded block. It is read on packed rows for peak and
  speed.
- **P143 HELD:** the double-quantized absmax closes most of what is left. The next registration asks whether the library should default
  to it on packed rows, the way amendment 28 read its speed.
- **Otherwise** the read names what is left, and the next registration targets it.
- **P141 FALSIFIED on an arm** (maintainer review, before any box): the census can't account for that arm's peak, so its excess is
  not attributed. The read reports the unattributed share, and no lever registration follows from that arm's groups until the
  instrument is fixed and re-read.
- Positions stay with the boxes that read them.

**Budget.** One RTX 5090 at the policy rate ($0.85/h), 3 h guard, TC1's 98 GB host floor, venv-unsloth built. About $1.5 with the
download.

### Amendment 56 (2026-10-06T23:37Z, after amendment 55's read, before any box): the double-quantized absmax as a library default on packed rows, with each run's peak split by phase (P144–P148)

**Why.** Amendment 55's census held P143. With `E4B_ABSMAX_DQ=1`, e4b's packed-row peak is 2.01 GB above Unsloth's, against 3.36 GB at the
library's defaults. By its rule, the next registration asks whether the library should default to the double-quantized absmax on packed
rows. The trainer (`python -m experts4bit_qlora.train`) already does, for resident training, on amendments 28 and 31's field-recipe
readings: 1.4 % of the step for 1.34 GB on Qwen3-30B-A3B. Its speed on packed rows has not been read.

The census also found something else. e4b's run peak is its held-out evaluation, where the stock LM loss holds the full fp32 logits;
Unsloth's is a training backward. So the packed peak comparison so far set e4b's evaluation against Unsloth's training step. The census
keeps only the snapshot that set the run's peak, so e4b's training-phase peak has never been recorded. This box records it.

**The instrument** (`tc1_arm.py --phase-peaks 1`). The run's peak allocated is split into three phases:

- `setup`: the load, up to the first held-out evaluation;
- `eval`: every held-out evaluation;
- `train`: the training steps between them.

At each boundary the allocator's max is folded into its phase and reset, so each phase is read on its own and `peak_vram_gb` stays the
run's max. The receipt field is `peak_vram_gb_phases`. The flag is refused with `--mem-census 1`, whose snapshots read that max.

**The box** (token `qwen3dqpack`). One RTX 5090, packed 4,096-token rows, 40 load-gated steps, held-out at steps 0 and 40. The matched
arm in venv-unsloth (torch 2.12), with e4b's defaults otherwise, avoiding machines 145701, 130223 and 55583. In this order:

- e4b `fused_attn4_m_a0`: the fp32 expert absmax (the library's default);
- e4b `fused_attn4_m_a1`: `E4B_ABSMAX_DQ=1`;
- e4b `fused_attn4_m_a1_d2` and `fused_attn4_m_a0_d2`: their second draws (A B B A);
- Unsloth `ckpt_unsloth_m_pp`: TC1's qwen3 Unsloth arm (grouped_mm), one draw.

Every arm runs with `--phase-peaks 1`. Validity (`dqpack_why`): torch 2.12. On e4b it also requires the absmax the side names, every
padded call bucketed under `auto` with its variable unset, and the chunked loss serving the rows unset. The per-expert loop is a recorded
route (≤ 5 %). The phase record is not part of validity: without it, P147 and P148 read UNTESTED.

**Predictions** (registered before the box):

- **P144:** s/step `a1` / `a0` ≤ **1.02**.
- **P145:** the run's peak (median of each side's draws) falls by at least **1.2 GB** from `a0` to `a1`.
- **P146:** |mean held-out at N, `a1` − `a0`| ≤ **0.005**.
- **P147:** e4b `a1`'s training-phase peak (median of its draws) is at most **1.0 GB** above Unsloth's training-phase peak.
- **P148:** on every e4b draw the evaluation phase's peak exceeds the training phase's.

**Decision rules.**

- **P144, P145 and P146 HELD:** a library PR makes `enable_fast_train` compress the expert absmax by default for resident training,
  with the trainer's guards: off under expert offload and the training arena, a model the compressor refuses keeps its fp32 absmax, and
  `E4B_ABSMAX_DQ=0` turns it off. That PR states the evidence scope in its changelog and STATUS (maintainer review, before any box):
  one model, one RTX 5090, torch 2.12 / triton 3.7 (venv-unsloth). Under the field image's torch 2.8 the packed-row speed of the
  compressed absmax is unread here, just as bucketed padding's was until amendment 52.
- **P144 FALSIFIED:** the library default stays fp32. STATUS gives the packed-row cost beside the trainer's default.
- **P147 HELD:** STATUS says that on packed rows e4b's training-phase memory is within 1 GB of Unsloth's, and that the run-peak gap is
  e4b's evaluation. The next candidate is an opt-in chunked loss for no-grad evaluation. It has to be opt-in, because such a forward
  returns no logits and a caller computing metrics from them would break.
- **P147 FALSIFIED:** the read gives the training-phase gap, and the next registration is a census of the training phase.
- **P148 FALSIFIED:** amendment 55's finding does not hold on every draw, and the read says where the run peak fell instead.
- Positions stay with the boxes that read them. **Any UNTESTED, none FALSIFIED:** a re-ask is allowed.

**Budget.** One RTX 5090 at the policy rate ($0.85/h), 4 h guard, TC1's 98 GB host floor, venv-unsloth built. Five arms: about $2 with
the download.

### Amendment 57 (2026-10-07T01:36Z, after amendment 56's read, before any box): the memory census of the training phase on packed rows (P149–P152)

**Why.** Amendment 56 split each run's peak by phase. On packed 4,096-token rows, with the double-quantized absmax, e4b's training phase
peaks at 26.79 GB against Unsloth's 24.86: +1.92 GB (P147 FALSIFIED). By its rule, the next registration is a census of the training
phase. Amendment 55's census could not attribute it, because its snapshot was taken at the run's peak, which was the held-out evaluation
after step 20, 0.09 GB above training.

**The box** (token `qwen3memc4kt`). Amendment 47's census box (packed rows, `TC1_FREE_OUTPUTS=1`, the matched set, 20 steps, one draw
per arm, `--mem-census 1`, every arm in venv-unsloth), with `TC1_EVAL_EVERY` above `TC1_STEPS`. No evaluation then runs inside the
census window: the step-0 evaluation comes before training, with no optimizer state, and the final one after the census closes. The
family refuses to run otherwise. In this order:

- e4b `fused_attn4_m_p4t`: `E4B_ABSMAX_DQ=0`, the fp32 expert absmax;
- e4b `fused_attn4_m_p4t_dq`: `E4B_ABSMAX_DQ=1`;
- Unsloth `ckpt_unsloth_m_p4t`: TC1's qwen3 Unsloth arm (grouped_mm).

The absmax is set explicitly on both e4b arms, so the box reads the same whatever the library default is when it runs. Validity:
amendment 55's `memc4kb_why` (the absmax the tag names, the bucketed default, the chunked loss unset, the census present). No speed is
read.

**Predictions** (registered before the box):

- **P149:** the census attributes at least 90 % of each arm's peak.
- **P150:** e4b fp32's peak allocated is **2.0 to 4.5 GB** above Unsloth's, set with amendment 56's 3.28 GB (phase peaks) in view.
- **P151:** e4b absmax-dq's peak is at most **2.5 GB** above Unsloth's (amendment 56: +1.92).
- **P152:** every arm's census peak falls in a training step (a phase `s<N>.mb<M>.forward|backward|optimizer`, never an evaluation).

**Decision rules.** This is a measurement, not a position. The read names the excess's largest class and e4b's largest live groups at the
training peak, and the next registration targets the largest e4b-only group, whichever it is.

- **P152 FALSIFIED:** the census did not read the training phase. The read says where the peak fell, and the box is re-asked with that
  phase excluded.
- **P149 FALSIFIED on an arm** (maintainer review, before any box), as amendment 55: that arm's excess is not attributed. The read
  reports the unattributed share, and no lever registration follows from its groups until the instrument is fixed and re-read.
- Positions stay with the boxes that read them. **Any UNTESTED, none FALSIFIED:** a re-ask is allowed.

**Budget.** One RTX 5090 at the policy rate ($0.85/h), 3 h guard, TC1's 98 GB host floor, venv-unsloth built. About $1.5 with the
download.

### Amendment 58 (2026-10-07T04:36Z, after amendment 57's read, before any box): checkpoint inputs in pinned host memory on packed rows (P153–P157)

**Why.** Amendment 57's census put e4b's 1.91 GB training-phase excess over Unsloth (packed rows, double-quantized absmax) in three
places:
- each checkpointed decoder layer's input kept on the GPU, 47 × 16.8 MB = 0.79 GB, the largest single group;
- the routed-expert combine's backward, 1.07 GB;
- grouped-nf4-gemm's bucketed delta block, 0.90 GB.

By its rule, the next registration targets the first. `E4B_CKPT_OFFLOAD=1` (#1298) keeps those inputs in pinned host memory: the reentrant
checkpoint inside `save_on_cpu`. Its gradients equal the default checkpointing's exactly on CPU, and on CUDA through e4b's fused experts.
Separately, #1296 cut the combine backward's temporaries by 0.40 GB with the same bytes, measured on its own on an RTX A2000. Both are on
main for this box, so the read also says what the combine change did to the training peak.

**The box** (token `qwen3ckptoff`). One RTX 5090, packed 4,096-token rows, 40 load-gated steps, held-out at 0 and 40. The matched arm in
venv-unsloth (torch 2.12), at e4b's defaults otherwise: the double-quantized absmax, buckets `auto`, the chunked loss `auto`. Avoiding
machines 145701, 130223 and 55583. In this order:
- e4b `fused_attn4_m_o0`: `E4B_CKPT_OFFLOAD=0`;
- e4b `fused_attn4_m_o1`: `E4B_CKPT_OFFLOAD=1`;
- their second draws, `_o1_d2` then `_o0_d2` (A B B A);
- Unsloth `ckpt_unsloth_m_oo`: one draw.

Every arm runs with `--phase-peaks 1`. Validity (`ckptoff_why`) requires torch 2.12 and e4b's defaults as recorded: the absmax compressed,
every padded call bucketed, the chunked loss serving the rows. It also requires the checkpoint the side names: `o1` routes all 48 layers
(the receipt's `ckpt_offload_layers`), `o0` routes none.

**Predictions** (registered before the box):
- **P153:** e4b's training-phase peak (median of the draws) falls by at least **0.6 GB** from `o0` to `o1`.
- **P154:** s/step `o1` / `o0` ≤ **1.05**. The copies are synchronous, one per layer each way.
- **P155:** |mean held-out at N, `o1` − `o0`| ≤ **0.005**.
- **P156:** `o0`'s training-phase peak is at most **1.7 GB** above Unsloth's. Amendment 57 read +1.92 before #1296.
- **P157:** `o1`'s training-phase peak is at most **1.0 GB** above Unsloth's.

**Decision rules.**
- **P153, P154 and P155 HELD:** the next registration reads it at TC1's field recipe, where each layer's input is smaller and the copies
  cost relatively more, before any default.
- **P154 FALSIFIED with P153 HELD:** the saving costs too much as written. The next step makes the copies asynchronous (a side stream
  and an event), and the box is re-asked.
- **P153 FALSIFIED:** the read says what the training peak held instead.
- **P155 FALSIFIED** (maintainer review, before any box): the offloaded checkpoint's gradients equal the in-GPU checkpoint's exactly
  (#1298's tests), so a held-out shift is a defect signal, not a trade-off. Before any further registration, a $0 correctness check of
  the offloaded path against the plain one comes first, at the packed-row shape on the RTX A2000.
- **P157 HELD:** STATUS says that with it, e4b's packed-row training memory is within 1 GB of Unsloth's.
- No default changes on this box. **Any UNTESTED, none FALSIFIED:** a re-ask is allowed.

**Budget.** One RTX 5090 at the policy rate ($0.85/h), 4 h guard, TC1's 98 GB host floor, venv-unsloth built. Five arms: about $2 with
the download.

### Amendment 59 (2026-10-07T06:07Z, after amendment 58's read, before any box): checkpoint inputs in pinned host memory at the field recipe (P158–P161)

**Why.** Amendment 58's box read `E4B_CKPT_OFFLOAD=1` on packed rows. P153, P154 and P155 HELD: the training-phase peak fell 0.739 GB
at 1.003× the step, and held-out moved +0.0002. By its rule, this registration reads the switch at TC1's field recipe before any default.
There each layer's input is a few hundred tokens, so the copies are small, but there are as many of them a step and the step is shorter.
The field box this reads against (amendment 50's, box 107) padded its micro-batches to 67–565 tokens, median 260. So the 47 inputs kept
across a step at the longest micro-batch are 2 × 565 × 2,048 × 2 bytes = 4.6 MB each, about 0.22 GB, against 16.8 MB on packed rows.

**The box** (token `qwen3ckptofff`). One RTX 5090 at TC1's field recipe: Qwen3-30B-A3B, seq 2048, micro-batch 2 × accum 4, 60
load-gated steps, held-out every 20. Every arm in venv-unsloth (torch 2.12), at e4b's defaults otherwise: the double-quantized absmax,
buckets `auto`, the chunked loss `auto` (both stay stock at this recipe, amendments 44 and 50). Avoiding machines 145701, 130223 and
55583. In this order:
- e4b `fused_attn4_shipped_f0` then `_f1`: the shipped arm, `E4B_CKPT_OFFLOAD=0` then `=1`;
- e4b `fused_attn4_m_f0` then `_f1`: the matched arm, the same pair;
- their second draws, `m_f1_d2`, `m_f0_d2`, `shipped_f1_d2`, `shipped_f0_d2` (A B B A on each arm).

Every arm runs with `--phase-peaks 1`. Validity (`ckptoff_why`, its field-recipe form) requires torch 2.12, the absmax compressed, every
padded call through the single block with none bucketed, no chunked loss call, and the checkpoint the side names: `f1` routes all 48
layers (`ckpt_offload_layers`), `f0` routes none.

**Predictions** (registered before the box):
- **P158:** the matched arm's s/step `f1` / `f0` ≤ **1.01**, amendment 49's bar for a default at this recipe.
- **P159:** the shipped arm's s/step `f1` / `f0` ≤ **1.01**.
- **P160:** on each arm, |mean held-out at N, `f1` − `f0`| ≤ **0.005**.
- **P161:** the matched arm's training-phase peak (median of the draws) falls by at least **0.10 GB** from `f0` to `f1`.

**Decision rules.**
- **All four HELD:** before offloading becomes e4b's default (`E4B_CKPT_OFFLOAD` unset = on, `=0` off), one more read is required
  (maintainer review, before any box). This box and amendment 58 read torch 2.12 only. The default would also apply to the field
  image's torch 2.8, where amendments 53-54 found the step host-bound on some hosts, which is the regime where synchronous host copies
  cost the most. So the next registration reads `f0` against `f1` at the field recipe in venv-e4b (torch 2.8), with a premise gate: the
  `f0` arm's device busy fraction against its timed step must be at most 0.9, or the speed reading is UNTESTED, not HELD. The default
  flip then follows in a library PR that cites all three reads and states their scope: one model, RTX 5090, both torches.
- **P158 or P159 FALSIFIED, with P160 HELD:** the copies cost the short rows too much. The next step is a size gate,
  `E4B_CKPT_OFFLOAD=auto`: offload only a micro-batch above a token threshold, as the chunked loss and the bucketing do. Its own field read
  (auto never engages here) comes before it becomes the default.
- **P161 FALSIFIED, the rest HELD:** the field recipe's training peak is not set while the inputs are alive. The read says what holds it,
  and the size gate above is the next step.
- **P160 FALSIFIED:** as amendment 58: a $0 correctness check of the offloaded path against the plain one on the RTX A2000 comes first.
- No default changes on this box. **Any UNTESTED, none FALSIFIED:** a re-ask is allowed on another host.

**Budget.** One RTX 5090 at the policy rate ($0.85/h), 4 h guard, TC1's 98 GB host floor, venv-unsloth built. Eight field-recipe arms:
about $2 with the download.

### Amendment 60 (2026-10-07T06:29Z, after amendment 58's read, before any box): the held-out loss from the logits in chunks on packed rows (P162–P165)

**Why.** Amendment 58's read left e4b's packed-row run peak in its held-out evaluation: 26.877 GB on every e4b draw, above the training
phase's 26.59 GB (25.85 GB with `E4B_CKPT_OFFLOAD=1`). Unsloth's evaluation phase peaked at 21.85 GB. A `torch.no_grad` forward with labels
runs Hugging Face's loss, which upcasts the whole logits to fp32 and takes the log-softmax of the copy. On an RTX A2000 at Qwen3's
vocabulary and 4,096 tokens that is 4.637 GiB above the bf16 logits. `E4B_CHUNKED_EVAL_LOSS=1` (#1302, opt-in) runs such a forward without
labels, returns the stock logits and takes the loss from them in 512-token fp32 chunks. That is 0.580 GiB above the logits, with the loss
bit-identical at 2,048 and 4,096 tokens there (`bench/chunked-lm-loss/receipts/eval_loss_peak_a2000.json`). This box reads it in the run.

**The box** (token `qwen3evalce`). One RTX 5090, packed 4,096-token rows, 40 load-gated steps, held-out at 0 and 40 (amendment 58's
command). e4b's matched arm in venv-unsloth with `E4B_CKPT_OFFLOAD=1` and its defaults otherwise: the double-quantized absmax, buckets
`auto`, the chunked training loss `auto`. Avoiding machines 145701, 130223 and 55583. In this order:
- e4b `fused_attn4_m_e0`: `E4B_CHUNKED_EVAL_LOSS=0`;
- e4b `fused_attn4_m_e1`: `E4B_CHUNKED_EVAL_LOSS=1`;
- their second draws, `_e1_d2` then `_e0_d2` (A B B A);
- Unsloth `ckpt_unsloth_m_ee`: one draw.

Every arm runs with `--phase-peaks 1`. Validity (`evalce_why`) requires amendment 58's predicates for its `o1` side: torch 2.12, the
absmax compressed, every padded call bucketed, the chunked training loss serving the rows, all 48 layers offloaded. It also requires
the eval switch the side names: `e1` records `E4B_CHUNKED_EVAL_LOSS=1` with at least one held-out forward through it, and `e0` records
`0` with none.

**Predictions** (registered before the box):
- **P162:** the evaluation-phase peak (median of each side's draws) falls by at least **3.5 GB** from `e0` to `e1`. The A2000 measured
  4.06 GiB (4.36 GB) of the loss's own transient.
- **P163:** on every `e1` draw the evaluation-phase peak is below the training-phase peak, so the run peak is training's.
- **P164:** |step-0 held-out `e1` − `e0`| ≤ **0.0001** on each draw pair (`e1` / `e0`, `e1_d2` / `e0_d2`). Amendment 58's four e4b
  draws read 1.28851 alike at step 0, so a larger difference is the loss, not the run.
- **P165:** |mean held-out at N, `e1` − `e0`| ≤ **0.005**.

The run peak against Unsloth's is reported, not scored.

**Decision rules.**
- **All four HELD:** `E4B_CHUNKED_EVAL_LOSS` becomes on by default (unset = on above the gate, `0` off), in a library PR that cites this
  read. At TC1's field recipe the evaluation rows stay under the 1 GiB gate, so the field recipe's held-out is unchanged by construction.
  That PR states its scope (maintainer review, before any box): one model, one RTX 5090, torch 2.12. Its changelog also says that,
  above the gate, a user's held-out loss can move at the scale of fp32 summation order, bounded by this box's P164 and P165 readings,
  so evaluations compared across the flip are not byte-identical.
- **P164 or P165 FALSIFIED:** the loss differs by more than rounding, a defect signal. A $0 check of the full evaluation forward against
  stock on the RTX A2000 comes before any further registration.
- **P162 or P163 FALSIFIED:** the read says what holds the evaluation phase instead, and the switch stays opt-in.
- No default changes on this box. **Any UNTESTED, none FALSIFIED:** a re-ask is allowed.

**Budget.** One RTX 5090 at the policy rate ($0.85/h), 4 h guard, TC1's 98 GB host floor, venv-unsloth built. Five packed arms (amendment
58's box cost $0.82): about $1–2 with the download.

### Amendment 61 (2026-10-07T06:48Z, after amendment 58's read, before any box): the routed-expert combine over row chunks on packed rows (P166–P169), and a CUDA host floor

**Why.** Amendment 57's census put the routed-expert combine's backward among the groups above Unsloth's training peak on packed rows.
#1296 trimmed one fp32 copy from it. #1304 runs the combine's row-wise fp32 work (the forward's scatter and weight multiply, the
backward's weight-gradient sum and down-gradient scale) over row chunks of about 32 MiB once the whole `[tokens*k, hidden]` image would
reach 128 MiB, at least 16 rows a chunk. On an RTX A2000 at the packed shape it is byte-identical (forward and both gradients
`torch.equal`, 1,024- to 32,768-row chunks), and the combine backward's peak fell from 0.805 to 0.269 GB, the forward's transient from
768 to 320 MiB. It is on by default above the gate, so this box reads a shipped default: what it does to the training peak and the step.

**The box** (token `qwen3combck`). One RTX 5090, packed 4,096-token rows, 40 load-gated steps, held-out at 0 and 40 (amendment 58's
command). e4b's matched arm in venv-unsloth with `E4B_CKPT_OFFLOAD=1` and its defaults otherwise. Avoiding machines 145701, 130223 and
55583. In this order:
- e4b `fused_attn4_m_c0`: `E4B_COMBINE_CHUNK=0` (the whole-tensor combine);
- e4b `fused_attn4_m_c1`: the default (row chunks);
- their second draws, `_c1_d2` then `_c0_d2` (A B B A);
- Unsloth `ckpt_unsloth_m_cc`: one draw.

Every arm runs with `--phase-peaks 1`. Validity (`combck_why`) requires amendment 58's predicates for its `o1` side, plus the combine
the side names. `c1` records the variable unset with chunked forwards and backwards (the receipt's new `combine_chunk`); `c0` records `0`
with none.

**Predictions** (registered before the box):
- **P166:** the training-phase peak (median of each side's draws) falls by at least **0.3 GB** from `c0` to `c1`.
- **P167:** s/step `c1` / `c0` ≤ **1.02**, amendment 48's packed-row bar. The chunks add launches: 8 a call at this shape.
- **P168:** |step-0 held-out `c1` − `c0`| ≤ **0.0001** on each draw pair, and |mean held-out at N| ≤ **0.005**. The combine is
  byte-identical, so a larger step-0 difference is a defect signal.
- **P169:** `c1`'s training-phase peak is at most **0.7 GB** above Unsloth's. Amendment 58 read +0.983 with the offload alone.

**Decision rules.**
- **P166, P167 and P168 HELD:** the default stands, and STATUS gives the packed-row training gap with it.
- **P167 FALSIFIED:** the chunks cost the step too much. The next step raises the chunk size (fewer launches) and the box is re-asked.
  Until then, `E4B_COMBINE_CHUNK=0` is the documented way out.
- **P168 FALSIFIED:** a $0 A2000 check of the full combine at this shape comes first.
- **P166 FALSIFIED:** the read says what holds the training peak instead.
- **Any UNTESTED, none FALSIFIED:** a re-ask is allowed.

**A CUDA host floor (harness).** `tc1-5090-119` drew machine 34887 (driver 595.58). It passed the driver gate, then the image's torch
raised CUDA error 803 ("system has unsupported display driver / cuda driver combination"). Every venv inherits that torch, so the e4b
tripwire failed with rc 9, a harness code that names no machine, and the next draw could land on it again. `tc1_run.sh` now probes the
image's torch right after the driver gate. When torch imports but cannot use the GPU, the box refuses with code 18 (`BOX_REFUSED
cuda=unusable`), a registered host floor beside amendment 1's driver floor, so the receipt names the machine for exclusion. An image
without an importable torch is not refused there: that is the image, not the host. Nothing in any fixture, arm, predicate or band changes.

**Budget.** One RTX 5090 at the policy rate ($0.85/h), 4 h guard, TC1's 98 GB host floor, venv-unsloth built. Five packed arms: about
$1–2 with the download.
