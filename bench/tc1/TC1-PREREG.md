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
