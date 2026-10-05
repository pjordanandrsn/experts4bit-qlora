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

