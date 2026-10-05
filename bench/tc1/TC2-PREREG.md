# TC2 — the other families at matched work on the current cuts: Granite-3.1-3B-A800M, OLMoE-1B-7B, gpt-oss-20b (box A, `tc2small`, N 60) and Qwen3.6-35B-A3B, Mixtral-8x7B (box B, `tc2big`, N 20), e4b vs Unsloth 2026.9.14 vs HF 5.18/PEFT 0.21.2 vs axolotl 0.20.0 (registered 2026-10-01, before any box; family tokens `tc2small` (box A) and `tc2big` (`TC1_BOX=B`) of the TC1 harness, this file at `bench/tc1/TC2-PREREG.md`)

Issue: experts4bit-qlora#835. Lineage: tp4 (2026-09-10/19): Granite and Qwen3.6 VOID against Unsloth (its arm
adapted attention only: 5.2 M vs 99.6 M; 3.4 M vs 926 M), OLMoE's Unsloth arm died before its first step (a grouped
matmul fault on its transformers pin), gpt-oss REFUSED in every framework, Mixtral's Unsloth arm ran resident at
29.2 GB against e4b's 3.2 GB under offload (tp2). TC1's corrections (fp32 adapters in every arm, matched init, two
draws, the verdict column) apply here unchanged.

## What each family asks

| fam | the question this lane can answer | what moved since tp4 |
|---|---|---|
| granite | Unsloth 2026.9.14 never discovers nor quantises `GraniteMoeParallelExperts` (`input_linear`/`output_linear`; read from source: `get_moe_target_parameters` looks for gate_up/down names only, the bnb patch needs `gate_up_proj`+`down_proj`): the registered arm reproduces tp4's attention-only VOID, and a manual-`target_parameters` arm trains PEFT's weight-side fold over bf16 experts — the same regime as the HF arm, labelled so. The family's 4-bit-regime comparators are HF (bf16 experts, recorded) and axolotl (`quantize_moe_experts` quantises any 3-D "expert" parameter, so Granite's stacks DO go NF4 there) | Unsloth 2 releases; HF 5.18; axolotl 0.20.0 |
| olmoe | Unsloth calls OLMoE "a stacked-expert family Unsloth does not patch": its arm takes PEFT's dense fold or a generic route (UNVERIFIED) or dies — a row either way, with the engaged-path counters. HF's bf16-expert row was FLAGGED (0.055) at N=20: at N=60 with matched init, is it COMPARABLE? axolotl quantises OLMoE's stacks | same |
| gptoss | Unsloth's MXFP4 path keeps the experts packed under a 16-bit load (`load_in_4bit=False`, `UNSLOTH_MXFP4_KEEP_PACKED` default on; needs the grouped_mm backend, i.e. the torch-2.12 venv) and trains LoRA on them through a fused MXFP4 grouped GEMM; its 4-bit load trains per-expert `Linear4bit` through a Python loop. e4b REFUSES expert LoRA on this family (bare experts) and trains attention-only: an e4b LOSS recorded at full weight; the kernel package's experimental `ExpertsMxfp4LoRA` route is NOT a comparator (never licensed). axolotl: `quantize_moe_experts` on the transposed/biased gpt-oss stacks — whether its parametrisation handles `is_transposed` is a row, not a prediction | Unsloth 2026.9.14 (2026-10-01) |
| qwen3_5 | Qwen3.6-35B-A3B (256 routed + 1 shared expert, linear attention): can any framework adapt the routed experts in 4-bit besides e4b? Unsloth with explicit expert targets (not UT4); HF target_parameters by structure | same |
| mixtral | the footprint trade: e4b under expert offload (3.2 GB) vs Unsloth resident (29 GB) — speed ratio AND peak VRAM in one sentence; HF/axolotl resident (OOM expected) | same |

Fixture: TC1's field recipe (`bench/tc1/TC1-PREREG.md`: matched init, fp32 adapters, double-quant HF on / Unsloth and
axolotl off, the validity predicates, the verdict column, the equivalence bands, the cross-draw interval); N = 60 on box A
(the small families; the eval instrument 48 rows every 20 steps, as tp4 registered for them), N = 20 with 8 rows on box
B (tp4 amendment 2's instrument). Matched set + e4b as-shipped + e4b reference per family; two draws of the primary pair
where a position is possible (granite, olmoe: e4b and HF; gptoss: e4b attention-only and Unsloth packed-MXFP4; qwen3_5
and mixtral: e4b and Unsloth). Every Unsloth matched arm runs in `venv-unsloth` (torch 2.12.1+cu130, grouped_mm); the
HF `t214` arms run in the axolotl venv's torch 2.14 with `experts_implementation="grouped_mm"` (the dispatch recorded).

## Arms per family (tags as the harness names them)

granite: `e4b/fused_attn4_m`, `hf/hf_peft_m`, `e4b/reference_attn4_m` (third, the equivalence anchor), `e4b/fused_attn4_m_d2`,
`hf/hf_peft_m_d2`, `unsloth/ckpt_unsloth_m` (the notebooks' seven), `unsloth/ckpt_unsloth_m_experts`
(`q,k,v,o,input_linear,output_linear`, tp4 amendment 4's second arm: expected attention-only or a bf16-expert PEFT fold,
recorded), `hf/hf_peft_m_t214` (HF's native-best experts kernel on torch 2.14, bf16 experts, regime recorded),
`axolotl/ckpt_axolotl_m`, `axolotl/ckpt_axolotl_best` (ScatterMoE), `e4b/fused_attn4_shipped`.
olmoe: the same list without the `_experts` Unsloth arm.
gptoss: `e4b/fused_attn4_m` = a `refused` stub citing tp1/tp2 and the probe (as tp4), `e4b/attn_only_m` (x2),
`unsloth/ckpt_unsloth_m` (`load_in_4bit=True`: its per-expert Linear4bit loop, or REFUSED on the MXFP4 conversion as tp4
recorded — re-asked on the new release), `unsloth/ckpt_unsloth_mxfp4` (`load_in_4bit=False`: the 16-bit load that keeps the
MXFP4 experts packed and trains them through its fused MXFP4 grouped GEMM; x2; the packed-expert parameter class and the
engaged backend recorded), `hf/hf_peft_m` (expected REFUSED or OOM), `axolotl/ckpt_axolotl_m`; `e4b/reference_attn4_m`
REFUSED as tp4's. The trainable counts differ by construction (e4b adapts attention only; Unsloth the experts too):
**no ratio is quoted across different adapter sets** — the family's line reads both s/step values beside both counts, and
the row enters STATUS as an e4b limitation: no expert-LoRA path on gpt-oss where a competitor now has one.
qwen3_5: `e4b/fused_attn4_m` (x2), `unsloth/ckpt_unsloth_m` (x2; the attention-only `q,k,v,o` targets as tp4 registered for
this family), `unsloth/ckpt_unsloth_m_experts` (the routed stacks `mlp.experts.gate_up_proj,mlp.experts.down_proj` + q/k/v/o:
Unsloth's own resolver names them; its shared dense expert gets ordinary Linear4bit LoRA, so the trainable count is read,
not assumed), `hf/hf_peft_m`, `axolotl/ckpt_axolotl_m`, `axolotl/ckpt_axolotl_best`, `e4b/fused_attn4_shipped`,
`e4b/reference_attn4_m` last (the longest alarm).
mixtral: `e4b/fused_attn4_m` under expert offload (x2), `unsloth/ckpt_unsloth_m` resident (x2), `hf/hf_peft_m`,
`axolotl/ckpt_axolotl_m`, `axolotl/ckpt_axolotl_best`, `e4b/fused_attn4_shipped` (offload), `e4b/reference_attn4_m`
(offload) last; the footprint line (e4b's peak VRAM under offload against Unsloth's resident peak) leads the block.

Alarms: tp4's per-family ceilings (granite 1800/1800/1800/2400, olmoe 2400/2400/2400/3000, gptoss 3600/2400/2400,
qwen3_5 3600/3600/1800/5400, mixtral 5400/2400/1800/6000 for fused / Unsloth / HF / reference; axolotl = HF + 900);
box A guard 4.5 h, box B guard 5.5 h, both at the $0.54/h ceiling TC1 moved to (its two slow-link hosts sit above it).

## Validity, verdicts, readings: TC1's. Positions only within a family on one box, both arms VALID and stable.

## Predictions

- P1 granite: Unsloth adapts attention only on both target lists -> UNSUPPORTED for the 4-bit MoE regime (P10 of tp4
  amendment 4, re-asked on the new release); HF/e4b in [1.1, 1.6] (tp4: 1.29); axolotl ≈ HF or UNSUPPORTED.
- P2 olmoe: Unsloth's arm either engages (ratio in [1.2, 3]) or dies again (HARNESS_ERROR/UNSUPPORTED row); HF/e4b in
  [1.5, 2.5] (tp4: 1.95) and, with matched init, the N=60 quality reads COMPARABLE (tp4's 0.055 FLAG was init scale).
- P3 gptoss: Unsloth's packed-MXFP4 arm trains the experts (OK, Params MXFP4 recorded) — e4b has no expert path: the
  family's row is an e4b loss; e4b attn-only s/step vs Unsloth's full-expert s/step reported, never as a ratio.
- P4 qwen3_5: Unsloth with explicit expert names either engages the 256 routed experts (ratio in [2, 6]) or VOIDs on
  trainable count again; HF OOM; e4b fused_m ≈ tp4's 6.33 s/step within 15 %.
- P5 mixtral: e4b offload / Unsloth resident ratio in [0.3, 0.5] (tp2: 0.361) at a ≥ 8x lower e4b peak; HF and axolotl
  OOM resident on 32 GB.
- P6 e4b internal parity PASS on every family with both arms; P7 matched sets EQUIVALENT where two frameworks train
  the same adapter set.

## Decision rules

A family whose only VALID comparator is HF carries an HF position; a family where no framework but e4b trains the
4-bit MoE regime carries a coverage statement, not a ratio; gpt-oss's row enters STATUS as an e4b limitation beside the
wins. Nothing licenses; `training_support` cells move only through the read PR.

## Budget: two RTX 5090 boxes on Vast verified-secure, $0.54/h ceilings, 4.5 h + 5.5 h, estimates $2.43 + $2.97, each run < $15; a host below driver 580 refuses with code 18 (TC1 amendment 1) and the lane redraws once with it excluded.

## Amendments

(none yet)

### Amendment 1 (2026-10-02T04:59Z, before box B launched): box B runs from TC3 amendment 2's merge (C1 under expert offload)

Mixtral's e4b arms run under expert offload (`--offload 1`, as tp4). TC3's first 12 GB arm showed that the phase-3 C1 hasher counts
e4b's evicted expert stacks (0-element GPU placeholders; the bytes live in the offload handle's pinned-CPU `home`) as empty frozen
tensors and refuses the arm. TC3 amendment 2 makes the hasher read the home copy; box B (`tc2big`) launches from that merge so its
Mixtral e4b rows are measured, not refused. Box A (`tc1-5090-17`, no offload arm) is unaffected. Nothing else moves.

### Amendment 2 (2026-10-02T10:58Z, after box B's Mixtral Unsloth arms, before any redraw): the Mixtral pair re-asked with the Unsloth alarm at 7,200 s

**What box B showed.** On `tc1-5090-22` both Unsloth matched arms on Mixtral-8x7B (`ckpt_unsloth_m` and its second draw) exited with the
alarm's code (142) with no step taken: the arm was still loading at about 30 GB on the card when its registered 2,400 s ran out, twice.
e4b's arms on the same box loaded and trained (two draws under expert offload), HF OOMed at load, so P5's pair -- e4b under offload against
Unsloth resident -- has no Unsloth reading; by the lane's rules P5 is UNTESTED on that box, and that is what its read records. The 2,400 s
was tp2's figure for Unsloth 2026.9.2's loader on a different host; whether 2026.9.14's load of this 93 GB checkpoint completes at all
under a budget that is not the host's disk is the open question.

**The amendment.** One more box, token `tc2mixtral` on `TC1_BOX=B` (the same `tc2_big_family` call with the Unsloth alarm raised from
2,400 s to 7,200 s and nothing else changed): e4b `fused_attn4_m` under offload x2, Unsloth `ckpt_unsloth_m` x2 (`grouped_mm`, the registered
targets), the e4b reference arm under offload once; HF, both axolotl arms and e4b as shipped are not re-run (box B holds those rows) and
appear as `not_run` stubs. P5, P6 (Mixtral) and P7 (the Mixtral pair) are scored on this box; the position, if any, is within it (two
draws each, the 5 % stability rule). A second alarm reads as the finding: Unsloth's Mixtral load does not complete in two hours on a
verified-secure RTX 5090 host, said so, no position. Budget: one RTX 5090, ceiling $0.54/h, guard 8 h, estimate under $5; the standing
no-ask tier. The launch waits for box B's teardown and for the launcher checkout's pending update, and the manifest pins both heads.

### Amendment 3 (2026-10-02T12:15Z, after box B was lost, before either re-run launches): box B's receipts died with its instance; the Qwen3.6 half re-runs on its own box, and the Mixtral redraw box gets a host-RAM floor

**What happened.** Box B (`tc1-5090-22`) stopped answering at 11:42Z during its Mixtral Unsloth micro-batch-1 secondary, with every
arm but that one and HF's secondary complete on the box. The provider's listing then read the instance as exited and stopped by the host
(`intended_status: stopped`, memory use 85 % at the stop, its last duration ending about 11:41Z), the ssh gateway refused, and the
controller ran out its deadline with no answer; the guard's teardown destroys the instance and the receipts on its disk with it. The
controller's own log holds only the status of each cell, never its numbers, so nothing from that box is registered. The session's
automation was not permitted to restart the stopped instance to fetch them; the loss is recorded in the private store's receipt for the run.

**The amendment.** Two boxes, each under the standing no-ask tier, launched together:

- `tc1-5090-24`, token `tc2big` with `TC1_SKIP=mixtral` (the harness's registered skip knob, forwarded by the driver; the Mixtral family
  appears as `not_run` stubs): Qwen3.6-35B-A3B exactly as box B ran it -- e4b fused x2, Unsloth with explicit expert names x2 and the
  `m_experts` arm, HF, both axolotl arms, e4b as shipped, the e4b reference, and the micro-batch-1 secondary of any framework whose matched
  arm OOMs. P4, P6 (Qwen3.6) and P7 (its pairs) are scored on it.
- `tc1-5090-23`, token `tc2mixtral` as amendment 2 registers it, with one addition to the box specification: the host must offer at least
  192 GB of RAM (box B's manifest asked for 98 GB). Unsloth's loader materialises the 93 GB bf16 checkpoint in host memory before it
  quantises, and a host that stops the container at that point cannot answer the question amendment 2 asks. If no verified-secure RTX
  5090 host at the ceiling offers that much RAM, the box is not drawn and the Mixtral position stays open, said so.

Nothing in the harness, the arms, the alarms or the readings moves beyond what amendment 2 registered; the budgets are box B's (/bin/zsh.54/h
ceiling, 4.5 h, under .11) and amendment 2's (8 h guard, under ).

### Amendment 4 (2026-10-02T17:29Z, after the box B re-run read, before any box): Qwen3.6-35B-A3B's matched set with e4b under expert offload

**What the re-run showed.** On `tc1-5090-27` the matched set (926,187,520 fp32 adapters over 20,520 slots) OOMed e4b's fused path resident
on the 32 GB card at both micro-batches and in its reference loop, while Unsloth 2026.9.14, given the family's expert target parameters,
trained it resident at 30.47 GB and 10.59 s/step (one draw). The read registered it as an e4b loss and named the row that was not asked:
e4b under expert offload, the lever e4b ships for exactly this case (lane TC3 measured it on Qwen3-30B-A3B: EQUIVALENT-TO-RESIDENT).

**The amendment.** Token `tc2qwen35off` on `TC1_BOX=B`: the same `tc2_big_family` call for Qwen3.6 with `--offload 1` on every e4b arm
(the e4b alarm 5,400 s, the reference 7,200 s) and the matched Unsloth arm given the expert target parameters on both draws
(`TC2_UNS_TARGET_PARAMS`), so the reducer's ordinary matched pair is e4b-under-offload against Unsloth-resident at the same 926 M
parameters. HF, both axolotl arms and e4b as shipped are `not_run` stubs (`tc1-5090-27` holds them). Host RAM floor 192 GB (e4b's offload
pins the expert stacks in host memory; box B was lost to a 98 GB host).

**Predictions** (registered before the box is drawn; each read off a line `tc1_reduce.py` already prints for that box, no new
reducer code: P8 from `e4b/fused_attn4_m`'s VERDICT and peak, P9 from the MATCHED POSITION line for Unsloth or its NO-POSITION reason,
P10 from the equivalence line and the e4b internal-parity line):
- **P8**: e4b's fused path under offload completes the matched set (VALID) at a peak under 24 GB.
- **P9**: Unsloth/e4b s/step in [0.4, 1.0] -- Unsloth faster per step, e4b paying the expert stream -- with both arms' draws within 5 %.
  A ratio outside the band refutes; an unstable pair is UNTESTED.
- **P10**: the matched pair EQUIVALENT at N = 20 (the lane's band), and e4b's parity control under offload PASSES.

**Decision rules.** P8 holds -> `e4b.train.h2h.unsloth.qwen3_5.5090.2026-10-02`'s resident OOM stands as measured and a new row
registers the offload fit; a stable P9 pair is a position on this family, quoted with its footprint and with "e4b under offload" in its
name. P8 refuted -> the e4b loss on this family stands, now with the lever tried. Budget: one RTX 5090, ceiling $0.69/h (the pool's
cheapest verified host was $0.57/h at the box B re-run), 4.5 h, estimate under $3.20; the standing no-ask tier.

### Amendment 5 (2026-10-02T17:49Z, a correction, before any re-run): the Mixtral Unsloth readings were this harness's counters inside Unsloth's compiled MoE block

**What was published.** The box B re-run read (`e4b.train.h2h.unsloth.mixtral.5090.2026-10-02` and its `.compile-warmup` row) reported
that Unsloth 2026.9.14 spends its first six Mixtral steps compiling kernels (76-89 minutes, about 5,000 compile batches per arm), then
steps in 6.2 / 7.0 s at about 32 GB with its two draws 12 % apart, and offered a cause only as a candidate.

**What the receipts and the source say** (unsloth 2026.9.14, unsloth_zoo 2026.9.9, transformers 5.5.0 in the Unsloth venv, torch 2.12.1,
read 2026-10-02). The harness's engagement counters were Python dict increments wrapped around `torch._grouped_mm` and Unsloth's MoE
backend functions. Unsloth compiles Mixtral's `MixtralSparseMoeBlock` (it wraps Qwen3's MoE block in
`torch_compiler_disable_unless_decode`, `unsloth_zoo/temporary_patches/qwen3_moe.py:92`, so Qwen3's runs uncompiled) and raises Dynamo's
recompile limit from 8 to 1024 (`unsloth_zoo/patching_utils.py:262-263`). Inside the compiled block each counter is a side effect:
Dynamo froze its current value into a guard, so every call failed the guard and recompiled, and inside Unsloth's
`_GroupedMMRecompute` autograd function the increment was a graph break. The receipts carry it: 3,547 unique graphs, 743 graph breaks
"HOP: Unsafe side effect ... Attempted to mutate ConstDictVariable()", two "recompile limit exceeded (1024)" messages, inductor
`fxgraph_cache_hit` 4,988 against 36 misses (the same graphs re-traced), on both draws; after the limit the affected frames ran eagerly.
Unsloth documents the same failure for its own call counter (`moe_utils.py:1153-1157`, off by default). Every other Unsloth arm of the
campaign is clean on the same counters: Qwen3-30B-A3B on the 5090, H100 and 4090 boxes and in TC1b 4 frames and 20 graphs with no graph
break; Qwen3.6 6 frames; OLMoE 29 frames with 2 breaks of Unsloth's own `compiler.disable` and no growth from step 10 to 60; gpt-oss 39.
**The Mixtral Unsloth numbers are the harness's, not Unsloth's**: both Mixtral rows are retired, P7 (decided by Unsloth's second Mixtral
draw) is withdrawn to UNTESTED, and P5 stays UNTESTED. e4b's Mixtral offload readings were not affected (its path is not compiled) and
are restated by the re-run below.

**The fix** (`bench/tc1/tc1_arm.py`, `TracedCounts`). A counted call bumps a registered custom op (`tc1::bump`) on a module-global CPU
counter tensor when Dynamo is tracing, and a plain integer otherwise; each wrapper holds only an int slot. To Dynamo the op is an ordinary
mutating op: no guard on the count, no graph break, executed on every run of the compiled graph. Tested under `torch.compile` with an
autograd.Function inside activation checkpointing (the shape of Unsloth's path): exact counts including the checkpoint replay, zero
graph breaks, no per-call recompile, where the dict counter graph-breaks. Eager paths (e4b's, Qwen3's MoE block) keep the plain
increment. Every Dynamo snapshot now also records frames, unique graphs, graph breaks and recompile-limit hits: torch 2.12 never fills
the `recompiles` key the receipt's `recompiles_total` read, which is why it said 0.

**The re-run.** Amendment 2's `tc2mixtral` token from this amendment's merge, on a host with at least 192 GB of RAM (amendment 3): e4b
`fused_attn4_m` under offload x2, Unsloth `ckpt_unsloth_m` x2, the e4b reference under offload. P5, P6 and P7 are scored on it as
registered. A Mixtral Unsloth arm that still recompiles per call, with the counters inert, is the finding and is quoted with its frame
counts. Budget: one RTX 5090, ceiling $0.69/h, 6 h (the launcher's cap), under $4.20; the standing no-ask tier.

### Amendment 6 (2026-10-04T10:54Z, after TC1 amendments 10-15 and TC1c, before any box): both big families with e4b RESIDENT on the 32 GB card

**Why now.** On 2026-10-02 e4b trained neither big family resident on the RTX 5090. Qwen3.6's matched set (926,187,520 fp32 adapters)
OOMed e4b's fused path at both micro-batches (`tc1-5090-27`, 32.52 GB at the failing allocation), and Mixtral only ever ran under
expert offload, where Unsloth, resident at about 29 GB, steps about 5.3x faster (`e4b.train.footprint.unsloth.mixtral.5090.2026-10-02`).
Both were registered as e4b losses on the footprint/speed trade. Since then e4b's training memory changed: the lean LoRA delta is on by
default, and `_ScatterCombine` saves the bf16 `down` instead of its fp32 copy (23.75 MB per MoE layer at 380 tokens on Qwen3-30B-A3B).
Whether either family now fits resident is the open question, and the resident pair is the comparison the footprint row stands in for.
A rough budget for Mixtral (not a measurement): about 26.7 GB of weights, including 2.8 GB of fp32 absmax, about 2.2 GB of adapters,
gradients and 8-bit optimizer state, plus activations under checkpointing. That is at the card's edge.

**The token.** `tc2resident` on `TC1_BOX=B`. It makes the same `tc2_big_family` calls with `--offload 0` on every e4b arm: Mixtral first,
then Qwen3.6 with the matched Unsloth arms given the family's expert target parameters (amendment 4's pair). Each family runs e4b
`fused_attn4_m` x2, Unsloth `ckpt_unsloth_m` x2 (grouped_mm) and the e4b reference. A primary arm that OOMs falls to the `_mb1` pair (micro-batch 1 x
accum 8, the same tokens per step), as in every family. HF, both axolotl arms and e4b as shipped are `not_run` stubs (box B and
`tc1-5090-27` hold them). The reducer no longer draws Mixtral's footprint line, or scores P5 (the offload pair's prediction), when
e4b's anchor ran resident. Host RAM floor 192 GB (amendment 3).

**Predictions** (registered before the box is drawn; each read off a line `tc1_reduce.py` prints for the box):
- **P11** (Mixtral fit): e4b's fused path completes Mixtral's matched set RESIDENT at micro-batch 2 (VALID), peak at most 31.5 GB.
  This is close to a coin flip; the budget above leans to a fit.
- **P12** (Mixtral position, when P11 holds and both pairs are stable): Unsloth/e4b s/step in **[0.45, 0.95]**, Unsloth faster.
  Mixtral routes 2 of 8 large experts, so the step is GEMM-bound, not launch-bound. That is where e4b's edge on Qwen3-30B-A3B
  (many small experts, a launch-bound step) does not carry, and where the H100 showed dequantize + grouped GEMM beating the fused kernels.
- **P13** (Qwen3.6 fit): e4b's fused path OOMs Qwen3.6 resident at micro-batch 2 again, and completes it resident at micro-batch 1
  (VALID). Each half is scored on its own.
- **P14**: each resident matched pair that runs is EQUIVALENT at N = 20 (the lane's band), and e4b's parity control PASSES wherever the
  resident reference fits. A reference that OOMs resident leaves parity UNTESTED on this box (amendment 5's offload parity stands).

**Decision rules.**
- P11 holds and a stable pair is quoted: Mixtral gets a resident position row, `e4b.train.h2h.unsloth.mixtral.5090.2026-10-04`. The
  footprint row stays as the offload lever's reading.
- P11 refuted at both micro-batches: the resident loss stands, with the peak at the failing allocation recorded. The next lever is
  e4b's fp32 absmax (about 2.8 GB on Mixtral, where Unsloth double-quantizes), its own registration.
- P13: a resident fit at either micro-batch registers a row beside `e4b.train.h2h.unsloth.qwen3_5.5090.2026-10-02`. A pair at
  micro-batch 1 is quoted with the micro-batch in its name.

**Budget.** One RTX 5090 at the policy rate ($0.85/h), 5 h guard. Estimate: $4.25 plus about 165 GB of checkpoint download at the
launcher's cap of $0.011/GB, under $6.10 in all. The standing no-ask tier for a single run under $15; the campaign's daily cap is $100.

### Amendment 7 (2026-10-04T14:03Z, after amendment 6's read and the two e4b memory switches, before any box): the small families re-read, and the big families with the absmax double-quantized

**Why.** Amendment 6 left two things open.

- **The small-family positions are stale.** Granite and OLMoE were read on 2026-10-02 at e4b 0.38.1, before TC1 amendments 10–15 made e4b's
  own step about 1.4× faster on Qwen3-30B-A3B (Unsloth/e4b 1.437 → 1.997 on a 5090). On Granite, HF read 0.971 and axolotl 0.903 against
  e4b, both faster than it then.
- **e4b's extra resident footprint has a measured cause.** e4b stores the expert absmax in fp32, where Unsloth trains on bitsandbytes'
  double-quantized absmax; on Qwen3.6, e4b also keeps 270 non-routed projections in bf16 that Unsloth stores in 4-bit. Two switches now
  exist. `E4B_ABSMAX_DQ=1` (#1040) stores the absmax as bitsandbytes does and expands one layer's to fp32 for the kernel. Its values
  are bitsandbytes' nested ones, and it measured 3.94× fewer absmax bytes on a Qwen3-30B-A3B layer. `--frozen-4bit` (#1037) stores
  the non-routed projections in NF4; it is a measurement hook that puts e4b on a 4-bit comparator's bytes, priced at 0.05 nats at
  step 0 on Qwen3.6.

**The boxes.** Three RTX 5090s, each from this amendment's merge, with grouped-nf4-gemm v0.37.0:

- **Box S** (`tc2small`, unchanged token): Granite, OLMoE and gpt-oss at the small instrument, every arm as on 2026-10-02.
- **Box D** (`tc2resident`, `TC1_E4B_ENV="E4B_ABSMAX_DQ=1"`): amendment 6's box with the absmax double-quantized on every e4b arm.
  This is e4b's quality-preserving configuration (the non-routed projections stay bf16).
- **Box F** (`tc2resident`, `TC1_E4B_ENV="E4B_ABSMAX_DQ=1 TRAIN_FROZEN_4BIT=1"`): box D plus the non-routed projections in NF4, the
  bytes Unsloth trains on. On Mixtral the second switch converts nothing, so box F's Mixtral half is a replication of box D's.

**Predictions** (registered before the boxes), read off each box's own lines:

- **P15** (box S, Granite): HF/e4b in **[1.05, 1.60]** and axolotl/e4b in **[1.00, 1.50]**, e4b now faster against both.
- **P16** (box S, OLMoE): Unsloth/e4b in **[1.30, 2.20]** and HF/e4b in **[1.40, 2.40]**.
- **P17** (box S): e4b's parity PASSES on both families, and each quoted pair reads as on 2026-10-02 (Granite axolotl EQUIVALENT, OLMoE
  Unsloth EQUIVALENT; HF COMPARABLE).
- **P18** (box D, Mixtral): e4b's resident peak at most **29.6 GB** (amendment 6: 31.07), its pair stable, Unsloth/e4b in
  **[0.62, 0.78]** (amendment 6: 0.697; the expansion costs a few percent), and the held-out within the pair's equivalence band.
- **P19** (box D, Qwen3.6): e4b's fused path completes the matched set resident at micro-batch 2 (VALID). Its pair is VOID by the
  step-0 rule (the bf16 projections), as registered: a fit row, not a position.
- **P20** (box F, Qwen3.6): e4b completes the matched set resident at micro-batch 2 at a peak at most **30.6 GB**.
- **P21** (box F, Qwen3.6): the step-0 held-out gap |e4b − Unsloth| is at most **0.01** nats (amendment 6's census: 0.05 with the
  projections in bf16).
- **P22** (box F, Qwen3.6): Unsloth/e4b in **[1.30, 2.50]**, e4b faster, both pairs stable.
- **P23** (boxes D and F): every resident matched pair that runs is EQUIVALENT at N = 20, and e4b's parity PASSES wherever its
  resident reference fits.

Each is FALSIFIED outside its band and UNTESTED where the box quotes no reading.

**Decision rules.**

- **P15/P16:** each quoted pair supersedes its 2026-10-02 row as the family's position. The old rows stay, labelled as e4b 0.38.1.
- **Default:** `E4B_ABSMAX_DQ` becomes e4b's training default only if P18 holds and box D's e4b held-out is within 0.005 of amendment
  6's e4b held-out on Mixtral. Until then it stays opt-in.
- **Qwen3.6 position:** box F's pair, with P20–P23 holding, is the family's first matched position. It is quoted as "e4b on the
  comparator's 4-bit bytes" (`--frozen-4bit`), since e4b's own default keeps those projections bf16.

**Budget.** Three RTX 5090s at the policy rate ($0.85/h). Box S has a 3 h guard; boxes D and F have 5 h each, with the 192 GB host floor.
Estimates: S about $1.5, D and F about $3.2 each with the download, about $8 in all. The standing no-ask tier per run; the campaign's daily
cap is $100, counted 9 am to 9 am Central.

### Amendment 8 (2026-10-04T18:00Z, after amendment 7's read and TC1 amendment 22, before any box): Mixtral at default settings with the dense route, and Qwen3.6's micro-batch-1 pair with two draws a side

**Why.**

- **Mixtral.** TC1 amendment 22 read grouped-nf4-gemm's dense route at 0.651× the fused kernels' step on Mixtral-8x7B, and grouped-nf4-gemm#463
  makes it `auto`'s choice off sm_90 for calls with at most 16 present groups. Mixtral routes 2 of 8 experts, so its calls qualify.
  Amendment 6's position (Unsloth/e4b 0.697) read the fused kernels; this box reads the position with e4b at its defaults on that release
  of the route.
- **Qwen3.6.** Amendment 7's box F trained Qwen3.6 resident at micro-batch 1 with the absmax double-quantized and the non-routed projections
  in NF4 (`--frozen-4bit`, the comparator's bytes): e4b 9.24 s/step against Unsloth's 18.49 on the same box. That was one draw each, the
  secondary pair, so it was not quoted. This amendment asks for that pair as a primary pair, two draws a side.

**The boxes.** Two RTX 5090s from this amendment's merge, with grouped-nf4-gemm at #463's merge.

- **Box M** (`tc2mixtralres`): Mixtral-8x7B alone, the field recipe, every e4b arm resident at **default settings** (nothing set, so `auto`
  takes the dense route). Unsloth resident ×2 and e4b ×2, then the e4b reference; HF, both axolotl arms and e4b as shipped are `not_run`
  stubs.
- **Box Q** (`tc2qwen35mb1`): Qwen3.6-35B-A3B alone, resident, the primary pair at **micro-batch 1 × accum 8** (the same tokens per step),
  e4b with `E4B_ABSMAX_DQ=1 TRAIN_FROZEN_4BIT=1` ×2, and Unsloth with the family's expert target parameters ×2. The e4b reference,
  HF, both axolotl arms and e4b as shipped are `not_run` stubs (the reference OOMed resident at micro-batch 1 on box F).

**Predictions** (registered before the boxes):

- **P24** (box M): e4b's fused path completes Mixtral resident at default settings (VALID), peak at most **31.6 GB** (the fp32 absmax,
  31.07 GB with the fused kernels, plus the dense route's one-expert transient).
- **P25** (box M): Unsloth/e4b lies in **[0.75, 1.20]**, both pairs stable. This band is wide on purpose: Unsloth's Mixtral step has
  ranged 3.0–4.1 s across hosts, and e4b's dense step is about 3.6–3.7 s.
- **P26** (box Q): both e4b micro-batch-1 draws complete resident (VALID), each peak at most **31.8 GB**.
- **P27** (box Q): Unsloth/e4b at micro-batch 1 lies in **[1.50, 2.60]**, both pairs stable (box F's single draws: 2.0).
- **P28:** each quoted pair reads EQUIVALENT or COMPARABLE at N = 20 (the reading is recorded, not a gate), and box Q's step-0 gap is at
  most 0.02 nats (box F: 0.011–0.014).

Each is FALSIFIED outside its band and UNTESTED where the box quotes no reading.

**Decision rules.**

- **P24 and P25:** box M's pair becomes Mixtral's position at default settings (`e4b.train.h2h.unsloth.mixtral.5090.<date>.dense-default`).
  It supersedes amendment 6's 0.697 as the family's position, and that row stays labelled as the fused kernels' reading.
- **P26 and P27:** box Q's pair is Qwen3.6's first quoted position, labelled "micro-batch 1, e4b on the comparator's bytes with the absmax
  double-quantized", since e4b's defaults keep those projections bf16 and do not fit at micro-batch 2.

**Budget.** Two RTX 5090s at the policy rate, 3 h (M) and 4 h (Q) guards, with a 192 GB host floor. About $1.5 and $2.5 with the downloads;
the standing no-ask tier; the campaign's daily cap is $100, counted 9 am to 9 am Central.

### Amendment 9 (2026-10-05T07:16Z, after TC1 amendments 33–35 and TC1c amendment 9, before any box): Mixtral's position with both frameworks on one stack (P29, P30, P31)

**Why.** Mixtral-8x7B is the one family where e4b's position at default settings is a loss: amendment 8's box M read Unsloth/e4b
**0.836**, Unsloth about 20 % faster per step. Two things in that box favour Unsloth, and neither is the frameworks' own work:

- **The stack.** e4b ran on the field image's torch 2.8.0 / transformers 5.18.0, Unsloth on torch 2.12.1 / transformers 5.5.0. On
  Qwen3-30B-A3B, TC1 amendments 33 and 34 found e4b's step 0.90–0.91× as long on torch 2.12, and TC1 amendment 25's same-stack position
  became that family's quoted one.
- **The host.** Box M ran on an Intel Core Ultra 9 285K. Its own row records Unsloth's Mixtral step at 3.0 s there, 3.7 s on an EPYC 7B13
  and 4.1 s on an EPYC 7C13, while e4b's moved much less. Vast machine 151350, a 285K, also lost speed pairs at low load for no cause found.

**The box** (token `mixtralsamestack`). TC1 amendment 25's same-stack family on Mixtral-8x7B-Instruct at TC2's pin and field recipe, as
TC1 amendment 33 ran it on Qwen3-30B-A3B:

- every e4b arm resident at **default settings** (nothing set: grouped-nf4-gemm's `auto` takes the dense route off sm_90, the prebound
  launches on, the expert absmax fp32);
- e4b's matched arm in venv-unsloth (two draws), Unsloth 2026.9.14 grouped_mm with TC2's seven targets (two draws), e4b's matched arm in
  venv-e4b (`_t28`, two draws), in amendment 25's order; e4b's reference arm not run;
- `TC1_STEPS=60`, load-gated draws (`TC1_LOAD_GATE=6.0`, `TC1_LOAD_RETRIES=2`), a 192 GB host floor, avoiding machines 151350, 45511 and
  138786.

Engagement: TC1 amendment 25's (each e4b receipt records the torch its tag names), plus the dense route on every fused e4b arm that ran
(`route_ab` counts dense forward and dense dgrad calls and no fused ones, with `GNF4_TRAIN_GEMM` unset).

**Predictions** (registered before the box):

- **P29:** with both frameworks on one stack, Unsloth/e4b lies in **[0.85, 1.25]**, both pairs stable. The basis: 0.836 on the 285K,
  Unsloth's step 20–35 % longer on the EPYC hosts it has run on, and an environment gain for e4b at or below Qwen3's.
- **P30:** e4b's matched arm in venv-unsloth over venv-e4b lies in **[0.85, 1.02]**, both sides stable. Mixtral's dense route issues far
  fewer launches per step than Qwen3's fused path, so its gain may be smaller than Qwen3's 0.909.
- **P31:** the dense route on every fused e4b arm that ran, as above.

Each is FALSIFIED outside its band and UNTESTED where a side is missing, unstable, not VALID or not engaged. The matched set's quality
reading is the reducer's (QUALITY_FAIL at |Δ held-out| > 0.05 blocks the position). An e4b OOM resident is a row (box M's default peak was
31.07 GB of the card's 32), and leaves P29 and P30 UNTESTED.

**Decision rules.**

- **P29 read with both pairs stable and P31 HELD, whichever side it favours:** the ratio becomes Mixtral's position to quote,
  `e4b.train.h2h.unsloth.mixtral.5090.<date>.same-stack`. Box M's 0.836 stays as the reading with e4b in the field image's environment on
  a 285K host, and STATUS names both. A reading below 1.00 is quoted as Unsloth faster on one stack, an e4b loss said as such.
- **P31 FALSIFIED:** the box ran a different route than the position it replaces; nothing is quoted from it.
- **P30** says whether the torch 2.12 gain carries to a family on the dense route; it moves no default.

**Budget.** One RTX 5090 at the policy rate ($0.85/h), 4 h guard, a 192 GB host floor, venv-unsloth built for e4b's same-stack arms and
Unsloth. About $2.50 with Mixtral's download and the gate's possible re-runs; this is in the standing no-ask tier.
