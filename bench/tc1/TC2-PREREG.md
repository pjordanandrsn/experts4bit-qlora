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
