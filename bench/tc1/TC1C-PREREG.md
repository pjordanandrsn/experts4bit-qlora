# TC1c — the same matched pair on the card the competitor is built for: Qwen3-30B-A3B on one H100 (registered 2026-10-02, before any box; the `qwen3` token of the TC1 harness with `TC1_GPU_CLASS="H100 NVL"`; may run before TC1 box A is read — the readings are independent)

Issue: experts4bit-qlora#835. Why: Unsloth's grouped GEMM is "tested and benched on H100" and `torch._grouped_mm` is
native on sm_90 even on torch 2.8; the RTX 5090 is the class where the comparator's intended path was unavailable
until TC1. A position that holds only on the card where the competitor was handicapped is not a position. One
draw, inside the no-ask tier.

## Arms (the TC1 `qwen3` token, unchanged, on an H100 NVL: `TC1_GPU_CLASS="H100 NVL"`, the spelling the policy and the harness's class check accept)

`e4b/fused_attn4_m`, `unsloth/ckpt_unsloth_m` (venv-unsloth, grouped_mm), `e4b/reference_attn4_m` third, the second draws of
the matched pair, `hf/hf_peft_m` (on 80 GB the bf16-expert HF stack is expected to FIT — the first HF position on this
family), `axolotl/ckpt_axolotl_m`, `e4b/fused_attn4_m_prof`, `unsloth/ckpt_unsloth_prof`; the `_mb1` pair on any OOM.
The torch-2.8 Unsloth row (`ckpt_unsloth_t28`, where `torch._grouped_mm` is native on sm_90 even on torch 2.8) is not in
this token; a native-rows draw on this class is a follow-up if TC1c's reading warrants it. Fixture, validity, verdicts,
readings: TC1's; every ratio within this box.
e4b's kernels on sm_90: grouped-nf4-gemm's fused path is sm_80+ Triton (its own register says which arch each kernel
was measured on; an engagement or tripwire failure on this class is a row).

## Predictions

- P1 Unsloth/e4b on the H100 is LOWER than on the 5090 (the comparator gains more from the card than e4b does); band
  [0.5, 2.0]. Below 1.0 on this card is a stated loss and goes beside the 5090 position wherever it is quoted.
- P2 stability and P3 equivalence as TC1.
- P4 HF+PEFT (bf16 experts, weight-side delta) trains on this card and is SLOWER than e4b: HF/e4b in [1.5, 4]; its
  held-out at N reads COMPARABLE or better (bf16 experts carry no quantisation loss) — the quality column states the
  regime beside the number.

## Budget

One H100 NVL on Vast verified-secure (2026-10-02T00:26Z: two verified offers, $2.67/h and $3.54/h, 188 GB RAM),
ceiling $2.80/h, guard 3 h, estimate $8.40 (< $15, the owner's standing tier). Pre-flight as TC1 (>= 320 GB disk,
>= 98 GB RAM, >= 40 MB/s); the driver floor (TC1 amendment 1) and the cu130 index (amendment 2) apply unchanged.

## Amendments

### Amendment 1 (2026-10-04T01:44Z, before any box): the H100 position again, with e4b after TC1 amendments 10–15 (P5, P6)

**Why.** TC1c's box (`tc1c-h100-2`) read Unsloth/e4b **0.621** [0.615, 0.628] on an H100 NVL, Unsloth 1.61× faster per step
(2.546 against 4.097 s/step). That was e4b before #945. Since then e4b's matched arm has stepped at 0.847 × 0.911 × 0.968 × 0.959
≈ 0.716 of itself on RTX 5090s, each factor within one box:

- #945's host syncs: 0.847;
- #440's trimmed LoRA delta: 0.911;
- #442's tile rule: 0.968;
- #975's fused RMSNorm: 0.959.

The fused rotary is exact. On the 5090 the matched position against Unsloth moved from 1.437 to 1.997 (TC1 amendment 19, P30). The
H100 is the class where the comparator's path is native, and the one TC1c position where e4b loses. Only a new box can say whether
it still does.

**The box.** TC1c's token, unchanged: TC1's `qwen3` with `TC1_GPU_CLASS="H100 NVL"`. e4b is pinned at the main commit carrying
this amendment, with every default from TC1 amendments 10–15, and grouped-nf4-gemm at `00929a4` (#442). No environment variables
are set. Two caveats are written down before the box:

- the factors above were measured on sm_120;
- the cost tile rule's D (96 rows) was fitted on sm_86.

So nothing about this card is assumed.

**Predictions** (registered before the box), read off the box's own lines:

- **P5:** the MATCHED POSITION unsloth/e4b on the H100 NVL lies in **[0.70, 1.20]**. The point estimate is 0.621 / 0.716 ≈ 0.87, and the
  band is wide because the factors come from another class. **Ordering reading**, beside the band:
  - interval wholly below 1.0: Unsloth is still faster on this card;
  - wholly above 1.0: e4b is now faster on it too;
  - spanning 1.0: parity within the draw noise.
- **P6:** the box's P3 line is HELD, with the matched set EQUIVALENT or inside the draw noise of e4b fused.

Each is FALSIFIED outside its band, and UNTESTED where the box quotes no position: an unstable pair, a non-VALID arm or a missing
receipt.

**Decision rules.** A HELD or FALSIFIED P5 becomes a new register row, `e4b.train.h2h.unsloth.qwen3.h100.2026-10-04` (the run's date),
and STATUS quotes it with its ordering. The 2026-10-02 row stays active, labelled as the code before #945. P6 FALSIFIED blocks quoting
the box's position.

**Budget.** One H100 NVL on Vast verified-secure, ceiling $2.80/h, guard 3 h, estimate $8.40 (under $15, the owner's standing tier),
as TC1c's original box.

### Amendment 2 (2026-10-04T03:41Z, after amendment 1's read, before any box): the H100 position with e4b keeping its MoE activations (P7, P8)

**Why.** Amendment 1 read Unsloth/e4b **0.817** [0.799, 0.836] on an H100 NVL (`tc1c-h100-3`): Unsloth still faster per step there,
by 1.22×. Its profiled arms say why. e4b's step is now mostly device-bound on that card (device-busy 0.665), and it spends about 2.4×
Unsloth's device time per step. Part of that device time is gradient checkpointing re-running every MoE forward in backward. On a
4-layer Qwen3-30B-A3B slice on an RTX A2000 that recompute was 232 of 917 ms of device time. Keeping all four layers' MoE activations
took the step to 0.773 of before, with every trainable gradient `torch.equal`.

That needs e4b#1007's `E4B_MOE_KEEP_LAYERS` and grouped-nf4-gemm#445's `NF4_QLORA_COMPACT_DELTA=1`. It costs about 121 MB of peak per
layer at this fixture's largest micro-batch, so about 5.8 GB for all 48 layers. That does not fit beside the 5090 arm's 27.2 GB (TC1
amendment 21 sizes it to 16–32 layers there), but it does on 80 GB. This box measures that setting on the card where e4b loses.

**The box.** TC1c's token, unchanged: TC1's `qwen3` family with `TC1_GPU_CLASS="H100 NVL"`. Two settings change:

- **Every e4b arm runs with**
  `TC1_E4B_ENV="E4B_MOE_KEEP_LAYERS=all NF4_QLORA_COMPACT_DELTA=1 GNF4_HOST_REUSE=1"`. This is the new knob, forwarded by the driver
  and applied to e4b arms only. Host reuse is named explicitly, so the box does not depend on when grouped-nf4-gemm#446 lands.
- **`TC1_SKIP="qwen3/hf/hf_peft_m qwen3/axolotl/ckpt_axolotl_m"`.** HF + PEFT hit its 1,800 s alarm with no step on both earlier H100
  boxes, and axolotl's row is not part of this question. Both become `not_run` stubs.

Unsloth's arms are byte-for-byte TC1c's. e4b is pinned at the main commit carrying this amendment, which carries #1007 and TC1
amendment 21's `keep_ab` record. grouped-nf4-gemm is pinned at a main commit carrying #445.

**Engagement**, read off the receipts before any prediction is scored. Each e4b fused arm's `keep_ab` must record 48 layers kept with
the compact delta on, and its `reuse_ab` must record the flag in force with hits. If not, the box is VOID for P7.

**Predictions** (registered before the box), read off the box's own lines:

- **P7:** the MATCHED POSITION unsloth/e4b lies in **[0.85, 1.30]**. The point estimate is 0.817 / 0.78 ≈ 1.05: the slice's 0.773,
  rounded up for the share of the H100 step that is not MoE recompute. The same ordering reading as P5 applies:
  - interval wholly below 1.0: Unsloth still faster;
  - wholly above 1.0: e4b faster with this setting;
  - spanning 1.0: parity within the draw noise.
- **P8:** the box's P3 line is HELD, with the matched set EQUIVALENT or inside the draw noise of e4b fused (e4b's reference arm runs
  with the same environment).

Each is FALSIFIED outside its band, and UNTESTED where the box quotes no position. The e4b fused arms' peak memory is reported beside
P7, not predicted.

**Decision rule.** A HELD or FALSIFIED P7 becomes a LABELLED register row,
`e4b.train.h2h.unsloth.qwen3.h100.2026-10-04.moe-keep`. It is e4b with an opt-in setting, quoted beside amendment 1's default-settings
position and never in place of it. STATUS states both and names the setting. P8 FALSIFIED blocks quoting the box's position.

**Budget.** One H100 NVL on Vast verified-secure, $2.80/h GPU ceiling (disk billed on top, as on every box), 2.5 h guard, estimate
about $4–5 with the HF arm skipped; the owner's standing tier (a single run under $15). The box launches only after TC1 amendment 21's
5090 box has produced a VALID keep arm, so a broken path is found on the cheaper card.

### Amendment 3 (2026-10-04T06:00Z, after amendment 2's read, before any box): which GEMM route the H100 wants — a kernel replay, not a position (P9, P10, P11)

**Why.** On the H100 NVL e4b is now mostly device-bound. Amendment 2's profiled e4b arm reads device-busy 0.659, about 1.58 s of device
time per step, against Unsloth's 0.306 at about 0.87 s. The step's device time is dominated by grouped-nf4-gemm's fused NF4 grouped
kernels: on an RTX A2000 the forward and dgrad kernels were 75 % of one MoE layer's device time. Unsloth's path on this card is a
bitsandbytes dequantize of the 4-bit stack followed by torch's grouped GEMM, and `torch._grouped_mm` is first-class on sm_90. Whether
e4b should take that route on Hopper is a kernel question, so it is measured as one before any training box depends on it.

**The box.** One H100 NVL, token `routebench`. No model is downloaded, and no Unsloth venv is built. The box runs
`bench/tc1/route_bench.py` on `bench/tc1/routecalls-qwen3.json` (staged with `TC1_EXTRA_STAGE`). That file holds the 192 group-size /
expert-id lists of e4b's fused forward and dgrad calls, recorded through the real checkpoint's router on a 4-layer Qwen3-30B-A3B slice
(TC1's alpaca rows, mb2, 8 micro-batches; recorder `bench/tc1/route_record.py`); the recompute repeats leave 128 unique calls. For each
unique call at Qwen3-30B-A3B's shapes (E 128; gate_up N 1536 / K 2048; down N 2048 / K 768) it times, by CUDA events at the median of 10:

- **(a)** grouped-nf4-gemm's fused forward and dgrad at their defaults;
- **(b)** one whole-stack `dequantize_4bit` to bf16;
- **(c)** `torch._grouped_mm` over all 128 experts' offsets. Empty experts are zero-size groups; the ids are ascending in every call, so
  there is no gather.

Weights are random NF4 stacks in the training layout, and the script refuses to run unless the dequantized stack matches
grouped-nf4-gemm's `dequant_ref` bitwise. The route's output is compared against (a) on every call. On the RTX A2000, where
`torch._grouped_mm` is unavailable, a per-group `torch.mm` on the same dequantized stack agreed with (a) to a relative Frobenius error of
at most 0.0024. e4b is pinned at this amendment's merge, and grouped-nf4-gemm at its main.

**Predictions** (registered before the box):

- **P9:** summed over the 64 unique forward calls, ((b) + (c)) / (a) lies in **[0.20, 0.80]**.
- **P10:** the same for the 64 dgrad calls, in **[0.20, 0.80]**.
- **P11:** on every call, the route's relative Frobenius error against (a) is at most **0.005**.

Each is FALSIFIED outside its band, and UNTESTED if `torch._grouped_mm` is unavailable on the box or the script does not complete.
(c) / (a) alone is reported beside each figure: the dequant could be shared between a layer's forward, recompute and dgrad.

**Basis.** If the device-time gap is the GEMM route, as the profiles suggest, Unsloth's route should run these calls in about half
the fused kernels' time. Below 0.20 would mean the fused kernels are badly mistuned for sm_90, and above 0.80 that the route is not
the gap.

**Decision rule.**

- **P9 and P10 at or below 0.80, with P11 HELD:** build the route into grouped-nf4-gemm as an opt-in for sm_90. Its value then gets
  measured on the full training step in an amendment 4.
- **Otherwise:** no route. The H100 gap is documented with this evidence.

This box quotes no position and changes no register row.

**Budget.** One H100 NVL on Vast verified-secure, $2.80/h GPU ceiling (disk billed on top), 1.0 h guard, estimate about $1.50–2.70;
the owner's standing tier (a single run under $15).

### Amendment 4 (2026-10-04T06:19Z, after amendment 3's read, before any box): the H100 position with grouped-nf4-gemm's grouped_mm route, two boxes (P12, P13, P14)

**Why.** Amendment 3 (`tc1c-h100-5`) replayed the 128 unique fused GEMM calls of e4b's training step on an H100 NVL. A dequantize
plus `torch._grouped_mm` took **0.596** (forward) and **0.494** (dgrad) of the fused kernels' time, within 0.0024 relative Frobenius
error on every call. By its decision rule grouped-nf4-gemm#450 takes that route as an sm_90 opt-in, `GNF4_TRAIN_GEMM=grouped_mm`. The
dequant is a Triton kernel over the present experts, bit-equal to `dequant_ref`. The route is not bit-identical to the fused kernels,
so its value and its numerics are measured on the full training step, against Unsloth, before anything is defaulted.

**The boxes.** Two H100 NVL boxes run TC1c's token unchanged (TC1's `qwen3` family with `TC1_GPU_CLASS="H100 NVL"`, HF and axolotl
skipped as in amendment 2). Each sets `TC1_E4B_ENV` on every e4b arm:

- **box R (`route`):** `TC1_E4B_ENV="GNF4_TRAIN_GEMM=grouped_mm"`. Every other e4b setting is a default, including host reuse (on
  since grouped-nf4-gemm 0.36.0). Its counterpart is amendment 1's 0.817.
- **box K (`keep + route`):**
  `TC1_E4B_ENV="E4B_MOE_KEEP_LAYERS=all NF4_QLORA_COMPACT_DELTA=1 GNF4_HOST_REUSE=1 GNF4_TRAIN_GEMM=grouped_mm"`. Its counterpart is
  amendment 2's 1.100.

e4b is pinned at this amendment's merge, and grouped-nf4-gemm at the merge of #450. A new `route_ab` record on each e4b arm names the
route in force and its call counts (`nf4_route.ROUTE_STATS`). Engagement is read off the receipts before any prediction is scored: on
every e4b fused arm `route_ab` must name `grouped_mm` with nonzero forward and dgrad counts, and box K's `keep_ab` must record 48
layers kept. If not, that box is VOID.

**Predictions** (registered before the boxes), read off each box's own lines:

- **P12** (box R): the MATCHED POSITION unsloth/e4b lies in **[0.85, 1.20]**. Amendment 1's e4b step was 3.146 s at device-busy 0.665,
  about 2.1 s of device time. If the fused GEMMs are about 60 % of that and the route takes them to about 0.55, the step loses about
  0.55 s, giving about 0.99.
- **P13** (box K): the MATCHED POSITION unsloth/e4b lies in **[1.15, 1.60]**. Amendment 2's step was 2.343 s at 0.659 busy. The same
  arithmetic over the forward and dgrad GEMMs, with no recompute, gives about 1.3.
- **P14:** both boxes' P3 lines are HELD. With the route on, e4b fused stays EQUIVALENT to e4b's reference and to Unsloth, or inside
  the draw noise. This is the route's numerics test on the training step.

The ordering reading of amendment 1 applies to P12 and P13. Each is FALSIFIED outside its band, and UNTESTED where the box quotes no
position.

**Decision rules.**

- **P12** HELD or FALSIFIED becomes a LABELLED row, `e4b.train.h2h.unsloth.qwen3.h100.<date>.route`. **P13** likewise becomes
  `...h100.<date>.moe-keep-route`. Both are quoted beside amendments 1 and 2, never in place of them.
- **Default on sm_90.** grouped-nf4-gemm makes the route its default there only if three things hold: P14 HELD, box R's position at
  or above 1.05 × 0.817 = 0.858, and box R's e4b held-out within 0.005 of amendment 1's e4b. Otherwise it stays opt-in.
- **P14 FALSIFIED** on either box blocks quoting that box's position.

**Budget.** Two H100 NVL boxes, each $2.80/h GPU ceiling (disk billed on top), 2.5 h guard, about $2.80 each as amendment 2's box was;
the owner's standing tier (each a single run under $15).

### Amendment 5 (2026-10-04T06:26Z, before any box): the fused kernels' own configs on the H100 — a kernel replay, not a position (P15, P16, P17)

**Why.** Amendment 3 found the grouped GEMM alone (`torch._grouped_mm` on an already-dequantized stack) running the recorded calls at
0.20 (forward) and 0.16 (dgrad) of grouped-nf4-gemm's fused kernels on an H100 NVL. Part of that may be configuration: the fused
kernels' defaults were chosen on sm_86 and sm_120. On an RTX A2000 the defaults were best, the variant-1 configs were all bit-identical
to the default, and the BLOCK_K 128 and most bf16-MMA (variant 3) configs did not fit its 101 KB of shared memory. The H100 has 228 KB
and wgmma, so its ranking can differ. A bit-identical speedup there would beat a route that is not bit-identical.

**The box.** One H100 NVL, token `fusedsweep`: no model, no Unsloth venv. It runs `bench/tc1/fused_sweep.py` on the same 128 unique
recorded calls (`bench/tc1/routecalls-qwen3.json`, staged with `TC1_EXTRA_STAGE`). The forward is swept over the default plus 24 configs:
`prefill_variant` ∈ {1, 3}, `prefill_groups` ∈ {1, 2}, and (BLOCK_N, warps, stages) ∈ {(128,4,3), (128,8,3), (128,8,4), (256,8,3),
(64,4,4), (128,4,4)}, at the cost-rule M-tile. The dgrad is swept over the default plus 8 configs. Each config's summed device time
(CUDA events, median of 5) and its bit-identity to the default are recorded. A config that does not compile or launch is recorded with
its error. e4b is pinned at this amendment's merge, and grouped-nf4-gemm at its main.

**Predictions** (registered before the box):

- **P15:** the best forward config that is bit-identical to the default takes at most **0.90** of the default's time.
- **P16:** the best forward config of any kind takes at most **0.60** of the default's time, i.e. at or below the route's 0.596.
- **P17:** the best dgrad config takes at most **0.90** of the default's time.

Each is FALSIFIED above its bound, and UNTESTED if the box does not complete.

**Basis.** These are bets, not estimates. A 5× gap to cuBLAS is more than tuning usually closes, so P16 is the long shot; P15 and P17
ask whether the sm_86 choices leave easy time on the table on sm_90.

**Decision rules.**

- **P15 or P17 HELD:** that bit-identical config becomes grouped-nf4-gemm's sm_90 default after a full-step box confirms it.
- **P16 HELD with a non-identical config:** it competes with the grouped_mm route in the next full-step box.
- **All FALSIFIED:** the route (amendment 4) is the H100 path.

This box quotes no position and changes no register row.

**Budget.** One H100 NVL, $2.80/h GPU ceiling (disk billed on top), 1.0 h guard, estimate under $1; the owner's standing tier.

### Amendment 6 (2026-10-04T08:39Z, after amendment 4's read, before any box): the grouped_mm route again, with a dequant at bandwidth (P18, P19, P20)

**Why.** Amendment 4's boxes made e4b slower with grouped-nf4-gemm's grouped_mm route: Unsloth/e4b 0.664 with the route alone and 0.934
with MoE activations kept, against 0.817 and 1.100 without it. Their profiled arm showed why. The route's own dequant kernel took 2,178
ms of device time per step, 1.89 ms per call, against the fused kernels' 1,424 ms. It gathered the NF4 LUT and the absmax per element,
and amendment 3's replay, whose ratios amendment 4 relied on, had timed bitsandbytes' dequant instead.

grouped-nf4-gemm#452 rewrites the kernel: coalesced byte tiles, a register LUT, block-broadcast absmax. It is still bit-equal to
`dequant_ref` in bf16. On an RTX A2000 it runs 3.1–5.1× faster than the old kernel and at 0.73–0.93× of bitsandbytes'
`dequantize_4bit`. Nothing else in the route changes.

**The boxes.** Amendment 4's two boxes are re-run unchanged, with grouped-nf4-gemm pinned at the merge of #452 and e4b at this
amendment's merge:

- **box R** (`route`): `TC1_E4B_ENV="GNF4_TRAIN_GEMM=grouped_mm"`;
- **box K** (`keep + route`): `TC1_E4B_ENV="E4B_MOE_KEEP_LAYERS=all NF4_QLORA_COMPACT_DELTA=1 GNF4_HOST_REUSE=1 GNF4_TRAIN_GEMM=grouped_mm"`.

Both run TC1c's token with HF and axolotl skipped. Engagement is read as in amendment 4 (`route_ab`, `keep_ab`). The profiled e4b arm
also reports the dequant kernel's device time, so its share of the step is read, not assumed.

**Predictions** (registered before the boxes), read off each box's own lines:

- **P18** (box R): the MATCHED POSITION unsloth/e4b lies in **[0.90, 1.25]**. At about 0.3 ms per dequant call, box R's route takes about
  0.6 s of device time per step against the fused kernels' 1.42 s. Only part of the saving reaches the step, which is partly host-bound,
  so the estimate is about 1.0 (amendment 1's e4b step was 3.146 s).
- **P19** (box K): the MATCHED POSITION unsloth/e4b lies in **[1.15, 1.65]**. There are 768 dequant calls per step, the forward and
  dgrad GEMMs come to about 1.0 s fused, and amendment 2's step was 2.343 s. The estimate is about 1.35.
- **P20:** both boxes' P3 lines are HELD (the route's training numerics again).

Each is FALSIFIED outside its band, and UNTESTED where the box quotes no position.

**Decision rules.**

- Each reading becomes a LABELLED row (`...h100.<date>.route-v2`, `.moe-keep-route-v2`), quoted beside amendments 1 and 2. Amendment 4's
  rows stay as they are, labelled as the first dequant kernel.
- **Default on sm_90.** grouped-nf4-gemm makes the route its default there only if three things hold, as amendment 4 registered: P20
  HELD, box R at or above 0.858 (5 % over amendment 1's 0.817), and box R's e4b held-out within 0.005 of amendment 1's e4b.
- **P20 FALSIFIED** blocks quoting that box.

**Budget.** Two H100 NVL boxes, each $2.80/h GPU ceiling (disk billed on top), 2.5 h guard. Each is about $2.50 invoiced, as amendment
4's were; the owner's standing tier (each a single run under $15).

### Amendment 7 (2026-10-04T12:45Z, after amendment 6's read and the 0.45.0 release, before any box): the H100 position at default settings on the release that carries the route

**Why.** Amendment 6's readings are LABELLED rows because the route was forced with `GNF4_TRAIN_GEMM=grouped_mm`. Their decision rule
held, and grouped-nf4-gemm#454 made the route its sm_90 default (`GNF4_TRAIN_GEMM=auto`). grouped-nf4-gemm 0.37.0 ships it, and
experts4bit-qlora 0.45.0 is the first release on it. Amendment 6 named this box as what turns the labelled reading into the H100
position: default settings, nothing set, on the release that carries the default. It also checks `auto` itself on real sm_90 hardware.
Amendment 6 forced the route, so `auto`'s capability check has not yet run on a card that takes it.

**The box.** TC1c's token on one H100 NVL with **no `TC1_E4B_ENV`** (`GNF4_TRAIN_GEMM` unset), grouped-nf4-gemm pinned at v0.37.0
(`71185d6`), and e4b at this amendment's merge, whose code is 0.45.0's. HF and axolotl are skipped, as in amendments 2, 4 and 6.

**Predictions** (registered before the box), read off its own lines:

- **P21:** the MATCHED POSITION unsloth/e4b lies in **[0.95, 1.12]**. Amendment 6's box R read 1.030 [1.016, 1.045] with the route
  forced. This box should take the same path through `auto`, and the spread between H100 NVL boxes has been about 2 %.
- **P22:** `auto` engages the route with nothing set. On every e4b fused arm `route_ab` names `grouped_mm` with `GNF4_TRAIN_GEMM` unset,
  and counts amendment 6's 16,896 forward and 7,680 dgrad calls; the reference arm counts none.
- **P23:** the box's P3 line is HELD.

Each is FALSIFIED outside its band or count, and UNTESTED where the box quotes no position.

**Decision rules.**

- **All three HELD:** this reading becomes the H100 position at default settings (`e4b.train.h2h.unsloth.qwen3.h100.release-0.45.0`). It supersedes
  amendment 1's 0.817 as the current default-settings row. Amendment 1's row is kept as the fused kernels' reading, and amendment 6's
  labelled rows stay beside it.
- **P22 FALSIFIED:** `auto` did not take the route on sm_90. That is a grouped-nf4-gemm defect; it is fixed before any position moves,
  and amendment 1's row stands.
- **P23 FALSIFIED** blocks quoting the box.

**Budget.** One H100 NVL, $2.80/h GPU ceiling (disk billed on top), 2.5 h guard, about $2.60 invoiced like amendment 6's boxes. The owner's
standing tier for a single run under $15; the campaign's daily cap is $100.

### Amendment 8 (2026-10-04T14:10Z, after amendment 7's read, before any box): amendment 7's box once more, and the last re-ask under these rules

**Why.** Amendment 7's box showed `auto` taking the route with nothing set (P22 HELD), but it could not be quoted. e4b's draws were 5.1 %
apart, over the 5 % rule (P21 UNTESTED), and Unsloth's first draw read COMPARABLE, at 0.0061 against a 0.0054 band (P23 FALSIFIED).
On this card e4b's step is now host-bound (amendment 6: device-busy 0.454), which widens its draw spread. Amendment 6's box R read
2.5 %.

**The box.** Amendment 7's box unchanged (`tc1c-h100-12`): TC1c's token, nothing set, grouped-nf4-gemm v0.37.0, e4b at this
amendment's merge (no code change from 0.45.0), and HF and axolotl skipped.

**Predictions:** amendment 7's three, renumbered.

- **P24:** the MATCHED POSITION unsloth/e4b lies in **[0.95, 1.12]**.
- **P25:** `route_ab` names `grouped_mm` with `GNF4_TRAIN_GEMM` unset (16,896 forward and 7,680 dgrad calls).
- **P26:** the box's P3 line is HELD.

**Decision rules.**

- **All three HELD:** this box becomes the H100 position at default settings (`e4b.train.h2h.unsloth.qwen3.h100.release-0.45.0`), as
  amendment 7 registered.
- **Otherwise:** there is no further re-ask under these rules. The register keeps amendment 1's row as the fused kernels' reading and
  amendment 6's labelled rows. STATUS reports the default's readings as measured and unquoted.
- A change to the instrument (more draws, or a band that accounts for Unsloth's own draw spread) would be its own registration,
  made before any further box.

**Budget.** One H100 NVL at a $2.80/h GPU ceiling, 2.5 h guard, about $2.15 invoiced like amendment 7's.

### Amendment 9 (2026-10-05T05:00Z, after TC1 amendment 33's read, before any box): the H100 position with both frameworks on one stack (P27, P28, P29)

**Why.** On the RTX 5090, TC1 amendment 33 read Unsloth/e4b at 2.352 with both frameworks on torch 2.12.1+cu130 / transformers 5.5.0,
and e4b's matched arm at 0.900× of its own step in the field image's environment. That became the 5090 position to quote. The H100
position of record (amendment 8, 1.061) still runs e4b in the field image's environment (torch 2.8.0, transformers 5.18.0, triton 3.4)
and Unsloth in its own. It is e4b's thinnest lead on any card, so it is the one most likely to move. Amendment 6 measured e4b's H100 step
as host-bound (device-busy 0.454), the regime where the 5090 found the environment gain. Since amendment 8, e4b's prebound Triton
launches became the default (TC1 amendments 26 and 30), and they now cover triton 3.7 (experts4bit-qlora#1108, grouped-nf4-gemm#471):
both e4b sides of this box take them.

**The box** (token `qwen3samestackh100`). TC1 amendment 25's same-stack family on one H100 NVL (`TC1_GPU_CLASS="H100 NVL"`), as TC1
amendment 33 ran it:

- `TC1_STEPS=60`, e4b's reference arm not run, load-gated draws (`TC1_LOAD_GATE=6.0`, `TC1_LOAD_RETRIES=2`);
- e4b's matched arm in venv-unsloth (two draws), Unsloth 2026.9.14 grouped_mm (two draws), e4b's matched arm in venv-e4b (`_t28`, two
  draws), in amendment 25's order;
- nothing else set: grouped-nf4-gemm's `auto` route on sm_90, the prebound launches at their default.

Engagement: amendment 25's (each e4b receipt records the torch its tag names), plus `route_ab` naming `grouped_mm` on every e4b arm.

**Predictions** (registered before the box):

- **P27:** with both frameworks on one stack, Unsloth/e4b lies in **[1.00, 1.35]**, both pairs stable. The basis: amendment 8's 1.061
  over an environment gain of 0.80–1.00 (the 5090 read 0.88–0.90).
- **P28:** e4b's matched arm in venv-unsloth over venv-e4b lies in **[0.80, 1.00]**, both sides stable.
- **P29:** `route_ab` names `grouped_mm` on every e4b arm in both environments, with `GNF4_TRAIN_GEMM` unset.

Each is FALSIFIED outside its band and UNTESTED where a side is missing, unstable or not engaged. The matched set's quality reading is
the reducer's (QUALITY_FAIL at |Δ held-out| > 0.05 blocks the position).

**Decision rules.**

- **P27 read with both pairs stable and P29 HELD, whichever side it favours:** the ratio becomes the H100 position to quote,
  `e4b.train.h2h.unsloth.qwen3.h100.<date>.same-stack`. Amendment 8's 1.061 stays as the reading with e4b in the field image's
  environment, and STATUS names both. A reading below 1.00 is quoted as Unsloth faster on one stack, not set aside.
- **P29 FALSIFIED:** the box ran a different route than the position it replaces; nothing is quoted from it.
- **P28** is a replication of TC1 amendment 33's P51 on another card; it moves no default.

**Budget.** One H100 NVL, $2.80/h GPU ceiling (disk billed on top), 3 h guard (a voided draw adds up to one arm's time), venv-unsloth
built for e4b's same-stack arms. About $4 invoiced; this is in the standing no-ask tier.
