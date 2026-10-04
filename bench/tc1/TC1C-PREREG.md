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
