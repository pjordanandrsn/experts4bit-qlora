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
