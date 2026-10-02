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

(none yet)
