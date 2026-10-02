# TC1c — the same matched set on one H100 NVL: the 5090 position reverses (lane TC1c of #835; 2026-10-02)

Pre-registration: [`../../tc1/TC1C-PREREG.md`](../../tc1/TC1C-PREREG.md) — TC1's `qwen3` token unchanged on an H100 NVL
(`TC1_GPU_CLASS="H100 NVL"`), the same harness at e4b `09b0f6a` (0.38.1; `experts4bit/` byte-identical to TC1's 079a422),
grouped-nf4-gemm `846b512`, the same tokens (sha `bfc742f67e37`), the same per-slot init (`f7832488926eda91`), fp32 adapters in both
frameworks, Unsloth 2026.9.14 on torch 2.12.1+cu130 with `grouped_mm` engaged on every step. Receipts: `receipts/tc1c-h100-2/`
(private store `receipts/experts4bit-qlora/2026-10-02/tc1c-h100-2/tc1/`); one box, Vast verified-secure instance 53802633
(AMD EPYC 9534, 224 vCPU, 1.58 TB host RAM, driver 595.71.05), $4.88. The first draw (`tc1c-h100-1`) was refused at $0 before any
instance existed (its manifest carried the 5090 pre-flight exclusion receipts, not same-class for an H100).

## The position (register `e4b.train.h2h.unsloth.qwen3.h100.2026-10-02`)

| | e4b `fused_attn4_m` | Unsloth `ckpt_unsloth_m` | reading |
|---|---|---|---|
| s/step, median of steps 11..20, two draws | 4.130 / 4.065 (**4.097**) | 2.539 / 2.553 (**2.546**) | **Unsloth/e4b 0.621 [0.615, 0.628]** over the four cross-draw ratios — **Unsloth faster per step by 1.61 ×**; both STABLE (1.6 % / 0.5 %) |
| tokens/s | 370.4 | 491.1 | |
| peak VRAM | 27.86 GB | **24.27 GB** | Unsloth lower by 3.59 GB |
| energy per step | 742.5 J | **466.8 J** | Unsloth ×0.63 |
| held-out loss at N = 20 | 1.9468 → 0.8476 / 0.8490 | 1.9305 → 0.8517 / 0.8498 | Δ +0.0041: **EQUIVALENT** (band 0.0078 = 3 × the in-draw fused-vs-reference Δ; floor 0.0019); step-0 Δ −0.0163 reads NEAR (the two quantisers' bytes on Hopper's kernels; ≤ 0.05) |

e4b's fused path against its own reference on the same box: Δ final 0.00414, median step |Δ| 0.00218 — PASS; ×12.58 faster than
the reference (51.94 s/step). The ratio's sign is the finding: **on the card where `torch._grouped_mm` is first-class, Unsloth's
dequant-then-dense-grouped-GEMM path beats e4b's fused kernel per step, at lower VRAM and lower energy, with the same loss.**
TC1c's registered P1 (Unsloth/e4b in [0.5, 2.0] and lower than on the 5090) holds on the number; its decision rule — below 1.0 the
public "e4b faster" position is withdrawn for this card class — applies. The 5090 position (1.437) stands as a 5090 position.

## Why the two cards disagree (what the profiles show)

| profiled matched arm | box | s/step | device-busy | device events / step | CPU-side ops / step |
|---|---|---|---|---|---|
| e4b `fused_attn4_m_prof` | RTX 5090 (Threadripper 3975WX) | 6.169 | 0.480 | 140,809 | 753,129 |
| e4b `fused_attn4_m_prof` | H100 NVL (EPYC 9534) | 4.306 | 0.562 | 139,181 | 744,465 |
| Unsloth `ckpt_unsloth_prof` | RTX 5090 | 9.868 | 0.234 | 612,179 | 7,430,931 |
| Unsloth `ckpt_unsloth_prof` | H100 NVL | 2.828 | 0.308 | 81,780 | 515,683 |

e4b issues the same ~139 k device events per step on both cards; Unsloth issues 612 k on the RTX 5090 and 82 k on the H100 — 7.5 ×
fewer — with 14 × fewer CPU-side ops. On sm_120 (torch 2.12.1+cu130) the grouped GEMM Unsloth routes through is not one launch per
call; on sm_90 it is. That is consistent with the two positions: on the 5090 both paths are launch-bound and e4b's single fused launch
wins; on the H100 Unsloth's path becomes a handful of large GEMMs and the dequant-then-GEMM arithmetic beats the fused NF4 kernel.
Stated as what the profiles show; which torch code path sm_120 takes is not established here.

## The other arms on this box

- HF + PEFT (bf16 experts, `target_parameters`, double-quant on the attention): loaded in 23.5 s at 62 GB resident — it fits on 80 GB,
  as TC1c predicted — then produced no training step inside its registered 1,800 s alarm: **ALARM, not measured** (P4 UNTESTED; the
  alarm was sized for the 5090's OOM, not for bf16 experts training through PEFT's weight-side fold). No HF position on this card.
- axolotl 0.20.0: its venv installed (amendment 3) and the arm died after load in transformers 5.17.0's Qwen3-MoE router **Corrected 2026-10-02 (TC1 amendment 4): the harness's no-autocast loop met axolotl's fp32 `.gate` router; not an axolotl result.**
  (`F.linear(hidden_states, self.weight)`: bf16 vs fp32) before writing a receipt — HARNESS_ERROR here, an UNSUPPORTED row with that
  message once TC3 amendment 4's wrapper is on the box. The same crash on the 24 GB box (`tc3-4090-1`).
- The mb1 secondary pair did not run (HF did not OOM); the native rows and the torch-2.8 row are not part of this token.

## Predictions scored (TC1's P-set on this box, mechanically; TC1c's own P1–P4 in the registration)

P1 (TC1's band) FALSIFIED at 0.621; P2 HELD (draws); P3 HELD (EQUIVALENT); P5 UNTESTED (ALARM, not OOM); P6 UNTESTED (HARNESS_ERROR);
P8 FALSIFIED (device-busy 0.308); P9 HELD (parity). TC1c: P1 HELD on the number ([0.5, 2.0], lower than the 5090) with its decision
rule applied; P2 HELD; P3 HELD; P4 UNTESTED (HF ALARM).
