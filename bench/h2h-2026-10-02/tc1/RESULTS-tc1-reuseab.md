# TC1 amendment 20 read (#945): grouped-nf4-gemm's per-pass host reuse makes e4b's training step 5–7 % faster on a 5090 — P33 and P34 HELD

**Box:** `tc1-5090-51` (instance 54095822, AMD EPYC 7C13, RTX 5090, driver 580.95.05; receipts in
[`receipts/tc1-5090-51/`](receipts/tc1-5090-51/)). Vast invoiced **$3.14** for it: GPU $0.39, storage $0.18, download $2.57 (65.7 GB at this host's $0.039/GB) and upload $0.004. Corrected 2026-10-04: this page first quoted the receipt's figure, $0.36, which was GPU rate × runtime. The instance billed $0.84/h: $0.573/h for the GPU, which
cleared the declared $0.69/h ceiling, plus $0.267/h for its 320 GB disk, which the launcher's ceiling does not count
(adertha-agents#140). The code was e4b `186265d` and grouped-nf4-gemm `192f63f` (#444). Pre-registration:
[`../../tc1/TC1-PREREG.md`](../../tc1/TC1-PREREG.md), amendment 20.

**What was asked.** e4b against itself at the field recipe, with every other current default in force: the post-#945 sync path, the
trimmed LoRA delta, the cost tile rule and the fused RMSNorm. The two sides were:

- `*_reuse0`: `GNF4_HOST_REUSE=0`;
- `*_reuse1`: `GNF4_HOST_REUSE=1`. Within each MoE layer pass the repeated index uploads come from a memo, the down LoRA delta reuses
  the gate_up delta's plan, and the adapters are gathered with a scatter backward instead of a sorted one.

It ran on the shipped and matched arms, two draws each, in ABBA order. Every arm is VALID, and each `reuse_ab` record names the flag
in force and the process's hit counts.

## Readings

| arm | side | s/step (steps 11+) d1 / d2 | held-out at N d1 / d2 | J/step d1 / d2 | peak GB | memo hits (uploads / plans) |
|---|---|---|---|---|---|---|
| shipped | off | 3.146 / 3.075 | 0.8146 / 0.8147 | 831 / 772 | 24.58 | 0 / 0 |
| shipped | on | 2.870 / 2.934 | 0.8154 / 0.8141 | 746 / 748 | 24.58 | 16,465 / 8,528 |
| matched | off | 3.875 / 4.008 | 0.8500 / 0.8493 | 977 / 989 | 27.23 | 0 / 0 |
| matched | on | 3.697 / 3.799 | 0.8481 / 0.8528 | 958 / 963 | 27.21 | 16,465 / 8,528 |

- **P33 HELD:** on / off is **0.933** on the shipped arm, with the four cross-draw ratios at 0.912–0.954 (band 0.92–0.99).
- **P34 HELD:** **0.951** on the matched arm, cross-draw 0.922–0.980 (band 0.93–0.99).
- Every pair is stable (within 3.4 %). Held-out loss at step 0 is 1.9441 on all eight arms. At N the mean of on minus off is
  +0.0001 (shipped) and +0.0008 (matched), inside the spread between one side's two draws (up to 0.0047). Energy per step is lower,
  and peak memory is unchanged.
- The plan memo hit once per MoE layer pass (8,528 hits against 8,368 misses), as designed: the down delta reuses the gate_up
  delta's plan. About 37 % of index uploads were answered by the memo.

## What follows

Both stable ratios are at or below 0.99, so by amendment 20's decision rule `GNF4_HOST_REUSE` becomes grouped-nf4-gemm's default
(grouped-nf4-gemm#446; `GNF4_HOST_REUSE=0` restores the old behaviour). Register: `e4b.train.host-reuse.qwen3.5090.2026-10-04`. e4b
against itself on one host; the cross-framework positions predate the change and stand as measured.
