# TC1 amendment 13 read (#945): grouped-nf4-gemm's trimmed LoRA delta makes e4b's training step 6–9 % faster on a 5090 — P20 and P21 HELD

**Box:** `tc1-5090-42` (instance 54006898, AMD Ryzen 9 7900, RTX 5090, driver 580.178.04, $0.33; receipts in
[`receipts/tc1-5090-42/`](receipts/tc1-5090-42/)). The code was e4b `e129650` and grouped-nf4-gemm `58fb19d` (#440).
Pre-registration: [`../../tc1/TC1-PREREG.md`](../../tc1/TC1-PREREG.md), amendment 13.

**What was asked.** e4b against itself at the field recipe, on the post-#945 sync path. The two sides were:

- `*_lean0`: `NF4_QLORA_LEAN_DELTA=0`, grouped-nf4-gemm's previous padded LoRA delta;
- `*_lean1`: its trimmed body (#440), which uses a flat row index, a scatter-backward gather and no zero fill, and puts `scaling`
  on the gathered rows, skipped at 1.

It ran on the shipped and matched arms, two draws each, in ABBA order. Every arm is VALID, and each record names the body it ran.
On every arm the padded path served all 16,896 delta calls.

## Readings

| arm | side | s/step (steps 11+) d1 / d2 | held-out at N d1 / d2 | peak GB | J/step d1 / d2 |
|---|---|---|---|---|---|
| shipped | previous body | 2.371 / 2.350 | 0.8113 / 0.8160 | 24.58 | 826 / 745 |
| shipped | trimmed | 2.218 / 2.217 | 0.8110 / 0.8127 | 24.58 | 730 / 728 |
| matched | previous body | 2.949 / 2.955 | 0.8497 / 0.8516 | 27.82 / 27.85 | 1,009 / 1,021 |
| matched | trimmed | 2.728 / 2.654 | 0.8489 / 0.8502 | 27.24 | 859 / 882 |

- **P20 HELD:** trimmed / previous is **0.939** on the shipped arm, with the four cross-draw ratios at 0.935–0.944 (band 0.90–0.99).
- **P21 HELD:** **0.911** on the matched arm, cross-draw 0.898–0.925.
- Every pair is stable, the held-out loss is unchanged, and the matched arm's peak VRAM falls 0.6 GB.

The matched arm gains more because its delta is fp32. The multiply that the trim removes is twice the bytes there, as are the padded
copies.

## What follows

By amendment 13's decision rule, the trimmed body stays grouped-nf4-gemm's default; it has been the default since #440, because it
is exact. Register: `e4b.train.lora-delta-lean.qwen3.5090.2026-10-03`. The cross-framework positions predate the change and stand as
measured.

The next target from the same profile is the grouped GEMM's M-tile rule: grouped-nf4-gemm#441 adds the opt-in
`GNF4_PREFILL_TILE_RULE=cost`, and amendment 14 (#957) registers its 5090 A/B.
