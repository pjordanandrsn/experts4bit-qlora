# TC1 amendment 14 read (#945): the grouped GEMM's M-tile height from the group sizes makes e4b's training step 3–8 % faster on a 5090 — P22 and P23 HELD

**Box:** `tc1-5090-43` (instance 54011443, AMD Ryzen 9 7950X, RTX 5090, driver 580.159.03, $0.41; receipts in
[`receipts/tc1-5090-43/`](receipts/tc1-5090-43/)). The code was e4b `43f36c7` and grouped-nf4-gemm `c8f0adc` (#441).
Pre-registration: [`../../tc1/TC1-PREREG.md`](../../tc1/TC1-PREREG.md), amendment 14.

**What was asked.** e4b against itself at the field recipe, on the trimmed LoRA delta and the post-#945 sync path. The two sides were:

- `*_tilemax`: `GNF4_PREFILL_TILE_RULE=max`, the grouped GEMM's M-tile height keyed on the largest group;
- `*_tilecost`: `GNF4_PREFILL_TILE_RULE=cost`, the height that minimises `tiles x (96 + BLOCK_M)` over the actual group sizes.

It ran on the shipped and matched arms, two draws each, in ABBA order. Every arm is VALID, and each `tile_ab` record names its rule
and the tile heights it launched.

## Readings

| arm | side | s/step (steps 11+) d1 / d2 | held-out at N d1 / d2 | J/step d1 / d2 | tile heights launched (128 / 64 / 32 / 16) |
|---|---|---|---|---|---|
| shipped | max | 2.232 / 2.233 | 0.8105 / 0.8115 | 763 / 700 | 15,592 / 1,190 / 114 / 0 |
| shipped | cost | 2.077 / 2.049 | 0.8095 / 0.8105 | 657 / 645 | 1,256 / 10,790 / 4,096 / 754 |
| matched | max | 2.618 / 2.628 | 0.8499 / 0.8533 | 849 / 891 | 15,562 / 1,224 / 110 / 0 |
| matched | cost | 2.533 / 2.543 | 0.8470 / 0.8484 | 793 / 804 | 1,320 / 10,768 / 4,052 / 756 |

- **P22 HELD:** cost / max is **0.924** on the shipped arm, with the four cross-draw ratios at 0.918–0.930 (band 0.85–0.97).
- **P23 HELD:** **0.968** on the matched arm, cross-draw 0.964–0.971 (band 0.88–0.98).
- Every pair is stable, held-out loss is unchanged, and energy per step is lower.
- Qwen3's real router puts one hot expert in nearly every call, so `max` launched 128-row tiles on about 92 % of calls. `cost`
  launched mostly 64-row (64 %) and 32-row (24 %) tiles.

## What follows

Both stable ratios are at or below 0.99, so by amendment 14's decision rule `cost` becomes grouped-nf4-gemm's default
(grouped-nf4-gemm#442; `GNF4_PREFILL_TILE_RULE=max` restores the old rule). Register: `e4b.train.prefill-tile-rule.qwen3.5090.2026-10-03`.
The cross-framework positions predate the change and stand as measured.
