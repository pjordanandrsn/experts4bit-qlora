# TC1 amendment 21 read (#945): keeping MoE activations makes e4b's training step 7–17 % faster on a 5090, at +2.3–4.5 GB — P35, P36 and P37 HELD

**Box:** `tc1-5090-52` (instance 54100315, AMD EPYC 7C13, RTX 5090, driver 580.95.05, $0.35; receipts in
[`receipts/tc1-5090-52/`](receipts/tc1-5090-52/)). The code was e4b `65d3fc1` (with #1007) and grouped-nf4-gemm `1e41298` (#445).
This predates grouped-nf4-gemm#446, so host reuse was off on both sides. Pre-registration:
[`../../tc1/TC1-PREREG.md`](../../tc1/TC1-PREREG.md), amendment 21.

**What was asked.** e4b against itself at the field recipe, with every other default in force:

- `*_keep0`: whole-layer gradient checkpointing, the default. Every MoE forward is recomputed in backward.
- `*_keep1`: `E4B_MOE_KEEP_LAYERS=n` with `NF4_QLORA_COMPACT_DELTA=1`. Only attention is checkpointed in the last n decoder layers,
  whose MoE activations are kept instead. n is 32 on the shipped arm and 16 on the matched arm, sized to the 5090's headroom.

Gradients are identical by construction (`torch.equal` under deterministic mode on an A2000 slice). It ran on two draws each, in
ABBA order. Every arm is VALID, and each `keep_ab` record names the layers kept and the compact delta's state.

## Readings

| arm | side | s/step (steps 11+) d1 / d2 | held-out at N d1 / d2 | J/step d1 / d2 | peak GB |
|---|---|---|---|---|---|
| shipped | keep0 | 3.083 / 2.989 | 0.8105 / 0.8134 | 829 / 753 | 24.58 |
| shipped | keep1 (32 of 48) | 2.536 / 2.536 | 0.8141 / 0.8173 | 643 / 652 | 29.09 |
| matched | keep0 | 3.908 / 3.914 | 0.8515 / 0.8513 | 993 / 986 | 27.12 / 27.16 |
| matched | keep1 (16 of 48) | 3.642 / 3.600 | 0.8504 / 0.8551 | 923 / 920 | 29.40 |

- **P35 HELD:** keep1 / keep0 is **0.835** on the shipped arm, with the four cross-draw ratios at 0.823–0.849 (band 0.80–0.95).
- **P36 HELD:** **0.926** on the matched arm, cross-draw 0.920–0.932 (band 0.86–0.98).
- **P37 HELD:** the keep1 peaks are 29.09 and 29.40 GB (bound 31.0). Mean held-out at N moved +0.0037 (shipped) and +0.0014
  (matched), inside the 0.005 bound and within the step's existing run-to-run variation. Step-0 held-out is 1.9441 on all eight arms.
- **Memory.** Each kept layer cost about 141 MB of peak on both arms: +4.51 GB for 32 layers and +2.26 GB for 16. That is above
  the 121 MB the A2000 slice predicted, and inside the bound. Energy per step is lower on both arms.

## What follows

By amendment 21's decision rule, P37 HELD with both ratios at or below 0.99 makes `E4B_MOE_KEEP_LAYERS` (with
`NF4_QLORA_COMPACT_DELTA=1`) the recommended setting for cards with the headroom. That is about 141 MB per layer at this fixture's
largest micro-batch, so 32 layers where 4.5 GB is spare and 16 where 2.3 GB is spare. It stays opt-in: the right n depends on the
card, the model and the micro-batch. Register: `e4b.train.moe-keep.qwen3.5090.2026-10-04`. TC1c amendment 2 measures the same
setting with all 48 layers kept on an H100 NVL (`tc1c-h100-4`). The cross-framework positions are quoted with the default settings
and stand as measured.
