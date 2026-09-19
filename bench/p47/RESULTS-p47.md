# Results — P47: where Gemma-4's 1.08 nats come from (H100 NVL, 2026-09-19)

Pre-registration: [`P47-PREREG.md`](P47-PREREG.md) (+ amendment 1). Run `p47-gemma4diag` (H100 NVL, vast 51554073, e4b `d8b2ceb`, gnf4 v0.32.0), receipt `receipts/experts4bit-qlora/2026-09-19/p47-gemma4diag/p47/`, read by `bench/p47/p47_reduce.py`. K0 controls passed on the box; control (i) again refused the decode scorer (reference self-KL 0.279) so every row is PREFILL-shaped on both sides, against the same cached bf16 reference and the same 200 prompts / 5,879 tokens as P44-b.

| arm | builder | NF4 experts in layers | KL nats/token | top-1 | general / technical / code / longctx | stacks quantised / bf16 |
|---|---|---|---|---|---|---|
| `served_nf4` | P44-b's served stack (anchor) | all 30 | **1.0770** | 0.638 | 1.032 / 1.234 / 1.388 / 0.868 | (served census) |
| `loader_nf4` | training loader alone | all 30 | **1.0837** | 0.640 | 1.244 / 1.502 / 1.386 / 0.776 | 30 / 0 |
| `loader_nf4_lo` | training loader | **0–14** | **1.0577** | 0.644 | 1.194 / 1.430 / 1.361 / 0.765 | 15 / 15 |
| `loader_nf4_hi` | training loader | **15–29** | **0.1334** | 0.862 | 0.133 / 0.198 / 0.174 / 0.095 | 15 / 15 |
| `loader_bf16experts` | training loader, `quantize_layers=set()` | none | NOT MEASURED | | | refused by the loader's zero-quantized-layers guard (amendment 1) |

## Verdicts

- **P1 (anchor) — HOLDS**: 1.0770, P44-b run 5's 1.077 to three digits.
- **P2 (the model, not the stack) — HOLDS**: loader − served = +0.0067 nats (rule < 0.05). The arena bake, the placement solver, `enable_hybrid_tier` and hot-residency dispatch add nothing; **the quantised model carries the whole gap**, and it is the same model a training step holds — so #558's Gemma-4 training-parity failure is the same defect seen through the loss.
- **P3 (modelling) — NOT READ** (the arm was refused, amendment 1). Partial evidence from the `hi` row: e4b's Gemma-4 model with bf16 experts in layers 0–14 and NF4 in 15–29 is 0.133 nats / top-1 0.86 from the checkpoint, so the modelling path is not grossly wrong — but 0.13 is still 6× gpt-oss's NF4 requant and includes 15 NF4 layers, so a modelling component is not excluded.
- **P4 (depth) — HOLDS, hard.** The first half carries **89 %** of lo + hi (rule > 70 %), and the halves add (|lo + hi − full| / full = 0.10, rule ≤ 0.30): **NF4 in layers 0–14 alone reproduces 98 % of the full gap; NF4 in layers 15–29 alone costs 0.13.** Per stratum the `lo` row is within 5 % of the full row everywhere; the `hi` row is 0.10–0.20 everywhere.
- **P5 (reference self-inconsistency replicates) — HOLDS**: 0.279 (0.279 in P44-b runs 3 and 5). A property of the checkpoint under transformers 5.16.1, recorded four times now.

## What this says about #597

The nat is **NF4 quantisation of the EARLY layers' experts** — not serving, not (grossly) modelling, and not "4-bit costs a nat" (the same instrument reads four other families' NF4 at 0.02–0.1, and this family's own layers 15–29 at 0.13). Something about layers 0–14's expert tensors, or how the loader fuses/scales them, does not survive block-64 NF4. A free CPU probe on the LAN copy of the checkpoint (per-layer, per-expert weight statistics and NF4 round-trip error) is running; its read and the per-layer lane it motivates are pre-registered as P48.

## Cost

`p47-gemma4diag`: 21 min on an H100 NVL at ≤ $3.10/h ≈ $1.10 (registered $6.20; the loader arms took ~80 s each including load — far under the served-arm estimate). Torn down with proof.
