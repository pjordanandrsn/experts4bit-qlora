# Results — P50: what the bf16-early-layers remedy costs for Gemma-4 (H100 NVL, 2026-09-19)

Pre-registration: [`P50-PREREG.md`](P50-PREREG.md). Run `p50-gemma4keep` (H100 NVL, vast 51577120, e4b `6979120`, gnf4 v0.32.0), receipt `receipts/experts4bit-qlora/2026-09-19/p50-gemma4keep/p47/`, read by `bench/p50/p50_reduce.py`. K0 passed; control (i) refused the decode scorer again (reference self-KL 0.2787) → prefill on both sides, the same cached bf16 reference and 200 prompts as P44-b / P47 / P48 / P49. Every row carries `verify_moe_4bit`'s k bf16 / 30−k quantised counts and an expert-bytes census.

## The curve

| k (bf16 experts in layers 0..k−1) | NF4 layers | KL nats/token | top-1 | expert store | × all-NF4 | × all-bf16 |
|---|---|---|---|---|---|---|
| 0 (all NF4 — P47's `loader_nf4`) | 30 | 1.0837 | 0.638 | 11.96 GB | 1.00× | 0.28× |
| **5** | 25 | 0.8425 | 0.678 | 17.06 | 1.43× | 0.40× |
| **10** | 20 | 0.4618 | 0.763 | 22.16 | 1.85× | 0.52× |
| **15** | 15 | 0.1334 | 0.862 | 27.25 | 2.28× | 0.64× |
| **20** | 10 | 0.0469 | 0.916 | 32.35 | 2.70× | 0.76× |
| **24** | 6 | 0.0258 | 0.933 | 36.42 | 3.04× | 0.86× |
| 30 (all bf16 — P48's bound is layer-27-only at 0.0056) | 0 | — | — | 42.5 GB | 3.56× | 1.00× |

## Verdicts

- **P1 (anchor) — HOLDS**: k = 15 reads **0.1334**, P47's `loader_nf4_hi` **0.1334** on a different box and a later e4b head. The two lanes' identical layer set produced an identical number to four digits.
- **P2 (monotone, and k = 15 is not enough) — HOLDS**: strictly decreasing, and k = 15 sits at 0.133, 2.7× the 0.05 floor.
- **P3 (k = 20 reaches the floor) — HOLDS**: **0.0469** ≤ 0.05. Twenty of thirty expert layers must stay bf16.
- **P4 (the remedy is expensive) — HOLDS**: at k = 20 the expert store is **32.3 GB against 12.0 GB all-NF4 = 2.70×** (rule ≥ 2.5×). Read the other way, it is **76% of the all-bf16 store**: the whole quantisation buys a 24% saving on the expert weights, for 0.047 nats.
- **P5 (top-1 ≥ 0.95 at the passing k) — REFUTED, and the threshold was mine and miscalibrated.** k = 20 reads top-1 0.916. But the calibration I should have done before registering it: **gpt-oss's NF4 requant — a position this project ships and the register licenses — reads top-1 0.9366 at KL 0.0222** on the same instrument, itself under 0.95. So 0.95 was stricter than the standard the shipped baseline meets, and the refutation says more about my threshold than about the configuration. The comparable statement, with both numbers on the table: Gemma-4 at k = 20 is 0.047 nats / top-1 0.916, where gpt-oss's shipped NF4 is 0.022 nats / top-1 0.937. (`feedback_calibrate_detector_against_gold` existed and I did not apply it; recorded.)

## What this decides

The remedy works and is not worth a default. Every step of the curve is a real quality gain and a real memory cost, and the point where Gemma-4 reaches the quality floor other families reach with plain NF4 is a model whose expert weights are **76% of bf16**. Quantising Gemma-4's experts at all, on this architecture, buys about a quarter of the expert memory and costs a measurable amount of fidelity; quantising them the way every other family is quantised costs a nat.

So, per the registered decision rule (P4 holds → say it plainly): **e4b ships no NF4 expert default for Gemma-4.** `quantize_layers` is the knob, this curve is the documentation, and a user who wants the memory picks their own k with the cost in front of them. **No Gemma-4 serving or training position is quoted, and none is proposed** — the K8-gated default that P50's rule would have allowed is not worth the gate. #597 closes as explained, with a measured curve rather than a fix.

What made it inevitable is P49's probe: Gemma-4's sensitivity is positional, not magnitudinal — the same 8 % expert-branch damage costs 159× more at layer 0 than at layer 27 — so the only lever is *how many early layers you leave alone*, and that lever is memory, one layer at a time.

## Cost

≈ 17 min on an H100 NVL at ≤ $3.10/h ≈ **$0.88** against a registered $4.65.
