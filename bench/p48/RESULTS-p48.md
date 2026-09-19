# Results — P48: one NF4 layer at a time on Gemma-4-26B-A4B-it (H100 NVL, 2026-09-19)

Pre-registration: [`P48-PREREG.md`](P48-PREREG.md) (amendment 1 = the disk floor after run 1). Run `p48-gemma4layer-3` (H100 NVL, vast 51566703, e4b `47e6f0b`, gnf4 v0.32.0; runs 1–2 were host-limited draws: a 32 GB overlay, then a box whose ssh never authenticated), receipt `receipts/experts4bit-qlora/2026-09-19/p48-gemma4layer-3/p47/`, read by `python bench/p48/p48_reduce.py <dir> --probe bench/p48/g4_expert_probe.jsonl`. K0 passed on the box; control (i) again refused the decode scorer (reference self-KL 0.2787) → prefill on both sides, the same cached bf16 reference and 200 prompts as P44-b/P47. Every one of the 30 rows carries `verify_moe_4bit` = 1 quantised / 29 bf16 stacks (the builder check).

## The profile — NF4 experts in exactly ONE layer, bf16 everywhere else

| layer | KL nats/token | top-1 | layer | KL | top-1 | layer | KL | top-1 |
|---|---|---|---|---|---|---|---|---|
| 0 | 0.8916 | 0.680 |  10 | 0.3317 | 0.818 |  20 | 0.0176 | 0.947 |
| 1 | 0.8681 | 0.684 |  11 | 0.2303 | 0.837 |  21 | 0.0144 | 0.952 |
| 2 | 0.7982 | 0.686 |  12 | 0.1920 | 0.843 |  22 | 0.0114 | 0.960 |
| 3 | 0.6804 | 0.727 |  13 | 0.1277 | 0.877 |  23 | 0.0098 | 0.962 |
| 4 | 0.6676 | 0.725 |  14 | 0.1007 | 0.891 |  24 | 0.0090 | 0.964 |
| 5 | 0.6214 | 0.739 |  15 | 0.0833 | 0.892 |  25 | 0.0063 | 0.970 |
| 6 | 0.5097 | 0.759 |  16 | 0.0470 | 0.914 |  26 | 0.0073 | 0.967 |
| 7 | 0.4436 | 0.767 |  17 | 0.0354 | 0.925 |  27 | 0.0056 | 0.970 |
| 8 | 0.4245 | 0.783 |  18 | 0.0254 | 0.939 |  28 | 0.0077 | 0.965 |
| 9 | 0.4011 | 0.789 |  19 | 0.0166 | 0.945 |  29 | 0.0072 | 0.961 |

Sum of the 30 rows: **7.59** nats; all 30 layers NF4 at once (P47 `loader_nf4`): **1.084**.

## Verdicts

- **P1 (near-additivity) — REFUTED.** The singles sum to 7.0× the all-layers row (rule [0.5, 1.5]). Strongly SUB-additive: once one early layer is NF4 the rest add little (layer 0 alone is 83 % of the all-layers gap). The profile is a ranking of sensitivity, not a decomposition of the 1.08.
- **P2 (concentration) — HOLDS**: the five largest rows are layers 0–4 and carry 51 % of the sum (rule ≥ 50 %); **P2b HOLDS**: layers 0–14 carry 96 % (rule ≥ 80 %). The profile is monotone in depth: 0.89 at layer 0, 0.10 at layer 14, 0.006 at layer 27 — 150×.
- **P3 (the modelling bound) — HOLDS**: the smallest row is **0.0056 nats at layer 27** (rule ≤ 0.02): e4b's Gemma-4 model with 29 of 30 expert stacks in bf16 matches the checkpoint to the other families' NF4 floor (gpt-oss NF4 requant reads 0.022 on this instrument). **The P3 that P47 could not read: e4b's Gemma-4 modelling is faithful; the whole #597 gap is NF4 on early layers' experts.**
- **P4 (the weights do not predict it) — REFUTED by its registered statistic, and the follow-up says the weights still do not explain it.** Spearman(per-layer KL, probe NF4 relative error) = 0.61 (rule |ρ| < 0.5). But the probe's per-layer relative error spans 0.0923–0.0929 (0.6 %) while the KL spans 150×: both are monotone in depth, which is all a rank correlation can see, and a 0.6 % weight-space difference cannot carry a 150× effect. As the registered rule required, the probe was re-run **per block** (`g4_block_probe.py`, layers 0 / 1 / 7 / 14 / 21 / 29, both projections, committed as `g4_block_probe.jsonl`):

| layer | proj | block rel err p50 | p99 | max | outlier blocks (absmax > 6·rms) | energy in them | sq-error share of the top 1 % blocks |
|---|---|---|---|---|---|---|---|
| 0 | `gate_up_proj` | 0.0922 | 0.1164 | 0.194 | 0.13 % | 0.37 % | 3.0 % |
| 0 | `down_proj` | 0.0919 | 0.1130 | 0.178 | 0.15 % | 1.25 % | 3.3 % |
| 1 | `gate_up_proj` | 0.0921 | 0.1135 | 0.184 | 0.04 % | 0.08 % | 2.3 % |
| 1 | `down_proj` | 0.0917 | 0.1124 | 0.177 | 0.25 % | 1.73 % | 3.7 % |
| 7 | `gate_up_proj` | 0.0920 | 0.1135 | 0.202 | 0.09 % | 0.21 % | 2.6 % |
| 7 | `down_proj` | 0.0922 | 0.1147 | 0.198 | 0.27 % | 1.63 % | 3.9 % |
| 14 | `gate_up_proj` | 0.0920 | 0.1134 | 0.194 | 0.13 % | 0.39 % | 3.1 % |
| 14 | `down_proj` | 0.0920 | 0.1135 | 0.202 | 0.19 % | 1.34 % | 3.7 % |
| 21 | `gate_up_proj` | 0.0920 | 0.1150 | 0.203 | 0.21 % | 0.53 % | 3.2 % |
| 21 | `down_proj` | 0.0919 | 0.1136 | 0.196 | 0.19 % | 1.61 % | 3.9 % |
| 29 | `gate_up_proj` | 0.0917 | 0.1123 | 0.183 | 0.11 % | 0.40 % | 3.0 % |
| 29 | `down_proj` | 0.0919 | 0.1127 | 0.207 | 0.26 % | 3.20 % | 5.3 % |

  Uniform: p50 0.092 and p99 0.113 in every layer, < 0.3 % outlier blocks, the worst 1 % of blocks carrying 3–5 % of the squared error everywhere. **Block-64 NF4 does the same ~9 % relative damage to layer 0's weights as to layer 27's; layer 0's costs 0.89 nats and layer 27's 0.006.** The mechanism is activation-side — the early layers' expert OUTPUTS carry, or are amplified into, far more of the residual stream than the late layers' — and is to be measured on the box (per-layer expert-output scale relative to the residual, and the KL of NF4-vs-bf16 outputs on real activations), not in weight space.

## Decision (per the registered rules)

P2 ∧ P3 hold → **the defect is specific layers' expert quantisation**, and the layers are the early ones, monotone in depth. P4's refutation routed through the per-block re-probe, which came back flat, so the next lane is the one the P2 ∧ P3 rule names, on **layer 0 alone** (where one row answers the question): expert-FORMAT arms — the loader's other stores and block sizes — against the same reference, plus the activation-scale probe. Registered as **P49**. Meanwhile the e4b-side remedy is bounded: Gemma-4's first k expert layers cannot be block-64 NF4; keeping them high-precision costs ~1.5 GB bf16 per layer against 0.4 GB NF4 (k = 10 would be ~11 GB on top of the 8.5 GB model — the same card class as today's all-VRAM tier), and any per-family default of that kind gets its own K8/KL gate before a position is quoted. **Gemma-4 stays unquoted.**

## Cost

`p48-gemma4layer` runs 1–3: $0.49 (ENOSPC) + $0 (ssh never authenticated, torn down at pre-flight) + ≈ $1.35 (26 min) ≈ $1.85 against a registered $4.65. The two CPU probes were free (QNAP, LAN checkpoint).
