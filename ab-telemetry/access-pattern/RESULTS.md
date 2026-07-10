# Phase 0 — access-pattern results (Claims 4/5, router axis)

*Measured 2026-07-10 on a rented RTX 3090 (~10 min, ~$0.10). Forward-only, counting distinct
experts — no training, no timing. Owned hardware couldn't cleanly host the 30B (38 GB QNAP free
vs ~61 GB), so the measurement ran on a cheap pod; the harness + analysis were $0/local.*

## Headline: real routing widens the sparse-read window far beyond the uniform null

The occupancy null (uniform routing) predicts the union saturates almost immediately — read
fraction 0.9 at ~36 tokens for Qwen3-30B — which would make the SSD-tier thesis useful only at
decode scale. **The real measurement says the opposite:**

| model | E | real 0.5-cross | null 0.5-cross | widening | real 0.9-cross | plateau @2048 tok |
|---|---|---|---|---|---|---|
| **Qwen3-30B-A3B** | 128 | **57.5 tok** | 10.7 | **5.4×** | **never reached** | **0.796** |
| OLMoE-1B-7B | 64 | 14.1 tok | 5.2 | 2.7× | ~110 tok | 0.999 |

**Qwen3-30B never reads more than ~80% of its experts per layer, even at a full 2048-token
forward** — ~28% stay persistently cold (stable load imbalance). At decode/short-context scale the
read fraction is far lower: 0.06 @ 1 token, 0.21 @ 8, 0.53 @ 64. So there is a broad regime where a
large fraction of experts can live on the slow tier — a real operating window for placement, and a
stable hot/cold split is exactly what a static placement map needs.

**The window widens with expert count.** 128 experts → 5.4× widening and a 0.80 plateau; 64 experts
→ 2.7× and near-full saturation. More experts ⇒ wider window ⇒ more SSD-tier benefit. This is the
quantitative argument that the wide-expert models (256/512, D2) are where the thesis is strongest —
and that 30B already shows a usable window at 128.

## Confound checked (measurement validity)

First runs used a single sentence repeated 500× and showed lower plateaus (Qwen3 0.72, OLMoE 0.95).
That is a real confound — repetitive text under-samples the expert space. Re-running on **diverse
wikitext-103** shifted the curves up (OLMoE to 0.999, Qwen3 to 0.80) but **the core finding
survived**: Qwen3 still crosses 0.5 at ~58 tokens (5.4× the null) and never reaches 0.9. The
diverse runs are canonical; the repetitive runs are kept (`access_*_.jsonl` without `_diverse`) as
the confound record.

## Sub-finding: batching reads wider than long-context at matched eff_tokens

At matched effective tokens, batched multi-sequence forwards gather a **wider** expert union than
one long sequence (Qwen3 @ 512 eff_tokens: batch-8×seq-64 = 0.82 vs batch-1×seq-512 = 0.73; OLMoE
0.995 vs 0.986). Diverse sequences sample more of the expert space than one continuation. So the
placement thesis is **most favorable for long-context single-stream** inference and less so for
batched serving — a distinction Session 4's slices should respect.

## What this scopes for Session 4 (placement axis)

The crossover for Qwen3-30B lives at **eff_tokens ≈ 30–256** (read fraction 0.4–0.7), NOT the
saturated batch grid. Session 4 measures placement `f` at those slices — e.g. eff_tokens ∈ {64,
256, 2048} (below-knee / at-knee / plateau) — and can skip the ≥1024-token cells the original grid
proposed, which the plateau shows are all in the same ~0.80-read regime.

## Artifacts

`access_reduction.json` (crossings + widening + plateau), `access_{qwen30b,olmoe}_diverse.jsonl`
(canonical), `access_{qwen30b,olmoe}.jsonl` (repetitive, confound record), `access_overlay.svg`.
Harness + pre-registration + null in the sibling files; the pre-registered null (36-token 0.9
crossing) is exactly the prediction the measurement falsified — recorded as such.
