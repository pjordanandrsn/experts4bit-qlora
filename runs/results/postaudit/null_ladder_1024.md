# n=1024 ∅-ladder re-pin — preregistered analysis

## Integrity

- jobs: 13 | GPU(s): ['NVIDIA RTX A5000']
- eval-set hash: ALL MATCH committed 3e836c1a01ab5cce…

## D2 + determinism at n=1024 (bitwise)

- fp4: 1024/1024 bitwise-identical resident vs offload
- nf4: 1024/1024 bitwise-identical resident vs offload
- int8: 1024/1024 bitwise-identical resident vs offload
- fp8: 1024/1024 bitwise-identical resident vs offload
- bf16: 1024/1024 bitwise-identical resident vs offload
- fp16: 1024/1024 bitwise-identical resident vs offload
- determinism repeat (int8-resident): 1024/1024 bitwise

## Ladder (resident, mean ± SE)

- fp4: 1.62542 ± 0.02455
- nf4: 1.61876 ± 0.02449
- int8: 1.60219 ± 0.02422
- fp8: 1.60194 ± 0.02428
- bf16: 1.60310 ± 0.02425
- fp16: 1.60266 ± 0.02427

## PRIMARY — nf4 − int8 (paired, two-sided, |t| ≥ 3)

- G_int8 = +0.01657 ± 0.00227 (sd 0.0727, n=1024, |t|=7.29)
- **PRIMARY CLEARS |t| ≥ 3**
- CO-PRIMARY Wilcoxon signed-rank: z = +12.43 (n_nonzero = 1024) — consistent

## SECONDARY (exploratory) — 15 resident pairs, Bonferroni |t| ≥ 3.0

| pair | mean d ± SE | sd | |t| | Bonferroni |
|---|---|---|---|---|
| fp4 − nf4 | +0.00666 ± 0.00298 | 0.0953 | 2.24 | — |
| fp4 − int8 | +0.02323 ± 0.00258 | 0.0826 | 9.00 | **survives** |
| fp4 − fp8 | +0.02347 ± 0.00247 | 0.0789 | 9.52 | **survives** |
| fp4 − bf16 | +0.02232 ± 0.00255 | 0.0817 | 8.74 | **survives** |
| fp4 − fp16 | +0.02276 ± 0.00251 | 0.0804 | 9.06 | **survives** |
| nf4 − int8 | +0.01657 ± 0.00227 | 0.0727 | 7.29 | **survives** |
| nf4 − fp8 | +0.01681 ± 0.00233 | 0.0747 | 7.20 | **survives** |
| nf4 − bf16 | +0.01566 ± 0.00217 | 0.0696 | 7.20 | **survives** |
| nf4 − fp16 | +0.01610 ± 0.00222 | 0.0712 | 7.24 | **survives** |
| int8 − fp8 | +0.00025 ± 0.00086 | 0.0275 | 0.29 | — |
| int8 − bf16 | -0.00090 ± 0.00062 | 0.0199 | 1.45 | — |
| int8 − fp16 | -0.00047 ± 0.00055 | 0.0175 | 0.86 | — |
| fp8 − bf16 | -0.00115 ± 0.00083 | 0.0266 | 1.39 | — |
| fp8 − fp16 | -0.00071 ± 0.00078 | 0.0250 | 0.92 | — |
| bf16 − fp16 | +0.00044 ± 0.00046 | 0.0148 | 0.95 | — |

- G_total (nf4 − bf16) = +0.01566 ± 0.00217 (sd 0.0696, n=1024, |t|=7.20)
- coverage = 106% (both clear 3σ — ratio licensed)

## Tail report (primary pair)

- top 5% of |d_i| carries 32% of Σ|d_i|; top 10% carries 45%
- d_i histogram [-0.4480 … +1.0168], 11 buckets: [3, 6, 45, 912, 44, 10, 1, 0, 1, 1, 1]

## Mechanism probe — routing disagreement

- int8/bf16: mean Jaccard 0.9955 | Spearman corr(|d_i|, disagreement) = +0.316 [within trio]
- nf4/bf16: mean Jaccard 0.9806 | Spearman corr(|d_i|, disagreement) = +0.274 [cross-cluster]
- nf4/int8: mean Jaccard 0.9804 | Spearman corr(|d_i|, disagreement) = +0.268 [cross-cluster]
- fp4/bf16: mean Jaccard 0.9764 | Spearman corr(|d_i|, disagreement) = +0.326 [cross-cluster]
- int8/fp16: mean Jaccard 0.9955 | Spearman corr(|d_i|, disagreement) = +0.317 [within trio]
- bf16/fp16: mean Jaccard 0.9975 | Spearman corr(|d_i|, disagreement) = +0.204 [within trio]
- fp4/nf4: mean Jaccard 0.9758 | Spearman corr(|d_i|, disagreement) = +0.432
- committed prediction Jaccard(int8,bf16) ≫ Jaccard(nf4,bf16): 0.9955 vs 0.9806 — HOLDS

## §4 branch table — outcomes

- G_int8 ≥ 3σ: **YES — precision gap real; program resumes**
- trio {int8,bf16,fp16} within ±3σ (resolution ±0.0019): **YES — flat above int8 ships**
- fp8 advantage over {nf4, fp4} at 3σ: **YES — promoted to finding**
- top-10% carries >50% of Σ|d_i|: **NO** (45%)
- cross-cluster corr(|d|, routing disagreement) > 0: 3/3 pairs with ρ > 0.1; within-trio ρ: ['+0.316', '+0.317', '+0.204']
