# Z1 — factor decomposition on the raw n=64 deviation vectors

Verification of `docs/SPECULATIVE_LANES_ADDENDUM_1.md` §1 (which was recovered from summary
statistics), plus the preregistered Pearson-vs-Spearman comparison. Deviations are
`loss_i(mode) − loss_i(bf16)`, resident, n=64. Raw scores:
`factor_structure_n64.json` (PC1/PC2 loadings + per-example factor scores).

## §1 verification: reproduces exactly

Pearson matrix matches the addendum to ±0.002 everywhere; eigenvalues 3.319 / 0.996 / 0.349 /
0.223 / 0.113 (PC1 66%, PC2 20%); PC2 bipolar {fp4 +0.53, nf4 +0.45} vs {int8 −0.60,
fp16 −0.41} with fp8 at +0.006; triad loadings nf4 1.003 / fp4 0.742 / fp8 0.788;
sd(int8−fp16) = 0.0132; means fp4 +0.0223, nf4 +0.0088, int8 −0.0007, fp8 −0.0094,
fp16 −0.0037. All as filed.

## Pearson vs Spearman — the shared-outlier alternative is LARGELY CONFIRMED for factor 1

| pair | Pearson | Spearman |
|---|---|---|
| fp4–nf4 | +0.744 | +0.473 |
| nf4–fp8 | +0.790 | **+0.145** |
| nf4–fp16 | +0.520 | **+0.032** |
| fp4–fp8 | +0.584 | +0.227 |
| int8–fp8 | +0.622 | +0.448 |
| **int8–fp16** | **+0.745** | **+0.636** |

The large cross-format Pearson correlations among the scattered formats collapse under
rank transformation — the PC1 "common core" is carried substantially by a small set of
**shared high-|d| outlier examples**, not by distribution-wide association. The int8–fp16
pair (factor 2) is the exception: it survives ranks (0.64), i.e. it is the
distribution-robust channel. Two consequences, both exploratory (n=64):

- The §2 FALSIFIED grading stands but sharpens: the scattered formats do not flip
  *different* subsets — they share a small *fragile-example* subpopulation that moves
  together under any weight perturbation. This is also the amendment's tail/subpopulation
  branch and raises the prior on P-A5 (probe-set enrichment).
- P-A1/P-A2 at n=1024 should be read alongside rank-based equivalents; the analysis will
  report both.

## Z3/Z4 resolution (source read, `docs/OFFLOAD_MEMORY_FACTS.md`)

All six modes share one matmul path (bf16 compute everywhere; identical `F.linear`); fp16 is
a minimal weight-perturbation point (subnormal-range rounding only), not an activation
channel. ρ(int8, fp16) cannot be a kernel-path artifact.
