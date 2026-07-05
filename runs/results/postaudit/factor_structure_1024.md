# n=1024 factor/telemetry analysis — P-A1..P-A6 (exploratory, lanes addendum 1)

- n = 1024 examples; modes vs bf16, resident

## Deviation correlations (Pearson / Spearman)

| pair | Pearson | Spearman |
|---|---|---|
| fp4–nf4 | +0.215 | +0.250 |
| fp4–int8 | +0.079 | +0.119 |
| fp4–fp8 | +0.267 | +0.134 |
| fp4–fp16 | +0.178 | +0.115 |
| nf4–int8 | -0.017 | +0.135 |
| nf4–fp8 | -0.009 | +0.093 |
| nf4–fp16 | -0.004 | +0.044 |
| int8–fp8 | +0.327 | +0.314 |
| int8–fp16 | +0.527 | +0.318 |
| fp8–fp16 | +0.385 | +0.311 |

- eigenvalues: [1.931, 1.184, 0.818, 0.606, 0.462] — PC1 share 39%
- PC1 loadings: {'fp4': 0.319, 'nf4': 0.056, 'int8': 0.537, 'fp8': 0.52, 'fp16': 0.58}
- **P-A1 (PC1 ≥ 50%): FAILS**

## Telemetry joins

- mean flip count vs bf16: fp4 20.4, nf4 16.7, int8 3.8, fp8 6.0, fp16 2.2
- **P-A2 (Spearman(PC1 score, scattered flips) > 0.4): ρ = -0.049 — FAILS** (sign convention: |ρ| judged, loadings sign-free: |ρ| = 0.049)
- **P-A3: ρ(int8, fp16) = +0.527 (≥0.5 ✓); within-trio int8: ρ(|d|, flips) = +0.273 | fp16: ρ(|d|, flips) = +0.161 — FAILS/PARTIAL**
- **P-A4 (fp8 flips within ±25% of nf4): nf4 16.7, fp8 6.0 — FAILS** (sign condition judged with the amendment's primary)

## P-A5 — probe-set cross-validation (threshold-transfer operationalization)

- n=1024 top-decile cutoff on |nf4−bf16|: 0.0827
- pilot top-decile examples above that cutoff: 3/6 = 50% (base rate 10%) — **HOLDS (≥3×)**
- interpretation note: disjoint sets, so 'membership prediction' is operationalized as cross-set cutoff transfer; stated in the script header before the join was computed.

## P-A6 — resolved by Z3 (no computation)

- Compute paths are SHARED by construction (OFFLOAD_MEMORY_FACTS.md Z3): ρ(int8, fp16) cannot be a kernel-path artifact; it stands as example-level smooth-sensitivity if it replicates (see P-A3).
