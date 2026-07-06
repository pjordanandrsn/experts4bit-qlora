# N2 routed-stream Phase 0-1 (reconstruction) — results

## nf4_base: locality + margins (4011 decode tokens, 16 layers)

- mean consecutive-token Jaccard 0.2983 (range 0.2076-0.3718); global low-margin threshold None
- **P-B1** Spearman corr(near-margin frac, Jaccard) across layers = **+0.000** (committed: < -0.40 → FAILS; n=16 layers)
- **P-B2** churn ratio top-vs-bottom near-margin quartile = **0.95x** (committed: >= 1.5x → FAILS)

## nf4_adapter: locality + margins (4027 decode tokens, 16 layers)

- mean consecutive-token Jaccard 0.2989 (range 0.1983-0.3716); global low-margin threshold None
- **P-B1** Spearman corr(near-margin frac, Jaccard) across layers = **+0.000** (committed: < -0.40 → FAILS; n=16 layers)
- **P-B2** churn ratio top-vs-bottom near-margin quartile = **0.94x** (committed: >= 1.5x → FAILS)

## int8_base: locality + margins (4027 decode tokens, 16 layers)

- mean consecutive-token Jaccard 0.2968 (range 0.2054-0.3715); global low-margin threshold None
- **P-B1** Spearman corr(near-margin frac, Jaccard) across layers = **+0.000** (committed: < -0.40 → FAILS; n=16 layers)
- **P-B2** churn ratio top-vs-bottom near-margin quartile = **0.95x** (committed: >= 1.5x → FAILS)

## O1 byproduct — base vs adapter routing on identical streams

- mean per-token routed-set Jaccard = **0.7706** (eval-set S-B number was 0.9418; trace-workload confirmation)

## h(S), economics, kill table (reconstructed rule: gain < 10% at S <= 2 GB kills)

| precision | T_ovh ms | t_fetch ms | ceiling gap | best h(S<=2GB) LRU | margin-LRU | best gain | verdict |
|---|---|---|---|---|---|---|---|
| nf4 | 79.7 | 34.0 | +43% | 0.854@2.0GB | 0.854 | +34.3% | **SPARE** |
| int8 | 95.6 | 63.9 | +67% | 0.626@2.0GB | 0.626 | +33.5% | **SPARE** |
| bf16 (proxy stream) | 72.9 | 179.6 | +246% | 0.424@2.0GB | 0.424 | +43.1% | **SPARE** |

- A2 prior shape (gaps +20/+38/+79%, kill nf4 / spare int8+16-bit) graded against the
  measured column above. bf16 h(S) uses the nf4-base token stream as a routing proxy
  (flagged; routing differs across precisions by ~2% of decisions).
