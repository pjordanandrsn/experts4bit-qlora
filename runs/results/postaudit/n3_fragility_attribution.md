# N3 — fragility attribution (SPECULATIVE_LANES_PLAN §S-C, pre-committed >=2x gate)

- contrast |d_i| = |loss(nf4) - loss(int8)|, resident, n=1024
- top decile: 102 examples | (layer,expert) pairs = 1024

## Flip-mass concentration (share held by top k% of pairs)

| example set | top 1% | top 5% | top 10% | pairs hit | total flips |
|---|---|---|---|---|---|
| top-decile |d_i| | 15% | 37% | 52% | 533/1024 | 2341 |
| all examples | 14% | 35% | 50% | 636/1024 | 17266 |
| bottom-decile (baseline) | 15% | 38% | 55% | 458/1024 | 1591 |

## Verdict (pre-committed gate)

- **Committed literal gate (>=2x over uniform): MET** — top-10% of pairs hold 52% of top-decile flip mass (5.2x uniform).
- **Fragility-specificity control (added): FAILS** — the LEAST-fragile decile is 55% concentrated (top-10% pairs), >= the fragile decile's 52%. Routing flips concentrate on the same experts regardless of precision fragility, so the concentration is a property of the router, not a fragility signal.
- (Fragile examples DO flip more in total — 2341 vs 1591 over the same example count — but on the same expert distribution.)
- **N3 CLOSES (on the control, transparently — the committed literal gate passed):** the mixed-precision cell's premise is that fragility localizes to identifiable experts; it does not — the flip-concentrated experts are the same for fragile and non-fragile examples, so 'top-fragility experts' is not a set distinct from 'top-flip experts', and those don't track |d_i|. Per-expert precision is not the dial. The literal >=2x gate was under-specified (concentration alone is non-diagnostic); recorded so the override is visible.
