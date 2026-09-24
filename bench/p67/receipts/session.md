P67 registered read (bench/p67/P67-PREREG.md): TOL=0.05, K=3, N_MIN=3. Quantities on the TRAIN loss; held-out beside, never gated.

### gemma4 — reading READ
- reference: loss_last 1.12894, held-out 1.19205
- plain repeat bit-identical to the reference on this box: False
| floor draw | admissible | D_final train | D_med train | Δ held-out | Δ step-0 | first differing step | why not |
|---|---|---|---|---|---|---|---|
| `reference_attn4_perm1` | True | 0.12100 | 0.08037 | 0.11233 | 0.217260 FLAG | 0 |  |
| `reference_attn4_perm2` | True | 0.01124 | 0.05226 | 0.02151 | 0.057440 FLAG | 0 |  |
| `reference_attn4_perm3` | True | 0.11469 | 0.12752 | 0.18907 | 0.051820 FLAG | 0 |  |
| `reference_attn4_perm4` | True | 0.02563 | 0.07790 | 0.05350 | 0.081940 FLAG | 0 |  |
| `reference_attn4_repeat` | True | 0.02236 | 0.04509 | 0.02474 | 0.000000 | 2 |  |
- floor (5 draws): F_hi final 0.12100 (lo 0.01124) -> band 0.36300; F_hi med 0.12752 (lo 0.04509) -> band 0.38257
| judged arm | status | D_final train | D_med train | Δ held-out | constant 0.05 | **floor band** | carried by | detectable | D/F_hi (final, med) |
|---|---|---|---|---|---|---|---|---|---|
| `fused_attn4` | VALID | 0.01331 | 0.03565 | 0.01136 | PASS | **PASS** | tolerance | False | 0.11, 0.28 |
| `batched_attn4` | VALID | 0.03694 | 0.04803 | 0.03786 | PASS | **PASS** | tolerance | False | 0.30, 0.38 |
| `fused_attn4_nodgrad` | VOID | — | — | — | — | — | — | — | status 'not_run'; C1 frozen bytes not bit-exact; 0 losses for 20 steps |
- consistency (/Users/jordananderson/adertha-receipts/receipts/experts4bit-qlora/2026-09-19/tp4-c-parity-2): `fused_attn4` D_final 0.08257 D_med 0.12421 -> PASS
- consistency (/Users/jordananderson/adertha-receipts/receipts/experts4bit-qlora/2026-09-22/p56-gemma4-ladder-3): `fused_attn4` D_final 0.10223 D_med 0.10639 -> PASS
- consistency (/Users/jordananderson/adertha-receipts/receipts/experts4bit-qlora/2026-09-22/p56-gemma4-ladder-3): `batched_attn4` D_final 0.05425 D_med 0.08487 -> PASS
- consistency (/Users/jordananderson/adertha-receipts/receipts/experts4bit-qlora/2026-09-22/p56-gemma4-ladder-3): `fused_attn4_nodgrad` D_final 0.10274 D_med 0.11678 -> PASS

