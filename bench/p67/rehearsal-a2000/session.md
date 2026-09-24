P67 registered read (bench/p67/P67-PREREG.md): TOL=0.05, K=3, N_MIN=3. Quantities on the TRAIN loss; held-out beside, never gated.

### granite — reading READ
- reference: loss_last 0.95459, held-out 0.88582
- plain repeat bit-identical to the reference on this box: False
| floor draw | admissible | D_final train | D_med train | Δ held-out | Δ step-0 | first differing step | why not |
|---|---|---|---|---|---|---|---|
| `reference_attn4_perm1` | True | 0.00134 | 0.00159 | 0.00336 | 0.004590 FLAG | 0 |  |
| `reference_attn4_perm2` | True | 0.00114 | 0.00133 | 0.00025 | 0.000000 | 2 |  |
| `reference_attn4_perm3` | True | 0.00380 | 0.00205 | 0.00026 | 0.004590 FLAG | 0 |  |
| `reference_attn4_repeat` | True | 0.00165 | 0.00082 | 0.00023 | 0.000000 | 5 |  |
- floor (4 draws): F_hi final 0.00380 (lo 0.00114) -> band 0.05000; F_hi med 0.00205 (lo 0.00082) -> band 0.05000
| judged arm | status | D_final train | D_med train | Δ held-out | constant 0.05 | **floor band** | carried by | detectable | D/F_hi (final, med) |
|---|---|---|---|---|---|---|---|---|---|
| `fused_attn4` | VALID | 0.00136 | 0.00339 | 0.00031 | PASS | **PASS** | tolerance | False | 0.36, 1.65 |
| `batched_attn4` | VALID | 0.00141 | 0.00447 | 0.00103 | PASS | **PASS** | tolerance | False | 0.37, 2.18 |

