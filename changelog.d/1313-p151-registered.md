### P151 registered (#1313): D1's served read of the dense single-row GEMV on Qwen3's attention projections

`bench/p151` registers the served read of `E4B_DENSE_GEMV_BW=1`. The knob routes Qwen3-30B-A3B's one-row decode GEMVs
on the fused q/k/v and o projections through grouped-nf4-gemm K35's kernel. The read runs P143 Amendment 3's form on one
RTX 5090 with a power limit of at least 575 W: eight ABBA arms at five reps, P143 A2's SANE gate, and the same bar
(FLIP iff the minimum of four W1 pairs >= 1.03 and g16 >= 0.99). The arms' engagement and power limits are checked.
P148 projected 3.35 % of the unprofiled 3.913 ms step, so the registration predicts NO_FLIP at least as likely as FLIP.
Nothing changes by default before that read.
