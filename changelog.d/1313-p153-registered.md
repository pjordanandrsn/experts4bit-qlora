### P153 registered (#1313): D2's served read of the int8 lm_head through grouped-nf4-gemm K36

`bench/p153` registers the served read of `E4B_INT8_HEAD_BW=1`. The knob rounds Qwen3-30B-A3B's lm_head in place to
the per-row int8 grid P150 ruled GO on quality, so every row count reads that grid. It routes the head's one-row decode
GEMV through K36's kernel. The read runs P151's form on one RTX 5090 with a recorded power cap of at least 575 W: eight
ABBA arms at five reps, P143 A2's SANE gate, and the same bar (FLIP iff the minimum of four W1 pairs >= 1.03 and
g16 >= 0.99).

A warm-up process now runs first, as in P131's Amendment 2. It is integrity-checked and reported, but never scored,
because P151's read went NOISY on a slow first process. P148 projected 4.84 % of the unprofiled 3.913 ms step; at P151's
realized share that is g1 about 1.036. Nothing changes by default before the read.
