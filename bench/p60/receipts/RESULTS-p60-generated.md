# Results -- P60: where the expert GEMV's B=16 headroom goes (real routing, replayed)

Pre-registration: [`P60-PREREG.md`](P60-PREREG.md). Every number below is read from the run's receipts by `p60_reduce.py`.

Box: NVIDIA GeForce RTX 5090 (170 SMs), torch 2.8.0+cu128; copy bandwidth **1512 GB/s**; plans (sk) {'gate_up': 16, 'down': 6}; 128 recorded steps replayed; eids [128, 48, 16, 8].

| arm | step ms (median) | mean | min | max | `_gemv_int4_b32` ms (profiler) | reduce ms |
|---|---|---|---|---|---|---|
| served | 6.479 | 6.475 | 5.982 | 6.875 | 6.155 | 0.303 |
| sorted | 6.444 | 6.462 | 6.028 | 6.874 | 6.359 | 0.316 |
| dedup | 5.560 | 5.564 | 4.801 | 6.142 | 5.12 | 0.171 |
| floor (distinct bytes / measured bandwidth) | 4.598 | 4.606 | | | | |
| floor at #564's 1528 GB/s | 4.550 | | | | | |

Mean distinct experts per layer per step: **54.66** (recorder: 54.65966796875; layer means 48.765625..70.21875).

## Verdicts (pre-registered)

- **P0: HOLDS** -- served _gemv_int4_b32 6.154557875 ms vs census 6.34 (-2.9%)
- **P1: HOLDS** -- served - dedup = 0.919 ms/step (hold >= 0.8, refute < 0.3)
- **P2: BETWEEN BANDS** -- dedup / floor = 1.209 (hold <= 1.15, refute > 1.30)
- **P3: HOLDS** -- sorted / served - 1 = -0.55% (hold within +-3 %)
- **P4: REFUTED** -- mean distinct 54.66 vs P57 58.67 (+-3)

**Decision rule:** P1 holds -> a grouped expert GEMV (one weight read per expert for all its rows) is licensed to BUILD (grouped-nf4-gemm lane K18, pre-registered before code)
