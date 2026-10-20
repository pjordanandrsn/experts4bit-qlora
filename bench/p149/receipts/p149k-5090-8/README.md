# `p149k-5090-8`: K37's int8 attention projections against cuBLAS bf16 at Mixtral-8x7B's shapes (K_GO)

One RTX 5090 (cc 12.0, 170 SMs, 32,607 MiB, driver 580.173.02, a **575 W** power limit read on the box before any
install, a 3,090 MHz maximum SM clock) on an Intel Core i9-9900K host (16 CPUs, 62 GiB RAM), machine 83029. The lane ran
2026-10-19T19:22:04Z to 19:25:48Z. Cost $0.038.
- **Code:** e4b `ffb1632c` (the launch commit, carrying Amendment 3's 0.75 h guard); grouped-nf4-gemm 0.47.1 at
  `38b30888` (the serve stack's pin). The prototype kernels are `bench/p149/p149_kernels.py`, staged by sha256.
- **Stack:** torch 2.8.0+cu128, triton 3.4.0, transformers 5.17.0, bitsandbytes 0.50.2 (`versions.txt`).

**Before the bench:** the power floor and refusals passed. The self-tests (`p149k_reduce.py` 22 cases, `p149k_bench.py`
8) and the kernel suite compiled on the card (135 passed, none skipped) passed.

**The bench** (`p149k.json`) ran rc 0 in 46 s (`BENCH_WALL_S`). One process, no model; every launch reads its own
layer's weight, above L2.

| shape, M | chosen config | cuBLAS A1 / A2 (ms) | kernel B1 / B2 (ms) | relative RMS error, kernel / cuBLAS |
|---|---|---|---|---|
| q, o (4096 × 4096), 1 | 4,512,4,1 | 1.4850 / 1.4849 | 0.7588 / 0.7589 | 1.670e-3 / 1.676e-3 |
| q, o (4096 × 4096), 16 | 32,256,4,1 | 1.5547 / 1.5551 | 0.7903 / 0.7906 | 1.668e-3 / 1.666e-3 |
| k, v (1024 × 4096), 1 | 4,2048,8,1 | 0.5799 / 0.5806 | 0.3064 / 0.3064 | 1.764e-3 / 1.802e-3 |
| k, v (1024 × 4096), 16 | 32,256,4,1 | 0.5686 / 0.5687 | 0.5225 / 0.5221 | 1.687e-3 / 2.361e-3 |

Times are the median of 200 timed replays of 64 launches (one shape's calls in a step). Configs read
BLOCK_N, KC (M == 1) or BK (M == 16), warps, split-K.
- **The contract** (error ≤ 1.05 × cuBLAS's): every shape passes.
- **Self-pairs:** every A2/A1 and B2/B1 is within [0.9993, 1.0012], inside the [0.985, 1.015] NOISY band.
- **The harness transfers:** cuBLAS at M == 1 reads 1.014 (q, o) and 1.022 (k, v) × P145's in-situ per-call times,
  inside [0.8, 1.25].
- **GPU state:** 575 W enforced throughout; SM clock 2,460 MHz at the start and 2,910 MHz at the end.

**Verdict: K_GO** (`verdict_p149k.json`). The W1 projected in-situ saving is 0.984 ms (q, o 0.716; k, v 0.268), against
the 0.1625 ms bar (2 % of P145's 8.125 ms W1 node period). The harness's own saving, reported only, is 0.999 ms. This is
a microbench premise: it opened the quality-and-speed box and quotes no gain into the served step.

**Files:** `p149k.json` (the bench), `verdict_p149k.json` (the reduce), `summary.txt` (the runner's summary),
`versions.txt`, and `logs/`. `SHA256SUMS` covers them and this README. `STORE-ONLY.sha256` names what stays in the
private receipt store.
