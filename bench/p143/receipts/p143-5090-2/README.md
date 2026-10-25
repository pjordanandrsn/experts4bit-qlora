# `p143-5090-2`: step 1, the attention decode microbench (GO_FIXED)

One RTX 5090 (cc 12.0, 170 SMs, driver 580.173.02, 575 W) on an Intel Core i9-9900K host (16 CPUs): Vast instance
redacted-b6a651a0 on machine 83029, 2026-10-18T18:05:55Z to 18:09:28Z. Cost $0.029. Code: e4b `adea7321` (Amendment 1's merge,
#1762), grouped-nf4-gemm 0.46.1 (`70bc03cb`). No model.

**Verdict: GO_FIXED.** The paged FP8 attention decode is e4b's own `Fp8PagedKV` at Qwen3-30B-A3B's geometry, 48 layers
per CUDA graph. With 16 splits it saves **0.171 ms** a step at 640 tokens against the library's plan (32 splits; bar 0.131
ms, 3 % of P142's step). It is 0.040 ms *faster* at 4096 (noise 0.001), so 16 beats 32 at every context. 8 splits save
0.179 ms at 640 but lose 0.149 ms at 4096. The best split per context (an upper bound) would add only 0.008 ms over a
fixed 16. Microseconds per launch, plan / 4 / 8 / 16 / 32 splits:

| context | plan | 4 | 8 | 16 | 32 |
|---|---|---|---|---|---|
| 256 | 11.09 | 5.11 | 5.88 | 7.59 | 11.26 |
| 640 | 11.19 | 8.18 | 7.46 | 7.63 | 11.17 |
| 1024 | 11.10 | 9.72 | 7.44 | 7.67 | 11.28 |
| 2048 | 12.53 | 16.92 | 11.31 | 10.12 | 12.45 |
| 4096 | 15.27 | 29.96 | 18.36 | 14.43 | 15.33 |

The anchor read 0.966x P142's served launch. The plan matched the 32-split arm at every context, and the self-pairs were
tight. The maintainer re-derived the verdict byte for byte (#1759). The launcher's receipt and ledger row are in the
receipt store (`fa624e53`). `SHA256SUMS` covers every file here.
