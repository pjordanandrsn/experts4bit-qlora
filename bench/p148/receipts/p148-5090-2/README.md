# `p148-5090-2`: the dense bf16 decode GEMVs microbench (D1_GO+D2_GO)

One RTX 5090 (cc 12.0, 170 SMs, driver 575.64.03, a **400 W** power limit) on an AMD Ryzen Threadripper PRO 7975WX
host (64 CPUs): Vast instance 56682743 on machine 40521, 2026-10-19T06:18:28Z to 06:21:57Z. Cost $0.025. Code: e4b
`caf5a28c` (the registration's merge, #1796), grouped-nf4-gemm 0.47.1 (`38b30888`); torch 2.8.0+cu128, triton 3.4.0.
No model. [`p148-5090-1`](../p148-5090-1/README.md) was refused at create ($0), so this read ran under the registered
pre-create spare.

**Verdict: D1_GO+D2_GO.** A microbench premise, never the quoted gain. The self-tests passed, P136's premise read 19
passed, and the kernels' own suite read 61 passed, compiled on the card. Medians of 200 replays of one CUDA graph per
shape (48 launches for q/k/v and o, one for the head), in ms:

| shape | cuBLAS grid | A1 / A2 | B1 / B2 | A1 / P142 | self-pairs A2/A1, B2/B1 | choice | projected saving | harness saving |
|---|---|---|---|---|---|---|---|---|
| q/k/v | 1280 | 0.7406 / 0.7410 | 0.6787 / 0.6787 | 1.036 | 1.0006 / 0.9999 | `2,1024,4,1` | 0.0597 | 0.0618 |
| o | 512 | 0.6336 / 0.6340 | 0.5577 / 0.5581 | 1.061 | 1.0006 / 1.0008 | `2,1024,4,1` | 0.0715 | 0.0759 |
| head (int8) | 37984 | 0.3805 / 0.3805 | 0.1892 / 0.1892 | 1.009 | 1.0000 / 0.9999 | `4,1024,4,1` | 0.1895 | 0.1913 |

A is cuBLAS, B the choice; a choice is `BLOCK_N,KC,warps,split_k` (`BLOCK_N,KC,warps` for the head). The projected
saving is the smaller pair's P142 in-situ time x (1 - B/A); the bars rule on it. The harness saving is the smaller raw
pair, reported only.
- **D1** = 0.0597 + 0.0715 = **0.131137 ms**, 2.99 % of P142's step (the bar is 0.087603 ms, 2 %): GO.
- **D2** = **0.189546 ms**, 4.33 % of the step: GO.
- **Contract** (relative RMS against fp64, 4 rows of layer 0): every config passes (36, 36 and 18). q/k/v and o read
  cuBLAS's own error (1.697e-3 and 1.705e-3). The int8 head reads 1.660e-3 against its dequantized weights, where cuBLAS
  reads 1.665e-3 on the bf16 head.
- **DRAM rate** (reported): the choices ran at 1.48, 1.44 and 1.65 TB/s, 83, 81 and 92 % of 1.792. cuBLAS in the
  harness ran at 1.36, 1.27 and 1.64.
- **Selection** (reported): split-K lost at every shape. Its best graph read 0.818 ms at q/k/v and 0.696 ms at o,
  against 0.679 and 0.558 ms unsplit (the prototype sums the partials in a separate pass). The int8 head was flat: all
  18 configs read between 0.188 and 0.192 ms.

The host's 400 W limit bounds the absolute rates. The verdict is ratios on one host, projected onto P142's in-situ
times, and the anchors put the harness's cuBLAS at 1.01-1.06x P142's in-situ time, not faster. The maintainer
re-derived the verdict byte for byte from the private receipt store and recomputed it without the reducer. The
launcher's receipt and ledger row are in that store. `SHA256SUMS` covers every file here.
