# `p143a3-5090-1`: Amendment 3, the binding served read of the split cap (FLIP)

One RTX 5090 (cc 12.0, 170 SMs, driver 615.71.09, a **450 W** power limit) on an AMD EPYC 9135 host (32 CPUs): Vast
instance redacted-0fb2ce02 on machine 151395, 2026-10-19T01:59:50Z to 02:29:33Z. Cost $0.479. Code: e4b `e8de1799` (the
registration's merge, #1788), grouped-nf4-gemm 0.47.1 (`38b30888`); torch 2.8.0+cu128, triton 3.4.0, transformers
5.17.0. Qwen/Qwen3-30B-A3B @ `ad44e777`.

**Verdict: FLIP.** Eight arms of the default server, ABBA twice (S0 = `E4B_ATTN_SPLIT_CAP=0`, S1 = `16`), five timed reps
each.
- **W1 pairs:** S1a/S0a 1.0442, S1b/S0b 1.0445, S1c/S0c 1.0448, S1d/S0d 1.0447. So **g1 = 1.044184** (unrounded,
  bar 1.03). One request: 255.6 -> 267.0 tok/s.
- **W16 pairs:** 0.9920 / 1.0052 / 1.0003 / 0.9992. So **g16 = 0.991994** (no-regression line 0.99). Geomean 0.9992:
  W16 runs the plan's 5 splits, so the cap does not bind there.
- **Noise:** W1 self-pairs 0.9991 / 0.9994 / 0.9995 / 0.9993, inside the NOISY band [0.985, 1.015]. Cross-block 0.9998
  / 1.0004.
- **Function:** every S0 arm decoded S0a's tokens and every S1 arm S1a's. W1 and W16 tokens were also identical S1
  against S0 (16 of 16 rows).
- **SANE:** bias 0.0, argmax 1.0, KL 0.0 (12 windows, T == 1). The pass KV ran at 4096 tokens, with B = 1 capped 32 ->
  16 on 73,152 calls.
- **Engagement:** as registered on every S1 arm, buckets 1 / 2 / 4 / 8 / 16 at 16 / 16 / 16 / 11 / 5 splits. No S0 arm
  carried a cap.

The host's 450 W limit bounds the absolute rates. The verdict is ratios within one host and one process family. The
maintainer re-derived the verdict byte for byte (#1788). The launcher's receipt and ledger row are in the receipt store
(`733f3941`). `SHA256SUMS` covers every file here.
Twelve logs carry carriage returns from the box's progress output. They stay only in that store, byte-exact,
and `STORE-ONLY.sha256` lists them.
