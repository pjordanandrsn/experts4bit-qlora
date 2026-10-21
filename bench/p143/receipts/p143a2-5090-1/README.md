# `p143a2-5090-1`: Amendment 2, the first served read of the split cap (NO_FLIP)

One RTX 5090 (cc 12.0, 170 SMs, driver 595.71.05, 575 W) on an AMD EPYC 7702P host (128 CPUs): Vast instance 56638605
on machine 45379, 2026-10-19T00:02:43Z to 00:31:40Z. Cost $0.406. Code: e4b `7ed1695e` (the registration's merge,
#1778), grouped-nf4-gemm 0.47.1 (`38b30888`); torch 2.8.0+cu128, triton 3.4.0, transformers 5.17.0. Qwen/Qwen3-30B-A3B
@ `ad44e777`.

**Verdict: NO_FLIP.**
- **Kernel contract:** held at all 15 cells. The cap's error against the fp64 reference was 0.9976-1.0000x the
  plan's, and **bitwise identical** at every context <= 1024 (all B) and at B 4 / 2048.
- **Kernel timing:** the cap was faster at B 2 and 4 in every cell, by 0.09-0.20 ms per 48 launches. B = 1 was
  0.161 ms faster at 640 tokens.
- **Served (default server, ABBA):** W1 pairs S1a/S0a 1.0405 and S1b/S0b 1.0297, so **g1 = 1.0297 < 1.03**. g16 =
  1.0113. The W1 self-pairs were 0.9949 (S0) and **0.9845** (S1): the second capped arm ran 1.55 % slower than the first.
- **SANE** (12 windows, T == 1, the pass KV at the server's 4096 tokens): bias 0.0, argmax 1.0, KL 0.0. The cap capped
  B = 1 on 73,152 calls and changed no output at these contexts.

The maintainer re-derived the verdict byte for byte (#1778). Amendment 3 (#1788) followed, on stricter terms and binding.
The launcher's receipt and ledger row are in the receipt store (`74f8955f`). `SHA256SUMS` covers every file here.
Eight logs carry carriage returns from the box's progress output. They stay only in that store, byte-exact,
and `STORE-ONLY.sha256` lists them.

Re-derive: `python bench/p143/p143a2_reduce.py --dir bench/p143/receipts/p143a2-5090-1 --out {out} --e4b-sha 7ed1695eceab0052078d43c14aed8da06381d8f6` writes `bench/p143/receipts/p143a2-5090-1/verdict_p143a2.json`
