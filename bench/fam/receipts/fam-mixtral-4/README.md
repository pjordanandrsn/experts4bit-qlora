# `fam-mixtral-4`: Mixtral-8x7B-Instruct, the B=1 fused stack at T == 1 (the reading)

One RTX 5090 (cc 12.0, driver 595.71.05) on an AMD EPYC 7C13 host (256 CPUs): Vast instance 55108544, launched
2026-10-09T22:54:13Z, torn down 2026-10-10T01:04:31Z. Cost $2.59. Code: e4b `68845bfc` (Amendment 9's merge, #1487),
grouped-nf4-gemm 0.43.0 (`6ee2e10`).

**Verdict: PASS on ON_glue, ON_r2, ON_epi and ON_auto, at a resolution of × 0.90 (Amendment 8).** `mut090` failed the
gate in every gated cell, so the claim is the null read of that size: no effect on Mixtral as large as a × 0.90 change
of the decode softmax scale.
- **The instrument held.** The census was exact on all five processes (OFF `0 / 0 / [0, 0] / 0`, ON_auto
  `0 / 65 / [32, 32] / 32`). `mut080` and `mutant_scale` failed every cell. `rep` was bit-identical, so the floor is
  chunk, half and split1.
- **Floors and margins.** A_f 0.9661–0.9694 and B_f 0.0039–0.0062. The closest ON entries: ON_epi at (wikitext, 12),
  |bias| 0.0094 against 0.0162; ON_auto at (c4val1, 1), agreement 0.9661 against 0.9611. KL at most 0.0062.
- **The ladder** (set A): × 0.95 passes 3 of 4 cells (it fails c4val1 at shape 1); × 0.98 passes all four.
- **Walls:** about 830 s from launch to the bake's end; OFF 4314 s; each ON 555–603 s. Peak 28.19 GiB.

The maintainer re-derived the verdict from the store (`9066d13f`) with main's reducer: identical, except the last unit
in the last place of two floor KL values (macOS against Linux summation). The launcher's receipt and ledger row are in
the receipt store. `SHA256SUMS` covers every file here.
