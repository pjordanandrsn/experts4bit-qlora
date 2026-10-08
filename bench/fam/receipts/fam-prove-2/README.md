# `fam-prove-2`: the proving rental under Amendment 1 (not a reading)

One RTX 5090 (cc 12.0, driver 595.91.07) on an Intel(R) Core(TM) Ultra 9 285K host (24 CPUs): Vast instance 54927731, launched
2026-10-08T23:11:50Z, lane done 23:29:25Z, torn down 23:29:26Z. Cost $0.214. Code: e4b `9766fb4c` (the #1403
merge, carrying #1395 and #1398), grouped-nf4-gemm 0.43.0 (`6ee2e10`). Granite-3.1-3b-a800m at 32 teacher-forced
positions, every process kind, the anchor included.

**Verdict: PROVED (rc 0).** The reducer reads PASS for ON_epi and ON_auto at proof scale; that is not a reading.
- **The 5090 regression check of #1398 holds.** The router epilogue licensed 32 of Granite's 32 routers in ON_epi,
  ON_auto and the anchor's ON_auto. `fam-prove-1` read 27 on the old probe.
- **Census:** ON_auto `0 / 65 / [32, 32] / 32` exactly; OFF zero.
- **Every integrity check held,** and the premise passed (36 tests).
- **The anchor's two SC2g processes ran:** Granite's `int4_b32` store, 6.45 GiB peak. The anchor's own floor is in
  `verdict.json`.
- **Process walls:** OFF 542 s, ON_epi 94 s, ON_auto 72 s. This host was faster than `fam-prove-1`'s; Amendment 1's
  guards come from the slower one.

The launcher's receipt and ledger row are in the receipt store. `SHA256SUMS` covers every file here.
