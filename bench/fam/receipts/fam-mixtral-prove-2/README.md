# `fam-mixtral-prove-2`: Mixtral's own proof under Amendment 7 (not a reading)

One RTX 5090 (cc 12.0, driver 595.91.07) on an AMD EPYC 7763 host (256 CPUs): Vast instance 55054214, launched
2026-10-09T16:32:50Z, torn down 17:31:37Z. Cost $1.775. Code: e4b `da17bbc5` (Amendment 7's merge, #1468),
grouped-nf4-gemm 0.43.0 (`6ee2e10`). The box end to end at 32 positions.

**Verdict: PROVED (rc 0).**
- The census was exact. OFF took 1513 s, ON_epi 261 s and ON_auto 234 s. Peak 28.12 GiB.
- At proof scale the reducer read UNRESOLVED: `mut090` passed the gate on c4val1 at both shapes. That is a property of
  the instrument on Mixtral, the mutant's sensitivity, and it triggered Amendment 8 (× 0.80 added behind × 0.90). At
  reading scale × 0.90 resolved (`fam-mixtral-4`).

The maintainer re-derived the proof from the store (`ff615c5c`), byte-identical. The launcher's receipt and ledger row
are in the receipt store. `SHA256SUMS` covers every file here.
