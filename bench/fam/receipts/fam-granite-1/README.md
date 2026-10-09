# `fam-granite-1`: Granite-3.1-3b-a800m, the B=1 fused stack at T == 1 (a reading)

One RTX 5090 (cc 12.0, driver 595.71.05) on an AMD EPYC 7C13 host (256 CPUs): Vast instance 54930667, launched
2026-10-08T23:32:43Z, lane done 2026-10-09T00:33:39Z, torn down 00:33:39Z. Cost $0.828. Code: e4b `9766fb4c` (the
#1403 merge), grouped-nf4-gemm 0.43.0 (`6ee2e10`). Granite at 128 teacher-forced positions; OFF and ON_auto.

**Verdict: ON_auto FAIL** under the registered rule. Granite's knobs stay off by default.
- **The miss.** One gated entry of 12: c4val1, shape 1, set C. Its argmax agreement was 0.95117 (1461 of 1536)
  against A_f − 0.005 = 0.95195, where A_f = 0.95703 is chunk's draw on set B. In that entry |bias| was 0.0026
  (gate 0.0144), spread 0.0084 (gate 0.0248) and KL 0.0081, inside the floor draws' 0.0071–0.0093. Every other entry
  passed.
- **The instrument held.** `mut090` and `mutant_scale` failed all four cells. On the ladder, `mut095` passed 3 of 4 and
  `mut098` 4 of 4. `rep` was bit-identical and `split1` was not.
- **Census:** ON_auto `0 / 65 / [32, 32] / 32` exactly; OFF zero. Every engagement check held, and the premise
  passed (36 tests).
- **Process walls:** OFF 2891 s, ON_auto 483 s. Peak 4.89 GB.

The maintainer re-derived the verdict from the store (`cceafa43`) byte for byte, and ruled no re-read under this
registration. The launcher's receipt and ledger row are in the receipt store. `SHA256SUMS` covers every file here.
