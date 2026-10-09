# `fam-gptoss-1`: gpt-oss-20b, the B=1 fused stack at T == 1, one knob per arm, and the anchor (a reading)

One RTX 5090 (cc 12.0, driver 610.57.04) on an AMD engineering-sample host (100-000000897-03, 64 CPUs): Vast instance
54930717, launched 2026-10-08T23:33:11Z, torn down 2026-10-09T00:41:58Z. Cost $1.007. Code: e4b `9766fb4c` (the #1403
merge), grouped-nf4-gemm 0.43.0 (`6ee2e10`). gpt-oss-20b at 128 teacher-forced positions: OFF, ON_glue, ON_r2, ON_epi
and ON_auto on the default path; the anchor's OFF and ON_auto on SC2g's.

**Verdict: ON_glue, ON_r2, ON_epi and ON_auto all FAIL** under the registered rule. gpt-oss's knobs stay off by
default.
- **Every failing entry fails only the absolute backstop**, never a gate relative to the family's own floor:
  - ON_glue fails 3 of 12 entries on argmax agreement < 0.90 (0.8848 to 0.8997);
  - ON_r2 fails 1 of 12: wikitext, shape 12, set B, |bias| 0.0201 > 0.020;
  - ON_epi and ON_auto fail 6 of 12 each on agreement < 0.90 (0.8848 to 0.8984).
- **The family's own floor sits below the backstop.** A_f is 0.8841 / 0.8737 / 0.8841 / 0.8743 at wikitext 1 / 12
  and c4val1 1 / 12. The chunk, half and `split1` draws span 0.874 to 0.934, and about half read under 0.90.
- **The instrument held.** `mut090` and `mutant_scale` failed all four cells. On the ladder (set A), `mut095` passed 2
  of 4 and `mut098` 1 of 4. `rep` was bit-identical and `split1` was not.
- **Census:** glue `0 / 49 / [0, 0] / 0`, r2 `0 / 0 / [24, 0] / 0`, epi `0 / 0 / [0, 0] / 24`, auto
  `0 / 49 / [24, 0] / 24`, exactly; OFF zero. Every engagement check held, and the premise passed (36 tests).
- **The anchor** (SC2g's path, an MXFP4 store on 24 layers; wikitext, shape 12, set A): its floor A_f is 0.9036 and
  ON_auto agrees 0.9238 with bias +0.0022. Phase C's 0.924 lies within that floor, so Phase C's miss came from the
  gate, not the knob.
- **Process walls:** OFF 2025 s, each ON 321 to 338 s, the anchor's two 70 s and 8 s. Peak 15.77 GB on the default
  path and 25.22 GB on SC2g.

The maintainer re-derived the verdict from the store (`12484a95`) byte for byte. The launcher's receipt and ledger row
are in the receipt store. `SHA256SUMS` covers every file here.
