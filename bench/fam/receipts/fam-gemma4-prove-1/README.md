# `fam-gemma4-prove-1`: Gemma-4-26B-A4B's own proof (not a reading); the family closed here

One RTX 5090 (cc 12.0, driver 595.71.05) on an AMD EPYC 7C13 host (256 CPUs): Vast instance 55124966, launched
2026-10-10T01:08:18Z, torn down 01:55:50Z. Cost $1.13. Code: e4b `68845bfc` (Amendment 9's merge, #1487),
grouped-nf4-gemm 0.43.0 (`6ee2e10`). The box end to end at 32 positions: OFF, ON_epi and ON_auto.

**The box's verdict, `verdict.json`: VOID.** Every process ran to rc 0, and the census was exact (ON_auto
`0 / 271 / [0, 0] / 30`). At T == 12, ON_auto called `rmsnorm_rows` 216 times per decode step against Amendment 6's
271. The glue fold sends Gemma-4's per-head q and k norms to their own torch forward above 64 rows. That is a re-route,
not a skip, and Amendment 10 (#1529) counts it per shape.

**Amendment 10's re-reduction, `verdict-amendment-10.json`: PROVED, with ON_epi and ON_auto FAIL at proof scale.**
- It was reduced by main's `fam_reduce.py` at `a15c8d22`. The maintainer re-derived it byte-identical.
- Every ON entry passes the relative gate: its worst agreement, 0.81–0.86, sits inside Gemma-4's own neutral floor.
  Every ON entry fails the absolute backstop (agreement ≥ 0.90).
- The family's floor is A_f 0.737–0.766 and B_f up to 0.51 nats. On the wikitext cells even ×0.90 passes the relative
  gate and fails only the backstop.
- So nothing here shows the knobs harm Gemma-4. This instrument cannot license them on Gemma-4's NF4 stack, which is
  1.077 nats of KL from bf16 (`e4b.serve.p44.gemma4.nf4-vs-bf16`). The maintainer closed the family at this proof,
  with no reading.
- **Walls:** about 680 s from launch to OFF; OFF 1535 s; ON_epi 254 s; ON_auto 240 s. Peak 26.18 GiB.

The launcher's receipt and ledger row are in the receipt store (`37e7f190`). `SHA256SUMS` covers every file here.
