# `fam-qw36-2`: Qwen3.6-35B-A3B, the B=1 fused stack at T == 1 (a reading; the registration's one rerun)

One RTX 5090 (cc 12.0, driver 595.71.05) on an AMD EPYC 7C13 host (256 CPUs): Vast instance 54944654, launched
2026-10-09T01:26:37Z, torn down 03:34:29Z. Cost $2.345. Code: e4b `98d59921` (Amendment 2's merge, #1420),
grouped-nf4-gemm 0.43.0 (`6ee2e10`). Qwen3.6 at 128 teacher-forced positions, OFF and ON_auto, the largest shape first.

**Verdict: ON_auto PASS** (`verdict-amendment-3.json`, the reducer at Amendment 3's merge `2ead4627`).
- **The box's own reduction VOIDed** (`verdict.json`, kept as written) on one engagement count, 613 times: 127 decode-shaped
  forwards against the 128 registered. The registration expected the hybrid's linear-state warm-up on every padded pass, but
  e4b runs it once per process, and each process here shows exactly one, on its first pass. Amendment 3 corrected the
  count. Per the maintainer's ruling, these records were re-reduced with no new rental, and no gate statistic was
  computed before that amendment merged.
- **Every gated entry passes with a wide margin.** The worst-set |bias| is at most 0.0012 against gates of 0.0128 to 0.0166.
  Agreement is at least 0.9876 against gates of 0.9527 to 0.9611. KL is at most 0.0010.
- **The instrument held.** The census read ON_auto `0 / 0 / [0, 0] / 40` exactly, and OFF zero. Every engagement check
  held under Amendment 3. `mut090` and `mutant_scale` failed all four cells. On the ladder (set A), `mut095` and `mut098`
  passed 4 of 4. `rep` was bit-identical and `split1` was not.
- **Process walls:** OFF 5966 s, ON_auto 922 s. Peak 24.38 GB.

The launcher's receipt and ledger row are in the receipt store (`555ba11f`). `SHA256SUMS` covers every file here.
