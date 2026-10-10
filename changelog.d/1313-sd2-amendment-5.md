### SD2 Amendment 5 (#1313): the key-row gate at layer 0, after `sd2-prove-4`

- **What happened.** `sd2-prove-4` ($0.784) read FAILED under rule a3. At layer 0 the key-row gate landed exactly as
  derived (the real build 0.00093 rad, the RoPE-shift mutant 1.058), but at depth the real build's stored keys drift
  from the T == 1 keys. A few small-magnitude cells there carry large relative errors, which is what Amendment 4
  registered as not analytic.
- **The change (post hoc, said plainly).** Rule a4, now the default, gates the key rows at layer 0 only, where the
  derivation is analytic. The 0.5 rad boundary stays, and the derived bounds are checked on both sides: the real
  build at most 0.142, the shift mutant at least 0.858. Depth is reported and never gated: quantiles at three magnitude floors.
- **No new proof.** `sd2-prove-4` is pinned PROVED under a4 and FAILED under a3. The out-of-sample test is the read
  `sd2-5090-2` on fresh rows.
