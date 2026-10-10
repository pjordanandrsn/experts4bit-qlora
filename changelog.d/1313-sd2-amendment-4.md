### SD2 Amendment 4 (#1313): a key-row position gate in V0, after `sd2-5090-1`

- **What happened.** `sd2-5090-1` ($0.566) read VOID: on rows outside calibration, the RoPE-shift mutant stayed inside
  the 0.25-nat logit bound (0.161 and 0.064), so V0 failed closed before any timing. Every build item held.
- **The finding.** A logit-level gate on short verify rows cannot see a uniform RoPE position shift out of sample,
  because the verify rows keep their relative positions among themselves.
- **The added gate** (rule a3, now the default). V0 compares the keys the verify step writes with the keys the T == 1
  path writes at the same positions, as a rotation on RoPE pair 0, which turns 1 rad per position. The boundary is
  0.5 rad, the midpoint, derived and not fitted: one e4m3 ulp bounds the real build at 0.142 rad at layer 0, and a
  one-position error reads at least 0.858. A CPU test checks the derivation with real e4m3. The logit gate is
  unchanged, and rules a1 and a2 still re-derive every earlier receipt.
- **Next:** `sd2-prove-4` on the calibration rows, then the read reruns as `sd2-5090-2`.
