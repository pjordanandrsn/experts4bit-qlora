### SD2 Amendment 2 (#1313): V0's addressing gate is a logit gate, after `sd2-prove-2`

- **What happened.** `sd2-prove-2` ($0.66) ran in full and read FAILED with one item, `GATE_TOO_WEAK:c`: the RoPE-shift
  mutant kept argmax agreement above 0.90. Every build item held.
- **The new gate.** `bench/sd2/sd2_reduce.py --rule a2` (now the default) makes the logit gate decide: each verify row's
  mean |Δ log p| of the oracle's token must be at most 0.25 nats. That bound is calibrated on `sd2-prove-2`: the real
  build peaked at 0.037 and the mutants at 1.18 or more.
- **The old rule stays reproducible.** `--rule a1` re-derives the registered verdict unchanged.
- **A fresh proof.** `sd2-prove-3`, at the same target and read by rule `a2`, must be PROVED before the builds merge.
- **The read also checks the gate out of sample.** Its V0 re-runs the three mutants before any timing, and a mutant
  inside the bound makes the read VOID.
