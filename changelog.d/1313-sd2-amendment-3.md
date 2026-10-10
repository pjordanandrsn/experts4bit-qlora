### SD2 Amendment 3 (#1313): the read `sd2-5090-N` -- its target, its harness and its budget

- **The target is merged main,** re-proven on the card: a package-diff audit against the proven `539a2d26`
  (`bench/sd2/sd2_audit.py` over `audit_read.tsv`, which refuses an unlisted or stale file), the target's GPU tests,
  and V0 on rows outside the gate's calibration, which fails closed.
- **The runner's read mode** (`SD2_MODE=read`): stage V, the gate (VOID / E_ONLY / ALL), stage E's six arms in the
  registered palindrome, stage Q unless the stop rule applies, and the registered verdict. The driver takes the mode
  from the run id (`sd2-prove-N` or `sd2-5090-N`).
- **The budget,** from receipts: Q at one window a pass is about 100 min (12.6 s per eager window-pass), so the read's
  guard is 3.5 h and its ceiling $3.66. Q keeps its registered 48 windows per text.
