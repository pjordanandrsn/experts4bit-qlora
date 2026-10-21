### Changed

- P222 Amendment 2 (#1313), before any reading: the paid shakedown `p222-shake-2` ran end to end with five arms
  bitwise identical, but read VOID on the self-pair band (T/T2 1.0217 on a 72 W L4 over 10 reps). In a shakedown the
  band is now reported, never ruling; a reading still VOIDs outside 0.99-1.01. Each arm's compiled metadata (registers,
  spills, shared memory) is now recorded for all five arms: block A's reference compile is kept, the first compile of
  the same plan stands in, and each row names its source pass. A new shakedown id, `p222-shake-3`, runs on the same
  terms. From P160's public 5090 receipt, P(VOID | no real difference) under the registered band at 120 reps is under 1
  in 20,000. The GO rule, the band for a reading, the seeds, the selection and the budget are unchanged. The shake-2
  record is a sha256-pinned test fixture.
