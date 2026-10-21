### Changed

- P222 Amendment 1 (#1313), before any reading: TM (the tree at M's plan, a reported-only arm) is dropped after the
  paid shakedown `p222-shake-1` found a selected plan whose tree variant cannot compile (210,944 B of shared memory
  against the 101,376 B per-block limit). Block B runs five arms in Williams' design for an odd count (10 orders a
  square), on 120 reps (12 squares). A compile, launch or capture failure in an arm is recorded in the bench record and
  the reducer VOIDs with the reason, never a crash. The GO rule, the seeds, the selection, the served-arm rule and the
  budget are unchanged.
