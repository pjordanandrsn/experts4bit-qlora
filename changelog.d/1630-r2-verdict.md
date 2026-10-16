### Added

- R2-Mixtral's mechanical milestone verdict (`bench/fam235/r2_verdict.py`, #1630). It runs Q235's verdict unchanged, by
  citation, with Amendment 1's depth-local band in place of Q235's single 0.775: `band_j = 2 × u_cal(round(32j / n))`, the same
  threshold for the plants (VOID below it) and the unplanted box (FAIL above it). It reports each layer's margin and names the
  shallow band (j = 5..14 on 56 layers) as where a noise FAIL is expected.
- Its self-test replays Amendment 1's numbers and reproduces the recorded VALIDATED. Until an amendment registers the box's class
  and estimates, the tool VOIDs every run. No box, rental or claim.
