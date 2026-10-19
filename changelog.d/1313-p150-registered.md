### P150 registered (#1313): D2's quality ruling, the int8 lm_head on held-out windows

P148 read an int8 lm_head GO in a microbench, a premise and never a gain. Before any kernel reaches the served path,
`bench/p150/` rules whether rounding only the lm_head to its per-row int8 grid (P148's lever, bit for bit) keeps P125's
quality gate. The read runs the W8A16 split premise's instrument on P125's subject, but on **held-out** windows: the
same corpus's validation split at P125's counts (t1 196, k16 123), with the windows' sha256s registered before data.
The rule is fixed first: GO iff argmax ≥ 0.95 and d + 1.645 SE ≤ P125's bound at both gates, else PARK. A GO licenses
quality only; the speed is a served read of its own. Nothing changes by default.
