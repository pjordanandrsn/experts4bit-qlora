### SMT fuse Amendment 1: the second hardware thread, pinned

`bench/smt-fuse/smt_sib_box.py` reads what `smt-fuse-2` could not: each rep runs in a fresh process pinned before torch
is imported, to K physical cores (PHYS), the same K cores with both hardware threads (SIB), or 2K physical cores (WIDE),
with K = min(16, floor(quota / 2.5)). The reading is SIB/PHYS on the fuse (15 timed blocks per process, their median), the
whole block reported, over 8 interleaved triples; the registration states the resolvable effect (a fuse SIB/PHYS of about 0.80) and a tested time fit (guard 2.5 h);
triples with a busy co-tenant on the arm's sibling threads, or CFS throttling, are reported but not read. Ids
`smt-sib-1..3`. The drive no longer fetches the box's venv.
