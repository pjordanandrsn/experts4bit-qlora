### P117 registered (#846): does decoding 32 or 64 rows in one graph (`E4B_PAGED_BUCKETS=auto`) cost quality against decode in pieces of at most 16 rows? (bench and tests only)

- **Why.** SC2e read 64 slots with buckets up to 64 holding the SLO to 12 req/s against 8 on the default list and 4 at
  16 slots (#1333). The ruling on #1320: `E4B_PAGED_BUCKETS=auto` becomes a default only after P110's teacher-forced
  read at buckets 32 and 64. Above 16 rows a decode step takes the prefill-side routes (`Int4Linear`'s cached bf16 weight
  with cuBLAS; above 256 routed rows K19's chained tile table).
- **The box** (`bench/p117/p117_box.py`): SC2e's int4 stack built eager with one slot on one RTX 5090, then ten
  teacher-forced paged passes over 64 wikitext windows decoded together: R (16-row pieces), rep, the floor (half, chunk,
  rev), the subjects W32, W64 and W64pad (48 windows padded to 64), the scale mutant, and G64, a captured bucket-64
  replay whose emitted tokens must equal W64's (FUNCTION).
- **The rule** (`bench/p117/p117_reduce.py`, 22 self-test cases): P110's bar unchanged; AT_PARITY iff W32, W64 and
  W64pad all pass. AT_PARITY licenses `E4B_PAGED_BUCKETS=auto` as a default in a separate PR; COST keeps it opt-in.
- **Proof** on Granite-3.1-3B-A800M (40 windows, 32 positions, guard 0.75 h); reading guard 1.5 h; lane ceiling $3.00.
