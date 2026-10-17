### P133 and P134 Amendment 1: the quality pass integrity reads the instrument's buckets

- `p133-4090-prove-1` ($0.17) ran every registered box step on the RTX 4090. The reducer VOIDed it: the quality pass
  integrity read the server's buckets from the record (1–64 at `max_seqs` 64), not P110's instrument buckets
  `(1, 2, 4, 8, 16)`.
- `p133_reduce.py` and `p134_reduce.py` now read `INSTRUMENT_BUCKETS` from `p110_box.py`'s own `BUCKETS`, as P116's rule
  did. The CI tests rerun each pre-amendment reducer on the proof's shape: it VOIDs, the amended one passes, and a pass
  missing bucket 8 still VOIDs.
- No rule, bar, prediction, size or budget changes. The proof's committed records are re-reduced after this merges.
