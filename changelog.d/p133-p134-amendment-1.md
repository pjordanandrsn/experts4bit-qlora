### P133 and P134 Amendment 1: the quality pass integrity reads the instrument's buckets

- `p133-4090-prove-1` ($0.17) ran every registered box step on the RTX 4090, and the reducer VOIDed it on one check. Each
  quality pass's `graph_status` was compared with the server's buckets (1–64 at `max_seqs` 64). It should have been
  compared with P110's instrument buckets, `(1, 2, 4, 8, 16)`.
- `p133_reduce.py` and `p134_reduce.py` now check `INSTRUMENT_BUCKETS`, which is P116's rule. Their self-tests add the
  proof's shape, and the CI tests pin the constant to `p110_box.BUCKETS`.
- Re-reduced, the proof's committed records read NOISY, not VOID, at its short proof lengths. No rule, bar, prediction,
  size or budget changes.
