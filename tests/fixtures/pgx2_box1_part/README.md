# `pgx2-4090-r2-2-1`'s part record, and the controller's re-derivation of it (PGX2 Amendment 2)

- `part_verdict.json`: the part record that box 1 (`pgx2-4090-r2-2-1`, 2026-11-07 23:50Z to 2026-11-08 06:36Z) wrote on
  the box, byte-identical to the fetched file.
- `rederived_seed0.json`: the controller's re-derivation of the same record. This is `pgx_reduce.reduce([read/],
  registration, draws=(1, 2), part=True)` on the box's fetched `read/` and `registration.json`, at the launch commit's
  `pgx_reduce.py` (sha256 `0b0adc4f…`, unchanged), run with `PYTHONHASHSEED=0` and round-tripped through JSON as
  `pgx_box_end.py` does.

The two hold the same six `token_invariance` findings in different orders. They agree in every other field. The box's
`read/` is about 32 MB and is not vendored, so `tests/test_pgx2_findings_order.py` pins both files by sha256.
