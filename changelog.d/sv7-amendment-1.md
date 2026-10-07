### SV7 amendment 1 (#1294): the serve measure's driver sampler takes a sole unmatched compute process (bench, prereg and tests only)

- `sv7-4090-1` (OK, $0.611) read V3, V4 and V5 HELD but V1 and V2 NO_READING. Its host's container listed
  host-namespace PIDs in `nvidia-smi --query-compute-apps`, so `bench/sv4/sv4_measure.py` matched no row and took zero
  driver samples.
- `sv4_measure.py`'s new `driver_sample` keeps the PID match. When no row matches and exactly one compute process is
  listed, it takes that one, and the receipt records how each sample matched (`driver_match`).
- `tests/test_sv4_measure_sampler.py` pins the cases.
- The next SV7 box launches from this amendment's merge.
