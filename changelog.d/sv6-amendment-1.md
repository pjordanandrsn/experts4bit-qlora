### SV6 amendment 1 (#1267): a registered driver floor, exit 18, before the lane's next box (bench, prereg and tests only)

- `sv6-4090-1` (NOT_RUN, pre-flight HF bandwidth, $0.023) and `sv6-4090-2` (tripwire rc 9, $0.029) both landed on Vast
  machine 29956. It runs driver 535 / `cuda_max_good` 12.2, where torch 2.8+cu128 cannot initialise CUDA (error 804).
  Neither reached an arm, so there is no data.
- `bench/sv6/sv6_run.sh` now refuses a driver older than 570 before installing anything and exits 18, the registered
  host-floor code, so a later launch can exclude the machine. `tests/test_sv6.py` pins it.
- Readings, consequences, the reducer and its pinned numbers are unchanged.
