### Fixed

- `tests/test_hostprobe.py`: the shared-budget test now drives the probe's monotonic clock instead of sleeping, so a
  loaded host can no longer change its outcome. Test only; the probe is unchanged.
