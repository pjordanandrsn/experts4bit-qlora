### bench/gpu-contract: a PR's own e4b test list for its GPU smoke

- `GC_E4B_TESTS` replaces the served-route premise list with space-separated `tests/test_<name>.py` paths. The controller
  and the box each refuse (78) any entry not of exactly that form. The default list is unchanged.
- `GC_NEED_E4B_S` now reaches the box. The README documented it, but `gc_drive.sh` did not pass it through, so setting it on
  the controller had no effect.
