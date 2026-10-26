### G3 GPU check: the closed kit's pin test reads its registered bytes, not the live tree

- `tests/test_g3_gpu_check.py` now checks `bench/kv-pool-g/g3-gpu-check/staged.sha256` against each staged source's bytes
  at the kit's merge (`47fac3ac`, the kit g3-gpu-check-4 ran), never against the live tree.
- The lane is closed (g3-gpu-check-4 PASS), so a later change to a shared staged file, such as
  `bench/common/card_clock.py` in TC1 amendment 83, no longer turns its test red.
- A tampered pin still fails, and a relaunch would re-pin by its own PR.
