### SC2f Amendment 6, SC2h Amendment 4, SC2i Amendment 3: the HOLD gate's declared exceptions (pre-data)

- `bench/sc2/sc2_kit_hold.py`'s HOLD gate keeps its byte identity with `7b8f1f69`, with exactly two declared
  exceptions: rule 8's one `--no-links` on `pull_box`'s rsync line, and box Q's own `sc2j_<name>` files changed or
  added (staged by name in the drivers, with INERT re-run at the launch commit). Anything else, a removed entry, an
  unreferenced non-sc2j file, or a dirty launch worktree refuses.
- The self-test has 11 new cases; a test reads the registered `main` commit `ac7e6ed1` as OK.
