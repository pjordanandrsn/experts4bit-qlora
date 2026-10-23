### SC2f Amendment 6, SC2h Amendment 4, SC2i Amendment 3: the HOLD gate's declared exceptions (pre-data)

- `bench/sc2/sc2_kit_hold.py`'s HOLD gate keeps its byte identity with `7b8f1f69`, with exactly two declared
  exceptions: rule 8's one `--no-links` on `pull_box`'s rsync line, and box Q's own `sc2j_<name>` files changed or
  added (staged by name in the drivers, with INERT re-run at the launch commit). Anything else, a removed entry, an
  unreferenced non-sc2j file, or a staging checkout that is dirty or not at the launch commit refuses.
- The self-test covers each refusal, including names only a prefix or substring match would pass; a test reads the registered `main` commit `ac7e6ed1` as OK.
