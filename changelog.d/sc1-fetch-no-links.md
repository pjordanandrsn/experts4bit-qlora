### Rule 8 for the SC1 harness and the G1/G3 smoke kits: fetches never import links; SC2j Amendment 2 (pre-data)

- `bench/sc1/sc1_drive.sh`'s `pull_box` and the G1/G3 smoke drivers fetch with `rsync -az --no-links`
  (bench/common rule 8: a fetch never imports links).
- SC2j Amendment 2: the launch checkout takes the fix. `bench/sc2/sc2_kit_hold.py package-hold` refuses unless the
  package under test (the `experts4bit_qlora/` tree, `pyproject.toml`, `MANIFEST.in`), the staged kit and `make_pin.sh`
  equal Amendment 1's merge's, and `sc1_drive.sh` differs only by that one token on `pull_box`'s rsync line. Its
  self-test removes or alters each; tests run it and the driver check on registered commits.
