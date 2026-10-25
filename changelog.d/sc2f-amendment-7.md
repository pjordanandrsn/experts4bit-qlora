### SC2f Amendment 7, SC2h Amendment 5, SC2i Amendment 4 (pre-data): qualify-at-proof, the release pair moves to e4b 0.58.1 and grouped-nf4-gemm 0.51.0

- **Qualify-at-proof replaces the receipt bar** for the three lanes. The bar's machines were never listed through four
  screen windows. The new gate qualifies the card actually rented, and history only orders the list.
  - `bench/sc2/sc2_qualify_drive.sh`, new: the launcher's command for every box N, O and P.
    - Before any install: an enforced 575 W power limit, compared as a float, and a `bench/common/card_clock.py` burn
      at the 2700 MHz floor.
    - Over every run: a controller-side `nvidia-smi` sampler, judged by `card_clock.py`'s rule.
    - After a clean proof: its own host records.

    A record read and failed is a host fault (rc 17 or 18). A record missing or unreadable is a harness fault (rc
    25) that never excludes the machine.
  - `bench/sc2/sc2_qualify.py`, new: the pin admission as code.
    - An open walk: continuity, then 52780 and 145701, then any machine listing 575 W by prior clean runs, then
      listed-power-unknown machines.
    - The exclusions, and an all-in offer in 3 consecutive reads.
    - Every create counts as one of 3 attempts, and no host is revisited.
  - A 72 h screen deadline from the amendment's merge or the G3 GPU check's PASS, whichever is later; at the deadline
    the three lanes park. In the basis window, only 52780 ever passed the stability rule (6 of 40 reads).
- **The release pair.** The stack under test is `v0.58.1` (`605d21e6`). `bench/sc1/sc1_run.sh` moves boxes N, O and P
  to grouped-nf4-gemm `v0.51.0` (`30adb1bf`); box Q keeps SC2j's `v0.50.0`.
  - `bench/sc2/sc2_kit_hold.py`: STACK names `605d21e6`, and HOLD exception 3 requires exactly the two gnf4 lines.
  - `bench/sc2/sc2_defaults.py dump --set` records an arm's resolved paths. `bench/sc2/sc2f-am7/` holds both arms'
    dumps at both stacks, and the basis window's reconstructed screen reads. P130's two knobs did not move, and the on/off contrast is identical in both trees.
- `tests/test_sc2f_am7.py`, new.
