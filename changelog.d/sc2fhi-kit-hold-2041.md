### SC2f Amendment 5, SC2h Amendment 3, SC2i Amendment 2: the kit is held at e4b#2041's merge (pre-data)

- The three lanes' kit moves from `1a25b209` to e4b#2041's merge, which carries the shared harness's `brotli>=1.2.0`
  pin and its single model-fetch retry. The stack under test stays `36cdd2d4`, set as `E4B_SHA` on the launch command.
- Everything else that merge changed in the flat staged kit is declared inert for boxes N, O and P: box Q's wiring in
  `sc1_run.sh`, seven new box Q files, and the staging lines in `sc1_drive.sh` and `make_pin.sh`.
- `bench/sc2/sc2_kit_hold.py` checks this before the create and fails closed. Its four gates: the declared delta exactly
  (DELTA), every pin matching its source at the hold (PIN), no N/O/P path sourcing or calling the new files (INERT), and
  the launch commit's kit identical to the hold (HOLD). Its self-test refuses each gate with its evidence removed;
  `tests/test_sc2_kit_hold.py` runs it on the registered commits.
