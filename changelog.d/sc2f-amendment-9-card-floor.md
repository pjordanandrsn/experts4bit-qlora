### SC2f amendment 9: the card floor follows TC1 amendment 83; the in-run judge rules bits only

- `bench/sc2/sc2_qualify_drive.sh`: the pre-install burn is `card_clock.py burn --floor 2700 --mode clock`, which is
  amendment 83's clock rule named explicitly, and the only clock screen.
- `bench/sc2/sc2_qualify.py judge-run`: the in-run sampler is judged by `card_clock.passes(..., mode="clock")` at
  `SERVING_JUDGE_FLOOR_MHZ = 0`, card_clock's own code. A slowdown bit on any sample, or Reliability on at least 95 % of
  the busy samples, refuses. SwPowerCap is recorded, and the clock is recorded but never ruled.
- The reason: the private receipt store's 51 busy serving snapshots read a busy median of 2,415 MHz (under the 5090
  floor) and Reliability on 4 of them. Pre-data: no SC2f proof has run. SC2h and SC2i take it by reference.
