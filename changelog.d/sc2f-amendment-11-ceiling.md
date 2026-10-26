### SC2f amendment 11: the price ceiling is $1.25 an hour for SC2f, SC2h and SC2i

- `bench/sc2/sc2_qualify.py`: `CEILING_USD_H` moves from 1.05 to 1.25, one ceiling for the three lanes. Pre-data; the power
  class, the clock floor, the recorded-clock rule, the attempts and the deadline are unchanged.
- Over the 228 committed screen reads, the new ceiling admits on 85 reads against 59
  (`bench/sc2/sc2f-am11/ceiling_replay.json`).
- Worst cases restated per lane: SC2f proof $1.38, reading $5.14, lane $14.42; SC2h $1.35 / $5.73 / $14.16; SC2i $1.35 /
  $6.36 / $15.41. Every run stays under $15.
- The launch orders 340 GB of disk and offers must list at least 340 GB: SC1's 320 GB free-space floor plus 20 GB. Ordering
  exactly the floor refused `sc2f-prove-12` on a host that enforced its quota. `bench/sc2/sc2f_launch.json` holds the ordered
  disk and price, and a test fails CI if either falls below SC1's floor plus 20 GB or drifts from `sc2_qualify.py`.
