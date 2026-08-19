# Spin-window measurement: the cold-entry cost recovered — B=8 wall 28.1 -> 22.1 ms

TR PRO 7965WX + 5090, capped harness, full stack (stealing + thin +
fx55), three spin budgets. Box destroyed; verified.

| spin budget | B=8 dram / ratio | B=16 dram / ratio |
|---|---|---|
| default (~100 us) | 28.1 ms / 0.563 | 49.0 ms / 0.327 |
| 1000 us | 23.3 / 0.682 | 33.5 / 0.482 |
| 3000 us | **22.1 / 0.727** | 34.2 / 0.484 |

- The intracall diagnosis's ~2.5x first-call-after-idle cost was real
  and recoverable: keeping workers warm across a forward's calls takes
  21% off the B=8 wall and 32% off B=16. 1 ms captures nearly all of
  it (the budget only needs to span a layer's GPU phase); 3 ms adds a
  sliver. Recommended serving setting: spin_us ~1500.
- **Bar arithmetic**: this host's fast GPU side (15.9-16.5 ms) sets
  0.80 at dram <= 19.8 ms — 2.3 ms short here. The reference 7975WX
  (gpu 17.9 ms) sets it at <= 22.4 ms, which spin3000's 22.1 CROSSES.
  The reference-host confirmation run is pending on offer availability;
  until it lands, G8's honest status is 0.727 measured, bar-crossing
  projected on the reference class.
- The balance ratio's GPU-side sensitivity (documented in the final-
  round receipts) is now the dominant term in cross-host variance:
  hosts with faster GPU paths need proportionally faster DRAM tiers
  for the same ratio.

Receipts: `g8_spin{0,1000,3000}.json`, `calib.json`, `rows_curve.json`.
