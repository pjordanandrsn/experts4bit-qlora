### RD1 amendment 1 registered: a load-gated train anchor, after three anchor refusals on launch jitter (bench and tests only)

- **Why.** RD1's three RTX 5090 draws were refused by the train anchor, each on `launch.self_pair` alone (1.042 on machine
  145701 twice, 1.060 on 36544; FLOPs and H2D in band), for $0.068 in all.
  - TC1 amendment 33 measured host load from other tenants driving this kind of instability on these multi-tenant hosts.
  - adertha's anchor-exclusion class cannot name the machines, because it accepts only P41-layout receipts.
- **What.** `bench/moegen/rd1/rd1_run.sh`:
  - runs the anchor only at host load1 ≤ 5.0 (waiting up to 600 s), at most 3 attempts, with the last attempt standing and
    every attempt's files kept;
  - samples `/proc/loadavg` every 5 s, and writes the probe window's load summary into its receipt.
  - `rd_table.py` takes no decision from a probe whose median load1 exceeded the gate, or that has no load summary.
  - The bar, the correctness gate and the probe are unchanged.
- **Tests.** `tests/test_rd1_lane.py` pins the constants to the registration, and shows a loaded or unrecorded probe decides
  nothing.
