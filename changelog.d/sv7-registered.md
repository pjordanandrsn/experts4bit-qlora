### SV7 registered (#1294): the planner's re-matched 24 GB plan for Qwen3-30B-A3B, solver tiers at VRAM 13.143 GiB and 8 × 8192, on an RTX 4090 at the longest prompts (bench and prereg only)

- `bench/sv7/` follows SV6's layout:
  - `SV7-PREREG.md`;
  - `sv7_run.sh`, the box side, which reuses `bench/sv4/sv4_measure.py`. It keeps SV6's host-only exit codes, its driver
    floor (18) and its estimate-pin tripwire.
  - `sv7_reduce.py`, with a 32-case self-test, also run by `tests/test_sv7.py`.
- **The plan:** loggetta `10cbf9f` now borrows a same-shape reserve (SV4 `t4_plan8`, 4.08%) instead of a 4 × 4096 NVMe
  arm's 6.7%. Its tier plan grows from SV6's VRAM 12.631 GiB to 13.143 GiB (5,316 rows), 22.344 GiB planned. The
  estimate is 20.987 GiB at `c07ea7f6`.
- **Readings:**
  - V1: the plan fits at 8 × 8,000-token prompts;
  - V2: the driver peak against the plan;
  - V3: the measured reserved-minus-allocated against the borrowed 0.856 GiB reserve;
  - V4: the estimate ±5% at long prompts;
  - V5: the server's tier rows equal the estimate's.
- Each reading's consequence is registered before the data. V3 decides whether loggetta keeps borrowing a serve slack
  across tier budgets.
- Spend is capped at $5 by #1294, within the owner's $50 approval.
