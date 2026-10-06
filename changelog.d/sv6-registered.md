### SV6 registered (#1267): the planner's revised 24 GB plan for Qwen3-30B-A3B, solver tiers at 8 × 8192, on an RTX 4090 at the longest prompts (bench and prereg only)

- `bench/sv6/`:
  - `SV6-PREREG.md`;
  - `sv6_run.sh`, the box side, which reuses `bench/sv4/sv4_measure.py`. Host-only exit codes: 13 disk, 18 RAM. A
    tripwire refuses the box if the registered estimate total moved.
  - `sv6_reduce.py`, with a 30-case self-test, also run by `tests/test_sv6.py`.
- **The plan:** loggetta `dd4783f`'s solver tiers, VRAM 12.631 / DRAM 15.187 GiB, eager decode, 22.343 GiB planned. The
  estimate is 20.476 GiB at `50e24f3c`, including #1247's bulk flush (526 MiB).
- **Arms:** `b6_short` (1,024-token prompts, the anchor) and `b6_long` (8,000).
- **Readings:**
  - Y1: the plan fits;
  - Y2: the driver peak against the plan;
  - Y3 / Y4: the estimate ±5%;
  - Y5: the server's tier rows equal the estimate's;
  - Y6: the long-minus-short allocator delta against the flush and staging items' 1,003.9 MiB, ±15%. That is the first
    measurement of #1247's item.
- Each reading's consequence is registered before the data.
- Spend is capped at $5 by #1267, within the owner's $50 approval.
