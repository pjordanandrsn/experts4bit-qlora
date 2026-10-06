### SV4 registered (#1236): the serve estimate and the planner's tiers on a real RTX 4090, Qwen3-30B-A3B (bench and prereg only)

- `bench/sv4/`: `SV4-PREREG.md`, `sv4_run.sh` (box side; one arena, which the NVMe tier also reads) and
  `sv4_measure.py`. The measure script records the estimate's device, host and NVMe items, pinned host memory, and the
  server's own tier split.
- Arms:
  - `t4_all1`: all-VRAM, 1 × 4096 (the anchor);
  - `t4_plan8`: the planner's tiers for 8 × 8192 on 24 GB;
  - `t4_deep4`: VRAM 8 / DRAM 3 GiB, 4.2 GiB on NVMe.
- Readings:
  - X1–X3: the estimate within ±5% of each peak;
  - X4: the server's split equals the estimate's;
  - X5: the pinned landing.
- Spend is capped at $10 by #1236, within the owner's $50 approval.
