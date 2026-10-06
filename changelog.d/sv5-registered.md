### SV5 registered (#1242): the planner's all-VRAM 8 × 8192 plan for Qwen3-30B-A3B on a 24 GB RTX 4090, at the longest prompts (bench and prereg only)

- `bench/sv5/`: `SV5-PREREG.md` and `sv5_run.sh` (box side; one `bake_nf4` arena; SV4's `sv4_measure.py`).
- Arms:
  - `a5_short`: 1,024-token prompts (the anchor);
  - `a5_long`: 8,000-token prompts.
- Readings:
  - Z1: the plan fits (no OOM);
  - Z2: the driver peak at or under the plan's 22.74 GiB;
  - Z3, Z4: the estimate within ±5%.
- Spend is capped at $5 by #1242, within the owner's $50 approval.
