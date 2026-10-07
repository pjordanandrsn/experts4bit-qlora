### Read: SV7 (#1294): the planner's re-matched 24 GB plan (VRAM tier 13.143 GiB) for Qwen3-30B-A3B served 8,000-token prompts inside its plan, and the borrowed same-shape reserve held (bench and receipts only)

- **The runs:**
  - `sv7-4090-1` ($0.611): V1 and V2 NO_READING, because its container listed host-namespace PIDs (amendment 1, #1308).
  - `sv7-4090-2` ($0.408): every driver sample matched by PID.
  - Lane total $1.019 (`bench/sv7/RESULTS-sv7.md`). Both runs' receipts and verdicts are committed and byte-identical to
    the store's.
- **Readings** (`sv7-4090-2`, through `bench/sv7/sv7_reduce.py`), all HELD:
  - V1: no OOM at 8 × 8,000-token prompts, and the driver peak is under the card (21.916 of 23.988 GiB);
  - V2: driver peak 21.916 GiB against the 22.344 GiB plan;
  - V3: reserved minus allocated 823 MiB against the borrowed 877 MiB, in both runs;
  - V4: −1.6%;
  - V5: tier rows exactly 5,316 / 828 / 0.
- **Consequences, as registered:**
  - the tier plan, the estimate's long-prompt total and its tier pricing stand;
  - loggetta's same-shape reserve rule stands for this shape and card class;
  - `sv7-4090-2`'s receipts become same-setup evidence for the planner's reserve and context on this class, and
    `sv7-4090-1`'s do not.
