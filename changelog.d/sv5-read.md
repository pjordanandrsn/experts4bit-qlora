### Read (exploratory): SV5 (#1242): the planner's all-VRAM 8 × 8192 plan for Qwen3-30B-A3B ran out of memory on a real RTX 4090 at 8,000-token prompts (bench and receipts only; licenses no change)

- `sv5-4090-1`: one RTX 4090, **$0.128**, teardown proven. Two arm receipts (`bench/sv5/RESULTS-sv5.md`).
- Read through `bench/sv5/sv5_reduce.py` (#1243), verdict `READ`:
  - Z1 MISSED: `a5_long` ran out of memory;
  - Z2 MISSED on a lower bound (23.53 GiB against the plan's 22.74);
  - Z3 NO_READING: the floor is +2.1%, inside the band;
  - Z4 HELD at −1.9%.
- The OOM's site is not recorded. The unpriced bulk KV flush is the leading candidate, not a measured cause; #1247
  prices it on its code.
- **Exploratory:** the box ran before #1243 was reviewed, and its rule was written after the data. Per #1243 this read
  licenses no change in this package or in the planner.
