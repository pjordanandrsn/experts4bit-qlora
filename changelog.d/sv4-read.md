### Read: SV4 (#1236): the serve estimate and the planner's tiers held at 30B on a real RTX 4090 (bench and receipts only)

- `sv4-4090-1`: one RTX 4090, **$0.153**, teardown proven. Three arms OK (`bench/sv4/RESULTS-sv4.md`).
- Held:
  - X1: all-VRAM 1 × 4096, −0.1%;
  - X2: the planner's tiers for 8 × 8192, −4.3%;
  - X3: VRAM / DRAM / NVMe, −3.2%, with 4.2 GiB streamed from NVMe;
  - X4: the server's tier split equalled the estimate's exactly in both solver arms.
- **X5 MISSED.** Pinned host memory was 1.9 GB against the 512 MiB landing. Every arm carries 1.1–1.3 GB of pinned
  memory beyond the priced landing and setup tier; it is not attributed yet.
- Tiered allocator slack at 30B was 4.1–6.7%, against the up-to-15% a planner had borrowed from OLMoE.
