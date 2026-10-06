# SV4 read — `sv4-4090-1`

Registered in `bench/sv4/SV4-PREREG.md` (#1239) before the box; work item #1236 (owner-authorized, $10 cap within an
owner-approved $50).

**The run.**
- One RTX 4090 (sm_89): Vast instance 54428318, driver 595.84, Intel i9-10980XE, 125 GB RAM, 320 GB disk.
- **$0.153**; teardown proven (destroy HTTP 200, instance absent from the list after).
- experts4bit-qlora `e80b04a` (#1239's head, 0.48.0, including #1228/#1229/#1233), grouped-nf4-gemm 0.41.0 (`dc8f94a`),
  torch 2.8.0+cu128, transformers 5.18.0.
- The arena was baked on the box (16.3 GB, 209 s). The lane finished `TC1_SUCCESS` with three receipts.

Receipts: `receipts/sv4-4090-1/*.json` (`sv4-arm/1`), `summary.txt`, `forensics.txt`.

| arm | placement | seqs × tokens | estimate | allocator peak | vs estimate | the server's split (VRAM / DRAM / NVMe rows) | slack |
|---|---|---|---|---|---|---|---|
| `t4_all1` | all-VRAM, graphs, bucket 1 | 1 × 4096 | 18.685 | 18.668 | −18 MiB (−0.1%) | all 6,144 | 0.42% |
| `t4_plan8` | the planner's tiers (10.933 / 15.187 GiB) | 8 × 8192 | 18.264 | 17.471 | −812 MiB (−4.3%) | 4,422 / 1,722 / 0, as priced | 4.08% |
| `t4_deep4` | tiers 8.0 / 3.0 GiB, `hot_rows` 128 | 4 × 4096 | 12.528 | 12.131 | −406 MiB (−3.2%) | 3,236 / 1,213 / 1,695, as priced | 6.68% |

GiB unless marked. "Estimate" is `estimate_serve_footprint`'s device total, read against the allocator peak.

**Readings.**
- **X1 (all-VRAM on 24 GB): HELD.** 18 MiB under, the closest any serve run has read.
- **X2 (the planner's tiers): HELD** at −4.3%.
  - The margin is the chunk-sized ceilings. Prefill staging is priced for 8,192 + 512 tokens (816 MiB) and these
    prompts staged 1,536.
  - The DRAM tier's prefill on the GPU (#1229) is priced for a full chunk of routed experts.
- **X3 (VRAM / DRAM / NVMe): HELD** at −3.2%. 1,695 expert rows (4.2 GiB) streamed from the arena on NVMe.
- **X4 (the split): HELD, exactly.** In both solver arms the server's tier rows equal the estimate's.
- **X5 (pinned host memory): MISSED.**
  - `t4_deep4`'s pinned (`RssShmem`) peak was 1,904 MiB against the 512 MiB landing (expected at most +256 MiB above
    it).
  - The other arms show the same: `t4_all1` held 1.55 GB, `t4_plan8` 1.14 GB.
  - Subtracting the priced landing and setup tier leaves 1.1–1.3 GB in every arm, whatever `hot_rows` was. That is a
    fixed pinned cost of building Qwen3-30B-A3B that the estimate does not price.
  - The RTX A2000's OLMoE builds left ~48 MB unexplained, so it grows with the model.
  - Not attributed by this lane.
- **Integrity: clean.** Every arm finished all its requests, and `t4_all1`'s decode bucket reported `graph`.

**Also recorded, not registered.**
- **Tiered allocator slack at 30B: 4.1% and 6.7%.** A planner borrowing the largest tiered slack it had measured
  (OLMoE on an A2000, up to 15%) over-reserves by about 2.9 GiB here, enough to push experts from VRAM to DRAM.
- **Anonymous host memory** during serving: 1.63 / 6.38 / 4.97 GB. The DRAM tier (4.4 / 3.1 GiB) is in the second
  and third.
- **tok/s**, one draw each: 2.8 / 6.8 / 3.1.
- **CUDA context:** 0.46–0.50 GiB.

## Maintainer note (2026-10-06, after the merge; nothing above is changed)

How this box came to run, from #1239's and #1240's events, their reviews and the launch receipt
(`receipts/experts4bit-qlora/2026-10-06/sv4-4090-1/receipt.json` in the receipt store):

| time (UTC) | event |
|---|---|
| 04:02:28 | #1236 (the work item) opened; the maintainer posted the registration requirements on it |
| 04:04:01 | #1239 (the registration) opened, labelled, auto-merge on |
| **04:05:42** | **`sv4-4090-1` launched**, from a registration that was not on main, before review |
| ~04:06 | maintainer review requested three pre-data changes: registered consequences for X1–X5, a reducer, exit codes (bake → not 13; RAM → 18) |
| 04:07:19 | auto-merge re-enabled at the unchanged head; disabled 04:08:07 |
| 04:25:06 | box finished, $0.153 |
| 04:27:01 | its receipt committed on the mini and never pushed; it blocked every lane's launches until another session pushed it |
| 04:29:09 | #1239 merged directly at the reviewed head, none of the changes made |
| 04:28:46 | this read (#1240) opened with auto-merge on; the maintainer held it for this timeline |
| 04:55:41 | #1240 merged directly, without the timeline |

**What follows.** X1–X5's expectations were public before the data (#1239 at 04:04:01Z), so the readings are not post-hoc.
But no consequence was registered and no reducer existed before the data, so:
- this read **licenses no change** to the estimate, the planner's slack or its tier rule; any change it suggests needs its
  own registration and box. X5's 1.9 GB of pinned memory against a 512 MiB landing is a finding (#1241 measures and fixes its
  source on its own A2000 probe), not a re-price;
- "HELD" above means "inside the registered expectation", not a registered verdict with a consequence;
- the X table has no registered reducer behind it;
- `sv4_run.sh` maps a bake failure to exit 13 and a RAM floor to 13 (adertha's host-floor code is 18); do not reuse the
  runner until both are fixed.
