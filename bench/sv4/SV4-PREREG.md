# SV4 — the serve estimate and the planner's tiers on a real RTX 4090 (Qwen3-30B-A3B)

Work item: experts4bit-qlora#1236 (the owner-authorization permalink; $10 cap within an owner-approved $50). Registered
2026-10-06 before any box, for one RTX 4090 (Vast, verified-secure) through `bench/tc1/tc1_drive.sh`
(`TC1_RUNNER=sv4_run.sh`).

**Question.** Does `estimate_serve_footprint` hold for Qwen3-30B-A3B on a 24 GB card, at all-VRAM and on the solver's
tiers, and does the server split the experts the way the estimate prices them? So far:
- the planner's 24 GB plans are what-ifs on a stated card;
- the tiers are checked on OLMoE-1B-7B and ERNIE-4.5-21B (RTX A2000) only;
- at 30B, nothing has measured the cold rows' device stack, the DRAM tier's prefill on the GPU (#1229), NVMe
  streaming or tiered allocator slack.

## Shape (fixed)

| | |
|---|---|
| model | `Qwen/Qwen3-30B-A3B` @ `ad44e777bcd18fa416d9da3bd8f70d33ebb85d39` |
| arena | baked on the box by grouped-nf4-gemm's `bake_nf4` (NF4, blocksize 64, fp32 absmax) under `/root/sv4-arena`, outside the fetched tree; the NVMe tier reads this file |
| calibration | `bench/p39/calib.json` (the solver's bandwidth calibration; with no routing profile it decides nothing) |
| arms, in order | `t4_all1`: all-VRAM, 1 × 4096, decode graphs, bucket 1 (the planner's plan for 4096 × 1 on 24 GB; the anchor). `t4_plan8`: solver, 8 × 8192, VRAM 10.933 / DRAM 15.187 GiB, `hot_rows` 1, graphs off (the planner's plan for 8192 × 8 on 24 GB). `t4_deep4`: solver, 4 × 4096, VRAM 8.0 / DRAM 3.0 GiB, `hot_rows` 128 (the cold tier's minimum), graphs off |
| workload | as many requests as sequences, 1,024 seeded random prompt tokens each (vocab-bounded, seed 0), 32 new tokens, served to idle |
| software | experts4bit-qlora at the launch head (includes #1228, #1229, #1233); grouped-nf4-gemm 0.41.0 (`dc8f94abfd868f149178623f6eb403dc8b892b02`); torch 2.8 (image), transformers 5.18.0, bitsandbytes 0.50.2; `E4B_PAGED_TORCH_THREADS=8` |

Each arm is one process (`sv4_measure.py`) and writes `receipts/<arm>.json` (`sv4-arm/1`). It records:
- the estimate's device, host and NVMe items;
- device peaks;
- anonymous and pinned (`RssShmem`) host memory;
- the server's own tier split (`solve_placement` exactly as `build_engine` calls it);
- graph status.

The runner refuses a box with under 120 GB of disk or under 40 GB of available RAM (rc 13).

## The estimate's numbers, computed before the box (experts4bit-qlora `041c9ac9`)

| arm | device total | host items | NVMe | expert rows VRAM / DRAM / NVMe |
|---|---|---|---|---|
| `t4_all1` | 18.685 GiB | 0.408 GiB | — | all 6,144 |
| `t4_plan8` | 18.264 GiB | 4.280 GiB | — | 4,422 / 1,722 / 0 |
| `t4_deep4` | 12.528 GiB | 3.973 GiB | 4.190 GiB | 3,236 / 1,213 / 1,695 |

Each total includes prefill staging at its ceiling and, on the tiers, the chunk-sized transients (the cold rows' stack,
the DRAM tier's prefill on the GPU). With 1,024-token prompts each estimate is therefore expected to sit at or a little
above its peak.

## Readings

- **X1 (all-VRAM on 24 GB).** `t4_all1`'s allocator peak against its device total. Expected within ±5%.
- **X2 (the planner's tiers).** `t4_plan8`'s allocator peak against its device total. Expected within ±5%.
- **X3 (VRAM / DRAM / NVMe).** `t4_deep4`'s allocator peak against its device total. Expected within ±5%.
- **X4 (the split).** In both solver arms, the server's tier rows equal the estimate's, exactly.
- **X5 (pinned host memory).** `t4_deep4`'s `RssShmem` peak against the estimate's pinned cold-tier landing (512 MiB,
  `pinned_request_cost`). Expected at or above it and within +256 MiB.
- **Integrity**, every arm:
  - every request reaches `done`;
  - where graphs are on, every captured bucket reports `graph`.

  An arm failing integrity is reported as ALARM and its memory readings are not used.
- **Recorded, not registered:** tok/s (one draw each), anonymous host growth, allocator slack per arm (what the
  planner borrows for unmeasured tiers).

## Outcomes

- **Success:** three arm receipts with integrity clean; X1–X5 read and reported as numbers. A miss is a result, not a
  failure.
- **Failure codes:**
  - 9: install or tripwire, including the CPU tier's native kernels not building;
  - 10: fetch;
  - 13: bake, or under 120 GB disk / 40 GB RAM;
  - 12: the `t4_all1` anchor not finishing;
  - 11: fewer than three receipts.

## Cost

Estimate: about 1.5 h on one RTX 4090 at the policy rate. That covers a ~57 GB download, one ~16 GB bake (133 s on
SV2's box) and three builds. Ceiling: wallclock 2.5 h, spend $10 (#1236), within the owner's $50 approval.
