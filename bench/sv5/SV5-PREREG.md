# SV5 — the planner's all-VRAM 8 × 8192 plan for Qwen3-30B-A3B on a 24 GB RTX 4090, at the longest prompts

Work item: experts4bit-qlora#1242 (the owner-authorization permalink; $5 cap within an owner-approved $50). Registered
2026-10-06 before any box, for one RTX 4090 (Vast, verified-secure) through `bench/tc1/tc1_drive.sh`
(`TC1_RUNNER=sv5_run.sh`).

**Question.** With SV4's receipts on file, a planner moves Qwen3-30B-A3B at 8 × 8192 on a 24 GB card from tiers to
all-VRAM with decode graphs. It plans 22.74 GiB:
- `estimate_serve_footprint`'s 22.150 GiB;
- a 0.09 GiB reserve learned from SV4's one-sequence arm;
- a 0.5 GiB CUDA context.

Does that plan bound the process when the prompts are as long as the setup admits? The estimate lists the bulk KV
flush transient (one prompt's K/V staged for its single write) as unmodelled; at 8,000 tokens it could be ~0.75 GiB.

## Shape (fixed)

| | |
|---|---|
| model | `Qwen/Qwen3-30B-A3B` @ `ad44e777bcd18fa416d9da3bd8f70d33ebb85d39` |
| arena | baked on the box by grouped-nf4-gemm's `bake_nf4` under `/root/sv5-arena`, outside the fetched tree |
| server setup | `ServeSetup(placement="all-vram", max_seqs=8, max_tokens_per_seq=8192, chunk_tokens=512, graphs=True, buckets=(1, 2, 4, 8), prefill_graph="0", hot_rows=64)` → `to_env()`; `E4B_PAGED_TORCH_THREADS=8` |
| arms, in order | `a5_short`: 8 requests of 1,024 seeded prompt tokens (the anchor). `a5_long`: 8 requests of 8,000 seeded prompt tokens. Both 32 new tokens, served to idle |
| measure | `bench/sv4/sv4_measure.py` (`sv4-arm/1`): the estimate's items, allocator / reserved / driver peaks, host memory, graph status |
| software | experts4bit-qlora at the launch head; grouped-nf4-gemm 0.41.0 (`dc8f94abfd868f149178623f6eb403dc8b892b02`); torch 2.8 (image), transformers 5.18.0, bitsandbytes 0.50.2 |

## The estimate's numbers, computed before the box (experts4bit-qlora `fd70f75b`)

Device total **22.150 GiB**:
- frozen expert stacks 15,552 MiB;
- dense 2,939 MiB;
- FP8 paged KV pool 3,320 MiB;
- prefill staging at its ceiling (8,704 tokens) 816 MiB;
- working set 55 MiB.

The plan's total is 22.74 GiB; the card reports 23.99 GiB.

## Readings

- **Z1 (the plan fits).** `a5_long` finishes all 8 requests with no out-of-memory error, and its driver-reported peak is
  at or under the card's total. Expected: yes.
- **Z2 (the plan bounds the process).** `a5_long`'s driver peak against the plan's 22.74 GiB. Expected: at or under it.
  A miss names what the plan did not carry.
- **Z3 (the estimate, long prompts).** `a5_long`'s allocator peak against 22.150 GiB. Expected within ±5%. The bulk
  KV flush transient is unpriced and only adds.
- **Z4 (the estimate, short prompts).** `a5_short`'s allocator peak against 22.150 GiB. Expected within ±5%, below it by
  about the staging ceiling (816 MiB priced against ~144 MiB staged).
- **Integrity**, both arms:
  - every request reaches `done`;
  - every captured decode bucket reports `graph`.

  An arm failing integrity is ALARM. An out-of-memory error in `a5_long` is Z1's reading, not an ALARM.
- **Recorded, not registered:** tok/s (one draw each), host memory, slack.

## Outcomes

- **Success:** two arm receipts with integrity clean; Z1–Z4 read. A miss is a result.
- **Failure codes:**
  - 9: install or tripwire;
  - 10: fetch;
  - 13: bake or under 120 GB disk;
  - 12: the `a5_short` anchor not finishing;
  - 11: fewer than two receipts. An OOM in `a5_long` still writes its receipt with status OOM.

## Cost

Estimate: about 1 h on one RTX 4090 at the declared $0.60/h: a ~57 GB download, one ~16 GB bake and two builds.
Ceiling: wallclock 2.5 h, spend $5 (#1242), within the owner's $50 approval.
