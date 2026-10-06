# SV3 — the hybrid model's linear-attention state pool and the decode-bucket cap, beside the estimate (+ gpt-oss)

Work item: experts4bit-qlora#1224 (the owner-authorization permalink; $10 cap). Registered 2026-10-06 before any box,
for one RTX 5090 (Vast, verified-secure) through `bench/tc1/tc1_drive.sh` (`TC1_RUNNER=sv3_run.sh`).

**Question.** Two things are priced but have not been measured on a full model.
- **The state pool.** #1219 priced the paged server's per-slot linear-attention state pool for the hybrid models
  (Qwen3.5 / 3.6 / Next): a bf16 conv window and an fp32 recurrent state per Gated DeltaNet layer per slot. A unit
  test pins the arithmetic against a real `LinearStatePool` on a tiny model. Qwen3.6-35B-A3B's own pool, and the
  estimate's total beside its peak, are unmeasured.
- **The bucket cap.** The planner now caps the decode-graph buckets at the sequences, because each scratch slot costs
  a full slot of that state. Its saving for one user is priced, not measured.
- **And a fourth family.** gpt-oss-20b's serve estimate (per-expert biases, an MXFP4 source) has not been checked
  beside a run.

## Shape (fixed)

| | |
|---|---|
| models | `Qwen/Qwen3.6-35B-A3B` @ `995ad96eacd98c81ed38be0c5b274b04031597b0` (P106's pin); `openai/gpt-oss-20b` @ `6cee5e81ee83917806bbde320786a8fb61efebee` (SC2g's pin) |
| arenas | baked on the box through experts4bit-qlora's loader (`bench/p98/p98_bake.py`, NF4), under `/root/sv3-work-*` (outside the fetched tree); Qwen3.6's freed before gpt-oss is fetched |
| calibration | `bench/p39/calib.json`; under `all-vram` the solver it feeds is overridden |
| server setup | `ServeSetup(max_tokens_per_seq=4096, chunk_tokens=512, placement="all-vram", hot_rows=64, prefill_graph="0", ...)` → `to_env()`; `E4B_PAGED_TORCH_THREADS=8` |
| arms, in order | `q36_e16` (16 seqs, eager); `q36_g16` (16 seqs, decode graphs, buckets 1–16); `q36_g1_default` (1 seq, graphs, buckets 1–16); `q36_g1_capped` (1 seq, graphs, bucket 1); `gptoss_g16` (16 seqs, graphs, buckets 1–16) |
| workload | as many requests as sequences, 1,024 seeded random prompt tokens each (vocab-bounded, seed 0), 32 new tokens, served to idle |
| software | experts4bit-qlora at the launch head (includes #1219); grouped-nf4-gemm 0.41.0 (`dc8f94abfd868f149178623f6eb403dc8b892b02`); torch 2.8 (image), transformers 5.18.0, bitsandbytes 0.50.2 |

Each arm is one process (`sv3_measure.py`) and writes `receipts/<arm>.json` (`sv3-arm/1`). It records:
- the estimate's items;
- device peaks;
- anonymous host memory;
- graph status;
- on a hybrid, the pool's own `nbytes()`, slot count and layer count at the end of the arm.

`q36_e16` is the anchor: if it is not OK the runner stops.

## The estimate's numbers, computed before the box (experts4bit-qlora `dfc5bdaf`)

| arm | device total | state pool | |
|---|---|---|---|
| `q36_e16` | 23.217 GiB | 990.0 MiB (16 slots) | |
| `q36_g16` | 24.187 GiB | 1,980.0 MiB (32 slots) | |
| `q36_g1_default` | 22.629 GiB | 1,051.9 MiB (17 slots) | |
| `q36_g1_capped` | 21.720 GiB | 123.8 MiB (2 slots) | the cap's saving: 931 MiB |
| `gptoss_g16` | 15.391 GiB | — | |

Every total includes prefill staging at its ceiling. On Qwen3.6, 10 attention layers with 2 KV heads × 256 cost
90 MiB at 16 sequences and 80 MiB at 1. On gpt-oss it is 216 MiB. These prompts stage 1,536 tokens (1,024 for one
sequence), so each estimate is expected to sit a little above its peak.

## Readings

- **W1 (estimate, Qwen3.6, eager).** `q36_e16`'s allocator peak against its device total. Expected within ±5%.
- **W2 (the pool, exact).** In every Qwen3.6 arm, the pool's `nbytes()` at the end equals the estimate's
  "linear-attention state pool" item, to the byte, with the slot count `max_seqs` + the largest bucket (graphs on) and
  30 layers. Expected: exact.
- **W3 (estimate, Qwen3.6, graphs).** `q36_g16`'s allocator peak against its device total. Expected within ±5%. The
  decode graphs' pools are unpriced (SV1: +60 MiB on OLMoE).
- **W4 (the bucket cap).** `q36_g1_default` − `q36_g1_capped`, allocator peak, against the priced 931 MiB. Expected
  between priced − 32 MiB and priced + 256 MiB; the four extra buckets' graph pools are unpriced and only add.
- **W5 (gpt-oss).** `gptoss_g16`'s allocator peak against its device total. Expected within ±5%.
- **Integrity**, every arm:
  - every request reaches `done`;
  - decode graphs report `graph` for every captured bucket where on.

  An arm failing integrity is reported as ALARM and its memory readings are not used.
- **Recorded, not registered:** tok/s (one draw per arm), load seconds, host memory.

## Outcomes

- **Success:** five arm receipts with integrity clean; W1–W5 read and reported as numbers. A miss is a result, not a
  failure.
- **Failure codes:**
  - 9: install or tripwire, including a card where decode graphs cannot run or a transformers without the per-slot
    state carrier;
  - 10: fetch;
  - 13: bake, or under 150 GB of free disk;
  - 12: the `q36_e16` anchor not finishing;
  - 11: fewer than five receipts. A gpt-oss bake failure leaves four, recorded as such.

## Cost

Estimate: about 1.5 h on one RTX 5090 at the policy rate. That covers a ~72 GB and a ~13 GB download, two loader bakes
and five builds. Ceiling: wallclock 2.5 h, spend $10 (#1224), within the owner's $50 approval for this stretch.
