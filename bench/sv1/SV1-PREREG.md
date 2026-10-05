# SV1 — `estimate_serve_footprint` with decode graphs and the prefill graph on

Work item: experts4bit-qlora#1152 (the owner-authorization permalink). Registered 2026-10-05 before any box, for one
RTX 5090 (Vast, verified-secure) through `bench/tc1/tc1_drive.sh` (`TC1_RUNNER=sv1_run.sh`).

**Question.** How close is `estimate_serve_footprint`'s device total to what `serve_paged.build_engine` holds once the
server runs its graphs? It lists two pools as not modelled:
- the bucketed decode graphs' memory (`E4B_PAGED_GRAPHS`);
- the first-chunk prefill graph's private pool (`E4B_PAGED_PREFILL_GRAPH`).

Neither has been measured beside the estimate:
- The estimate's checks so far ran on an RTX A2000 (sm_86), where decode graphs cannot run (#1090), and on P109's
  receipts, which predate the prefill graph.
- SC2b measured the prefill graph's pool at +3.3 GiB on Qwen3-30B-A3B int4, not at NF4.

## Shape (fixed)

| | |
|---|---|
| models | `allenai/OLMoE-1B-7B-0924` @ `6d84c48581ece794365f2b8e9cfb043c68ade9c5` (anchor); `Qwen/Qwen3-30B-A3B` @ `ad44e777bcd18fa416d9da3bd8f70d33ebb85d39` |
| arena | baked on the box from the checkpoint by grouped-nf4-gemm's `bake_nf4` (NF4, blocksize 64, fp32 absmax) |
| calibration | `bench/p39/calib.json` (an RTX 5090 box's blob); under `all-vram` the solver it feeds is overridden |
| server setup | `ServeSetup(max_seqs=16, max_tokens_per_seq=4096, chunk_tokens=512, placement="all-vram", hot_rows=64)` → `to_env()`; `E4B_PAGED_TORCH_THREADS=8` |
| arms | OLMoE: `olmoe_eager` (graphs 0, prefill graph 0), `olmoe_graphs` (graphs 1, prefill graph 0), `olmoe_prefill` (graphs 1, prefill graph 1). Qwen3: `qwen3_graphs`, `qwen3_prefill` |
| workload | 16 requests of 1,024 seeded random prompt tokens (vocab-bounded, seed 0), 32 new tokens each, served to idle |
| software | experts4bit-qlora at the launch head; grouped-nf4-gemm 0.41.0 (`dc8f94abfd868f149178623f6eb403dc8b892b02`, CI's pin); torch 2.8 (image), transformers 5.18.0, bitsandbytes 0.50.2 |

Each arm is one process (`sv1_measure.py`). It builds the engine in-process exactly as `serve_paged` does, with the
environment `ServeSetup.to_env()` gives, and writes `receipts/<arm>.json` (`sv1-arm/1`).

## Readings

- **S1 (estimate, eager).** `olmoe_eager`'s `measured.device_peak_bytes` against the estimate's device total, which
  includes prefill staging at its ceiling. Expected: within ±5%. The A2000's eager runs read −0.1% to +0.2% after
  #1139/#1141.
- **S2 (decode graphs).** `olmoe_graphs` minus `olmoe_eager`, allocator peak and reserved peak. No expectation is
  registered: this is the unpriced pool. The runner's `graph_status` per bucket is recorded.
- **S3 (prefill graph).**
  - `olmoe_prefill` minus `olmoe_graphs`, and the runner's own `prefill_graph_stats()["pool_mib"]`.
  - No expectation is registered at NF4 / OLMoE. SC2b's +3.3 GiB is Qwen3-30B int4.
- **S4 (30B).** The same differences at Qwen3-30B-A3B NF4 (`qwen3_prefill` − `qwen3_graphs`, and the estimate
  against `qwen3_graphs`).
- **Integrity**, every arm:
  - all 16 requests reach `done`;
  - decode graphs report `graph` for the buckets captured, where graphs are on;
  - the prefill graph reports `status: on` with replays > 0, where forced on.

  An arm failing integrity is reported as ALARM and its memory readings are not used.

## Outcomes

- **Success:** five arm receipts with integrity clean; S1–S4 read and reported as numbers. A miss on S1 is a result,
  not a failure.
- **Failure:**
  - install or tripwire (rc 9);
  - fetch (10);
  - bake (13);
  - an OLMoE arm not finishing (12): the runner stops before the Qwen3 download, so a broken instrument costs minutes;
  - fewer than five receipts (11).

  A refused prefill graph under `=1` is recorded with the runner's reason and counts as a finished arm.

## Cost

Estimate: about 1.5 h on one RTX 5090 at the policy rate. That covers ~14 GB + ~57 GB of downloads, two arena bakes
(3.6 GB, ~16 GB) and five builds. Ceiling: wallclock 2.5 h, spend $5 (#1152).
