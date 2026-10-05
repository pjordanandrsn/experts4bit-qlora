# SV2 — `estimate_serve_footprint` with the int4 serving levers, Qwen3-30B-A3B, decode graphs on

Work item: experts4bit-qlora#1207 (the owner-authorization permalink; $5 cap). Registered 2026-10-05 before any box, for
one RTX 5090 (Vast, verified-secure) through `bench/tc1/tc1_drive.sh` (`TC1_RUNNER=sv2_run.sh`).

**Question.** How close is `estimate_serve_footprint`'s device total, and its host repack term, to what
`serve_paged.build_engine` holds with the int4 levers on, at 30B and with decode graphs?

#1182 priced the levers and checked them on an RTX A2000 (OLMoE-1B-7B, eager). It cannot show three things:
- the int4 expert store under decode graphs, whose batched decode allocates through the graph pools;
- the first-chunk prefill graph's pool at int4 (SC2b: +3.3 GiB on Qwen3-30B-A3B, never measured beside the estimate);
- the repack's host peak on 604M-parameter layers (its price, 12 B per parameter, came from OLMoE's 403M).

## Shape (fixed)

| | |
|---|---|
| model | `Qwen/Qwen3-30B-A3B` @ `ad44e777bcd18fa416d9da3bd8f70d33ebb85d39` |
| arena | baked on the box from the checkpoint by grouped-nf4-gemm's `bake_nf4` (NF4, blocksize 64, fp32 absmax), under `/root/sv2-arena` (outside the fetched tree) |
| calibration | `bench/p39/calib.json`; under `all-vram` the solver it feeds is overridden |
| server setup | `ServeSetup(max_seqs=16, max_tokens_per_seq=4096, chunk_tokens=512, placement="all-vram", hot_rows=64, graphs=True, ...)` → `to_env()`; `E4B_PAGED_TORCH_THREADS=8` |
| arms, in order | `q_nf4` (levers off, prefill graph 0), `q_exp` (`exp_int4`), `q_both` (`exp_int4` + `attn_int4`), `q_exp_prefill` (`exp_int4`, prefill graph 1) |
| workload | 16 requests of 1,024 seeded random prompt tokens (vocab-bounded, seed 0), 32 new tokens each, served to idle |
| software | experts4bit-qlora at the launch head (includes #1182); grouped-nf4-gemm 0.41.0 (`dc8f94abfd868f149178623f6eb403dc8b892b02`, CI's pin); torch 2.8 (image), transformers 5.18.0, bitsandbytes 0.50.2 |

Each arm is one process (`sv2_measure.py`) and writes `receipts/<arm>.json` (`sv2-arm/1`). It records:
- the estimate's items;
- allocator, reserved and driver peaks, at load and overall;
- anonymous host memory: the load peak, after load, and serving's own peak;
- the runner's graph status, prefill-graph stats and expert routes.

`q_nf4` is the anchor: if it is not OK the runner stops before the three repacks.

## The estimate's numbers, computed before the box (#1182 at `0bf98cf3`)

| arm | device total | vs `q_nf4` | host repack item |
|---|---|---|---|
| `q_nf4` | 21.786 GiB | — | — |
| `q_exp` | 21.839 GiB | +54 MiB (the int4 stores' split-K partials) | 6,912 MiB |
| `q_both` | 22.410 GiB | +625 MiB (+571 over `q_exp`: the attention grid, its kept bf16 copy, its workspaces) | 6,912 MiB |
| `q_exp_prefill` | 21.839 GiB | +54 MiB (the prefill graph's pool is unpriced) | 6,912 MiB |

Every arm's total includes prefill staging at its ceiling. That ceiling is 4,608 tokens = 432 MiB; these prompts stage
1,536 tokens = 144 MiB. So the estimate is expected to sit about 288 MiB above each peak, as SV1's S1 read.

## Readings

- **V1 (int4 experts, estimate).** `q_exp`'s allocator peak against its device total. Expected within ±5%.
- **V2 (the int4 stores against NF4).** `q_exp` − `q_nf4`:
  - load peak (allocated): expected within ±32 MiB of the priced +54 MiB;
  - serving peak: reported. It includes the int4 batched decode's allocations through the decode graphs' pools,
    which are unpriced.
- **V3 (int4 attention).** `q_both` − `q_exp`, allocator peak. Expected within ±64 MiB of the priced +571 MiB.
- **V4 (the prefill graph at int4).** `q_exp_prefill` − `q_exp`, allocator and reserved peaks, and the runner's own
  `prefill_graph_stats()["pool_mib"]`. No expectation is registered: this is the unpriced pool, and SC2b read +3.3 GiB.
- **V5 (the repack's host peak).** `q_exp`'s `host_anon_load_peak_bytes` − `q_nf4`'s `host_anon_after_load_bytes`.
  Expected at or under the priced 6,912 MiB. The item is a ceiling. #1182 measured 10.4–11.2 B per parameter of a
  layer against the 12 priced.
- **V6 (the heap handed back).** `q_exp`'s and `q_both`'s `host_anon_after_load_bytes` against `q_nf4`'s. Expected
  within +0.25 GB (#1182's trims).
- **Integrity**, every arm:
  - all 16 requests reach `done`;
  - decode graphs report `graph` for the buckets captured;
  - `int4_expert_layers` = 48 where `exp_int4`, and `int4_attn_projections` = 192 where `attn_int4`;
  - the prefill graph reports `status: on` with replays > 0 in `q_exp_prefill`.

  An arm failing integrity is reported as ALARM and its memory readings are not used. A refused prefill graph under
  `=1` is recorded with the runner's reason and counts as a finished arm.
- **Recorded, not registered:** tok/s per arm (one draw each, no self-pair) and the expert routes seen.

## Outcomes

- **Success:** four arm receipts with integrity clean; V1–V6 read and reported as numbers. A miss on V1, V2, V3, V5 or
  V6 is a result, not a failure.
- **Failure codes:**
  - 9: install or tripwire, including a card where decode graphs cannot run, or a package without the int4 levers;
  - 10: fetch;
  - 13: bake, or under 120 GB of free disk;
  - 12: the `q_nf4` anchor not finishing;
  - 11: fewer than four receipts.

## Cost

Estimate: about 1.5–2 h on one RTX 5090 at the policy rate. That covers a ~57 GB download, one ~16 GB arena bake
(260 s in SV1), one NF4 build and three builds with the int4 repack (the whole bf16 checkpoint read at each). Ceiling:
wallclock 2.5 h, spend $5 (#1207).
