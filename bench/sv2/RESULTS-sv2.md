# SV2 read — `sv2-5090-1`

Registered in `bench/sv2/SV2-PREREG.md` (#1208) before the box; work item #1207 (owner-authorized, $5 cap).

**The run.**
- One RTX 5090: Vast instance 54392386, driver 595.91.07, AMD EPYC 7K62, 504 GB RAM, 320 GB disk.
- **$0.45**; teardown proven (destroy HTTP 200, instance absent from the list after).
- experts4bit-qlora `467f25d` (#1208's head, 0.48.0, including #1182), grouped-nf4-gemm 0.41.0 (`dc8f94a`),
  torch 2.8.0+cu128, transformers 5.18.0.
- The 57 GB checkpoint fetched and the arena baked on the box (16.3 GB, 133 s). The lane finished `TC1_SUCCESS` with
  four receipts, and the final fetch carried no arena (it lived outside the fetched tree).
- Every arm: Qwen3-30B-A3B, all-VRAM, decode graphs on, 16 sequences × 4,096 tokens, 16 requests of 1,024 seeded
  prompt tokens, 32 new tokens each.

Receipts: `receipts/sv2-5090-1/*.json` (`sv2-arm/1`), `summary.txt`, `forensics.txt`.

| arm | levers | prefill graph | estimate | allocator peak | load peak | reserved peak | driver peak | host anon after load |
|---|---|---|---|---|---|---|---|---|
| `q_nf4` | — | off | 21.786 | 21.612 | 21.362 | 21.895 | 22.619 | 3.21 GB |
| `q_exp` | int4 experts | off | 21.839 | 21.664 | 21.415 | 21.932 | 22.656 | 1.68 GB |
| `q_both` | int4 experts + attention | off | 22.410 | 22.236 | 20.299 | 22.531 | 23.252 | 1.72 GB |
| `q_exp_prefill` | int4 experts | on, 16 replays, pool 294 MiB | 21.839 | 22.222 | 22.222 | 22.660 | 23.398 | 1.88 GB |

GiB unless marked. "Estimate" is `estimate_serve_footprint`'s device total for the arm's `ServeSetup`. It excludes the
CUDA context and allocator slack, so it is read against the allocator peak.

**Readings.**
- **V1 (int4 experts, estimate): HELD.** `q_exp` peaked 0.8% under its estimate, inside ±5%. The margin is prefill
  staging priced at its ceiling: 432 MiB priced against the 144 MiB these prompts staged.
- **V2 (the int4 stores against NF4): HELD.** The load peak rose **+54.0 MiB** against +54 priced (±32). The serving
  peak also rose +54.0 MiB. Batched int4 decode under the decode graphs added nothing the NF4 graphs did not already
  hold.
- **V3 (int4 attention): HELD.** `q_both` − `q_exp` = **+585.2 MiB** against +585.0 priced.
  - The prereg wrote the price as "+571 MiB". That was a unit slip: the estimate's difference is 0.571 GiB, which is
    585 MiB. Its "+625 MiB" for `q_both` − `q_nf4` should likewise read +639 MiB.
  - The reading holds against the registered number too (14 MiB from +571, inside ±64).
  - The load peak fell to 20.299 GiB because the bf16 attention is freed at the swap; the kept bf16 copy is built at
    the first prefill.
- **V4 (the prefill graph at int4).** `q_exp_prefill` − `q_exp`:
  - **+570.8 MiB** allocated, +746 MiB reserved;
  - the runner's own pool: 294 MiB.
  - That is SV1's NF4 figure (+571 MiB allocated, pool 310 MiB). At this head the prefill graph costs the same at
    int4 as at NF4.
  - SC2b's +3.3 GiB, read at 0.47.0, is not reproduced. This lane does not say what changed.
- **V5 (the repack's host peak): HELD.** `q_exp`'s anonymous load peak exceeded `q_nf4`'s after-load memory by
  **3.30 GB** (5.7 B per parameter of a 604M-parameter layer), at or under the 6,912 MiB priced.
  - The baseline moves, though: the repack's per-layer trims also hand back heap the NF4 load left, which `q_nf4`
    keeps (V6).
  - So the repack's own peak above its starting point is not separable here. It is at most 6.68 − 1.68 = 5.0 GB,
    8.3 B per parameter, still under the price.
- **V6 (the heap handed back): HELD.** After load, `q_exp` holds **1.53 GB less** anonymous memory than `q_nf4`, and
  `q_both` 1.49 GB less (expected: no more than +0.25 GB).
  - The NF4 build itself kept 3.21 GB of anonymous memory after load. The int4 builds' trims release that too.
  - Trimming after every build, NF4 included, would return it. That is a separate change.
- **Integrity: clean.**
  - Every arm finished all 16 requests, and every decode bucket (1–16) reported `graph`.
  - `int4_expert_layers` was 48 wherever `exp_int4` was set, and `int4_attn_projections` 192 in `q_both`.
  - The forced prefill graph reported `status: on` with 16 replays.

**Also recorded, not registered.** One draw per arm, no self-pair.
- **tok/s** (including the prefill of 16 prompts): 37.3 / 39.4 / 41.6 / 49.3.
- **Expert routes:**
  - NF4: K25 for decode rows, the captured M-tile for chunks;
  - int4: K19 for both, and the singleton GEMV for single rows.
- **Load:** 54 s for NF4, 165–182 s with the repack (the whole bf16 checkpoint read and repacked).
- **CUDA context:** 0.72–0.74 GiB. **Allocator slack:** 1.2–2.0%.
