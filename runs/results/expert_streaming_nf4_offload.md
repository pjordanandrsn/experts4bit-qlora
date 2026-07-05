# Expert-streaming profile

- model: allenai/OLMoE-1B-7B-0924 | phase: train | storage: nf4 | offload: True | seed: 1337
- host: NVIDIA RTX A5000 | torch 2.8.0+cu128 | bnb 0.49.2
- methodology: CUDA events on the staging stream per copy, reduced at flush with a single synchronize; routing via per-forward on-device bincount; staging is layer-granular (whole fused stack per visit)

**Staging is layer-granular.** Measured H2D stall is per layer; per-(layer,expert) stall below is a PROJECTION (layer stall shared by token fraction) — the number a per-expert pinning policy would have to beat, not a measurement of isolated transfer.

- layers profiled: 16 | routed (layer,expert) pairs: 1024
- total measured H2D stall: 261264.1 ms across 3504.29 GB staged

## Top (layer, expert) by projected stall ms

| layer | expert | projected_stall_ms | hits | tokens |
|---|---|---|---|---|
| 7 | 1 | 987.59 | 992 | 37610 |
| 8 | 18 | 971.33 | 992 | 36981 |
| 6 | 18 | 835.81 | 992 | 31813 |
| 0 | 6 | 800.88 | 992 | 51103 |
| 6 | 36 | 767.69 | 992 | 29220 |
| 5 | 13 | 762.36 | 992 | 29014 |
| 6 | 33 | 659.15 | 974 | 25089 |
| 12 | 43 | 622.71 | 988 | 23707 |
| 5 | 21 | 610.09 | 992 | 23219 |
| 14 | 28 | 607.00 | 936 | 23116 |
| 8 | 51 | 606.21 | 992 | 23080 |
| 13 | 49 | 589.44 | 918 | 22446 |

## Top (layer, expert) by hits

| layer | expert | hits | hits | tokens |
|---|---|---|---|---|
| 0 | 1 | 992 | 992 | 5389 |
| 0 | 6 | 992 | 992 | 51103 |
| 0 | 7 | 992 | 992 | 14115 |
| 0 | 8 | 992 | 992 | 20391 |
| 0 | 13 | 992 | 992 | 13002 |
| 0 | 18 | 992 | 992 | 15708 |
| 0 | 20 | 992 | 992 | 8079 |
| 0 | 21 | 992 | 992 | 29892 |
| 0 | 22 | 992 | 992 | 10966 |
| 0 | 24 | 992 | 992 | 9097 |
| 0 | 31 | 992 | 992 | 10098 |
| 0 | 39 | 992 | 992 | 15218 |

## Top (layer, expert) by tokens routed

| layer | expert | tokens_routed | hits | tokens |
|---|---|---|---|---|
| 0 | 6 | 51103 | 992 | 51103 |
| 7 | 1 | 37610 | 992 | 37610 |
| 8 | 18 | 36981 | 992 | 36981 |
| 6 | 18 | 31813 | 992 | 31813 |
| 0 | 21 | 29892 | 992 | 29892 |
| 0 | 41 | 29708 | 992 | 29708 |
| 6 | 36 | 29220 | 992 | 29220 |
| 5 | 13 | 29014 | 992 | 29014 |
| 6 | 33 | 25089 | 974 | 25089 |
| 12 | 43 | 23707 | 988 | 23707 |
| 5 | 21 | 23219 | 992 | 23219 |
| 14 | 28 | 23116 | 936 | 23116 |

## Concentration (share of total held by the hottest pairs)

| metric | top 1% | top 5% | top 10% | top 20% |
|---|---|---|---|---|
| projected_stall_ms | 3% | 11% | 20% | 35% |
| hits | 1% | 6% | 11% | 23% |
| tokens_routed | 3% | 11% | 20% | 35% |

## Projected pinning budgets (greedy by stall-per-byte)

Estimated coverage if the hottest experts were held resident within a GPU cache budget (projections under the attribution rule, not measured speedups). Offload is not binary: this table is the dial a user would spend spare VRAM on.

| budget GB | pinned experts | added VRAM GB | projected stall covered | H2D GB avoided | hit % covered |
|---|---|---|---|---|---|
| 0.25 | 70 | 0.25 | 14% | 241.50 | 8% |
| 0.50 | 141 | 0.50 | 26% | 490.75 | 15% |
| 1.00 | 282 | 1.00 | 45% | 984.34 | 31% |
| 2.00 | 565 | 2.00 | 74% | 1970.77 | 59% |

## Decision

DO NOT build hot-static: stall is diffuse (top 10% = 20%, top 20% = 35%); the offload wall is not hot-expert concentrated for this model/path.

## What is not claimed

- No measurement of isolated per-expert transfer (staging is layer-granular).
- No speedup is claimed here — pinning coverage is a projection; a hot-static validation run is required to measure real s/step / stall change.
- Concentration is observed for this model/path/host; it is not asserted to generalize (Qwen3 gets a sentinel profile only if OLMoE shows concentration worth scaling).
