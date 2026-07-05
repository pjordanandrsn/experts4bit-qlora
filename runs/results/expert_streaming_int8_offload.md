# Expert-streaming profile

- model: allenai/OLMoE-1B-7B-0924 | phase: train | storage: int8 | offload: True | seed: 1337
- host: NVIDIA RTX A5000 | torch 2.8.0+cu128 | bnb 0.49.2
- methodology: CUDA events on the staging stream per copy, reduced at flush with a single synchronize; routing via per-forward on-device bincount; staging is layer-granular (whole fused stack per visit)

**Staging is layer-granular.** Measured H2D stall is per layer; per-(layer,expert) stall below is a PROJECTION (layer stall shared by token fraction) — the number a per-expert pinning policy would have to beat, not a measurement of isolated transfer.

- layers profiled: 16 | routed (layer,expert) pairs: 1024
- total measured H2D stall: 492952.6 ms across 6619.22 GB staged

## Top (layer, expert) by projected stall ms

| layer | expert | projected_stall_ms | hits | tokens |
|---|---|---|---|---|
| 8 | 18 | 1751.69 | 992 | 35352 |
| 0 | 6 | 1750.05 | 992 | 59177 |
| 6 | 18 | 1535.99 | 992 | 30997 |
| 7 | 1 | 1524.01 | 992 | 30755 |
| 6 | 36 | 1489.27 | 992 | 30054 |
| 5 | 13 | 1389.59 | 992 | 28040 |
| 6 | 23 | 1327.18 | 992 | 26783 |
| 1 | 15 | 1260.79 | 992 | 25437 |
| 3 | 15 | 1257.80 | 992 | 25383 |
| 14 | 28 | 1234.04 | 980 | 24895 |
| 4 | 43 | 1227.35 | 985 | 24767 |
| 3 | 61 | 1181.49 | 992 | 23843 |

## Top (layer, expert) by hits

| layer | expert | hits | hits | tokens |
|---|---|---|---|---|
| 0 | 1 | 992 | 992 | 5219 |
| 0 | 6 | 992 | 992 | 59177 |
| 0 | 7 | 992 | 992 | 13597 |
| 0 | 8 | 992 | 992 | 20899 |
| 0 | 13 | 992 | 992 | 12358 |
| 0 | 18 | 992 | 992 | 13810 |
| 0 | 21 | 992 | 992 | 29308 |
| 0 | 24 | 992 | 992 | 8811 |
| 0 | 31 | 992 | 992 | 10772 |
| 0 | 39 | 992 | 992 | 12394 |
| 0 | 40 | 992 | 992 | 11630 |
| 0 | 41 | 992 | 992 | 29706 |

## Top (layer, expert) by tokens routed

| layer | expert | tokens_routed | hits | tokens |
|---|---|---|---|---|
| 0 | 6 | 59177 | 992 | 59177 |
| 8 | 18 | 35352 | 992 | 35352 |
| 6 | 18 | 30997 | 992 | 30997 |
| 7 | 1 | 30755 | 992 | 30755 |
| 6 | 36 | 30054 | 992 | 30054 |
| 0 | 41 | 29706 | 992 | 29706 |
| 0 | 21 | 29308 | 992 | 29308 |
| 5 | 13 | 28040 | 992 | 28040 |
| 6 | 23 | 26783 | 992 | 26783 |
| 1 | 15 | 25437 | 992 | 25437 |
| 3 | 15 | 25383 | 992 | 25383 |
| 14 | 28 | 24895 | 980 | 24895 |

## Concentration (share of total held by the hottest pairs)

| metric | top 1% | top 5% | top 10% | top 20% |
|---|---|---|---|---|
| projected_stall_ms | 3% | 11% | 20% | 36% |
| hits | 1% | 6% | 11% | 23% |
| tokens_routed | 3% | 12% | 20% | 35% |

## Projected pinning budgets (greedy by stall-per-byte)

Estimated coverage if the hottest experts were held resident within a GPU cache budget (projections under the attribution rule, not measured speedups). Offload is not binary: this table is the dial a user would spend spare VRAM on.

| budget GB | pinned experts | added VRAM GB | projected stall covered | H2D GB avoided | hit % covered |
|---|---|---|---|---|---|
| 0.25 | 37 | 0.25 | 9% | 242.68 | 4% |
| 0.50 | 74 | 0.49 | 15% | 482.69 | 8% |
| 1.00 | 149 | 1.00 | 28% | 980.03 | 17% |
| 2.00 | 299 | 2.00 | 48% | 1972.03 | 32% |

## Decision

DO NOT build hot-static: stall is diffuse (top 10% = 20%, top 20% = 36%); the offload wall is not hot-expert concentrated for this model/path.

## What is not claimed

- No measurement of isolated per-expert transfer (staging is layer-granular).
- No speedup is claimed here — pinning coverage is a projection; a hot-static validation run is required to measure real s/step / stall change.
- Concentration is observed for this model/path/host; it is not asserted to generalize (Qwen3 gets a sentinel profile only if OLMoE shows concentration worth scaling).
