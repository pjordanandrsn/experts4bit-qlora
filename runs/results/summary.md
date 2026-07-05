# Distributed validation summary (controller aggregate)

jobs with results: 15 (train 12, decode 3, query 0) | non-pass: 0

Observed on the recorded hosts/pods — not universal claims.

## Training repeats: mode x seed

| mode | seed 1337 | seed 2027 | seed 3407 |
|---|---|---|---|
| int8-offload | best 1.0181 / final 1.0240 / 2.72 GB / 17.0 s/step | best 1.0339 / final 1.0416 / 2.72 GB / 15.9 s/step | best 1.0262 / final 1.0299 / 2.72 GB / 18.0 s/step |
| int8-resident | best 1.0213 / final 1.0213 / 8.50 GB / 11.9 s/step | best 1.0383 / final 1.0435 / 8.50 GB / 11.9 s/step | best 1.0343 / final 1.0360 / 8.50 GB / 11.9 s/step |
| nf4-offload | best 1.0214 / final 1.0214 / 2.52 GB / 14.6 s/step | best 1.0346 / final 1.0469 / 2.52 GB / 14.8 s/step | best 1.0315 / final 1.0382 / 2.52 GB / 14.7 s/step |
| nf4-resident | best 1.0211 / final 1.0215 / 5.28 GB / 12.0 s/step | best 1.0441 / final 1.0467 / 5.28 GB / 11.9 s/step | best 1.0287 / final 1.0388 / 5.28 GB / 12.2 s/step |

## Training aggregate (across seeds)

| mode | n | best eval mean ± std [min, max] | peak GB mean | s/step mean |
|---|---|---|---|---|
| int8-offload | 3 | 1.0261 ± 0.0079 [1.0181, 1.0339] | 2.72 | 17.0 |
| int8-resident | 3 | 1.0313 ± 0.0089 [1.0213, 1.0383] | 8.50 | 11.9 |
| nf4-offload | 3 | 1.0292 ± 0.0069 [1.0214, 1.0346] | 2.52 | 14.7 |
| nf4-resident | 3 | 1.0313 ± 0.0117 [1.0211, 1.0441] | 5.28 | 12.0 |

## Decode repeats (resident; N measured samples after 1 discarded warmup)

| mode | samples | tok/s mean ± std [min, max] | peak GB |
|---|---|---|---|
| fp4-resident | 5 | 12.87 ± 0.20 [12.61, 13.10] | 4.72 |
| int8-resident | 5 | 11.63 ± 0.02 [11.61, 11.65] | 7.95 |
| nf4-resident | 5 | 12.68 ± 0.22 [12.29, 12.86] | 4.72 |

## Portability: eval with adapter (train row x query mode)

*no query jobs completed yet*

## Portability: delta vs query-mode no-adapter baseline

*no query jobs completed yet*

## Claim status (rule shown per claim; all host-specific observations)

| claim | evidence | rule | status |
|---|---|---|---|
| int8-offload posts the best training eval | best-eval wins in 3/3 seeds | win in all seeds vs every other mode | **Stable (host-specific)** |
| fp4 resident decode faster than nf4 (this host/path) | fp4 12.87±0.20 vs nf4 12.68±0.22 tok/s | means separated by >1 std each | **Candidate** |
| offload collapses the storage-width memory difference | resident delta 3.22 GB vs offload delta 0.20 GB (ratio 0.06) | offload delta < 0.25x resident delta | **Stable (host-specific)** |
| resident training memory scales with storage width | int8 8.50 GB vs nf4 5.28 GB resident | int8 resident peak > 1.4x nf4 resident peak | **Stable (host-specific)** |
| BEFORE-training eval tracks fidelity ordering (int8 < nf4) | holds in 3/3 seed-matched pairs | holds in all pairs | **Stable (host-specific)** |
