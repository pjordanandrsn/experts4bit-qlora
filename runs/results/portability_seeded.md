# Seeded phase-3 portability (bundle olmoe-qlora-grid-20260705-1351)

OLMoE-1B-7B, 3 seeds (1337/2027/3407), 24 query jobs, all `claim_usable` (commit-attested).
Cell = mean +/- std of held-out eval-with-adapter across seeds (n=3).

| train \ query | int8-resident | nf4-resident |
|---|---|---|
| nf4 | 1.0316 ± 0.0094 | 1.0319 ± 0.0107 |
| nf4-offload | 1.0325 ± 0.0123 | 1.0280 ± 0.0057 |
| int8 | 1.0323 ± 0.0085 | 1.0398 ± 0.0098 |
| int8-offload | 1.0260 ± 0.0079 | 1.0321 ± 0.0106 |
