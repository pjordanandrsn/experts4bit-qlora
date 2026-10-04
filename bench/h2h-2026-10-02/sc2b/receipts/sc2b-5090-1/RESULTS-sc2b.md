# sc2b-5090-1: the registered rule's tables (rendered from `sc2/verdict_sc2b.json`, which `sc2b_reduce.py` re-derives identically from the committed run files)

### Gates (in order)

| gate | result | detail |
|---|---|---|
| ROUTES | pass | `{}` |
| ENGAGED | pass | `{"1": [], "2": []}` |
| PROMPTS | pass | `{}` |
| DETERMINISM (OFF d1 vs repeat) | IDENTICAL | n 24, 17701 chars |
| IDENTITY (OFF vs ON, serial) | **IDENTICAL** | d1: IDENTICAL, d2: IDENTICAL |

### Rows per arm (SC2's rule)

| arm | serial p50 TTFT d1 / d2 | serial p50 TPOT d1 / d2 (ms) | attainment @1 / @2 / @4 / @8 (d1, d2) | ceiling |
|---|---|---|---|---|
| OFF | 0.263 / 0.222 (UNSTABLE) | 4.51 / 4.48 | 1.00, 0.98 / 0.57, 0.82 (UNSTABLE) / 0.10, 0.19 (UNSTABLE) / 0.04, 0.03 | **1** |
| ON | 0.160 / 0.171 (VALID) | 4.46 / 4.52 | 1.00, 1.00 / 0.90, 1.00 / 0.12, 0.29 (UNSTABLE) / 0.06, 0.06 | **1** |

### Predictions and the licence

| # | verdict | detail |
|---|---|---|
| P1 | **REFUTED** | `{"ttft_off_over_on": [1.645, 1.2994]}` |
| P2 | **HOLDS** | `{"tpot_on_over_off": [0.9876, 1.0089]}` |
| P3 | **REFUTED** | `{"ceiling_on": 1}` |
| P4 | **HOLDS** | `{"pairs": {"r1_d1": [1.0, 1.0], "r1_d2": [0.9833, 1.0], "r2_d1": [0.5667, 0.9], "r2_d2": [0.825, 1.0], "r4_d1": [0.1, 0.1167], "r4_d2": [0.1917, 0.2917], "r8_d1": [0.0417, 0.0583], "r8_d2": [0.0333, 0.0583]}, "regression` |
| licence | **DEFAULT_LICENSED** | failed: none; void: False; TTFT OFF/ON per draw: [1.645, 1.2994] |
