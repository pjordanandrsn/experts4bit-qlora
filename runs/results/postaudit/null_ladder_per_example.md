# Per-example ∅ ladder (debts D1 + D2)

- jobs: 13 | GPU(s): ['NVIDIA RTX A5000'] | host(s): ['41a04ccbc77d']

## Ladder (mean ± SE over examples)

| mode | resident | offload |
|---|---|---|
| fp4 | 1.5041 ± 0.1065 (n=64) | 1.5041 ± 0.1065 (n=64) |
| nf4 | 1.4905 ± 0.1055 (n=64) | 1.4905 ± 0.1055 (n=64) |
| int8 | 1.4811 ± 0.1069 (n=64) | 1.4811 ± 0.1069 (n=64) |
| fp8 | 1.4724 ± 0.1018 (n=64) | 1.4724 ± 0.1018 (n=64) |
| bf16 | 1.4818 ± 0.1082 (n=64) | 1.4818 ± 0.1082 (n=64) |
| fp16 | 1.4780 ± 0.1067 (n=64) | 1.4780 ± 0.1067 (n=64) |

## D1 — paired mode deltas, resident (rowmode − colmode)

| pair | mean d | sd | SE | |t| |
|---|---|---|---|---|
| fp4 − nf4 | +0.0136 | 0.0601 | 0.0075 | 1.81 |
| fp4 − int8 | +0.0230 | 0.0862 | 0.0108 | 2.13 |
| fp4 − fp8 | +0.0317 | 0.0773 | 0.0097 | 3.28 |
| fp4 − bf16 | +0.0223 | 0.0899 | 0.0112 | 1.99 |
| fp4 − fp16 | +0.0260 | 0.0843 | 0.0105 | 2.47 |
| nf4 − int8 | +0.0094 | 0.0610 | 0.0076 | 1.24 |
| nf4 − fp8 | +0.0181 | 0.0480 | 0.0060 | 3.02 |
| nf4 − bf16 | +0.0088 | 0.0643 | 0.0080 | 1.09 |
| nf4 − fp16 | +0.0125 | 0.0568 | 0.0071 | 1.76 |
| int8 − fp8 | +0.0087 | 0.0684 | 0.0086 | 1.01 |
| int8 − bf16 | -0.0007 | 0.0180 | 0.0022 | 0.30 |
| int8 − fp16 | +0.0030 | 0.0132 | 0.0016 | 1.85 |
| fp8 − bf16 | -0.0094 | 0.0781 | 0.0098 | 0.96 |
| fp8 − fp16 | -0.0056 | 0.0659 | 0.0082 | 0.68 |
| bf16 − fp16 | +0.0037 | 0.0188 | 0.0024 | 1.58 |

## D2 — placement delta per mode (resident − offload, example-paired)

| mode | mean d | SE | |t| | verdict |
|---|---|---|---|---|
| fp4 | +0.000000 | 0.000000 | inf | identical |
| nf4 | +0.000000 | 0.000000 | inf | identical |
| int8 | +0.000000 | 0.000000 | inf | identical |
| fp8 | +0.000000 | 0.000000 | inf | identical |
| bf16 | +0.000000 | 0.000000 | inf | identical |
| fp16 | +0.000000 | 0.000000 | inf | identical |

## Eval-determinism repeat (int8-resident, run twice)

- per-example max |d| = 0.000e+00, mean |d| = 0.000e+00, bitwise-identical examples: 64/64

## S9 check — G_int8 against paired SE

- G_int8 (nf4−int8) = +0.0094 ± 0.0076 (|t| = 1.24)
- G_total (nf4−bf16) = +0.0088 ± 0.0080 (|t| = 1.09)
- coverage = 108%
- **S9 FIRES — G_int8 indistinguishable from 0; precision program drops to screening**

