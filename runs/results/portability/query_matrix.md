# Train/query storage-mode matrix

- base model: allenai/OLMoE-1B-7B-0924
- host: 5efb3005656c | GPU: NVIDIA RTX A5000
- legs: 25 pass / 0 fail-or-skip

Read as a storage-mode portability test observed on this host/model/dataset/run — not a benchmark, not a universal compatibility claim.

## Held-out eval loss (with adapter)

| train \ query | bf16 | fp16 | fp4 | int8 | nf4 |
|---|---|---|---|---|---|
| fp4 | 1.0314 | 1.0315 | 1.0348 | 1.0301 | 1.0210 |
| int8 | 1.0179 | 1.0173 | 1.0378 | 1.0171 | 1.0237 |
| int8-offload | 1.0147 | 1.0153 | 1.0360 | 1.0126 | 1.0226 |
| nf4 | 1.0271 | 1.0265 | 1.0354 | 1.0217 | 1.0208 |
| nf4-offload | 1.0245 | 1.0245 | 1.0361 | 1.0242 | 1.0216 |

## Delta vs same-query-mode no-adapter baseline (negative = adapter helps)

| train \ query | bf16 | fp16 | fp4 | int8 | nf4 |
|---|---|---|---|---|---|
| fp4 | -0.4504 | -0.4465 | -0.4692 | -0.4510 | -0.4695 |
| int8 | -0.4639 | -0.4608 | -0.4663 | -0.4640 | -0.4669 |
| int8-offload | -0.4671 | -0.4627 | -0.4680 | -0.4685 | -0.4679 |
| nf4 | -0.4546 | -0.4515 | -0.4687 | -0.4594 | -0.4697 |
| nf4-offload | -0.4572 | -0.4535 | -0.4680 | -0.4568 | -0.4689 |

## Query cost (peak GPU GB; decode tok/s not measured this pass)

| train \ query | bf16 | fp16 | fp4 | int8 | nf4 |
|---|---|---|---|---|---|
| fp4 | 14.22 | 14.22 | 4.96 | 8.18 | 4.96 |
| int8 | 14.22 | 14.22 | 4.96 | 8.18 | 4.96 |
| int8-offload | 14.22 | 14.22 | 4.96 | 8.18 | 4.96 |
| nf4 | 14.22 | 14.22 | 4.96 | 8.18 | 4.96 |
| nf4-offload | 14.22 | 14.22 | 4.96 | 8.18 | 4.96 |

## Best query mode per train mode

- `fp4` -> `nf4`: eval 1.0210
- `int8` -> `int8`: eval 1.0171
- `int8-offload` -> `int8`: eval 1.0126
- `nf4` -> `nf4`: eval 1.0208
- `nf4-offload` -> `nf4`: eval 1.0216

## Best train mode per query mode

- `bf16` -> `int8-offload`: eval 1.0147
- `fp16` -> `int8-offload`: eval 1.0153
- `fp4` -> `fp4`: eval 1.0348
- `int8` -> `int8-offload`: eval 1.0126
- `nf4` -> `nf4`: eval 1.0208

## Transfer observations (this run)

- same-mode: fp4 -> fp4: eval 1.0348 (base-no-adapter 1.5041)
- upward: fp4 -> nf4: eval 1.0210 (-0.0138 vs same-mode)
- upward: fp4 -> int8: eval 1.0301 (-0.0048 vs same-mode)
- upward: fp4 -> bf16: eval 1.0314 (-0.0035 vs same-mode)
- upward: fp4 -> fp16: eval 1.0315 (-0.0033 vs same-mode)
- same-mode: int8 -> int8: eval 1.0171 (base-no-adapter 1.4811)
- downward: int8 -> nf4: eval 1.0237 (+0.0066 vs same-mode)
- downward: int8 -> fp4: eval 1.0378 (+0.0207 vs same-mode)
- upward: int8 -> bf16: eval 1.0179 (+0.0008 vs same-mode)
- upward: int8 -> fp16: eval 1.0173 (+0.0002 vs same-mode)
- same-mode: int8-offload -> int8: eval 1.0126 (base-no-adapter 1.4811)
- downward: int8-offload -> nf4: eval 1.0226 (+0.0100 vs same-mode)
- downward: int8-offload -> fp4: eval 1.0360 (+0.0235 vs same-mode)
- upward: int8-offload -> bf16: eval 1.0147 (+0.0021 vs same-mode)
- upward: int8-offload -> fp16: eval 1.0153 (+0.0028 vs same-mode)
- same-mode: nf4 -> nf4: eval 1.0208 (base-no-adapter 1.4905)
- downward: nf4 -> fp4: eval 1.0354 (+0.0146 vs same-mode)
- upward: nf4 -> int8: eval 1.0217 (+0.0009 vs same-mode)
- upward: nf4 -> bf16: eval 1.0271 (+0.0063 vs same-mode)
- upward: nf4 -> fp16: eval 1.0265 (+0.0057 vs same-mode)
- same-mode: nf4-offload -> nf4: eval 1.0216 (base-no-adapter 1.4905)
- downward: nf4-offload -> fp4: eval 1.0361 (+0.0145 vs same-mode)
- upward: nf4-offload -> int8: eval 1.0242 (+0.0026 vs same-mode)
- upward: nf4-offload -> bf16: eval 1.0245 (+0.0029 vs same-mode)
- upward: nf4-offload -> fp16: eval 1.0245 (+0.0029 vs same-mode)
- symmetry: fp4->int8 eval 1.0301 vs int8->fp4 eval 1.0378 (observed in this run)
- symmetry: fp4->nf4 eval 1.0210 vs nf4->fp4 eval 1.0354 (observed in this run)
- symmetry: int8->nf4 eval 1.0237 vs nf4->int8 eval 1.0217 (observed in this run)
