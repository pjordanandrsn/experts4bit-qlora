# P129 Amendment 4 (`tc1-5090-150`): the reduction by main's `tc1_reduce.py` at `d7ba80d`

Machine 152440 (AMD EPYC 7K62, RTX 5090), e4b `2d0ed91`, grouped-nf4-gemm `d1f64ba`, 60 load-gated steps. The rows below are the
reducer's own output, unedited.

## P129 Amendment 3: E4B_TRAIN_FUSE_QKV 0 vs 1 at the field recipe (descriptive)
| arm | VERDICT | s/step (11..N) | device ms / profiled step | busy_t | launches / profiled step | fused modules | peak GB | held-out 0 / N |
|---|---|---|---|---|---|---|---|---|
| e4b/fused_attn4_shipped_q0 | VALID | 3.231 | 1332.4 | 0.412 | 75833 | 0 | 23.321 | 1.95917 / 0.7554 |
| e4b/fused_attn4_shipped_q1 | VALID | 2.809 | 1312.6 | 0.467 | 64761 | 48 | 23.344 | 1.94684 / 0.75533 |
| e4b/fused_attn4_m_q0 | VALID | 3.319 | 1791.7 | 0.540 | 79614 | 0 | 26.495 | 1.95917 / 0.75857 |
| e4b/fused_attn4_m_q1 | VALID | 3.012 | 1769.6 | 0.588 | 68524 | 48 | 26.500 | 1.94684 / 0.7602 |
| e4b/fused_attn4_m_q1_d2 | VALID | 2.998 | 1765.6 | 0.589 | 68518 | 48 | 26.500 | 1.94684 / 0.75533 |
| e4b/fused_attn4_m_q0_d2 | VALID | 3.313 | 1786.1 | 0.539 | 79572 | 0 | 26.495 | 1.95917 / 0.75333 |
| e4b/fused_attn4_shipped_q1_d2 | VALID | 2.831 | 1321.8 | 0.467 | 64761 | 48 | 23.344 | 1.94684 / 0.75644 |
| e4b/fused_attn4_shipped_q0_d2 | VALID | 3.153 | 1333.6 | 0.423 | 75833 | 0 | 23.321 | 1.95917 / 0.75494 |

## P129 Amendment 3's gates and verdict (scored mechanically)
| row | family | verdict | evidence |
|---|---|---|---|
| R_m | qwen3fqkv3 | **HELD** | launches per profiled step 79593 -> 68521 = -13.9 % vs >= 11.2 % (0.8 of Phase 1's 14.0 %) |
| R_shipped | qwen3fqkv3 | **HELD** | launches per profiled step 75833 -> 64761 = -14.6 % vs >= 11.2 % (0.8 of Phase 1's 14.0 %) |
| PREMISE | qwen3fqkv3 | **HELD** | matched q0 busy_t 0.539 vs <= 0.85 |
| W_m | qwen3fqkv3 | **HELD** | q1 / q0 0.906 [0.903, 0.909 over 4 cross-draw ratios] vs <= 0.98; s/step q1 3.012 / 2.998, q0 3.319 / 3.313 |
| W_shipped | qwen3fqkv3 | **HELD** | q1 / q0 0.884 [0.870, 0.898 over 4 cross-draw ratios] vs <= 0.98; s/step q1 2.809 / 2.831, q0 3.231 / 3.153 |
| DEVICE | qwen3fqkv3 | **REPORTED** | m: device 1788.9 -> 1767.6 ms = 0.988, peak 26.495 -> 26.5 GB; shipped: device 1333.0 -> 1317.2 ms = 0.988, peak 23.321 -> 23.344 GB |
| QUALITY | qwen3fqkv3 | **HELD** | m: step 0 e_B 0.02313 vs max(e_A, e_D2, e_D3, e_D4) 0.03905 (e_A 0.03138, e_D2 0.03138, e_D3 0.03523, e_D4 0.03905); N +0.00182; shipped: step 0 e_B 0.02313 vs max(e_A, e_D2, e_D3, e_D4) 0.03905 (e_A 0.03138, e_D2 0.03138, e_D3 0.03523, e_D4 0.03905); N +0.00071 (step 0 e_B <= max(e_A, e_D2, e_D3, e_D4), e_x = mean |row - D1|, N |.| <= 0.005) |
| FQKV | qwen3fqkv3 | **GAIN** | the first rung that applies: VOID / NOISY / QUALITY_FAIL / NO_GAIN / GAIN (DEFAULT_ON needs a second host) |

## Load gate (TC1-PREREG amendment 33): 0 draw(s) voided for host load and run again
Each line: the arm, the attempt, the median host load1 over that attempt's run, the gate. A VOID attempt's files are in loadvoid/; the last attempt of each arm stands whatever its load.

- `qwen3fqkv3/e4b/fused_attn4_shipped_q0 attempt 0 load1_median 3.92 gate 6.0 status ok over 0`
- `qwen3fqkv3/e4b/fused_attn4_shipped_q1 attempt 0 load1_median 4.81 gate 6.0 status ok over 0`
- `qwen3fqkv3/e4b/fused_attn4_m_q0 attempt 0 load1_median 4.83 gate 6.0 status ok over 0`
- `qwen3fqkv3/e4b/fused_attn4_m_q1 attempt 0 load1_median 3.84 gate 6.0 status ok over 0`
- `qwen3fqkv3/e4b/fused_attn4_m_q1_d2 attempt 0 load1_median 2.99 gate 6.0 status ok over 0`
- `qwen3fqkv3/e4b/fused_attn4_m_q0_d2 attempt 0 load1_median 2.28 gate 6.0 status ok over 0`
- `qwen3fqkv3/e4b/fused_attn4_shipped_q1_d2 attempt 0 load1_median 2.74 gate 6.0 status ok over 0`
- `qwen3fqkv3/e4b/fused_attn4_shipped_q0_d2 attempt 0 load1_median 2.73 gate 6.0 status ok over 0`
