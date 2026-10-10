# P129 Amendment 3 (`tc1-5090-147`): the reduction by main's `tc1_reduce.py` at `19bfbe1`

Machine 27708 (AMD EPYC 7B13, RTX 5090), e4b `51e5ae1`, grouped-nf4-gemm `d1f64ba`, 60 load-gated steps. The rows below are the
reducer's own output, unedited.

## P129 Amendment 3: E4B_TRAIN_FUSE_QKV 0 vs 1 at the field recipe (descriptive)
| arm | VERDICT | s/step (11..N) | device ms / profiled step | busy_t | launches / profiled step | fused modules | peak GB | held-out 0 / N |
|---|---|---|---|---|---|---|---|---|
| e4b/fused_attn4_shipped_q0 | VALID | 3.902 | 1381.0 | 0.354 | 75833 | 0 | 23.321 | 1.95917 / 0.75709 |
| e4b/fused_attn4_shipped_q1 | VALID | 3.364 | 1363.9 | 0.405 | 64761 | 48 | 23.344 | 1.94684 / 0.75825 |
| e4b/fused_attn4_m_q0 | VALID | 3.985 | 1843.5 | 0.463 | 79572 | 0 | 26.495 | 1.95917 / 0.75583 |
| e4b/fused_attn4_m_q1 | VALID | 3.546 | 1819.0 | 0.513 | 68510 | 48 | 26.500 | 1.94684 / 0.75757 |
| e4b/fused_attn4_m_q1_d2 | VALID | 3.555 | 1809.9 | 0.509 | 68516 | 48 | 26.500 | 1.94684 / 0.75444 |
| e4b/fused_attn4_m_q0_d2 | VALID | 3.944 | 1833.9 | 0.465 | 79580 | 0 | 26.495 | 1.95917 / 0.75619 |
| e4b/fused_attn4_shipped_q1_d2 | VALID | 3.385 | 1364.1 | 0.403 | 64761 | 48 | 23.344 | 1.94684 / 0.75648 |
| e4b/fused_attn4_shipped_q0_d2 | VALID | 3.869 | 1384.2 | 0.358 | 75833 | 0 | 23.321 | 1.95917 / 0.75638 |

## P129 Amendment 3's gates and verdict (scored mechanically)
| row | family | verdict | evidence |
|---|---|---|---|
| R_m | qwen3fqkv3 | **HELD** | launches per profiled step 79576 -> 68513 = -13.9 % vs >= 11.2 % (0.8 of Phase 1's 14.0 %) |
| R_shipped | qwen3fqkv3 | **HELD** | launches per profiled step 75833 -> 64761 = -14.6 % vs >= 11.2 % (0.8 of Phase 1's 14.0 %) |
| PREMISE | qwen3fqkv3 | **HELD** | matched q0 busy_t 0.464 vs <= 0.85 |
| W_m | qwen3fqkv3 | **HELD** | q1 / q0 0.895 [0.890, 0.901 over 4 cross-draw ratios] vs <= 0.98; s/step q1 3.546 / 3.555, q0 3.985 / 3.944 |
| W_shipped | qwen3fqkv3 | **HELD** | q1 / q0 0.868 [0.862, 0.875 over 4 cross-draw ratios] vs <= 0.98; s/step q1 3.364 / 3.385, q0 3.902 / 3.869 |
| DEVICE | qwen3fqkv3 | **REPORTED** | m: device 1838.7 -> 1814.4 ms = 0.987, peak 26.495 -> 26.5 GB; shipped: device 1382.6 -> 1364.0 ms = 0.987, peak 23.321 -> 23.344 GB |
| QUALITY | qwen3fqkv3 | **HELD** | m: step 0 e_B 0.02313 vs max(e_A, e_D2, e_D3, e_D4) 0.03905 (e_A 0.03138, e_D2 0.03138, e_D3 0.03523, e_D4 0.03905); N -0.00001; shipped: step 0 e_B 0.02313 vs max(e_A, e_D2, e_D3, e_D4) 0.03905 (e_A 0.03138, e_D2 0.03138, e_D3 0.03523, e_D4 0.03905); N +0.00063 (step 0 e_B <= max(e_A, e_D2, e_D3, e_D4), e_x = mean |row - D1|, N |.| <= 0.005) |
| FQKV | qwen3fqkv3 | **GAIN** | the first rung that applies: VOID / NOISY / QUALITY_FAIL / NO_GAIN / GAIN (DEFAULT_ON needs a second host) |

## Load gate (TC1-PREREG amendment 33): 14 draw(s) voided for host load and run again
Each line: the arm, the attempt, the median host load1 over that attempt's run, the gate. A VOID attempt's files are in loadvoid/; the last attempt of each arm stands whatever its load.

- `qwen3fqkv3/e4b/fused_attn4_shipped_q0 attempt 0 load1_median 17.18 gate 6.0 status ok over 1`
- `qwen3fqkv3/e4b/fused_attn4_shipped_q0 attempt 0 VOID (host load1 median 17.18 > 6.0): re-run 1 of 2`
- `qwen3fqkv3/e4b/fused_attn4_shipped_q0 attempt 1 load1_median 13.58 gate 6.0 status ok over 1`
- `qwen3fqkv3/e4b/fused_attn4_shipped_q0 attempt 1 VOID (host load1 median 13.58 > 6.0): re-run 2 of 2`
- `qwen3fqkv3/e4b/fused_attn4_shipped_q0 attempt 2 load1_median 31.5 gate 6.0 status ok over 1`
- `qwen3fqkv3/e4b/fused_attn4_shipped_q1 attempt 0 load1_median 27.12 gate 6.0 status ok over 1`
- `qwen3fqkv3/e4b/fused_attn4_shipped_q1 attempt 0 VOID (host load1 median 27.12 > 6.0): re-run 1 of 2`
- `qwen3fqkv3/e4b/fused_attn4_shipped_q1 attempt 1 load1_median 28.62 gate 6.0 status ok over 1`
- `qwen3fqkv3/e4b/fused_attn4_shipped_q1 attempt 1 VOID (host load1 median 28.62 > 6.0): re-run 2 of 2`
- `qwen3fqkv3/e4b/fused_attn4_shipped_q1 attempt 2 load1_median 26.65 gate 6.0 status ok over 1`
- `qwen3fqkv3/e4b/fused_attn4_m_q0 attempt 0 load1_median 56.72 gate 6.0 status ok over 1`
- `qwen3fqkv3/e4b/fused_attn4_m_q0 attempt 0 VOID (host load1 median 56.72 > 6.0): re-run 1 of 2`
- `qwen3fqkv3/e4b/fused_attn4_m_q0 attempt 1 load1_median 26.19 gate 6.0 status ok over 1`
- `qwen3fqkv3/e4b/fused_attn4_m_q0 attempt 1 VOID (host load1 median 26.19 > 6.0): re-run 2 of 2`
- `qwen3fqkv3/e4b/fused_attn4_m_q0 attempt 2 load1_median 12.95 gate 6.0 status ok over 1`
- `qwen3fqkv3/e4b/fused_attn4_m_q1 attempt 0 load1_median 23.49 gate 6.0 status ok over 1`
- `qwen3fqkv3/e4b/fused_attn4_m_q1 attempt 0 VOID (host load1 median 23.49 > 6.0): re-run 1 of 2`
- `qwen3fqkv3/e4b/fused_attn4_m_q1 attempt 1 load1_median 23.5 gate 6.0 status ok over 1`
- `qwen3fqkv3/e4b/fused_attn4_m_q1 attempt 1 VOID (host load1 median 23.5 > 6.0): re-run 2 of 2`
- `qwen3fqkv3/e4b/fused_attn4_m_q1 attempt 2 load1_median 30.04 gate 6.0 status ok over 1`
- `qwen3fqkv3/e4b/fused_attn4_m_q1_d2 attempt 0 load1_median 33.03 gate 6.0 status ok over 1`
- `qwen3fqkv3/e4b/fused_attn4_m_q1_d2 attempt 0 VOID (host load1 median 33.03 > 6.0): re-run 1 of 2`
- `qwen3fqkv3/e4b/fused_attn4_m_q1_d2 attempt 1 load1_median 30.72 gate 6.0 status ok over 1`
- `qwen3fqkv3/e4b/fused_attn4_m_q1_d2 attempt 1 VOID (host load1 median 30.72 > 6.0): re-run 2 of 2`
- `qwen3fqkv3/e4b/fused_attn4_m_q1_d2 attempt 2 load1_median 31.98 gate 6.0 status ok over 1`
- `qwen3fqkv3/e4b/fused_attn4_m_q0_d2 attempt 0 load1_median 45.78 gate 6.0 status ok over 1`
- `qwen3fqkv3/e4b/fused_attn4_m_q0_d2 attempt 0 VOID (host load1 median 45.78 > 6.0): re-run 1 of 2`
- `qwen3fqkv3/e4b/fused_attn4_m_q0_d2 attempt 1 load1_median 19.5 gate 6.0 status ok over 1`
- `qwen3fqkv3/e4b/fused_attn4_m_q0_d2 attempt 1 VOID (host load1 median 19.5 > 6.0): re-run 2 of 2`
- `qwen3fqkv3/e4b/fused_attn4_m_q0_d2 attempt 2 load1_median 12.3 gate 6.0 status ok over 1`
- `qwen3fqkv3/e4b/fused_attn4_shipped_q1_d2 attempt 0 load1_median 26.69 gate 6.0 status ok over 1`
- `qwen3fqkv3/e4b/fused_attn4_shipped_q1_d2 attempt 0 VOID (host load1 median 26.69 > 6.0): re-run 1 of 2`
- `qwen3fqkv3/e4b/fused_attn4_shipped_q1_d2 attempt 1 load1_median 27.23 gate 6.0 status ok over 1`
- `qwen3fqkv3/e4b/fused_attn4_shipped_q1_d2 attempt 1 VOID (host load1 median 27.23 > 6.0): re-run 2 of 2`
- `qwen3fqkv3/e4b/fused_attn4_shipped_q1_d2 attempt 2 load1_median 27.86 gate 6.0 status ok over 1`
- `qwen3fqkv3/e4b/fused_attn4_shipped_q0_d2 attempt 0 load1_median 29.41 gate 6.0 status ok over 1`
