# P129 Amendment 2, box 1 (`tc1-5090-146`): the reduction by main's `tc1_reduce.py` at `d14bcb1`

Machine 152440 (AMD EPYC 7K62, RTX 5090), e4b `be7a88d`, grouped-nf4-gemm `d1f64ba`, 60 load-gated steps. The rows below are the
reducer's own output, unedited.

## P129 Amendment 2: E4B_TRAIN_FUSE_QKV 0 vs 1 at the field recipe (descriptive)
| arm | VERDICT | s/step (11..N) | device ms / profiled step | busy_t | launches / profiled step | fused modules | peak GB | held-out 0 / N |
|---|---|---|---|---|---|---|---|---|
| e4b/fused_attn4_shipped_q0 | VALID | 3.193 | 1282.7 | 0.402 | 75833 | 0 | 23.321 | 1.95917 / 0.75512 |
| e4b/fused_attn4_shipped_q1 | VALID | 2.842 | 1264.4 | 0.445 | 64761 | 48 | 23.344 | 1.94684 / 0.7572 |
| e4b/fused_attn4_m_q0 | VALID | 3.302 | 1731.6 | 0.524 | 79582 | 0 | 26.495 | 1.95917 / 0.75805 |
| e4b/fused_attn4_m_q1 | VALID | 2.939 | 1712.6 | 0.583 | 68524 | 48 | 26.500 | 1.94684 / 0.75478 |
| e4b/fused_attn4_m_q1_d2 | VALID | 3.003 | 1709.7 | 0.569 | 68510 | 48 | 26.500 | 1.94684 / 0.75722 |
| e4b/fused_attn4_m_q0_d2 | VALID | 3.332 | 1737.4 | 0.522 | 79580 | 0 | 26.495 | 1.95917 / 0.75549 |
| e4b/fused_attn4_shipped_q1_d2 | VALID | 2.801 | 1262.5 | 0.451 | 64761 | 48 | 23.344 | 1.94684 / 0.7538 |
| e4b/fused_attn4_shipped_q0_d2 | VALID | 3.227 | 1281.5 | 0.397 | 75833 | 0 | 23.321 | 1.95917 / 0.75895 |

## P129 Amendment 2's gates and verdict (scored mechanically)
| row | family | verdict | evidence |
|---|---|---|---|
| R_m | qwen3fqkv | **HELD** | launches per profiled step 79581 -> 68517 = -13.9 % vs >= 11.2 % (0.8 of Phase 1's 14.0 %) |
| R_shipped | qwen3fqkv | **HELD** | launches per profiled step 75833 -> 64761 = -14.6 % vs >= 11.2 % (0.8 of Phase 1's 14.0 %) |
| PREMISE | qwen3fqkv | **HELD** | matched q0 busy_t 0.523 vs <= 0.85 |
| W_m | qwen3fqkv | **HELD** | q1 / q0 0.896 [0.882, 0.909 over 4 cross-draw ratios] vs <= 0.98; s/step q1 2.939 / 3.003, q0 3.302 / 3.332 |
| W_shipped | qwen3fqkv | **HELD** | q1 / q0 0.879 [0.868, 0.890 over 4 cross-draw ratios] vs <= 0.98; s/step q1 2.842 / 2.801, q0 3.193 / 3.227 |
| DEVICE | qwen3fqkv | **REPORTED** | m: device 1734.5 -> 1711.2 ms = 0.987, peak 26.495 -> 26.5 GB; shipped: device 1282.1 -> 1263.4 ms = 0.985, peak 23.321 -> 23.344 GB |
| QUALITY | qwen3fqkv | **FALSIFIED** | m: step 0 -0.01233, -0.01233; N -0.00077; shipped: step 0 -0.01233, -0.01233; N -0.00153 (step 0 |.| <= 0.0005, N |.| <= 0.005) |
| FQKV | qwen3fqkv | **QUALITY_FAIL** | the first rung that applies: VOID / NOISY / QUALITY_FAIL / NO_GAIN / GAIN (DEFAULT_ON needs a second host) |

## Load gate (TC1-PREREG amendment 33): 6 draw(s) voided for host load and run again
Each line: the arm, the attempt, the median host load1 over that attempt's run, the gate. A VOID attempt's files are in loadvoid/; the last attempt of each arm stands whatever its load.

- `qwen3fqkv/e4b/fused_attn4_shipped_q0 attempt 0 load1_median 7.92 gate 6.0 status ok over 1`
- `qwen3fqkv/e4b/fused_attn4_shipped_q0 attempt 0 VOID (host load1 median 7.92 > 6.0): re-run 1 of 2`
- `qwen3fqkv/e4b/fused_attn4_shipped_q0 attempt 1 load1_median 6.48 gate 6.0 status ok over 1`
- `qwen3fqkv/e4b/fused_attn4_shipped_q0 attempt 1 VOID (host load1 median 6.48 > 6.0): re-run 2 of 2`
- `qwen3fqkv/e4b/fused_attn4_shipped_q0 attempt 2 load1_median 12.21 gate 6.0 status ok over 1`
- `qwen3fqkv/e4b/fused_attn4_shipped_q1 attempt 0 load1_median 8.81 gate 6.0 status ok over 1`
- `qwen3fqkv/e4b/fused_attn4_shipped_q1 attempt 0 VOID (host load1 median 8.81 > 6.0): re-run 1 of 2`
- `qwen3fqkv/e4b/fused_attn4_shipped_q1 attempt 1 load1_median 9.1 gate 6.0 status ok over 1`
- `qwen3fqkv/e4b/fused_attn4_shipped_q1 attempt 1 VOID (host load1 median 9.1 > 6.0): re-run 2 of 2`
- `qwen3fqkv/e4b/fused_attn4_shipped_q1 attempt 2 load1_median 8.96 gate 6.0 status ok over 1`
- `qwen3fqkv/e4b/fused_attn4_m_q0 attempt 0 load1_median 7.5 gate 6.0 status ok over 1`
- `qwen3fqkv/e4b/fused_attn4_m_q0 attempt 0 VOID (host load1 median 7.5 > 6.0): re-run 1 of 2`
- `qwen3fqkv/e4b/fused_attn4_m_q0 attempt 1 load1_median 8.14 gate 6.0 status ok over 1`
- `qwen3fqkv/e4b/fused_attn4_m_q0 attempt 1 VOID (host load1 median 8.14 > 6.0): re-run 2 of 2`
- `qwen3fqkv/e4b/fused_attn4_m_q0 attempt 2 load1_median 6.63 gate 6.0 status ok over 1`
- `qwen3fqkv/e4b/fused_attn4_m_q1 attempt 0 load1_median 5.51 gate 6.0 status ok over 0`
- `qwen3fqkv/e4b/fused_attn4_m_q1_d2 attempt 0 load1_median 3.71 gate 6.0 status ok over 0`
- `qwen3fqkv/e4b/fused_attn4_m_q0_d2 attempt 0 load1_median 4.43 gate 6.0 status ok over 0`
- `qwen3fqkv/e4b/fused_attn4_shipped_q1_d2 attempt 0 load1_median 4.03 gate 6.0 status ok over 0`
- `qwen3fqkv/e4b/fused_attn4_shipped_q0_d2 attempt 0 load1_median 3.98 gate 6.0 status ok over 0`
