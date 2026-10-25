# TC1 amendment 81: the device-time re-derivation

26 runs; reproduced 26; refused 0; VERDICT FLIPS (incl. unpaired verdict lines): 0; REWORDED lines: 2.
Annotation rows removed, by name and family, over all corrected cells: {'Optimizer.step# in optimizer': 119}.

## tc1-5090-110 (bench/h2h-2026-10-02/tc1/receipts/tc1-5090-110): REPRODUCED, 20 changed lines, 0 flips, 0 unpaired verdict lines, annotation share 0.0054
- cell qwen3prof28_e4b_fused_attn4_m_q212.json: device ms (all profiled steps) 28848.2 -> 28693.2 (annotation 155.0 in optimizer)
- cell qwen3prof28_e4b_fused_attn4_m_q212_d2.json: device ms (all profiled steps) 28792.0 -> 28641.1 (annotation 150.9 in optimizer)
- cell qwen3prof28_e4b_fused_attn4_m_q28.json: device ms (all profiled steps) 32008.1 -> 31861.8 (annotation 146.3 in optimizer)
- cell qwen3prof28_e4b_fused_attn4_m_q28_d2.json: device ms (all profiled steps) 31996.0 -> 31850.9 (annotation 145.1 in optimizer)
- cell qwen3prof28_e4b_fused_attn4_m_q28k0.json: device ms (all profiled steps) 35758.5 -> 35618.5 (annotation 140.0 in optimizer)
- cell qwen3prof28_e4b_fused_attn4_m_q28k0_d2.json: device ms (all profiled steps) 35781.0 -> 35640.3 (annotation 140.7 in optimizer)
- **REWORDED** (a branch may have moved) `- device ms per profiled step by family, q28 - q212 (median of two draws, largest increase first): fused_kernel +988.5, matmul +67.8, other +29.0, autograd +0.6, routing -0.6, optimizer -2.4, memcpy -` -> `- device ms per profiled step by family, q28 - q212 (median of two draws, largest increase first): fused_kernel +988.5, matmul +67.8, other +29.0, autograd +0.6, optimizer +0.0, routing -0.6, memcpy -`
- **REWORDED** (a branch may have moved) `- device ms per profiled step by family, q28 - q28k0 (median of two draws, largest increase first): optimizer +2.0, autograd +0.7, norm_act -1.0, routing -1.3, fused_kernel -10.5, memcpy -13.6, other ` -> `- device ms per profiled step by family, q28 - q28k0 (median of two draws, largest increase first): autograd +0.7, optimizer +0.2, norm_act -1.0, routing -1.3, fused_kernel -10.5, memcpy -13.6, other `
```diff
-| fused_attn4_m_q212 | VALID | 9.924 | True | 10648.0 | 9616.1 | 0.9031 | 137438 | 963645 |
-| fused_attn4_m_q28 | VALID | 12.562 | True | 14733.4 | 10669.4 | 0.7242 | 137656 | 962394 |
-| fused_attn4_m_q28k0 | VALID | 12.337 | True | 12830.4 | 11919.5 | 0.929 | 111643 | 697181 |
-| fused_attn4_m_q28k0_d2 | VALID | 12.375 | True | 12858.4 | 11927.0 | 0.9276 | 112823 | 703776 |
-| fused_attn4_m_q28_d2 | VALID | 12.535 | True | 14772.3 | 10665.3 | 0.722 | 137777 | 963712 |
-| fused_attn4_m_q212_d2 | VALID | 9.914 | True | 10611.9 | 9597.3 | 0.9044 | 137438 | 963761 |
+| fused_attn4_m_q212 | VALID | 9.924 | True | 10648.0 | 9564.4 | 0.8982 | 137438 | 963645 |
+| fused_attn4_m_q28 | VALID | 12.562 | True | 14733.4 | 10620.6 | 0.7209 | 137656 | 962394 |
+| fused_attn4_m_q28k0 | VALID | 12.337 | True | 12830.4 | 11872.8 | 0.9254 | 111643 | 697181 |
+| fused_attn4_m_q28k0_d2 | VALID | 12.375 | True | 12858.4 | 11880.1 | 0.9239 | 112823 | 703776 |
+| fused_attn4_m_q28_d2 | VALID | 12.535 | True | 14772.3 | 10617.0 | 0.7187 | 137777 | 963712 |
+| fused_attn4_m_q212_d2 | VALID | 9.914 | True | 10611.9 | 9547.0 | 0.8997 | 137438 | 963761 |
-- device ms per profiled step by family, q28 - q212 (median of two draws, largest increase first): fused_kernel +988.5, matmul +67.8, other +29.0, autograd +0.6, routing -0.6, optimizer -2.4, memcpy -10.4, norm_act -11.2
-- device ms per profiled step by family, q28 - q28k0 (median of two draws, largest increase first): optimizer +2.0, autograd +0.7, norm_act -1.0, routing -1.3, fused_kernel -10.5, memcpy -13.6, other -20.2, matmul -1210.9
+- device ms per profiled step by family, q28 - q212 (median of two draws, largest increase first): fused_kernel +988.5, matmul +67.8, other +29.0, autograd +0.6, optimizer +0.0, routing -0.6, memcpy -10.4, norm_act -11.2
+- device ms per profiled step by family, q28 - q28k0 (median of two draws, largest increase first): autograd +0.7, optimizer +0.2, norm_act -1.0, routing -1.3, fused_kernel -10.5, memcpy -13.6, other -20.2, matmul -1210.9
-| P135 | qwen3prof28 | **HELD** | timed ms/step q28 12548.5 vs q212 9918.9 (D +2629.6); device/step 10667.3 vs 9606.7 ms (+1060.6); share not device time 0.597 vs >= 0.5; profiled wall/step 14752.8 vs 10630.0 (gap +4122.9, reported); largest CPU-self increases (ms/step): matmul +4816.5, fused_kernel +21.8, optimizer +1.4 |
-| P136 | qwen3prof28 | **HELD** | device busy fraction (device / timed step) q28 0.850 vs q28k0 0.965 (drop +0.115 vs >= 0.03); timed ms/step 12548.5 vs 12356.1, device/step 10667.3 vs 11923.2 ms; against the profiled wall (reported) 0.723 vs 0.928 |
+| P135 | qwen3prof28 | **HELD** | timed ms/step q28 12548.5 vs q212 9918.9 (D +2629.6); device/step 10618.8 vs 9555.7 ms (+1063.1); share not device time 0.596 vs >= 0.5; profiled wall/step 14752.8 vs 10630.0 (gap +4122.9, reported); largest CPU-self increases (ms/step): matmul +4816.5, fused_kernel +21.8, optimizer +1.4 |
+| P136 | qwen3prof28 | **HELD** | device busy fraction (device / timed step) q28 0.846 vs q28k0 0.961 (drop +0.115 vs >= 0.03); timed ms/step 12548.5 vs 12356.1, device/step 10618.8 vs 11876.5 ms; against the profiled wall (reported) 0.720 vs 0.925 |
```

## tc1-5090-111 (bench/h2h-2026-10-02/tc1/receipts/tc1-5090-111): REPRODUCED, 12 changed lines, 0 flips, 0 unpaired verdict lines, annotation share 0.003
- cell qwen3ladder28_e4b_fused_attn4_m_c0.json: device ms (all profiled steps) 31206.9 -> 31121.1 (annotation 85.8 in optimizer)
- cell qwen3ladder28_e4b_fused_attn4_m_c0_d2.json: device ms (all profiled steps) 31229.5 -> 31137.3 (annotation 92.2 in optimizer)
- cell qwen3ladder28_e4b_fused_attn4_m_c1.json: device ms (all profiled steps) 31209.6 -> 31124.2 (annotation 85.4 in optimizer)
- cell qwen3ladder28_e4b_fused_attn4_m_c1_d2.json: device ms (all profiled steps) 31354.0 -> 31267.1 (annotation 86.9 in optimizer)
- cell qwen3ladder28_e4b_fused_attn4_m_c2.json: device ms (all profiled steps) 31747.5 -> 31662.9 (annotation 84.6 in optimizer)
- cell qwen3ladder28_e4b_fused_attn4_m_c2_d2.json: device ms (all profiled steps) 31936.4 -> 31851.7 (annotation 84.7 in optimizer)
```diff
-| fused_attn4_m_c0 | VALID | 10.507 | 28.18 | 10402.3 |
-| fused_attn4_m_c1 | VALID | 10.477 | 28.18 | 10403.2 |
-| fused_attn4_m_c2 | VALID | 10.584 | 28.50 | 10582.5 |
-| fused_attn4_m_c2_d2 | VALID | 10.632 | 28.53 | 10645.5 |
-| fused_attn4_m_c1_d2 | VALID | 10.446 | 28.18 | 10451.3 |
-| fused_attn4_m_c0_d2 | VALID | 10.492 | 28.18 | 10409.8 |
+| fused_attn4_m_c0 | VALID | 10.507 | 28.18 | 10373.7 |
+| fused_attn4_m_c1 | VALID | 10.477 | 28.18 | 10374.7 |
+| fused_attn4_m_c2 | VALID | 10.584 | 28.50 | 10554.3 |
+| fused_attn4_m_c2_d2 | VALID | 10.632 | 28.53 | 10617.2 |
+| fused_attn4_m_c1_d2 | VALID | 10.446 | 28.18 | 10422.4 |
+| fused_attn4_m_c0_d2 | VALID | 10.492 | 28.18 | 10379.1 |
```

## tc1-5090-125 (bench/h2h-2026-10-02/tc1/receipts/tc1-5090-125): REPRODUCED, 20 changed lines, 0 flips, 0 unpaired verdict lines, annotation share 0.0168
- cell qwen3ckptoff28_e4b_fused_attn4_m_g0.json: device ms (all profiled steps) 5745.4 -> 5664.5 (annotation 80.9 in optimizer)
- cell qwen3ckptoff28_e4b_fused_attn4_m_g0_d2.json: device ms (all profiled steps) 5755.9 -> 5674.9 (annotation 81.0 in optimizer)
- cell qwen3ckptoff28_e4b_fused_attn4_m_g1.json: device ms (all profiled steps) 5814.0 -> 5733.5 (annotation 80.5 in optimizer)
- cell qwen3ckptoff28_e4b_fused_attn4_m_g1_d2.json: device ms (all profiled steps) 5820.2 -> 5738.1 (annotation 82.1 in optimizer)
- cell qwen3ckptoff28_e4b_fused_attn4_shipped_g0.json: device ms (all profiled steps) 4636.8 -> 4558.9 (annotation 77.9 in optimizer)
- cell qwen3ckptoff28_e4b_fused_attn4_shipped_g0_d2.json: device ms (all profiled steps) 4647.9 -> 4569.9 (annotation 78.0 in optimizer)
- cell qwen3ckptoff28_e4b_fused_attn4_shipped_g1.json: device ms (all profiled steps) 4708.1 -> 4629.7 (annotation 78.4 in optimizer)
- cell qwen3ckptoff28_e4b_fused_attn4_shipped_g1_d2.json: device ms (all profiled steps) 4718.3 -> 4639.5 (annotation 78.8 in optimizer)
- NEAR BAR (margin below the share 0.0168): 0.994 <= 1.01 (margin 0.0158) :: `| P174 | qwen3ckptoff28 | **HELD** | m: g1 / g0 0.994 [0.993, 0.995 over 4 cross-draw ratios] vs <= 1.01; s/step g0 2.474 / 2.470, g1 2.457 / 2.458; g0 device b`
```diff
-| e4b/fused_attn4_shipped_g0 | VALID | 1.947 | 23.271 | 23.271 | 0.75431 | 0.794 |
-| e4b/fused_attn4_shipped_g1 | VALID | 1.916 | 23.054 | 23.054 | 0.75568 | 0.819 |
-| e4b/fused_attn4_m_g0 | VALID | 2.474 | 26.106 | 26.106 | 0.75744 | 0.774 |
-| e4b/fused_attn4_m_g1 | VALID | 2.457 | 25.918 | 25.918 | 0.75532 | 0.789 |
-| e4b/fused_attn4_m_g1_d2 | VALID | 2.458 | 25.928 | 25.928 | 0.75633 | 0.789 |
-| e4b/fused_attn4_m_g0_d2 | VALID | 2.470 | 26.102 | 26.102 | 0.7563 | 0.777 |
-| e4b/fused_attn4_shipped_g1_d2 | VALID | 1.919 | 23.054 | 23.054 | 0.75523 | 0.819 |
-| e4b/fused_attn4_shipped_g0_d2 | VALID | 1.945 | 23.271 | 23.271 | 0.7557 | 0.796 |
+| e4b/fused_attn4_shipped_g0 | VALID | 1.947 | 23.271 | 23.271 | 0.75431 | 0.780 |
+| e4b/fused_attn4_shipped_g1 | VALID | 1.916 | 23.054 | 23.054 | 0.75568 | 0.805 |
+| e4b/fused_attn4_m_g0 | VALID | 2.474 | 26.106 | 26.106 | 0.75744 | 0.763 |
+| e4b/fused_attn4_m_g1 | VALID | 2.457 | 25.918 | 25.918 | 0.75532 | 0.778 |
+| e4b/fused_attn4_m_g1_d2 | VALID | 2.458 | 25.928 | 25.928 | 0.75633 | 0.778 |
+| e4b/fused_attn4_m_g0_d2 | VALID | 2.470 | 26.102 | 26.102 | 0.7563 | 0.766 |
+| e4b/fused_attn4_shipped_g1_d2 | VALID | 1.919 | 23.054 | 23.054 | 0.75523 | 0.806 |
+| e4b/fused_attn4_shipped_g0_d2 | VALID | 1.945 | 23.271 | 23.271 | 0.7557 | 0.783 |
-| P174 | qwen3ckptoff28 | **HELD** | m: g1 / g0 0.994 [0.993, 0.995 over 4 cross-draw ratios] vs <= 1.01; s/step g0 2.474 / 2.470, g1 2.457 / 2.458; g0 device busy vs the timed step 0.775 (premise <= 0.9) |
-| P175 | qwen3ckptoff28 | **HELD** | shipped: g1 / g0 0.985 [0.984, 0.987 over 4 cross-draw ratios] vs <= 1.01; s/step g0 1.947 / 1.945, g1 1.916 / 1.919; g0 device busy vs the timed step 0.795 (premise <= 0.9) |
+| P174 | qwen3ckptoff28 | **HELD** | m: g1 / g0 0.994 [0.993, 0.995 over 4 cross-draw ratios] vs <= 1.01; s/step g0 2.474 / 2.470, g1 2.457 / 2.458; g0 device busy vs the timed step 0.764 (premise <= 0.9) |
+| P175 | qwen3ckptoff28 | **HELD** | shipped: g1 / g0 0.985 [0.984, 0.987 over 4 cross-draw ratios] vs <= 1.01; s/step g0 1.947 / 1.945, g1 1.916 / 1.919; g0 device busy vs the timed step 0.782 (premise <= 0.9) |
```

## tc1-5090-128 (bench/h2h-2026-10-02/tc1/receipts/tc1-5090-128): REPRODUCED, 2 changed lines, 0 flips, 0 unpaired verdict lines, annotation share 0.037
- cell qwen3ckptre28_e4b_fused_attn4_shipped_r0.json: device ms (all profiled steps) 4663.4 -> 4490.8 (annotation 172.6 in optimizer)
- cell qwen3ckptre28_e4b_fused_attn4_shipped_r0_d2.json: device ms (all profiled steps) 4677.5 -> 4505.7 (annotation 171.8 in optimizer)
- cell qwen3ckptre28_e4b_fused_attn4_shipped_r1.json: device ms (all profiled steps) 4813.1 -> 4641.5 (annotation 171.6 in optimizer)
- cell qwen3ckptre28_e4b_fused_attn4_shipped_r1_d2.json: device ms (all profiled steps) 4818.4 -> 4645.1 (annotation 173.3 in optimizer)
- cell qwen3ckptre28_e4b_fused_attn4_shipped_rr.json: device ms (all profiled steps) 4669.5 -> 4498.4 (annotation 171.1 in optimizer)
- cell qwen3ckptre28_e4b_fused_attn4_shipped_rr_d2.json: device ms (all profiled steps) 4675.0 -> 4503.8 (annotation 171.2 in optimizer)
```diff
-| P182 | qwen3ckptre28 | **HELD** | rr / r0 0.838 [0.825, 0.852 over 4 cross-draw ratios] vs <= 1.0; s/step rr 3.247 / 3.215, r0 3.811 / 3.899; r0 device busy vs the timed step 0.404 (premise <= 0.9) |
+| P182 | qwen3ckptre28 | **HELD** | rr / r0 0.838 [0.825, 0.852 over 4 cross-draw ratios] vs <= 1.0; s/step rr 3.247 / 3.215, r0 3.811 / 3.899; r0 device busy vs the timed step 0.389 (premise <= 0.9) |
```

## tc1-5090-133 (bench/h2h-2026-10-02/tc1/receipts/tc1-5090-133): REPRODUCED, 24 changed lines, 0 flips, 0 unpaired verdict lines, annotation share 0.0034
- cell qwen3cbk_e4b_fused_attn4_m_k0.json: device ms (all profiled steps) 27387.5 -> 27307.4 (annotation 80.1 in optimizer)
- cell qwen3cbk_e4b_fused_attn4_m_k0_d2.json: device ms (all profiled steps) 27476.4 -> 27397.4 (annotation 79.0 in optimizer)
- cell qwen3cbk_e4b_fused_attn4_m_k1.json: device ms (all profiled steps) 26677.5 -> 26598.8 (annotation 78.7 in optimizer)
- cell qwen3cbk_e4b_fused_attn4_m_k1_d2.json: device ms (all profiled steps) 26704.3 -> 26625.0 (annotation 79.3 in optimizer)
- cell qwen3cbk_e4b_fused_attn4_shipped_k0.json: device ms (all profiled steps) 21565.5 -> 21493.0 (annotation 72.5 in optimizer)
- cell qwen3cbk_e4b_fused_attn4_shipped_k0_d2.json: device ms (all profiled steps) 21609.9 -> 21538.7 (annotation 71.2 in optimizer)
- cell qwen3cbk_e4b_fused_attn4_shipped_k1.json: device ms (all profiled steps) 21110.5 -> 21038.4 (annotation 72.1 in optimizer)
- cell qwen3cbk_e4b_fused_attn4_shipped_k1_d2.json: device ms (all profiled steps) 21119.3 -> 21046.7 (annotation 72.6 in optimizer)
- cell qwen3cbk_unsloth_ckpt_unsloth_m_kk.json: device ms (all profiled steps) 30456.6 -> 30375.6 (annotation 81.0 in optimizer)
```diff
-| e4b/fused_attn4_shipped_k0 | VALID | 7.320 | 23.003 | 23.003 | 21.245 | 1.28851 | 0.94519 | 0 | 7188.5 |
-| e4b/fused_attn4_shipped_k1 | VALID | 7.146 | 22.664 | 22.664 | 21.245 | 1.28851 | 0.94514 | 31824 | 7036.8 |
-| e4b/fused_attn4_m_k0 | VALID | 9.230 | 26.561 | 26.561 | 22.504 | 1.28851 | 0.95412 | 0 | 9129.2 |
-| e4b/fused_attn4_m_k1 | VALID | 8.972 | 25.906 | 25.906 | 22.504 | 1.28851 | 0.95426 | 31764 | 8892.5 |
-| e4b/fused_attn4_m_k1_d2 | VALID | 8.970 | 25.911 | 25.911 | 22.504 | 1.28851 | 0.95383 | 31766 | 8901.4 |
-| e4b/fused_attn4_m_k0_d2 | VALID | 9.229 | 26.564 | 26.564 | 22.504 | 1.28851 | 0.95424 | 0 | 9158.8 |
-| e4b/fused_attn4_shipped_k1_d2 | VALID | 7.149 | 22.660 | 22.66 | 21.245 | 1.28851 | 0.94516 | 31828 | 7039.8 |
-| e4b/fused_attn4_shipped_k0_d2 | VALID | 7.313 | 23.014 | 23.014 | 21.245 | 1.28851 | 0.94514 | 0 | 7203.3 |
-| unsloth/ckpt_unsloth_m_kk | VALID | 11.542 | 24.864 | 24.864 | 21.852 | 1.28707 | 0.95449 | None | 10152.2 |
+| e4b/fused_attn4_shipped_k0 | VALID | 7.320 | 23.003 | 23.003 | 21.245 | 1.28851 | 0.94519 | 0 | 7164.3 |
+| e4b/fused_attn4_shipped_k1 | VALID | 7.146 | 22.664 | 22.664 | 21.245 | 1.28851 | 0.94514 | 31824 | 7012.8 |
+| e4b/fused_attn4_m_k0 | VALID | 9.230 | 26.561 | 26.561 | 22.504 | 1.28851 | 0.95412 | 0 | 9102.5 |
+| e4b/fused_attn4_m_k1 | VALID | 8.972 | 25.906 | 25.906 | 22.504 | 1.28851 | 0.95426 | 31764 | 8866.3 |
+| e4b/fused_attn4_m_k1_d2 | VALID | 8.970 | 25.911 | 25.911 | 22.504 | 1.28851 | 0.95383 | 31766 | 8875.0 |
+| e4b/fused_attn4_m_k0_d2 | VALID | 9.229 | 26.564 | 26.564 | 22.504 | 1.28851 | 0.95424 | 0 | 9132.5 |
+| e4b/fused_attn4_shipped_k1_d2 | VALID | 7.149 | 22.660 | 22.66 | 21.245 | 1.28851 | 0.94516 | 31828 | 7015.6 |
+| e4b/fused_attn4_shipped_k0_d2 | VALID | 7.313 | 23.014 | 23.014 | 21.245 | 1.28851 | 0.94514 | 0 | 7179.6 |
+| unsloth/ckpt_unsloth_m_kk | VALID | 11.542 | 24.864 | 24.864 | 21.852 | 1.28707 | 0.95449 | None | 10125.2 |
-| P190 | qwen3cbk | **HELD** | k1 / k0 0.972 [0.972, 0.972 over 4 cross-draw ratios] vs <= 1.01; s/step k1 8.972 / 8.970, k0 9.230 / 9.229; k0 device busy vs the timed step 0.991 (descriptive) |
-| P191 | qwen3cbk | **HELD** | k1 / k0 0.977 [0.976, 0.978 over 4 cross-draw ratios] vs <= 1.01; s/step k1 7.146 / 7.149, k0 7.320 / 7.313; k0 device busy vs the timed step 0.984 (descriptive) |
+| P190 | qwen3cbk | **HELD** | k1 / k0 0.972 [0.972, 0.972 over 4 cross-draw ratios] vs <= 1.01; s/step k1 8.972 / 8.970, k0 9.230 / 9.229; k0 device busy vs the timed step 0.988 (descriptive) |
+| P191 | qwen3cbk | **HELD** | k1 / k0 0.977 [0.976, 0.978 over 4 cross-draw ratios] vs <= 1.01; s/step k1 7.146 / 7.149, k0 7.320 / 7.313; k0 device busy vs the timed step 0.980 (descriptive) |
-| P194 | qwen3cbk | **HELD** | m: k1 / k0 0.973 (device ms per profiled step k0 9129.2 / 9158.8, k1 8892.5 / 8901.4); shipped: k1 / k0 0.978 (device ms per profiled step k0 7188.5 / 7203.3, k1 7036.8 / 7039.8) (each arm <= 1.01) |
+| P194 | qwen3cbk | **HELD** | m: k1 / k0 0.973 (device ms per profiled step k0 9102.5 / 9132.5, k1 8866.3 / 8875.0); shipped: k1 / k0 0.978 (device ms per profiled step k0 7164.3 / 7179.6, k1 7012.8 / 7015.6) (each arm <= 1.01) |
```

## tc1-5090-136 (bench/h2h-2026-10-02/tc1/receipts/tc1-5090-136): REPRODUCED, 14 changed lines, 0 flips, 0 unpaired verdict lines, annotation share 0.0036
- cell qwen3pos68_e4b_fused_attn4_m_pd.json: device ms (all profiled steps) 30086.0 -> 29977.5 (annotation 108.5 in optimizer)
- cell qwen3pos68_e4b_fused_attn4_m_pd_d2.json: device ms (all profiled steps) 30171.0 -> 30062.8 (annotation 108.2 in optimizer)
- cell qwen3pos68_unsloth_ckpt_unsloth_m_pv.json: device ms (all profiled steps) 34949.6 -> 34842.2 (annotation 107.4 in optimizer)
- cell qwen3pos68_unsloth_ckpt_unsloth_m_pv_d2.json: device ms (all profiled steps) 34946.8 -> 34837.1 (annotation 109.7 in optimizer)
```diff
-| e4b/fused_attn4_m_pd | VALID | 10.108 | 10028.7 | 0.992 | 1075993 | 159754 | 25.898 | 0.95446 |
-| unsloth/ckpt_unsloth_m_pv | VALID | 14.891 | 11649.9 | 0.782 | 7659165 | 759464 | 24.864 | 0.95426 |
-| unsloth/ckpt_unsloth_m_pv_d2 | VALID | 14.880 | 11648.9 | 0.783 | 7659024 | 759406 | 24.864 | 0.95418 |
-| e4b/fused_attn4_m_pd_d2 | VALID | 10.114 | 10057.0 | 0.994 | 1083338 | 161055 | 25.919 | 0.95421 |
+| e4b/fused_attn4_m_pd | VALID | 10.108 | 9992.5 | 0.989 | 1075993 | 159754 | 25.898 | 0.95446 |
+| unsloth/ckpt_unsloth_m_pv | VALID | 14.891 | 11614.1 | 0.780 | 7659165 | 759464 | 24.864 | 0.95426 |
+| unsloth/ckpt_unsloth_m_pv_d2 | VALID | 14.880 | 11612.4 | 0.780 | 7659024 | 759406 | 24.864 | 0.95418 |
+| e4b/fused_attn4_m_pd_d2 | VALID | 10.114 | 10020.9 | 0.991 | 1083338 | 161055 | 25.919 | 0.95421 |
-| P200 | qwen3pos68 | **HELD** | device ms per profiled step Unsloth 11649.4 / e4b 10042.8 = 1.160 vs in [1.06, 1.24]; s/step Unsloth / e4b 1.472 on this host (descriptive) |
-| P201 | qwen3pos68 | **HELD** | busy_t e4b 0.993, Unsloth 0.783 (gap +0.211 vs >= 0.08) |
+| P200 | qwen3pos68 | **HELD** | device ms per profiled step Unsloth 11613.2 / e4b 10006.7 = 1.161 vs in [1.06, 1.24]; s/step Unsloth / e4b 1.472 on this host (descriptive) |
+| P201 | qwen3pos68 | **HELD** | busy_t e4b 0.990, Unsloth 0.780 (gap +0.210 vs >= 0.08) |
-| P203 | qwen3pos68 | **FALSIFIED** | Unsloth s/step 14.886 >= 14.0: busy_t 0.783 vs <= 0.75; device ms per profiled step 11649.4 vs 10152.2 (+14.7% vs |.| <= 8%) |
+| P203 | qwen3pos68 | **FALSIFIED** | Unsloth s/step 14.886 >= 14.0: busy_t 0.780 vs <= 0.75; device ms per profiled step 11613.2 vs 10152.2 (+14.4% vs |.| <= 8%) |
```

## tc1-5090-137 (bench/h2h-2026-10-02/tc1/receipts/tc1-5090-137): REPRODUCED, 12 changed lines, 0 flips, 0 unpaired verdict lines, annotation share 0.0327
- cell qwen3pos69_e4b_fused_attn4_m_fd.json: device ms (all profiled steps) 5076.8 -> 4919.0 (annotation 157.8 in optimizer)
- cell qwen3pos69_e4b_fused_attn4_m_fd_d2.json: device ms (all profiled steps) 5086.8 -> 4920.3 (annotation 166.5 in optimizer)
- cell qwen3pos69_unsloth_ckpt_unsloth_m_fv.json: device ms (all profiled steps) 9783.8 -> 9605.8 (annotation 178.0 in optimizer)
- cell qwen3pos69_unsloth_ckpt_unsloth_m_fv_d2.json: device ms (all profiled steps) 9770.9 -> 9613.3 (annotation 157.6 in optimizer)
- UNPARSED prediction line, read by hand: `| P208 | qwen3pos69 | **HELD** | s/step ratio 2.803 over device ratio 1.953 = 1.435 (busy_t e4b 0.470, Unsloth 0.328) vs >= 1.05 |`
```diff
-| e4b/fused_attn4_m_fd | VALID | 3.430 | 1692.3 | 0.493 | 527263 | 83691 | 26.156 | 0.75689 |
-| unsloth/ckpt_unsloth_m_fv | VALID | 9.833 | 3261.3 | 0.332 | 7430405 | 611890 | 24.273 | 0.75825 |
-| unsloth/ckpt_unsloth_m_fv_d2 | VALID | 9.727 | 3257.0 | 0.335 | 7430785 | 611956 | 24.273 | 0.75697 |
-| e4b/fused_attn4_m_fd_d2 | VALID | 3.548 | 1695.6 | 0.478 | 527202 | 83690 | 26.165 | 0.75598 |
+| e4b/fused_attn4_m_fd | VALID | 3.430 | 1639.7 | 0.478 | 527263 | 83691 | 26.156 | 0.75689 |
+| unsloth/ckpt_unsloth_m_fv | VALID | 9.833 | 3201.9 | 0.326 | 7430405 | 611890 | 24.273 | 0.75825 |
+| unsloth/ckpt_unsloth_m_fv_d2 | VALID | 9.727 | 3204.4 | 0.329 | 7430785 | 611956 | 24.273 | 0.75697 |
+| e4b/fused_attn4_m_fd_d2 | VALID | 3.548 | 1640.1 | 0.462 | 527202 | 83690 | 26.165 | 0.75598 |
-| P205 | qwen3pos69 | **HELD** | device ms per profiled step Unsloth 3259.1 / e4b 1693.9 = 1.924 vs in [1.7, 2.7] |
+| P205 | qwen3pos69 | **HELD** | device ms per profiled step Unsloth 3203.2 / e4b 1639.9 = 1.953 vs in [1.7, 2.7] |
-| P208 | qwen3pos69 | **HELD** | s/step ratio 2.803 over device ratio 1.924 = 1.457 (busy_t e4b 0.486, Unsloth 0.333) vs >= 1.05 |
+| P208 | qwen3pos69 | **HELD** | s/step ratio 2.803 over device ratio 1.953 = 1.435 (busy_t e4b 0.470, Unsloth 0.328) vs >= 1.05 |
```

## tc1-5090-138 (bench/h2h-2026-10-02/tc1/receipts/tc1-5090-138): REPRODUCED, 22 changed lines, 0 flips, 0 unpaired verdict lines, annotation share 0.0417
- cell qwen3sladder_e4b_fused_attn4_m_l0.json: device ms (all profiled steps) 5211.6 -> 5015.6 (annotation 196.0 in optimizer)
- cell qwen3sladder_e4b_fused_attn4_m_l0_d2.json: device ms (all profiled steps) 5173.2 -> 5004.6 (annotation 168.6 in optimizer)
- cell qwen3sladder_e4b_fused_attn4_m_l1.json: device ms (all profiled steps) 5403.5 -> 5237.8 (annotation 165.7 in optimizer)
- cell qwen3sladder_e4b_fused_attn4_m_l1_d2.json: device ms (all profiled steps) 5410.9 -> 5243.2 (annotation 167.7 in optimizer)
- cell qwen3sladder_e4b_fused_attn4_shipped_l0.json: device ms (all profiled steps) 4043.0 -> 3880.2 (annotation 162.8 in optimizer)
- cell qwen3sladder_e4b_fused_attn4_shipped_l0_d2.json: device ms (all profiled steps) 4049.4 -> 3886.3 (annotation 163.1 in optimizer)
- cell qwen3sladder_e4b_fused_attn4_shipped_l1.json: device ms (all profiled steps) 4167.9 -> 4009.4 (annotation 158.5 in optimizer)
- cell qwen3sladder_e4b_fused_attn4_shipped_l1_d2.json: device ms (all profiled steps) 4184.2 -> 4009.6 (annotation 174.6 in optimizer)
- NEAR BAR (margin below the share 0.0417): 1.046 <= 1.05 (margin 0.0038), 1.033 <= 1.05 (margin 0.0162) :: `| P213 | qwen3sladder | **HELD** | device ms per profiled step l1 / l0: m: 1670.0 -> 1746.8 ms = 1.046; shipped: 1294.4 -> 1336.5 ms = 1.033 (each <= 1.05) |`
```diff
-| e4b/fused_attn4_shipped_l0 | VALID | 3.231 | 1347.7 | 0.417 | 27.9 | 0 | 23.321 | 1.95917 / 0.7552 |
-| e4b/fused_attn4_shipped_l1 | VALID | 3.251 | 1389.3 | 0.427 | 23.5 | 49152 | 23.321 | 1.95917 / 0.75578 |
-| e4b/fused_attn4_m_l0 | VALID | 4.137 | 1737.2 | 0.420 | 306.0 | 0 | 26.168 | 1.95917 / 0.75871 |
-| e4b/fused_attn4_m_l1 | VALID | 3.286 | 1801.2 | 0.548 | 24.6 | 49152 | 26.495 | 1.95917 / 0.75814 |
-| e4b/fused_attn4_m_l1_d2 | VALID | 3.334 | 1803.6 | 0.541 | 24.3 | 49152 | 26.495 | 1.95917 / 0.75721 |
-| e4b/fused_attn4_m_l0_d2 | VALID | 4.167 | 1724.4 | 0.414 | 304.8 | 0 | 26.156 | 1.95917 / 0.75561 |
-| e4b/fused_attn4_shipped_l1_d2 | VALID | 3.280 | 1394.7 | 0.425 | 24.0 | 49152 | 23.321 | 1.95917 / 0.75286 |
-| e4b/fused_attn4_shipped_l0_d2 | VALID | 3.201 | 1349.8 | 0.422 | 27.6 | 0 | 23.321 | 1.95917 / 0.75614 |
+| e4b/fused_attn4_shipped_l0 | VALID | 3.231 | 1293.4 | 0.400 | 27.9 | 0 | 23.321 | 1.95917 / 0.7552 |
+| e4b/fused_attn4_shipped_l1 | VALID | 3.251 | 1336.5 | 0.411 | 23.5 | 49152 | 23.321 | 1.95917 / 0.75578 |
+| e4b/fused_attn4_m_l0 | VALID | 4.137 | 1671.9 | 0.404 | 306.0 | 0 | 26.168 | 1.95917 / 0.75871 |
+| e4b/fused_attn4_m_l1 | VALID | 3.286 | 1745.9 | 0.531 | 24.6 | 49152 | 26.495 | 1.95917 / 0.75814 |
+| e4b/fused_attn4_m_l1_d2 | VALID | 3.334 | 1747.7 | 0.524 | 24.3 | 49152 | 26.495 | 1.95917 / 0.75721 |
+| e4b/fused_attn4_m_l0_d2 | VALID | 4.167 | 1668.2 | 0.400 | 304.8 | 0 | 26.156 | 1.95917 / 0.75561 |
+| e4b/fused_attn4_shipped_l1_d2 | VALID | 3.280 | 1336.5 | 0.408 | 24.0 | 49152 | 23.321 | 1.95917 / 0.75286 |
+| e4b/fused_attn4_shipped_l0_d2 | VALID | 3.201 | 1295.4 | 0.405 | 27.6 | 0 | 23.321 | 1.95917 / 0.75614 |
-| P210 | qwen3sladder | **HELD** | l1 / l0 0.797 [0.789, 0.806 over 4 cross-draw ratios] vs <= 0.97; s/step l1 3.286 / 3.334, l0 4.137 / 4.167; matched l0 busy_t 0.417 (premise <= 0.85) |
-| P211 | qwen3sladder | **FALSIFIED** | l1 / l0 1.015 [1.006, 1.025 over 4 cross-draw ratios] vs <= 0.97; s/step l1 3.251 / 3.280, l0 3.231 / 3.201; matched l0 busy_t 0.417 (premise <= 0.85) |
+| P210 | qwen3sladder | **HELD** | l1 / l0 0.797 [0.789, 0.806 over 4 cross-draw ratios] vs <= 0.97; s/step l1 3.286 / 3.334, l0 4.137 / 4.167; matched l0 busy_t 0.402 (premise <= 0.85) |
+| P211 | qwen3sladder | **FALSIFIED** | l1 / l0 1.015 [1.006, 1.025 over 4 cross-draw ratios] vs <= 0.97; s/step l1 3.251 / 3.280, l0 3.231 / 3.201; matched l0 busy_t 0.402 (premise <= 0.85) |
-| P213 | qwen3sladder | **HELD** | device ms per profiled step l1 / l0: m: 1730.8 -> 1802.4 ms = 1.041; shipped: 1348.7 -> 1392.0 ms = 1.032 (each <= 1.05) |
+| P213 | qwen3sladder | **HELD** | device ms per profiled step l1 / l0: m: 1670.0 -> 1746.8 ms = 1.046; shipped: 1294.4 -> 1336.5 ms = 1.033 (each <= 1.05) |
```

## tc1-5090-139 (bench/h2h-2026-10-02/tc1/receipts/tc1-5090-139): REPRODUCED, 20 changed lines, 0 flips, 0 unpaired verdict lines, annotation share 0.042
- cell qwen3slauto_e4b_fused_attn4_m_l0.json: device ms (all profiled steps) 5109.7 -> 4954.4 (annotation 155.3 in optimizer)
- cell qwen3slauto_e4b_fused_attn4_m_l0_d2.json: device ms (all profiled steps) 5118.8 -> 4950.8 (annotation 168.0 in optimizer)
- cell qwen3slauto_e4b_fused_attn4_m_la.json: device ms (all profiled steps) 5331.9 -> 5172.8 (annotation 159.1 in optimizer)
- cell qwen3slauto_e4b_fused_attn4_m_la_d2.json: device ms (all profiled steps) 5333.1 -> 5179.0 (annotation 154.1 in optimizer)
- cell qwen3slauto_e4b_fused_attn4_shipped_l0.json: device ms (all profiled steps) 3982.3 -> 3830.4 (annotation 151.9 in optimizer)
- cell qwen3slauto_e4b_fused_attn4_shipped_l0_d2.json: device ms (all profiled steps) 3993.5 -> 3826.7 (annotation 166.8 in optimizer)
- cell qwen3slauto_e4b_fused_attn4_shipped_la.json: device ms (all profiled steps) 3990.9 -> 3840.9 (annotation 150.0 in optimizer)
- cell qwen3slauto_e4b_fused_attn4_shipped_la_d2.json: device ms (all profiled steps) 3997.9 -> 3830.0 (annotation 167.9 in optimizer)
- NEAR BAR (margin below the share 0.042): 1.002 in [0.98, 1.02] (margin 0.0176), 1.045 <= 1.06 (margin 0.0142) :: `| P218 | qwen3slauto | **HELD** | device ms per profiled step la / l0: m: 1650.9 -> 1725.3 ms = 1.045 (<= 1.06); shipped: 1276.2 -> 1278.5 ms = 1.002 (in [0.98,`
- UNPARSED prediction line, read by hand: `| P215 | qwen3slauto | **UNTESTED** | la / l0: two stable VALID draws a side are registered -- l0 UNSTABLE: draws 3.463 / 3.718 s/step differ by 7.1% > 5% (UNSTABLE: reported, not quoted); matched l0 `
```diff
-| e4b/fused_attn4_shipped_l0 | VALID | 2.991 | 1327.4 | 0.444 | 24.4 | 0 | 23.321 | 1.95917 / 0.75653 |
-| e4b/fused_attn4_shipped_la | VALID | 2.983 | 1330.3 | 0.446 | 24.4 | 0 | 23.321 | 1.95917 / 0.75692 |
-| e4b/fused_attn4_m_l0 | VALID | 3.463 | 1703.2 | 0.492 | 171.8 | 0 | 26.141 | 1.95917 / 0.75434 |
-| e4b/fused_attn4_m_la | VALID | 3.033 | 1777.3 | 0.586 | 21.0 | 49152 | 26.495 | 1.95917 / 0.75576 |
-| e4b/fused_attn4_m_la_d2 | VALID | 3.024 | 1777.7 | 0.588 | 21.1 | 49152 | 26.495 | 1.95917 / 0.75989 |
-| e4b/fused_attn4_m_l0_d2 | VALID | 3.718 | 1706.3 | 0.459 | 178.4 | 0 | 26.159 | 1.95917 / 0.75862 |
-| e4b/fused_attn4_shipped_la_d2 | VALID | 3.279 | 1332.6 | 0.406 | 28.9 | 0 | 23.321 | 1.95917 / 0.75686 |
-| e4b/fused_attn4_shipped_l0_d2 | VALID | 3.289 | 1331.2 | 0.405 | 30.2 | 0 | 23.321 | 1.95917 / 0.75438 |
+| e4b/fused_attn4_shipped_l0 | VALID | 2.991 | 1276.8 | 0.427 | 24.4 | 0 | 23.321 | 1.95917 / 0.75653 |
+| e4b/fused_attn4_shipped_la | VALID | 2.983 | 1280.3 | 0.429 | 24.4 | 0 | 23.321 | 1.95917 / 0.75692 |
+| e4b/fused_attn4_m_l0 | VALID | 3.463 | 1651.5 | 0.477 | 171.8 | 0 | 26.141 | 1.95917 / 0.75434 |
+| e4b/fused_attn4_m_la | VALID | 3.033 | 1724.3 | 0.569 | 21.0 | 49152 | 26.495 | 1.95917 / 0.75576 |
+| e4b/fused_attn4_m_la_d2 | VALID | 3.024 | 1726.3 | 0.571 | 21.1 | 49152 | 26.495 | 1.95917 / 0.75989 |
+| e4b/fused_attn4_m_l0_d2 | VALID | 3.718 | 1650.3 | 0.444 | 178.4 | 0 | 26.159 | 1.95917 / 0.75862 |
+| e4b/fused_attn4_shipped_la_d2 | VALID | 3.279 | 1276.7 | 0.389 | 28.9 | 0 | 23.321 | 1.95917 / 0.75686 |
+| e4b/fused_attn4_shipped_l0_d2 | VALID | 3.289 | 1275.6 | 0.388 | 30.2 | 0 | 23.321 | 1.95917 / 0.75438 |
-| P215 | qwen3slauto | **UNTESTED** | la / l0: two stable VALID draws a side are registered -- l0 UNSTABLE: draws 3.463 / 3.718 s/step differ by 7.1% > 5% (UNSTABLE: reported, not quoted); matched l0 busy_t 0.475: a host-bound box |
+| P215 | qwen3slauto | **UNTESTED** | la / l0: two stable VALID draws a side are registered -- l0 UNSTABLE: draws 3.463 / 3.718 s/step differ by 7.1% > 5% (UNSTABLE: reported, not quoted); matched l0 busy_t 0.460: a host-bound box |
-| P218 | qwen3slauto | **HELD** | device ms per profiled step la / l0: m: 1704.8 -> 1777.5 ms = 1.043 (<= 1.06); shipped: 1329.3 -> 1331.5 ms = 1.002 (in [0.98, 1.02]) |
+| P218 | qwen3slauto | **HELD** | device ms per profiled step la / l0: m: 1650.9 -> 1725.3 ms = 1.045 (<= 1.06); shipped: 1276.2 -> 1278.5 ms = 1.002 (in [0.98, 1.02]) |
```

## tc1-5090-141 (bench/h2h-2026-10-02/tc1/receipts/tc1-5090-141): REPRODUCED, 20 changed lines, 0 flips, 0 unpaired verdict lines, annotation share 0.0177
- cell qwen3slauto_e4b_fused_attn4_m_l0.json: device ms (all profiled steps) 5418.0 -> 5338.6 (annotation 79.4 in optimizer)
- cell qwen3slauto_e4b_fused_attn4_m_l0_d2.json: device ms (all profiled steps) 5417.7 -> 5336.8 (annotation 80.9 in optimizer)
- cell qwen3slauto_e4b_fused_attn4_m_la.json: device ms (all profiled steps) 5715.2 -> 5633.0 (annotation 82.2 in optimizer)
- cell qwen3slauto_e4b_fused_attn4_m_la_d2.json: device ms (all profiled steps) 5718.7 -> 5636.7 (annotation 82.0 in optimizer)
- cell qwen3slauto_e4b_fused_attn4_shipped_l0.json: device ms (all profiled steps) 4238.2 -> 4163.0 (annotation 75.2 in optimizer)
- cell qwen3slauto_e4b_fused_attn4_shipped_l0_d2.json: device ms (all profiled steps) 4212.6 -> 4138.6 (annotation 74.0 in optimizer)
- cell qwen3slauto_e4b_fused_attn4_shipped_la.json: device ms (all profiled steps) 4216.8 -> 4144.5 (annotation 72.3 in optimizer)
- cell qwen3slauto_e4b_fused_attn4_shipped_la_d2.json: device ms (all profiled steps) 4226.4 -> 4153.6 (annotation 72.8 in optimizer)
- NEAR BAR (margin below the share 0.0177): 1.056 <= 1.06 (margin 0.0038) :: `| P218 | qwen3slauto | **HELD** | device ms per profiled step la / l0: m: 1779.2 -> 1878.3 ms = 1.056 (<= 1.06); shipped: 1383.6 -> 1383.0 ms = 1.000 (in [0.98,`
```diff
-| e4b/fused_attn4_shipped_l0 | VALID | 1.594 | 1412.7 | 0.886 | 10.0 | 0 | 23.321 | 1.95917 / 0.75583 |
-| e4b/fused_attn4_shipped_la | VALID | 1.599 | 1405.6 | 0.879 | 10.1 | 0 | 23.321 | 1.95917 / 0.75558 |
-| e4b/fused_attn4_m_l0 | VALID | 2.041 | 1806.0 | 0.885 | 96.7 | 0 | 26.154 | 1.95917 / 0.7561 |
-| e4b/fused_attn4_m_la | VALID | 2.110 | 1905.1 | 0.903 | 8.8 | 49152 | 26.495 | 1.95917 / 0.75662 |
-| e4b/fused_attn4_m_la_d2 | VALID | 2.105 | 1906.2 | 0.906 | 8.8 | 49152 | 26.495 | 1.95917 / 0.75824 |
-| e4b/fused_attn4_m_l0_d2 | VALID | 2.047 | 1805.9 | 0.882 | 95.2 | 0 | 26.151 | 1.95917 / 0.75831 |
-| e4b/fused_attn4_shipped_la_d2 | VALID | 1.598 | 1408.8 | 0.882 | 10.2 | 0 | 23.321 | 1.95917 / 0.75742 |
-| e4b/fused_attn4_shipped_l0_d2 | VALID | 1.596 | 1404.2 | 0.880 | 10.3 | 0 | 23.321 | 1.95917 / 0.75762 |
+| e4b/fused_attn4_shipped_l0 | VALID | 1.594 | 1387.7 | 0.870 | 10.0 | 0 | 23.321 | 1.95917 / 0.75583 |
+| e4b/fused_attn4_shipped_la | VALID | 1.599 | 1381.5 | 0.864 | 10.1 | 0 | 23.321 | 1.95917 / 0.75558 |
+| e4b/fused_attn4_m_l0 | VALID | 2.041 | 1779.5 | 0.872 | 96.7 | 0 | 26.154 | 1.95917 / 0.7561 |
+| e4b/fused_attn4_m_la | VALID | 2.110 | 1877.7 | 0.890 | 8.8 | 49152 | 26.495 | 1.95917 / 0.75662 |
+| e4b/fused_attn4_m_la_d2 | VALID | 2.105 | 1878.9 | 0.893 | 8.8 | 49152 | 26.495 | 1.95917 / 0.75824 |
+| e4b/fused_attn4_m_l0_d2 | VALID | 2.047 | 1778.9 | 0.869 | 95.2 | 0 | 26.151 | 1.95917 / 0.75831 |
+| e4b/fused_attn4_shipped_la_d2 | VALID | 1.598 | 1384.5 | 0.866 | 10.2 | 0 | 23.321 | 1.95917 / 0.75742 |
+| e4b/fused_attn4_shipped_l0_d2 | VALID | 1.596 | 1379.5 | 0.864 | 10.3 | 0 | 23.321 | 1.95917 / 0.75762 |
-| P215 | qwen3slauto | **HELD** | la / l0 1.031 [1.028, 1.034 over 4 cross-draw ratios] vs <= 1.05; s/step la 2.110 / 2.105, l0 2.041 / 2.047; matched l0 busy_t 0.884: a GPU-bound box |
+| P215 | qwen3slauto | **HELD** | la / l0 1.031 [1.028, 1.034 over 4 cross-draw ratios] vs <= 1.05; s/step la 2.110 / 2.105, l0 2.041 / 2.047; matched l0 busy_t 0.871: a GPU-bound box |
-| P218 | qwen3slauto | **HELD** | device ms per profiled step la / l0: m: 1805.9 -> 1905.7 ms = 1.055 (<= 1.06); shipped: 1408.5 -> 1407.2 ms = 0.999 (in [0.98, 1.02]) |
+| P218 | qwen3slauto | **HELD** | device ms per profiled step la / l0: m: 1779.2 -> 1878.3 ms = 1.056 (<= 1.06); shipped: 1383.6 -> 1383.0 ms = 1.000 (in [0.98, 1.02]) |
```

## tc1-5090-16 (bench/h2h-2026-10-02/tc1/receipts/tc1-5090-16): REPRODUCED, 4 changed lines, 0 flips, 0 unpaired verdict lines, annotation share 0.0136
- cell qwen3_e4b_fused_attn4_m_prof.json: device ms (all profiled steps) 10477.6 -> 10334.6 (annotation 143.0 in optimizer)
- cell qwen3_unsloth_ckpt_unsloth_prof.json: device ms (all profiled steps) 12654.2 -> 12503.0 (annotation 151.2 in optimizer)
```diff
-- where Unsloth's step goes (`unsloth/ckpt_unsloth_prof`, descriptive): device busy fraction 0.234, device events/step 612179, CPU ops/step 7430931, CPU self by family {'other': 0.2907, 'matmul': 0.2565, 'norm_act': 0.1707, 'fused_kernel': 0.1576, 'autograd': 0.0665, 'memcpy': 0.0509, 'routing': 0.0054, 'optimizer': 0.0017}
+- where Unsloth's step goes (`unsloth/ckpt_unsloth_prof`, descriptive): device busy fraction 0.232, device events/step 612179, CPU ops/step 7430931, CPU self by family {'other': 0.2907, 'matmul': 0.2565, 'norm_act': 0.1707, 'fused_kernel': 0.1576, 'autograd': 0.0665, 'memcpy': 0.0509, 'routing': 0.0054, 'optimizer': 0.0017}
-| P8 | qwen3 | **FALSIFIED** | device_busy_fraction 0.234 (>= 0.5 predicted on the grouped_mm arm); device events/step 612179 |
+| P8 | qwen3 | **FALSIFIED** | device_busy_fraction 0.232 (>= 0.5 predicted on the grouped_mm arm); device events/step 612179 |
```

## tc1-5090-41 (bench/h2h-2026-10-02/tc1/receipts/tc1-5090-41): REPRODUCED, 8 changed lines, 0 flips, 0 unpaired verdict lines, annotation share 0.012
- cell qwen3prof945_e4b_fused_attn4_m_prof.json: device ms (all profiled steps) 7886.1 -> 7791.7 (annotation 94.4 in optimizer)
- cell qwen3prof945_e4b_fused_attn4_m_prof_legacy.json: device ms (all profiled steps) 7895.6 -> 7800.7 (annotation 94.9 in optimizer)
- cell qwen3prof945_e4b_fused_attn4_shipped_prof.json: device ms (all profiled steps) 6146.5 -> 6072.6 (annotation 73.9 in optimizer)
- UNPARSED prediction line, read by hand: `| fused_attn4_shipped_prof | **VALID** | 2.207 | 0.820 | 133898 | 733073 | {"autograd": 0.1858, "fused_kernel": 0.0287, "matmul": 0.076, "memcpy": 0.2042, "norm_act": 0.1733, "optimizer": 0.0054, "oth`
- UNPARSED prediction line, read by hand: `| fused_attn4_m_prof | **VALID** | 3.119 | 0.731 | 136969 | 750409 | {"autograd": 0.168, "fused_kernel": 0.0217, "matmul": 0.2438, "memcpy": 0.09, "norm_act": 0.1706, "optimizer": 0.0058, "other": 0.2`
- UNPARSED prediction line, read by hand: `| fused_attn4_m_prof_legacy | **VALID** | 3.924 | 0.588 | 140809 | 753087 | {"autograd": 0.1279, "fused_kernel": 0.0168, "matmul": 0.199, "memcpy": 0.0481, "norm_act": 0.1297, "optimizer": 0.0045, "ot`
- UNPARSED prediction line, read by hand: `| P19 | qwen3prof945 | **HELD** | device busy fraction matched new 0.731 vs legacy 0.588 (+0.143; >= +0.05 predicted); device events/step 136969 vs 140809; CPU ops/step 750409 vs 753087 |`
```diff
-| fused_attn4_shipped_prof | **VALID** | 2.207 | 0.830 | 133898 | 733073 | {"autograd": 0.1858, "fused_kernel": 0.0287, "matmul": 0.076, "memcpy": 0.2042, "norm_act": 0.1733, "optimizer": 0.0054, "other": 0.286, "routing": 0.0405} |
-| fused_attn4_m_prof | **VALID** | 3.119 | 0.740 | 136969 | 750409 | {"autograd": 0.168, "fused_kernel": 0.0217, "matmul": 0.2438, "memcpy": 0.09, "norm_act": 0.1706, "optimizer": 0.0058, "other": 0.2656, "routing": 0.0346} |
-| fused_attn4_m_prof_legacy | **VALID** | 3.924 | 0.595 | 140809 | 753087 | {"autograd": 0.1279, "fused_kernel": 0.0168, "matmul": 0.199, "memcpy": 0.0481, "norm_act": 0.1297, "optimizer": 0.0045, "other": 0.4453, "routing": 0.0286} |
+| fused_attn4_shipped_prof | **VALID** | 2.207 | 0.820 | 133898 | 733073 | {"autograd": 0.1858, "fused_kernel": 0.0287, "matmul": 0.076, "memcpy": 0.2042, "norm_act": 0.1733, "optimizer": 0.0054, "other": 0.286, "routing": 0.0405} |
+| fused_attn4_m_prof | **VALID** | 3.119 | 0.731 | 136969 | 750409 | {"autograd": 0.168, "fused_kernel": 0.0217, "matmul": 0.2438, "memcpy": 0.09, "norm_act": 0.1706, "optimizer": 0.0058, "other": 0.2656, "routing": 0.0346} |
+| fused_attn4_m_prof_legacy | **VALID** | 3.924 | 0.588 | 140809 | 753087 | {"autograd": 0.1279, "fused_kernel": 0.0168, "matmul": 0.199, "memcpy": 0.0481, "norm_act": 0.1297, "optimizer": 0.0045, "other": 0.4453, "routing": 0.0286} |
-| P19 | qwen3prof945 | **HELD** | device busy fraction matched new 0.740 vs legacy 0.595 (+0.145; >= +0.05 predicted); device events/step 136969 vs 140809; CPU ops/step 750409 vs 753087 |
+| P19 | qwen3prof945 | **HELD** | device busy fraction matched new 0.731 vs legacy 0.588 (+0.143; >= +0.05 predicted); device events/step 136969 vs 140809; CPU ops/step 750409 vs 753087 |
```

## tc1-5090-49 (bench/h2h-2026-10-02/tc1/receipts/tc1-5090-49): REPRODUCED, 0 changed lines, 0 flips, 0 unpaired verdict lines, annotation share 0.0267
- cell qwen3_e4b_fused_attn4_m_prof.json: device ms (all profiled steps) 5689.9 -> 5538.2 (annotation 151.7 in optimizer)
```diff
```

## tc1-5090-pos79-r1 (bench/h2h-2026-10-02/tc1/receipts/tc1-5090-pos79-r1): REPRODUCED, 8 changed lines, 0 flips, 0 unpaired verdict lines, annotation share 0.0413
- cell qwen3pos79_e4b_fused_attn4_m_fd.json: device ms (all profiled steps) 6194.3 -> 5938.5 (annotation 255.8 in optimizer)
- cell qwen3pos79_e4b_fused_attn4_m_fd_d2.json: device ms (all profiled steps) 6149.7 -> 5915.3 (annotation 234.4 in optimizer)
- cell qwen3pos79_unsloth_ckpt_unsloth_m_fv.json: device ms (all profiled steps) 5177.7 -> 5010.6 (annotation 167.1 in optimizer)
- cell qwen3pos79_unsloth_ckpt_unsloth_m_fv_d2.json: device ms (all profiled steps) 4406.6 -> 4250.9 (annotation 155.7 in optimizer)
```diff
-| e4b/fused_attn4_m_fd | VALID | 4.490 | 2064.8 | 0.460 | 497556 | 26.500 | 0.75455 |
-| unsloth/ckpt_unsloth_m_fv | VOID | 3.275 | 1725.9 | 0.527 | 200437 | 24.221 | 0.75511 |
-| unsloth/ckpt_unsloth_m_fv_d2 | VOID | 3.235 | 1468.9 | 0.454 | 159119 | 24.221 | 0.75436 |
-| e4b/fused_attn4_m_fd_d2 | VALID | 4.377 | 2049.9 | 0.468 | 497517 | 26.500 | 0.75629 |
+| e4b/fused_attn4_m_fd | VALID | 4.490 | 1979.5 | 0.441 | 497556 | 26.500 | 0.75455 |
+| unsloth/ckpt_unsloth_m_fv | VOID | 3.275 | 1670.2 | 0.510 | 200437 | 24.221 | 0.75511 |
+| unsloth/ckpt_unsloth_m_fv_d2 | VOID | 3.235 | 1417.0 | 0.438 | 159119 | 24.221 | 0.75436 |
+| e4b/fused_attn4_m_fd_d2 | VALID | 4.377 | 1971.8 | 0.451 | 497517 | 26.500 | 0.75629 |
```

## tc1c-h100-10 (bench/h2h-2026-10-02/tc1c/receipts/tc1c-h100-10): REPRODUCED, 4 changed lines, 0 flips, 0 unpaired verdict lines, annotation share 0.038
- cell qwen3_e4b_fused_attn4_m_prof.json: device ms (all profiled steps) 3503.7 -> 3370.6 (annotation 133.1 in optimizer)
- cell qwen3_unsloth_ckpt_unsloth_prof.json: device ms (all profiled steps) 3440.1 -> 3314.4 (annotation 125.7 in optimizer)
```diff
-- where Unsloth's step goes (`unsloth/ckpt_unsloth_prof`, descriptive): device busy fraction 0.302, device events/step 81780, CPU ops/step 515750, CPU self by family {'other': 0.3704, 'autograd': 0.2989, 'norm_act': 0.1292, 'matmul': 0.072, 'fused_kernel': 0.0556, 'memcpy': 0.0517, 'routing': 0.0169, 'optimizer': 0.0052}
+- where Unsloth's step goes (`unsloth/ckpt_unsloth_prof`, descriptive): device busy fraction 0.291, device events/step 81780, CPU ops/step 515750, CPU self by family {'other': 0.3704, 'autograd': 0.2989, 'norm_act': 0.1292, 'matmul': 0.072, 'fused_kernel': 0.0556, 'memcpy': 0.0517, 'routing': 0.0169, 'optimizer': 0.0052}
-| P8 | qwen3 | **FALSIFIED** | device_busy_fraction 0.302 (>= 0.5 predicted on the grouped_mm arm); device events/step 81780 |
+| P8 | qwen3 | **FALSIFIED** | device_busy_fraction 0.291 (>= 0.5 predicted on the grouped_mm arm); device events/step 81780 |
```

## tc1c-h100-11 (bench/h2h-2026-10-02/tc1c/receipts/tc1c-h100-11): REPRODUCED, 4 changed lines, 0 flips, 0 unpaired verdict lines, annotation share 0.0403
- cell qwen3_e4b_fused_attn4_m_prof.json: device ms (all profiled steps) 4531.5 -> 4397.9 (annotation 133.6 in optimizer)
- cell qwen3_unsloth_ckpt_unsloth_prof.json: device ms (all profiled steps) 3473.5 -> 3333.5 (annotation 140.0 in optimizer)
```diff
-- where Unsloth's step goes (`unsloth/ckpt_unsloth_prof`, descriptive): device busy fraction 0.309, device events/step 81780, CPU ops/step 515750, CPU self by family {'other': 0.3793, 'autograd': 0.2954, 'norm_act': 0.1311, 'matmul': 0.0733, 'memcpy': 0.0524, 'fused_kernel': 0.044, 'routing': 0.0179, 'optimizer': 0.0066}
+- where Unsloth's step goes (`unsloth/ckpt_unsloth_prof`, descriptive): device busy fraction 0.297, device events/step 81780, CPU ops/step 515750, CPU self by family {'other': 0.3793, 'autograd': 0.2954, 'norm_act': 0.1311, 'matmul': 0.0733, 'memcpy': 0.0524, 'fused_kernel': 0.044, 'routing': 0.0179, 'optimizer': 0.0066}
-| P8 | qwen3 | **FALSIFIED** | device_busy_fraction 0.309 (>= 0.5 predicted on the grouped_mm arm); device events/step 81780 |
+| P8 | qwen3 | **FALSIFIED** | device_busy_fraction 0.297 (>= 0.5 predicted on the grouped_mm arm); device events/step 81780 |
```

## tc1c-h100-15 (bench/h2h-2026-10-02/tc1c/receipts/tc1c-h100-15): REPRODUCED, 4 changed lines, 0 flips, 0 unpaired verdict lines, annotation share 0.0355
- cell qwen3_e4b_fused_attn4_m_prof.json: device ms (all profiled steps) 4472.2 -> 4351.1 (annotation 121.1 in optimizer)
- cell qwen3_unsloth_ckpt_unsloth_prof.json: device ms (all profiled steps) 3377.0 -> 3257.0 (annotation 120.0 in optimizer)
```diff
-- where Unsloth's step goes (`unsloth/ckpt_unsloth_prof`, descriptive): device busy fraction 0.308, device events/step 81780, CPU ops/step 515750, CPU self by family {'other': 0.3739, 'autograd': 0.3047, 'norm_act': 0.1228, 'matmul': 0.0756, 'memcpy': 0.0539, 'fused_kernel': 0.0459, 'routing': 0.0179, 'optimizer': 0.0052}
+- where Unsloth's step goes (`unsloth/ckpt_unsloth_prof`, descriptive): device busy fraction 0.297, device events/step 81780, CPU ops/step 515750, CPU self by family {'other': 0.3739, 'autograd': 0.3047, 'norm_act': 0.1228, 'matmul': 0.0756, 'memcpy': 0.0539, 'fused_kernel': 0.0459, 'routing': 0.0179, 'optimizer': 0.0052}
-| P8 | qwen3 | **FALSIFIED** | device_busy_fraction 0.308 (>= 0.5 predicted on the grouped_mm arm); device events/step 81780 |
+| P8 | qwen3 | **FALSIFIED** | device_busy_fraction 0.297 (>= 0.5 predicted on the grouped_mm arm); device events/step 81780 |
```

## tc1c-h100-2 (bench/h2h-2026-10-02/tc1c/receipts/tc1c-h100-2): REPRODUCED, 4 changed lines, 0 flips, 0 unpaired verdict lines, annotation share 0.0378
- cell qwen3_e4b_fused_attn4_m_prof.json: device ms (all profiled steps) 8403.9 -> 8276.5 (annotation 127.4 in optimizer)
- cell qwen3_unsloth_ckpt_unsloth_prof.json: device ms (all profiled steps) 3475.4 -> 3344.2 (annotation 131.2 in optimizer)
```diff
-- where Unsloth's step goes (`unsloth/ckpt_unsloth_prof`, descriptive): device busy fraction 0.308, device events/step 81780, CPU ops/step 515683, CPU self by family {'other': 0.3824, 'autograd': 0.2934, 'norm_act': 0.1197, 'matmul': 0.0855, 'memcpy': 0.0508, 'fused_kernel': 0.0447, 'routing': 0.0178, 'optimizer': 0.0058}
+- where Unsloth's step goes (`unsloth/ckpt_unsloth_prof`, descriptive): device busy fraction 0.297, device events/step 81780, CPU ops/step 515683, CPU self by family {'other': 0.3824, 'autograd': 0.2934, 'norm_act': 0.1197, 'matmul': 0.0855, 'memcpy': 0.0508, 'fused_kernel': 0.0447, 'routing': 0.0178, 'optimizer': 0.0058}
-| P8 | qwen3 | **FALSIFIED** | device_busy_fraction 0.308 (>= 0.5 predicted on the grouped_mm arm); device events/step 81780 |
+| P8 | qwen3 | **FALSIFIED** | device_busy_fraction 0.297 (>= 0.5 predicted on the grouped_mm arm); device events/step 81780 |
```

## tc1c-h100-3 (bench/h2h-2026-10-02/tc1c/receipts/tc1c-h100-3): REPRODUCED, 4 changed lines, 0 flips, 0 unpaired verdict lines, annotation share 0.0378
- cell qwen3_e4b_fused_attn4_m_prof.json: device ms (all profiled steps) 7319.5 -> 7188.5 (annotation 131.0 in optimizer)
- cell qwen3_unsloth_ckpt_unsloth_prof.json: device ms (all profiled steps) 3476.0 -> 3344.7 (annotation 131.3 in optimizer)
```diff
-- where Unsloth's step goes (`unsloth/ckpt_unsloth_prof`, descriptive): device busy fraction 0.305, device events/step 81780, CPU ops/step 515750, CPU self by family {'other': 0.3896, 'autograd': 0.2884, 'norm_act': 0.132, 'matmul': 0.0735, 'memcpy': 0.0504, 'fused_kernel': 0.043, 'routing': 0.0175, 'optimizer': 0.0057}
+- where Unsloth's step goes (`unsloth/ckpt_unsloth_prof`, descriptive): device busy fraction 0.294, device events/step 81780, CPU ops/step 515750, CPU self by family {'other': 0.3896, 'autograd': 0.2884, 'norm_act': 0.132, 'matmul': 0.0735, 'memcpy': 0.0504, 'fused_kernel': 0.043, 'routing': 0.0175, 'optimizer': 0.0057}
-| P8 | qwen3 | **FALSIFIED** | device_busy_fraction 0.305 (>= 0.5 predicted on the grouped_mm arm); device events/step 81780 |
+| P8 | qwen3 | **FALSIFIED** | device_busy_fraction 0.294 (>= 0.5 predicted on the grouped_mm arm); device events/step 81780 |
```

## tc1c-h100-4 (bench/h2h-2026-10-02/tc1c/receipts/tc1c-h100-4): REPRODUCED, 4 changed lines, 0 flips, 0 unpaired verdict lines, annotation share 0.0387
- cell qwen3_e4b_fused_attn4_m_prof.json: device ms (all profiled steps) 5489.0 -> 5361.1 (annotation 127.9 in optimizer)
- cell qwen3_unsloth_ckpt_unsloth_prof.json: device ms (all profiled steps) 3476.0 -> 3341.5 (annotation 134.5 in optimizer)
```diff
-- where Unsloth's step goes (`unsloth/ckpt_unsloth_prof`, descriptive): device busy fraction 0.306, device events/step 81780, CPU ops/step 515750, CPU self by family {'other': 0.3831, 'autograd': 0.3039, 'norm_act': 0.122, 'matmul': 0.0738, 'memcpy': 0.0507, 'fused_kernel': 0.0431, 'routing': 0.0175, 'optimizer': 0.006}
+- where Unsloth's step goes (`unsloth/ckpt_unsloth_prof`, descriptive): device busy fraction 0.294, device events/step 81780, CPU ops/step 515750, CPU self by family {'other': 0.3831, 'autograd': 0.3039, 'norm_act': 0.122, 'matmul': 0.0738, 'memcpy': 0.0507, 'fused_kernel': 0.0431, 'routing': 0.0175, 'optimizer': 0.006}
-| P8 | qwen3 | **FALSIFIED** | device_busy_fraction 0.306 (>= 0.5 predicted on the grouped_mm arm); device events/step 81780 |
+| P8 | qwen3 | **FALSIFIED** | device_busy_fraction 0.294 (>= 0.5 predicted on the grouped_mm arm); device events/step 81780 |
```

## tc1c-h100-6 (bench/h2h-2026-10-02/tc1c/receipts/tc1c-h100-6): REPRODUCED, 4 changed lines, 0 flips, 0 unpaired verdict lines, annotation share 0.042
- cell qwen3_e4b_fused_attn4_m_prof.json: device ms (all profiled steps) 10295.5 -> 10170.3 (annotation 125.2 in optimizer)
- cell qwen3_unsloth_ckpt_unsloth_prof.json: device ms (all profiled steps) 3488.4 -> 3341.9 (annotation 146.5 in optimizer)
```diff
-- where Unsloth's step goes (`unsloth/ckpt_unsloth_prof`, descriptive): device busy fraction 0.308, device events/step 81780, CPU ops/step 515750, CPU self by family {'other': 0.3817, 'autograd': 0.2923, 'norm_act': 0.13, 'matmul': 0.0744, 'memcpy': 0.0519, 'fused_kernel': 0.0446, 'routing': 0.0178, 'optimizer': 0.0071}
+- where Unsloth's step goes (`unsloth/ckpt_unsloth_prof`, descriptive): device busy fraction 0.295, device events/step 81780, CPU ops/step 515750, CPU self by family {'other': 0.3817, 'autograd': 0.2923, 'norm_act': 0.13, 'matmul': 0.0744, 'memcpy': 0.0519, 'fused_kernel': 0.0446, 'routing': 0.0178, 'optimizer': 0.0071}
-| P8 | qwen3 | **FALSIFIED** | device_busy_fraction 0.308 (>= 0.5 predicted on the grouped_mm arm); device events/step 81780 |
+| P8 | qwen3 | **FALSIFIED** | device_busy_fraction 0.295 (>= 0.5 predicted on the grouped_mm arm); device events/step 81780 |
```

## tc1c-h100-7 (bench/h2h-2026-10-02/tc1c/receipts/tc1c-h100-7): REPRODUCED, 4 changed lines, 0 flips, 0 unpaired verdict lines, annotation share 0.0378
- cell qwen3_e4b_fused_attn4_m_prof.json: device ms (all profiled steps) 7367.5 -> 7222.3 (annotation 145.2 in optimizer)
- cell qwen3_unsloth_ckpt_unsloth_prof.json: device ms (all profiled steps) 3429.8 -> 3300.1 (annotation 129.7 in optimizer)
```diff
-- where Unsloth's step goes (`unsloth/ckpt_unsloth_prof`, descriptive): device busy fraction 0.301, device events/step 81780, CPU ops/step 515750, CPU self by family {'other': 0.3783, 'autograd': 0.3001, 'norm_act': 0.1192, 'matmul': 0.0847, 'memcpy': 0.051, 'fused_kernel': 0.0435, 'routing': 0.0176, 'optimizer': 0.0056}
+- where Unsloth's step goes (`unsloth/ckpt_unsloth_prof`, descriptive): device busy fraction 0.290, device events/step 81780, CPU ops/step 515750, CPU self by family {'other': 0.3783, 'autograd': 0.3001, 'norm_act': 0.1192, 'matmul': 0.0847, 'memcpy': 0.051, 'fused_kernel': 0.0435, 'routing': 0.0176, 'optimizer': 0.0056}
-| P8 | qwen3 | **FALSIFIED** | device_busy_fraction 0.301 (>= 0.5 predicted on the grouped_mm arm); device events/step 81780 |
+| P8 | qwen3 | **FALSIFIED** | device_busy_fraction 0.290 (>= 0.5 predicted on the grouped_mm arm); device events/step 81780 |
```

## tc1c-h100-9 (bench/h2h-2026-10-02/tc1c/receipts/tc1c-h100-9): REPRODUCED, 4 changed lines, 0 flips, 0 unpaired verdict lines, annotation share 0.0365
- cell qwen3_e4b_fused_attn4_m_prof.json: device ms (all profiled steps) 4524.1 -> 4398.5 (annotation 125.6 in optimizer)
- cell qwen3_unsloth_ckpt_unsloth_prof.json: device ms (all profiled steps) 3468.5 -> 3341.8 (annotation 126.7 in optimizer)
```diff
-- where Unsloth's step goes (`unsloth/ckpt_unsloth_prof`, descriptive): device busy fraction 0.306, device events/step 81780, CPU ops/step 515750, CPU self by family {'other': 0.3897, 'autograd': 0.2892, 'norm_act': 0.1302, 'matmul': 0.0725, 'memcpy': 0.0521, 'fused_kernel': 0.0438, 'routing': 0.0171, 'optimizer': 0.0053}
+- where Unsloth's step goes (`unsloth/ckpt_unsloth_prof`, descriptive): device busy fraction 0.295, device events/step 81780, CPU ops/step 515750, CPU self by family {'other': 0.3897, 'autograd': 0.2892, 'norm_act': 0.1302, 'matmul': 0.0725, 'memcpy': 0.0521, 'fused_kernel': 0.0438, 'routing': 0.0171, 'optimizer': 0.0053}
-| P8 | qwen3 | **FALSIFIED** | device_busy_fraction 0.306 (>= 0.5 predicted on the grouped_mm arm); device events/step 81780 |
+| P8 | qwen3 | **FALSIFIED** | device_busy_fraction 0.295 (>= 0.5 predicted on the grouped_mm arm); device events/step 81780 |
```

## tc1-5090-146 (bench/p129/receipts/tc1-5090-146): REPRODUCED, 20 changed lines, 0 flips, 0 unpaired verdict lines, annotation share 0.0415
- cell qwen3fqkv_e4b_fused_attn4_m_q0.json: device ms (all profiled steps) 5194.8 -> 5016.6 (annotation 178.2 in optimizer)
- cell qwen3fqkv_e4b_fused_attn4_m_q0_d2.json: device ms (all profiled steps) 5212.3 -> 5017.3 (annotation 195.0 in optimizer)
- cell qwen3fqkv_e4b_fused_attn4_m_q1.json: device ms (all profiled steps) 5137.9 -> 4974.4 (annotation 163.5 in optimizer)
- cell qwen3fqkv_e4b_fused_attn4_m_q1_d2.json: device ms (all profiled steps) 5129.1 -> 4963.2 (annotation 165.9 in optimizer)
- cell qwen3fqkv_e4b_fused_attn4_shipped_q0.json: device ms (all profiled steps) 3848.1 -> 3691.2 (annotation 156.9 in optimizer)
- cell qwen3fqkv_e4b_fused_attn4_shipped_q0_d2.json: device ms (all profiled steps) 3844.5 -> 3685.4 (annotation 159.1 in optimizer)
- cell qwen3fqkv_e4b_fused_attn4_shipped_q1.json: device ms (all profiled steps) 3793.2 -> 3635.7 (annotation 157.5 in optimizer)
- cell qwen3fqkv_e4b_fused_attn4_shipped_q1_d2.json: device ms (all profiled steps) 3787.4 -> 3630.1 (annotation 157.3 in optimizer)
- UNPARSED prediction line, read by hand: `| DEVICE | qwen3fqkv | **REPORTED** | m: device 1672.3 -> 1656.3 ms = 0.990, peak 26.495 -> 26.5 GB; shipped: device 1229.4 -> 1211.0 ms = 0.985, peak 23.321 -> 23.344 GB |`
```diff
-| e4b/fused_attn4_shipped_q0 | VALID | 3.193 | 1282.7 | 0.402 | 75833 | 0 | 23.321 | 1.95917 / 0.75512 |
-| e4b/fused_attn4_shipped_q1 | VALID | 2.842 | 1264.4 | 0.445 | 64761 | 48 | 23.344 | 1.94684 / 0.7572 |
-| e4b/fused_attn4_m_q0 | VALID | 3.302 | 1731.6 | 0.524 | 79582 | 0 | 26.495 | 1.95917 / 0.75805 |
-| e4b/fused_attn4_m_q1 | VALID | 2.939 | 1712.6 | 0.583 | 68524 | 48 | 26.500 | 1.94684 / 0.75478 |
-| e4b/fused_attn4_m_q1_d2 | VALID | 3.003 | 1709.7 | 0.569 | 68510 | 48 | 26.500 | 1.94684 / 0.75722 |
-| e4b/fused_attn4_m_q0_d2 | VALID | 3.332 | 1737.4 | 0.522 | 79580 | 0 | 26.495 | 1.95917 / 0.75549 |
-| e4b/fused_attn4_shipped_q1_d2 | VALID | 2.801 | 1262.5 | 0.451 | 64761 | 48 | 23.344 | 1.94684 / 0.7538 |
-| e4b/fused_attn4_shipped_q0_d2 | VALID | 3.227 | 1281.5 | 0.397 | 75833 | 0 | 23.321 | 1.95917 / 0.75895 |
+| e4b/fused_attn4_shipped_q0 | VALID | 3.193 | 1230.4 | 0.385 | 75833 | 0 | 23.321 | 1.95917 / 0.75512 |
+| e4b/fused_attn4_shipped_q1 | VALID | 2.842 | 1211.9 | 0.426 | 64761 | 48 | 23.344 | 1.94684 / 0.7572 |
+| e4b/fused_attn4_m_q0 | VALID | 3.302 | 1672.2 | 0.506 | 79582 | 0 | 26.495 | 1.95917 / 0.75805 |
+| e4b/fused_attn4_m_q1 | VALID | 2.939 | 1658.1 | 0.564 | 68524 | 48 | 26.500 | 1.94684 / 0.75478 |
+| e4b/fused_attn4_m_q1_d2 | VALID | 3.003 | 1654.4 | 0.551 | 68510 | 48 | 26.500 | 1.94684 / 0.75722 |
+| e4b/fused_attn4_m_q0_d2 | VALID | 3.332 | 1672.4 | 0.502 | 79580 | 0 | 26.495 | 1.95917 / 0.75549 |
+| e4b/fused_attn4_shipped_q1_d2 | VALID | 2.801 | 1210.0 | 0.432 | 64761 | 48 | 23.344 | 1.94684 / 0.7538 |
+| e4b/fused_attn4_shipped_q0_d2 | VALID | 3.227 | 1228.5 | 0.381 | 75833 | 0 | 23.321 | 1.95917 / 0.75895 |
-| PREMISE | qwen3fqkv | **HELD** | matched q0 busy_t 0.523 vs <= 0.85 |
+| PREMISE | qwen3fqkv | **HELD** | matched q0 busy_t 0.504 vs <= 0.85 |
-| DEVICE | qwen3fqkv | **REPORTED** | m: device 1734.5 -> 1711.2 ms = 0.987, peak 26.495 -> 26.5 GB; shipped: device 1282.1 -> 1263.4 ms = 0.985, peak 23.321 -> 23.344 GB |
+| DEVICE | qwen3fqkv | **REPORTED** | m: device 1672.3 -> 1656.3 ms = 0.990, peak 26.495 -> 26.5 GB; shipped: device 1229.4 -> 1211.0 ms = 0.985, peak 23.321 -> 23.344 GB |
```

## tc1-5090-147 (bench/p129/receipts/tc1-5090-147): REPRODUCED, 20 changed lines, 0 flips, 0 unpaired verdict lines, annotation share 0.0481
- cell qwen3fqkv3_e4b_fused_attn4_m_q0.json: device ms (all profiled steps) 5530.6 -> 5309.7 (annotation 220.9 in optimizer)
- cell qwen3fqkv3_e4b_fused_attn4_m_q0_d2.json: device ms (all profiled steps) 5501.8 -> 5308.1 (annotation 193.7 in optimizer)
- cell qwen3fqkv3_e4b_fused_attn4_m_q1.json: device ms (all profiled steps) 5456.9 -> 5245.6 (annotation 211.3 in optimizer)
- cell qwen3fqkv3_e4b_fused_attn4_m_q1_d2.json: device ms (all profiled steps) 5429.7 -> 5240.5 (annotation 189.2 in optimizer)
- cell qwen3fqkv3_e4b_fused_attn4_shipped_q0.json: device ms (all profiled steps) 4142.9 -> 3957.4 (annotation 185.5 in optimizer)
- cell qwen3fqkv3_e4b_fused_attn4_shipped_q0_d2.json: device ms (all profiled steps) 4152.7 -> 3959.5 (annotation 193.2 in optimizer)
- cell qwen3fqkv3_e4b_fused_attn4_shipped_q1.json: device ms (all profiled steps) 4091.6 -> 3898.2 (annotation 193.4 in optimizer)
- cell qwen3fqkv3_e4b_fused_attn4_shipped_q1_d2.json: device ms (all profiled steps) 4092.4 -> 3895.6 (annotation 196.8 in optimizer)
- UNPARSED prediction line, read by hand: `| DEVICE | qwen3fqkv3 | **REPORTED** | m: device 1769.6 -> 1747.7 ms = 0.988, peak 26.495 -> 26.5 GB; shipped: device 1319.5 -> 1299.0 ms = 0.984, peak 23.321 -> 23.344 GB |`
```diff
-| e4b/fused_attn4_shipped_q0 | VALID | 3.902 | 1381.0 | 0.354 | 75833 | 0 | 23.321 | 1.95917 / 0.75709 |
-| e4b/fused_attn4_shipped_q1 | VALID | 3.364 | 1363.9 | 0.405 | 64761 | 48 | 23.344 | 1.94684 / 0.75825 |
-| e4b/fused_attn4_m_q0 | VALID | 3.985 | 1843.5 | 0.463 | 79572 | 0 | 26.495 | 1.95917 / 0.75583 |
-| e4b/fused_attn4_m_q1 | VALID | 3.546 | 1819.0 | 0.513 | 68510 | 48 | 26.500 | 1.94684 / 0.75757 |
-| e4b/fused_attn4_m_q1_d2 | VALID | 3.555 | 1809.9 | 0.509 | 68516 | 48 | 26.500 | 1.94684 / 0.75444 |
-| e4b/fused_attn4_m_q0_d2 | VALID | 3.944 | 1833.9 | 0.465 | 79580 | 0 | 26.495 | 1.95917 / 0.75619 |
-| e4b/fused_attn4_shipped_q1_d2 | VALID | 3.385 | 1364.1 | 0.403 | 64761 | 48 | 23.344 | 1.94684 / 0.75648 |
-| e4b/fused_attn4_shipped_q0_d2 | VALID | 3.869 | 1384.2 | 0.358 | 75833 | 0 | 23.321 | 1.95917 / 0.75638 |
+| e4b/fused_attn4_shipped_q0 | VALID | 3.902 | 1319.1 | 0.338 | 75833 | 0 | 23.321 | 1.95917 / 0.75709 |
+| e4b/fused_attn4_shipped_q1 | VALID | 3.364 | 1299.4 | 0.386 | 64761 | 48 | 23.344 | 1.94684 / 0.75825 |
+| e4b/fused_attn4_m_q0 | VALID | 3.985 | 1769.9 | 0.444 | 79572 | 0 | 26.495 | 1.95917 / 0.75583 |
+| e4b/fused_attn4_m_q1 | VALID | 3.546 | 1748.5 | 0.493 | 68510 | 48 | 26.500 | 1.94684 / 0.75757 |
+| e4b/fused_attn4_m_q1_d2 | VALID | 3.555 | 1746.8 | 0.491 | 68516 | 48 | 26.500 | 1.94684 / 0.75444 |
+| e4b/fused_attn4_m_q0_d2 | VALID | 3.944 | 1769.4 | 0.449 | 79580 | 0 | 26.495 | 1.95917 / 0.75619 |
+| e4b/fused_attn4_shipped_q1_d2 | VALID | 3.385 | 1298.5 | 0.384 | 64761 | 48 | 23.344 | 1.94684 / 0.75648 |
+| e4b/fused_attn4_shipped_q0_d2 | VALID | 3.869 | 1319.8 | 0.341 | 75833 | 0 | 23.321 | 1.95917 / 0.75638 |
-| PREMISE | qwen3fqkv3 | **HELD** | matched q0 busy_t 0.464 vs <= 0.85 |
+| PREMISE | qwen3fqkv3 | **HELD** | matched q0 busy_t 0.446 vs <= 0.85 |
-| DEVICE | qwen3fqkv3 | **REPORTED** | m: device 1838.7 -> 1814.4 ms = 0.987, peak 26.495 -> 26.5 GB; shipped: device 1382.6 -> 1364.0 ms = 0.987, peak 23.321 -> 23.344 GB |
+| DEVICE | qwen3fqkv3 | **REPORTED** | m: device 1769.6 -> 1747.7 ms = 0.988, peak 26.495 -> 26.5 GB; shipped: device 1319.5 -> 1299.0 ms = 0.984, peak 23.321 -> 23.344 GB |
```

## tc1-5090-150 (bench/p129/receipts/tc1-5090-150): REPRODUCED, 20 changed lines, 0 flips, 0 unpaired verdict lines, annotation share 0.0468
- cell qwen3fqkv3_e4b_fused_attn4_m_q0.json: device ms (all profiled steps) 5375.0 -> 5193.4 (annotation 181.6 in optimizer)
- cell qwen3fqkv3_e4b_fused_attn4_m_q0_d2.json: device ms (all profiled steps) 5358.4 -> 5182.3 (annotation 176.1 in optimizer)
- cell qwen3fqkv3_e4b_fused_attn4_m_q1.json: device ms (all profiled steps) 5308.8 -> 5123.8 (annotation 185.0 in optimizer)
- cell qwen3fqkv3_e4b_fused_attn4_m_q1_d2.json: device ms (all profiled steps) 5296.7 -> 5126.2 (annotation 170.5 in optimizer)
- cell qwen3fqkv3_e4b_fused_attn4_shipped_q0.json: device ms (all profiled steps) 3997.2 -> 3837.3 (annotation 159.9 in optimizer)
- cell qwen3fqkv3_e4b_fused_attn4_shipped_q0_d2.json: device ms (all profiled steps) 4000.9 -> 3843.7 (annotation 157.2 in optimizer)
- cell qwen3fqkv3_e4b_fused_attn4_shipped_q1.json: device ms (all profiled steps) 3937.9 -> 3779.3 (annotation 158.6 in optimizer)
- cell qwen3fqkv3_e4b_fused_attn4_shipped_q1_d2.json: device ms (all profiled steps) 3965.4 -> 3779.8 (annotation 185.6 in optimizer)
- UNPARSED prediction line, read by hand: `| DEVICE | qwen3fqkv3 | **REPORTED** | m: device 1729.3 -> 1708.3 ms = 0.988, peak 26.495 -> 26.5 GB; shipped: device 1280.2 -> 1259.8 ms = 0.984, peak 23.321 -> 23.344 GB |`
```diff
-| e4b/fused_attn4_shipped_q0 | VALID | 3.231 | 1332.4 | 0.412 | 75833 | 0 | 23.321 | 1.95917 / 0.7554 |
-| e4b/fused_attn4_shipped_q1 | VALID | 2.809 | 1312.6 | 0.467 | 64761 | 48 | 23.344 | 1.94684 / 0.75533 |
-| e4b/fused_attn4_m_q0 | VALID | 3.319 | 1791.7 | 0.540 | 79614 | 0 | 26.495 | 1.95917 / 0.75857 |
-| e4b/fused_attn4_m_q1 | VALID | 3.012 | 1769.6 | 0.588 | 68524 | 48 | 26.500 | 1.94684 / 0.7602 |
-| e4b/fused_attn4_m_q1_d2 | VALID | 2.998 | 1765.6 | 0.589 | 68518 | 48 | 26.500 | 1.94684 / 0.75533 |
-| e4b/fused_attn4_m_q0_d2 | VALID | 3.313 | 1786.1 | 0.539 | 79572 | 0 | 26.495 | 1.95917 / 0.75333 |
-| e4b/fused_attn4_shipped_q1_d2 | VALID | 2.831 | 1321.8 | 0.467 | 64761 | 48 | 23.344 | 1.94684 / 0.75644 |
-| e4b/fused_attn4_shipped_q0_d2 | VALID | 3.153 | 1333.6 | 0.423 | 75833 | 0 | 23.321 | 1.95917 / 0.75494 |
+| e4b/fused_attn4_shipped_q0 | VALID | 3.231 | 1279.1 | 0.396 | 75833 | 0 | 23.321 | 1.95917 / 0.7554 |
+| e4b/fused_attn4_shipped_q1 | VALID | 2.809 | 1259.8 | 0.448 | 64761 | 48 | 23.344 | 1.94684 / 0.75533 |
+| e4b/fused_attn4_m_q0 | VALID | 3.319 | 1731.1 | 0.522 | 79614 | 0 | 26.495 | 1.95917 / 0.75857 |
+| e4b/fused_attn4_m_q1 | VALID | 3.012 | 1707.9 | 0.567 | 68524 | 48 | 26.500 | 1.94684 / 0.7602 |
+| e4b/fused_attn4_m_q1_d2 | VALID | 2.998 | 1708.7 | 0.570 | 68518 | 48 | 26.500 | 1.94684 / 0.75533 |
+| e4b/fused_attn4_m_q0_d2 | VALID | 3.313 | 1727.4 | 0.521 | 79572 | 0 | 26.495 | 1.95917 / 0.75333 |
+| e4b/fused_attn4_shipped_q1_d2 | VALID | 2.831 | 1259.9 | 0.445 | 64761 | 48 | 23.344 | 1.94684 / 0.75644 |
+| e4b/fused_attn4_shipped_q0_d2 | VALID | 3.153 | 1281.2 | 0.406 | 75833 | 0 | 23.321 | 1.95917 / 0.75494 |
-| PREMISE | qwen3fqkv3 | **HELD** | matched q0 busy_t 0.539 vs <= 0.85 |
+| PREMISE | qwen3fqkv3 | **HELD** | matched q0 busy_t 0.521 vs <= 0.85 |
-| DEVICE | qwen3fqkv3 | **REPORTED** | m: device 1788.9 -> 1767.6 ms = 0.988, peak 26.495 -> 26.5 GB; shipped: device 1333.0 -> 1317.2 ms = 0.988, peak 23.321 -> 23.344 GB |
+| DEVICE | qwen3fqkv3 | **REPORTED** | m: device 1729.3 -> 1708.3 ms = 0.988, peak 26.495 -> 26.5 GB; shipped: device 1280.2 -> 1259.8 ms = 0.984, peak 23.321 -> 23.344 GB |
```

