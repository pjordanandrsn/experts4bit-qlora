# TC1 — Qwen3-30B-A3B on one RTX 5090: e4b vs Unsloth vs HF+PEFT vs axolotl at matched work, init and adapter precision (tc1dec-5090-4)
Rule (tc1/TC1-PREREG.md): status per attempt in the vocabulary OK / REFUSED / OOM / INSTALL_FAILED / LOAD_FAULT / HARNESS_ERROR / ALARM / NOT_RUN; VALID/VOID per tp4's predicates plus the matched-set predicates (R3: matched init complete, fp32 adapters, step-0 held-out within 0.005 of e4b/reference_attn4_m); VERDICT exactly one of VALID / VOID / QUALITY_FAIL / OOM / UNSUPPORTED / HARNESS_ERROR / ALARM / NOT_RUN (QUALITY_FAIL = held-out |Δ| at N > 0.05 vs e4b/fused_attn4_m); positions = s/step ratios other/e4b from the medians over both draws (stability |d1-d2|/mean ≤ 5 % e4b / 10 % others, else UNSTABLE), quoted only when both arms are VALID; equivalence vs e4b/reference_attn4_m EQUIVALENT ≤ 0.02 / COMPARABLE ≤ 0.05 / else DIVERGENT; frozen base SAME-BYTES / DIFFERENT / N-A per slot; predictions P1–P10 (+ P1b) scored HELD / FALSIFIED / UNTESTED. VOID never enters a ratio or an equivalence reading. Nothing is licensed; no cross-box number is divided into these.
`versions.txt`
```
e4b 0.48.0 @ac3adf9608b47be7f3663985dab7874ef626a68f (GitHub main)
gnf4 0.41.0 @aaefbf8d1a20d22a18114b0a2091ffd5235777cf (GitHub main)
torch(e4b/hf) 2.8.0+cu128
triton(e4b/hf) 3.4.0
transformers(e4b/hf) 5.18.0
bitsandbytes(e4b/hf) 0.50.2
peft(hf) 0.21.2
axolotl 0.20.0
torch(axolotl) 2.14.0+cu130
transformers(axolotl) 5.17.0
peft(axolotl) 0.21.0
bitsandbytes(axolotl) 0.50.2
python(axolotl) 3.12.15
deepspeed(axolotl) None
```
`box.json`
```
{
 "box": "A",
 "run_id": "tc1dec-5090-4",
 "instance_id": "54403153",
 "gpu": "NVIDIA GeForce RTX 5090",
 "driver": "580.126.09",
 "cpu": "AMD EPYC 7713 64-Core Processor",
 "nproc": 128,
 "mem_total_kb": "527972068",
 "cgroup_memory_max": "",
 "disk_root": "overlay         320G  2.4M  320G   1% /",
 "hostname": "8243068a5ba3",
 "cgroup_cpu_max": "",
 "affinity_cpus": 128,
 "cgroup_cpuset_effective": "",
 "registered_gpu_class": "5090",
 "prereg": "tc1/TC1-PREREG.md",
 "local_box": null,
 "local_snapshot": null
}
```

### OLMoE-1B-7B-0924-Instruct (amendment 46: grouped-nf4-gemm's fused 4-bit kernels vs its decoded route, matched arm, venv-e4b, TC2's pin) (`olmoedecab`, registered n_layers 16, attention census 64)
- model `allenai/OLMoE-1B-7B-0924-Instruct` @ `7f1c97f440f0`; tokens sha `5c1d5386d339`; N=60; fixture template alpaca seq 2048 micro-batch 2 × accum 4 lr 0.0002 r 16 α 16 optimizer adamw_8bit(lr=0.0002, weight_decay=0.001) schedule=linear warmup_steps=5 autocast False; e4b trainable 121634816; box_class RTX 5090 gpu NVIDIA GeForce RTX 5090
| framework | arm | status | validity | **VERDICT** | matched | init / dtype | step-0 class | N | s/step med(11+) | tok/s | peak GB | J/step | train first→last | held-out 0→final | quality Δ | engagement | trainable | regime | reason / why |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| e4b | fused_attn4_m_dec0 | **OK** | VALID | **VALID** | yes | matched:3407 (complete 2112/2112) / float32 | SAME-BYTES-CLASS (0.0000) | 60 | 1.040 | 1512.5 | 7.672 | 333.4 | 2.8756→0.6999 | 2.8929→0.7292 | 0.0000 | patched 16 / kcalls 256 | 121634816 | 4-bit experts (e4b NF4) + NF4 attention |  |
| e4b | fused_attn4_m_dec1 | **OK** | VALID | **VALID** | yes | matched:3407 (complete 2112/2112) / float32 | NEAR (0.0114) | 60 | 1.018 | 1601.3 | 7.666 | 308.9 | 2.8856→0.7017 | 2.9043→0.7295 | 0.0003 | patched 16 / kcalls 256 | 121634816 | 4-bit experts (e4b NF4) + NF4 attention |  |
| e4b | fused_attn4_m_dec1_d2 | **OK** | VALID | **VALID** | yes | matched:3407 (complete 2112/2112) / float32 | NEAR (0.0114) | 60 | 1.058 | 1555.9 | 7.664 | 308.9 | 2.8856→0.7012 | 2.9043→0.7325 | 0.0033 | patched 16 / kcalls 256 | 121634816 | 4-bit experts (e4b NF4) + NF4 attention |  |
| e4b | fused_attn4_m_dec0_d2 | **OK** | VALID | **VALID** | yes | matched:3407 (complete 2112/2112) / float32 | SAME-BYTES-CLASS (0.0000) | 60 | 1.026 | 1627.3 | 7.668 | 316.0 | 2.8756→0.7009 | 2.8929→0.7286 | -0.0006 | patched 16 / kcalls 256 | 121634816 | 4-bit experts (e4b NF4) + NF4 attention |  |
- prologue `e4b/fused_attn4_m_dec0` **29.1 s** before step 1 (29% of the arm): c1_before 12.7, c1_after 6.6, load_weights 4.9, preamble 4.0; unattributed 1.194; budget 840.0
- prologue `e4b/fused_attn4_m_dec1` **27.1 s** before step 1 (29% of the arm): c1_before 13.1, c1_after 6.5, load_weights 4.8, preamble 3.9; unattributed 1.176; budget 840.0
- prologue `e4b/fused_attn4_m_dec1_d2` **27.3 s** before step 1 (29% of the arm): c1_before 13.3, c1_after 6.6, load_weights 4.8, preamble 3.9; unattributed 1.21; budget 840.0
- prologue `e4b/fused_attn4_m_dec0_d2` **26.9 s** before step 1 (30% of the arm): c1_before 13.0, c1_after 6.5, load_weights 4.8, preamble 3.9; unattributed 1.183; budget 840.0
- draws (R1): `e4b/fused_attn4_m_dec0` STABLE (1.040/1.026 s, |Δ|/mean 1.3% vs 5%); `e4b/fused_attn4_m_dec1` STABLE (1.018/1.058 s, |Δ|/mean 3.9% vs 5%)
- e4b internal parity: NO-ARM
- **NO MATCHED POSITION QUOTED (unsloth)** — not quoted: e4b STABLE / unsloth no receipt
- **NO MATCHED POSITION QUOTED (hf)** — not quoted: e4b STABLE / hf no receipt
- **NO MATCHED POSITION QUOTED (axolotl)** — not quoted: e4b STABLE / axolotl no receipt
- equivalence vs `e4b/fused_attn4_m` (I/H; EQUIVALENT iff both ≤ band = max(0.005, 3 × fused-vs-reference |Δ|) = none (no reference arm: COMPARABLE is the ceiling); COMPARABLE ≤ 0.05; draw-noise floor 0.0030): `e4b/fused_attn4_m_dec1` **COMPARABLE** (median step |Δ| 0.0013, |Δ held-out at N| 0.0003, step-0 0.0114 NEAR, |Δ loss at step 2| 0.0075, paired rows mean +0.0003 ± 0.0011 SE over 8, favouring arm 4 / anchor 4) — no reference arm ran: the EQUIVALENT band cannot be set, COMPARABLE is the ceiling; `e4b/fused_attn4_m_dec1_d2` **COMPARABLE** (median step |Δ| 0.0013, |Δ held-out at N| 0.0033, step-0 0.0114 NEAR, |Δ loss at step 2| 0.0088, paired rows mean +0.0033 ± 0.0018 SE over 8, favouring arm 2 / anchor 6) — no reference arm ran: the EQUIVALENT band cannot be set, COMPARABLE is the ceiling; `e4b/fused_attn4_m_dec0_d2` **COMPARABLE** (median step |Δ| 0.0009, |Δ held-out at N| 0.0006, step-0 0.0000 SAME-BYTES-CLASS, |Δ loss at step 2| 0.0005, paired rows mean -0.0006 ± 0.0013 SE over 8, favouring arm 5 / anchor 3) — no reference arm ran: the EQUIVALENT band cannot be set, COMPARABLE is the ceiling
- matched_init_sha (B, name-free, canonical slot order): anchor `6f306a59541d98c8`; `e4b/fused_attn4_m_dec0` same; `e4b/fused_attn4_m_dec1` same; `e4b/fused_attn4_m_dec1_d2` same; `e4b/fused_attn4_m_dec0_d2` same
- frozen base (R5), anchor `e4b/fused_attn4_m`: gate_up nf4/64 sha 65b09e0d8053 control detects, down nf4/64 sha 38efe81dfa95 control detects, q_proj nf4/64+dq sha a089b2a76723 control detects
- frozen base `e4b/fused_attn4_m_dec1`: gate_up **SAME-BYTES** (nf4/64) control detects, down **SAME-BYTES** (nf4/64) control detects, q_proj **SAME-BYTES** (nf4/64+dq) control detects
- frozen base `e4b/fused_attn4_m_dec1_d2`: gate_up **SAME-BYTES** (nf4/64) control detects, down **SAME-BYTES** (nf4/64) control detects, q_proj **SAME-BYTES** (nf4/64+dq) control detects
- frozen base `e4b/fused_attn4_m_dec0_d2`: gate_up **SAME-BYTES** (nf4/64) control detects, down **SAME-BYTES** (nf4/64) control detects, q_proj **SAME-BYTES** (nf4/64+dq) control detects

### Qwen3-30B-A3B (amendment 46: grouped-nf4-gemm's fused 4-bit kernels vs its decoded route, matched arm, venv-e4b) (`qwen3decab`, registered n_layers 48, attention census 192)
- model `Qwen/Qwen3-30B-A3B` @ `ad44e777bcd1`; tokens sha `bfc742f67e37`; N=60; fixture template alpaca seq 2048 micro-batch 2 × accum 4 lr 0.0002 r 16 α 16 optimizer adamw_8bit(lr=0.0002, weight_decay=0.001) schedule=linear warmup_steps=5 autocast False; e4b trainable 642514944; box_class RTX 5090 gpu NVIDIA GeForce RTX 5090
| framework | arm | status | validity | **VERDICT** | matched | init / dtype | step-0 class | N | s/step med(11+) | tok/s | peak GB | J/step | train first→last | held-out 0→final | quality Δ | engagement | trainable | regime | reason / why |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| e4b | fused_attn4_m_dec0 | **OK** | VALID | **VALID** | yes | matched:3407 (complete 12480/12480) / float32 | SAME-BYTES-CLASS (0.0000) | 60 | 3.682 | 425.4 | 27.445 | 1061.8 | 2.0614→0.8171 | 1.9441→0.7604 | 0.0000 | patched 48 / kcalls 768 | 642514944 | 4-bit experts (e4b NF4) + NF4 attention |  |
| e4b | fused_attn4_m_dec1 | **OK** | VALID | **VALID** | yes | matched:3407 (complete 12480/12480) / float32 | SAME-BYTES-CLASS (0.0094) | 60 | 3.943 | 403.2 | 27.443 | 1066.1 | 2.0802→0.8150 | 1.9535→0.7570 | -0.0035 | patched 48 / kcalls 768 | 642514944 | 4-bit experts (e4b NF4) + NF4 attention |  |
| e4b | fused_attn4_m_dec1_d2 | **OK** | VALID | **VALID** | yes | matched:3407 (complete 12480/12480) / float32 | SAME-BYTES-CLASS (0.0094) | 60 | 3.868 | 406.9 | 27.448 | 1042.5 | 2.0802→0.8216 | 1.9535→0.7582 | -0.0022 | patched 48 / kcalls 768 | 642514944 | 4-bit experts (e4b NF4) + NF4 attention |  |
| e4b | fused_attn4_m_dec0_d2 | **OK** | VALID | **VALID** | yes | matched:3407 (complete 12480/12480) / float32 | SAME-BYTES-CLASS (0.0000) | 60 | 3.647 | 434.5 | 27.446 | 1026.4 | 2.0614→0.8197 | 1.9441→0.7578 | -0.0027 | patched 48 / kcalls 768 | 642514944 | 4-bit experts (e4b NF4) + NF4 attention |  |
- prologue `e4b/fused_attn4_m_dec0` **115.6 s** before step 1 (34% of the arm): c1_before 62.7, load_weights 30.6, c1_after 29.8, adapter_save 7.2; unattributed 1.156; budget 1260.0
- prologue `e4b/fused_attn4_m_dec1` **105.4 s** before step 1 (30% of the arm): c1_before 62.8, c1_after 29.2, load_weights 23.0, adapter_save 6.2; unattributed 1.204; budget 1260.0
- prologue `e4b/fused_attn4_m_dec1_d2` **107.4 s** before step 1 (31% of the arm): c1_before 64.4, c1_after 31.6, load_weights 22.9, attn4 6.1; unattributed 1.228; budget 1260.0
- prologue `e4b/fused_attn4_m_dec0_d2` **105.5 s** before step 1 (32% of the arm): c1_before 63.5, c1_after 32.1, load_weights 22.8, attn4 5.8; unattributed 1.222; budget 1260.0
- draws (R1): `e4b/fused_attn4_m_dec0` STABLE (3.682/3.647 s, |Δ|/mean 1.0% vs 5%); `e4b/fused_attn4_m_dec1` STABLE (3.943/3.868 s, |Δ|/mean 1.9% vs 5%)
- e4b internal parity: NO-ARM
- **NO MATCHED POSITION QUOTED (unsloth)** — not quoted: e4b STABLE / unsloth no receipt
- **NO MATCHED POSITION QUOTED (hf)** — not quoted: e4b STABLE / hf no receipt
- **NO MATCHED POSITION QUOTED (axolotl)** — not quoted: e4b STABLE / axolotl no receipt
- equivalence vs `e4b/fused_attn4_m` (I/H; EQUIVALENT iff both ≤ band = max(0.005, 3 × fused-vs-reference |Δ|) = none (no reference arm: COMPARABLE is the ceiling); COMPARABLE ≤ 0.05; draw-noise floor 0.0027): `e4b/fused_attn4_m_dec1` **COMPARABLE** (median step |Δ| 0.0014, |Δ held-out at N| 0.0035, step-0 0.0094 SAME-BYTES-CLASS, |Δ loss at step 2| 0.0019, paired rows mean -0.0035 ± 0.0020 SE over 8, favouring arm 5 / anchor 3) — no reference arm ran: the EQUIVALENT band cannot be set, COMPARABLE is the ceiling; `e4b/fused_attn4_m_dec1_d2` **COMPARABLE** (median step |Δ| 0.0013, |Δ held-out at N| 0.0022, step-0 0.0094 SAME-BYTES-CLASS, |Δ loss at step 2| 0.0146, paired rows mean -0.0022 ± 0.0019 SE over 8, favouring arm 6 / anchor 2) — no reference arm ran: the EQUIVALENT band cannot be set, COMPARABLE is the ceiling; `e4b/fused_attn4_m_dec0_d2` **COMPARABLE** (median step |Δ| 0.0012, |Δ held-out at N| 0.0027, step-0 0.0000 SAME-BYTES-CLASS, |Δ loss at step 2| 0.0111, paired rows mean -0.0027 ± 0.0014 SE over 8, favouring arm 6 / anchor 2) — no reference arm ran: the EQUIVALENT band cannot be set, COMPARABLE is the ceiling
- matched_init_sha (B, name-free, canonical slot order): anchor `f7832488926eda91`; `e4b/fused_attn4_m_dec0` same; `e4b/fused_attn4_m_dec1` same; `e4b/fused_attn4_m_dec1_d2` same; `e4b/fused_attn4_m_dec0_d2` same
- frozen base (R5), anchor `e4b/fused_attn4_m`: gate_up nf4/64 sha 8c3b3fd78e89 control detects, down nf4/64 sha 59569d810a1a control detects, q_proj nf4/64+dq sha 5c5e402ce51d control detects
- frozen base `e4b/fused_attn4_m_dec1`: gate_up **SAME-BYTES** (nf4/64) control detects, down **SAME-BYTES** (nf4/64) control detects, q_proj **SAME-BYTES** (nf4/64+dq) control detects
- frozen base `e4b/fused_attn4_m_dec1_d2`: gate_up **SAME-BYTES** (nf4/64) control detects, down **SAME-BYTES** (nf4/64) control detects, q_proj **SAME-BYTES** (nf4/64+dq) control detects
- frozen base `e4b/fused_attn4_m_dec0_d2`: gate_up **SAME-BYTES** (nf4/64) control detects, down **SAME-BYTES** (nf4/64) control detects, q_proj **SAME-BYTES** (nf4/64+dq) control detects

## Verdicts
| family | arm | VERDICT | validity | quality | note |
|---|---|---|---|---|---|
| olmoedecab | e4b/fused_attn4_m_dec0 | **VALID** | VALID | 0.0000 |  |
| olmoedecab | e4b/fused_attn4_m_dec1 | **VALID** | VALID | 0.0003 |  |
| olmoedecab | e4b/fused_attn4_m_dec1_d2 | **VALID** | VALID | 0.0033 |  |
| olmoedecab | e4b/fused_attn4_m_dec0_d2 | **VALID** | VALID | -0.0006 |  |
| qwen3decab | e4b/fused_attn4_m_dec0 | **VALID** | VALID | 0.0000 |  |
| qwen3decab | e4b/fused_attn4_m_dec1 | **VALID** | VALID | -0.0035 |  |
| qwen3decab | e4b/fused_attn4_m_dec1_d2 | **VALID** | VALID | -0.0022 |  |
| qwen3decab | e4b/fused_attn4_m_dec0_d2 | **VALID** | VALID | -0.0027 |  |

## Predictions P107 / P108 / P109 / P110 / P111 (TC1-PREREG amendment 46: grouped-nf4-gemm's decoded route vs its fused kernels, the sm_120 gate first, two stable draws a side; scored mechanically)
| prediction | family | verdict | evidence |
|---|---|---|---|
| P107 | decgate | **HELD** | grouped-nf4-gemm @aaefbf8d1a20 on NVIDIA GeForce RTX 5090 (torch 2.8.0+cu128, triton 3.4.0): test_nf4_route.py -k "decoded or cap_bounds or dequant_groups" rc 0: 24 tests, 0 failed, 0 errors, 0 skipped; test_nf4_route_decision.py rc 0: 38 tests, 0 failed, 0 errors, 0 skipped |
| P108 | olmoedecab | **FALSIFIED** | OLMoE-1B-7B: dec1 / dec0 1.005 [0.979, 1.031 over 4 cross-draw ratios] vs <= 0.95 on the median and every cross-draw ratio; s/step dec0 1.040 / 1.026 (within 1.3%), dec1 1.018 / 1.058 (within 3.9%); dec1 route counts (process) {"decoded_dgrad": 7680, "decoded_fwd": 16384, "dense_dgrad": 0, "dense_fwd": 0, "dgrad": 0, "fwd": 0} |
| P109 | qwen3decab | **HELD** | Qwen3-30B-A3B: dec1 / dec0 1.066 [1.050, 1.081 over 4 cross-draw ratios] vs in [0.97, 1.25]; s/step dec0 3.682 / 3.647 (within 1.0%), dec1 3.943 / 3.868 (within 1.9%); dec1 route counts (process) {"decoded_dgrad": 23040, "decoded_fwd": 49152, "dense_dgrad": 0, "dense_fwd": 0, "dgrad": 0, "fwd": 0} |
| P110 | decodedab | **HELD** | olmoedecab: mean held-out dec1 - dec0 +0.0021 (bound 0.01); held-out at N dec0 [0.7292, 0.7286] dec1 [0.7295, 0.7325]; qwen3decab: mean held-out dec1 - dec0 -0.0015 (bound 0.01); held-out at N dec0 [0.7604, 0.7578] dec1 [0.757, 0.7582] |
| P111 | decodedab | **HELD** | olmoedecab: peak dec1 - dec0 GB -0.0050 (bound 0.3); peak dec0 7.67 / dec1 7.67 GB; qwen3decab: peak dec1 - dec0 GB +0.0000 (bound 0.3); peak dec0 27.45 / dec1 27.45 GB |

## Load gate (TC1-PREREG amendment 33): 0 draw(s) voided for host load and run again
Each line: the arm, the attempt, the median host load1 over that attempt's run, the gate. A VOID attempt's files are in loadvoid/; the last attempt of each arm stands whatever its load.

- `olmoedecab/e4b/fused_attn4_m_dec0 attempt 0 load1_median 2.96 gate 6.0 status ok over 0`
- `olmoedecab/e4b/fused_attn4_m_dec1 attempt 0 load1_median 2.28 gate 6.0 status ok over 0`
- `olmoedecab/e4b/fused_attn4_m_dec1_d2 attempt 0 load1_median 2.59 gate 6.0 status ok over 0`
- `olmoedecab/e4b/fused_attn4_m_dec0_d2 attempt 0 load1_median 2.34 gate 6.0 status ok over 0`
- `qwen3decab/e4b/fused_attn4_m_dec0 attempt 0 load1_median 2.43 gate 6.0 status ok over 0`
- `qwen3decab/e4b/fused_attn4_m_dec1 attempt 0 load1_median 2.13 gate 6.0 status ok over 0`
- `qwen3decab/e4b/fused_attn4_m_dec1_d2 attempt 0 load1_median 3.1 gate 6.0 status ok over 0`
- `qwen3decab/e4b/fused_attn4_m_dec0_d2 attempt 0 load1_median 1.94 gate 6.0 status ok over 0`
