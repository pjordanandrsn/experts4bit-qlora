# TC1 — Qwen3-30B-A3B on one RTX 5090: e4b vs Unsloth vs HF+PEFT vs axolotl at matched work, init and adapter precision (tc1)
Rule (tc1/TC1-PREREG.md): status per attempt in the vocabulary OK / REFUSED / OOM / INSTALL_FAILED / LOAD_FAULT / HARNESS_ERROR / ALARM / NOT_RUN; VALID/VOID per tp4's predicates plus the matched-set predicates (R3: matched init complete, fp32 adapters, step-0 held-out within 0.005 of e4b/reference_attn4_m); VERDICT exactly one of VALID / VOID / QUALITY_FAIL / OOM / UNSUPPORTED / HARNESS_ERROR / ALARM / NOT_RUN (QUALITY_FAIL = held-out |Δ| at N > 0.05 vs e4b/fused_attn4_m); positions = s/step ratios other/e4b from the medians over both draws (stability |d1-d2|/mean ≤ 5 % e4b / 10 % others, else UNSTABLE), quoted only when both arms are VALID; equivalence vs e4b/reference_attn4_m EQUIVALENT ≤ 0.02 / COMPARABLE ≤ 0.05 / else DIVERGENT; frozen base SAME-BYTES / DIFFERENT / N-A per slot; predictions P1–P10 (+ P1b) scored HELD / FALSIFIED / UNTESTED. VOID never enters a ratio or an equivalence reading. Nothing is licensed; no cross-box number is divided into these.
`versions.txt`
```
e4b 0.48.0 @655f626e86cdd872f67242d54c2fe023872c8253 (GitHub main)
gnf4 0.42.0 @0e6bff33a56a0277b39a0560c37e5af9bee23cf6 (GitHub main)
torch(e4b/hf) 2.8.0+cu128
triton(e4b/hf) 3.4.0
transformers(e4b/hf) 5.18.0
bitsandbytes(e4b/hf) 0.50.2
peft(hf) 0.21.2
unsloth(unsloth-t28) 2026.9.14
unsloth_zoo(unsloth-t28) 2026.9.9
torch(unsloth-t28) 2.8.0+cu128
triton(unsloth-t28) 3.4.0
transformers(unsloth-t28) 5.5.0
bitsandbytes(unsloth-t28) 0.50.2
peft(unsloth-t28) 0.21.2
torchao(unsloth-t28) None
moe_backend(unsloth-t28) native_torch
unsloth(unsloth) 2026.9.14
unsloth_zoo(unsloth) 2026.9.9
torch(unsloth) 2.12.1+cu130
triton(unsloth) 3.7.1
transformers(unsloth) 5.5.0
bitsandbytes(unsloth) 0.50.2
peft(unsloth) 0.21.2
torchao(unsloth) 0.18.0+cu130
moe_backend(unsloth) grouped_mm
e4b(t212) 0.48.0 @655f626e86cdd872f67242d54c2fe023872c8253
gnf4(t212) 0.42.0 @0e6bff33a56a0277b39a0560c37e5af9bee23cf6
torch(e4b-t212) 2.12.1+cu130
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
 "run_id": "tc1-5090-120",
 "instance_id": "54597943",
 "gpu": "NVIDIA GeForce RTX 5090",
 "driver": "595.91.07",
 "cpu": "AMD Ryzen 9 9950X 16-Core Processor",
 "nproc": 32,
 "mem_total_kb": "129453600",
 "cgroup_memory_max": "127257280512",
 "disk_root": "overlay         320G  2.5M  320G   1% /",
 "hostname": "0c27e452b94f",
 "cgroup_cpu_max": "3071999 100000",
 "affinity_cpus": 32,
 "cgroup_cpuset_effective": "0-31",
 "registered_gpu_class": "5090",
 "prereg": "tc1/TC1-PREREG.md",
 "local_box": null,
 "local_snapshot": null
}
```

### Qwen3-30B-A3B (amendment 59: checkpoint inputs on the GPU vs in pinned host memory at the field recipe, shipped and matched arms; peaks by phase) (`qwen3ckptofff`, registered n_layers 48, attention census 192)
- model `Qwen/Qwen3-30B-A3B` @ `ad44e777bcd1`; tokens sha `bfc742f67e37`; N=60; fixture template alpaca seq 2048 micro-batch 2 × accum 4 lr 0.0002 r 16 α 16 optimizer adamw_8bit(lr=0.0002, weight_decay=0.001) schedule=linear warmup_steps=5 autocast False; e4b trainable 642514944; box_class RTX 5090 gpu NVIDIA GeForce RTX 5090
| framework | arm | status | validity | **VERDICT** | matched | init / dtype | step-0 class | N | s/step med(11+) | tok/s | peak GB | J/step | train first→last | held-out 0→final | quality Δ | engagement | trainable | regime | reason / why |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| e4b | fused_attn4_shipped_f0 | **OK** | VALID | **VALID** | native | native / bfloat16,float32 | — | 60 | 1.713 | 875.0 | 23.321 | 694.6 | 2.0722→0.8170 | 1.9592→0.7564 | -0.0000 | patched 48 / kcalls 768 | 642514944 | 4-bit experts (e4b NF4) + NF4 attention |  |
| e4b | fused_attn4_shipped_f1 | **OK** | VALID | **VALID** | native | native / bfloat16,float32 | — | 60 | 1.554 | 1030.9 | 23.104 | 647.0 | 2.0722→0.8199 | 1.9592→0.7537 | -0.0027 | patched 48 / kcalls 768 | 642514944 | 4-bit experts (e4b NF4) + NF4 attention |  |
| e4b | fused_attn4_m_f0 | **OK** | VALID | **VALID** | yes | matched:3407 (complete 12480/12480) / float32 | SAME-BYTES-CLASS (0.0000) | 60 | 2.150 | 734.0 | 26.148 | 810.3 | 2.0722→0.8172 | 1.9592→0.7564 | 0.0000 | patched 48 / kcalls 768 | 642514944 | 4-bit experts (e4b NF4) + NF4 attention |  |
| e4b | fused_attn4_m_f1 | **OK** | VALID | **VALID** | yes | matched:3407 (complete 12480/12480) / float32 | SAME-BYTES-CLASS (0.0000) | 60 | 2.040 | 792.0 | 25.978 | 808.1 | 2.0722→0.8190 | 1.9592→0.7564 | 0.0000 | patched 48 / kcalls 768 | 642514944 | 4-bit experts (e4b NF4) + NF4 attention |  |
| e4b | fused_attn4_m_f1_d2 | **OK** | VALID | **VALID** | yes | matched:3407 (complete 12480/12480) / float32 | SAME-BYTES-CLASS (0.0000) | 60 | 2.032 | 795.0 | 25.968 | 806.6 | 2.0722→0.8199 | 1.9592→0.7573 | 0.0008 | patched 48 / kcalls 768 | 642514944 | 4-bit experts (e4b NF4) + NF4 attention |  |
| e4b | fused_attn4_m_f0_d2 | **OK** | VALID | **VALID** | yes | matched:3407 (complete 12480/12480) / float32 | SAME-BYTES-CLASS (0.0000) | 60 | 2.147 | 734.9 | 26.139 | 812.4 | 2.0722→0.8189 | 1.9592→0.7570 | 0.0006 | patched 48 / kcalls 768 | 642514944 | 4-bit experts (e4b NF4) + NF4 attention |  |
| e4b | fused_attn4_shipped_f1_d2 | **OK** | VALID | **VALID** | native | native / bfloat16,float32 | — | 60 | 1.558 | 1029.3 | 23.104 | 646.6 | 2.0722→0.8157 | 1.9592→0.7546 | -0.0018 | patched 48 / kcalls 768 | 642514944 | 4-bit experts (e4b NF4) + NF4 attention |  |
| e4b | fused_attn4_shipped_f0_d2 | **OK** | VALID | **VALID** | native | native / bfloat16,float32 | — | 60 | 1.686 | 933.1 | 23.321 | 646.0 | 2.0722→0.8197 | 1.9592→0.7577 | 0.0012 | patched 48 / kcalls 768 | 642514944 | 4-bit experts (e4b NF4) + NF4 attention |  |
- prologue `e4b/fused_attn4_shipped_f0` **57.3 s** before step 1 (34% of the arm): c1_before 29.1, c1_after 14.6, load_weights 10.4, eval0 7.5; unattributed 1.154; budget 1260.0
- prologue `e4b/fused_attn4_shipped_f1` **50.6 s** before step 1 (35% of the arm): c1_before 29.0, c1_after 14.6, load_weights 10.5, attn4 2.9; unattributed 1.158; budget 1260.0
- prologue `e4b/fused_attn4_m_f0` **51.8 s** before step 1 (28% of the arm): c1_before 29.2, c1_after 14.7, load_weights 10.4, attn4 2.9; unattributed 1.15; budget 1260.0
- prologue `e4b/fused_attn4_m_f1` **51.6 s** before step 1 (30% of the arm): c1_before 29.1, c1_after 14.7, load_weights 10.3, attn4 2.9; unattributed 1.162; budget 1260.0
- prologue `e4b/fused_attn4_m_f1_d2` **51.7 s** before step 1 (30% of the arm): c1_before 29.1, c1_after 14.6, load_weights 10.4, attn4 2.9; unattributed 1.165; budget 1260.0
- prologue `e4b/fused_attn4_m_f0_d2` **51.7 s** before step 1 (28% of the arm): c1_before 29.1, c1_after 14.7, load_weights 10.4, attn4 2.9; unattributed 1.165; budget 1260.0
- prologue `e4b/fused_attn4_shipped_f1_d2` **50.6 s** before step 1 (35% of the arm): c1_before 29.1, c1_after 14.7, load_weights 10.3, attn4 2.9; unattributed 1.162; budget 1260.0
- prologue `e4b/fused_attn4_shipped_f0_d2` **50.7 s** before step 1 (33% of the arm): c1_before 29.2, c1_after 14.6, load_weights 10.3, attn4 2.9; unattributed 1.167; budget 1260.0
- draws (R1): `e4b/fused_attn4_shipped_f0` STABLE (1.713/1.686 s, |Δ|/mean 1.6% vs 5%); `e4b/fused_attn4_shipped_f1` STABLE (1.554/1.558 s, |Δ|/mean 0.2% vs 5%); `e4b/fused_attn4_m_f0` STABLE (2.150/2.147 s, |Δ|/mean 0.1% vs 5%); `e4b/fused_attn4_m_f1` STABLE (2.040/2.032 s, |Δ|/mean 0.4% vs 5%)
- e4b internal parity: NO-ARM
- **NO MATCHED POSITION QUOTED (unsloth)** — not quoted: e4b STABLE / unsloth no receipt
- **NO MATCHED POSITION QUOTED (hf)** — not quoted: e4b STABLE / hf no receipt
- **NO MATCHED POSITION QUOTED (axolotl)** — not quoted: e4b STABLE / axolotl no receipt
- equivalence vs `e4b/fused_attn4_m` (I/H; EQUIVALENT iff both ≤ band = max(0.005, 3 × fused-vs-reference |Δ|) = none (no reference arm: COMPARABLE is the ceiling); COMPARABLE ≤ 0.05; draw-noise floor 0.0013): `e4b/fused_attn4_m_f1` **COMPARABLE** (median step |Δ| 0.0012, |Δ held-out at N| 0.0000, step-0 0.0000 SAME-BYTES-CLASS, |Δ loss at step 2| 0.0054, paired rows mean +0.0000 ± 0.0013 SE over 8, favouring arm 3 / anchor 5) — no reference arm ran: the EQUIVALENT band cannot be set, COMPARABLE is the ceiling; `e4b/fused_attn4_m_f1_d2` **COMPARABLE** (median step |Δ| 0.0012, |Δ held-out at N| 0.0008, step-0 0.0000 SAME-BYTES-CLASS, |Δ loss at step 2| 0.0106, paired rows mean +0.0009 ± 0.0013 SE over 8, favouring arm 2 / anchor 6) — no reference arm ran: the EQUIVALENT band cannot be set, COMPARABLE is the ceiling; `e4b/fused_attn4_m_f0_d2` **COMPARABLE** (median step |Δ| 0.0015, |Δ held-out at N| 0.0006, step-0 0.0000 SAME-BYTES-CLASS, |Δ loss at step 2| 0.0041, paired rows mean +0.0006 ± 0.0021 SE over 8, favouring arm 4 / anchor 4) — no reference arm ran: the EQUIVALENT band cannot be set, COMPARABLE is the ceiling
- matched_init_sha (B, name-free, canonical slot order): anchor `f7832488926eda91`; `e4b/fused_attn4_m_f0` same; `e4b/fused_attn4_m_f1` same; `e4b/fused_attn4_m_f1_d2` same; `e4b/fused_attn4_m_f0_d2` same
- frozen base (R5), anchor `e4b/fused_attn4_m`: gate_up nf4/64+dq sha e2bed944143e control detects, down nf4/64+dq sha f2ade29dafdb control detects, q_proj nf4/64+dq sha 5c5e402ce51d control detects
- frozen base `e4b/fused_attn4_shipped_f0`: gate_up **SAME-BYTES** (nf4/64+dq) control detects, down **SAME-BYTES** (nf4/64+dq) control detects, q_proj **SAME-BYTES** (nf4/64+dq) control detects
- frozen base `e4b/fused_attn4_shipped_f1`: gate_up **SAME-BYTES** (nf4/64+dq) control detects, down **SAME-BYTES** (nf4/64+dq) control detects, q_proj **SAME-BYTES** (nf4/64+dq) control detects
- frozen base `e4b/fused_attn4_m_f1`: gate_up **SAME-BYTES** (nf4/64+dq) control detects, down **SAME-BYTES** (nf4/64+dq) control detects, q_proj **SAME-BYTES** (nf4/64+dq) control detects
- frozen base `e4b/fused_attn4_m_f1_d2`: gate_up **SAME-BYTES** (nf4/64+dq) control detects, down **SAME-BYTES** (nf4/64+dq) control detects, q_proj **SAME-BYTES** (nf4/64+dq) control detects
- frozen base `e4b/fused_attn4_m_f0_d2`: gate_up **SAME-BYTES** (nf4/64+dq) control detects, down **SAME-BYTES** (nf4/64+dq) control detects, q_proj **SAME-BYTES** (nf4/64+dq) control detects
- frozen base `e4b/fused_attn4_shipped_f1_d2`: gate_up **SAME-BYTES** (nf4/64+dq) control detects, down **SAME-BYTES** (nf4/64+dq) control detects, q_proj **SAME-BYTES** (nf4/64+dq) control detects
- frozen base `e4b/fused_attn4_shipped_f0_d2`: gate_up **SAME-BYTES** (nf4/64+dq) control detects, down **SAME-BYTES** (nf4/64+dq) control detects, q_proj **SAME-BYTES** (nf4/64+dq) control detects

## Verdicts
| family | arm | VERDICT | validity | quality | note |
|---|---|---|---|---|---|
| qwen3ckptofff | e4b/fused_attn4_shipped_f0 | **VALID** | VALID | -0.0000 |  |
| qwen3ckptofff | e4b/fused_attn4_shipped_f1 | **VALID** | VALID | -0.0027 |  |
| qwen3ckptofff | e4b/fused_attn4_m_f0 | **VALID** | VALID | 0.0000 |  |
| qwen3ckptofff | e4b/fused_attn4_m_f1 | **VALID** | VALID | 0.0000 |  |
| qwen3ckptofff | e4b/fused_attn4_m_f1_d2 | **VALID** | VALID | 0.0008 |  |
| qwen3ckptofff | e4b/fused_attn4_m_f0_d2 | **VALID** | VALID | 0.0006 |  |
| qwen3ckptofff | e4b/fused_attn4_shipped_f1_d2 | **VALID** | VALID | -0.0018 |  |
| qwen3ckptofff | e4b/fused_attn4_shipped_f0_d2 | **VALID** | VALID | 0.0012 |  |

## Amendment 59: checkpoint inputs on the GPU vs in pinned host memory at the field recipe, peaks by phase (descriptive)
| arm | VERDICT | s/step (11..N) | run peak GB | setup | eval | train |
|---|---|---|---|---|---|---|
| e4b/fused_attn4_shipped_f0 | VALID | 1.713 | 23.321 | 21.856 | 20.236 | 23.321 |
| e4b/fused_attn4_shipped_f1 | VALID | 1.554 | 23.104 | 21.856 | 20.236 | 23.104 |
| e4b/fused_attn4_m_f0 | VALID | 2.150 | 26.148 | 21.856 | 21.507 | 26.148 |
| e4b/fused_attn4_m_f1 | VALID | 2.040 | 25.978 | 21.856 | 21.517 | 25.978 |
| e4b/fused_attn4_m_f1_d2 | VALID | 2.032 | 25.968 | 21.856 | 21.513 | 25.968 |
| e4b/fused_attn4_m_f0_d2 | VALID | 2.147 | 26.139 | 21.856 | 21.51 | 26.139 |
| e4b/fused_attn4_shipped_f1_d2 | VALID | 1.558 | 23.104 | 21.856 | 20.236 | 23.104 |
| e4b/fused_attn4_shipped_f0_d2 | VALID | 1.686 | 23.321 | 21.856 | 20.236 | 23.321 |

## Predictions P158 / P159 / P160 / P161 (TC1-PREREG amendment 59: checkpoint inputs in pinned host memory at the field recipe; scored mechanically)
| prediction | family | verdict | evidence |
|---|---|---|---|
| P158 | qwen3ckptofff | **HELD** | m: f1 / f0 0.948 [0.945, 0.951 over 4 cross-draw ratios] vs <= 1.01; s/step f0 2.150 / 2.147, f1 2.040 / 2.032 |
| P159 | qwen3ckptofff | **HELD** | shipped: f1 / f0 0.916 [0.907, 0.924 over 4 cross-draw ratios] vs <= 1.01; s/step f0 1.713 / 1.686, f1 1.554 / 1.558 |
| P160 | qwen3ckptofff | **HELD** | mean held-out at N f1 - f0: m +0.0001, shipped -0.0029 (|.| <= 0.005) |
| P161 | qwen3ckptofff | **HELD** | matched training-phase peak f0 26.143 -> f1 25.973 GB (drop +0.171 vs >= 0.1) |

## Load gate (TC1-PREREG amendment 33): 0 draw(s) voided for host load and run again
Each line: the arm, the attempt, the median host load1 over that attempt's run, the gate. A VOID attempt's files are in loadvoid/; the last attempt of each arm stands whatever its load.

- `qwen3ckptofff/e4b/fused_attn4_shipped_f0 attempt 0 load1_median 0.86 gate 6.0 status ok over 0`
- `qwen3ckptofff/e4b/fused_attn4_shipped_f1 attempt 0 load1_median 1.43 gate 6.0 status ok over 0`
- `qwen3ckptofff/e4b/fused_attn4_m_f0 attempt 0 load1_median 1.14 gate 6.0 status ok over 0`
- `qwen3ckptofff/e4b/fused_attn4_m_f1 attempt 0 load1_median 1.04 gate 6.0 status ok over 0`
- `qwen3ckptofff/e4b/fused_attn4_m_f1_d2 attempt 0 load1_median 1.06 gate 6.0 status ok over 0`
- `qwen3ckptofff/e4b/fused_attn4_m_f0_d2 attempt 0 load1_median 1.1 gate 6.0 status ok over 0`
- `qwen3ckptofff/e4b/fused_attn4_shipped_f1_d2 attempt 0 load1_median 1.06 gate 6.0 status ok over 0`
- `qwen3ckptofff/e4b/fused_attn4_shipped_f0_d2 attempt 0 load1_median 1.82 gate 6.0 status ok over 0`
