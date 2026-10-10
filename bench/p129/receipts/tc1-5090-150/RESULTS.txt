# TC1 — Qwen3-30B-A3B on one RTX 5090: e4b vs Unsloth vs HF+PEFT vs axolotl at matched work, init and adapter precision (tc1)
Rule (tc1/TC1-PREREG.md): status per attempt in the vocabulary OK / REFUSED / OOM / INSTALL_FAILED / LOAD_FAULT / HARNESS_ERROR / ALARM / NOT_RUN; VALID/VOID per tp4's predicates plus the matched-set predicates (R3: matched init complete, fp32 adapters, step-0 held-out within 0.005 of e4b/reference_attn4_m); VERDICT exactly one of VALID / VOID / QUALITY_FAIL / OOM / UNSUPPORTED / HARNESS_ERROR / ALARM / NOT_RUN (QUALITY_FAIL = held-out |Δ| at N > 0.05 vs e4b/fused_attn4_m); positions = s/step ratios other/e4b from the medians over both draws (stability |d1-d2|/mean ≤ 5 % e4b / 10 % others, else UNSTABLE), quoted only when both arms are VALID; equivalence vs e4b/reference_attn4_m EQUIVALENT ≤ 0.02 / COMPARABLE ≤ 0.05 / else DIVERGENT; frozen base SAME-BYTES / DIFFERENT / N-A per slot; predictions P1–P10 (+ P1b) scored HELD / FALSIFIED / UNTESTED. VOID never enters a ratio or an equivalence reading. Nothing is licensed; no cross-box number is divided into these.
`versions.txt`
```
e4b 0.52.0 @2d0ed913867b9cbed42af1ccc9f3bfc47b6dd6c1 (GitHub main)
gnf4 0.44.0 @d1f64ba50afce94533e0166ba3332ef075aa43bb (GitHub main)
torch(e4b/hf) 2.8.0+cu128
triton(e4b/hf) 3.4.0
transformers(e4b/hf) 5.18.0
bitsandbytes(e4b/hf) 0.50.2
peft(hf) 0.21.2
huggingface_hub(e4b/hf) 2.2.0
httpx2(e4b/hf) 2.13.1
brotli(e4b/hf) 1.2.0
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
e4b(t212) 0.52.0 @2d0ed913867b9cbed42af1ccc9f3bfc47b6dd6c1
gnf4(t212) 0.44.0 @d1f64ba50afce94533e0166ba3332ef075aa43bb
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
 "run_id": "tc1-5090-150",
 "instance_id": "55149329",
 "gpu": "NVIDIA GeForce RTX 5090",
 "driver": "595.91.07",
 "cpu": "AMD EPYC 7K62 48-Core Processor",
 "nproc": 96,
 "mem_total_kb": "527994556",
 "cgroup_memory_max": "259519414272",
 "disk_root": "overlay         320G   58M  320G   1% /",
 "hostname": "bee71551dd30",
 "cgroup_cpu_max": "2304000 100000",
 "affinity_cpus": 96,
 "cgroup_cpuset_effective": "0-95",
 "registered_gpu_class": "5090",
 "prereg": "tc1/TC1-PREREG.md",
 "local_box": null,
 "local_snapshot": null
}
```

### Qwen3-30B-A3B (P129 Amendment 3: E4B_TRAIN_FUSE_QKV 0 vs 1 at the field recipe, shipped and matched arms; profiled; the step-0 clause against the box's fp32-anchored floor) (`qwen3fqkv3`, registered n_layers 48, attention census 192)
- model `Qwen/Qwen3-30B-A3B` @ `ad44e777bcd1`; tokens sha `bfc742f67e37`; N=60; fixture template alpaca seq 2048 micro-batch 2 × accum 4 lr 0.0002 r 16 α 16 optimizer adamw_8bit(lr=0.0002, weight_decay=0.001) schedule=linear warmup_steps=5 autocast False; e4b trainable 642514944; box_class RTX 5090 gpu NVIDIA GeForce RTX 5090
| framework | arm | status | validity | **VERDICT** | matched | init / dtype | step-0 class | N | s/step med(11+) | tok/s | peak GB | J/step | train first→last | held-out 0→final | quality Δ | engagement | trainable | regime | reason / why |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| e4b | fused_attn4_shipped_q0 | **OK** | VALID | **VALID** | native | native / bfloat16,float32 | — | 60 | 3.231 | 456.2 | 23.321 | 556.1 | 2.0722→0.8164 | 1.9592→0.7554 | -0.0032 | patched 48 / kcalls 768 | 642514944 | 4-bit experts (e4b NF4) + NF4 attention |  |
| e4b | fused_attn4_shipped_q1 | **OK** | VALID | **VALID** | native | native / bfloat16,float32 | — | 60 | 2.809 | 557.0 | 23.344 | 501.4 | 2.0463→0.8177 | 1.9468→0.7553 | -0.0032 | patched 48 / kcalls 768 | 642514944 | 4-bit experts (e4b NF4) + NF4 attention |  |
| e4b | fused_attn4_m_q0 | **OK** | VALID | **VALID** | yes | matched:3407 (complete 12480/12480) / float32 | SAME-BYTES-CLASS (0.0000) | 60 | 3.319 | 467.8 | 26.495 | 645.5 | 2.0722→0.8196 | 1.9592→0.7586 | 0.0000 | patched 48 / kcalls 768 | 642514944 | 4-bit experts (e4b NF4) + NF4 attention |  |
| e4b | fused_attn4_m_q1 | **OK** | VALID | **VALID** | yes | matched:3407 (complete 12480/12480) / float32 | NEAR (0.0123) | 60 | 3.012 | 514.3 | 26.500 | 623.9 | 2.0463→0.8233 | 1.9468→0.7602 | 0.0016 | patched 48 / kcalls 768 | 642514944 | 4-bit experts (e4b NF4) + NF4 attention |  |
| e4b | fused_attn4_m_q1_d2 | **OK** | VALID | **VALID** | yes | matched:3407 (complete 12480/12480) / float32 | NEAR (0.0123) | 60 | 2.998 | 516.1 | 26.500 | 603.1 | 2.0463→0.8258 | 1.9468→0.7553 | -0.0032 | patched 48 / kcalls 768 | 642514944 | 4-bit experts (e4b NF4) + NF4 attention |  |
| e4b | fused_attn4_m_q0_d2 | **OK** | VALID | **VALID** | yes | matched:3407 (complete 12480/12480) / float32 | SAME-BYTES-CLASS (0.0000) | 60 | 3.313 | 467.1 | 26.495 | 608.0 | 2.0722→0.8162 | 1.9592→0.7533 | -0.0052 | patched 48 / kcalls 768 | 642514944 | 4-bit experts (e4b NF4) + NF4 attention |  |
| e4b | fused_attn4_shipped_q1_d2 | **OK** | VALID | **VALID** | native | native / bfloat16,float32 | — | 60 | 2.831 | 550.2 | 23.344 | 504.7 | 2.0463→0.8185 | 1.9468→0.7564 | -0.0021 | patched 48 / kcalls 768 | 642514944 | 4-bit experts (e4b NF4) + NF4 attention |  |
| e4b | fused_attn4_shipped_q0_d2 | **OK** | VALID | **VALID** | native | native / bfloat16,float32 | — | 60 | 3.153 | 495.5 | 23.321 | 515.9 | 2.0722→0.8198 | 1.9592→0.7549 | -0.0036 | patched 48 / kcalls 768 | 642514944 | 4-bit experts (e4b NF4) + NF4 attention |  |
- prologue `e4b/fused_attn4_shipped_q0` **133.7 s** before step 1 (24% of the arm): c1_before 67.9, load_weights 21.1, c1_after 18.7, eval0 16.2; unattributed 2.406; budget 1260.0
- prologue `e4b/fused_attn4_shipped_q1` **66.8 s** before step 1 (16% of the arm): c1_before 23.7, load_weights 21.0, c1_after 18.7, attn4 7.3; unattributed 2.234; budget 1260.0
- prologue `e4b/fused_attn4_m_q0` **121.9 s** before step 1 (22% of the arm): c1_before 67.9, load_weights 20.8, c1_after 18.7, qkv_floor 8.1; unattributed 2.28; budget 1260.0
- prologue `e4b/fused_attn4_m_q1` **67.3 s** before step 1 (15% of the arm): c1_before 23.7, load_weights 21.0, c1_after 18.4, attn4 7.3; unattributed 2.291; budget 1260.0
- prologue `e4b/fused_attn4_m_q1_d2` **67.4 s** before step 1 (15% of the arm): c1_before 23.4, load_weights 20.9, c1_after 18.9, attn4 7.4; unattributed 2.254; budget 1260.0
- prologue `e4b/fused_attn4_m_q0_d2` **120.6 s** before step 1 (22% of the arm): c1_before 68.6, load_weights 20.7, c1_after 18.8, qkv_floor 7.9; unattributed 2.198; budget 1260.0
- prologue `e4b/fused_attn4_shipped_q1_d2` **66.6 s** before step 1 (15% of the arm): c1_before 23.2, load_weights 20.9, c1_after 18.8, attn4 7.4; unattributed 2.255; budget 1260.0
- prologue `e4b/fused_attn4_shipped_q0_d2` **119.2 s** before step 1 (23% of the arm): c1_before 68.0, load_weights 20.8, c1_after 19.1, qkv_floor 7.6; unattributed 2.286; budget 1260.0
- draws (R1): `e4b/fused_attn4_shipped_q0` STABLE (3.231/3.153 s, |Δ|/mean 2.4% vs 5%); `e4b/fused_attn4_shipped_q1` STABLE (2.809/2.831 s, |Δ|/mean 0.8% vs 5%); `e4b/fused_attn4_m_q0` STABLE (3.319/3.313 s, |Δ|/mean 0.2% vs 5%); `e4b/fused_attn4_m_q1` STABLE (3.012/2.998 s, |Δ|/mean 0.5% vs 5%)
- e4b internal parity: NO-ARM
- **NO MATCHED POSITION QUOTED (unsloth)** — not quoted: e4b STABLE / unsloth no receipt
- **NO MATCHED POSITION QUOTED (hf)** — not quoted: e4b STABLE / hf no receipt
- **NO MATCHED POSITION QUOTED (axolotl)** — not quoted: e4b STABLE / axolotl no receipt
- equivalence vs `e4b/fused_attn4_m` (I/H; EQUIVALENT iff both ≤ band = max(0.005, 3 × fused-vs-reference |Δ|) = none (no reference arm: COMPARABLE is the ceiling); COMPARABLE ≤ 0.05; draw-noise floor 0.0052): `e4b/fused_attn4_m_q1` **COMPARABLE** (median step |Δ| 0.0012, |Δ held-out at N| 0.0016, step-0 0.0123 NEAR, |Δ loss at step 2| 0.0161, paired rows mean +0.0016 ± 0.0021 SE over 8, favouring arm 4 / anchor 4) — no reference arm ran: the EQUIVALENT band cannot be set, COMPARABLE is the ceiling; `e4b/fused_attn4_m_q1_d2` **COMPARABLE** (median step |Δ| 0.0014, |Δ held-out at N| 0.0032, step-0 0.0123 NEAR, |Δ loss at step 2| 0.0008, paired rows mean -0.0032 ± 0.0019 SE over 8, favouring arm 6 / anchor 2) — no reference arm ran: the EQUIVALENT band cannot be set, COMPARABLE is the ceiling; `e4b/fused_attn4_m_q0_d2` **COMPARABLE** (median step |Δ| 0.0015, |Δ held-out at N| 0.0052, step-0 0.0000 SAME-BYTES-CLASS, |Δ loss at step 2| 0.0150, paired rows mean -0.0052 ± 0.0016 SE over 8, favouring arm 7 / anchor 1) — no reference arm ran: the EQUIVALENT band cannot be set, COMPARABLE is the ceiling
- matched_init_sha (B, name-free, canonical slot order): anchor `f7832488926eda91`; `e4b/fused_attn4_m_q0` same; `e4b/fused_attn4_m_q1` same; `e4b/fused_attn4_m_q1_d2` same; `e4b/fused_attn4_m_q0_d2` same
- frozen base (R5), anchor `e4b/fused_attn4_m`: gate_up nf4/64+dq sha e2bed944143e control detects, down nf4/64+dq sha f2ade29dafdb control detects, q_proj nf4/64+dq sha 5c5e402ce51d control detects
- frozen base `e4b/fused_attn4_shipped_q0`: gate_up **SAME-BYTES** (nf4/64+dq) control detects, down **SAME-BYTES** (nf4/64+dq) control detects, q_proj **SAME-BYTES** (nf4/64+dq) control detects
- frozen base `e4b/fused_attn4_shipped_q1`: gate_up **SAME-BYTES** (nf4/64+dq) control detects, down **SAME-BYTES** (nf4/64+dq) control detects, q_proj **N-A** (slot missing on this arm) control FAILS
- frozen base `e4b/fused_attn4_m_q1`: gate_up **SAME-BYTES** (nf4/64+dq) control detects, down **SAME-BYTES** (nf4/64+dq) control detects, q_proj **N-A** (slot missing on this arm) control FAILS
- frozen base `e4b/fused_attn4_m_q1_d2`: gate_up **SAME-BYTES** (nf4/64+dq) control detects, down **SAME-BYTES** (nf4/64+dq) control detects, q_proj **N-A** (slot missing on this arm) control FAILS
- frozen base `e4b/fused_attn4_m_q0_d2`: gate_up **SAME-BYTES** (nf4/64+dq) control detects, down **SAME-BYTES** (nf4/64+dq) control detects, q_proj **SAME-BYTES** (nf4/64+dq) control detects
- frozen base `e4b/fused_attn4_shipped_q1_d2`: gate_up **SAME-BYTES** (nf4/64+dq) control detects, down **SAME-BYTES** (nf4/64+dq) control detects, q_proj **N-A** (slot missing on this arm) control FAILS
- frozen base `e4b/fused_attn4_shipped_q0_d2`: gate_up **SAME-BYTES** (nf4/64+dq) control detects, down **SAME-BYTES** (nf4/64+dq) control detects, q_proj **SAME-BYTES** (nf4/64+dq) control detects

## Verdicts
| family | arm | VERDICT | validity | quality | note |
|---|---|---|---|---|---|
| qwen3fqkv3 | e4b/fused_attn4_shipped_q0 | **VALID** | VALID | -0.0032 |  |
| qwen3fqkv3 | e4b/fused_attn4_shipped_q1 | **VALID** | VALID | -0.0032 |  |
| qwen3fqkv3 | e4b/fused_attn4_m_q0 | **VALID** | VALID | 0.0000 |  |
| qwen3fqkv3 | e4b/fused_attn4_m_q1 | **VALID** | VALID | 0.0016 |  |
| qwen3fqkv3 | e4b/fused_attn4_m_q1_d2 | **VALID** | VALID | -0.0032 |  |
| qwen3fqkv3 | e4b/fused_attn4_m_q0_d2 | **VALID** | VALID | -0.0052 |  |
| qwen3fqkv3 | e4b/fused_attn4_shipped_q1_d2 | **VALID** | VALID | -0.0021 |  |
| qwen3fqkv3 | e4b/fused_attn4_shipped_q0_d2 | **VALID** | VALID | -0.0036 |  |

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
