# TC1 — Qwen3-30B-A3B on one RTX 5090: e4b vs Unsloth vs HF+PEFT vs axolotl at matched work, init and adapter precision (tc1)
Rule (tc1/TC1-PREREG.md): status per attempt in the vocabulary OK / REFUSED / OOM / INSTALL_FAILED / LOAD_FAULT / HARNESS_ERROR / ALARM / NOT_RUN; VALID/VOID per tp4's predicates plus the matched-set predicates (R3: matched init complete, fp32 adapters, step-0 held-out within 0.005 of e4b/reference_attn4_m); VERDICT exactly one of VALID / VOID / QUALITY_FAIL / OOM / UNSUPPORTED / HARNESS_ERROR / ALARM / NOT_RUN (QUALITY_FAIL = held-out |Δ| at N > 0.05 vs e4b/fused_attn4_m); positions = s/step ratios other/e4b from the medians over both draws (stability |d1-d2|/mean ≤ 5 % e4b / 10 % others, else UNSTABLE), quoted only when both arms are VALID; equivalence vs e4b/reference_attn4_m EQUIVALENT ≤ 0.02 / COMPARABLE ≤ 0.05 / else DIVERGENT; frozen base SAME-BYTES / DIFFERENT / N-A per slot; predictions P1–P10 (+ P1b) scored HELD / FALSIFIED / UNTESTED. VOID never enters a ratio or an equivalence reading. Nothing is licensed; no cross-box number is divided into these.
`versions.txt`
```
e4b 0.51.0 @51e5ae1b5b10f47e2dafe16ef92cb97489e4669d (GitHub main)
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
e4b(t212) 0.51.0 @51e5ae1b5b10f47e2dafe16ef92cb97489e4669d
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
 "run_id": "tc1-5090-147",
 "instance_id": "55102607",
 "gpu": "NVIDIA GeForce RTX 5090",
 "driver": "580.82.09",
 "cpu": "AMD EPYC 7B13 64-Core Processor",
 "nproc": 128,
 "mem_total_kb": "527975568",
 "cgroup_memory_max": "",
 "disk_root": "overlay         320G   81M  320G   1% /",
 "hostname": "3ea83242a53b",
 "cgroup_cpu_max": "",
 "affinity_cpus": 128,
 "cgroup_cpuset_effective": "",
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
| e4b | fused_attn4_shipped_q0 | **OK** | VALID | **VALID** | native | native / bfloat16,float32 | — | 60 | 3.902 | 403.9 | 23.321 | 472.9 | 2.0722→0.8172 | 1.9592→0.7571 | 0.0013 | patched 48 / kcalls 768 | 642514944 | 4-bit experts (e4b NF4) + NF4 attention |  |
| e4b | fused_attn4_shipped_q1 | **OK** | VALID | **VALID** | native | native / bfloat16,float32 | — | 60 | 3.364 | 467.8 | 23.344 | 449.0 | 2.0463→0.8166 | 1.9468→0.7582 | 0.0024 | patched 48 / kcalls 768 | 642514944 | 4-bit experts (e4b NF4) + NF4 attention |  |
| e4b | fused_attn4_m_q0 | **OK** | VALID | **VALID** | yes | matched:3407 (complete 12480/12480) / float32 | SAME-BYTES-CLASS (0.0000) | 60 | 3.985 | 395.0 | 26.495 | 495.2 | 2.0722→0.8169 | 1.9592→0.7558 | 0.0000 | patched 48 / kcalls 768 | 642514944 | 4-bit experts (e4b NF4) + NF4 attention |  |
| e4b | fused_attn4_m_q1 | **OK** | VALID | **VALID** | yes | matched:3407 (complete 12480/12480) / float32 | NEAR (0.0123) | 60 | 3.546 | 443.6 | 26.500 | 559.0 | 2.0463→0.8193 | 1.9468→0.7576 | 0.0017 | patched 48 / kcalls 768 | 642514944 | 4-bit experts (e4b NF4) + NF4 attention |  |
| e4b | fused_attn4_m_q1_d2 | **OK** | VALID | **VALID** | yes | matched:3407 (complete 12480/12480) / float32 | NEAR (0.0123) | 60 | 3.555 | 440.8 | 26.500 | 483.8 | 2.0463→0.8159 | 1.9468→0.7544 | -0.0014 | patched 48 / kcalls 768 | 642514944 | 4-bit experts (e4b NF4) + NF4 attention |  |
| e4b | fused_attn4_m_q0_d2 | **OK** | VALID | **VALID** | yes | matched:3407 (complete 12480/12480) / float32 | SAME-BYTES-CLASS (0.0000) | 60 | 3.944 | 399.3 | 26.495 | 571.9 | 2.0722→0.8220 | 1.9592→0.7562 | 0.0004 | patched 48 / kcalls 768 | 642514944 | 4-bit experts (e4b NF4) + NF4 attention |  |
| e4b | fused_attn4_shipped_q1_d2 | **OK** | VALID | **VALID** | native | native / bfloat16,float32 | — | 60 | 3.385 | 466.6 | 23.344 | 449.3 | 2.0463→0.8185 | 1.9468→0.7565 | 0.0007 | patched 48 / kcalls 768 | 642514944 | 4-bit experts (e4b NF4) + NF4 attention |  |
| e4b | fused_attn4_shipped_q0_d2 | **OK** | VALID | **VALID** | native | native / bfloat16,float32 | — | 60 | 3.869 | 407.0 | 23.321 | 432.1 | 2.0722→0.8165 | 1.9592→0.7564 | 0.0006 | patched 48 / kcalls 768 | 642514944 | 4-bit experts (e4b NF4) + NF4 attention |  |
- prologue `e4b/fused_attn4_shipped_q0` **126.8 s** before step 1 (23% of the arm): c1_before 70.4, load_weights 22.6, c1_after 21.3, qkv_floor 10.0; unattributed 2.511; budget 1260.0
- prologue `e4b/fused_attn4_shipped_q1` **73.4 s** before step 1 (16% of the arm): c1_before 27.2, load_weights 22.9, c1_after 19.7, attn4 8.7; unattributed 2.493; budget 1260.0
- prologue `e4b/fused_attn4_m_q0` **128.8 s** before step 1 (22% of the arm): c1_before 70.3, load_weights 22.9, c1_after 20.0, qkv_floor 10.1; unattributed 2.568; budget 1260.0
- prologue `e4b/fused_attn4_m_q1` **77.5 s** before step 1 (16% of the arm): c1_before 29.7, c1_after 27.7, load_weights 22.9, attn4 8.3; unattributed 2.644; budget 1260.0
- prologue `e4b/fused_attn4_m_q1_d2` **77.7 s** before step 1 (16% of the arm): c1_before 29.6, load_weights 22.8, c1_after 22.5, attn4 8.4; unattributed 2.546; budget 1260.0
- prologue `e4b/fused_attn4_m_q0_d2` **126.4 s** before step 1 (22% of the arm): c1_before 68.2, load_weights 22.8, c1_after 21.0, qkv_floor 10.0; unattributed 2.562; budget 954.4
- prologue `e4b/fused_attn4_shipped_q1_d2` **77.9 s** before step 1 (17% of the arm): c1_before 31.8, load_weights 22.9, c1_after 19.7, attn4 8.6; unattributed 2.553; budget 391.3
- prologue `e4b/fused_attn4_shipped_q0_d2` **125.5 s** before step 1 (23% of the arm): c1_before 68.9, load_weights 23.3, c1_after 20.0, qkv_floor 9.7; unattributed 2.562; budget 217.0
- draws (R1): `e4b/fused_attn4_shipped_q0` STABLE (3.902/3.869 s, |Δ|/mean 0.9% vs 5%); `e4b/fused_attn4_shipped_q1` STABLE (3.364/3.385 s, |Δ|/mean 0.6% vs 5%); `e4b/fused_attn4_m_q0` STABLE (3.985/3.944 s, |Δ|/mean 1.0% vs 5%); `e4b/fused_attn4_m_q1` STABLE (3.546/3.555 s, |Δ|/mean 0.3% vs 5%)
- e4b internal parity: NO-ARM
- **NO MATCHED POSITION QUOTED (unsloth)** — not quoted: e4b STABLE / unsloth no receipt
- **NO MATCHED POSITION QUOTED (hf)** — not quoted: e4b STABLE / hf no receipt
- **NO MATCHED POSITION QUOTED (axolotl)** — not quoted: e4b STABLE / axolotl no receipt
- equivalence vs `e4b/fused_attn4_m` (I/H; EQUIVALENT iff both ≤ band = max(0.005, 3 × fused-vs-reference |Δ|) = none (no reference arm: COMPARABLE is the ceiling); COMPARABLE ≤ 0.05; draw-noise floor 0.0031): `e4b/fused_attn4_m_q1` **COMPARABLE** (median step |Δ| 0.0015, |Δ held-out at N| 0.0017, step-0 0.0123 NEAR, |Δ loss at step 2| 0.0106, paired rows mean +0.0017 ± 0.0036 SE over 8, favouring arm 4 / anchor 4) — no reference arm ran: the EQUIVALENT band cannot be set, COMPARABLE is the ceiling; `e4b/fused_attn4_m_q1_d2` **COMPARABLE** (median step |Δ| 0.0015, |Δ held-out at N| 0.0014, step-0 0.0123 NEAR, |Δ loss at step 2| 0.0104, paired rows mean -0.0014 ± 0.0027 SE over 8, favouring arm 6 / anchor 2) — no reference arm ran: the EQUIVALENT band cannot be set, COMPARABLE is the ceiling; `e4b/fused_attn4_m_q0_d2` **COMPARABLE** (median step |Δ| 0.0012, |Δ held-out at N| 0.0004, step-0 0.0000 SAME-BYTES-CLASS, |Δ loss at step 2| 0.0226, paired rows mean +0.0004 ± 0.0009 SE over 8, favouring arm 3 / anchor 5) — no reference arm ran: the EQUIVALENT band cannot be set, COMPARABLE is the ceiling
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
| qwen3fqkv3 | e4b/fused_attn4_shipped_q0 | **VALID** | VALID | 0.0013 |  |
| qwen3fqkv3 | e4b/fused_attn4_shipped_q1 | **VALID** | VALID | 0.0024 |  |
| qwen3fqkv3 | e4b/fused_attn4_m_q0 | **VALID** | VALID | 0.0000 |  |
| qwen3fqkv3 | e4b/fused_attn4_m_q1 | **VALID** | VALID | 0.0017 |  |
| qwen3fqkv3 | e4b/fused_attn4_m_q1_d2 | **VALID** | VALID | -0.0014 |  |
| qwen3fqkv3 | e4b/fused_attn4_m_q0_d2 | **VALID** | VALID | 0.0004 |  |
| qwen3fqkv3 | e4b/fused_attn4_shipped_q1_d2 | **VALID** | VALID | 0.0007 |  |
| qwen3fqkv3 | e4b/fused_attn4_shipped_q0_d2 | **VALID** | VALID | 0.0006 |  |

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
