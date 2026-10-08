# TC1 — Qwen3-30B-A3B on one RTX 5090: e4b vs Unsloth vs HF+PEFT vs axolotl at matched work, init and adapter precision (tc1)
Rule (tc1/TC1-PREREG.md): status per attempt in the vocabulary OK / REFUSED / OOM / INSTALL_FAILED / LOAD_FAULT / HARNESS_ERROR / ALARM / NOT_RUN; VALID/VOID per tp4's predicates plus the matched-set predicates (R3: matched init complete, fp32 adapters, step-0 held-out within 0.005 of e4b/reference_attn4_m); VERDICT exactly one of VALID / VOID / QUALITY_FAIL / OOM / UNSUPPORTED / HARNESS_ERROR / ALARM / NOT_RUN (QUALITY_FAIL = held-out |Δ| at N > 0.05 vs e4b/fused_attn4_m); positions = s/step ratios other/e4b from the medians over both draws (stability |d1-d2|/mean ≤ 5 % e4b / 10 % others, else UNSTABLE), quoted only when both arms are VALID; equivalence vs e4b/reference_attn4_m EQUIVALENT ≤ 0.02 / COMPARABLE ≤ 0.05 / else DIVERGENT; frozen base SAME-BYTES / DIFFERENT / N-A per slot; predictions P1–P10 (+ P1b) scored HELD / FALSIFIED / UNTESTED. VOID never enters a ratio or an equivalence reading. Nothing is licensed; no cross-box number is divided into these.
`versions.txt`
```
e4b 0.49.0 @822884d7a6513fa69e7cf2b43cddddc0b9416bee (GitHub main)
gnf4 0.43.0 @f8b4a0a8ebc0093834015745dea3307f7a05de21 (GitHub main)
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
e4b(t212) 0.49.0 @822884d7a6513fa69e7cf2b43cddddc0b9416bee
gnf4(t212) 0.43.0 @f8b4a0a8ebc0093834015745dea3307f7a05de21
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
 "run_id": "tc1-5090-138",
 "instance_id": "54853789",
 "gpu": "NVIDIA GeForce RTX 5090",
 "driver": "595.91.07",
 "cpu": "AMD EPYC 7K62 48-Core Processor",
 "nproc": 96,
 "mem_total_kb": "527994556",
 "cgroup_memory_max": "259519414272",
 "disk_root": "overlay         320G   57M  320G   1% /",
 "hostname": "bc054401501e",
 "cgroup_cpu_max": "2304000 100000",
 "affinity_cpus": 96,
 "cgroup_cpuset_effective": "0-95",
 "registered_gpu_class": "5090",
 "prereg": "tc1/TC1-PREREG.md",
 "local_box": null,
 "local_snapshot": null
}
```

### Qwen3-30B-A3B (amendment 70: grouped-nf4-gemm's single padded block off vs on its ladder at the field recipe, shipped and matched arms; profiled) (`qwen3sladder`, registered n_layers 48, attention census 192)
- model `Qwen/Qwen3-30B-A3B` @ `ad44e777bcd1`; tokens sha `bfc742f67e37`; N=60; fixture template alpaca seq 2048 micro-batch 2 × accum 4 lr 0.0002 r 16 α 16 optimizer adamw_8bit(lr=0.0002, weight_decay=0.001) schedule=linear warmup_steps=5 autocast False; e4b trainable 642514944; box_class RTX 5090 gpu NVIDIA GeForce RTX 5090
| framework | arm | status | validity | **VERDICT** | matched | init / dtype | step-0 class | N | s/step med(11+) | tok/s | peak GB | J/step | train first→last | held-out 0→final | quality Δ | engagement | trainable | regime | reason / why |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| e4b | fused_attn4_shipped_l0 | **OK** | VALID | **VALID** | native | native / bfloat16,float32 | — | 60 | 3.231 | 461.6 | 23.321 | 508.9 | 2.0722→0.8194 | 1.9592→0.7552 | -0.0035 | patched 48 / kcalls 768 | 642514944 | 4-bit experts (e4b NF4) + NF4 attention |  |
| e4b | fused_attn4_shipped_l1 | **OK** | VALID | **VALID** | native | native / bfloat16,float32 | — | 60 | 3.251 | 484.1 | 23.321 | 430.9 | 2.0722→0.8177 | 1.9592→0.7558 | -0.0029 | patched 48 / kcalls 768 | 642514944 | 4-bit experts (e4b NF4) + NF4 attention |  |
| e4b | fused_attn4_m_l0 | **OK** | VALID | **VALID** | yes | matched:3407 (complete 12480/12480) / float32 | SAME-BYTES-CLASS (0.0000) | 60 | 4.137 | 381.4 | 26.168 | 639.6 | 2.0722→0.8184 | 1.9592→0.7587 | 0.0000 | patched 48 / kcalls 768 | 642514944 | 4-bit experts (e4b NF4) + NF4 attention |  |
| e4b | fused_attn4_m_l1 | **OK** | VALID | **VALID** | yes | matched:3407 (complete 12480/12480) / float32 | SAME-BYTES-CLASS (0.0000) | 60 | 3.286 | 475.1 | 26.495 | 601.6 | 2.0722→0.8156 | 1.9592→0.7581 | -0.0006 | patched 48 / kcalls 768 | 642514944 | 4-bit experts (e4b NF4) + NF4 attention |  |
| e4b | fused_attn4_m_l1_d2 | **OK** | VALID | **VALID** | yes | matched:3407 (complete 12480/12480) / float32 | SAME-BYTES-CLASS (0.0000) | 60 | 3.334 | 470.0 | 26.495 | 592.1 | 2.0722→0.8221 | 1.9592→0.7572 | -0.0015 | patched 48 / kcalls 768 | 642514944 | 4-bit experts (e4b NF4) + NF4 attention |  |
| e4b | fused_attn4_m_l0_d2 | **OK** | VALID | **VALID** | yes | matched:3407 (complete 12480/12480) / float32 | SAME-BYTES-CLASS (0.0000) | 60 | 4.167 | 378.4 | 26.156 | 678.3 | 2.0722→0.8178 | 1.9592→0.7556 | -0.0031 | patched 48 / kcalls 768 | 642514944 | 4-bit experts (e4b NF4) + NF4 attention |  |
| e4b | fused_attn4_shipped_l1_d2 | **OK** | VALID | **VALID** | native | native / bfloat16,float32 | — | 60 | 3.280 | 479.0 | 23.321 | 501.6 | 2.0722→0.8173 | 1.9592→0.7529 | -0.0059 | patched 48 / kcalls 768 | 642514944 | 4-bit experts (e4b NF4) + NF4 attention |  |
| e4b | fused_attn4_shipped_l0_d2 | **OK** | VALID | **VALID** | native | native / bfloat16,float32 | — | 60 | 3.201 | 491.6 | 23.321 | 492.0 | 2.0722→0.8135 | 1.9592→0.7561 | -0.0026 | patched 48 / kcalls 768 | 642514944 | 4-bit experts (e4b NF4) + NF4 attention |  |
- prologue `e4b/fused_attn4_shipped_l0` **120.8 s** before step 1 (23% of the arm): c1_before 64.5, load_weights 20.9, c1_after 18.6, eval0 15.6; unattributed 2.243; budget 1260.0
- prologue `e4b/fused_attn4_shipped_l1` **112.9 s** before step 1 (22% of the arm): c1_before 64.9, load_weights 20.7, c1_after 18.5, attn4 12.1; unattributed 2.357; budget 1260.0
- prologue `e4b/fused_attn4_m_l0` **110.2 s** before step 1 (19% of the arm): c1_before 66.6, load_weights 20.7, c1_after 18.5, attn4 7.2; unattributed 2.215; budget 1260.0
- prologue `e4b/fused_attn4_m_l1` **109.1 s** before step 1 (21% of the arm): c1_before 65.7, load_weights 20.8, c1_after 18.4, attn4 7.2; unattributed 2.229; budget 1260.0
- prologue `e4b/fused_attn4_m_l1_d2` **109.6 s** before step 1 (21% of the arm): c1_before 66.0, load_weights 20.8, c1_after 18.3, attn4 7.2; unattributed 2.176; budget 1260.0
- prologue `e4b/fused_attn4_m_l0_d2` **108.6 s** before step 1 (19% of the arm): c1_before 64.8, load_weights 20.8, c1_after 18.3, attn4 7.2; unattributed 2.226; budget 1260.0
- prologue `e4b/fused_attn4_shipped_l1_d2` **109.3 s** before step 1 (21% of the arm): c1_before 66.5, load_weights 20.8, c1_after 18.5, attn4 7.2; unattributed 2.344; budget 1260.0
- prologue `e4b/fused_attn4_shipped_l0_d2` **110.8 s** before step 1 (22% of the arm): c1_before 67.6, load_weights 20.8, c1_after 18.5, attn4 7.4; unattributed 2.348; budget 1260.0
- draws (R1): `e4b/fused_attn4_shipped_l0` STABLE (3.231/3.201 s, |Δ|/mean 0.9% vs 5%); `e4b/fused_attn4_shipped_l1` STABLE (3.251/3.280 s, |Δ|/mean 0.9% vs 5%); `e4b/fused_attn4_m_l0` STABLE (4.137/4.167 s, |Δ|/mean 0.7% vs 5%); `e4b/fused_attn4_m_l1` STABLE (3.286/3.334 s, |Δ|/mean 1.4% vs 5%)
- e4b internal parity: NO-ARM
- **NO MATCHED POSITION QUOTED (unsloth)** — not quoted: e4b STABLE / unsloth no receipt
- **NO MATCHED POSITION QUOTED (hf)** — not quoted: e4b STABLE / hf no receipt
- **NO MATCHED POSITION QUOTED (axolotl)** — not quoted: e4b STABLE / axolotl no receipt
- equivalence vs `e4b/fused_attn4_m` (I/H; EQUIVALENT iff both ≤ band = max(0.005, 3 × fused-vs-reference |Δ|) = none (no reference arm: COMPARABLE is the ceiling); COMPARABLE ≤ 0.05; draw-noise floor 0.0031): `e4b/fused_attn4_m_l1` **COMPARABLE** (median step |Δ| 0.0010, |Δ held-out at N| 0.0006, step-0 0.0000 SAME-BYTES-CLASS, |Δ loss at step 2| 0.0119, paired rows mean -0.0006 ± 0.0017 SE over 8, favouring arm 4 / anchor 4) — no reference arm ran: the EQUIVALENT band cannot be set, COMPARABLE is the ceiling; `e4b/fused_attn4_m_l1_d2` **COMPARABLE** (median step |Δ| 0.0016, |Δ held-out at N| 0.0015, step-0 0.0000 SAME-BYTES-CLASS, |Δ loss at step 2| 0.0088, paired rows mean -0.0015 ± 0.0021 SE over 8, favouring arm 3 / anchor 5) — no reference arm ran: the EQUIVALENT band cannot be set, COMPARABLE is the ceiling; `e4b/fused_attn4_m_l0_d2` **COMPARABLE** (median step |Δ| 0.0012, |Δ held-out at N| 0.0031, step-0 0.0000 SAME-BYTES-CLASS, |Δ loss at step 2| 0.0013, paired rows mean -0.0031 ± 0.0027 SE over 8, favouring arm 4 / anchor 4) — no reference arm ran: the EQUIVALENT band cannot be set, COMPARABLE is the ceiling
- matched_init_sha (B, name-free, canonical slot order): anchor `f7832488926eda91`; `e4b/fused_attn4_m_l0` same; `e4b/fused_attn4_m_l1` same; `e4b/fused_attn4_m_l1_d2` same; `e4b/fused_attn4_m_l0_d2` same
- frozen base (R5), anchor `e4b/fused_attn4_m`: gate_up nf4/64+dq sha e2bed944143e control detects, down nf4/64+dq sha f2ade29dafdb control detects, q_proj nf4/64+dq sha 5c5e402ce51d control detects
- frozen base `e4b/fused_attn4_shipped_l0`: gate_up **SAME-BYTES** (nf4/64+dq) control detects, down **SAME-BYTES** (nf4/64+dq) control detects, q_proj **SAME-BYTES** (nf4/64+dq) control detects
- frozen base `e4b/fused_attn4_shipped_l1`: gate_up **SAME-BYTES** (nf4/64+dq) control detects, down **SAME-BYTES** (nf4/64+dq) control detects, q_proj **SAME-BYTES** (nf4/64+dq) control detects
- frozen base `e4b/fused_attn4_m_l1`: gate_up **SAME-BYTES** (nf4/64+dq) control detects, down **SAME-BYTES** (nf4/64+dq) control detects, q_proj **SAME-BYTES** (nf4/64+dq) control detects
- frozen base `e4b/fused_attn4_m_l1_d2`: gate_up **SAME-BYTES** (nf4/64+dq) control detects, down **SAME-BYTES** (nf4/64+dq) control detects, q_proj **SAME-BYTES** (nf4/64+dq) control detects
- frozen base `e4b/fused_attn4_m_l0_d2`: gate_up **SAME-BYTES** (nf4/64+dq) control detects, down **SAME-BYTES** (nf4/64+dq) control detects, q_proj **SAME-BYTES** (nf4/64+dq) control detects
- frozen base `e4b/fused_attn4_shipped_l1_d2`: gate_up **SAME-BYTES** (nf4/64+dq) control detects, down **SAME-BYTES** (nf4/64+dq) control detects, q_proj **SAME-BYTES** (nf4/64+dq) control detects
- frozen base `e4b/fused_attn4_shipped_l0_d2`: gate_up **SAME-BYTES** (nf4/64+dq) control detects, down **SAME-BYTES** (nf4/64+dq) control detects, q_proj **SAME-BYTES** (nf4/64+dq) control detects

## Verdicts
| family | arm | VERDICT | validity | quality | note |
|---|---|---|---|---|---|
| qwen3sladder | e4b/fused_attn4_shipped_l0 | **VALID** | VALID | -0.0035 |  |
| qwen3sladder | e4b/fused_attn4_shipped_l1 | **VALID** | VALID | -0.0029 |  |
| qwen3sladder | e4b/fused_attn4_m_l0 | **VALID** | VALID | 0.0000 |  |
| qwen3sladder | e4b/fused_attn4_m_l1 | **VALID** | VALID | -0.0006 |  |
| qwen3sladder | e4b/fused_attn4_m_l1_d2 | **VALID** | VALID | -0.0015 |  |
| qwen3sladder | e4b/fused_attn4_m_l0_d2 | **VALID** | VALID | -0.0031 |  |
| qwen3sladder | e4b/fused_attn4_shipped_l1_d2 | **VALID** | VALID | -0.0059 |  |
| qwen3sladder | e4b/fused_attn4_shipped_l0_d2 | **VALID** | VALID | -0.0026 |  |

## Amendment 70: the single padded block off vs on its ladder at the field recipe (descriptive)
| arm | VERDICT | s/step (11..N) | device ms / profiled step | busy_t | aten::bmm CPU us / call | laddered calls | peak GB | held-out 0 / N |
|---|---|---|---|---|---|---|---|---|
| e4b/fused_attn4_shipped_l0 | VALID | 3.231 | 1347.7 | 0.417 | 27.9 | 0 | 23.321 | 1.95917 / 0.7552 |
| e4b/fused_attn4_shipped_l1 | VALID | 3.251 | 1389.3 | 0.427 | 23.5 | 49152 | 23.321 | 1.95917 / 0.75578 |
| e4b/fused_attn4_m_l0 | VALID | 4.137 | 1737.2 | 0.420 | 306.0 | 0 | 26.168 | 1.95917 / 0.75871 |
| e4b/fused_attn4_m_l1 | VALID | 3.286 | 1801.2 | 0.548 | 24.6 | 49152 | 26.495 | 1.95917 / 0.75814 |
| e4b/fused_attn4_m_l1_d2 | VALID | 3.334 | 1803.6 | 0.541 | 24.3 | 49152 | 26.495 | 1.95917 / 0.75721 |
| e4b/fused_attn4_m_l0_d2 | VALID | 4.167 | 1724.4 | 0.414 | 304.8 | 0 | 26.156 | 1.95917 / 0.75561 |
| e4b/fused_attn4_shipped_l1_d2 | VALID | 3.280 | 1394.7 | 0.425 | 24.0 | 49152 | 23.321 | 1.95917 / 0.75286 |
| e4b/fused_attn4_shipped_l0_d2 | VALID | 3.201 | 1349.8 | 0.422 | 27.6 | 0 | 23.321 | 1.95917 / 0.75614 |

## Predictions P210-P214 (TC1-PREREG amendment 70: the single-block ladder at the field recipe; scored mechanically)
| prediction | family | verdict | evidence |
|---|---|---|---|
| P210 | qwen3sladder | **HELD** | l1 / l0 0.797 [0.789, 0.806 over 4 cross-draw ratios] vs <= 0.97; s/step l1 3.286 / 3.334, l0 4.137 / 4.167; matched l0 busy_t 0.417 (premise <= 0.85) |
| P211 | qwen3sladder | **FALSIFIED** | l1 / l0 1.015 [1.006, 1.025 over 4 cross-draw ratios] vs <= 0.97; s/step l1 3.251 / 3.280, l0 3.231 / 3.201; matched l0 busy_t 0.417 (premise <= 0.85) |
| P212 | qwen3sladder | **HELD** | aten::bmm CPU self per call 305.4 -> 24.5 us = 0.080 vs <= 0.5 (matched) |
| P213 | qwen3sladder | **HELD** | device ms per profiled step l1 / l0: m: 1730.8 -> 1802.4 ms = 1.041; shipped: 1348.7 -> 1392.0 ms = 1.032 (each <= 1.05) |
| P214 | qwen3sladder | **HELD** | m: step 0 +0.00000, +0.00000; N +0.00052; shipped: step 0 +0.00000, +0.00000; N -0.00135 (step 0 |.| <= 0.0005, N |.| <= 0.005) |

## Load gate (TC1-PREREG amendment 33): 2 draw(s) voided for host load and run again
Each line: the arm, the attempt, the median host load1 over that attempt's run, the gate. A VOID attempt's files are in loadvoid/; the last attempt of each arm stands whatever its load.

- `qwen3sladder/e4b/fused_attn4_shipped_l0 attempt 0 load1_median 3.4 gate 6.0 status ok over 0`
- `qwen3sladder/e4b/fused_attn4_shipped_l1 attempt 0 load1_median 3.51 gate 6.0 status ok over 0`
- `qwen3sladder/e4b/fused_attn4_m_l0 attempt 0 load1_median 3.96 gate 6.0 status ok over 0`
- `qwen3sladder/e4b/fused_attn4_m_l1 attempt 0 load1_median 3.4 gate 6.0 status ok over 0`
- `qwen3sladder/e4b/fused_attn4_m_l1_d2 attempt 0 load1_median 4.9 gate 6.0 status ok over 0`
- `qwen3sladder/e4b/fused_attn4_m_l0_d2 attempt 0 load1_median 4.94 gate 6.0 status ok over 0`
- `qwen3sladder/e4b/fused_attn4_shipped_l1_d2 attempt 0 load1_median 7.26 gate 6.0 status ok over 1`
- `qwen3sladder/e4b/fused_attn4_shipped_l1_d2 attempt 0 VOID (host load1 median 7.26 > 6.0): re-run 1 of 2`
- `qwen3sladder/e4b/fused_attn4_shipped_l1_d2 attempt 1 load1_median 8.22 gate 6.0 status ok over 1`
- `qwen3sladder/e4b/fused_attn4_shipped_l1_d2 attempt 1 VOID (host load1 median 8.22 > 6.0): re-run 2 of 2`
- `qwen3sladder/e4b/fused_attn4_shipped_l1_d2 attempt 2 load1_median 6.43 gate 6.0 status ok over 1`
- `qwen3sladder/e4b/fused_attn4_shipped_l0_d2 attempt 0 load1_median 3.55 gate 6.0 status ok over 0`
