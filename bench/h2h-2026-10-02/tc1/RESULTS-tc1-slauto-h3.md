# TC1 — Qwen3-30B-A3B on one RTX 5090: e4b vs Unsloth vs HF+PEFT vs axolotl at matched work, init and adapter precision (tc1)
Rule (tc1/TC1-PREREG.md): status per attempt in the vocabulary OK / REFUSED / OOM / INSTALL_FAILED / LOAD_FAULT / HARNESS_ERROR / ALARM / NOT_RUN; VALID/VOID per tp4's predicates plus the matched-set predicates (R3: matched init complete, fp32 adapters, step-0 held-out within 0.005 of e4b/reference_attn4_m); VERDICT exactly one of VALID / VOID / QUALITY_FAIL / OOM / UNSUPPORTED / HARNESS_ERROR / ALARM / NOT_RUN (QUALITY_FAIL = held-out |Δ| at N > 0.05 vs e4b/fused_attn4_m); positions = s/step ratios other/e4b from the medians over both draws (stability |d1-d2|/mean ≤ 5 % e4b / 10 % others, else UNSTABLE), quoted only when both arms are VALID; equivalence vs e4b/reference_attn4_m EQUIVALENT ≤ 0.02 / COMPARABLE ≤ 0.05 / else DIVERGENT; frozen base SAME-BYTES / DIFFERENT / N-A per slot; predictions P1–P10 (+ P1b) scored HELD / FALSIFIED / UNTESTED. VOID never enters a ratio or an equivalence reading. Nothing is licensed; no cross-box number is divided into these.
`versions.txt`
```
e4b 0.50.0 @9629874fd6273c23260d04a6e29375df21358079 (GitHub main)
gnf4 0.43.0 @3ce2ecd8ae20725a6b13ccc6d06c6a85ba5ce3ac (GitHub main)
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
e4b(t212) 0.50.0 @9629874fd6273c23260d04a6e29375df21358079
gnf4(t212) 0.43.0 @3ce2ecd8ae20725a6b13ccc6d06c6a85ba5ce3ac
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
 "run_id": "tc1-5090-141",
 "instance_id": "54916667",
 "gpu": "NVIDIA GeForce RTX 5090",
 "driver": "595.71.05",
 "cpu": "AMD Ryzen 9 9950X3D 16-Core Processor",
 "nproc": 32,
 "mem_total_kb": "129399712",
 "cgroup_memory_max": "127203803136",
 "disk_root": "overlay         320G   58M  320G   1% /",
 "hostname": "728eb49293a8",
 "cgroup_cpu_max": "3071999 100000",
 "affinity_cpus": 32,
 "cgroup_cpuset_effective": "0-31",
 "registered_gpu_class": "5090",
 "prereg": "tc1/TC1-PREREG.md",
 "local_box": null,
 "local_snapshot": null
}
```

### Qwen3-30B-A3B (amendment 71: NF4_QLORA_SINGLE_LADDER 0 vs auto at the field recipe, shipped and matched arms; profiled) (`qwen3slauto`, registered n_layers 48, attention census 192)
- model `Qwen/Qwen3-30B-A3B` @ `ad44e777bcd1`; tokens sha `bfc742f67e37`; N=60; fixture template alpaca seq 2048 micro-batch 2 × accum 4 lr 0.0002 r 16 α 16 optimizer adamw_8bit(lr=0.0002, weight_decay=0.001) schedule=linear warmup_steps=5 autocast False; e4b trainable 642514944; box_class RTX 5090 gpu NVIDIA GeForce RTX 5090
| framework | arm | status | validity | **VERDICT** | matched | init / dtype | step-0 class | N | s/step med(11+) | tok/s | peak GB | J/step | train first→last | held-out 0→final | quality Δ | engagement | trainable | regime | reason / why |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| e4b | fused_attn4_shipped_l0 | **OK** | VALID | **VALID** | native | native / bfloat16,float32 | — | 60 | 1.594 | 964.7 | 23.321 | 426.2 | 2.0722→0.8156 | 1.9592→0.7558 | -0.0003 | patched 48 / kcalls 768 | 642514944 | 4-bit experts (e4b NF4) + NF4 attention |  |
| e4b | fused_attn4_shipped_la | **OK** | VALID | **VALID** | native | native / bfloat16,float32 | — | 60 | 1.599 | 1006.8 | 23.321 | 365.8 | 2.0722→0.8172 | 1.9592→0.7556 | -0.0005 | patched 48 / kcalls 768 | 642514944 | 4-bit experts (e4b NF4) + NF4 attention |  |
| e4b | fused_attn4_m_l0 | **OK** | VALID | **VALID** | yes | matched:3407 (complete 12480/12480) / float32 | SAME-BYTES-CLASS (0.0000) | 60 | 2.041 | 788.3 | 26.154 | 498.4 | 2.0722→0.8170 | 1.9592→0.7561 | 0.0000 | patched 48 / kcalls 768 | 642514944 | 4-bit experts (e4b NF4) + NF4 attention |  |
| e4b | fused_attn4_m_la | **OK** | VALID | **VALID** | yes | matched:3407 (complete 12480/12480) / float32 | SAME-BYTES-CLASS (0.0000) | 60 | 2.110 | 766.8 | 26.495 | 506.1 | 2.0722→0.8190 | 1.9592→0.7566 | 0.0005 | patched 48 / kcalls 768 | 642514944 | 4-bit experts (e4b NF4) + NF4 attention |  |
| e4b | fused_attn4_m_la_d2 | **OK** | VALID | **VALID** | yes | matched:3407 (complete 12480/12480) / float32 | SAME-BYTES-CLASS (0.0000) | 60 | 2.105 | 768.1 | 26.495 | 482.7 | 2.0722→0.8163 | 1.9592→0.7582 | 0.0021 | patched 48 / kcalls 768 | 642514944 | 4-bit experts (e4b NF4) + NF4 attention |  |
| e4b | fused_attn4_m_l0_d2 | **OK** | VALID | **VALID** | yes | matched:3407 (complete 12480/12480) / float32 | SAME-BYTES-CLASS (0.0000) | 60 | 2.047 | 787.6 | 26.151 | 468.9 | 2.0722→0.8185 | 1.9592→0.7583 | 0.0022 | patched 48 / kcalls 768 | 642514944 | 4-bit experts (e4b NF4) + NF4 attention |  |
| e4b | fused_attn4_shipped_la_d2 | **OK** | VALID | **VALID** | native | native / bfloat16,float32 | — | 60 | 1.598 | 1003.6 | 23.321 | 368.5 | 2.0722→0.8176 | 1.9592→0.7574 | 0.0013 | patched 48 / kcalls 768 | 642514944 | 4-bit experts (e4b NF4) + NF4 attention |  |
| e4b | fused_attn4_shipped_l0_d2 | **OK** | VALID | **VALID** | native | native / bfloat16,float32 | — | 60 | 1.596 | 1004.8 | 23.321 | 351.9 | 2.0722→0.8184 | 1.9592→0.7576 | 0.0015 | patched 48 / kcalls 768 | 642514944 | 4-bit experts (e4b NF4) + NF4 attention |  |
- prologue `e4b/fused_attn4_shipped_l0` **45.7 s** before step 1 (21% of the arm): c1_before 22.2, load_weights 9.6, c1_after 8.6, eval0 5.6; unattributed 0.865; budget 1260.0
- prologue `e4b/fused_attn4_shipped_la` **42.1 s** before step 1 (20% of the arm): c1_before 22.4, load_weights 10.6, c1_after 8.6, preamble 2.8; unattributed 0.868; budget 1260.0
- prologue `e4b/fused_attn4_m_l0` **41.9 s** before step 1 (17% of the arm): c1_before 22.3, load_weights 9.6, c1_after 8.7, preamble 2.8; unattributed 0.866; budget 1260.0
- prologue `e4b/fused_attn4_m_la` **42.1 s** before step 1 (17% of the arm): c1_before 22.5, load_weights 9.6, c1_after 8.7, preamble 2.8; unattributed 0.869; budget 1260.0
- prologue `e4b/fused_attn4_m_la_d2` **42.0 s** before step 1 (17% of the arm): c1_before 22.4, load_weights 9.6, c1_after 8.7, preamble 2.8; unattributed 0.875; budget 1260.0
- prologue `e4b/fused_attn4_m_l0_d2` **42.2 s** before step 1 (18% of the arm): c1_before 22.4, load_weights 9.7, c1_after 8.7, preamble 2.8; unattributed 0.92; budget 1260.0
- prologue `e4b/fused_attn4_shipped_la_d2` **41.4 s** before step 1 (20% of the arm): c1_before 22.5, load_weights 9.7, c1_after 8.7, preamble 2.7; unattributed 0.931; budget 1260.0
- prologue `e4b/fused_attn4_shipped_l0_d2` **41.2 s** before step 1 (19% of the arm): c1_before 22.5, load_weights 9.6, c1_after 8.7, preamble 2.7; unattributed 0.884; budget 1260.0
- draws (R1): `e4b/fused_attn4_shipped_l0` STABLE (1.594/1.596 s, |Δ|/mean 0.1% vs 5%); `e4b/fused_attn4_shipped_la` STABLE (1.599/1.598 s, |Δ|/mean 0.1% vs 5%); `e4b/fused_attn4_m_l0` STABLE (2.041/2.047 s, |Δ|/mean 0.3% vs 5%); `e4b/fused_attn4_m_la` STABLE (2.110/2.105 s, |Δ|/mean 0.3% vs 5%)
- e4b internal parity: NO-ARM
- **NO MATCHED POSITION QUOTED (unsloth)** — not quoted: e4b STABLE / unsloth no receipt
- **NO MATCHED POSITION QUOTED (hf)** — not quoted: e4b STABLE / hf no receipt
- **NO MATCHED POSITION QUOTED (axolotl)** — not quoted: e4b STABLE / axolotl no receipt
- equivalence vs `e4b/fused_attn4_m` (I/H; EQUIVALENT iff both ≤ band = max(0.005, 3 × fused-vs-reference |Δ|) = none (no reference arm: COMPARABLE is the ceiling); COMPARABLE ≤ 0.05; draw-noise floor 0.0022): `e4b/fused_attn4_m_la` **COMPARABLE** (median step |Δ| 0.0011, |Δ held-out at N| 0.0005, step-0 0.0000 SAME-BYTES-CLASS, |Δ loss at step 2| 0.0010, paired rows mean +0.0005 ± 0.0011 SE over 8, favouring arm 5 / anchor 3) — no reference arm ran: the EQUIVALENT band cannot be set, COMPARABLE is the ceiling; `e4b/fused_attn4_m_la_d2` **COMPARABLE** (median step |Δ| 0.0010, |Δ held-out at N| 0.0021, step-0 0.0000 SAME-BYTES-CLASS, |Δ loss at step 2| 0.0033, paired rows mean +0.0021 ± 0.0014 SE over 8, favouring arm 1 / anchor 7) — no reference arm ran: the EQUIVALENT band cannot be set, COMPARABLE is the ceiling; `e4b/fused_attn4_m_l0_d2` **COMPARABLE** (median step |Δ| 0.0010, |Δ held-out at N| 0.0022, step-0 0.0000 SAME-BYTES-CLASS, |Δ loss at step 2| 0.0258, paired rows mean +0.0022 ± 0.0021 SE over 8, favouring arm 3 / anchor 5) — no reference arm ran: the EQUIVALENT band cannot be set, COMPARABLE is the ceiling
- matched_init_sha (B, name-free, canonical slot order): anchor `f7832488926eda91`; `e4b/fused_attn4_m_l0` same; `e4b/fused_attn4_m_la` same; `e4b/fused_attn4_m_la_d2` same; `e4b/fused_attn4_m_l0_d2` same
- frozen base (R5), anchor `e4b/fused_attn4_m`: gate_up nf4/64+dq sha e2bed944143e control detects, down nf4/64+dq sha f2ade29dafdb control detects, q_proj nf4/64+dq sha 5c5e402ce51d control detects
- frozen base `e4b/fused_attn4_shipped_l0`: gate_up **SAME-BYTES** (nf4/64+dq) control detects, down **SAME-BYTES** (nf4/64+dq) control detects, q_proj **SAME-BYTES** (nf4/64+dq) control detects
- frozen base `e4b/fused_attn4_shipped_la`: gate_up **SAME-BYTES** (nf4/64+dq) control detects, down **SAME-BYTES** (nf4/64+dq) control detects, q_proj **SAME-BYTES** (nf4/64+dq) control detects
- frozen base `e4b/fused_attn4_m_la`: gate_up **SAME-BYTES** (nf4/64+dq) control detects, down **SAME-BYTES** (nf4/64+dq) control detects, q_proj **SAME-BYTES** (nf4/64+dq) control detects
- frozen base `e4b/fused_attn4_m_la_d2`: gate_up **SAME-BYTES** (nf4/64+dq) control detects, down **SAME-BYTES** (nf4/64+dq) control detects, q_proj **SAME-BYTES** (nf4/64+dq) control detects
- frozen base `e4b/fused_attn4_m_l0_d2`: gate_up **SAME-BYTES** (nf4/64+dq) control detects, down **SAME-BYTES** (nf4/64+dq) control detects, q_proj **SAME-BYTES** (nf4/64+dq) control detects
- frozen base `e4b/fused_attn4_shipped_la_d2`: gate_up **SAME-BYTES** (nf4/64+dq) control detects, down **SAME-BYTES** (nf4/64+dq) control detects, q_proj **SAME-BYTES** (nf4/64+dq) control detects
- frozen base `e4b/fused_attn4_shipped_l0_d2`: gate_up **SAME-BYTES** (nf4/64+dq) control detects, down **SAME-BYTES** (nf4/64+dq) control detects, q_proj **SAME-BYTES** (nf4/64+dq) control detects

## Verdicts
| family | arm | VERDICT | validity | quality | note |
|---|---|---|---|---|---|
| qwen3slauto | e4b/fused_attn4_shipped_l0 | **VALID** | VALID | -0.0003 |  |
| qwen3slauto | e4b/fused_attn4_shipped_la | **VALID** | VALID | -0.0005 |  |
| qwen3slauto | e4b/fused_attn4_m_l0 | **VALID** | VALID | 0.0000 |  |
| qwen3slauto | e4b/fused_attn4_m_la | **VALID** | VALID | 0.0005 |  |
| qwen3slauto | e4b/fused_attn4_m_la_d2 | **VALID** | VALID | 0.0021 |  |
| qwen3slauto | e4b/fused_attn4_m_l0_d2 | **VALID** | VALID | 0.0022 |  |
| qwen3slauto | e4b/fused_attn4_shipped_la_d2 | **VALID** | VALID | 0.0013 |  |
| qwen3slauto | e4b/fused_attn4_shipped_l0_d2 | **VALID** | VALID | 0.0015 |  |

## Amendment 71: NF4_QLORA_SINGLE_LADDER 0 vs auto at the field recipe (descriptive)
| arm | VERDICT | s/step (11..N) | device ms / profiled step | busy_t | aten::bmm CPU us / call | laddered calls | peak GB | held-out 0 / N |
|---|---|---|---|---|---|---|---|---|
| e4b/fused_attn4_shipped_l0 | VALID | 1.594 | 1412.7 | 0.886 | 10.0 | 0 | 23.321 | 1.95917 / 0.75583 |
| e4b/fused_attn4_shipped_la | VALID | 1.599 | 1405.6 | 0.879 | 10.1 | 0 | 23.321 | 1.95917 / 0.75558 |
| e4b/fused_attn4_m_l0 | VALID | 2.041 | 1806.0 | 0.885 | 96.7 | 0 | 26.154 | 1.95917 / 0.7561 |
| e4b/fused_attn4_m_la | VALID | 2.110 | 1905.1 | 0.903 | 8.8 | 49152 | 26.495 | 1.95917 / 0.75662 |
| e4b/fused_attn4_m_la_d2 | VALID | 2.105 | 1906.2 | 0.906 | 8.8 | 49152 | 26.495 | 1.95917 / 0.75824 |
| e4b/fused_attn4_m_l0_d2 | VALID | 2.047 | 1805.9 | 0.882 | 95.2 | 0 | 26.151 | 1.95917 / 0.75831 |
| e4b/fused_attn4_shipped_la_d2 | VALID | 1.598 | 1408.8 | 0.882 | 10.2 | 0 | 23.321 | 1.95917 / 0.75742 |
| e4b/fused_attn4_shipped_l0_d2 | VALID | 1.596 | 1404.2 | 0.880 | 10.3 | 0 | 23.321 | 1.95917 / 0.75762 |

## Predictions P215-P219 (TC1-PREREG amendment 71: NF4_QLORA_SINGLE_LADDER=auto at the field recipe; scored mechanically)
| prediction | family | verdict | evidence |
|---|---|---|---|
| P215 | qwen3slauto | **HELD** | la / l0 1.031 [1.028, 1.034 over 4 cross-draw ratios] vs <= 1.05; s/step la 2.110 / 2.105, l0 2.041 / 2.047; matched l0 busy_t 0.884: a GPU-bound box |
| P216 | qwen3slauto | **HELD** | la / l0 1.002 [1.001, 1.003 over 4 cross-draw ratios] vs in [0.98, 1.02]; s/step la 1.599 / 1.598, l0 1.594 / 1.596 |
| P217 | qwen3slauto | **HELD** | aten::bmm CPU self per call 95.9 -> 8.8 us = 0.092 vs <= 0.5 (matched) |
| P218 | qwen3slauto | **HELD** | device ms per profiled step la / l0: m: 1805.9 -> 1905.7 ms = 1.055 (<= 1.06); shipped: 1408.5 -> 1407.2 ms = 0.999 (in [0.98, 1.02]) |
| P219 | qwen3slauto | **HELD** | m: step 0 +0.00000, +0.00000; N +0.00023; shipped: step 0 +0.00000, +0.00000; N -0.00023 (step 0 |.| <= 0.0005, N |.| <= 0.005) |

## Load gate (TC1-PREREG amendment 33): 0 draw(s) voided for host load and run again
Each line: the arm, the attempt, the median host load1 over that attempt's run, the gate. A VOID attempt's files are in loadvoid/; the last attempt of each arm stands whatever its load.

- `qwen3slauto/e4b/fused_attn4_shipped_l0 attempt 0 load1_median 1.16 gate 6.0 status ok over 0`
- `qwen3slauto/e4b/fused_attn4_shipped_la attempt 0 load1_median 1.07 gate 6.0 status ok over 0`
- `qwen3slauto/e4b/fused_attn4_m_l0 attempt 0 load1_median 1.02 gate 6.0 status ok over 0`
- `qwen3slauto/e4b/fused_attn4_m_la attempt 0 load1_median 1.2 gate 6.0 status ok over 0`
- `qwen3slauto/e4b/fused_attn4_m_la_d2 attempt 0 load1_median 1.2 gate 6.0 status ok over 0`
- `qwen3slauto/e4b/fused_attn4_m_l0_d2 attempt 0 load1_median 1.12 gate 6.0 status ok over 0`
- `qwen3slauto/e4b/fused_attn4_shipped_la_d2 attempt 0 load1_median 1.15 gate 6.0 status ok over 0`
- `qwen3slauto/e4b/fused_attn4_shipped_l0_d2 attempt 0 load1_median 1.31 gate 6.0 status ok over 0`
