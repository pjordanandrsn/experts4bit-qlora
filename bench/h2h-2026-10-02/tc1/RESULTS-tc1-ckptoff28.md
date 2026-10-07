# TC1 — Qwen3-30B-A3B on one RTX 5090: e4b vs Unsloth vs HF+PEFT vs axolotl at matched work, init and adapter precision (tc1)
Rule (tc1/TC1-PREREG.md): status per attempt in the vocabulary OK / REFUSED / OOM / INSTALL_FAILED / LOAD_FAULT / HARNESS_ERROR / ALARM / NOT_RUN; VALID/VOID per tp4's predicates plus the matched-set predicates (R3: matched init complete, fp32 adapters, step-0 held-out within 0.005 of e4b/reference_attn4_m); VERDICT exactly one of VALID / VOID / QUALITY_FAIL / OOM / UNSUPPORTED / HARNESS_ERROR / ALARM / NOT_RUN (QUALITY_FAIL = held-out |Δ| at N > 0.05 vs e4b/fused_attn4_m); positions = s/step ratios other/e4b from the medians over both draws (stability |d1-d2|/mean ≤ 5 % e4b / 10 % others, else UNSTABLE), quoted only when both arms are VALID; equivalence vs e4b/reference_attn4_m EQUIVALENT ≤ 0.02 / COMPARABLE ≤ 0.05 / else DIVERGENT; frozen base SAME-BYTES / DIFFERENT / N-A per slot; predictions P1–P10 (+ P1b) scored HELD / FALSIFIED / UNTESTED. VOID never enters a ratio or an equivalence reading. Nothing is licensed; no cross-box number is divided into these.
`versions.txt`
```
e4b 0.48.0 @53b3fb38ffa766f581c33d1e974b7f0dd8ae2110 (GitHub main)
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
e4b(t212) 0.48.0 @53b3fb38ffa766f581c33d1e974b7f0dd8ae2110
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
 "run_id": "tc1-5090-125",
 "instance_id": "54627003",
 "gpu": "NVIDIA GeForce RTX 5090",
 "driver": "595.91.07",
 "cpu": "AMD Ryzen 9 9950X 16-Core Processor",
 "nproc": 32,
 "mem_total_kb": "129453600",
 "cgroup_memory_max": "127257280512",
 "disk_root": "overlay         320G  2.6M  320G   1% /",
 "hostname": "676a6cbb4270",
 "cgroup_cpu_max": "3071999 100000",
 "affinity_cpus": 32,
 "cgroup_cpuset_effective": "0-31",
 "registered_gpu_class": "5090",
 "prereg": "tc1/TC1-PREREG.md",
 "local_box": null,
 "local_snapshot": null
}
```

### Qwen3-30B-A3B (amendment 63: checkpoint inputs on the GPU vs in pinned host memory at the field recipe in torch 2.8, shipped and matched arms, profiled) (`qwen3ckptoff28`, registered n_layers 48, attention census 192)
- model `Qwen/Qwen3-30B-A3B` @ `ad44e777bcd1`; tokens sha `bfc742f67e37`; N=60; fixture template alpaca seq 2048 micro-batch 2 × accum 4 lr 0.0002 r 16 α 16 optimizer adamw_8bit(lr=0.0002, weight_decay=0.001) schedule=linear warmup_steps=5 autocast False; e4b trainable 642514944; box_class RTX 5090 gpu NVIDIA GeForce RTX 5090
| framework | arm | status | validity | **VERDICT** | matched | init / dtype | step-0 class | N | s/step med(11+) | tok/s | peak GB | J/step | train first→last | held-out 0→final | quality Δ | engagement | trainable | regime | reason / why |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| e4b | fused_attn4_shipped_g0 | **OK** | VALID | **VALID** | native | native / bfloat16,float32 | — | 60 | 1.947 | 729.4 | 23.271 | 572.6 | 2.0506→0.8198 | 1.9251→0.7543 | -0.0031 | patched 48 / kcalls 768 | 642514944 | 4-bit experts (e4b NF4) + NF4 attention |  |
| e4b | fused_attn4_shipped_g1 | **OK** | VALID | **VALID** | native | native / bfloat16,float32 | — | 60 | 1.916 | 829.6 | 23.054 | 474.3 | 2.0506→0.8160 | 1.9251→0.7557 | -0.0018 | patched 48 / kcalls 768 | 642514944 | 4-bit experts (e4b NF4) + NF4 attention |  |
| e4b | fused_attn4_m_g0 | **OK** | VALID | **VALID** | yes | matched:3407 (complete 12480/12480) / float32 | SAME-BYTES-CLASS (0.0000) | 60 | 2.474 | 644.1 | 26.106 | 647.9 | 2.0506→0.8165 | 1.9251→0.7574 | 0.0000 | patched 48 / kcalls 768 | 642514944 | 4-bit experts (e4b NF4) + NF4 attention |  |
| e4b | fused_attn4_m_g1 | **OK** | VALID | **VALID** | yes | matched:3407 (complete 12480/12480) / float32 | SAME-BYTES-CLASS (0.0000) | 60 | 2.457 | 654.6 | 25.918 | 617.8 | 2.0506→0.8171 | 1.9251→0.7553 | -0.0021 | patched 48 / kcalls 768 | 642514944 | 4-bit experts (e4b NF4) + NF4 attention |  |
| e4b | fused_attn4_m_g1_d2 | **OK** | VALID | **VALID** | yes | matched:3407 (complete 12480/12480) / float32 | SAME-BYTES-CLASS (0.0000) | 60 | 2.458 | 657.5 | 25.928 | 615.8 | 2.0506→0.8186 | 1.9251→0.7563 | -0.0011 | patched 48 / kcalls 768 | 642514944 | 4-bit experts (e4b NF4) + NF4 attention |  |
| e4b | fused_attn4_m_g0_d2 | **OK** | VALID | **VALID** | yes | matched:3407 (complete 12480/12480) / float32 | SAME-BYTES-CLASS (0.0000) | 60 | 2.470 | 643.2 | 26.102 | 652.9 | 2.0506→0.8175 | 1.9251→0.7563 | -0.0011 | patched 48 / kcalls 768 | 642514944 | 4-bit experts (e4b NF4) + NF4 attention |  |
| e4b | fused_attn4_shipped_g1_d2 | **OK** | VALID | **VALID** | native | native / bfloat16,float32 | — | 60 | 1.919 | 829.5 | 23.054 | 479.0 | 2.0506→0.8179 | 1.9251→0.7552 | -0.0022 | patched 48 / kcalls 768 | 642514944 | 4-bit experts (e4b NF4) + NF4 attention |  |
| e4b | fused_attn4_shipped_g0_d2 | **OK** | VALID | **VALID** | native | native / bfloat16,float32 | — | 60 | 1.945 | 761.8 | 23.271 | 490.3 | 2.0506→0.8173 | 1.9251→0.7557 | -0.0017 | patched 48 / kcalls 768 | 642514944 | 4-bit experts (e4b NF4) + NF4 attention |  |
- prologue `e4b/fused_attn4_shipped_g0` **58.1 s** before step 1 (22% of the arm): c1_before 29.8, load_weights 11.8, c1_after 9.7, eval0 6.5; unattributed 0.753; budget 1260.0
- prologue `e4b/fused_attn4_shipped_g1` **52.2 s** before step 1 (21% of the arm): c1_before 29.4, load_weights 11.8, c1_after 11.3, attn4 3.1; unattributed 0.796; budget 1260.0
- prologue `e4b/fused_attn4_m_g0` **53.5 s** before step 1 (19% of the arm): c1_before 29.5, load_weights 11.8, c1_after 11.3, attn4 3.1; unattributed 0.744; budget 1260.0
- prologue `e4b/fused_attn4_m_g1` **53.5 s** before step 1 (19% of the arm): c1_before 29.4, load_weights 11.8, c1_after 11.4, attn4 3.1; unattributed 0.741; budget 1260.0
- prologue `e4b/fused_attn4_m_g1_d2` **54.4 s** before step 1 (19% of the arm): c1_before 29.8, load_weights 11.8, c1_after 11.3, attn4 3.3; unattributed 0.745; budget 1260.0
- prologue `e4b/fused_attn4_m_g0_d2` **53.9 s** before step 1 (19% of the arm): c1_before 29.8, load_weights 11.9, c1_after 11.3, attn4 3.1; unattributed 0.744; budget 1260.0
- prologue `e4b/fused_attn4_shipped_g1_d2` **52.6 s** before step 1 (21% of the arm): c1_before 29.9, load_weights 11.8, c1_after 11.3, attn4 3.1; unattributed 0.744; budget 1260.0
- prologue `e4b/fused_attn4_shipped_g0_d2` **52.8 s** before step 1 (20% of the arm): c1_before 30.0, load_weights 11.8, c1_after 9.6, attn4 3.1; unattributed 0.742; budget 1260.0
- draws (R1): `e4b/fused_attn4_shipped_g0` STABLE (1.947/1.945 s, |Δ|/mean 0.1% vs 5%); `e4b/fused_attn4_shipped_g1` STABLE (1.916/1.919 s, |Δ|/mean 0.2% vs 5%); `e4b/fused_attn4_m_g0` STABLE (2.474/2.470 s, |Δ|/mean 0.1% vs 5%); `e4b/fused_attn4_m_g1` STABLE (2.457/2.458 s, |Δ|/mean 0.0% vs 5%)
- e4b internal parity: NO-ARM
- **NO MATCHED POSITION QUOTED (unsloth)** — not quoted: e4b STABLE / unsloth no receipt
- **NO MATCHED POSITION QUOTED (hf)** — not quoted: e4b STABLE / hf no receipt
- **NO MATCHED POSITION QUOTED (axolotl)** — not quoted: e4b STABLE / axolotl no receipt
- equivalence vs `e4b/fused_attn4_m` (I/H; EQUIVALENT iff both ≤ band = max(0.005, 3 × fused-vs-reference |Δ|) = none (no reference arm: COMPARABLE is the ceiling); COMPARABLE ≤ 0.05; draw-noise floor 0.0014): `e4b/fused_attn4_m_g1` **COMPARABLE** (median step |Δ| 0.0012, |Δ held-out at N| 0.0021, step-0 0.0000 SAME-BYTES-CLASS, |Δ loss at step 2| 0.0250, paired rows mean -0.0021 ± 0.0033 SE over 8, favouring arm 5 / anchor 3) — no reference arm ran: the EQUIVALENT band cannot be set, COMPARABLE is the ceiling; `e4b/fused_attn4_m_g1_d2` **COMPARABLE** (median step |Δ| 0.0009, |Δ held-out at N| 0.0011, step-0 0.0000 SAME-BYTES-CLASS, |Δ loss at step 2| 0.0272, paired rows mean -0.0011 ± 0.0019 SE over 8, favouring arm 6 / anchor 2) — no reference arm ran: the EQUIVALENT band cannot be set, COMPARABLE is the ceiling; `e4b/fused_attn4_m_g0_d2` **COMPARABLE** (median step |Δ| 0.0012, |Δ held-out at N| 0.0011, step-0 0.0000 SAME-BYTES-CLASS, |Δ loss at step 2| 0.0059, paired rows mean -0.0011 ± 0.0028 SE over 8, favouring arm 5 / anchor 3) — no reference arm ran: the EQUIVALENT band cannot be set, COMPARABLE is the ceiling
- matched_init_sha (B, name-free, canonical slot order): anchor `f7832488926eda91`; `e4b/fused_attn4_m_g0` same; `e4b/fused_attn4_m_g1` same; `e4b/fused_attn4_m_g1_d2` same; `e4b/fused_attn4_m_g0_d2` same
- frozen base (R5), anchor `e4b/fused_attn4_m`: gate_up nf4/64+dq sha e2bed944143e control detects, down nf4/64+dq sha f2ade29dafdb control detects, q_proj nf4/64+dq sha 5c5e402ce51d control detects
- frozen base `e4b/fused_attn4_shipped_g0`: gate_up **SAME-BYTES** (nf4/64+dq) control detects, down **SAME-BYTES** (nf4/64+dq) control detects, q_proj **SAME-BYTES** (nf4/64+dq) control detects
- frozen base `e4b/fused_attn4_shipped_g1`: gate_up **SAME-BYTES** (nf4/64+dq) control detects, down **SAME-BYTES** (nf4/64+dq) control detects, q_proj **SAME-BYTES** (nf4/64+dq) control detects
- frozen base `e4b/fused_attn4_m_g1`: gate_up **SAME-BYTES** (nf4/64+dq) control detects, down **SAME-BYTES** (nf4/64+dq) control detects, q_proj **SAME-BYTES** (nf4/64+dq) control detects
- frozen base `e4b/fused_attn4_m_g1_d2`: gate_up **SAME-BYTES** (nf4/64+dq) control detects, down **SAME-BYTES** (nf4/64+dq) control detects, q_proj **SAME-BYTES** (nf4/64+dq) control detects
- frozen base `e4b/fused_attn4_m_g0_d2`: gate_up **SAME-BYTES** (nf4/64+dq) control detects, down **SAME-BYTES** (nf4/64+dq) control detects, q_proj **SAME-BYTES** (nf4/64+dq) control detects
- frozen base `e4b/fused_attn4_shipped_g1_d2`: gate_up **SAME-BYTES** (nf4/64+dq) control detects, down **SAME-BYTES** (nf4/64+dq) control detects, q_proj **SAME-BYTES** (nf4/64+dq) control detects
- frozen base `e4b/fused_attn4_shipped_g0_d2`: gate_up **SAME-BYTES** (nf4/64+dq) control detects, down **SAME-BYTES** (nf4/64+dq) control detects, q_proj **SAME-BYTES** (nf4/64+dq) control detects

## Verdicts
| family | arm | VERDICT | validity | quality | note |
|---|---|---|---|---|---|
| qwen3ckptoff28 | e4b/fused_attn4_shipped_g0 | **VALID** | VALID | -0.0031 |  |
| qwen3ckptoff28 | e4b/fused_attn4_shipped_g1 | **VALID** | VALID | -0.0018 |  |
| qwen3ckptoff28 | e4b/fused_attn4_m_g0 | **VALID** | VALID | 0.0000 |  |
| qwen3ckptoff28 | e4b/fused_attn4_m_g1 | **VALID** | VALID | -0.0021 |  |
| qwen3ckptoff28 | e4b/fused_attn4_m_g1_d2 | **VALID** | VALID | -0.0011 |  |
| qwen3ckptoff28 | e4b/fused_attn4_m_g0_d2 | **VALID** | VALID | -0.0011 |  |
| qwen3ckptoff28 | e4b/fused_attn4_shipped_g1_d2 | **VALID** | VALID | -0.0022 |  |
| qwen3ckptoff28 | e4b/fused_attn4_shipped_g0_d2 | **VALID** | VALID | -0.0017 |  |

## Amendment 63: checkpoint inputs on the GPU vs in pinned host memory at the field recipe in torch 2.8, profiled (descriptive)
| arm | VERDICT | s/step (11..N) | run peak GB | train | held-out N | device busy vs timed step |
|---|---|---|---|---|---|---|
| e4b/fused_attn4_shipped_g0 | VALID | 1.947 | 23.271 | 23.271 | 0.75431 | 0.794 |
| e4b/fused_attn4_shipped_g1 | VALID | 1.916 | 23.054 | 23.054 | 0.75568 | 0.819 |
| e4b/fused_attn4_m_g0 | VALID | 2.474 | 26.106 | 26.106 | 0.75744 | 0.774 |
| e4b/fused_attn4_m_g1 | VALID | 2.457 | 25.918 | 25.918 | 0.75532 | 0.789 |
| e4b/fused_attn4_m_g1_d2 | VALID | 2.458 | 25.928 | 25.928 | 0.75633 | 0.789 |
| e4b/fused_attn4_m_g0_d2 | VALID | 2.470 | 26.102 | 26.102 | 0.7563 | 0.777 |
| e4b/fused_attn4_shipped_g1_d2 | VALID | 1.919 | 23.054 | 23.054 | 0.75523 | 0.819 |
| e4b/fused_attn4_shipped_g0_d2 | VALID | 1.945 | 23.271 | 23.271 | 0.7557 | 0.796 |

## Predictions P174 / P175 / P176 / P177 (TC1-PREREG amendment 63: the offload in torch 2.8 at the field recipe; scored mechanically)
| prediction | family | verdict | evidence |
|---|---|---|---|
| P174 | qwen3ckptoff28 | **HELD** | m: g1 / g0 0.994 [0.993, 0.995 over 4 cross-draw ratios] vs <= 1.01; s/step g0 2.474 / 2.470, g1 2.457 / 2.458; g0 device busy vs the timed step 0.775 (premise <= 0.9) |
| P175 | qwen3ckptoff28 | **HELD** | shipped: g1 / g0 0.985 [0.984, 0.987 over 4 cross-draw ratios] vs <= 1.01; s/step g0 1.947 / 1.945, g1 1.916 / 1.919; g0 device busy vs the timed step 0.795 (premise <= 0.9) |
| P176 | qwen3ckptoff28 | **HELD** | mean held-out at N g1 - g0: m -0.0010, shipped +0.0004 (|.| <= 0.005) |
| P177 | qwen3ckptoff28 | **HELD** | matched training-phase peak g0 26.104 -> g1 25.923 GB (drop +0.181 vs >= 0.1) |

## Load gate (TC1-PREREG amendment 33): 0 draw(s) voided for host load and run again
Each line: the arm, the attempt, the median host load1 over that attempt's run, the gate. A VOID attempt's files are in loadvoid/; the last attempt of each arm stands whatever its load.

- `qwen3ckptoff28/e4b/fused_attn4_shipped_g0 attempt 0 load1_median 1.01 gate 6.0 status ok over 0`
- `qwen3ckptoff28/e4b/fused_attn4_shipped_g1 attempt 0 load1_median 1.27 gate 6.0 status ok over 0`
- `qwen3ckptoff28/e4b/fused_attn4_m_g0 attempt 0 load1_median 1.08 gate 6.0 status ok over 0`
- `qwen3ckptoff28/e4b/fused_attn4_m_g1 attempt 0 load1_median 1.03 gate 6.0 status ok over 0`
- `qwen3ckptoff28/e4b/fused_attn4_m_g1_d2 attempt 0 load1_median 1.08 gate 6.0 status ok over 0`
- `qwen3ckptoff28/e4b/fused_attn4_m_g0_d2 attempt 0 load1_median 1.08 gate 6.0 status ok over 0`
- `qwen3ckptoff28/e4b/fused_attn4_shipped_g1_d2 attempt 0 load1_median 1.09 gate 6.0 status ok over 0`
- `qwen3ckptoff28/e4b/fused_attn4_shipped_g0_d2 attempt 0 load1_median 1.2 gate 6.0 status ok over 0`
