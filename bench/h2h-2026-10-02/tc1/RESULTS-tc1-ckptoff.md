# TC1 — Qwen3-30B-A3B on one RTX 5090: e4b vs Unsloth vs HF+PEFT vs axolotl at matched work, init and adapter precision (tc1-5090-115)
Rule (tc1/TC1-PREREG.md): status per attempt in the vocabulary OK / REFUSED / OOM / INSTALL_FAILED / LOAD_FAULT / HARNESS_ERROR / ALARM / NOT_RUN; VALID/VOID per tp4's predicates plus the matched-set predicates (R3: matched init complete, fp32 adapters, step-0 held-out within 0.005 of e4b/reference_attn4_m); VERDICT exactly one of VALID / VOID / QUALITY_FAIL / OOM / UNSUPPORTED / HARNESS_ERROR / ALARM / NOT_RUN (QUALITY_FAIL = held-out |Δ| at N > 0.05 vs e4b/fused_attn4_m); positions = s/step ratios other/e4b from the medians over both draws (stability |d1-d2|/mean ≤ 5 % e4b / 10 % others, else UNSTABLE), quoted only when both arms are VALID; equivalence vs e4b/reference_attn4_m EQUIVALENT ≤ 0.02 / COMPARABLE ≤ 0.05 / else DIVERGENT; frozen base SAME-BYTES / DIFFERENT / N-A per slot; predictions P1–P10 (+ P1b) scored HELD / FALSIFIED / UNTESTED. VOID never enters a ratio or an equivalence reading. Nothing is licensed; no cross-box number is divided into these.
`versions.txt`
```
e4b 0.48.0 @666b775e6d97dea39ac359434e01b333b53d6d03 (GitHub main)
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
e4b(t212) 0.48.0 @666b775e6d97dea39ac359434e01b333b53d6d03
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
 "run_id": "tc1-5090-115",
 "instance_id": "54587486",
 "gpu": "NVIDIA GeForce RTX 5090",
 "driver": "580.178.04",
 "cpu": "AMD Ryzen 9 7900 12-Core Processor",
 "nproc": 24,
 "mem_total_kb": "130952932",
 "cgroup_memory_max": "128730529792",
 "disk_root": "overlay         320G  2.5M  320G   1% /",
 "hostname": "dc4720fed68c",
 "cgroup_cpu_max": "2304000 100000",
 "affinity_cpus": 24,
 "cgroup_cpuset_effective": "0-23",
 "registered_gpu_class": "5090",
 "prereg": "tc1/TC1-PREREG.md",
 "local_box": null,
 "local_snapshot": null
}
```

### Qwen3-30B-A3B (amendment 58: e4b's matched arm with checkpoint inputs on the GPU vs in pinned host memory on packed rows, Unsloth beside; peaks by phase) (`qwen3ckptoff`, registered n_layers 48, attention census 192)
- model `Qwen/Qwen3-30B-A3B` @ `ad44e777bcd1`; tokens sha `d2a501eba57d`; N=40; fixture template alpaca seq 4096 (packed rows, amendment 39) micro-batch 1 × accum 4 lr 0.0002 r 16 α 16 optimizer adamw_8bit(lr=0.0002, weight_decay=0.001) schedule=linear warmup_steps=5 autocast False; e4b trainable 642514944; box_class RTX 5090 gpu NVIDIA GeForce RTX 5090
| framework | arm | status | validity | **VERDICT** | matched | init / dtype | step-0 class | N | s/step med(11+) | tok/s | peak GB | J/step | train first→last | held-out 0→final | quality Δ | engagement | trainable | regime | reason / why |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| e4b | fused_attn4_m_o0 | **OK** | VALID | **VALID** | yes | matched:3407 (complete 12480/12480) / float32 | SAME-BYTES-CLASS (0.0000) | 40 | 9.697 | 1677.4 | 26.877 | 4676.9 | 1.2578→0.9127 | 1.2885→0.9541 | 0.0000 | patched 48 / kcalls 768 | 642514944 | 4-bit experts (e4b NF4) + NF4 attention |  |
| e4b | fused_attn4_m_o1 | **OK** | VALID | **VALID** | yes | matched:3407 (complete 12480/12480) / float32 | SAME-BYTES-CLASS (0.0000) | 40 | 9.708 | 1677.1 | 26.877 | 4312.4 | 1.2578→0.9126 | 1.2885→0.9545 | 0.0004 | patched 48 / kcalls 768 | 642514944 | 4-bit experts (e4b NF4) + NF4 attention |  |
| e4b | fused_attn4_m_o1_d2 | **OK** | VALID | **VALID** | yes | matched:3407 (complete 12480/12480) / float32 | SAME-BYTES-CLASS (0.0000) | 40 | 9.700 | 1677.1 | 26.877 | 4409.3 | 1.2578→0.9131 | 1.2885→0.9542 | 0.0001 | patched 48 / kcalls 768 | 642514944 | 4-bit experts (e4b NF4) + NF4 attention |  |
| e4b | fused_attn4_m_o0_d2 | **OK** | VALID | **VALID** | yes | matched:3407 (complete 12480/12480) / float32 | SAME-BYTES-CLASS (0.0000) | 40 | 9.652 | 1686.1 | 26.877 | 4350.3 | 1.2578→0.9133 | 1.2885→0.9542 | 0.0001 | patched 48 / kcalls 768 | 642514944 | 4-bit experts (e4b NF4) + NF4 attention |  |
| unsloth | ckpt_unsloth_m_oo | **OK** | VALID | **VALID** | yes | matched:3407 (complete 12480/12480) / float32 | SAME-BYTES-CLASS (0.0014) | 40 | 11.300 | 1434.4 | 24.864 | 3896.1 | 1.2587→0.9135 | 1.2871→0.9542 | 0.0001 | stacks 96 / fwd 384 / u8 48 | 642514944 | 4-bit expert stacks + bnb-4bit attention (Params4bit stacks 96, Linear4bit 384) |  |
- prologue `e4b/fused_attn4_m_o0` **86.9 s** before step 1 (18% of the arm): c1_before 37.5, load_weights 26.4, c1_after 19.2, eval0 9.4; unattributed 1.506; budget 1890.0
- prologue `e4b/fused_attn4_m_o1` **72.1 s** before step 1 (15% of the arm): c1_before 37.2, c1_after 18.6, load_weights 15.1, eval0 6.3; unattributed 1.483; budget 1890.0
- prologue `e4b/fused_attn4_m_o1_d2` **68.8 s** before step 1 (15% of the arm): c1_before 37.1, c1_after 18.6, load_weights 13.9, eval0 6.0; unattributed 1.256; budget 1890.0
- prologue `e4b/fused_attn4_m_o0_d2` **68.9 s** before step 1 (15% of the arm): c1_before 37.2, c1_after 18.7, load_weights 13.9, eval0 6.0; unattributed 1.257; budget 1890.0
- prologue `unsloth/ckpt_unsloth_m_oo` **79.7 s** before step 1 (15% of the arm): c1_before 36.5, c1_after 18.4, load_weights 17.3, eval0 13.8; unattributed 6.414; budget 1890.0
- draws (R1): `e4b/fused_attn4_m_o0` STABLE (9.697/9.652 s, |Δ|/mean 0.5% vs 5%); `e4b/fused_attn4_m_o1` STABLE (9.708/9.700 s, |Δ|/mean 0.1% vs 5%); `unsloth/ckpt_unsloth_m_oo` SINGLE (single draw (no second draw registered))
- e4b internal parity: NO-ARM
- **NO MATCHED POSITION QUOTED (unsloth)** — not quoted: e4b STABLE / unsloth no receipt
- **NO MATCHED POSITION QUOTED (hf)** — not quoted: e4b STABLE / hf no receipt
- **NO MATCHED POSITION QUOTED (axolotl)** — not quoted: e4b STABLE / axolotl no receipt
- equivalence vs `e4b/fused_attn4_m` (I/H; EQUIVALENT iff both ≤ band = max(0.005, 3 × fused-vs-reference |Δ|) = none (no reference arm: COMPARABLE is the ceiling); COMPARABLE ≤ 0.05; draw-noise floor 0.0003): `e4b/fused_attn4_m_o1` **COMPARABLE** (median step |Δ| 0.0004, |Δ held-out at N| 0.0004, step-0 0.0000 SAME-BYTES-CLASS, |Δ loss at step 2| 0.0006, paired rows mean +0.0005 ± 0.0002 SE over 8, favouring arm 3 / anchor 5) — no reference arm ran: the EQUIVALENT band cannot be set, COMPARABLE is the ceiling; `e4b/fused_attn4_m_o1_d2` **COMPARABLE** (median step |Δ| 0.0002, |Δ held-out at N| 0.0001, step-0 0.0000 SAME-BYTES-CLASS, |Δ loss at step 2| 0.0042, paired rows mean +0.0001 ± 0.0004 SE over 8, favouring arm 4 / anchor 4) — no reference arm ran: the EQUIVALENT band cannot be set, COMPARABLE is the ceiling; `e4b/fused_attn4_m_o0_d2` **COMPARABLE** (median step |Δ| 0.0003, |Δ held-out at N| 0.0001, step-0 0.0000 SAME-BYTES-CLASS, |Δ loss at step 2| 0.0022, paired rows mean +0.0001 ± 0.0003 SE over 8, favouring arm 3 / anchor 5) — no reference arm ran: the EQUIVALENT band cannot be set, COMPARABLE is the ceiling; `unsloth/ckpt_unsloth_m_oo` **COMPARABLE** (median step |Δ| 0.0004, |Δ held-out at N| 0.0001, step-0 0.0014 SAME-BYTES-CLASS, |Δ loss at step 2| 0.0021, paired rows mean +0.0001 ± 0.0001 SE over 8, favouring arm 3 / anchor 5) — no reference arm ran: the EQUIVALENT band cannot be set, COMPARABLE is the ceiling
- matched_init_sha (B, name-free, canonical slot order): anchor `f7832488926eda91`; `e4b/fused_attn4_m_o0` same; `e4b/fused_attn4_m_o1` same; `e4b/fused_attn4_m_o1_d2` same; `e4b/fused_attn4_m_o0_d2` same; `unsloth/ckpt_unsloth_m_oo` same
- frozen base (R5), anchor `e4b/fused_attn4_m`: gate_up nf4/64+dq sha e2bed944143e control detects, down nf4/64+dq sha f2ade29dafdb control detects, q_proj nf4/64+dq sha 5c5e402ce51d control detects
- frozen base `e4b/fused_attn4_m_o1`: gate_up **SAME-BYTES** (nf4/64+dq) control detects, down **SAME-BYTES** (nf4/64+dq) control detects, q_proj **SAME-BYTES** (nf4/64+dq) control detects
- frozen base `e4b/fused_attn4_m_o1_d2`: gate_up **SAME-BYTES** (nf4/64+dq) control detects, down **SAME-BYTES** (nf4/64+dq) control detects, q_proj **SAME-BYTES** (nf4/64+dq) control detects
- frozen base `e4b/fused_attn4_m_o0_d2`: gate_up **SAME-BYTES** (nf4/64+dq) control detects, down **SAME-BYTES** (nf4/64+dq) control detects, q_proj **SAME-BYTES** (nf4/64+dq) control detects
- frozen base `unsloth/ckpt_unsloth_m_oo`: gate_up **SAME-BYTES** (nf4/64+dq) control detects, down **SAME-BYTES** (nf4/64+dq) control detects, q_proj **SAME-BYTES** (nf4/64+dq) control detects

## Verdicts
| family | arm | VERDICT | validity | quality | note |
|---|---|---|---|---|---|
| qwen3ckptoff | e4b/fused_attn4_m_o0 | **VALID** | VALID | 0.0000 |  |
| qwen3ckptoff | e4b/fused_attn4_m_o1 | **VALID** | VALID | 0.0004 |  |
| qwen3ckptoff | e4b/fused_attn4_m_o1_d2 | **VALID** | VALID | 0.0001 |  |
| qwen3ckptoff | e4b/fused_attn4_m_o0_d2 | **VALID** | VALID | 0.0001 |  |
| qwen3ckptoff | unsloth/ckpt_unsloth_m_oo | **VALID** | VALID | 0.0001 |  |

## Amendment 58: checkpoint inputs on the GPU vs in pinned host memory on packed rows, peaks by phase (descriptive)
| arm | VERDICT | s/step (11..N) | run peak GB | setup | eval | train |
|---|---|---|---|---|---|---|
| e4b/fused_attn4_m_o0 | VALID | 9.697 | 26.877 | 21.856 | 26.877 | 26.587 |
| e4b/fused_attn4_m_o1 | VALID | 9.708 | 26.877 | 21.856 | 26.877 | 25.842 |
| e4b/fused_attn4_m_o1_d2 | VALID | 9.700 | 26.877 | 21.856 | 26.877 | 25.853 |
| e4b/fused_attn4_m_o0_d2 | VALID | 9.652 | 26.877 | 21.856 | 26.877 | 26.586 |
| unsloth/ckpt_unsloth_m_oo | VALID | 11.300 | 24.864 | 21.139 | 21.852 | 24.864 |

## Predictions P153 / P154 / P155 / P156 / P157 (TC1-PREREG amendment 58: checkpoint inputs in pinned host memory on packed rows; scored mechanically)
| prediction | family | verdict | evidence |
|---|---|---|---|
| P153 | qwen3ckptoff | **HELD** | training-phase peak o0 26.587 -> o1 25.848 GB (drop +0.739 vs >= 0.6); draws o0 [26.587, 26.586], o1 [25.842, 25.853] |
| P154 | qwen3ckptoff | **HELD** | o1 / o0 1.003 [1.000, 1.006 over 4 cross-draw ratios] vs <= 1.05; s/step o0 9.697 / 9.652, o1 9.708 / 9.700 |
| P155 | qwen3ckptoff | **HELD** | mean held-out at N o0 0.9541, o1 +0.0002 (|.| <= 0.005) |
| P156 | qwen3ckptoff | **FALSIFIED** | training-phase peak e4b o0 26.587 - Unsloth 24.864 = +1.723 GB vs <= +1.7 |
| P157 | qwen3ckptoff | **HELD** | training-phase peak e4b o1 25.848 - Unsloth 24.864 = +0.983 GB vs <= +1.0 |

## Load gate (TC1-PREREG amendment 33): 0 draw(s) voided for host load and run again
Each line: the arm, the attempt, the median host load1 over that attempt's run, the gate. A VOID attempt's files are in loadvoid/; the last attempt of each arm stands whatever its load.

- `qwen3ckptoff/e4b/fused_attn4_m_o0 attempt 0 load1_median 1.19 gate 6.0 status ok over 0`
- `qwen3ckptoff/e4b/fused_attn4_m_o1 attempt 0 load1_median 1.25 gate 6.0 status ok over 0`
- `qwen3ckptoff/e4b/fused_attn4_m_o1_d2 attempt 0 load1_median 1.34 gate 6.0 status ok over 0`
- `qwen3ckptoff/e4b/fused_attn4_m_o0_d2 attempt 0 load1_median 1.19 gate 6.0 status ok over 0`
- `qwen3ckptoff/unsloth/ckpt_unsloth_m_oo attempt 0 load1_median 1.23 gate 6.0 status ok over 0`
