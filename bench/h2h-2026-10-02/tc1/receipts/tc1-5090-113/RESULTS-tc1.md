# TC1 — Qwen3-30B-A3B on one RTX 5090: e4b vs Unsloth vs HF+PEFT vs axolotl at matched work, init and adapter precision (tc1)
Rule (tc1/TC1-PREREG.md): status per attempt in the vocabulary OK / REFUSED / OOM / INSTALL_FAILED / LOAD_FAULT / HARNESS_ERROR / ALARM / NOT_RUN; VALID/VOID per tp4's predicates plus the matched-set predicates (R3: matched init complete, fp32 adapters, step-0 held-out within 0.005 of e4b/reference_attn4_m); VERDICT exactly one of VALID / VOID / QUALITY_FAIL / OOM / UNSUPPORTED / HARNESS_ERROR / ALARM / NOT_RUN (QUALITY_FAIL = held-out |Δ| at N > 0.05 vs e4b/fused_attn4_m); positions = s/step ratios other/e4b from the medians over both draws (stability |d1-d2|/mean ≤ 5 % e4b / 10 % others, else UNSTABLE), quoted only when both arms are VALID; equivalence vs e4b/reference_attn4_m EQUIVALENT ≤ 0.02 / COMPARABLE ≤ 0.05 / else DIVERGENT; frozen base SAME-BYTES / DIFFERENT / N-A per slot; predictions P1–P10 (+ P1b) scored HELD / FALSIFIED / UNTESTED. VOID never enters a ratio or an equivalence reading. Nothing is licensed; no cross-box number is divided into these.
`versions.txt`
```
e4b 0.48.0 @52acaa30bf92e8e975c5897494510e90100e54d7 (GitHub main)
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
e4b(t212) 0.48.0 @52acaa30bf92e8e975c5897494510e90100e54d7
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
 "run_id": "tc1-5090-113",
 "instance_id": "54558732",
 "gpu": "NVIDIA GeForce RTX 5090",
 "driver": "580.95.05",
 "cpu": "AMD EPYC 7713 64-Core Processor",
 "nproc": 128,
 "mem_total_kb": "527977828",
 "cgroup_memory_max": "367003697152",
 "disk_root": "overlay         320G   78M  320G   1% /",
 "hostname": "1310c66e56e9",
 "cgroup_cpu_max": "6143999 100000",
 "affinity_cpus": 128,
 "cgroup_cpuset_effective": "0-127",
 "registered_gpu_class": "5090",
 "prereg": "tc1/TC1-PREREG.md",
 "local_box": null,
 "local_snapshot": null
}
```

### Qwen3-30B-A3B (amendment 56: e4b's matched arm with the fp32 vs the double-quantized expert absmax on packed rows, Unsloth beside; peaks by phase) (`qwen3dqpack`, registered n_layers 48, attention census 192)
- model `Qwen/Qwen3-30B-A3B` @ `ad44e777bcd1`; tokens sha `d2a501eba57d`; N=40; fixture template alpaca seq 4096 (packed rows, amendment 39) micro-batch 1 × accum 4 lr 0.0002 r 16 α 16 optimizer adamw_8bit(lr=0.0002, weight_decay=0.001) schedule=linear warmup_steps=5 autocast False; e4b trainable 642514944; box_class RTX 5090 gpu NVIDIA GeForce RTX 5090
| framework | arm | status | validity | **VERDICT** | matched | init / dtype | step-0 class | N | s/step med(11+) | tok/s | peak GB | J/step | train first→last | held-out 0→final | quality Δ | engagement | trainable | regime | reason / why |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| e4b | fused_attn4_m_a0 | **OK** | VALID | **VALID** | yes | matched:3407 (complete 12480/12480) / float32 | SAME-BYTES-CLASS (0.0000) | 40 | 10.513 | 1542.2 | 28.228 | 4546.2 | 1.2644→0.9124 | 1.2893→0.9543 | 0.0000 | patched 48 / kcalls 768 | 642514944 | 4-bit experts (e4b NF4) + NF4 attention |  |
| e4b | fused_attn4_m_a1 | **OK** | VALID | **VALID** | yes | matched:3407 (complete 12480/12480) / float32 | SAME-BYTES-CLASS (0.0008) | 40 | 10.530 | 1546.9 | 26.877 | 4398.4 | 1.2578→0.9133 | 1.2885→0.9544 | 0.0001 | patched 48 / kcalls 768 | 642514944 | 4-bit experts (e4b NF4) + NF4 attention |  |
| e4b | fused_attn4_m_a1_d2 | **OK** | VALID | **VALID** | yes | matched:3407 (complete 12480/12480) / float32 | SAME-BYTES-CLASS (0.0008) | 40 | 10.543 | 1543.1 | 26.877 | 4453.2 | 1.2578→0.9130 | 1.2885→0.9547 | 0.0004 | patched 48 / kcalls 768 | 642514944 | 4-bit experts (e4b NF4) + NF4 attention |  |
| e4b | fused_attn4_m_a0_d2 | **OK** | VALID | **VALID** | yes | matched:3407 (complete 12480/12480) / float32 | SAME-BYTES-CLASS (0.0000) | 40 | 10.526 | 1551.4 | 28.228 | 4409.5 | 1.2644→0.9130 | 1.2893→0.9543 | 0.0000 | patched 48 / kcalls 768 | 642514944 | 4-bit experts (e4b NF4) + NF4 attention |  |
| unsloth | ckpt_unsloth_m_pp | **OK** | VALID | **VALID** | yes | matched:3407 (complete 12480/12480) / float32 | SAME-BYTES-CLASS (0.0022) | 40 | 14.085 | 1142.2 | 24.864 | 4243.0 | 1.2587→0.9129 | 1.2871→0.9543 | 0.0000 | stacks 96 / fwd 384 / u8 48 | 642514944 | 4-bit expert stacks + bnb-4bit attention (Params4bit stacks 96, Linear4bit 384) |  |
- prologue `e4b/fused_attn4_m_a0` **103.9 s** before step 1 (19% of the arm): c1_before 57.4, c1_after 28.9, load_weights 17.9, eval0 11.7; unattributed 2.014; budget 1890.0
- prologue `e4b/fused_attn4_m_a1` **92.9 s** before step 1 (18% of the arm): c1_before 51.9, c1_after 26.0, load_weights 17.7, eval0 6.5; unattributed 1.995; budget 1890.0
- prologue `e4b/fused_attn4_m_a1_d2` **92.7 s** before step 1 (18% of the arm): c1_before 51.9, c1_after 26.3, load_weights 17.6, eval0 6.5; unattributed 1.998; budget 1890.0
- prologue `e4b/fused_attn4_m_a0_d2` **98.1 s** before step 1 (19% of the arm): c1_before 57.4, c1_after 28.8, load_weights 17.6, eval0 6.5; unattributed 2.002; budget 1890.0
- prologue `unsloth/ckpt_unsloth_m_pp` **111.9 s** before step 1 (16% of the arm): c1_before 51.9, c1_after 25.6, load_weights 22.4, eval0 20.6; unattributed 9.37; budget 1890.0
- draws (R1): `e4b/fused_attn4_m_a0` STABLE (10.513/10.526 s, |Δ|/mean 0.1% vs 5%); `e4b/fused_attn4_m_a1` STABLE (10.530/10.543 s, |Δ|/mean 0.1% vs 5%); `unsloth/ckpt_unsloth_m_pp` SINGLE (single draw (no second draw registered))
- e4b internal parity: NO-ARM
- **NO MATCHED POSITION QUOTED (unsloth)** — not quoted: e4b STABLE / unsloth no receipt
- **NO MATCHED POSITION QUOTED (hf)** — not quoted: e4b STABLE / hf no receipt
- **NO MATCHED POSITION QUOTED (axolotl)** — not quoted: e4b STABLE / axolotl no receipt
- equivalence vs `e4b/fused_attn4_m` (I/H; EQUIVALENT iff both ≤ band = max(0.005, 3 × fused-vs-reference |Δ|) = none (no reference arm: COMPARABLE is the ceiling); COMPARABLE ≤ 0.05; draw-noise floor 0.0003): `e4b/fused_attn4_m_a1` **COMPARABLE** (median step |Δ| 0.0002, |Δ held-out at N| 0.0001, step-0 0.0008 SAME-BYTES-CLASS, |Δ loss at step 2| 0.0025, paired rows mean +0.0001 ± 0.0002 SE over 8, favouring arm 3 / anchor 5) — no reference arm ran: the EQUIVALENT band cannot be set, COMPARABLE is the ceiling; `e4b/fused_attn4_m_a1_d2` **COMPARABLE** (median step |Δ| 0.0003, |Δ held-out at N| 0.0004, step-0 0.0008 SAME-BYTES-CLASS, |Δ loss at step 2| 0.0051, paired rows mean +0.0004 ± 0.0003 SE over 8, favouring arm 2 / anchor 6) — no reference arm ran: the EQUIVALENT band cannot be set, COMPARABLE is the ceiling; `e4b/fused_attn4_m_a0_d2` **COMPARABLE** (median step |Δ| 0.0002, |Δ held-out at N| 0.0000, step-0 0.0000 SAME-BYTES-CLASS, |Δ loss at step 2| 0.0031, paired rows mean -0.0000 ± 0.0003 SE over 8, favouring arm 5 / anchor 3) — no reference arm ran: the EQUIVALENT band cannot be set, COMPARABLE is the ceiling; `unsloth/ckpt_unsloth_m_pp` **COMPARABLE** (median step |Δ| 0.0003, |Δ held-out at N| 0.0000, step-0 0.0022 SAME-BYTES-CLASS, |Δ loss at step 2| 0.0015, paired rows mean +0.0000 ± 0.0004 SE over 8, favouring arm 4 / anchor 4) — no reference arm ran: the EQUIVALENT band cannot be set, COMPARABLE is the ceiling
- matched_init_sha (B, name-free, canonical slot order): anchor `f7832488926eda91`; `e4b/fused_attn4_m_a0` same; `e4b/fused_attn4_m_a1` same; `e4b/fused_attn4_m_a1_d2` same; `e4b/fused_attn4_m_a0_d2` same; `unsloth/ckpt_unsloth_m_pp` same
- frozen base (R5), anchor `e4b/fused_attn4_m`: gate_up nf4/64 sha 8c3b3fd78e89 control detects, down nf4/64 sha 59569d810a1a control detects, q_proj nf4/64+dq sha 5c5e402ce51d control detects
- frozen base `e4b/fused_attn4_m_a1`: gate_up **N-A** (regime differs: anchor nf4/64 vs nf4/64+dq) control detects, down **N-A** (regime differs: anchor nf4/64 vs nf4/64+dq) control detects, q_proj **SAME-BYTES** (nf4/64+dq) control detects
- frozen base `e4b/fused_attn4_m_a1_d2`: gate_up **N-A** (regime differs: anchor nf4/64 vs nf4/64+dq) control detects, down **N-A** (regime differs: anchor nf4/64 vs nf4/64+dq) control detects, q_proj **SAME-BYTES** (nf4/64+dq) control detects
- frozen base `e4b/fused_attn4_m_a0_d2`: gate_up **SAME-BYTES** (nf4/64) control detects, down **SAME-BYTES** (nf4/64) control detects, q_proj **SAME-BYTES** (nf4/64+dq) control detects
- frozen base `unsloth/ckpt_unsloth_m_pp`: gate_up **N-A** (regime differs: anchor nf4/64 vs nf4/64+dq) control detects, down **N-A** (regime differs: anchor nf4/64 vs nf4/64+dq) control detects, q_proj **SAME-BYTES** (nf4/64+dq) control detects

## Verdicts
| family | arm | VERDICT | validity | quality | note |
|---|---|---|---|---|---|
| qwen3dqpack | e4b/fused_attn4_m_a0 | **VALID** | VALID | 0.0000 |  |
| qwen3dqpack | e4b/fused_attn4_m_a1 | **VALID** | VALID | 0.0001 |  |
| qwen3dqpack | e4b/fused_attn4_m_a1_d2 | **VALID** | VALID | 0.0004 |  |
| qwen3dqpack | e4b/fused_attn4_m_a0_d2 | **VALID** | VALID | 0.0000 |  |
| qwen3dqpack | unsloth/ckpt_unsloth_m_pp | **VALID** | VALID | 0.0000 |  |

## Amendment 56: the double-quantized absmax on packed rows, peaks by phase (descriptive)
| arm | VERDICT | s/step (11..N) | run peak GB | setup | eval | train |
|---|---|---|---|---|---|---|
| e4b/fused_attn4_m_a0 | VALID | 10.513 | 28.228 | 21.856 | 28.228 | 28.137 |
| e4b/fused_attn4_m_a1 | VALID | 10.530 | 26.877 | 21.856 | 26.877 | 26.784 |
| e4b/fused_attn4_m_a1_d2 | VALID | 10.543 | 26.877 | 21.856 | 26.877 | 26.788 |
| e4b/fused_attn4_m_a0_d2 | VALID | 10.526 | 28.228 | 21.856 | 28.228 | 28.14 |
| unsloth/ckpt_unsloth_m_pp | VALID | 14.085 | 24.864 | 21.139 | 21.852 | 24.864 |

## Predictions P144 / P145 / P146 / P147 / P148 (TC1-PREREG amendment 56: the double-quantized absmax on packed rows, peaks by phase; scored mechanically)
| prediction | family | verdict | evidence |
|---|---|---|---|
| P144 | qwen3dqpack | **HELD** | a1 / a0 1.002 [1.000, 1.003 over 4 cross-draw ratios] vs <= 1.02; s/step a0 10.513 / 10.526, a1 10.530 / 10.543 |
| P145 | qwen3dqpack | **HELD** | run peak a0 28.228 -> a1 26.877 GB (drop +1.351 vs >= 1.2) |
| P146 | qwen3dqpack | **HELD** | mean held-out at N a0 0.9543, a1 +0.0002 (|.| <= 0.005) |
| P147 | qwen3dqpack | **FALSIFIED** | training-phase peak e4b a1 26.784 / 26.788 GB (median 26.786) - Unsloth 24.864 = +1.922 GB vs <= +1.0; Unsloth's phases {'setup': 21.139, 'eval': 21.852, 'train': 24.864} |
| P148 | qwen3dqpack | **HELD** | eval vs train phase peak (GB) per e4b draw: fused_attn4_m_a0 28.228 vs 28.137; fused_attn4_m_a1 26.877 vs 26.784; fused_attn4_m_a1_d2 26.877 vs 26.788; fused_attn4_m_a0_d2 28.228 vs 28.140 |

## Load gate (TC1-PREREG amendment 33): 0 draw(s) voided for host load and run again
Each line: the arm, the attempt, the median host load1 over that attempt's run, the gate. A VOID attempt's files are in loadvoid/; the last attempt of each arm stands whatever its load.

- `qwen3dqpack/e4b/fused_attn4_m_a0 attempt 0 load1_median 1.41 gate 6.0 status ok over 0`
- `qwen3dqpack/e4b/fused_attn4_m_a1 attempt 0 load1_median 1.33 gate 6.0 status ok over 0`
- `qwen3dqpack/e4b/fused_attn4_m_a1_d2 attempt 0 load1_median 1.29 gate 6.0 status ok over 0`
- `qwen3dqpack/e4b/fused_attn4_m_a0_d2 attempt 0 load1_median 1.25 gate 6.0 status ok over 0`
- `qwen3dqpack/unsloth/ckpt_unsloth_m_pp attempt 0 load1_median 1.38 gate 6.0 status ok over 0`
