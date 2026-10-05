# TC1 — Qwen3-30B-A3B on one RTX 5090: e4b vs Unsloth vs HF+PEFT vs axolotl at matched work, init and adapter precision (tc1)
Rule (tc1/TC1-PREREG.md): status per attempt in the vocabulary OK / REFUSED / OOM / INSTALL_FAILED / LOAD_FAULT / HARNESS_ERROR / ALARM / NOT_RUN; VALID/VOID per tp4's predicates plus the matched-set predicates (R3: matched init complete, fp32 adapters, step-0 held-out within 0.005 of e4b/reference_attn4_m); VERDICT exactly one of VALID / VOID / QUALITY_FAIL / OOM / UNSUPPORTED / HARNESS_ERROR / ALARM / NOT_RUN (QUALITY_FAIL = held-out |Δ| at N > 0.05 vs e4b/fused_attn4_m); positions = s/step ratios other/e4b from the medians over both draws (stability |d1-d2|/mean ≤ 5 % e4b / 10 % others, else UNSTABLE), quoted only when both arms are VALID; equivalence vs e4b/reference_attn4_m EQUIVALENT ≤ 0.02 / COMPARABLE ≤ 0.05 / else DIVERGENT; frozen base SAME-BYTES / DIFFERENT / N-A per slot; predictions P1–P10 (+ P1b) scored HELD / FALSIFIED / UNTESTED. VOID never enters a ratio or an equivalence reading. Nothing is licensed; no cross-box number is divided into these.
`versions.txt`
```
e4b 0.48.0 @64afc6da16cf0abcfc3c57d07e4b275311656ab8 (GitHub main)
gnf4 0.41.0 @9622144c734395de4d40332f06dea92135d858de (GitHub main)
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
e4b(t212) 0.48.0 @64afc6da16cf0abcfc3c57d07e4b275311656ab8
gnf4(t212) 0.41.0 @9622144c734395de4d40332f06dea92135d858de
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
 "run_id": "tc1-5090-83",
 "instance_id": "54275103",
 "gpu": "NVIDIA GeForce RTX 5090",
 "driver": "595.71.05",
 "cpu": "AMD EPYC 7702P 64-Core Processor",
 "nproc": 126,
 "mem_total_kb": "527981296",
 "cgroup_memory_max": "259512074240",
 "disk_root": "overlay         320G  9.8M  320G   1% /",
 "hostname": "aaa3a4b1b8e4",
 "registered_gpu_class": "5090",
 "prereg": "tc1/TC1-PREREG.md",
 "local_box": null,
 "local_snapshot": null
}
```

### Qwen3-30B-A3B (amendment 37: the compact padded LoRA delta off vs on with its backward freeing early, venv-unsloth) (`qwen3compactab2`, registered n_layers 48, attention census 192)
- model `Qwen/Qwen3-30B-A3B` @ `ad44e777bcd1`; tokens sha `bfc742f67e37`; N=60; fixture template alpaca seq 2048 micro-batch 2 × accum 4 lr 0.0002 r 16 α 16 optimizer adamw_8bit(lr=0.0002, weight_decay=0.001) schedule=linear warmup_steps=5 autocast False; e4b trainable 642514944; box_class RTX 5090 gpu NVIDIA GeForce RTX 5090
| framework | arm | status | validity | **VERDICT** | matched | init / dtype | step-0 class | N | s/step med(11+) | tok/s | peak GB | J/step | train first→last | held-out 0→final | quality Δ | engagement | trainable | regime | reason / why |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| e4b | fused_attn4_shipped_cd0 | **OK** | VALID | **VALID** | native | native / bfloat16,float32 | — | 60 | 3.026 | 486.8 | 24.673 | 879.4 | 2.0554→0.8159 | 1.9478→0.7570 | 0.0025 | patched 48 / kcalls 768 | 642514944 | 4-bit experts (e4b NF4) + NF4 attention |  |
| e4b | fused_attn4_shipped_cd1 | **OK** | VALID | **VALID** | native | native / bfloat16,float32 | — | 60 | 2.824 | 551.1 | 24.673 | 829.4 | 2.0554→0.8179 | 1.9478→0.7564 | 0.0020 | patched 48 / kcalls 768 | 642514944 | 4-bit experts (e4b NF4) + NF4 attention |  |
| e4b | fused_attn4_m_cd0 | **OK** | VALID | **VALID** | yes | matched:3407 (complete 12480/12480) / float32 | SAME-BYTES-CLASS (0.0000) | 60 | 3.914 | 402.9 | 27.478 | 1050.5 | 2.0554→0.8168 | 1.9478→0.7544 | 0.0000 | patched 48 / kcalls 768 | 642514944 | 4-bit experts (e4b NF4) + NF4 attention |  |
| e4b | fused_attn4_m_cd1 | **OK** | VALID | **VALID** | yes | matched:3407 (complete 12480/12480) / float32 | SAME-BYTES-CLASS (0.0000) | 60 | 3.752 | 416.6 | 27.189 | 1071.6 | 2.0554→0.8182 | 1.9478→0.7595 | 0.0051 | patched 48 / kcalls 768 | 642514944 | 4-bit experts (e4b NF4) + NF4 attention |  |
| e4b | fused_attn4_m_cd1_d2 | **OK** | VALID | **VALID** | yes | matched:3407 (complete 12480/12480) / float32 | SAME-BYTES-CLASS (0.0000) | 60 | 3.763 | 418.1 | 27.189 | 1070.6 | 2.0554→0.8182 | 1.9478→0.7570 | 0.0025 | patched 48 / kcalls 768 | 642514944 | 4-bit experts (e4b NF4) + NF4 attention |  |
| e4b | fused_attn4_m_cd0_d2 | **OK** | VALID | **VALID** | yes | matched:3407 (complete 12480/12480) / float32 | SAME-BYTES-CLASS (0.0000) | 60 | 3.860 | 406.4 | 27.476 | 1063.8 | 2.0554→0.8189 | 1.9478→0.7563 | 0.0019 | patched 48 / kcalls 768 | 642514944 | 4-bit experts (e4b NF4) + NF4 attention |  |
| e4b | fused_attn4_shipped_cd1_d2 | **OK** | VALID | **VALID** | native | native / bfloat16,float32 | — | 60 | 2.864 | 543.3 | 24.673 | 839.5 | 2.0554→0.8192 | 1.9478→0.7584 | 0.0039 | patched 48 / kcalls 768 | 642514944 | 4-bit experts (e4b NF4) + NF4 attention |  |
| e4b | fused_attn4_shipped_cd0_d2 | **OK** | VALID | **VALID** | native | native / bfloat16,float32 | — | 60 | 2.970 | 523.1 | 24.673 | 842.1 | 2.0554→0.8177 | 1.9478→0.7585 | 0.0041 | patched 48 / kcalls 768 | 642514944 | 4-bit experts (e4b NF4) + NF4 attention |  |
- prologue `e4b/fused_attn4_shipped_cd0` **119.2 s** before step 1 (37% of the arm): c1_before 61.9, c1_after 31.1, load_weights 23.2, eval0 15.4; unattributed 2.141; budget 1260.0
- prologue `e4b/fused_attn4_shipped_cd1` **110.0 s** before step 1 (38% of the arm): c1_before 61.8, c1_after 31.0, load_weights 27.6, attn4 6.4; unattributed 2.163; budget 1260.0
- prologue `e4b/fused_attn4_m_cd0` **107.9 s** before step 1 (31% of the arm): c1_before 63.0, c1_after 32.0, load_weights 23.3, attn4 6.5; unattributed 2.156; budget 1260.0
- prologue `e4b/fused_attn4_m_cd1` **107.2 s** before step 1 (31% of the arm): c1_before 62.6, c1_after 30.8, load_weights 23.2, attn4 6.5; unattributed 2.141; budget 1260.0
- prologue `e4b/fused_attn4_m_cd1_d2` **107.0 s** before step 1 (31% of the arm): c1_before 62.5, c1_after 31.8, load_weights 23.2, attn4 6.4; unattributed 2.175; budget 1260.0
- prologue `e4b/fused_attn4_m_cd0_d2` **106.3 s** before step 1 (31% of the arm): c1_before 61.5, c1_after 30.9, load_weights 23.2, attn4 6.5; unattributed 2.217; budget 1260.0
- prologue `e4b/fused_attn4_shipped_cd1_d2` **105.6 s** before step 1 (37% of the arm): c1_before 61.8, c1_after 31.1, load_weights 23.2, attn4 6.5; unattributed 2.134; budget 1260.0
- prologue `e4b/fused_attn4_shipped_cd0_d2` **105.5 s** before step 1 (36% of the arm): c1_before 61.7, c1_after 30.9, load_weights 23.2, attn4 6.5; unattributed 2.128; budget 1260.0
- draws (R1): `e4b/fused_attn4_shipped_cd0` STABLE (3.026/2.970 s, |Δ|/mean 1.9% vs 5%); `e4b/fused_attn4_shipped_cd1` STABLE (2.824/2.864 s, |Δ|/mean 1.4% vs 5%); `e4b/fused_attn4_m_cd0` STABLE (3.914/3.860 s, |Δ|/mean 1.4% vs 5%); `e4b/fused_attn4_m_cd1` STABLE (3.752/3.763 s, |Δ|/mean 0.3% vs 5%)
- e4b internal parity: NO-ARM
- **NO MATCHED POSITION QUOTED (unsloth)** — not quoted: e4b STABLE / unsloth no receipt
- **NO MATCHED POSITION QUOTED (hf)** — not quoted: e4b STABLE / hf no receipt
- **NO MATCHED POSITION QUOTED (axolotl)** — not quoted: e4b STABLE / axolotl no receipt
- equivalence vs `e4b/fused_attn4_m` (I/H; EQUIVALENT iff both ≤ band = max(0.005, 3 × fused-vs-reference |Δ|) = none (no reference arm: COMPARABLE is the ceiling); COMPARABLE ≤ 0.05; draw-noise floor 0.0025): `e4b/fused_attn4_m_cd1` **COMPARABLE** (median step |Δ| 0.0012, |Δ held-out at N| 0.0051, step-0 0.0000 SAME-BYTES-CLASS, |Δ loss at step 2| 0.0089, paired rows mean +0.0051 ± 0.0019 SE over 8, favouring arm 1 / anchor 7) — no reference arm ran: the EQUIVALENT band cannot be set, COMPARABLE is the ceiling; `e4b/fused_attn4_m_cd1_d2` **COMPARABLE** (median step |Δ| 0.0012, |Δ held-out at N| 0.0025, step-0 0.0000 SAME-BYTES-CLASS, |Δ loss at step 2| 0.0093, paired rows mean +0.0025 ± 0.0020 SE over 8, favouring arm 4 / anchor 4) — no reference arm ran: the EQUIVALENT band cannot be set, COMPARABLE is the ceiling; `e4b/fused_attn4_m_cd0_d2` **COMPARABLE** (median step |Δ| 0.0013, |Δ held-out at N| 0.0019, step-0 0.0000 SAME-BYTES-CLASS, |Δ loss at step 2| 0.0204, paired rows mean +0.0019 ± 0.0027 SE over 8, favouring arm 3 / anchor 5) — no reference arm ran: the EQUIVALENT band cannot be set, COMPARABLE is the ceiling
- matched_init_sha (B, name-free, canonical slot order): anchor `f7832488926eda91`; `e4b/fused_attn4_m_cd0` same; `e4b/fused_attn4_m_cd1` same; `e4b/fused_attn4_m_cd1_d2` same; `e4b/fused_attn4_m_cd0_d2` same
- frozen base (R5), anchor `e4b/fused_attn4_m`: gate_up nf4/64 sha 8c3b3fd78e89 control detects, down nf4/64 sha 59569d810a1a control detects, q_proj nf4/64+dq sha 5c5e402ce51d control detects
- frozen base `e4b/fused_attn4_shipped_cd0`: gate_up **SAME-BYTES** (nf4/64) control detects, down **SAME-BYTES** (nf4/64) control detects, q_proj **SAME-BYTES** (nf4/64+dq) control detects
- frozen base `e4b/fused_attn4_shipped_cd1`: gate_up **SAME-BYTES** (nf4/64) control detects, down **SAME-BYTES** (nf4/64) control detects, q_proj **SAME-BYTES** (nf4/64+dq) control detects
- frozen base `e4b/fused_attn4_m_cd1`: gate_up **SAME-BYTES** (nf4/64) control detects, down **SAME-BYTES** (nf4/64) control detects, q_proj **SAME-BYTES** (nf4/64+dq) control detects
- frozen base `e4b/fused_attn4_m_cd1_d2`: gate_up **SAME-BYTES** (nf4/64) control detects, down **SAME-BYTES** (nf4/64) control detects, q_proj **SAME-BYTES** (nf4/64+dq) control detects
- frozen base `e4b/fused_attn4_m_cd0_d2`: gate_up **SAME-BYTES** (nf4/64) control detects, down **SAME-BYTES** (nf4/64) control detects, q_proj **SAME-BYTES** (nf4/64+dq) control detects
- frozen base `e4b/fused_attn4_shipped_cd1_d2`: gate_up **SAME-BYTES** (nf4/64) control detects, down **SAME-BYTES** (nf4/64) control detects, q_proj **SAME-BYTES** (nf4/64+dq) control detects
- frozen base `e4b/fused_attn4_shipped_cd0_d2`: gate_up **SAME-BYTES** (nf4/64) control detects, down **SAME-BYTES** (nf4/64) control detects, q_proj **SAME-BYTES** (nf4/64+dq) control detects

## Verdicts
| family | arm | VERDICT | validity | quality | note |
|---|---|---|---|---|---|
| qwen3compactab2 | e4b/fused_attn4_shipped_cd0 | **VALID** | VALID | 0.0025 |  |
| qwen3compactab2 | e4b/fused_attn4_shipped_cd1 | **VALID** | VALID | 0.0020 |  |
| qwen3compactab2 | e4b/fused_attn4_m_cd0 | **VALID** | VALID | 0.0000 |  |
| qwen3compactab2 | e4b/fused_attn4_m_cd1 | **VALID** | VALID | 0.0051 |  |
| qwen3compactab2 | e4b/fused_attn4_m_cd1_d2 | **VALID** | VALID | 0.0025 |  |
| qwen3compactab2 | e4b/fused_attn4_m_cd0_d2 | **VALID** | VALID | 0.0019 |  |
| qwen3compactab2 | e4b/fused_attn4_shipped_cd1_d2 | **VALID** | VALID | 0.0039 |  |
| qwen3compactab2 | e4b/fused_attn4_shipped_cd0_d2 | **VALID** | VALID | 0.0041 |  |

## Predictions P73 / P74 / P75 / P76 (TC1-PREREG amendment 37: the compact padded LoRA delta off vs on with its backward freeing early, venv-unsloth; scored mechanically)
| prediction | family | verdict | evidence |
|---|---|---|---|
| P73 | qwen3compactab2 | **HELD** | matched: peak cd0 27.477 / cd1 27.189 GB, drop 0.288 vs [-0.05, 0.5] |
| P74 | qwen3compactab2 | **HELD** | matched: cd1 / cd0 0.967 [0.959, 0.975 over 4 cross-draw ratios] vs [0.95, 0.99]; s/step cd0 3.914 / 3.860 (within 1.4%), cd1 3.752 / 3.763 (within 0.3%); peak cd0 27.48 / cd1 27.19 GB |
| P75 | qwen3compactab2 | **FALSIFIED** | shipped: cd1 / cd0 0.948 [0.933, 0.964 over 4 cross-draw ratios] vs [0.95, 0.99]; s/step cd0 3.026 / 2.970 (within 1.9%), cd1 2.824 / 2.864 (within 1.4%); peak cd0 24.67 / cd1 24.67 GB |
| P76 | qwen3compactab2 | **HELD** | matched: mean held-out cd1 - cd0 +0.0029 (|.| <= 0.005); held-out at N cd0 [0.7544, 0.7563] cd1 [0.7595, 0.757]; shipped: mean held-out cd1 - cd0 -0.0003 (|.| <= 0.005); held-out at N cd0 [0.757, 0.7585] cd1 [0.7564, 0.7584] |

## Load gate (TC1-PREREG amendment 33): 0 draw(s) voided for host load and run again
Each line: the arm, the attempt, the median host load1 over that attempt's run, the gate. A VOID attempt's files are in loadvoid/; the last attempt of each arm stands whatever its load.

- `qwen3compactab2/e4b/fused_attn4_shipped_cd0 attempt 0 load1_median 1.39 gate 6.0 status ok over 0`
- `qwen3compactab2/e4b/fused_attn4_shipped_cd1 attempt 0 load1_median 1.89 gate 6.0 status ok over 0`
- `qwen3compactab2/e4b/fused_attn4_m_cd0 attempt 0 load1_median 2.84 gate 6.0 status ok over 0`
- `qwen3compactab2/e4b/fused_attn4_m_cd1 attempt 0 load1_median 3.43 gate 6.0 status ok over 0`
- `qwen3compactab2/e4b/fused_attn4_m_cd1_d2 attempt 0 load1_median 2.88 gate 6.0 status ok over 0`
- `qwen3compactab2/e4b/fused_attn4_m_cd0_d2 attempt 0 load1_median 2.12 gate 6.0 status ok over 0`
- `qwen3compactab2/e4b/fused_attn4_shipped_cd1_d2 attempt 0 load1_median 1.92 gate 6.0 status ok over 0`
- `qwen3compactab2/e4b/fused_attn4_shipped_cd0_d2 attempt 0 load1_median 1.61 gate 6.0 status ok over 0`
