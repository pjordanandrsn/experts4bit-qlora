# TC1 — Qwen3-30B-A3B on one RTX 5090: e4b vs Unsloth vs HF+PEFT vs axolotl at matched work, init and adapter precision (tc1)
Rule (tc1/TC1-PREREG.md): status per attempt in the vocabulary OK / REFUSED / OOM / INSTALL_FAILED / LOAD_FAULT / HARNESS_ERROR / ALARM / NOT_RUN; VALID/VOID per tp4's predicates plus the matched-set predicates (R3: matched init complete, fp32 adapters, step-0 held-out within 0.005 of e4b/reference_attn4_m); VERDICT exactly one of VALID / VOID / QUALITY_FAIL / OOM / UNSUPPORTED / HARNESS_ERROR / ALARM / NOT_RUN (QUALITY_FAIL = held-out |Δ| at N > 0.05 vs e4b/fused_attn4_m); positions = s/step ratios other/e4b from the medians over both draws (stability |d1-d2|/mean ≤ 5 % e4b / 10 % others, else UNSTABLE), quoted only when both arms are VALID; equivalence vs e4b/reference_attn4_m EQUIVALENT ≤ 0.02 / COMPARABLE ≤ 0.05 / else DIVERGENT; frozen base SAME-BYTES / DIFFERENT / N-A per slot; predictions P1–P10 (+ P1b) scored HELD / FALSIFIED / UNTESTED. VOID never enters a ratio or an equivalence reading. Nothing is licensed; no cross-box number is divided into these.
`versions.txt`
```
e4b 0.47.0 @23edeff2c52543d52aa9b461137e2fede163067a (GitHub main)
gnf4 0.40.0 @c4a683b30af23e4a15e6428ce1e691b7bcce1117 (GitHub main)
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
e4b(t212) 0.47.0 @23edeff2c52543d52aa9b461137e2fede163067a
gnf4(t212) 0.40.0 @c4a683b30af23e4a15e6428ce1e691b7bcce1117
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
 "run_id": "tc1-5090-80",
 "instance_id": "54259219",
 "gpu": "NVIDIA GeForce RTX 5090",
 "driver": "595.84",
 "cpu": "AMD EPYC 7B13 64-Core Processor",
 "nproc": 256,
 "mem_total_kb": "2101193408",
 "cgroup_memory_max": "730283900928",
 "disk_root": "overlay         320G   77M  320G   1% /",
 "hostname": "ad70abae4bcd",
 "registered_gpu_class": "5090",
 "prereg": "tc1/TC1-PREREG.md",
 "local_box": null,
 "local_snapshot": null
}
```

### Qwen3-30B-A3B (amendment 36: grouped-nf4-gemm's padded LoRA delta saving its block vs its input, venv-unsloth) (`qwen3compactab`, registered n_layers 48, attention census 192)
- model `Qwen/Qwen3-30B-A3B` @ `ad44e777bcd1`; tokens sha `bfc742f67e37`; N=60; fixture template alpaca seq 2048 micro-batch 2 × accum 4 lr 0.0002 r 16 α 16 optimizer adamw_8bit(lr=0.0002, weight_decay=0.001) schedule=linear warmup_steps=5 autocast False; e4b trainable 642514944; box_class RTX 5090 gpu NVIDIA GeForce RTX 5090
| framework | arm | status | validity | **VERDICT** | matched | init / dtype | step-0 class | N | s/step med(11+) | tok/s | peak GB | J/step | train first→last | held-out 0→final | quality Δ | engagement | trainable | regime | reason / why |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| e4b | fused_attn4_shipped_cd0 | **OK** | VALID | **VALID** | native | native / bfloat16,float32 | — | 60 | 2.919 | 538.0 | 24.673 | 807.7 | 2.0554→0.8175 | 1.9478→0.7560 | 0.0000 | patched 48 / kcalls 768 | 642514944 | 4-bit experts (e4b NF4) + NF4 attention |  |
| e4b | fused_attn4_shipped_cd1 | **OK** | VALID | **VALID** | native | native / bfloat16,float32 | — | 60 | 2.793 | 565.1 | 24.673 | 803.6 | 2.0554→0.8201 | 1.9478→0.7557 | -0.0003 | patched 48 / kcalls 768 | 642514944 | 4-bit experts (e4b NF4) + NF4 attention |  |
| e4b | fused_attn4_m_cd0 | **OK** | VALID | **VALID** | yes | matched:3407 (complete 12480/12480) / float32 | SAME-BYTES-CLASS (0.0000) | 60 | 3.431 | 460.5 | 27.490 | 1002.3 | 2.0554→0.8199 | 1.9478→0.7560 | 0.0000 | patched 48 / kcalls 768 | 642514944 | 4-bit experts (e4b NF4) + NF4 attention |  |
| e4b | fused_attn4_m_cd1 | **OK** | VALID | **VALID** | yes | matched:3407 (complete 12480/12480) / float32 | SAME-BYTES-CLASS (0.0000) | 60 | 3.348 | 473.9 | 27.722 | 1024.0 | 2.0554→0.8163 | 1.9478→0.7572 | 0.0012 | patched 48 / kcalls 768 | 642514944 | 4-bit experts (e4b NF4) + NF4 attention |  |
| e4b | fused_attn4_m_cd1_d2 | **OK** | VALID | **VALID** | yes | matched:3407 (complete 12480/12480) / float32 | SAME-BYTES-CLASS (0.0000) | 60 | 3.299 | 476.0 | 27.716 | 1013.5 | 2.0554→0.8179 | 1.9478→0.7576 | 0.0016 | patched 48 / kcalls 768 | 642514944 | 4-bit experts (e4b NF4) + NF4 attention |  |
| e4b | fused_attn4_m_cd0_d2 | **OK** | VALID | **VALID** | yes | matched:3407 (complete 12480/12480) / float32 | SAME-BYTES-CLASS (0.0000) | 60 | 3.428 | 455.2 | 27.490 | 1013.6 | 2.0554→0.8177 | 1.9478→0.7579 | 0.0019 | patched 48 / kcalls 768 | 642514944 | 4-bit experts (e4b NF4) + NF4 attention |  |
| e4b | fused_attn4_shipped_cd1_d2 | **OK** | VALID | **VALID** | native | native / bfloat16,float32 | — | 60 | 2.810 | 558.9 | 24.673 | 812.6 | 2.0554→0.8159 | 1.9478→0.7552 | -0.0008 | patched 48 / kcalls 768 | 642514944 | 4-bit experts (e4b NF4) + NF4 attention |  |
| e4b | fused_attn4_shipped_cd0_d2 | **OK** | VALID | **VALID** | native | native / bfloat16,float32 | — | 60 | 2.858 | 544.9 | 24.673 | 810.1 | 2.0554→0.8155 | 1.9478→0.7561 | 0.0001 | patched 48 / kcalls 768 | 642514944 | 4-bit experts (e4b NF4) + NF4 attention |  |
- prologue `e4b/fused_attn4_shipped_cd0` **107.1 s** before step 1 (37% of the arm): c1_before 55.1, load_weights 29.2, c1_after 27.6, trainable_sha 6.2; unattributed 2.0; budget 1260.0
- prologue `e4b/fused_attn4_shipped_cd1` **104.2 s** before step 1 (38% of the arm): c1_before 55.9, c1_after 27.9, load_weights 25.9, trainable_sha 6.1; unattributed 2.004; budget 1260.0
- prologue `e4b/fused_attn4_m_cd0` **101.0 s** before step 1 (32% of the arm): c1_before 55.2, c1_after 27.5, load_weights 25.7, attn4 5.5; unattributed 1.994; budget 1260.0
- prologue `e4b/fused_attn4_m_cd1` **101.2 s** before step 1 (33% of the arm): c1_before 55.1, c1_after 27.4, load_weights 25.8, attn4 5.7; unattributed 2.004; budget 1260.0
- prologue `e4b/fused_attn4_m_cd1_d2` **101.2 s** before step 1 (33% of the arm): c1_before 54.6, c1_after 28.3, load_weights 26.6, attn4 5.5; unattributed 1.993; budget 1260.0
- prologue `e4b/fused_attn4_m_cd0_d2` **101.5 s** before step 1 (32% of the arm): c1_before 54.7, c1_after 27.7, load_weights 26.1, attn4 5.8; unattributed 2.077; budget 1260.0
- prologue `e4b/fused_attn4_shipped_cd1_d2` **104.4 s** before step 1 (37% of the arm): c1_before 56.0, c1_after 27.9, load_weights 26.0, trainable_sha 5.9; unattributed 2.011; budget 1260.0
- prologue `e4b/fused_attn4_shipped_cd0_d2` **106.0 s** before step 1 (37% of the arm): c1_before 57.0, c1_after 27.6, load_weights 26.6, trainable_sha 6.2; unattributed 1.976; budget 1260.0
- draws (R1): `e4b/fused_attn4_shipped_cd0` STABLE (2.919/2.858 s, |Δ|/mean 2.1% vs 5%); `e4b/fused_attn4_shipped_cd1` STABLE (2.793/2.810 s, |Δ|/mean 0.6% vs 5%); `e4b/fused_attn4_m_cd0` STABLE (3.431/3.428 s, |Δ|/mean 0.1% vs 5%); `e4b/fused_attn4_m_cd1` STABLE (3.348/3.299 s, |Δ|/mean 1.5% vs 5%)
- e4b internal parity: NO-ARM
- **NO MATCHED POSITION QUOTED (unsloth)** — not quoted: e4b STABLE / unsloth no receipt
- **NO MATCHED POSITION QUOTED (hf)** — not quoted: e4b STABLE / hf no receipt
- **NO MATCHED POSITION QUOTED (axolotl)** — not quoted: e4b STABLE / axolotl no receipt
- equivalence vs `e4b/fused_attn4_m` (I/H; EQUIVALENT iff both ≤ band = max(0.005, 3 × fused-vs-reference |Δ|) = none (no reference arm: COMPARABLE is the ceiling); COMPARABLE ≤ 0.05; draw-noise floor 0.0019): `e4b/fused_attn4_m_cd1` **COMPARABLE** (median step |Δ| 0.0011, |Δ held-out at N| 0.0012, step-0 0.0000 SAME-BYTES-CLASS, |Δ loss at step 2| 0.0179, paired rows mean +0.0012 ± 0.0020 SE over 8, favouring arm 4 / anchor 4) — no reference arm ran: the EQUIVALENT band cannot be set, COMPARABLE is the ceiling; `e4b/fused_attn4_m_cd1_d2` **COMPARABLE** (median step |Δ| 0.0013, |Δ held-out at N| 0.0016, step-0 0.0000 SAME-BYTES-CLASS, |Δ loss at step 2| 0.0104, paired rows mean +0.0016 ± 0.0021 SE over 8, favouring arm 3 / anchor 5) — no reference arm ran: the EQUIVALENT band cannot be set, COMPARABLE is the ceiling; `e4b/fused_attn4_m_cd0_d2` **COMPARABLE** (median step |Δ| 0.0013, |Δ held-out at N| 0.0019, step-0 0.0000 SAME-BYTES-CLASS, |Δ loss at step 2| 0.0005, paired rows mean +0.0019 ± 0.0018 SE over 8, favouring arm 4 / anchor 4) — no reference arm ran: the EQUIVALENT band cannot be set, COMPARABLE is the ceiling
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
| qwen3compactab | e4b/fused_attn4_shipped_cd0 | **VALID** | VALID | 0.0000 |  |
| qwen3compactab | e4b/fused_attn4_shipped_cd1 | **VALID** | VALID | -0.0003 |  |
| qwen3compactab | e4b/fused_attn4_m_cd0 | **VALID** | VALID | 0.0000 |  |
| qwen3compactab | e4b/fused_attn4_m_cd1 | **VALID** | VALID | 0.0012 |  |
| qwen3compactab | e4b/fused_attn4_m_cd1_d2 | **VALID** | VALID | 0.0016 |  |
| qwen3compactab | e4b/fused_attn4_m_cd0_d2 | **VALID** | VALID | 0.0019 |  |
| qwen3compactab | e4b/fused_attn4_shipped_cd1_d2 | **VALID** | VALID | -0.0008 |  |
| qwen3compactab | e4b/fused_attn4_shipped_cd0_d2 | **VALID** | VALID | 0.0001 |  |

## Predictions P69 / P70 / P71 / P72 (TC1-PREREG amendment 36: the compact padded LoRA delta off vs on, venv-unsloth, two stable draws a side; scored mechanically)
| prediction | family | verdict | evidence |
|---|---|---|---|
| P69 | qwen3compactab | **FALSIFIED** | matched: peak cd0 27.490 / cd1 27.719 GB, drop -0.229 vs [0.3, 2.0] |
| P70 | qwen3compactab | **FALSIFIED** | matched: cd1 / cd0 0.969 [0.962, 0.977 over 4 cross-draw ratios] vs [0.97, 1.02]; s/step cd0 3.431 / 3.428 (within 0.1%), cd1 3.348 / 3.299 (within 1.5%); peak cd0 27.49 / cd1 27.72 GB |
| P71 | qwen3compactab | **FALSIFIED** | shipped: cd1 / cd0 0.970 [0.957, 0.983 over 4 cross-draw ratios] vs [0.97, 1.02]; s/step cd0 2.919 / 2.858 (within 2.1%), cd1 2.793 / 2.810 (within 0.6%); peak cd0 24.67 / cd1 24.67 GB |
| P72 | qwen3compactab | **HELD** | matched: mean held-out cd1 - cd0 +0.0004 (|.| <= 0.005); held-out at N cd0 [0.756, 0.7579] cd1 [0.7572, 0.7576]; shipped: mean held-out cd1 - cd0 -0.0006 (|.| <= 0.005); held-out at N cd0 [0.756, 0.7561] cd1 [0.7557, 0.7552] |

## Load gate (TC1-PREREG amendment 33): 2 draw(s) voided for host load and run again
Each line: the arm, the attempt, the median host load1 over that attempt's run, the gate. A VOID attempt's files are in loadvoid/; the last attempt of each arm stands whatever its load.

- `qwen3compactab/e4b/fused_attn4_shipped_cd0 attempt 0 load1_median 8.24 gate 6.0 status ok over 1`
- `qwen3compactab/e4b/fused_attn4_shipped_cd0 attempt 0 VOID (host load1 median 8.24 > 6.0): re-run 1 of 2`
- `qwen3compactab/e4b/fused_attn4_shipped_cd0 attempt 1 load1_median 4.5 gate 6.0 status ok over 0`
- `qwen3compactab/e4b/fused_attn4_shipped_cd1 attempt 0 load1_median 8.16 gate 6.0 status ok over 1`
- `qwen3compactab/e4b/fused_attn4_shipped_cd1 attempt 0 VOID (host load1 median 8.16 > 6.0): re-run 1 of 2`
- `qwen3compactab/e4b/fused_attn4_shipped_cd1 attempt 1 load1_median 5.43 gate 6.0 status ok over 0`
- `qwen3compactab/e4b/fused_attn4_m_cd0 attempt 0 load1_median 3.49 gate 6.0 status ok over 0`
- `qwen3compactab/e4b/fused_attn4_m_cd1 attempt 0 load1_median 3.34 gate 6.0 status ok over 0`
- `qwen3compactab/e4b/fused_attn4_m_cd1_d2 attempt 0 load1_median 3.24 gate 6.0 status ok over 0`
- `qwen3compactab/e4b/fused_attn4_m_cd0_d2 attempt 0 load1_median 4.58 gate 6.0 status ok over 0`
- `qwen3compactab/e4b/fused_attn4_shipped_cd1_d2 attempt 0 load1_median 3.59 gate 6.0 status ok over 0`
- `qwen3compactab/e4b/fused_attn4_shipped_cd0_d2 attempt 0 load1_median 1.84 gate 6.0 status ok over 0`
