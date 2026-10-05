# TC1 — Qwen3-30B-A3B on one RTX 5090: e4b vs Unsloth vs HF+PEFT vs axolotl at matched work, init and adapter precision (tc1)
Rule (tc1/TC1-PREREG.md): status per attempt in the vocabulary OK / REFUSED / OOM / INSTALL_FAILED / LOAD_FAULT / HARNESS_ERROR / ALARM / NOT_RUN; VALID/VOID per tp4's predicates plus the matched-set predicates (R3: matched init complete, fp32 adapters, step-0 held-out within 0.005 of e4b/reference_attn4_m); VERDICT exactly one of VALID / VOID / QUALITY_FAIL / OOM / UNSUPPORTED / HARNESS_ERROR / ALARM / NOT_RUN (QUALITY_FAIL = held-out |Δ| at N > 0.05 vs e4b/fused_attn4_m); positions = s/step ratios other/e4b from the medians over both draws (stability |d1-d2|/mean ≤ 5 % e4b / 10 % others, else UNSTABLE), quoted only when both arms are VALID; equivalence vs e4b/reference_attn4_m EQUIVALENT ≤ 0.02 / COMPARABLE ≤ 0.05 / else DIVERGENT; frozen base SAME-BYTES / DIFFERENT / N-A per slot; predictions P1–P10 (+ P1b) scored HELD / FALSIFIED / UNTESTED. VOID never enters a ratio or an equivalence reading. Nothing is licensed; no cross-box number is divided into these.
`versions.txt`
```
e4b 0.47.0 @96811ff4697619656b39290546821554559c1b09 (GitHub main)
gnf4 0.40.0 @23a0153ab452cea3131705ce1815f72462f8bf3f (GitHub main)
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
e4b(t212) 0.47.0 @96811ff4697619656b39290546821554559c1b09
gnf4(t212) 0.40.0 @23a0153ab452cea3131705ce1815f72462f8bf3f
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
 "run_id": "tc1-5090-78",
 "instance_id": "54248510",
 "gpu": "NVIDIA GeForce RTX 5090",
 "driver": "595.84",
 "cpu": "AMD EPYC 7B13 64-Core Processor",
 "nproc": 256,
 "mem_total_kb": "2101193408",
 "cgroup_memory_max": "730283900928",
 "disk_root": "overlay         320G   77M  320G   1% /",
 "hostname": "3805a78af83e",
 "registered_gpu_class": "5090",
 "prereg": "tc1/TC1-PREREG.md",
 "local_box": null,
 "local_snapshot": null
}
```

### Qwen3-30B-A3B (amendment 34: the matched arm in three environments -- transformers 5.18 / 5.5 on torch 2.8, and torch 2.12) (`qwen3envsplit`, registered n_layers 48, attention census 192)
- model `Qwen/Qwen3-30B-A3B` @ `ad44e777bcd1`; tokens sha `bfc742f67e37`; N=60; fixture template alpaca seq 2048 micro-batch 2 × accum 4 lr 0.0002 r 16 α 16 optimizer adamw_8bit(lr=0.0002, weight_decay=0.001) schedule=linear warmup_steps=5 autocast False; e4b trainable 642514944; box_class RTX 5090 gpu NVIDIA GeForce RTX 5090
| framework | arm | status | validity | **VERDICT** | matched | init / dtype | step-0 class | N | s/step med(11+) | tok/s | peak GB | J/step | train first→last | held-out 0→final | quality Δ | engagement | trainable | regime | reason / why |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| e4b | fused_attn4_m_e0 | **OK** | VALID | **VALID** | yes | matched:3407 (complete 12480/12480) / float32 | SAME-BYTES-CLASS (0.0000) | 60 | 3.840 | 387.2 | 27.431 | 1152.9 | 2.0614→0.8182 | 1.9441→0.7569 | 0.0000 | patched 48 / kcalls 768 | 642514944 | 4-bit experts (e4b NF4) + NF4 attention |  |
| e4b | fused_attn4_m_e1 | **OK** | VALID | **VALID** | yes | matched:3407 (complete 12480/12480) / float32 | NEAR (0.0292) | 60 | 3.821 | 414.2 | 27.440 | 1123.6 | 2.0832→0.8181 | 1.9733→0.7580 | 0.0011 | patched 48 / kcalls 768 | 642514944 | 4-bit experts (e4b NF4) + NF4 attention |  |
| e4b | fused_attn4_m_e2 | **OK** | VALID | **VALID** | yes | matched:3407 (complete 12480/12480) / float32 | SAME-BYTES-CLASS (0.0038) | 60 | 3.507 | 431.8 | 27.489 | 1073.0 | 2.0554→0.8165 | 1.9478→0.7558 | -0.0011 | patched 48 / kcalls 768 | 642514944 | 4-bit experts (e4b NF4) + NF4 attention |  |
| e4b | fused_attn4_m_e2_d2 | **OK** | VALID | **VALID** | yes | matched:3407 (complete 12480/12480) / float32 | SAME-BYTES-CLASS (0.0038) | 60 | 3.452 | 455.2 | 27.505 | 1054.1 | 2.0554→0.8192 | 1.9478→0.7558 | -0.0010 | patched 48 / kcalls 768 | 642514944 | 4-bit experts (e4b NF4) + NF4 attention |  |
| e4b | fused_attn4_m_e1_d2 | **OK** | VALID | **VALID** | yes | matched:3407 (complete 12480/12480) / float32 | NEAR (0.0292) | 60 | 3.871 | 408.1 | 27.440 | 1131.1 | 2.0832→0.8163 | 1.9733→0.7582 | 0.0014 | patched 48 / kcalls 768 | 642514944 | 4-bit experts (e4b NF4) + NF4 attention |  |
| e4b | fused_attn4_m_e0_d2 | **OK** | VALID | **VALID** | yes | matched:3407 (complete 12480/12480) / float32 | SAME-BYTES-CLASS (0.0000) | 60 | 3.816 | 416.0 | 27.444 | 1118.8 | 2.0614→0.8179 | 1.9441→0.7568 | -0.0001 | patched 48 / kcalls 768 | 642514944 | 4-bit experts (e4b NF4) + NF4 attention |  |
- prologue `e4b/fused_attn4_m_e0` **108.6 s** before step 1 (30% of the arm): c1_before 54.6, c1_after 26.5, load_weights 25.8, eval0 11.2; unattributed 1.331; budget 1260.0
- prologue `e4b/fused_attn4_m_e1` **100.7 s** before step 1 (30% of the arm): c1_before 54.4, c1_after 27.0, load_weights 26.2, attn4 6.2; unattributed 1.143; budget 1260.0
- prologue `e4b/fused_attn4_m_e2` **108.5 s** before step 1 (32% of the arm): c1_before 54.3, c1_after 27.2, load_weights 24.2, eval0 12.8; unattributed 2.011; budget 1260.0
- prologue `e4b/fused_attn4_m_e2_d2` **100.4 s** before step 1 (32% of the arm): c1_before 54.9, c1_after 27.4, load_weights 24.6, attn4 6.1; unattributed 2.13; budget 1260.0
- prologue `e4b/fused_attn4_m_e1_d2` **100.9 s** before step 1 (30% of the arm): c1_before 54.2, c1_after 26.9, load_weights 25.8, attn4 6.4; unattributed 1.422; budget 1260.0
- prologue `e4b/fused_attn4_m_e0_d2` **102.8 s** before step 1 (30% of the arm): c1_before 55.9, c1_after 28.0, load_weights 26.2, attn4 5.8; unattributed 1.266; budget 1260.0
- draws (R1): `e4b/fused_attn4_m_e0` STABLE (3.840/3.816 s, |Δ|/mean 0.6% vs 5%); `e4b/fused_attn4_m_e1` STABLE (3.821/3.871 s, |Δ|/mean 1.3% vs 5%); `e4b/fused_attn4_m_e2` STABLE (3.507/3.452 s, |Δ|/mean 1.6% vs 5%)
- e4b internal parity: NO-ARM
- **NO MATCHED POSITION QUOTED (unsloth)** — not quoted: e4b STABLE / unsloth no receipt
- **NO MATCHED POSITION QUOTED (hf)** — not quoted: e4b STABLE / hf no receipt
- **NO MATCHED POSITION QUOTED (axolotl)** — not quoted: e4b STABLE / axolotl no receipt
- equivalence vs `e4b/fused_attn4_m` (I/H; EQUIVALENT iff both ≤ band = max(0.005, 3 × fused-vs-reference |Δ|) = none (no reference arm: COMPARABLE is the ceiling); COMPARABLE ≤ 0.05; draw-noise floor 0.0003): `e4b/fused_attn4_m_e1` **COMPARABLE** (median step |Δ| 0.0015, |Δ held-out at N| 0.0011, step-0 0.0292 NEAR, |Δ loss at step 2| 0.0017, paired rows mean +0.0011 ± 0.0014 SE over 8, favouring arm 2 / anchor 6) — no reference arm ran: the EQUIVALENT band cannot be set, COMPARABLE is the ceiling; `e4b/fused_attn4_m_e2` **COMPARABLE** (median step |Δ| 0.0014, |Δ held-out at N| 0.0011, step-0 0.0038 SAME-BYTES-CLASS, |Δ loss at step 2| 0.0226, paired rows mean -0.0011 ± 0.0010 SE over 8, favouring arm 4 / anchor 4) — no reference arm ran: the EQUIVALENT band cannot be set, COMPARABLE is the ceiling; `e4b/fused_attn4_m_e2_d2` **COMPARABLE** (median step |Δ| 0.0014, |Δ held-out at N| 0.0010, step-0 0.0038 SAME-BYTES-CLASS, |Δ loss at step 2| 0.0153, paired rows mean -0.0010 ± 0.0015 SE over 8, favouring arm 4 / anchor 4) — no reference arm ran: the EQUIVALENT band cannot be set, COMPARABLE is the ceiling; `e4b/fused_attn4_m_e1_d2` **COMPARABLE** (median step |Δ| 0.0016, |Δ held-out at N| 0.0014, step-0 0.0292 NEAR, |Δ loss at step 2| 0.0114, paired rows mean +0.0014 ± 0.0019 SE over 8, favouring arm 4 / anchor 4) — no reference arm ran: the EQUIVALENT band cannot be set, COMPARABLE is the ceiling; `e4b/fused_attn4_m_e0_d2` **COMPARABLE** (median step |Δ| 0.0016, |Δ held-out at N| 0.0001, step-0 0.0000 SAME-BYTES-CLASS, |Δ loss at step 2| 0.0100, paired rows mean -0.0001 ± 0.0014 SE over 8, favouring arm 6 / anchor 2) — no reference arm ran: the EQUIVALENT band cannot be set, COMPARABLE is the ceiling
- matched_init_sha (B, name-free, canonical slot order): anchor `f7832488926eda91`; `e4b/fused_attn4_m_e0` same; `e4b/fused_attn4_m_e1` same; `e4b/fused_attn4_m_e2` same; `e4b/fused_attn4_m_e2_d2` same; `e4b/fused_attn4_m_e1_d2` same; `e4b/fused_attn4_m_e0_d2` same
- frozen base (R5), anchor `e4b/fused_attn4_m`: gate_up nf4/64 sha 8c3b3fd78e89 control detects, down nf4/64 sha 59569d810a1a control detects, q_proj nf4/64+dq sha 5c5e402ce51d control detects
- frozen base `e4b/fused_attn4_m_e1`: gate_up **SAME-BYTES** (nf4/64) control detects, down **SAME-BYTES** (nf4/64) control detects, q_proj **SAME-BYTES** (nf4/64+dq) control detects
- frozen base `e4b/fused_attn4_m_e2`: gate_up **SAME-BYTES** (nf4/64) control detects, down **SAME-BYTES** (nf4/64) control detects, q_proj **SAME-BYTES** (nf4/64+dq) control detects
- frozen base `e4b/fused_attn4_m_e2_d2`: gate_up **SAME-BYTES** (nf4/64) control detects, down **SAME-BYTES** (nf4/64) control detects, q_proj **SAME-BYTES** (nf4/64+dq) control detects
- frozen base `e4b/fused_attn4_m_e1_d2`: gate_up **SAME-BYTES** (nf4/64) control detects, down **SAME-BYTES** (nf4/64) control detects, q_proj **SAME-BYTES** (nf4/64+dq) control detects
- frozen base `e4b/fused_attn4_m_e0_d2`: gate_up **SAME-BYTES** (nf4/64) control detects, down **SAME-BYTES** (nf4/64) control detects, q_proj **SAME-BYTES** (nf4/64+dq) control detects

## Verdicts
| family | arm | VERDICT | validity | quality | note |
|---|---|---|---|---|---|
| qwen3envsplit | e4b/fused_attn4_m_e0 | **VALID** | VALID | 0.0000 |  |
| qwen3envsplit | e4b/fused_attn4_m_e1 | **VALID** | VALID | 0.0011 |  |
| qwen3envsplit | e4b/fused_attn4_m_e2 | **VALID** | VALID | -0.0011 |  |
| qwen3envsplit | e4b/fused_attn4_m_e2_d2 | **VALID** | VALID | -0.0010 |  |
| qwen3envsplit | e4b/fused_attn4_m_e1_d2 | **VALID** | VALID | 0.0014 |  |
| qwen3envsplit | e4b/fused_attn4_m_e0_d2 | **VALID** | VALID | -0.0001 |  |

## Predictions P62 / P63 / P64 / P65 (TC1-PREREG amendment 34: the matched arm in three environments, two stable draws each; scored mechanically)
| prediction | family | verdict | evidence |
|---|---|---|---|
| P62 | qwen3envsplit | **FALSIFIED** | transformers 5.5 over 5.18 (torch 2.8): e1 / e0 1.005 [0.995, 1.014] vs [0.9, 1.0]; s/step e1 3.821 / 3.871, e0 3.840 / 3.816 |
| P63 | qwen3envsplit | **HELD** | torch 2.12 + triton 3.7 over torch 2.8 + triton 3.4 (transformers 5.5): e2 / e1 0.905 [0.892, 0.918] vs [0.85, 0.99]; s/step e2 3.507 / 3.452, e1 3.821 / 3.871 |
| P64 | qwen3envsplit | **HELD** | the whole environment (amendment 24's 0.882, re-read): e2 / e0 0.909 [0.899, 0.919] vs [0.8, 0.95]; s/step e2 3.507 / 3.452, e0 3.840 / 3.816 |
| P65 | qwen3envsplit | **HELD** | e1 - e0 +0.0013; e2 - e0 -0.0010 (|.| <= 0.005) |

## Load gate (TC1-PREREG amendment 33): 6 draw(s) voided for host load and run again
Each line: the arm, the attempt, the median host load1 over that attempt's run, the gate. A VOID attempt's files are in loadvoid/; the last attempt of each arm stands whatever its load.

- `qwen3envsplit/e4b/fused_attn4_m_e0 attempt 0 load1_median 4.2 gate 6.0 status ok over 0`
- `qwen3envsplit/e4b/fused_attn4_m_e1 attempt 0 load1_median 14.32 gate 6.0 status ok over 1`
- `qwen3envsplit/e4b/fused_attn4_m_e1 attempt 0 VOID (host load1 median 14.32 > 6.0): re-run 1 of 2`
- `qwen3envsplit/e4b/fused_attn4_m_e1 attempt 1 load1_median 3.84 gate 6.0 status ok over 0`
- `qwen3envsplit/e4b/fused_attn4_m_e2 attempt 0 load1_median 4.01 gate 6.0 status ok over 0`
- `qwen3envsplit/e4b/fused_attn4_m_e2_d2 attempt 0 load1_median 6.55 gate 6.0 status ok over 1`
- `qwen3envsplit/e4b/fused_attn4_m_e2_d2 attempt 0 VOID (host load1 median 6.55 > 6.0): re-run 1 of 2`
- `qwen3envsplit/e4b/fused_attn4_m_e2_d2 attempt 1 load1_median 11.26 gate 6.0 status ok over 1`
- `qwen3envsplit/e4b/fused_attn4_m_e2_d2 attempt 1 VOID (host load1 median 11.26 > 6.0): re-run 2 of 2`
- `qwen3envsplit/e4b/fused_attn4_m_e2_d2 attempt 2 load1_median 4.57 gate 6.0 status ok over 0`
- `qwen3envsplit/e4b/fused_attn4_m_e1_d2 attempt 0 load1_median 6.02 gate 6.0 status ok over 1`
- `qwen3envsplit/e4b/fused_attn4_m_e1_d2 attempt 0 VOID (host load1 median 6.02 > 6.0): re-run 1 of 2`
- `qwen3envsplit/e4b/fused_attn4_m_e1_d2 attempt 1 load1_median 4.33 gate 6.0 status ok over 0`
- `qwen3envsplit/e4b/fused_attn4_m_e0_d2 attempt 0 load1_median 12.01 gate 6.0 status ok over 1`
- `qwen3envsplit/e4b/fused_attn4_m_e0_d2 attempt 0 VOID (host load1 median 12.01 > 6.0): re-run 1 of 2`
- `qwen3envsplit/e4b/fused_attn4_m_e0_d2 attempt 1 load1_median 9.5 gate 6.0 status ok over 1`
- `qwen3envsplit/e4b/fused_attn4_m_e0_d2 attempt 1 VOID (host load1 median 9.5 > 6.0): re-run 2 of 2`
- `qwen3envsplit/e4b/fused_attn4_m_e0_d2 attempt 2 load1_median 9.15 gate 6.0 status ok over 1`
