# TC1 — Qwen3-30B-A3B on one RTX 5090: e4b vs Unsloth vs HF+PEFT vs axolotl at matched work, init and adapter precision (tc1)
Rule (tc1/TC1-PREREG.md): status per attempt in the vocabulary OK / REFUSED / OOM / INSTALL_FAILED / LOAD_FAULT / HARNESS_ERROR / ALARM / NOT_RUN; VALID/VOID per tp4's predicates plus the matched-set predicates (R3: matched init complete, fp32 adapters, step-0 held-out within 0.005 of e4b/reference_attn4_m); VERDICT exactly one of VALID / VOID / QUALITY_FAIL / OOM / UNSUPPORTED / HARNESS_ERROR / ALARM / NOT_RUN (QUALITY_FAIL = held-out |Δ| at N > 0.05 vs e4b/fused_attn4_m); positions = s/step ratios other/e4b from the medians over both draws (stability |d1-d2|/mean ≤ 5 % e4b / 10 % others, else UNSTABLE), quoted only when both arms are VALID; equivalence vs e4b/reference_attn4_m EQUIVALENT ≤ 0.02 / COMPARABLE ≤ 0.05 / else DIVERGENT; frozen base SAME-BYTES / DIFFERENT / N-A per slot; predictions P1–P10 (+ P1b) scored HELD / FALSIFIED / UNTESTED. VOID never enters a ratio or an equivalence reading. Nothing is licensed; no cross-box number is divided into these.
`versions.txt`
```
e4b 0.47.0 @3cc7f106da3f58d72366027cc434f2c41120b5f7 (GitHub main)
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
e4b(t212) 0.47.0 @3cc7f106da3f58d72366027cc434f2c41120b5f7
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
 "run_id": "tc1-5090-79",
 "instance_id": "54255834",
 "gpu": "NVIDIA GeForce RTX 5090",
 "driver": "595.84",
 "cpu": "AMD EPYC 7B13 64-Core Processor",
 "nproc": 256,
 "mem_total_kb": "2101193408",
 "cgroup_memory_max": "730283900928",
 "disk_root": "overlay         320G   77M  320G   1% /",
 "hostname": "2dce83b1792c",
 "registered_gpu_class": "5090",
 "prereg": "tc1/TC1-PREREG.md",
 "local_box": null,
 "local_snapshot": null
}
```

### Qwen3-30B-A3B (amendment 35: Triton launches prebound off vs on under triton 3.7.1, e4b + grouped-nf4-gemm, venv-unsloth) (`qwen3prebind37`, registered n_layers 48, attention census 192)
- model `Qwen/Qwen3-30B-A3B` @ `ad44e777bcd1`; tokens sha `bfc742f67e37`; N=60; fixture template alpaca seq 2048 micro-batch 2 × accum 4 lr 0.0002 r 16 α 16 optimizer adamw_8bit(lr=0.0002, weight_decay=0.001) schedule=linear warmup_steps=5 autocast False; e4b trainable 642514944; box_class RTX 5090 gpu NVIDIA GeForce RTX 5090
| framework | arm | status | validity | **VERDICT** | matched | init / dtype | step-0 class | N | s/step med(11+) | tok/s | peak GB | J/step | train first→last | held-out 0→final | quality Δ | engagement | trainable | regime | reason / why |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| e4b | fused_attn4_shipped_pb0 | **OK** | VALID | **VALID** | native | native / bfloat16,float32 | — | 60 | 2.876 | 547.5 | 24.673 | 753.1 | 2.0554→0.8176 | 1.9478→0.7598 | 0.0022 | patched 48 / kcalls 768 | 642514944 | 4-bit experts (e4b NF4) + NF4 attention |  |
| e4b | fused_attn4_shipped_pb1 | **OK** | VALID | **VALID** | native | native / bfloat16,float32 | — | 60 | 2.839 | 557.9 | 24.673 | 745.0 | 2.0554→0.8142 | 1.9478→0.7560 | -0.0015 | patched 48 / kcalls 768 | 642514944 | 4-bit experts (e4b NF4) + NF4 attention |  |
| e4b | fused_attn4_m_pb0 | **OK** | VALID | **VALID** | yes | matched:3407 (complete 12480/12480) / float32 | SAME-BYTES-CLASS (0.0000) | 60 | 3.444 | 460.4 | 27.500 | 922.7 | 2.0554→0.8181 | 1.9478→0.7576 | 0.0000 | patched 48 / kcalls 768 | 642514944 | 4-bit experts (e4b NF4) + NF4 attention |  |
| e4b | fused_attn4_m_pb1 | **OK** | VALID | **VALID** | yes | matched:3407 (complete 12480/12480) / float32 | SAME-BYTES-CLASS (0.0000) | 60 | 3.408 | 464.1 | 27.507 | 931.0 | 2.0554→0.8156 | 1.9478→0.7580 | 0.0005 | patched 48 / kcalls 768 | 642514944 | 4-bit experts (e4b NF4) + NF4 attention |  |
| e4b | fused_attn4_m_pb1_d2 | **OK** | VALID | **VALID** | yes | matched:3407 (complete 12480/12480) / float32 | SAME-BYTES-CLASS (0.0000) | 60 | 3.360 | 465.3 | 27.481 | 930.6 | 2.0554→0.8201 | 1.9478→0.7561 | -0.0014 | patched 48 / kcalls 768 | 642514944 | 4-bit experts (e4b NF4) + NF4 attention |  |
| e4b | fused_attn4_m_pb0_d2 | **OK** | VALID | **VALID** | yes | matched:3407 (complete 12480/12480) / float32 | SAME-BYTES-CLASS (0.0000) | 60 | 3.418 | 463.7 | 27.496 | 932.3 | 2.0554→0.8183 | 1.9478→0.7548 | -0.0028 | patched 48 / kcalls 768 | 642514944 | 4-bit experts (e4b NF4) + NF4 attention |  |
| e4b | fused_attn4_shipped_pb1_d2 | **OK** | VALID | **VALID** | native | native / bfloat16,float32 | — | 60 | 2.913 | 544.1 | 24.673 | 746.1 | 2.0554→0.8178 | 1.9478→0.7551 | -0.0025 | patched 48 / kcalls 768 | 642514944 | 4-bit experts (e4b NF4) + NF4 attention |  |
| e4b | fused_attn4_shipped_pb0_d2 | **OK** | VALID | **VALID** | native | native / bfloat16,float32 | — | 60 | 2.902 | 541.4 | 24.673 | 751.7 | 2.0554→0.8192 | 1.9478→0.7564 | -0.0012 | patched 48 / kcalls 768 | 642514944 | 4-bit experts (e4b NF4) + NF4 attention |  |
- prologue `e4b/fused_attn4_shipped_pb0` **101.2 s** before step 1 (36% of the arm): c1_before 55.2, c1_after 27.5, load_weights 24.0, trainable_sha 6.0; unattributed 2.071; budget 1260.0
- prologue `e4b/fused_attn4_shipped_pb1` **99.3 s** before step 1 (36% of the arm): c1_before 54.5, c1_after 28.3, load_weights 23.0, trainable_sha 6.1; unattributed 2.005; budget 1260.0
- prologue `e4b/fused_attn4_m_pb0` **98.1 s** before step 1 (32% of the arm): c1_before 54.7, c1_after 28.1, load_weights 23.7, attn4 5.6; unattributed 2.108; budget 1260.0
- prologue `e4b/fused_attn4_m_pb1` **98.1 s** before step 1 (32% of the arm): c1_before 55.3, c1_after 27.5, load_weights 23.5, attn4 5.4; unattributed 1.99; budget 1260.0
- prologue `e4b/fused_attn4_m_pb1_d2` **98.1 s** before step 1 (32% of the arm): c1_before 55.1, c1_after 27.7, load_weights 23.4, attn4 5.6; unattributed 2.027; budget 1260.0
- prologue `e4b/fused_attn4_m_pb0_d2` **98.2 s** before step 1 (32% of the arm): c1_before 55.0, c1_after 27.1, load_weights 23.6, attn4 5.5; unattributed 2.131; budget 1260.0
- prologue `e4b/fused_attn4_shipped_pb1_d2` **98.3 s** before step 1 (35% of the arm): c1_before 53.6, c1_after 27.4, load_weights 22.7, trainable_sha 6.0; unattributed 1.976; budget 1260.0
- prologue `e4b/fused_attn4_shipped_pb0_d2` **100.4 s** before step 1 (36% of the arm): c1_before 54.6, c1_after 28.6, load_weights 23.7, trainable_sha 5.9; unattributed 2.165; budget 1260.0
- draws (R1): `e4b/fused_attn4_shipped_pb0` STABLE (2.876/2.902 s, |Δ|/mean 0.9% vs 5%); `e4b/fused_attn4_shipped_pb1` STABLE (2.839/2.913 s, |Δ|/mean 2.6% vs 5%); `e4b/fused_attn4_m_pb0` STABLE (3.444/3.418 s, |Δ|/mean 0.8% vs 5%); `e4b/fused_attn4_m_pb1` STABLE (3.408/3.360 s, |Δ|/mean 1.4% vs 5%)
- e4b internal parity: NO-ARM
- **NO MATCHED POSITION QUOTED (unsloth)** — not quoted: e4b STABLE / unsloth no receipt
- **NO MATCHED POSITION QUOTED (hf)** — not quoted: e4b STABLE / hf no receipt
- **NO MATCHED POSITION QUOTED (axolotl)** — not quoted: e4b STABLE / axolotl no receipt
- equivalence vs `e4b/fused_attn4_m` (I/H; EQUIVALENT iff both ≤ band = max(0.005, 3 × fused-vs-reference |Δ|) = none (no reference arm: COMPARABLE is the ceiling); COMPARABLE ≤ 0.05; draw-noise floor 0.0034): `e4b/fused_attn4_m_pb1` **COMPARABLE** (median step |Δ| 0.0010, |Δ held-out at N| 0.0005, step-0 0.0000 SAME-BYTES-CLASS, |Δ loss at step 2| 0.0058, paired rows mean +0.0005 ± 0.0021 SE over 8, favouring arm 5 / anchor 3) — no reference arm ran: the EQUIVALENT band cannot be set, COMPARABLE is the ceiling; `e4b/fused_attn4_m_pb1_d2` **COMPARABLE** (median step |Δ| 0.0014, |Δ held-out at N| 0.0014, step-0 0.0000 SAME-BYTES-CLASS, |Δ loss at step 2| 0.0133, paired rows mean -0.0014 ± 0.0015 SE over 8, favouring arm 6 / anchor 2) — no reference arm ran: the EQUIVALENT band cannot be set, COMPARABLE is the ceiling; `e4b/fused_attn4_m_pb0_d2` **COMPARABLE** (median step |Δ| 0.0013, |Δ held-out at N| 0.0028, step-0 0.0000 SAME-BYTES-CLASS, |Δ loss at step 2| 0.0034, paired rows mean -0.0028 ± 0.0014 SE over 8, favouring arm 6 / anchor 2) — no reference arm ran: the EQUIVALENT band cannot be set, COMPARABLE is the ceiling
- matched_init_sha (B, name-free, canonical slot order): anchor `f7832488926eda91`; `e4b/fused_attn4_m_pb0` same; `e4b/fused_attn4_m_pb1` same; `e4b/fused_attn4_m_pb1_d2` same; `e4b/fused_attn4_m_pb0_d2` same
- frozen base (R5), anchor `e4b/fused_attn4_m`: gate_up nf4/64 sha 8c3b3fd78e89 control detects, down nf4/64 sha 59569d810a1a control detects, q_proj nf4/64+dq sha 5c5e402ce51d control detects
- frozen base `e4b/fused_attn4_shipped_pb0`: gate_up **SAME-BYTES** (nf4/64) control detects, down **SAME-BYTES** (nf4/64) control detects, q_proj **SAME-BYTES** (nf4/64+dq) control detects
- frozen base `e4b/fused_attn4_shipped_pb1`: gate_up **SAME-BYTES** (nf4/64) control detects, down **SAME-BYTES** (nf4/64) control detects, q_proj **SAME-BYTES** (nf4/64+dq) control detects
- frozen base `e4b/fused_attn4_m_pb1`: gate_up **SAME-BYTES** (nf4/64) control detects, down **SAME-BYTES** (nf4/64) control detects, q_proj **SAME-BYTES** (nf4/64+dq) control detects
- frozen base `e4b/fused_attn4_m_pb1_d2`: gate_up **SAME-BYTES** (nf4/64) control detects, down **SAME-BYTES** (nf4/64) control detects, q_proj **SAME-BYTES** (nf4/64+dq) control detects
- frozen base `e4b/fused_attn4_m_pb0_d2`: gate_up **SAME-BYTES** (nf4/64) control detects, down **SAME-BYTES** (nf4/64) control detects, q_proj **SAME-BYTES** (nf4/64+dq) control detects
- frozen base `e4b/fused_attn4_shipped_pb1_d2`: gate_up **SAME-BYTES** (nf4/64) control detects, down **SAME-BYTES** (nf4/64) control detects, q_proj **SAME-BYTES** (nf4/64+dq) control detects
- frozen base `e4b/fused_attn4_shipped_pb0_d2`: gate_up **SAME-BYTES** (nf4/64) control detects, down **SAME-BYTES** (nf4/64) control detects, q_proj **SAME-BYTES** (nf4/64+dq) control detects

## Verdicts
| family | arm | VERDICT | validity | quality | note |
|---|---|---|---|---|---|
| qwen3prebind37 | e4b/fused_attn4_shipped_pb0 | **VALID** | VALID | 0.0022 |  |
| qwen3prebind37 | e4b/fused_attn4_shipped_pb1 | **VALID** | VALID | -0.0015 |  |
| qwen3prebind37 | e4b/fused_attn4_m_pb0 | **VALID** | VALID | 0.0000 |  |
| qwen3prebind37 | e4b/fused_attn4_m_pb1 | **VALID** | VALID | 0.0005 |  |
| qwen3prebind37 | e4b/fused_attn4_m_pb1_d2 | **VALID** | VALID | -0.0014 |  |
| qwen3prebind37 | e4b/fused_attn4_m_pb0_d2 | **VALID** | VALID | -0.0028 |  |
| qwen3prebind37 | e4b/fused_attn4_shipped_pb1_d2 | **VALID** | VALID | -0.0025 |  |
| qwen3prebind37 | e4b/fused_attn4_shipped_pb0_d2 | **VALID** | VALID | -0.0012 |  |

## Predictions P66 / P67 / P68 (TC1-PREREG amendment 35: prebound Triton launches off vs on under triton 3.7.1, venv-unsloth, two stable draws a side; scored mechanically)
| prediction | family | verdict | evidence |
|---|---|---|---|
| P66 | qwen3prebind37 | **HELD** | shipped: pb1 / pb0 0.996 [0.978, 1.013 over 4 cross-draw ratios] vs [0.97, 1.0]; s/step pb0 2.876 / 2.902 (within 0.9%), pb1 2.839 / 2.913 (within 2.6%); pb1 launch counts (process) {"e4b": {"prebound": 216335, "triton": 753}, "gnf4": {"prebound": 72162, "triton": 30}}; triton 3.7.1 |
| P67 | qwen3prebind37 | **HELD** | matched: pb1 / pb0 0.986 [0.975, 0.997 over 4 cross-draw ratios] vs [0.97, 1.0]; s/step pb0 3.444 / 3.418 (within 0.8%), pb1 3.408 / 3.360 (within 1.4%); pb1 launch counts (process) {"e4b": {"prebound": 216335, "triton": 753}, "gnf4": {"prebound": 72162, "triton": 30}}; triton 3.7.1 |
| P68 | qwen3prebind37 | **HELD** | shipped: mean held-out pb1 - pb0 -0.0026 (|.| <= 0.005); held-out at N pb0 [0.7598, 0.7564] pb1 [0.756, 0.7551]; matched: mean held-out pb1 - pb0 +0.0009 (|.| <= 0.005); held-out at N pb0 [0.7576, 0.7548] pb1 [0.758, 0.7561] |

## Load gate (TC1-PREREG amendment 33): 5 draw(s) voided for host load and run again
Each line: the arm, the attempt, the median host load1 over that attempt's run, the gate. A VOID attempt's files are in loadvoid/; the last attempt of each arm stands whatever its load.

- `qwen3prebind37/e4b/fused_attn4_shipped_pb0 attempt 0 load1_median 7.89 gate 6.0 status ok over 1`
- `qwen3prebind37/e4b/fused_attn4_shipped_pb0 attempt 0 VOID (host load1 median 7.89 > 6.0): re-run 1 of 2`
- `qwen3prebind37/e4b/fused_attn4_shipped_pb0 attempt 1 load1_median 18.9 gate 6.0 status ok over 1`
- `qwen3prebind37/e4b/fused_attn4_shipped_pb0 attempt 1 VOID (host load1 median 18.9 > 6.0): re-run 2 of 2`
- `qwen3prebind37/e4b/fused_attn4_shipped_pb0 attempt 2 load1_median 8.17 gate 6.0 status ok over 1`
- `qwen3prebind37/e4b/fused_attn4_shipped_pb1 attempt 0 load1_median 9.37 gate 6.0 status ok over 1`
- `qwen3prebind37/e4b/fused_attn4_shipped_pb1 attempt 0 VOID (host load1 median 9.37 > 6.0): re-run 1 of 2`
- `qwen3prebind37/e4b/fused_attn4_shipped_pb1 attempt 1 load1_median 6.84 gate 6.0 status ok over 1`
- `qwen3prebind37/e4b/fused_attn4_shipped_pb1 attempt 1 VOID (host load1 median 6.84 > 6.0): re-run 2 of 2`
- `qwen3prebind37/e4b/fused_attn4_shipped_pb1 attempt 2 load1_median 8.24 gate 6.0 status ok over 1`
- `qwen3prebind37/e4b/fused_attn4_m_pb0 attempt 0 load1_median 3.86 gate 6.0 status ok over 0`
- `qwen3prebind37/e4b/fused_attn4_m_pb1 attempt 0 load1_median 7.14 gate 6.0 status ok over 1`
- `qwen3prebind37/e4b/fused_attn4_m_pb1 attempt 0 VOID (host load1 median 7.14 > 6.0): re-run 1 of 2`
- `qwen3prebind37/e4b/fused_attn4_m_pb1 attempt 1 load1_median 4.48 gate 6.0 status ok over 0`
- `qwen3prebind37/e4b/fused_attn4_m_pb1_d2 attempt 0 load1_median 2.99 gate 6.0 status ok over 0`
- `qwen3prebind37/e4b/fused_attn4_m_pb0_d2 attempt 0 load1_median 3.29 gate 6.0 status ok over 0`
- `qwen3prebind37/e4b/fused_attn4_shipped_pb1_d2 attempt 0 load1_median 4.95 gate 6.0 status ok over 0`
- `qwen3prebind37/e4b/fused_attn4_shipped_pb0_d2 attempt 0 load1_median 4.04 gate 6.0 status ok over 0`
