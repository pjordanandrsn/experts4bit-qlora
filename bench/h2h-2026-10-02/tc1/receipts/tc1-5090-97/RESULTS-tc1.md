# TC1 — Qwen3-30B-A3B on one RTX 5090: e4b vs Unsloth vs HF+PEFT vs axolotl at matched work, init and adapter precision (tc1)
Rule (tc1/TC1-PREREG.md): status per attempt in the vocabulary OK / REFUSED / OOM / INSTALL_FAILED / LOAD_FAULT / HARNESS_ERROR / ALARM / NOT_RUN; VALID/VOID per tp4's predicates plus the matched-set predicates (R3: matched init complete, fp32 adapters, step-0 held-out within 0.005 of e4b/reference_attn4_m); VERDICT exactly one of VALID / VOID / QUALITY_FAIL / OOM / UNSUPPORTED / HARNESS_ERROR / ALARM / NOT_RUN (QUALITY_FAIL = held-out |Δ| at N > 0.05 vs e4b/fused_attn4_m); positions = s/step ratios other/e4b from the medians over both draws (stability |d1-d2|/mean ≤ 5 % e4b / 10 % others, else UNSTABLE), quoted only when both arms are VALID; equivalence vs e4b/reference_attn4_m EQUIVALENT ≤ 0.02 / COMPARABLE ≤ 0.05 / else DIVERGENT; frozen base SAME-BYTES / DIFFERENT / N-A per slot; predictions P1–P10 (+ P1b) scored HELD / FALSIFIED / UNTESTED. VOID never enters a ratio or an equivalence reading. Nothing is licensed; no cross-box number is divided into these.
`versions.txt`
```
e4b 0.48.0 @298224fd2f63b067fcb901bc5a1650bfb6a627f3 (GitHub main)
gnf4 0.41.0 @f127981fded98c2fa24acd6b4ae1eb0929ddb192 (GitHub main)
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
e4b(t212) 0.48.0 @298224fd2f63b067fcb901bc5a1650bfb6a627f3
gnf4(t212) 0.41.0 @f127981fded98c2fa24acd6b4ae1eb0929ddb192
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
 "run_id": "tc1-5090-97",
 "instance_id": "54366046",
 "gpu": "NVIDIA GeForce RTX 5090",
 "driver": "590.48.01",
 "cpu": "AMD Ryzen Threadripper PRO 3955WX 16-Cores",
 "nproc": 32,
 "mem_total_kb": "263771780",
 "cgroup_memory_max": "183350853632",
 "disk_root": "overlay         320G   48M  320G   1% /",
 "hostname": "a542e679703e",
 "registered_gpu_class": "5090",
 "prereg": "tc1/TC1-PREREG.md",
 "local_box": null,
 "local_snapshot": null
}
```

### Qwen3-30B-A3B (amendment 44: e4b's chunked LM loss off vs auto at the field recipe, venv-unsloth) (`qwen3chunkauto`, registered n_layers 48, attention census 192)
- model `Qwen/Qwen3-30B-A3B` @ `ad44e777bcd1`; tokens sha `bfc742f67e37`; N=60; fixture template alpaca seq 2048 micro-batch 2 × accum 4 lr 0.0002 r 16 α 16 optimizer adamw_8bit(lr=0.0002, weight_decay=0.001) schedule=linear warmup_steps=5 autocast False; e4b trainable 642514944; box_class RTX 5090 gpu NVIDIA GeForce RTX 5090
| framework | arm | status | validity | **VERDICT** | matched | init / dtype | step-0 class | N | s/step med(11+) | tok/s | peak GB | J/step | train first→last | held-out 0→final | quality Δ | engagement | trainable | regime | reason / why |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| e4b | fused_attn4_shipped_ca0 | **OK** | VALID | **VALID** | native | native / bfloat16,float32 | — | 60 | 2.483 | 607.3 | 24.673 | 748.3 | 2.0554→0.8176 | 1.9478→0.7564 | -0.0000 | patched 48 / kcalls 768 | 642514944 | 4-bit experts (e4b NF4) + NF4 attention |  |
| e4b | fused_attn4_shipped_ca1 | **OK** | VALID | **VALID** | native | native / bfloat16,float32 | — | 60 | 2.429 | 660.0 | 24.673 | 726.6 | 2.0554→0.8188 | 1.9478→0.7569 | 0.0004 | patched 48 / kcalls 768 | 642514944 | 4-bit experts (e4b NF4) + NF4 attention |  |
| e4b | fused_attn4_m_ca0 | **OK** | VALID | **VALID** | yes | matched:3407 (complete 12480/12480) / float32 | SAME-BYTES-CLASS (0.0000) | 60 | 3.135 | 503.4 | 27.496 | 919.3 | 2.0554→0.8166 | 1.9478→0.7565 | 0.0000 | patched 48 / kcalls 768 | 642514944 | 4-bit experts (e4b NF4) + NF4 attention |  |
| e4b | fused_attn4_m_ca1 | **OK** | VALID | **VALID** | yes | matched:3407 (complete 12480/12480) / float32 | SAME-BYTES-CLASS (0.0000) | 60 | 3.139 | 507.5 | 27.493 | 908.5 | 2.0554→0.8157 | 1.9478→0.7560 | -0.0005 | patched 48 / kcalls 768 | 642514944 | 4-bit experts (e4b NF4) + NF4 attention |  |
| e4b | fused_attn4_m_ca1_d2 | **OK** | VALID | **VALID** | yes | matched:3407 (complete 12480/12480) / float32 | SAME-BYTES-CLASS (0.0000) | 60 | 3.131 | 507.1 | 27.495 | 927.9 | 2.0554→0.8190 | 1.9478→0.7584 | 0.0020 | patched 48 / kcalls 768 | 642514944 | 4-bit experts (e4b NF4) + NF4 attention |  |
| e4b | fused_attn4_m_ca0_d2 | **OK** | VALID | **VALID** | yes | matched:3407 (complete 12480/12480) / float32 | SAME-BYTES-CLASS (0.0000) | 60 | 3.142 | 504.5 | 27.495 | 929.6 | 2.0554→0.8222 | 1.9478→0.7577 | 0.0012 | patched 48 / kcalls 768 | 642514944 | 4-bit experts (e4b NF4) + NF4 attention |  |
| e4b | fused_attn4_shipped_ca1_d2 | **OK** | VALID | **VALID** | native | native / bfloat16,float32 | — | 60 | 2.435 | 656.9 | 24.673 | 711.4 | 2.0554→0.8175 | 1.9478→0.7582 | 0.0018 | patched 48 / kcalls 768 | 642514944 | 4-bit experts (e4b NF4) + NF4 attention |  |
| e4b | fused_attn4_shipped_ca0_d2 | **OK** | VALID | **VALID** | native | native / bfloat16,float32 | — | 60 | 2.421 | 658.8 | 24.673 | 714.5 | 2.0554→0.8180 | 1.9478→0.7590 | 0.0025 | patched 48 / kcalls 768 | 642514944 | 4-bit experts (e4b NF4) + NF4 attention |  |
- prologue `e4b/fused_attn4_shipped_ca0` **98.6 s** before step 1 (38% of the arm): c1_before 53.7, c1_after 27.3, load_weights 18.6, eval0 11.8; unattributed 1.689; budget 1260.0
- prologue `e4b/fused_attn4_shipped_ca1` **88.3 s** before step 1 (37% of the arm): c1_before 53.8, c1_after 27.1, load_weights 18.6, attn4 5.5; unattributed 1.652; budget 1260.0
- prologue `e4b/fused_attn4_m_ca0` **89.6 s** before step 1 (32% of the arm): c1_before 53.6, c1_after 27.0, load_weights 18.5, attn4 5.5; unattributed 1.671; budget 1260.0
- prologue `e4b/fused_attn4_m_ca1` **89.0 s** before step 1 (32% of the arm): c1_before 53.0, c1_after 27.2, load_weights 18.4, attn4 5.5; unattributed 1.704; budget 1260.0
- prologue `e4b/fused_attn4_m_ca1_d2` **89.5 s** before step 1 (32% of the arm): c1_before 53.4, c1_after 28.6, load_weights 18.5, attn4 5.5; unattributed 1.676; budget 1260.0
- prologue `e4b/fused_attn4_m_ca0_d2` **95.1 s** before step 1 (33% of the arm): c1_before 55.2, c1_after 26.5, load_weights 21.6, attn4 5.8; unattributed 1.851; budget 1260.0
- prologue `e4b/fused_attn4_shipped_ca1_d2` **88.5 s** before step 1 (37% of the arm): c1_before 53.9, c1_after 26.9, load_weights 18.6, attn4 5.5; unattributed 1.689; budget 1260.0
- prologue `e4b/fused_attn4_shipped_ca0_d2` **88.2 s** before step 1 (37% of the arm): c1_before 53.7, c1_after 26.5, load_weights 18.4, attn4 5.6; unattributed 1.681; budget 1260.0
- draws (R1): `e4b/fused_attn4_shipped_ca0` STABLE (2.483/2.421 s, |Δ|/mean 2.5% vs 5%); `e4b/fused_attn4_shipped_ca1` STABLE (2.429/2.435 s, |Δ|/mean 0.3% vs 5%); `e4b/fused_attn4_m_ca0` STABLE (3.135/3.142 s, |Δ|/mean 0.2% vs 5%); `e4b/fused_attn4_m_ca1` STABLE (3.139/3.131 s, |Δ|/mean 0.3% vs 5%)
- e4b internal parity: NO-ARM
- **NO MATCHED POSITION QUOTED (unsloth)** — not quoted: e4b STABLE / unsloth no receipt
- **NO MATCHED POSITION QUOTED (hf)** — not quoted: e4b STABLE / hf no receipt
- **NO MATCHED POSITION QUOTED (axolotl)** — not quoted: e4b STABLE / axolotl no receipt
- equivalence vs `e4b/fused_attn4_m` (I/H; EQUIVALENT iff both ≤ band = max(0.005, 3 × fused-vs-reference |Δ|) = none (no reference arm: COMPARABLE is the ceiling); COMPARABLE ≤ 0.05; draw-noise floor 0.0025): `e4b/fused_attn4_m_ca1` **COMPARABLE** (median step |Δ| 0.0012, |Δ held-out at N| 0.0005, step-0 0.0000 SAME-BYTES-CLASS, |Δ loss at step 2| 0.0147, paired rows mean -0.0005 ± 0.0010 SE over 8, favouring arm 5 / anchor 3) — no reference arm ran: the EQUIVALENT band cannot be set, COMPARABLE is the ceiling; `e4b/fused_attn4_m_ca1_d2` **COMPARABLE** (median step |Δ| 0.0010, |Δ held-out at N| 0.0020, step-0 0.0000 SAME-BYTES-CLASS, |Δ loss at step 2| 0.0009, paired rows mean +0.0020 ± 0.0018 SE over 8, favouring arm 4 / anchor 4) — no reference arm ran: the EQUIVALENT band cannot be set, COMPARABLE is the ceiling; `e4b/fused_attn4_m_ca0_d2` **COMPARABLE** (median step |Δ| 0.0013, |Δ held-out at N| 0.0012, step-0 0.0000 SAME-BYTES-CLASS, |Δ loss at step 2| 0.0017, paired rows mean +0.0012 ± 0.0009 SE over 8, favouring arm 2 / anchor 6) — no reference arm ran: the EQUIVALENT band cannot be set, COMPARABLE is the ceiling
- matched_init_sha (B, name-free, canonical slot order): anchor `f7832488926eda91`; `e4b/fused_attn4_m_ca0` same; `e4b/fused_attn4_m_ca1` same; `e4b/fused_attn4_m_ca1_d2` same; `e4b/fused_attn4_m_ca0_d2` same
- frozen base (R5), anchor `e4b/fused_attn4_m`: gate_up nf4/64 sha 8c3b3fd78e89 control detects, down nf4/64 sha 59569d810a1a control detects, q_proj nf4/64+dq sha 5c5e402ce51d control detects
- frozen base `e4b/fused_attn4_shipped_ca0`: gate_up **SAME-BYTES** (nf4/64) control detects, down **SAME-BYTES** (nf4/64) control detects, q_proj **SAME-BYTES** (nf4/64+dq) control detects
- frozen base `e4b/fused_attn4_shipped_ca1`: gate_up **SAME-BYTES** (nf4/64) control detects, down **SAME-BYTES** (nf4/64) control detects, q_proj **SAME-BYTES** (nf4/64+dq) control detects
- frozen base `e4b/fused_attn4_m_ca1`: gate_up **SAME-BYTES** (nf4/64) control detects, down **SAME-BYTES** (nf4/64) control detects, q_proj **SAME-BYTES** (nf4/64+dq) control detects
- frozen base `e4b/fused_attn4_m_ca1_d2`: gate_up **SAME-BYTES** (nf4/64) control detects, down **SAME-BYTES** (nf4/64) control detects, q_proj **SAME-BYTES** (nf4/64+dq) control detects
- frozen base `e4b/fused_attn4_m_ca0_d2`: gate_up **SAME-BYTES** (nf4/64) control detects, down **SAME-BYTES** (nf4/64) control detects, q_proj **SAME-BYTES** (nf4/64+dq) control detects
- frozen base `e4b/fused_attn4_shipped_ca1_d2`: gate_up **SAME-BYTES** (nf4/64) control detects, down **SAME-BYTES** (nf4/64) control detects, q_proj **SAME-BYTES** (nf4/64+dq) control detects
- frozen base `e4b/fused_attn4_shipped_ca0_d2`: gate_up **SAME-BYTES** (nf4/64) control detects, down **SAME-BYTES** (nf4/64) control detects, q_proj **SAME-BYTES** (nf4/64+dq) control detects

## Verdicts
| family | arm | VERDICT | validity | quality | note |
|---|---|---|---|---|---|
| qwen3chunkauto | e4b/fused_attn4_shipped_ca0 | **VALID** | VALID | -0.0000 |  |
| qwen3chunkauto | e4b/fused_attn4_shipped_ca1 | **VALID** | VALID | 0.0004 |  |
| qwen3chunkauto | e4b/fused_attn4_m_ca0 | **VALID** | VALID | 0.0000 |  |
| qwen3chunkauto | e4b/fused_attn4_m_ca1 | **VALID** | VALID | -0.0005 |  |
| qwen3chunkauto | e4b/fused_attn4_m_ca1_d2 | **VALID** | VALID | 0.0020 |  |
| qwen3chunkauto | e4b/fused_attn4_m_ca0_d2 | **VALID** | VALID | 0.0012 |  |
| qwen3chunkauto | e4b/fused_attn4_shipped_ca1_d2 | **VALID** | VALID | 0.0018 |  |
| qwen3chunkauto | e4b/fused_attn4_shipped_ca0_d2 | **VALID** | VALID | 0.0025 |  |

## Predictions P99 / P100 / P101 / P102 / P103 (TC1-PREREG amendment 44: e4b's chunked LM loss off vs auto at the field recipe; scored mechanically)
| prediction | family | verdict | evidence |
|---|---|---|---|
| P99 | qwen3chunkauto | **HELD** | want chunked 0 / small 240 on every ca1 arm: `fused_attn4_shipped_ca1` chunked 0 / small 240; `fused_attn4_m_ca1` chunked 0 / small 240; `fused_attn4_m_ca1_d2` chunked 0 / small 240; `fused_attn4_shipped_ca1_d2` chunked 0 / small 240 |
| P102 | qwen3chunkauto | **HELD** | matched: peak ca0 27.495 / ca1 27.494 GB, drop 0.002 vs [-0.05, 99.0] |
| P101 | qwen3chunkauto | **HELD** | matched: ca1 / ca0 0.999 [0.997, 1.001 over 4 cross-draw ratios] vs [0.0, 1.02]; s/step ca0 3.135 / 3.142 (within 0.2%), ca1 3.139 / 3.131 (within 0.3%); peak ca0 27.50 / ca1 27.49 GB |
| P100 | qwen3chunkauto | **HELD** | shipped: ca1 / ca0 0.992 [0.978, 1.006 over 4 cross-draw ratios] vs [0.0, 1.02]; s/step ca0 2.483 / 2.421 (within 2.5%), ca1 2.429 / 2.435 (within 0.3%); peak ca0 24.67 / ca1 24.67 GB |
| P103 | qwen3chunkauto | **HELD** | matched: mean held-out ca1 - ca0 +0.0001 (|.| <= 0.005); held-out at N ca0 [0.7565, 0.7577] ca1 [0.756, 0.7584]; shipped: mean held-out ca1 - ca0 -0.0001 (|.| <= 0.005); held-out at N ca0 [0.7564, 0.759] ca1 [0.7569, 0.7582] |

## Load gate (TC1-PREREG amendment 33): 0 draw(s) voided for host load and run again
Each line: the arm, the attempt, the median host load1 over that attempt's run, the gate. A VOID attempt's files are in loadvoid/; the last attempt of each arm stands whatever its load.

- `qwen3chunkauto/e4b/fused_attn4_shipped_ca0 attempt 0 load1_median 1.24 gate 6.0 status ok over 0`
- `qwen3chunkauto/e4b/fused_attn4_shipped_ca1 attempt 0 load1_median 1.35 gate 6.0 status ok over 0`
- `qwen3chunkauto/e4b/fused_attn4_m_ca0 attempt 0 load1_median 1.31 gate 6.0 status ok over 0`
- `qwen3chunkauto/e4b/fused_attn4_m_ca1 attempt 0 load1_median 1.46 gate 6.0 status ok over 0`
- `qwen3chunkauto/e4b/fused_attn4_m_ca1_d2 attempt 0 load1_median 1.3 gate 6.0 status ok over 0`
- `qwen3chunkauto/e4b/fused_attn4_m_ca0_d2 attempt 0 load1_median 1.85 gate 6.0 status ok over 0`
- `qwen3chunkauto/e4b/fused_attn4_shipped_ca1_d2 attempt 0 load1_median 1.66 gate 6.0 status ok over 0`
- `qwen3chunkauto/e4b/fused_attn4_shipped_ca0_d2 attempt 0 load1_median 1.13 gate 6.0 status ok over 0`
