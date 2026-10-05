# TC1 — Qwen3-30B-A3B on one RTX 5090: e4b vs Unsloth vs HF+PEFT vs axolotl at matched work, init and adapter precision (tc1)
Rule (tc1/TC1-PREREG.md): status per attempt in the vocabulary OK / REFUSED / OOM / INSTALL_FAILED / LOAD_FAULT / HARNESS_ERROR / ALARM / NOT_RUN; VALID/VOID per tp4's predicates plus the matched-set predicates (R3: matched init complete, fp32 adapters, step-0 held-out within 0.005 of e4b/reference_attn4_m); VERDICT exactly one of VALID / VOID / QUALITY_FAIL / OOM / UNSUPPORTED / HARNESS_ERROR / ALARM / NOT_RUN (QUALITY_FAIL = held-out |Δ| at N > 0.05 vs e4b/fused_attn4_m); positions = s/step ratios other/e4b from the medians over both draws (stability |d1-d2|/mean ≤ 5 % e4b / 10 % others, else UNSTABLE), quoted only when both arms are VALID; equivalence vs e4b/reference_attn4_m EQUIVALENT ≤ 0.02 / COMPARABLE ≤ 0.05 / else DIVERGENT; frozen base SAME-BYTES / DIFFERENT / N-A per slot; predictions P1–P10 (+ P1b) scored HELD / FALSIFIED / UNTESTED. VOID never enters a ratio or an equivalence reading. Nothing is licensed; no cross-box number is divided into these.
`versions.txt`
```
e4b 0.47.0 @f7f736e5ae310fd0f3ac93b97d6d08e644747193 (GitHub main)
gnf4 0.39.0 @f0c1ece95902a7b7d004b0f2af331dcd9f06e7c7 (GitHub main)
torch(e4b/hf) 2.8.0+cu128
triton(e4b/hf) 3.4.0
transformers(e4b/hf) 5.18.0
bitsandbytes(e4b/hf) 0.50.2
peft(hf) 0.21.2
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
 "run_id": "tc1-5090-75",
 "instance_id": "54238515",
 "gpu": "NVIDIA GeForce RTX 5090",
 "driver": "595.84",
 "cpu": "AMD EPYC 7B13 64-Core Processor",
 "nproc": 256,
 "mem_total_kb": "2101193408",
 "cgroup_memory_max": "730283900928",
 "disk_root": "overlay         320G   77M  320G   1% /",
 "hostname": "c9e288a640e1",
 "registered_gpu_class": "5090",
 "prereg": "tc1/TC1-PREREG.md",
 "local_box": null,
 "local_snapshot": null
}
```

### Qwen3-30B-A3B (amendment 28: the expert absmax fp32 vs double-quantized, matched arm, resident) (`qwen3dqab`, registered n_layers 48, attention census 192)
- model `Qwen/Qwen3-30B-A3B` @ `ad44e777bcd1`; tokens sha `bfc742f67e37`; N=60; fixture template alpaca seq 2048 micro-batch 2 × accum 4 lr 0.0002 r 16 α 16 optimizer adamw_8bit(lr=0.0002, weight_decay=0.001) schedule=linear warmup_steps=5 autocast False; e4b trainable 642514944; box_class RTX 5090 gpu NVIDIA GeForce RTX 5090
| framework | arm | status | validity | **VERDICT** | matched | init / dtype | step-0 class | N | s/step med(11+) | tok/s | peak GB | J/step | train first→last | held-out 0→final | quality Δ | engagement | trainable | regime | reason / why |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| e4b | fused_attn4_m_dq0 | **OK** | VALID | **VALID** | yes | matched:3407 (complete 12480/12480) / float32 | SAME-BYTES-CLASS (0.0000) | 60 | 3.751 | 405.6 | 27.441 | 1141.8 | 2.0614→0.8201 | 1.9441→0.7597 | 0.0000 | patched 48 / kcalls 768 | 642514944 | 4-bit experts (e4b NF4) + NF4 attention |  |
| e4b | fused_attn4_m_dq1 | **OK** | VALID | **VALID** | yes | matched:3407 (complete 12480/12480) / float32 | NEAR (0.0190) | 60 | 3.777 | 419.7 | 26.109 | 1130.6 | 2.0506→0.8215 | 1.9251→0.7567 | -0.0030 | patched 48 / kcalls 768 | 642514944 | 4-bit experts (e4b NF4) + NF4 attention |  |
| e4b | fused_attn4_m_dq1_d2 | **OK** | VALID | **VALID** | yes | matched:3407 (complete 12480/12480) / float32 | NEAR (0.0190) | 60 | 3.805 | 412.3 | 26.099 | 1135.5 | 2.0506→0.8200 | 1.9251→0.7572 | -0.0026 | patched 48 / kcalls 768 | 642514944 | 4-bit experts (e4b NF4) + NF4 attention |  |
| e4b | fused_attn4_m_dq0_d2 | **OK** | VALID | **VALID** | yes | matched:3407 (complete 12480/12480) / float32 | SAME-BYTES-CLASS (0.0000) | 60 | 3.727 | 422.5 | 27.445 | 1112.7 | 2.0614→0.8198 | 1.9441→0.7583 | -0.0014 | patched 48 / kcalls 768 | 642514944 | 4-bit experts (e4b NF4) + NF4 attention |  |
- prologue `e4b/fused_attn4_m_dq0` **108.3 s** before step 1 (31% of the arm): c1_before 53.9, c1_after 27.9, load_weights 24.9, eval0 11.1; unattributed 1.253; budget 1260.0
- prologue `e4b/fused_attn4_m_dq1` **98.6 s** before step 1 (30% of the arm): c1_before 50.9, load_weights 28.1, c1_after 25.4, attn4 5.6; unattributed 1.248; budget 1260.0
- prologue `e4b/fused_attn4_m_dq1_d2` **95.9 s** before step 1 (29% of the arm): c1_before 50.8, c1_after 26.0, load_weights 25.2, attn4 5.9; unattributed 1.24; budget 1260.0
- prologue `e4b/fused_attn4_m_dq0_d2` **100.8 s** before step 1 (30% of the arm): c1_before 55.2, c1_after 27.2, load_weights 25.4, attn4 5.6; unattributed 1.237; budget 1260.0
- draws (R1): `e4b/fused_attn4_m_dq0` STABLE (3.751/3.727 s, |Δ|/mean 0.6% vs 5%); `e4b/fused_attn4_m_dq1` STABLE (3.777/3.805 s, |Δ|/mean 0.7% vs 5%)
- e4b internal parity: NO-ARM
- **NO MATCHED POSITION QUOTED (unsloth)** — not quoted: e4b STABLE / unsloth no receipt
- **NO MATCHED POSITION QUOTED (hf)** — not quoted: e4b STABLE / hf no receipt
- **NO MATCHED POSITION QUOTED (axolotl)** — not quoted: e4b STABLE / axolotl no receipt
- equivalence vs `e4b/fused_attn4_m` (I/H; EQUIVALENT iff both ≤ band = max(0.005, 3 × fused-vs-reference |Δ|) = none (no reference arm: COMPARABLE is the ceiling); COMPARABLE ≤ 0.05; draw-noise floor 0.0014): `e4b/fused_attn4_m_dq1` **COMPARABLE** (median step |Δ| 0.0013, |Δ held-out at N| 0.0030, step-0 0.0190 NEAR, |Δ loss at step 2| 0.0006, paired rows mean -0.0031 ± 0.0017 SE over 8, favouring arm 7 / anchor 1) — no reference arm ran: the EQUIVALENT band cannot be set, COMPARABLE is the ceiling; `e4b/fused_attn4_m_dq1_d2` **COMPARABLE** (median step |Δ| 0.0012, |Δ held-out at N| 0.0026, step-0 0.0190 NEAR, |Δ loss at step 2| 0.0136, paired rows mean -0.0026 ± 0.0021 SE over 8, favouring arm 6 / anchor 2) — no reference arm ran: the EQUIVALENT band cannot be set, COMPARABLE is the ceiling; `e4b/fused_attn4_m_dq0_d2` **COMPARABLE** (median step |Δ| 0.0010, |Δ held-out at N| 0.0014, step-0 0.0000 SAME-BYTES-CLASS, |Δ loss at step 2| 0.0147, paired rows mean -0.0014 ± 0.0015 SE over 8, favouring arm 6 / anchor 2) — no reference arm ran: the EQUIVALENT band cannot be set, COMPARABLE is the ceiling
- matched_init_sha (B, name-free, canonical slot order): anchor `f7832488926eda91`; `e4b/fused_attn4_m_dq0` same; `e4b/fused_attn4_m_dq1` same; `e4b/fused_attn4_m_dq1_d2` same; `e4b/fused_attn4_m_dq0_d2` same
- frozen base (R5), anchor `e4b/fused_attn4_m`: gate_up nf4/64 sha 8c3b3fd78e89 control detects, down nf4/64 sha 59569d810a1a control detects, q_proj nf4/64+dq sha 5c5e402ce51d control detects
- frozen base `e4b/fused_attn4_m_dq1`: gate_up **N-A** (regime differs: anchor nf4/64 vs nf4/64+dq) control detects, down **N-A** (regime differs: anchor nf4/64 vs nf4/64+dq) control detects, q_proj **SAME-BYTES** (nf4/64+dq) control detects
- frozen base `e4b/fused_attn4_m_dq1_d2`: gate_up **N-A** (regime differs: anchor nf4/64 vs nf4/64+dq) control detects, down **N-A** (regime differs: anchor nf4/64 vs nf4/64+dq) control detects, q_proj **SAME-BYTES** (nf4/64+dq) control detects
- frozen base `e4b/fused_attn4_m_dq0_d2`: gate_up **SAME-BYTES** (nf4/64) control detects, down **SAME-BYTES** (nf4/64) control detects, q_proj **SAME-BYTES** (nf4/64+dq) control detects

## Verdicts
| family | arm | VERDICT | validity | quality | note |
|---|---|---|---|---|---|
| qwen3dqab | e4b/fused_attn4_m_dq0 | **VALID** | VALID | 0.0000 |  |
| qwen3dqab | e4b/fused_attn4_m_dq1 | **VALID** | VALID | -0.0030 |  |
| qwen3dqab | e4b/fused_attn4_m_dq1_d2 | **VALID** | VALID | -0.0026 |  |
| qwen3dqab | e4b/fused_attn4_m_dq0_d2 | **VALID** | VALID | -0.0014 |  |

## Predictions P56 / P57 / P58 (TC1-PREREG amendment 28: the expert absmax fp32 vs double-quantized, two stable draws a side; scored mechanically)
| prediction | family | verdict | evidence |
|---|---|---|---|
| P56 | qwen3dqab | **HELD** | Qwen3-30B-A3B: dq1 / dq0 1.014 [1.007, 1.021 over 4 cross-draw ratios] vs [0.97, 1.03]; peak dq0 27.44 / dq1 26.10 GB, drop 1.339 vs [1.25, 1.45]; s/step dq0 3.751 / 3.727 (within 0.6%), dq1 3.777 / 3.805 (within 0.7%) |
| P57 | mixtraldqab | **UNTESTED** | Mixtral-8x7B: no mixtraldqab receipts in this directory |
| P58 | dqab | **UNTESTED** | qwen3dqab: mean held-out dq1 - dq0 -0.0021 (|.| <= 0.005); held-out at N dq0 [0.7597, 0.7583] dq1 [0.7567, 0.7572]; mixtraldqab: no receipts |
