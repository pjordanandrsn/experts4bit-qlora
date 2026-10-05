# TC1 — Qwen3-30B-A3B on one RTX 5090: e4b vs Unsloth vs HF+PEFT vs axolotl at matched work, init and adapter precision (tc1)
Rule (tc1/TC1-PREREG.md): status per attempt in the vocabulary OK / REFUSED / OOM / INSTALL_FAILED / LOAD_FAULT / HARNESS_ERROR / ALARM / NOT_RUN; VALID/VOID per tp4's predicates plus the matched-set predicates (R3: matched init complete, fp32 adapters, step-0 held-out within 0.005 of e4b/reference_attn4_m); VERDICT exactly one of VALID / VOID / QUALITY_FAIL / OOM / UNSUPPORTED / HARNESS_ERROR / ALARM / NOT_RUN (QUALITY_FAIL = held-out |Δ| at N > 0.05 vs e4b/fused_attn4_m); positions = s/step ratios other/e4b from the medians over both draws (stability |d1-d2|/mean ≤ 5 % e4b / 10 % others, else UNSTABLE), quoted only when both arms are VALID; equivalence vs e4b/reference_attn4_m EQUIVALENT ≤ 0.02 / COMPARABLE ≤ 0.05 / else DIVERGENT; frozen base SAME-BYTES / DIFFERENT / N-A per slot; predictions P1–P10 (+ P1b) scored HELD / FALSIFIED / UNTESTED. VOID never enters a ratio or an equivalence reading. Nothing is licensed; no cross-box number is divided into these.
`versions.txt`
```
e4b 0.46.0 @cdea8ea92e45f486d7cb945a7b390d184abea3b5 (GitHub main)
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
 "run_id": "tc1-5090-70",
 "instance_id": "54223173",
 "gpu": "NVIDIA GeForce RTX 5090",
 "driver": "595.71.05",
 "cpu": "AMD EPYC 7C13 64-Core Processor",
 "nproc": 256,
 "mem_total_kb": "1056596104",
 "cgroup_memory_max": "367227043840",
 "disk_root": "overlay         320G   56M  320G   1% /",
 "hostname": "04139ce920e0",
 "registered_gpu_class": "5090",
 "prereg": "tc1/TC1-PREREG.md",
 "local_box": null,
 "local_snapshot": null
}
```

### Qwen3-30B-A3B (amendment 28: the expert absmax fp32 vs double-quantized, matched arm, resident) (`qwen3dqab`, registered n_layers 48, attention census 192)
- model `Qwen/Qwen3-30B-A3B` @ `ad44e777bcd1`; tokens sha `bfc742f67e37`; N=20; fixture template alpaca seq 2048 micro-batch 2 × accum 4 lr 0.0002 r 16 α 16 optimizer adamw_8bit(lr=0.0002, weight_decay=0.001) schedule=linear warmup_steps=5 autocast False; e4b trainable 642514944; box_class RTX 5090 gpu NVIDIA GeForce RTX 5090
| framework | arm | status | validity | **VERDICT** | matched | init / dtype | step-0 class | N | s/step med(11+) | tok/s | peak GB | J/step | train first→last | held-out 0→final | quality Δ | engagement | trainable | regime | reason / why |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| e4b | fused_attn4_m_dq0 | **OK** | VALID | **VALID** | yes | matched:3407 (complete 12480/12480) / float32 | SAME-BYTES-CLASS (0.0000) | 20 | 3.801 | 349.8 | 27.137 | 957.3 | 2.0614→0.8331 | 1.9441→0.8516 | 0.0000 | patched 48 / kcalls 768 | 642514944 | 4-bit experts (e4b NF4) + NF4 attention |  |
| e4b | fused_attn4_m_dq1 | **OK** | VALID | **VALID** | yes | matched:3407 (complete 12480/12480) / float32 | NEAR (0.0190) | 20 | 3.947 | 367.2 | 25.817 | 872.8 | 2.0506→0.8323 | 1.9251→0.8538 | 0.0022 | patched 48 / kcalls 768 | 642514944 | 4-bit experts (e4b NF4) + NF4 attention |  |
| e4b | fused_attn4_m_dq1_d2 | **OK** | VALID | **VALID** | yes | matched:3407 (complete 12480/12480) / float32 | NEAR (0.0190) | 20 | 5.014 | 320.8 | 25.817 | 977.0 | 2.0506→0.8354 | 1.9251→0.8537 | 0.0022 | patched 48 / kcalls 768 | 642514944 | 4-bit experts (e4b NF4) + NF4 attention |  |
| e4b | fused_attn4_m_dq0_d2 | **OK** | VALID | **VALID** | yes | matched:3407 (complete 12480/12480) / float32 | SAME-BYTES-CLASS (0.0000) | 20 | 4.019 | 377.3 | 27.157 | 917.4 | 2.0614→0.8349 | 1.9441→0.8508 | -0.0007 | patched 48 / kcalls 768 | 642514944 | 4-bit experts (e4b NF4) + NF4 attention |  |
- prologue `e4b/fused_attn4_m_dq0` **162.0 s** before step 1 (65% of the arm): c1_before 73.5, load_weights 54.0, c1_after 33.2, adapter_save 32.7; unattributed 1.44; budget 1260.0
- prologue `e4b/fused_attn4_m_dq1` **139.1 s** before step 1 (62% of the arm): c1_before 61.8, load_weights 54.3, adapter_save 48.1, c1_after 29.5; unattributed 1.67; budget 1260.0
- prologue `e4b/fused_attn4_m_dq1_d2` **138.7 s** before step 1 (59% of the arm): c1_before 62.3, load_weights 50.8, c1_after 34.7, adapter_save 21.3; unattributed 1.755; budget 1260.0
- prologue `e4b/fused_attn4_m_dq0_d2` **134.1 s** before step 1 (62% of the arm): c1_before 59.7, load_weights 50.7, c1_after 30.6, adapter_save 26.8; unattributed 1.645; budget 1260.0
- draws (R1): `e4b/fused_attn4_m_dq0` UNSTABLE (3.801/4.019 s, |Δ|/mean 5.6% vs 5%); `e4b/fused_attn4_m_dq1` UNSTABLE (3.947/5.014 s, |Δ|/mean 23.8% vs 5%)
- e4b internal parity: NO-ARM
- **NO MATCHED POSITION QUOTED (unsloth)** — not quoted: e4b draws 3.801 / 4.019 s/step differ by 5.6% > 5% (UNSTABLE: reported, not quoted) / unsloth no receipt
- **NO MATCHED POSITION QUOTED (hf)** — not quoted: e4b draws 3.801 / 4.019 s/step differ by 5.6% > 5% (UNSTABLE: reported, not quoted) / hf no receipt
- **NO MATCHED POSITION QUOTED (axolotl)** — not quoted: e4b draws 3.801 / 4.019 s/step differ by 5.6% > 5% (UNSTABLE: reported, not quoted) / axolotl no receipt
- equivalence vs `e4b/fused_attn4_m` (I/H; EQUIVALENT iff both ≤ band = max(0.005, 3 × fused-vs-reference |Δ|) = none (no reference arm: COMPARABLE is the ceiling); COMPARABLE ≤ 0.05; draw-noise floor —): `e4b/fused_attn4_m_dq1` **COMPARABLE** (median step |Δ| 0.0016, |Δ held-out at N| 0.0022, step-0 0.0190 NEAR, |Δ loss at step 2| 0.0104, paired rows mean +0.0022 ± 0.0034 SE over 8, favouring arm 5 / anchor 3) — no reference arm ran: the EQUIVALENT band cannot be set, COMPARABLE is the ceiling; `e4b/fused_attn4_m_dq1_d2` **COMPARABLE** (median step |Δ| 0.0015, |Δ held-out at N| 0.0022, step-0 0.0190 NEAR, |Δ loss at step 2| 0.0010, paired rows mean +0.0022 ± 0.0015 SE over 8, favouring arm 3 / anchor 5) — no reference arm ran: the EQUIVALENT band cannot be set, COMPARABLE is the ceiling; `e4b/fused_attn4_m_dq0_d2` **COMPARABLE** (median step |Δ| 0.0010, |Δ held-out at N| 0.0007, step-0 0.0000 SAME-BYTES-CLASS, |Δ loss at step 2| 0.0038, paired rows mean -0.0007 ± 0.0020 SE over 8, favouring arm 5 / anchor 3) — no reference arm ran: the EQUIVALENT band cannot be set, COMPARABLE is the ceiling
- matched_init_sha (B, name-free, canonical slot order): anchor `f7832488926eda91`; `e4b/fused_attn4_m_dq0` same; `e4b/fused_attn4_m_dq1` same; `e4b/fused_attn4_m_dq1_d2` same; `e4b/fused_attn4_m_dq0_d2` same
- frozen base (R5), anchor `e4b/fused_attn4_m`: gate_up nf4/64 sha 8c3b3fd78e89 control detects, down nf4/64 sha 59569d810a1a control detects, q_proj nf4/64+dq sha 5c5e402ce51d control detects
- frozen base `e4b/fused_attn4_m_dq1`: gate_up **N-A** (regime differs: anchor nf4/64 vs nf4/64+dq) control detects, down **N-A** (regime differs: anchor nf4/64 vs nf4/64+dq) control detects, q_proj **SAME-BYTES** (nf4/64+dq) control detects
- frozen base `e4b/fused_attn4_m_dq1_d2`: gate_up **N-A** (regime differs: anchor nf4/64 vs nf4/64+dq) control detects, down **N-A** (regime differs: anchor nf4/64 vs nf4/64+dq) control detects, q_proj **SAME-BYTES** (nf4/64+dq) control detects
- frozen base `e4b/fused_attn4_m_dq0_d2`: gate_up **SAME-BYTES** (nf4/64) control detects, down **SAME-BYTES** (nf4/64) control detects, q_proj **SAME-BYTES** (nf4/64+dq) control detects

## Verdicts
| family | arm | VERDICT | validity | quality | note |
|---|---|---|---|---|---|
| qwen3dqab | e4b/fused_attn4_m_dq0 | **VALID** | VALID | 0.0000 |  |
| qwen3dqab | e4b/fused_attn4_m_dq1 | **VALID** | VALID | 0.0022 |  |
| qwen3dqab | e4b/fused_attn4_m_dq1_d2 | **VALID** | VALID | 0.0022 |  |
| qwen3dqab | e4b/fused_attn4_m_dq0_d2 | **VALID** | VALID | -0.0007 |  |

## Predictions P56 / P57 / P58 (TC1-PREREG amendment 28: the expert absmax fp32 vs double-quantized, two stable draws a side; scored mechanically)
| prediction | family | verdict | evidence |
|---|---|---|---|
| P56 | qwen3dqab | **UNTESTED** | Qwen3-30B-A3B: two stable VALID draws a side are registered -- dq0 UNSTABLE: draws 3.801 / 4.019 s/step differ by 5.6% > 5% (UNSTABLE: reported, not quoted); dq1 UNSTABLE: draws 3.947 / 5.014 s/step differ by 23.8% > 5% (UNSTABLE: reported, not quoted) |
| P57 | mixtraldqab | **UNTESTED** | Mixtral-8x7B: no mixtraldqab receipts in this directory |
| P58 | dqab | **UNTESTED** | qwen3dqab: dq0 UNSTABLE: draws 3.801 / 4.019 s/step differ by 5.6% > 5% (UNSTABLE: reported, not quoted); dq1 UNSTABLE: draws 3.947 / 5.014 s/step differ by 23.8% > 5% (UNSTABLE: reported, not quoted); mixtraldqab: no receipts |
