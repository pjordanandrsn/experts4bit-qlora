# TC1 — Qwen3-30B-A3B on one RTX 5090: e4b vs Unsloth vs HF+PEFT vs axolotl at matched work, init and adapter precision (tc1)
Rule (tc1/TC1-PREREG.md): status per attempt in the vocabulary OK / REFUSED / OOM / INSTALL_FAILED / LOAD_FAULT / HARNESS_ERROR / ALARM / NOT_RUN; VALID/VOID per tp4's predicates plus the matched-set predicates (R3: matched init complete, fp32 adapters, step-0 held-out within 0.005 of e4b/reference_attn4_m); VERDICT exactly one of VALID / VOID / QUALITY_FAIL / OOM / UNSUPPORTED / HARNESS_ERROR / ALARM / NOT_RUN (QUALITY_FAIL = held-out |Δ| at N > 0.05 vs e4b/fused_attn4_m); positions = s/step ratios other/e4b from the medians over both draws (stability |d1-d2|/mean ≤ 5 % e4b / 10 % others, else UNSTABLE), quoted only when both arms are VALID; equivalence vs e4b/reference_attn4_m EQUIVALENT ≤ 0.02 / COMPARABLE ≤ 0.05 / else DIVERGENT; frozen base SAME-BYTES / DIFFERENT / N-A per slot; predictions P1–P10 (+ P1b) scored HELD / FALSIFIED / UNTESTED. VOID never enters a ratio or an equivalence reading. Nothing is licensed; no cross-box number is divided into these.
`versions.txt`
```
e4b 0.47.0 @2f5301336f2a9f717768593739c3db4361a7c45c (GitHub main)
gnf4 0.40.0 @23a0153ab452cea3131705ce1815f72462f8bf3f (GitHub main)
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
 "run_id": "tc1-5090-74",
 "instance_id": "54237146",
 "gpu": "NVIDIA GeForce RTX 5090",
 "driver": "595.84",
 "cpu": "AMD EPYC 7B13 64-Core Processor",
 "nproc": 256,
 "mem_total_kb": "2101193408",
 "cgroup_memory_max": "730283900928",
 "disk_root": "overlay         320G   77M  320G   1% /",
 "hostname": "003aa5f039d5",
 "registered_gpu_class": "5090",
 "prereg": "tc1/TC1-PREREG.md",
 "local_box": null,
 "local_snapshot": null
}
```

### Qwen3-30B-A3B (amendment 32: venv-e4b with triton 3.4 vs 3.7.1, prebound launches off, matched and shipped arms) (`qwen3tritonab`, registered n_layers 48, attention census 192)
- model `Qwen/Qwen3-30B-A3B` @ `ad44e777bcd1`; tokens sha `bfc742f67e37`; N=60; fixture template alpaca seq 2048 micro-batch 2 × accum 4 lr 0.0002 r 16 α 16 optimizer adamw_8bit(lr=0.0002, weight_decay=0.001) schedule=linear warmup_steps=5 autocast False; e4b trainable 642514944; box_class RTX 5090 gpu NVIDIA GeForce RTX 5090
| framework | arm | status | validity | **VERDICT** | matched | init / dtype | step-0 class | N | s/step med(11+) | tok/s | peak GB | J/step | train first→last | held-out 0→final | quality Δ | engagement | trainable | regime | reason / why |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| e4b | fused_attn4_m_tr0 | **OK** | VALID | **VALID** | yes | matched:3407 (complete 12480/12480) / float32 | SAME-BYTES-CLASS (0.0000) | 60 | 3.827 | 396.1 | 27.433 | 1139.0 | 2.0614→0.8208 | 1.9441→0.7594 | 0.0000 | patched 48 / kcalls 768 | 642514944 | 4-bit experts (e4b NF4) + NF4 attention |  |
| e4b | fused_attn4_m_tr1 | **OK** | VALID | **VALID** | yes | matched:3407 (complete 12480/12480) / float32 | SAME-BYTES-CLASS (0.0000) | 60 | 3.894 | 392.1 | 27.455 | 1068.5 | 2.0614→0.8145 | 1.9441→0.7584 | -0.0010 | patched 48 / kcalls 768 | 642514944 | 4-bit experts (e4b NF4) + NF4 attention |  |
| e4b | fused_attn4_m_tr1_d2 | **OK** | VALID | **VALID** | yes | matched:3407 (complete 12480/12480) / float32 | SAME-BYTES-CLASS (0.0000) | 60 | 3.746 | 419.8 | 27.436 | 1045.3 | 2.0614→0.8159 | 1.9441→0.7586 | -0.0008 | patched 48 / kcalls 768 | 642514944 | 4-bit experts (e4b NF4) + NF4 attention |  |
| e4b | fused_attn4_m_tr0_d2 | **OK** | VALID | **VALID** | yes | matched:3407 (complete 12480/12480) / float32 | SAME-BYTES-CLASS (0.0000) | 60 | 3.871 | 408.5 | 27.441 | 1097.0 | 2.0614→0.8165 | 1.9441→0.7592 | -0.0001 | patched 48 / kcalls 768 | 642514944 | 4-bit experts (e4b NF4) + NF4 attention |  |
| e4b | fused_attn4_shipped_tr0 | **OK** | VALID | **VALID** | native | native / bfloat16,float32 | — | 60 | 3.117 | 513.0 | 24.623 | 874.4 | 2.0614→0.8161 | 1.9441→0.7567 | -0.0027 | patched 48 / kcalls 768 | 642514944 | 4-bit experts (e4b NF4) + NF4 attention |  |
| e4b | fused_attn4_shipped_tr1 | **OK** | VALID | **VALID** | native | native / bfloat16,float32 | — | 60 | 2.976 | 530.7 | 24.623 | 831.6 | 2.0614→0.8163 | 1.9441→0.7581 | -0.0012 | patched 48 / kcalls 768 | 642514944 | 4-bit experts (e4b NF4) + NF4 attention |  |
| e4b | fused_attn4_shipped_tr1_d2 | **OK** | VALID | **VALID** | native | native / bfloat16,float32 | — | 60 | 2.986 | 529.8 | 24.623 | 816.7 | 2.0614→0.8178 | 1.9441→0.7569 | -0.0024 | patched 48 / kcalls 768 | 642514944 | 4-bit experts (e4b NF4) + NF4 attention |  |
| e4b | fused_attn4_shipped_tr0_d2 | **OK** | VALID | **VALID** | native | native / bfloat16,float32 | — | 60 | 3.023 | 525.4 | 24.623 | 871.4 | 2.0614→0.8194 | 1.9441→0.7573 | -0.0020 | patched 48 / kcalls 768 | 642514944 | 4-bit experts (e4b NF4) + NF4 attention |  |
- prologue `e4b/fused_attn4_m_tr0` **111.1 s** before step 1 (31% of the arm): c1_before 54.2, c1_after 27.2, load_weights 27.2, eval0 11.5; unattributed 1.305; budget 1260.0
- prologue `e4b/fused_attn4_m_tr1` **119.4 s** before step 1 (32% of the arm): c1_before 55.4, load_weights 31.9, c1_after 26.9, eval0 13.5; unattributed 1.27; budget 1260.0
- prologue `e4b/fused_attn4_m_tr1_d2` **101.5 s** before step 1 (30% of the arm): c1_before 54.1, c1_after 28.1, load_weights 26.7, attn4 6.0; unattributed 1.286; budget 1260.0
- prologue `e4b/fused_attn4_m_tr0_d2` **103.3 s** before step 1 (30% of the arm): c1_before 54.7, c1_after 27.4, load_weights 26.9, attn4 6.2; unattributed 1.286; budget 1260.0
- prologue `e4b/fused_attn4_shipped_tr0` **106.4 s** before step 1 (36% of the arm): c1_before 55.1, load_weights 28.5, c1_after 27.5, trainable_sha 6.3; unattributed 1.286; budget 1260.0
- prologue `e4b/fused_attn4_shipped_tr1` **104.5 s** before step 1 (36% of the arm): c1_before 55.8, c1_after 27.6, load_weights 26.7, trainable_sha 6.0; unattributed 1.314; budget 1260.0
- prologue `e4b/fused_attn4_shipped_tr1_d2` **106.3 s** before step 1 (37% of the arm): c1_before 56.3, load_weights 27.9, c1_after 27.9, trainable_sha 6.2; unattributed 1.267; budget 1260.0
- prologue `e4b/fused_attn4_shipped_tr0_d2` **103.3 s** before step 1 (36% of the arm): c1_before 54.4, c1_after 27.8, load_weights 26.4, trainable_sha 6.2; unattributed 1.277; budget 1260.0
- draws (R1): `e4b/fused_attn4_m_tr0` STABLE (3.827/3.871 s, |Δ|/mean 1.1% vs 5%); `e4b/fused_attn4_m_tr1` STABLE (3.894/3.746 s, |Δ|/mean 3.9% vs 5%); `e4b/fused_attn4_shipped_tr0` STABLE (3.117/3.023 s, |Δ|/mean 3.1% vs 5%); `e4b/fused_attn4_shipped_tr1` STABLE (2.976/2.986 s, |Δ|/mean 0.3% vs 5%)
- e4b internal parity: NO-ARM
- **NO MATCHED POSITION QUOTED (unsloth)** — not quoted: e4b STABLE / unsloth no receipt
- **NO MATCHED POSITION QUOTED (hf)** — not quoted: e4b STABLE / hf no receipt
- **NO MATCHED POSITION QUOTED (axolotl)** — not quoted: e4b STABLE / axolotl no receipt
- equivalence vs `e4b/fused_attn4_m` (I/H; EQUIVALENT iff both ≤ band = max(0.005, 3 × fused-vs-reference |Δ|) = none (no reference arm: COMPARABLE is the ceiling); COMPARABLE ≤ 0.05; draw-noise floor 0.0012): `e4b/fused_attn4_m_tr1` **COMPARABLE** (median step |Δ| 0.0014, |Δ held-out at N| 0.0010, step-0 0.0000 SAME-BYTES-CLASS, |Δ loss at step 2| 0.0032, paired rows mean -0.0010 ± 0.0021 SE over 8, favouring arm 4 / anchor 4) — no reference arm ran: the EQUIVALENT band cannot be set, COMPARABLE is the ceiling; `e4b/fused_attn4_m_tr1_d2` **COMPARABLE** (median step |Δ| 0.0010, |Δ held-out at N| 0.0008, step-0 0.0000 SAME-BYTES-CLASS, |Δ loss at step 2| 0.0015, paired rows mean -0.0008 ± 0.0007 SE over 8, favouring arm 5 / anchor 3) — no reference arm ran: the EQUIVALENT band cannot be set, COMPARABLE is the ceiling; `e4b/fused_attn4_m_tr0_d2` **COMPARABLE** (median step |Δ| 0.0012, |Δ held-out at N| 0.0001, step-0 0.0000 SAME-BYTES-CLASS, |Δ loss at step 2| 0.0036, paired rows mean -0.0001 ± 0.0013 SE over 8, favouring arm 4 / anchor 4) — no reference arm ran: the EQUIVALENT band cannot be set, COMPARABLE is the ceiling
- matched_init_sha (B, name-free, canonical slot order): anchor `f7832488926eda91`; `e4b/fused_attn4_m_tr0` same; `e4b/fused_attn4_m_tr1` same; `e4b/fused_attn4_m_tr1_d2` same; `e4b/fused_attn4_m_tr0_d2` same
- frozen base (R5), anchor `e4b/fused_attn4_m`: gate_up nf4/64 sha 8c3b3fd78e89 control detects, down nf4/64 sha 59569d810a1a control detects, q_proj nf4/64+dq sha 5c5e402ce51d control detects
- frozen base `e4b/fused_attn4_m_tr1`: gate_up **SAME-BYTES** (nf4/64) control detects, down **SAME-BYTES** (nf4/64) control detects, q_proj **SAME-BYTES** (nf4/64+dq) control detects
- frozen base `e4b/fused_attn4_m_tr1_d2`: gate_up **SAME-BYTES** (nf4/64) control detects, down **SAME-BYTES** (nf4/64) control detects, q_proj **SAME-BYTES** (nf4/64+dq) control detects
- frozen base `e4b/fused_attn4_m_tr0_d2`: gate_up **SAME-BYTES** (nf4/64) control detects, down **SAME-BYTES** (nf4/64) control detects, q_proj **SAME-BYTES** (nf4/64+dq) control detects
- frozen base `e4b/fused_attn4_shipped_tr0`: gate_up **SAME-BYTES** (nf4/64) control detects, down **SAME-BYTES** (nf4/64) control detects, q_proj **SAME-BYTES** (nf4/64+dq) control detects
- frozen base `e4b/fused_attn4_shipped_tr1`: gate_up **SAME-BYTES** (nf4/64) control detects, down **SAME-BYTES** (nf4/64) control detects, q_proj **SAME-BYTES** (nf4/64+dq) control detects
- frozen base `e4b/fused_attn4_shipped_tr1_d2`: gate_up **SAME-BYTES** (nf4/64) control detects, down **SAME-BYTES** (nf4/64) control detects, q_proj **SAME-BYTES** (nf4/64+dq) control detects
- frozen base `e4b/fused_attn4_shipped_tr0_d2`: gate_up **SAME-BYTES** (nf4/64) control detects, down **SAME-BYTES** (nf4/64) control detects, q_proj **SAME-BYTES** (nf4/64+dq) control detects

## Verdicts
| family | arm | VERDICT | validity | quality | note |
|---|---|---|---|---|---|
| qwen3tritonab | e4b/fused_attn4_m_tr0 | **VALID** | VALID | 0.0000 |  |
| qwen3tritonab | e4b/fused_attn4_m_tr1 | **VALID** | VALID | -0.0010 |  |
| qwen3tritonab | e4b/fused_attn4_m_tr1_d2 | **VALID** | VALID | -0.0008 |  |
| qwen3tritonab | e4b/fused_attn4_m_tr0_d2 | **VALID** | VALID | -0.0001 |  |
| qwen3tritonab | e4b/fused_attn4_shipped_tr0 | **VALID** | VALID | -0.0027 |  |
| qwen3tritonab | e4b/fused_attn4_shipped_tr1 | **VALID** | VALID | -0.0012 |  |
| qwen3tritonab | e4b/fused_attn4_shipped_tr1_d2 | **VALID** | VALID | -0.0024 |  |
| qwen3tritonab | e4b/fused_attn4_shipped_tr0_d2 | **VALID** | VALID | -0.0020 |  |

## Predictions P59 / P60 / P61 (TC1-PREREG amendment 32: venv-e4b with triton 3.4 vs 3.7.1, two stable draws a side; scored mechanically)
| prediction | family | verdict | evidence |
|---|---|---|---|
| P59 | qwen3tritonab | **FALSIFIED** | matched: tr1 / tr0 0.992 [0.968, 1.017 over 4 cross-draw ratios] vs [0.82, 0.95]; s/step tr0 3.827 / 3.871 (within 1.1%), tr1 3.894 / 3.746 (within 3.9%); amendment 24's environment A/B (torch, transformers and triton together) read 0.882 on the matched arm |
| P60 | qwen3tritonab | **HELD** | shipped: tr1 / tr0 0.971 [0.955, 0.988 over 4 cross-draw ratios] vs [0.85, 0.98]; s/step tr0 3.117 / 3.023 (within 3.1%), tr1 2.976 / 2.986 (within 0.3%); amendment 24's environment A/B (torch, transformers and triton together) read 0.882 on the matched arm |
| P61 | qwen3tritonab | **HELD** | matched: mean held-out tr1 - tr0 -0.0008 (|.| <= 0.005); held-out at N tr0 [0.7594, 0.7592] tr1 [0.7584, 0.7586]; shipped: mean held-out tr1 - tr0 +0.0005 (|.| <= 0.005); held-out at N tr0 [0.7567, 0.7573] tr1 [0.7581, 0.7569] |
