# TC1 — Qwen3-30B-A3B on one RTX 5090: e4b vs Unsloth vs HF+PEFT vs axolotl at matched work, init and adapter precision (tc1)
Rule (tc1/TC1-PREREG.md): status per attempt in the vocabulary OK / REFUSED / OOM / INSTALL_FAILED / LOAD_FAULT / HARNESS_ERROR / ALARM / NOT_RUN; VALID/VOID per tp4's predicates plus the matched-set predicates (R3: matched init complete, fp32 adapters, step-0 held-out within 0.005 of e4b/reference_attn4_m); VERDICT exactly one of VALID / VOID / QUALITY_FAIL / OOM / UNSUPPORTED / HARNESS_ERROR / ALARM / NOT_RUN (QUALITY_FAIL = held-out |Δ| at N > 0.05 vs e4b/fused_attn4_m); positions = s/step ratios other/e4b from the medians over both draws (stability |d1-d2|/mean ≤ 5 % e4b / 10 % others, else UNSTABLE), quoted only when both arms are VALID; equivalence vs e4b/reference_attn4_m EQUIVALENT ≤ 0.02 / COMPARABLE ≤ 0.05 / else DIVERGENT; frozen base SAME-BYTES / DIFFERENT / N-A per slot; predictions P1–P10 (+ P1b) scored HELD / FALSIFIED / UNTESTED. VOID never enters a ratio or an equivalence reading. Nothing is licensed; no cross-box number is divided into these.
`versions.txt`
```
e4b 0.41.0 @e1296508aa4b75c9a86c606b358f6cd57704e113 (GitHub main)
gnf4 0.34.1 @58fb19d4c3f61889255dc4ed588f30e3adc3be0d (GitHub main)
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
 "run_id": "tc1-5090-42",
 "instance_id": "54006898",
 "gpu": "NVIDIA GeForce RTX 5090",
 "driver": "580.178.04",
 "cpu": "AMD Ryzen 9 7900 12-Core Processor",
 "nproc": 24,
 "mem_total_kb": "130952932",
 "cgroup_memory_max": "128730529792",
 "disk_root": "overlay         320G  9.3M  320G   1% /",
 "hostname": "90606a9f8d52",
 "registered_gpu_class": "5090",
 "prereg": "tc1/TC1-PREREG.md",
 "local_box": null,
 "local_snapshot": null
}
```

### Qwen3-30B-A3B (amendment 13: gnf4's previous padded LoRA delta vs its trimmed body, both on the post-#945 sync path) (`qwen3leanab`, registered n_layers 48, attention census 192)
- model `Qwen/Qwen3-30B-A3B` @ `ad44e777bcd1`; tokens sha `bfc742f67e37`; N=20; fixture template alpaca seq 2048 micro-batch 2 × accum 4 lr 0.0002 r 16 α 16 optimizer adamw_8bit(lr=0.0002, weight_decay=0.001) schedule=linear warmup_steps=5 autocast False; e4b trainable 642514944; box_class RTX 5090 gpu NVIDIA GeForce RTX 5090
| framework | arm | status | validity | **VERDICT** | matched | init / dtype | step-0 class | N | s/step med(11+) | tok/s | peak GB | J/step | train first→last | held-out 0→final | quality Δ | engagement | trainable | regime | reason / why |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| e4b | fused_attn4_shipped_lean0 | **OK** | VALID | **VALID** | native | native / bfloat16,float32 | — | 20 | 2.371 | 631.3 | 24.581 | 826.0 | 2.0705→0.7985 | 1.9505→0.8113 | -0.0384 | patched 48 / kcalls 768 | 642514944 | 4-bit experts (e4b NF4) + NF4 attention |  |
| e4b | fused_attn4_shipped_lean1 | **OK** | VALID | **VALID** | native | native / bfloat16,float32 | — | 20 | 2.218 | 694.7 | 24.581 | 729.8 | 2.0705→0.7994 | 1.9505→0.8110 | -0.0387 | patched 48 / kcalls 768 | 642514944 | 4-bit experts (e4b NF4) + NF4 attention |  |
| e4b | fused_attn4_m_lean0 | **OK** | VALID | **VALID** | yes | matched:3407 (complete 12480/12480) / float32 | SAME-BYTES-CLASS (0.0000) | 20 | 2.949 | 517.1 | 27.822 | 1008.6 | 2.0705→0.8329 | 1.9505→0.8497 | 0.0000 | patched 48 / kcalls 768 | 642514944 | 4-bit experts (e4b NF4) + NF4 attention |  |
| e4b | fused_attn4_m_lean1 | **OK** | VALID | **VALID** | yes | matched:3407 (complete 12480/12480) / float32 | SAME-BYTES-CLASS (0.0000) | 20 | 2.728 | 577.7 | 27.237 | 859.0 | 2.0705→0.8283 | 1.9505→0.8489 | -0.0008 | patched 48 / kcalls 768 | 642514944 | 4-bit experts (e4b NF4) + NF4 attention |  |
| e4b | fused_attn4_m_lean1_d2 | **OK** | VALID | **VALID** | yes | matched:3407 (complete 12480/12480) / float32 | SAME-BYTES-CLASS (0.0000) | 20 | 2.654 | 580.0 | 27.239 | 881.9 | 2.0705→0.8327 | 1.9505→0.8502 | 0.0004 | patched 48 / kcalls 768 | 642514944 | 4-bit experts (e4b NF4) + NF4 attention |  |
| e4b | fused_attn4_m_lean0_d2 | **OK** | VALID | **VALID** | yes | matched:3407 (complete 12480/12480) / float32 | SAME-BYTES-CLASS (0.0000) | 20 | 2.955 | 516.9 | 27.846 | 1020.5 | 2.0705→0.8326 | 1.9505→0.8516 | 0.0018 | patched 48 / kcalls 768 | 642514944 | 4-bit experts (e4b NF4) + NF4 attention |  |
| e4b | fused_attn4_shipped_lean1_d2 | **OK** | VALID | **VALID** | native | native / bfloat16,float32 | — | 20 | 2.217 | 695.0 | 24.581 | 728.3 | 2.0705→0.7967 | 1.9505→0.8127 | -0.0371 | patched 48 / kcalls 768 | 642514944 | 4-bit experts (e4b NF4) + NF4 attention |  |
| e4b | fused_attn4_shipped_lean0_d2 | **OK** | VALID | **VALID** | native | native / bfloat16,float32 | — | 20 | 2.350 | 653.6 | 24.581 | 745.3 | 2.0705→0.7967 | 1.9505→0.8160 | -0.0337 | patched 48 / kcalls 768 | 642514944 | 4-bit experts (e4b NF4) + NF4 attention |  |
- prologue `e4b/fused_attn4_shipped_lean0` **94.5 s** before step 1 (66% of the arm): c1_before 42.1, load_weights 32.9, c1_after 21.0, eval0 8.1; unattributed 0.778; budget 1241.1
- prologue `e4b/fused_attn4_shipped_lean1` **68.9 s** before step 1 (61% of the arm): c1_before 40.5, c1_after 20.5, load_weights 16.0, attn4 3.7; unattributed 0.865; budget 1181.9
- prologue `e4b/fused_attn4_m_lean0` **68.6 s** before step 1 (53% of the arm): c1_before 40.6, c1_after 20.5, load_weights 15.3, attn4 3.7; unattributed 0.739; budget 1133.6
- prologue `e4b/fused_attn4_m_lean1` **70.9 s** before step 1 (57% of the arm): c1_before 43.0, c1_after 21.2, load_weights 15.3, attn4 3.7; unattributed 0.774; budget 1079.0
- prologue `e4b/fused_attn4_m_lean1_d2` **70.7 s** before step 1 (57% of the arm): c1_before 41.6, c1_after 20.6, load_weights 15.4, attn4 3.7; unattributed 0.743; budget 1026.2
- prologue `e4b/fused_attn4_m_lean0_d2` **69.6 s** before step 1 (54% of the arm): c1_before 41.5, c1_after 20.7, load_weights 15.4, attn4 3.7; unattributed 0.746; budget 973.7
- prologue `e4b/fused_attn4_shipped_lean1_d2` **68.2 s** before step 1 (60% of the arm): c1_before 41.2, c1_after 20.8, load_weights 15.4, attn4 3.7; unattributed 0.745; budget 919.1
- prologue `e4b/fused_attn4_shipped_lean0_d2` **67.8 s** before step 1 (59% of the arm): c1_before 41.0, c1_after 20.6, load_weights 15.4, attn4 3.6; unattributed 0.743; budget 870.8
- draws (R1): `e4b/fused_attn4_shipped_lean0` STABLE (2.371/2.350 s, |Δ|/mean 0.9% vs 5%); `e4b/fused_attn4_shipped_lean1` STABLE (2.218/2.217 s, |Δ|/mean 0.0% vs 5%); `e4b/fused_attn4_m_lean0` STABLE (2.949/2.955 s, |Δ|/mean 0.2% vs 5%); `e4b/fused_attn4_m_lean1` STABLE (2.728/2.654 s, |Δ|/mean 2.7% vs 5%)
- e4b internal parity: NO-ARM
- **NO MATCHED POSITION QUOTED (unsloth)** — not quoted: e4b STABLE / unsloth no receipt
- **NO MATCHED POSITION QUOTED (hf)** — not quoted: e4b STABLE / hf no receipt
- **NO MATCHED POSITION QUOTED (axolotl)** — not quoted: e4b STABLE / axolotl no receipt
- equivalence vs `e4b/fused_attn4_m` (I/H; EQUIVALENT iff both ≤ band = max(0.005, 3 × fused-vs-reference |Δ|) = none (no reference arm: COMPARABLE is the ceiling); COMPARABLE ≤ 0.05; draw-noise floor 0.0047): `e4b/fused_attn4_m_lean1` **COMPARABLE** (median step |Δ| 0.0013, |Δ held-out at N| 0.0008, step-0 0.0000 SAME-BYTES-CLASS, |Δ loss at step 2| 0.0013, paired rows mean -0.0008 ± 0.0019 SE over 8, favouring arm 4 / anchor 4) — no reference arm ran: the EQUIVALENT band cannot be set, COMPARABLE is the ceiling; `e4b/fused_attn4_m_lean1_d2` **COMPARABLE** (median step |Δ| 0.0024, |Δ held-out at N| 0.0004, step-0 0.0000 SAME-BYTES-CLASS, |Δ loss at step 2| 0.0028, paired rows mean +0.0004 ± 0.0017 SE over 8, favouring arm 3 / anchor 5) — no reference arm ran: the EQUIVALENT band cannot be set, COMPARABLE is the ceiling; `e4b/fused_attn4_m_lean0_d2` **COMPARABLE** (median step |Δ| 0.0018, |Δ held-out at N| 0.0018, step-0 0.0000 SAME-BYTES-CLASS, |Δ loss at step 2| 0.0011, paired rows mean +0.0018 ± 0.0025 SE over 8, favouring arm 3 / anchor 5) — no reference arm ran: the EQUIVALENT band cannot be set, COMPARABLE is the ceiling
- matched_init_sha (B, name-free, canonical slot order): anchor `f7832488926eda91`; `e4b/fused_attn4_m_lean0` same; `e4b/fused_attn4_m_lean1` same; `e4b/fused_attn4_m_lean1_d2` same; `e4b/fused_attn4_m_lean0_d2` same
- frozen base (R5), anchor `e4b/fused_attn4_m`: gate_up nf4/64 sha 8c3b3fd78e89 control detects, down nf4/64 sha 59569d810a1a control detects, q_proj nf4/64+dq sha 5c5e402ce51d control detects
- frozen base `e4b/fused_attn4_shipped_lean0`: gate_up **SAME-BYTES** (nf4/64) control detects, down **SAME-BYTES** (nf4/64) control detects, q_proj **SAME-BYTES** (nf4/64+dq) control detects
- frozen base `e4b/fused_attn4_shipped_lean1`: gate_up **SAME-BYTES** (nf4/64) control detects, down **SAME-BYTES** (nf4/64) control detects, q_proj **SAME-BYTES** (nf4/64+dq) control detects
- frozen base `e4b/fused_attn4_m_lean1`: gate_up **SAME-BYTES** (nf4/64) control detects, down **SAME-BYTES** (nf4/64) control detects, q_proj **SAME-BYTES** (nf4/64+dq) control detects
- frozen base `e4b/fused_attn4_m_lean1_d2`: gate_up **SAME-BYTES** (nf4/64) control detects, down **SAME-BYTES** (nf4/64) control detects, q_proj **SAME-BYTES** (nf4/64+dq) control detects
- frozen base `e4b/fused_attn4_m_lean0_d2`: gate_up **SAME-BYTES** (nf4/64) control detects, down **SAME-BYTES** (nf4/64) control detects, q_proj **SAME-BYTES** (nf4/64+dq) control detects
- frozen base `e4b/fused_attn4_shipped_lean1_d2`: gate_up **SAME-BYTES** (nf4/64) control detects, down **SAME-BYTES** (nf4/64) control detects, q_proj **SAME-BYTES** (nf4/64+dq) control detects
- frozen base `e4b/fused_attn4_shipped_lean0_d2`: gate_up **SAME-BYTES** (nf4/64) control detects, down **SAME-BYTES** (nf4/64) control detects, q_proj **SAME-BYTES** (nf4/64+dq) control detects

## Verdicts
| family | arm | VERDICT | validity | quality | note |
|---|---|---|---|---|---|
| qwen3leanab | e4b/fused_attn4_shipped_lean0 | **VALID** | VALID | -0.0384 |  |
| qwen3leanab | e4b/fused_attn4_shipped_lean1 | **VALID** | VALID | -0.0387 |  |
| qwen3leanab | e4b/fused_attn4_m_lean0 | **VALID** | VALID | 0.0000 |  |
| qwen3leanab | e4b/fused_attn4_m_lean1 | **VALID** | VALID | -0.0008 |  |
| qwen3leanab | e4b/fused_attn4_m_lean1_d2 | **VALID** | VALID | 0.0004 |  |
| qwen3leanab | e4b/fused_attn4_m_lean0_d2 | **VALID** | VALID | 0.0018 |  |
| qwen3leanab | e4b/fused_attn4_shipped_lean1_d2 | **VALID** | VALID | -0.0371 |  |
| qwen3leanab | e4b/fused_attn4_shipped_lean0_d2 | **VALID** | VALID | -0.0337 |  |

## Predictions P20 / P21 (TC1-PREREG amendment 13, #945: gnf4's trimmed padded LoRA delta vs its previous body, two stable draws a side; scored mechanically)
| prediction | family | verdict | evidence |
|---|---|---|---|
| P20 | qwen3leanab | **HELD** | shipped: lean1 / lean0 0.939 [0.935, 0.944 over 4 cross-draw ratios] vs [0.9, 0.99]; s/step lean0 2.371 / 2.350 (within 0.9%), lean1 2.218 / 2.217 (within 0.0%); held-out at N lean0 0.8113 / lean1 0.8110; delta paths (lean1, process) {"grouped_mm": 0, "loop": 0, "padded": 16896} |
| P21 | qwen3leanab | **HELD** | matched: lean1 / lean0 0.911 [0.898, 0.925 over 4 cross-draw ratios] vs [0.9, 0.99]; s/step lean0 2.949 / 2.955 (within 0.2%), lean1 2.728 / 2.654 (within 2.7%); held-out at N lean0 0.8497 / lean1 0.8489; delta paths (lean1, process) {"grouped_mm": 0, "loop": 0, "padded": 16896} |
