# TC1 — Qwen3-30B-A3B on one RTX 5090: e4b vs Unsloth vs HF+PEFT vs axolotl at matched work, init and adapter precision (tc1)
Rule (tc1/TC1-PREREG.md): status per attempt in the vocabulary OK / REFUSED / OOM / INSTALL_FAILED / LOAD_FAULT / HARNESS_ERROR / ALARM / NOT_RUN; VALID/VOID per tp4's predicates plus the matched-set predicates (R3: matched init complete, fp32 adapters, step-0 held-out within 0.005 of e4b/reference_attn4_m); VERDICT exactly one of VALID / VOID / QUALITY_FAIL / OOM / UNSUPPORTED / HARNESS_ERROR / ALARM / NOT_RUN (QUALITY_FAIL = held-out |Δ| at N > 0.05 vs e4b/fused_attn4_m); positions = s/step ratios other/e4b from the medians over both draws (stability |d1-d2|/mean ≤ 5 % e4b / 10 % others, else UNSTABLE), quoted only when both arms are VALID; equivalence vs e4b/reference_attn4_m EQUIVALENT ≤ 0.02 / COMPARABLE ≤ 0.05 / else DIVERGENT; frozen base SAME-BYTES / DIFFERENT / N-A per slot; predictions P1–P10 (+ P1b) scored HELD / FALSIFIED / UNTESTED. VOID never enters a ratio or an equivalence reading. Nothing is licensed; no cross-box number is divided into these.
`versions.txt`
```
e4b 0.41.0 @4dd72be133c014bb1086f817a2495e4fbec8d8a0 (GitHub main)
gnf4 0.34.1 @b670e611314b082592c4107048433bd3ed48a82e (GitHub main)
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
 "run_id": "tc1-5090-38",
 "instance_id": "53991731",
 "gpu": "NVIDIA GeForce RTX 5090",
 "driver": "580.95.05",
 "cpu": "AMD EPYC 7663 56-Core Processor",
 "nproc": 224,
 "mem_total_kb": "792385364",
 "cgroup_memory_max": "294412877824",
 "disk_root": "overlay         320G   55M  320G   1% /",
 "hostname": "82191bef7c39",
 "registered_gpu_class": "5090",
 "prereg": "tc1/TC1-PREREG.md",
 "local_box": null,
 "local_snapshot": null
}
```

### Qwen3-30B-A3B (amendment 10: e4b legacy grouping + pageable copies vs single-read grouping + pinned ring, #945) (`qwen3syncab`, registered n_layers 48, attention census 192)
- model `Qwen/Qwen3-30B-A3B` @ `ad44e777bcd1`; tokens sha `bfc742f67e37`; N=20; fixture template alpaca seq 2048 micro-batch 2 × accum 4 lr 0.0002 r 16 α 16 optimizer adamw_8bit(lr=0.0002, weight_decay=0.001) schedule=linear warmup_steps=5 autocast False; e4b trainable 642514944; box_class RTX 5090 gpu NVIDIA GeForce RTX 5090
| framework | arm | status | validity | **VERDICT** | matched | init / dtype | step-0 class | N | s/step med(11+) | tok/s | peak GB | J/step | train first→last | held-out 0→final | quality Δ | engagement | trainable | regime | reason / why |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| e4b | fused_attn4_shipped_legacy | **OK** | VALID | **VALID** | native | native / bfloat16,float32 | — | 20 | 4.110 | 362.1 | 24.581 | 1062.5 | 2.0705→0.7996 | 1.9505→0.8131 | -0.0374 | patched 48 / kcalls 768 | 642514944 | 4-bit experts (e4b NF4) + NF4 attention |  |
| e4b | fused_attn4_shipped_sync1 | **OK** | VALID | **VALID** | native | native / bfloat16,float32 | — | 20 | 3.508 | 439.2 | 24.581 | 969.5 | 2.0705→0.7964 | 1.9505→0.8133 | -0.0372 | patched 48 / kcalls 768 | 642514944 | 4-bit experts (e4b NF4) + NF4 attention |  |
| e4b | fused_attn4_m_legacy | **OK** | VALID | **VALID** | yes | matched:3407 (complete 12480/12480) / float32 | SAME-BYTES-CLASS (0.0000) | 20 | 5.267 | 293.5 | 27.822 | 1322.4 | 2.0705→0.8314 | 1.9505→0.8505 | 0.0000 | patched 48 / kcalls 768 | 642514944 | 4-bit experts (e4b NF4) + NF4 attention |  |
| e4b | fused_attn4_m_sync1 | **OK** | VALID | **VALID** | yes | matched:3407 (complete 12480/12480) / float32 | SAME-BYTES-CLASS (0.0000) | 20 | 4.426 | 348.9 | 27.822 | 1261.4 | 2.0705→0.8309 | 1.9505→0.8506 | 0.0001 | patched 48 / kcalls 768 | 642514944 | 4-bit experts (e4b NF4) + NF4 attention |  |
| e4b | fused_attn4_m_sync1_d2 | **OK** | VALID | **VALID** | yes | matched:3407 (complete 12480/12480) / float32 | SAME-BYTES-CLASS (0.0000) | 20 | 4.453 | 348.2 | 27.849 | 1260.4 | 2.0705→0.8331 | 1.9505→0.8535 | 0.0030 | patched 48 / kcalls 768 | 642514944 | 4-bit experts (e4b NF4) + NF4 attention |  |
| e4b | fused_attn4_m_legacy_d2 | **OK** | VALID | **VALID** | yes | matched:3407 (complete 12480/12480) / float32 | SAME-BYTES-CLASS (0.0000) | 20 | 5.213 | 293.0 | 27.849 | 1313.9 | 2.0705→0.8321 | 1.9505→0.8529 | 0.0024 | patched 48 / kcalls 768 | 642514944 | 4-bit experts (e4b NF4) + NF4 attention |  |
| e4b | fused_attn4_shipped_sync1_d2 | **OK** | VALID | **VALID** | native | native / bfloat16,float32 | — | 20 | 3.654 | 414.3 | 24.581 | 999.5 | 2.0705→0.7972 | 1.9505→0.8153 | -0.0352 | patched 48 / kcalls 768 | 642514944 | 4-bit experts (e4b NF4) + NF4 attention |  |
| e4b | fused_attn4_shipped_legacy_d2 | **OK** | VALID | **VALID** | native | native / bfloat16,float32 | — | 20 | 4.162 | 370.5 | 24.581 | 1028.2 | 2.0705→0.7997 | 1.9505→0.8122 | -0.0383 | patched 48 / kcalls 768 | 642514944 | 4-bit experts (e4b NF4) + NF4 attention |  |
- prologue `e4b/fused_attn4_shipped_legacy` **119.3 s** before step 1 (58% of the arm): c1_before 58.5, c1_after 28.6, load_weights 27.7, eval0 14.8; unattributed 1.238; budget 1260.0
- prologue `e4b/fused_attn4_shipped_sync1` **112.2 s** before step 1 (61% of the arm): c1_before 58.3, load_weights 33.1, c1_after 29.9, attn4 5.6; unattributed 1.247; budget 1260.0
- prologue `e4b/fused_attn4_m_legacy` **108.8 s** before step 1 (51% of the arm): c1_before 58.1, load_weights 30.8, c1_after 30.5, attn4 5.8; unattributed 1.274; budget 1260.0
- prologue `e4b/fused_attn4_m_sync1` **107.3 s** before step 1 (55% of the arm): c1_before 58.7, c1_after 29.2, load_weights 28.6, attn4 6.0; unattributed 1.301; budget 1260.0
- prologue `e4b/fused_attn4_m_sync1_d2` **110.0 s** before step 1 (55% of the arm): c1_before 60.6, c1_after 29.2, load_weights 28.9, attn4 6.0; unattributed 1.297; budget 1260.0
- prologue `e4b/fused_attn4_m_legacy_d2` **106.3 s** before step 1 (50% of the arm): c1_before 57.8, c1_after 30.8, load_weights 28.1, attn4 6.0; unattributed 1.215; budget 1260.0
- prologue `e4b/fused_attn4_shipped_sync1_d2` **114.3 s** before step 1 (60% of the arm): c1_before 65.6, c1_after 33.9, load_weights 27.1, attn4 5.9; unattributed 1.403; budget 1260.0
- prologue `e4b/fused_attn4_shipped_legacy_d2` **115.0 s** before step 1 (58% of the arm): c1_before 60.8, load_weights 32.5, c1_after 31.2, attn4 6.0; unattributed 1.255; budget 1260.0
- draws (R1): `e4b/fused_attn4_shipped_legacy` STABLE (4.110/4.162 s, |Δ|/mean 1.3% vs 5%); `e4b/fused_attn4_shipped_sync1` STABLE (3.508/3.654 s, |Δ|/mean 4.1% vs 5%); `e4b/fused_attn4_m_legacy` STABLE (5.267/5.213 s, |Δ|/mean 1.0% vs 5%); `e4b/fused_attn4_m_sync1` STABLE (4.426/4.453 s, |Δ|/mean 0.6% vs 5%)
- e4b internal parity: NO-ARM
- **NO MATCHED POSITION QUOTED (unsloth)** — not quoted: e4b STABLE / unsloth no receipt
- **NO MATCHED POSITION QUOTED (hf)** — not quoted: e4b STABLE / hf no receipt
- **NO MATCHED POSITION QUOTED (axolotl)** — not quoted: e4b STABLE / axolotl no receipt
- equivalence vs `e4b/fused_attn4_m` (I/H; EQUIVALENT iff both ≤ band = max(0.005, 3 × fused-vs-reference |Δ|) = none (no reference arm: COMPARABLE is the ceiling); COMPARABLE ≤ 0.05; draw-noise floor 0.0029): `e4b/fused_attn4_m_sync1` **COMPARABLE** (median step |Δ| 0.0023, |Δ held-out at N| 0.0001, step-0 0.0000 SAME-BYTES-CLASS, |Δ loss at step 2| 0.0143, paired rows mean +0.0001 ± 0.0017 SE over 8, favouring arm 4 / anchor 4) — no reference arm ran: the EQUIVALENT band cannot be set, COMPARABLE is the ceiling; `e4b/fused_attn4_m_sync1_d2` **COMPARABLE** (median step |Δ| 0.0019, |Δ held-out at N| 0.0030, step-0 0.0000 SAME-BYTES-CLASS, |Δ loss at step 2| 0.0009, paired rows mean +0.0030 ± 0.0021 SE over 8, favouring arm 2 / anchor 6) — no reference arm ran: the EQUIVALENT band cannot be set, COMPARABLE is the ceiling; `e4b/fused_attn4_m_legacy_d2` **COMPARABLE** (median step |Δ| 0.0022, |Δ held-out at N| 0.0024, step-0 0.0000 SAME-BYTES-CLASS, |Δ loss at step 2| 0.0066, paired rows mean +0.0024 ± 0.0019 SE over 8, favouring arm 3 / anchor 5) — no reference arm ran: the EQUIVALENT band cannot be set, COMPARABLE is the ceiling
- matched_init_sha (B, name-free, canonical slot order): anchor `f7832488926eda91`; `e4b/fused_attn4_m_legacy` same; `e4b/fused_attn4_m_sync1` same; `e4b/fused_attn4_m_sync1_d2` same; `e4b/fused_attn4_m_legacy_d2` same
- frozen base (R5), anchor `e4b/fused_attn4_m`: gate_up nf4/64 sha 8c3b3fd78e89 control detects, down nf4/64 sha 59569d810a1a control detects, q_proj nf4/64+dq sha 5c5e402ce51d control detects
- frozen base `e4b/fused_attn4_shipped_legacy`: gate_up **SAME-BYTES** (nf4/64) control detects, down **SAME-BYTES** (nf4/64) control detects, q_proj **SAME-BYTES** (nf4/64+dq) control detects
- frozen base `e4b/fused_attn4_shipped_sync1`: gate_up **SAME-BYTES** (nf4/64) control detects, down **SAME-BYTES** (nf4/64) control detects, q_proj **SAME-BYTES** (nf4/64+dq) control detects
- frozen base `e4b/fused_attn4_m_sync1`: gate_up **SAME-BYTES** (nf4/64) control detects, down **SAME-BYTES** (nf4/64) control detects, q_proj **SAME-BYTES** (nf4/64+dq) control detects
- frozen base `e4b/fused_attn4_m_sync1_d2`: gate_up **SAME-BYTES** (nf4/64) control detects, down **SAME-BYTES** (nf4/64) control detects, q_proj **SAME-BYTES** (nf4/64+dq) control detects
- frozen base `e4b/fused_attn4_m_legacy_d2`: gate_up **SAME-BYTES** (nf4/64) control detects, down **SAME-BYTES** (nf4/64) control detects, q_proj **SAME-BYTES** (nf4/64+dq) control detects
- frozen base `e4b/fused_attn4_shipped_sync1_d2`: gate_up **SAME-BYTES** (nf4/64) control detects, down **SAME-BYTES** (nf4/64) control detects, q_proj **SAME-BYTES** (nf4/64+dq) control detects
- frozen base `e4b/fused_attn4_shipped_legacy_d2`: gate_up **SAME-BYTES** (nf4/64) control detects, down **SAME-BYTES** (nf4/64) control detects, q_proj **SAME-BYTES** (nf4/64+dq) control detects

## Verdicts
| family | arm | VERDICT | validity | quality | note |
|---|---|---|---|---|---|
| qwen3syncab | e4b/fused_attn4_shipped_legacy | **VALID** | VALID | -0.0374 |  |
| qwen3syncab | e4b/fused_attn4_shipped_sync1 | **VALID** | VALID | -0.0372 |  |
| qwen3syncab | e4b/fused_attn4_m_legacy | **VALID** | VALID | 0.0000 |  |
| qwen3syncab | e4b/fused_attn4_m_sync1 | **VALID** | VALID | 0.0001 |  |
| qwen3syncab | e4b/fused_attn4_m_sync1_d2 | **VALID** | VALID | 0.0030 |  |
| qwen3syncab | e4b/fused_attn4_m_legacy_d2 | **VALID** | VALID | 0.0024 |  |
| qwen3syncab | e4b/fused_attn4_shipped_sync1_d2 | **VALID** | VALID | -0.0352 |  |
| qwen3syncab | e4b/fused_attn4_shipped_legacy_d2 | **VALID** | VALID | -0.0383 |  |

## Predictions P16 / P17 (TC1-PREREG amendment 10, #945: single-read grouping + pinned ring vs legacy, two stable draws a side; scored mechanically)
| prediction | family | verdict | evidence |
|---|---|---|---|
| P16 | qwen3syncab | **HELD** | shipped: sync1 / legacy 0.866 [0.843, 0.889 over 4 cross-draw ratios] vs [0.75, 0.95]; s/step legacy 4.110 / 4.162 (within 1.3%), sync1 3.508 / 3.654 (within 4.1%); held-out at N legacy 0.8131 / sync1 0.8133 |
| P17 | qwen3syncab | **HELD** | matched: sync1 / legacy 0.847 [0.840, 0.854 over 4 cross-draw ratios] vs [0.75, 0.95]; s/step legacy 5.267 / 5.213 (within 1.0%), sync1 4.426 / 4.453 (within 0.6%); held-out at N legacy 0.8505 / sync1 0.8506 |
