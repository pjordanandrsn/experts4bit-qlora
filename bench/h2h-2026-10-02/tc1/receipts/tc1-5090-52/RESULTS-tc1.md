# TC1 — Qwen3-30B-A3B on one RTX 5090: e4b vs Unsloth vs HF+PEFT vs axolotl at matched work, init and adapter precision (tc1)
Rule (tc1/TC1-PREREG.md): status per attempt in the vocabulary OK / REFUSED / OOM / INSTALL_FAILED / LOAD_FAULT / HARNESS_ERROR / ALARM / NOT_RUN; VALID/VOID per tp4's predicates plus the matched-set predicates (R3: matched init complete, fp32 adapters, step-0 held-out within 0.005 of e4b/reference_attn4_m); VERDICT exactly one of VALID / VOID / QUALITY_FAIL / OOM / UNSUPPORTED / HARNESS_ERROR / ALARM / NOT_RUN (QUALITY_FAIL = held-out |Δ| at N > 0.05 vs e4b/fused_attn4_m); positions = s/step ratios other/e4b from the medians over both draws (stability |d1-d2|/mean ≤ 5 % e4b / 10 % others, else UNSTABLE), quoted only when both arms are VALID; equivalence vs e4b/reference_attn4_m EQUIVALENT ≤ 0.02 / COMPARABLE ≤ 0.05 / else DIVERGENT; frozen base SAME-BYTES / DIFFERENT / N-A per slot; predictions P1–P10 (+ P1b) scored HELD / FALSIFIED / UNTESTED. VOID never enters a ratio or an equivalence reading. Nothing is licensed; no cross-box number is divided into these.
`versions.txt`
```
e4b 0.43.0 @65d3fc179b37188a3f88b69d64899f5be21cb34f (GitHub main)
gnf4 0.35.0 @1e4129826901dcfaa5dc91efbdc0d62d4b27a9ee (GitHub main)
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
 "run_id": "tc1-5090-52",
 "instance_id": "54100315",
 "gpu": "NVIDIA GeForce RTX 5090",
 "driver": "580.95.05",
 "cpu": "AMD EPYC 7C13 64-Core Processor",
 "nproc": 256,
 "mem_total_kb": "858436212",
 "cgroup_memory_max": "377392988160",
 "disk_root": "overlay         320G   52M  320G   1% /",
 "hostname": "6b85fe562c49",
 "registered_gpu_class": "5090",
 "prereg": "tc1/TC1-PREREG.md",
 "local_box": null,
 "local_snapshot": null
}
```

### Qwen3-30B-A3B (amendment 21: whole-layer gradient checkpointing vs keeping the last n layers' MoE activations) (`qwen3keepab`, registered n_layers 48, attention census 192)
- model `Qwen/Qwen3-30B-A3B` @ `ad44e777bcd1`; tokens sha `bfc742f67e37`; N=20; fixture template alpaca seq 2048 micro-batch 2 × accum 4 lr 0.0002 r 16 α 16 optimizer adamw_8bit(lr=0.0002, weight_decay=0.001) schedule=linear warmup_steps=5 autocast False; e4b trainable 642514944; box_class RTX 5090 gpu NVIDIA GeForce RTX 5090
| framework | arm | status | validity | **VERDICT** | matched | init / dtype | step-0 class | N | s/step med(11+) | tok/s | peak GB | J/step | train first→last | held-out 0→final | quality Δ | engagement | trainable | regime | reason / why |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| e4b | fused_attn4_shipped_keep0 | **OK** | VALID | **VALID** | native | native / bfloat16,float32 | — | 20 | 3.083 | 432.7 | 24.576 | 828.7 | 2.0614→0.7978 | 1.9441→0.8105 | -0.0409 | patched 48 / kcalls 768 | 642514944 | 4-bit experts (e4b NF4) + NF4 attention |  |
| e4b | fused_attn4_shipped_keep1 | **OK** | VALID | **VALID** | native | native / bfloat16,float32 | — | 20 | 2.536 | 588.9 | 29.085 | 643.2 | 2.0614→0.7968 | 1.9441→0.8141 | -0.0374 | patched 48 / kcalls 512 | 642514944 | 4-bit experts (e4b NF4) + NF4 attention |  |
| e4b | fused_attn4_m_keep0 | **OK** | VALID | **VALID** | yes | matched:3407 (complete 12480/12480) / float32 | SAME-BYTES-CLASS (0.0000) | 20 | 3.908 | 387.4 | 27.120 | 992.6 | 2.0614→0.8337 | 1.9441→0.8515 | 0.0000 | patched 48 / kcalls 768 | 642514944 | 4-bit experts (e4b NF4) + NF4 attention |  |
| e4b | fused_attn4_m_keep1 | **OK** | VALID | **VALID** | yes | matched:3407 (complete 12480/12480) / float32 | SAME-BYTES-CLASS (0.0000) | 20 | 3.642 | 413.5 | 29.401 | 922.9 | 2.0614→0.8322 | 1.9441→0.8504 | -0.0010 | patched 48 / kcalls 640 | 642514944 | 4-bit experts (e4b NF4) + NF4 attention |  |
| e4b | fused_attn4_m_keep1_d2 | **OK** | VALID | **VALID** | yes | matched:3407 (complete 12480/12480) / float32 | SAME-BYTES-CLASS (0.0000) | 20 | 3.600 | 416.2 | 29.401 | 920.3 | 2.0614→0.8310 | 1.9441→0.8551 | 0.0037 | patched 48 / kcalls 640 | 642514944 | 4-bit experts (e4b NF4) + NF4 attention |  |
| e4b | fused_attn4_m_keep0_d2 | **OK** | VALID | **VALID** | yes | matched:3407 (complete 12480/12480) / float32 | SAME-BYTES-CLASS (0.0000) | 20 | 3.914 | 386.3 | 27.157 | 986.0 | 2.0614→0.8323 | 1.9441→0.8513 | -0.0002 | patched 48 / kcalls 768 | 642514944 | 4-bit experts (e4b NF4) + NF4 attention |  |
| e4b | fused_attn4_shipped_keep1_d2 | **OK** | VALID | **VALID** | native | native / bfloat16,float32 | — | 20 | 2.536 | 580.9 | 29.085 | 651.7 | 2.0614→0.7975 | 1.9441→0.8173 | -0.0342 | patched 48 / kcalls 512 | 642514944 | 4-bit experts (e4b NF4) + NF4 attention |  |
| e4b | fused_attn4_shipped_keep0_d2 | **OK** | VALID | **VALID** | native | native / bfloat16,float32 | — | 20 | 2.989 | 500.4 | 24.576 | 752.9 | 2.0614→0.7972 | 1.9441→0.8134 | -0.0381 | patched 48 / kcalls 768 | 642514944 | 4-bit experts (e4b NF4) + NF4 attention |  |
- prologue `e4b/fused_attn4_shipped_keep0` **106.7 s** before step 1 (60% of the arm): c1_before 50.9, load_weights 27.5, c1_after 25.0, eval0 10.9; unattributed 1.244; budget 1260.0
- prologue `e4b/fused_attn4_shipped_keep1` **101.1 s** before step 1 (66% of the arm): c1_before 50.4, load_weights 30.7, c1_after 25.5, attn4 5.6; unattributed 1.246; budget 1260.0
- prologue `e4b/fused_attn4_m_keep0` **99.0 s** before step 1 (55% of the arm): c1_before 51.1, load_weights 27.5, c1_after 25.1, attn4 5.9; unattributed 1.237; budget 1260.0
- prologue `e4b/fused_attn4_m_keep1` **100.5 s** before step 1 (57% of the arm): c1_before 52.4, load_weights 27.6, c1_after 25.2, attn4 5.9; unattributed 1.257; budget 1202.9
- prologue `e4b/fused_attn4_m_keep1_d2` **98.4 s** before step 1 (57% of the arm): c1_before 51.5, load_weights 26.9, c1_after 25.3, attn4 5.7; unattributed 1.265; budget 1129.1
- prologue `e4b/fused_attn4_m_keep0_d2` **98.6 s** before step 1 (55% of the arm): c1_before 51.8, c1_after 27.4, load_weights 26.8, attn4 5.9; unattributed 1.223; budget 1056.3
- prologue `e4b/fused_attn4_shipped_keep1_d2` **104.0 s** before step 1 (66% of the arm): c1_before 56.2, load_weights 27.5, c1_after 25.3, attn4 5.7; unattributed 1.241; budget 980.3
- prologue `e4b/fused_attn4_shipped_keep0_d2` **102.1 s** before step 1 (62% of the arm): c1_before 53.7, load_weights 28.2, c1_after 25.2, attn4 6.1; unattributed 1.267; budget 913.8
- draws (R1): `e4b/fused_attn4_shipped_keep0` STABLE (3.083/2.989 s, |Δ|/mean 3.1% vs 5%); `e4b/fused_attn4_shipped_keep1` STABLE (2.536/2.536 s, |Δ|/mean 0.0% vs 5%); `e4b/fused_attn4_m_keep0` STABLE (3.908/3.914 s, |Δ|/mean 0.2% vs 5%); `e4b/fused_attn4_m_keep1` STABLE (3.642/3.600 s, |Δ|/mean 1.2% vs 5%)
- e4b internal parity: NO-ARM
- **NO MATCHED POSITION QUOTED (unsloth)** — not quoted: e4b STABLE / unsloth no receipt
- **NO MATCHED POSITION QUOTED (hf)** — not quoted: e4b STABLE / hf no receipt
- **NO MATCHED POSITION QUOTED (axolotl)** — not quoted: e4b STABLE / axolotl no receipt
- equivalence vs `e4b/fused_attn4_m` (I/H; EQUIVALENT iff both ≤ band = max(0.005, 3 × fused-vs-reference |Δ|) = none (no reference arm: COMPARABLE is the ceiling); COMPARABLE ≤ 0.05; draw-noise floor 0.0047): `e4b/fused_attn4_m_keep1` **COMPARABLE** (median step |Δ| 0.0020, |Δ held-out at N| 0.0010, step-0 0.0000 SAME-BYTES-CLASS, |Δ loss at step 2| 0.0048, paired rows mean -0.0010 ± 0.0027 SE over 8, favouring arm 5 / anchor 3) — no reference arm ran: the EQUIVALENT band cannot be set, COMPARABLE is the ceiling; `e4b/fused_attn4_m_keep1_d2` **COMPARABLE** (median step |Δ| 0.0022, |Δ held-out at N| 0.0037, step-0 0.0000 SAME-BYTES-CLASS, |Δ loss at step 2| 0.0026, paired rows mean +0.0037 ± 0.0039 SE over 8, favouring arm 2 / anchor 6) — no reference arm ran: the EQUIVALENT band cannot be set, COMPARABLE is the ceiling; `e4b/fused_attn4_m_keep0_d2` **COMPARABLE** (median step |Δ| 0.0021, |Δ held-out at N| 0.0002, step-0 0.0000 SAME-BYTES-CLASS, |Δ loss at step 2| 0.0083, paired rows mean -0.0002 ± 0.0024 SE over 8, favouring arm 6 / anchor 2) — no reference arm ran: the EQUIVALENT band cannot be set, COMPARABLE is the ceiling
- matched_init_sha (B, name-free, canonical slot order): anchor `f7832488926eda91`; `e4b/fused_attn4_m_keep0` same; `e4b/fused_attn4_m_keep1` same; `e4b/fused_attn4_m_keep1_d2` same; `e4b/fused_attn4_m_keep0_d2` same
- frozen base (R5), anchor `e4b/fused_attn4_m`: gate_up nf4/64 sha 8c3b3fd78e89 control detects, down nf4/64 sha 59569d810a1a control detects, q_proj nf4/64+dq sha 5c5e402ce51d control detects
- frozen base `e4b/fused_attn4_shipped_keep0`: gate_up **SAME-BYTES** (nf4/64) control detects, down **SAME-BYTES** (nf4/64) control detects, q_proj **SAME-BYTES** (nf4/64+dq) control detects
- frozen base `e4b/fused_attn4_shipped_keep1`: gate_up **SAME-BYTES** (nf4/64) control detects, down **SAME-BYTES** (nf4/64) control detects, q_proj **SAME-BYTES** (nf4/64+dq) control detects
- frozen base `e4b/fused_attn4_m_keep1`: gate_up **SAME-BYTES** (nf4/64) control detects, down **SAME-BYTES** (nf4/64) control detects, q_proj **SAME-BYTES** (nf4/64+dq) control detects
- frozen base `e4b/fused_attn4_m_keep1_d2`: gate_up **SAME-BYTES** (nf4/64) control detects, down **SAME-BYTES** (nf4/64) control detects, q_proj **SAME-BYTES** (nf4/64+dq) control detects
- frozen base `e4b/fused_attn4_m_keep0_d2`: gate_up **SAME-BYTES** (nf4/64) control detects, down **SAME-BYTES** (nf4/64) control detects, q_proj **SAME-BYTES** (nf4/64+dq) control detects
- frozen base `e4b/fused_attn4_shipped_keep1_d2`: gate_up **SAME-BYTES** (nf4/64) control detects, down **SAME-BYTES** (nf4/64) control detects, q_proj **SAME-BYTES** (nf4/64+dq) control detects
- frozen base `e4b/fused_attn4_shipped_keep0_d2`: gate_up **SAME-BYTES** (nf4/64) control detects, down **SAME-BYTES** (nf4/64) control detects, q_proj **SAME-BYTES** (nf4/64+dq) control detects

## Verdicts
| family | arm | VERDICT | validity | quality | note |
|---|---|---|---|---|---|
| qwen3keepab | e4b/fused_attn4_shipped_keep0 | **VALID** | VALID | -0.0409 |  |
| qwen3keepab | e4b/fused_attn4_shipped_keep1 | **VALID** | VALID | -0.0374 |  |
| qwen3keepab | e4b/fused_attn4_m_keep0 | **VALID** | VALID | 0.0000 |  |
| qwen3keepab | e4b/fused_attn4_m_keep1 | **VALID** | VALID | -0.0010 |  |
| qwen3keepab | e4b/fused_attn4_m_keep1_d2 | **VALID** | VALID | 0.0037 |  |
| qwen3keepab | e4b/fused_attn4_m_keep0_d2 | **VALID** | VALID | -0.0002 |  |
| qwen3keepab | e4b/fused_attn4_shipped_keep1_d2 | **VALID** | VALID | -0.0342 |  |
| qwen3keepab | e4b/fused_attn4_shipped_keep0_d2 | **VALID** | VALID | -0.0381 |  |

## Predictions P35 / P36 / P37 (TC1-PREREG amendment 21, #945: whole-layer checkpointing vs keeping MoE activations, two stable draws a side; scored mechanically)
| prediction | family | verdict | evidence |
|---|---|---|---|
| P35 | qwen3keepab | **HELD** | shipped (32 of 48 layers kept): keep1 / keep0 0.835 [0.823, 0.849 over 4 cross-draw ratios] vs [0.8, 0.95]; s/step keep0 3.083 / 2.989 (within 3.1%), keep1 2.536 / 2.536 (within 0.0%); peak keep0 24.58 / keep1 29.09 GB |
| P36 | qwen3keepab | **HELD** | matched (16 of 48 layers kept): keep1 / keep0 0.926 [0.920, 0.932 over 4 cross-draw ratios] vs [0.86, 0.98]; s/step keep0 3.908 / 3.914 (within 0.2%), keep1 3.642 / 3.600 (within 1.2%); peak keep0 27.14 / keep1 29.40 GB |
| P37 | qwen3keepab | **HELD** | shipped: keep1 peak 29.09 GB (<= 31.0), mean held-out keep1 - keep0 +0.0037 (|.| <= 0.005); peak keep0 24.58 / keep1 29.09 GB; held-out at N keep0 [0.8105, 0.8134] keep1 [0.8141, 0.8173]; matched: keep1 peak 29.40 GB (<= 31.0), mean held-out keep1 - keep0 +0.0014 (|.| <= 0.005); peak keep0 27.14 / keep1 29.40 GB; held-out at N keep0 [0.8515, 0.8513] keep1 [0.8504, 0.8551] |
