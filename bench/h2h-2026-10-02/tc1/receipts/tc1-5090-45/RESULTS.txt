# TC1 — Qwen3-30B-A3B on one RTX 5090: e4b vs Unsloth vs HF+PEFT vs axolotl at matched work, init and adapter precision (tc1)
Rule (tc1/TC1-PREREG.md): status per attempt in the vocabulary OK / REFUSED / OOM / INSTALL_FAILED / LOAD_FAULT / HARNESS_ERROR / ALARM / NOT_RUN; VALID/VOID per tp4's predicates plus the matched-set predicates (R3: matched init complete, fp32 adapters, step-0 held-out within 0.005 of e4b/reference_attn4_m); VERDICT exactly one of VALID / VOID / QUALITY_FAIL / OOM / UNSUPPORTED / HARNESS_ERROR / ALARM / NOT_RUN (QUALITY_FAIL = held-out |Δ| at N > 0.05 vs e4b/fused_attn4_m); positions = s/step ratios other/e4b from the medians over both draws (stability |d1-d2|/mean ≤ 5 % e4b / 10 % others, else UNSTABLE), quoted only when both arms are VALID; equivalence vs e4b/reference_attn4_m EQUIVALENT ≤ 0.02 / COMPARABLE ≤ 0.05 / else DIVERGENT; frozen base SAME-BYTES / DIFFERENT / N-A per slot; predictions P1–P10 (+ P1b) scored HELD / FALSIFIED / UNTESTED. VOID never enters a ratio or an equivalence reading. Nothing is licensed; no cross-box number is divided into these.
`versions.txt`
```
e4b 0.41.0 @aba67eab3e7a239f0be29c71edc5c613866d3992 (GitHub main)
gnf4 0.34.1 @00929a493f8ca6ef60c950d666eb5fa5cee38df6 (GitHub main)
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
 "run_id": "tc1-5090-45",
 "instance_id": "54018669",
 "gpu": "NVIDIA GeForce RTX 5090",
 "driver": "580.95.05",
 "cpu": "AMD EPYC 7663 56-Core Processor",
 "nproc": 224,
 "mem_total_kb": "792385364",
 "cgroup_memory_max": "294412877824",
 "disk_root": "overlay         320G   55M  320G   1% /",
 "hostname": "1ec946a721a1",
 "registered_gpu_class": "5090",
 "prereg": "tc1/TC1-PREREG.md",
 "local_box": null,
 "local_snapshot": null
}
```

### Qwen3-30B-A3B (amendment 15: the Hugging Face RMSNorm composite vs e4b's fused training RMSNorm, trimmed delta, post-#945 sync path) (`qwen3rmsab`, registered n_layers 48, attention census 192)
- model `Qwen/Qwen3-30B-A3B` @ `ad44e777bcd1`; tokens sha `bfc742f67e37`; N=20; fixture template alpaca seq 2048 micro-batch 2 × accum 4 lr 0.0002 r 16 α 16 optimizer adamw_8bit(lr=0.0002, weight_decay=0.001) schedule=linear warmup_steps=5 autocast False; e4b trainable 642514944; box_class RTX 5090 gpu NVIDIA GeForce RTX 5090
| framework | arm | status | validity | **VERDICT** | matched | init / dtype | step-0 class | N | s/step med(11+) | tok/s | peak GB | J/step | train first→last | held-out 0→final | quality Δ | engagement | trainable | regime | reason / why |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| e4b | fused_attn4_shipped_rms0 | **OK** | VALID | **VALID** | native | native / bfloat16,float32 | — | 20 | 3.537 | 385.3 | 24.581 | 932.7 | 2.0705→0.7990 | 1.9505→0.8143 | -0.0391 | patched 48 / kcalls 768 | 642514944 | 4-bit experts (e4b NF4) + NF4 attention |  |
| e4b | fused_attn4_shipped_rms1 | **OK** | VALID | **VALID** | native | native / bfloat16,float32 | — | 20 | 3.234 | 465.8 | 24.576 | 833.9 | 2.0614→0.7982 | 1.9441→0.8136 | -0.0398 | patched 48 / kcalls 768 | 642514944 | 4-bit experts (e4b NF4) + NF4 attention |  |
| e4b | fused_attn4_m_rms0 | **OK** | VALID | **VALID** | yes | matched:3407 (complete 12480/12480) / float32 | SAME-BYTES-CLASS (0.0000) | 20 | 4.258 | 356.2 | 27.239 | 1089.2 | 2.0705→0.8324 | 1.9505→0.8534 | 0.0000 | patched 48 / kcalls 768 | 642514944 | 4-bit experts (e4b NF4) + NF4 attention |  |
| e4b | fused_attn4_m_rms1 | **OK** | VALID | **VALID** | yes | matched:3407 (complete 12480/12480) / float32 | SAME-BYTES-CLASS (0.0064) | 20 | 4.110 | 364.3 | 27.210 | 1060.4 | 2.0614→0.8308 | 1.9441→0.8482 | -0.0052 | patched 48 / kcalls 768 | 642514944 | 4-bit experts (e4b NF4) + NF4 attention |  |
| e4b | fused_attn4_m_rms1_d2 | **OK** | VALID | **VALID** | yes | matched:3407 (complete 12480/12480) / float32 | SAME-BYTES-CLASS (0.0064) | 20 | 4.192 | 359.2 | 27.230 | 1067.4 | 2.0614→0.8375 | 1.9441→0.8499 | -0.0036 | patched 48 / kcalls 768 | 642514944 | 4-bit experts (e4b NF4) + NF4 attention |  |
| e4b | fused_attn4_m_rms0_d2 | **OK** | VALID | **VALID** | yes | matched:3407 (complete 12480/12480) / float32 | SAME-BYTES-CLASS (0.0000) | 20 | 4.396 | 345.0 | 27.258 | 1099.7 | 2.0705→0.8308 | 1.9505→0.8477 | -0.0058 | patched 48 / kcalls 768 | 642514944 | 4-bit experts (e4b NF4) + NF4 attention |  |
| e4b | fused_attn4_shipped_rms1_d2 | **OK** | VALID | **VALID** | native | native / bfloat16,float32 | — | 20 | 3.159 | 461.1 | 24.576 | 836.6 | 2.0614→0.7973 | 1.9441→0.8103 | -0.0431 | patched 48 / kcalls 768 | 642514944 | 4-bit experts (e4b NF4) + NF4 attention |  |
| e4b | fused_attn4_shipped_rms0_d2 | **OK** | VALID | **VALID** | native | native / bfloat16,float32 | — | 20 | 3.386 | 447.7 | 24.581 | 871.5 | 2.0705→0.7989 | 1.9505→0.8139 | -0.0396 | patched 48 / kcalls 768 | 642514944 | 4-bit experts (e4b NF4) + NF4 attention |  |
- prologue `e4b/fused_attn4_shipped_rms0` **116.3 s** before step 1 (59% of the arm): c1_before 58.4, c1_after 30.7, load_weights 27.9, eval0 11.0; unattributed 1.274; budget 1260.0
- prologue `e4b/fused_attn4_shipped_rms1` **112.8 s** before step 1 (63% of the arm): c1_before 59.5, c1_after 31.2, load_weights 30.7, attn4 6.3; unattributed 1.235; budget 1228.1
- prologue `e4b/fused_attn4_m_rms0` **110.2 s** before step 1 (56% of the arm): c1_before 60.9, c1_after 31.3, load_weights 27.4, attn4 6.3; unattributed 1.291; budget 1151.5
- prologue `e4b/fused_attn4_m_rms1` **105.2 s** before step 1 (55% of the arm): c1_before 57.3, c1_after 29.4, load_weights 27.5, attn4 5.6; unattributed 1.299; budget 1067.8
- prologue `e4b/fused_attn4_m_rms1_d2` **106.1 s** before step 1 (55% of the arm): c1_before 58.6, c1_after 30.9, load_weights 27.4, attn4 5.8; unattributed 1.232; budget 987.3
- prologue `e4b/fused_attn4_m_rms0_d2` **121.1 s** before step 1 (57% of the arm): c1_before 67.3, load_weights 33.2, c1_after 31.6, attn4 6.0; unattributed 1.385; budget 905.4
- prologue `e4b/fused_attn4_shipped_rms1_d2` **110.9 s** before step 1 (62% of the arm): c1_before 61.5, c1_after 29.5, load_weights 27.6, attn4 5.8; unattributed 1.282; budget 816.5
- prologue `e4b/fused_attn4_shipped_rms0_d2` **110.8 s** before step 1 (61% of the arm): c1_before 61.5, c1_after 29.5, load_weights 27.9, attn4 6.2; unattributed 1.284; budget 740.9
- draws (R1): `e4b/fused_attn4_shipped_rms0` STABLE (3.537/3.386 s, |Δ|/mean 4.3% vs 5%); `e4b/fused_attn4_shipped_rms1` STABLE (3.234/3.159 s, |Δ|/mean 2.3% vs 5%); `e4b/fused_attn4_m_rms0` STABLE (4.258/4.396 s, |Δ|/mean 3.2% vs 5%); `e4b/fused_attn4_m_rms1` STABLE (4.110/4.192 s, |Δ|/mean 2.0% vs 5%)
- e4b internal parity: NO-ARM
- **NO MATCHED POSITION QUOTED (unsloth)** — not quoted: e4b STABLE / unsloth no receipt
- **NO MATCHED POSITION QUOTED (hf)** — not quoted: e4b STABLE / hf no receipt
- **NO MATCHED POSITION QUOTED (axolotl)** — not quoted: e4b STABLE / axolotl no receipt
- equivalence vs `e4b/fused_attn4_m` (I/H; EQUIVALENT iff both ≤ band = max(0.005, 3 × fused-vs-reference |Δ|) = none (no reference arm: COMPARABLE is the ceiling); COMPARABLE ≤ 0.05; draw-noise floor 0.0058): `e4b/fused_attn4_m_rms1` **COMPARABLE** (median step |Δ| 0.0018, |Δ held-out at N| 0.0052, step-0 0.0064 SAME-BYTES-CLASS, |Δ loss at step 2| 0.0062, paired rows mean -0.0053 ± 0.0020 SE over 8, favouring arm 6 / anchor 2) — no reference arm ran: the EQUIVALENT band cannot be set, COMPARABLE is the ceiling; `e4b/fused_attn4_m_rms1_d2` **COMPARABLE** (median step |Δ| 0.0020, |Δ held-out at N| 0.0036, step-0 0.0064 SAME-BYTES-CLASS, |Δ loss at step 2| 0.0087, paired rows mean -0.0036 ± 0.0024 SE over 8, favouring arm 4 / anchor 4) — no reference arm ran: the EQUIVALENT band cannot be set, COMPARABLE is the ceiling; `e4b/fused_attn4_m_rms0_d2` **COMPARABLE** (median step |Δ| 0.0012, |Δ held-out at N| 0.0058, step-0 0.0000 SAME-BYTES-CLASS, |Δ loss at step 2| 0.0082, paired rows mean -0.0058 ± 0.0015 SE over 8, favouring arm 8 / anchor 0) — no reference arm ran: the EQUIVALENT band cannot be set, COMPARABLE is the ceiling
- matched_init_sha (B, name-free, canonical slot order): anchor `f7832488926eda91`; `e4b/fused_attn4_m_rms0` same; `e4b/fused_attn4_m_rms1` same; `e4b/fused_attn4_m_rms1_d2` same; `e4b/fused_attn4_m_rms0_d2` same
- frozen base (R5), anchor `e4b/fused_attn4_m`: gate_up nf4/64 sha 8c3b3fd78e89 control detects, down nf4/64 sha 59569d810a1a control detects, q_proj nf4/64+dq sha 5c5e402ce51d control detects
- frozen base `e4b/fused_attn4_shipped_rms0`: gate_up **SAME-BYTES** (nf4/64) control detects, down **SAME-BYTES** (nf4/64) control detects, q_proj **SAME-BYTES** (nf4/64+dq) control detects
- frozen base `e4b/fused_attn4_shipped_rms1`: gate_up **SAME-BYTES** (nf4/64) control detects, down **SAME-BYTES** (nf4/64) control detects, q_proj **SAME-BYTES** (nf4/64+dq) control detects
- frozen base `e4b/fused_attn4_m_rms1`: gate_up **SAME-BYTES** (nf4/64) control detects, down **SAME-BYTES** (nf4/64) control detects, q_proj **SAME-BYTES** (nf4/64+dq) control detects
- frozen base `e4b/fused_attn4_m_rms1_d2`: gate_up **SAME-BYTES** (nf4/64) control detects, down **SAME-BYTES** (nf4/64) control detects, q_proj **SAME-BYTES** (nf4/64+dq) control detects
- frozen base `e4b/fused_attn4_m_rms0_d2`: gate_up **SAME-BYTES** (nf4/64) control detects, down **SAME-BYTES** (nf4/64) control detects, q_proj **SAME-BYTES** (nf4/64+dq) control detects
- frozen base `e4b/fused_attn4_shipped_rms1_d2`: gate_up **SAME-BYTES** (nf4/64) control detects, down **SAME-BYTES** (nf4/64) control detects, q_proj **SAME-BYTES** (nf4/64+dq) control detects
- frozen base `e4b/fused_attn4_shipped_rms0_d2`: gate_up **SAME-BYTES** (nf4/64) control detects, down **SAME-BYTES** (nf4/64) control detects, q_proj **SAME-BYTES** (nf4/64+dq) control detects

## Verdicts
| family | arm | VERDICT | validity | quality | note |
|---|---|---|---|---|---|
| qwen3rmsab | e4b/fused_attn4_shipped_rms0 | **VALID** | VALID | -0.0391 |  |
| qwen3rmsab | e4b/fused_attn4_shipped_rms1 | **VALID** | VALID | -0.0398 |  |
| qwen3rmsab | e4b/fused_attn4_m_rms0 | **VALID** | VALID | 0.0000 |  |
| qwen3rmsab | e4b/fused_attn4_m_rms1 | **VALID** | VALID | -0.0052 |  |
| qwen3rmsab | e4b/fused_attn4_m_rms1_d2 | **VALID** | VALID | -0.0036 |  |
| qwen3rmsab | e4b/fused_attn4_m_rms0_d2 | **VALID** | VALID | -0.0058 |  |
| qwen3rmsab | e4b/fused_attn4_shipped_rms1_d2 | **VALID** | VALID | -0.0431 |  |
| qwen3rmsab | e4b/fused_attn4_shipped_rms0_d2 | **VALID** | VALID | -0.0396 |  |

## Predictions P24 / P25 / P26 (TC1-PREREG amendment 15, #945: e4b's fused training RMSNorm vs the composite, two stable draws a side; scored mechanically)
| prediction | family | verdict | evidence |
|---|---|---|---|
| P24 | qwen3rmsab | **HELD** | shipped: rms1 / rms0 0.924 [0.893, 0.955 over 4 cross-draw ratios] vs [0.85, 0.97]; s/step rms0 3.537 / 3.386 (within 4.3%), rms1 3.234 / 3.159 (within 2.3%); flip-eligible on speed |
| P25 | qwen3rmsab | **HELD** | matched: rms1 / rms0 0.959 [0.935, 0.984 over 4 cross-draw ratios] vs [0.88, 0.98]; s/step rms0 4.258 / 4.396 (within 3.2%), rms1 4.110 / 4.192 (within 2.0%); flip-eligible on speed |
| P26 | qwen3rmsab | **HELD** | shipped: mean held-out rms1 - rms0 -0.0021 (|.| <= 0.01); held-out at N rms0 [0.8143, 0.8139] rms1 [0.8136, 0.8103]; matched: mean held-out rms1 - rms0 -0.0015 (|.| <= 0.01); held-out at N rms0 [0.8534, 0.8477] rms1 [0.8482, 0.8499] |
