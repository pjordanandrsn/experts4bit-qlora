# TC1 — Qwen3-30B-A3B on one RTX 5090: e4b vs Unsloth vs HF+PEFT vs axolotl at matched work, init and adapter precision (tc1)
Rule (tc1/TC1-PREREG.md): status per attempt in the vocabulary OK / REFUSED / OOM / INSTALL_FAILED / LOAD_FAULT / HARNESS_ERROR / ALARM / NOT_RUN; VALID/VOID per tp4's predicates plus the matched-set predicates (R3: matched init complete, fp32 adapters, step-0 held-out within 0.005 of e4b/reference_attn4_m); VERDICT exactly one of VALID / VOID / QUALITY_FAIL / OOM / UNSUPPORTED / HARNESS_ERROR / ALARM / NOT_RUN (QUALITY_FAIL = held-out |Δ| at N > 0.05 vs e4b/fused_attn4_m); positions = s/step ratios other/e4b from the medians over both draws (stability |d1-d2|/mean ≤ 5 % e4b / 10 % others, else UNSTABLE), quoted only when both arms are VALID; equivalence vs e4b/reference_attn4_m EQUIVALENT ≤ 0.02 / COMPARABLE ≤ 0.05 / else DIVERGENT; frozen base SAME-BYTES / DIFFERENT / N-A per slot; predictions P1–P10 (+ P1b) scored HELD / FALSIFIED / UNTESTED. VOID never enters a ratio or an equivalence reading. Nothing is licensed; no cross-box number is divided into these.
`versions.txt`
```
e4b 0.43.0 @186265dc04769c6937b206b9152f900ac6fee7f4 (GitHub main)
gnf4 0.35.0 @192f63f1416b25f0ea5f5ba8eb0903d2ece19512 (GitHub main)
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
 "run_id": "tc1-5090-51",
 "instance_id": "54095822",
 "gpu": "NVIDIA GeForce RTX 5090",
 "driver": "580.95.05",
 "cpu": "AMD EPYC 7C13 64-Core Processor",
 "nproc": 256,
 "mem_total_kb": "858436212",
 "cgroup_memory_max": "377392988160",
 "disk_root": "overlay         320G   52M  320G   1% /",
 "hostname": "b89add6363e7",
 "registered_gpu_class": "5090",
 "prereg": "tc1/TC1-PREREG.md",
 "local_box": null,
 "local_snapshot": null
}
```

### Qwen3-30B-A3B (amendment 20: gnf4's per-pass host reuse off vs on, on every current default) (`qwen3reuseab`, registered n_layers 48, attention census 192)
- model `Qwen/Qwen3-30B-A3B` @ `ad44e777bcd1`; tokens sha `bfc742f67e37`; N=20; fixture template alpaca seq 2048 micro-batch 2 × accum 4 lr 0.0002 r 16 α 16 optimizer adamw_8bit(lr=0.0002, weight_decay=0.001) schedule=linear warmup_steps=5 autocast False; e4b trainable 642514944; box_class RTX 5090 gpu NVIDIA GeForce RTX 5090
| framework | arm | status | validity | **VERDICT** | matched | init / dtype | step-0 class | N | s/step med(11+) | tok/s | peak GB | J/step | train first→last | held-out 0→final | quality Δ | engagement | trainable | regime | reason / why |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| e4b | fused_attn4_shipped_reuse0 | **OK** | VALID | **VALID** | native | native / bfloat16,float32 | — | 20 | 3.146 | 429.3 | 24.576 | 831.1 | 2.0614→0.7992 | 1.9441→0.8145 | -0.0354 | patched 48 / kcalls 768 | 642514944 | 4-bit experts (e4b NF4) + NF4 attention |  |
| e4b | fused_attn4_shipped_reuse1 | **OK** | VALID | **VALID** | native | native / bfloat16,float32 | — | 20 | 2.870 | 526.0 | 24.576 | 746.1 | 2.0614→0.7980 | 1.9441→0.8154 | -0.0346 | patched 48 / kcalls 768 | 642514944 | 4-bit experts (e4b NF4) + NF4 attention |  |
| e4b | fused_attn4_m_reuse0 | **OK** | VALID | **VALID** | yes | matched:3407 (complete 12480/12480) / float32 | SAME-BYTES-CLASS (0.0000) | 20 | 3.875 | 389.8 | 27.228 | 977.0 | 2.0614→0.8334 | 1.9441→0.8500 | 0.0000 | patched 48 / kcalls 768 | 642514944 | 4-bit experts (e4b NF4) + NF4 attention |  |
| e4b | fused_attn4_m_reuse1 | **OK** | VALID | **VALID** | yes | matched:3407 (complete 12480/12480) / float32 | SAME-BYTES-CLASS (0.0000) | 20 | 3.697 | 405.8 | 27.210 | 958.1 | 2.0614→0.8327 | 1.9441→0.8481 | -0.0019 | patched 48 / kcalls 768 | 642514944 | 4-bit experts (e4b NF4) + NF4 attention |  |
| e4b | fused_attn4_m_reuse1_d2 | **OK** | VALID | **VALID** | yes | matched:3407 (complete 12480/12480) / float32 | SAME-BYTES-CLASS (0.0000) | 20 | 3.799 | 394.6 | 27.210 | 962.6 | 2.0614→0.8330 | 1.9441→0.8528 | 0.0028 | patched 48 / kcalls 768 | 642514944 | 4-bit experts (e4b NF4) + NF4 attention |  |
| e4b | fused_attn4_m_reuse0_d2 | **OK** | VALID | **VALID** | yes | matched:3407 (complete 12480/12480) / float32 | SAME-BYTES-CLASS (0.0000) | 20 | 4.008 | 376.8 | 27.228 | 989.4 | 2.0614→0.8316 | 1.9441→0.8493 | -0.0007 | patched 48 / kcalls 768 | 642514944 | 4-bit experts (e4b NF4) + NF4 attention |  |
| e4b | fused_attn4_shipped_reuse1_d2 | **OK** | VALID | **VALID** | native | native / bfloat16,float32 | — | 20 | 2.934 | 517.4 | 24.576 | 747.5 | 2.0614→0.7979 | 1.9441→0.8141 | -0.0359 | patched 48 / kcalls 768 | 642514944 | 4-bit experts (e4b NF4) + NF4 attention |  |
| e4b | fused_attn4_shipped_reuse0_d2 | **OK** | VALID | **VALID** | native | native / bfloat16,float32 | — | 20 | 3.075 | 485.1 | 24.576 | 771.9 | 2.0614→0.7974 | 1.9441→0.8147 | -0.0353 | patched 48 / kcalls 768 | 642514944 | 4-bit experts (e4b NF4) + NF4 attention |  |
- prologue `e4b/fused_attn4_shipped_reuse0` **107.3 s** before step 1 (60% of the arm): c1_before 51.1, load_weights 27.6, c1_after 25.7, eval0 11.0; unattributed 1.326; budget 1260.0
- prologue `e4b/fused_attn4_shipped_reuse1` **101.0 s** before step 1 (63% of the arm): c1_before 50.6, load_weights 31.0, c1_after 24.9, attn4 5.7; unattributed 1.246; budget 1260.0
- prologue `e4b/fused_attn4_m_reuse0` **102.7 s** before step 1 (56% of the arm): c1_before 52.8, load_weights 28.9, c1_after 27.2, attn4 6.6; unattributed 1.239; budget 1260.0
- prologue `e4b/fused_attn4_m_reuse1` **100.5 s** before step 1 (57% of the arm): c1_before 52.0, load_weights 28.4, c1_after 25.0, attn4 5.9; unattributed 1.253; budget 1194.2
- prologue `e4b/fused_attn4_m_reuse1_d2` **110.5 s** before step 1 (58% of the arm): c1_before 61.4, c1_after 30.6, load_weights 28.2, attn4 5.9; unattributed 1.507; budget 1120.3
- prologue `e4b/fused_attn4_m_reuse0_d2` **100.1 s** before step 1 (55% of the arm): c1_before 51.0, load_weights 29.0, c1_after 26.4, attn4 5.9; unattributed 1.227; budget 1039.8
- prologue `e4b/fused_attn4_shipped_reuse1_d2` **105.7 s** before step 1 (64% of the arm): c1_before 56.8, load_weights 28.9, c1_after 26.3, attn4 5.7; unattributed 1.261; budget 963.2
- prologue `e4b/fused_attn4_shipped_reuse0_d2` **111.4 s** before step 1 (63% of the arm): c1_before 63.0, c1_after 31.2, load_weights 27.8, attn4 6.1; unattributed 1.363; budget 893.5
- draws (R1): `e4b/fused_attn4_shipped_reuse0` STABLE (3.146/3.075 s, |Δ|/mean 2.3% vs 5%); `e4b/fused_attn4_shipped_reuse1` STABLE (2.870/2.934 s, |Δ|/mean 2.2% vs 5%); `e4b/fused_attn4_m_reuse0` STABLE (3.875/4.008 s, |Δ|/mean 3.4% vs 5%); `e4b/fused_attn4_m_reuse1` STABLE (3.697/3.799 s, |Δ|/mean 2.7% vs 5%)
- e4b internal parity: NO-ARM
- **NO MATCHED POSITION QUOTED (unsloth)** — not quoted: e4b STABLE / unsloth no receipt
- **NO MATCHED POSITION QUOTED (hf)** — not quoted: e4b STABLE / hf no receipt
- **NO MATCHED POSITION QUOTED (axolotl)** — not quoted: e4b STABLE / axolotl no receipt
- equivalence vs `e4b/fused_attn4_m` (I/H; EQUIVALENT iff both ≤ band = max(0.005, 3 × fused-vs-reference |Δ|) = none (no reference arm: COMPARABLE is the ceiling); COMPARABLE ≤ 0.05; draw-noise floor 0.0047): `e4b/fused_attn4_m_reuse1` **COMPARABLE** (median step |Δ| 0.0014, |Δ held-out at N| 0.0019, step-0 0.0000 SAME-BYTES-CLASS, |Δ loss at step 2| 0.0070, paired rows mean -0.0019 ± 0.0023 SE over 8, favouring arm 5 / anchor 3) — no reference arm ran: the EQUIVALENT band cannot be set, COMPARABLE is the ceiling; `e4b/fused_attn4_m_reuse1_d2` **COMPARABLE** (median step |Δ| 0.0014, |Δ held-out at N| 0.0028, step-0 0.0000 SAME-BYTES-CLASS, |Δ loss at step 2| 0.0133, paired rows mean +0.0028 ± 0.0026 SE over 8, favouring arm 3 / anchor 5) — no reference arm ran: the EQUIVALENT band cannot be set, COMPARABLE is the ceiling; `e4b/fused_attn4_m_reuse0_d2` **COMPARABLE** (median step |Δ| 0.0016, |Δ held-out at N| 0.0007, step-0 0.0000 SAME-BYTES-CLASS, |Δ loss at step 2| 0.0103, paired rows mean -0.0007 ± 0.0021 SE over 8, favouring arm 5 / anchor 3) — no reference arm ran: the EQUIVALENT band cannot be set, COMPARABLE is the ceiling
- matched_init_sha (B, name-free, canonical slot order): anchor `f7832488926eda91`; `e4b/fused_attn4_m_reuse0` same; `e4b/fused_attn4_m_reuse1` same; `e4b/fused_attn4_m_reuse1_d2` same; `e4b/fused_attn4_m_reuse0_d2` same
- frozen base (R5), anchor `e4b/fused_attn4_m`: gate_up nf4/64 sha 8c3b3fd78e89 control detects, down nf4/64 sha 59569d810a1a control detects, q_proj nf4/64+dq sha 5c5e402ce51d control detects
- frozen base `e4b/fused_attn4_shipped_reuse0`: gate_up **SAME-BYTES** (nf4/64) control detects, down **SAME-BYTES** (nf4/64) control detects, q_proj **SAME-BYTES** (nf4/64+dq) control detects
- frozen base `e4b/fused_attn4_shipped_reuse1`: gate_up **SAME-BYTES** (nf4/64) control detects, down **SAME-BYTES** (nf4/64) control detects, q_proj **SAME-BYTES** (nf4/64+dq) control detects
- frozen base `e4b/fused_attn4_m_reuse1`: gate_up **SAME-BYTES** (nf4/64) control detects, down **SAME-BYTES** (nf4/64) control detects, q_proj **SAME-BYTES** (nf4/64+dq) control detects
- frozen base `e4b/fused_attn4_m_reuse1_d2`: gate_up **SAME-BYTES** (nf4/64) control detects, down **SAME-BYTES** (nf4/64) control detects, q_proj **SAME-BYTES** (nf4/64+dq) control detects
- frozen base `e4b/fused_attn4_m_reuse0_d2`: gate_up **SAME-BYTES** (nf4/64) control detects, down **SAME-BYTES** (nf4/64) control detects, q_proj **SAME-BYTES** (nf4/64+dq) control detects
- frozen base `e4b/fused_attn4_shipped_reuse1_d2`: gate_up **SAME-BYTES** (nf4/64) control detects, down **SAME-BYTES** (nf4/64) control detects, q_proj **SAME-BYTES** (nf4/64+dq) control detects
- frozen base `e4b/fused_attn4_shipped_reuse0_d2`: gate_up **SAME-BYTES** (nf4/64) control detects, down **SAME-BYTES** (nf4/64) control detects, q_proj **SAME-BYTES** (nf4/64+dq) control detects

## Verdicts
| family | arm | VERDICT | validity | quality | note |
|---|---|---|---|---|---|
| qwen3reuseab | e4b/fused_attn4_shipped_reuse0 | **VALID** | VALID | -0.0354 |  |
| qwen3reuseab | e4b/fused_attn4_shipped_reuse1 | **VALID** | VALID | -0.0346 |  |
| qwen3reuseab | e4b/fused_attn4_m_reuse0 | **VALID** | VALID | 0.0000 |  |
| qwen3reuseab | e4b/fused_attn4_m_reuse1 | **VALID** | VALID | -0.0019 |  |
| qwen3reuseab | e4b/fused_attn4_m_reuse1_d2 | **VALID** | VALID | 0.0028 |  |
| qwen3reuseab | e4b/fused_attn4_m_reuse0_d2 | **VALID** | VALID | -0.0007 |  |
| qwen3reuseab | e4b/fused_attn4_shipped_reuse1_d2 | **VALID** | VALID | -0.0359 |  |
| qwen3reuseab | e4b/fused_attn4_shipped_reuse0_d2 | **VALID** | VALID | -0.0353 |  |

## Predictions P33 / P34 (TC1-PREREG amendment 20, #945: gnf4's per-pass host reuse off vs on, two stable draws a side; scored mechanically)
| prediction | family | verdict | evidence |
|---|---|---|---|
| P33 | qwen3reuseab | **HELD** | shipped: reuse1 / reuse0 0.933 [0.912, 0.954 over 4 cross-draw ratios] vs [0.92, 0.99]; decision reading: flip-eligible; s/step reuse0 3.146 / 3.075 (within 2.3%), reuse1 2.870 / 2.934 (within 2.2%); held-out at N reuse0 0.8145 / reuse1 0.8154; reuse1 hits (process) {"plan_hits": 8528, "plan_misses": 8368, "upload_hits": 16465, "upload_misses": 27795} |
| P34 | qwen3reuseab | **HELD** | matched: reuse1 / reuse0 0.951 [0.922, 0.980 over 4 cross-draw ratios] vs [0.93, 0.99]; decision reading: flip-eligible; s/step reuse0 3.875 / 4.008 (within 3.4%), reuse1 3.697 / 3.799 (within 2.7%); held-out at N reuse0 0.8500 / reuse1 0.8481; reuse1 hits (process) {"plan_hits": 8528, "plan_misses": 8368, "upload_hits": 16465, "upload_misses": 27800} |
