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
 "run_id": "tc1-5090-71",
 "instance_id": "54223342",
 "gpu": "NVIDIA GeForce RTX 5090",
 "driver": "595.84",
 "cpu": "AMD EPYC 7B13 64-Core Processor",
 "nproc": 256,
 "mem_total_kb": "2101193408",
 "cgroup_memory_max": "730283900928",
 "disk_root": "overlay         320G   77M  320G   1% /",
 "hostname": "560d314d2251",
 "registered_gpu_class": "5090",
 "prereg": "tc1/TC1-PREREG.md",
 "local_box": null,
 "local_snapshot": null
}
```

### Mixtral-8x7B-Instruct-v0.1 (amendment 28: the expert absmax fp32 vs double-quantized, matched arm, resident) (`mixtraldqab`, registered n_layers 32, attention census 128)
- model `mistralai/Mixtral-8x7B-Instruct-v0.1` @ `eba92302a286`; tokens sha `4a41b3f4a561`; N=20; fixture template alpaca seq 2048 micro-batch 2 × accum 4 lr 0.0002 r 16 α 16 optimizer adamw_8bit(lr=0.0002, weight_decay=0.001) schedule=linear warmup_steps=5 autocast False; e4b trainable 223346688; box_class RTX 5090 gpu NVIDIA GeForce RTX 5090
| framework | arm | status | validity | **VERDICT** | matched | init / dtype | step-0 class | N | s/step med(11+) | tok/s | peak GB | J/step | train first→last | held-out 0→final | quality Δ | engagement | trainable | regime | reason / why |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| e4b | fused_attn4_m_dq0 | **OK** | VALID | **VALID** | yes | matched:3407 (complete 640/640) / float32 | SAME-BYTES-CLASS (0.0000) | 20 | 3.807 | 415.0 | 31.070 | 1609.7 | 1.4174→0.6955 | 1.4249→0.7138 | 0.0000 | patched 32 / kcalls 512 | 223346688 | 4-bit experts (e4b NF4) + NF4 attention |  |
| e4b | fused_attn4_m_dq1 | **OK** | VALID | **VALID** | yes | matched:3407 (complete 640/640) / float32 | SAME-BYTES-CLASS (0.0041) | 20 | 3.932 | 415.2 | 29.024 | 1621.7 | 1.4104→0.6949 | 1.4207→0.7130 | -0.0008 | patched 32 / kcalls 512 | 223346688 | 4-bit experts (e4b NF4) + NF4 attention |  |
| e4b | fused_attn4_m_dq1_d2 | **OK** | VALID | **VALID** | yes | matched:3407 (complete 640/640) / float32 | SAME-BYTES-CLASS (0.0041) | 20 | 3.970 | 415.6 | 29.032 | 1627.2 | 1.4104→0.6960 | 1.4207→0.7138 | 0.0000 | patched 32 / kcalls 512 | 223346688 | 4-bit experts (e4b NF4) + NF4 attention |  |
| e4b | fused_attn4_m_dq0_d2 | **OK** | VALID | **VALID** | yes | matched:3407 (complete 640/640) / float32 | SAME-BYTES-CLASS (0.0000) | 20 | 3.917 | 418.8 | 31.074 | 1595.8 | 1.4174→0.6955 | 1.4249→0.7186 | 0.0048 | patched 32 / kcalls 512 | 223346688 | 4-bit experts (e4b NF4) + NF4 attention |  |
- prologue `e4b/fused_attn4_m_dq0` **144.3 s** before step 1 (62% of the arm): c1_before 93.8, c1_after 45.4, load_weights 28.4, attn4 8.7; unattributed 1.241; budget 1260.0
- prologue `e4b/fused_attn4_m_dq1` **136.5 s** before step 1 (61% of the arm): c1_before 84.0, c1_after 41.7, load_weights 32.0, attn4 8.7; unattributed 1.23; budget 1260.0
- prologue `e4b/fused_attn4_m_dq1_d2` **132.4 s** before step 1 (60% of the arm): c1_before 84.4, c1_after 41.9, load_weights 27.8, attn4 8.8; unattributed 1.219; budget 1260.0
- prologue `e4b/fused_attn4_m_dq0_d2` **155.3 s** before step 1 (64% of the arm): c1_before 104.7, c1_after 51.9, load_weights 28.5, attn4 9.9; unattributed 1.462; budget 1260.0
- draws (R1): `e4b/fused_attn4_m_dq0` STABLE (3.807/3.917 s, |Δ|/mean 2.8% vs 5%); `e4b/fused_attn4_m_dq1` STABLE (3.932/3.970 s, |Δ|/mean 1.0% vs 5%)
- e4b internal parity: NO-ARM
- **NO MATCHED POSITION QUOTED (unsloth)** — not quoted: e4b STABLE / unsloth no receipt
- **NO MATCHED POSITION QUOTED (hf)** — not quoted: e4b STABLE / hf no receipt
- **NO MATCHED POSITION QUOTED (axolotl)** — not quoted: e4b STABLE / axolotl no receipt
- equivalence vs `e4b/fused_attn4_m` (I/H; EQUIVALENT iff both ≤ band = max(0.005, 3 × fused-vs-reference |Δ|) = none (no reference arm: COMPARABLE is the ceiling); COMPARABLE ≤ 0.05; draw-noise floor 0.0048): `e4b/fused_attn4_m_dq1` **COMPARABLE** (median step |Δ| 0.0015, |Δ held-out at N| 0.0008, step-0 0.0041 SAME-BYTES-CLASS, |Δ loss at step 2| 0.0000, paired rows mean -0.0008 ± 0.0013 SE over 8, favouring arm 5 / anchor 3) — no reference arm ran: the EQUIVALENT band cannot be set, COMPARABLE is the ceiling; `e4b/fused_attn4_m_dq1_d2` **COMPARABLE** (median step |Δ| 0.0020, |Δ held-out at N| 0.0000, step-0 0.0041 SAME-BYTES-CLASS, |Δ loss at step 2| 0.0048, paired rows mean +0.0000 ± 0.0017 SE over 8, favouring arm 4 / anchor 4) — no reference arm ran: the EQUIVALENT band cannot be set, COMPARABLE is the ceiling; `e4b/fused_attn4_m_dq0_d2` **COMPARABLE** (median step |Δ| 0.0011, |Δ held-out at N| 0.0048, step-0 0.0000 SAME-BYTES-CLASS, |Δ loss at step 2| 0.0067, paired rows mean +0.0048 ± 0.0022 SE over 8, favouring arm 1 / anchor 7) — no reference arm ran: the EQUIVALENT band cannot be set, COMPARABLE is the ceiling
- matched_init_sha (B, name-free, canonical slot order): anchor `8d6f46fd1d279996`; `e4b/fused_attn4_m_dq0` same; `e4b/fused_attn4_m_dq1` same; `e4b/fused_attn4_m_dq1_d2` same; `e4b/fused_attn4_m_dq0_d2` same
- frozen base (R5), anchor `e4b/fused_attn4_m`: gate_up nf4/64 sha c56e6fc0fd22 control detects, down nf4/64 sha 7e63cab94f9c control detects, q_proj nf4/64+dq sha d5fbebd64960 control detects
- frozen base `e4b/fused_attn4_m_dq1`: gate_up **N-A** (regime differs: anchor nf4/64 vs nf4/64+dq) control detects, down **N-A** (regime differs: anchor nf4/64 vs nf4/64+dq) control detects, q_proj **SAME-BYTES** (nf4/64+dq) control detects
- frozen base `e4b/fused_attn4_m_dq1_d2`: gate_up **N-A** (regime differs: anchor nf4/64 vs nf4/64+dq) control detects, down **N-A** (regime differs: anchor nf4/64 vs nf4/64+dq) control detects, q_proj **SAME-BYTES** (nf4/64+dq) control detects
- frozen base `e4b/fused_attn4_m_dq0_d2`: gate_up **SAME-BYTES** (nf4/64) control detects, down **SAME-BYTES** (nf4/64) control detects, q_proj **SAME-BYTES** (nf4/64+dq) control detects

## Verdicts
| family | arm | VERDICT | validity | quality | note |
|---|---|---|---|---|---|
| mixtraldqab | e4b/fused_attn4_m_dq0 | **VALID** | VALID | 0.0000 |  |
| mixtraldqab | e4b/fused_attn4_m_dq1 | **VALID** | VALID | -0.0008 |  |
| mixtraldqab | e4b/fused_attn4_m_dq1_d2 | **VALID** | VALID | 0.0000 |  |
| mixtraldqab | e4b/fused_attn4_m_dq0_d2 | **VALID** | VALID | 0.0048 |  |

## Predictions P56 / P57 / P58 (TC1-PREREG amendment 28: the expert absmax fp32 vs double-quantized, two stable draws a side; scored mechanically)
| prediction | family | verdict | evidence |
|---|---|---|---|
| P56 | qwen3dqab | **UNTESTED** | Qwen3-30B-A3B: no qwen3dqab receipts in this directory |
| P57 | mixtraldqab | **HELD** | Mixtral-8x7B: dq1 / dq0 1.023 [1.004, 1.043 over 4 cross-draw ratios] vs [0.97, 1.03]; peak dq0 31.07 / dq1 29.03 GB, drop 2.044 vs [1.9, 2.3]; s/step dq0 3.807 / 3.917 (within 2.8%), dq1 3.932 / 3.970 (within 1.0%) |
| P58 | dqab | **UNTESTED** | qwen3dqab: no receipts; mixtraldqab: mean held-out dq1 - dq0 -0.0028 (|.| <= 0.005); held-out at N dq0 [0.7138, 0.7186] dq1 [0.713, 0.7138] |
