# TC1 — Qwen3-30B-A3B on one RTX 5090: e4b vs Unsloth vs HF+PEFT vs axolotl at matched work, init and adapter precision (tc1)
Rule (tc1/TC1-PREREG.md): status per attempt in the vocabulary OK / REFUSED / OOM / INSTALL_FAILED / LOAD_FAULT / HARNESS_ERROR / ALARM / NOT_RUN; VALID/VOID per tp4's predicates plus the matched-set predicates (R3: matched init complete, fp32 adapters, step-0 held-out within 0.005 of e4b/reference_attn4_m); VERDICT exactly one of VALID / VOID / QUALITY_FAIL / OOM / UNSUPPORTED / HARNESS_ERROR / ALARM / NOT_RUN (QUALITY_FAIL = held-out |Δ| at N > 0.05 vs e4b/fused_attn4_m); positions = s/step ratios other/e4b from the medians over both draws (stability |d1-d2|/mean ≤ 5 % e4b / 10 % others, else UNSTABLE), quoted only when both arms are VALID; equivalence vs e4b/reference_attn4_m EQUIVALENT ≤ 0.02 / COMPARABLE ≤ 0.05 / else DIVERGENT; frozen base SAME-BYTES / DIFFERENT / N-A per slot; predictions P1–P10 (+ P1b) scored HELD / FALSIFIED / UNTESTED. VOID never enters a ratio or an equivalence reading. Nothing is licensed; no cross-box number is divided into these.
`versions.txt`
```
e4b 0.48.0 @a41c857dd7d5c68f8735af16634ce08426e01271 (GitHub main)
gnf4 0.42.0 @b4f93f1c62d1e3436ed45bec8ccd608c90433737 (GitHub main)
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
e4b(t212) 0.48.0 @a41c857dd7d5c68f8735af16634ce08426e01271
gnf4(t212) 0.42.0 @b4f93f1c62d1e3436ed45bec8ccd608c90433737
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
 "run_id": "tc1-5090-109",
 "instance_id": "54497288",
 "gpu": "NVIDIA GeForce RTX 5090",
 "driver": "580.126.09",
 "cpu": "AMD EPYC 7713 64-Core Processor",
 "nproc": 128,
 "mem_total_kb": "527972068",
 "cgroup_memory_max": "",
 "disk_root": "overlay         320G   22M  320G   1% /",
 "hostname": "e694370ae73b",
 "cgroup_cpu_max": "",
 "affinity_cpus": 128,
 "cgroup_cpuset_effective": "",
 "registered_gpu_class": "5090",
 "prereg": "tc1/TC1-PREREG.md",
 "local_box": null,
 "local_snapshot": null
}
```

### Qwen3-30B-A3B (amendment 52: grouped-nf4-gemm's LoRA delta one padded block vs buckets on packed rows, venv-e4b / torch 2.8) (`qwen3padbk28`, registered n_layers 48, attention census 192)
- model `Qwen/Qwen3-30B-A3B` @ `ad44e777bcd1`; tokens sha `d2a501eba57d`; N=40; fixture template alpaca seq 4096 (packed rows, amendment 39) micro-batch 1 × accum 4 lr 0.0002 r 16 α 16 optimizer adamw_8bit(lr=0.0002, weight_decay=0.001) schedule=linear warmup_steps=5 autocast False; e4b trainable 642514944; box_class RTX 5090 gpu NVIDIA GeForce RTX 5090
| framework | arm | status | validity | **VERDICT** | matched | init / dtype | step-0 class | N | s/step med(11+) | tok/s | peak GB | J/step | train first→last | held-out 0→final | quality Δ | engagement | trainable | regime | reason / why |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| e4b | fused_attn4_shipped_k0 | **OK** | VALID | **VALID** | native | native / bfloat16,float32 | — | 40 | 9.598 | 1680.9 | 26.920 | 4620.9 | 1.2594→0.9056 | 1.2914→0.9446 | -0.0098 | patched 48 / kcalls 768 | 642514944 | 4-bit experts (e4b NF4) + NF4 attention |  |
| e4b | fused_attn4_shipped_k1 | **OK** | VALID | **VALID** | native | native / bfloat16,float32 | — | 40 | 9.071 | 1793.4 | 26.920 | 4244.1 | 1.2594→0.9051 | 1.2914→0.9448 | -0.0096 | patched 48 / kcalls 768 | 642514944 | 4-bit experts (e4b NF4) + NF4 attention |  |
| e4b | fused_attn4_m_k0 | **OK** | VALID | **VALID** | yes | matched:3407 (complete 12480/12480) / float32 | SAME-BYTES-CLASS (0.0000) | 40 | 12.219 | 1338.0 | 32.422 | 5885.3 | 1.2594→0.9130 | 1.2914→0.9544 | 0.0000 | patched 48 / kcalls 768 | 642514944 | 4-bit experts (e4b NF4) + NF4 attention |  |
| e4b | fused_attn4_m_k1 | **OK** | VALID | **VALID** | yes | matched:3407 (complete 12480/12480) / float32 | SAME-BYTES-CLASS (0.0000) | 40 | 12.129 | 1339.8 | 28.178 | 5118.1 | 1.2594→0.9124 | 1.2914→0.9540 | -0.0004 | patched 48 / kcalls 768 | 642514944 | 4-bit experts (e4b NF4) + NF4 attention |  |
| e4b | fused_attn4_m_k1_d2 | **OK** | VALID | **VALID** | yes | matched:3407 (complete 12480/12480) / float32 | SAME-BYTES-CLASS (0.0000) | 40 | 11.971 | 1327.8 | 28.178 | 5153.3 | 1.2594→0.9127 | 1.2914→0.9543 | -0.0001 | patched 48 / kcalls 768 | 642514944 | 4-bit experts (e4b NF4) + NF4 attention |  |
| e4b | fused_attn4_m_k0_d2 | **OK** | VALID | **VALID** | yes | matched:3407 (complete 12480/12480) / float32 | SAME-BYTES-CLASS (0.0000) | 40 | 12.287 | 1315.6 | 32.413 | 5976.9 | 1.2594→0.9129 | 1.2914→0.9544 | 0.0000 | patched 48 / kcalls 768 | 642514944 | 4-bit experts (e4b NF4) + NF4 attention |  |
| e4b | fused_attn4_shipped_k1_d2 | **OK** | VALID | **VALID** | native | native / bfloat16,float32 | — | 40 | 9.106 | 1764.9 | 26.920 | 4284.6 | 1.2594→0.9054 | 1.2914→0.9448 | -0.0096 | patched 48 / kcalls 768 | 642514944 | 4-bit experts (e4b NF4) + NF4 attention |  |
| e4b | fused_attn4_shipped_k0_d2 | **OK** | VALID | **VALID** | native | native / bfloat16,float32 | — | 40 | 9.760 | 1666.5 | 26.920 | 4657.9 | 1.2594→0.9055 | 1.2914→0.9447 | -0.0097 | patched 48 / kcalls 768 | 642514944 | 4-bit experts (e4b NF4) + NF4 attention |  |
- prologue `e4b/fused_attn4_shipped_k0` **115.7 s** before step 1 (23% of the arm): c1_before 64.0, c1_after 32.2, load_weights 22.3, eval0 12.7; unattributed 1.227; budget 1260.0
- prologue `e4b/fused_attn4_shipped_k1` **108.8 s** before step 1 (23% of the arm): c1_before 64.4, c1_after 31.2, load_weights 22.4, attn4 5.9; unattributed 1.203; budget 1260.0
- prologue `e4b/fused_attn4_m_k0` **109.1 s** before step 1 (18% of the arm): c1_before 62.0, c1_after 35.9, load_weights 22.6, eval0 7.4; unattributed 1.182; budget 1260.0
- prologue `e4b/fused_attn4_m_k1` **111.1 s** before step 1 (18% of the arm): c1_before 63.8, c1_after 34.5, load_weights 22.7, eval0 7.5; unattributed 1.225; budget 1260.0
- prologue `e4b/fused_attn4_m_k1_d2` **112.6 s** before step 1 (18% of the arm): c1_before 64.1, c1_after 31.8, load_weights 22.9, eval0 7.6; unattributed 1.257; budget 1260.0
- prologue `e4b/fused_attn4_m_k0_d2` **123.5 s** before step 1 (20% of the arm): c1_before 72.3, c1_after 36.4, load_weights 24.3, eval0 7.5; unattributed 1.203; budget 1260.0
- prologue `e4b/fused_attn4_shipped_k1_d2` **108.7 s** before step 1 (22% of the arm): c1_before 62.7, c1_after 36.3, load_weights 23.1, attn4 5.9; unattributed 1.221; budget 1260.0
- prologue `e4b/fused_attn4_shipped_k0_d2` **110.5 s** before step 1 (22% of the arm): c1_before 65.2, c1_after 31.3, load_weights 22.9, eval0 6.0; unattributed 1.195; budget 1260.0
- draws (R1): `e4b/fused_attn4_shipped_k0` STABLE (9.598/9.760 s, |Δ|/mean 1.7% vs 5%); `e4b/fused_attn4_shipped_k1` STABLE (9.071/9.106 s, |Δ|/mean 0.4% vs 5%); `e4b/fused_attn4_m_k0` STABLE (12.219/12.287 s, |Δ|/mean 0.6% vs 5%); `e4b/fused_attn4_m_k1` STABLE (12.129/11.971 s, |Δ|/mean 1.3% vs 5%)
- e4b internal parity: NO-ARM
- **NO MATCHED POSITION QUOTED (unsloth)** — not quoted: e4b STABLE / unsloth no receipt
- **NO MATCHED POSITION QUOTED (hf)** — not quoted: e4b STABLE / hf no receipt
- **NO MATCHED POSITION QUOTED (axolotl)** — not quoted: e4b STABLE / axolotl no receipt
- equivalence vs `e4b/fused_attn4_m` (I/H; EQUIVALENT iff both ≤ band = max(0.005, 3 × fused-vs-reference |Δ|) = none (no reference arm: COMPARABLE is the ceiling); COMPARABLE ≤ 0.05; draw-noise floor 0.0003): `e4b/fused_attn4_m_k1` **COMPARABLE** (median step |Δ| 0.0003, |Δ held-out at N| 0.0004, step-0 0.0000 SAME-BYTES-CLASS, |Δ loss at step 2| 0.0005, paired rows mean -0.0004 ± 0.0003 SE over 8, favouring arm 5 / anchor 3) — no reference arm ran: the EQUIVALENT band cannot be set, COMPARABLE is the ceiling; `e4b/fused_attn4_m_k1_d2` **COMPARABLE** (median step |Δ| 0.0002, |Δ held-out at N| 0.0001, step-0 0.0000 SAME-BYTES-CLASS, |Δ loss at step 2| 0.0053, paired rows mean -0.0001 ± 0.0002 SE over 8, favouring arm 4 / anchor 4) — no reference arm ran: the EQUIVALENT band cannot be set, COMPARABLE is the ceiling; `e4b/fused_attn4_m_k0_d2` **COMPARABLE** (median step |Δ| 0.0002, |Δ held-out at N| 0.0000, step-0 0.0000 SAME-BYTES-CLASS, |Δ loss at step 2| 0.0017, paired rows mean +0.0000 ± 0.0002 SE over 8, favouring arm 3 / anchor 5) — no reference arm ran: the EQUIVALENT band cannot be set, COMPARABLE is the ceiling
- matched_init_sha (B, name-free, canonical slot order): anchor `f7832488926eda91`; `e4b/fused_attn4_m_k0` same; `e4b/fused_attn4_m_k1` same; `e4b/fused_attn4_m_k1_d2` same; `e4b/fused_attn4_m_k0_d2` same
- frozen base (R5), anchor `e4b/fused_attn4_m`: gate_up nf4/64 sha 8c3b3fd78e89 control detects, down nf4/64 sha 59569d810a1a control detects, q_proj nf4/64+dq sha 5c5e402ce51d control detects
- frozen base `e4b/fused_attn4_shipped_k0`: gate_up **SAME-BYTES** (nf4/64) control detects, down **SAME-BYTES** (nf4/64) control detects, q_proj **SAME-BYTES** (nf4/64+dq) control detects
- frozen base `e4b/fused_attn4_shipped_k1`: gate_up **SAME-BYTES** (nf4/64) control detects, down **SAME-BYTES** (nf4/64) control detects, q_proj **SAME-BYTES** (nf4/64+dq) control detects
- frozen base `e4b/fused_attn4_m_k1`: gate_up **SAME-BYTES** (nf4/64) control detects, down **SAME-BYTES** (nf4/64) control detects, q_proj **SAME-BYTES** (nf4/64+dq) control detects
- frozen base `e4b/fused_attn4_m_k1_d2`: gate_up **SAME-BYTES** (nf4/64) control detects, down **SAME-BYTES** (nf4/64) control detects, q_proj **SAME-BYTES** (nf4/64+dq) control detects
- frozen base `e4b/fused_attn4_m_k0_d2`: gate_up **SAME-BYTES** (nf4/64) control detects, down **SAME-BYTES** (nf4/64) control detects, q_proj **SAME-BYTES** (nf4/64+dq) control detects
- frozen base `e4b/fused_attn4_shipped_k1_d2`: gate_up **SAME-BYTES** (nf4/64) control detects, down **SAME-BYTES** (nf4/64) control detects, q_proj **SAME-BYTES** (nf4/64+dq) control detects
- frozen base `e4b/fused_attn4_shipped_k0_d2`: gate_up **SAME-BYTES** (nf4/64) control detects, down **SAME-BYTES** (nf4/64) control detects, q_proj **SAME-BYTES** (nf4/64+dq) control detects

## Verdicts
| family | arm | VERDICT | validity | quality | note |
|---|---|---|---|---|---|
| qwen3padbk28 | e4b/fused_attn4_shipped_k0 | **VALID** | VALID | -0.0098 |  |
| qwen3padbk28 | e4b/fused_attn4_shipped_k1 | **VALID** | VALID | -0.0096 |  |
| qwen3padbk28 | e4b/fused_attn4_m_k0 | **VALID** | VALID | 0.0000 |  |
| qwen3padbk28 | e4b/fused_attn4_m_k1 | **VALID** | VALID | -0.0004 |  |
| qwen3padbk28 | e4b/fused_attn4_m_k1_d2 | **VALID** | VALID | -0.0001 |  |
| qwen3padbk28 | e4b/fused_attn4_m_k0_d2 | **VALID** | VALID | 0.0000 |  |
| qwen3padbk28 | e4b/fused_attn4_shipped_k1_d2 | **VALID** | VALID | -0.0096 |  |
| qwen3padbk28 | e4b/fused_attn4_shipped_k0_d2 | **VALID** | VALID | -0.0097 |  |

## Predictions P130 / P131 / P132 / P133 (TC1-PREREG amendment 52: the LoRA delta one padded block vs buckets on packed rows in torch 2.8; scored mechanically)
| prediction | family | verdict | evidence |
|---|---|---|---|
| P132 | qwen3padbk28 | **HELD** | matched: peak k0 32.417 / k1 28.178 GB, drop 4.239 vs [3.0, 99.0] |
| P130 | qwen3padbk28 | **HELD** | matched: k1 / k0 0.983 [0.974, 0.993 over 4 cross-draw ratios] vs [0.0, 1.02]; s/step k0 12.219 / 12.287 (within 0.6%), k1 12.129 / 11.971 (within 1.3%); peak k0 32.42 / k1 28.18 GB |
| P131 | qwen3padbk28 | **HELD** | shipped: k1 / k0 0.939 [0.929, 0.949 over 4 cross-draw ratios] vs [0.0, 1.02]; s/step k0 9.598 / 9.760 (within 1.7%), k1 9.071 / 9.106 (within 0.4%); peak k0 26.92 / k1 26.92 GB |
| P133 | qwen3padbk28 | **HELD** | matched: mean held-out k1 - k0 -0.0002 (|.| <= 0.005); held-out at N k0 [0.9544, 0.9544] k1 [0.954, 0.9543]; shipped: mean held-out k1 - k0 +0.0002 (|.| <= 0.005); held-out at N k0 [0.9446, 0.9447] k1 [0.9448, 0.9448] |

## Load gate (TC1-PREREG amendment 33): 5 draw(s) voided for host load and run again
Each line: the arm, the attempt, the median host load1 over that attempt's run, the gate. A VOID attempt's files are in loadvoid/; the last attempt of each arm stands whatever its load.

- `qwen3padbk28/e4b/fused_attn4_shipped_k0 attempt 0 load1_median 3.15 gate 6.0 status ok over 0`
- `qwen3padbk28/e4b/fused_attn4_shipped_k1 attempt 0 load1_median 3.8 gate 6.0 status ok over 0`
- `qwen3padbk28/e4b/fused_attn4_m_k0 attempt 0 load1_median 3.11 gate 6.0 status ok over 0`
- `qwen3padbk28/e4b/fused_attn4_m_k1 attempt 0 load1_median 7.83 gate 6.0 status ok over 1`
- `qwen3padbk28/e4b/fused_attn4_m_k1 attempt 0 VOID (host load1 median 7.83 > 6.0): re-run 1 of 2`
- `qwen3padbk28/e4b/fused_attn4_m_k1 attempt 1 load1_median 3.75 gate 6.0 status ok over 0`
- `qwen3padbk28/e4b/fused_attn4_m_k1_d2 attempt 0 load1_median 6.76 gate 6.0 status ok over 1`
- `qwen3padbk28/e4b/fused_attn4_m_k1_d2 attempt 0 VOID (host load1 median 6.76 > 6.0): re-run 1 of 2`
- `qwen3padbk28/e4b/fused_attn4_m_k1_d2 attempt 1 load1_median 5.86 gate 6.0 status ok over 0`
- `qwen3padbk28/e4b/fused_attn4_m_k0_d2 attempt 0 load1_median 6.94 gate 6.0 status ok over 1`
- `qwen3padbk28/e4b/fused_attn4_m_k0_d2 attempt 0 VOID (host load1 median 6.94 > 6.0): re-run 1 of 2`
- `qwen3padbk28/e4b/fused_attn4_m_k0_d2 attempt 1 load1_median 7.95 gate 6.0 status ok over 1`
- `qwen3padbk28/e4b/fused_attn4_m_k0_d2 attempt 1 VOID (host load1 median 7.95 > 6.0): re-run 2 of 2`
- `qwen3padbk28/e4b/fused_attn4_m_k0_d2 attempt 2 load1_median 7.11 gate 6.0 status ok over 1`
- `qwen3padbk28/e4b/fused_attn4_shipped_k1_d2 attempt 0 load1_median 5.53 gate 6.0 status ok over 0`
- `qwen3padbk28/e4b/fused_attn4_shipped_k0_d2 attempt 0 load1_median 6.45 gate 6.0 status ok over 1`
- `qwen3padbk28/e4b/fused_attn4_shipped_k0_d2 attempt 0 VOID (host load1 median 6.45 > 6.0): re-run 1 of 2`
- `qwen3padbk28/e4b/fused_attn4_shipped_k0_d2 attempt 1 load1_median 5.74 gate 6.0 status ok over 0`
