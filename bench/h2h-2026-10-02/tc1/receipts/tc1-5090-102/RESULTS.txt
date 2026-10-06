# TC1 — Qwen3-30B-A3B on one RTX 5090: e4b vs Unsloth vs HF+PEFT vs axolotl at matched work, init and adapter precision (tc1)
Rule (tc1/TC1-PREREG.md): status per attempt in the vocabulary OK / REFUSED / OOM / INSTALL_FAILED / LOAD_FAULT / HARNESS_ERROR / ALARM / NOT_RUN; VALID/VOID per tp4's predicates plus the matched-set predicates (R3: matched init complete, fp32 adapters, step-0 held-out within 0.005 of e4b/reference_attn4_m); VERDICT exactly one of VALID / VOID / QUALITY_FAIL / OOM / UNSUPPORTED / HARNESS_ERROR / ALARM / NOT_RUN (QUALITY_FAIL = held-out |Δ| at N > 0.05 vs e4b/fused_attn4_m); positions = s/step ratios other/e4b from the medians over both draws (stability |d1-d2|/mean ≤ 5 % e4b / 10 % others, else UNSTABLE), quoted only when both arms are VALID; equivalence vs e4b/reference_attn4_m EQUIVALENT ≤ 0.02 / COMPARABLE ≤ 0.05 / else DIVERGENT; frozen base SAME-BYTES / DIFFERENT / N-A per slot; predictions P1–P10 (+ P1b) scored HELD / FALSIFIED / UNTESTED. VOID never enters a ratio or an equivalence reading. Nothing is licensed; no cross-box number is divided into these.
`versions.txt`
```
e4b 0.48.0 @0afa032ac9333a89bd977c50c23c19df7e00c070 (GitHub main)
gnf4 0.41.0 @d3e7788939fbad98a63a8c06badd253764c6dac0 (GitHub main)
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
e4b(t212) 0.48.0 @0afa032ac9333a89bd977c50c23c19df7e00c070
gnf4(t212) 0.41.0 @d3e7788939fbad98a63a8c06badd253764c6dac0
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
 "run_id": "tc1-5090-102",
 "instance_id": "54437423",
 "gpu": "NVIDIA GeForce RTX 5090",
 "driver": "595.71.05",
 "cpu": "Intel(R) Xeon(R) W-2145 CPU @ 3.70GHz",
 "nproc": 16,
 "mem_total_kb": "131573680",
 "cgroup_memory_max": "129340801024",
 "disk_root": "overlay         320G  2.4M  320G   1% /",
 "hostname": "e80889b4a2de",
 "cgroup_cpu_max": "1536000 100000",
 "affinity_cpus": 16,
 "cgroup_cpuset_effective": "0-15",
 "registered_gpu_class": "5090",
 "prereg": "tc1/TC1-PREREG.md",
 "local_box": null,
 "local_snapshot": null
}
```

### Qwen3-30B-A3B (amendment 48: grouped-nf4-gemm's LoRA delta one padded block vs buckets on packed 4,096-token rows, venv-unsloth) (`qwen3padbk`, registered n_layers 48, attention census 192)
- model `Qwen/Qwen3-30B-A3B` @ `ad44e777bcd1`; tokens sha `d2a501eba57d`; N=40; fixture template alpaca seq 4096 (packed rows, amendment 39) micro-batch 1 × accum 4 lr 0.0002 r 16 α 16 optimizer adamw_8bit(lr=0.0002, weight_decay=0.001) schedule=linear warmup_steps=5 autocast False; e4b trainable 642514944; box_class RTX 5090 gpu NVIDIA GeForce RTX 5090
| framework | arm | status | validity | **VERDICT** | matched | init / dtype | step-0 class | N | s/step med(11+) | tok/s | peak GB | J/step | train first→last | held-out 0→final | quality Δ | engagement | trainable | regime | reason / why |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| e4b | fused_attn4_shipped_pk0 | **OK** | VALID | **VALID** | native | native / bfloat16,float32 | — | 40 | 8.742 | 1847.1 | 26.970 | 4265.2 | 1.2644→0.9054 | 1.2893→0.9452 | -0.0092 | patched 48 / kcalls 768 | 642514944 | 4-bit experts (e4b NF4) + NF4 attention |  |
| e4b | fused_attn4_shipped_pk1 | **OK** | VALID | **VALID** | native | native / bfloat16,float32 | — | 40 | 8.166 | 1996.0 | 26.970 | 3871.6 | 1.2644→0.9054 | 1.2893→0.9451 | -0.0093 | patched 48 / kcalls 768 | 642514944 | 4-bit experts (e4b NF4) + NF4 attention |  |
| e4b | fused_attn4_m_pk0 | **OK** | VALID | **VALID** | yes | matched:3407 (complete 12480/12480) / float32 | SAME-BYTES-CLASS (0.0000) | 40 | 11.306 | 1442.7 | 32.563 | 5419.6 | 1.2644→0.9128 | 1.2893→0.9544 | 0.0000 | patched 48 / kcalls 768 | 642514944 | 4-bit experts (e4b NF4) + NF4 attention |  |
| e4b | fused_attn4_m_pk1 | **OK** | VALID | **VALID** | yes | matched:3407 (complete 12480/12480) / float32 | SAME-BYTES-CLASS (0.0000) | 40 | 10.112 | 1607.7 | 28.228 | 4689.6 | 1.2644→0.9127 | 1.2893→0.9540 | -0.0004 | patched 48 / kcalls 768 | 642514944 | 4-bit experts (e4b NF4) + NF4 attention |  |
| e4b | fused_attn4_m_pk1_d2 | **OK** | VALID | **VALID** | yes | matched:3407 (complete 12480/12480) / float32 | SAME-BYTES-CLASS (0.0000) | 40 | 10.118 | 1607.7 | 28.228 | 4714.2 | 1.2644→0.9128 | 1.2893→0.9541 | -0.0003 | patched 48 / kcalls 768 | 642514944 | 4-bit experts (e4b NF4) + NF4 attention |  |
| e4b | fused_attn4_m_pk0_d2 | **OK** | VALID | **VALID** | yes | matched:3407 (complete 12480/12480) / float32 | SAME-BYTES-CLASS (0.0000) | 40 | 11.338 | 1440.8 | 32.477 | 5434.5 | 1.2644→0.9127 | 1.2893→0.9545 | 0.0001 | patched 48 / kcalls 768 | 642514944 | 4-bit experts (e4b NF4) + NF4 attention |  |
| e4b | fused_attn4_shipped_pk1_d2 | **OK** | VALID | **VALID** | native | native / bfloat16,float32 | — | 40 | 8.154 | 1999.6 | 26.970 | 3824.5 | 1.2644→0.9057 | 1.2893→0.9456 | -0.0088 | patched 48 / kcalls 768 | 642514944 | 4-bit experts (e4b NF4) + NF4 attention |  |
| e4b | fused_attn4_shipped_pk0_d2 | **OK** | VALID | **VALID** | native | native / bfloat16,float32 | — | 40 | 8.748 | 1866.9 | 26.970 | 4170.9 | 1.2644→0.9057 | 1.2893→0.9454 | -0.0090 | patched 48 / kcalls 768 | 642514944 | 4-bit experts (e4b NF4) + NF4 attention |  |
- prologue `e4b/fused_attn4_shipped_pk0` **158.9 s** before step 1 (31% of the arm): c1_before 96.0, c1_after 48.4, load_weights 28.5, eval0 12.1; unattributed 2.189; budget 1260.0
- prologue `e4b/fused_attn4_shipped_pk1` **145.2 s** before step 1 (30% of the arm): c1_before 97.9, c1_after 48.8, load_weights 20.9, attn4 6.0; unattributed 1.965; budget 1260.0
- prologue `e4b/fused_attn4_m_pk0` **149.2 s** before step 1 (24% of the arm): c1_before 97.9, c1_after 48.8, load_weights 20.9, eval0 7.5; unattributed 2.163; budget 1260.0
- prologue `e4b/fused_attn4_m_pk1` **148.5 s** before step 1 (26% of the arm): c1_before 98.3, c1_after 49.4, load_weights 20.7, eval0 7.0; unattributed 2.031; budget 1260.0
- prologue `e4b/fused_attn4_m_pk1_d2` **148.2 s** before step 1 (26% of the arm): c1_before 98.1, c1_after 49.3, load_weights 20.8, eval0 7.0; unattributed 1.964; budget 1260.0
- prologue `e4b/fused_attn4_m_pk0_d2` **147.3 s** before step 1 (24% of the arm): c1_before 97.5, c1_after 49.1, load_weights 20.4, eval0 7.5; unattributed 1.963; budget 1260.0
- prologue `e4b/fused_attn4_shipped_pk1_d2` **145.7 s** before step 1 (30% of the arm): c1_before 98.4, c1_after 49.1, load_weights 20.8, attn4 5.9; unattributed 1.962; budget 1260.0
- prologue `e4b/fused_attn4_shipped_pk0_d2` **144.8 s** before step 1 (29% of the arm): c1_before 98.0, c1_after 49.1, load_weights 20.2, eval0 6.1; unattributed 1.952; budget 1260.0
- draws (R1): `e4b/fused_attn4_shipped_pk0` STABLE (8.742/8.748 s, |Δ|/mean 0.1% vs 5%); `e4b/fused_attn4_shipped_pk1` STABLE (8.166/8.154 s, |Δ|/mean 0.1% vs 5%); `e4b/fused_attn4_m_pk0` STABLE (11.306/11.338 s, |Δ|/mean 0.3% vs 5%); `e4b/fused_attn4_m_pk1` STABLE (10.112/10.118 s, |Δ|/mean 0.1% vs 5%)
- e4b internal parity: NO-ARM
- **NO MATCHED POSITION QUOTED (unsloth)** — not quoted: e4b STABLE / unsloth no receipt
- **NO MATCHED POSITION QUOTED (hf)** — not quoted: e4b STABLE / hf no receipt
- **NO MATCHED POSITION QUOTED (axolotl)** — not quoted: e4b STABLE / axolotl no receipt
- equivalence vs `e4b/fused_attn4_m` (I/H; EQUIVALENT iff both ≤ band = max(0.005, 3 × fused-vs-reference |Δ|) = none (no reference arm: COMPARABLE is the ceiling); COMPARABLE ≤ 0.05; draw-noise floor 0.0005): `e4b/fused_attn4_m_pk1` **COMPARABLE** (median step |Δ| 0.0003, |Δ held-out at N| 0.0004, step-0 0.0000 SAME-BYTES-CLASS, |Δ loss at step 2| 0.0003, paired rows mean -0.0004 ± 0.0003 SE over 8, favouring arm 6 / anchor 2) — no reference arm ran: the EQUIVALENT band cannot be set, COMPARABLE is the ceiling; `e4b/fused_attn4_m_pk1_d2` **COMPARABLE** (median step |Δ| 0.0004, |Δ held-out at N| 0.0003, step-0 0.0000 SAME-BYTES-CLASS, |Δ loss at step 2| 0.0015, paired rows mean -0.0003 ± 0.0004 SE over 8, favouring arm 4 / anchor 4) — no reference arm ran: the EQUIVALENT band cannot be set, COMPARABLE is the ceiling; `e4b/fused_attn4_m_pk0_d2` **COMPARABLE** (median step |Δ| 0.0003, |Δ held-out at N| 0.0001, step-0 0.0000 SAME-BYTES-CLASS, |Δ loss at step 2| 0.0005, paired rows mean +0.0001 ± 0.0002 SE over 8, favouring arm 4 / anchor 4) — no reference arm ran: the EQUIVALENT band cannot be set, COMPARABLE is the ceiling
- matched_init_sha (B, name-free, canonical slot order): anchor `f7832488926eda91`; `e4b/fused_attn4_m_pk0` same; `e4b/fused_attn4_m_pk1` same; `e4b/fused_attn4_m_pk1_d2` same; `e4b/fused_attn4_m_pk0_d2` same
- frozen base (R5), anchor `e4b/fused_attn4_m`: gate_up nf4/64 sha 8c3b3fd78e89 control detects, down nf4/64 sha 59569d810a1a control detects, q_proj nf4/64+dq sha 5c5e402ce51d control detects
- frozen base `e4b/fused_attn4_shipped_pk0`: gate_up **SAME-BYTES** (nf4/64) control detects, down **SAME-BYTES** (nf4/64) control detects, q_proj **SAME-BYTES** (nf4/64+dq) control detects
- frozen base `e4b/fused_attn4_shipped_pk1`: gate_up **SAME-BYTES** (nf4/64) control detects, down **SAME-BYTES** (nf4/64) control detects, q_proj **SAME-BYTES** (nf4/64+dq) control detects
- frozen base `e4b/fused_attn4_m_pk1`: gate_up **SAME-BYTES** (nf4/64) control detects, down **SAME-BYTES** (nf4/64) control detects, q_proj **SAME-BYTES** (nf4/64+dq) control detects
- frozen base `e4b/fused_attn4_m_pk1_d2`: gate_up **SAME-BYTES** (nf4/64) control detects, down **SAME-BYTES** (nf4/64) control detects, q_proj **SAME-BYTES** (nf4/64+dq) control detects
- frozen base `e4b/fused_attn4_m_pk0_d2`: gate_up **SAME-BYTES** (nf4/64) control detects, down **SAME-BYTES** (nf4/64) control detects, q_proj **SAME-BYTES** (nf4/64+dq) control detects
- frozen base `e4b/fused_attn4_shipped_pk1_d2`: gate_up **SAME-BYTES** (nf4/64) control detects, down **SAME-BYTES** (nf4/64) control detects, q_proj **SAME-BYTES** (nf4/64+dq) control detects
- frozen base `e4b/fused_attn4_shipped_pk0_d2`: gate_up **SAME-BYTES** (nf4/64) control detects, down **SAME-BYTES** (nf4/64) control detects, q_proj **SAME-BYTES** (nf4/64+dq) control detects

## Verdicts
| family | arm | VERDICT | validity | quality | note |
|---|---|---|---|---|---|
| qwen3padbk | e4b/fused_attn4_shipped_pk0 | **VALID** | VALID | -0.0092 |  |
| qwen3padbk | e4b/fused_attn4_shipped_pk1 | **VALID** | VALID | -0.0093 |  |
| qwen3padbk | e4b/fused_attn4_m_pk0 | **VALID** | VALID | 0.0000 |  |
| qwen3padbk | e4b/fused_attn4_m_pk1 | **VALID** | VALID | -0.0004 |  |
| qwen3padbk | e4b/fused_attn4_m_pk1_d2 | **VALID** | VALID | -0.0003 |  |
| qwen3padbk | e4b/fused_attn4_m_pk0_d2 | **VALID** | VALID | 0.0001 |  |
| qwen3padbk | e4b/fused_attn4_shipped_pk1_d2 | **VALID** | VALID | -0.0088 |  |
| qwen3padbk | e4b/fused_attn4_shipped_pk0_d2 | **VALID** | VALID | -0.0090 |  |

## Predictions P115 / P116 / P117 / P118 (TC1-PREREG amendment 48: the LoRA delta one padded block vs buckets on packed rows; scored mechanically)
| prediction | family | verdict | evidence |
|---|---|---|---|
| P115 | qwen3padbk | **HELD** | matched: peak pk0 32.520 / pk1 28.228 GB, drop 4.292 vs [3.0, 99.0] |
| P116 | qwen3padbk | **HELD** | matched: pk1 / pk0 0.893 [0.892, 0.895 over 4 cross-draw ratios] vs [0.0, 1.02]; s/step pk0 11.306 / 11.338 (within 0.3%), pk1 10.112 / 10.118 (within 0.1%); peak pk0 32.52 / pk1 28.23 GB |
| P117 | qwen3padbk | **HELD** | shipped: pk1 / pk0 0.933 [0.932, 0.934 over 4 cross-draw ratios] vs [0.0, 1.02]; s/step pk0 8.742 / 8.748 (within 0.1%), pk1 8.166 / 8.154 (within 0.1%); peak pk0 26.97 / pk1 26.97 GB |
| P118 | qwen3padbk | **HELD** | matched: mean held-out pk1 - pk0 -0.0004 (|.| <= 0.005); held-out at N pk0 [0.9544, 0.9545] pk1 [0.954, 0.9541]; shipped: mean held-out pk1 - pk0 +0.0000 (|.| <= 0.005); held-out at N pk0 [0.9452, 0.9454] pk1 [0.9451, 0.9456] |

## Load gate (TC1-PREREG amendment 33): 0 draw(s) voided for host load and run again
Each line: the arm, the attempt, the median host load1 over that attempt's run, the gate. A VOID attempt's files are in loadvoid/; the last attempt of each arm stands whatever its load.

- `qwen3padbk/e4b/fused_attn4_shipped_pk0 attempt 0 load1_median 1.33 gate 6.0 status ok over 0`
- `qwen3padbk/e4b/fused_attn4_shipped_pk1 attempt 0 load1_median 1.38 gate 6.0 status ok over 0`
- `qwen3padbk/e4b/fused_attn4_m_pk0 attempt 0 load1_median 1.31 gate 6.0 status ok over 0`
- `qwen3padbk/e4b/fused_attn4_m_pk1 attempt 0 load1_median 1.32 gate 6.0 status ok over 0`
- `qwen3padbk/e4b/fused_attn4_m_pk1_d2 attempt 0 load1_median 1.34 gate 6.0 status ok over 0`
- `qwen3padbk/e4b/fused_attn4_m_pk0_d2 attempt 0 load1_median 1.33 gate 6.0 status ok over 0`
- `qwen3padbk/e4b/fused_attn4_shipped_pk1_d2 attempt 0 load1_median 1.42 gate 6.0 status ok over 0`
- `qwen3padbk/e4b/fused_attn4_shipped_pk0_d2 attempt 0 load1_median 1.28 gate 6.0 status ok over 0`
