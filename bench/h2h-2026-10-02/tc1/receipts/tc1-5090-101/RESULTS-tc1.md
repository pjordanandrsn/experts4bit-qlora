# TC1 — Qwen3-30B-A3B on one RTX 5090: e4b vs Unsloth vs HF+PEFT vs axolotl at matched work, init and adapter precision (tc1)
Rule (tc1/TC1-PREREG.md): status per attempt in the vocabulary OK / REFUSED / OOM / INSTALL_FAILED / LOAD_FAULT / HARNESS_ERROR / ALARM / NOT_RUN; VALID/VOID per tp4's predicates plus the matched-set predicates (R3: matched init complete, fp32 adapters, step-0 held-out within 0.005 of e4b/reference_attn4_m); VERDICT exactly one of VALID / VOID / QUALITY_FAIL / OOM / UNSUPPORTED / HARNESS_ERROR / ALARM / NOT_RUN (QUALITY_FAIL = held-out |Δ| at N > 0.05 vs e4b/fused_attn4_m); positions = s/step ratios other/e4b from the medians over both draws (stability |d1-d2|/mean ≤ 5 % e4b / 10 % others, else UNSTABLE), quoted only when both arms are VALID; equivalence vs e4b/reference_attn4_m EQUIVALENT ≤ 0.02 / COMPARABLE ≤ 0.05 / else DIVERGENT; frozen base SAME-BYTES / DIFFERENT / N-A per slot; predictions P1–P10 (+ P1b) scored HELD / FALSIFIED / UNTESTED. VOID never enters a ratio or an equivalence reading. Nothing is licensed; no cross-box number is divided into these.
`versions.txt`
```
e4b 0.48.0 @087e04570df153ffab300c50a4dcd8a38788f7b2 (GitHub main)
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
e4b(t212) 0.48.0 @087e04570df153ffab300c50a4dcd8a38788f7b2
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
 "run_id": "tc1-5090-101",
 "instance_id": "54425329",
 "gpu": "NVIDIA GeForce RTX 5090",
 "driver": "580.65.06",
 "cpu": "AMD Ryzen 9 7950X 16-Core Processor",
 "nproc": 32,
 "mem_total_kb": "130979720",
 "cgroup_memory_max": "128756744192",
 "disk_root": "overlay         320G  2.4M  320G   1% /",
 "hostname": "5802a89d3f48",
 "cgroup_cpu_max": "3071999 100000",
 "affinity_cpus": 32,
 "cgroup_cpuset_effective": "0-31",
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
| e4b | fused_attn4_shipped_pk0 | **OK** | VALID | **VALID** | native | native / bfloat16,float32 | — | 40 | 8.458 | 1919.6 | 26.970 | 4448.7 | 1.2644→0.9054 | 1.2893→0.9454 | -0.0089 | patched 48 / kcalls 768 | 642514944 | 4-bit experts (e4b NF4) + NF4 attention |  |
| e4b | fused_attn4_shipped_pk1 | **OK** | VOID | **VOID** | native | native / bfloat16,float32 | — | 40 | 7.916 | 2060.9 | 26.970 | 3777.2 | 1.2644→0.9055 | 1.2893→0.9450 | -0.0093 | patched 48 / kcalls 768 | 642514944 | 4-bit experts (e4b NF4) + NF4 attention | the per-expert LoRA loop ran on step(s) [1, 2, 3, 4, 5, 6]... (lora_loop_share max 1.000) |
| e4b | fused_attn4_m_pk0 | **OK** | VALID | **VALID** | yes | matched:3407 (complete 12480/12480) / float32 | SAME-BYTES-CLASS (0.0000) | 40 | 10.948 | 1489.2 | 32.530 | 5436.4 | 1.2644→0.9130 | 1.2893→0.9544 | 0.0000 | patched 48 / kcalls 768 | 642514944 | 4-bit experts (e4b NF4) + NF4 attention |  |
| e4b | fused_attn4_m_pk1 | **OK** | VOID | **VOID** | yes | matched:3407 (complete 12480/12480) / float32 | SAME-BYTES-CLASS (0.0000) | 40 | 9.771 | 1660.1 | 28.228 | 4558.6 | 1.2644→0.9133 | 1.2893→0.9547 | 0.0003 | patched 48 / kcalls 768 | 642514944 | 4-bit experts (e4b NF4) + NF4 attention | the per-expert LoRA loop ran on step(s) [1, 2, 3, 4, 5, 6]... (lora_loop_share max 1.000) |
| e4b | fused_attn4_m_pk1_d2 | **OK** | VOID | **VOID** | yes | matched:3407 (complete 12480/12480) / float32 | SAME-BYTES-CLASS (0.0000) | 40 | 9.777 | 1661.3 | 28.228 | 4578.0 | 1.2644→0.9128 | 1.2893→0.9540 | -0.0004 | patched 48 / kcalls 768 | 642514944 | 4-bit experts (e4b NF4) + NF4 attention | the per-expert LoRA loop ran on step(s) [1, 2, 3, 4, 5, 6]... (lora_loop_share max 1.000) |
| e4b | fused_attn4_m_pk0_d2 | **OK** | VALID | **VALID** | yes | matched:3407 (complete 12480/12480) / float32 | SAME-BYTES-CLASS (0.0000) | 40 | 10.985 | 1487.0 | 32.570 | 5406.8 | 1.2644→0.9127 | 1.2893→0.9545 | 0.0001 | patched 48 / kcalls 768 | 642514944 | 4-bit experts (e4b NF4) + NF4 attention |  |
| e4b | fused_attn4_shipped_pk1_d2 | **OK** | VOID | **VOID** | native | native / bfloat16,float32 | — | 40 | 7.925 | 2059.5 | 26.970 | 3780.4 | 1.2644→0.9056 | 1.2893→0.9454 | -0.0089 | patched 48 / kcalls 768 | 642514944 | 4-bit experts (e4b NF4) + NF4 attention | the per-expert LoRA loop ran on step(s) [1, 2, 3, 4, 5, 6]... (lora_loop_share max 1.000) |
| e4b | fused_attn4_shipped_pk0_d2 | **OK** | VALID | **VALID** | native | native / bfloat16,float32 | — | 40 | 8.456 | 1927.6 | 26.970 | 4142.5 | 1.2644→0.9050 | 1.2893→0.9454 | -0.0090 | patched 48 / kcalls 768 | 642514944 | 4-bit experts (e4b NF4) + NF4 attention |  |
- prologue `e4b/fused_attn4_shipped_pk0` **65.6 s** before step 1 (16% of the arm): c1_before 33.7, c1_after 17.0, load_weights 13.3, eval0 8.1; unattributed 1.216; budget 1260.0
- prologue `e4b/fused_attn4_shipped_pk1` **65.6 s** before step 1 (17% of the arm): c1_before 34.8, c1_after 17.6, load_weights 15.4, eval0 5.0; unattributed 1.232; budget 1260.0
- prologue `e4b/fused_attn4_m_pk0` **67.3 s** before step 1 (13% of the arm): c1_before 35.5, c1_after 17.9, load_weights 13.8, eval0 6.6; unattributed 1.261; budget 1260.0
- prologue `e4b/fused_attn4_m_pk1` **64.9 s** before step 1 (14% of the arm): c1_before 34.1, c1_after 17.8, load_weights 13.6, eval0 6.1; unattributed 1.234; budget 1260.0
- prologue `e4b/fused_attn4_m_pk1_d2` **64.9 s** before step 1 (14% of the arm): c1_before 34.2, c1_after 17.2, load_weights 13.4, eval0 6.1; unattributed 1.214; budget 1260.0
- prologue `e4b/fused_attn4_m_pk0_d2` **65.1 s** before step 1 (13% of the arm): c1_before 33.9, c1_after 17.2, load_weights 13.4, eval0 6.6; unattributed 1.211; budget 1260.0
- prologue `e4b/fused_attn4_shipped_pk1_d2` **62.7 s** before step 1 (16% of the arm): c1_before 33.9, c1_after 17.0, load_weights 13.5, eval0 5.0; unattributed 1.221; budget 1260.0
- prologue `e4b/fused_attn4_shipped_pk0_d2` **63.6 s** before step 1 (16% of the arm): c1_before 34.4, c1_after 17.4, load_weights 13.5, eval0 5.2; unattributed 1.217; budget 1260.0
- draws (R1): `e4b/fused_attn4_shipped_pk0` STABLE (8.458/8.456 s, |Δ|/mean 0.0% vs 5%); `e4b/fused_attn4_shipped_pk1` — (e4b/fused_attn4_shipped_pk1 is VOID); `e4b/fused_attn4_m_pk0` STABLE (10.948/10.985 s, |Δ|/mean 0.3% vs 5%); `e4b/fused_attn4_m_pk1` — (e4b/fused_attn4_m_pk1 is VOID)
- e4b internal parity: NO-ARM
- **NO MATCHED POSITION QUOTED (unsloth)** — not quoted: e4b STABLE / unsloth no receipt
- **NO MATCHED POSITION QUOTED (hf)** — not quoted: e4b STABLE / hf no receipt
- **NO MATCHED POSITION QUOTED (axolotl)** — not quoted: e4b STABLE / axolotl no receipt
- equivalence vs `e4b/fused_attn4_m` (I/H; EQUIVALENT iff both ≤ band = max(0.005, 3 × fused-vs-reference |Δ|) = none (no reference arm: COMPARABLE is the ceiling); COMPARABLE ≤ 0.05; draw-noise floor 0.0001): `e4b/fused_attn4_m_pk1` **N-A** — validity: anchor VALID, arm VOID (VOID never enters an equivalence reading); `e4b/fused_attn4_m_pk1_d2` **N-A** — validity: anchor VALID, arm VOID (VOID never enters an equivalence reading); `e4b/fused_attn4_m_pk0_d2` **COMPARABLE** (median step |Δ| 0.0002, |Δ held-out at N| 0.0001, step-0 0.0000 SAME-BYTES-CLASS, |Δ loss at step 2| 0.0046, paired rows mean +0.0001 ± 0.0003 SE over 8, favouring arm 3 / anchor 5) — no reference arm ran: the EQUIVALENT band cannot be set, COMPARABLE is the ceiling
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
| qwen3padbk | e4b/fused_attn4_shipped_pk0 | **VALID** | VALID | -0.0089 |  |
| qwen3padbk | e4b/fused_attn4_shipped_pk1 | **VOID** | VOID | -0.0093 | the per-expert LoRA loop ran on step(s) [1, 2, 3, 4, 5, 6]... (lora_loop_share max 1.000) |
| qwen3padbk | e4b/fused_attn4_m_pk0 | **VALID** | VALID | 0.0000 |  |
| qwen3padbk | e4b/fused_attn4_m_pk1 | **VOID** | VOID | 0.0003 | the per-expert LoRA loop ran on step(s) [1, 2, 3, 4, 5, 6]... (lora_loop_share max 1.000) |
| qwen3padbk | e4b/fused_attn4_m_pk1_d2 | **VOID** | VOID | -0.0004 | the per-expert LoRA loop ran on step(s) [1, 2, 3, 4, 5, 6]... (lora_loop_share max 1.000) |
| qwen3padbk | e4b/fused_attn4_m_pk0_d2 | **VALID** | VALID | 0.0001 |  |
| qwen3padbk | e4b/fused_attn4_shipped_pk1_d2 | **VOID** | VOID | -0.0089 | the per-expert LoRA loop ran on step(s) [1, 2, 3, 4, 5, 6]... (lora_loop_share max 1.000) |
| qwen3padbk | e4b/fused_attn4_shipped_pk0_d2 | **VALID** | VALID | -0.0090 |  |

## Predictions P115 / P116 / P117 / P118 (TC1-PREREG amendment 48: the LoRA delta one padded block vs buckets on packed rows; scored mechanically)
| prediction | family | verdict | evidence |
|---|---|---|---|
| P115 | qwen3padbk | **UNTESTED** | matched: two stable VALID draws a side are registered -- pk0 STABLE:; pk1 —: e4b/fused_attn4_m_pk1 is VOID |
| P116 | qwen3padbk | **UNTESTED** | matched: two stable VALID draws a side are registered -- pk0 STABLE:; pk1 —: e4b/fused_attn4_m_pk1 is VOID |
| P117 | qwen3padbk | **UNTESTED** | shipped: two stable VALID draws a side are registered -- pk0 STABLE:; pk1 —: e4b/fused_attn4_shipped_pk1 is VOID |
| P118 | qwen3padbk | **UNTESTED** | matched: pk0 STABLE:; pk1 —: e4b/fused_attn4_m_pk1 is VOID; shipped: pk0 STABLE:; pk1 —: e4b/fused_attn4_shipped_pk1 is VOID |

## Load gate (TC1-PREREG amendment 33): 0 draw(s) voided for host load and run again
Each line: the arm, the attempt, the median host load1 over that attempt's run, the gate. A VOID attempt's files are in loadvoid/; the last attempt of each arm stands whatever its load.

- `qwen3padbk/e4b/fused_attn4_shipped_pk0 attempt 0 load1_median 1.26 gate 6.0 status ok over 0`
- `qwen3padbk/e4b/fused_attn4_shipped_pk1 attempt 0 load1_median 1.3 gate 6.0 status ok over 0`
- `qwen3padbk/e4b/fused_attn4_m_pk0 attempt 0 load1_median 1.15 gate 6.0 status ok over 0`
- `qwen3padbk/e4b/fused_attn4_m_pk1 attempt 0 load1_median 1.17 gate 6.0 status ok over 0`
- `qwen3padbk/e4b/fused_attn4_m_pk1_d2 attempt 0 load1_median 1.18 gate 6.0 status ok over 0`
- `qwen3padbk/e4b/fused_attn4_m_pk0_d2 attempt 0 load1_median 1.12 gate 6.0 status ok over 0`
- `qwen3padbk/e4b/fused_attn4_shipped_pk1_d2 attempt 0 load1_median 1.21 gate 6.0 status ok over 0`
- `qwen3padbk/e4b/fused_attn4_shipped_pk0_d2 attempt 0 load1_median 1.19 gate 6.0 status ok over 0`
