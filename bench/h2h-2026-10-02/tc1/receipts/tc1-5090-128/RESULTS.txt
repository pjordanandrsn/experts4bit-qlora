# TC1 — Qwen3-30B-A3B on one RTX 5090: e4b vs Unsloth vs HF+PEFT vs axolotl at matched work, init and adapter precision (tc1)
Rule (tc1/TC1-PREREG.md): status per attempt in the vocabulary OK / REFUSED / OOM / INSTALL_FAILED / LOAD_FAULT / HARNESS_ERROR / ALARM / NOT_RUN; VALID/VOID per tp4's predicates plus the matched-set predicates (R3: matched init complete, fp32 adapters, step-0 held-out within 0.005 of e4b/reference_attn4_m); VERDICT exactly one of VALID / VOID / QUALITY_FAIL / OOM / UNSUPPORTED / HARNESS_ERROR / ALARM / NOT_RUN (QUALITY_FAIL = held-out |Δ| at N > 0.05 vs e4b/fused_attn4_m); positions = s/step ratios other/e4b from the medians over both draws (stability |d1-d2|/mean ≤ 5 % e4b / 10 % others, else UNSTABLE), quoted only when both arms are VALID; equivalence vs e4b/reference_attn4_m EQUIVALENT ≤ 0.02 / COMPARABLE ≤ 0.05 / else DIVERGENT; frozen base SAME-BYTES / DIFFERENT / N-A per slot; predictions P1–P10 (+ P1b) scored HELD / FALSIFIED / UNTESTED. VOID never enters a ratio or an equivalence reading. Nothing is licensed; no cross-box number is divided into these.
`versions.txt`
```
e4b 0.48.0 @434db5f6df5c25ef687a49b08c439d84ced5c574 (GitHub main)
gnf4 0.42.0 @0e6bff33a56a0277b39a0560c37e5af9bee23cf6 (GitHub main)
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
e4b(t212) 0.48.0 @434db5f6df5c25ef687a49b08c439d84ced5c574
gnf4(t212) 0.42.0 @0e6bff33a56a0277b39a0560c37e5af9bee23cf6
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
 "run_id": "tc1-5090-128",
 "instance_id": "54640352",
 "gpu": "NVIDIA GeForce RTX 5090",
 "driver": "595.91.07",
 "cpu": "AMD EPYC 7K62 48-Core Processor",
 "nproc": 96,
 "mem_total_kb": "527994548",
 "cgroup_memory_max": "259519414272",
 "disk_root": "overlay         320G   53M  320G   1% /",
 "hostname": "44dcb1a7a167",
 "cgroup_cpu_max": "2304000 100000",
 "affinity_cpus": 96,
 "cgroup_cpuset_effective": "0-95",
 "registered_gpu_class": "5090",
 "prereg": "tc1/TC1-PREREG.md",
 "local_box": null,
 "local_snapshot": null
}
```

### Qwen3-30B-A3B (amendment 64: the shipped arm at the field recipe in torch 2.8 with Hugging Face's checkpoint, the reentrant checkpoint alone, and with its inputs in host memory) (`qwen3ckptre28`, registered n_layers 48, attention census 192)
- model `Qwen/Qwen3-30B-A3B` @ `ad44e777bcd1`; tokens sha `bfc742f67e37`; N=60; fixture template alpaca seq 2048 micro-batch 2 × accum 4 lr 0.0002 r 16 α 16 optimizer adamw_8bit(lr=0.0002, weight_decay=0.001) schedule=linear warmup_steps=5 autocast False; e4b trainable 642514944; box_class RTX 5090 gpu NVIDIA GeForce RTX 5090
| framework | arm | status | validity | **VERDICT** | matched | init / dtype | step-0 class | N | s/step med(11+) | tok/s | peak GB | J/step | train first→last | held-out 0→final | quality Δ | engagement | trainable | regime | reason / why |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| e4b | fused_attn4_shipped_r0 | **OK** | VALID | **VALID** | native | native / bfloat16,float32 | — | 60 | 3.811 | 388.3 | 23.271 | 584.8 | 2.0506→0.8154 | 1.9251→0.7578 | 0.0000 | patched 48 / kcalls 768 | 642514944 | 4-bit experts (e4b NF4) + NF4 attention |  |
| e4b | fused_attn4_shipped_rr | **OK** | VALID | **VALID** | native | native / bfloat16,float32 | — | 60 | 3.247 | 480.9 | 23.271 | 463.2 | 2.0506→0.8177 | 1.9251→0.7553 | -0.0024 | patched 48 / kcalls 768 | 642514944 | 4-bit experts (e4b NF4) + NF4 attention |  |
| e4b | fused_attn4_shipped_r1 | **OK** | VALID | **VALID** | native | native / bfloat16,float32 | — | 60 | 3.353 | 464.9 | 23.054 | 513.3 | 2.0506→0.8160 | 1.9251→0.7542 | -0.0035 | patched 48 / kcalls 768 | 642514944 | 4-bit experts (e4b NF4) + NF4 attention |  |
| e4b | fused_attn4_shipped_r1_d2 | **OK** | VALID | **VALID** | native | native / bfloat16,float32 | — | 60 | 3.356 | 463.4 | 23.054 | 513.8 | 2.0506→0.8172 | 1.9251→0.7532 | -0.0045 | patched 48 / kcalls 768 | 642514944 | 4-bit experts (e4b NF4) + NF4 attention |  |
| e4b | fused_attn4_shipped_rr_d2 | **OK** | VALID | **VALID** | native | native / bfloat16,float32 | — | 60 | 3.215 | 484.4 | 23.271 | 463.2 | 2.0506→0.8153 | 1.9251→0.7545 | -0.0032 | patched 48 / kcalls 768 | 642514944 | 4-bit experts (e4b NF4) + NF4 attention |  |
| e4b | fused_attn4_shipped_r0_d2 | **OK** | VALID | **VALID** | native | native / bfloat16,float32 | — | 60 | 3.899 | 375.4 | 23.271 | 613.7 | 2.0506→0.8153 | 1.9251→0.7574 | -0.0004 | patched 48 / kcalls 768 | 642514944 | 4-bit experts (e4b NF4) + NF4 attention |  |
- prologue `e4b/fused_attn4_shipped_r0` **128.2 s** before step 1 (22% of the arm): c1_before 69.3, load_weights 24.5, c1_after 19.0, eval0 14.7; unattributed 1.478; budget 1260.0
- prologue `e4b/fused_attn4_shipped_rr` **115.0 s** before step 1 (23% of the arm): c1_before 69.7, load_weights 23.7, c1_after 19.4, attn4 7.4; unattributed 1.441; budget 1260.0
- prologue `e4b/fused_attn4_shipped_r1` **114.7 s** before step 1 (22% of the arm): c1_before 69.0, load_weights 23.6, c1_after 18.9, attn4 7.4; unattributed 1.448; budget 1260.0
- prologue `e4b/fused_attn4_shipped_r1_d2` **114.4 s** before step 1 (22% of the arm): c1_before 68.0, c1_after 27.1, load_weights 24.5, attn4 7.7; unattributed 1.498; budget 1260.0
- prologue `e4b/fused_attn4_shipped_rr_d2` **114.6 s** before step 1 (23% of the arm): c1_before 69.7, load_weights 23.5, c1_after 18.9, attn4 7.3; unattributed 1.418; budget 1260.0
- prologue `e4b/fused_attn4_shipped_r0_d2` **114.7 s** before step 1 (21% of the arm): c1_before 69.2, load_weights 23.3, c1_after 12.3, attn4 7.4; unattributed 1.454; budget 1260.0
- draws (R1): `e4b/fused_attn4_shipped_r0` STABLE (3.811/3.899 s, |Δ|/mean 2.3% vs 5%); `e4b/fused_attn4_shipped_rr` STABLE (3.247/3.215 s, |Δ|/mean 1.0% vs 5%); `e4b/fused_attn4_shipped_r1` STABLE (3.353/3.356 s, |Δ|/mean 0.1% vs 5%)
- e4b internal parity: NO-ARM
- **NO MATCHED POSITION QUOTED (unsloth)** — not quoted: e4b STABLE / unsloth no receipt
- **NO MATCHED POSITION QUOTED (hf)** — not quoted: e4b STABLE / hf no receipt
- **NO MATCHED POSITION QUOTED (axolotl)** — not quoted: e4b STABLE / axolotl no receipt
- frozen base (R5), anchor `e4b/fused_attn4_m`: gate_up nf4/64+dq sha e2bed944143e control detects, down nf4/64+dq sha f2ade29dafdb control detects, q_proj nf4/64+dq sha 5c5e402ce51d control detects
- frozen base `e4b/fused_attn4_shipped_rr`: gate_up **SAME-BYTES** (nf4/64+dq) control detects, down **SAME-BYTES** (nf4/64+dq) control detects, q_proj **SAME-BYTES** (nf4/64+dq) control detects
- frozen base `e4b/fused_attn4_shipped_r1`: gate_up **SAME-BYTES** (nf4/64+dq) control detects, down **SAME-BYTES** (nf4/64+dq) control detects, q_proj **SAME-BYTES** (nf4/64+dq) control detects
- frozen base `e4b/fused_attn4_shipped_r1_d2`: gate_up **SAME-BYTES** (nf4/64+dq) control detects, down **SAME-BYTES** (nf4/64+dq) control detects, q_proj **SAME-BYTES** (nf4/64+dq) control detects
- frozen base `e4b/fused_attn4_shipped_rr_d2`: gate_up **SAME-BYTES** (nf4/64+dq) control detects, down **SAME-BYTES** (nf4/64+dq) control detects, q_proj **SAME-BYTES** (nf4/64+dq) control detects
- frozen base `e4b/fused_attn4_shipped_r0_d2`: gate_up **SAME-BYTES** (nf4/64+dq) control detects, down **SAME-BYTES** (nf4/64+dq) control detects, q_proj **SAME-BYTES** (nf4/64+dq) control detects

## Verdicts
| family | arm | VERDICT | validity | quality | note |
|---|---|---|---|---|---|
| qwen3ckptre28 | e4b/fused_attn4_shipped_r0 | **VALID** | VALID | 0.0000 |  |
| qwen3ckptre28 | e4b/fused_attn4_shipped_rr | **VALID** | VALID | -0.0024 |  |
| qwen3ckptre28 | e4b/fused_attn4_shipped_r1 | **VALID** | VALID | -0.0035 |  |
| qwen3ckptre28 | e4b/fused_attn4_shipped_r1_d2 | **VALID** | VALID | -0.0045 |  |
| qwen3ckptre28 | e4b/fused_attn4_shipped_rr_d2 | **VALID** | VALID | -0.0032 |  |
| qwen3ckptre28 | e4b/fused_attn4_shipped_r0_d2 | **VALID** | VALID | -0.0004 |  |

## Amendment 64: Hugging Face's checkpoint, the reentrant checkpoint alone, and with its inputs in host memory -- packed rows (torch 2.12) and the field recipe in torch 2.8 (descriptive)
| family | arm | VERDICT | s/step (11..N) | run peak GB | train | held-out N | checkpoint |
|---|---|---|---|---|---|---|---|
| qwen3ckptre28 | e4b/fused_attn4_shipped_r0 | VALID | 3.811 | 23.271 | 23.271 | 0.75775 | 0 [] |
| qwen3ckptre28 | e4b/fused_attn4_shipped_rr | VALID | 3.247 | 23.271 | 23.271 | 0.75532 | reentrant ['reentrant_checkpoint'] |
| qwen3ckptre28 | e4b/fused_attn4_shipped_r1 | VALID | 3.353 | 23.054 | 23.054 | 0.75423 | 1 ['offloaded_checkpoint'] |
| qwen3ckptre28 | e4b/fused_attn4_shipped_r1_d2 | VALID | 3.356 | 23.054 | 23.054 | 0.75322 | 1 ['offloaded_checkpoint'] |
| qwen3ckptre28 | e4b/fused_attn4_shipped_rr_d2 | VALID | 3.215 | 23.271 | 23.271 | 0.75452 | reentrant ['reentrant_checkpoint'] |
| qwen3ckptre28 | e4b/fused_attn4_shipped_r0_d2 | VALID | 3.899 | 23.271 | 23.271 | 0.75739 | 0 [] |

## Predictions P178-P183 (TC1-PREREG amendment 64: the reentrant checkpoint as the default checkpoint; scored mechanically)
| prediction | family | verdict | evidence |
|---|---|---|---|
| P182 | qwen3ckptre28 | **HELD** | rr / r0 0.838 [0.825, 0.852 over 4 cross-draw ratios] vs <= 1.0; s/step rr 3.247 / 3.215, r0 3.811 / 3.899; r0 device busy vs the timed step 0.404 (premise <= 0.9) |
| P183 | qwen3ckptre28 | **HELD** | mean held-out at N r0 0.75757, rr 0.75492, r1 0.75372; largest difference 0.00385 (<= 0.005) |

## Load gate (TC1-PREREG amendment 33): 3 draw(s) voided for host load and run again
Each line: the arm, the attempt, the median host load1 over that attempt's run, the gate. A VOID attempt's files are in loadvoid/; the last attempt of each arm stands whatever its load.

- `qwen3ckptre28/e4b/fused_attn4_shipped_r0 attempt 0 load1_median 4.57 gate 6.0 status ok over 0`
- `qwen3ckptre28/e4b/fused_attn4_shipped_rr attempt 0 load1_median 5.41 gate 6.0 status ok over 0`
- `qwen3ckptre28/e4b/fused_attn4_shipped_r1 attempt 0 load1_median 8.56 gate 6.0 status ok over 1`
- `qwen3ckptre28/e4b/fused_attn4_shipped_r1 attempt 0 VOID (host load1 median 8.56 > 6.0): re-run 1 of 2`
- `qwen3ckptre28/e4b/fused_attn4_shipped_r1 attempt 1 load1_median 4.25 gate 6.0 status ok over 0`
- `qwen3ckptre28/e4b/fused_attn4_shipped_r1_d2 attempt 0 load1_median 7.87 gate 6.0 status ok over 1`
- `qwen3ckptre28/e4b/fused_attn4_shipped_r1_d2 attempt 0 VOID (host load1 median 7.87 > 6.0): re-run 1 of 2`
- `qwen3ckptre28/e4b/fused_attn4_shipped_r1_d2 attempt 1 load1_median 8.21 gate 6.0 status ok over 1`
- `qwen3ckptre28/e4b/fused_attn4_shipped_r1_d2 attempt 1 VOID (host load1 median 8.21 > 6.0): re-run 2 of 2`
- `qwen3ckptre28/e4b/fused_attn4_shipped_r1_d2 attempt 2 load1_median 10.12 gate 6.0 status ok over 1`
- `qwen3ckptre28/e4b/fused_attn4_shipped_rr_d2 attempt 0 load1_median 4.86 gate 6.0 status ok over 0`
- `qwen3ckptre28/e4b/fused_attn4_shipped_r0_d2 attempt 0 load1_median 3.16 gate 6.0 status ok over 0`
