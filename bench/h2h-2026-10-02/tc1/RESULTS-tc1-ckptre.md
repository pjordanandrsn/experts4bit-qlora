# TC1 — Qwen3-30B-A3B on one RTX 5090: e4b vs Unsloth vs HF+PEFT vs axolotl at matched work, init and adapter precision (tc1)
Rule (tc1/TC1-PREREG.md): status per attempt in the vocabulary OK / REFUSED / OOM / INSTALL_FAILED / LOAD_FAULT / HARNESS_ERROR / ALARM / NOT_RUN; VALID/VOID per tp4's predicates plus the matched-set predicates (R3: matched init complete, fp32 adapters, step-0 held-out within 0.005 of e4b/reference_attn4_m); VERDICT exactly one of VALID / VOID / QUALITY_FAIL / OOM / UNSUPPORTED / HARNESS_ERROR / ALARM / NOT_RUN (QUALITY_FAIL = held-out |Δ| at N > 0.05 vs e4b/fused_attn4_m); positions = s/step ratios other/e4b from the medians over both draws (stability |d1-d2|/mean ≤ 5 % e4b / 10 % others, else UNSTABLE), quoted only when both arms are VALID; equivalence vs e4b/reference_attn4_m EQUIVALENT ≤ 0.02 / COMPARABLE ≤ 0.05 / else DIVERGENT; frozen base SAME-BYTES / DIFFERENT / N-A per slot; predictions P1–P10 (+ P1b) scored HELD / FALSIFIED / UNTESTED. VOID never enters a ratio or an equivalence reading. Nothing is licensed; no cross-box number is divided into these.
`versions.txt`
```
e4b 0.48.0 @53b3fb38ffa766f581c33d1e974b7f0dd8ae2110 (GitHub main)
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
e4b(t212) 0.48.0 @53b3fb38ffa766f581c33d1e974b7f0dd8ae2110
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
 "run_id": "tc1-5090-127",
 "instance_id": "54628275",
 "gpu": "NVIDIA GeForce RTX 5090",
 "driver": "580.119.02",
 "cpu": "AMD EPYC 9655 96-Core Processor",
 "nproc": 48,
 "mem_total_kb": "113303332",
 "cgroup_memory_max": "111380791296",
 "disk_root": "overlay         320G   53M  320G   1% /",
 "hostname": "fd4f26d496f3",
 "cgroup_cpu_max": "4608000 100000",
 "affinity_cpus": 48,
 "cgroup_cpuset_effective": "0-47",
 "registered_gpu_class": "5090",
 "prereg": "tc1/TC1-PREREG.md",
 "local_box": null,
 "local_snapshot": null
}
```

### Qwen3-30B-A3B (amendment 62: the shipped arm at the field recipe with Hugging Face's checkpoint, the reentrant checkpoint alone, and the reentrant checkpoint with its inputs in host memory) (`qwen3ckptre`, registered n_layers 48, attention census 192)
- model `Qwen/Qwen3-30B-A3B` @ `ad44e777bcd1`; tokens sha `bfc742f67e37`; N=60; fixture template alpaca seq 2048 micro-batch 2 × accum 4 lr 0.0002 r 16 α 16 optimizer adamw_8bit(lr=0.0002, weight_decay=0.001) schedule=linear warmup_steps=5 autocast False; e4b trainable 642514944; box_class RTX 5090 gpu NVIDIA GeForce RTX 5090
| framework | arm | status | validity | **VERDICT** | matched | init / dtype | step-0 class | N | s/step med(11+) | tok/s | peak GB | J/step | train first→last | held-out 0→final | quality Δ | engagement | trainable | regime | reason / why |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| e4b | fused_attn4_shipped_r0 | **OK** | VALID | **VALID** | native | native / bfloat16,float32 | — | 60 | 1.773 | 851.6 | 23.321 | 722.0 | 2.0722→0.8180 | 1.9592→0.7548 | 0.0000 | patched 48 / kcalls 768 | 642514944 | 4-bit experts (e4b NF4) + NF4 attention |  |
| e4b | fused_attn4_shipped_rr | **OK** | VALID | **VALID** | native | native / bfloat16,float32 | — | 60 | 1.579 | 1002.8 | 23.321 | 636.5 | 2.0722→0.8170 | 1.9592→0.7557 | 0.0009 | patched 48 / kcalls 768 | 642514944 | 4-bit experts (e4b NF4) + NF4 attention |  |
| e4b | fused_attn4_shipped_r1 | **OK** | VALID | **VALID** | native | native / bfloat16,float32 | — | 60 | 1.618 | 981.8 | 23.104 | 644.8 | 2.0722→0.8188 | 1.9592→0.7577 | 0.0029 | patched 48 / kcalls 768 | 642514944 | 4-bit experts (e4b NF4) + NF4 attention |  |
| e4b | fused_attn4_shipped_r1_d2 | **OK** | VALID | **VALID** | native | native / bfloat16,float32 | — | 60 | 1.619 | 982.8 | 23.104 | 658.3 | 2.0722→0.8184 | 1.9592→0.7556 | 0.0008 | patched 48 / kcalls 768 | 642514944 | 4-bit experts (e4b NF4) + NF4 attention |  |
| e4b | fused_attn4_shipped_rr_d2 | **OK** | VALID | **VALID** | native | native / bfloat16,float32 | — | 60 | 1.598 | 998.0 | 23.321 | 649.2 | 2.0722→0.8171 | 1.9592→0.7570 | 0.0022 | patched 48 / kcalls 768 | 642514944 | 4-bit experts (e4b NF4) + NF4 attention |  |
| e4b | fused_attn4_shipped_r0_d2 | **OK** | VALID | **VALID** | native | native / bfloat16,float32 | — | 60 | 1.756 | 902.5 | 23.321 | 667.8 | 2.0722→0.8194 | 1.9592→0.7564 | 0.0017 | patched 48 / kcalls 768 | 642514944 | 4-bit experts (e4b NF4) + NF4 attention |  |
- prologue `e4b/fused_attn4_shipped_r0` **61.0 s** before step 1 (35% of the arm): c1_before 26.9, load_weights 16.1, c1_after 13.8, eval0 7.7; unattributed 1.454; budget 1260.0
- prologue `e4b/fused_attn4_shipped_rr` **56.4 s** before step 1 (37% of the arm): c1_before 27.9, load_weights 18.1, c1_after 13.9, preamble 2.8; unattributed 1.093; budget 1260.0
- prologue `e4b/fused_attn4_shipped_r1` **48.9 s** before step 1 (33% of the arm): c1_before 27.4, c1_after 13.8, load_weights 11.1, preamble 2.7; unattributed 1.09; budget 1260.0
- prologue `e4b/fused_attn4_shipped_r1_d2` **48.7 s** before step 1 (33% of the arm): c1_before 27.2, c1_after 13.7, load_weights 11.1, preamble 2.7; unattributed 1.088; budget 1260.0
- prologue `e4b/fused_attn4_shipped_rr_d2` **49.0 s** before step 1 (33% of the arm): c1_before 27.4, c1_after 13.8, load_weights 11.0, preamble 2.8; unattributed 1.098; budget 1260.0
- prologue `e4b/fused_attn4_shipped_r0_d2` **48.9 s** before step 1 (31% of the arm): c1_before 27.4, c1_after 13.6, load_weights 11.0, preamble 2.8; unattributed 1.089; budget 1260.0
- draws (R1): `e4b/fused_attn4_shipped_r0` STABLE (1.773/1.756 s, |Δ|/mean 1.0% vs 5%); `e4b/fused_attn4_shipped_rr` STABLE (1.579/1.598 s, |Δ|/mean 1.2% vs 5%); `e4b/fused_attn4_shipped_r1` STABLE (1.618/1.619 s, |Δ|/mean 0.1% vs 5%)
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
| qwen3ckptre | e4b/fused_attn4_shipped_r0 | **VALID** | VALID | 0.0000 |  |
| qwen3ckptre | e4b/fused_attn4_shipped_rr | **VALID** | VALID | 0.0009 |  |
| qwen3ckptre | e4b/fused_attn4_shipped_r1 | **VALID** | VALID | 0.0029 |  |
| qwen3ckptre | e4b/fused_attn4_shipped_r1_d2 | **VALID** | VALID | 0.0008 |  |
| qwen3ckptre | e4b/fused_attn4_shipped_rr_d2 | **VALID** | VALID | 0.0022 |  |
| qwen3ckptre | e4b/fused_attn4_shipped_r0_d2 | **VALID** | VALID | 0.0017 |  |

## Amendment 62: the shipped arm at the field recipe -- Hugging Face's checkpoint, the reentrant checkpoint alone, and with its inputs in host memory (descriptive)
| arm | VERDICT | s/step (11..N) | run peak GB | train | held-out N | checkpoint |
|---|---|---|---|---|---|---|
| e4b/fused_attn4_shipped_r0 | VALID | 1.773 | 23.321 | 23.321 | 0.75479 | 0 [] |
| e4b/fused_attn4_shipped_rr | VALID | 1.579 | 23.321 | 23.321 | 0.75566 | reentrant ['reentrant_checkpoint'] |
| e4b/fused_attn4_shipped_r1 | VALID | 1.618 | 23.104 | 23.104 | 0.75768 | 1 ['offloaded_checkpoint'] |
| e4b/fused_attn4_shipped_r1_d2 | VALID | 1.619 | 23.104 | 23.104 | 0.75562 | 1 ['offloaded_checkpoint'] |
| e4b/fused_attn4_shipped_rr_d2 | VALID | 1.598 | 23.321 | 23.321 | 0.75699 | reentrant ['reentrant_checkpoint'] |
| e4b/fused_attn4_shipped_r0_d2 | VALID | 1.756 | 23.321 | 23.321 | 0.75645 | 0 [] |

## Predictions P170 / P171 / P172 / P173 (TC1-PREREG amendment 62: what made the field recipe's step faster; scored mechanically)
| prediction | family | verdict | evidence |
|---|---|---|---|
| P170 | qwen3ckptre | **HELD** | rr / r0 0.900 [0.891, 0.910 over 4 cross-draw ratios] vs <= 0.96; s/step rr 1.579 / 1.598, r0 1.773 / 1.756 |
| P171 | qwen3ckptre | **HELD** | r1 / rr 1.019 [1.012, 1.025 over 4 cross-draw ratios] vs in [0.98, 1.02]; s/step r1 1.618 / 1.619, rr 1.579 / 1.598 |
| P172 | qwen3ckptre | **HELD** | r1 / r0 0.917 [0.912, 0.922 over 4 cross-draw ratios] vs <= 0.96; s/step r1 1.618 / 1.619, r0 1.773 / 1.756 |
| P173 | qwen3ckptre | **HELD** | mean held-out at N r0 0.75562, rr 0.75633, r1 0.75665; largest difference 0.00103 (<= 0.005) |

## Load gate (TC1-PREREG amendment 33): 0 draw(s) voided for host load and run again
Each line: the arm, the attempt, the median host load1 over that attempt's run, the gate. A VOID attempt's files are in loadvoid/; the last attempt of each arm stands whatever its load.

- `qwen3ckptre/e4b/fused_attn4_shipped_r0 attempt 0 load1_median 1.02 gate 6.0 status ok over 0`
- `qwen3ckptre/e4b/fused_attn4_shipped_rr attempt 0 load1_median 1.29 gate 6.0 status ok over 0`
- `qwen3ckptre/e4b/fused_attn4_shipped_r1 attempt 0 load1_median 2.18 gate 6.0 status ok over 0`
- `qwen3ckptre/e4b/fused_attn4_shipped_r1_d2 attempt 0 load1_median 1.22 gate 6.0 status ok over 0`
- `qwen3ckptre/e4b/fused_attn4_shipped_rr_d2 attempt 0 load1_median 1.11 gate 6.0 status ok over 0`
- `qwen3ckptre/e4b/fused_attn4_shipped_r0_d2 attempt 0 load1_median 1.83 gate 6.0 status ok over 0`
