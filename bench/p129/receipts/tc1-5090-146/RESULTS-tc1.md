# TC1 — Qwen3-30B-A3B on one RTX 5090: e4b vs Unsloth vs HF+PEFT vs axolotl at matched work, init and adapter precision (tc1)
Rule (tc1/TC1-PREREG.md): status per attempt in the vocabulary OK / REFUSED / OOM / INSTALL_FAILED / LOAD_FAULT / HARNESS_ERROR / ALARM / NOT_RUN; VALID/VOID per tp4's predicates plus the matched-set predicates (R3: matched init complete, fp32 adapters, step-0 held-out within 0.005 of e4b/reference_attn4_m); VERDICT exactly one of VALID / VOID / QUALITY_FAIL / OOM / UNSUPPORTED / HARNESS_ERROR / ALARM / NOT_RUN (QUALITY_FAIL = held-out |Δ| at N > 0.05 vs e4b/fused_attn4_m); positions = s/step ratios other/e4b from the medians over both draws (stability |d1-d2|/mean ≤ 5 % e4b / 10 % others, else UNSTABLE), quoted only when both arms are VALID; equivalence vs e4b/reference_attn4_m EQUIVALENT ≤ 0.02 / COMPARABLE ≤ 0.05 / else DIVERGENT; frozen base SAME-BYTES / DIFFERENT / N-A per slot; predictions P1–P10 (+ P1b) scored HELD / FALSIFIED / UNTESTED. VOID never enters a ratio or an equivalence reading. Nothing is licensed; no cross-box number is divided into these.
`versions.txt`
```
e4b 0.51.0 @be7a88d3cc3dbe4b2902059abc43cf92c6d1dd32 (GitHub main)
gnf4 0.44.0 @d1f64ba50afce94533e0166ba3332ef075aa43bb (GitHub main)
torch(e4b/hf) 2.8.0+cu128
triton(e4b/hf) 3.4.0
transformers(e4b/hf) 5.18.0
bitsandbytes(e4b/hf) 0.50.2
peft(hf) 0.21.2
huggingface_hub(e4b/hf) 2.2.0
httpx2(e4b/hf) 2.13.1
brotli(e4b/hf) 1.2.0
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
e4b(t212) 0.51.0 @be7a88d3cc3dbe4b2902059abc43cf92c6d1dd32
gnf4(t212) 0.44.0 @d1f64ba50afce94533e0166ba3332ef075aa43bb
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
 "run_id": "tc1-5090-146",
 "instance_id": "55065443",
 "gpu": "NVIDIA GeForce RTX 5090",
 "driver": "595.91.07",
 "cpu": "AMD EPYC 7K62 48-Core Processor",
 "nproc": 96,
 "mem_total_kb": "527994556",
 "cgroup_memory_max": "259519414272",
 "disk_root": "overlay         320G   58M  320G   1% /",
 "hostname": "17b0daaf652c",
 "cgroup_cpu_max": "2304000 100000",
 "affinity_cpus": 96,
 "cgroup_cpuset_effective": "0-95",
 "registered_gpu_class": "5090",
 "prereg": "tc1/TC1-PREREG.md",
 "local_box": null,
 "local_snapshot": null
}
```

### Qwen3-30B-A3B (P129 Amendment 2: E4B_TRAIN_FUSE_QKV 0 vs 1 at the field recipe, shipped and matched arms; profiled) (`qwen3fqkv`, registered n_layers 48, attention census 192)
- model `Qwen/Qwen3-30B-A3B` @ `ad44e777bcd1`; tokens sha `bfc742f67e37`; N=60; fixture template alpaca seq 2048 micro-batch 2 × accum 4 lr 0.0002 r 16 α 16 optimizer adamw_8bit(lr=0.0002, weight_decay=0.001) schedule=linear warmup_steps=5 autocast False; e4b trainable 642514944; box_class RTX 5090 gpu NVIDIA GeForce RTX 5090
| framework | arm | status | validity | **VERDICT** | matched | init / dtype | step-0 class | N | s/step med(11+) | tok/s | peak GB | J/step | train first→last | held-out 0→final | quality Δ | engagement | trainable | regime | reason / why |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| e4b | fused_attn4_shipped_q0 | **OK** | VALID | **VALID** | native | native / bfloat16,float32 | — | 60 | 3.193 | 459.8 | 23.321 | 542.0 | 2.0722→0.8173 | 1.9592→0.7551 | -0.0029 | patched 48 / kcalls 768 | 642514944 | 4-bit experts (e4b NF4) + NF4 attention |  |
| e4b | fused_attn4_shipped_q1 | **OK** | VALID | **VALID** | native | native / bfloat16,float32 | — | 60 | 2.842 | 539.0 | 23.344 | 505.4 | 2.0463→0.8206 | 1.9468→0.7572 | -0.0009 | patched 48 / kcalls 768 | 642514944 | 4-bit experts (e4b NF4) + NF4 attention |  |
| e4b | fused_attn4_m_q0 | **OK** | VALID | **VALID** | yes | matched:3407 (complete 12480/12480) / float32 | SAME-BYTES-CLASS (0.0000) | 60 | 3.302 | 468.9 | 26.495 | 648.0 | 2.0722→0.8190 | 1.9592→0.7581 | 0.0000 | patched 48 / kcalls 768 | 642514944 | 4-bit experts (e4b NF4) + NF4 attention |  |
| e4b | fused_attn4_m_q1 | **OK** | VALID | **VALID** | yes | matched:3407 (complete 12480/12480) / float32 | NEAR (0.0123) | 60 | 2.939 | 527.3 | 26.500 | 634.5 | 2.0463→0.8180 | 1.9468→0.7548 | -0.0033 | patched 48 / kcalls 768 | 642514944 | 4-bit experts (e4b NF4) + NF4 attention |  |
| e4b | fused_attn4_m_q1_d2 | **OK** | VALID | **VALID** | yes | matched:3407 (complete 12480/12480) / float32 | NEAR (0.0123) | 60 | 3.003 | 519.8 | 26.500 | 631.3 | 2.0463→0.8157 | 1.9468→0.7572 | -0.0008 | patched 48 / kcalls 768 | 642514944 | 4-bit experts (e4b NF4) + NF4 attention |  |
| e4b | fused_attn4_m_q0_d2 | **OK** | VALID | **VALID** | yes | matched:3407 (complete 12480/12480) / float32 | SAME-BYTES-CLASS (0.0000) | 60 | 3.332 | 465.6 | 26.495 | 657.0 | 2.0722→0.8153 | 1.9592→0.7555 | -0.0026 | patched 48 / kcalls 768 | 642514944 | 4-bit experts (e4b NF4) + NF4 attention |  |
| e4b | fused_attn4_shipped_q1_d2 | **OK** | VALID | **VALID** | native | native / bfloat16,float32 | — | 60 | 2.801 | 554.6 | 23.344 | 501.1 | 2.0463→0.8171 | 1.9468→0.7538 | -0.0042 | patched 48 / kcalls 768 | 642514944 | 4-bit experts (e4b NF4) + NF4 attention |  |
| e4b | fused_attn4_shipped_q0_d2 | **OK** | VALID | **VALID** | native | native / bfloat16,float32 | — | 60 | 3.227 | 482.5 | 23.321 | 534.7 | 2.0722→0.8169 | 1.9592→0.7590 | 0.0009 | patched 48 / kcalls 768 | 642514944 | 4-bit experts (e4b NF4) + NF4 attention |  |
- prologue `e4b/fused_attn4_shipped_q0` **110.4 s** before step 1 (21% of the arm): c1_before 67.4, load_weights 20.9, c1_after 18.3, attn4 7.2; unattributed 2.362; budget 1260.0
- prologue `e4b/fused_attn4_shipped_q1` **68.4 s** before step 1 (16% of the arm): c1_before 24.5, load_weights 21.5, c1_after 19.0, attn4 7.4; unattributed 2.237; budget 1260.0
- prologue `e4b/fused_attn4_m_q0` **111.1 s** before step 1 (21% of the arm): c1_before 66.9, load_weights 21.0, c1_after 18.7, attn4 7.3; unattributed 2.38; budget 1260.0
- prologue `e4b/fused_attn4_m_q1` **67.2 s** before step 1 (15% of the arm): c1_before 23.1, load_weights 20.9, c1_after 18.6, attn4 7.2; unattributed 2.262; budget 1260.0
- prologue `e4b/fused_attn4_m_q1_d2` **74.2 s** before step 1 (16% of the arm): c1_before 30.0, load_weights 20.9, c1_after 18.3, attn4 7.6; unattributed 2.362; budget 1260.0
- prologue `e4b/fused_attn4_m_q0_d2` **111.2 s** before step 1 (21% of the arm): c1_before 67.0, load_weights 20.9, c1_after 18.8, attn4 7.3; unattributed 2.356; budget 1260.0
- prologue `e4b/fused_attn4_shipped_q1_d2` **67.1 s** before step 1 (16% of the arm): c1_before 23.2, load_weights 20.9, c1_after 18.5, attn4 7.7; unattributed 2.343; budget 1260.0
- prologue `e4b/fused_attn4_shipped_q0_d2` **111.4 s** before step 1 (22% of the arm): c1_before 67.8, load_weights 21.0, c1_after 18.8, attn4 7.3; unattributed 2.353; budget 1260.0
- draws (R1): `e4b/fused_attn4_shipped_q0` STABLE (3.193/3.227 s, |Δ|/mean 1.0% vs 5%); `e4b/fused_attn4_shipped_q1` STABLE (2.842/2.801 s, |Δ|/mean 1.5% vs 5%); `e4b/fused_attn4_m_q0` STABLE (3.302/3.332 s, |Δ|/mean 0.9% vs 5%); `e4b/fused_attn4_m_q1` STABLE (2.939/3.003 s, |Δ|/mean 2.1% vs 5%)
- e4b internal parity: NO-ARM
- **NO MATCHED POSITION QUOTED (unsloth)** — not quoted: e4b STABLE / unsloth no receipt
- **NO MATCHED POSITION QUOTED (hf)** — not quoted: e4b STABLE / hf no receipt
- **NO MATCHED POSITION QUOTED (axolotl)** — not quoted: e4b STABLE / axolotl no receipt
- equivalence vs `e4b/fused_attn4_m` (I/H; EQUIVALENT iff both ≤ band = max(0.005, 3 × fused-vs-reference |Δ|) = none (no reference arm: COMPARABLE is the ceiling); COMPARABLE ≤ 0.05; draw-noise floor 0.0038): `e4b/fused_attn4_m_q1` **COMPARABLE** (median step |Δ| 0.0013, |Δ held-out at N| 0.0033, step-0 0.0123 NEAR, |Δ loss at step 2| 0.0290, paired rows mean -0.0033 ± 0.0012 SE over 8, favouring arm 6 / anchor 2) — no reference arm ran: the EQUIVALENT band cannot be set, COMPARABLE is the ceiling; `e4b/fused_attn4_m_q1_d2` **COMPARABLE** (median step |Δ| 0.0010, |Δ held-out at N| 0.0008, step-0 0.0123 NEAR, |Δ loss at step 2| 0.0298, paired rows mean -0.0008 ± 0.0017 SE over 8, favouring arm 7 / anchor 1) — no reference arm ran: the EQUIVALENT band cannot be set, COMPARABLE is the ceiling; `e4b/fused_attn4_m_q0_d2` **COMPARABLE** (median step |Δ| 0.0012, |Δ held-out at N| 0.0026, step-0 0.0000 SAME-BYTES-CLASS, |Δ loss at step 2| 0.0285, paired rows mean -0.0026 ± 0.0016 SE over 8, favouring arm 6 / anchor 2) — no reference arm ran: the EQUIVALENT band cannot be set, COMPARABLE is the ceiling
- matched_init_sha (B, name-free, canonical slot order): anchor `f7832488926eda91`; `e4b/fused_attn4_m_q0` same; `e4b/fused_attn4_m_q1` same; `e4b/fused_attn4_m_q1_d2` same; `e4b/fused_attn4_m_q0_d2` same
- frozen base (R5), anchor `e4b/fused_attn4_m`: gate_up nf4/64+dq sha e2bed944143e control detects, down nf4/64+dq sha f2ade29dafdb control detects, q_proj nf4/64+dq sha 5c5e402ce51d control detects
- frozen base `e4b/fused_attn4_shipped_q0`: gate_up **SAME-BYTES** (nf4/64+dq) control detects, down **SAME-BYTES** (nf4/64+dq) control detects, q_proj **SAME-BYTES** (nf4/64+dq) control detects
- frozen base `e4b/fused_attn4_shipped_q1`: gate_up **SAME-BYTES** (nf4/64+dq) control detects, down **SAME-BYTES** (nf4/64+dq) control detects, q_proj **N-A** (slot missing on this arm) control FAILS
- frozen base `e4b/fused_attn4_m_q1`: gate_up **SAME-BYTES** (nf4/64+dq) control detects, down **SAME-BYTES** (nf4/64+dq) control detects, q_proj **N-A** (slot missing on this arm) control FAILS
- frozen base `e4b/fused_attn4_m_q1_d2`: gate_up **SAME-BYTES** (nf4/64+dq) control detects, down **SAME-BYTES** (nf4/64+dq) control detects, q_proj **N-A** (slot missing on this arm) control FAILS
- frozen base `e4b/fused_attn4_m_q0_d2`: gate_up **SAME-BYTES** (nf4/64+dq) control detects, down **SAME-BYTES** (nf4/64+dq) control detects, q_proj **SAME-BYTES** (nf4/64+dq) control detects
- frozen base `e4b/fused_attn4_shipped_q1_d2`: gate_up **SAME-BYTES** (nf4/64+dq) control detects, down **SAME-BYTES** (nf4/64+dq) control detects, q_proj **N-A** (slot missing on this arm) control FAILS
- frozen base `e4b/fused_attn4_shipped_q0_d2`: gate_up **SAME-BYTES** (nf4/64+dq) control detects, down **SAME-BYTES** (nf4/64+dq) control detects, q_proj **SAME-BYTES** (nf4/64+dq) control detects

## Verdicts
| family | arm | VERDICT | validity | quality | note |
|---|---|---|---|---|---|
| qwen3fqkv | e4b/fused_attn4_shipped_q0 | **VALID** | VALID | -0.0029 |  |
| qwen3fqkv | e4b/fused_attn4_shipped_q1 | **VALID** | VALID | -0.0009 |  |
| qwen3fqkv | e4b/fused_attn4_m_q0 | **VALID** | VALID | 0.0000 |  |
| qwen3fqkv | e4b/fused_attn4_m_q1 | **VALID** | VALID | -0.0033 |  |
| qwen3fqkv | e4b/fused_attn4_m_q1_d2 | **VALID** | VALID | -0.0008 |  |
| qwen3fqkv | e4b/fused_attn4_m_q0_d2 | **VALID** | VALID | -0.0026 |  |
| qwen3fqkv | e4b/fused_attn4_shipped_q1_d2 | **VALID** | VALID | -0.0042 |  |
| qwen3fqkv | e4b/fused_attn4_shipped_q0_d2 | **VALID** | VALID | 0.0009 |  |

## P129 Amendment 2: E4B_TRAIN_FUSE_QKV 0 vs 1 at the field recipe (descriptive)
| arm | VERDICT | s/step (11..N) | device ms / profiled step | busy_t | launches / profiled step | fused modules | peak GB | held-out 0 / N |
|---|---|---|---|---|---|---|---|---|
| e4b/fused_attn4_shipped_q0 | VALID | 3.193 | 1282.7 | 0.402 | 75833 | 0 | 23.321 | 1.95917 / 0.75512 |
| e4b/fused_attn4_shipped_q1 | VALID | 2.842 | 1264.4 | 0.445 | 64761 | 48 | 23.344 | 1.94684 / 0.7572 |
| e4b/fused_attn4_m_q0 | VALID | 3.302 | 1731.6 | 0.524 | 79582 | 0 | 26.495 | 1.95917 / 0.75805 |
| e4b/fused_attn4_m_q1 | VALID | 2.939 | 1712.6 | 0.583 | 68524 | 48 | 26.500 | 1.94684 / 0.75478 |
| e4b/fused_attn4_m_q1_d2 | VALID | 3.003 | 1709.7 | 0.569 | 68510 | 48 | 26.500 | 1.94684 / 0.75722 |
| e4b/fused_attn4_m_q0_d2 | VALID | 3.332 | 1737.4 | 0.522 | 79580 | 0 | 26.495 | 1.95917 / 0.75549 |
| e4b/fused_attn4_shipped_q1_d2 | VALID | 2.801 | 1262.5 | 0.451 | 64761 | 48 | 23.344 | 1.94684 / 0.7538 |
| e4b/fused_attn4_shipped_q0_d2 | VALID | 3.227 | 1281.5 | 0.397 | 75833 | 0 | 23.321 | 1.95917 / 0.75895 |

## P129 Amendment 2's gates and verdict (scored mechanically)
| row | family | verdict | evidence |
|---|---|---|---|
| R_m | qwen3fqkv | **HELD** | launches per profiled step 79581 -> 68517 = -13.9 % vs >= 11.2 % (0.8 of Phase 1's 14.0 %) |
| R_shipped | qwen3fqkv | **HELD** | launches per profiled step 75833 -> 64761 = -14.6 % vs >= 11.2 % (0.8 of Phase 1's 14.0 %) |
| PREMISE | qwen3fqkv | **HELD** | matched q0 busy_t 0.523 vs <= 0.85 |
| W_m | qwen3fqkv | **HELD** | q1 / q0 0.896 [0.882, 0.909 over 4 cross-draw ratios] vs <= 0.98; s/step q1 2.939 / 3.003, q0 3.302 / 3.332 |
| W_shipped | qwen3fqkv | **HELD** | q1 / q0 0.879 [0.868, 0.890 over 4 cross-draw ratios] vs <= 0.98; s/step q1 2.842 / 2.801, q0 3.193 / 3.227 |
| DEVICE | qwen3fqkv | **REPORTED** | m: device 1734.5 -> 1711.2 ms = 0.987, peak 26.495 -> 26.5 GB; shipped: device 1282.1 -> 1263.4 ms = 0.985, peak 23.321 -> 23.344 GB |
| QUALITY | qwen3fqkv | **FALSIFIED** | m: step 0 -0.01233, -0.01233; N -0.00077; shipped: step 0 -0.01233, -0.01233; N -0.00153 (step 0 |.| <= 0.0005, N |.| <= 0.005) |
| FQKV | qwen3fqkv | **QUALITY_FAIL** | the first rung that applies: VOID / NOISY / QUALITY_FAIL / NO_GAIN / GAIN (DEFAULT_ON needs a second host) |

## Load gate (TC1-PREREG amendment 33): 6 draw(s) voided for host load and run again
Each line: the arm, the attempt, the median host load1 over that attempt's run, the gate. A VOID attempt's files are in loadvoid/; the last attempt of each arm stands whatever its load.

- `qwen3fqkv/e4b/fused_attn4_shipped_q0 attempt 0 load1_median 7.92 gate 6.0 status ok over 1`
- `qwen3fqkv/e4b/fused_attn4_shipped_q0 attempt 0 VOID (host load1 median 7.92 > 6.0): re-run 1 of 2`
- `qwen3fqkv/e4b/fused_attn4_shipped_q0 attempt 1 load1_median 6.48 gate 6.0 status ok over 1`
- `qwen3fqkv/e4b/fused_attn4_shipped_q0 attempt 1 VOID (host load1 median 6.48 > 6.0): re-run 2 of 2`
- `qwen3fqkv/e4b/fused_attn4_shipped_q0 attempt 2 load1_median 12.21 gate 6.0 status ok over 1`
- `qwen3fqkv/e4b/fused_attn4_shipped_q1 attempt 0 load1_median 8.81 gate 6.0 status ok over 1`
- `qwen3fqkv/e4b/fused_attn4_shipped_q1 attempt 0 VOID (host load1 median 8.81 > 6.0): re-run 1 of 2`
- `qwen3fqkv/e4b/fused_attn4_shipped_q1 attempt 1 load1_median 9.1 gate 6.0 status ok over 1`
- `qwen3fqkv/e4b/fused_attn4_shipped_q1 attempt 1 VOID (host load1 median 9.1 > 6.0): re-run 2 of 2`
- `qwen3fqkv/e4b/fused_attn4_shipped_q1 attempt 2 load1_median 8.96 gate 6.0 status ok over 1`
- `qwen3fqkv/e4b/fused_attn4_m_q0 attempt 0 load1_median 7.5 gate 6.0 status ok over 1`
- `qwen3fqkv/e4b/fused_attn4_m_q0 attempt 0 VOID (host load1 median 7.5 > 6.0): re-run 1 of 2`
- `qwen3fqkv/e4b/fused_attn4_m_q0 attempt 1 load1_median 8.14 gate 6.0 status ok over 1`
- `qwen3fqkv/e4b/fused_attn4_m_q0 attempt 1 VOID (host load1 median 8.14 > 6.0): re-run 2 of 2`
- `qwen3fqkv/e4b/fused_attn4_m_q0 attempt 2 load1_median 6.63 gate 6.0 status ok over 1`
- `qwen3fqkv/e4b/fused_attn4_m_q1 attempt 0 load1_median 5.51 gate 6.0 status ok over 0`
- `qwen3fqkv/e4b/fused_attn4_m_q1_d2 attempt 0 load1_median 3.71 gate 6.0 status ok over 0`
- `qwen3fqkv/e4b/fused_attn4_m_q0_d2 attempt 0 load1_median 4.43 gate 6.0 status ok over 0`
- `qwen3fqkv/e4b/fused_attn4_shipped_q1_d2 attempt 0 load1_median 4.03 gate 6.0 status ok over 0`
- `qwen3fqkv/e4b/fused_attn4_shipped_q0_d2 attempt 0 load1_median 3.98 gate 6.0 status ok over 0`
