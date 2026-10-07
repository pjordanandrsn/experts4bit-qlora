# TC1 — Qwen3-30B-A3B on one RTX 5090: e4b vs Unsloth vs HF+PEFT vs axolotl at matched work, init and adapter precision (tc1)
Rule (tc1/TC1-PREREG.md): status per attempt in the vocabulary OK / REFUSED / OOM / INSTALL_FAILED / LOAD_FAULT / HARNESS_ERROR / ALARM / NOT_RUN; VALID/VOID per tp4's predicates plus the matched-set predicates (R3: matched init complete, fp32 adapters, step-0 held-out within 0.005 of e4b/reference_attn4_m); VERDICT exactly one of VALID / VOID / QUALITY_FAIL / OOM / UNSUPPORTED / HARNESS_ERROR / ALARM / NOT_RUN (QUALITY_FAIL = held-out |Δ| at N > 0.05 vs e4b/fused_attn4_m); positions = s/step ratios other/e4b from the medians over both draws (stability |d1-d2|/mean ≤ 5 % e4b / 10 % others, else UNSTABLE), quoted only when both arms are VALID; equivalence vs e4b/reference_attn4_m EQUIVALENT ≤ 0.02 / COMPARABLE ≤ 0.05 / else DIVERGENT; frozen base SAME-BYTES / DIFFERENT / N-A per slot; predictions P1–P10 (+ P1b) scored HELD / FALSIFIED / UNTESTED. VOID never enters a ratio or an equivalence reading. Nothing is licensed; no cross-box number is divided into these.
`versions.txt`
```
e4b 0.48.0 @82ec6f15a28eb03c1826dd6f7c582c755694ebbf (GitHub main)
gnf4 0.42.0 @ab1a342d501d7d2f2a08ca7b3abb7a2537210d08 (GitHub main)
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
e4b(t212) 0.48.0 @82ec6f15a28eb03c1826dd6f7c582c755694ebbf
gnf4(t212) 0.42.0 @ab1a342d501d7d2f2a08ca7b3abb7a2537210d08
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
 "run_id": "tc1-5090-133",
 "instance_id": "54714145",
 "gpu": "NVIDIA GeForce RTX 5090",
 "driver": "580.159.03",
 "cpu": "Intel(R) Core(TM) i9-14900K",
 "nproc": 32,
 "mem_total_kb": "129221160",
 "cgroup_memory_max": "127028690944",
 "disk_root": "overlay         320G  2.6M  320G   1% /",
 "hostname": "62b8997a745d",
 "cgroup_cpu_max": "3072000 100000",
 "affinity_cpus": 32,
 "cgroup_cpuset_effective": "0-31",
 "registered_gpu_class": "5090",
 "prereg": "tc1/TC1-PREREG.md",
 "local_box": null,
 "local_snapshot": null
}
```

### Qwen3-30B-A3B (amendment 66: grouped-nf4-gemm's bucketed delta as autograd ops vs one compact node on packed rows, shipped and matched arms, Unsloth beside; peaks by phase) (`qwen3cbk`, registered n_layers 48, attention census 192)
- model `Qwen/Qwen3-30B-A3B` @ `ad44e777bcd1`; tokens sha `d2a501eba57d`; N=40; fixture template alpaca seq 4096 (packed rows, amendment 39) micro-batch 1 × accum 4 lr 0.0002 r 16 α 16 optimizer adamw_8bit(lr=0.0002, weight_decay=0.001) schedule=linear warmup_steps=5 autocast False; e4b trainable 642514944; box_class RTX 5090 gpu NVIDIA GeForce RTX 5090
| framework | arm | status | validity | **VERDICT** | matched | init / dtype | step-0 class | N | s/step med(11+) | tok/s | peak GB | J/step | train first→last | held-out 0→final | quality Δ | engagement | trainable | regime | reason / why |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| e4b | fused_attn4_shipped_k0 | **OK** | VALID | **VALID** | native | native / bfloat16,float32 | — | 40 | 7.320 | 2207.3 | 23.003 | 2864.9 | 1.2578→0.9058 | 1.2885→0.9452 | -0.0089 | patched 48 / kcalls 768 | 642514944 | 4-bit experts (e4b NF4) + NF4 attention |  |
| e4b | fused_attn4_shipped_k1 | **OK** | VALID | **VALID** | native | native / bfloat16,float32 | — | 40 | 7.146 | 2269.9 | 22.664 | 2475.4 | 1.2578→0.9058 | 1.2885→0.9451 | -0.0090 | patched 48 / kcalls 768 | 642514944 | 4-bit experts (e4b NF4) + NF4 attention |  |
| e4b | fused_attn4_m_k0 | **OK** | VALID | **VALID** | yes | matched:3407 (complete 12480/12480) / float32 | SAME-BYTES-CLASS (0.0000) | 40 | 9.230 | 1759.1 | 26.561 | 3286.7 | 1.2578→0.9134 | 1.2885→0.9541 | 0.0000 | patched 48 / kcalls 768 | 642514944 | 4-bit experts (e4b NF4) + NF4 attention |  |
| e4b | fused_attn4_m_k1 | **OK** | VALID | **VALID** | yes | matched:3407 (complete 12480/12480) / float32 | SAME-BYTES-CLASS (0.0000) | 40 | 8.972 | 1810.6 | 25.906 | 3174.6 | 1.2578→0.9130 | 1.2885→0.9543 | 0.0001 | patched 48 / kcalls 768 | 642514944 | 4-bit experts (e4b NF4) + NF4 attention |  |
| e4b | fused_attn4_m_k1_d2 | **OK** | VALID | **VALID** | yes | matched:3407 (complete 12480/12480) / float32 | SAME-BYTES-CLASS (0.0000) | 40 | 8.970 | 1810.7 | 25.911 | 3178.5 | 1.2578→0.9134 | 1.2885→0.9538 | -0.0003 | patched 48 / kcalls 768 | 642514944 | 4-bit experts (e4b NF4) + NF4 attention |  |
| e4b | fused_attn4_m_k0_d2 | **OK** | VALID | **VALID** | yes | matched:3407 (complete 12480/12480) / float32 | SAME-BYTES-CLASS (0.0000) | 40 | 9.229 | 1758.8 | 26.564 | 3255.6 | 1.2578→0.9131 | 1.2885→0.9542 | 0.0001 | patched 48 / kcalls 768 | 642514944 | 4-bit experts (e4b NF4) + NF4 attention |  |
| e4b | fused_attn4_shipped_k1_d2 | **OK** | VALID | **VALID** | native | native / bfloat16,float32 | — | 40 | 7.149 | 2270.1 | 22.660 | 2498.9 | 1.2578→0.9061 | 1.2885→0.9452 | -0.0090 | patched 48 / kcalls 768 | 642514944 | 4-bit experts (e4b NF4) + NF4 attention |  |
| e4b | fused_attn4_shipped_k0_d2 | **OK** | VALID | **VALID** | native | native / bfloat16,float32 | — | 40 | 7.313 | 2223.3 | 23.014 | 2658.0 | 1.2578→0.9056 | 1.2885→0.9451 | -0.0090 | patched 48 / kcalls 768 | 642514944 | 4-bit experts (e4b NF4) + NF4 attention |  |
| unsloth | ckpt_unsloth_m_kk | **OK** | VALID | **VALID** | yes | matched:3407 (complete 12480/12480) / float32 | SAME-BYTES-CLASS (0.0014) | 40 | 11.542 | 1391.2 | 24.864 | 1261.2 | 1.2587→0.9130 | 1.2871→0.9545 | 0.0004 | stacks 96 / fwd 384 / u8 48 | 642514944 | 4-bit expert stacks + bnb-4bit attention (Params4bit stacks 96, Linear4bit 384) |  |
- prologue `e4b/fused_attn4_shipped_k0` **60.0 s** before step 1 (12% of the arm): c1_before 25.5, load_weights 17.0, eval0 7.9, c1_after 7.1; unattributed 1.229; budget 1890.0
- prologue `e4b/fused_attn4_shipped_k1` **52.8 s** before step 1 (11% of the arm): c1_before 25.8, load_weights 13.4, c1_after 7.1, eval0 4.6; unattributed 1.099; budget 1890.0
- prologue `e4b/fused_attn4_m_k0` **55.6 s** before step 1 (10% of the arm): c1_before 25.8, load_weights 14.0, c1_after 7.1, eval0 5.9; unattributed 1.101; budget 1890.0
- prologue `e4b/fused_attn4_m_k1` **56.2 s** before step 1 (10% of the arm): c1_before 26.9, load_weights 13.2, c1_after 7.1, eval0 5.7; unattributed 1.184; budget 1890.0
- prologue `e4b/fused_attn4_m_k1_d2` **56.0 s** before step 1 (10% of the arm): c1_before 27.1, load_weights 12.9, c1_after 7.1, eval0 5.6; unattributed 1.089; budget 1890.0
- prologue `e4b/fused_attn4_m_k0_d2` **55.8 s** before step 1 (10% of the arm): c1_before 27.1, load_weights 12.9, c1_after 7.1, eval0 5.9; unattributed 1.104; budget 1890.0
- prologue `e4b/fused_attn4_shipped_k1_d2` **53.5 s** before step 1 (11% of the arm): c1_before 26.8, load_weights 12.9, c1_after 7.1, eval0 4.6; unattributed 1.094; budget 1890.0
- prologue `e4b/fused_attn4_shipped_k0_d2` **53.8 s** before step 1 (11% of the arm): c1_before 27.0, load_weights 12.9, c1_after 7.3, eval0 4.8; unattributed 1.076; budget 1890.0
- prologue `unsloth/ckpt_unsloth_m_kk` **63.2 s** before step 1 (4% of the arm): c1_before 26.5, load_weights 13.8, eval0 12.9, c1_after 7.1; unattributed 5.131; budget 1890.0
- draws (R1): `e4b/fused_attn4_shipped_k0` STABLE (7.320/7.313 s, |Δ|/mean 0.1% vs 5%); `e4b/fused_attn4_shipped_k1` STABLE (7.146/7.149 s, |Δ|/mean 0.0% vs 5%); `e4b/fused_attn4_m_k0` STABLE (9.230/9.229 s, |Δ|/mean 0.0% vs 5%); `e4b/fused_attn4_m_k1` STABLE (8.972/8.970 s, |Δ|/mean 0.0% vs 5%); `unsloth/ckpt_unsloth_m_kk` SINGLE (single draw (no second draw registered))
- e4b internal parity: NO-ARM
- **NO MATCHED POSITION QUOTED (unsloth)** — not quoted: e4b STABLE / unsloth no receipt
- **NO MATCHED POSITION QUOTED (hf)** — not quoted: e4b STABLE / hf no receipt
- **NO MATCHED POSITION QUOTED (axolotl)** — not quoted: e4b STABLE / axolotl no receipt
- equivalence vs `e4b/fused_attn4_m` (I/H; EQUIVALENT iff both ≤ band = max(0.005, 3 × fused-vs-reference |Δ|) = none (no reference arm: COMPARABLE is the ceiling); COMPARABLE ≤ 0.05; draw-noise floor 0.0004): `e4b/fused_attn4_m_k1` **COMPARABLE** (median step |Δ| 0.0004, |Δ held-out at N| 0.0001, step-0 0.0000 SAME-BYTES-CLASS, |Δ loss at step 2| 0.0012, paired rows mean +0.0001 ± 0.0004 SE over 8, favouring arm 4 / anchor 4) — no reference arm ran: the EQUIVALENT band cannot be set, COMPARABLE is the ceiling; `e4b/fused_attn4_m_k1_d2` **COMPARABLE** (median step |Δ| 0.0003, |Δ held-out at N| 0.0003, step-0 0.0000 SAME-BYTES-CLASS, |Δ loss at step 2| 0.0003, paired rows mean -0.0003 ± 0.0004 SE over 8, favouring arm 4 / anchor 4) — no reference arm ran: the EQUIVALENT band cannot be set, COMPARABLE is the ceiling; `e4b/fused_attn4_m_k0_d2` **COMPARABLE** (median step |Δ| 0.0003, |Δ held-out at N| 0.0001, step-0 0.0000 SAME-BYTES-CLASS, |Δ loss at step 2| 0.0037, paired rows mean +0.0001 ± 0.0004 SE over 8, favouring arm 4 / anchor 4) — no reference arm ran: the EQUIVALENT band cannot be set, COMPARABLE is the ceiling; `unsloth/ckpt_unsloth_m_kk` **COMPARABLE** (median step |Δ| 0.0003, |Δ held-out at N| 0.0004, step-0 0.0014 SAME-BYTES-CLASS, |Δ loss at step 2| 0.0002, paired rows mean +0.0004 ± 0.0004 SE over 8, favouring arm 3 / anchor 5) — no reference arm ran: the EQUIVALENT band cannot be set, COMPARABLE is the ceiling
- matched_init_sha (B, name-free, canonical slot order): anchor `f7832488926eda91`; `e4b/fused_attn4_m_k0` same; `e4b/fused_attn4_m_k1` same; `e4b/fused_attn4_m_k1_d2` same; `e4b/fused_attn4_m_k0_d2` same; `unsloth/ckpt_unsloth_m_kk` same
- frozen base (R5), anchor `e4b/fused_attn4_m`: gate_up nf4/64+dq sha e2bed944143e control detects, down nf4/64+dq sha f2ade29dafdb control detects, q_proj nf4/64+dq sha 5c5e402ce51d control detects
- frozen base `e4b/fused_attn4_shipped_k0`: gate_up **SAME-BYTES** (nf4/64+dq) control detects, down **SAME-BYTES** (nf4/64+dq) control detects, q_proj **SAME-BYTES** (nf4/64+dq) control detects
- frozen base `e4b/fused_attn4_shipped_k1`: gate_up **SAME-BYTES** (nf4/64+dq) control detects, down **SAME-BYTES** (nf4/64+dq) control detects, q_proj **SAME-BYTES** (nf4/64+dq) control detects
- frozen base `e4b/fused_attn4_m_k1`: gate_up **SAME-BYTES** (nf4/64+dq) control detects, down **SAME-BYTES** (nf4/64+dq) control detects, q_proj **SAME-BYTES** (nf4/64+dq) control detects
- frozen base `e4b/fused_attn4_m_k1_d2`: gate_up **SAME-BYTES** (nf4/64+dq) control detects, down **SAME-BYTES** (nf4/64+dq) control detects, q_proj **SAME-BYTES** (nf4/64+dq) control detects
- frozen base `e4b/fused_attn4_m_k0_d2`: gate_up **SAME-BYTES** (nf4/64+dq) control detects, down **SAME-BYTES** (nf4/64+dq) control detects, q_proj **SAME-BYTES** (nf4/64+dq) control detects
- frozen base `e4b/fused_attn4_shipped_k1_d2`: gate_up **SAME-BYTES** (nf4/64+dq) control detects, down **SAME-BYTES** (nf4/64+dq) control detects, q_proj **SAME-BYTES** (nf4/64+dq) control detects
- frozen base `e4b/fused_attn4_shipped_k0_d2`: gate_up **SAME-BYTES** (nf4/64+dq) control detects, down **SAME-BYTES** (nf4/64+dq) control detects, q_proj **SAME-BYTES** (nf4/64+dq) control detects
- frozen base `unsloth/ckpt_unsloth_m_kk`: gate_up **SAME-BYTES** (nf4/64+dq) control detects, down **SAME-BYTES** (nf4/64+dq) control detects, q_proj **SAME-BYTES** (nf4/64+dq) control detects

## Verdicts
| family | arm | VERDICT | validity | quality | note |
|---|---|---|---|---|---|
| qwen3cbk | e4b/fused_attn4_shipped_k0 | **VALID** | VALID | -0.0089 |  |
| qwen3cbk | e4b/fused_attn4_shipped_k1 | **VALID** | VALID | -0.0090 |  |
| qwen3cbk | e4b/fused_attn4_m_k0 | **VALID** | VALID | 0.0000 |  |
| qwen3cbk | e4b/fused_attn4_m_k1 | **VALID** | VALID | 0.0001 |  |
| qwen3cbk | e4b/fused_attn4_m_k1_d2 | **VALID** | VALID | -0.0003 |  |
| qwen3cbk | e4b/fused_attn4_m_k0_d2 | **VALID** | VALID | 0.0001 |  |
| qwen3cbk | e4b/fused_attn4_shipped_k1_d2 | **VALID** | VALID | -0.0090 |  |
| qwen3cbk | e4b/fused_attn4_shipped_k0_d2 | **VALID** | VALID | -0.0090 |  |
| qwen3cbk | unsloth/ckpt_unsloth_m_kk | **VALID** | VALID | 0.0004 |  |

## Amendment 66: grouped-nf4-gemm's bucketed delta as autograd ops vs one compact node on packed rows, peaks by phase (descriptive)
| arm | VERDICT | s/step (11..N) | run peak GB | train | eval | held-out step 0 | held-out N | compact calls | device ms/profiled step |
|---|---|---|---|---|---|---|---|---|---|
| e4b/fused_attn4_shipped_k0 | VALID | 7.320 | 23.003 | 23.003 | 21.245 | 1.28851 | 0.94519 | 0 | 7188.5 |
| e4b/fused_attn4_shipped_k1 | VALID | 7.146 | 22.664 | 22.664 | 21.245 | 1.28851 | 0.94514 | 31824 | 7036.8 |
| e4b/fused_attn4_m_k0 | VALID | 9.230 | 26.561 | 26.561 | 22.504 | 1.28851 | 0.95412 | 0 | 9129.2 |
| e4b/fused_attn4_m_k1 | VALID | 8.972 | 25.906 | 25.906 | 22.504 | 1.28851 | 0.95426 | 31764 | 8892.5 |
| e4b/fused_attn4_m_k1_d2 | VALID | 8.970 | 25.911 | 25.911 | 22.504 | 1.28851 | 0.95383 | 31766 | 8901.4 |
| e4b/fused_attn4_m_k0_d2 | VALID | 9.229 | 26.564 | 26.564 | 22.504 | 1.28851 | 0.95424 | 0 | 9158.8 |
| e4b/fused_attn4_shipped_k1_d2 | VALID | 7.149 | 22.660 | 22.66 | 21.245 | 1.28851 | 0.94516 | 31828 | 7039.8 |
| e4b/fused_attn4_shipped_k0_d2 | VALID | 7.313 | 23.014 | 23.014 | 21.245 | 1.28851 | 0.94514 | 0 | 7203.3 |
| unsloth/ckpt_unsloth_m_kk | VALID | 11.542 | 24.864 | 24.864 | 21.852 | 1.28707 | 0.95449 | None | 10152.2 |

## Predictions P189-P194 (TC1-PREREG amendment 66: the compact bucketed delta on packed rows; scored mechanically)
| prediction | family | verdict | evidence |
|---|---|---|---|
| P189 | qwen3cbk | **HELD** | matched training-phase peak k0 26.562 -> k1 25.909 GB (drop +0.654 vs >= 0.5) |
| P190 | qwen3cbk | **HELD** | k1 / k0 0.972 [0.972, 0.972 over 4 cross-draw ratios] vs <= 1.01; s/step k1 8.972 / 8.970, k0 9.230 / 9.229; k0 device busy vs the timed step 0.991 (descriptive) |
| P191 | qwen3cbk | **HELD** | k1 / k0 0.977 [0.976, 0.978 over 4 cross-draw ratios] vs <= 1.01; s/step k1 7.146 / 7.149, k0 7.320 / 7.313; k0 device busy vs the timed step 0.984 (descriptive) |
| P192 | qwen3cbk | **HELD** | m: step 0 +0.00000, +0.00000; N -0.00013; shipped: step 0 +0.00000, +0.00000; N -0.00002 (step 0 |.| <= 0.0001, N |.| <= 0.005) |
| P193 | qwen3cbk | **FALSIFIED** | matched k1 training-phase peak 25.909 vs Unsloth 24.864 GB (gap +1.044 vs <= 1.0) |
| P194 | qwen3cbk | **HELD** | m: k1 / k0 0.973 (device ms per profiled step k0 9129.2 / 9158.8, k1 8892.5 / 8901.4); shipped: k1 / k0 0.978 (device ms per profiled step k0 7188.5 / 7203.3, k1 7036.8 / 7039.8) (each arm <= 1.01) |

## Load gate (TC1-PREREG amendment 33): 0 draw(s) voided for host load and run again
Each line: the arm, the attempt, the median host load1 over that attempt's run, the gate. A VOID attempt's files are in loadvoid/; the last attempt of each arm stands whatever its load.

- `qwen3cbk/e4b/fused_attn4_shipped_k0 attempt 0 load1_median 1.06 gate 6.0 status ok over 0`
- `qwen3cbk/e4b/fused_attn4_shipped_k1 attempt 0 load1_median 1.16 gate 6.0 status ok over 0`
- `qwen3cbk/e4b/fused_attn4_m_k0 attempt 0 load1_median 1.13 gate 6.0 status ok over 0`
- `qwen3cbk/e4b/fused_attn4_m_k1 attempt 0 load1_median 1.27 gate 6.0 status ok over 0`
- `qwen3cbk/e4b/fused_attn4_m_k1_d2 attempt 0 load1_median 1.18 gate 6.0 status ok over 0`
- `qwen3cbk/e4b/fused_attn4_m_k0_d2 attempt 0 load1_median 1.16 gate 6.0 status ok over 0`
- `qwen3cbk/e4b/fused_attn4_shipped_k1_d2 attempt 0 load1_median 1.15 gate 6.0 status ok over 0`
- `qwen3cbk/e4b/fused_attn4_shipped_k0_d2 attempt 0 load1_median 1.19 gate 6.0 status ok over 0`
- `qwen3cbk/unsloth/ckpt_unsloth_m_kk attempt 0 load1_median 1.17 gate 6.0 status ok over 0`
