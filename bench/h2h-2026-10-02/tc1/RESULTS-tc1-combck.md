# TC1 — Qwen3-30B-A3B on one RTX 5090: e4b vs Unsloth vs HF+PEFT vs axolotl at matched work, init and adapter precision (tc1)
Rule (tc1/TC1-PREREG.md): status per attempt in the vocabulary OK / REFUSED / OOM / INSTALL_FAILED / LOAD_FAULT / HARNESS_ERROR / ALARM / NOT_RUN; VALID/VOID per tp4's predicates plus the matched-set predicates (R3: matched init complete, fp32 adapters, step-0 held-out within 0.005 of e4b/reference_attn4_m); VERDICT exactly one of VALID / VOID / QUALITY_FAIL / OOM / UNSUPPORTED / HARNESS_ERROR / ALARM / NOT_RUN (QUALITY_FAIL = held-out |Δ| at N > 0.05 vs e4b/fused_attn4_m); positions = s/step ratios other/e4b from the medians over both draws (stability |d1-d2|/mean ≤ 5 % e4b / 10 % others, else UNSTABLE), quoted only when both arms are VALID; equivalence vs e4b/reference_attn4_m EQUIVALENT ≤ 0.02 / COMPARABLE ≤ 0.05 / else DIVERGENT; frozen base SAME-BYTES / DIFFERENT / N-A per slot; predictions P1–P10 (+ P1b) scored HELD / FALSIFIED / UNTESTED. VOID never enters a ratio or an equivalence reading. Nothing is licensed; no cross-box number is divided into these.
`versions.txt`
```
e4b 0.48.0 @faf65aee8267cea4d1c30cbd85c92347fcc01149 (GitHub main)
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
e4b(t212) 0.48.0 @faf65aee8267cea4d1c30cbd85c92347fcc01149
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
 "run_id": "tc1-5090-122",
 "instance_id": "54601318",
 "gpu": "NVIDIA GeForce RTX 5090",
 "driver": "580.119.02",
 "cpu": "AMD EPYC 9655 96-Core Processor",
 "nproc": 48,
 "mem_total_kb": "113302936",
 "cgroup_memory_max": "111380791296",
 "disk_root": "overlay         320G  2.5M  320G   1% /",
 "hostname": "93a1ae0f0f31",
 "cgroup_cpu_max": "4608000 100000",
 "affinity_cpus": 48,
 "cgroup_cpuset_effective": "0-47",
 "registered_gpu_class": "5090",
 "prereg": "tc1/TC1-PREREG.md",
 "local_box": null,
 "local_snapshot": null
}
```

### Qwen3-30B-A3B (amendment 61: the routed-expert combine whole vs over row chunks on packed rows, e4b with checkpoint inputs in host memory, Unsloth beside; peaks by phase) (`qwen3combck`, registered n_layers 48, attention census 192)
- model `Qwen/Qwen3-30B-A3B` @ `ad44e777bcd1`; tokens sha `d2a501eba57d`; N=40; fixture template alpaca seq 4096 (packed rows, amendment 39) micro-batch 1 × accum 4 lr 0.0002 r 16 α 16 optimizer adamw_8bit(lr=0.0002, weight_decay=0.001) schedule=linear warmup_steps=5 autocast False; e4b trainable 642514944; box_class RTX 5090 gpu NVIDIA GeForce RTX 5090
| framework | arm | status | validity | **VERDICT** | matched | init / dtype | step-0 class | N | s/step med(11+) | tok/s | peak GB | J/step | train first→last | held-out 0→final | quality Δ | engagement | trainable | regime | reason / why |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| e4b | fused_attn4_m_c0 | **OK** | VALID | **VALID** | yes | matched:3407 (complete 12480/12480) / float32 | SAME-BYTES-CLASS (0.0000) | 40 | 10.036 | 1617.3 | 26.877 | 4666.1 | 1.2578→0.9129 | 1.2885→0.9540 | 0.0000 | patched 48 / kcalls 768 | 642514944 | 4-bit experts (e4b NF4) + NF4 attention |  |
| e4b | fused_attn4_m_c1 | **OK** | VALID | **VALID** | yes | matched:3407 (complete 12480/12480) / float32 | SAME-BYTES-CLASS (0.0000) | 40 | 9.805 | 1661.3 | 26.877 | 4387.7 | 1.2578→0.9127 | 1.2885→0.9542 | 0.0002 | patched 48 / kcalls 768 | 642514944 | 4-bit experts (e4b NF4) + NF4 attention |  |
| e4b | fused_attn4_m_c1_d2 | **OK** | VALID | **VALID** | yes | matched:3407 (complete 12480/12480) / float32 | SAME-BYTES-CLASS (0.0000) | 40 | 9.813 | 1660.7 | 26.877 | 4320.0 | 1.2578→0.9131 | 1.2885→0.9545 | 0.0005 | patched 48 / kcalls 768 | 642514944 | 4-bit experts (e4b NF4) + NF4 attention |  |
| e4b | fused_attn4_m_c0_d2 | **OK** | VALID | **VALID** | yes | matched:3407 (complete 12480/12480) / float32 | SAME-BYTES-CLASS (0.0000) | 40 | 10.028 | 1626.8 | 26.877 | 4414.0 | 1.2578→0.9128 | 1.2885→0.9541 | 0.0001 | patched 48 / kcalls 768 | 642514944 | 4-bit experts (e4b NF4) + NF4 attention |  |
| unsloth | ckpt_unsloth_m_cc | **OK** | VALID | **VALID** | yes | matched:3407 (complete 12480/12480) / float32 | SAME-BYTES-CLASS (0.0014) | 40 | 11.651 | 1390.8 | 24.864 | 4361.0 | 1.2587→0.9132 | 1.2871→0.9544 | 0.0005 | stacks 96 / fwd 384 / u8 48 | 642514944 | 4-bit expert stacks + bnb-4bit attention (Params4bit stacks 96, Linear4bit 384) |  |
- prologue `e4b/fused_attn4_m_c0` **65.8 s** before step 1 (14% of the arm): c1_before 27.2, load_weights 16.6, c1_after 13.7, eval0 10.1; unattributed 2.237; budget 1890.0
- prologue `e4b/fused_attn4_m_c1` **65.8 s** before step 1 (14% of the arm): c1_before 27.5, load_weights 21.0, c1_after 14.2, eval0 6.0; unattributed 1.28; budget 1890.0
- prologue `e4b/fused_attn4_m_c1_d2` **55.0 s** before step 1 (12% of the arm): c1_before 27.7, c1_after 13.6, load_weights 10.9, eval0 6.0; unattributed 1.283; budget 1890.0
- prologue `e4b/fused_attn4_m_c0_d2` **54.6 s** before step 1 (12% of the arm): c1_before 26.9, c1_after 13.6, load_weights 10.7, eval0 6.1; unattributed 1.285; budget 1890.0
- prologue `unsloth/ckpt_unsloth_m_cc` **70.2 s** before step 1 (13% of the arm): c1_before 26.6, load_weights 16.4, eval0 14.7, c1_after 13.5; unattributed 7.059; budget 1890.0
- draws (R1): `e4b/fused_attn4_m_c0` STABLE (10.036/10.028 s, |Δ|/mean 0.1% vs 5%); `e4b/fused_attn4_m_c1` STABLE (9.805/9.813 s, |Δ|/mean 0.1% vs 5%); `unsloth/ckpt_unsloth_m_cc` SINGLE (single draw (no second draw registered))
- e4b internal parity: NO-ARM
- **NO MATCHED POSITION QUOTED (unsloth)** — not quoted: e4b STABLE / unsloth no receipt
- **NO MATCHED POSITION QUOTED (hf)** — not quoted: e4b STABLE / hf no receipt
- **NO MATCHED POSITION QUOTED (axolotl)** — not quoted: e4b STABLE / axolotl no receipt
- equivalence vs `e4b/fused_attn4_m` (I/H; EQUIVALENT iff both ≤ band = max(0.005, 3 × fused-vs-reference |Δ|) = none (no reference arm: COMPARABLE is the ceiling); COMPARABLE ≤ 0.05; draw-noise floor 0.0003): `e4b/fused_attn4_m_c1` **COMPARABLE** (median step |Δ| 0.0004, |Δ held-out at N| 0.0002, step-0 0.0000 SAME-BYTES-CLASS, |Δ loss at step 2| 0.0081, paired rows mean +0.0002 ± 0.0002 SE over 8, favouring arm 2 / anchor 6) — no reference arm ran: the EQUIVALENT band cannot be set, COMPARABLE is the ceiling; `e4b/fused_attn4_m_c1_d2` **COMPARABLE** (median step |Δ| 0.0003, |Δ held-out at N| 0.0005, step-0 0.0000 SAME-BYTES-CLASS, |Δ loss at step 2| 0.0030, paired rows mean +0.0005 ± 0.0002 SE over 8, favouring arm 2 / anchor 6) — no reference arm ran: the EQUIVALENT band cannot be set, COMPARABLE is the ceiling; `e4b/fused_attn4_m_c0_d2` **COMPARABLE** (median step |Δ| 0.0003, |Δ held-out at N| 0.0001, step-0 0.0000 SAME-BYTES-CLASS, |Δ loss at step 2| 0.0004, paired rows mean +0.0001 ± 0.0002 SE over 8, favouring arm 4 / anchor 4) — no reference arm ran: the EQUIVALENT band cannot be set, COMPARABLE is the ceiling; `unsloth/ckpt_unsloth_m_cc` **COMPARABLE** (median step |Δ| 0.0003, |Δ held-out at N| 0.0005, step-0 0.0014 SAME-BYTES-CLASS, |Δ loss at step 2| 0.0004, paired rows mean +0.0005 ± 0.0002 SE over 8, favouring arm 1 / anchor 7) — no reference arm ran: the EQUIVALENT band cannot be set, COMPARABLE is the ceiling
- matched_init_sha (B, name-free, canonical slot order): anchor `f7832488926eda91`; `e4b/fused_attn4_m_c0` same; `e4b/fused_attn4_m_c1` same; `e4b/fused_attn4_m_c1_d2` same; `e4b/fused_attn4_m_c0_d2` same; `unsloth/ckpt_unsloth_m_cc` same
- frozen base (R5), anchor `e4b/fused_attn4_m`: gate_up nf4/64+dq sha e2bed944143e control detects, down nf4/64+dq sha f2ade29dafdb control detects, q_proj nf4/64+dq sha 5c5e402ce51d control detects
- frozen base `e4b/fused_attn4_m_c1`: gate_up **SAME-BYTES** (nf4/64+dq) control detects, down **SAME-BYTES** (nf4/64+dq) control detects, q_proj **SAME-BYTES** (nf4/64+dq) control detects
- frozen base `e4b/fused_attn4_m_c1_d2`: gate_up **SAME-BYTES** (nf4/64+dq) control detects, down **SAME-BYTES** (nf4/64+dq) control detects, q_proj **SAME-BYTES** (nf4/64+dq) control detects
- frozen base `e4b/fused_attn4_m_c0_d2`: gate_up **SAME-BYTES** (nf4/64+dq) control detects, down **SAME-BYTES** (nf4/64+dq) control detects, q_proj **SAME-BYTES** (nf4/64+dq) control detects
- frozen base `unsloth/ckpt_unsloth_m_cc`: gate_up **SAME-BYTES** (nf4/64+dq) control detects, down **SAME-BYTES** (nf4/64+dq) control detects, q_proj **SAME-BYTES** (nf4/64+dq) control detects

## Verdicts
| family | arm | VERDICT | validity | quality | note |
|---|---|---|---|---|---|
| qwen3combck | e4b/fused_attn4_m_c0 | **VALID** | VALID | 0.0000 |  |
| qwen3combck | e4b/fused_attn4_m_c1 | **VALID** | VALID | 0.0002 |  |
| qwen3combck | e4b/fused_attn4_m_c1_d2 | **VALID** | VALID | 0.0005 |  |
| qwen3combck | e4b/fused_attn4_m_c0_d2 | **VALID** | VALID | 0.0001 |  |
| qwen3combck | unsloth/ckpt_unsloth_m_cc | **VALID** | VALID | 0.0005 |  |

## Amendment 61: the routed-expert combine whole vs over row chunks on packed rows, peaks by phase (descriptive)
| arm | VERDICT | s/step (11..N) | run peak GB | setup | eval | train | held-out step 0 | held-out N | chunked fwd / bwd |
|---|---|---|---|---|---|---|---|---|---|
| e4b/fused_attn4_m_c0 | VALID | 10.036 | 26.877 | 21.856 | 26.877 | 25.849 | 1.28851 | 0.95397 | 0 / 0 |
| e4b/fused_attn4_m_c1 | VALID | 9.805 | 26.877 | 21.856 | 26.877 | 25.842 | 1.28851 | 0.95421 | 16128 / 7680 |
| e4b/fused_attn4_m_c1_d2 | VALID | 9.813 | 26.877 | 21.856 | 26.877 | 25.844 | 1.28851 | 0.9545 | 16128 / 7680 |
| e4b/fused_attn4_m_c0_d2 | VALID | 10.028 | 26.877 | 21.856 | 26.877 | 25.856 | 1.28851 | 0.95407 | 0 / 0 |
| unsloth/ckpt_unsloth_m_cc | VALID | 11.651 | 24.864 | 21.139 | 21.852 | 24.864 | 1.28707 | 0.95444 | None / None |

## Predictions P166 / P167 / P168 / P169 (TC1-PREREG amendment 61: the combine over row chunks; scored mechanically)
| prediction | family | verdict | evidence |
|---|---|---|---|
| P166 | qwen3combck | **FALSIFIED** | training-phase peak c0 25.852 -> c1 25.843 GB (drop +0.009 vs >= 0.3) |
| P167 | qwen3combck | **HELD** | c1 / c0 0.978 [0.977, 0.979 over 4 cross-draw ratios] vs <= 1.02; s/step c0 10.036 / 10.028, c1 9.805 / 9.813 |
| P168 | qwen3combck | **HELD** | step-0 held-out c1 - c0 per draw pair +0.00000, +0.00000 (|.| <= 0.0001); mean held-out at N c1 - c0 +0.00034 (|.| <= 0.005) |
| P169 | qwen3combck | **FALSIFIED** | c1 training-phase peak 25.843 vs Unsloth 24.864 GB (gap +0.979 vs <= 0.7) |

## Load gate (TC1-PREREG amendment 33): 0 draw(s) voided for host load and run again
Each line: the arm, the attempt, the median host load1 over that attempt's run, the gate. A VOID attempt's files are in loadvoid/; the last attempt of each arm stands whatever its load.

- `qwen3combck/e4b/fused_attn4_m_c0 attempt 0 load1_median 1.19 gate 6.0 status ok over 0`
- `qwen3combck/e4b/fused_attn4_m_c1 attempt 0 load1_median 1.11 gate 6.0 status ok over 0`
- `qwen3combck/e4b/fused_attn4_m_c1_d2 attempt 0 load1_median 1.11 gate 6.0 status ok over 0`
- `qwen3combck/e4b/fused_attn4_m_c0_d2 attempt 0 load1_median 1.13 gate 6.0 status ok over 0`
- `qwen3combck/unsloth/ckpt_unsloth_m_cc attempt 0 load1_median 1.09 gate 6.0 status ok over 0`
