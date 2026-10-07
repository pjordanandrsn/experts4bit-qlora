# TC1 — Qwen3-30B-A3B on one RTX 5090: e4b vs Unsloth vs HF+PEFT vs axolotl at matched work, init and adapter precision (tc1)
Rule (tc1/TC1-PREREG.md): status per attempt in the vocabulary OK / REFUSED / OOM / INSTALL_FAILED / LOAD_FAULT / HARNESS_ERROR / ALARM / NOT_RUN; VALID/VOID per tp4's predicates plus the matched-set predicates (R3: matched init complete, fp32 adapters, step-0 held-out within 0.005 of e4b/reference_attn4_m); VERDICT exactly one of VALID / VOID / QUALITY_FAIL / OOM / UNSUPPORTED / HARNESS_ERROR / ALARM / NOT_RUN (QUALITY_FAIL = held-out |Δ| at N > 0.05 vs e4b/fused_attn4_m); positions = s/step ratios other/e4b from the medians over both draws (stability |d1-d2|/mean ≤ 5 % e4b / 10 % others, else UNSTABLE), quoted only when both arms are VALID; equivalence vs e4b/reference_attn4_m EQUIVALENT ≤ 0.02 / COMPARABLE ≤ 0.05 / else DIVERGENT; frozen base SAME-BYTES / DIFFERENT / N-A per slot; predictions P1–P10 (+ P1b) scored HELD / FALSIFIED / UNTESTED. VOID never enters a ratio or an equivalence reading. Nothing is licensed; no cross-box number is divided into these.
`versions.txt`
```
e4b 0.48.0 @5f353069f52d96dc82ffe33457b06f103da9e542 (GitHub main)
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
e4b(t212) 0.48.0 @5f353069f52d96dc82ffe33457b06f103da9e542
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
 "run_id": "tc1-5090-121",
 "instance_id": "54599380",
 "gpu": "NVIDIA GeForce RTX 5090",
 "driver": "580.119.02",
 "cpu": "AMD EPYC 9755 128-Core Processor",
 "nproc": 48,
 "mem_total_kb": "113302944",
 "cgroup_memory_max": "111380791296",
 "disk_root": "overlay         320G  2.5M  320G   1% /",
 "hostname": "6944be7830aa",
 "cgroup_cpu_max": "4608000 100000",
 "affinity_cpus": 48,
 "cgroup_cpuset_effective": "0-47",
 "registered_gpu_class": "5090",
 "prereg": "tc1/TC1-PREREG.md",
 "local_box": null,
 "local_snapshot": null
}
```

### Qwen3-30B-A3B (amendment 60: the held-out loss stock vs from the logits in chunks on packed rows, e4b with checkpoint inputs in host memory, Unsloth beside; peaks by phase) (`qwen3evalce`, registered n_layers 48, attention census 192)
- model `Qwen/Qwen3-30B-A3B` @ `ad44e777bcd1`; tokens sha `d2a501eba57d`; N=40; fixture template alpaca seq 4096 (packed rows, amendment 39) micro-batch 1 × accum 4 lr 0.0002 r 16 α 16 optimizer adamw_8bit(lr=0.0002, weight_decay=0.001) schedule=linear warmup_steps=5 autocast False; e4b trainable 642514944; box_class RTX 5090 gpu NVIDIA GeForce RTX 5090
| framework | arm | status | validity | **VERDICT** | matched | init / dtype | step-0 class | N | s/step med(11+) | tok/s | peak GB | J/step | train first→last | held-out 0→final | quality Δ | engagement | trainable | regime | reason / why |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| e4b | fused_attn4_m_e0 | **OK** | VALID | **VALID** | yes | matched:3407 (complete 12480/12480) / float32 | SAME-BYTES-CLASS (0.0000) | 40 | 10.194 | 1591.2 | 26.877 | 4715.6 | 1.2578→0.9130 | 1.2885→0.9545 | 0.0000 | patched 48 / kcalls 768 | 642514944 | 4-bit experts (e4b NF4) + NF4 attention |  |
| e4b | fused_attn4_m_e1 | **OK** | VALID | **VALID** | yes | matched:3407 (complete 12480/12480) / float32 | SAME-BYTES-CLASS (0.0000) | 40 | 10.203 | 1599.0 | 25.851 | 4556.6 | 1.2578→0.9135 | 1.2885→0.9544 | -0.0001 | patched 48 / kcalls 768 | 642514944 | 4-bit experts (e4b NF4) + NF4 attention |  |
| e4b | fused_attn4_m_e1_d2 | **OK** | VALID | **VALID** | yes | matched:3407 (complete 12480/12480) / float32 | SAME-BYTES-CLASS (0.0000) | 40 | 10.188 | 1599.8 | 25.852 | 4490.5 | 1.2578→0.9131 | 1.2885→0.9541 | -0.0004 | patched 48 / kcalls 768 | 642514944 | 4-bit experts (e4b NF4) + NF4 attention |  |
| e4b | fused_attn4_m_e0_d2 | **OK** | VALID | **VALID** | yes | matched:3407 (complete 12480/12480) / float32 | SAME-BYTES-CLASS (0.0000) | 40 | 10.200 | 1599.3 | 26.877 | 4504.8 | 1.2578→0.9127 | 1.2885→0.9539 | -0.0005 | patched 48 / kcalls 768 | 642514944 | 4-bit experts (e4b NF4) + NF4 attention |  |
| unsloth | ckpt_unsloth_m_ee | **OK** | VALID | **VALID** | yes | matched:3407 (complete 12480/12480) / float32 | SAME-BYTES-CLASS (0.0014) | 40 | 12.543 | 1289.2 | 24.864 | 4452.6 | 1.2587→0.9131 | 1.2871→0.9540 | -0.0004 | stacks 96 / fwd 384 / u8 48 | 642514944 | 4-bit expert stacks + bnb-4bit attention (Params4bit stacks 96, Linear4bit 384) |  |
- prologue `e4b/fused_attn4_m_e0` **70.7 s** before step 1 (14% of the arm): c1_before 35.6, c1_after 17.1, load_weights 12.2, eval0 10.1; unattributed 1.571; budget 1890.0
- prologue `e4b/fused_attn4_m_e1` **67.4 s** before step 1 (14% of the arm): c1_before 34.6, c1_after 17.1, load_weights 13.2, eval0 6.3; unattributed 1.686; budget 1890.0
- prologue `e4b/fused_attn4_m_e1_d2` **65.8 s** before step 1 (14% of the arm): c1_before 34.2, c1_after 17.2, load_weights 12.2, eval0 6.3; unattributed 1.561; budget 1890.0
- prologue `e4b/fused_attn4_m_e0_d2` **65.8 s** before step 1 (14% of the arm): c1_before 34.4, c1_after 17.4, load_weights 12.1, eval0 6.4; unattributed 1.681; budget 1890.0
- prologue `unsloth/ckpt_unsloth_m_ee` **87.1 s** before step 1 (14% of the arm): c1_before 34.1, load_weights 22.0, c1_after 17.4, eval0 16.5; unattributed 8.429; budget 1890.0
- draws (R1): `e4b/fused_attn4_m_e0` STABLE (10.194/10.200 s, |Δ|/mean 0.1% vs 5%); `e4b/fused_attn4_m_e1` STABLE (10.203/10.188 s, |Δ|/mean 0.2% vs 5%); `unsloth/ckpt_unsloth_m_ee` SINGLE (single draw (no second draw registered))
- e4b internal parity: NO-ARM
- **NO MATCHED POSITION QUOTED (unsloth)** — not quoted: e4b STABLE / unsloth no receipt
- **NO MATCHED POSITION QUOTED (hf)** — not quoted: e4b STABLE / hf no receipt
- **NO MATCHED POSITION QUOTED (axolotl)** — not quoted: e4b STABLE / axolotl no receipt
- equivalence vs `e4b/fused_attn4_m` (I/H; EQUIVALENT iff both ≤ band = max(0.005, 3 × fused-vs-reference |Δ|) = none (no reference arm: COMPARABLE is the ceiling); COMPARABLE ≤ 0.05; draw-noise floor 0.0005): `e4b/fused_attn4_m_e1` **COMPARABLE** (median step |Δ| 0.0002, |Δ held-out at N| 0.0001, step-0 0.0000 SAME-BYTES-CLASS, |Δ loss at step 2| 0.0000, paired rows mean -0.0001 ± 0.0003 SE over 8, favouring arm 4 / anchor 4) — no reference arm ran: the EQUIVALENT band cannot be set, COMPARABLE is the ceiling; `e4b/fused_attn4_m_e1_d2` **COMPARABLE** (median step |Δ| 0.0002, |Δ held-out at N| 0.0004, step-0 0.0000 SAME-BYTES-CLASS, |Δ loss at step 2| 0.0002, paired rows mean -0.0004 ± 0.0004 SE over 8, favouring arm 6 / anchor 2) — no reference arm ran: the EQUIVALENT band cannot be set, COMPARABLE is the ceiling; `e4b/fused_attn4_m_e0_d2` **COMPARABLE** (median step |Δ| 0.0003, |Δ held-out at N| 0.0005, step-0 0.0000 SAME-BYTES-CLASS, |Δ loss at step 2| 0.0022, paired rows mean -0.0005 ± 0.0003 SE over 8, favouring arm 6 / anchor 2) — no reference arm ran: the EQUIVALENT band cannot be set, COMPARABLE is the ceiling; `unsloth/ckpt_unsloth_m_ee` **COMPARABLE** (median step |Δ| 0.0005, |Δ held-out at N| 0.0004, step-0 0.0014 SAME-BYTES-CLASS, |Δ loss at step 2| 0.0019, paired rows mean -0.0004 ± 0.0002 SE over 8, favouring arm 7 / anchor 1) — no reference arm ran: the EQUIVALENT band cannot be set, COMPARABLE is the ceiling
- matched_init_sha (B, name-free, canonical slot order): anchor `f7832488926eda91`; `e4b/fused_attn4_m_e0` same; `e4b/fused_attn4_m_e1` same; `e4b/fused_attn4_m_e1_d2` same; `e4b/fused_attn4_m_e0_d2` same; `unsloth/ckpt_unsloth_m_ee` same
- frozen base (R5), anchor `e4b/fused_attn4_m`: gate_up nf4/64+dq sha e2bed944143e control detects, down nf4/64+dq sha f2ade29dafdb control detects, q_proj nf4/64+dq sha 5c5e402ce51d control detects
- frozen base `e4b/fused_attn4_m_e1`: gate_up **SAME-BYTES** (nf4/64+dq) control detects, down **SAME-BYTES** (nf4/64+dq) control detects, q_proj **SAME-BYTES** (nf4/64+dq) control detects
- frozen base `e4b/fused_attn4_m_e1_d2`: gate_up **SAME-BYTES** (nf4/64+dq) control detects, down **SAME-BYTES** (nf4/64+dq) control detects, q_proj **SAME-BYTES** (nf4/64+dq) control detects
- frozen base `e4b/fused_attn4_m_e0_d2`: gate_up **SAME-BYTES** (nf4/64+dq) control detects, down **SAME-BYTES** (nf4/64+dq) control detects, q_proj **SAME-BYTES** (nf4/64+dq) control detects
- frozen base `unsloth/ckpt_unsloth_m_ee`: gate_up **SAME-BYTES** (nf4/64+dq) control detects, down **SAME-BYTES** (nf4/64+dq) control detects, q_proj **SAME-BYTES** (nf4/64+dq) control detects

## Verdicts
| family | arm | VERDICT | validity | quality | note |
|---|---|---|---|---|---|
| qwen3evalce | e4b/fused_attn4_m_e0 | **VALID** | VALID | 0.0000 |  |
| qwen3evalce | e4b/fused_attn4_m_e1 | **VALID** | VALID | -0.0001 |  |
| qwen3evalce | e4b/fused_attn4_m_e1_d2 | **VALID** | VALID | -0.0004 |  |
| qwen3evalce | e4b/fused_attn4_m_e0_d2 | **VALID** | VALID | -0.0005 |  |
| qwen3evalce | unsloth/ckpt_unsloth_m_ee | **VALID** | VALID | -0.0004 |  |

## Amendment 60: the held-out loss stock vs from the logits in chunks on packed rows, peaks by phase (descriptive)
| arm | VERDICT | s/step (11..N) | run peak GB | setup | eval | train | held-out step 0 | held-out N | eval forwards chunked |
|---|---|---|---|---|---|---|---|---|---|
| e4b/fused_attn4_m_e0 | VALID | 10.194 | 26.877 | 21.856 | 26.877 | 25.854 | 1.28851 | 0.95449 | 0 |
| e4b/fused_attn4_m_e1 | VALID | 10.203 | 25.851 | 21.856 | 22.504 | 25.851 | 1.28851 | 0.95443 | 16 |
| e4b/fused_attn4_m_e1_d2 | VALID | 10.188 | 25.852 | 21.856 | 22.504 | 25.852 | 1.28851 | 0.95411 | 16 |
| e4b/fused_attn4_m_e0_d2 | VALID | 10.200 | 26.877 | 21.856 | 26.877 | 25.852 | 1.28851 | 0.95395 | 0 |
| unsloth/ckpt_unsloth_m_ee | VALID | 12.543 | 24.864 | 21.139 | 21.852 | 24.864 | 1.28707 | 0.95405 | None |

## Predictions P162 / P163 / P164 / P165 (TC1-PREREG amendment 60: the held-out loss from the logits in chunks; scored mechanically)
| prediction | family | verdict | evidence |
|---|---|---|---|
| P162 | qwen3evalce | **HELD** | evaluation-phase peak e0 26.877 -> e1 22.504 GB (drop +4.373 vs >= 3.5) |
| P163 | qwen3evalce | **HELD** | e1 evaluation vs training phase peak: 22.504 vs 25.851; 22.504 vs 25.852 |
| P164 | qwen3evalce | **HELD** | step-0 held-out e1 - e0 per draw pair: +0.00000, +0.00000 (|.| <= 0.0001) |
| P165 | qwen3evalce | **HELD** | mean held-out at N e0 0.95422, e1 0.95427 (diff +0.00005 vs |.| <= 0.005) |

## Load gate (TC1-PREREG amendment 33): 0 draw(s) voided for host load and run again
Each line: the arm, the attempt, the median host load1 over that attempt's run, the gate. A VOID attempt's files are in loadvoid/; the last attempt of each arm stands whatever its load.

- `qwen3evalce/e4b/fused_attn4_m_e0 attempt 0 load1_median 1.22 gate 6.0 status ok over 0`
- `qwen3evalce/e4b/fused_attn4_m_e1 attempt 0 load1_median 1.14 gate 6.0 status ok over 0`
- `qwen3evalce/e4b/fused_attn4_m_e1_d2 attempt 0 load1_median 1.3 gate 6.0 status ok over 0`
- `qwen3evalce/e4b/fused_attn4_m_e0_d2 attempt 0 load1_median 1.24 gate 6.0 status ok over 0`
- `qwen3evalce/unsloth/ckpt_unsloth_m_ee attempt 0 load1_median 1.19 gate 6.0 status ok over 0`
