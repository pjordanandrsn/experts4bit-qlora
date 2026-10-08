# TC1 — Qwen3-30B-A3B on one RTX 5090: e4b vs Unsloth vs HF+PEFT vs axolotl at matched work, init and adapter precision (tc1)
Rule (tc1/TC1-PREREG.md): status per attempt in the vocabulary OK / REFUSED / OOM / INSTALL_FAILED / LOAD_FAULT / HARNESS_ERROR / ALARM / NOT_RUN; VALID/VOID per tp4's predicates plus the matched-set predicates (R3: matched init complete, fp32 adapters, step-0 held-out within 0.005 of e4b/reference_attn4_m); VERDICT exactly one of VALID / VOID / QUALITY_FAIL / OOM / UNSUPPORTED / HARNESS_ERROR / ALARM / NOT_RUN (QUALITY_FAIL = held-out |Δ| at N > 0.05 vs e4b/fused_attn4_m); positions = s/step ratios other/e4b from the medians over both draws (stability |d1-d2|/mean ≤ 5 % e4b / 10 % others, else UNSTABLE), quoted only when both arms are VALID; equivalence vs e4b/reference_attn4_m EQUIVALENT ≤ 0.02 / COMPARABLE ≤ 0.05 / else DIVERGENT; frozen base SAME-BYTES / DIFFERENT / N-A per slot; predictions P1–P10 (+ P1b) scored HELD / FALSIFIED / UNTESTED. VOID never enters a ratio or an equivalence reading. Nothing is licensed; no cross-box number is divided into these.
`versions.txt`
```
e4b 0.49.0 @737f02ff45319f06116f75c6d4e5ed29970e5c5e (GitHub main)
gnf4 0.43.0 @8371c38dc6a4ebfbf54c850bf9cbb53a4120a258 (GitHub main)
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
e4b(t212) 0.49.0 @737f02ff45319f06116f75c6d4e5ed29970e5c5e
gnf4(t212) 0.43.0 @8371c38dc6a4ebfbf54c850bf9cbb53a4120a258
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
 "run_id": "tc1-5090-137",
 "instance_id": "54809007",
 "gpu": "NVIDIA GeForce RTX 5090",
 "driver": "580.126.09",
 "cpu": "AMD EPYC 7713 64-Core Processor",
 "nproc": 128,
 "mem_total_kb": "527972068",
 "cgroup_memory_max": "",
 "disk_root": "overlay         320G   45M  320G   1% /",
 "hostname": "b1de4496f26a",
 "cgroup_cpu_max": "",
 "affinity_cpus": 128,
 "cgroup_cpuset_effective": "",
 "registered_gpu_class": "5090",
 "prereg": "tc1/TC1-PREREG.md",
 "local_box": null,
 "local_snapshot": null
}
```

### Qwen3-30B-A3B (amendment 69: the field recipe's same-stack position at the new defaults, profiled -- e4b defaults vs Unsloth, two draws each) (`qwen3pos69`, registered n_layers 48, attention census 192)
- model `Qwen/Qwen3-30B-A3B` @ `ad44e777bcd1`; tokens sha `bfc742f67e37`; N=60; fixture template alpaca seq 2048 micro-batch 2 × accum 4 lr 0.0002 r 16 α 16 optimizer adamw_8bit(lr=0.0002, weight_decay=0.001) schedule=linear warmup_steps=5 autocast False; e4b trainable 642514944; box_class RTX 5090 gpu NVIDIA GeForce RTX 5090
| framework | arm | status | validity | **VERDICT** | matched | init / dtype | step-0 class | N | s/step med(11+) | tok/s | peak GB | J/step | train first→last | held-out 0→final | quality Δ | engagement | trainable | regime | reason / why |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| e4b | fused_attn4_m_fd | **OK** | VALID | **VALID** | yes | matched:3407 (complete 12480/12480) / float32 | SAME-BYTES-CLASS (0.0000) | 60 | 3.430 | 439.1 | 26.156 | 682.4 | 2.0722→0.8177 | 1.9592→0.7569 | 0.0000 | patched 48 / kcalls 768 | 642514944 | 4-bit experts (e4b NF4) + NF4 attention |  |
| unsloth | ckpt_unsloth_m_fv | **OK** | VALID | **VALID** | yes | matched:3407 (complete 12480/12480) / float32 | SAME-BYTES-CLASS (0.0034) | 60 | 9.833 | 144.8 | 24.273 | 483.2 | 2.0705→0.8202 | 1.9558→0.7582 | 0.0014 | stacks 96 / fwd 384 / u8 48 | 642514944 | 4-bit expert stacks + bnb-4bit attention (Params4bit stacks 96, Linear4bit 384) |  |
| unsloth | ckpt_unsloth_m_fv_d2 | **OK** | VALID | **VALID** | yes | matched:3407 (complete 12480/12480) / float32 | SAME-BYTES-CLASS (0.0034) | 60 | 9.727 | 156.0 | 24.273 | 307.7 | 2.0705→0.8207 | 1.9558→0.7570 | 0.0001 | stacks 96 / fwd 384 / u8 48 | 642514944 | 4-bit expert stacks + bnb-4bit attention (Params4bit stacks 96, Linear4bit 384) |  |
| e4b | fused_attn4_m_fd_d2 | **OK** | VALID | **VALID** | yes | matched:3407 (complete 12480/12480) / float32 | SAME-BYTES-CLASS (0.0000) | 60 | 3.548 | 445.4 | 26.165 | 698.4 | 2.0722→0.8158 | 1.9592→0.7560 | -0.0009 | patched 48 / kcalls 768 | 642514944 | 4-bit experts (e4b NF4) + NF4 attention |  |
- prologue `e4b/fused_attn4_m_fd` **117.6 s** before step 1 (24% of the arm): c1_before 58.8, load_weights 27.6, c1_after 18.5, eval0 12.9; unattributed 2.232; budget 1260.0
- prologue `unsloth/ckpt_unsloth_m_fv` **122.9 s** before step 1 (5% of the arm): c1_before 60.4, load_weights 26.2, eval0 17.5, c1_after 11.8; unattributed 10.805; budget 1260.0
- prologue `unsloth/ckpt_unsloth_m_fv_d2` **141.9 s** before step 1 (5% of the arm): c1_before 59.6, load_weights 52.4, c1_after 11.0, eval0 9.6; unattributed 11.026; budget 1260.0
- prologue `e4b/fused_attn4_m_fd_d2` **101.0 s** before step 1 (22% of the arm): c1_before 58.3, load_weights 22.5, c1_after 18.4, attn4 5.8; unattributed 2.024; budget 1260.0
- draws (R1): `e4b/fused_attn4_m_fd` STABLE (3.430/3.548 s, |Δ|/mean 3.4% vs 5%); `unsloth/ckpt_unsloth_m_fv` STABLE (9.833/9.727 s, |Δ|/mean 1.1% vs 5%)
- e4b internal parity: NO-ARM
- **NO MATCHED POSITION QUOTED (unsloth)** — not quoted: e4b STABLE / unsloth no receipt
- **NO MATCHED POSITION QUOTED (hf)** — not quoted: e4b STABLE / hf no receipt
- **NO MATCHED POSITION QUOTED (axolotl)** — not quoted: e4b STABLE / axolotl no receipt
- equivalence vs `e4b/fused_attn4_m` (I/H; EQUIVALENT iff both ≤ band = max(0.005, 3 × fused-vs-reference |Δ|) = none (no reference arm: COMPARABLE is the ceiling); COMPARABLE ≤ 0.05; draw-noise floor 0.0013): `unsloth/ckpt_unsloth_m_fv` **COMPARABLE** (median step |Δ| 0.0012, |Δ held-out at N| 0.0014, step-0 0.0034 SAME-BYTES-CLASS, |Δ loss at step 2| 0.0119, paired rows mean +0.0014 ± 0.0018 SE over 8, favouring arm 2 / anchor 6) — no reference arm ran: the EQUIVALENT band cannot be set, COMPARABLE is the ceiling; `unsloth/ckpt_unsloth_m_fv_d2` **COMPARABLE** (median step |Δ| 0.0017, |Δ held-out at N| 0.0001, step-0 0.0034 SAME-BYTES-CLASS, |Δ loss at step 2| 0.0120, paired rows mean +0.0001 ± 0.0026 SE over 8, favouring arm 2 / anchor 6) — no reference arm ran: the EQUIVALENT band cannot be set, COMPARABLE is the ceiling; `e4b/fused_attn4_m_fd_d2` **COMPARABLE** (median step |Δ| 0.0011, |Δ held-out at N| 0.0009, step-0 0.0000 SAME-BYTES-CLASS, |Δ loss at step 2| 0.0125, paired rows mean -0.0009 ± 0.0017 SE over 8, favouring arm 4 / anchor 4) — no reference arm ran: the EQUIVALENT band cannot be set, COMPARABLE is the ceiling
- matched_init_sha (B, name-free, canonical slot order): anchor `f7832488926eda91`; `e4b/fused_attn4_m_fd` same; `unsloth/ckpt_unsloth_m_fv` same; `unsloth/ckpt_unsloth_m_fv_d2` same; `e4b/fused_attn4_m_fd_d2` same
- frozen base (R5), anchor `e4b/fused_attn4_m`: gate_up nf4/64+dq sha e2bed944143e control detects, down nf4/64+dq sha f2ade29dafdb control detects, q_proj nf4/64+dq sha 5c5e402ce51d control detects
- frozen base `unsloth/ckpt_unsloth_m_fv`: gate_up **SAME-BYTES** (nf4/64+dq) control detects, down **SAME-BYTES** (nf4/64+dq) control detects, q_proj **SAME-BYTES** (nf4/64+dq) control detects
- frozen base `unsloth/ckpt_unsloth_m_fv_d2`: gate_up **SAME-BYTES** (nf4/64+dq) control detects, down **SAME-BYTES** (nf4/64+dq) control detects, q_proj **SAME-BYTES** (nf4/64+dq) control detects
- frozen base `e4b/fused_attn4_m_fd_d2`: gate_up **SAME-BYTES** (nf4/64+dq) control detects, down **SAME-BYTES** (nf4/64+dq) control detects, q_proj **SAME-BYTES** (nf4/64+dq) control detects

## Verdicts
| family | arm | VERDICT | validity | quality | note |
|---|---|---|---|---|---|
| qwen3pos69 | e4b/fused_attn4_m_fd | **VALID** | VALID | 0.0000 |  |
| qwen3pos69 | unsloth/ckpt_unsloth_m_fv | **VALID** | VALID | 0.0014 |  |
| qwen3pos69 | unsloth/ckpt_unsloth_m_fv_d2 | **VALID** | VALID | 0.0001 |  |
| qwen3pos69 | e4b/fused_attn4_m_fd_d2 | **VALID** | VALID | -0.0009 |  |

## Amendment 69: the field recipe's same-stack position profiled -- device time and host share (descriptive)
| arm | VERDICT | s/step (11..N) | device ms / profiled step | busy_t | CPU ops / profiled step | device events / profiled step | peak GB | held-out N |
|---|---|---|---|---|---|---|---|---|
| e4b/fused_attn4_m_fd | VALID | 3.430 | 1692.3 | 0.493 | 527263 | 83691 | 26.156 | 0.75689 |
| unsloth/ckpt_unsloth_m_fv | VALID | 9.833 | 3261.3 | 0.332 | 7430405 | 611890 | 24.273 | 0.75825 |
| unsloth/ckpt_unsloth_m_fv_d2 | VALID | 9.727 | 3257.0 | 0.335 | 7430785 | 611956 | 24.273 | 0.75697 |
| e4b/fused_attn4_m_fd_d2 | VALID | 3.548 | 1695.6 | 0.478 | 527202 | 83690 | 26.165 | 0.75598 |

## Predictions P205-P209 (TC1-PREREG amendment 69: the field recipe's same-stack position profiled; scored mechanically)
| prediction | family | verdict | evidence |
|---|---|---|---|
| P205 | qwen3pos69 | **HELD** | device ms per profiled step Unsloth 3259.1 / e4b 1693.9 = 1.924 vs in [1.7, 2.7] |
| P206 | qwen3pos69 | **HELD** | s/step Unsloth 9.780 / e4b 3.489 = 2.803 vs in [2.3, 3.3] on this host |
| P207 | qwen3pos69 | **HELD** | CPU ops per profiled step Unsloth 7430595 / e4b 527232 = 14.09 vs >= 8.0 |
| P208 | qwen3pos69 | **HELD** | s/step ratio 2.803 over device ratio 1.924 = 1.457 (busy_t e4b 0.486, Unsloth 0.333) vs >= 1.05 |
| P209 | qwen3pos69 | **HELD** | mean held-out at N e4b defaults 0.75643, Unsloth 0.75761 (diff -0.00118 vs |.| <= 0.01) |

## Load gate (TC1-PREREG amendment 33): 0 draw(s) voided for host load and run again
Each line: the arm, the attempt, the median host load1 over that attempt's run, the gate. A VOID attempt's files are in loadvoid/; the last attempt of each arm stands whatever its load.

- `qwen3pos69/e4b/fused_attn4_m_fd attempt 0 load1_median 4.44 gate 6.0 status ok over 0`
- `qwen3pos69/unsloth/ckpt_unsloth_m_fv attempt 0 load1_median 5.07 gate 6.0 status ok over 0`
- `qwen3pos69/unsloth/ckpt_unsloth_m_fv_d2 attempt 0 load1_median 4.36 gate 6.0 status ok over 0`
- `qwen3pos69/e4b/fused_attn4_m_fd_d2 attempt 0 load1_median 5.33 gate 6.0 status ok over 0`
