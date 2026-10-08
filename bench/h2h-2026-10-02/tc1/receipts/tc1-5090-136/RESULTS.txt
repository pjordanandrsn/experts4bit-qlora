# TC1 — Qwen3-30B-A3B on one RTX 5090: e4b vs Unsloth vs HF+PEFT vs axolotl at matched work, init and adapter precision (tc1)
Rule (tc1/TC1-PREREG.md): status per attempt in the vocabulary OK / REFUSED / OOM / INSTALL_FAILED / LOAD_FAULT / HARNESS_ERROR / ALARM / NOT_RUN; VALID/VOID per tp4's predicates plus the matched-set predicates (R3: matched init complete, fp32 adapters, step-0 held-out within 0.005 of e4b/reference_attn4_m); VERDICT exactly one of VALID / VOID / QUALITY_FAIL / OOM / UNSUPPORTED / HARNESS_ERROR / ALARM / NOT_RUN (QUALITY_FAIL = held-out |Δ| at N > 0.05 vs e4b/fused_attn4_m); positions = s/step ratios other/e4b from the medians over both draws (stability |d1-d2|/mean ≤ 5 % e4b / 10 % others, else UNSTABLE), quoted only when both arms are VALID; equivalence vs e4b/reference_attn4_m EQUIVALENT ≤ 0.02 / COMPARABLE ≤ 0.05 / else DIVERGENT; frozen base SAME-BYTES / DIFFERENT / N-A per slot; predictions P1–P10 (+ P1b) scored HELD / FALSIFIED / UNTESTED. VOID never enters a ratio or an equivalence reading. Nothing is licensed; no cross-box number is divided into these.
`versions.txt`
```
e4b 0.48.0 @0b2995236f7a5d446299a252db0a7beb1d212280 (GitHub main)
gnf4 0.42.0 @8944aa84857349cb98dce30cdf21f56c5a3640b6 (GitHub main)
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
e4b(t212) 0.48.0 @0b2995236f7a5d446299a252db0a7beb1d212280
gnf4(t212) 0.42.0 @8944aa84857349cb98dce30cdf21f56c5a3640b6
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
 "run_id": "tc1-5090-136",
 "instance_id": "54768641",
 "gpu": "NVIDIA GeForce RTX 5090",
 "driver": "610.57.04",
 "cpu": "AMD Eng Sample",
 "nproc": 64,
 "mem_total_kb": "263621836",
 "cgroup_memory_max": "183247044608",
 "disk_root": "overlay         320G   36M  320G   1% /",
 "hostname": "e6f2605a2f49",
 "cgroup_cpu_max": "3071999 100000",
 "affinity_cpus": 64,
 "cgroup_cpuset_effective": "0-63",
 "registered_gpu_class": "5090",
 "prereg": "tc1/TC1-PREREG.md",
 "local_box": null,
 "local_snapshot": null
}
```

### Qwen3-30B-A3B (amendment 68: amendment 67's packed-row position profiled -- e4b defaults vs Unsloth, two draws each; device time and host share) (`qwen3pos68`, registered n_layers 48, attention census 192)
- model `Qwen/Qwen3-30B-A3B` @ `ad44e777bcd1`; tokens sha `d2a501eba57d`; N=40; fixture template alpaca seq 4096 (packed rows, amendment 39) micro-batch 1 × accum 4 lr 0.0002 r 16 α 16 optimizer adamw_8bit(lr=0.0002, weight_decay=0.001) schedule=linear warmup_steps=5 autocast False; e4b trainable 642514944; box_class RTX 5090 gpu NVIDIA GeForce RTX 5090
| framework | arm | status | validity | **VERDICT** | matched | init / dtype | step-0 class | N | s/step med(11+) | tok/s | peak GB | J/step | train first→last | held-out 0→final | quality Δ | engagement | trainable | regime | reason / why |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| e4b | fused_attn4_m_pd | **OK** | VALID | **VALID** | yes | matched:3407 (complete 12480/12480) / float32 | SAME-BYTES-CLASS (0.0000) | 40 | 10.108 | 1595.6 | 25.898 | 2656.9 | 1.2578→0.9131 | 1.2885→0.9545 | 0.0000 | patched 48 / kcalls 768 | 642514944 | 4-bit experts (e4b NF4) + NF4 attention |  |
| unsloth | ckpt_unsloth_m_pv | **OK** | VALID | **VALID** | yes | matched:3407 (complete 12480/12480) / float32 | SAME-BYTES-CLASS (0.0014) | 40 | 14.891 | 1055.2 | 24.864 | 1020.1 | 1.2587→0.9123 | 1.2871→0.9543 | -0.0002 | stacks 96 / fwd 384 / u8 48 | 642514944 | 4-bit expert stacks + bnb-4bit attention (Params4bit stacks 96, Linear4bit 384) |  |
| unsloth | ckpt_unsloth_m_pv_d2 | **OK** | VALID | **VALID** | yes | matched:3407 (complete 12480/12480) / float32 | SAME-BYTES-CLASS (0.0014) | 40 | 14.880 | 1066.6 | 24.864 | 1017.8 | 1.2587→0.9134 | 1.2871→0.9542 | -0.0003 | stacks 96 / fwd 384 / u8 48 | 642514944 | 4-bit expert stacks + bnb-4bit attention (Params4bit stacks 96, Linear4bit 384) |  |
| e4b | fused_attn4_m_pd_d2 | **OK** | VALID | **VALID** | yes | matched:3407 (complete 12480/12480) / float32 | SAME-BYTES-CLASS (0.0000) | 40 | 10.114 | 1604.0 | 25.919 | 2526.4 | 1.2578→0.9127 | 1.2885→0.9542 | -0.0002 | patched 48 / kcalls 768 | 642514944 | 4-bit experts (e4b NF4) + NF4 attention |  |
- prologue `e4b/fused_attn4_m_pd` **88.3 s** before step 1 (12% of the arm): c1_before 45.6, load_weights 16.4, eval0 11.0, c1_after 10.4; unattributed 1.745; budget 1890.0
- prologue `unsloth/ckpt_unsloth_m_pv` **102.1 s** before step 1 (4% of the arm): c1_before 45.2, load_weights 23.6, eval0 17.9, c1_after 10.5; unattributed 8.478; budget 1890.0
- prologue `unsloth/ckpt_unsloth_m_pv_d2` **101.7 s** before step 1 (4% of the arm): c1_before 47.7, load_weights 26.2, eval0 11.5, c1_after 10.1; unattributed 9.339; budget 1890.0
- prologue `e4b/fused_attn4_m_pd_d2` **90.7 s** before step 1 (12% of the arm): c1_before 47.0, load_weights 22.0, c1_after 10.4, eval0 6.5; unattributed 1.807; budget 1890.0
- draws (R1): `e4b/fused_attn4_m_pd` STABLE (10.108/10.114 s, |Δ|/mean 0.0% vs 5%); `unsloth/ckpt_unsloth_m_pv` STABLE (14.891/14.880 s, |Δ|/mean 0.1% vs 5%)
- e4b internal parity: NO-ARM
- **NO MATCHED POSITION QUOTED (unsloth)** — not quoted: e4b STABLE / unsloth no receipt
- **NO MATCHED POSITION QUOTED (hf)** — not quoted: e4b STABLE / hf no receipt
- **NO MATCHED POSITION QUOTED (axolotl)** — not quoted: e4b STABLE / axolotl no receipt
- equivalence vs `e4b/fused_attn4_m` (I/H; EQUIVALENT iff both ≤ band = max(0.005, 3 × fused-vs-reference |Δ|) = none (no reference arm: COMPARABLE is the ceiling); COMPARABLE ≤ 0.05; draw-noise floor 0.0002): `unsloth/ckpt_unsloth_m_pv` **COMPARABLE** (median step |Δ| 0.0003, |Δ held-out at N| 0.0002, step-0 0.0014 SAME-BYTES-CLASS, |Δ loss at step 2| 0.0075, paired rows mean -0.0002 ± 0.0002 SE over 8, favouring arm 5 / anchor 3) — no reference arm ran: the EQUIVALENT band cannot be set, COMPARABLE is the ceiling; `unsloth/ckpt_unsloth_m_pv_d2` **COMPARABLE** (median step |Δ| 0.0003, |Δ held-out at N| 0.0003, step-0 0.0014 SAME-BYTES-CLASS, |Δ loss at step 2| 0.0031, paired rows mean -0.0003 ± 0.0002 SE over 8, favouring arm 5 / anchor 3) — no reference arm ran: the EQUIVALENT band cannot be set, COMPARABLE is the ceiling; `e4b/fused_attn4_m_pd_d2` **COMPARABLE** (median step |Δ| 0.0003, |Δ held-out at N| 0.0002, step-0 0.0000 SAME-BYTES-CLASS, |Δ loss at step 2| 0.0019, paired rows mean -0.0003 ± 0.0002 SE over 8, favouring arm 5 / anchor 3) — no reference arm ran: the EQUIVALENT band cannot be set, COMPARABLE is the ceiling
- matched_init_sha (B, name-free, canonical slot order): anchor `f7832488926eda91`; `e4b/fused_attn4_m_pd` same; `unsloth/ckpt_unsloth_m_pv` same; `unsloth/ckpt_unsloth_m_pv_d2` same; `e4b/fused_attn4_m_pd_d2` same
- frozen base (R5), anchor `e4b/fused_attn4_m`: gate_up nf4/64+dq sha e2bed944143e control detects, down nf4/64+dq sha f2ade29dafdb control detects, q_proj nf4/64+dq sha 5c5e402ce51d control detects
- frozen base `unsloth/ckpt_unsloth_m_pv`: gate_up **SAME-BYTES** (nf4/64+dq) control detects, down **SAME-BYTES** (nf4/64+dq) control detects, q_proj **SAME-BYTES** (nf4/64+dq) control detects
- frozen base `unsloth/ckpt_unsloth_m_pv_d2`: gate_up **SAME-BYTES** (nf4/64+dq) control detects, down **SAME-BYTES** (nf4/64+dq) control detects, q_proj **SAME-BYTES** (nf4/64+dq) control detects
- frozen base `e4b/fused_attn4_m_pd_d2`: gate_up **SAME-BYTES** (nf4/64+dq) control detects, down **SAME-BYTES** (nf4/64+dq) control detects, q_proj **SAME-BYTES** (nf4/64+dq) control detects

## Verdicts
| family | arm | VERDICT | validity | quality | note |
|---|---|---|---|---|---|
| qwen3pos68 | e4b/fused_attn4_m_pd | **VALID** | VALID | 0.0000 |  |
| qwen3pos68 | unsloth/ckpt_unsloth_m_pv | **VALID** | VALID | -0.0002 |  |
| qwen3pos68 | unsloth/ckpt_unsloth_m_pv_d2 | **VALID** | VALID | -0.0003 |  |
| qwen3pos68 | e4b/fused_attn4_m_pd_d2 | **VALID** | VALID | -0.0002 |  |

## Amendment 68: the packed-row position profiled -- device time and host share (descriptive)
| arm | VERDICT | s/step (11..N) | device ms / profiled step | busy_t | CPU ops / profiled step | device events / profiled step | train peak GB | held-out N |
|---|---|---|---|---|---|---|---|---|
| e4b/fused_attn4_m_pd | VALID | 10.108 | 10028.7 | 0.992 | 1075993 | 159754 | 25.898 | 0.95446 |
| unsloth/ckpt_unsloth_m_pv | VALID | 14.891 | 11649.9 | 0.782 | 7659165 | 759464 | 24.864 | 0.95426 |
| unsloth/ckpt_unsloth_m_pv_d2 | VALID | 14.880 | 11648.9 | 0.783 | 7659024 | 759406 | 24.864 | 0.95418 |
| e4b/fused_attn4_m_pd_d2 | VALID | 10.114 | 10057.0 | 0.994 | 1083338 | 161055 | 25.919 | 0.95421 |

## Predictions P200-P204 (TC1-PREREG amendment 68: the packed-row position profiled; scored mechanically)
| prediction | family | verdict | evidence |
|---|---|---|---|
| P200 | qwen3pos68 | **HELD** | device ms per profiled step Unsloth 11649.4 / e4b 10042.8 = 1.160 vs in [1.06, 1.24]; s/step Unsloth / e4b 1.472 on this host (descriptive) |
| P201 | qwen3pos68 | **HELD** | busy_t e4b 0.993, Unsloth 0.783 (gap +0.211 vs >= 0.08) |
| P202 | qwen3pos68 | **HELD** | CPU ops per profiled step Unsloth 7659094 / e4b 1079666 = 7.09 vs >= 5.0 |
| P203 | qwen3pos68 | **FALSIFIED** | Unsloth s/step 14.886 >= 14.0: busy_t 0.783 vs <= 0.75; device ms per profiled step 11649.4 vs 10152.2 (+14.7% vs |.| <= 8%) |
| P204 | qwen3pos68 | **HELD** | mean held-out at N e4b defaults 0.95433, Unsloth 0.95422 (diff +0.00011 vs |.| <= 0.01) |

## Load gate (TC1-PREREG amendment 33): 0 draw(s) voided for host load and run again
Each line: the arm, the attempt, the median host load1 over that attempt's run, the gate. A VOID attempt's files are in loadvoid/; the last attempt of each arm stands whatever its load.

- `qwen3pos68/e4b/fused_attn4_m_pd attempt 0 load1_median 1.47 gate 6.0 status ok over 0`
- `qwen3pos68/unsloth/ckpt_unsloth_m_pv attempt 0 load1_median 1.49 gate 6.0 status ok over 0`
- `qwen3pos68/unsloth/ckpt_unsloth_m_pv_d2 attempt 0 load1_median 1.4 gate 6.0 status ok over 0`
- `qwen3pos68/e4b/fused_attn4_m_pd_d2 attempt 0 load1_median 1.36 gate 6.0 status ok over 0`
