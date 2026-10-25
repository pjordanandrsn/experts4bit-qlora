# TC1 — Qwen3-30B-A3B on one RTX 5090: e4b vs Unsloth vs HF+PEFT vs axolotl at matched work, init and adapter precision (tc1)
Rule (tc1/TC1-PREREG.md): status per attempt in the vocabulary OK / REFUSED / OOM / INSTALL_FAILED / LOAD_FAULT / HARNESS_ERROR / ALARM / NOT_RUN; VALID/VOID per tp4's predicates plus the matched-set predicates (R3: matched init complete, fp32 adapters, step-0 held-out within 0.005 of e4b/reference_attn4_m); VERDICT exactly one of VALID / VOID / QUALITY_FAIL / OOM / UNSUPPORTED / HARNESS_ERROR / ALARM / NOT_RUN (QUALITY_FAIL = held-out |Δ| at N > 0.05 vs e4b/fused_attn4_m); positions = s/step ratios other/e4b from the medians over both draws (stability |d1-d2|/mean ≤ 5 % e4b / 10 % others, else UNSTABLE), quoted only when both arms are VALID; equivalence vs e4b/reference_attn4_m EQUIVALENT ≤ 0.02 / COMPARABLE ≤ 0.05 / else DIVERGENT; frozen base SAME-BYTES / DIFFERENT / N-A per slot; predictions P1–P10 (+ P1b) scored HELD / FALSIFIED / UNTESTED. VOID never enters a ratio or an equivalence reading. Nothing is licensed; no cross-box number is divided into these.
`versions.txt`
```
e4b 0.58.0 @f15cb7d695cf7148067c6bcea543484f31dc3abd (GitHub main)
gnf4 0.51.0 @30adb1bfa76f95f9ce4a4c29734b8f36de907eca (GitHub main)
torch(e4b/hf) 2.8.0+cu128
triton(e4b/hf) 3.4.0
transformers(e4b/hf) 5.18.0
bitsandbytes(e4b/hf) 0.50.2
peft(hf) 0.21.2
huggingface_hub(e4b/hf) 2.3.0
httpx2(e4b/hf) 2.16.0
brotli(e4b/hf) 1.2.1
unsloth(unsloth-t28) 2026.10.8
unsloth_zoo(unsloth-t28) 2026.10.8
torch(unsloth-t28) 2.8.0+cu128
triton(unsloth-t28) 3.4.0
transformers(unsloth-t28) 5.17.0
bitsandbytes(unsloth-t28) 0.50.2
peft(unsloth-t28) 0.21.2
torchao(unsloth-t28) None
moe_backend(unsloth-t28) unsloth_triton
unsloth(unsloth) 2026.10.8
unsloth_zoo(unsloth) 2026.10.8
torch(unsloth) 2.12.1+cu130
triton(unsloth) 3.7.1
transformers(unsloth) 5.17.0
bitsandbytes(unsloth) 0.50.2
peft(unsloth) 0.21.2
torchao(unsloth) 0.18.0+cu130
moe_backend(unsloth) grouped_mm
e4b(t212) 0.58.0 @f15cb7d695cf7148067c6bcea543484f31dc3abd
gnf4(t212) 0.51.0 @30adb1bfa76f95f9ce4a4c29734b8f36de907eca
torch(e4b-t212) 2.12.1+cu130
```
`box.json`
```
{
 "box": "A",
 "run_id": "tc1-5090-pos79-r1",
 "gpu": "NVIDIA GeForce RTX 5090",
 "driver": "615.71.09",
 "cpu": "Intel(R) Xeon(R) CPU E5-2698 v4 @ 2.20GHz",
 "nproc": 40,
 "mem_total_kb": "263747500",
 "cgroup_memory_max": "183333027840",
 "disk_root": "overlay         220G  3.0M  220G   1% /",
 "cgroup_cpu_max": "1920000 100000",
 "affinity_cpus": 40,
 "cgroup_cpuset_effective": "0-39",
 "registered_gpu_class": "5090",
 "prereg": "tc1/TC1-PREREG.md",
 "local_box": null,
 "local_snapshot": null
}
```

### Qwen3-30B-A3B (amendment 79: amendment 69's field-recipe position at the 0.58.0 release against Unsloth 2026.10.8, profiled, two draws each) (`qwen3pos79`, registered n_layers 48, attention census 192)
- model `Qwen/Qwen3-30B-A3B` @ `ad44e777bcd1`; tokens sha `bfc742f67e37`; N=60; fixture template alpaca seq 2048 micro-batch 2 × accum 4 lr 0.0002 r 16 α 16 optimizer adamw_8bit(lr=0.0002, weight_decay=0.001) schedule=linear warmup_steps=5 autocast False; e4b trainable 642514944; box_class RTX 5090 gpu NVIDIA GeForce RTX 5090
| framework | arm | status | validity | **VERDICT** | matched | init / dtype | step-0 class | N | s/step med(11+) | tok/s | peak GB | J/step | train first→last | held-out 0→final | quality Δ | engagement | trainable | regime | reason / why |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| e4b | fused_attn4_m_fd | **OK** | VALID | **VALID** | yes | matched:3407 (complete 12480/12480) / float32 | SAME-BYTES-CLASS (0.0000) | 60 | 4.490 | 336.2 | 26.500 | 658.0 | 2.0685→0.8187 | 1.9784→0.7546 | 0.0000 | patched 48 / kcalls 768 | 642514944 | 4-bit experts (e4b NF4) + NF4 attention |  |
| unsloth | ckpt_unsloth_m_fv | **OK** | VOID | **VOID** | yes | matched:3407 (complete 12480/12480) / float32 | NEAR (0.0284) | 60 | 3.275 | 382.8 | 24.221 | 433.9 | 2.0717→0.8187 | 1.9500→0.7551 | 0.0006 | stacks 96 / fwd 384 / u8 48 | 642514944 | 4-bit expert stacks + bnb-4bit attention (Params4bit stacks 96, Linear4bit 384) | torch._grouped_mm 0 calls/step < 6*48*accum 4 = 1152 |
| unsloth | ckpt_unsloth_m_fv_d2 | **OK** | VOID | **VOID** | yes | matched:3407 (complete 12480/12480) / float32 | NEAR (0.0241) | 60 | 3.235 | 430.5 | 24.221 | 422.8 | 2.0583→0.8166 | 1.9543→0.7544 | -0.0002 | stacks 96 / fwd 384 / u8 48 | 642514944 | 4-bit expert stacks + bnb-4bit attention (Params4bit stacks 96, Linear4bit 384) | torch._grouped_mm 0 calls/step < 6*48*accum 4 = 1152 |
| e4b | fused_attn4_m_fd_d2 | **OK** | VALID | **VALID** | yes | matched:3407 (complete 12480/12480) / float32 | SAME-BYTES-CLASS (0.0000) | 60 | 4.377 | 362.2 | 26.500 | 591.7 | 2.0685→0.8164 | 1.9784→0.7563 | 0.0017 | patched 48 / kcalls 768 | 642514944 | 4-bit experts (e4b NF4) + NF4 attention |  |
- prologue `e4b/fused_attn4_m_fd` **191.8 s** before step 1 (29% of the arm): c1_before 93.7, c1_after 44.5, load_weights 28.3, eval0 18.4; unattributed 5.952; budget 1260.0
- prologue `unsloth/ckpt_unsloth_m_fv` **211.9 s** before step 1 (38% of the arm): c1_before 106.8, c1_after 53.1, load_weights 40.6, eval0 23.0; unattributed 24.449; budget 1260.0
- prologue `unsloth/ckpt_unsloth_m_fv_d2` **195.8 s** before step 1 (40% of the arm): c1_before 103.9, c1_after 52.1, load_weights 34.6, eval0 22.3; unattributed 16.29; budget 1260.0
- prologue `e4b/fused_attn4_m_fd_d2` **170.0 s** before step 1 (27% of the arm): c1_before 94.4, c1_after 44.9, load_weights 23.0, eval0 12.2; unattributed 2.822; budget 1260.0
- draws (R1): `e4b/fused_attn4_m_fd` STABLE (4.490/4.377 s, |Δ|/mean 2.6% vs 5%); `unsloth/ckpt_unsloth_m_fv` — (unsloth/ckpt_unsloth_m_fv is VOID)
- e4b internal parity: NO-ARM
- **NO MATCHED POSITION QUOTED (unsloth)** — not quoted: e4b STABLE / unsloth no receipt
- **NO MATCHED POSITION QUOTED (hf)** — not quoted: e4b STABLE / hf no receipt
- **NO MATCHED POSITION QUOTED (axolotl)** — not quoted: e4b STABLE / axolotl no receipt
- equivalence vs `e4b/fused_attn4_m` (I/H; EQUIVALENT iff both ≤ band = max(0.005, 3 × fused-vs-reference |Δ|) = none (no reference arm: COMPARABLE is the ceiling); COMPARABLE ≤ 0.05; draw-noise floor 0.0017): `unsloth/ckpt_unsloth_m_fv` **N-A** — validity: anchor VALID, arm VOID (VOID never enters an equivalence reading); `unsloth/ckpt_unsloth_m_fv_d2` **N-A** — validity: anchor VALID, arm VOID (VOID never enters an equivalence reading); `e4b/fused_attn4_m_fd_d2` **COMPARABLE** (median step |Δ| 0.0015, |Δ held-out at N| 0.0017, step-0 0.0000 SAME-BYTES-CLASS, |Δ loss at step 2| 0.0230, paired rows mean +0.0018 ± 0.0020 SE over 8, favouring arm 4 / anchor 4) — no reference arm ran: the EQUIVALENT band cannot be set, COMPARABLE is the ceiling
- matched_init_sha (B, name-free, canonical slot order): anchor `f7832488926eda91`; `e4b/fused_attn4_m_fd` same; `unsloth/ckpt_unsloth_m_fv` same; `unsloth/ckpt_unsloth_m_fv_d2` same; `e4b/fused_attn4_m_fd_d2` same
- frozen base (R5), anchor `e4b/fused_attn4_m`: gate_up nf4/64+dq sha e2bed944143e control detects, down nf4/64+dq sha f2ade29dafdb control detects, q_proj nf4/64 sha 5c5e402ce51d control detects
- frozen base `unsloth/ckpt_unsloth_m_fv`: gate_up **SAME-BYTES** (nf4/64+dq) control detects, down **SAME-BYTES** (nf4/64+dq) control detects, q_proj **N-A** (regime differs: anchor nf4/64 vs nf4/64+dq) control detects
- frozen base `unsloth/ckpt_unsloth_m_fv_d2`: gate_up **SAME-BYTES** (nf4/64+dq) control detects, down **SAME-BYTES** (nf4/64+dq) control detects, q_proj **N-A** (regime differs: anchor nf4/64 vs nf4/64+dq) control detects
- frozen base `e4b/fused_attn4_m_fd_d2`: gate_up **SAME-BYTES** (nf4/64+dq) control detects, down **SAME-BYTES** (nf4/64+dq) control detects, q_proj **SAME-BYTES** (nf4/64) control detects

## Verdicts
| family | arm | VERDICT | validity | quality | note |
|---|---|---|---|---|---|
| qwen3pos79 | e4b/fused_attn4_m_fd | **VALID** | VALID | 0.0000 |  |
| qwen3pos79 | unsloth/ckpt_unsloth_m_fv | **VOID** | VOID | 0.0006 | torch._grouped_mm 0 calls/step < 6*48*accum 4 = 1152 |
| qwen3pos79 | unsloth/ckpt_unsloth_m_fv_d2 | **VOID** | VOID | -0.0002 | torch._grouped_mm 0 calls/step < 6*48*accum 4 = 1152 |
| qwen3pos79 | e4b/fused_attn4_m_fd_d2 | **VALID** | VALID | 0.0017 |  |

## Amendment 79: the field recipe's same-stack position at the 0.58.0 release against Unsloth 2026.10.8 (profiled)
| arm | VERDICT | s/step (11..N) | device ms / profiled step | busy_t | CPU ops / profiled step | peak GB | held-out N |
|---|---|---|---|---|---|---|---|
| e4b/fused_attn4_m_fd | VALID | 4.490 | 2064.8 | 0.460 | 497556 | 26.500 | 0.75455 |
| unsloth/ckpt_unsloth_m_fv | VOID | 3.275 | 1725.9 | 0.527 | 200437 | 24.221 | 0.75511 |
| unsloth/ckpt_unsloth_m_fv_d2 | VOID | 3.235 | 1468.9 | 0.454 | 159119 | 24.221 | 0.75436 |
| e4b/fused_attn4_m_fd_d2 | VALID | 4.377 | 2049.9 | 0.468 | 497517 | 26.500 | 0.75629 |

**Amendment 79 rule: NOT_READ**
- two VALID profiled draws a side are registered: e4b ok, Unsloth missing

## Load gate (TC1-PREREG amendment 33): 0 draw(s) voided for host load and run again
Each line: the arm, the attempt, the median host load1 over that attempt's run, the gate. A VOID attempt's files are in loadvoid/; the last attempt of each arm stands whatever its load.

- `qwen3pos79/e4b/fused_attn4_m_fd attempt 0 load1_median 2.0 gate 6.0 status ok over 0`
- `qwen3pos79/unsloth/ckpt_unsloth_m_fv attempt 0 load1_median 2.21 gate 6.0 status ok over 0`
- `qwen3pos79/unsloth/ckpt_unsloth_m_fv_d2 attempt 0 load1_median 2.37 gate 6.0 status ok over 0`
- `qwen3pos79/e4b/fused_attn4_m_fd_d2 attempt 0 load1_median 2.23 gate 6.0 status ok over 0`
