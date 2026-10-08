# TC1 — Qwen3-30B-A3B on one RTX 5090: e4b vs Unsloth vs HF+PEFT vs axolotl at matched work, init and adapter precision (tc1)
Rule (tc1/TC1-PREREG.md): status per attempt in the vocabulary OK / REFUSED / OOM / INSTALL_FAILED / LOAD_FAULT / HARNESS_ERROR / ALARM / NOT_RUN; VALID/VOID per tp4's predicates plus the matched-set predicates (R3: matched init complete, fp32 adapters, step-0 held-out within 0.005 of e4b/reference_attn4_m); VERDICT exactly one of VALID / VOID / QUALITY_FAIL / OOM / UNSUPPORTED / HARNESS_ERROR / ALARM / NOT_RUN (QUALITY_FAIL = held-out |Δ| at N > 0.05 vs e4b/fused_attn4_m); positions = s/step ratios other/e4b from the medians over both draws (stability |d1-d2|/mean ≤ 5 % e4b / 10 % others, else UNSTABLE), quoted only when both arms are VALID; equivalence vs e4b/reference_attn4_m EQUIVALENT ≤ 0.02 / COMPARABLE ≤ 0.05 / else DIVERGENT; frozen base SAME-BYTES / DIFFERENT / N-A per slot; predictions P1–P10 (+ P1b) scored HELD / FALSIFIED / UNTESTED. VOID never enters a ratio or an equivalence reading. Nothing is licensed; no cross-box number is divided into these.
`versions.txt`
```
e4b 0.48.0 @08ff61cd9acc2b8c758a02c414ab5670d550d9ad (GitHub main)
gnf4 0.42.0 @9953bab9a94f0d5b3c932fe2e90c22f927d4b550 (GitHub main)
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
e4b(t212) 0.48.0 @08ff61cd9acc2b8c758a02c414ab5670d550d9ad
gnf4(t212) 0.42.0 @9953bab9a94f0d5b3c932fe2e90c22f927d4b550
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
 "run_id": "tc1-5090-135",
 "instance_id": "54735496",
 "gpu": "NVIDIA GeForce RTX 5090",
 "driver": "595.91.07",
 "cpu": "AMD EPYC 7K62 48-Core Processor",
 "nproc": 96,
 "mem_total_kb": "527994548",
 "cgroup_memory_max": "259519414272",
 "disk_root": "overlay         320G   57M  320G   1% /",
 "hostname": "7848a180d442",
 "cgroup_cpu_max": "2304000 100000",
 "affinity_cpus": 96,
 "cgroup_cpuset_effective": "0-95",
 "registered_gpu_class": "5090",
 "prereg": "tc1/TC1-PREREG.md",
 "local_box": null,
 "local_snapshot": null
}
```

### Qwen3-30B-A3B (amendment 67: the packed-row position at the new defaults -- e4b defaults, e4b with the offload, Unsloth; two draws each; peaks by phase) (`qwen3pos67`, registered n_layers 48, attention census 192)
- model `Qwen/Qwen3-30B-A3B` @ `ad44e777bcd1`; tokens sha `d2a501eba57d`; N=40; fixture template alpaca seq 4096 (packed rows, amendment 39) micro-batch 1 × accum 4 lr 0.0002 r 16 α 16 optimizer adamw_8bit(lr=0.0002, weight_decay=0.001) schedule=linear warmup_steps=5 autocast False; e4b trainable 642514944; box_class RTX 5090 gpu NVIDIA GeForce RTX 5090
| framework | arm | status | validity | **VERDICT** | matched | init / dtype | step-0 class | N | s/step med(11+) | tok/s | peak GB | J/step | train first→last | held-out 0→final | quality Δ | engagement | trainable | regime | reason / why |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| e4b | fused_attn4_m_pd | **OK** | VALID | **VALID** | yes | matched:3407 (complete 12480/12480) / float32 | SAME-BYTES-CLASS (0.0000) | 40 | 9.052 | 1770.6 | 25.913 | 4612.7 | 1.2578→0.9132 | 1.2885→0.9543 | 0.0000 | patched 48 / kcalls 768 | 642514944 | 4-bit experts (e4b NF4) + NF4 attention |  |
| unsloth | ckpt_unsloth_m_pv | **OK** | VALID | **VALID** | yes | matched:3407 (complete 12480/12480) / float32 | SAME-BYTES-CLASS (0.0018) | 40 | 16.014 | 1000.5 | 24.864 | 4615.8 | 1.2601→0.9129 | 1.2903→0.9540 | -0.0003 | stacks 96 / fwd 384 / u8 48 | 642514944 | 4-bit expert stacks + bnb-4bit attention (Params4bit stacks 96, Linear4bit 384) |  |
| e4b | fused_attn4_m_po | **OK** | VALID | **VALID** | yes | matched:3407 (complete 12480/12480) / float32 | SAME-BYTES-CLASS (0.0000) | 40 | 9.404 | 1718.6 | 25.164 | 4570.0 | 1.2578→0.9133 | 1.2885→0.9544 | 0.0001 | patched 48 / kcalls 768 | 642514944 | 4-bit experts (e4b NF4) + NF4 attention |  |
| e4b | fused_attn4_m_po_d2 | **OK** | VALID | **VALID** | yes | matched:3407 (complete 12480/12480) / float32 | SAME-BYTES-CLASS (0.0000) | 40 | 9.415 | 1720.9 | 25.170 | 4550.9 | 1.2578→0.9132 | 1.2885→0.9542 | -0.0002 | patched 48 / kcalls 768 | 642514944 | 4-bit experts (e4b NF4) + NF4 attention |  |
| unsloth | ckpt_unsloth_m_pv_d2 | **OK** | VALID | **VALID** | yes | matched:3407 (complete 12480/12480) / float32 | SAME-BYTES-CLASS (0.0018) | 40 | 16.036 | 1012.0 | 24.864 | 4668.7 | 1.2601→0.9130 | 1.2903→0.9544 | 0.0000 | stacks 96 / fwd 384 / u8 48 | 642514944 | 4-bit expert stacks + bnb-4bit attention (Params4bit stacks 96, Linear4bit 384) |  |
| e4b | fused_attn4_m_pd_d2 | **OK** | VALID | **VALID** | yes | matched:3407 (complete 12480/12480) / float32 | SAME-BYTES-CLASS (0.0000) | 40 | 9.025 | 1795.3 | 25.894 | 4578.9 | 1.2578→0.9127 | 1.2885→0.9545 | 0.0002 | patched 48 / kcalls 768 | 642514944 | 4-bit experts (e4b NF4) + NF4 attention |  |
- prologue `e4b/fused_attn4_m_pd` **123.8 s** before step 1 (25% of the arm): c1_before 68.4, c1_after 34.8, load_weights 20.6, eval0 12.8; unattributed 2.598; budget 1890.0
- prologue `unsloth/ckpt_unsloth_m_pv` **147.7 s** before step 1 (18% of the arm): c1_before 68.2, c1_after 34.3, load_weights 33.3, eval0 24.5; unattributed 12.978; budget 1890.0
- prologue `e4b/fused_attn4_m_po` **117.4 s** before step 1 (23% of the arm): c1_before 69.5, c1_after 35.4, load_weights 20.6, attn4 7.4; unattributed 2.585; budget 1890.0
- prologue `e4b/fused_attn4_m_po_d2` **118.4 s** before step 1 (23% of the arm): c1_before 69.9, c1_after 35.2, load_weights 20.9, attn4 7.5; unattributed 2.622; budget 1890.0
- prologue `unsloth/ckpt_unsloth_m_pv_d2` **138.1 s** before step 1 (17% of the arm): c1_before 68.1, c1_after 34.6, load_weights 33.7, eval0 14.6; unattributed 13.135; budget 1890.0
- prologue `e4b/fused_attn4_m_pd_d2` **116.8 s** before step 1 (24% of the arm): c1_before 68.9, c1_after 35.2, load_weights 20.7, attn4 7.4; unattributed 2.587; budget 1890.0
- draws (R1): `e4b/fused_attn4_m_pd` STABLE (9.052/9.025 s, |Δ|/mean 0.3% vs 5%); `unsloth/ckpt_unsloth_m_pv` STABLE (16.014/16.036 s, |Δ|/mean 0.1% vs 5%); `e4b/fused_attn4_m_po` STABLE (9.404/9.415 s, |Δ|/mean 0.1% vs 5%)
- e4b internal parity: NO-ARM
- **NO MATCHED POSITION QUOTED (unsloth)** — not quoted: e4b STABLE / unsloth no receipt
- **NO MATCHED POSITION QUOTED (hf)** — not quoted: e4b STABLE / hf no receipt
- **NO MATCHED POSITION QUOTED (axolotl)** — not quoted: e4b STABLE / axolotl no receipt
- equivalence vs `e4b/fused_attn4_m` (I/H; EQUIVALENT iff both ≤ band = max(0.005, 3 × fused-vs-reference |Δ|) = none (no reference arm: COMPARABLE is the ceiling); COMPARABLE ≤ 0.05; draw-noise floor 0.0003): `unsloth/ckpt_unsloth_m_pv` **COMPARABLE** (median step |Δ| 0.0003, |Δ held-out at N| 0.0003, step-0 0.0018 SAME-BYTES-CLASS, |Δ loss at step 2| 0.0039, paired rows mean -0.0003 ± 0.0003 SE over 8, favouring arm 5 / anchor 3) — no reference arm ran: the EQUIVALENT band cannot be set, COMPARABLE is the ceiling; `e4b/fused_attn4_m_po` **COMPARABLE** (median step |Δ| 0.0003, |Δ held-out at N| 0.0001, step-0 0.0000 SAME-BYTES-CLASS, |Δ loss at step 2| 0.0031, paired rows mean +0.0001 ± 0.0003 SE over 8, favouring arm 2 / anchor 6) — no reference arm ran: the EQUIVALENT band cannot be set, COMPARABLE is the ceiling; `e4b/fused_attn4_m_po_d2` **COMPARABLE** (median step |Δ| 0.0002, |Δ held-out at N| 0.0002, step-0 0.0000 SAME-BYTES-CLASS, |Δ loss at step 2| 0.0014, paired rows mean -0.0002 ± 0.0003 SE over 8, favouring arm 3 / anchor 5) — no reference arm ran: the EQUIVALENT band cannot be set, COMPARABLE is the ceiling; `unsloth/ckpt_unsloth_m_pv_d2` **COMPARABLE** (median step |Δ| 0.0005, |Δ held-out at N| 0.0000, step-0 0.0018 SAME-BYTES-CLASS, |Δ loss at step 2| 0.0027, paired rows mean +0.0000 ± 0.0002 SE over 8, favouring arm 4 / anchor 4) — no reference arm ran: the EQUIVALENT band cannot be set, COMPARABLE is the ceiling; `e4b/fused_attn4_m_pd_d2` **COMPARABLE** (median step |Δ| 0.0003, |Δ held-out at N| 0.0002, step-0 0.0000 SAME-BYTES-CLASS, |Δ loss at step 2| 0.0008, paired rows mean +0.0002 ± 0.0001 SE over 8, favouring arm 3 / anchor 5) — no reference arm ran: the EQUIVALENT band cannot be set, COMPARABLE is the ceiling
- matched_init_sha (B, name-free, canonical slot order): anchor `f7832488926eda91`; `e4b/fused_attn4_m_pd` same; `unsloth/ckpt_unsloth_m_pv` same; `e4b/fused_attn4_m_po` same; `e4b/fused_attn4_m_po_d2` same; `unsloth/ckpt_unsloth_m_pv_d2` same; `e4b/fused_attn4_m_pd_d2` same
- frozen base (R5), anchor `e4b/fused_attn4_m`: gate_up nf4/64+dq sha e2bed944143e control detects, down nf4/64+dq sha f2ade29dafdb control detects, q_proj nf4/64+dq sha 5c5e402ce51d control detects
- frozen base `unsloth/ckpt_unsloth_m_pv`: gate_up **SAME-BYTES** (nf4/64+dq) control detects, down **SAME-BYTES** (nf4/64+dq) control detects, q_proj **SAME-BYTES** (nf4/64+dq) control detects
- frozen base `e4b/fused_attn4_m_po`: gate_up **SAME-BYTES** (nf4/64+dq) control detects, down **SAME-BYTES** (nf4/64+dq) control detects, q_proj **SAME-BYTES** (nf4/64+dq) control detects
- frozen base `e4b/fused_attn4_m_po_d2`: gate_up **SAME-BYTES** (nf4/64+dq) control detects, down **SAME-BYTES** (nf4/64+dq) control detects, q_proj **SAME-BYTES** (nf4/64+dq) control detects
- frozen base `unsloth/ckpt_unsloth_m_pv_d2`: gate_up **SAME-BYTES** (nf4/64+dq) control detects, down **SAME-BYTES** (nf4/64+dq) control detects, q_proj **SAME-BYTES** (nf4/64+dq) control detects
- frozen base `e4b/fused_attn4_m_pd_d2`: gate_up **SAME-BYTES** (nf4/64+dq) control detects, down **SAME-BYTES** (nf4/64+dq) control detects, q_proj **SAME-BYTES** (nf4/64+dq) control detects

## Verdicts
| family | arm | VERDICT | validity | quality | note |
|---|---|---|---|---|---|
| qwen3pos67 | e4b/fused_attn4_m_pd | **VALID** | VALID | 0.0000 |  |
| qwen3pos67 | unsloth/ckpt_unsloth_m_pv | **VALID** | VALID | -0.0003 |  |
| qwen3pos67 | e4b/fused_attn4_m_po | **VALID** | VALID | 0.0001 |  |
| qwen3pos67 | e4b/fused_attn4_m_po_d2 | **VALID** | VALID | -0.0002 |  |
| qwen3pos67 | unsloth/ckpt_unsloth_m_pv_d2 | **VALID** | VALID | 0.0000 |  |
| qwen3pos67 | e4b/fused_attn4_m_pd_d2 | **VALID** | VALID | 0.0002 |  |

## Amendment 67: the packed-row position at the new defaults, peaks by phase (descriptive)
| arm | VERDICT | s/step (11..N) | run peak GB | train | eval | held-out N |
|---|---|---|---|---|---|---|
| e4b/fused_attn4_m_pd | VALID | 9.052 | 25.913 | 25.913 | 22.504 | 0.95432 |
| unsloth/ckpt_unsloth_m_pv | VALID | 16.014 | 24.864 | 24.864 | 21.852 | 0.95401 |
| e4b/fused_attn4_m_po | VALID | 9.404 | 25.164 | 25.164 | 22.504 | 0.95437 |
| e4b/fused_attn4_m_po_d2 | VALID | 9.415 | 25.170 | 25.17 | 22.504 | 0.95415 |
| unsloth/ckpt_unsloth_m_pv_d2 | VALID | 16.036 | 24.864 | 24.864 | 21.852 | 0.95436 |
| e4b/fused_attn4_m_pd_d2 | VALID | 9.025 | 25.894 | 25.894 | 22.504 | 0.95448 |

## Predictions P195-P199 (TC1-PREREG amendment 67: the packed-row position at the new defaults; scored mechanically)
| prediction | family | verdict | evidence |
|---|---|---|---|
| P195 | qwen3pos67 | **FALSIFIED** | pv / pd 1.773 [1.769, 1.777 over 4 cross-draw ratios] vs in [1.15, 1.45]; s/step pv 16.014 / 16.036, pd 9.052 / 9.025 |
| P196 | qwen3pos67 | **HELD** | fused_attn4_m_pd training-phase peak 25.904 vs Unsloth 24.864 GB (gap +1.040 vs <= 1.2) |
| P197 | qwen3pos67 | **HELD** | fused_attn4_m_po training-phase peak 25.167 vs Unsloth 24.864 GB (gap +0.303 vs <= 0.5) |
| P198 | qwen3pos67 | **FALSIFIED** | po / pd 1.041 [1.039, 1.043 over 4 cross-draw ratios] vs <= 1.03; s/step po 9.404 / 9.415, pd 9.052 / 9.025 |
| P199 | qwen3pos67 | **HELD** | mean held-out at N e4b defaults 0.95440, Unsloth 0.95419 (diff +0.00021 vs |.| <= 0.01) |

## Load gate (TC1-PREREG amendment 33): 1 draw(s) voided for host load and run again
Each line: the arm, the attempt, the median host load1 over that attempt's run, the gate. A VOID attempt's files are in loadvoid/; the last attempt of each arm stands whatever its load.

- `qwen3pos67/e4b/fused_attn4_m_pd attempt 0 load1_median 4.06 gate 6.0 status ok over 0`
- `qwen3pos67/unsloth/ckpt_unsloth_m_pv attempt 0 load1_median 3.91 gate 6.0 status ok over 0`
- `qwen3pos67/e4b/fused_attn4_m_po attempt 0 load1_median 3.4 gate 6.0 status ok over 0`
- `qwen3pos67/e4b/fused_attn4_m_po_d2 attempt 0 load1_median 2.89 gate 6.0 status ok over 0`
- `qwen3pos67/unsloth/ckpt_unsloth_m_pv_d2 attempt 0 load1_median 3.46 gate 6.0 status ok over 0`
- `qwen3pos67/e4b/fused_attn4_m_pd_d2 attempt 0 load1_median 12.85 gate 6.0 status ok over 1`
- `qwen3pos67/e4b/fused_attn4_m_pd_d2 attempt 0 VOID (host load1 median 12.85 > 6.0): re-run 1 of 2`
- `qwen3pos67/e4b/fused_attn4_m_pd_d2 attempt 1 load1_median 3.66 gate 6.0 status ok over 0`
