# TC1 — Qwen3-30B-A3B on one RTX 5090: e4b vs Unsloth vs HF+PEFT vs axolotl at matched work, init and adapter precision (tc1)
Rule (tc1/TC1-PREREG.md): status per attempt in the vocabulary OK / REFUSED / OOM / INSTALL_FAILED / LOAD_FAULT / HARNESS_ERROR / ALARM / NOT_RUN; VALID/VOID per tp4's predicates plus the matched-set predicates (R3: matched init complete, fp32 adapters, step-0 held-out within 0.005 of e4b/reference_attn4_m); VERDICT exactly one of VALID / VOID / QUALITY_FAIL / OOM / UNSUPPORTED / HARNESS_ERROR / ALARM / NOT_RUN (QUALITY_FAIL = held-out |Δ| at N > 0.05 vs e4b/fused_attn4_m); positions = s/step ratios other/e4b from the medians over both draws (stability |d1-d2|/mean ≤ 5 % e4b / 10 % others, else UNSTABLE), quoted only when both arms are VALID; equivalence vs e4b/reference_attn4_m EQUIVALENT ≤ 0.02 / COMPARABLE ≤ 0.05 / else DIVERGENT; frozen base SAME-BYTES / DIFFERENT / N-A per slot; predictions P1–P10 (+ P1b) scored HELD / FALSIFIED / UNTESTED. VOID never enters a ratio or an equivalence reading. Nothing is licensed; no cross-box number is divided into these.
`versions.txt`
```
e4b 0.48.0 @5c908866f73ab7d63f87ff8ecac291e2f92d1c7c (GitHub main)
gnf4 0.41.0 @ccf4de91e9138764e4635d42dc10cca005ef26f4 (GitHub main)
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
e4b(t212) 0.48.0 @5c908866f73ab7d63f87ff8ecac291e2f92d1c7c
gnf4(t212) 0.41.0 @ccf4de91e9138764e4635d42dc10cca005ef26f4
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
 "run_id": "tc1-5090-94",
 "instance_id": "54328898",
 "gpu": "NVIDIA GeForce RTX 5090",
 "driver": "595.91.07",
 "cpu": "AMD EPYC 7K62 48-Core Processor",
 "nproc": 96,
 "mem_total_kb": "527994548",
 "cgroup_memory_max": "259519414272",
 "disk_root": "overlay         320G   25M  320G   1% /",
 "hostname": "c8845f0930e7",
 "registered_gpu_class": "5090",
 "prereg": "tc1/TC1-PREREG.md",
 "local_box": null,
 "local_snapshot": null
}
```

### Qwen3-30B-A3B (amendment 42: the matched set with e4b and Unsloth on one stack, a second host, the current code) (`qwen3samestackh2`, registered n_layers 48, attention census 192)
- model `Qwen/Qwen3-30B-A3B` @ `ad44e777bcd1`; tokens sha `bfc742f67e37`; N=60; fixture template alpaca seq 2048 micro-batch 2 × accum 4 lr 0.0002 r 16 α 16 optimizer adamw_8bit(lr=0.0002, weight_decay=0.001) schedule=linear warmup_steps=5 autocast False; e4b trainable 642514944; box_class RTX 5090 gpu NVIDIA GeForce RTX 5090
| framework | arm | status | validity | **VERDICT** | matched | init / dtype | step-0 class | N | s/step med(11+) | tok/s | peak GB | J/step | train first→last | held-out 0→final | quality Δ | engagement | trainable | regime | reason / why |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| e4b | fused_attn4_m | **OK** | VALID | **VALID** | yes | matched:3407 (complete 12480/12480) / float32 | SAME-BYTES-CLASS (0.0000) | 60 | 4.143 | 362.3 | 27.502 | 1075.0 | 2.0554→0.8182 | 1.9478→0.7578 | 0.0000 | patched 48 / kcalls 768 | 642514944 | 4-bit experts (e4b NF4) + NF4 attention |  |
| unsloth | ckpt_unsloth_m | **OK** | VALID | **VALID** | yes | matched:3407 (complete 12480/12480) / float32 | SAME-BYTES-CLASS (0.0045) | 60 | 10.155 | 151.8 | 24.273 | 1523.0 | 2.0593→0.8182 | 1.9523→0.7566 | -0.0012 | stacks 96 / fwd 384 / u8 48 | 642514944 | 4-bit expert stacks + bnb-4bit attention (Params4bit stacks 96, Linear4bit 384) |  |
| e4b | reference_attn4_m | **NOT_RUN** | — | **NOT_RUN** | yes | — / — | — | 60 | — | — | — | — | —→— | —→— | — | patched — / kcalls — | — | — | skipped by TC1_SKIP |
| e4b | fused_attn4_m_t28 | **OK** | VALID | **VALID** | yes | matched:3407 (complete 12480/12480) / float32 | SAME-BYTES-CLASS (0.0038) | 60 | 4.734 | 318.1 | 27.448 | 1133.7 | 2.0614→0.8192 | 1.9441→0.7583 | 0.0005 | patched 48 / kcalls 768 | 642514944 | 4-bit experts (e4b NF4) + NF4 attention |  |
| e4b | fused_attn4_m_t28_d2 | **OK** | VALID | **VALID** | yes | matched:3407 (complete 12480/12480) / float32 | SAME-BYTES-CLASS (0.0038) | 60 | 4.709 | 333.0 | 27.445 | 1121.9 | 2.0614→0.8189 | 1.9441→0.7552 | -0.0026 | patched 48 / kcalls 768 | 642514944 | 4-bit experts (e4b NF4) + NF4 attention |  |
| unsloth | ckpt_unsloth_m_d2 | **OK** | VALID | **VALID** | yes | matched:3407 (complete 12480/12480) / float32 | SAME-BYTES-CLASS (0.0045) | 60 | 10.120 | 156.0 | 24.273 | 1464.5 | 2.0593→0.8167 | 1.9523→0.7589 | 0.0010 | stacks 96 / fwd 384 / u8 48 | 642514944 | 4-bit expert stacks + bnb-4bit attention (Params4bit stacks 96, Linear4bit 384) |  |
| e4b | fused_attn4_m_d2 | **OK** | VALID | **VALID** | yes | matched:3407 (complete 12480/12480) / float32 | SAME-BYTES-CLASS (0.0000) | 60 | 4.073 | 384.1 | 27.486 | 1045.7 | 2.0554→0.8210 | 1.9478→0.7561 | -0.0017 | patched 48 / kcalls 768 | 642514944 | 4-bit experts (e4b NF4) + NF4 attention |  |
- prologue `e4b/fused_attn4_m` **130.2 s** before step 1 (33% of the arm): c1_before 72.6, c1_after 37.1, load_weights 20.9, eval0 16.0; unattributed 2.269; budget 1260.0
- prologue `unsloth/ckpt_unsloth_m` **139.9 s** before step 1 (18% of the arm): c1_before 67.1, c1_after 33.8, load_weights 33.6, eval0 18.3; unattributed 12.385; budget 1260.0
- prologue `e4b/fused_attn4_m_t28` **132.2 s** before step 1 (30% of the arm): c1_before 73.6, c1_after 37.1, load_weights 23.3, eval0 14.6; unattributed 1.454; budget 1260.0
- prologue `e4b/fused_attn4_m_t28_d2` **121.3 s** before step 1 (29% of the arm): c1_before 73.5, c1_after 37.2, load_weights 23.0, attn4 7.4; unattributed 1.45; budget 1260.0
- prologue `unsloth/ckpt_unsloth_m_d2` **133.8 s** before step 1 (18% of the arm): c1_before 68.2, c1_after 34.0, load_weights 33.5, eval0 10.8; unattributed 12.937; budget 1260.0
- prologue `e4b/fused_attn4_m_d2` **118.8 s** before step 1 (32% of the arm): c1_before 74.5, c1_after 37.4, load_weights 20.8, attn4 7.5; unattributed 2.362; budget 1260.0
- draws (R1): `e4b/fused_attn4_m` STABLE (4.143/4.073 s, |Δ|/mean 1.7% vs 5%); `unsloth/ckpt_unsloth_m` STABLE (10.155/10.120 s, |Δ|/mean 0.3% vs 5%); `e4b/fused_attn4_m_t28` STABLE (4.734/4.709 s, |Δ|/mean 0.5% vs 5%)
- e4b internal parity: NO-REF — reference arm missing or not OK
- **MATCHED POSITION: s/step ratio unsloth/e4b = 2.468** [2.443, 2.493 over 4 cross-draw ratios] (10.138 vs 4.108 s, medians over 2/2 draws; e4b faster per step); peak VRAM unsloth 24.27 vs e4b 27.49 GB (Δ -3.22); J/step unsloth 1493.7 vs e4b 1060.4 (×1.409); tok/s unsloth 153.9 vs e4b 373.2; unsloth regime: 4-bit expert stacks + bnb-4bit attention (Params4bit stacks 96, Linear4bit 384)
- quality reading at N=60 (unsloth): held-out e4b 0.7578 / unsloth 0.7566 (Δ -0.0012) → **COMPARABLE** (|Δ| ≤ 0.05 reads COMPARABLE); step-0 Δ +0.0045
- **NO MATCHED POSITION QUOTED (hf)** — not quoted: e4b STABLE / hf no receipt
- **NO MATCHED POSITION QUOTED (axolotl)** — not quoted: e4b STABLE / axolotl no receipt
- equivalence vs `e4b/fused_attn4_m` (I/H; EQUIVALENT iff both ≤ band = max(0.005, 3 × fused-vs-reference |Δ|) = none (no reference arm: COMPARABLE is the ceiling); COMPARABLE ≤ 0.05; draw-noise floor 0.0030): `unsloth/ckpt_unsloth_m` **COMPARABLE** (median step |Δ| 0.0013, |Δ held-out at N| 0.0012, step-0 0.0045 SAME-BYTES-CLASS, |Δ loss at step 2| 0.0253, paired rows mean -0.0013 ± 0.0030 SE over 8, favouring arm 4 / anchor 4) — no reference arm ran: the EQUIVALENT band cannot be set, COMPARABLE is the ceiling; `e4b/reference_attn4_m` **—** — no OK receipt; `e4b/fused_attn4_m_t28` **COMPARABLE** (median step |Δ| 0.0015, |Δ held-out at N| 0.0005, step-0 0.0038 SAME-BYTES-CLASS, |Δ loss at step 2| 0.0208, paired rows mean +0.0004 ± 0.0013 SE over 8, favouring arm 4 / anchor 4) — no reference arm ran: the EQUIVALENT band cannot be set, COMPARABLE is the ceiling; `e4b/fused_attn4_m_t28_d2` **COMPARABLE** (median step |Δ| 0.0013, |Δ held-out at N| 0.0026, step-0 0.0038 SAME-BYTES-CLASS, |Δ loss at step 2| 0.0150, paired rows mean -0.0026 ± 0.0018 SE over 8, favouring arm 5 / anchor 3) — no reference arm ran: the EQUIVALENT band cannot be set, COMPARABLE is the ceiling; `unsloth/ckpt_unsloth_m_d2` **COMPARABLE** (median step |Δ| 0.0012, |Δ held-out at N| 0.0010, step-0 0.0045 SAME-BYTES-CLASS, |Δ loss at step 2| 0.0119, paired rows mean +0.0010 ± 0.0010 SE over 8, favouring arm 3 / anchor 5) — no reference arm ran: the EQUIVALENT band cannot be set, COMPARABLE is the ceiling; `e4b/fused_attn4_m_d2` **COMPARABLE** (median step |Δ| 0.0010, |Δ held-out at N| 0.0017, step-0 0.0000 SAME-BYTES-CLASS, |Δ loss at step 2| 0.0079, paired rows mean -0.0017 ± 0.0021 SE over 8, favouring arm 6 / anchor 2) — no reference arm ran: the EQUIVALENT band cannot be set, COMPARABLE is the ceiling
- matched_init_sha (B, name-free, canonical slot order): anchor `f7832488926eda91`; `e4b/fused_attn4_m` same; `unsloth/ckpt_unsloth_m` same; `e4b/fused_attn4_m_t28` same; `e4b/fused_attn4_m_t28_d2` same; `unsloth/ckpt_unsloth_m_d2` same; `e4b/fused_attn4_m_d2` same
- frozen base (R5), anchor `e4b/fused_attn4_m`: gate_up nf4/64 sha 8c3b3fd78e89 control detects, down nf4/64 sha 59569d810a1a control detects, q_proj nf4/64+dq sha 5c5e402ce51d control detects
- frozen base `unsloth/ckpt_unsloth_m`: gate_up **N-A** (regime differs: anchor nf4/64 vs nf4/64+dq) control detects, down **N-A** (regime differs: anchor nf4/64 vs nf4/64+dq) control detects, q_proj **SAME-BYTES** (nf4/64+dq) control detects
- frozen base `e4b/fused_attn4_m_t28`: gate_up **SAME-BYTES** (nf4/64) control detects, down **SAME-BYTES** (nf4/64) control detects, q_proj **SAME-BYTES** (nf4/64+dq) control detects
- frozen base `e4b/fused_attn4_m_t28_d2`: gate_up **SAME-BYTES** (nf4/64) control detects, down **SAME-BYTES** (nf4/64) control detects, q_proj **SAME-BYTES** (nf4/64+dq) control detects
- frozen base `unsloth/ckpt_unsloth_m_d2`: gate_up **N-A** (regime differs: anchor nf4/64 vs nf4/64+dq) control detects, down **N-A** (regime differs: anchor nf4/64 vs nf4/64+dq) control detects, q_proj **SAME-BYTES** (nf4/64+dq) control detects
- frozen base `e4b/fused_attn4_m_d2`: gate_up **SAME-BYTES** (nf4/64) control detects, down **SAME-BYTES** (nf4/64) control detects, q_proj **SAME-BYTES** (nf4/64+dq) control detects

## Verdicts
| family | arm | VERDICT | validity | quality | note |
|---|---|---|---|---|---|
| qwen3samestackh2 | e4b/fused_attn4_m | **VALID** | VALID | 0.0000 |  |
| qwen3samestackh2 | unsloth/ckpt_unsloth_m | **VALID** | VALID | -0.0012 |  |
| qwen3samestackh2 | e4b/reference_attn4_m | **NOT_RUN** | — | — | skipped by TC1_SKIP |
| qwen3samestackh2 | e4b/fused_attn4_m_t28 | **VALID** | VALID | 0.0005 |  |
| qwen3samestackh2 | e4b/fused_attn4_m_t28_d2 | **VALID** | VALID | -0.0026 |  |
| qwen3samestackh2 | unsloth/ckpt_unsloth_m_d2 | **VALID** | VALID | 0.0010 |  |
| qwen3samestackh2 | e4b/fused_attn4_m_d2 | **VALID** | VALID | -0.0017 |  |

## Predictions P94 / P95 (TC1-PREREG amendment 42: the same-stack position on a second host, the current code; scored mechanically)
| prediction | family | verdict | evidence |
|---|---|---|---|
| P94 | qwen3samestackh2 | **HELD** | Unsloth/e4b on one stack 2.468 [2.443, 2.493 over 4 cross-draw ratios] vs [1.9, 2.9]; s/step e4b 4.108 (within 1.7%), Unsloth 10.138 (within 0.3%); peak e4b 27.49 / Unsloth 24.27 GB; held-out at N e4b 0.7578 / Unsloth 0.7566 |
| P95 | qwen3samestackh2 | **HELD** | e4b matched arm venv-unsloth / venv-e4b 0.870 [0.860, 0.880] vs [0.8, 0.95]; s/step venv-unsloth 4.143 / 4.073, venv-e4b 4.734 / 4.709; amendment 33 read 0.900 on an EPYC 7B13 |

## Load gate (TC1-PREREG amendment 33): 0 draw(s) voided for host load and run again
Each line: the arm, the attempt, the median host load1 over that attempt's run, the gate. A VOID attempt's files are in loadvoid/; the last attempt of each arm stands whatever its load.

- `qwen3samestackh2/e4b/fused_attn4_m attempt 0 load1_median 2.77 gate 6.0 status ok over 0`
- `qwen3samestackh2/unsloth/ckpt_unsloth_m attempt 0 load1_median 2.93 gate 6.0 status ok over 0`
- `qwen3samestackh2/e4b/fused_attn4_m_t28 attempt 0 load1_median 3.01 gate 6.0 status ok over 0`
- `qwen3samestackh2/e4b/fused_attn4_m_t28_d2 attempt 0 load1_median 2.95 gate 6.0 status ok over 0`
- `qwen3samestackh2/unsloth/ckpt_unsloth_m_d2 attempt 0 load1_median 1.72 gate 6.0 status ok over 0`
- `qwen3samestackh2/e4b/fused_attn4_m_d2 attempt 0 load1_median 1.65 gate 6.0 status ok over 0`
