# TC1 — Qwen3-30B-A3B on one RTX 5090: e4b vs Unsloth vs HF+PEFT vs axolotl at matched work, init and adapter precision (tc1c-h100-22)
Rule (tc1/TC1-PREREG.md): status per attempt in the vocabulary OK / REFUSED / OOM / INSTALL_FAILED / LOAD_FAULT / HARNESS_ERROR / ALARM / NOT_RUN; VALID/VOID per tp4's predicates plus the matched-set predicates (R3: matched init complete, fp32 adapters, step-0 held-out within 0.005 of e4b/reference_attn4_m); VERDICT exactly one of VALID / VOID / QUALITY_FAIL / OOM / UNSUPPORTED / HARNESS_ERROR / ALARM / NOT_RUN (QUALITY_FAIL = held-out |Δ| at N > 0.05 vs e4b/fused_attn4_m); positions = s/step ratios other/e4b from the medians over both draws (stability |d1-d2|/mean ≤ 5 % e4b / 10 % others, else UNSTABLE), quoted only when both arms are VALID; equivalence vs e4b/reference_attn4_m EQUIVALENT ≤ 0.02 / COMPARABLE ≤ 0.05 / else DIVERGENT; frozen base SAME-BYTES / DIFFERENT / N-A per slot; predictions P1–P10 (+ P1b) scored HELD / FALSIFIED / UNTESTED. VOID never enters a ratio or an equivalence reading. Nothing is licensed; no cross-box number is divided into these.
`versions.txt`
```
e4b 0.48.0 @16074418a7e171f55dd983c4a1ba5c7d91df9948 (GitHub main)
gnf4 0.41.0 @f127981fded98c2fa24acd6b4ae1eb0929ddb192 (GitHub main)
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
e4b(t212) 0.48.0 @16074418a7e171f55dd983c4a1ba5c7d91df9948
gnf4(t212) 0.41.0 @f127981fded98c2fa24acd6b4ae1eb0929ddb192
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
 "run_id": "tc1c-h100-22",
 "instance_id": "lj6bxnvudzxiv7",
 "gpu": "NVIDIA H100 NVL",
 "driver": "580.159.04",
 "cpu": "Intel(R) Xeon(R) Platinum 8452Y",
 "nproc": 144,
 "mem_total_kb": "1188615704",
 "cgroup_memory_max": "",
 "disk_root": "overlay         320G  124M  320G   1% /",
 "hostname": "47052476a018",
 "registered_gpu_class": "H100 NVL",
 "prereg": "tc1/TC1-PREREG.md",
 "local_box": null,
 "local_snapshot": null
}
```

### Qwen3-30B-A3B on an H100 NVL (TC1c amendment 9: the matched set with e4b and Unsloth on one stack, torch 2.12.1+cu130 / transformers 5.5.0) (`qwen3samestackh100`, registered n_layers 48, attention census 192)
- model `Qwen/Qwen3-30B-A3B` @ `ad44e777bcd1`; tokens sha `bfc742f67e37`; N=60; fixture template alpaca seq 2048 micro-batch 2 × accum 4 lr 0.0002 r 16 α 16 optimizer adamw_8bit(lr=0.0002, weight_decay=0.001) schedule=linear warmup_steps=5 autocast False; e4b trainable 642514944; box_class H100 NVL gpu NVIDIA H100 NVL
| framework | arm | status | validity | **VERDICT** | matched | init / dtype | step-0 class | N | s/step med(11+) | tok/s | peak GB | J/step | train first→last | held-out 0→final | quality Δ | engagement | trainable | regime | reason / why |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| e4b | fused_attn4_m | **OK** | VALID | **VALID** | yes | matched:3407 (complete 12480/12480) / float32 | SAME-BYTES-CLASS (0.0000) | 60 | 2.968 | 512.0 | 27.489 | 518.7 | 2.0761→0.8192 | 1.9602→0.7558 | 0.0000 | patched 48 / kcalls 768 | 642514944 | 4-bit experts (e4b NF4) + NF4 attention |  |
| unsloth | ckpt_unsloth_m | **OK** | VALID | **VALID** | yes | matched:3407 (complete 12480/12480) / float32 | NEAR (0.0313) | 60 | 3.127 | 454.2 | 24.273 | 430.1 | 2.0566→0.8149 | 1.9289→0.7557 | -0.0001 | stacks 96 / fwd 384 / u8 48 | 642514944 | 4-bit expert stacks + bnb-4bit attention (Params4bit stacks 96, Linear4bit 384) |  |
| e4b | reference_attn4_m | **NOT_RUN** | — | **NOT_RUN** | yes | — / — | — | 60 | — | — | — | — | —→— | —→— | — | patched — / kcalls — | — | — | skipped by TC1_SKIP |
| e4b | fused_attn4_m_t28 | **OK** | VALID | **VALID** | yes | matched:3407 (complete 12480/12480) / float32 | SAME-BYTES-CLASS (0.0012) | 60 | 3.428 | 468.1 | 27.514 | 539.7 | 2.0773→0.8190 | 1.9614→0.7557 | -0.0001 | patched 48 / kcalls 768 | 642514944 | 4-bit experts (e4b NF4) + NF4 attention |  |
| e4b | fused_attn4_m_t28_d2 | **OK** | VALID | **VALID** | yes | matched:3407 (complete 12480/12480) / float32 | SAME-BYTES-CLASS (0.0012) | 60 | 2.568 | 603.7 | 27.491 | 507.1 | 2.0773→0.8210 | 1.9614→0.7552 | -0.0006 | patched 48 / kcalls 768 | 642514944 | 4-bit experts (e4b NF4) + NF4 attention |  |
| unsloth | ckpt_unsloth_m_d2 | **OK** | VALID | **VALID** | yes | matched:3407 (complete 12480/12480) / float32 | NEAR (0.0313) | 60 | 3.021 | 487.5 | 24.273 | 423.3 | 2.0566→0.8180 | 1.9289→0.7565 | 0.0007 | stacks 96 / fwd 384 / u8 48 | 642514944 | 4-bit expert stacks + bnb-4bit attention (Params4bit stacks 96, Linear4bit 384) |  |
| e4b | fused_attn4_m_d2 | **OK** | VALID | **VALID** | yes | matched:3407 (complete 12480/12480) / float32 | SAME-BYTES-CLASS (0.0000) | 60 | 3.557 | 450.6 | 27.493 | 531.6 | 2.0761→0.8169 | 1.9602→0.7561 | 0.0003 | patched 48 / kcalls 768 | 642514944 | 4-bit experts (e4b NF4) + NF4 attention |  |
- prologue `e4b/fused_attn4_m` **105.6 s** before step 1 (36% of the arm): c1_before 63.7, c1_after 30.0, load_weights 21.1, attn4 5.5; unattributed 2.861; budget 1260.0
- prologue `unsloth/ckpt_unsloth_m` **126.3 s** before step 1 (37% of the arm): c1_before 63.3, load_weights 33.9, c1_after 32.0, eval0 8.6; unattributed 12.029; budget 1260.0
- prologue `e4b/fused_attn4_m_t28` **132.8 s** before step 1 (39% of the arm): c1_before 77.1, c1_after 39.2, load_weights 28.4, attn4 8.7; unattributed 1.85; budget 1260.0
- prologue `e4b/fused_attn4_m_t28_d2` **105.2 s** before step 1 (39% of the arm): c1_before 63.0, c1_after 32.8, load_weights 22.3, attn4 5.6; unattributed 1.706; budget 1260.0
- prologue `unsloth/ckpt_unsloth_m_d2` **123.6 s** before step 1 (38% of the arm): c1_before 59.8, load_weights 34.9, c1_after 27.7, eval0 7.0; unattributed 11.142; budget 1260.0
- prologue `e4b/fused_attn4_m_d2` **107.2 s** before step 1 (33% of the arm): c1_before 63.1, c1_after 30.8, load_weights 21.5, attn4 8.0; unattributed 2.131; budget 970.2
- draws (R1): `e4b/fused_attn4_m` UNSTABLE (2.968/3.557 s, |Δ|/mean 18.1% vs 5%); `unsloth/ckpt_unsloth_m` STABLE (3.127/3.021 s, |Δ|/mean 3.5% vs 5%); `e4b/fused_attn4_m_t28` UNSTABLE (3.428/2.568 s, |Δ|/mean 28.7% vs 5%)
- e4b internal parity: NO-REF — reference arm missing or not OK
- **NO MATCHED POSITION QUOTED (unsloth)** — not quoted: e4b draws 2.968 / 3.557 s/step differ by 18.1% > 5% (UNSTABLE: reported, not quoted) / unsloth STABLE
- **NO MATCHED POSITION QUOTED (hf)** — not quoted: e4b draws 2.968 / 3.557 s/step differ by 18.1% > 5% (UNSTABLE: reported, not quoted) / hf no receipt
- **NO MATCHED POSITION QUOTED (axolotl)** — not quoted: e4b draws 2.968 / 3.557 s/step differ by 18.1% > 5% (UNSTABLE: reported, not quoted) / axolotl no receipt
- equivalence vs `e4b/fused_attn4_m` (I/H; EQUIVALENT iff both ≤ band = max(0.005, 3 × fused-vs-reference |Δ|) = none (no reference arm: COMPARABLE is the ceiling); COMPARABLE ≤ 0.05; draw-noise floor 0.0008): `unsloth/ckpt_unsloth_m` **COMPARABLE** (median step |Δ| 0.0014, |Δ held-out at N| 0.0001, step-0 0.0313 NEAR, |Δ loss at step 2| 0.0022, paired rows mean -0.0001 ± 0.0016 SE over 8, favouring arm 4 / anchor 4) — no reference arm ran: the EQUIVALENT band cannot be set, COMPARABLE is the ceiling; `e4b/reference_attn4_m` **—** — no OK receipt; `e4b/fused_attn4_m_t28` **COMPARABLE** (median step |Δ| 0.0010, |Δ held-out at N| 0.0001, step-0 0.0012 SAME-BYTES-CLASS, |Δ loss at step 2| 0.0048, paired rows mean -0.0001 ± 0.0028 SE over 8, favouring arm 3 / anchor 5) — no reference arm ran: the EQUIVALENT band cannot be set, COMPARABLE is the ceiling; `e4b/fused_attn4_m_t28_d2` **COMPARABLE** (median step |Δ| 0.0013, |Δ held-out at N| 0.0006, step-0 0.0012 SAME-BYTES-CLASS, |Δ loss at step 2| 0.0013, paired rows mean -0.0006 ± 0.0023 SE over 8, favouring arm 5 / anchor 3) — no reference arm ran: the EQUIVALENT band cannot be set, COMPARABLE is the ceiling; `unsloth/ckpt_unsloth_m_d2` **COMPARABLE** (median step |Δ| 0.0016, |Δ held-out at N| 0.0007, step-0 0.0313 NEAR, |Δ loss at step 2| 0.0137, paired rows mean +0.0007 ± 0.0017 SE over 8, favouring arm 4 / anchor 4) — no reference arm ran: the EQUIVALENT band cannot be set, COMPARABLE is the ceiling; `e4b/fused_attn4_m_d2` **COMPARABLE** (median step |Δ| 0.0012, |Δ held-out at N| 0.0003, step-0 0.0000 SAME-BYTES-CLASS, |Δ loss at step 2| 0.0071, paired rows mean +0.0003 ± 0.0011 SE over 8, favouring arm 4 / anchor 4) — no reference arm ran: the EQUIVALENT band cannot be set, COMPARABLE is the ceiling
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
| qwen3samestackh100 | e4b/fused_attn4_m | **VALID** | VALID | 0.0000 |  |
| qwen3samestackh100 | unsloth/ckpt_unsloth_m | **VALID** | VALID | -0.0001 |  |
| qwen3samestackh100 | e4b/reference_attn4_m | **NOT_RUN** | — | — | skipped by TC1_SKIP |
| qwen3samestackh100 | e4b/fused_attn4_m_t28 | **VALID** | VALID | -0.0001 |  |
| qwen3samestackh100 | e4b/fused_attn4_m_t28_d2 | **VALID** | VALID | -0.0006 |  |
| qwen3samestackh100 | unsloth/ckpt_unsloth_m_d2 | **VALID** | VALID | 0.0007 |  |
| qwen3samestackh100 | e4b/fused_attn4_m_d2 | **VALID** | VALID | 0.0003 |  |

## Predictions P27 / P28 / P29 (TC1C-PREREG amendment 9: the H100 position with both frameworks on one stack; scored mechanically)
| prediction | family | verdict | evidence |
|---|---|---|---|
| P27 | qwen3samestackh100 | **UNTESTED** | two stable VALID draws a side are registered -- not quoted: e4b draws 2.968 / 3.557 s/step differ by 18.1% > 5% (UNSTABLE: reported, not quoted) / unsloth STABLE |
| P28 | qwen3samestackh100 | **UNTESTED** | two stable VALID draws a side are registered -- venv-unsloth UNSTABLE: draws 2.968 / 3.557 s/step differ by 18.1% > 5% (UNSTABLE: reported, not quoted); venv-e4b UNSTABLE: draws 3.428 / 2.568 s/step differ by 28.7% > 5% (UNSTABLE: reported, not quoted) |
| P29 | qwen3samestackh100 | **HELD** | 4 e4b receipts; route grouped_mm with GNF4_TRAIN_GEMM unset on all |

## Load gate (TC1-PREREG amendment 33): 12 draw(s) voided for host load and run again
Each line: the arm, the attempt, the median host load1 over that attempt's run, the gate. A VOID attempt's files are in loadvoid/; the last attempt of each arm stands whatever its load.

- `qwen3samestackh100/e4b/fused_attn4_m attempt 0 load1_median 18.7 gate 6.0 status ok over 1`
- `qwen3samestackh100/e4b/fused_attn4_m attempt 0 VOID (host load1 median 18.7 > 6.0): re-run 1 of 2`
- `qwen3samestackh100/e4b/fused_attn4_m attempt 1 load1_median 13.74 gate 6.0 status ok over 1`
- `qwen3samestackh100/e4b/fused_attn4_m attempt 1 VOID (host load1 median 13.74 > 6.0): re-run 2 of 2`
- `qwen3samestackh100/e4b/fused_attn4_m attempt 2 load1_median 14.01 gate 6.0 status ok over 1`
- `qwen3samestackh100/unsloth/ckpt_unsloth_m attempt 0 load1_median 11.1 gate 6.0 status ok over 1`
- `qwen3samestackh100/unsloth/ckpt_unsloth_m attempt 0 VOID (host load1 median 11.1 > 6.0): re-run 1 of 2`
- `qwen3samestackh100/unsloth/ckpt_unsloth_m attempt 1 load1_median 16.87 gate 6.0 status ok over 1`
- `qwen3samestackh100/unsloth/ckpt_unsloth_m attempt 1 VOID (host load1 median 16.87 > 6.0): re-run 2 of 2`
- `qwen3samestackh100/unsloth/ckpt_unsloth_m attempt 2 load1_median 24.29 gate 6.0 status ok over 1`
- `qwen3samestackh100/e4b/fused_attn4_m_t28 attempt 0 load1_median 33.31 gate 6.0 status ok over 1`
- `qwen3samestackh100/e4b/fused_attn4_m_t28 attempt 0 VOID (host load1 median 33.31 > 6.0): re-run 1 of 2`
- `qwen3samestackh100/e4b/fused_attn4_m_t28 attempt 1 load1_median 28.13 gate 6.0 status ok over 1`
- `qwen3samestackh100/e4b/fused_attn4_m_t28 attempt 1 VOID (host load1 median 28.13 > 6.0): re-run 2 of 2`
- `qwen3samestackh100/e4b/fused_attn4_m_t28 attempt 2 load1_median 33.79 gate 6.0 status ok over 1`
- `qwen3samestackh100/e4b/fused_attn4_m_t28_d2 attempt 0 load1_median 42.03 gate 6.0 status ok over 1`
- `qwen3samestackh100/e4b/fused_attn4_m_t28_d2 attempt 0 VOID (host load1 median 42.03 > 6.0): re-run 1 of 2`
- `qwen3samestackh100/e4b/fused_attn4_m_t28_d2 attempt 1 load1_median 27.72 gate 6.0 status ok over 1`
- `qwen3samestackh100/e4b/fused_attn4_m_t28_d2 attempt 1 VOID (host load1 median 27.72 > 6.0): re-run 2 of 2`
- `qwen3samestackh100/e4b/fused_attn4_m_t28_d2 attempt 2 load1_median 14.14 gate 6.0 status ok over 1`
- `qwen3samestackh100/unsloth/ckpt_unsloth_m_d2 attempt 0 load1_median 19.68 gate 6.0 status ok over 1`
- `qwen3samestackh100/unsloth/ckpt_unsloth_m_d2 attempt 0 VOID (host load1 median 19.68 > 6.0): re-run 1 of 2`
- `qwen3samestackh100/unsloth/ckpt_unsloth_m_d2 attempt 1 load1_median 7.52 gate 6.0 status ok over 1`
- `qwen3samestackh100/unsloth/ckpt_unsloth_m_d2 attempt 1 VOID (host load1 median 7.52 > 6.0): re-run 2 of 2`
- `qwen3samestackh100/unsloth/ckpt_unsloth_m_d2 attempt 2 load1_median 10.16 gate 6.0 status ok over 1`
- `qwen3samestackh100/e4b/fused_attn4_m_d2 attempt 0 load1_median 9.4 gate 6.0 status ok over 1`
- `qwen3samestackh100/e4b/fused_attn4_m_d2 attempt 0 VOID (host load1 median 9.4 > 6.0): re-run 1 of 2`
- `qwen3samestackh100/e4b/fused_attn4_m_d2 attempt 1 load1_median 11.26 gate 6.0 status ok over 1`
- `qwen3samestackh100/e4b/fused_attn4_m_d2 attempt 1 VOID (host load1 median 11.26 > 6.0): re-run 2 of 2`
- `qwen3samestackh100/e4b/fused_attn4_m_d2 attempt 2 load1_median 8.72 gate 6.0 status ok over 1`
