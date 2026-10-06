# TC1 — Qwen3-30B-A3B on one RTX 5090: e4b vs Unsloth vs HF+PEFT vs axolotl at matched work, init and adapter precision (tc1-5090-98)
Rule (tc1/TC1-PREREG.md): status per attempt in the vocabulary OK / REFUSED / OOM / INSTALL_FAILED / LOAD_FAULT / HARNESS_ERROR / ALARM / NOT_RUN; VALID/VOID per tp4's predicates plus the matched-set predicates (R3: matched init complete, fp32 adapters, step-0 held-out within 0.005 of e4b/reference_attn4_m); VERDICT exactly one of VALID / VOID / QUALITY_FAIL / OOM / UNSUPPORTED / HARNESS_ERROR / ALARM / NOT_RUN (QUALITY_FAIL = held-out |Δ| at N > 0.05 vs e4b/fused_attn4_m); positions = s/step ratios other/e4b from the medians over both draws (stability |d1-d2|/mean ≤ 5 % e4b / 10 % others, else UNSTABLE), quoted only when both arms are VALID; equivalence vs e4b/reference_attn4_m EQUIVALENT ≤ 0.02 / COMPARABLE ≤ 0.05 / else DIVERGENT; frozen base SAME-BYTES / DIFFERENT / N-A per slot; predictions P1–P10 (+ P1b) scored HELD / FALSIFIED / UNTESTED. VOID never enters a ratio or an equivalence reading. Nothing is licensed; no cross-box number is divided into these.
`versions.txt`
```
e4b 0.48.0 @ee9d17944027227f5188f3427e3bc0963b2ff989 (GitHub main)
gnf4 0.41.0 @054be19fbb919bc9598d756068a2aed74a8e46ca (GitHub main)
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
e4b(t212) 0.48.0 @ee9d17944027227f5188f3427e3bc0963b2ff989
gnf4(t212) 0.41.0 @054be19fbb919bc9598d756068a2aed74a8e46ca
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
 "run_id": "tc1-5090-98",
 "instance_id": "54379183",
 "gpu": "NVIDIA GeForce RTX 5090",
 "driver": "595.84",
 "cpu": "AMD EPYC 7B13 64-Core Processor",
 "nproc": 256,
 "mem_total_kb": "2101193408",
 "cgroup_memory_max": "730283900928",
 "disk_root": "overlay         320G   77M  320G   1% /",
 "hostname": "32e10d5d029b",
 "cgroup_cpu_max": "3071999 100000",
 "affinity_cpus": 256,
 "cgroup_cpuset_effective": "0-255",
 "registered_gpu_class": "5090",
 "prereg": "tc1/TC1-PREREG.md",
 "local_box": null,
 "local_snapshot": null
}
```

### Qwen3-30B-A3B (amendment 45: OMP_NUM_THREADS at the physical cores vs the container's CPU allotment, e4b and Unsloth in venv-unsloth) (`qwen3ompab`, registered n_layers 48, attention census 192)
- model `Qwen/Qwen3-30B-A3B` @ `ad44e777bcd1`; tokens sha `bfc742f67e37`; N=60; fixture template alpaca seq 2048 micro-batch 2 × accum 4 lr 0.0002 r 16 α 16 optimizer adamw_8bit(lr=0.0002, weight_decay=0.001) schedule=linear warmup_steps=5 autocast False; e4b trainable 642514944; box_class RTX 5090 gpu NVIDIA GeForce RTX 5090
| framework | arm | status | validity | **VERDICT** | matched | init / dtype | step-0 class | N | s/step med(11+) | tok/s | peak GB | J/step | train first→last | held-out 0→final | quality Δ | engagement | trainable | regime | reason / why |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| e4b | fused_attn4_m_om0 | **OK** | VALID | **VALID** | yes | matched:3407 (complete 12480/12480) / float32 | SAME-BYTES-CLASS (0.0000) | 60 | 3.464 | 454.8 | 27.490 | 1069.4 | 2.0554→0.8196 | 1.9478→0.7557 | 0.0000 | patched 48 / kcalls 768 | 642514944 | 4-bit experts (e4b NF4) + NF4 attention |  |
| e4b | fused_attn4_m_om1 | **OK** | VALID | **VALID** | yes | matched:3407 (complete 12480/12480) / float32 | SAME-BYTES-CLASS (0.0000) | 60 | 3.513 | 444.8 | 27.503 | 1067.7 | 2.0554→0.8184 | 1.9478→0.7590 | 0.0033 | patched 48 / kcalls 768 | 642514944 | 4-bit experts (e4b NF4) + NF4 attention |  |
| unsloth | ckpt_unsloth_m_om0 | **OK** | VALID | **VALID** | yes | matched:3407 (complete 12480/12480) / float32 | SAME-BYTES-CLASS (0.0045) | 60 | 8.416 | 184.7 | 24.273 | 1563.3 | 2.0593→0.8163 | 1.9523→0.7568 | 0.0011 | stacks 96 / fwd 384 / u8 48 | 642514944 | 4-bit expert stacks + bnb-4bit attention (Params4bit stacks 96, Linear4bit 384) |  |
| unsloth | ckpt_unsloth_m_om1 | **OK** | VALID | **VALID** | yes | matched:3407 (complete 12480/12480) / float32 | SAME-BYTES-CLASS (0.0045) | 60 | 8.866 | 177.7 | 24.273 | 1597.6 | 2.0593→0.8189 | 1.9523→0.7570 | 0.0013 | stacks 96 / fwd 384 / u8 48 | 642514944 | 4-bit expert stacks + bnb-4bit attention (Params4bit stacks 96, Linear4bit 384) |  |
| unsloth | ckpt_unsloth_m_om1_d2 | **OK** | VALID | **VALID** | yes | matched:3407 (complete 12480/12480) / float32 | SAME-BYTES-CLASS (0.0045) | 60 | 8.437 | 175.8 | 24.273 | 1538.8 | 2.0593→0.8191 | 1.9523→0.7568 | 0.0011 | stacks 96 / fwd 384 / u8 48 | 642514944 | 4-bit expert stacks + bnb-4bit attention (Params4bit stacks 96, Linear4bit 384) |  |
| unsloth | ckpt_unsloth_m_om0_d2 | **OK** | VALID | **VALID** | yes | matched:3407 (complete 12480/12480) / float32 | SAME-BYTES-CLASS (0.0045) | 60 | 8.868 | 178.6 | 24.273 | 1535.8 | 2.0593→0.8163 | 1.9523→0.7600 | 0.0043 | stacks 96 / fwd 384 / u8 48 | 642514944 | 4-bit expert stacks + bnb-4bit attention (Params4bit stacks 96, Linear4bit 384) |  |
| e4b | fused_attn4_m_om1_d2 | **OK** | VALID | **VALID** | yes | matched:3407 (complete 12480/12480) / float32 | SAME-BYTES-CLASS (0.0000) | 60 | 4.024 | 396.9 | 27.481 | 1109.5 | 2.0554→0.8179 | 1.9478→0.7579 | 0.0022 | patched 48 / kcalls 768 | 642514944 | 4-bit experts (e4b NF4) + NF4 attention |  |
| e4b | fused_attn4_m_om0_d2 | **NOT_RUN** | — | **NOT_RUN** | yes | — / — | — | — | — | — | — | — | —→— | —→— | — | patched — / kcalls — | — | — | no receipt and no attempt line |
- prologue `e4b/fused_attn4_m_om0` **111.6 s** before step 1 (34% of the arm): c1_before 58.7, load_weights 31.5, c1_after 28.1, attn4 5.9; unattributed 2.236; budget 1260.0
- prologue `e4b/fused_attn4_m_om1` **112.9 s** before step 1 (34% of the arm): c1_before 63.5, c1_after 30.5, load_weights 27.3, attn4 6.0; unattributed 2.154; budget 1260.0
- prologue `unsloth/ckpt_unsloth_m_om0` **108.9 s** before step 1 (17% of the arm): c1_before 50.2, load_weights 31.8, c1_after 28.3, eval0 9.4; unattributed 10.042; budget 1890.0
- prologue `unsloth/ckpt_unsloth_m_om1` **120.4 s** before step 1 (18% of the arm): c1_before 58.0, load_weights 32.5, c1_after 29.0, eval0 10.3; unattributed 11.492; budget 1890.0
- prologue `unsloth/ckpt_unsloth_m_om1_d2` **123.4 s** before step 1 (18% of the arm): c1_before 58.3, load_weights 35.5, c1_after 30.3, eval0 9.9; unattributed 10.784; budget 1610.7
- prologue `unsloth/ckpt_unsloth_m_om0_d2` **123.9 s** before step 1 (18% of the arm): c1_before 57.1, load_weights 37.2, c1_after 28.4, eval0 10.3; unattributed 10.581; budget 856.4
- prologue `e4b/fused_attn4_m_om1_d2` **116.8 s** before step 1 (32% of the arm): c1_before 66.5, c1_after 35.1, load_weights 28.3, attn4 6.2; unattributed 2.024; budget 346.5
- draws (R1): `e4b/fused_attn4_m_om0` UNMEASURED (second draw fused_attn4_m_om0_d2 is NOT_RUN: stability unmeasured, position not quoted); `e4b/fused_attn4_m_om1` UNSTABLE (3.513/4.024 s, |Δ|/mean 13.6% vs 5%); `unsloth/ckpt_unsloth_m_om0` UNSTABLE (8.416/8.868 s, |Δ|/mean 5.2% vs 5%); `unsloth/ckpt_unsloth_m_om1` STABLE (8.866/8.437 s, |Δ|/mean 5.0% vs 5%)
- e4b internal parity: NO-ARM
- **NO MATCHED POSITION QUOTED (unsloth)** — not quoted: e4b second draw fused_attn4_m_om0_d2 is NOT_RUN: stability unmeasured, position not quoted / unsloth no receipt
- **NO MATCHED POSITION QUOTED (hf)** — not quoted: e4b second draw fused_attn4_m_om0_d2 is NOT_RUN: stability unmeasured, position not quoted / hf no receipt
- **NO MATCHED POSITION QUOTED (axolotl)** — not quoted: e4b second draw fused_attn4_m_om0_d2 is NOT_RUN: stability unmeasured, position not quoted / axolotl no receipt
- equivalence vs `e4b/fused_attn4_m` (I/H; EQUIVALENT iff both ≤ band = max(0.005, 3 × fused-vs-reference |Δ|) = none (no reference arm: COMPARABLE is the ceiling); COMPARABLE ≤ 0.05; draw-noise floor 0.0002): `e4b/fused_attn4_m_om1` **COMPARABLE** (median step |Δ| 0.0011, |Δ held-out at N| 0.0033, step-0 0.0000 SAME-BYTES-CLASS, |Δ loss at step 2| 0.0001, paired rows mean +0.0033 ± 0.0025 SE over 8, favouring arm 3 / anchor 5) — no reference arm ran: the EQUIVALENT band cannot be set, COMPARABLE is the ceiling; `unsloth/ckpt_unsloth_m_om0` **COMPARABLE** (median step |Δ| 0.0019, |Δ held-out at N| 0.0011, step-0 0.0045 SAME-BYTES-CLASS, |Δ loss at step 2| 0.0228, paired rows mean +0.0011 ± 0.0023 SE over 8, favouring arm 3 / anchor 5) — no reference arm ran: the EQUIVALENT band cannot be set, COMPARABLE is the ceiling; `unsloth/ckpt_unsloth_m_om1` **COMPARABLE** (median step |Δ| 0.0015, |Δ held-out at N| 0.0013, step-0 0.0045 SAME-BYTES-CLASS, |Δ loss at step 2| 0.0157, paired rows mean +0.0013 ± 0.0014 SE over 8, favouring arm 3 / anchor 5) — no reference arm ran: the EQUIVALENT band cannot be set, COMPARABLE is the ceiling; `unsloth/ckpt_unsloth_m_om1_d2` **COMPARABLE** (median step |Δ| 0.0010, |Δ held-out at N| 0.0011, step-0 0.0045 SAME-BYTES-CLASS, |Δ loss at step 2| 0.0006, paired rows mean +0.0011 ± 0.0013 SE over 8, favouring arm 4 / anchor 4) — no reference arm ran: the EQUIVALENT band cannot be set, COMPARABLE is the ceiling; `unsloth/ckpt_unsloth_m_om0_d2` **COMPARABLE** (median step |Δ| 0.0013, |Δ held-out at N| 0.0043, step-0 0.0045 SAME-BYTES-CLASS, |Δ loss at step 2| 0.0059, paired rows mean +0.0044 ± 0.0016 SE over 8, favouring arm 1 / anchor 7) — no reference arm ran: the EQUIVALENT band cannot be set, COMPARABLE is the ceiling; `e4b/fused_attn4_m_om1_d2` **COMPARABLE** (median step |Δ| 0.0012, |Δ held-out at N| 0.0022, step-0 0.0000 SAME-BYTES-CLASS, |Δ loss at step 2| 0.0131, paired rows mean +0.0022 ± 0.0019 SE over 8, favouring arm 2 / anchor 6) — no reference arm ran: the EQUIVALENT band cannot be set, COMPARABLE is the ceiling
- matched_init_sha (B, name-free, canonical slot order): anchor `f7832488926eda91`; `e4b/fused_attn4_m_om0` same; `e4b/fused_attn4_m_om1` same; `unsloth/ckpt_unsloth_m_om0` same; `unsloth/ckpt_unsloth_m_om1` same; `unsloth/ckpt_unsloth_m_om1_d2` same; `unsloth/ckpt_unsloth_m_om0_d2` same; `e4b/fused_attn4_m_om1_d2` same
- frozen base (R5), anchor `e4b/fused_attn4_m`: gate_up nf4/64 sha 8c3b3fd78e89 control detects, down nf4/64 sha 59569d810a1a control detects, q_proj nf4/64+dq sha 5c5e402ce51d control detects
- frozen base `e4b/fused_attn4_m_om1`: gate_up **SAME-BYTES** (nf4/64) control detects, down **SAME-BYTES** (nf4/64) control detects, q_proj **SAME-BYTES** (nf4/64+dq) control detects
- frozen base `unsloth/ckpt_unsloth_m_om0`: gate_up **N-A** (regime differs: anchor nf4/64 vs nf4/64+dq) control detects, down **N-A** (regime differs: anchor nf4/64 vs nf4/64+dq) control detects, q_proj **SAME-BYTES** (nf4/64+dq) control detects
- frozen base `unsloth/ckpt_unsloth_m_om1`: gate_up **N-A** (regime differs: anchor nf4/64 vs nf4/64+dq) control detects, down **N-A** (regime differs: anchor nf4/64 vs nf4/64+dq) control detects, q_proj **SAME-BYTES** (nf4/64+dq) control detects
- frozen base `unsloth/ckpt_unsloth_m_om1_d2`: gate_up **N-A** (regime differs: anchor nf4/64 vs nf4/64+dq) control detects, down **N-A** (regime differs: anchor nf4/64 vs nf4/64+dq) control detects, q_proj **SAME-BYTES** (nf4/64+dq) control detects
- frozen base `unsloth/ckpt_unsloth_m_om0_d2`: gate_up **N-A** (regime differs: anchor nf4/64 vs nf4/64+dq) control detects, down **N-A** (regime differs: anchor nf4/64 vs nf4/64+dq) control detects, q_proj **SAME-BYTES** (nf4/64+dq) control detects
- frozen base `e4b/fused_attn4_m_om1_d2`: gate_up **SAME-BYTES** (nf4/64) control detects, down **SAME-BYTES** (nf4/64) control detects, q_proj **SAME-BYTES** (nf4/64+dq) control detects

## Verdicts
| family | arm | VERDICT | validity | quality | note |
|---|---|---|---|---|---|
| qwen3ompab | e4b/fused_attn4_m_om0 | **VALID** | VALID | 0.0000 |  |
| qwen3ompab | e4b/fused_attn4_m_om1 | **VALID** | VALID | 0.0033 |  |
| qwen3ompab | unsloth/ckpt_unsloth_m_om0 | **VALID** | VALID | 0.0011 |  |
| qwen3ompab | unsloth/ckpt_unsloth_m_om1 | **VALID** | VALID | 0.0013 |  |
| qwen3ompab | unsloth/ckpt_unsloth_m_om1_d2 | **VALID** | VALID | 0.0011 |  |
| qwen3ompab | unsloth/ckpt_unsloth_m_om0_d2 | **VALID** | VALID | 0.0043 |  |
| qwen3ompab | e4b/fused_attn4_m_om1_d2 | **VALID** | VALID | 0.0022 |  |
| qwen3ompab | e4b/fused_attn4_m_om0_d2 | **NOT_RUN** | — | — | no receipt and no attempt line |

## Predictions P104 / P105 / P106 (TC1-PREREG amendment 45: OMP_NUM_THREADS at the physical cores vs the container's allotment; scored mechanically)
| prediction | family | verdict | evidence |
|---|---|---|---|
| P104 | qwen3ompab | **UNTESTED** | e4b: two stable VALID draws a side are registered -- om0 UNMEASURED: second draw fused_attn4_m_om0_d2 is NOT_RUN: stability unmeasured, position not quoted; om1 UNSTABLE: draws 3.513 / 4.024 s/step differ by 13.6% > 5% (UNSTABLE: reported, not quoted) |
| P105 | qwen3ompab | **UNTESTED** | unsloth: two stable VALID draws a side are registered -- om0 UNSTABLE: draws 8.416 / 8.868 s/step differ by 5.2% > 5% (UNSTABLE: reported, not quoted); om1 STABLE: |
| P106 | qwen3ompab | **UNTESTED** | e4b: om0 UNMEASURED: second draw fused_attn4_m_om0_d2 is NOT_RUN: stability unmeasured, position not quoted; om1 UNSTABLE: draws 3.513 / 4.024 s/step differ by 13.6% > 5% (UNSTABLE: reported, not quoted); unsloth: om0 UNSTABLE: draws 8.416 / 8.868 s/step differ by 5.2% > 5% (UNSTABLE: reported, not quoted); om1 STABLE: |

## Load gate (TC1-PREREG amendment 33): 13 draw(s) voided for host load and run again
Each line: the arm, the attempt, the median host load1 over that attempt's run, the gate. A VOID attempt's files are in loadvoid/; the last attempt of each arm stands whatever its load.

- `qwen3ompab/e4b/fused_attn4_m_om0 attempt 0 load1_median 65.22 gate 6.0 status ok over 1`
- `qwen3ompab/e4b/fused_attn4_m_om0 attempt 0 VOID (host load1 median 65.22 > 6.0): re-run 1 of 2`
- `qwen3ompab/e4b/fused_attn4_m_om0 attempt 1 load1_median 4.15 gate 6.0 status ok over 0`
- `qwen3ompab/e4b/fused_attn4_m_om1 attempt 0 load1_median 17.45 gate 6.0 status ok over 1`
- `qwen3ompab/e4b/fused_attn4_m_om1 attempt 0 VOID (host load1 median 17.45 > 6.0): re-run 1 of 2`
- `qwen3ompab/e4b/fused_attn4_m_om1 attempt 1 load1_median 24.75 gate 6.0 status ok over 1`
- `qwen3ompab/e4b/fused_attn4_m_om1 attempt 1 VOID (host load1 median 24.75 > 6.0): re-run 2 of 2`
- `qwen3ompab/e4b/fused_attn4_m_om1 attempt 2 load1_median 32.98 gate 6.0 status ok over 1`
- `qwen3ompab/unsloth/ckpt_unsloth_m_om0 attempt 0 load1_median 28.06 gate 6.0 status ok over 1`
- `qwen3ompab/unsloth/ckpt_unsloth_m_om0 attempt 0 VOID (host load1 median 28.06 > 6.0): re-run 1 of 2`
- `qwen3ompab/unsloth/ckpt_unsloth_m_om0 attempt 1 load1_median 10.25 gate 6.0 status ok over 1`
- `qwen3ompab/unsloth/ckpt_unsloth_m_om0 attempt 1 VOID (host load1 median 10.25 > 6.0): re-run 2 of 2`
- `qwen3ompab/unsloth/ckpt_unsloth_m_om0 attempt 2 load1_median 13.4 gate 6.0 status ok over 1`
- `qwen3ompab/unsloth/ckpt_unsloth_m_om1 attempt 0 load1_median 33.85 gate 6.0 status ok over 1`
- `qwen3ompab/unsloth/ckpt_unsloth_m_om1 attempt 0 VOID (host load1 median 33.85 > 6.0): re-run 1 of 2`
- `qwen3ompab/unsloth/ckpt_unsloth_m_om1 attempt 1 load1_median 74.84 gate 6.0 status ok over 1`
- `qwen3ompab/unsloth/ckpt_unsloth_m_om1 attempt 1 VOID (host load1 median 74.84 > 6.0): re-run 2 of 2`
- `qwen3ompab/unsloth/ckpt_unsloth_m_om1 attempt 2 load1_median 24.62 gate 6.0 status ok over 1`
- `qwen3ompab/unsloth/ckpt_unsloth_m_om1_d2 attempt 0 load1_median 29.99 gate 6.0 status ok over 1`
- `qwen3ompab/unsloth/ckpt_unsloth_m_om1_d2 attempt 0 VOID (host load1 median 29.99 > 6.0): re-run 1 of 2`
- `qwen3ompab/unsloth/ckpt_unsloth_m_om1_d2 attempt 1 load1_median 32.56 gate 6.0 status ok over 1`
- `qwen3ompab/unsloth/ckpt_unsloth_m_om1_d2 attempt 1 VOID (host load1 median 32.56 > 6.0): re-run 2 of 2`
- `qwen3ompab/unsloth/ckpt_unsloth_m_om1_d2 attempt 2 load1_median 47.29 gate 6.0 status ok over 1`
- `qwen3ompab/unsloth/ckpt_unsloth_m_om0_d2 attempt 0 load1_median 36.78 gate 6.0 status ok over 1`
- `qwen3ompab/unsloth/ckpt_unsloth_m_om0_d2 attempt 0 VOID (host load1 median 36.78 > 6.0): re-run 1 of 2`
- `qwen3ompab/unsloth/ckpt_unsloth_m_om0_d2 attempt 1 load1_median 24.51 gate 6.0 status ok over 1`
- `qwen3ompab/unsloth/ckpt_unsloth_m_om0_d2 attempt 1 VOID (host load1 median 24.51 > 6.0): re-run 2 of 2`
- `qwen3ompab/unsloth/ckpt_unsloth_m_om0_d2 attempt 2 load1_median 23.98 gate 6.0 status ok over 1`
- `qwen3ompab/e4b/fused_attn4_m_om1_d2 attempt 0 load1_median 30.5 gate 6.0 status ok over 1`
- `qwen3ompab/e4b/fused_attn4_m_om1_d2 attempt 0 VOID (host load1 median 30.5 > 6.0): re-run 1 of 2`
- `qwen3ompab/e4b/fused_attn4_m_om1_d2 attempt 1 load1_median 25.04 gate 6.0 status ok over 1`
- `qwen3ompab/e4b/fused_attn4_m_om1_d2 attempt 1 VOID (host load1 median 25.04 > 6.0): re-run 2 of 2`
- `qwen3ompab/e4b/fused_attn4_m_om1_d2 attempt 2 load1_median 53.99 gate 6.0 status ok over 1`
