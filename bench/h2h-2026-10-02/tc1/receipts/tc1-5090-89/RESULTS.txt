# TC1 — Qwen3-30B-A3B on one RTX 5090: e4b vs Unsloth vs HF+PEFT vs axolotl at matched work, init and adapter precision (tc1)
Rule (tc1/TC1-PREREG.md): status per attempt in the vocabulary OK / REFUSED / OOM / INSTALL_FAILED / LOAD_FAULT / HARNESS_ERROR / ALARM / NOT_RUN; VALID/VOID per tp4's predicates plus the matched-set predicates (R3: matched init complete, fp32 adapters, step-0 held-out within 0.005 of e4b/reference_attn4_m); VERDICT exactly one of VALID / VOID / QUALITY_FAIL / OOM / UNSUPPORTED / HARNESS_ERROR / ALARM / NOT_RUN (QUALITY_FAIL = held-out |Δ| at N > 0.05 vs e4b/fused_attn4_m); positions = s/step ratios other/e4b from the medians over both draws (stability |d1-d2|/mean ≤ 5 % e4b / 10 % others, else UNSTABLE), quoted only when both arms are VALID; equivalence vs e4b/reference_attn4_m EQUIVALENT ≤ 0.02 / COMPARABLE ≤ 0.05 / else DIVERGENT; frozen base SAME-BYTES / DIFFERENT / N-A per slot; predictions P1–P10 (+ P1b) scored HELD / FALSIFIED / UNTESTED. VOID never enters a ratio or an equivalence reading. Nothing is licensed; no cross-box number is divided into these.
`versions.txt`
```
e4b 0.48.0 @054cb8c8881cf5fa75cd41961f3c55b63b4dcf33 (GitHub main)
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
e4b(t212) 0.48.0 @054cb8c8881cf5fa75cd41961f3c55b63b4dcf33
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
 "run_id": "tc1-5090-89",
 "instance_id": "54325899",
 "gpu": "NVIDIA GeForce RTX 5090",
 "driver": "595.84",
 "cpu": "AMD EPYC 7B13 64-Core Processor",
 "nproc": 256,
 "mem_total_kb": "2101193408",
 "cgroup_memory_max": "730283900928",
 "disk_root": "overlay         320G   77M  320G   1% /",
 "hostname": "011252bddf05",
 "registered_gpu_class": "5090",
 "prereg": "tc1/TC1-PREREG.md",
 "local_box": null,
 "local_snapshot": null
}
```

### Qwen3-30B-A3B (amendment 41: e4b's chunked LM loss off vs on at the field recipe, venv-unsloth) (`qwen3chunkab`, registered n_layers 48, attention census 192)
- model `Qwen/Qwen3-30B-A3B` @ `ad44e777bcd1`; tokens sha `bfc742f67e37`; N=60; fixture template alpaca seq 2048 micro-batch 2 × accum 4 lr 0.0002 r 16 α 16 optimizer adamw_8bit(lr=0.0002, weight_decay=0.001) schedule=linear warmup_steps=5 autocast False; e4b trainable 642514944; box_class RTX 5090 gpu NVIDIA GeForce RTX 5090
| framework | arm | status | validity | **VERDICT** | matched | init / dtype | step-0 class | N | s/step med(11+) | tok/s | peak GB | J/step | train first→last | held-out 0→final | quality Δ | engagement | trainable | regime | reason / why |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| e4b | fused_attn4_shipped_ce0 | **OK** | VALID | **VALID** | native | native / bfloat16,float32 | — | 60 | 2.918 | 535.0 | 24.673 | 850.3 | 2.0554→0.8170 | 1.9478→0.7582 | 0.0012 | patched 48 / kcalls 768 | 642514944 | 4-bit experts (e4b NF4) + NF4 attention |  |
| e4b | fused_attn4_shipped_ce1 | **OK** | VALID | **VALID** | native | native / bfloat16,float32 | — | 60 | 2.967 | 528.7 | 23.508 | 848.5 | 2.0554→0.8186 | 1.9478→0.7548 | -0.0022 | patched 48 / kcalls 768 | 642514944 | 4-bit experts (e4b NF4) + NF4 attention |  |
| e4b | fused_attn4_m_ce0 | **OK** | VALID | **VALID** | yes | matched:3407 (complete 12480/12480) / float32 | SAME-BYTES-CLASS (0.0000) | 60 | 3.688 | 420.6 | 27.498 | 1080.7 | 2.0554→0.8181 | 1.9478→0.7570 | 0.0000 | patched 48 / kcalls 768 | 642514944 | 4-bit experts (e4b NF4) + NF4 attention |  |
| e4b | fused_attn4_m_ce1 | **OK** | VALID | **VALID** | yes | matched:3407 (complete 12480/12480) / float32 | SAME-BYTES-CLASS (0.0000) | 60 | 3.896 | 406.6 | 27.145 | 1083.4 | 2.0554→0.8167 | 1.9478→0.7569 | -0.0002 | patched 48 / kcalls 768 | 642514944 | 4-bit experts (e4b NF4) + NF4 attention |  |
| e4b | fused_attn4_m_ce1_d2 | **OK** | VALID | **VALID** | yes | matched:3407 (complete 12480/12480) / float32 | SAME-BYTES-CLASS (0.0000) | 60 | 3.692 | 431.4 | 27.145 | 1054.9 | 2.0554→0.8187 | 1.9478→0.7578 | 0.0008 | patched 48 / kcalls 768 | 642514944 | 4-bit experts (e4b NF4) + NF4 attention |  |
| e4b | fused_attn4_m_ce0_d2 | **OK** | VALID | **VALID** | yes | matched:3407 (complete 12480/12480) / float32 | SAME-BYTES-CLASS (0.0000) | 60 | 3.717 | 426.3 | 27.498 | 1044.2 | 2.0554→0.8201 | 1.9478→0.7591 | 0.0021 | patched 48 / kcalls 768 | 642514944 | 4-bit experts (e4b NF4) + NF4 attention |  |
| e4b | fused_attn4_shipped_ce1_d2 | **OK** | VALID | **VALID** | native | native / bfloat16,float32 | — | 60 | 3.113 | 505.7 | 23.506 | 822.1 | 2.0554→0.8174 | 1.9478→0.7569 | -0.0001 | patched 48 / kcalls 768 | 642514944 | 4-bit experts (e4b NF4) + NF4 attention |  |
| e4b | fused_attn4_shipped_ce0_d2 | **OK** | VALID | **VALID** | native | native / bfloat16,float32 | — | 60 | 2.878 | 544.9 | 24.673 | 827.5 | 2.0554→0.8177 | 1.9478→0.7565 | -0.0005 | patched 48 / kcalls 768 | 642514944 | 4-bit experts (e4b NF4) + NF4 attention |  |
- prologue `e4b/fused_attn4_shipped_ce0` **103.6 s** before step 1 (36% of the arm): c1_before 56.3, c1_after 27.1, load_weights 24.7, trainable_sha 5.9; unattributed 2.054; budget 1260.0
- prologue `e4b/fused_attn4_shipped_ce1` **97.4 s** before step 1 (34% of the arm): c1_before 55.1, c1_after 27.0, load_weights 19.9, trainable_sha 6.1; unattributed 2.056; budget 1260.0
- prologue `e4b/fused_attn4_m_ce0` **97.5 s** before step 1 (30% of the arm): c1_before 56.3, c1_after 29.7, load_weights 20.1, attn4 5.8; unattributed 2.111; budget 1260.0
- prologue `e4b/fused_attn4_m_ce1` **100.5 s** before step 1 (29% of the arm): c1_before 60.5, c1_after 32.1, load_weights 19.6, attn4 5.8; unattributed 1.958; budget 1260.0
- prologue `e4b/fused_attn4_m_ce1_d2` **108.5 s** before step 1 (32% of the arm): c1_before 62.4, c1_after 27.8, load_weights 25.3, attn4 5.4; unattributed 2.319; budget 1260.0
- prologue `e4b/fused_attn4_m_ce0_d2` **107.4 s** before step 1 (32% of the arm): c1_before 59.3, c1_after 29.3, load_weights 26.1, attn4 5.8; unattributed 2.386; budget 1260.0
- prologue `e4b/fused_attn4_shipped_ce1_d2` **109.8 s** before step 1 (36% of the arm): c1_before 60.4, c1_after 31.1, load_weights 25.2, trainable_sha 6.7; unattributed 2.028; budget 1260.0
- prologue `e4b/fused_attn4_shipped_ce0_d2` **108.4 s** before step 1 (38% of the arm): c1_before 60.2, c1_after 30.8, load_weights 24.1, attn4 6.9; unattributed 2.132; budget 1192.8
- draws (R1): `e4b/fused_attn4_shipped_ce0` STABLE (2.918/2.878 s, |Δ|/mean 1.4% vs 5%); `e4b/fused_attn4_shipped_ce1` STABLE (2.967/3.113 s, |Δ|/mean 4.8% vs 5%); `e4b/fused_attn4_m_ce0` STABLE (3.688/3.717 s, |Δ|/mean 0.8% vs 5%); `e4b/fused_attn4_m_ce1` UNSTABLE (3.896/3.692 s, |Δ|/mean 5.4% vs 5%)
- e4b internal parity: NO-ARM
- **NO MATCHED POSITION QUOTED (unsloth)** — not quoted: e4b STABLE / unsloth no receipt
- **NO MATCHED POSITION QUOTED (hf)** — not quoted: e4b STABLE / hf no receipt
- **NO MATCHED POSITION QUOTED (axolotl)** — not quoted: e4b STABLE / axolotl no receipt
- equivalence vs `e4b/fused_attn4_m` (I/H; EQUIVALENT iff both ≤ band = max(0.005, 3 × fused-vs-reference |Δ|) = none (no reference arm: COMPARABLE is the ceiling); COMPARABLE ≤ 0.05; draw-noise floor 0.0021): `e4b/fused_attn4_m_ce1` **COMPARABLE** (median step |Δ| 0.0014, |Δ held-out at N| 0.0002, step-0 0.0000 SAME-BYTES-CLASS, |Δ loss at step 2| 0.0147, paired rows mean -0.0002 ± 0.0017 SE over 8, favouring arm 5 / anchor 3) — no reference arm ran: the EQUIVALENT band cannot be set, COMPARABLE is the ceiling; `e4b/fused_attn4_m_ce1_d2` **COMPARABLE** (median step |Δ| 0.0015, |Δ held-out at N| 0.0008, step-0 0.0000 SAME-BYTES-CLASS, |Δ loss at step 2| 0.0170, paired rows mean +0.0008 ± 0.0009 SE over 8, favouring arm 3 / anchor 5) — no reference arm ran: the EQUIVALENT band cannot be set, COMPARABLE is the ceiling; `e4b/fused_attn4_m_ce0_d2` **COMPARABLE** (median step |Δ| 0.0014, |Δ held-out at N| 0.0021, step-0 0.0000 SAME-BYTES-CLASS, |Δ loss at step 2| 0.0104, paired rows mean +0.0021 ± 0.0027 SE over 8, favouring arm 4 / anchor 4) — no reference arm ran: the EQUIVALENT band cannot be set, COMPARABLE is the ceiling
- matched_init_sha (B, name-free, canonical slot order): anchor `f7832488926eda91`; `e4b/fused_attn4_m_ce0` same; `e4b/fused_attn4_m_ce1` same; `e4b/fused_attn4_m_ce1_d2` same; `e4b/fused_attn4_m_ce0_d2` same
- frozen base (R5), anchor `e4b/fused_attn4_m`: gate_up nf4/64 sha 8c3b3fd78e89 control detects, down nf4/64 sha 59569d810a1a control detects, q_proj nf4/64+dq sha 5c5e402ce51d control detects
- frozen base `e4b/fused_attn4_shipped_ce0`: gate_up **SAME-BYTES** (nf4/64) control detects, down **SAME-BYTES** (nf4/64) control detects, q_proj **SAME-BYTES** (nf4/64+dq) control detects
- frozen base `e4b/fused_attn4_shipped_ce1`: gate_up **SAME-BYTES** (nf4/64) control detects, down **SAME-BYTES** (nf4/64) control detects, q_proj **SAME-BYTES** (nf4/64+dq) control detects
- frozen base `e4b/fused_attn4_m_ce1`: gate_up **SAME-BYTES** (nf4/64) control detects, down **SAME-BYTES** (nf4/64) control detects, q_proj **SAME-BYTES** (nf4/64+dq) control detects
- frozen base `e4b/fused_attn4_m_ce1_d2`: gate_up **SAME-BYTES** (nf4/64) control detects, down **SAME-BYTES** (nf4/64) control detects, q_proj **SAME-BYTES** (nf4/64+dq) control detects
- frozen base `e4b/fused_attn4_m_ce0_d2`: gate_up **SAME-BYTES** (nf4/64) control detects, down **SAME-BYTES** (nf4/64) control detects, q_proj **SAME-BYTES** (nf4/64+dq) control detects
- frozen base `e4b/fused_attn4_shipped_ce1_d2`: gate_up **SAME-BYTES** (nf4/64) control detects, down **SAME-BYTES** (nf4/64) control detects, q_proj **SAME-BYTES** (nf4/64+dq) control detects
- frozen base `e4b/fused_attn4_shipped_ce0_d2`: gate_up **SAME-BYTES** (nf4/64) control detects, down **SAME-BYTES** (nf4/64) control detects, q_proj **SAME-BYTES** (nf4/64+dq) control detects

## Verdicts
| family | arm | VERDICT | validity | quality | note |
|---|---|---|---|---|---|
| qwen3chunkab | e4b/fused_attn4_shipped_ce0 | **VALID** | VALID | 0.0012 |  |
| qwen3chunkab | e4b/fused_attn4_shipped_ce1 | **VALID** | VALID | -0.0022 |  |
| qwen3chunkab | e4b/fused_attn4_m_ce0 | **VALID** | VALID | 0.0000 |  |
| qwen3chunkab | e4b/fused_attn4_m_ce1 | **VALID** | VALID | -0.0002 |  |
| qwen3chunkab | e4b/fused_attn4_m_ce1_d2 | **VALID** | VALID | 0.0008 |  |
| qwen3chunkab | e4b/fused_attn4_m_ce0_d2 | **VALID** | VALID | 0.0021 |  |
| qwen3chunkab | e4b/fused_attn4_shipped_ce1_d2 | **VALID** | VALID | -0.0001 |  |
| qwen3chunkab | e4b/fused_attn4_shipped_ce0_d2 | **VALID** | VALID | -0.0005 |  |

## Predictions P90 / P91 / P92 / P93 (TC1-PREREG amendment 41: e4b's chunked LM loss off vs on at the field recipe; scored mechanically)
| prediction | family | verdict | evidence |
|---|---|---|---|
| P92 | qwen3chunkab | **UNTESTED** | matched: two stable VALID draws a side are registered -- ce0 STABLE:; ce1 UNSTABLE: draws 3.896 / 3.692 s/step differ by 5.4% > 5% (UNSTABLE: reported, not quoted) |
| P90 | qwen3chunkab | **UNTESTED** | matched: two stable VALID draws a side are registered -- ce0 STABLE:; ce1 UNSTABLE: draws 3.896 / 3.692 s/step differ by 5.4% > 5% (UNSTABLE: reported, not quoted) |
| P91 | qwen3chunkab | **FALSIFIED** | shipped: ce1 / ce0 1.049 [1.017, 1.082 over 4 cross-draw ratios] vs [0.0, 1.01]; s/step ce0 2.918 / 2.878 (within 1.4%), ce1 2.967 / 3.113 (within 4.8%); peak ce0 24.67 / ce1 23.51 GB |
| P93 | qwen3chunkab | **UNTESTED** | matched: ce0 STABLE:; ce1 UNSTABLE: draws 3.896 / 3.692 s/step differ by 5.4% > 5% (UNSTABLE: reported, not quoted); shipped: mean held-out ce1 - ce0 -0.0015 (|.| <= 0.005); held-out at N ce0 [0.7582, 0.7565] ce1 [0.7548, 0.7569] |

## Load gate (TC1-PREREG amendment 33): 16 draw(s) voided for host load and run again
Each line: the arm, the attempt, the median host load1 over that attempt's run, the gate. A VOID attempt's files are in loadvoid/; the last attempt of each arm stands whatever its load.

- `qwen3chunkab/e4b/fused_attn4_shipped_ce0 attempt 0 load1_median 7.77 gate 6.0 status ok over 1`
- `qwen3chunkab/e4b/fused_attn4_shipped_ce0 attempt 0 VOID (host load1 median 7.77 > 6.0): re-run 1 of 2`
- `qwen3chunkab/e4b/fused_attn4_shipped_ce0 attempt 1 load1_median 10.04 gate 6.0 status ok over 1`
- `qwen3chunkab/e4b/fused_attn4_shipped_ce0 attempt 1 VOID (host load1 median 10.04 > 6.0): re-run 2 of 2`
- `qwen3chunkab/e4b/fused_attn4_shipped_ce0 attempt 2 load1_median 8.31 gate 6.0 status ok over 1`
- `qwen3chunkab/e4b/fused_attn4_shipped_ce1 attempt 0 load1_median 7.23 gate 6.0 status ok over 1`
- `qwen3chunkab/e4b/fused_attn4_shipped_ce1 attempt 0 VOID (host load1 median 7.23 > 6.0): re-run 1 of 2`
- `qwen3chunkab/e4b/fused_attn4_shipped_ce1 attempt 1 load1_median 12.9 gate 6.0 status ok over 1`
- `qwen3chunkab/e4b/fused_attn4_shipped_ce1 attempt 1 VOID (host load1 median 12.9 > 6.0): re-run 2 of 2`
- `qwen3chunkab/e4b/fused_attn4_shipped_ce1 attempt 2 load1_median 6.99 gate 6.0 status ok over 1`
- `qwen3chunkab/e4b/fused_attn4_m_ce0 attempt 0 load1_median 32.91 gate 6.0 status ok over 1`
- `qwen3chunkab/e4b/fused_attn4_m_ce0 attempt 0 VOID (host load1 median 32.91 > 6.0): re-run 1 of 2`
- `qwen3chunkab/e4b/fused_attn4_m_ce0 attempt 1 load1_median 31.09 gate 6.0 status ok over 1`
- `qwen3chunkab/e4b/fused_attn4_m_ce0 attempt 1 VOID (host load1 median 31.09 > 6.0): re-run 2 of 2`
- `qwen3chunkab/e4b/fused_attn4_m_ce0 attempt 2 load1_median 12.14 gate 6.0 status ok over 1`
- `qwen3chunkab/e4b/fused_attn4_m_ce1 attempt 0 load1_median 10.03 gate 6.0 status ok over 1`
- `qwen3chunkab/e4b/fused_attn4_m_ce1 attempt 0 VOID (host load1 median 10.03 > 6.0): re-run 1 of 2`
- `qwen3chunkab/e4b/fused_attn4_m_ce1 attempt 1 load1_median 28.97 gate 6.0 status ok over 1`
- `qwen3chunkab/e4b/fused_attn4_m_ce1 attempt 1 VOID (host load1 median 28.97 > 6.0): re-run 2 of 2`
- `qwen3chunkab/e4b/fused_attn4_m_ce1 attempt 2 load1_median 14.86 gate 6.0 status ok over 1`
- `qwen3chunkab/e4b/fused_attn4_m_ce1_d2 attempt 0 load1_median 25.04 gate 6.0 status ok over 1`
- `qwen3chunkab/e4b/fused_attn4_m_ce1_d2 attempt 0 VOID (host load1 median 25.04 > 6.0): re-run 1 of 2`
- `qwen3chunkab/e4b/fused_attn4_m_ce1_d2 attempt 1 load1_median 25.45 gate 6.0 status ok over 1`
- `qwen3chunkab/e4b/fused_attn4_m_ce1_d2 attempt 1 VOID (host load1 median 25.45 > 6.0): re-run 2 of 2`
- `qwen3chunkab/e4b/fused_attn4_m_ce1_d2 attempt 2 load1_median 17.87 gate 6.0 status ok over 1`
- `qwen3chunkab/e4b/fused_attn4_m_ce0_d2 attempt 0 load1_median 13.08 gate 6.0 status ok over 1`
- `qwen3chunkab/e4b/fused_attn4_m_ce0_d2 attempt 0 VOID (host load1 median 13.08 > 6.0): re-run 1 of 2`
- `qwen3chunkab/e4b/fused_attn4_m_ce0_d2 attempt 1 load1_median 16.27 gate 6.0 status ok over 1`
- `qwen3chunkab/e4b/fused_attn4_m_ce0_d2 attempt 1 VOID (host load1 median 16.27 > 6.0): re-run 2 of 2`
- `qwen3chunkab/e4b/fused_attn4_m_ce0_d2 attempt 2 load1_median 22.51 gate 6.0 status ok over 1`
- `qwen3chunkab/e4b/fused_attn4_shipped_ce1_d2 attempt 0 load1_median 18.37 gate 6.0 status ok over 1`
- `qwen3chunkab/e4b/fused_attn4_shipped_ce1_d2 attempt 0 VOID (host load1 median 18.37 > 6.0): re-run 1 of 2`
- `qwen3chunkab/e4b/fused_attn4_shipped_ce1_d2 attempt 1 load1_median 15.64 gate 6.0 status ok over 1`
- `qwen3chunkab/e4b/fused_attn4_shipped_ce1_d2 attempt 1 VOID (host load1 median 15.64 > 6.0): re-run 2 of 2`
- `qwen3chunkab/e4b/fused_attn4_shipped_ce1_d2 attempt 2 load1_median 21.28 gate 6.0 status ok over 1`
- `qwen3chunkab/e4b/fused_attn4_shipped_ce0_d2 attempt 0 load1_median 17.38 gate 6.0 status ok over 1`
- `qwen3chunkab/e4b/fused_attn4_shipped_ce0_d2 attempt 0 VOID (host load1 median 17.38 > 6.0): re-run 1 of 2`
- `qwen3chunkab/e4b/fused_attn4_shipped_ce0_d2 attempt 1 load1_median 7.19 gate 6.0 status ok over 1`
- `qwen3chunkab/e4b/fused_attn4_shipped_ce0_d2 attempt 1 VOID (host load1 median 7.19 > 6.0): re-run 2 of 2`
- `qwen3chunkab/e4b/fused_attn4_shipped_ce0_d2 attempt 2 load1_median 8.75 gate 6.0 status ok over 1`
