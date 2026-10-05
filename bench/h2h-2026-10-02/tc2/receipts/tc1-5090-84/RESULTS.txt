# TC1 — Qwen3-30B-A3B on one RTX 5090: e4b vs Unsloth vs HF+PEFT vs axolotl at matched work, init and adapter precision (tc1)
Rule (tc1/TC1-PREREG.md): status per attempt in the vocabulary OK / REFUSED / OOM / INSTALL_FAILED / LOAD_FAULT / HARNESS_ERROR / ALARM / NOT_RUN; VALID/VOID per tp4's predicates plus the matched-set predicates (R3: matched init complete, fp32 adapters, step-0 held-out within 0.005 of e4b/reference_attn4_m); VERDICT exactly one of VALID / VOID / QUALITY_FAIL / OOM / UNSUPPORTED / HARNESS_ERROR / ALARM / NOT_RUN (QUALITY_FAIL = held-out |Δ| at N > 0.05 vs e4b/fused_attn4_m); positions = s/step ratios other/e4b from the medians over both draws (stability |d1-d2|/mean ≤ 5 % e4b / 10 % others, else UNSTABLE), quoted only when both arms are VALID; equivalence vs e4b/reference_attn4_m EQUIVALENT ≤ 0.02 / COMPARABLE ≤ 0.05 / else DIVERGENT; frozen base SAME-BYTES / DIFFERENT / N-A per slot; predictions P1–P10 (+ P1b) scored HELD / FALSIFIED / UNTESTED. VOID never enters a ratio or an equivalence reading. Nothing is licensed; no cross-box number is divided into these.
`versions.txt`
```
e4b 0.48.0 @63400e1ab0af233aeb09e9e59017492da8ca3694 (GitHub main)
gnf4 0.41.0 @9622144c734395de4d40332f06dea92135d858de (GitHub main)
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
e4b(t212) 0.48.0 @63400e1ab0af233aeb09e9e59017492da8ca3694
gnf4(t212) 0.41.0 @9622144c734395de4d40332f06dea92135d858de
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
 "run_id": "tc1-5090-84",
 "instance_id": "54277905",
 "gpu": "NVIDIA GeForce RTX 5090",
 "driver": "595.84",
 "cpu": "AMD EPYC 7B13 64-Core Processor",
 "nproc": 256,
 "mem_total_kb": "2101193408",
 "cgroup_memory_max": "730283900928",
 "disk_root": "overlay         320G   77M  320G   1% /",
 "hostname": "bc077f853cc4",
 "registered_gpu_class": "5090",
 "prereg": "tc1/TC1-PREREG.md",
 "local_box": null,
 "local_snapshot": null
}
```

### Mixtral-8x7B-Instruct-v0.1 (TC2 amendment 9: the matched set with e4b and Unsloth on one stack, resident, e4b at default settings) (`mixtralsamestack`, registered n_layers 32, attention census 128)
- model `mistralai/Mixtral-8x7B-Instruct-v0.1` @ `eba92302a286`; tokens sha `4a41b3f4a561`; N=60; fixture template alpaca seq 2048 micro-batch 2 × accum 4 lr 0.0002 r 16 α 16 optimizer adamw_8bit(lr=0.0002, weight_decay=0.001) schedule=linear warmup_steps=5 autocast False; e4b trainable 223346688; box_class RTX 5090 gpu NVIDIA GeForce RTX 5090
| framework | arm | status | validity | **VERDICT** | matched | init / dtype | step-0 class | N | s/step med(11+) | tok/s | peak GB | J/step | train first→last | held-out 0→final | quality Δ | engagement | trainable | regime | reason / why |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| e4b | fused_attn4_m | **OK** | VALID | **VALID** | yes | matched:3407 (complete 640/640) / float32 | SAME-BYTES-CLASS (0.0000) | 60 | 3.233 | 580.2 | 31.253 | 1641.7 | 1.4134→0.7136 | 1.4268→0.6318 | 0.0000 | patched 32 / kcalls 512 | 223346688 | 4-bit experts (e4b NF4) + NF4 attention |  |
| unsloth | ckpt_unsloth_m | **OK** | VALID | **VALID** | yes | matched:3407 (complete 640/640) / float32 | SAME-BYTES-CLASS (0.0023) | 60 | 3.701 | 444.5 | 29.158 | 1528.8 | 1.4108→0.7068 | 1.4291→0.6308 | -0.0010 | stacks 64 / fwd 256 / u8 32 | 223346688 | 4-bit expert stacks + bnb-4bit attention (Params4bit stacks 64, Linear4bit 256) |  |
| e4b | reference_attn4_m | **NOT_RUN** | — | **NOT_RUN** | yes | — / — | — | 60 | — | — | — | — | —→— | —→— | — | patched — / kcalls — | — | — | skipped by TC1_SKIP |
| e4b | fused_attn4_m_t28 | **OK** | VALID | **VALID** | yes | matched:3407 (complete 640/640) / float32 | SAME-BYTES-CLASS (0.0019) | 60 | 3.681 | 478.3 | 31.182 | 1800.8 | 1.4174→0.7112 | 1.4249→0.6308 | -0.0010 | patched 32 / kcalls 512 | 223346688 | 4-bit experts (e4b NF4) + NF4 attention |  |
| e4b | fused_attn4_m_t28_d2 | **OK** | VALID | **VALID** | yes | matched:3407 (complete 640/640) / float32 | SAME-BYTES-CLASS (0.0019) | 60 | 3.636 | 479.7 | 31.197 | 1786.1 | 1.4174→0.7104 | 1.4249→0.6325 | 0.0008 | patched 32 / kcalls 512 | 223346688 | 4-bit experts (e4b NF4) + NF4 attention |  |
| unsloth | ckpt_unsloth_m_d2 | **OK** | VALID | **VALID** | yes | matched:3407 (complete 640/640) / float32 | SAME-BYTES-CLASS (0.0023) | 60 | 3.711 | 477.4 | 29.184 | 1506.2 | 1.4108→0.7057 | 1.4291→0.6318 | 0.0000 | stacks 64 / fwd 256 / u8 32 | 223346688 | 4-bit expert stacks + bnb-4bit attention (Params4bit stacks 64, Linear4bit 256) |  |
| e4b | fused_attn4_m_d2 | **OK** | VALID | **VALID** | yes | matched:3407 (complete 640/640) / float32 | SAME-BYTES-CLASS (0.0000) | 60 | 3.248 | 578.4 | 31.222 | 1620.9 | 1.4134→0.7136 | 1.4268→0.6297 | -0.0021 | patched 32 / kcalls 512 | 223346688 | 4-bit experts (e4b NF4) + NF4 attention |  |
- prologue `e4b/fused_attn4_m` **135.6 s** before step 1 (41% of the arm): c1_before 89.2, c1_after 44.4, load_weights 25.6, attn4 8.1; unattributed 1.998; budget 1260.0
- prologue `unsloth/ckpt_unsloth_m` **148.9 s** before step 1 (37% of the arm): c1_before 78.9, c1_after 39.7, load_weights 35.5, eval0 18.5; unattributed 10.117; budget 1260.0
- prologue `e4b/fused_attn4_m_t28` **135.8 s** before step 1 (36% of the arm): c1_before 88.5, c1_after 44.4, load_weights 25.8, attn4 8.5; unattributed 1.234; budget 1260.0
- prologue `e4b/fused_attn4_m_t28_d2` **135.1 s** before step 1 (36% of the arm): c1_before 89.0, c1_after 44.2, load_weights 26.0, attn4 8.4; unattributed 1.248; budget 1260.0
- prologue `unsloth/ckpt_unsloth_m_d2` **135.2 s** before step 1 (36% of the arm): c1_before 78.9, c1_after 40.7, load_weights 31.1, eval0 9.7; unattributed 9.915; budget 1260.0
- prologue `e4b/fused_attn4_m_d2` **134.8 s** before step 1 (41% of the arm): c1_before 89.4, c1_after 42.6, load_weights 25.4, attn4 8.0; unattributed 1.992; budget 1260.0
- draws (R1): `e4b/fused_attn4_m` STABLE (3.233/3.248 s, |Δ|/mean 0.4% vs 5%); `unsloth/ckpt_unsloth_m` STABLE (3.701/3.711 s, |Δ|/mean 0.3% vs 5%); `e4b/fused_attn4_m_t28` STABLE (3.681/3.636 s, |Δ|/mean 1.2% vs 5%)
- e4b internal parity: NO-REF — reference arm missing or not OK
- **MATCHED POSITION: s/step ratio unsloth/e4b = 1.144** [1.140, 1.148 over 4 cross-draw ratios] (3.706 vs 3.241 s, medians over 2/2 draws; e4b faster per step); peak VRAM unsloth 29.17 vs e4b 31.24 GB (Δ -2.07); J/step unsloth 1517.5 vs e4b 1631.3 (×0.930); tok/s unsloth 460.9 vs e4b 579.3; unsloth regime: 4-bit expert stacks + bnb-4bit attention (Params4bit stacks 64, Linear4bit 256)
- quality reading at N=60 (unsloth): held-out e4b 0.6318 / unsloth 0.6308 (Δ -0.0010) → **COMPARABLE** (|Δ| ≤ 0.05 reads COMPARABLE); step-0 Δ +0.0023
- **NO MATCHED POSITION QUOTED (hf)** — not quoted: e4b STABLE / hf no receipt
- **NO MATCHED POSITION QUOTED (axolotl)** — not quoted: e4b STABLE / axolotl no receipt
- equivalence vs `e4b/fused_attn4_m` (I/H; EQUIVALENT iff both ≤ band = max(0.005, 3 × fused-vs-reference |Δ|) = none (no reference arm: COMPARABLE is the ceiling); COMPARABLE ≤ 0.05; draw-noise floor 0.0021): `unsloth/ckpt_unsloth_m` **COMPARABLE** (median step |Δ| 0.0015, |Δ held-out at N| 0.0010, step-0 0.0023 SAME-BYTES-CLASS, |Δ loss at step 2| 0.0648, paired rows mean -0.0010 ± 0.0016 SE over 8, favouring arm 5 / anchor 3) — no reference arm ran: the EQUIVALENT band cannot be set, COMPARABLE is the ceiling; `e4b/reference_attn4_m` **—** — no OK receipt; `e4b/fused_attn4_m_t28` **COMPARABLE** (median step |Δ| 0.0012, |Δ held-out at N| 0.0010, step-0 0.0019 SAME-BYTES-CLASS, |Δ loss at step 2| 0.0077, paired rows mean -0.0010 ± 0.0020 SE over 8, favouring arm 5 / anchor 3) — no reference arm ran: the EQUIVALENT band cannot be set, COMPARABLE is the ceiling; `e4b/fused_attn4_m_t28_d2` **COMPARABLE** (median step |Δ| 0.0011, |Δ held-out at N| 0.0008, step-0 0.0019 SAME-BYTES-CLASS, |Δ loss at step 2| 0.0026, paired rows mean +0.0008 ± 0.0012 SE over 8, favouring arm 3 / anchor 5) — no reference arm ran: the EQUIVALENT band cannot be set, COMPARABLE is the ceiling; `unsloth/ckpt_unsloth_m_d2` **COMPARABLE** (median step |Δ| 0.0016, |Δ held-out at N| 0.0000, step-0 0.0023 SAME-BYTES-CLASS, |Δ loss at step 2| 0.0801, paired rows mean +0.0000 ± 0.0023 SE over 8, favouring arm 5 / anchor 3) — no reference arm ran: the EQUIVALENT band cannot be set, COMPARABLE is the ceiling; `e4b/fused_attn4_m_d2` **COMPARABLE** (median step |Δ| 0.0007, |Δ held-out at N| 0.0021, step-0 0.0000 SAME-BYTES-CLASS, |Δ loss at step 2| 0.0008, paired rows mean -0.0021 ± 0.0023 SE over 8, favouring arm 5 / anchor 3) — no reference arm ran: the EQUIVALENT band cannot be set, COMPARABLE is the ceiling
- matched_init_sha (B, name-free, canonical slot order): anchor `8d6f46fd1d279996`; `e4b/fused_attn4_m` same; `unsloth/ckpt_unsloth_m` same; `e4b/fused_attn4_m_t28` same; `e4b/fused_attn4_m_t28_d2` same; `unsloth/ckpt_unsloth_m_d2` same; `e4b/fused_attn4_m_d2` same
- frozen base (R5), anchor `e4b/fused_attn4_m`: gate_up nf4/64 sha c56e6fc0fd22 control detects, down nf4/64 sha 7e63cab94f9c control detects, q_proj nf4/64+dq sha d5fbebd64960 control detects
- frozen base `unsloth/ckpt_unsloth_m`: gate_up **N-A** (regime differs: anchor nf4/64 vs nf4/64+dq) control detects, down **N-A** (regime differs: anchor nf4/64 vs nf4/64+dq) control detects, q_proj **SAME-BYTES** (nf4/64+dq) control detects
- frozen base `e4b/fused_attn4_m_t28`: gate_up **SAME-BYTES** (nf4/64) control detects, down **SAME-BYTES** (nf4/64) control detects, q_proj **SAME-BYTES** (nf4/64+dq) control detects
- frozen base `e4b/fused_attn4_m_t28_d2`: gate_up **SAME-BYTES** (nf4/64) control detects, down **SAME-BYTES** (nf4/64) control detects, q_proj **SAME-BYTES** (nf4/64+dq) control detects
- frozen base `unsloth/ckpt_unsloth_m_d2`: gate_up **N-A** (regime differs: anchor nf4/64 vs nf4/64+dq) control detects, down **N-A** (regime differs: anchor nf4/64 vs nf4/64+dq) control detects, q_proj **SAME-BYTES** (nf4/64+dq) control detects
- frozen base `e4b/fused_attn4_m_d2`: gate_up **SAME-BYTES** (nf4/64) control detects, down **SAME-BYTES** (nf4/64) control detects, q_proj **SAME-BYTES** (nf4/64+dq) control detects

## Verdicts
| family | arm | VERDICT | validity | quality | note |
|---|---|---|---|---|---|
| mixtralsamestack | e4b/fused_attn4_m | **VALID** | VALID | 0.0000 |  |
| mixtralsamestack | unsloth/ckpt_unsloth_m | **VALID** | VALID | -0.0010 |  |
| mixtralsamestack | e4b/reference_attn4_m | **NOT_RUN** | — | — | skipped by TC1_SKIP |
| mixtralsamestack | e4b/fused_attn4_m_t28 | **VALID** | VALID | -0.0010 |  |
| mixtralsamestack | e4b/fused_attn4_m_t28_d2 | **VALID** | VALID | 0.0008 |  |
| mixtralsamestack | unsloth/ckpt_unsloth_m_d2 | **VALID** | VALID | 0.0000 |  |
| mixtralsamestack | e4b/fused_attn4_m_d2 | **VALID** | VALID | -0.0021 |  |

## Predictions P29 / P30 / P31 (TC2-PREREG amendment 9: Mixtral's position with both frameworks on one stack; scored mechanically)
| prediction | family | verdict | evidence |
|---|---|---|---|
| P29 | mixtralsamestack | **HELD** | Unsloth/e4b on one stack 1.144 [1.140, 1.148 over 4 cross-draw ratios] vs [0.85, 1.25]; s/step e4b 3.241 (within 0.4%), Unsloth 3.706 (within 0.3%); peak e4b 31.24 / Unsloth 29.17 GB; held-out at N e4b 0.6318 / Unsloth 0.6308 |
| P30 | mixtralsamestack | **HELD** | e4b matched arm venv-unsloth / venv-e4b 0.886 [0.878, 0.893] vs [0.85, 1.02]; s/step venv-unsloth 3.233 / 3.248, venv-e4b 3.681 / 3.636; TC1 amendment 34's whole environment read 0.909 on Qwen3-30B-A3B |
| P31 | mixtralsamestack | **HELD** | 4 e4b receipts; route dense with GNF4_TRAIN_GEMM unset on all |

## Load gate (TC1-PREREG amendment 33): 0 draw(s) voided for host load and run again
Each line: the arm, the attempt, the median host load1 over that attempt's run, the gate. A VOID attempt's files are in loadvoid/; the last attempt of each arm stands whatever its load.

- `mixtralsamestack/e4b/fused_attn4_m attempt 0 load1_median 5.7 gate 6.0 status ok over 0`
- `mixtralsamestack/unsloth/ckpt_unsloth_m attempt 0 load1_median 3.31 gate 6.0 status ok over 0`
- `mixtralsamestack/e4b/fused_attn4_m_t28 attempt 0 load1_median 3.96 gate 6.0 status ok over 0`
- `mixtralsamestack/e4b/fused_attn4_m_t28_d2 attempt 0 load1_median 2.7 gate 6.0 status ok over 0`
- `mixtralsamestack/unsloth/ckpt_unsloth_m_d2 attempt 0 load1_median 4.42 gate 6.0 status ok over 0`
- `mixtralsamestack/e4b/fused_attn4_m_d2 attempt 0 load1_median 4.09 gate 6.0 status ok over 0`
