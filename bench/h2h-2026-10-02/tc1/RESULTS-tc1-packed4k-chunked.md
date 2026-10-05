# TC1 — Qwen3-30B-A3B on one RTX 5090: e4b vs Unsloth vs HF+PEFT vs axolotl at matched work, init and adapter precision (tc1)
Rule (tc1/TC1-PREREG.md): status per attempt in the vocabulary OK / REFUSED / OOM / INSTALL_FAILED / LOAD_FAULT / HARNESS_ERROR / ALARM / NOT_RUN; VALID/VOID per tp4's predicates plus the matched-set predicates (R3: matched init complete, fp32 adapters, step-0 held-out within 0.005 of e4b/reference_attn4_m); VERDICT exactly one of VALID / VOID / QUALITY_FAIL / OOM / UNSUPPORTED / HARNESS_ERROR / ALARM / NOT_RUN (QUALITY_FAIL = held-out |Δ| at N > 0.05 vs e4b/fused_attn4_m); positions = s/step ratios other/e4b from the medians over both draws (stability |d1-d2|/mean ≤ 5 % e4b / 10 % others, else UNSTABLE), quoted only when both arms are VALID; equivalence vs e4b/reference_attn4_m EQUIVALENT ≤ 0.02 / COMPARABLE ≤ 0.05 / else DIVERGENT; frozen base SAME-BYTES / DIFFERENT / N-A per slot; predictions P1–P10 (+ P1b) scored HELD / FALSIFIED / UNTESTED. VOID never enters a ratio or an equivalence reading. Nothing is licensed; no cross-box number is divided into these.
`versions.txt`
```
e4b 0.48.0 @78faf5da706ef6635b451f2fb1f2930a8179b187 (GitHub main)
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
e4b(t212) 0.48.0 @78faf5da706ef6635b451f2fb1f2930a8179b187
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
 "run_id": "tc1-5090-91",
 "instance_id": "54323622",
 "gpu": "NVIDIA GeForce RTX 5090",
 "driver": "595.91.07",
 "cpu": "AMD EPYC 7K62 48-Core Processor",
 "nproc": 96,
 "mem_total_kb": "527994548",
 "cgroup_memory_max": "259519414272",
 "disk_root": "overlay         320G  1.9M  320G   1% /",
 "hostname": "60b5fedb86b2",
 "registered_gpu_class": "5090",
 "prereg": "tc1/TC1-PREREG.md",
 "local_box": null,
 "local_snapshot": null
}
```

### Qwen3-30B-A3B (amendment 40: the packed 4,096-token regime on one stack, e4b with its chunked LM loss) (`qwen3samestack4kce`, registered n_layers 48, attention census 192)
- model `Qwen/Qwen3-30B-A3B` @ `ad44e777bcd1`; tokens sha `d2a501eba57d`; N=40; fixture template alpaca seq 4096 (packed rows, amendment 39) micro-batch 1 × accum 4 lr 0.0002 r 16 α 16 optimizer adamw_8bit(lr=0.0002, weight_decay=0.001) schedule=linear warmup_steps=5 autocast False; e4b trainable 642514944; box_class RTX 5090 gpu NVIDIA GeForce RTX 5090
| framework | arm | status | validity | **VERDICT** | matched | init / dtype | step-0 class | N | s/step med(11+) | tok/s | peak GB | J/step | train first→last | held-out 0→final | quality Δ | engagement | trainable | regime | reason / why |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| e4b | fused_attn4_m | **OK** | VOID | **VOID** | yes | matched:3407 (complete 12480/12480) / float32 | SAME-BYTES-CLASS (0.0000) | 40 | 11.188 | 1438.3 | 32.486 | 5593.3 | 1.2644→0.9129 | 1.2893→0.9537 | N-A (anchor VOID) | patched 48 / kcalls 768 | 642514944 | 4-bit experts (e4b NF4) + NF4 attention | the per-expert LoRA loop ran on step(s) [1, 2, 3, 4, 5, 6]... (lora_loop_share max 0.029) |
| unsloth | ckpt_unsloth_m | **OK** | VALID | **VALID** | yes | matched:3407 (complete 12480/12480) / float32 | SAME-BYTES-CLASS (0.0022) | 40 | 16.027 | 1003.8 | 24.864 | 4553.6 | 1.2587→0.9125 | 1.2871→0.9542 | N-A (anchor VOID) | stacks 96 / fwd 384 / u8 48 | 642514944 | 4-bit expert stacks + bnb-4bit attention (Params4bit stacks 96, Linear4bit 384) |  |
| e4b | reference_attn4_m | **NOT_RUN** | — | **NOT_RUN** | yes | — / — | — | 40 | — | — | — | — | —→— | —→— | — | patched — / kcalls — | — | — | skipped by TC1_SKIP |
| e4b | fused_attn4_m_t28 | **OK** | VOID | **VOID** | yes | matched:3407 (complete 12480/12480) / float32 | SAME-BYTES-CLASS (0.0021) | 40 | 12.562 | 1282.0 | 32.521 | 5772.2 | 1.2594→0.9133 | 1.2914→0.9547 | N-A (anchor VOID) | patched 48 / kcalls 768 | 642514944 | 4-bit experts (e4b NF4) + NF4 attention | the per-expert LoRA loop ran on step(s) [1, 2, 3, 4, 5, 6]... (lora_loop_share max 0.026) |
| e4b | fused_attn4_m_t28_d2 | **OK** | VOID | **VOID** | yes | matched:3407 (complete 12480/12480) / float32 | SAME-BYTES-CLASS (0.0021) | 40 | 12.553 | 1294.0 | 32.439 | 5755.3 | 1.2594→0.9133 | 1.2914→0.9542 | N-A (anchor VOID) | patched 48 / kcalls 768 | 642514944 | 4-bit experts (e4b NF4) + NF4 attention | the per-expert LoRA loop ran on step(s) [1, 2, 3, 4, 5, 6]... (lora_loop_share max 0.029) |
| unsloth | ckpt_unsloth_m_d2 | **OK** | VALID | **VALID** | yes | matched:3407 (complete 12480/12480) / float32 | SAME-BYTES-CLASS (0.0022) | 40 | 16.030 | 1016.2 | 24.864 | 4568.9 | 1.2587→0.9138 | 1.2871→0.9540 | N-A (anchor VOID) | stacks 96 / fwd 384 / u8 48 | 642514944 | 4-bit expert stacks + bnb-4bit attention (Params4bit stacks 96, Linear4bit 384) |  |
| e4b | fused_attn4_m_d2 | **OK** | VOID | **VOID** | yes | matched:3407 (complete 12480/12480) / float32 | SAME-BYTES-CLASS (0.0000) | 40 | 11.210 | 1452.3 | 32.568 | 5613.5 | 1.2644→0.9125 | 1.2893→0.9545 | N-A (anchor VOID) | patched 48 / kcalls 768 | 642514944 | 4-bit experts (e4b NF4) + NF4 attention | the per-expert LoRA loop ran on step(s) [1, 2, 3, 4, 5, 6]... (lora_loop_share max 0.029) |
- prologue `e4b/fused_attn4_m` **127.4 s** before step 1 (22% of the arm): c1_before 71.6, c1_after 36.0, load_weights 20.7, eval0 14.0; unattributed 2.592; budget 1260.0
- prologue `unsloth/ckpt_unsloth_m` **143.1 s** before step 1 (18% of the arm): c1_before 65.5, c1_after 34.2, load_weights 32.0, eval0 24.2; unattributed 12.763; budget 1890.0
- prologue `e4b/fused_attn4_m_t28` **135.8 s** before step 1 (21% of the arm): c1_before 74.2, c1_after 36.0, load_weights 23.3, eval0 17.2; unattributed 1.59; budget 1260.0
- prologue `e4b/fused_attn4_m_t28_d2` **127.6 s** before step 1 (20% of the arm): c1_before 73.7, c1_after 37.3, load_weights 23.2, eval0 7.9; unattributed 1.451; budget 1260.0
- prologue `unsloth/ckpt_unsloth_m_d2` **138.9 s** before step 1 (18% of the arm): c1_before 67.1, c1_after 33.6, load_weights 33.5, eval0 14.6; unattributed 13.079; budget 1890.0
- prologue `e4b/fused_attn4_m_d2` **123.0 s** before step 1 (21% of the arm): c1_before 74.1, c1_after 36.8, load_weights 20.6, attn4 7.4; unattributed 2.597; budget 1260.0
- draws (R1): `e4b/fused_attn4_m` — (e4b/fused_attn4_m is VOID); `unsloth/ckpt_unsloth_m` STABLE (16.027/16.030 s, |Δ|/mean 0.0% vs 5%); `e4b/fused_attn4_m_t28` — (e4b/fused_attn4_m_t28 is VOID)
- e4b internal parity: NO-REF — reference arm missing or not OK
- **NO MATCHED POSITION QUOTED (unsloth)** — not quoted: e4b e4b/fused_attn4_m is VOID / unsloth STABLE
- **NO MATCHED POSITION QUOTED (hf)** — not quoted: e4b e4b/fused_attn4_m is VOID / hf no receipt
- **NO MATCHED POSITION QUOTED (axolotl)** — not quoted: e4b e4b/fused_attn4_m is VOID / axolotl no receipt
- equivalence vs `e4b/fused_attn4_m` (I/H; EQUIVALENT iff both ≤ band = max(0.005, 3 × fused-vs-reference |Δ|) = none (no reference arm: COMPARABLE is the ceiling); COMPARABLE ≤ 0.05; draw-noise floor 0.0002): `unsloth/ckpt_unsloth_m` **N-A** — validity: anchor VOID, arm VALID (VOID never enters an equivalence reading); `e4b/reference_attn4_m` **—** — no OK receipt; `e4b/fused_attn4_m_t28` **N-A** — validity: anchor VOID, arm VOID (VOID never enters an equivalence reading); `e4b/fused_attn4_m_t28_d2` **N-A** — validity: anchor VOID, arm VOID (VOID never enters an equivalence reading); `unsloth/ckpt_unsloth_m_d2` **N-A** — validity: anchor VOID, arm VALID (VOID never enters an equivalence reading); `e4b/fused_attn4_m_d2` **N-A** — validity: anchor VOID, arm VOID (VOID never enters an equivalence reading)
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
| qwen3samestack4kce | e4b/fused_attn4_m | **VOID** | VOID | N-A (anchor VOID) | the per-expert LoRA loop ran on step(s) [1, 2, 3, 4, 5, 6]... (lora_loop_share max 0.029) |
| qwen3samestack4kce | unsloth/ckpt_unsloth_m | **VALID** | VALID | N-A (anchor VOID) |  |
| qwen3samestack4kce | e4b/reference_attn4_m | **NOT_RUN** | — | — | skipped by TC1_SKIP |
| qwen3samestack4kce | e4b/fused_attn4_m_t28 | **VOID** | VOID | N-A (anchor VOID) | the per-expert LoRA loop ran on step(s) [1, 2, 3, 4, 5, 6]... (lora_loop_share max 0.026) |
| qwen3samestack4kce | e4b/fused_attn4_m_t28_d2 | **VOID** | VOID | N-A (anchor VOID) | the per-expert LoRA loop ran on step(s) [1, 2, 3, 4, 5, 6]... (lora_loop_share max 0.029) |
| qwen3samestack4kce | unsloth/ckpt_unsloth_m_d2 | **VALID** | VALID | N-A (anchor VOID) |  |
| qwen3samestack4kce | e4b/fused_attn4_m_d2 | **VOID** | VOID | N-A (anchor VOID) | the per-expert LoRA loop ran on step(s) [1, 2, 3, 4, 5, 6]... (lora_loop_share max 0.029) |

## Predictions P87 / P88 / P89 (TC1-PREREG amendment 40: the packed 4,096-token regime on one stack, e4b with its chunked LM loss; scored mechanically)
| prediction | family | verdict | evidence |
|---|---|---|---|
| P87 | qwen3samestack4kce | **UNTESTED** | two stable VALID draws a side are registered -- not quoted: e4b e4b/fused_attn4_m is VOID / unsloth STABLE |
| P88 | qwen3samestack4kce | **UNTESTED** | two stable VALID draws a side are registered -- venv-unsloth —: e4b/fused_attn4_m is VOID; venv-e4b —: e4b/fused_attn4_m_t28 is VOID |
| P89 | qwen3samestack4kce | **UNTESTED** | not every e4b arm that ran is VALID and none OOMed: fused_attn4_m VOID peak 32.49 GB; fused_attn4_m_t28 VOID peak 32.52 GB; fused_attn4_m_t28_d2 VOID peak 32.44 GB; fused_attn4_m_d2 VOID peak 32.57 GB |

## Load gate (TC1-PREREG amendment 33): 0 draw(s) voided for host load and run again
Each line: the arm, the attempt, the median host load1 over that attempt's run, the gate. A VOID attempt's files are in loadvoid/; the last attempt of each arm stands whatever its load.

- `qwen3samestack4kce/e4b/fused_attn4_m attempt 0 load1_median 2.3 gate 6.0 status ok over 0`
- `qwen3samestack4kce/unsloth/ckpt_unsloth_m attempt 0 load1_median 2.49 gate 6.0 status ok over 0`
- `qwen3samestack4kce/e4b/fused_attn4_m_t28 attempt 0 load1_median 2.04 gate 6.0 status ok over 0`
- `qwen3samestack4kce/e4b/fused_attn4_m_t28_d2 attempt 0 load1_median 2.81 gate 6.0 status ok over 0`
- `qwen3samestack4kce/unsloth/ckpt_unsloth_m_d2 attempt 0 load1_median 3.03 gate 6.0 status ok over 0`
- `qwen3samestack4kce/e4b/fused_attn4_m_d2 attempt 0 load1_median 2.92 gate 6.0 status ok over 0`
