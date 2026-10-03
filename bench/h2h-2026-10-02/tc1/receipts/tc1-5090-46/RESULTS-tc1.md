# TC1 — Qwen3-30B-A3B on one RTX 5090: e4b vs Unsloth vs HF+PEFT vs axolotl at matched work, init and adapter precision (tc1)
Rule (tc1/TC1-PREREG.md): status per attempt in the vocabulary OK / REFUSED / OOM / INSTALL_FAILED / LOAD_FAULT / HARNESS_ERROR / ALARM / NOT_RUN; VALID/VOID per tp4's predicates plus the matched-set predicates (R3: matched init complete, fp32 adapters, step-0 held-out within 0.005 of e4b/reference_attn4_m); VERDICT exactly one of VALID / VOID / QUALITY_FAIL / OOM / UNSUPPORTED / HARNESS_ERROR / ALARM / NOT_RUN (QUALITY_FAIL = held-out |Δ| at N > 0.05 vs e4b/fused_attn4_m); positions = s/step ratios other/e4b from the medians over both draws (stability |d1-d2|/mean ≤ 5 % e4b / 10 % others, else UNSTABLE), quoted only when both arms are VALID; equivalence vs e4b/reference_attn4_m EQUIVALENT ≤ 0.02 / COMPARABLE ≤ 0.05 / else DIVERGENT; frozen base SAME-BYTES / DIFFERENT / N-A per slot; predictions P1–P10 (+ P1b) scored HELD / FALSIFIED / UNTESTED. VOID never enters a ratio or an equivalence reading. Nothing is licensed; no cross-box number is divided into these.
`versions.txt`
```
e4b 0.41.1 @e5859e995db0e6614bf6f4128263cafea922608a (GitHub main)
gnf4 0.34.1 @00929a493f8ca6ef60c950d666eb5fa5cee38df6 (GitHub main)
torch(e4b/hf) 2.8.0+cu128
triton(e4b/hf) 3.4.0
transformers(e4b/hf) 5.18.0
bitsandbytes(e4b/hf) 0.50.2
peft(hf) 0.21.2
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
 "run_id": "tc1-5090-46",
 "instance_id": "54024744",
 "gpu": "NVIDIA GeForce RTX 5090",
 "driver": "580.95.05",
 "cpu": "AMD EPYC 7663 56-Core Processor",
 "nproc": 224,
 "mem_total_kb": "792385364",
 "cgroup_memory_max": "294412877824",
 "disk_root": "overlay         320G   55M  320G   1% /",
 "hostname": "9837d066428a",
 "registered_gpu_class": "5090",
 "prereg": "tc1/TC1-PREREG.md",
 "local_box": null,
 "local_snapshot": null
}
```

### Qwen3-30B-A3B (amendment 8: e4b shipped vs axolotl scattermoe over 200 steps, two draws each) (`qwen3nativebest200`, registered n_layers 48, attention census 192)
- model `Qwen/Qwen3-30B-A3B` @ `ad44e777bcd1`; tokens sha `f07accfe6fcf`; N=20; fixture template alpaca seq 2048 micro-batch 2 × accum 4 lr 0.0002 r 16 α 16 optimizer adamw_8bit(lr=0.0002, weight_decay=0.001) schedule=linear warmup_steps=5 autocast False; e4b trainable 642514944; box_class RTX 5090 gpu NVIDIA GeForce RTX 5090
| framework | arm | status | validity | **VERDICT** | matched | init / dtype | step-0 class | N | s/step med(11+) | tok/s | peak GB | J/step | train first→last | held-out 0→final | quality Δ | engagement | trainable | regime | reason / why |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| e4b | fused_attn4_shipped_200 | **OK** | VOID | **VOID** | native | native / bfloat16,float32 | — | 200 | 3.212 | 481.5 | 25.508 | 827.1 | 2.0614→0.6069 | 1.9514→0.7948 | N-A (anchor VOID) | patched 48 / kcalls 768 | 642514944 | 4-bit experts (e4b NF4) + NF4 attention | step count 200/200 != N=20 |
| axolotl | ckpt_axolotl_best_200 | **OK** | VOID | **VOID** | native | native / float32 | — | 200 | 4.012 | 229.4 | 26.769 | 1474.8 | 2.0621→0.7371 | 1.9546→0.7705 | N-A (anchor VOID) | peft mods 192 / params 96 / fwd 384 | 642514944 | 4-bit expert stacks (axolotl quantize_moe_experts: 96 parametrized stacks, nf4/64+dq) + bnb-4bit attention (Linear4bit 384) | step count 200/200 != N=20 |
| e4b | fused_attn4_shipped_200_d2 | **OK** | VOID | **VOID** | native | native / bfloat16,float32 | — | 200 | 3.211 | 484.6 | 25.508 | 809.2 | 2.0614→0.6052 | 1.9514→0.7937 | N-A (anchor VOID) | patched 48 / kcalls 768 | 642514944 | 4-bit experts (e4b NF4) + NF4 attention | step count 200/200 != N=20 |
| axolotl | ckpt_axolotl_best_200_d2 | **OK** | VOID | **VOID** | native | native / float32 | — | 200 | 4.029 | 226.8 | 26.769 | 1461.1 | 2.0621→0.7365 | 1.9546→0.7704 | N-A (anchor VOID) | peft mods 192 / params 96 / fwd 384 | 642514944 | 4-bit expert stacks (axolotl quantize_moe_experts: 96 parametrized stacks, nf4/64+dq) + bnb-4bit attention (Linear4bit 384) | step count 200/200 != N=20 |
| e4b | fused_attn4_m_200 | **OK** | VOID | **VOID** | yes | matched:3407 (complete 12480/12480) / float32 | SAME-BYTES-CLASS (0.0000) | 200 | 3.769 | 411.4 | 28.287 | 990.9 | 2.0614→0.7377 | 1.9514→0.7673 | N-A (anchor VOID) | patched 48 / kcalls 768 | 642514944 | 4-bit experts (e4b NF4) + NF4 attention | step count 200/200 != N=20 |
- prologue `e4b/fused_attn4_shipped_200` **116.3 s** before step 1 (15% of the arm): c1_before 58.0, c1_after 30.7, load_weights 26.5, eval0 13.0; unattributed 1.319; budget 1680.0
- prologue `axolotl/ckpt_axolotl_best_200` **327.3 s** before step 1 (19% of the arm): eval0 217.4, c1_before 52.4, load_weights 44.6, c1_after 26.6; unattributed 5.657; budget 1890.0
- prologue `e4b/fused_attn4_shipped_200_d2` **119.5 s** before step 1 (15% of the arm): c1_before 63.8, c1_after 32.6, load_weights 31.6, attn4 6.1; unattributed 1.371; budget 1680.0
- prologue `axolotl/ckpt_axolotl_best_200_d2` **328.4 s** before step 1 (19% of the arm): eval0 220.9, c1_before 54.3, load_weights 39.9, c1_after 25.5; unattributed 6.032; budget 1890.0
- prologue `e4b/fused_attn4_m_200` **112.8 s** before step 1 (13% of the arm): c1_before 61.7, c1_after 30.8, load_weights 27.7, attn4 6.0; unattributed 1.285; budget 1680.0
- draws (R1): `e4b/fused_attn4_shipped_200` — (e4b/fused_attn4_shipped_200 is VOID); `axolotl/ckpt_axolotl_best_200` — (axolotl/ckpt_axolotl_best_200 is VOID)
- e4b internal parity: NO-ARM
- **NO MATCHED POSITION QUOTED (unsloth)** — not quoted: e4b e4b/fused_attn4_m_200 is VOID / unsloth no receipt
- **NO MATCHED POSITION QUOTED (hf)** — not quoted: e4b e4b/fused_attn4_m_200 is VOID / hf no receipt
- **NO MATCHED POSITION QUOTED (axolotl)** — not quoted: e4b e4b/fused_attn4_m_200 is VOID / axolotl no receipt
- **NO NATIVE-BEST (reported beside, never instead of, the matched position) QUOTED (axolotl native-best vs e4b shipped)** — not quoted: e4b e4b/fused_attn4_shipped_200 is VOID / axolotl native-best vs e4b shipped axolotl/ckpt_axolotl_best_200 is VOID
- frozen base (R5), anchor `e4b/fused_attn4_m`: gate_up nf4/64 sha 8c3b3fd78e89 control detects, down nf4/64 sha 59569d810a1a control detects, q_proj nf4/64+dq sha 5c5e402ce51d control detects
- frozen base `e4b/fused_attn4_shipped_200`: gate_up **SAME-BYTES** (nf4/64) control detects, down **SAME-BYTES** (nf4/64) control detects, q_proj **SAME-BYTES** (nf4/64+dq) control detects
- frozen base `axolotl/ckpt_axolotl_best_200`: gate_up **N-A** (regime differs: anchor nf4/64 vs nf4/64+dq) control detects, down **N-A** (regime differs: anchor nf4/64 vs nf4/64+dq) control detects, q_proj **N-A** (regime differs: anchor nf4/64+dq vs nf4/64) control detects
- frozen base `e4b/fused_attn4_shipped_200_d2`: gate_up **SAME-BYTES** (nf4/64) control detects, down **SAME-BYTES** (nf4/64) control detects, q_proj **SAME-BYTES** (nf4/64+dq) control detects
- frozen base `axolotl/ckpt_axolotl_best_200_d2`: gate_up **N-A** (regime differs: anchor nf4/64 vs nf4/64+dq) control detects, down **N-A** (regime differs: anchor nf4/64 vs nf4/64+dq) control detects, q_proj **N-A** (regime differs: anchor nf4/64+dq vs nf4/64) control detects

## Verdicts
| family | arm | VERDICT | validity | quality | note |
|---|---|---|---|---|---|
| qwen3nativebest200 | e4b/fused_attn4_shipped_200 | **VOID** | VOID | N-A (anchor VOID) | step count 200/200 != N=20 |
| qwen3nativebest200 | axolotl/ckpt_axolotl_best_200 | **VOID** | VOID | N-A (anchor VOID) | step count 200/200 != N=20 |
| qwen3nativebest200 | e4b/fused_attn4_shipped_200_d2 | **VOID** | VOID | N-A (anchor VOID) | step count 200/200 != N=20 |
| qwen3nativebest200 | axolotl/ckpt_axolotl_best_200_d2 | **VOID** | VOID | N-A (anchor VOID) | step count 200/200 != N=20 |
| qwen3nativebest200 | e4b/fused_attn4_m_200 | **VOID** | VOID | N-A (anchor VOID) | step count 200/200 != N=20 |

## Prediction P14 (TC1-PREREG amendment 8: e4b shipped vs axolotl scattermoe over steps 101..200, two stable draws a side; scored mechanically)
| prediction | family | verdict | evidence |
|---|---|---|---|
| P14 | qwen3nativebest200 | **UNTESTED** | e4b draws fused_attn4_shipped_200 VOID / fused_attn4_shipped_200_d2 VOID: two VALID draws a side are registered |
