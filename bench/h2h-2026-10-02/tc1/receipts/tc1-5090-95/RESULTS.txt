# TC1 — Qwen3-30B-A3B on one RTX 5090: e4b vs Unsloth vs HF+PEFT vs axolotl at matched work, init and adapter precision (tc1)
Rule (tc1/TC1-PREREG.md): status per attempt in the vocabulary OK / REFUSED / OOM / INSTALL_FAILED / LOAD_FAULT / HARNESS_ERROR / ALARM / NOT_RUN; VALID/VOID per tp4's predicates plus the matched-set predicates (R3: matched init complete, fp32 adapters, step-0 held-out within 0.005 of e4b/reference_attn4_m); VERDICT exactly one of VALID / VOID / QUALITY_FAIL / OOM / UNSUPPORTED / HARNESS_ERROR / ALARM / NOT_RUN (QUALITY_FAIL = held-out |Δ| at N > 0.05 vs e4b/fused_attn4_m); positions = s/step ratios other/e4b from the medians over both draws (stability |d1-d2|/mean ≤ 5 % e4b / 10 % others, else UNSTABLE), quoted only when both arms are VALID; equivalence vs e4b/reference_attn4_m EQUIVALENT ≤ 0.02 / COMPARABLE ≤ 0.05 / else DIVERGENT; frozen base SAME-BYTES / DIFFERENT / N-A per slot; predictions P1–P10 (+ P1b) scored HELD / FALSIFIED / UNTESTED. VOID never enters a ratio or an equivalence reading. Nothing is licensed; no cross-box number is divided into these.
`versions.txt`
```
e4b 0.48.0 @da897e3fdb4cc29f81d733d524e62f0232d82091 (GitHub main)
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
e4b(t212) 0.48.0 @da897e3fdb4cc29f81d733d524e62f0232d82091
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
 "run_id": "tc1-5090-95",
 "instance_id": "54341612",
 "gpu": "NVIDIA GeForce RTX 5090",
 "driver": "595.84",
 "cpu": "AMD EPYC 7B13 64-Core Processor",
 "nproc": 256,
 "mem_total_kb": "2101193408",
 "cgroup_memory_max": "730283900928",
 "disk_root": "overlay         320G   77M  320G   1% /",
 "hostname": "ed1887455550",
 "registered_gpu_class": "5090",
 "prereg": "tc1/TC1-PREREG.md",
 "local_box": null,
 "local_snapshot": null
}
```

### Qwen3-30B-A3B (amendment 43: the packed 4,096-token regime on one stack, e4b with its chunked LM loss, the LoRA loop a recorded route) (`qwen3samestack4kce2`, registered n_layers 48, attention census 192)
- model `Qwen/Qwen3-30B-A3B` @ `ad44e777bcd1`; tokens sha `d2a501eba57d`; N=40; fixture template alpaca seq 4096 (packed rows, amendment 39) micro-batch 1 × accum 4 lr 0.0002 r 16 α 16 optimizer adamw_8bit(lr=0.0002, weight_decay=0.001) schedule=linear warmup_steps=5 autocast False; e4b trainable 642514944; box_class RTX 5090 gpu NVIDIA GeForce RTX 5090
| framework | arm | status | validity | **VERDICT** | matched | init / dtype | step-0 class | N | s/step med(11+) | tok/s | peak GB | J/step | train first→last | held-out 0→final | quality Δ | engagement | trainable | regime | reason / why |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| e4b | fused_attn4_m | **OK** | VALID | **VALID** | yes | matched:3407 (complete 12480/12480) / float32 | SAME-BYTES-CLASS (0.0000) | 40 | 11.188 | 1458.9 | 32.471 | 5697.9 | 1.2644→0.9131 | 1.2893→0.9548 | 0.0000 | patched 48 / kcalls 768 | 642514944 | 4-bit experts (e4b NF4) + NF4 attention |  |
| unsloth | ckpt_unsloth_m | **OK** | VALID | **VALID** | yes | matched:3407 (complete 12480/12480) / float32 | SAME-BYTES-CLASS (0.0010) | 40 | 14.266 | 1128.1 | 24.864 | 4636.6 | 1.2601→0.9134 | 1.2903→0.9544 | -0.0004 | stacks 96 / fwd 384 / u8 48 | 642514944 | 4-bit expert stacks + bnb-4bit attention (Params4bit stacks 96, Linear4bit 384) |  |
| e4b | reference_attn4_m | **NOT_RUN** | — | **NOT_RUN** | yes | — / — | — | 40 | — | — | — | — | —→— | —→— | — | patched — / kcalls — | — | — | skipped by TC1_SKIP |
| e4b | fused_attn4_m_t28 | **OK** | VALID | **VALID** | yes | matched:3407 (complete 12480/12480) / float32 | SAME-BYTES-CLASS (0.0021) | 40 | 12.236 | 1333.2 | 32.453 | 5898.2 | 1.2594→0.9127 | 1.2914→0.9542 | -0.0006 | patched 48 / kcalls 768 | 642514944 | 4-bit experts (e4b NF4) + NF4 attention |  |
| e4b | fused_attn4_m_t28_d2 | **OK** | VALID | **VALID** | yes | matched:3407 (complete 12480/12480) / float32 | SAME-BYTES-CLASS (0.0021) | 40 | 12.257 | 1332.5 | 32.459 | 5877.4 | 1.2594→0.9133 | 1.2914→0.9542 | -0.0006 | patched 48 / kcalls 768 | 642514944 | 4-bit experts (e4b NF4) + NF4 attention |  |
| unsloth | ckpt_unsloth_m_d2 | **OK** | VALID | **VALID** | yes | matched:3407 (complete 12480/12480) / float32 | SAME-BYTES-CLASS (0.0010) | 40 | 14.366 | 1103.4 | 24.864 | 4723.5 | 1.2601→0.9129 | 1.2903→0.9548 | 0.0001 | stacks 96 / fwd 384 / u8 48 | 642514944 | 4-bit expert stacks + bnb-4bit attention (Params4bit stacks 96, Linear4bit 384) |  |
| e4b | fused_attn4_m_d2 | **OK** | VALID | **VALID** | yes | matched:3407 (complete 12480/12480) / float32 | SAME-BYTES-CLASS (0.0000) | 40 | 11.220 | 1444.4 | 32.530 | 5649.1 | 1.2644→0.9135 | 1.2893→0.9542 | -0.0005 | patched 48 / kcalls 768 | 642514944 | 4-bit experts (e4b NF4) + NF4 attention |  |
- prologue `e4b/fused_attn4_m` **110.8 s** before step 1 (20% of the arm): c1_before 57.9, c1_after 28.4, load_weights 26.2, eval0 7.3; unattributed 2.398; budget 1260.0
- prologue `unsloth/ckpt_unsloth_m` **126.4 s** before step 1 (18% of the arm): c1_before 54.5, load_weights 32.7, c1_after 25.6, eval0 20.9; unattributed 10.082; budget 1890.0
- prologue `e4b/fused_attn4_m_t28` **109.9 s** before step 1 (18% of the arm): c1_before 55.3, c1_after 30.9, load_weights 28.2, eval0 7.7; unattributed 1.272; budget 1260.0
- prologue `e4b/fused_attn4_m_t28_d2` **112.4 s** before step 1 (18% of the arm): c1_before 57.5, load_weights 28.6, c1_after 28.5, eval0 7.7; unattributed 1.286; budget 1260.0
- prologue `unsloth/ckpt_unsloth_m_d2` **114.8 s** before step 1 (16% of the arm): c1_before 49.9, load_weights 33.7, c1_after 24.9, eval0 12.8; unattributed 9.852; budget 1801.4
- prologue `e4b/fused_attn4_m_d2` **111.3 s** before step 1 (19% of the arm): c1_before 57.9, c1_after 33.1, load_weights 27.6, eval0 7.2; unattributed 2.211; budget 1109.5
- draws (R1): `e4b/fused_attn4_m` STABLE (11.188/11.220 s, |Δ|/mean 0.3% vs 5%); `unsloth/ckpt_unsloth_m` STABLE (14.266/14.366 s, |Δ|/mean 0.7% vs 5%); `e4b/fused_attn4_m_t28` STABLE (12.236/12.257 s, |Δ|/mean 0.2% vs 5%)
- e4b internal parity: NO-REF — reference arm missing or not OK
- **MATCHED POSITION (micro-batch 1 × accum 4): s/step ratio unsloth/e4b = 1.278** [1.271, 1.284 over 4 cross-draw ratios] (14.316 vs 11.204 s, medians over 2/2 draws; e4b faster per step); peak VRAM unsloth 24.86 vs e4b 32.50 GB (Δ -7.64); J/step unsloth 4680.0 vs e4b 5673.5 (×0.825); tok/s unsloth 1115.8 vs e4b 1451.7; unsloth regime: 4-bit expert stacks + bnb-4bit attention (Params4bit stacks 96, Linear4bit 384)
- quality reading at N=40 (unsloth): held-out e4b 0.9548 / unsloth 0.9544 (Δ -0.0004) → **COMPARABLE** (|Δ| ≤ 0.05 reads COMPARABLE); step-0 Δ +0.0010
- **NO MATCHED POSITION QUOTED (hf)** — not quoted: e4b STABLE / hf no receipt
- **NO MATCHED POSITION QUOTED (axolotl)** — not quoted: e4b STABLE / axolotl no receipt
- equivalence vs `e4b/fused_attn4_m` (I/H; EQUIVALENT iff both ≤ band = max(0.005, 3 × fused-vs-reference |Δ|) = none (no reference arm: COMPARABLE is the ceiling); COMPARABLE ≤ 0.05; draw-noise floor 0.0005): `unsloth/ckpt_unsloth_m` **COMPARABLE** (median step |Δ| 0.0003, |Δ held-out at N| 0.0004, step-0 0.0010 SAME-BYTES-CLASS, |Δ loss at step 2| 0.0038, paired rows mean -0.0004 ± 0.0003 SE over 8, favouring arm 7 / anchor 1) — no reference arm ran: the EQUIVALENT band cannot be set, COMPARABLE is the ceiling; `e4b/reference_attn4_m` **—** — no OK receipt; `e4b/fused_attn4_m_t28` **COMPARABLE** (median step |Δ| 0.0004, |Δ held-out at N| 0.0006, step-0 0.0021 SAME-BYTES-CLASS, |Δ loss at step 2| 0.0009, paired rows mean -0.0006 ± 0.0002 SE over 8, favouring arm 7 / anchor 1) — no reference arm ran: the EQUIVALENT band cannot be set, COMPARABLE is the ceiling; `e4b/fused_attn4_m_t28_d2` **COMPARABLE** (median step |Δ| 0.0003, |Δ held-out at N| 0.0006, step-0 0.0021 SAME-BYTES-CLASS, |Δ loss at step 2| 0.0005, paired rows mean -0.0006 ± 0.0003 SE over 8, favouring arm 5 / anchor 3) — no reference arm ran: the EQUIVALENT band cannot be set, COMPARABLE is the ceiling; `unsloth/ckpt_unsloth_m_d2` **COMPARABLE** (median step |Δ| 0.0003, |Δ held-out at N| 0.0001, step-0 0.0010 SAME-BYTES-CLASS, |Δ loss at step 2| 0.0052, paired rows mean +0.0001 ± 0.0002 SE over 8, favouring arm 4 / anchor 4) — no reference arm ran: the EQUIVALENT band cannot be set, COMPARABLE is the ceiling; `e4b/fused_attn4_m_d2` **COMPARABLE** (median step |Δ| 0.0003, |Δ held-out at N| 0.0005, step-0 0.0000 SAME-BYTES-CLASS, |Δ loss at step 2| 0.0033, paired rows mean -0.0005 ± 0.0002 SE over 8, favouring arm 6 / anchor 2) — no reference arm ran: the EQUIVALENT band cannot be set, COMPARABLE is the ceiling
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
| qwen3samestack4kce2 | e4b/fused_attn4_m | **VALID** | VALID | 0.0000 |  |
| qwen3samestack4kce2 | unsloth/ckpt_unsloth_m | **VALID** | VALID | -0.0004 |  |
| qwen3samestack4kce2 | e4b/reference_attn4_m | **NOT_RUN** | — | — | skipped by TC1_SKIP |
| qwen3samestack4kce2 | e4b/fused_attn4_m_t28 | **VALID** | VALID | -0.0006 |  |
| qwen3samestack4kce2 | e4b/fused_attn4_m_t28_d2 | **VALID** | VALID | -0.0006 |  |
| qwen3samestack4kce2 | unsloth/ckpt_unsloth_m_d2 | **VALID** | VALID | 0.0001 |  |
| qwen3samestack4kce2 | e4b/fused_attn4_m_d2 | **VALID** | VALID | -0.0005 |  |

## Predictions P96 / P97 / P98 (TC1-PREREG amendment 43: the packed 4,096-token regime on one stack, e4b with its chunked LM loss, the LoRA loop a recorded route; scored mechanically)
| prediction | family | verdict | evidence |
|---|---|---|---|
| P96 | qwen3samestack4kce2 | **HELD** | Unsloth/e4b on one stack 1.278 [1.271, 1.284 over 4 cross-draw ratios] vs [1.25, 1.65]; s/step e4b 11.204 (within 0.3%), Unsloth 14.316 (within 0.7%); peak e4b 32.50 / Unsloth 24.86 GB; held-out at N e4b 0.9548 / Unsloth 0.9544 |
| P97 | qwen3samestack4kce2 | **HELD** | e4b matched arm venv-unsloth / venv-e4b 0.915 [0.913, 0.917] vs [0.84, 0.95]; s/step venv-unsloth 11.188 / 11.220, venv-e4b 12.236 / 12.257; amendment 40's unquotable readings: 1.43 / 0.892 |
| P98 | qwen3samestack4kce2 | **HELD** | all 4 e4b arms that ran completed resident and VALID: fused_attn4_m VALID peak 32.47 GB; fused_attn4_m_t28 VALID peak 32.45 GB; fused_attn4_m_t28_d2 VALID peak 32.46 GB; fused_attn4_m_d2 VALID peak 32.53 GB |
- the per-expert LoRA loop's share of a step's delta calls (max per arm, the recorded route): `fused_attn4_m` 0.026; `fused_attn4_m_t28` 0.029; `fused_attn4_m_t28_d2` 0.029; `fused_attn4_m_d2` 0.026

## Load gate (TC1-PREREG amendment 33): 8 draw(s) voided for host load and run again
Each line: the arm, the attempt, the median host load1 over that attempt's run, the gate. A VOID attempt's files are in loadvoid/; the last attempt of each arm stands whatever its load.

- `qwen3samestack4kce2/e4b/fused_attn4_m attempt 0 load1_median 17.05 gate 6.0 status ok over 1`
- `qwen3samestack4kce2/e4b/fused_attn4_m attempt 0 VOID (host load1 median 17.05 > 6.0): re-run 1 of 2`
- `qwen3samestack4kce2/e4b/fused_attn4_m attempt 1 load1_median 19.35 gate 6.0 status ok over 1`
- `qwen3samestack4kce2/e4b/fused_attn4_m attempt 1 VOID (host load1 median 19.35 > 6.0): re-run 2 of 2`
- `qwen3samestack4kce2/e4b/fused_attn4_m attempt 2 load1_median 7.98 gate 6.0 status ok over 1`
- `qwen3samestack4kce2/unsloth/ckpt_unsloth_m attempt 0 load1_median 4.7 gate 6.0 status ok over 0`
- `qwen3samestack4kce2/e4b/fused_attn4_m_t28 attempt 0 load1_median 11.42 gate 6.0 status ok over 1`
- `qwen3samestack4kce2/e4b/fused_attn4_m_t28 attempt 0 VOID (host load1 median 11.42 > 6.0): re-run 1 of 2`
- `qwen3samestack4kce2/e4b/fused_attn4_m_t28 attempt 1 load1_median 5.7 gate 6.0 status ok over 0`
- `qwen3samestack4kce2/e4b/fused_attn4_m_t28_d2 attempt 0 load1_median 15.26 gate 6.0 status ok over 1`
- `qwen3samestack4kce2/e4b/fused_attn4_m_t28_d2 attempt 0 VOID (host load1 median 15.26 > 6.0): re-run 1 of 2`
- `qwen3samestack4kce2/e4b/fused_attn4_m_t28_d2 attempt 1 load1_median 3.56 gate 6.0 status ok over 0`
- `qwen3samestack4kce2/unsloth/ckpt_unsloth_m_d2 attempt 0 load1_median 14.49 gate 6.0 status ok over 1`
- `qwen3samestack4kce2/unsloth/ckpt_unsloth_m_d2 attempt 0 VOID (host load1 median 14.49 > 6.0): re-run 1 of 2`
- `qwen3samestack4kce2/unsloth/ckpt_unsloth_m_d2 attempt 1 load1_median 7.61 gate 6.0 status ok over 1`
- `qwen3samestack4kce2/unsloth/ckpt_unsloth_m_d2 attempt 1 VOID (host load1 median 7.61 > 6.0): re-run 2 of 2`
- `qwen3samestack4kce2/unsloth/ckpt_unsloth_m_d2 attempt 2 load1_median 18.16 gate 6.0 status ok over 1`
- `qwen3samestack4kce2/e4b/fused_attn4_m_d2 attempt 0 load1_median 9.72 gate 6.0 status ok over 1`
- `qwen3samestack4kce2/e4b/fused_attn4_m_d2 attempt 0 VOID (host load1 median 9.72 > 6.0): re-run 1 of 2`
- `qwen3samestack4kce2/e4b/fused_attn4_m_d2 attempt 1 load1_median 67.18 gate 6.0 status ok over 1`
- `qwen3samestack4kce2/e4b/fused_attn4_m_d2 attempt 1 VOID (host load1 median 67.18 > 6.0): re-run 2 of 2`
- `qwen3samestack4kce2/e4b/fused_attn4_m_d2 attempt 2 load1_median 52.16 gate 6.0 status ok over 1`
