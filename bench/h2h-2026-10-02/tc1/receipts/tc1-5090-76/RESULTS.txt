# TC1 — Qwen3-30B-A3B on one RTX 5090: e4b vs Unsloth vs HF+PEFT vs axolotl at matched work, init and adapter precision (tc1)
Rule (tc1/TC1-PREREG.md): status per attempt in the vocabulary OK / REFUSED / OOM / INSTALL_FAILED / LOAD_FAULT / HARNESS_ERROR / ALARM / NOT_RUN; VALID/VOID per tp4's predicates plus the matched-set predicates (R3: matched init complete, fp32 adapters, step-0 held-out within 0.005 of e4b/reference_attn4_m); VERDICT exactly one of VALID / VOID / QUALITY_FAIL / OOM / UNSUPPORTED / HARNESS_ERROR / ALARM / NOT_RUN (QUALITY_FAIL = held-out |Δ| at N > 0.05 vs e4b/fused_attn4_m); positions = s/step ratios other/e4b from the medians over both draws (stability |d1-d2|/mean ≤ 5 % e4b / 10 % others, else UNSTABLE), quoted only when both arms are VALID; equivalence vs e4b/reference_attn4_m EQUIVALENT ≤ 0.02 / COMPARABLE ≤ 0.05 / else DIVERGENT; frozen base SAME-BYTES / DIFFERENT / N-A per slot; predictions P1–P10 (+ P1b) scored HELD / FALSIFIED / UNTESTED. VOID never enters a ratio or an equivalence reading. Nothing is licensed; no cross-box number is divided into these.
`versions.txt`
```
e4b 0.47.0 @c8925bbdb2ec94ad5d65146440b70d62db664423 (GitHub main)
gnf4 0.38.0 @bb56b42c574d043df9428b85b17066f5bf6c092a (GitHub main)
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
e4b(t212) 0.47.0 @c8925bbdb2ec94ad5d65146440b70d62db664423
gnf4(t212) 0.38.0 @bb56b42c574d043df9428b85b17066f5bf6c092a
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
 "run_id": "tc1-5090-76",
 "instance_id": "54239673",
 "gpu": "NVIDIA GeForce RTX 5090",
 "driver": "595.84",
 "cpu": "AMD EPYC 7B13 64-Core Processor",
 "nproc": 256,
 "mem_total_kb": "2101193408",
 "cgroup_memory_max": "730283900928",
 "disk_root": "overlay         320G   77M  320G   1% /",
 "hostname": "5010842bfccd",
 "registered_gpu_class": "5090",
 "prereg": "tc1/TC1-PREREG.md",
 "local_box": null,
 "local_snapshot": null
}
```

### Qwen3-30B-A3B (amendment 25: the matched set with e4b and Unsloth on one stack, torch 2.12.1+cu130 / transformers 5.5.0) (`qwen3samestack`, registered n_layers 48, attention census 192)
- model `Qwen/Qwen3-30B-A3B` @ `ad44e777bcd1`; tokens sha `bfc742f67e37`; N=60; fixture template alpaca seq 2048 micro-batch 2 × accum 4 lr 0.0002 r 16 α 16 optimizer adamw_8bit(lr=0.0002, weight_decay=0.001) schedule=linear warmup_steps=5 autocast False; e4b trainable 642514944; box_class RTX 5090 gpu NVIDIA GeForce RTX 5090
| framework | arm | status | validity | **VERDICT** | matched | init / dtype | step-0 class | N | s/step med(11+) | tok/s | peak GB | J/step | train first→last | held-out 0→final | quality Δ | engagement | trainable | regime | reason / why |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| e4b | fused_attn4_m | **OK** | VALID | **VALID** | yes | matched:3407 (complete 12480/12480) / float32 | SAME-BYTES-CLASS (0.0000) | 60 | 3.496 | 436.2 | 27.493 | 1062.0 | 2.0554→0.8158 | 1.9478→0.7569 | 0.0000 | patched 48 / kcalls 768 | 642514944 | 4-bit experts (e4b NF4) + NF4 attention |  |
| unsloth | ckpt_unsloth_m | **OK** | VALID | **VALID** | yes | matched:3407 (complete 12480/12480) / float32 | SAME-BYTES-CLASS (0.0080) | 60 | 8.208 | 187.3 | 24.273 | 1436.7 | 2.0705→0.8176 | 1.9558→0.7557 | -0.0012 | stacks 96 / fwd 384 / u8 48 | 642514944 | 4-bit expert stacks + bnb-4bit attention (Params4bit stacks 96, Linear4bit 384) |  |
| e4b | reference_attn4_m | **NOT_RUN** | — | **NOT_RUN** | yes | — / — | — | 60 | — | — | — | — | —→— | —→— | — | patched — / kcalls — | — | — | skipped by TC1_SKIP |
| e4b | fused_attn4_m_t28 | **OK** | VALID | **VALID** | yes | matched:3407 (complete 12480/12480) / float32 | SAME-BYTES-CLASS (0.0038) | 60 | 3.855 | 392.5 | 27.443 | 1130.0 | 2.0614→0.8165 | 1.9441→0.7575 | 0.0006 | patched 48 / kcalls 768 | 642514944 | 4-bit experts (e4b NF4) + NF4 attention |  |
| e4b | fused_attn4_m_t28_d2 | **OK** | VALID | **VALID** | yes | matched:3407 (complete 12480/12480) / float32 | SAME-BYTES-CLASS (0.0038) | 60 | 3.913 | 404.2 | 27.442 | 1108.9 | 2.0614→0.8156 | 1.9441→0.7558 | -0.0011 | patched 48 / kcalls 768 | 642514944 | 4-bit experts (e4b NF4) + NF4 attention |  |
| unsloth | ckpt_unsloth_m_d2 | **OK** | VALID | **VALID** | yes | matched:3407 (complete 12480/12480) / float32 | SAME-BYTES-CLASS (0.0080) | 60 | 8.228 | 190.7 | 24.273 | 1435.0 | 2.0705→0.8188 | 1.9558→0.7535 | -0.0034 | stacks 96 / fwd 384 / u8 48 | 642514944 | 4-bit expert stacks + bnb-4bit attention (Params4bit stacks 96, Linear4bit 384) |  |
| e4b | fused_attn4_m_d2 | **OK** | VALID | **VALID** | yes | matched:3407 (complete 12480/12480) / float32 | SAME-BYTES-CLASS (0.0000) | 60 | 3.492 | 446.1 | 27.493 | 1040.2 | 2.0554→0.8169 | 1.9478→0.7573 | 0.0004 | patched 48 / kcalls 768 | 642514944 | 4-bit experts (e4b NF4) + NF4 attention |  |
- prologue `e4b/fused_attn4_m` **112.2 s** before step 1 (33% of the arm): c1_before 56.6, c1_after 27.7, load_weights 24.7, eval0 12.9; unattributed 2.04; budget 1260.0
- prologue `unsloth/ckpt_unsloth_m` **118.4 s** before step 1 (18% of the arm): c1_before 51.4, load_weights 34.3, c1_after 25.3, eval0 15.5; unattributed 9.75; budget 1260.0
- prologue `e4b/fused_attn4_m_t28` **112.3 s** before step 1 (31% of the arm): c1_before 56.3, load_weights 26.8, c1_after 26.7, eval0 11.1; unattributed 1.262; budget 1260.0
- prologue `e4b/fused_attn4_m_t28_d2` **112.5 s** before step 1 (32% of the arm): c1_before 63.3, c1_after 31.6, load_weights 27.7, attn4 6.1; unattributed 1.374; budget 1260.0
- prologue `unsloth/ckpt_unsloth_m_d2` **108.0 s** before step 1 (17% of the arm): c1_before 49.3, load_weights 31.3, c1_after 25.5, eval0 9.4; unattributed 9.906; budget 1260.0
- prologue `e4b/fused_attn4_m_d2` **101.2 s** before step 1 (32% of the arm): c1_before 56.1, c1_after 27.9, load_weights 24.4, attn4 5.6; unattributed 2.029; budget 1260.0
- draws (R1): `e4b/fused_attn4_m` STABLE (3.496/3.492 s, |Δ|/mean 0.1% vs 5%); `unsloth/ckpt_unsloth_m` STABLE (8.208/8.228 s, |Δ|/mean 0.3% vs 5%); `e4b/fused_attn4_m_t28` STABLE (3.855/3.913 s, |Δ|/mean 1.5% vs 5%)
- e4b internal parity: NO-REF — reference arm missing or not OK
- **MATCHED POSITION: s/step ratio unsloth/e4b = 2.352** [2.348, 2.356 over 4 cross-draw ratios] (8.218 vs 3.494 s, medians over 2/2 draws; e4b faster per step); peak VRAM unsloth 24.27 vs e4b 27.49 GB (Δ -3.22); J/step unsloth 1435.9 vs e4b 1051.1 (×1.366); tok/s unsloth 189.0 vs e4b 441.1; unsloth regime: 4-bit expert stacks + bnb-4bit attention (Params4bit stacks 96, Linear4bit 384)
- quality reading at N=60 (unsloth): held-out e4b 0.7569 / unsloth 0.7557 (Δ -0.0012) → **COMPARABLE** (|Δ| ≤ 0.05 reads COMPARABLE); step-0 Δ +0.0080
- **NO MATCHED POSITION QUOTED (hf)** — not quoted: e4b STABLE / hf no receipt
- **NO MATCHED POSITION QUOTED (axolotl)** — not quoted: e4b STABLE / axolotl no receipt
- equivalence vs `e4b/fused_attn4_m` (I/H; EQUIVALENT iff both ≤ band = max(0.005, 3 × fused-vs-reference |Δ|) = none (no reference arm: COMPARABLE is the ceiling); COMPARABLE ≤ 0.05; draw-noise floor 0.0022): `unsloth/ckpt_unsloth_m` **COMPARABLE** (median step |Δ| 0.0011, |Δ held-out at N| 0.0012, step-0 0.0080 SAME-BYTES-CLASS, |Δ loss at step 2| 0.0011, paired rows mean -0.0012 ± 0.0020 SE over 8, favouring arm 4 / anchor 4) — no reference arm ran: the EQUIVALENT band cannot be set, COMPARABLE is the ceiling; `e4b/reference_attn4_m` **—** — no OK receipt; `e4b/fused_attn4_m_t28` **COMPARABLE** (median step |Δ| 0.0012, |Δ held-out at N| 0.0006, step-0 0.0038 SAME-BYTES-CLASS, |Δ loss at step 2| 0.0196, paired rows mean +0.0006 ± 0.0018 SE over 8, favouring arm 4 / anchor 4) — no reference arm ran: the EQUIVALENT band cannot be set, COMPARABLE is the ceiling; `e4b/fused_attn4_m_t28_d2` **COMPARABLE** (median step |Δ| 0.0011, |Δ held-out at N| 0.0011, step-0 0.0038 SAME-BYTES-CLASS, |Δ loss at step 2| 0.0018, paired rows mean -0.0011 ± 0.0017 SE over 8, favouring arm 3 / anchor 5) — no reference arm ran: the EQUIVALENT band cannot be set, COMPARABLE is the ceiling; `unsloth/ckpt_unsloth_m_d2` **COMPARABLE** (median step |Δ| 0.0010, |Δ held-out at N| 0.0034, step-0 0.0080 SAME-BYTES-CLASS, |Δ loss at step 2| 0.0080, paired rows mean -0.0034 ± 0.0026 SE over 8, favouring arm 5 / anchor 3) — no reference arm ran: the EQUIVALENT band cannot be set, COMPARABLE is the ceiling; `e4b/fused_attn4_m_d2` **COMPARABLE** (median step |Δ| 0.0011, |Δ held-out at N| 0.0004, step-0 0.0000 SAME-BYTES-CLASS, |Δ loss at step 2| 0.0242, paired rows mean +0.0005 ± 0.0028 SE over 8, favouring arm 4 / anchor 4) — no reference arm ran: the EQUIVALENT band cannot be set, COMPARABLE is the ceiling
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
| qwen3samestack | e4b/fused_attn4_m | **VALID** | VALID | 0.0000 |  |
| qwen3samestack | unsloth/ckpt_unsloth_m | **VALID** | VALID | -0.0012 |  |
| qwen3samestack | e4b/reference_attn4_m | **NOT_RUN** | — | — | skipped by TC1_SKIP |
| qwen3samestack | e4b/fused_attn4_m_t28 | **VALID** | VALID | 0.0006 |  |
| qwen3samestack | e4b/fused_attn4_m_t28_d2 | **VALID** | VALID | -0.0011 |  |
| qwen3samestack | unsloth/ckpt_unsloth_m_d2 | **VALID** | VALID | -0.0034 |  |
| qwen3samestack | e4b/fused_attn4_m_d2 | **VALID** | VALID | 0.0004 |  |

## Predictions P50 / P51 / P52 (TC1-PREREG amendment 25: the matched position with both frameworks on one stack; scored mechanically)
| prediction | family | verdict | evidence |
|---|---|---|---|
| P50 | qwen3samestack | **HELD** | Unsloth/e4b on one stack 2.352 [2.348, 2.356 over 4 cross-draw ratios] vs [1.9, 2.9]; s/step e4b 3.494 (within 0.1%), Unsloth 8.218 (within 0.3%); peak e4b 27.49 / Unsloth 24.27 GB; held-out at N e4b 0.7569 / Unsloth 0.7557 |
| P51 | qwen3samestack | **HELD** | e4b matched arm venv-unsloth / venv-e4b 0.900 [0.893, 0.907] vs [0.8, 0.95]; s/step venv-unsloth 3.496 / 3.492, venv-e4b 3.855 / 3.913; amendment 24's P47 read 0.882 |
| P52 | qwen3samestack | **UNTESTED** | `e4b/reference_attn4_m` —; `unsloth/ckpt_unsloth_m` COMPARABLE; e4b parity NO-REF |

## Load gate (TC1-PREREG amendment 33): 3 draw(s) voided for host load and run again
Each line: the arm, the attempt, the median host load1 over that attempt's run, the gate. A VOID attempt's files are in loadvoid/; the last attempt of each arm stands whatever its load.

- `qwen3samestack/e4b/fused_attn4_m attempt 0 load1_median 4.41 gate 6.0 status ok over 0`
- `qwen3samestack/unsloth/ckpt_unsloth_m attempt 0 load1_median 4.08 gate 6.0 status ok over 0`
- `qwen3samestack/e4b/fused_attn4_m_t28 attempt 0 load1_median 2.94 gate 6.0 status ok over 0`
- `qwen3samestack/e4b/fused_attn4_m_t28_d2 attempt 0 load1_median 6.26 gate 6.0 status ok over 1`
- `qwen3samestack/e4b/fused_attn4_m_t28_d2 attempt 0 VOID (host load1 median 6.26 > 6.0): re-run 1 of 2`
- `qwen3samestack/e4b/fused_attn4_m_t28_d2 attempt 1 load1_median 8.44 gate 6.0 status ok over 1`
- `qwen3samestack/e4b/fused_attn4_m_t28_d2 attempt 1 VOID (host load1 median 8.44 > 6.0): re-run 2 of 2`
- `qwen3samestack/e4b/fused_attn4_m_t28_d2 attempt 2 load1_median 5.85 gate 6.0 status ok over 0`
- `qwen3samestack/unsloth/ckpt_unsloth_m_d2 attempt 0 load1_median 6.33 gate 6.0 status ok over 1`
- `qwen3samestack/unsloth/ckpt_unsloth_m_d2 attempt 0 VOID (host load1 median 6.33 > 6.0): re-run 1 of 2`
- `qwen3samestack/unsloth/ckpt_unsloth_m_d2 attempt 1 load1_median 5.22 gate 6.0 status ok over 0`
- `qwen3samestack/e4b/fused_attn4_m_d2 attempt 0 load1_median 5.83 gate 6.0 status ok over 0`
