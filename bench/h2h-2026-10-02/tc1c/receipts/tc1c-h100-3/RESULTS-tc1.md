# TC1 — Qwen3-30B-A3B on one RTX 5090: e4b vs Unsloth vs HF+PEFT vs axolotl at matched work, init and adapter precision (tc1)
Rule (tc1/TC1-PREREG.md): status per attempt in the vocabulary OK / REFUSED / OOM / INSTALL_FAILED / LOAD_FAULT / HARNESS_ERROR / ALARM / NOT_RUN; VALID/VOID per tp4's predicates plus the matched-set predicates (R3: matched init complete, fp32 adapters, step-0 held-out within 0.005 of e4b/reference_attn4_m); VERDICT exactly one of VALID / VOID / QUALITY_FAIL / OOM / UNSUPPORTED / HARNESS_ERROR / ALARM / NOT_RUN (QUALITY_FAIL = held-out |Δ| at N > 0.05 vs e4b/fused_attn4_m); positions = s/step ratios other/e4b from the medians over both draws (stability |d1-d2|/mean ≤ 5 % e4b / 10 % others, else UNSTABLE), quoted only when both arms are VALID; equivalence vs e4b/reference_attn4_m EQUIVALENT ≤ 0.02 / COMPARABLE ≤ 0.05 / else DIVERGENT; frozen base SAME-BYTES / DIFFERENT / N-A per slot; predictions P1–P10 (+ P1b) scored HELD / FALSIFIED / UNTESTED. VOID never enters a ratio or an equivalence reading. Nothing is licensed; no cross-box number is divided into these.
`versions.txt`
```
e4b 0.43.0 @e1837cff1d0faef7b3e04b3891c76dc22af5678c (GitHub main)
gnf4 0.34.1 @00929a493f8ca6ef60c950d666eb5fa5cee38df6 (GitHub main)
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
e4b(t212) 0.43.0 @e1837cff1d0faef7b3e04b3891c76dc22af5678c
gnf4(t212) 0.34.1 @00929a493f8ca6ef60c950d666eb5fa5cee38df6
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
 "run_id": "tc1c-h100-3",
 "instance_id": "54091206",
 "gpu": "NVIDIA H100 NVL",
 "driver": "595.71.05",
 "cpu": "AMD EPYC 9534 64-Core Processor",
 "nproc": 224,
 "mem_total_kb": "1584958328",
 "cgroup_memory_max": "",
 "disk_root": "overlay         320G   77M  320G   1% /",
 "hostname": "99610503f250",
 "registered_gpu_class": "H100 NVL",
 "prereg": "tc1/TC1-PREREG.md",
 "local_box": null,
 "local_snapshot": null
}
```

### Qwen3-30B-A3B (`qwen3`, registered n_layers 48, attention census 192)
- model `Qwen/Qwen3-30B-A3B` @ `ad44e777bcd1`; tokens sha `bfc742f67e37`; N=20; fixture template alpaca seq 2048 micro-batch 2 × accum 4 lr 0.0002 r 16 α 16 optimizer adamw_8bit(lr=0.0002, weight_decay=0.001) schedule=linear warmup_steps=5 autocast False; e4b trainable 642514944; box_class H100 NVL gpu NVIDIA H100 NVL
| framework | arm | status | validity | **VERDICT** | matched | init / dtype | step-0 class | N | s/step med(11+) | tok/s | peak GB | J/step | train first→last | held-out 0→final | quality Δ | engagement | trainable | regime | reason / why |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| e4b | fused_attn4_m | **OK** | VALID | **VALID** | yes | matched:3407 (complete 12480/12480) / float32 | SAME-BYTES-CLASS (0.0000) | 20 | 3.188 | 433.4 | 27.260 | 548.9 | 2.0735→0.8334 | 1.9714→0.8523 | 0.0000 | patched 48 / kcalls 768 | 642514944 | 4-bit experts (e4b NF4) + NF4 attention |  |
| unsloth | ckpt_unsloth_m | **OK** | VALID | **VALID** | yes | matched:3407 (complete 12480/12480) / float32 | NEAR (0.0409) | 20 | 2.548 | 433.7 | 24.269 | 435.7 | 2.0577→0.8315 | 1.9305→0.8472 | -0.0051 | stacks 96 / fwd 384 / u8 48 | 642514944 | 4-bit expert stacks + bnb-4bit attention (Params4bit stacks 96, Linear4bit 384) |  |
| e4b | reference_attn4_m | **OK** | VALID | **VALID** | yes | matched:3407 (complete 12480/12480) / float32 | NEAR (0.0163) | 20 | 52.091 | 28.7 | 27.147 | 2182.1 | 2.0461→0.8320 | 1.9551→0.8501 | -0.0021 | patched 0 / kcalls 0 | 642514944 | 4-bit experts (e4b NF4) + NF4 attention |  |
| e4b | fused_attn4_m_d2 | **OK** | VALID | **VALID** | yes | matched:3407 (complete 12480/12480) / float32 | SAME-BYTES-CLASS (0.0000) | 20 | 3.104 | 492.7 | 27.241 | 521.2 | 2.0735→0.8323 | 1.9714→0.8520 | -0.0003 | patched 48 / kcalls 768 | 642514944 | 4-bit experts (e4b NF4) + NF4 attention |  |
| unsloth | ckpt_unsloth_m_d2 | **OK** | VALID | **VALID** | yes | matched:3407 (complete 12480/12480) / float32 | NEAR (0.0409) | 20 | 2.594 | 551.0 | 24.269 | 412.0 | 2.0577→0.8345 | 1.9305→0.8504 | -0.0019 | stacks 96 / fwd 384 / u8 48 | 642514944 | 4-bit expert stacks + bnb-4bit attention (Params4bit stacks 96, Linear4bit 384) |  |
| hf | hf_peft_m | **ALARM** | — | **ALARM** | yes | — / — | — | 20 | — | — | — | — | —→— | —→— | — | peft mods — / params — / fwd — | — | — | arm alarm 1800 s (SIGALRM; the process could not write its own stub) |
| axolotl | ckpt_axolotl_m | **OK** | VALID | **VALID** | yes | matched:3407 (complete 12480/12480) / float32 | NEAR (0.0113) | 20 | 4.087 | 349.3 | 26.882 | 1158.2 | 2.0510→0.8387 | 1.9601→0.8549 | 0.0026 | peft mods 192 / params 96 / fwd 384 | 642514944 | 4-bit expert stacks (axolotl quantize_moe_experts: 96 parametrized stacks, nf4/64+dq) + bnb-4bit attention (Linear4bit 384) |  |
| e4b | fused_attn4_m_prof | **OK** | VALID | **VALID** | yes | matched:3407 (complete 12480/12480) / float32 | SAME-BYTES-CLASS (0.0000) | 20 | 3.182 | 468.7 | 27.259 | 288.2 | 2.0735→0.8340 | 1.9714→0.8517 | -0.0006 | patched 48 / kcalls 768 | 642514944 | 4-bit experts (e4b NF4) + NF4 attention |  |
| unsloth | ckpt_unsloth_prof | **OK** | VALID | **VALID** | yes | matched:3407 (complete 12480/12480) / float32 | NEAR (0.0409) | 20 | 2.861 | 487.4 | 24.269 | 238.9 | 2.0577→0.8319 | 1.9305→0.8490 | -0.0033 | stacks 96 / fwd 384 / u8 48 | 642514944 | 4-bit expert stacks + bnb-4bit attention (Params4bit stacks 96, Linear4bit 384) |  |
- prologue `e4b/fused_attn4_m` **102.7 s** before step 1 (59% of the arm): c1_before 54.5, c1_after 26.0, load_weights 22.5, eval0 9.9; unattributed 1.139; budget 1260.0
- prologue `unsloth/ckpt_unsloth_m` **106.9 s** before step 1 (60% of the arm): c1_before 49.8, load_weights 29.1, c1_after 24.6, eval0 12.3; unattributed 8.619; budget 1260.0
- prologue `e4b/reference_attn4_m` **112.4 s** before step 1 (9% of the arm): c1_before 54.0, c1_after 28.4, load_weights 23.0, eval0 19.7; unattributed 1.194; budget 1890.0
- prologue `e4b/fused_attn4_m_d2` **95.0 s** before step 1 (60% of the arm): c1_before 54.1, c1_after 26.4, load_weights 23.0, attn4 5.2; unattributed 1.202; budget 1260.0
- prologue `unsloth/ckpt_unsloth_m_d2` **97.5 s** before step 1 (63% of the arm): c1_before 50.1, load_weights 26.3, c1_after 24.8, eval0 5.4; unattributed 8.745; budget 1260.0
- prologue `axolotl/ckpt_axolotl_m` **99.4 s** before step 1 (52% of the arm): c1_before 53.6, load_weights 27.5, c1_after 26.2, eval0 5.6; unattributed 5.016; budget 945.0
- prologue `e4b/fused_attn4_m_prof` **101.1 s** before step 1 (34% of the arm): c1_before 58.6, load_weights 22.9, c1_after 16.9, attn4 5.4; unattributed 1.276; budget 840.0
- prologue `unsloth/ckpt_unsloth_prof` **100.5 s** before step 1 (34% of the arm): c1_before 52.5, load_weights 26.4, c1_after 15.5, eval0 5.5; unattributed 9.081; budget 840.0
- draws (R1): `e4b/fused_attn4_m` STABLE (3.188/3.104 s, |Δ|/mean 2.7% vs 5%); `unsloth/ckpt_unsloth_m` STABLE (2.548/2.594 s, |Δ|/mean 1.8% vs 5%); `e4b/reference_attn4_m` SINGLE (single draw (no second draw registered)); `axolotl/ckpt_axolotl_m` SINGLE (single draw (no second draw registered)); `e4b/fused_attn4_m_prof` SINGLE (single draw (no second draw registered)); `unsloth/ckpt_unsloth_prof` SINGLE (single draw (no second draw registered))
- e4b internal parity (tp1's rule, informational): fused_attn4_m vs reference_attn4_m Δfinal 0.00137, median step |Δ| 0.00229 → **PASS** (band 0.05/0.05); ×16.34 faster per step, peak ×1.004
- **MATCHED POSITION: s/step ratio unsloth/e4b = 0.817** [0.799, 0.836 over 4 cross-draw ratios] (2.571 vs 3.146 s, medians over 2/2 draws; unsloth faster per step); peak VRAM unsloth 24.27 vs e4b 27.25 GB (Δ -2.98); J/step unsloth 423.8 vs e4b 535.1 (×0.792); tok/s unsloth 492.4 vs e4b 463.0; unsloth regime: 4-bit expert stacks + bnb-4bit attention (Params4bit stacks 96, Linear4bit 384)
- quality reading at N=20 (unsloth): held-out e4b 0.8523 / unsloth 0.8472 (Δ -0.0051) → **COMPARABLE** (|Δ| ≤ 0.05 reads COMPARABLE); step-0 Δ -0.0409
- **NO MATCHED POSITION QUOTED (hf)** — not quoted: e4b STABLE / hf hf/hf_peft_m is ALARM
- **MATCHED POSITION: s/step ratio axolotl/e4b = 1.299** [1.282, 1.317 over 2 cross-draw ratios] (4.087 vs 3.146 s, medians over 1/2 draws; e4b faster per step); peak VRAM axolotl 26.88 vs e4b 27.25 GB (Δ -0.37); J/step axolotl 1158.2 vs e4b 535.1 (×2.165); tok/s axolotl 349.3 vs e4b 463.0; axolotl regime: 4-bit expert stacks (axolotl quantize_moe_experts: 96 parametrized stacks, nf4/64+dq) + bnb-4bit attention (Linear4bit 384)
- quality reading at N=20 (axolotl): held-out e4b 0.8523 / axolotl 0.8549 (Δ 0.0026) → **COMPARABLE** (|Δ| ≤ 0.05 reads COMPARABLE); step-0 Δ -0.0113
- equivalence vs `e4b/fused_attn4_m` (I/H; EQUIVALENT iff both ≤ band = max(0.005, 3 × fused-vs-reference |Δ|) = train 0.0069 / held-out 0.0064; COMPARABLE ≤ 0.05; draw-noise floor 0.0032): `unsloth/ckpt_unsloth_m` **EQUIVALENT** (median step |Δ| 0.0045, |Δ held-out at N| 0.0051, step-0 0.0409 NEAR, |Δ loss at step 2| 0.0031, paired rows mean -0.0051 ± 0.0034 SE over 8, favouring arm 5 / anchor 3); `e4b/reference_attn4_m` **INSIDE-DRAW-NOISE** (median step |Δ| 0.0023, |Δ held-out at N| 0.0021, step-0 0.0163 NEAR, |Δ loss at step 2| 0.0241, paired rows mean -0.0021 ± 0.0035 SE over 8, favouring arm 6 / anchor 2) — |delta| 0.0021 / 0.0023 are narrower than the draw-noise floor 0.0032: inside the draw noise, not a precision statement; `e4b/fused_attn4_m_d2` **INSIDE-DRAW-NOISE** (median step |Δ| 0.0011, |Δ held-out at N| 0.0003, step-0 0.0000 SAME-BYTES-CLASS, |Δ loss at step 2| 0.0070, paired rows mean -0.0003 ± 0.0020 SE over 8, favouring arm 3 / anchor 5) — |delta| 0.0003 / 0.0011 are narrower than the draw-noise floor 0.0032: inside the draw noise, not a precision statement; `unsloth/ckpt_unsloth_m_d2` **INSIDE-DRAW-NOISE** (median step |Δ| 0.0028, |Δ held-out at N| 0.0019, step-0 0.0409 NEAR, |Δ loss at step 2| 0.0028, paired rows mean -0.0019 ± 0.0032 SE over 8, favouring arm 5 / anchor 3) — |delta| 0.0019 / 0.0028 are narrower than the draw-noise floor 0.0032: inside the draw noise, not a precision statement; `hf/hf_peft_m` **—** — no OK receipt; `axolotl/ckpt_axolotl_m` **COMPARABLE** (median step |Δ| 0.0100, |Δ held-out at N| 0.0026, step-0 0.0113 NEAR, |Δ loss at step 2| 0.0027, paired rows mean +0.0026 ± 0.0038 SE over 8, favouring arm 2 / anchor 6); `e4b/fused_attn4_m_prof` **INSIDE-DRAW-NOISE** (median step |Δ| 0.0011, |Δ held-out at N| 0.0006, step-0 0.0000 SAME-BYTES-CLASS, |Δ loss at step 2| 0.0051, paired rows mean -0.0006 ± 0.0017 SE over 8, favouring arm 3 / anchor 5) — |delta| 0.0006 / 0.0011 are narrower than the draw-noise floor 0.0032: inside the draw noise, not a precision statement; `unsloth/ckpt_unsloth_prof` **EQUIVALENT** (median step |Δ| 0.0036, |Δ held-out at N| 0.0033, step-0 0.0409 NEAR, |Δ loss at step 2| 0.0245, paired rows mean -0.0033 ± 0.0039 SE over 8, favouring arm 4 / anchor 4)
- matched_init_sha (B, name-free, canonical slot order): anchor `f7832488926eda91`; `e4b/fused_attn4_m` same; `unsloth/ckpt_unsloth_m` same; `e4b/reference_attn4_m` same; `e4b/fused_attn4_m_d2` same; `unsloth/ckpt_unsloth_m_d2` same; `axolotl/ckpt_axolotl_m` same; `e4b/fused_attn4_m_prof` same; `unsloth/ckpt_unsloth_prof` same
- frozen base (R5), anchor `e4b/fused_attn4_m`: gate_up nf4/64 sha 8c3b3fd78e89 control detects, down nf4/64 sha 59569d810a1a control detects, q_proj nf4/64+dq sha 5c5e402ce51d control detects
- frozen base `unsloth/ckpt_unsloth_m`: gate_up **N-A** (regime differs: anchor nf4/64 vs nf4/64+dq) control detects, down **N-A** (regime differs: anchor nf4/64 vs nf4/64+dq) control detects, q_proj **SAME-BYTES** (nf4/64+dq) control detects
- frozen base `e4b/reference_attn4_m`: gate_up **SAME-BYTES** (nf4/64) control detects, down **SAME-BYTES** (nf4/64) control detects, q_proj **SAME-BYTES** (nf4/64+dq) control detects
- frozen base `e4b/fused_attn4_m_d2`: gate_up **SAME-BYTES** (nf4/64) control detects, down **SAME-BYTES** (nf4/64) control detects, q_proj **SAME-BYTES** (nf4/64+dq) control detects
- frozen base `unsloth/ckpt_unsloth_m_d2`: gate_up **N-A** (regime differs: anchor nf4/64 vs nf4/64+dq) control detects, down **N-A** (regime differs: anchor nf4/64 vs nf4/64+dq) control detects, q_proj **SAME-BYTES** (nf4/64+dq) control detects
- frozen base `axolotl/ckpt_axolotl_m`: gate_up **N-A** (regime differs: anchor nf4/64 vs nf4/64+dq) control detects, down **N-A** (regime differs: anchor nf4/64 vs nf4/64+dq) control detects, q_proj **N-A** (regime differs: anchor nf4/64+dq vs nf4/64) control detects
- frozen base `e4b/fused_attn4_m_prof`: gate_up **SAME-BYTES** (nf4/64) control detects, down **SAME-BYTES** (nf4/64) control detects, q_proj **SAME-BYTES** (nf4/64+dq) control detects
- frozen base `unsloth/ckpt_unsloth_prof`: gate_up **N-A** (regime differs: anchor nf4/64 vs nf4/64+dq) control detects, down **N-A** (regime differs: anchor nf4/64 vs nf4/64+dq) control detects, q_proj **SAME-BYTES** (nf4/64+dq) control detects
- where Unsloth's step goes (`unsloth/ckpt_unsloth_prof`, descriptive): device busy fraction 0.305, device events/step 81780, CPU ops/step 515750, CPU self by family {'other': 0.3896, 'autograd': 0.2884, 'norm_act': 0.132, 'matmul': 0.0735, 'memcpy': 0.0504, 'fused_kernel': 0.043, 'routing': 0.0175, 'optimizer': 0.0057}

## Verdicts
| family | arm | VERDICT | validity | quality | note |
|---|---|---|---|---|---|
| qwen3 | e4b/fused_attn4_m | **VALID** | VALID | 0.0000 |  |
| qwen3 | unsloth/ckpt_unsloth_m | **VALID** | VALID | -0.0051 |  |
| qwen3 | e4b/reference_attn4_m | **VALID** | VALID | -0.0021 |  |
| qwen3 | e4b/fused_attn4_m_d2 | **VALID** | VALID | -0.0003 |  |
| qwen3 | unsloth/ckpt_unsloth_m_d2 | **VALID** | VALID | -0.0019 |  |
| qwen3 | hf/hf_peft_m | **ALARM** | — | — | arm alarm 1800 s (SIGALRM; the process could not write its own stub) |
| qwen3 | axolotl/ckpt_axolotl_m | **VALID** | VALID | 0.0026 |  |
| qwen3 | e4b/fused_attn4_m_prof | **VALID** | VALID | -0.0006 |  |
| qwen3 | unsloth/ckpt_unsloth_prof | **VALID** | VALID | -0.0033 |  |

## Predictions P1–P10 (+ P1b) (TC1-PREREG.md + phase 2, scored mechanically)
| prediction | family | verdict | evidence |
|---|---|---|---|
| P1 | qwen3 | **FALSIFIED** | matched unsloth/e4b 0.817 vs [2.0, 5.0]; BELOW 1.5: the standing position is refuted and superseded |
| P1b | qwen3 | **UNTESTED** | ckpt_unsloth_t28 missing: no receipt |
| P2 | qwen3 | **HELD** | e4b |d1-d2|/mean 2.7% vs 5% -> STABLE; unsloth |d1-d2|/mean 1.8% vs 5% -> STABLE |
| P3 | qwen3 | **HELD** | e4b/reference_attn4_m INSIDE-DRAW-NOISE (median step |Δ| 0.0023, |Δ held-out at N| 0.0021, band {'train': 0.006870000000000376, 'heldout': 0.006390000000000229}); unsloth/ckpt_unsloth_m EQUIVALENT (median step |Δ| 0.0045, |Δ held-out at N| 0.0051, band {'train': 0.006870000000000376, 'heldout': 0.006390000000000229}); draw-noise floor 0.0032 |
| P4 | qwen3 | **UNTESTED** | shipped missing / fused_m VALID: both must be VALID on the same box |
| P5 | qwen3 | **UNTESTED** | hf_peft_m ALARM, hf_peft_m_mb1 missing: neither an OOM pair nor a trained arm |
| P6 | qwen3 | **FALSIFIED** | axolotl trained; axolotl/e4b 1.299 vs [1.5, 6] |
| P7 | qwen3 | **UNTESTED** | unsloth_best missing / unsloth_m VALID / e4b_m VALID: all three must be VALID |
| P8 | qwen3 | **FALSIFIED** | device_busy_fraction 0.305 (>= 0.5 predicted on the grouped_mm arm); device events/step 81780 |
| P9 | qwen3 | **HELD** | fused_m vs reference_m PASS (Δfinal 0.00137, median 0.00229) |
| P10 | qwen3 | **UNTESTED** | C1 bit-exact; expert slots ['N-A', 'N-A'] (regime differs: anchor nf4/64 vs nf4/64+dq); attention SAME-BYTES |
