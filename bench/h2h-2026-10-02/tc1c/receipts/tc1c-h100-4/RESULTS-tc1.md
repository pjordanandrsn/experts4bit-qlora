# TC1 — Qwen3-30B-A3B on one RTX 5090: e4b vs Unsloth vs HF+PEFT vs axolotl at matched work, init and adapter precision (tc1)
Rule (tc1/TC1-PREREG.md): status per attempt in the vocabulary OK / REFUSED / OOM / INSTALL_FAILED / LOAD_FAULT / HARNESS_ERROR / ALARM / NOT_RUN; VALID/VOID per tp4's predicates plus the matched-set predicates (R3: matched init complete, fp32 adapters, step-0 held-out within 0.005 of e4b/reference_attn4_m); VERDICT exactly one of VALID / VOID / QUALITY_FAIL / OOM / UNSUPPORTED / HARNESS_ERROR / ALARM / NOT_RUN (QUALITY_FAIL = held-out |Δ| at N > 0.05 vs e4b/fused_attn4_m); positions = s/step ratios other/e4b from the medians over both draws (stability |d1-d2|/mean ≤ 5 % e4b / 10 % others, else UNSTABLE), quoted only when both arms are VALID; equivalence vs e4b/reference_attn4_m EQUIVALENT ≤ 0.02 / COMPARABLE ≤ 0.05 / else DIVERGENT; frozen base SAME-BYTES / DIFFERENT / N-A per slot; predictions P1–P10 (+ P1b) scored HELD / FALSIFIED / UNTESTED. VOID never enters a ratio or an equivalence reading. Nothing is licensed; no cross-box number is divided into these.
`versions.txt`
```
e4b 0.43.0 @8846764207dfdfac079bf332a004e9294c9d7c10 (GitHub main)
gnf4 0.35.0 @ac848187746f20ccb1ae154ea8ec9fa1448465ed (GitHub main)
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
e4b(t212) 0.43.0 @8846764207dfdfac079bf332a004e9294c9d7c10
gnf4(t212) 0.35.0 @ac848187746f20ccb1ae154ea8ec9fa1448465ed
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
 "run_id": "tc1c-h100-4",
 "instance_id": "54103635",
 "gpu": "NVIDIA H100 NVL",
 "driver": "595.71.05",
 "cpu": "AMD EPYC 9534 64-Core Processor",
 "nproc": 224,
 "mem_total_kb": "1584958328",
 "cgroup_memory_max": "",
 "disk_root": "overlay         320G   77M  320G   1% /",
 "hostname": "70a5114d560f",
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
| e4b | fused_attn4_m | **OK** | VALID | **VALID** | yes | matched:3407 (complete 12480/12480) / float32 | SAME-BYTES-CLASS (0.0000) | 20 | 2.333 | 560.4 | 34.078 | 475.1 | 2.0735→0.8311 | 1.9714→0.8486 | 0.0000 | patched 48 / kcalls 384 | 642514944 | 4-bit experts (e4b NF4) + NF4 attention |  |
| unsloth | ckpt_unsloth_m | **OK** | VALID | **VALID** | yes | matched:3407 (complete 12480/12480) / float32 | NEAR (0.0185) | 20 | 2.593 | 430.2 | 24.269 | 480.1 | 2.0319→0.8330 | 1.9529→0.8507 | 0.0022 | stacks 96 / fwd 384 / u8 48 | 642514944 | 4-bit expert stacks + bnb-4bit attention (Params4bit stacks 96, Linear4bit 384) |  |
| e4b | reference_attn4_m | **OK** | VALID | **VALID** | yes | matched:3407 (complete 12480/12480) / float32 | NEAR (0.0163) | 20 | 50.726 | 29.5 | 27.147 | 2303.0 | 2.0461→0.8320 | 1.9551→0.8501 | 0.0016 | patched 0 / kcalls 0 | 642514944 | 4-bit experts (e4b NF4) + NF4 attention |  |
| e4b | fused_attn4_m_d2 | **OK** | VALID | **VALID** | yes | matched:3407 (complete 12480/12480) / float32 | SAME-BYTES-CLASS (0.0000) | 20 | 2.353 | 664.5 | 34.077 | 443.8 | 2.0735→0.8341 | 1.9714→0.8500 | 0.0014 | patched 48 / kcalls 384 | 642514944 | 4-bit experts (e4b NF4) + NF4 attention |  |
| unsloth | ckpt_unsloth_m_d2 | **OK** | VALID | **VALID** | yes | matched:3407 (complete 12480/12480) / float32 | NEAR (0.0185) | 20 | 2.560 | 552.6 | 24.269 | 451.1 | 2.0319→0.8314 | 1.9529→0.8483 | -0.0003 | stacks 96 / fwd 384 / u8 48 | 642514944 | 4-bit expert stacks + bnb-4bit attention (Params4bit stacks 96, Linear4bit 384) |  |
| hf | hf_peft_m | **NOT_RUN** | — | **NOT_RUN** | yes | — / — | — | 20 | — | — | — | — | —→— | —→— | — | peft mods — / params — / fwd — | — | — | skipped by TC1_SKIP |
| axolotl | ckpt_axolotl_m | **NOT_RUN** | — | **NOT_RUN** | yes | — / — | — | 20 | — | — | — | — | —→— | —→— | — | peft mods — / params — / fwd — | — | — | skipped by TC1_SKIP |
| e4b | fused_attn4_m_prof | **OK** | VALID | **VALID** | yes | matched:3407 (complete 12480/12480) / float32 | SAME-BYTES-CLASS (0.0000) | 20 | 2.402 | 524.5 | 34.078 | 298.9 | 2.0735→0.8354 | 1.9714→0.8518 | 0.0032 | patched 48 / kcalls 384 | 642514944 | 4-bit experts (e4b NF4) + NF4 attention |  |
| unsloth | ckpt_unsloth_prof | **OK** | VALID | **VALID** | yes | matched:3407 (complete 12480/12480) / float32 | NEAR (0.0185) | 20 | 2.827 | 489.8 | 24.269 | 259.4 | 2.0319→0.8341 | 1.9529→0.8518 | 0.0033 | stacks 96 / fwd 384 / u8 48 | 642514944 | 4-bit expert stacks + bnb-4bit attention (Params4bit stacks 96, Linear4bit 384) |  |
- prologue `e4b/fused_attn4_m` **115.0 s** before step 1 (67% of the arm): c1_before 59.7, c1_after 28.1, load_weights 27.6, eval0 9.8; unattributed 1.471; budget 1260.0
- prologue `unsloth/ckpt_unsloth_m` **109.9 s** before step 1 (60% of the arm): c1_before 52.2, load_weights 29.9, c1_after 26.2, eval0 12.2; unattributed 8.636; budget 1260.0
- prologue `e4b/reference_attn4_m` **117.7 s** before step 1 (10% of the arm): c1_before 57.8, c1_after 28.8, load_weights 23.7, eval0 19.0; unattributed 1.131; budget 1890.0
- prologue `e4b/fused_attn4_m_d2` **93.4 s** before step 1 (67% of the arm): c1_before 54.8, c1_after 26.9, load_weights 19.9, attn4 5.0; unattributed 1.148; budget 1260.0
- prologue `unsloth/ckpt_unsloth_m_d2` **95.1 s** before step 1 (63% of the arm): c1_before 49.0, load_weights 24.9, c1_after 24.7, eval0 5.3; unattributed 8.762; budget 1260.0
- prologue `e4b/fused_attn4_m_prof` **94.1 s** before step 1 (39% of the arm): c1_before 53.8, load_weights 22.0, c1_after 11.6, attn4 5.3; unattributed 1.195; budget 840.0
- prologue `unsloth/ckpt_unsloth_prof` **101.0 s** before step 1 (34% of the arm): c1_before 52.5, load_weights 27.3, c1_after 15.4, eval0 5.3; unattributed 8.931; budget 840.0
- draws (R1): `e4b/fused_attn4_m` STABLE (2.333/2.353 s, |Δ|/mean 0.8% vs 5%); `unsloth/ckpt_unsloth_m` STABLE (2.593/2.560 s, |Δ|/mean 1.3% vs 5%); `e4b/reference_attn4_m` SINGLE (single draw (no second draw registered)); `e4b/fused_attn4_m_prof` SINGLE (single draw (no second draw registered)); `unsloth/ckpt_unsloth_prof` SINGLE (single draw (no second draw registered))
- e4b internal parity (tp1's rule, informational): fused_attn4_m vs reference_attn4_m Δfinal 0.00086, median step |Δ| 0.00220 → **PASS** (band 0.05/0.05); ×21.74 faster per step, peak ×1.255
- **MATCHED POSITION: s/step ratio unsloth/e4b = 1.100** [1.088, 1.111 over 4 cross-draw ratios] (2.577 vs 2.343 s, medians over 2/2 draws; e4b faster per step); peak VRAM unsloth 24.27 vs e4b 34.08 GB (Δ -9.81); J/step unsloth 465.6 vs e4b 459.4 (×1.013); tok/s unsloth 491.4 vs e4b 612.5; unsloth regime: 4-bit expert stacks + bnb-4bit attention (Params4bit stacks 96, Linear4bit 384)
- quality reading at N=20 (unsloth): held-out e4b 0.8486 / unsloth 0.8507 (Δ 0.0022) → **COMPARABLE** (|Δ| ≤ 0.05 reads COMPARABLE); step-0 Δ -0.0185
- **NO MATCHED POSITION QUOTED (hf)** — not quoted: e4b STABLE / hf hf/hf_peft_m is NOT_RUN
- **NO MATCHED POSITION QUOTED (axolotl)** — not quoted: e4b STABLE / axolotl axolotl/ckpt_axolotl_m is NOT_RUN
- equivalence vs `e4b/fused_attn4_m` (I/H; EQUIVALENT iff both ≤ band = max(0.005, 3 × fused-vs-reference |Δ|) = train 0.0066 / held-out 0.0050; COMPARABLE ≤ 0.05; draw-noise floor 0.0024): `unsloth/ckpt_unsloth_m` **INSIDE-DRAW-NOISE** (median step |Δ| 0.0022, |Δ held-out at N| 0.0022, step-0 0.0185 NEAR, |Δ loss at step 2| 0.0093, paired rows mean +0.0021 ± 0.0048 SE over 8, favouring arm 4 / anchor 4) — |delta| 0.0022 / 0.0022 are narrower than the draw-noise floor 0.0024: inside the draw noise, not a precision statement; `e4b/reference_attn4_m` **INSIDE-DRAW-NOISE** (median step |Δ| 0.0022, |Δ held-out at N| 0.0016, step-0 0.0163 NEAR, |Δ loss at step 2| 0.0097, paired rows mean +0.0016 ± 0.0037 SE over 8, favouring arm 5 / anchor 3) — |delta| 0.0016 / 0.0022 are narrower than the draw-noise floor 0.0024: inside the draw noise, not a precision statement; `e4b/fused_attn4_m_d2` **INSIDE-DRAW-NOISE** (median step |Δ| 0.0015, |Δ held-out at N| 0.0014, step-0 0.0000 SAME-BYTES-CLASS, |Δ loss at step 2| 0.0051, paired rows mean +0.0014 ± 0.0051 SE over 8, favouring arm 4 / anchor 4) — |delta| 0.0014 / 0.0015 are narrower than the draw-noise floor 0.0024: inside the draw noise, not a precision statement; `unsloth/ckpt_unsloth_m_d2` **INSIDE-DRAW-NOISE** (median step |Δ| 0.0022, |Δ held-out at N| 0.0003, step-0 0.0185 NEAR, |Δ loss at step 2| 0.0000, paired rows mean -0.0003 ± 0.0033 SE over 8, favouring arm 4 / anchor 4) — |delta| 0.0003 / 0.0022 are narrower than the draw-noise floor 0.0024: inside the draw noise, not a precision statement; `hf/hf_peft_m` **—** — no OK receipt; `axolotl/ckpt_axolotl_m` **—** — no OK receipt; `e4b/fused_attn4_m_prof` **EQUIVALENT** (median step |Δ| 0.0020, |Δ held-out at N| 0.0032, step-0 0.0000 SAME-BYTES-CLASS, |Δ loss at step 2| 0.0190, paired rows mean +0.0032 ± 0.0045 SE over 8, favouring arm 3 / anchor 5); `unsloth/ckpt_unsloth_prof` **EQUIVALENT** (median step |Δ| 0.0031, |Δ held-out at N| 0.0033, step-0 0.0185 NEAR, |Δ loss at step 2| 0.0084, paired rows mean +0.0032 ± 0.0045 SE over 8, favouring arm 4 / anchor 4)
- matched_init_sha (B, name-free, canonical slot order): anchor `f7832488926eda91`; `e4b/fused_attn4_m` same; `unsloth/ckpt_unsloth_m` same; `e4b/reference_attn4_m` same; `e4b/fused_attn4_m_d2` same; `unsloth/ckpt_unsloth_m_d2` same; `e4b/fused_attn4_m_prof` same; `unsloth/ckpt_unsloth_prof` same
- frozen base (R5), anchor `e4b/fused_attn4_m`: gate_up nf4/64 sha 8c3b3fd78e89 control detects, down nf4/64 sha 59569d810a1a control detects, q_proj nf4/64+dq sha 5c5e402ce51d control detects
- frozen base `unsloth/ckpt_unsloth_m`: gate_up **N-A** (regime differs: anchor nf4/64 vs nf4/64+dq) control detects, down **N-A** (regime differs: anchor nf4/64 vs nf4/64+dq) control detects, q_proj **SAME-BYTES** (nf4/64+dq) control detects
- frozen base `e4b/reference_attn4_m`: gate_up **SAME-BYTES** (nf4/64) control detects, down **SAME-BYTES** (nf4/64) control detects, q_proj **SAME-BYTES** (nf4/64+dq) control detects
- frozen base `e4b/fused_attn4_m_d2`: gate_up **SAME-BYTES** (nf4/64) control detects, down **SAME-BYTES** (nf4/64) control detects, q_proj **SAME-BYTES** (nf4/64+dq) control detects
- frozen base `unsloth/ckpt_unsloth_m_d2`: gate_up **N-A** (regime differs: anchor nf4/64 vs nf4/64+dq) control detects, down **N-A** (regime differs: anchor nf4/64 vs nf4/64+dq) control detects, q_proj **SAME-BYTES** (nf4/64+dq) control detects
- frozen base `e4b/fused_attn4_m_prof`: gate_up **SAME-BYTES** (nf4/64) control detects, down **SAME-BYTES** (nf4/64) control detects, q_proj **SAME-BYTES** (nf4/64+dq) control detects
- frozen base `unsloth/ckpt_unsloth_prof`: gate_up **N-A** (regime differs: anchor nf4/64 vs nf4/64+dq) control detects, down **N-A** (regime differs: anchor nf4/64 vs nf4/64+dq) control detects, q_proj **SAME-BYTES** (nf4/64+dq) control detects
- where Unsloth's step goes (`unsloth/ckpt_unsloth_prof`, descriptive): device busy fraction 0.306, device events/step 81780, CPU ops/step 515750, CPU self by family {'other': 0.3831, 'autograd': 0.3039, 'norm_act': 0.122, 'matmul': 0.0738, 'memcpy': 0.0507, 'fused_kernel': 0.0431, 'routing': 0.0175, 'optimizer': 0.006}

## Verdicts
| family | arm | VERDICT | validity | quality | note |
|---|---|---|---|---|---|
| qwen3 | e4b/fused_attn4_m | **VALID** | VALID | 0.0000 |  |
| qwen3 | unsloth/ckpt_unsloth_m | **VALID** | VALID | 0.0022 |  |
| qwen3 | e4b/reference_attn4_m | **VALID** | VALID | 0.0016 |  |
| qwen3 | e4b/fused_attn4_m_d2 | **VALID** | VALID | 0.0014 |  |
| qwen3 | unsloth/ckpt_unsloth_m_d2 | **VALID** | VALID | -0.0003 |  |
| qwen3 | hf/hf_peft_m | **NOT_RUN** | — | — | skipped by TC1_SKIP |
| qwen3 | axolotl/ckpt_axolotl_m | **NOT_RUN** | — | — | skipped by TC1_SKIP |
| qwen3 | e4b/fused_attn4_m_prof | **VALID** | VALID | 0.0032 |  |
| qwen3 | unsloth/ckpt_unsloth_prof | **VALID** | VALID | 0.0033 |  |

## Predictions P1–P10 (+ P1b) (TC1-PREREG.md + phase 2, scored mechanically)
| prediction | family | verdict | evidence |
|---|---|---|---|
| P1 | qwen3 | **FALSIFIED** | matched unsloth/e4b 1.100 vs [2.0, 5.0]; BELOW 1.5: the standing position is refuted and superseded |
| P1b | qwen3 | **UNTESTED** | ckpt_unsloth_t28 missing: no receipt |
| P2 | qwen3 | **HELD** | e4b |d1-d2|/mean 0.8% vs 5% -> STABLE; unsloth |d1-d2|/mean 1.3% vs 5% -> STABLE |
| P3 | qwen3 | **HELD** | e4b/reference_attn4_m INSIDE-DRAW-NOISE (median step |Δ| 0.0022, |Δ held-out at N| 0.0016, band {'train': 0.006600000000000272, 'heldout': 0.005}); unsloth/ckpt_unsloth_m INSIDE-DRAW-NOISE (median step |Δ| 0.0022, |Δ held-out at N| 0.0022, band {'train': 0.006600000000000272, 'heldout': 0.005}); draw-noise floor 0.0024 |
| P4 | qwen3 | **UNTESTED** | shipped missing / fused_m VALID: both must be VALID on the same box |
| P5 | qwen3 | **UNTESTED** | hf_peft_m NOT_RUN, hf_peft_m_mb1 missing: neither an OOM pair nor a trained arm |
| P6 | qwen3 | **UNTESTED** | axolotl NOT_RUN (NOT_RUN / HARNESS_ERROR / ALARM is not a reading) |
| P7 | qwen3 | **UNTESTED** | unsloth_best missing / unsloth_m VALID / e4b_m VALID: all three must be VALID |
| P8 | qwen3 | **FALSIFIED** | device_busy_fraction 0.306 (>= 0.5 predicted on the grouped_mm arm); device events/step 81780 |
| P9 | qwen3 | **HELD** | fused_m vs reference_m PASS (Δfinal 0.00086, median 0.00220) |
| P10 | qwen3 | **UNTESTED** | C1 bit-exact; expert slots ['N-A', 'N-A'] (regime differs: anchor nf4/64 vs nf4/64+dq); attention SAME-BYTES |
