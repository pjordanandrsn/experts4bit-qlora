# TC1 — Qwen3-30B-A3B on one RTX 5090: e4b vs Unsloth vs HF+PEFT vs axolotl at matched work, init and adapter precision (tc1)
Rule (tc1/TC1-PREREG.md): status per attempt in the vocabulary OK / REFUSED / OOM / INSTALL_FAILED / LOAD_FAULT / HARNESS_ERROR / ALARM / NOT_RUN; VALID/VOID per tp4's predicates plus the matched-set predicates (R3: matched init complete, fp32 adapters, step-0 held-out within 0.005 of e4b/reference_attn4_m); VERDICT exactly one of VALID / VOID / QUALITY_FAIL / OOM / UNSUPPORTED / HARNESS_ERROR / ALARM / NOT_RUN (QUALITY_FAIL = held-out |Δ| at N > 0.05 vs e4b/fused_attn4_m); positions = s/step ratios other/e4b from the medians over both draws (stability |d1-d2|/mean ≤ 5 % e4b / 10 % others, else UNSTABLE), quoted only when both arms are VALID; equivalence vs e4b/reference_attn4_m EQUIVALENT ≤ 0.02 / COMPARABLE ≤ 0.05 / else DIVERGENT; frozen base SAME-BYTES / DIFFERENT / N-A per slot; predictions P1–P10 (+ P1b) scored HELD / FALSIFIED / UNTESTED. VOID never enters a ratio or an equivalence reading. Nothing is licensed; no cross-box number is divided into these.
`versions.txt`
```
e4b 0.44.0 @e775eff806df92e7da0ee0fef6f3a25b049023b1 (GitHub main)
gnf4 0.36.0 @81706a9c6673ed37b4e01480d318ec7d549434e9 (GitHub main)
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
e4b(t212) 0.44.0 @e775eff806df92e7da0ee0fef6f3a25b049023b1
gnf4(t212) 0.36.0 @81706a9c6673ed37b4e01480d318ec7d549434e9
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
 "run_id": "tc1c-h100-9",
 "instance_id": "54128342",
 "gpu": "NVIDIA H100 NVL",
 "driver": "595.71.05",
 "cpu": "AMD EPYC 9534 64-Core Processor",
 "nproc": 224,
 "mem_total_kb": "1584958328",
 "cgroup_memory_max": "",
 "disk_root": "overlay         320G   77M  320G   1% /",
 "hostname": "a0e46207002b",
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
| e4b | fused_attn4_m | **OK** | VALID | **VALID** | yes | matched:3407 (complete 12480/12480) / float32 | SAME-BYTES-CLASS (0.0000) | 20 | 2.512 | 591.1 | 27.186 | 578.4 | 2.0773→0.8326 | 1.9614→0.8467 | 0.0000 | patched 48 / kcalls 768 | 642514944 | 4-bit experts (e4b NF4) + NF4 attention |  |
| unsloth | ckpt_unsloth_m | **OK** | VALID | **VALID** | yes | matched:3407 (complete 12480/12480) / float32 | NEAR (0.0325) | 20 | 2.561 | 438.4 | 24.269 | 487.1 | 2.0566→0.8348 | 1.9289→0.8513 | 0.0046 | stacks 96 / fwd 384 / u8 48 | 642514944 | 4-bit expert stacks + bnb-4bit attention (Params4bit stacks 96, Linear4bit 384) |  |
| e4b | reference_attn4_m | **OK** | VALID | **VALID** | yes | matched:3407 (complete 12480/12480) / float32 | SAME-BYTES-CLASS (0.0063) | 20 | 50.299 | 29.7 | 27.147 | 2239.1 | 2.0461→0.8320 | 1.9551→0.8501 | 0.0035 | patched 0 / kcalls 0 | 642514944 | 4-bit experts (e4b NF4) + NF4 attention |  |
| e4b | fused_attn4_m_d2 | **OK** | VALID | **VALID** | yes | matched:3407 (complete 12480/12480) / float32 | SAME-BYTES-CLASS (0.0000) | 20 | 2.450 | 613.9 | 27.187 | 561.3 | 2.0773→0.8318 | 1.9614→0.8495 | 0.0028 | patched 48 / kcalls 768 | 642514944 | 4-bit experts (e4b NF4) + NF4 attention |  |
| unsloth | ckpt_unsloth_m_d2 | **OK** | VALID | **VALID** | yes | matched:3407 (complete 12480/12480) / float32 | NEAR (0.0325) | 20 | 2.552 | 551.6 | 24.269 | 449.3 | 2.0566→0.8361 | 1.9289→0.8515 | 0.0048 | stacks 96 / fwd 384 / u8 48 | 642514944 | 4-bit expert stacks + bnb-4bit attention (Params4bit stacks 96, Linear4bit 384) |  |
| hf | hf_peft_m | **NOT_RUN** | — | **NOT_RUN** | yes | — / — | — | 20 | — | — | — | — | —→— | —→— | — | peft mods — / params — / fwd — | — | — | skipped by TC1_SKIP |
| axolotl | ckpt_axolotl_m | **NOT_RUN** | — | **NOT_RUN** | yes | — / — | — | 20 | — | — | — | — | —→— | —→— | — | peft mods — / params — / fwd — | — | — | skipped by TC1_SKIP |
| e4b | fused_attn4_m_prof | **OK** | VALID | **VALID** | yes | matched:3407 (complete 12480/12480) / float32 | SAME-BYTES-CLASS (0.0000) | 20 | 2.741 | 455.1 | 27.205 | 366.4 | 2.0773→0.8319 | 1.9614→0.8480 | 0.0013 | patched 48 / kcalls 768 | 642514944 | 4-bit experts (e4b NF4) + NF4 attention |  |
| unsloth | ckpt_unsloth_prof | **OK** | VALID | **VALID** | yes | matched:3407 (complete 12480/12480) / float32 | NEAR (0.0325) | 20 | 2.902 | 482.9 | 24.269 | 265.9 | 2.0566→0.8356 | 1.9289→0.8529 | 0.0062 | stacks 96 / fwd 384 / u8 48 | 642514944 | 4-bit expert stacks + bnb-4bit attention (Params4bit stacks 96, Linear4bit 384) |  |
- prologue `e4b/fused_attn4_m` **92.1 s** before step 1 (64% of the arm): c1_before 51.7, c1_after 25.9, load_weights 20.1, attn4 5.0; unattributed 1.108; budget 1260.0
- prologue `unsloth/ckpt_unsloth_m` **110.8 s** before step 1 (61% of the arm): c1_before 53.6, load_weights 29.6, c1_after 26.0, eval0 11.8; unattributed 8.563; budget 1260.0
- prologue `e4b/reference_attn4_m` **115.2 s** before step 1 (10% of the arm): c1_before 57.2, c1_after 28.4, load_weights 23.6, eval0 18.8; unattributed 1.122; budget 1890.0
- prologue `e4b/fused_attn4_m_d2` **102.1 s** before step 1 (67% of the arm): c1_before 57.7, c1_after 27.8, load_weights 25.1, attn4 5.2; unattributed 1.395; budget 1260.0
- prologue `unsloth/ckpt_unsloth_m_d2` **114.5 s** before step 1 (67% of the arm): c1_before 51.9, c1_after 26.2, load_weights 25.7, eval0 5.6; unattributed 24.199; budget 1260.0
- prologue `e4b/fused_attn4_m_prof` **101.6 s** before step 1 (36% of the arm): c1_before 58.4, load_weights 23.9, c1_after 11.6, attn4 5.4; unattributed 1.387; budget 840.0
- prologue `unsloth/ckpt_unsloth_prof` **98.3 s** before step 1 (34% of the arm): c1_before 52.1, load_weights 25.0, c1_after 15.5, eval0 5.4; unattributed 8.923; budget 840.0
- draws (R1): `e4b/fused_attn4_m` STABLE (2.512/2.450 s, |Δ|/mean 2.5% vs 5%); `unsloth/ckpt_unsloth_m` STABLE (2.561/2.552 s, |Δ|/mean 0.4% vs 5%); `e4b/reference_attn4_m` SINGLE (single draw (no second draw registered)); `e4b/fused_attn4_m_prof` SINGLE (single draw (no second draw registered)); `unsloth/ckpt_unsloth_prof` SINGLE (single draw (no second draw registered))
- e4b internal parity (tp1's rule, informational): fused_attn4_m vs reference_attn4_m Δfinal 0.00057, median step |Δ| 0.00263 → **PASS** (band 0.05/0.05); ×20.02 faster per step, peak ×1.001
- **MATCHED POSITION: s/step ratio unsloth/e4b = 1.030** [1.016, 1.045 over 4 cross-draw ratios] (2.557 vs 2.481 s, medians over 2/2 draws; e4b faster per step); peak VRAM unsloth 24.27 vs e4b 27.19 GB (Δ -2.92); J/step unsloth 468.2 vs e4b 569.8 (×0.822); tok/s unsloth 495.0 vs e4b 602.5; unsloth regime: 4-bit expert stacks + bnb-4bit attention (Params4bit stacks 96, Linear4bit 384)
- quality reading at N=20 (unsloth): held-out e4b 0.8467 / unsloth 0.8513 (Δ 0.0046) → **COMPARABLE** (|Δ| ≤ 0.05 reads COMPARABLE); step-0 Δ -0.0325
- **NO MATCHED POSITION QUOTED (hf)** — not quoted: e4b STABLE / hf hf/hf_peft_m is NOT_RUN
- **NO MATCHED POSITION QUOTED (axolotl)** — not quoted: e4b STABLE / axolotl axolotl/ckpt_axolotl_m is NOT_RUN
- equivalence vs `e4b/fused_attn4_m` (I/H; EQUIVALENT iff both ≤ band = max(0.005, 3 × fused-vs-reference |Δ|) = train 0.0079 / held-out 0.0104; COMPARABLE ≤ 0.05; draw-noise floor 0.0028): `unsloth/ckpt_unsloth_m` **EQUIVALENT** (median step |Δ| 0.0053, |Δ held-out at N| 0.0046, step-0 0.0325 NEAR, |Δ loss at step 2| 0.0243, paired rows mean +0.0046 ± 0.0025 SE over 8, favouring arm 1 / anchor 7); `e4b/reference_attn4_m` **EQUIVALENT** (median step |Δ| 0.0026, |Δ held-out at N| 0.0035, step-0 0.0063 SAME-BYTES-CLASS, |Δ loss at step 2| 0.0178, paired rows mean +0.0035 ± 0.0023 SE over 8, favouring arm 2 / anchor 6); `e4b/fused_attn4_m_d2` **INSIDE-DRAW-NOISE** (median step |Δ| 0.0013, |Δ held-out at N| 0.0028, step-0 0.0000 SAME-BYTES-CLASS, |Δ loss at step 2| 0.0045, paired rows mean +0.0028 ± 0.0026 SE over 8, favouring arm 3 / anchor 5) — |delta| 0.0028 / 0.0013 are narrower than the draw-noise floor 0.0028: inside the draw noise, not a precision statement; `unsloth/ckpt_unsloth_m_d2` **EQUIVALENT** (median step |Δ| 0.0034, |Δ held-out at N| 0.0048, step-0 0.0325 NEAR, |Δ loss at step 2| 0.0106, paired rows mean +0.0048 ± 0.0040 SE over 8, favouring arm 3 / anchor 5); `hf/hf_peft_m` **—** — no OK receipt; `axolotl/ckpt_axolotl_m` **—** — no OK receipt; `e4b/fused_attn4_m_prof` **INSIDE-DRAW-NOISE** (median step |Δ| 0.0020, |Δ held-out at N| 0.0013, step-0 0.0000 SAME-BYTES-CLASS, |Δ loss at step 2| 0.0175, paired rows mean +0.0013 ± 0.0020 SE over 8, favouring arm 3 / anchor 5) — |delta| 0.0013 / 0.0020 are narrower than the draw-noise floor 0.0028: inside the draw noise, not a precision statement; `unsloth/ckpt_unsloth_prof` **EQUIVALENT** (median step |Δ| 0.0039, |Δ held-out at N| 0.0062, step-0 0.0325 NEAR, |Δ loss at step 2| 0.0093, paired rows mean +0.0062 ± 0.0038 SE over 8, favouring arm 4 / anchor 4)
- matched_init_sha (B, name-free, canonical slot order): anchor `f7832488926eda91`; `e4b/fused_attn4_m` same; `unsloth/ckpt_unsloth_m` same; `e4b/reference_attn4_m` same; `e4b/fused_attn4_m_d2` same; `unsloth/ckpt_unsloth_m_d2` same; `e4b/fused_attn4_m_prof` same; `unsloth/ckpt_unsloth_prof` same
- frozen base (R5), anchor `e4b/fused_attn4_m`: gate_up nf4/64 sha 8c3b3fd78e89 control detects, down nf4/64 sha 59569d810a1a control detects, q_proj nf4/64+dq sha 5c5e402ce51d control detects
- frozen base `unsloth/ckpt_unsloth_m`: gate_up **N-A** (regime differs: anchor nf4/64 vs nf4/64+dq) control detects, down **N-A** (regime differs: anchor nf4/64 vs nf4/64+dq) control detects, q_proj **SAME-BYTES** (nf4/64+dq) control detects
- frozen base `e4b/reference_attn4_m`: gate_up **SAME-BYTES** (nf4/64) control detects, down **SAME-BYTES** (nf4/64) control detects, q_proj **SAME-BYTES** (nf4/64+dq) control detects
- frozen base `e4b/fused_attn4_m_d2`: gate_up **SAME-BYTES** (nf4/64) control detects, down **SAME-BYTES** (nf4/64) control detects, q_proj **SAME-BYTES** (nf4/64+dq) control detects
- frozen base `unsloth/ckpt_unsloth_m_d2`: gate_up **N-A** (regime differs: anchor nf4/64 vs nf4/64+dq) control detects, down **N-A** (regime differs: anchor nf4/64 vs nf4/64+dq) control detects, q_proj **SAME-BYTES** (nf4/64+dq) control detects
- frozen base `e4b/fused_attn4_m_prof`: gate_up **SAME-BYTES** (nf4/64) control detects, down **SAME-BYTES** (nf4/64) control detects, q_proj **SAME-BYTES** (nf4/64+dq) control detects
- frozen base `unsloth/ckpt_unsloth_prof`: gate_up **N-A** (regime differs: anchor nf4/64 vs nf4/64+dq) control detects, down **N-A** (regime differs: anchor nf4/64 vs nf4/64+dq) control detects, q_proj **SAME-BYTES** (nf4/64+dq) control detects
- where Unsloth's step goes (`unsloth/ckpt_unsloth_prof`, descriptive): device busy fraction 0.306, device events/step 81780, CPU ops/step 515750, CPU self by family {'other': 0.3897, 'autograd': 0.2892, 'norm_act': 0.1302, 'matmul': 0.0725, 'memcpy': 0.0521, 'fused_kernel': 0.0438, 'routing': 0.0171, 'optimizer': 0.0053}

## Verdicts
| family | arm | VERDICT | validity | quality | note |
|---|---|---|---|---|---|
| qwen3 | e4b/fused_attn4_m | **VALID** | VALID | 0.0000 |  |
| qwen3 | unsloth/ckpt_unsloth_m | **VALID** | VALID | 0.0046 |  |
| qwen3 | e4b/reference_attn4_m | **VALID** | VALID | 0.0035 |  |
| qwen3 | e4b/fused_attn4_m_d2 | **VALID** | VALID | 0.0028 |  |
| qwen3 | unsloth/ckpt_unsloth_m_d2 | **VALID** | VALID | 0.0048 |  |
| qwen3 | hf/hf_peft_m | **NOT_RUN** | — | — | skipped by TC1_SKIP |
| qwen3 | axolotl/ckpt_axolotl_m | **NOT_RUN** | — | — | skipped by TC1_SKIP |
| qwen3 | e4b/fused_attn4_m_prof | **VALID** | VALID | 0.0013 |  |
| qwen3 | unsloth/ckpt_unsloth_prof | **VALID** | VALID | 0.0062 |  |

## Predictions P1–P10 (+ P1b) (TC1-PREREG.md + phase 2, scored mechanically)
| prediction | family | verdict | evidence |
|---|---|---|---|
| P1 | qwen3 | **FALSIFIED** | matched unsloth/e4b 1.030 vs [2.0, 5.0]; BELOW 1.5: the standing position is refuted and superseded |
| P1b | qwen3 | **UNTESTED** | ckpt_unsloth_t28 missing: no receipt |
| P2 | qwen3 | **HELD** | e4b |d1-d2|/mean 2.5% vs 5% -> STABLE; unsloth |d1-d2|/mean 0.4% vs 5% -> STABLE |
| P3 | qwen3 | **HELD** | e4b/reference_attn4_m EQUIVALENT (median step |Δ| 0.0026, |Δ held-out at N| 0.0035, band {'train': 0.007904999999999995, 'heldout': 0.010439999999999783}); unsloth/ckpt_unsloth_m EQUIVALENT (median step |Δ| 0.0053, |Δ held-out at N| 0.0046, band {'train': 0.007904999999999995, 'heldout': 0.010439999999999783}); draw-noise floor 0.0028 |
| P4 | qwen3 | **UNTESTED** | shipped missing / fused_m VALID: both must be VALID on the same box |
| P5 | qwen3 | **UNTESTED** | hf_peft_m NOT_RUN, hf_peft_m_mb1 missing: neither an OOM pair nor a trained arm |
| P6 | qwen3 | **UNTESTED** | axolotl NOT_RUN (NOT_RUN / HARNESS_ERROR / ALARM is not a reading) |
| P7 | qwen3 | **UNTESTED** | unsloth_best missing / unsloth_m VALID / e4b_m VALID: all three must be VALID |
| P8 | qwen3 | **FALSIFIED** | device_busy_fraction 0.306 (>= 0.5 predicted on the grouped_mm arm); device events/step 81780 |
| P9 | qwen3 | **HELD** | fused_m vs reference_m PASS (Δfinal 0.00057, median 0.00263) |
| P10 | qwen3 | **UNTESTED** | C1 bit-exact; expert slots ['N-A', 'N-A'] (regime differs: anchor nf4/64 vs nf4/64+dq); attention SAME-BYTES |
