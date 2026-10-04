# TC1 — Qwen3-30B-A3B on one RTX 5090: e4b vs Unsloth vs HF+PEFT vs axolotl at matched work, init and adapter precision (tc1)
Rule (tc1/TC1-PREREG.md): status per attempt in the vocabulary OK / REFUSED / OOM / INSTALL_FAILED / LOAD_FAULT / HARNESS_ERROR / ALARM / NOT_RUN; VALID/VOID per tp4's predicates plus the matched-set predicates (R3: matched init complete, fp32 adapters, step-0 held-out within 0.005 of e4b/reference_attn4_m); VERDICT exactly one of VALID / VOID / QUALITY_FAIL / OOM / UNSUPPORTED / HARNESS_ERROR / ALARM / NOT_RUN (QUALITY_FAIL = held-out |Δ| at N > 0.05 vs e4b/fused_attn4_m); positions = s/step ratios other/e4b from the medians over both draws (stability |d1-d2|/mean ≤ 5 % e4b / 10 % others, else UNSTABLE), quoted only when both arms are VALID; equivalence vs e4b/reference_attn4_m EQUIVALENT ≤ 0.02 / COMPARABLE ≤ 0.05 / else DIVERGENT; frozen base SAME-BYTES / DIFFERENT / N-A per slot; predictions P1–P10 (+ P1b) scored HELD / FALSIFIED / UNTESTED. VOID never enters a ratio or an equivalence reading. Nothing is licensed; no cross-box number is divided into these.
`versions.txt`
```
e4b 0.44.0 @8208f5edcdc2d76094a71fac1f8dfae743d7490f (GitHub main)
gnf4 0.36.0 @e7ec90aad70c3e84c151fdb8412871f0ea2bc60b (GitHub main)
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
e4b(t212) 0.44.0 @8208f5edcdc2d76094a71fac1f8dfae743d7490f
gnf4(t212) 0.36.0 @e7ec90aad70c3e84c151fdb8412871f0ea2bc60b
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
 "run_id": "tc1c-h100-6",
 "instance_id": "54119104",
 "gpu": "NVIDIA H100 NVL",
 "driver": "595.71.05",
 "cpu": "AMD EPYC 9534 64-Core Processor",
 "nproc": 224,
 "mem_total_kb": "1584958328",
 "cgroup_memory_max": "",
 "disk_root": "overlay         320G   77M  320G   1% /",
 "hostname": "ed89a31aea63",
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
| e4b | fused_attn4_m | **OK** | VALID | **VALID** | yes | matched:3407 (complete 12480/12480) / float32 | SAME-BYTES-CLASS (0.0000) | 20 | 3.956 | 379.7 | 27.189 | 827.5 | 2.0773→0.8330 | 1.9614→0.8460 | 0.0000 | patched 48 / kcalls 768 | 642514944 | 4-bit experts (e4b NF4) + NF4 attention |  |
| unsloth | ckpt_unsloth_m | **OK** | VALID | **VALID** | yes | matched:3407 (complete 12480/12480) / float32 | NEAR (0.0309) | 20 | 2.650 | 420.2 | 24.269 | 480.5 | 2.0577→0.8344 | 1.9305→0.8492 | 0.0031 | stacks 96 / fwd 384 / u8 48 | 642514944 | 4-bit expert stacks + bnb-4bit attention (Params4bit stacks 96, Linear4bit 384) |  |
| e4b | reference_attn4_m | **OK** | VALID | **VALID** | yes | matched:3407 (complete 12480/12480) / float32 | SAME-BYTES-CLASS (0.0063) | 20 | 50.536 | 29.7 | 27.147 | 2291.6 | 2.0461→0.8320 | 1.9551→0.8501 | 0.0041 | patched 0 / kcalls 0 | 642514944 | 4-bit experts (e4b NF4) + NF4 attention |  |
| e4b | fused_attn4_m_d2 | **OK** | VALID | **VALID** | yes | matched:3407 (complete 12480/12480) / float32 | SAME-BYTES-CLASS (0.0000) | 20 | 3.908 | 386.5 | 27.207 | 824.1 | 2.0773→0.8321 | 1.9614→0.8538 | 0.0077 | patched 48 / kcalls 768 | 642514944 | 4-bit experts (e4b NF4) + NF4 attention |  |
| unsloth | ckpt_unsloth_m_d2 | **OK** | VALID | **VALID** | yes | matched:3407 (complete 12480/12480) / float32 | NEAR (0.0309) | 20 | 2.573 | 549.2 | 24.269 | 447.0 | 2.0577→0.8317 | 1.9305→0.8511 | 0.0051 | stacks 96 / fwd 384 / u8 48 | 642514944 | 4-bit expert stacks + bnb-4bit attention (Params4bit stacks 96, Linear4bit 384) |  |
| hf | hf_peft_m | **NOT_RUN** | — | **NOT_RUN** | yes | — / — | — | 20 | — | — | — | — | —→— | —→— | — | peft mods — / params — / fwd — | — | — | skipped by TC1_SKIP |
| axolotl | ckpt_axolotl_m | **NOT_RUN** | — | **NOT_RUN** | yes | — / — | — | 20 | — | — | — | — | —→— | —→— | — | peft mods — / params — / fwd — | — | — | skipped by TC1_SKIP |
| e4b | fused_attn4_m_prof | **OK** | VALID | **VALID** | yes | matched:3407 (complete 12480/12480) / float32 | SAME-BYTES-CLASS (0.0000) | 20 | 3.983 | 333.6 | 27.189 | 561.3 | 2.0773→0.8318 | 1.9614→0.8505 | 0.0045 | patched 48 / kcalls 768 | 642514944 | 4-bit experts (e4b NF4) + NF4 attention |  |
| unsloth | ckpt_unsloth_prof | **OK** | VALID | **VALID** | yes | matched:3407 (complete 12480/12480) / float32 | NEAR (0.0309) | 20 | 2.859 | 485.3 | 24.269 | 255.4 | 2.0577→0.8327 | 1.9305→0.8505 | 0.0045 | stacks 96 / fwd 384 / u8 48 | 642514944 | 4-bit expert stacks + bnb-4bit attention (Params4bit stacks 96, Linear4bit 384) |  |
- prologue `e4b/fused_attn4_m` **100.1 s** before step 1 (55% of the arm): c1_before 57.5, c1_after 27.6, load_weights 22.1, attn4 5.2; unattributed 1.213; budget 1260.0
- prologue `unsloth/ckpt_unsloth_m` **109.0 s** before step 1 (60% of the arm): c1_before 52.6, load_weights 28.6, c1_after 26.0, eval0 12.0; unattributed 8.725; budget 1260.0
- prologue `e4b/reference_attn4_m` **114.8 s** before step 1 (10% of the arm): c1_before 56.8, c1_after 29.5, load_weights 22.3, eval0 18.8; unattributed 1.151; budget 1890.0
- prologue `e4b/fused_attn4_m_d2` **98.7 s** before step 1 (55% of the arm): c1_before 58.5, c1_after 27.6, load_weights 21.8, attn4 5.4; unattributed 1.282; budget 1260.0
- prologue `unsloth/ckpt_unsloth_m_d2` **98.8 s** before step 1 (64% of the arm): c1_before 52.2, c1_after 27.0, load_weights 25.3, eval0 5.4; unattributed 8.782; budget 1260.0
- prologue `e4b/fused_attn4_m_prof` **98.0 s** before step 1 (33% of the arm): c1_before 58.0, load_weights 21.7, c1_after 11.6, attn4 5.3; unattributed 1.123; budget 840.0
- prologue `unsloth/ckpt_unsloth_prof` **100.7 s** before step 1 (34% of the arm): c1_before 53.6, load_weights 25.7, c1_after 15.3, eval0 5.7; unattributed 8.661; budget 840.0
- draws (R1): `e4b/fused_attn4_m` STABLE (3.956/3.908 s, |Δ|/mean 1.2% vs 5%); `unsloth/ckpt_unsloth_m` STABLE (2.650/2.573 s, |Δ|/mean 3.0% vs 5%); `e4b/reference_attn4_m` SINGLE (single draw (no second draw registered)); `e4b/fused_attn4_m_prof` SINGLE (single draw (no second draw registered)); `unsloth/ckpt_unsloth_prof` SINGLE (single draw (no second draw registered))
- e4b internal parity (tp1's rule, informational): fused_attn4_m vs reference_attn4_m Δfinal 0.00105, median step |Δ| 0.00306 → **PASS** (band 0.05/0.05); ×12.77 faster per step, peak ×1.002
- **MATCHED POSITION: s/step ratio unsloth/e4b = 0.664** [0.650, 0.678 over 4 cross-draw ratios] (2.612 vs 3.932 s, medians over 2/2 draws; unsloth faster per step); peak VRAM unsloth 24.27 vs e4b 27.20 GB (Δ -2.93); J/step unsloth 463.8 vs e4b 825.8 (×0.562); tok/s unsloth 484.7 vs e4b 383.1; unsloth regime: 4-bit expert stacks + bnb-4bit attention (Params4bit stacks 96, Linear4bit 384)
- quality reading at N=20 (unsloth): held-out e4b 0.8460 / unsloth 0.8492 (Δ 0.0031) → **COMPARABLE** (|Δ| ≤ 0.05 reads COMPARABLE); step-0 Δ -0.0309
- **NO MATCHED POSITION QUOTED (hf)** — not quoted: e4b STABLE / hf hf/hf_peft_m is NOT_RUN
- **NO MATCHED POSITION QUOTED (axolotl)** — not quoted: e4b STABLE / axolotl axolotl/ckpt_axolotl_m is NOT_RUN
- equivalence vs `e4b/fused_attn4_m` (I/H; EQUIVALENT iff both ≤ band = max(0.005, 3 × fused-vs-reference |Δ|) = train 0.0092 / held-out 0.0123; COMPARABLE ≤ 0.05; draw-noise floor 0.0077): `unsloth/ckpt_unsloth_m` **INSIDE-DRAW-NOISE** (median step |Δ| 0.0068, |Δ held-out at N| 0.0031, step-0 0.0309 NEAR, |Δ loss at step 2| 0.0132, paired rows mean +0.0031 ± 0.0020 SE over 8, favouring arm 2 / anchor 5) — |delta| 0.0031 / 0.0068 are narrower than the draw-noise floor 0.0077: inside the draw noise, not a precision statement; `e4b/reference_attn4_m` **INSIDE-DRAW-NOISE** (median step |Δ| 0.0031, |Δ held-out at N| 0.0041, step-0 0.0063 SAME-BYTES-CLASS, |Δ loss at step 2| 0.0192, paired rows mean +0.0041 ± 0.0027 SE over 8, favouring arm 2 / anchor 6) — |delta| 0.0041 / 0.0031 are narrower than the draw-noise floor 0.0077: inside the draw noise, not a precision statement; `e4b/fused_attn4_m_d2` **INSIDE-DRAW-NOISE** (median step |Δ| 0.0009, |Δ held-out at N| 0.0077, step-0 0.0000 SAME-BYTES-CLASS, |Δ loss at step 2| 0.0058, paired rows mean +0.0077 ± 0.0028 SE over 8, favouring arm 1 / anchor 7) — |delta| 0.0077 / 0.0009 are narrower than the draw-noise floor 0.0077: inside the draw noise, not a precision statement; `unsloth/ckpt_unsloth_m_d2` **INSIDE-DRAW-NOISE** (median step |Δ| 0.0023, |Δ held-out at N| 0.0051, step-0 0.0309 NEAR, |Δ loss at step 2| 0.0020, paired rows mean +0.0051 ± 0.0038 SE over 8, favouring arm 3 / anchor 5) — |delta| 0.0051 / 0.0023 are narrower than the draw-noise floor 0.0077: inside the draw noise, not a precision statement; `hf/hf_peft_m` **—** — no OK receipt; `axolotl/ckpt_axolotl_m` **—** — no OK receipt; `e4b/fused_attn4_m_prof` **INSIDE-DRAW-NOISE** (median step |Δ| 0.0009, |Δ held-out at N| 0.0045, step-0 0.0000 SAME-BYTES-CLASS, |Δ loss at step 2| 0.0061, paired rows mean +0.0045 ± 0.0023 SE over 8, favouring arm 2 / anchor 6) — |delta| 0.0045 / 0.0009 are narrower than the draw-noise floor 0.0077: inside the draw noise, not a precision statement; `unsloth/ckpt_unsloth_prof` **INSIDE-DRAW-NOISE** (median step |Δ| 0.0024, |Δ held-out at N| 0.0045, step-0 0.0309 NEAR, |Δ loss at step 2| 0.0213, paired rows mean +0.0045 ± 0.0035 SE over 8, favouring arm 2 / anchor 6) — |delta| 0.0045 / 0.0024 are narrower than the draw-noise floor 0.0077: inside the draw noise, not a precision statement
- matched_init_sha (B, name-free, canonical slot order): anchor `f7832488926eda91`; `e4b/fused_attn4_m` same; `unsloth/ckpt_unsloth_m` same; `e4b/reference_attn4_m` same; `e4b/fused_attn4_m_d2` same; `unsloth/ckpt_unsloth_m_d2` same; `e4b/fused_attn4_m_prof` same; `unsloth/ckpt_unsloth_prof` same
- frozen base (R5), anchor `e4b/fused_attn4_m`: gate_up nf4/64 sha 8c3b3fd78e89 control detects, down nf4/64 sha 59569d810a1a control detects, q_proj nf4/64+dq sha 5c5e402ce51d control detects
- frozen base `unsloth/ckpt_unsloth_m`: gate_up **N-A** (regime differs: anchor nf4/64 vs nf4/64+dq) control detects, down **N-A** (regime differs: anchor nf4/64 vs nf4/64+dq) control detects, q_proj **SAME-BYTES** (nf4/64+dq) control detects
- frozen base `e4b/reference_attn4_m`: gate_up **SAME-BYTES** (nf4/64) control detects, down **SAME-BYTES** (nf4/64) control detects, q_proj **SAME-BYTES** (nf4/64+dq) control detects
- frozen base `e4b/fused_attn4_m_d2`: gate_up **SAME-BYTES** (nf4/64) control detects, down **SAME-BYTES** (nf4/64) control detects, q_proj **SAME-BYTES** (nf4/64+dq) control detects
- frozen base `unsloth/ckpt_unsloth_m_d2`: gate_up **N-A** (regime differs: anchor nf4/64 vs nf4/64+dq) control detects, down **N-A** (regime differs: anchor nf4/64 vs nf4/64+dq) control detects, q_proj **SAME-BYTES** (nf4/64+dq) control detects
- frozen base `e4b/fused_attn4_m_prof`: gate_up **SAME-BYTES** (nf4/64) control detects, down **SAME-BYTES** (nf4/64) control detects, q_proj **SAME-BYTES** (nf4/64+dq) control detects
- frozen base `unsloth/ckpt_unsloth_prof`: gate_up **N-A** (regime differs: anchor nf4/64 vs nf4/64+dq) control detects, down **N-A** (regime differs: anchor nf4/64 vs nf4/64+dq) control detects, q_proj **SAME-BYTES** (nf4/64+dq) control detects
- where Unsloth's step goes (`unsloth/ckpt_unsloth_prof`, descriptive): device busy fraction 0.308, device events/step 81780, CPU ops/step 515750, CPU self by family {'other': 0.3817, 'autograd': 0.2923, 'norm_act': 0.13, 'matmul': 0.0744, 'memcpy': 0.0519, 'fused_kernel': 0.0446, 'routing': 0.0178, 'optimizer': 0.0071}

## Verdicts
| family | arm | VERDICT | validity | quality | note |
|---|---|---|---|---|---|
| qwen3 | e4b/fused_attn4_m | **VALID** | VALID | 0.0000 |  |
| qwen3 | unsloth/ckpt_unsloth_m | **VALID** | VALID | 0.0031 |  |
| qwen3 | e4b/reference_attn4_m | **VALID** | VALID | 0.0041 |  |
| qwen3 | e4b/fused_attn4_m_d2 | **VALID** | VALID | 0.0077 |  |
| qwen3 | unsloth/ckpt_unsloth_m_d2 | **VALID** | VALID | 0.0051 |  |
| qwen3 | hf/hf_peft_m | **NOT_RUN** | — | — | skipped by TC1_SKIP |
| qwen3 | axolotl/ckpt_axolotl_m | **NOT_RUN** | — | — | skipped by TC1_SKIP |
| qwen3 | e4b/fused_attn4_m_prof | **VALID** | VALID | 0.0045 |  |
| qwen3 | unsloth/ckpt_unsloth_prof | **VALID** | VALID | 0.0045 |  |

## Predictions P1–P10 (+ P1b) (TC1-PREREG.md + phase 2, scored mechanically)
| prediction | family | verdict | evidence |
|---|---|---|---|
| P1 | qwen3 | **FALSIFIED** | matched unsloth/e4b 0.664 vs [2.0, 5.0]; BELOW 1.5: the standing position is refuted and superseded |
| P1b | qwen3 | **UNTESTED** | ckpt_unsloth_t28 missing: no receipt |
| P2 | qwen3 | **HELD** | e4b |d1-d2|/mean 1.2% vs 5% -> STABLE; unsloth |d1-d2|/mean 3.0% vs 5% -> STABLE |
| P3 | qwen3 | **HELD** | e4b/reference_attn4_m INSIDE-DRAW-NOISE (median step |Δ| 0.0031, |Δ held-out at N| 0.0041, band {'train': 0.00916500000000009, 'heldout': 0.012299999999999978}); unsloth/ckpt_unsloth_m INSIDE-DRAW-NOISE (median step |Δ| 0.0068, |Δ held-out at N| 0.0031, band {'train': 0.00916500000000009, 'heldout': 0.012299999999999978}); draw-noise floor 0.0077 |
| P4 | qwen3 | **UNTESTED** | shipped missing / fused_m VALID: both must be VALID on the same box |
| P5 | qwen3 | **UNTESTED** | hf_peft_m NOT_RUN, hf_peft_m_mb1 missing: neither an OOM pair nor a trained arm |
| P6 | qwen3 | **UNTESTED** | axolotl NOT_RUN (NOT_RUN / HARNESS_ERROR / ALARM is not a reading) |
| P7 | qwen3 | **UNTESTED** | unsloth_best missing / unsloth_m VALID / e4b_m VALID: all three must be VALID |
| P8 | qwen3 | **FALSIFIED** | device_busy_fraction 0.308 (>= 0.5 predicted on the grouped_mm arm); device events/step 81780 |
| P9 | qwen3 | **HELD** | fused_m vs reference_m PASS (Δfinal 0.00105, median 0.00306) |
| P10 | qwen3 | **UNTESTED** | C1 bit-exact; expert slots ['N-A', 'N-A'] (regime differs: anchor nf4/64 vs nf4/64+dq); attention SAME-BYTES |
