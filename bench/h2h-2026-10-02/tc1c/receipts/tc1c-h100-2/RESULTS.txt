# TC1 — Qwen3-30B-A3B on one RTX 5090: e4b vs Unsloth vs HF+PEFT vs axolotl at matched work, init and adapter precision (/root/tc1)
Rule (tc1/TC1-PREREG.md): status per attempt in the vocabulary OK / REFUSED / OOM / INSTALL_FAILED / LOAD_FAULT / HARNESS_ERROR / ALARM / NOT_RUN; VALID/VOID per tp4's predicates plus the matched-set predicates (R3: matched init complete, fp32 adapters, step-0 held-out within 0.005 of e4b/reference_attn4_m); VERDICT exactly one of VALID / VOID / QUALITY_FAIL / OOM / UNSUPPORTED / HARNESS_ERROR / ALARM / NOT_RUN (QUALITY_FAIL = held-out |Δ| at N > 0.05 vs e4b/fused_attn4_m); positions = s/step ratios other/e4b from the medians over both draws (stability |d1-d2|/mean ≤ 5 % e4b / 10 % others, else UNSTABLE), quoted only when both arms are VALID; equivalence vs e4b/reference_attn4_m EQUIVALENT ≤ 0.02 / COMPARABLE ≤ 0.05 / else DIVERGENT; frozen base SAME-BYTES / DIFFERENT / N-A per slot; predictions P1–P10 (+ P1b) scored HELD / FALSIFIED / UNTESTED. VOID never enters a ratio or an equivalence reading. Nothing is licensed; no cross-box number is divided into these.
`versions.txt`
```
e4b 0.38.1 @09b0f6a6586d755dccad3550f68ae1b2e57773ad (GitHub main)
gnf4 0.34.0 @846b512b905468c08f5748943d08769b572affa2 (GitHub main)
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
e4b(t212) 0.38.1 @09b0f6a6586d755dccad3550f68ae1b2e57773ad
gnf4(t212) 0.34.0 @846b512b905468c08f5748943d08769b572affa2
torch(e4b-t212) 2.12.1+cu130
axolotl 0.20.0
torch(axolotl) 2.14.0+cu130
transformers(axolotl) 5.17.0
peft(axolotl) 0.21.0
bitsandbytes(axolotl) 0.50.2
python(axolotl) 3.12.15
```
`box.json`
```
{
 "box": "A",
 "run_id": "tc1c-h100-2",
 "instance_id": "53802633",
 "gpu": "NVIDIA H100 NVL",
 "driver": "595.71.05",
 "cpu": "AMD EPYC 9534 64-Core Processor",
 "nproc": 224,
 "mem_total_kb": "1584958328",
 "cgroup_memory_max": "",
 "disk_root": "overlay         320G   77M  320G   1% /",
 "hostname": "65fc9f51f611",
 "registered_gpu_class": "H100 NVL",
 "prereg": "tc1/TC1-PREREG.md"
}
```

### Qwen3-30B-A3B (`qwen3`, registered n_layers 48, attention census 192)
- model `Qwen/Qwen3-30B-A3B` @ `ad44e777bcd1`; tokens sha `bfc742f67e37`; N=20; fixture template alpaca seq 2048 micro-batch 2 × accum 4 lr 0.0002 r 16 α 16 optimizer adamw_8bit(lr=0.0002, weight_decay=0.001) schedule=linear warmup_steps=5 autocast False; e4b trainable 642514944; box_class H100 NVL gpu NVIDIA H100 NVL
| framework | arm | status | validity | **VERDICT** | matched | init / dtype | step-0 class | N | s/step med(11+) | tok/s | peak GB | J/step | train first→last | held-out 0→final | quality Δ | engagement | trainable | regime | reason / why |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| e4b | fused_attn4_m | **OK** | VALID | **VALID** | yes | matched:3407 (complete 12480/12480) / float32 | SAME-BYTES-CLASS (0.0000) | 20 | 4.130 | 363.0 | 27.820 | 748.7 | 2.0717→0.8361 | 1.9468→0.8476 | 0.0000 | patched 48 / kcalls 768 | 642514944 | 4-bit experts (e4b NF4) + NF4 attention |  |
| unsloth | ckpt_unsloth_m | **OK** | VALID | **VALID** | yes | matched:3407 (complete 12480/12480) / float32 | NEAR (0.0163) | 20 | 2.539 | 428.1 | 24.269 | 483.6 | 2.0577→0.8337 | 1.9305→0.8517 | 0.0041 | stacks 96 / fwd 384 / u8 48 | 642514944 | 4-bit expert stacks + bnb-4bit attention (Params4bit stacks 96, Linear4bit 384) |  |
| e4b | reference_attn4_m | **OK** | VALID | **VALID** | yes | matched:3407 (complete 12480/12480) / float32 | SAME-BYTES-CLASS (0.0083) | 20 | 51.940 | 28.7 | 27.147 | 2395.0 | 2.0461→0.8320 | 1.9551→0.8501 | 0.0026 | patched 0 / kcalls 0 | 642514944 | 4-bit experts (e4b NF4) + NF4 attention |  |
| e4b | fused_attn4_m_d2 | **OK** | VALID | **VALID** | yes | matched:3407 (complete 12480/12480) / float32 | SAME-BYTES-CLASS (0.0000) | 20 | 4.065 | 377.8 | 27.899 | 736.4 | 2.0717→0.8322 | 1.9468→0.8490 | 0.0015 | patched 48 / kcalls 768 | 642514944 | 4-bit experts (e4b NF4) + NF4 attention |  |
| unsloth | ckpt_unsloth_m_d2 | **OK** | VALID | **VALID** | yes | matched:3407 (complete 12480/12480) / float32 | NEAR (0.0163) | 20 | 2.553 | 554.1 | 24.269 | 450.0 | 2.0577→0.8336 | 1.9305→0.8498 | 0.0023 | stacks 96 / fwd 384 / u8 48 | 642514944 | 4-bit expert stacks + bnb-4bit attention (Params4bit stacks 96, Linear4bit 384) |  |
| hf | hf_peft_m | **ALARM** | — | **ALARM** | yes | — / — | — | 20 | — | — | — | — | —→— | —→— | — | peft mods — / params — / fwd — | — | — | arm alarm 1800 s (SIGALRM; the process could not write its own stub) |
| axolotl | ckpt_axolotl_m | **HARNESS_ERROR** | — | **HARNESS_ERROR** | yes | — / — | — | — | — | — | — | — | —→— | —→— | — | peft mods — / params — / fwd — | — | — | rc=1 and no receipt (the process died before its first write; attempts [1]) |
| e4b | fused_attn4_m_prof | **OK** | VALID | **VALID** | yes | matched:3407 (complete 12480/12480) / float32 | SAME-BYTES-CLASS (0.0000) | 20 | 4.306 | 347.6 | 27.845 | 439.9 | 2.0717→0.8292 | 1.9468→0.8500 | 0.0024 | patched 48 / kcalls 768 | 642514944 | 4-bit experts (e4b NF4) + NF4 attention |  |
| unsloth | ckpt_unsloth_prof | **OK** | VALID | **VALID** | yes | matched:3407 (complete 12480/12480) / float32 | NEAR (0.0163) | 20 | 2.828 | 491.9 | 24.269 | 258.4 | 2.0577→0.8308 | 1.9305→0.8510 | 0.0034 | stacks 96 / fwd 384 / u8 48 | 642514944 | 4-bit expert stacks + bnb-4bit attention (Params4bit stacks 96, Linear4bit 384) |  |
- prologue `e4b/fused_attn4_m` **103.7 s** before step 1 (55% of the arm): c1_before 52.8, c1_after 25.6, load_weights 22.6, eval0 12.6; unattributed 1.183; budget 1260.0
- prologue `unsloth/ckpt_unsloth_m` **107.1 s** before step 1 (60% of the arm): c1_before 49.8, load_weights 28.0, c1_after 24.9, eval0 12.6; unattributed 8.839; budget 1260.0
- prologue `e4b/reference_attn4_m` **109.7 s** before step 1 (9% of the arm): c1_before 52.5, c1_after 28.6, load_weights 22.4, eval0 19.7; unattributed 1.158; budget 1890.0
- prologue `e4b/fused_attn4_m_d2` **98.1 s** before step 1 (54% of the arm): c1_before 57.0, c1_after 27.9, load_weights 22.9, attn4 5.1; unattributed 1.159; budget 1260.0
- prologue `unsloth/ckpt_unsloth_m_d2` **100.8 s** before step 1 (64% of the arm): c1_before 52.0, load_weights 27.7, c1_after 25.7, eval0 5.4; unattributed 8.569; budget 1260.0
- prologue `e4b/fused_attn4_m_prof` **97.0 s** before step 1 (29% of the arm): c1_before 53.5, load_weights 24.0, c1_after 11.6, attn4 5.1; unattributed 1.094; budget 840.0
- prologue `unsloth/ckpt_unsloth_prof` **99.1 s** before step 1 (34% of the arm): c1_before 50.0, load_weights 27.8, c1_after 15.3, eval0 5.4; unattributed 8.807; budget 840.0
- draws (R1): `e4b/fused_attn4_m` STABLE (4.130/4.065 s, |Δ|/mean 1.6% vs 5%); `unsloth/ckpt_unsloth_m` STABLE (2.539/2.553 s, |Δ|/mean 0.5% vs 5%); `e4b/reference_attn4_m` SINGLE (single draw (no second draw registered)); `e4b/fused_attn4_m_prof` SINGLE (single draw (no second draw registered)); `unsloth/ckpt_unsloth_prof` SINGLE (single draw (no second draw registered))
- e4b internal parity (tp1's rule, informational): fused_attn4_m vs reference_attn4_m Δfinal 0.00414, median step |Δ| 0.00218 → **PASS** (band 0.05/0.05); ×12.58 faster per step, peak ×1.025
- **MATCHED POSITION: s/step ratio unsloth/e4b = 0.621** [0.615, 0.628 over 4 cross-draw ratios] (2.546 vs 4.097 s, medians over 2/2 draws; unsloth faster per step); peak VRAM unsloth 24.27 vs e4b 27.86 GB (Δ -3.59); J/step unsloth 466.8 vs e4b 742.5 (×0.629); tok/s unsloth 491.1 vs e4b 370.4; unsloth regime: 4-bit expert stacks + bnb-4bit attention (Params4bit stacks 96, Linear4bit 384)
- quality reading at N=20 (unsloth): held-out e4b 0.8476 / unsloth 0.8517 (Δ 0.0041) → **COMPARABLE** (|Δ| ≤ 0.05 reads COMPARABLE); step-0 Δ -0.0163
- **NO MATCHED POSITION QUOTED (hf)** — not quoted: e4b STABLE / hf hf/hf_peft_m is ALARM
- **NO MATCHED POSITION QUOTED (axolotl)** — not quoted: e4b STABLE / axolotl axolotl/ckpt_axolotl_m is HARNESS_ERROR
- equivalence vs `e4b/fused_attn4_m` (I/H; EQUIVALENT iff both ≤ band = max(0.005, 3 × fused-vs-reference |Δ|) = train 0.0066 / held-out 0.0078; COMPARABLE ≤ 0.05; draw-noise floor 0.0019): `unsloth/ckpt_unsloth_m` **EQUIVALENT** (median step |Δ| 0.0038, |Δ held-out at N| 0.0041, step-0 0.0163 NEAR, |Δ loss at step 2| 0.0230, paired rows mean +0.0042 ± 0.0033 SE over 8, favouring arm 3 / anchor 5); `e4b/reference_attn4_m` **EQUIVALENT** (median step |Δ| 0.0022, |Δ held-out at N| 0.0026, step-0 0.0083 SAME-BYTES-CLASS, |Δ loss at step 2| 0.0161, paired rows mean +0.0026 ± 0.0025 SE over 8, favouring arm 3 / anchor 5); `e4b/fused_attn4_m_d2` **INSIDE-DRAW-NOISE** (median step |Δ| 0.0017, |Δ held-out at N| 0.0015, step-0 0.0000 SAME-BYTES-CLASS, |Δ loss at step 2| 0.0054, paired rows mean +0.0015 ± 0.0016 SE over 8, favouring arm 3 / anchor 5) — |delta| 0.0015 / 0.0017 are narrower than the draw-noise floor 0.0019: inside the draw noise, not a precision statement; `unsloth/ckpt_unsloth_m_d2` **EQUIVALENT** (median step |Δ| 0.0051, |Δ held-out at N| 0.0023, step-0 0.0163 NEAR, |Δ loss at step 2| 0.0326, paired rows mean +0.0023 ± 0.0031 SE over 8, favouring arm 4 / anchor 4); `hf/hf_peft_m` **—** — no OK receipt; `e4b/fused_attn4_m_prof` **EQUIVALENT** (median step |Δ| 0.0023, |Δ held-out at N| 0.0024, step-0 0.0000 SAME-BYTES-CLASS, |Δ loss at step 2| 0.0146, paired rows mean +0.0024 ± 0.0015 SE over 8, favouring arm 1 / anchor 7); `unsloth/ckpt_unsloth_prof` **EQUIVALENT** (median step |Δ| 0.0054, |Δ held-out at N| 0.0034, step-0 0.0163 NEAR, |Δ loss at step 2| 0.0123, paired rows mean +0.0034 ± 0.0023 SE over 8, favouring arm 4 / anchor 4)
- matched_init_sha (B, name-free, canonical slot order): anchor `f7832488926eda91`; `e4b/fused_attn4_m` same; `unsloth/ckpt_unsloth_m` same; `e4b/reference_attn4_m` same; `e4b/fused_attn4_m_d2` same; `unsloth/ckpt_unsloth_m_d2` same; `e4b/fused_attn4_m_prof` same; `unsloth/ckpt_unsloth_prof` same
- frozen base (R5), anchor `e4b/fused_attn4_m`: gate_up nf4/64 sha 8c3b3fd78e89 control detects, down nf4/64 sha 59569d810a1a control detects, q_proj nf4/64+dq sha 5c5e402ce51d control detects
- frozen base `unsloth/ckpt_unsloth_m`: gate_up **N-A** (regime differs: anchor nf4/64 vs nf4/64+dq) control detects, down **N-A** (regime differs: anchor nf4/64 vs nf4/64+dq) control detects, q_proj **SAME-BYTES** (nf4/64+dq) control detects
- frozen base `e4b/reference_attn4_m`: gate_up **SAME-BYTES** (nf4/64) control detects, down **SAME-BYTES** (nf4/64) control detects, q_proj **SAME-BYTES** (nf4/64+dq) control detects
- frozen base `e4b/fused_attn4_m_d2`: gate_up **SAME-BYTES** (nf4/64) control detects, down **SAME-BYTES** (nf4/64) control detects, q_proj **SAME-BYTES** (nf4/64+dq) control detects
- frozen base `unsloth/ckpt_unsloth_m_d2`: gate_up **N-A** (regime differs: anchor nf4/64 vs nf4/64+dq) control detects, down **N-A** (regime differs: anchor nf4/64 vs nf4/64+dq) control detects, q_proj **SAME-BYTES** (nf4/64+dq) control detects
- frozen base `e4b/fused_attn4_m_prof`: gate_up **SAME-BYTES** (nf4/64) control detects, down **SAME-BYTES** (nf4/64) control detects, q_proj **SAME-BYTES** (nf4/64+dq) control detects
- frozen base `unsloth/ckpt_unsloth_prof`: gate_up **N-A** (regime differs: anchor nf4/64 vs nf4/64+dq) control detects, down **N-A** (regime differs: anchor nf4/64 vs nf4/64+dq) control detects, q_proj **SAME-BYTES** (nf4/64+dq) control detects
- where Unsloth's step goes (`unsloth/ckpt_unsloth_prof`, descriptive): device busy fraction 0.308, device events/step 81780, CPU ops/step 515683, CPU self by family {'other': 0.3824, 'autograd': 0.2934, 'norm_act': 0.1197, 'matmul': 0.0855, 'memcpy': 0.0508, 'fused_kernel': 0.0447, 'routing': 0.0178, 'optimizer': 0.0058}

## Verdicts
| family | arm | VERDICT | validity | quality | note |
|---|---|---|---|---|---|
| qwen3 | e4b/fused_attn4_m | **VALID** | VALID | 0.0000 |  |
| qwen3 | unsloth/ckpt_unsloth_m | **VALID** | VALID | 0.0041 |  |
| qwen3 | e4b/reference_attn4_m | **VALID** | VALID | 0.0026 |  |
| qwen3 | e4b/fused_attn4_m_d2 | **VALID** | VALID | 0.0015 |  |
| qwen3 | unsloth/ckpt_unsloth_m_d2 | **VALID** | VALID | 0.0023 |  |
| qwen3 | hf/hf_peft_m | **ALARM** | — | — | arm alarm 1800 s (SIGALRM; the process could not write its own stub) |
| qwen3 | axolotl/ckpt_axolotl_m | **HARNESS_ERROR** | — | — | rc=1 and no receipt (the process died before its first write; attempts [1]) |
| qwen3 | e4b/fused_attn4_m_prof | **VALID** | VALID | 0.0024 |  |
| qwen3 | unsloth/ckpt_unsloth_prof | **VALID** | VALID | 0.0034 |  |

## Predictions P1–P10 (+ P1b) (TC1-PREREG.md + phase 2, scored mechanically)
| prediction | family | verdict | evidence |
|---|---|---|---|
| P1 | qwen3 | **FALSIFIED** | matched unsloth/e4b 0.621 vs [2.0, 5.0]; BELOW 1.5: the standing position is refuted and superseded |
| P1b | qwen3 | **UNTESTED** | ckpt_unsloth_t28 missing: no receipt |
| P2 | qwen3 | **HELD** | e4b |d1-d2|/mean 1.6% vs 5% -> STABLE; unsloth |d1-d2|/mean 0.5% vs 5% -> STABLE |
| P3 | qwen3 | **HELD** | e4b/reference_attn4_m EQUIVALENT (median step |Δ| 0.0022, |Δ held-out at N| 0.0026, band {'train': 0.0065549999999999775, 'heldout': 0.007799999999999807}); unsloth/ckpt_unsloth_m EQUIVALENT (median step |Δ| 0.0038, |Δ held-out at N| 0.0041, band {'train': 0.0065549999999999775, 'heldout': 0.007799999999999807}); draw-noise floor 0.0019 |
| P4 | qwen3 | **UNTESTED** | shipped missing / fused_m VALID: both must be VALID on the same box |
| P5 | qwen3 | **UNTESTED** | hf_peft_m ALARM, hf_peft_m_mb1 missing: neither an OOM pair nor a trained arm |
| P6 | qwen3 | **UNTESTED** | axolotl HARNESS_ERROR (NOT_RUN / HARNESS_ERROR / ALARM is not a reading) |
| P7 | qwen3 | **UNTESTED** | unsloth_best missing / unsloth_m VALID / e4b_m VALID: all three must be VALID |
| P8 | qwen3 | **FALSIFIED** | device_busy_fraction 0.308 (>= 0.5 predicted on the grouped_mm arm); device events/step 81780 |
| P9 | qwen3 | **HELD** | fused_m vs reference_m PASS (Δfinal 0.00414, median 0.00218) |
| P10 | qwen3 | **UNTESTED** | C1 bit-exact; expert slots ['N-A', 'N-A'] (regime differs: anchor nf4/64 vs nf4/64+dq); attention SAME-BYTES |
