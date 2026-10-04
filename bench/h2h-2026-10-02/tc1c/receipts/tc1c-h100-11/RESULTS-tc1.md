# TC1 — Qwen3-30B-A3B on one RTX 5090: e4b vs Unsloth vs HF+PEFT vs axolotl at matched work, init and adapter precision (tc1)
Rule (tc1/TC1-PREREG.md): status per attempt in the vocabulary OK / REFUSED / OOM / INSTALL_FAILED / LOAD_FAULT / HARNESS_ERROR / ALARM / NOT_RUN; VALID/VOID per tp4's predicates plus the matched-set predicates (R3: matched init complete, fp32 adapters, step-0 held-out within 0.005 of e4b/reference_attn4_m); VERDICT exactly one of VALID / VOID / QUALITY_FAIL / OOM / UNSUPPORTED / HARNESS_ERROR / ALARM / NOT_RUN (QUALITY_FAIL = held-out |Δ| at N > 0.05 vs e4b/fused_attn4_m); positions = s/step ratios other/e4b from the medians over both draws (stability |d1-d2|/mean ≤ 5 % e4b / 10 % others, else UNSTABLE), quoted only when both arms are VALID; equivalence vs e4b/reference_attn4_m EQUIVALENT ≤ 0.02 / COMPARABLE ≤ 0.05 / else DIVERGENT; frozen base SAME-BYTES / DIFFERENT / N-A per slot; predictions P1–P10 (+ P1b) scored HELD / FALSIFIED / UNTESTED. VOID never enters a ratio or an equivalence reading. Nothing is licensed; no cross-box number is divided into these.
`versions.txt`
```
e4b 0.45.0 @317162120b9f1e95ccc91f77e2ebf15423b47a9a (GitHub main)
gnf4 0.37.0 @71185d6b52f6fc4d09a881d04e0be2ac196d15fe (GitHub main)
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
e4b(t212) 0.45.0 @317162120b9f1e95ccc91f77e2ebf15423b47a9a
gnf4(t212) 0.37.0 @71185d6b52f6fc4d09a881d04e0be2ac196d15fe
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
 "run_id": "tc1c-h100-11",
 "instance_id": "54150922",
 "gpu": "NVIDIA H100 NVL",
 "driver": "595.71.05",
 "cpu": "AMD EPYC 9534 64-Core Processor",
 "nproc": 224,
 "mem_total_kb": "1584958328",
 "cgroup_memory_max": "",
 "disk_root": "overlay         320G   77M  320G   1% /",
 "hostname": "81e8a1a0ade4",
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
| e4b | fused_attn4_m | **OK** | VALID | **VALID** | yes | matched:3407 (complete 12480/12480) / float32 | SAME-BYTES-CLASS (0.0000) | 20 | 2.540 | 606.7 | 27.205 | 512.2 | 2.0773→0.8327 | 1.9614→0.8484 | 0.0000 | patched 48 / kcalls 768 | 642514944 | 4-bit experts (e4b NF4) + NF4 attention |  |
| unsloth | ckpt_unsloth_m | **OK** | VALID | **VALID** | yes | matched:3407 (complete 12480/12480) / float32 | NEAR (0.0325) | 20 | 2.537 | 436.2 | 24.269 | 436.2 | 2.0319→0.8374 | 1.9289→0.8545 | 0.0061 | stacks 96 / fwd 384 / u8 48 | 642514944 | 4-bit expert stacks + bnb-4bit attention (Params4bit stacks 96, Linear4bit 384) |  |
| e4b | reference_attn4_m | **OK** | VALID | **VALID** | yes | matched:3407 (complete 12480/12480) / float32 | SAME-BYTES-CLASS (0.0063) | 20 | 51.623 | 29.2 | 27.147 | 2107.5 | 2.0461→0.8320 | 1.9551→0.8501 | 0.0018 | patched 0 / kcalls 0 | 642514944 | 4-bit experts (e4b NF4) + NF4 attention |  |
| e4b | fused_attn4_m_d2 | **OK** | VALID | **VALID** | yes | matched:3407 (complete 12480/12480) / float32 | SAME-BYTES-CLASS (0.0000) | 20 | 2.414 | 622.2 | 27.187 | 501.3 | 2.0773→0.8312 | 1.9614→0.8474 | -0.0010 | patched 48 / kcalls 768 | 642514944 | 4-bit experts (e4b NF4) + NF4 attention |  |
| unsloth | ckpt_unsloth_m_d2 | **OK** | VALID | **VALID** | yes | matched:3407 (complete 12480/12480) / float32 | NEAR (0.0325) | 20 | 2.552 | 556.6 | 24.269 | 405.7 | 2.0319→0.8363 | 1.9289→0.8510 | 0.0027 | stacks 96 / fwd 384 / u8 48 | 642514944 | 4-bit expert stacks + bnb-4bit attention (Params4bit stacks 96, Linear4bit 384) |  |
| hf | hf_peft_m | **NOT_RUN** | — | **NOT_RUN** | yes | — / — | — | 20 | — | — | — | — | —→— | —→— | — | peft mods — / params — / fwd — | — | — | skipped by TC1_SKIP |
| axolotl | ckpt_axolotl_m | **NOT_RUN** | — | **NOT_RUN** | yes | — / — | — | 20 | — | — | — | — | —→— | —→— | — | peft mods — / params — / fwd — | — | — | skipped by TC1_SKIP |
| e4b | fused_attn4_m_prof | **OK** | VALID | **VALID** | yes | matched:3407 (complete 12480/12480) / float32 | SAME-BYTES-CLASS (0.0000) | 20 | 2.680 | 440.6 | 27.187 | 319.8 | 2.0773→0.8348 | 1.9614→0.8529 | 0.0046 | patched 48 / kcalls 768 | 642514944 | 4-bit experts (e4b NF4) + NF4 attention |  |
| unsloth | ckpt_unsloth_prof | **OK** | VALID | **VALID** | yes | matched:3407 (complete 12480/12480) / float32 | NEAR (0.0325) | 20 | 2.809 | 492.2 | 24.269 | 229.5 | 2.0319→0.8350 | 1.9289→0.8493 | 0.0009 | stacks 96 / fwd 384 / u8 48 | 642514944 | 4-bit expert stacks + bnb-4bit attention (Params4bit stacks 96, Linear4bit 384) |  |
- prologue `e4b/fused_attn4_m` **102.0 s** before step 1 (67% of the arm): c1_before 57.5, c1_after 27.3, load_weights 25.3, attn4 5.2; unattributed 1.129; budget 1260.0
- prologue `unsloth/ckpt_unsloth_m` **112.5 s** before step 1 (61% of the arm): c1_before 52.3, load_weights 32.0, c1_after 26.0, eval0 12.4; unattributed 8.591; budget 1260.0
- prologue `e4b/reference_attn4_m` **119.0 s** before step 1 (10% of the arm): c1_before 58.2, c1_after 28.2, load_weights 25.4, eval0 19.4; unattributed 1.228; budget 1890.0
- prologue `e4b/fused_attn4_m_d2` **101.0 s** before step 1 (67% of the arm): c1_before 57.8, c1_after 25.4, load_weights 25.4, attn4 5.3; unattributed 1.237; budget 1260.0
- prologue `unsloth/ckpt_unsloth_m_d2` **101.2 s** before step 1 (64% of the arm): c1_before 50.8, load_weights 29.2, c1_after 23.6, eval0 5.4; unattributed 8.824; budget 1260.0
- prologue `e4b/fused_attn4_m_prof` **95.1 s** before step 1 (34% of the arm): c1_before 52.1, load_weights 25.2, c1_after 11.7, attn4 5.2; unattributed 1.23; budget 840.0
- prologue `unsloth/ckpt_unsloth_prof` **99.5 s** before step 1 (35% of the arm): c1_before 49.5, load_weights 29.1, c1_after 14.5, eval0 5.3; unattributed 8.615; budget 840.0
- draws (R1): `e4b/fused_attn4_m` UNSTABLE (2.540/2.414 s, |Δ|/mean 5.1% vs 5%); `unsloth/ckpt_unsloth_m` STABLE (2.537/2.552 s, |Δ|/mean 0.6% vs 5%); `e4b/reference_attn4_m` SINGLE (single draw (no second draw registered)); `e4b/fused_attn4_m_prof` SINGLE (single draw (no second draw registered)); `unsloth/ckpt_unsloth_prof` SINGLE (single draw (no second draw registered))
- e4b internal parity (tp1's rule, informational): fused_attn4_m vs reference_attn4_m Δfinal 0.00071, median step |Δ| 0.00180 → **PASS** (band 0.05/0.05); ×20.33 faster per step, peak ×1.002
- **NO MATCHED POSITION QUOTED (unsloth)** — not quoted: e4b draws 2.540 / 2.414 s/step differ by 5.1% > 5% (UNSTABLE: reported, not quoted) / unsloth STABLE
- **NO MATCHED POSITION QUOTED (hf)** — not quoted: e4b draws 2.540 / 2.414 s/step differ by 5.1% > 5% (UNSTABLE: reported, not quoted) / hf hf/hf_peft_m is NOT_RUN
- **NO MATCHED POSITION QUOTED (axolotl)** — not quoted: e4b draws 2.540 / 2.414 s/step differ by 5.1% > 5% (UNSTABLE: reported, not quoted) / axolotl axolotl/ckpt_axolotl_m is NOT_RUN
- equivalence vs `e4b/fused_attn4_m` (I/H; EQUIVALENT iff both ≤ band = max(0.005, 3 × fused-vs-reference |Δ|) = train 0.0054 / held-out 0.0054; COMPARABLE ≤ 0.05; draw-noise floor 0.0034): `unsloth/ckpt_unsloth_m` **COMPARABLE** (median step |Δ| 0.0031, |Δ held-out at N| 0.0061, step-0 0.0325 NEAR, |Δ loss at step 2| 0.0247, paired rows mean +0.0061 ± 0.0032 SE over 8, favouring arm 3 / anchor 5); `e4b/reference_attn4_m` **INSIDE-DRAW-NOISE** (median step |Δ| 0.0018, |Δ held-out at N| 0.0018, step-0 0.0063 SAME-BYTES-CLASS, |Δ loss at step 2| 0.0171, paired rows mean +0.0018 ± 0.0028 SE over 8, favouring arm 4 / anchor 4) — |delta| 0.0018 / 0.0018 are narrower than the draw-noise floor 0.0034: inside the draw noise, not a precision statement; `e4b/fused_attn4_m_d2` **INSIDE-DRAW-NOISE** (median step |Δ| 0.0019, |Δ held-out at N| 0.0010, step-0 0.0000 SAME-BYTES-CLASS, |Δ loss at step 2| 0.0105, paired rows mean -0.0010 ± 0.0037 SE over 8, favouring arm 3 / anchor 5) — |delta| 0.0010 / 0.0019 are narrower than the draw-noise floor 0.0034: inside the draw noise, not a precision statement; `unsloth/ckpt_unsloth_m_d2` **EQUIVALENT** (median step |Δ| 0.0036, |Δ held-out at N| 0.0027, step-0 0.0325 NEAR, |Δ loss at step 2| 0.0250, paired rows mean +0.0027 ± 0.0049 SE over 8, favouring arm 4 / anchor 4); `hf/hf_peft_m` **—** — no OK receipt; `axolotl/ckpt_axolotl_m` **—** — no OK receipt; `e4b/fused_attn4_m_prof` **EQUIVALENT** (median step |Δ| 0.0012, |Δ held-out at N| 0.0046, step-0 0.0000 SAME-BYTES-CLASS, |Δ loss at step 2| 0.0034, paired rows mean +0.0046 ± 0.0023 SE over 8, favouring arm 2 / anchor 6); `unsloth/ckpt_unsloth_prof` **INSIDE-DRAW-NOISE** (median step |Δ| 0.0023, |Δ held-out at N| 0.0009, step-0 0.0325 NEAR, |Δ loss at step 2| 0.0081, paired rows mean +0.0009 ± 0.0041 SE over 8, favouring arm 3 / anchor 5) — |delta| 0.0009 / 0.0023 are narrower than the draw-noise floor 0.0034: inside the draw noise, not a precision statement
- matched_init_sha (B, name-free, canonical slot order): anchor `f7832488926eda91`; `e4b/fused_attn4_m` same; `unsloth/ckpt_unsloth_m` same; `e4b/reference_attn4_m` same; `e4b/fused_attn4_m_d2` same; `unsloth/ckpt_unsloth_m_d2` same; `e4b/fused_attn4_m_prof` same; `unsloth/ckpt_unsloth_prof` same
- frozen base (R5), anchor `e4b/fused_attn4_m`: gate_up nf4/64 sha 8c3b3fd78e89 control detects, down nf4/64 sha 59569d810a1a control detects, q_proj nf4/64+dq sha 5c5e402ce51d control detects
- frozen base `unsloth/ckpt_unsloth_m`: gate_up **N-A** (regime differs: anchor nf4/64 vs nf4/64+dq) control detects, down **N-A** (regime differs: anchor nf4/64 vs nf4/64+dq) control detects, q_proj **SAME-BYTES** (nf4/64+dq) control detects
- frozen base `e4b/reference_attn4_m`: gate_up **SAME-BYTES** (nf4/64) control detects, down **SAME-BYTES** (nf4/64) control detects, q_proj **SAME-BYTES** (nf4/64+dq) control detects
- frozen base `e4b/fused_attn4_m_d2`: gate_up **SAME-BYTES** (nf4/64) control detects, down **SAME-BYTES** (nf4/64) control detects, q_proj **SAME-BYTES** (nf4/64+dq) control detects
- frozen base `unsloth/ckpt_unsloth_m_d2`: gate_up **N-A** (regime differs: anchor nf4/64 vs nf4/64+dq) control detects, down **N-A** (regime differs: anchor nf4/64 vs nf4/64+dq) control detects, q_proj **SAME-BYTES** (nf4/64+dq) control detects
- frozen base `e4b/fused_attn4_m_prof`: gate_up **SAME-BYTES** (nf4/64) control detects, down **SAME-BYTES** (nf4/64) control detects, q_proj **SAME-BYTES** (nf4/64+dq) control detects
- frozen base `unsloth/ckpt_unsloth_prof`: gate_up **N-A** (regime differs: anchor nf4/64 vs nf4/64+dq) control detects, down **N-A** (regime differs: anchor nf4/64 vs nf4/64+dq) control detects, q_proj **SAME-BYTES** (nf4/64+dq) control detects
- where Unsloth's step goes (`unsloth/ckpt_unsloth_prof`, descriptive): device busy fraction 0.309, device events/step 81780, CPU ops/step 515750, CPU self by family {'other': 0.3793, 'autograd': 0.2954, 'norm_act': 0.1311, 'matmul': 0.0733, 'memcpy': 0.0524, 'fused_kernel': 0.044, 'routing': 0.0179, 'optimizer': 0.0066}

## Verdicts
| family | arm | VERDICT | validity | quality | note |
|---|---|---|---|---|---|
| qwen3 | e4b/fused_attn4_m | **VALID** | VALID | 0.0000 |  |
| qwen3 | unsloth/ckpt_unsloth_m | **VALID** | VALID | 0.0061 |  |
| qwen3 | e4b/reference_attn4_m | **VALID** | VALID | 0.0018 |  |
| qwen3 | e4b/fused_attn4_m_d2 | **VALID** | VALID | -0.0010 |  |
| qwen3 | unsloth/ckpt_unsloth_m_d2 | **VALID** | VALID | 0.0027 |  |
| qwen3 | hf/hf_peft_m | **NOT_RUN** | — | — | skipped by TC1_SKIP |
| qwen3 | axolotl/ckpt_axolotl_m | **NOT_RUN** | — | — | skipped by TC1_SKIP |
| qwen3 | e4b/fused_attn4_m_prof | **VALID** | VALID | 0.0046 |  |
| qwen3 | unsloth/ckpt_unsloth_prof | **VALID** | VALID | 0.0009 |  |

## Predictions P1–P10 (+ P1b) (TC1-PREREG.md + phase 2, scored mechanically)
| prediction | family | verdict | evidence |
|---|---|---|---|
| P1 | qwen3 | **UNTESTED** | no quoted matched position: not quoted: e4b draws 2.540 / 2.414 s/step differ by 5.1% > 5% (UNSTABLE: reported, not quoted) / unsloth STABLE |
| P1b | qwen3 | **UNTESTED** | ckpt_unsloth_t28 missing: no receipt |
| P2 | qwen3 | **FALSIFIED** | e4b |d1-d2|/mean 5.1% vs 5% -> UNSTABLE; unsloth |d1-d2|/mean 0.6% vs 5% -> STABLE |
| P3 | qwen3 | **FALSIFIED** | e4b/reference_attn4_m INSIDE-DRAW-NOISE (median step |Δ| 0.0018, |Δ held-out at N| 0.0018, band {'train': 0.005385000000000306, 'heldout': 0.005399999999999738}); unsloth/ckpt_unsloth_m COMPARABLE (median step |Δ| 0.0031, |Δ held-out at N| 0.0061, band {'train': 0.005385000000000306, 'heldout': 0.005399999999999738}); draw-noise floor 0.0034; COMPARABLE, not EQUIVALENT, and not DIVERGENT: the decision rule does not fire |
| P4 | qwen3 | **UNTESTED** | shipped missing / fused_m VALID: both must be VALID on the same box |
| P5 | qwen3 | **UNTESTED** | hf_peft_m NOT_RUN, hf_peft_m_mb1 missing: neither an OOM pair nor a trained arm |
| P6 | qwen3 | **UNTESTED** | axolotl NOT_RUN (NOT_RUN / HARNESS_ERROR / ALARM is not a reading) |
| P7 | qwen3 | **UNTESTED** | unsloth_best missing / unsloth_m VALID / e4b_m VALID: all three must be VALID |
| P8 | qwen3 | **FALSIFIED** | device_busy_fraction 0.309 (>= 0.5 predicted on the grouped_mm arm); device events/step 81780 |
| P9 | qwen3 | **HELD** | fused_m vs reference_m PASS (Δfinal 0.00071, median 0.00180) |
| P10 | qwen3 | **UNTESTED** | C1 bit-exact; expert slots ['N-A', 'N-A'] (regime differs: anchor nf4/64 vs nf4/64+dq); attention SAME-BYTES |
