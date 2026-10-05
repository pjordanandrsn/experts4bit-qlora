# TC1 — Qwen3-30B-A3B on one RTX 5090: e4b vs Unsloth vs HF+PEFT vs axolotl at matched work, init and adapter precision (tc1)
Rule (tc1/TC1-PREREG.md): status per attempt in the vocabulary OK / REFUSED / OOM / INSTALL_FAILED / LOAD_FAULT / HARNESS_ERROR / ALARM / NOT_RUN; VALID/VOID per tp4's predicates plus the matched-set predicates (R3: matched init complete, fp32 adapters, step-0 held-out within 0.005 of e4b/reference_attn4_m); VERDICT exactly one of VALID / VOID / QUALITY_FAIL / OOM / UNSUPPORTED / HARNESS_ERROR / ALARM / NOT_RUN (QUALITY_FAIL = held-out |Δ| at N > 0.05 vs e4b/fused_attn4_m); positions = s/step ratios other/e4b from the medians over both draws (stability |d1-d2|/mean ≤ 5 % e4b / 10 % others, else UNSTABLE), quoted only when both arms are VALID; equivalence vs e4b/reference_attn4_m EQUIVALENT ≤ 0.02 / COMPARABLE ≤ 0.05 / else DIVERGENT; frozen base SAME-BYTES / DIFFERENT / N-A per slot; predictions P1–P10 (+ P1b) scored HELD / FALSIFIED / UNTESTED. VOID never enters a ratio or an equivalence reading. Nothing is licensed; no cross-box number is divided into these.
`versions.txt`
```
e4b 0.46.0 @285ed4493d0f8345186390f7a215452b5b19459e (GitHub main)
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
e4b(t212) 0.46.0 @285ed4493d0f8345186390f7a215452b5b19459e
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
 "run_id": "tc1-5090-68",
 "instance_id": "54214853",
 "gpu": "NVIDIA GeForce RTX 5090",
 "driver": "595.71.05",
 "cpu": "AMD EPYC 7663 56-Core Processor",
 "nproc": 112,
 "mem_total_kb": "263686464",
 "cgroup_memory_max": "",
 "disk_root": "overlay         320G  9.5M  320G   1% /",
 "hostname": "9f22c7469418",
 "registered_gpu_class": "5090",
 "prereg": "tc1/TC1-PREREG.md",
 "local_box": null,
 "local_snapshot": null
}
```

### Qwen3-30B-A3B (amendment 25: the matched set with e4b and Unsloth on one stack, torch 2.12.1+cu130 / transformers 5.5.0) (`qwen3samestack`, registered n_layers 48, attention census 192)
- model `Qwen/Qwen3-30B-A3B` @ `ad44e777bcd1`; tokens sha `bfc742f67e37`; N=20; fixture template alpaca seq 2048 micro-batch 2 × accum 4 lr 0.0002 r 16 α 16 optimizer adamw_8bit(lr=0.0002, weight_decay=0.001) schedule=linear warmup_steps=5 autocast False; e4b trainable 642514944; box_class RTX 5090 gpu NVIDIA GeForce RTX 5090
| framework | arm | status | validity | **VERDICT** | matched | init / dtype | step-0 class | N | s/step med(11+) | tok/s | peak GB | J/step | train first→last | held-out 0→final | quality Δ | engagement | trainable | regime | reason / why |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| e4b | fused_attn4_m | **OK** | VALID | **VALID** | yes | matched:3407 (complete 12480/12480) / float32 | SAME-BYTES-CLASS (0.0000) | 20 | 4.060 | 334.2 | 27.187 | 910.6 | 2.0554→0.8330 | 1.9478→0.8479 | 0.0000 | patched 48 / kcalls 768 | 642514944 | 4-bit experts (e4b NF4) + NF4 attention |  |
| unsloth | ckpt_unsloth_m | **OK** | VALID | **VALID** | yes | matched:3407 (complete 12480/12480) / float32 | NEAR (0.0105) | 20 | 8.989 | 154.1 | 24.269 | 1155.1 | 2.0705→0.8321 | 1.9583→0.8451 | -0.0028 | stacks 96 / fwd 384 / u8 48 | 642514944 | 4-bit expert stacks + bnb-4bit attention (Params4bit stacks 96, Linear4bit 384) |  |
| e4b | reference_attn4_m | **OK** | VALID | **VALID** | yes | matched:3407 (complete 12480/12480) / float32 | SAME-BYTES-CLASS (0.0042) | 20 | 71.621 | 20.9 | 27.147 | 5036.9 | 2.0769→0.8338 | 1.9521→0.8503 | 0.0024 | patched 0 / kcalls 0 | 642514944 | 4-bit experts (e4b NF4) + NF4 attention |  |
| e4b | fused_attn4_m_t28 | **OK** | VALID | **VALID** | yes | matched:3407 (complete 12480/12480) / float32 | SAME-BYTES-CLASS (0.0038) | 20 | 4.388 | 305.9 | 27.135 | 1039.9 | 2.0614→0.8333 | 1.9441→0.8498 | 0.0020 | patched 48 / kcalls 768 | 642514944 | 4-bit experts (e4b NF4) + NF4 attention |  |
| e4b | fused_attn4_m_t28_d2 | **OK** | VALID | **VALID** | yes | matched:3407 (complete 12480/12480) / float32 | SAME-BYTES-CLASS (0.0038) | 20 | 4.314 | 352.8 | 27.157 | 919.0 | 2.0614→0.8328 | 1.9441→0.8491 | 0.0013 | patched 48 / kcalls 768 | 642514944 | 4-bit experts (e4b NF4) + NF4 attention |  |
| unsloth | ckpt_unsloth_m_d2 | **OK** | VALID | **VALID** | yes | matched:3407 (complete 12480/12480) / float32 | NEAR (0.0105) | 20 | 9.246 | 158.8 | 24.269 | 1030.6 | 2.0705→0.8312 | 1.9583→0.8487 | 0.0008 | stacks 96 / fwd 384 / u8 48 | 642514944 | 4-bit expert stacks + bnb-4bit attention (Params4bit stacks 96, Linear4bit 384) |  |
| e4b | fused_attn4_m_d2 | **OK** | VALID | **VALID** | yes | matched:3407 (complete 12480/12480) / float32 | SAME-BYTES-CLASS (0.0000) | 20 | 3.755 | 395.8 | 27.206 | 882.5 | 2.0554→0.8315 | 1.9478→0.8497 | 0.0018 | patched 48 / kcalls 768 | 642514944 | 4-bit experts (e4b NF4) + NF4 attention |  |
- prologue `e4b/fused_attn4_m` **132.7 s** before step 1 (59% of the arm): c1_before 73.8, c1_after 35.7, load_weights 23.2, eval0 14.3; unattributed 2.316; budget 1260.0
- prologue `unsloth/ckpt_unsloth_m` **129.8 s** before step 1 (39% of the arm): c1_before 63.9, c1_after 31.1, load_weights 28.4, eval0 16.6; unattributed 12.637; budget 1260.0
- prologue `e4b/reference_attn4_m` **136.6 s** before step 1 (8% of the arm): c1_before 71.7, c1_after 36.2, eval0 25.6, load_weights 20.3; unattributed 2.059; budget 1890.0
- prologue `e4b/fused_attn4_m_t28` **124.2 s** before step 1 (55% of the arm): c1_before 70.5, c1_after 36.5, load_weights 22.9, eval0 11.4; unattributed 1.585; budget 1260.0
- prologue `e4b/fused_attn4_m_t28_d2` **111.4 s** before step 1 (56% of the arm): c1_before 68.7, c1_after 34.6, load_weights 21.7, attn4 6.5; unattributed 1.251; budget 1260.0
- prologue `unsloth/ckpt_unsloth_m_d2` **119.9 s** before step 1 (38% of the arm): c1_before 62.0, c1_after 31.1, load_weights 28.8, eval0 9.6; unattributed 11.18; budget 1260.0
- prologue `e4b/fused_attn4_m_d2` **109.7 s** before step 1 (58% of the arm): c1_before 68.9, c1_after 36.0, load_weights 19.9, attn4 6.3; unattributed 2.102; budget 1260.0
- draws (R1): `e4b/fused_attn4_m` UNSTABLE (4.060/3.755 s, |Δ|/mean 7.8% vs 5%); `unsloth/ckpt_unsloth_m` STABLE (8.989/9.246 s, |Δ|/mean 2.8% vs 5%); `e4b/reference_attn4_m` SINGLE (single draw (no second draw registered)); `e4b/fused_attn4_m_t28` STABLE (4.388/4.314 s, |Δ|/mean 1.7% vs 5%)
- e4b internal parity (tp1's rule, informational): fused_attn4_m vs reference_attn4_m Δfinal 0.00084, median step |Δ| 0.00185 → **PASS** (band 0.05/0.05); ×17.64 faster per step, peak ×1.001
- **NO MATCHED POSITION QUOTED (unsloth)** — not quoted: e4b draws 4.060 / 3.755 s/step differ by 7.8% > 5% (UNSTABLE: reported, not quoted) / unsloth STABLE
- **NO MATCHED POSITION QUOTED (hf)** — not quoted: e4b draws 4.060 / 3.755 s/step differ by 7.8% > 5% (UNSTABLE: reported, not quoted) / hf no receipt
- **NO MATCHED POSITION QUOTED (axolotl)** — not quoted: e4b draws 4.060 / 3.755 s/step differ by 7.8% > 5% (UNSTABLE: reported, not quoted) / axolotl no receipt
- equivalence vs `e4b/fused_attn4_m` (I/H; EQUIVALENT iff both ≤ band = max(0.005, 3 × fused-vs-reference |Δ|) = train 0.0055 / held-out 0.0073; COMPARABLE ≤ 0.05; draw-noise floor 0.0036): `unsloth/ckpt_unsloth_m` **INSIDE-DRAW-NOISE** (median step |Δ| 0.0025, |Δ held-out at N| 0.0028, step-0 0.0105 NEAR, |Δ loss at step 2| 0.0042, paired rows mean -0.0028 ± 0.0035 SE over 8, favouring arm 5 / anchor 3) — |delta| 0.0028 / 0.0025 are narrower than the draw-noise floor 0.0036: inside the draw noise, not a precision statement; `e4b/reference_attn4_m` **INSIDE-DRAW-NOISE** (median step |Δ| 0.0018, |Δ held-out at N| 0.0024, step-0 0.0042 SAME-BYTES-CLASS, |Δ loss at step 2| 0.0097, paired rows mean +0.0024 ± 0.0013 SE over 8, favouring arm 1 / anchor 7) — |delta| 0.0024 / 0.0018 are narrower than the draw-noise floor 0.0036: inside the draw noise, not a precision statement; `e4b/fused_attn4_m_t28` **INSIDE-DRAW-NOISE** (median step |Δ| 0.0013, |Δ held-out at N| 0.0020, step-0 0.0038 SAME-BYTES-CLASS, |Δ loss at step 2| 0.0154, paired rows mean +0.0019 ± 0.0016 SE over 8, favouring arm 3 / anchor 5) — |delta| 0.0020 / 0.0013 are narrower than the draw-noise floor 0.0036: inside the draw noise, not a precision statement; `e4b/fused_attn4_m_t28_d2` **INSIDE-DRAW-NOISE** (median step |Δ| 0.0028, |Δ held-out at N| 0.0013, step-0 0.0038 SAME-BYTES-CLASS, |Δ loss at step 2| 0.0169, paired rows mean +0.0013 ± 0.0019 SE over 8, favouring arm 3 / anchor 5) — |delta| 0.0013 / 0.0028 are narrower than the draw-noise floor 0.0036: inside the draw noise, not a precision statement; `unsloth/ckpt_unsloth_m_d2` **INSIDE-DRAW-NOISE** (median step |Δ| 0.0025, |Δ held-out at N| 0.0008, step-0 0.0105 NEAR, |Δ loss at step 2| 0.0099, paired rows mean +0.0008 ± 0.0036 SE over 8, favouring arm 2 / anchor 6) — |delta| 0.0008 / 0.0025 are narrower than the draw-noise floor 0.0036: inside the draw noise, not a precision statement; `e4b/fused_attn4_m_d2` **INSIDE-DRAW-NOISE** (median step |Δ| 0.0014, |Δ held-out at N| 0.0018, step-0 0.0000 SAME-BYTES-CLASS, |Δ loss at step 2| 0.0052, paired rows mean +0.0018 ± 0.0039 SE over 8, favouring arm 3 / anchor 5) — |delta| 0.0018 / 0.0014 are narrower than the draw-noise floor 0.0036: inside the draw noise, not a precision statement
- matched_init_sha (B, name-free, canonical slot order): anchor `f7832488926eda91`; `e4b/fused_attn4_m` same; `unsloth/ckpt_unsloth_m` same; `e4b/reference_attn4_m` same; `e4b/fused_attn4_m_t28` same; `e4b/fused_attn4_m_t28_d2` same; `unsloth/ckpt_unsloth_m_d2` same; `e4b/fused_attn4_m_d2` same
- frozen base (R5), anchor `e4b/fused_attn4_m`: gate_up nf4/64 sha 8c3b3fd78e89 control detects, down nf4/64 sha 59569d810a1a control detects, q_proj nf4/64+dq sha 5c5e402ce51d control detects
- frozen base `unsloth/ckpt_unsloth_m`: gate_up **N-A** (regime differs: anchor nf4/64 vs nf4/64+dq) control detects, down **N-A** (regime differs: anchor nf4/64 vs nf4/64+dq) control detects, q_proj **SAME-BYTES** (nf4/64+dq) control detects
- frozen base `e4b/reference_attn4_m`: gate_up **SAME-BYTES** (nf4/64) control detects, down **SAME-BYTES** (nf4/64) control detects, q_proj **SAME-BYTES** (nf4/64+dq) control detects
- frozen base `e4b/fused_attn4_m_t28`: gate_up **SAME-BYTES** (nf4/64) control detects, down **SAME-BYTES** (nf4/64) control detects, q_proj **SAME-BYTES** (nf4/64+dq) control detects
- frozen base `e4b/fused_attn4_m_t28_d2`: gate_up **SAME-BYTES** (nf4/64) control detects, down **SAME-BYTES** (nf4/64) control detects, q_proj **SAME-BYTES** (nf4/64+dq) control detects
- frozen base `unsloth/ckpt_unsloth_m_d2`: gate_up **N-A** (regime differs: anchor nf4/64 vs nf4/64+dq) control detects, down **N-A** (regime differs: anchor nf4/64 vs nf4/64+dq) control detects, q_proj **SAME-BYTES** (nf4/64+dq) control detects
- frozen base `e4b/fused_attn4_m_d2`: gate_up **SAME-BYTES** (nf4/64) control detects, down **SAME-BYTES** (nf4/64) control detects, q_proj **SAME-BYTES** (nf4/64+dq) control detects

## Verdicts
| family | arm | VERDICT | validity | quality | note |
|---|---|---|---|---|---|
| qwen3samestack | e4b/fused_attn4_m | **VALID** | VALID | 0.0000 |  |
| qwen3samestack | unsloth/ckpt_unsloth_m | **VALID** | VALID | -0.0028 |  |
| qwen3samestack | e4b/reference_attn4_m | **VALID** | VALID | 0.0024 |  |
| qwen3samestack | e4b/fused_attn4_m_t28 | **VALID** | VALID | 0.0020 |  |
| qwen3samestack | e4b/fused_attn4_m_t28_d2 | **VALID** | VALID | 0.0013 |  |
| qwen3samestack | unsloth/ckpt_unsloth_m_d2 | **VALID** | VALID | 0.0008 |  |
| qwen3samestack | e4b/fused_attn4_m_d2 | **VALID** | VALID | 0.0018 |  |

## Predictions P50 / P51 / P52 (TC1-PREREG amendment 25: the matched position with both frameworks on one stack; scored mechanically)
| prediction | family | verdict | evidence |
|---|---|---|---|
| P50 | qwen3samestack | **UNTESTED** | two stable VALID draws a side are registered -- not quoted: e4b draws 4.060 / 3.755 s/step differ by 7.8% > 5% (UNSTABLE: reported, not quoted) / unsloth STABLE |
| P51 | qwen3samestack | **UNTESTED** | two stable VALID draws a side are registered -- venv-unsloth UNSTABLE: draws 4.060 / 3.755 s/step differ by 7.8% > 5% (UNSTABLE: reported, not quoted); venv-e4b STABLE: |
| P52 | qwen3samestack | **HELD** | `e4b/reference_attn4_m` INSIDE-DRAW-NOISE; `unsloth/ckpt_unsloth_m` INSIDE-DRAW-NOISE; e4b parity PASS |
