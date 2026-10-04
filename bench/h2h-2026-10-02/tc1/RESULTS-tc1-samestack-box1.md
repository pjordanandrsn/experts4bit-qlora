# TC1 — Qwen3-30B-A3B on one RTX 5090: e4b vs Unsloth vs HF+PEFT vs axolotl at matched work, init and adapter precision (tc1)
Rule (tc1/TC1-PREREG.md): status per attempt in the vocabulary OK / REFUSED / OOM / INSTALL_FAILED / LOAD_FAULT / HARNESS_ERROR / ALARM / NOT_RUN; VALID/VOID per tp4's predicates plus the matched-set predicates (R3: matched init complete, fp32 adapters, step-0 held-out within 0.005 of e4b/reference_attn4_m); VERDICT exactly one of VALID / VOID / QUALITY_FAIL / OOM / UNSUPPORTED / HARNESS_ERROR / ALARM / NOT_RUN (QUALITY_FAIL = held-out |Δ| at N > 0.05 vs e4b/fused_attn4_m); positions = s/step ratios other/e4b from the medians over both draws (stability |d1-d2|/mean ≤ 5 % e4b / 10 % others, else UNSTABLE), quoted only when both arms are VALID; equivalence vs e4b/reference_attn4_m EQUIVALENT ≤ 0.02 / COMPARABLE ≤ 0.05 / else DIVERGENT; frozen base SAME-BYTES / DIFFERENT / N-A per slot; predictions P1–P10 (+ P1b) scored HELD / FALSIFIED / UNTESTED. VOID never enters a ratio or an equivalence reading. Nothing is licensed; no cross-box number is divided into these.
`versions.txt`
```
e4b 0.46.0 @89e5757603cc27ee41c27804918880bfdefe113b (GitHub main)
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
e4b(t212) 0.46.0 @89e5757603cc27ee41c27804918880bfdefe113b
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
 "run_id": "tc1-5090-67",
 "instance_id": "54209084",
 "gpu": "NVIDIA GeForce RTX 5090",
 "driver": "595.91.07",
 "cpu": "Intel(R) Core(TM) Ultra 9 285K",
 "nproc": 24,
 "mem_total_kb": "197216440",
 "cgroup_memory_max": "193871216640",
 "disk_root": "overlay         320G  9.5M  320G   1% /",
 "hostname": "a5043f393dbe",
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
| e4b | fused_attn4_m | **OK** | VALID | **VALID** | yes | matched:3407 (complete 12480/12480) / float32 | SAME-BYTES-CLASS (0.0000) | 20 | 2.220 | 623.2 | 27.187 | 818.4 | 2.0554→0.8338 | 1.9478→0.8505 | 0.0000 | patched 48 / kcalls 768 | 642514944 | 4-bit experts (e4b NF4) + NF4 attention |  |
| unsloth | ckpt_unsloth_m | **OK** | VALID | **VALID** | yes | matched:3407 (complete 12480/12480) / float32 | NEAR (0.0105) | 20 | 4.913 | 283.7 | 24.269 | 939.9 | 2.0705→0.8324 | 1.9583→0.8466 | -0.0039 | stacks 96 / fwd 384 / u8 48 | 642514944 | 4-bit expert stacks + bnb-4bit attention (Params4bit stacks 96, Linear4bit 384) |  |
| e4b | reference_attn4_m | **OK** | VALID | **VALID** | yes | matched:3407 (complete 12480/12480) / float32 | SAME-BYTES-CLASS (0.0042) | 20 | 29.582 | 49.4 | 27.147 | 2516.8 | 2.0769→0.8300 | 1.9521→0.8470 | -0.0035 | patched 0 / kcalls 0 | 642514944 | 4-bit experts (e4b NF4) + NF4 attention |  |
| e4b | fused_attn4_m_t28 | **OK** | VALID | **VALID** | yes | matched:3407 (complete 12480/12480) / float32 | SAME-BYTES-CLASS (0.0038) | 20 | 2.305 | 603.2 | 27.135 | 802.9 | 2.0614→0.8313 | 1.9441→0.8518 | 0.0013 | patched 48 / kcalls 768 | 642514944 | 4-bit experts (e4b NF4) + NF4 attention |  |
| e4b | fused_attn4_m_t28_d2 | **OK** | VALID | **VALID** | yes | matched:3407 (complete 12480/12480) / float32 | SAME-BYTES-CLASS (0.0038) | 20 | 2.507 | 619.4 | 27.157 | 763.5 | 2.0614→0.8338 | 1.9441→0.8478 | -0.0027 | patched 48 / kcalls 768 | 642514944 | 4-bit experts (e4b NF4) + NF4 attention |  |
| unsloth | ckpt_unsloth_m_d2 | **OK** | VALID | **VALID** | yes | matched:3407 (complete 12480/12480) / float32 | NEAR (0.0105) | 20 | 5.899 | 275.2 | 24.269 | 959.1 | 2.0705→0.8331 | 1.9583→0.8495 | -0.0010 | stacks 96 / fwd 384 / u8 48 | 642514944 | 4-bit expert stacks + bnb-4bit attention (Params4bit stacks 96, Linear4bit 384) |  |
| e4b | fused_attn4_m_d2 | **OK** | VALID | **VALID** | yes | matched:3407 (complete 12480/12480) / float32 | SAME-BYTES-CLASS (0.0000) | 20 | 2.188 | 692.0 | 27.187 | 740.9 | 2.0554→0.8327 | 1.9478→0.8496 | -0.0010 | patched 48 / kcalls 768 | 642514944 | 4-bit experts (e4b NF4) + NF4 attention |  |
- prologue `e4b/fused_attn4_m` **61.6 s** before step 1 (55% of the arm): c1_before 26.4, load_weights 17.1, c1_after 13.6, eval0 7.4; unattributed 1.264; budget 1260.0
- prologue `unsloth/ckpt_unsloth_m` **62.4 s** before step 1 (36% of the arm): c1_before 23.2, load_weights 18.2, c1_after 12.3, eval0 10.1; unattributed 6.161; budget 1260.0
- prologue `e4b/reference_attn4_m` **62.3 s** before step 1 (9% of the arm): c1_before 26.4, load_weights 13.5, c1_after 13.2, eval0 12.3; unattributed 1.152; budget 1890.0
- prologue `e4b/fused_attn4_m_t28` **54.9 s** before step 1 (52% of the arm): c1_before 27.2, c1_after 13.6, load_weights 13.0, eval0 5.6; unattributed 0.619; budget 1260.0
- prologue `e4b/fused_attn4_m_t28_d2` **49.8 s** before step 1 (50% of the arm): c1_before 26.8, c1_after 13.4, load_weights 12.9, preamble 2.9; unattributed 0.583; budget 1260.0
- prologue `unsloth/ckpt_unsloth_m_d2` **57.4 s** before step 1 (34% of the arm): c1_before 23.8, load_weights 16.6, c1_after 12.0, eval0 6.2; unattributed 5.909; budget 1260.0
- prologue `e4b/fused_attn4_m_d2` **50.0 s** before step 1 (53% of the arm): c1_before 26.6, c1_after 13.3, load_weights 13.2, preamble 3.1; unattributed 1.139; budget 1260.0
- draws (R1): `e4b/fused_attn4_m` STABLE (2.220/2.188 s, |Δ|/mean 1.5% vs 5%); `unsloth/ckpt_unsloth_m` UNSTABLE (4.913/5.899 s, |Δ|/mean 18.2% vs 5%); `e4b/reference_attn4_m` SINGLE (single draw (no second draw registered)); `e4b/fused_attn4_m_t28` UNSTABLE (2.305/2.507 s, |Δ|/mean 8.4% vs 5%)
- e4b internal parity (tp1's rule, informational): fused_attn4_m vs reference_attn4_m Δfinal 0.00380, median step |Δ| 0.00180 → **PASS** (band 0.05/0.05); ×13.32 faster per step, peak ×1.001
- **NO MATCHED POSITION QUOTED (unsloth)** — not quoted: e4b STABLE / unsloth draws 4.913 / 5.899 s/step differ by 18.2% > 5% (UNSTABLE: reported, not quoted)
- **NO MATCHED POSITION QUOTED (hf)** — not quoted: e4b STABLE / hf no receipt
- **NO MATCHED POSITION QUOTED (axolotl)** — not quoted: e4b STABLE / axolotl no receipt
- equivalence vs `e4b/fused_attn4_m` (I/H; EQUIVALENT iff both ≤ band = max(0.005, 3 × fused-vs-reference |Δ|) = train 0.0054 / held-out 0.0105; COMPARABLE ≤ 0.05; draw-noise floor 0.0010): `unsloth/ckpt_unsloth_m` **EQUIVALENT** (median step |Δ| 0.0015, |Δ held-out at N| 0.0039, step-0 0.0105 NEAR, |Δ loss at step 2| 0.0045, paired rows mean -0.0039 ± 0.0016 SE over 8, favouring arm 7 / anchor 1); `e4b/reference_attn4_m` **EQUIVALENT** (median step |Δ| 0.0018, |Δ held-out at N| 0.0035, step-0 0.0042 SAME-BYTES-CLASS, |Δ loss at step 2| 0.0161, paired rows mean -0.0035 ± 0.0020 SE over 8, favouring arm 7 / anchor 1); `e4b/fused_attn4_m_t28` **EQUIVALENT** (median step |Δ| 0.0033, |Δ held-out at N| 0.0013, step-0 0.0038 SAME-BYTES-CLASS, |Δ loss at step 2| 0.0074, paired rows mean +0.0013 ± 0.0012 SE over 8, favouring arm 3 / anchor 5); `e4b/fused_attn4_m_t28_d2` **EQUIVALENT** (median step |Δ| 0.0038, |Δ held-out at N| 0.0027, step-0 0.0038 SAME-BYTES-CLASS, |Δ loss at step 2| 0.0131, paired rows mean -0.0027 ± 0.0016 SE over 8, favouring arm 4 / anchor 4); `unsloth/ckpt_unsloth_m_d2` **EQUIVALENT** (median step |Δ| 0.0032, |Δ held-out at N| 0.0010, step-0 0.0105 NEAR, |Δ loss at step 2| 0.0080, paired rows mean -0.0011 ± 0.0023 SE over 8, favouring arm 4 / anchor 4); `e4b/fused_attn4_m_d2` **EQUIVALENT** (median step |Δ| 0.0013, |Δ held-out at N| 0.0010, step-0 0.0000 SAME-BYTES-CLASS, |Δ loss at step 2| 0.0183, paired rows mean -0.0010 ± 0.0019 SE over 8, favouring arm 4 / anchor 4)
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
| qwen3samestack | unsloth/ckpt_unsloth_m | **VALID** | VALID | -0.0039 |  |
| qwen3samestack | e4b/reference_attn4_m | **VALID** | VALID | -0.0035 |  |
| qwen3samestack | e4b/fused_attn4_m_t28 | **VALID** | VALID | 0.0013 |  |
| qwen3samestack | e4b/fused_attn4_m_t28_d2 | **VALID** | VALID | -0.0027 |  |
| qwen3samestack | unsloth/ckpt_unsloth_m_d2 | **VALID** | VALID | -0.0010 |  |
| qwen3samestack | e4b/fused_attn4_m_d2 | **VALID** | VALID | -0.0010 |  |

## Predictions P50 / P51 / P52 (TC1-PREREG amendment 25: the matched position with both frameworks on one stack; scored mechanically)
| prediction | family | verdict | evidence |
|---|---|---|---|
| P50 | qwen3samestack | **UNTESTED** | two stable VALID draws a side are registered -- not quoted: e4b STABLE / unsloth draws 4.913 / 5.899 s/step differ by 18.2% > 5% (UNSTABLE: reported, not quoted) |
| P51 | qwen3samestack | **UNTESTED** | two stable VALID draws a side are registered -- venv-unsloth STABLE:; venv-e4b UNSTABLE: draws 2.305 / 2.507 s/step differ by 8.4% > 5% (UNSTABLE: reported, not quoted) |
| P52 | qwen3samestack | **HELD** | `e4b/reference_attn4_m` EQUIVALENT; `unsloth/ckpt_unsloth_m` EQUIVALENT; e4b parity PASS |
