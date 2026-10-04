# TC1 — Qwen3-30B-A3B on one RTX 5090: e4b vs Unsloth vs HF+PEFT vs axolotl at matched work, init and adapter precision (tc1)
Rule (tc1/TC1-PREREG.md): status per attempt in the vocabulary OK / REFUSED / OOM / INSTALL_FAILED / LOAD_FAULT / HARNESS_ERROR / ALARM / NOT_RUN; VALID/VOID per tp4's predicates plus the matched-set predicates (R3: matched init complete, fp32 adapters, step-0 held-out within 0.005 of e4b/reference_attn4_m); VERDICT exactly one of VALID / VOID / QUALITY_FAIL / OOM / UNSUPPORTED / HARNESS_ERROR / ALARM / NOT_RUN (QUALITY_FAIL = held-out |Δ| at N > 0.05 vs e4b/fused_attn4_m); positions = s/step ratios other/e4b from the medians over both draws (stability |d1-d2|/mean ≤ 5 % e4b / 10 % others, else UNSTABLE), quoted only when both arms are VALID; equivalence vs e4b/reference_attn4_m EQUIVALENT ≤ 0.02 / COMPARABLE ≤ 0.05 / else DIVERGENT; frozen base SAME-BYTES / DIFFERENT / N-A per slot; predictions P1–P10 (+ P1b) scored HELD / FALSIFIED / UNTESTED. VOID never enters a ratio or an equivalence reading. Nothing is licensed; no cross-box number is divided into these.
`versions.txt`
```
e4b 0.45.0 @6c2f42cb72b08d9257b7357446056110000983c5 (GitHub main)
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
e4b(t212) 0.45.0 @6c2f42cb72b08d9257b7357446056110000983c5
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
 "run_id": "tc1-5090-66",
 "instance_id": "54201179",
 "gpu": "NVIDIA GeForce RTX 5090",
 "driver": "595.84",
 "cpu": "AMD EPYC 7B13 64-Core Processor",
 "nproc": 256,
 "mem_total_kb": "2101193408",
 "cgroup_memory_max": "730283900928",
 "disk_root": "overlay         320G   77M  320G   1% /",
 "hostname": "64dce77226c3",
 "registered_gpu_class": "5090",
 "prereg": "tc1/TC1-PREREG.md",
 "local_box": null,
 "local_snapshot": null
}
```

### Qwen3-30B-A3B (amendment 24: venv-e4b torch 2.8.0+cu128 vs venv-unsloth torch 2.12.1+cu130, matched and shipped arms) (`qwen3bmmab`, registered n_layers 48, attention census 192)
- model `Qwen/Qwen3-30B-A3B` @ `ad44e777bcd1`; tokens sha `bfc742f67e37`; N=20; fixture template alpaca seq 2048 micro-batch 2 × accum 4 lr 0.0002 r 16 α 16 optimizer adamw_8bit(lr=0.0002, weight_decay=0.001) schedule=linear warmup_steps=5 autocast False; e4b trainable 642514944; box_class RTX 5090 gpu NVIDIA GeForce RTX 5090
| framework | arm | status | validity | **VERDICT** | matched | init / dtype | step-0 class | N | s/step med(11+) | tok/s | peak GB | J/step | train first→last | held-out 0→final | quality Δ | engagement | trainable | regime | reason / why |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| e4b | fused_attn4_m_tv0 | **OK** | VALID | **VALID** | yes | matched:3407 (complete 12480/12480) / float32 | SAME-BYTES-CLASS (0.0000) | 20 | 3.915 | 340.9 | 27.157 | 1078.3 | 2.0614→0.8330 | 1.9441→0.8522 | 0.0000 | patched 48 / kcalls 768 | 642514944 | 4-bit experts (e4b NF4) + NF4 attention |  |
| e4b | fused_attn4_m_tv1 | **OK** | VALID | **VALID** | yes | matched:3407 (complete 12480/12480) / float32 | SAME-BYTES-CLASS (0.0038) | 20 | 3.495 | 386.5 | 27.187 | 986.9 | 2.0554→0.8311 | 1.9478→0.8467 | -0.0056 | patched 48 / kcalls 768 | 642514944 | 4-bit experts (e4b NF4) + NF4 attention |  |
| e4b | fused_attn4_m_tv1_d2 | **OK** | VALID | **VALID** | yes | matched:3407 (complete 12480/12480) / float32 | SAME-BYTES-CLASS (0.0038) | 20 | 3.394 | 438.6 | 27.187 | 956.5 | 2.0554→0.8308 | 1.9478→0.8483 | -0.0039 | patched 48 / kcalls 768 | 642514944 | 4-bit experts (e4b NF4) + NF4 attention |  |
| e4b | fused_attn4_m_tv0_d2 | **OK** | VALID | **VALID** | yes | matched:3407 (complete 12480/12480) / float32 | SAME-BYTES-CLASS (0.0000) | 20 | 3.895 | 376.4 | 27.157 | 1027.6 | 2.0614→0.8344 | 1.9441→0.8531 | 0.0009 | patched 48 / kcalls 768 | 642514944 | 4-bit experts (e4b NF4) + NF4 attention |  |
| e4b | fused_attn4_shipped_tv0 | **OK** | VALID | **VALID** | native | native / bfloat16,float32 | — | 20 | 3.009 | 487.8 | 24.576 | 810.4 | 2.0614→0.7986 | 1.9441→0.8159 | -0.0363 | patched 48 / kcalls 768 | 642514944 | 4-bit experts (e4b NF4) + NF4 attention |  |
| e4b | fused_attn4_shipped_tv1 | **OK** | VALID | **VALID** | native | native / bfloat16,float32 | — | 20 | 2.950 | 506.7 | 24.626 | 778.2 | 2.0554→0.8007 | 1.9478→0.8124 | -0.0399 | patched 48 / kcalls 768 | 642514944 | 4-bit experts (e4b NF4) + NF4 attention |  |
| e4b | fused_attn4_shipped_tv1_d2 | **OK** | VALID | **VALID** | native | native / bfloat16,float32 | — | 20 | 2.897 | 508.3 | 24.626 | 781.0 | 2.0554→0.7985 | 1.9478→0.8133 | -0.0390 | patched 48 / kcalls 768 | 642514944 | 4-bit experts (e4b NF4) + NF4 attention |  |
| e4b | fused_attn4_shipped_tv0_d2 | **OK** | VALID | **VALID** | native | native / bfloat16,float32 | — | 20 | 3.171 | 474.0 | 24.576 | 823.3 | 2.0614→0.7984 | 1.9441→0.8104 | -0.0418 | patched 48 / kcalls 768 | 642514944 | 4-bit experts (e4b NF4) + NF4 attention |  |
- prologue `e4b/fused_attn4_m_tv0` **109.2 s** before step 1 (55% of the arm): c1_before 54.8, c1_after 27.0, load_weights 25.8, eval0 11.3; unattributed 1.321; budget 1260.0
- prologue `e4b/fused_attn4_m_tv1` **114.9 s** before step 1 (59% of the arm): c1_before 56.0, load_weights 28.9, c1_after 26.6, eval0 13.3; unattributed 1.982; budget 1260.0
- prologue `e4b/fused_attn4_m_tv1_d2` **95.7 s** before step 1 (57% of the arm): c1_before 52.6, c1_after 27.1, load_weights 23.9, attn4 5.2; unattributed 1.994; budget 1260.0
- prologue `e4b/fused_attn4_m_tv0_d2` **104.7 s** before step 1 (56% of the arm): c1_before 56.1, load_weights 29.1, c1_after 26.3, attn4 5.5; unattributed 1.342; budget 1260.0
- prologue `e4b/fused_attn4_shipped_tv0` **102.7 s** before step 1 (62% of the arm): c1_before 55.5, c1_after 26.7, load_weights 25.2, trainable_sha 6.3; unattributed 1.264; budget 1260.0
- prologue `e4b/fused_attn4_shipped_tv1` **98.3 s** before step 1 (62% of the arm): c1_before 52.7, c1_after 26.3, load_weights 23.9, trainable_sha 6.0; unattributed 2.023; budget 1260.0
- prologue `e4b/fused_attn4_shipped_tv1_d2` **100.1 s** before step 1 (62% of the arm): c1_before 53.9, c1_after 26.9, load_weights 23.4, trainable_sha 6.3; unattributed 2.116; budget 1260.0
- prologue `e4b/fused_attn4_shipped_tv0_d2` **102.2 s** before step 1 (61% of the arm): c1_before 54.4, c1_after 26.0, load_weights 25.8, trainable_sha 6.4; unattributed 1.288; budget 1260.0
- draws (R1): `e4b/fused_attn4_m_tv0` STABLE (3.915/3.895 s, |Δ|/mean 0.5% vs 5%); `e4b/fused_attn4_m_tv1` STABLE (3.495/3.394 s, |Δ|/mean 2.9% vs 5%); `e4b/fused_attn4_shipped_tv0` UNSTABLE (3.009/3.171 s, |Δ|/mean 5.3% vs 5%); `e4b/fused_attn4_shipped_tv1` STABLE (2.950/2.897 s, |Δ|/mean 1.8% vs 5%)
- e4b internal parity: NO-ARM
- **NO MATCHED POSITION QUOTED (unsloth)** — not quoted: e4b STABLE / unsloth no receipt
- **NO MATCHED POSITION QUOTED (hf)** — not quoted: e4b STABLE / hf no receipt
- **NO MATCHED POSITION QUOTED (axolotl)** — not quoted: e4b STABLE / axolotl no receipt
- equivalence vs `e4b/fused_attn4_m` (I/H; EQUIVALENT iff both ≤ band = max(0.005, 3 × fused-vs-reference |Δ|) = none (no reference arm: COMPARABLE is the ceiling); COMPARABLE ≤ 0.05; draw-noise floor 0.0016): `e4b/fused_attn4_m_tv1` **COMPARABLE** (median step |Δ| 0.0020, |Δ held-out at N| 0.0056, step-0 0.0038 SAME-BYTES-CLASS, |Δ loss at step 2| 0.0071, paired rows mean -0.0055 ± 0.0029 SE over 8, favouring arm 6 / anchor 2) — no reference arm ran: the EQUIVALENT band cannot be set, COMPARABLE is the ceiling; `e4b/fused_attn4_m_tv1_d2` **COMPARABLE** (median step |Δ| 0.0016, |Δ held-out at N| 0.0039, step-0 0.0038 SAME-BYTES-CLASS, |Δ loss at step 2| 0.0029, paired rows mean -0.0039 ± 0.0022 SE over 8, favouring arm 5 / anchor 3) — no reference arm ran: the EQUIVALENT band cannot be set, COMPARABLE is the ceiling; `e4b/fused_attn4_m_tv0_d2` **COMPARABLE** (median step |Δ| 0.0013, |Δ held-out at N| 0.0009, step-0 0.0000 SAME-BYTES-CLASS, |Δ loss at step 2| 0.0008, paired rows mean +0.0009 ± 0.0023 SE over 8, favouring arm 4 / anchor 4) — no reference arm ran: the EQUIVALENT band cannot be set, COMPARABLE is the ceiling
- matched_init_sha (B, name-free, canonical slot order): anchor `f7832488926eda91`; `e4b/fused_attn4_m_tv0` same; `e4b/fused_attn4_m_tv1` same; `e4b/fused_attn4_m_tv1_d2` same; `e4b/fused_attn4_m_tv0_d2` same
- frozen base (R5), anchor `e4b/fused_attn4_m`: gate_up nf4/64 sha 8c3b3fd78e89 control detects, down nf4/64 sha 59569d810a1a control detects, q_proj nf4/64+dq sha 5c5e402ce51d control detects
- frozen base `e4b/fused_attn4_m_tv1`: gate_up **SAME-BYTES** (nf4/64) control detects, down **SAME-BYTES** (nf4/64) control detects, q_proj **SAME-BYTES** (nf4/64+dq) control detects
- frozen base `e4b/fused_attn4_m_tv1_d2`: gate_up **SAME-BYTES** (nf4/64) control detects, down **SAME-BYTES** (nf4/64) control detects, q_proj **SAME-BYTES** (nf4/64+dq) control detects
- frozen base `e4b/fused_attn4_m_tv0_d2`: gate_up **SAME-BYTES** (nf4/64) control detects, down **SAME-BYTES** (nf4/64) control detects, q_proj **SAME-BYTES** (nf4/64+dq) control detects
- frozen base `e4b/fused_attn4_shipped_tv0`: gate_up **SAME-BYTES** (nf4/64) control detects, down **SAME-BYTES** (nf4/64) control detects, q_proj **SAME-BYTES** (nf4/64+dq) control detects
- frozen base `e4b/fused_attn4_shipped_tv1`: gate_up **SAME-BYTES** (nf4/64) control detects, down **SAME-BYTES** (nf4/64) control detects, q_proj **SAME-BYTES** (nf4/64+dq) control detects
- frozen base `e4b/fused_attn4_shipped_tv1_d2`: gate_up **SAME-BYTES** (nf4/64) control detects, down **SAME-BYTES** (nf4/64) control detects, q_proj **SAME-BYTES** (nf4/64+dq) control detects
- frozen base `e4b/fused_attn4_shipped_tv0_d2`: gate_up **SAME-BYTES** (nf4/64) control detects, down **SAME-BYTES** (nf4/64) control detects, q_proj **SAME-BYTES** (nf4/64+dq) control detects

## Verdicts
| family | arm | VERDICT | validity | quality | note |
|---|---|---|---|---|---|
| qwen3bmmab | e4b/fused_attn4_m_tv0 | **VALID** | VALID | 0.0000 |  |
| qwen3bmmab | e4b/fused_attn4_m_tv1 | **VALID** | VALID | -0.0056 |  |
| qwen3bmmab | e4b/fused_attn4_m_tv1_d2 | **VALID** | VALID | -0.0039 |  |
| qwen3bmmab | e4b/fused_attn4_m_tv0_d2 | **VALID** | VALID | 0.0009 |  |
| qwen3bmmab | e4b/fused_attn4_shipped_tv0 | **VALID** | VALID | -0.0363 |  |
| qwen3bmmab | e4b/fused_attn4_shipped_tv1 | **VALID** | VALID | -0.0399 |  |
| qwen3bmmab | e4b/fused_attn4_shipped_tv1_d2 | **VALID** | VALID | -0.0390 |  |
| qwen3bmmab | e4b/fused_attn4_shipped_tv0_d2 | **VALID** | VALID | -0.0418 |  |

## Predictions P44–P49 (TC1-PREREG amendment 24: the RTX 5090's fp32 bmm host cost, replay and venv-e4b vs venv-unsloth A/B; scored mechanically)
| prediction | family | verdict | evidence |
|---|---|---|---|
| P44 | bmmbench | **FALSIFIED** | torch 2.8.0+cu128 on NVIDIA GeForce RTX 5090: fp32 88.9 us vs bf16 38.3 us per forward bmm at the recorded shapes (x2.32; registered >= 100 us and >= x3) |
| P45 | bmmbench | **HELD** | torch 2.8.0+cu128 fp32: cold 119.4 us (384 calls, 384 new shapes) vs repeated 37.9 us (x3.15; registered >= x2); fixed shape 37.4 us, bucketed (multiple of 32) 37.9 / again 37.4 us |
| P46 | bmmbench | **FALSIFIED** | fp32 cold median torch 2.12.1+cu130 120.9 us vs torch 2.8.0+cu128 119.4 us (x1.013; registered <= x0.5) |
| P47 | qwen3bmmab | **HELD** | matched: tv1 / tv0 0.882 [0.867, 0.897 over 4 cross-draw ratios]; s/step tv0 3.915 / 3.895 (within 0.5%), tv1 3.495 / 3.394 (within 2.9%); peak tv0 27.16 / tv1 27.19 GB vs [0.7, 0.95] |
| P48 | qwen3bmmab | **UNTESTED** | shipped: two stable VALID draws a side are registered -- tv0 UNSTABLE: draws 3.009 / 3.171 s/step differ by 5.3% > 5% (UNSTABLE: reported, not quoted); tv1 STABLE: |
| P49 | qwen3bmmab | **UNTESTED** | matched: mean held-out tv1 - tv0 -0.0052 (|.| <= 0.01); held-out at N tv0 [0.8522, 0.8531] tv1 [0.8467, 0.8483]; shipped: tv0 UNSTABLE: draws 3.009 / 3.171 s/step differ by 5.3% > 5% (UNSTABLE: reported, not quoted); tv1 STABLE: |

**Replay rows** (median host us per forward bmm; `bwd` = backward host / device ms at the recorded shapes)
| env | torch | dtype | blas | recorded | again | cold | fixed | bucket | bucket again | bwd host / device |
|---|---|---|---|---|---|---|---|---|---|---|
| t28 | 2.8.0+cu128 | fp32 | default | 88.9 | 37.9 | 119.4 | 37.4 | 37.9 | 37.4 | 0.924 / 1.070 |
| t28 | 2.8.0+cu128 | bf16 | default | 38.3 | 34.3 | 42.5 | 37.3 | 37.3 | 35.5 | 0.271 / 0.267 |
| t28 | 2.8.0+cu128 | fp32 | cublaslt | 89.7 | 40.6 | 240.1 | 72.8 | 89.8 | 50.6 | 0.620 / 1.003 |
| t28 | 2.8.0+cu128 | fp32tf32 | default | 80.2 | 35.3 | 269.4 | 37.9 | 38.8 | 33.3 | 0.521 / 0.736 |
| t212 | 2.12.1+cu130 | fp32 | default | 93.4 | 37.9 | 120.9 | 35.9 | 37.4 | 35.9 | 0.720 / 1.103 |
| t212 | 2.12.1+cu130 | bf16 | default | 38.3 | 37.8 | 43.5 | 37.8 | 38.8 | 39.0 | 0.330 / 0.348 |
| t212 | 2.12.1+cu130 | fp32 | cublaslt | 89.4 | 33.3 | 223.0 | 37.4 | 39.3 | 37.9 | 0.563 / 1.013 |
| t212 | 2.12.1+cu130 | fp32tf32 | default | 61.9 | 34.0 | 268.2 | 67.4 | 69.1 | 37.6 | 0.490 / 0.720 |
