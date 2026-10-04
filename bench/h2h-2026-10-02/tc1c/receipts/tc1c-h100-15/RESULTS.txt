# TC1 — Qwen3-30B-A3B on one RTX 5090: e4b vs Unsloth vs HF+PEFT vs axolotl at matched work, init and adapter precision (tc1)
Rule (tc1/TC1-PREREG.md): status per attempt in the vocabulary OK / REFUSED / OOM / INSTALL_FAILED / LOAD_FAULT / HARNESS_ERROR / ALARM / NOT_RUN; VALID/VOID per tp4's predicates plus the matched-set predicates (R3: matched init complete, fp32 adapters, step-0 held-out within 0.005 of e4b/reference_attn4_m); VERDICT exactly one of VALID / VOID / QUALITY_FAIL / OOM / UNSUPPORTED / HARNESS_ERROR / ALARM / NOT_RUN (QUALITY_FAIL = held-out |Δ| at N > 0.05 vs e4b/fused_attn4_m); positions = s/step ratios other/e4b from the medians over both draws (stability |d1-d2|/mean ≤ 5 % e4b / 10 % others, else UNSTABLE), quoted only when both arms are VALID; equivalence vs e4b/reference_attn4_m EQUIVALENT ≤ 0.02 / COMPARABLE ≤ 0.05 / else DIVERGENT; frozen base SAME-BYTES / DIFFERENT / N-A per slot; predictions P1–P10 (+ P1b) scored HELD / FALSIFIED / UNTESTED. VOID never enters a ratio or an equivalence reading. Nothing is licensed; no cross-box number is divided into these.
`versions.txt`
```
e4b 0.45.0 @3888fcabf20b144e75e32be12a99d2ae0f9053c2 (GitHub main)
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
e4b(t212) 0.45.0 @3888fcabf20b144e75e32be12a99d2ae0f9053c2
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
 "run_id": "tc1c-h100-15",
 "instance_id": "54181574",
 "gpu": "NVIDIA H100 NVL",
 "driver": "580.159.03",
 "cpu": "AMD EPYC 9V84 96-Core Processor",
 "nproc": 40,
 "mem_total_kb": "329973116",
 "cgroup_memory_max": "324375937024",
 "disk_root": "overlay         788G   89G  700G  12% /",
 "hostname": "7df3cc2aea4e",
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
| e4b | fused_attn4_m | **OK** | VALID | **VALID** | yes | matched:3407 (complete 12480/12480) / float32 | SAME-BYTES-CLASS (0.0000) | 20 | 2.428 | 630.5 | 27.207 | 492.9 | 2.0773→0.8317 | 1.9614→0.8491 | 0.0000 | patched 48 / kcalls 768 | 642514944 | 4-bit experts (e4b NF4) + NF4 attention |  |
| unsloth | ckpt_unsloth_m | **OK** | VALID | **VALID** | yes | matched:3407 (complete 12480/12480) / float32 | NEAR (0.0309) | 20 | 2.545 | 452.3 | 24.269 | 410.4 | 2.0403→0.8322 | 1.9305→0.8487 | -0.0003 | stacks 96 / fwd 384 / u8 48 | 642514944 | 4-bit expert stacks + bnb-4bit attention (Params4bit stacks 96, Linear4bit 384) |  |
| e4b | reference_attn4_m | **OK** | VALID | **VALID** | yes | matched:3407 (complete 12480/12480) / float32 | SAME-BYTES-CLASS (0.0063) | 20 | 48.248 | 31.1 | 27.147 | 1785.5 | 2.0461→0.8320 | 1.9551→0.8501 | 0.0011 | patched 0 / kcalls 0 | 642514944 | 4-bit experts (e4b NF4) + NF4 attention |  |
| e4b | fused_attn4_m_d2 | **OK** | VALID | **VALID** | yes | matched:3407 (complete 12480/12480) / float32 | SAME-BYTES-CLASS (0.0000) | 20 | 2.367 | 639.0 | 27.187 | 488.8 | 2.0773→0.8344 | 1.9614→0.8484 | -0.0007 | patched 48 / kcalls 768 | 642514944 | 4-bit experts (e4b NF4) + NF4 attention |  |
| unsloth | ckpt_unsloth_m_d2 | **OK** | VALID | **VALID** | yes | matched:3407 (complete 12480/12480) / float32 | NEAR (0.0309) | 20 | 2.543 | 565.2 | 24.269 | 389.1 | 2.0403→0.8362 | 1.9305→0.8494 | 0.0004 | stacks 96 / fwd 384 / u8 48 | 642514944 | 4-bit expert stacks + bnb-4bit attention (Params4bit stacks 96, Linear4bit 384) |  |
| hf | hf_peft_m | **NOT_RUN** | — | **NOT_RUN** | yes | — / — | — | 20 | — | — | — | — | —→— | —→— | — | peft mods — / params — / fwd — | — | — | skipped by TC1_SKIP |
| axolotl | ckpt_axolotl_m | **NOT_RUN** | — | **NOT_RUN** | yes | — / — | — | 20 | — | — | — | — | —→— | —→— | — | peft mods — / params — / fwd — | — | — | skipped by TC1_SKIP |
| e4b | fused_attn4_m_prof | **OK** | VALID | **VALID** | yes | matched:3407 (complete 12480/12480) / float32 | SAME-BYTES-CLASS (0.0000) | 20 | 2.657 | 483.8 | 27.187 | 298.2 | 2.0773→0.8298 | 1.9614→0.8479 | -0.0012 | patched 48 / kcalls 768 | 642514944 | 4-bit experts (e4b NF4) + NF4 attention |  |
| unsloth | ckpt_unsloth_prof | **OK** | VALID | **VALID** | yes | matched:3407 (complete 12480/12480) / float32 | NEAR (0.0309) | 20 | 2.806 | 502.0 | 24.269 | 217.9 | 2.0403→0.8335 | 1.9305→0.8505 | 0.0014 | stacks 96 / fwd 384 / u8 48 | 642514944 | 4-bit expert stacks + bnb-4bit attention (Params4bit stacks 96, Linear4bit 384) |  |
- prologue `e4b/fused_attn4_m` **54.0 s** before step 1 (52% of the arm): c1_before 24.2, load_weights 14.6, c1_after 11.9, preamble 3.6; unattributed 1.02; budget 1260.0
- prologue `unsloth/ckpt_unsloth_m` **65.3 s** before step 1 (49% of the arm): c1_before 21.6, load_weights 17.6, eval0 11.6, c1_after 10.8; unattributed 8.199; budget 1260.0
- prologue `e4b/reference_attn4_m` **69.5 s** before step 1 (7% of the arm): c1_before 24.2, eval0 18.1, load_weights 14.8, c1_after 12.1; unattributed 1.023; budget 1890.0
- prologue `e4b/fused_attn4_m_d2` **53.2 s** before step 1 (52% of the arm): c1_before 24.2, load_weights 14.8, c1_after 12.1, preamble 3.6; unattributed 1.019; budget 1260.0
- prologue `unsloth/ckpt_unsloth_m_d2` **58.7 s** before step 1 (52% of the arm): c1_before 21.5, load_weights 17.7, c1_after 10.8, eval0 5.0; unattributed 8.251; budget 1260.0
- prologue `e4b/fused_attn4_m_prof` **53.2 s** before step 1 (24% of the arm): c1_before 24.3, load_weights 14.6, c1_after 11.1, preamble 3.6; unattributed 1.047; budget 840.0
- prologue `unsloth/ckpt_unsloth_prof` **58.7 s** before step 1 (25% of the arm): c1_before 21.5, load_weights 17.7, c1_after 10.3, eval0 5.0; unattributed 8.155; budget 840.0
- draws (R1): `e4b/fused_attn4_m` STABLE (2.428/2.367 s, |Δ|/mean 2.5% vs 5%); `unsloth/ckpt_unsloth_m` STABLE (2.545/2.543 s, |Δ|/mean 0.1% vs 5%); `e4b/reference_attn4_m` SINGLE (single draw (no second draw registered)); `e4b/fused_attn4_m_prof` SINGLE (single draw (no second draw registered)); `unsloth/ckpt_unsloth_prof` SINGLE (single draw (no second draw registered))
- e4b internal parity (tp1's rule, informational): fused_attn4_m vs reference_attn4_m Δfinal 0.00024, median step |Δ| 0.00170 → **PASS** (band 0.05/0.05); ×19.87 faster per step, peak ×1.002
- **MATCHED POSITION: s/step ratio unsloth/e4b = 1.061** [1.047, 1.075 over 4 cross-draw ratios] (2.544 vs 2.397 s, medians over 2/2 draws; e4b faster per step); peak VRAM unsloth 24.27 vs e4b 27.20 GB (Δ -2.93); J/step unsloth 399.8 vs e4b 490.9 (×0.814); tok/s unsloth 508.8 vs e4b 634.8; unsloth regime: 4-bit expert stacks + bnb-4bit attention (Params4bit stacks 96, Linear4bit 384)
- quality reading at N=20 (unsloth): held-out e4b 0.8491 / unsloth 0.8487 (Δ -0.0003) → **COMPARABLE** (|Δ| ≤ 0.05 reads COMPARABLE); step-0 Δ -0.0309
- **NO MATCHED POSITION QUOTED (hf)** — not quoted: e4b STABLE / hf hf/hf_peft_m is NOT_RUN
- **NO MATCHED POSITION QUOTED (axolotl)** — not quoted: e4b STABLE / axolotl axolotl/ckpt_axolotl_m is NOT_RUN
- equivalence vs `e4b/fused_attn4_m` (I/H; EQUIVALENT iff both ≤ band = max(0.005, 3 × fused-vs-reference |Δ|) = train 0.0051 / held-out 0.0050; COMPARABLE ≤ 0.05; draw-noise floor 0.0007): `unsloth/ckpt_unsloth_m` **EQUIVALENT** (median step |Δ| 0.0027, |Δ held-out at N| 0.0003, step-0 0.0309 NEAR, |Δ loss at step 2| 0.0131, paired rows mean -0.0003 ± 0.0024 SE over 8, favouring arm 4 / anchor 4); `e4b/reference_attn4_m` **EQUIVALENT** (median step |Δ| 0.0017, |Δ held-out at N| 0.0011, step-0 0.0063 SAME-BYTES-CLASS, |Δ loss at step 2| 0.0214, paired rows mean +0.0011 ± 0.0018 SE over 8, favouring arm 4 / anchor 4); `e4b/fused_attn4_m_d2` **EQUIVALENT** (median step |Δ| 0.0017, |Δ held-out at N| 0.0007, step-0 0.0000 SAME-BYTES-CLASS, |Δ loss at step 2| 0.0069, paired rows mean -0.0007 ± 0.0021 SE over 8, favouring arm 4 / anchor 4); `unsloth/ckpt_unsloth_m_d2` **EQUIVALENT** (median step |Δ| 0.0024, |Δ held-out at N| 0.0004, step-0 0.0309 NEAR, |Δ loss at step 2| 0.0155, paired rows mean +0.0004 ± 0.0017 SE over 8, favouring arm 4 / anchor 4); `hf/hf_peft_m` **—** — no OK receipt; `axolotl/ckpt_axolotl_m` **—** — no OK receipt; `e4b/fused_attn4_m_prof` **EQUIVALENT** (median step |Δ| 0.0016, |Δ held-out at N| 0.0012, step-0 0.0000 SAME-BYTES-CLASS, |Δ loss at step 2| 0.0118, paired rows mean -0.0012 ± 0.0023 SE over 8, favouring arm 3 / anchor 5); `unsloth/ckpt_unsloth_prof` **EQUIVALENT** (median step |Δ| 0.0026, |Δ held-out at N| 0.0014, step-0 0.0309 NEAR, |Δ loss at step 2| 0.0046, paired rows mean +0.0014 ± 0.0030 SE over 8, favouring arm 3 / anchor 5)
- matched_init_sha (B, name-free, canonical slot order): anchor `f7832488926eda91`; `e4b/fused_attn4_m` same; `unsloth/ckpt_unsloth_m` same; `e4b/reference_attn4_m` same; `e4b/fused_attn4_m_d2` same; `unsloth/ckpt_unsloth_m_d2` same; `e4b/fused_attn4_m_prof` same; `unsloth/ckpt_unsloth_prof` same
- frozen base (R5), anchor `e4b/fused_attn4_m`: gate_up nf4/64 sha 8c3b3fd78e89 control detects, down nf4/64 sha 59569d810a1a control detects, q_proj nf4/64+dq sha 5c5e402ce51d control detects
- frozen base `unsloth/ckpt_unsloth_m`: gate_up **N-A** (regime differs: anchor nf4/64 vs nf4/64+dq) control detects, down **N-A** (regime differs: anchor nf4/64 vs nf4/64+dq) control detects, q_proj **SAME-BYTES** (nf4/64+dq) control detects
- frozen base `e4b/reference_attn4_m`: gate_up **SAME-BYTES** (nf4/64) control detects, down **SAME-BYTES** (nf4/64) control detects, q_proj **SAME-BYTES** (nf4/64+dq) control detects
- frozen base `e4b/fused_attn4_m_d2`: gate_up **SAME-BYTES** (nf4/64) control detects, down **SAME-BYTES** (nf4/64) control detects, q_proj **SAME-BYTES** (nf4/64+dq) control detects
- frozen base `unsloth/ckpt_unsloth_m_d2`: gate_up **N-A** (regime differs: anchor nf4/64 vs nf4/64+dq) control detects, down **N-A** (regime differs: anchor nf4/64 vs nf4/64+dq) control detects, q_proj **SAME-BYTES** (nf4/64+dq) control detects
- frozen base `e4b/fused_attn4_m_prof`: gate_up **SAME-BYTES** (nf4/64) control detects, down **SAME-BYTES** (nf4/64) control detects, q_proj **SAME-BYTES** (nf4/64+dq) control detects
- frozen base `unsloth/ckpt_unsloth_prof`: gate_up **N-A** (regime differs: anchor nf4/64 vs nf4/64+dq) control detects, down **N-A** (regime differs: anchor nf4/64 vs nf4/64+dq) control detects, q_proj **SAME-BYTES** (nf4/64+dq) control detects
- where Unsloth's step goes (`unsloth/ckpt_unsloth_prof`, descriptive): device busy fraction 0.308, device events/step 81780, CPU ops/step 515750, CPU self by family {'other': 0.3739, 'autograd': 0.3047, 'norm_act': 0.1228, 'matmul': 0.0756, 'memcpy': 0.0539, 'fused_kernel': 0.0459, 'routing': 0.0179, 'optimizer': 0.0052}

## Verdicts
| family | arm | VERDICT | validity | quality | note |
|---|---|---|---|---|---|
| qwen3 | e4b/fused_attn4_m | **VALID** | VALID | 0.0000 |  |
| qwen3 | unsloth/ckpt_unsloth_m | **VALID** | VALID | -0.0003 |  |
| qwen3 | e4b/reference_attn4_m | **VALID** | VALID | 0.0011 |  |
| qwen3 | e4b/fused_attn4_m_d2 | **VALID** | VALID | -0.0007 |  |
| qwen3 | unsloth/ckpt_unsloth_m_d2 | **VALID** | VALID | 0.0004 |  |
| qwen3 | hf/hf_peft_m | **NOT_RUN** | — | — | skipped by TC1_SKIP |
| qwen3 | axolotl/ckpt_axolotl_m | **NOT_RUN** | — | — | skipped by TC1_SKIP |
| qwen3 | e4b/fused_attn4_m_prof | **VALID** | VALID | -0.0012 |  |
| qwen3 | unsloth/ckpt_unsloth_prof | **VALID** | VALID | 0.0014 |  |

## Predictions P1–P10 (+ P1b) (TC1-PREREG.md + phase 2, scored mechanically)
| prediction | family | verdict | evidence |
|---|---|---|---|
| P1 | qwen3 | **FALSIFIED** | matched unsloth/e4b 1.061 vs [2.0, 5.0]; BELOW 1.5: the standing position is refuted and superseded |
| P1b | qwen3 | **UNTESTED** | ckpt_unsloth_t28 missing: no receipt |
| P2 | qwen3 | **HELD** | e4b |d1-d2|/mean 2.5% vs 5% -> STABLE; unsloth |d1-d2|/mean 0.1% vs 5% -> STABLE |
| P3 | qwen3 | **HELD** | e4b/reference_attn4_m EQUIVALENT (median step |Δ| 0.0017, |Δ held-out at N| 0.0011, band {'train': 0.005085000000000173, 'heldout': 0.005}); unsloth/ckpt_unsloth_m EQUIVALENT (median step |Δ| 0.0027, |Δ held-out at N| 0.0003, band {'train': 0.005085000000000173, 'heldout': 0.005}); draw-noise floor 0.0007 |
| P4 | qwen3 | **UNTESTED** | shipped missing / fused_m VALID: both must be VALID on the same box |
| P5 | qwen3 | **UNTESTED** | hf_peft_m NOT_RUN, hf_peft_m_mb1 missing: neither an OOM pair nor a trained arm |
| P6 | qwen3 | **UNTESTED** | axolotl NOT_RUN (NOT_RUN / HARNESS_ERROR / ALARM is not a reading) |
| P7 | qwen3 | **UNTESTED** | unsloth_best missing / unsloth_m VALID / e4b_m VALID: all three must be VALID |
| P8 | qwen3 | **FALSIFIED** | device_busy_fraction 0.308 (>= 0.5 predicted on the grouped_mm arm); device events/step 81780 |
| P9 | qwen3 | **HELD** | fused_m vs reference_m PASS (Δfinal 0.00024, median 0.00170) |
| P10 | qwen3 | **UNTESTED** | C1 bit-exact; expert slots ['N-A', 'N-A'] (regime differs: anchor nf4/64 vs nf4/64+dq); attention SAME-BYTES |
