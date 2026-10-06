# TC1 — Qwen3-30B-A3B on one RTX 5090: e4b vs Unsloth vs HF+PEFT vs axolotl at matched work, init and adapter precision (tc1-5090-107)
Rule (tc1/TC1-PREREG.md): status per attempt in the vocabulary OK / REFUSED / OOM / INSTALL_FAILED / LOAD_FAULT / HARNESS_ERROR / ALARM / NOT_RUN; VALID/VOID per tp4's predicates plus the matched-set predicates (R3: matched init complete, fp32 adapters, step-0 held-out within 0.005 of e4b/reference_attn4_m); VERDICT exactly one of VALID / VOID / QUALITY_FAIL / OOM / UNSUPPORTED / HARNESS_ERROR / ALARM / NOT_RUN (QUALITY_FAIL = held-out |Δ| at N > 0.05 vs e4b/fused_attn4_m); positions = s/step ratios other/e4b from the medians over both draws (stability |d1-d2|/mean ≤ 5 % e4b / 10 % others, else UNSTABLE), quoted only when both arms are VALID; equivalence vs e4b/reference_attn4_m EQUIVALENT ≤ 0.02 / COMPARABLE ≤ 0.05 / else DIVERGENT; frozen base SAME-BYTES / DIFFERENT / N-A per slot; predictions P1–P10 (+ P1b) scored HELD / FALSIFIED / UNTESTED. VOID never enters a ratio or an equivalence reading. Nothing is licensed; no cross-box number is divided into these.
`versions.txt`
```
e4b 0.48.0 @9d3732a26b1ed01daddcc629118ac41ee1c0edd8 (GitHub main)
gnf4 0.41.0 @706d84ff7b2199632d1bac7cb9d2982325220b86 (GitHub main)
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
e4b(t212) 0.48.0 @9d3732a26b1ed01daddcc629118ac41ee1c0edd8
gnf4(t212) 0.41.0 @706d84ff7b2199632d1bac7cb9d2982325220b86
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
 "run_id": "tc1-5090-107",
 "instance_id": "54473350",
 "gpu": "NVIDIA GeForce RTX 5090",
 "driver": "590.48.01",
 "cpu": "AMD Ryzen Threadripper PRO 3955WX 16-Cores",
 "nproc": 32,
 "mem_total_kb": "263771780",
 "cgroup_memory_max": "183350853632",
 "disk_root": "overlay         320G   48M  320G   1% /",
 "hostname": "c7410c541105",
 "cgroup_cpu_max": "1536000 100000",
 "affinity_cpus": 32,
 "cgroup_cpuset_effective": "0-31",
 "registered_gpu_class": "5090",
 "prereg": "tc1/TC1-PREREG.md",
 "local_box": null,
 "local_snapshot": null
}
```

### Qwen3-30B-A3B (amendment 50: NF4_QLORA_PAD_BUCKETS 0 vs auto at the field recipe, venv-unsloth; the gate must not fire) (`qwen3fieldauto`, registered n_layers 48, attention census 192)
- model `Qwen/Qwen3-30B-A3B` @ `ad44e777bcd1`; tokens sha `bfc742f67e37`; N=60; fixture template alpaca seq 2048 micro-batch 2 × accum 4 lr 0.0002 r 16 α 16 optimizer adamw_8bit(lr=0.0002, weight_decay=0.001) schedule=linear warmup_steps=5 autocast False; e4b trainable 642514944; box_class RTX 5090 gpu NVIDIA GeForce RTX 5090
| framework | arm | status | validity | **VERDICT** | matched | init / dtype | step-0 class | N | s/step med(11+) | tok/s | peak GB | J/step | train first→last | held-out 0→final | quality Δ | engagement | trainable | regime | reason / why |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| e4b | fused_attn4_shipped_fa0 | **OK** | VALID | **VALID** | native | native / bfloat16,float32 | — | 60 | 2.513 | 595.1 | 24.673 | 739.6 | 2.0554→0.8192 | 1.9478→0.7575 | 0.0044 | patched 48 / kcalls 768 | 642514944 | 4-bit experts (e4b NF4) + NF4 attention |  |
| e4b | fused_attn4_shipped_fa1 | **OK** | VALID | **VALID** | native | native / bfloat16,float32 | — | 60 | 2.446 | 647.3 | 24.673 | 703.6 | 2.0554→0.8186 | 1.9478→0.7576 | 0.0045 | patched 48 / kcalls 768 | 642514944 | 4-bit experts (e4b NF4) + NF4 attention |  |
| e4b | fused_attn4_m_fa0 | **OK** | VALID | **VALID** | yes | matched:3407 (complete 12480/12480) / float32 | SAME-BYTES-CLASS (0.0000) | 60 | 3.175 | 496.8 | 27.505 | 898.0 | 2.0554→0.8190 | 1.9478→0.7532 | 0.0000 | patched 48 / kcalls 768 | 642514944 | 4-bit experts (e4b NF4) + NF4 attention |  |
| e4b | fused_attn4_m_fa1 | **OK** | VALID | **VALID** | yes | matched:3407 (complete 12480/12480) / float32 | SAME-BYTES-CLASS (0.0000) | 60 | 3.184 | 500.3 | 27.483 | 888.8 | 2.0554→0.8175 | 1.9478→0.7570 | 0.0038 | patched 48 / kcalls 768 | 642514944 | 4-bit experts (e4b NF4) + NF4 attention |  |
| e4b | fused_attn4_m_fa1_d2 | **OK** | VALID | **VALID** | yes | matched:3407 (complete 12480/12480) / float32 | SAME-BYTES-CLASS (0.0000) | 60 | 3.195 | 493.7 | 27.487 | 883.3 | 2.0554→0.8199 | 1.9478→0.7571 | 0.0039 | patched 48 / kcalls 768 | 642514944 | 4-bit experts (e4b NF4) + NF4 attention |  |
| e4b | fused_attn4_m_fa0_d2 | **OK** | VALID | **VALID** | yes | matched:3407 (complete 12480/12480) / float32 | SAME-BYTES-CLASS (0.0000) | 60 | 3.213 | 494.9 | 27.490 | 893.7 | 2.0554→0.8174 | 1.9478→0.7565 | 0.0033 | patched 48 / kcalls 768 | 642514944 | 4-bit experts (e4b NF4) + NF4 attention |  |
| e4b | fused_attn4_shipped_fa1_d2 | **OK** | VALID | **VALID** | native | native / bfloat16,float32 | — | 60 | 2.448 | 641.7 | 24.673 | 712.8 | 2.0554→0.8175 | 1.9478→0.7557 | 0.0025 | patched 48 / kcalls 768 | 642514944 | 4-bit experts (e4b NF4) + NF4 attention |  |
| e4b | fused_attn4_shipped_fa0_d2 | **OK** | VALID | **VALID** | native | native / bfloat16,float32 | — | 60 | 2.466 | 642.9 | 24.673 | 708.4 | 2.0554→0.8186 | 1.9478→0.7554 | 0.0023 | patched 48 / kcalls 768 | 642514944 | 4-bit experts (e4b NF4) + NF4 attention |  |
- prologue `e4b/fused_attn4_shipped_fa0` **98.8 s** before step 1 (38% of the arm): c1_before 54.4, c1_after 27.4, load_weights 17.6, eval0 12.0; unattributed 1.76; budget 1260.0
- prologue `e4b/fused_attn4_shipped_fa1` **89.0 s** before step 1 (37% of the arm): c1_before 55.3, c1_after 27.5, load_weights 17.6, attn4 5.4; unattributed 1.673; budget 1260.0
- prologue `e4b/fused_attn4_m_fa0` **90.1 s** before step 1 (31% of the arm): c1_before 54.9, c1_after 27.7, load_weights 17.6, attn4 5.5; unattributed 1.697; budget 1260.0
- prologue `e4b/fused_attn4_m_fa1` **90.4 s** before step 1 (32% of the arm): c1_before 54.9, c1_after 27.8, load_weights 17.7, attn4 5.7; unattributed 1.714; budget 1260.0
- prologue `e4b/fused_attn4_m_fa1_d2` **89.6 s** before step 1 (31% of the arm): c1_before 54.4, c1_after 27.7, load_weights 17.6, attn4 5.5; unattributed 1.713; budget 1260.0
- prologue `e4b/fused_attn4_m_fa0_d2` **90.0 s** before step 1 (31% of the arm): c1_before 54.5, c1_after 27.5, load_weights 17.7, attn4 5.6; unattributed 1.725; budget 1260.0
- prologue `e4b/fused_attn4_shipped_fa1_d2` **88.7 s** before step 1 (37% of the arm): c1_before 54.7, c1_after 28.1, load_weights 17.8, attn4 5.6; unattributed 1.712; budget 1260.0
- prologue `e4b/fused_attn4_shipped_fa0_d2` **89.2 s** before step 1 (37% of the arm): c1_before 55.4, c1_after 27.7, load_weights 17.6, attn4 5.6; unattributed 1.728; budget 1260.0
- draws (R1): `e4b/fused_attn4_shipped_fa0` STABLE (2.513/2.466 s, |Δ|/mean 1.9% vs 5%); `e4b/fused_attn4_shipped_fa1` STABLE (2.446/2.448 s, |Δ|/mean 0.1% vs 5%); `e4b/fused_attn4_m_fa0` STABLE (3.175/3.213 s, |Δ|/mean 1.2% vs 5%); `e4b/fused_attn4_m_fa1` STABLE (3.184/3.195 s, |Δ|/mean 0.3% vs 5%)
- e4b internal parity: NO-ARM
- **NO MATCHED POSITION QUOTED (unsloth)** — not quoted: e4b STABLE / unsloth no receipt
- **NO MATCHED POSITION QUOTED (hf)** — not quoted: e4b STABLE / hf no receipt
- **NO MATCHED POSITION QUOTED (axolotl)** — not quoted: e4b STABLE / axolotl no receipt
- equivalence vs `e4b/fused_attn4_m` (I/H; EQUIVALENT iff both ≤ band = max(0.005, 3 × fused-vs-reference |Δ|) = none (no reference arm: COMPARABLE is the ceiling); COMPARABLE ≤ 0.05; draw-noise floor 0.0033): `e4b/fused_attn4_m_fa1` **COMPARABLE** (median step |Δ| 0.0012, |Δ held-out at N| 0.0038, step-0 0.0000 SAME-BYTES-CLASS, |Δ loss at step 2| 0.0017, paired rows mean +0.0038 ± 0.0013 SE over 8, favouring arm 0 / anchor 8) — no reference arm ran: the EQUIVALENT band cannot be set, COMPARABLE is the ceiling; `e4b/fused_attn4_m_fa1_d2` **COMPARABLE** (median step |Δ| 0.0009, |Δ held-out at N| 0.0039, step-0 0.0000 SAME-BYTES-CLASS, |Δ loss at step 2| 0.0059, paired rows mean +0.0039 ± 0.0018 SE over 8, favouring arm 2 / anchor 6) — no reference arm ran: the EQUIVALENT band cannot be set, COMPARABLE is the ceiling; `e4b/fused_attn4_m_fa0_d2` **COMPARABLE** (median step |Δ| 0.0012, |Δ held-out at N| 0.0033, step-0 0.0000 SAME-BYTES-CLASS, |Δ loss at step 2| 0.0037, paired rows mean +0.0033 ± 0.0008 SE over 8, favouring arm 1 / anchor 7) — no reference arm ran: the EQUIVALENT band cannot be set, COMPARABLE is the ceiling
- matched_init_sha (B, name-free, canonical slot order): anchor `f7832488926eda91`; `e4b/fused_attn4_m_fa0` same; `e4b/fused_attn4_m_fa1` same; `e4b/fused_attn4_m_fa1_d2` same; `e4b/fused_attn4_m_fa0_d2` same
- frozen base (R5), anchor `e4b/fused_attn4_m`: gate_up nf4/64 sha 8c3b3fd78e89 control detects, down nf4/64 sha 59569d810a1a control detects, q_proj nf4/64+dq sha 5c5e402ce51d control detects
- frozen base `e4b/fused_attn4_shipped_fa0`: gate_up **SAME-BYTES** (nf4/64) control detects, down **SAME-BYTES** (nf4/64) control detects, q_proj **SAME-BYTES** (nf4/64+dq) control detects
- frozen base `e4b/fused_attn4_shipped_fa1`: gate_up **SAME-BYTES** (nf4/64) control detects, down **SAME-BYTES** (nf4/64) control detects, q_proj **SAME-BYTES** (nf4/64+dq) control detects
- frozen base `e4b/fused_attn4_m_fa1`: gate_up **SAME-BYTES** (nf4/64) control detects, down **SAME-BYTES** (nf4/64) control detects, q_proj **SAME-BYTES** (nf4/64+dq) control detects
- frozen base `e4b/fused_attn4_m_fa1_d2`: gate_up **SAME-BYTES** (nf4/64) control detects, down **SAME-BYTES** (nf4/64) control detects, q_proj **SAME-BYTES** (nf4/64+dq) control detects
- frozen base `e4b/fused_attn4_m_fa0_d2`: gate_up **SAME-BYTES** (nf4/64) control detects, down **SAME-BYTES** (nf4/64) control detects, q_proj **SAME-BYTES** (nf4/64+dq) control detects
- frozen base `e4b/fused_attn4_shipped_fa1_d2`: gate_up **SAME-BYTES** (nf4/64) control detects, down **SAME-BYTES** (nf4/64) control detects, q_proj **SAME-BYTES** (nf4/64+dq) control detects
- frozen base `e4b/fused_attn4_shipped_fa0_d2`: gate_up **SAME-BYTES** (nf4/64) control detects, down **SAME-BYTES** (nf4/64) control detects, q_proj **SAME-BYTES** (nf4/64+dq) control detects

## Verdicts
| family | arm | VERDICT | validity | quality | note |
|---|---|---|---|---|---|
| qwen3fieldauto | e4b/fused_attn4_shipped_fa0 | **VALID** | VALID | 0.0044 |  |
| qwen3fieldauto | e4b/fused_attn4_shipped_fa1 | **VALID** | VALID | 0.0045 |  |
| qwen3fieldauto | e4b/fused_attn4_m_fa0 | **VALID** | VALID | 0.0000 |  |
| qwen3fieldauto | e4b/fused_attn4_m_fa1 | **VALID** | VALID | 0.0038 |  |
| qwen3fieldauto | e4b/fused_attn4_m_fa1_d2 | **VALID** | VALID | 0.0039 |  |
| qwen3fieldauto | e4b/fused_attn4_m_fa0_d2 | **VALID** | VALID | 0.0033 |  |
| qwen3fieldauto | e4b/fused_attn4_shipped_fa1_d2 | **VALID** | VALID | 0.0025 |  |
| qwen3fieldauto | e4b/fused_attn4_shipped_fa0_d2 | **VALID** | VALID | 0.0023 |  |

## Predictions P123 / P124 / P125 (TC1-PREREG amendment 50: NF4_QLORA_PAD_BUCKETS 0 vs auto at the field recipe; structure and values, speed reported; scored mechanically)
| prediction | family | verdict | evidence |
|---|---|---|---|
| P123 | qwen3fieldauto | **HELD** | auto never buckets at the field recipe: `fused_attn4_shipped_fa1` bucketed 0 / single 49152 (fa0 single [49152, 49152]); `fused_attn4_shipped_fa1_d2` bucketed 0 / single 49152 (fa0 single [49152, 49152]); `fused_attn4_m_fa1` bucketed 0 / single 49152 (fa0 single [49152, 49152]); `fused_attn4_m_fa1_d2` bucketed 0 / single 49152 (fa0 single [49152, 49152]) |
| P124 | qwen3fieldauto | **HELD** | fused_attn4_m: mean held-out fa1 - fa0 +0.0022 (fa0 [0.7532, 0.7565], fa1 [0.757, 0.7571]); fused_attn4_shipped: mean held-out fa1 - fa0 +0.0002 (fa0 [0.7575, 0.7554], fa1 [0.7576, 0.7557]) |
| P125 | qwen3fieldauto | **HELD** | matched peak fa0 [27.49, 27.505] / fa1 [27.483, 27.487] GB, fa1 - fa0 -0.012 vs <= 0.05 |
| speed (reported, not scored) | qwen3fieldauto | **—** | fused_attn4_m: fa0 [3.1753, 3.2133], fa1 [3.1844, 3.1947] s/step; fused_attn4_shipped: fa0 [2.5127, 2.4659], fa1 [2.4459, 2.4484] s/step |

## Load gate (TC1-PREREG amendment 33): 0 draw(s) voided for host load and run again
Each line: the arm, the attempt, the median host load1 over that attempt's run, the gate. A VOID attempt's files are in loadvoid/; the last attempt of each arm stands whatever its load.

- `qwen3fieldauto/e4b/fused_attn4_shipped_fa0 attempt 0 load1_median 2.08 gate 6.0 status ok over 0`
- `qwen3fieldauto/e4b/fused_attn4_shipped_fa1 attempt 0 load1_median 2.2 gate 6.0 status ok over 0`
- `qwen3fieldauto/e4b/fused_attn4_m_fa0 attempt 0 load1_median 2.1 gate 6.0 status ok over 0`
- `qwen3fieldauto/e4b/fused_attn4_m_fa1 attempt 0 load1_median 2.15 gate 6.0 status ok over 0`
- `qwen3fieldauto/e4b/fused_attn4_m_fa1_d2 attempt 0 load1_median 2.18 gate 6.0 status ok over 0`
- `qwen3fieldauto/e4b/fused_attn4_m_fa0_d2 attempt 0 load1_median 2.11 gate 6.0 status ok over 0`
- `qwen3fieldauto/e4b/fused_attn4_shipped_fa1_d2 attempt 0 load1_median 2.29 gate 6.0 status ok over 0`
- `qwen3fieldauto/e4b/fused_attn4_shipped_fa0_d2 attempt 0 load1_median 2.25 gate 6.0 status ok over 0`
