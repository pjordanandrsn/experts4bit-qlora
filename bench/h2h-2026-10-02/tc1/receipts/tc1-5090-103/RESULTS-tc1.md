# TC1 — Qwen3-30B-A3B on one RTX 5090: e4b vs Unsloth vs HF+PEFT vs axolotl at matched work, init and adapter precision (tc1)
Rule (tc1/TC1-PREREG.md): status per attempt in the vocabulary OK / REFUSED / OOM / INSTALL_FAILED / LOAD_FAULT / HARNESS_ERROR / ALARM / NOT_RUN; VALID/VOID per tp4's predicates plus the matched-set predicates (R3: matched init complete, fp32 adapters, step-0 held-out within 0.005 of e4b/reference_attn4_m); VERDICT exactly one of VALID / VOID / QUALITY_FAIL / OOM / UNSUPPORTED / HARNESS_ERROR / ALARM / NOT_RUN (QUALITY_FAIL = held-out |Δ| at N > 0.05 vs e4b/fused_attn4_m); positions = s/step ratios other/e4b from the medians over both draws (stability |d1-d2|/mean ≤ 5 % e4b / 10 % others, else UNSTABLE), quoted only when both arms are VALID; equivalence vs e4b/reference_attn4_m EQUIVALENT ≤ 0.02 / COMPARABLE ≤ 0.05 / else DIVERGENT; frozen base SAME-BYTES / DIFFERENT / N-A per slot; predictions P1–P10 (+ P1b) scored HELD / FALSIFIED / UNTESTED. VOID never enters a ratio or an equivalence reading. Nothing is licensed; no cross-box number is divided into these.
`versions.txt`
```
e4b 0.48.0 @9170c926630e3005efe4a944f4d1a9c0600ed42c (GitHub main)
gnf4 0.41.0 @d3e7788939fbad98a63a8c06badd253764c6dac0 (GitHub main)
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
e4b(t212) 0.48.0 @9170c926630e3005efe4a944f4d1a9c0600ed42c
gnf4(t212) 0.41.0 @d3e7788939fbad98a63a8c06badd253764c6dac0
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
 "run_id": "tc1-5090-103",
 "instance_id": "54449986",
 "gpu": "NVIDIA GeForce RTX 5090",
 "driver": "590.48.01",
 "cpu": "AMD Ryzen Threadripper 3990X 64-Core Processor",
 "nproc": 128,
 "mem_total_kb": "263750200",
 "cgroup_memory_max": "183335124992",
 "disk_root": "overlay         3.6T  2.5T  949G  73% /",
 "hostname": "c2644294b965",
 "cgroup_cpu_max": "6143999 100000",
 "affinity_cpus": 128,
 "cgroup_cpuset_effective": "0-127",
 "registered_gpu_class": "5090",
 "prereg": "tc1/TC1-PREREG.md",
 "local_box": null,
 "local_snapshot": null
}
```

### Qwen3-30B-A3B (amendment 49: grouped-nf4-gemm's LoRA delta one padded block vs buckets at the field recipe, venv-unsloth) (`qwen3fieldbk`, registered n_layers 48, attention census 192)
- model `Qwen/Qwen3-30B-A3B` @ `ad44e777bcd1`; tokens sha `bfc742f67e37`; N=60; fixture template alpaca seq 2048 micro-batch 2 × accum 4 lr 0.0002 r 16 α 16 optimizer adamw_8bit(lr=0.0002, weight_decay=0.001) schedule=linear warmup_steps=5 autocast False; e4b trainable 642514944; box_class RTX 5090 gpu NVIDIA GeForce RTX 5090
| framework | arm | status | validity | **VERDICT** | matched | init / dtype | step-0 class | N | s/step med(11+) | tok/s | peak GB | J/step | train first→last | held-out 0→final | quality Δ | engagement | trainable | regime | reason / why |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| e4b | fused_attn4_shipped_fb0 | **OK** | VALID | **VALID** | native | native / bfloat16,float32 | — | 60 | 2.970 | 502.3 | 24.673 | 828.6 | 2.0554→0.8183 | 1.9478→0.7547 | -0.0033 | patched 48 / kcalls 768 | 642514944 | 4-bit experts (e4b NF4) + NF4 attention |  |
| e4b | fused_attn4_shipped_fb1 | **OK** | VALID | **VALID** | native | native / bfloat16,float32 | — | 60 | 3.685 | 425.2 | 24.673 | 820.4 | 2.0554→0.8168 | 1.9478→0.7553 | -0.0028 | patched 48 / kcalls 768 | 642514944 | 4-bit experts (e4b NF4) + NF4 attention |  |
| e4b | fused_attn4_m_fb0 | **OK** | VALID | **VALID** | yes | matched:3407 (complete 12480/12480) / float32 | SAME-BYTES-CLASS (0.0000) | 60 | 3.524 | 451.9 | 27.490 | 1020.4 | 2.0554→0.8180 | 1.9478→0.7581 | 0.0000 | patched 48 / kcalls 768 | 642514944 | 4-bit experts (e4b NF4) + NF4 attention |  |
| e4b | fused_attn4_m_fb1 | **OK** | VALID | **VALID** | yes | matched:3407 (complete 12480/12480) / float32 | SAME-BYTES-CLASS (0.0000) | 60 | 4.218 | 375.3 | 27.189 | 942.8 | 2.0554→0.8169 | 1.9478→0.7558 | -0.0023 | patched 48 / kcalls 768 | 642514944 | 4-bit experts (e4b NF4) + NF4 attention |  |
| e4b | fused_attn4_m_fb1_d2 | **OK** | VALID | **VALID** | yes | matched:3407 (complete 12480/12480) / float32 | SAME-BYTES-CLASS (0.0000) | 60 | 4.159 | 385.8 | 27.189 | 925.7 | 2.0554→0.8173 | 1.9478→0.7587 | 0.0006 | patched 48 / kcalls 768 | 642514944 | 4-bit experts (e4b NF4) + NF4 attention |  |
| e4b | fused_attn4_m_fb0_d2 | **OK** | VALID | **VALID** | yes | matched:3407 (complete 12480/12480) / float32 | SAME-BYTES-CLASS (0.0000) | 60 | 4.342 | 389.9 | 27.505 | 1035.4 | 2.0554→0.8186 | 1.9478→0.7573 | -0.0008 | patched 48 / kcalls 768 | 642514944 | 4-bit experts (e4b NF4) + NF4 attention |  |
| e4b | fused_attn4_shipped_fb1_d2 | **OK** | VALID | **VALID** | native | native / bfloat16,float32 | — | 60 | 3.709 | 421.2 | 24.673 | 825.6 | 2.0554→0.8186 | 1.9478→0.7558 | -0.0023 | patched 48 / kcalls 768 | 642514944 | 4-bit experts (e4b NF4) + NF4 attention |  |
| e4b | fused_attn4_shipped_fb0_d2 | **OK** | VALID | **VALID** | native | native / bfloat16,float32 | — | 60 | 3.358 | 483.0 | 24.673 | 838.7 | 2.0554→0.8182 | 1.9478→0.7562 | -0.0018 | patched 48 / kcalls 768 | 642514944 | 4-bit experts (e4b NF4) + NF4 attention |  |
- prologue `e4b/fused_attn4_shipped_fb0` **127.6 s** before step 1 (40% of the arm): c1_before 57.1, load_weights 40.7, c1_after 29.1, eval0 13.7; unattributed 2.119; budget 1260.0
- prologue `e4b/fused_attn4_shipped_fb1` **102.0 s** before step 1 (31% of the arm): c1_before 60.7, c1_after 28.8, load_weights 23.0, attn4 6.1; unattributed 2.158; budget 1260.0
- prologue `e4b/fused_attn4_m_fb0` **100.0 s** before step 1 (32% of the arm): c1_before 58.6, c1_after 30.1, load_weights 21.5, attn4 6.2; unattributed 2.102; budget 1260.0
- prologue `e4b/fused_attn4_m_fb1` **99.0 s** before step 1 (28% of the arm): c1_before 59.5, c1_after 29.4, load_weights 19.6, attn4 6.0; unattributed 2.207; budget 1260.0
- prologue `e4b/fused_attn4_m_fb1_d2` **99.4 s** before step 1 (28% of the arm): c1_before 59.7, c1_after 29.3, load_weights 20.1, attn4 6.2; unattributed 1.895; budget 1260.0
- prologue `e4b/fused_attn4_m_fb0_d2` **97.9 s** before step 1 (28% of the arm): c1_before 59.3, c1_after 36.7, load_weights 19.1, attn4 6.2; unattributed 1.901; budget 1260.0
- prologue `e4b/fused_attn4_shipped_fb1_d2` **107.4 s** before step 1 (32% of the arm): c1_before 60.1, c1_after 28.5, load_weights 24.1, attn4 8.5; unattributed 2.47; budget 1260.0
- prologue `e4b/fused_attn4_shipped_fb0_d2` **96.3 s** before step 1 (32% of the arm): c1_before 57.1, c1_after 36.4, load_weights 21.7, attn4 6.1; unattributed 1.897; budget 1260.0
- draws (R1): `e4b/fused_attn4_shipped_fb0` UNSTABLE (2.970/3.358 s, |Δ|/mean 12.3% vs 5%); `e4b/fused_attn4_shipped_fb1` STABLE (3.685/3.709 s, |Δ|/mean 0.7% vs 5%); `e4b/fused_attn4_m_fb0` UNSTABLE (3.524/4.342 s, |Δ|/mean 20.8% vs 5%); `e4b/fused_attn4_m_fb1` STABLE (4.218/4.159 s, |Δ|/mean 1.4% vs 5%)
- e4b internal parity: NO-ARM
- **NO MATCHED POSITION QUOTED (unsloth)** — not quoted: e4b draws 3.524 / 4.342 s/step differ by 20.8% > 5% (UNSTABLE: reported, not quoted) / unsloth no receipt
- **NO MATCHED POSITION QUOTED (hf)** — not quoted: e4b draws 3.524 / 4.342 s/step differ by 20.8% > 5% (UNSTABLE: reported, not quoted) / hf no receipt
- **NO MATCHED POSITION QUOTED (axolotl)** — not quoted: e4b draws 3.524 / 4.342 s/step differ by 20.8% > 5% (UNSTABLE: reported, not quoted) / axolotl no receipt
- equivalence vs `e4b/fused_attn4_m` (I/H; EQUIVALENT iff both ≤ band = max(0.005, 3 × fused-vs-reference |Δ|) = none (no reference arm: COMPARABLE is the ceiling); COMPARABLE ≤ 0.05; draw-noise floor 0.0029): `e4b/fused_attn4_m_fb1` **COMPARABLE** (median step |Δ| 0.0013, |Δ held-out at N| 0.0023, step-0 0.0000 SAME-BYTES-CLASS, |Δ loss at step 2| 0.0067, paired rows mean -0.0023 ± 0.0026 SE over 8, favouring arm 6 / anchor 2) — no reference arm ran: the EQUIVALENT band cannot be set, COMPARABLE is the ceiling; `e4b/fused_attn4_m_fb1_d2` **COMPARABLE** (median step |Δ| 0.0012, |Δ held-out at N| 0.0006, step-0 0.0000 SAME-BYTES-CLASS, |Δ loss at step 2| 0.0042, paired rows mean +0.0006 ± 0.0015 SE over 8, favouring arm 3 / anchor 5) — no reference arm ran: the EQUIVALENT band cannot be set, COMPARABLE is the ceiling; `e4b/fused_attn4_m_fb0_d2` **COMPARABLE** (median step |Δ| 0.0012, |Δ held-out at N| 0.0008, step-0 0.0000 SAME-BYTES-CLASS, |Δ loss at step 2| 0.0122, paired rows mean -0.0008 ± 0.0022 SE over 8, favouring arm 5 / anchor 3) — no reference arm ran: the EQUIVALENT band cannot be set, COMPARABLE is the ceiling
- matched_init_sha (B, name-free, canonical slot order): anchor `f7832488926eda91`; `e4b/fused_attn4_m_fb0` same; `e4b/fused_attn4_m_fb1` same; `e4b/fused_attn4_m_fb1_d2` same; `e4b/fused_attn4_m_fb0_d2` same
- frozen base (R5), anchor `e4b/fused_attn4_m`: gate_up nf4/64 sha 8c3b3fd78e89 control detects, down nf4/64 sha 59569d810a1a control detects, q_proj nf4/64+dq sha 5c5e402ce51d control detects
- frozen base `e4b/fused_attn4_shipped_fb0`: gate_up **SAME-BYTES** (nf4/64) control detects, down **SAME-BYTES** (nf4/64) control detects, q_proj **SAME-BYTES** (nf4/64+dq) control detects
- frozen base `e4b/fused_attn4_shipped_fb1`: gate_up **SAME-BYTES** (nf4/64) control detects, down **SAME-BYTES** (nf4/64) control detects, q_proj **SAME-BYTES** (nf4/64+dq) control detects
- frozen base `e4b/fused_attn4_m_fb1`: gate_up **SAME-BYTES** (nf4/64) control detects, down **SAME-BYTES** (nf4/64) control detects, q_proj **SAME-BYTES** (nf4/64+dq) control detects
- frozen base `e4b/fused_attn4_m_fb1_d2`: gate_up **SAME-BYTES** (nf4/64) control detects, down **SAME-BYTES** (nf4/64) control detects, q_proj **SAME-BYTES** (nf4/64+dq) control detects
- frozen base `e4b/fused_attn4_m_fb0_d2`: gate_up **SAME-BYTES** (nf4/64) control detects, down **SAME-BYTES** (nf4/64) control detects, q_proj **SAME-BYTES** (nf4/64+dq) control detects
- frozen base `e4b/fused_attn4_shipped_fb1_d2`: gate_up **SAME-BYTES** (nf4/64) control detects, down **SAME-BYTES** (nf4/64) control detects, q_proj **SAME-BYTES** (nf4/64+dq) control detects
- frozen base `e4b/fused_attn4_shipped_fb0_d2`: gate_up **SAME-BYTES** (nf4/64) control detects, down **SAME-BYTES** (nf4/64) control detects, q_proj **SAME-BYTES** (nf4/64+dq) control detects

## Verdicts
| family | arm | VERDICT | validity | quality | note |
|---|---|---|---|---|---|
| qwen3fieldbk | e4b/fused_attn4_shipped_fb0 | **VALID** | VALID | -0.0033 |  |
| qwen3fieldbk | e4b/fused_attn4_shipped_fb1 | **VALID** | VALID | -0.0028 |  |
| qwen3fieldbk | e4b/fused_attn4_m_fb0 | **VALID** | VALID | 0.0000 |  |
| qwen3fieldbk | e4b/fused_attn4_m_fb1 | **VALID** | VALID | -0.0023 |  |
| qwen3fieldbk | e4b/fused_attn4_m_fb1_d2 | **VALID** | VALID | 0.0006 |  |
| qwen3fieldbk | e4b/fused_attn4_m_fb0_d2 | **VALID** | VALID | -0.0008 |  |
| qwen3fieldbk | e4b/fused_attn4_shipped_fb1_d2 | **VALID** | VALID | -0.0023 |  |
| qwen3fieldbk | e4b/fused_attn4_shipped_fb0_d2 | **VALID** | VALID | -0.0018 |  |

## Predictions P119 / P120 / P121 / P122 (TC1-PREREG amendment 49: the LoRA delta one padded block vs buckets at the field recipe; scored mechanically)
| prediction | family | verdict | evidence |
|---|---|---|---|
| P121 | qwen3fieldbk | **UNTESTED** | matched: two stable VALID draws a side are registered -- fb0 UNSTABLE: draws 3.524 / 4.342 s/step differ by 20.8% > 5% (UNSTABLE: reported, not quoted); fb1 STABLE: |
| P119 | qwen3fieldbk | **UNTESTED** | matched: two stable VALID draws a side are registered -- fb0 UNSTABLE: draws 3.524 / 4.342 s/step differ by 20.8% > 5% (UNSTABLE: reported, not quoted); fb1 STABLE: |
| P120 | qwen3fieldbk | **UNTESTED** | shipped: two stable VALID draws a side are registered -- fb0 UNSTABLE: draws 2.970 / 3.358 s/step differ by 12.3% > 5% (UNSTABLE: reported, not quoted); fb1 STABLE: |
| P122 | qwen3fieldbk | **UNTESTED** | matched: fb0 UNSTABLE: draws 3.524 / 4.342 s/step differ by 20.8% > 5% (UNSTABLE: reported, not quoted); fb1 STABLE:; shipped: fb0 UNSTABLE: draws 2.970 / 3.358 s/step differ by 12.3% > 5% (UNSTABLE: reported, not quoted); fb1 STABLE: |

## Load gate (TC1-PREREG amendment 33): 1 draw(s) voided for host load and run again
Each line: the arm, the attempt, the median host load1 over that attempt's run, the gate. A VOID attempt's files are in loadvoid/; the last attempt of each arm stands whatever its load.

- `qwen3fieldbk/e4b/fused_attn4_shipped_fb0 attempt 0 load1_median 1.82 gate 6.0 status ok over 0`
- `qwen3fieldbk/e4b/fused_attn4_shipped_fb1 attempt 0 load1_median 4.38 gate 6.0 status ok over 0`
- `qwen3fieldbk/e4b/fused_attn4_m_fb0 attempt 0 load1_median 1.82 gate 6.0 status ok over 0`
- `qwen3fieldbk/e4b/fused_attn4_m_fb1 attempt 0 load1_median 1.93 gate 6.0 status ok over 0`
- `qwen3fieldbk/e4b/fused_attn4_m_fb1_d2 attempt 0 load1_median 1.82 gate 6.0 status ok over 0`
- `qwen3fieldbk/e4b/fused_attn4_m_fb0_d2 attempt 0 load1_median 3.78 gate 6.0 status ok over 0`
- `qwen3fieldbk/e4b/fused_attn4_shipped_fb1_d2 attempt 0 load1_median 78.11 gate 6.0 status ok over 1`
- `qwen3fieldbk/e4b/fused_attn4_shipped_fb1_d2 attempt 0 VOID (host load1 median 78.11 > 6.0): re-run 1 of 2`
- `qwen3fieldbk/e4b/fused_attn4_shipped_fb1_d2 attempt 1 load1_median 4.68 gate 6.0 status ok over 0`
- `qwen3fieldbk/e4b/fused_attn4_shipped_fb0_d2 attempt 0 load1_median 1.65 gate 6.0 status ok over 0`
