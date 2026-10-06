# TC1 — Qwen3-30B-A3B on one RTX 5090: e4b vs Unsloth vs HF+PEFT vs axolotl at matched work, init and adapter precision (tc1)
Rule (tc1/TC1-PREREG.md): status per attempt in the vocabulary OK / REFUSED / OOM / INSTALL_FAILED / LOAD_FAULT / HARNESS_ERROR / ALARM / NOT_RUN; VALID/VOID per tp4's predicates plus the matched-set predicates (R3: matched init complete, fp32 adapters, step-0 held-out within 0.005 of e4b/reference_attn4_m); VERDICT exactly one of VALID / VOID / QUALITY_FAIL / OOM / UNSUPPORTED / HARNESS_ERROR / ALARM / NOT_RUN (QUALITY_FAIL = held-out |Δ| at N > 0.05 vs e4b/fused_attn4_m); positions = s/step ratios other/e4b from the medians over both draws (stability |d1-d2|/mean ≤ 5 % e4b / 10 % others, else UNSTABLE), quoted only when both arms are VALID; equivalence vs e4b/reference_attn4_m EQUIVALENT ≤ 0.02 / COMPARABLE ≤ 0.05 / else DIVERGENT; frozen base SAME-BYTES / DIFFERENT / N-A per slot; predictions P1–P10 (+ P1b) scored HELD / FALSIFIED / UNTESTED. VOID never enters a ratio or an equivalence reading. Nothing is licensed; no cross-box number is divided into these.
`versions.txt`
```
e4b 0.48.0 @3556f463535e8413968500f8f1a149405d8c1735 (GitHub main)
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
e4b(t212) 0.48.0 @3556f463535e8413968500f8f1a149405d8c1735
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
 "run_id": "tc1-5090-106",
 "instance_id": "54460139",
 "gpu": "NVIDIA GeForce RTX 5090",
 "driver": "595.58.03",
 "cpu": "Intel(R) Xeon(R) Platinum 8347C CPU @ 2.10GHz",
 "nproc": 72,
 "mem_total_kb": "263547312",
 "cgroup_memory_max": "",
 "disk_root": "overlay         320G   53M  320G   1% /",
 "hostname": "75e93dc939ca",
 "cgroup_cpu_max": "",
 "affinity_cpus": 72,
 "cgroup_cpuset_effective": "",
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
| e4b | fused_attn4_shipped_fb0 | **OK** | VALID | **VALID** | native | native / bfloat16,float32 | — | 60 | 4.575 | 312.9 | 24.673 | 949.9 | 2.0554→0.8173 | 1.9478→0.7571 | 0.0017 | patched 48 / kcalls 768 | 642514944 | 4-bit experts (e4b NF4) + NF4 attention |  |
| e4b | fused_attn4_shipped_fb1 | **OK** | VALID | **VALID** | native | native / bfloat16,float32 | — | 60 | 5.717 | 278.2 | 24.673 | 962.4 | 2.0554→0.8162 | 1.9478→0.7557 | 0.0003 | patched 48 / kcalls 768 | 642514944 | 4-bit experts (e4b NF4) + NF4 attention |  |
| e4b | fused_attn4_m_fb0 | **OK** | VALID | **VALID** | yes | matched:3407 (complete 12480/12480) / float32 | SAME-BYTES-CLASS (0.0000) | 60 | 5.359 | 286.7 | 27.487 | 1111.6 | 2.0554→0.8202 | 1.9478→0.7553 | 0.0000 | patched 48 / kcalls 768 | 642514944 | 4-bit experts (e4b NF4) + NF4 attention |  |
| e4b | fused_attn4_m_fb1 | **OK** | VALID | **VALID** | yes | matched:3407 (complete 12480/12480) / float32 | SAME-BYTES-CLASS (0.0000) | 60 | 5.997 | 265.2 | 27.189 | 1072.5 | 2.0554→0.8159 | 1.9478→0.7580 | 0.0027 | patched 48 / kcalls 768 | 642514944 | 4-bit experts (e4b NF4) + NF4 attention |  |
| e4b | fused_attn4_m_fb1_d2 | **OK** | VALID | **VALID** | yes | matched:3407 (complete 12480/12480) / float32 | SAME-BYTES-CLASS (0.0000) | 60 | 6.205 | 260.0 | 27.189 | 1091.3 | 2.0554→0.8190 | 1.9478→0.7580 | 0.0026 | patched 48 / kcalls 768 | 642514944 | 4-bit experts (e4b NF4) + NF4 attention |  |
| e4b | fused_attn4_m_fb0_d2 | **OK** | VALID | **VALID** | yes | matched:3407 (complete 12480/12480) / float32 | SAME-BYTES-CLASS (0.0000) | 60 | 6.129 | 278.3 | 27.493 | 1137.6 | 2.0554→0.8245 | 1.9478→0.7552 | -0.0001 | patched 48 / kcalls 768 | 642514944 | 4-bit experts (e4b NF4) + NF4 attention |  |
| e4b | fused_attn4_shipped_fb1_d2 | **OK** | VALID | **VALID** | native | native / bfloat16,float32 | — | 60 | 5.567 | 270.4 | 24.673 | 982.5 | 2.0554→0.8175 | 1.9478→0.7541 | -0.0012 | patched 48 / kcalls 768 | 642514944 | 4-bit experts (e4b NF4) + NF4 attention |  |
| e4b | fused_attn4_shipped_fb0_d2 | **OK** | VALID | **VALID** | native | native / bfloat16,float32 | — | 60 | 4.222 | 373.2 | 24.673 | 852.5 | 2.0554→0.8168 | 1.9478→0.7598 | 0.0045 | patched 48 / kcalls 768 | 642514944 | 4-bit experts (e4b NF4) + NF4 attention |  |
- prologue `e4b/fused_attn4_shipped_fb0` **160.2 s** before step 1 (34% of the arm): load_weights 64.2, c1_before 58.0, c1_after 29.2, eval0 20.7; unattributed 2.368; budget 1260.0
- prologue `e4b/fused_attn4_shipped_fb1` **95.3 s** before step 1 (21% of the arm): c1_before 59.6, c1_after 29.2, load_weights 16.3, attn4 5.2; unattributed 2.481; budget 1260.0
- prologue `e4b/fused_attn4_m_fb0` **94.2 s** before step 1 (22% of the arm): c1_before 58.3, c1_after 29.2, load_weights 17.0, adapter_save 10.2; unattributed 1.772; budget 1260.0
- prologue `e4b/fused_attn4_m_fb1` **96.5 s** before step 1 (21% of the arm): c1_before 58.5, c1_after 28.6, load_weights 17.8, adapter_save 6.3; unattributed 2.046; budget 1260.0
- prologue `e4b/fused_attn4_m_fb1_d2` **98.3 s** before step 1 (21% of the arm): c1_before 59.6, c1_after 29.1, load_weights 17.8, attn4 5.3; unattributed 2.493; budget 1260.0
- prologue `e4b/fused_attn4_m_fb0_d2` **97.6 s** before step 1 (22% of the arm): c1_before 58.6, c1_after 28.5, load_weights 18.0, attn4 6.3; unattributed 1.993; budget 1260.0
- prologue `e4b/fused_attn4_shipped_fb1_d2` **94.2 s** before step 1 (21% of the arm): c1_before 59.0, c1_after 30.1, load_weights 16.7, attn4 5.3; unattributed 1.758; budget 1260.0
- prologue `e4b/fused_attn4_shipped_fb0_d2` **91.9 s** before step 1 (26% of the arm): c1_before 57.5, c1_after 29.2, load_weights 16.9, attn4 5.2; unattributed 1.788; budget 1260.0
- draws (R1): `e4b/fused_attn4_shipped_fb0` UNSTABLE (4.575/4.222 s, |Δ|/mean 8.0% vs 5%); `e4b/fused_attn4_shipped_fb1` STABLE (5.717/5.567 s, |Δ|/mean 2.7% vs 5%); `e4b/fused_attn4_m_fb0` UNSTABLE (5.359/6.129 s, |Δ|/mean 13.4% vs 5%); `e4b/fused_attn4_m_fb1` STABLE (5.997/6.205 s, |Δ|/mean 3.4% vs 5%)
- e4b internal parity: NO-ARM
- **NO MATCHED POSITION QUOTED (unsloth)** — not quoted: e4b draws 5.359 / 6.129 s/step differ by 13.4% > 5% (UNSTABLE: reported, not quoted) / unsloth no receipt
- **NO MATCHED POSITION QUOTED (hf)** — not quoted: e4b draws 5.359 / 6.129 s/step differ by 13.4% > 5% (UNSTABLE: reported, not quoted) / hf no receipt
- **NO MATCHED POSITION QUOTED (axolotl)** — not quoted: e4b draws 5.359 / 6.129 s/step differ by 13.4% > 5% (UNSTABLE: reported, not quoted) / axolotl no receipt
- equivalence vs `e4b/fused_attn4_m` (I/H; EQUIVALENT iff both ≤ band = max(0.005, 3 × fused-vs-reference |Δ|) = none (no reference arm: COMPARABLE is the ceiling); COMPARABLE ≤ 0.05; draw-noise floor 0.0015): `e4b/fused_attn4_m_fb1` **COMPARABLE** (median step |Δ| 0.0012, |Δ held-out at N| 0.0027, step-0 0.0000 SAME-BYTES-CLASS, |Δ loss at step 2| 0.0117, paired rows mean +0.0027 ± 0.0018 SE over 8, favouring arm 2 / anchor 6) — no reference arm ran: the EQUIVALENT band cannot be set, COMPARABLE is the ceiling; `e4b/fused_attn4_m_fb1_d2` **COMPARABLE** (median step |Δ| 0.0014, |Δ held-out at N| 0.0026, step-0 0.0000 SAME-BYTES-CLASS, |Δ loss at step 2| 0.0135, paired rows mean +0.0026 ± 0.0019 SE over 8, favouring arm 2 / anchor 6) — no reference arm ran: the EQUIVALENT band cannot be set, COMPARABLE is the ceiling; `e4b/fused_attn4_m_fb0_d2` **COMPARABLE** (median step |Δ| 0.0012, |Δ held-out at N| 0.0001, step-0 0.0000 SAME-BYTES-CLASS, |Δ loss at step 2| 0.0129, paired rows mean -0.0001 ± 0.0015 SE over 8, favouring arm 4 / anchor 4) — no reference arm ran: the EQUIVALENT band cannot be set, COMPARABLE is the ceiling
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
| qwen3fieldbk | e4b/fused_attn4_shipped_fb0 | **VALID** | VALID | 0.0017 |  |
| qwen3fieldbk | e4b/fused_attn4_shipped_fb1 | **VALID** | VALID | 0.0003 |  |
| qwen3fieldbk | e4b/fused_attn4_m_fb0 | **VALID** | VALID | 0.0000 |  |
| qwen3fieldbk | e4b/fused_attn4_m_fb1 | **VALID** | VALID | 0.0027 |  |
| qwen3fieldbk | e4b/fused_attn4_m_fb1_d2 | **VALID** | VALID | 0.0026 |  |
| qwen3fieldbk | e4b/fused_attn4_m_fb0_d2 | **VALID** | VALID | -0.0001 |  |
| qwen3fieldbk | e4b/fused_attn4_shipped_fb1_d2 | **VALID** | VALID | -0.0012 |  |
| qwen3fieldbk | e4b/fused_attn4_shipped_fb0_d2 | **VALID** | VALID | 0.0045 |  |

## Predictions P119 / P120 / P121 / P122 (TC1-PREREG amendment 49: the LoRA delta one padded block vs buckets at the field recipe; scored mechanically)
| prediction | family | verdict | evidence |
|---|---|---|---|
| P121 | qwen3fieldbk | **UNTESTED** | matched: two stable VALID draws a side are registered -- fb0 UNSTABLE: draws 5.359 / 6.129 s/step differ by 13.4% > 5% (UNSTABLE: reported, not quoted); fb1 STABLE: |
| P119 | qwen3fieldbk | **UNTESTED** | matched: two stable VALID draws a side are registered -- fb0 UNSTABLE: draws 5.359 / 6.129 s/step differ by 13.4% > 5% (UNSTABLE: reported, not quoted); fb1 STABLE: |
| P120 | qwen3fieldbk | **UNTESTED** | shipped: two stable VALID draws a side are registered -- fb0 UNSTABLE: draws 4.575 / 4.222 s/step differ by 8.0% > 5% (UNSTABLE: reported, not quoted); fb1 STABLE: |
| P122 | qwen3fieldbk | **UNTESTED** | matched: fb0 UNSTABLE: draws 5.359 / 6.129 s/step differ by 13.4% > 5% (UNSTABLE: reported, not quoted); fb1 STABLE:; shipped: fb0 UNSTABLE: draws 4.575 / 4.222 s/step differ by 8.0% > 5% (UNSTABLE: reported, not quoted); fb1 STABLE: |

## Load gate (TC1-PREREG amendment 33): 0 draw(s) voided for host load and run again
Each line: the arm, the attempt, the median host load1 over that attempt's run, the gate. A VOID attempt's files are in loadvoid/; the last attempt of each arm stands whatever its load.

- `qwen3fieldbk/e4b/fused_attn4_shipped_fb0 attempt 0 load1_median 2.05 gate 6.0 status ok over 0`
- `qwen3fieldbk/e4b/fused_attn4_shipped_fb1 attempt 0 load1_median 1.96 gate 6.0 status ok over 0`
- `qwen3fieldbk/e4b/fused_attn4_m_fb0 attempt 0 load1_median 2.06 gate 6.0 status ok over 0`
- `qwen3fieldbk/e4b/fused_attn4_m_fb1 attempt 0 load1_median 2.33 gate 6.0 status ok over 0`
- `qwen3fieldbk/e4b/fused_attn4_m_fb1_d2 attempt 0 load1_median 2.99 gate 6.0 status ok over 0`
- `qwen3fieldbk/e4b/fused_attn4_m_fb0_d2 attempt 0 load1_median 1.65 gate 6.0 status ok over 0`
- `qwen3fieldbk/e4b/fused_attn4_shipped_fb1_d2 attempt 0 load1_median 1.83 gate 6.0 status ok over 0`
- `qwen3fieldbk/e4b/fused_attn4_shipped_fb0_d2 attempt 0 load1_median 2.27 gate 6.0 status ok over 0`
