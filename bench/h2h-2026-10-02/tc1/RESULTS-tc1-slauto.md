# TC1 — Qwen3-30B-A3B on one RTX 5090: e4b vs Unsloth vs HF+PEFT vs axolotl at matched work, init and adapter precision (tc1)
Rule (tc1/TC1-PREREG.md): status per attempt in the vocabulary OK / REFUSED / OOM / INSTALL_FAILED / LOAD_FAULT / HARNESS_ERROR / ALARM / NOT_RUN; VALID/VOID per tp4's predicates plus the matched-set predicates (R3: matched init complete, fp32 adapters, step-0 held-out within 0.005 of e4b/reference_attn4_m); VERDICT exactly one of VALID / VOID / QUALITY_FAIL / OOM / UNSUPPORTED / HARNESS_ERROR / ALARM / NOT_RUN (QUALITY_FAIL = held-out |Δ| at N > 0.05 vs e4b/fused_attn4_m); positions = s/step ratios other/e4b from the medians over both draws (stability |d1-d2|/mean ≤ 5 % e4b / 10 % others, else UNSTABLE), quoted only when both arms are VALID; equivalence vs e4b/reference_attn4_m EQUIVALENT ≤ 0.02 / COMPARABLE ≤ 0.05 / else DIVERGENT; frozen base SAME-BYTES / DIFFERENT / N-A per slot; predictions P1–P10 (+ P1b) scored HELD / FALSIFIED / UNTESTED. VOID never enters a ratio or an equivalence reading. Nothing is licensed; no cross-box number is divided into these.
`versions.txt`
```
e4b 0.50.0 @d0ae24368609d2c631eaaa544bb7f88a3c8019b5 (GitHub main)
gnf4 0.43.0 @3ce2ecd8ae20725a6b13ccc6d06c6a85ba5ce3ac (GitHub main)
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
e4b(t212) 0.50.0 @d0ae24368609d2c631eaaa544bb7f88a3c8019b5
gnf4(t212) 0.43.0 @3ce2ecd8ae20725a6b13ccc6d06c6a85ba5ce3ac
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
 "run_id": "tc1-5090-139",
 "instance_id": "54893119",
 "gpu": "NVIDIA GeForce RTX 5090",
 "driver": "595.91.07",
 "cpu": "AMD EPYC 7763 64-Core Processor",
 "nproc": 256,
 "mem_total_kb": "1056597456",
 "cgroup_memory_max": "519337672704",
 "disk_root": "overlay         320G   59M  320G   1% /",
 "hostname": "ca447cbc5d8c",
 "cgroup_cpu_max": "6143999 100000",
 "affinity_cpus": 256,
 "cgroup_cpuset_effective": "0-255",
 "registered_gpu_class": "5090",
 "prereg": "tc1/TC1-PREREG.md",
 "local_box": null,
 "local_snapshot": null
}
```

### Qwen3-30B-A3B (amendment 71: NF4_QLORA_SINGLE_LADDER 0 vs auto at the field recipe, shipped and matched arms; profiled) (`qwen3slauto`, registered n_layers 48, attention census 192)
- model `Qwen/Qwen3-30B-A3B` @ `ad44e777bcd1`; tokens sha `bfc742f67e37`; N=60; fixture template alpaca seq 2048 micro-batch 2 × accum 4 lr 0.0002 r 16 α 16 optimizer adamw_8bit(lr=0.0002, weight_decay=0.001) schedule=linear warmup_steps=5 autocast False; e4b trainable 642514944; box_class RTX 5090 gpu NVIDIA GeForce RTX 5090
| framework | arm | status | validity | **VERDICT** | matched | init / dtype | step-0 class | N | s/step med(11+) | tok/s | peak GB | J/step | train first→last | held-out 0→final | quality Δ | engagement | trainable | regime | reason / why |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| e4b | fused_attn4_shipped_l0 | **OK** | VALID | **VALID** | native | native / bfloat16,float32 | — | 60 | 2.991 | 500.9 | 23.321 | 504.6 | 2.0722→0.8197 | 1.9592→0.7565 | 0.0022 | patched 48 / kcalls 768 | 642514944 | 4-bit experts (e4b NF4) + NF4 attention |  |
| e4b | fused_attn4_shipped_la | **OK** | VALID | **VALID** | native | native / bfloat16,float32 | — | 60 | 2.983 | 510.2 | 23.321 | 488.3 | 2.0722→0.8177 | 1.9592→0.7569 | 0.0026 | patched 48 / kcalls 768 | 642514944 | 4-bit experts (e4b NF4) + NF4 attention |  |
| e4b | fused_attn4_m_l0 | **OK** | VALID | **VALID** | yes | matched:3407 (complete 12480/12480) / float32 | SAME-BYTES-CLASS (0.0000) | 60 | 3.463 | 454.6 | 26.141 | 622.4 | 2.0722→0.8142 | 1.9592→0.7543 | 0.0000 | patched 48 / kcalls 768 | 642514944 | 4-bit experts (e4b NF4) + NF4 attention |  |
| e4b | fused_attn4_m_la | **OK** | VALID | **VALID** | yes | matched:3407 (complete 12480/12480) / float32 | SAME-BYTES-CLASS (0.0000) | 60 | 3.033 | 515.7 | 26.495 | 562.9 | 2.0722→0.8187 | 1.9592→0.7558 | 0.0014 | patched 48 / kcalls 768 | 642514944 | 4-bit experts (e4b NF4) + NF4 attention |  |
| e4b | fused_attn4_m_la_d2 | **OK** | VALID | **VALID** | yes | matched:3407 (complete 12480/12480) / float32 | SAME-BYTES-CLASS (0.0000) | 60 | 3.024 | 519.9 | 26.495 | 597.0 | 2.0722→0.8236 | 1.9592→0.7599 | 0.0055 | patched 48 / kcalls 768 | 642514944 | 4-bit experts (e4b NF4) + NF4 attention |  |
| e4b | fused_attn4_m_l0_d2 | **OK** | VALID | **VALID** | yes | matched:3407 (complete 12480/12480) / float32 | SAME-BYTES-CLASS (0.0000) | 60 | 3.718 | 425.2 | 26.159 | 650.8 | 2.0722→0.8169 | 1.9592→0.7586 | 0.0043 | patched 48 / kcalls 768 | 642514944 | 4-bit experts (e4b NF4) + NF4 attention |  |
| e4b | fused_attn4_shipped_la_d2 | **OK** | VALID | **VALID** | native | native / bfloat16,float32 | — | 60 | 3.279 | 478.3 | 23.321 | 496.9 | 2.0722→0.8181 | 1.9592→0.7569 | 0.0025 | patched 48 / kcalls 768 | 642514944 | 4-bit experts (e4b NF4) + NF4 attention |  |
| e4b | fused_attn4_shipped_l0_d2 | **OK** | VALID | **VALID** | native | native / bfloat16,float32 | — | 60 | 3.289 | 474.1 | 23.321 | 453.6 | 2.0722→0.8182 | 1.9592→0.7544 | 0.0000 | patched 48 / kcalls 768 | 642514944 | 4-bit experts (e4b NF4) + NF4 attention |  |
- prologue `e4b/fused_attn4_shipped_l0` **103.8 s** before step 1 (24% of the arm): c1_before 54.8, load_weights 19.8, c1_after 19.6, eval0 12.2; unattributed 1.968; budget 1260.0
- prologue `e4b/fused_attn4_shipped_la` **104.2 s** before step 1 (23% of the arm): c1_before 58.4, load_weights 23.2, c1_after 18.1, attn4 6.5; unattributed 2.217; budget 1260.0
- prologue `e4b/fused_attn4_m_l0` **95.4 s** before step 1 (21% of the arm): c1_before 55.6, load_weights 19.9, c1_after 17.7, attn4 5.8; unattributed 1.919; budget 1260.0
- prologue `e4b/fused_attn4_m_la` **95.8 s** before step 1 (22% of the arm): c1_before 56.0, load_weights 20.3, c1_after 17.2, attn4 5.6; unattributed 1.952; budget 1260.0
- prologue `e4b/fused_attn4_m_la_d2` **94.7 s** before step 1 (22% of the arm): c1_before 55.5, load_weights 19.6, c1_after 19.2, attn4 5.7; unattributed 2.007; budget 1260.0
- prologue `e4b/fused_attn4_m_l0_d2` **98.8 s** before step 1 (21% of the arm): c1_before 56.8, load_weights 21.0, c1_after 17.9, attn4 6.2; unattributed 2.006; budget 1260.0
- prologue `e4b/fused_attn4_shipped_la_d2` **102.7 s** before step 1 (23% of the arm): c1_before 61.9, c1_after 24.7, load_weights 19.9, attn4 6.4; unattributed 2.235; budget 1260.0
- prologue `e4b/fused_attn4_shipped_l0_d2` **101.3 s** before step 1 (22% of the arm): c1_before 61.1, load_weights 19.7, c1_after 19.5, attn4 6.3; unattributed 2.183; budget 1260.0
- draws (R1): `e4b/fused_attn4_shipped_l0` UNSTABLE (2.991/3.289 s, |Δ|/mean 9.5% vs 5%); `e4b/fused_attn4_shipped_la` UNSTABLE (2.983/3.279 s, |Δ|/mean 9.5% vs 5%); `e4b/fused_attn4_m_l0` UNSTABLE (3.463/3.718 s, |Δ|/mean 7.1% vs 5%); `e4b/fused_attn4_m_la` STABLE (3.033/3.024 s, |Δ|/mean 0.3% vs 5%)
- e4b internal parity: NO-ARM
- **NO MATCHED POSITION QUOTED (unsloth)** — not quoted: e4b draws 3.463 / 3.718 s/step differ by 7.1% > 5% (UNSTABLE: reported, not quoted) / unsloth no receipt
- **NO MATCHED POSITION QUOTED (hf)** — not quoted: e4b draws 3.463 / 3.718 s/step differ by 7.1% > 5% (UNSTABLE: reported, not quoted) / hf no receipt
- **NO MATCHED POSITION QUOTED (axolotl)** — not quoted: e4b draws 3.463 / 3.718 s/step differ by 7.1% > 5% (UNSTABLE: reported, not quoted) / axolotl no receipt
- equivalence vs `e4b/fused_attn4_m` (I/H; EQUIVALENT iff both ≤ band = max(0.005, 3 × fused-vs-reference |Δ|) = none (no reference arm: COMPARABLE is the ceiling); COMPARABLE ≤ 0.05; draw-noise floor 0.0041): `e4b/fused_attn4_m_la` **COMPARABLE** (median step |Δ| 0.0010, |Δ held-out at N| 0.0014, step-0 0.0000 SAME-BYTES-CLASS, |Δ loss at step 2| 0.0145, paired rows mean +0.0014 ± 0.0017 SE over 8, favouring arm 3 / anchor 5) — no reference arm ran: the EQUIVALENT band cannot be set, COMPARABLE is the ceiling; `e4b/fused_attn4_m_la_d2` **COMPARABLE** (median step |Δ| 0.0011, |Δ held-out at N| 0.0055, step-0 0.0000 SAME-BYTES-CLASS, |Δ loss at step 2| 0.0058, paired rows mean +0.0055 ± 0.0023 SE over 8, favouring arm 2 / anchor 6) — no reference arm ran: the EQUIVALENT band cannot be set, COMPARABLE is the ceiling; `e4b/fused_attn4_m_l0_d2` **COMPARABLE** (median step |Δ| 0.0016, |Δ held-out at N| 0.0043, step-0 0.0000 SAME-BYTES-CLASS, |Δ loss at step 2| 0.0118, paired rows mean +0.0043 ± 0.0015 SE over 8, favouring arm 1 / anchor 7) — no reference arm ran: the EQUIVALENT band cannot be set, COMPARABLE is the ceiling
- matched_init_sha (B, name-free, canonical slot order): anchor `f7832488926eda91`; `e4b/fused_attn4_m_l0` same; `e4b/fused_attn4_m_la` same; `e4b/fused_attn4_m_la_d2` same; `e4b/fused_attn4_m_l0_d2` same
- frozen base (R5), anchor `e4b/fused_attn4_m`: gate_up nf4/64+dq sha e2bed944143e control detects, down nf4/64+dq sha f2ade29dafdb control detects, q_proj nf4/64+dq sha 5c5e402ce51d control detects
- frozen base `e4b/fused_attn4_shipped_l0`: gate_up **SAME-BYTES** (nf4/64+dq) control detects, down **SAME-BYTES** (nf4/64+dq) control detects, q_proj **SAME-BYTES** (nf4/64+dq) control detects
- frozen base `e4b/fused_attn4_shipped_la`: gate_up **SAME-BYTES** (nf4/64+dq) control detects, down **SAME-BYTES** (nf4/64+dq) control detects, q_proj **SAME-BYTES** (nf4/64+dq) control detects
- frozen base `e4b/fused_attn4_m_la`: gate_up **SAME-BYTES** (nf4/64+dq) control detects, down **SAME-BYTES** (nf4/64+dq) control detects, q_proj **SAME-BYTES** (nf4/64+dq) control detects
- frozen base `e4b/fused_attn4_m_la_d2`: gate_up **SAME-BYTES** (nf4/64+dq) control detects, down **SAME-BYTES** (nf4/64+dq) control detects, q_proj **SAME-BYTES** (nf4/64+dq) control detects
- frozen base `e4b/fused_attn4_m_l0_d2`: gate_up **SAME-BYTES** (nf4/64+dq) control detects, down **SAME-BYTES** (nf4/64+dq) control detects, q_proj **SAME-BYTES** (nf4/64+dq) control detects
- frozen base `e4b/fused_attn4_shipped_la_d2`: gate_up **SAME-BYTES** (nf4/64+dq) control detects, down **SAME-BYTES** (nf4/64+dq) control detects, q_proj **SAME-BYTES** (nf4/64+dq) control detects
- frozen base `e4b/fused_attn4_shipped_l0_d2`: gate_up **SAME-BYTES** (nf4/64+dq) control detects, down **SAME-BYTES** (nf4/64+dq) control detects, q_proj **SAME-BYTES** (nf4/64+dq) control detects

## Verdicts
| family | arm | VERDICT | validity | quality | note |
|---|---|---|---|---|---|
| qwen3slauto | e4b/fused_attn4_shipped_l0 | **VALID** | VALID | 0.0022 |  |
| qwen3slauto | e4b/fused_attn4_shipped_la | **VALID** | VALID | 0.0026 |  |
| qwen3slauto | e4b/fused_attn4_m_l0 | **VALID** | VALID | 0.0000 |  |
| qwen3slauto | e4b/fused_attn4_m_la | **VALID** | VALID | 0.0014 |  |
| qwen3slauto | e4b/fused_attn4_m_la_d2 | **VALID** | VALID | 0.0055 |  |
| qwen3slauto | e4b/fused_attn4_m_l0_d2 | **VALID** | VALID | 0.0043 |  |
| qwen3slauto | e4b/fused_attn4_shipped_la_d2 | **VALID** | VALID | 0.0025 |  |
| qwen3slauto | e4b/fused_attn4_shipped_l0_d2 | **VALID** | VALID | 0.0000 |  |

## Amendment 71: NF4_QLORA_SINGLE_LADDER 0 vs auto at the field recipe (descriptive)
| arm | VERDICT | s/step (11..N) | device ms / profiled step | busy_t | aten::bmm CPU us / call | laddered calls | peak GB | held-out 0 / N |
|---|---|---|---|---|---|---|---|---|
| e4b/fused_attn4_shipped_l0 | VALID | 2.991 | 1327.4 | 0.444 | 24.4 | 0 | 23.321 | 1.95917 / 0.75653 |
| e4b/fused_attn4_shipped_la | VALID | 2.983 | 1330.3 | 0.446 | 24.4 | 0 | 23.321 | 1.95917 / 0.75692 |
| e4b/fused_attn4_m_l0 | VALID | 3.463 | 1703.2 | 0.492 | 171.8 | 0 | 26.141 | 1.95917 / 0.75434 |
| e4b/fused_attn4_m_la | VALID | 3.033 | 1777.3 | 0.586 | 21.0 | 49152 | 26.495 | 1.95917 / 0.75576 |
| e4b/fused_attn4_m_la_d2 | VALID | 3.024 | 1777.7 | 0.588 | 21.1 | 49152 | 26.495 | 1.95917 / 0.75989 |
| e4b/fused_attn4_m_l0_d2 | VALID | 3.718 | 1706.3 | 0.459 | 178.4 | 0 | 26.159 | 1.95917 / 0.75862 |
| e4b/fused_attn4_shipped_la_d2 | VALID | 3.279 | 1332.6 | 0.406 | 28.9 | 0 | 23.321 | 1.95917 / 0.75686 |
| e4b/fused_attn4_shipped_l0_d2 | VALID | 3.289 | 1331.2 | 0.405 | 30.2 | 0 | 23.321 | 1.95917 / 0.75438 |

## Predictions P215-P219 (TC1-PREREG amendment 71: NF4_QLORA_SINGLE_LADDER=auto at the field recipe; scored mechanically)
| prediction | family | verdict | evidence |
|---|---|---|---|
| P215 | qwen3slauto | **UNTESTED** | la / l0: two stable VALID draws a side are registered -- l0 UNSTABLE: draws 3.463 / 3.718 s/step differ by 7.1% > 5% (UNSTABLE: reported, not quoted); matched l0 busy_t 0.475: a host-bound box |
| P216 | qwen3slauto | **UNTESTED** | la / l0: two stable VALID draws a side are registered -- la UNSTABLE: draws 2.983 / 3.279 s/step differ by 9.5% > 5% (UNSTABLE: reported, not quoted); l0 UNSTABLE: draws 2.991 / 3.289 s/step differ by 9.5% > 5% (UNSTABLE: reported, not quoted) |
| P217 | qwen3slauto | **HELD** | aten::bmm CPU self per call 175.1 -> 21.1 us = 0.120 vs <= 0.5 (matched) |
| P218 | qwen3slauto | **HELD** | device ms per profiled step la / l0: m: 1704.8 -> 1777.5 ms = 1.043 (<= 1.06); shipped: 1329.3 -> 1331.5 ms = 1.002 (in [0.98, 1.02]) |
| P219 | qwen3slauto | **HELD** | m: step 0 +0.00000, +0.00000; N +0.00134; shipped: step 0 +0.00000, +0.00000; N +0.00144 (step 0 |.| <= 0.0005, N |.| <= 0.005) |

## Load gate (TC1-PREREG amendment 33): 7 draw(s) voided for host load and run again
Each line: the arm, the attempt, the median host load1 over that attempt's run, the gate. A VOID attempt's files are in loadvoid/; the last attempt of each arm stands whatever its load.

- `qwen3slauto/e4b/fused_attn4_shipped_l0 attempt 0 load1_median 4.42 gate 6.0 status ok over 0`
- `qwen3slauto/e4b/fused_attn4_shipped_la attempt 0 load1_median 4.43 gate 6.0 status ok over 0`
- `qwen3slauto/e4b/fused_attn4_m_l0 attempt 0 load1_median 6.8 gate 6.0 status ok over 1`
- `qwen3slauto/e4b/fused_attn4_m_l0 attempt 0 VOID (host load1 median 6.8 > 6.0): re-run 1 of 2`
- `qwen3slauto/e4b/fused_attn4_m_l0 attempt 1 load1_median 5.0 gate 6.0 status ok over 0`
- `qwen3slauto/e4b/fused_attn4_m_la attempt 0 load1_median 3.57 gate 6.0 status ok over 0`
- `qwen3slauto/e4b/fused_attn4_m_la_d2 attempt 0 load1_median 3.43 gate 6.0 status ok over 0`
- `qwen3slauto/e4b/fused_attn4_m_l0_d2 attempt 0 load1_median 9.01 gate 6.0 status ok over 1`
- `qwen3slauto/e4b/fused_attn4_m_l0_d2 attempt 0 VOID (host load1 median 9.01 > 6.0): re-run 1 of 2`
- `qwen3slauto/e4b/fused_attn4_m_l0_d2 attempt 1 load1_median 15.16 gate 6.0 status ok over 1`
- `qwen3slauto/e4b/fused_attn4_m_l0_d2 attempt 1 VOID (host load1 median 15.16 > 6.0): re-run 2 of 2`
- `qwen3slauto/e4b/fused_attn4_m_l0_d2 attempt 2 load1_median 13.61 gate 6.0 status ok over 1`
- `qwen3slauto/e4b/fused_attn4_shipped_la_d2 attempt 0 load1_median 14.16 gate 6.0 status ok over 1`
- `qwen3slauto/e4b/fused_attn4_shipped_la_d2 attempt 0 VOID (host load1 median 14.16 > 6.0): re-run 1 of 2`
- `qwen3slauto/e4b/fused_attn4_shipped_la_d2 attempt 1 load1_median 16.74 gate 6.0 status ok over 1`
- `qwen3slauto/e4b/fused_attn4_shipped_la_d2 attempt 1 VOID (host load1 median 16.74 > 6.0): re-run 2 of 2`
- `qwen3slauto/e4b/fused_attn4_shipped_la_d2 attempt 2 load1_median 13.01 gate 6.0 status ok over 1`
- `qwen3slauto/e4b/fused_attn4_shipped_l0_d2 attempt 0 load1_median 17.19 gate 6.0 status ok over 1`
- `qwen3slauto/e4b/fused_attn4_shipped_l0_d2 attempt 0 VOID (host load1 median 17.19 > 6.0): re-run 1 of 2`
- `qwen3slauto/e4b/fused_attn4_shipped_l0_d2 attempt 1 load1_median 12.71 gate 6.0 status ok over 1`
- `qwen3slauto/e4b/fused_attn4_shipped_l0_d2 attempt 1 VOID (host load1 median 12.71 > 6.0): re-run 2 of 2`
- `qwen3slauto/e4b/fused_attn4_shipped_l0_d2 attempt 2 load1_median 14.1 gate 6.0 status ok over 1`
