# TC1 — Qwen3-30B-A3B on one RTX 5090: e4b vs Unsloth vs HF+PEFT vs axolotl at matched work, init and adapter precision (tc1)
Rule (tc1/TC1-PREREG.md): status per attempt in the vocabulary OK / REFUSED / OOM / INSTALL_FAILED / LOAD_FAULT / HARNESS_ERROR / ALARM / NOT_RUN; VALID/VOID per tp4's predicates plus the matched-set predicates (R3: matched init complete, fp32 adapters, step-0 held-out within 0.005 of e4b/reference_attn4_m); VERDICT exactly one of VALID / VOID / QUALITY_FAIL / OOM / UNSUPPORTED / HARNESS_ERROR / ALARM / NOT_RUN (QUALITY_FAIL = held-out |Δ| at N > 0.05 vs e4b/fused_attn4_m); positions = s/step ratios other/e4b from the medians over both draws (stability |d1-d2|/mean ≤ 5 % e4b / 10 % others, else UNSTABLE), quoted only when both arms are VALID; equivalence vs e4b/reference_attn4_m EQUIVALENT ≤ 0.02 / COMPARABLE ≤ 0.05 / else DIVERGENT; frozen base SAME-BYTES / DIFFERENT / N-A per slot; predictions P1–P10 (+ P1b) scored HELD / FALSIFIED / UNTESTED. VOID never enters a ratio or an equivalence reading. Nothing is licensed; no cross-box number is divided into these.
`versions.txt`
```
e4b 0.48.0 @aeb5bd62ed9b018c2ca1c9bbb1906e8f131d2b3b (GitHub main)
gnf4 0.42.0 @71847e5fc1ab17af083bcf5d9992793f1ffd573c (GitHub main)
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
e4b(t212) 0.48.0 @aeb5bd62ed9b018c2ca1c9bbb1906e8f131d2b3b
gnf4(t212) 0.42.0 @71847e5fc1ab17af083bcf5d9992793f1ffd573c
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
 "run_id": "tc1-5090-111",
 "instance_id": "54537794",
 "gpu": "NVIDIA GeForce RTX 5090",
 "driver": "595.91.07",
 "cpu": "AMD Ryzen Threadripper PRO 7965WX 24-Cores",
 "nproc": 48,
 "mem_total_kb": "263402624",
 "cgroup_memory_max": "183093952512",
 "disk_root": "overlay         320G   56M  320G   1% /",
 "hostname": "74c7552436ab",
 "cgroup_cpu_max": "2304000 100000",
 "affinity_cpus": 48,
 "cgroup_cpuset_effective": "0-47",
 "registered_gpu_class": "5090",
 "prereg": "tc1/TC1-PREREG.md",
 "local_box": null,
 "local_snapshot": null
}
```

### Qwen3-30B-A3B (amendment 54: e4b's matched arm at its defaults in torch 2.8 on packed rows -- the defaults, cuBLASLt's heuristics cache raised, grouped-nf4-gemm's bucket ladder) (`qwen3ladder28`, registered n_layers 48, attention census 192)
- model `Qwen/Qwen3-30B-A3B` @ `ad44e777bcd1`; tokens sha `d2a501eba57d`; N=40; fixture template alpaca seq 4096 (packed rows, amendment 39) micro-batch 1 × accum 4 lr 0.0002 r 16 α 16 optimizer adamw_8bit(lr=0.0002, weight_decay=0.001) schedule=linear warmup_steps=5 autocast False; e4b trainable 642514944; box_class RTX 5090 gpu NVIDIA GeForce RTX 5090
| framework | arm | status | validity | **VERDICT** | matched | init / dtype | step-0 class | N | s/step med(11+) | tok/s | peak GB | J/step | train first→last | held-out 0→final | quality Δ | engagement | trainable | regime | reason / why |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| e4b | fused_attn4_m_c0 | **OK** | VALID | **VALID** | yes | matched:3407 (complete 12480/12480) / float32 | SAME-BYTES-CLASS (0.0000) | 40 | 10.507 | 1539.6 | 28.178 | 4158.0 | 1.2594→0.9129 | 1.2914→0.9540 | 0.0000 | patched 48 / kcalls 768 | 642514944 | 4-bit experts (e4b NF4) + NF4 attention |  |
| e4b | fused_attn4_m_c1 | **OK** | VALID | **VALID** | yes | matched:3407 (complete 12480/12480) / float32 | SAME-BYTES-CLASS (0.0000) | 40 | 10.477 | 1548.1 | 28.178 | 4096.5 | 1.2594→0.9130 | 1.2914→0.9538 | -0.0003 | patched 48 / kcalls 768 | 642514944 | 4-bit experts (e4b NF4) + NF4 attention |  |
| e4b | fused_attn4_m_c2 | **OK** | VALID | **VALID** | yes | matched:3407 (complete 12480/12480) / float32 | SAME-BYTES-CLASS (0.0000) | 40 | 10.584 | 1533.3 | 28.498 | 3839.8 | 1.2594→0.9129 | 1.2914→0.9542 | 0.0002 | patched 48 / kcalls 768 | 642514944 | 4-bit experts (e4b NF4) + NF4 attention |  |
| e4b | fused_attn4_m_c2_d2 | **OK** | VALID | **VALID** | yes | matched:3407 (complete 12480/12480) / float32 | SAME-BYTES-CLASS (0.0000) | 40 | 10.632 | 1476.0 | 28.527 | 4082.2 | 1.2594→0.9127 | 1.2914→0.9540 | -0.0001 | patched 48 / kcalls 768 | 642514944 | 4-bit experts (e4b NF4) + NF4 attention |  |
| e4b | fused_attn4_m_c1_d2 | **OK** | VALID | **VALID** | yes | matched:3407 (complete 12480/12480) / float32 | SAME-BYTES-CLASS (0.0000) | 40 | 10.446 | 1547.6 | 28.178 | 3849.9 | 1.2594→0.9126 | 1.2914→0.9542 | 0.0001 | patched 48 / kcalls 768 | 642514944 | 4-bit experts (e4b NF4) + NF4 attention |  |
| e4b | fused_attn4_m_c0_d2 | **OK** | VALID | **VALID** | yes | matched:3407 (complete 12480/12480) / float32 | SAME-BYTES-CLASS (0.0000) | 40 | 10.492 | 1545.9 | 28.178 | 3978.0 | 1.2594→0.9133 | 1.2914→0.9543 | 0.0003 | patched 48 / kcalls 768 | 642514944 | 4-bit experts (e4b NF4) + NF4 attention |  |
- prologue `e4b/fused_attn4_m_c0` **103.7 s** before step 1 (15% of the arm): c1_before 42.1, load_weights 37.2, eval0 10.9, c1_after 8.2; unattributed 0.953; budget 1260.0
- prologue `e4b/fused_attn4_m_c1` **74.2 s** before step 1 (12% of the arm): c1_before 42.0, load_weights 13.3, c1_after 8.2, eval0 6.4; unattributed 0.785; budget 1260.0
- prologue `e4b/fused_attn4_m_c2` **74.6 s** before step 1 (11% of the arm): c1_before 42.6, load_weights 13.2, c1_after 8.1, eval0 6.4; unattributed 0.8; budget 1260.0
- prologue `e4b/fused_attn4_m_c2_d2` **73.6 s** before step 1 (11% of the arm): c1_before 42.9, load_weights 12.9, c1_after 8.3, eval0 6.4; unattributed 0.776; budget 1260.0
- prologue `e4b/fused_attn4_m_c1_d2` **74.6 s** before step 1 (12% of the arm): c1_before 42.3, load_weights 13.1, c1_after 8.1, eval0 6.4; unattributed 0.77; budget 1260.0
- prologue `e4b/fused_attn4_m_c0_d2` **73.6 s** before step 1 (11% of the arm): c1_before 43.0, c1_after 15.4, load_weights 12.9, eval0 6.3; unattributed 0.789; budget 1260.0
- draws (R1): `e4b/fused_attn4_m_c0` STABLE (10.507/10.492 s, |Δ|/mean 0.1% vs 5%); `e4b/fused_attn4_m_c1` STABLE (10.477/10.446 s, |Δ|/mean 0.3% vs 5%); `e4b/fused_attn4_m_c2` STABLE (10.584/10.632 s, |Δ|/mean 0.5% vs 5%)
- e4b internal parity: NO-ARM
- **NO MATCHED POSITION QUOTED (unsloth)** — not quoted: e4b STABLE / unsloth no receipt
- **NO MATCHED POSITION QUOTED (hf)** — not quoted: e4b STABLE / hf no receipt
- **NO MATCHED POSITION QUOTED (axolotl)** — not quoted: e4b STABLE / axolotl no receipt
- equivalence vs `e4b/fused_attn4_m` (I/H; EQUIVALENT iff both ≤ band = max(0.005, 3 × fused-vs-reference |Δ|) = none (no reference arm: COMPARABLE is the ceiling); COMPARABLE ≤ 0.05; draw-noise floor 0.0004): `e4b/fused_attn4_m_c1` **COMPARABLE** (median step |Δ| 0.0004, |Δ held-out at N| 0.0003, step-0 0.0000 SAME-BYTES-CLASS, |Δ loss at step 2| 0.0064, paired rows mean -0.0003 ± 0.0003 SE over 8, favouring arm 5 / anchor 3) — no reference arm ran: the EQUIVALENT band cannot be set, COMPARABLE is the ceiling; `e4b/fused_attn4_m_c2` **COMPARABLE** (median step |Δ| 0.0002, |Δ held-out at N| 0.0002, step-0 0.0000 SAME-BYTES-CLASS, |Δ loss at step 2| 0.0061, paired rows mean +0.0002 ± 0.0002 SE over 8, favouring arm 4 / anchor 4) — no reference arm ran: the EQUIVALENT band cannot be set, COMPARABLE is the ceiling; `e4b/fused_attn4_m_c2_d2` **COMPARABLE** (median step |Δ| 0.0002, |Δ held-out at N| 0.0001, step-0 0.0000 SAME-BYTES-CLASS, |Δ loss at step 2| 0.0094, paired rows mean -0.0001 ± 0.0004 SE over 8, favouring arm 3 / anchor 5) — no reference arm ran: the EQUIVALENT band cannot be set, COMPARABLE is the ceiling; `e4b/fused_attn4_m_c1_d2` **COMPARABLE** (median step |Δ| 0.0003, |Δ held-out at N| 0.0001, step-0 0.0000 SAME-BYTES-CLASS, |Δ loss at step 2| 0.0061, paired rows mean +0.0001 ± 0.0004 SE over 8, favouring arm 3 / anchor 5) — no reference arm ran: the EQUIVALENT band cannot be set, COMPARABLE is the ceiling; `e4b/fused_attn4_m_c0_d2` **COMPARABLE** (median step |Δ| 0.0002, |Δ held-out at N| 0.0003, step-0 0.0000 SAME-BYTES-CLASS, |Δ loss at step 2| 0.0050, paired rows mean +0.0003 ± 0.0002 SE over 8, favouring arm 2 / anchor 6) — no reference arm ran: the EQUIVALENT band cannot be set, COMPARABLE is the ceiling
- matched_init_sha (B, name-free, canonical slot order): anchor `f7832488926eda91`; `e4b/fused_attn4_m_c0` same; `e4b/fused_attn4_m_c1` same; `e4b/fused_attn4_m_c2` same; `e4b/fused_attn4_m_c2_d2` same; `e4b/fused_attn4_m_c1_d2` same; `e4b/fused_attn4_m_c0_d2` same
- frozen base (R5), anchor `e4b/fused_attn4_m`: gate_up nf4/64 sha 8c3b3fd78e89 control detects, down nf4/64 sha 59569d810a1a control detects, q_proj nf4/64+dq sha 5c5e402ce51d control detects
- frozen base `e4b/fused_attn4_m_c1`: gate_up **SAME-BYTES** (nf4/64) control detects, down **SAME-BYTES** (nf4/64) control detects, q_proj **SAME-BYTES** (nf4/64+dq) control detects
- frozen base `e4b/fused_attn4_m_c2`: gate_up **SAME-BYTES** (nf4/64) control detects, down **SAME-BYTES** (nf4/64) control detects, q_proj **SAME-BYTES** (nf4/64+dq) control detects
- frozen base `e4b/fused_attn4_m_c2_d2`: gate_up **SAME-BYTES** (nf4/64) control detects, down **SAME-BYTES** (nf4/64) control detects, q_proj **SAME-BYTES** (nf4/64+dq) control detects
- frozen base `e4b/fused_attn4_m_c1_d2`: gate_up **SAME-BYTES** (nf4/64) control detects, down **SAME-BYTES** (nf4/64) control detects, q_proj **SAME-BYTES** (nf4/64+dq) control detects
- frozen base `e4b/fused_attn4_m_c0_d2`: gate_up **SAME-BYTES** (nf4/64) control detects, down **SAME-BYTES** (nf4/64) control detects, q_proj **SAME-BYTES** (nf4/64+dq) control detects

## Verdicts
| family | arm | VERDICT | validity | quality | note |
|---|---|---|---|---|---|
| qwen3ladder28 | e4b/fused_attn4_m_c0 | **VALID** | VALID | 0.0000 |  |
| qwen3ladder28 | e4b/fused_attn4_m_c1 | **VALID** | VALID | -0.0003 |  |
| qwen3ladder28 | e4b/fused_attn4_m_c2 | **VALID** | VALID | 0.0002 |  |
| qwen3ladder28 | e4b/fused_attn4_m_c2_d2 | **VALID** | VALID | -0.0001 |  |
| qwen3ladder28 | e4b/fused_attn4_m_c1_d2 | **VALID** | VALID | 0.0001 |  |
| qwen3ladder28 | e4b/fused_attn4_m_c0_d2 | **VALID** | VALID | 0.0003 |  |

## Amendment 54: e4b's matched arm at its defaults in torch 2.8 -- the cuBLASLt cache and the bucket ladder (profiled, descriptive)
| arm | VERDICT | s/step (timed, 11..N) | peak GB | device ms/step (profiled) |
|---|---|---|---|---|
| fused_attn4_m_c0 | VALID | 10.507 | 28.18 | 10402.3 |
| fused_attn4_m_c1 | VALID | 10.477 | 28.18 | 10403.2 |
| fused_attn4_m_c2 | VALID | 10.584 | 28.50 | 10582.5 |
| fused_attn4_m_c2_d2 | VALID | 10.632 | 28.53 | 10645.5 |
| fused_attn4_m_c1_d2 | VALID | 10.446 | 28.18 | 10451.3 |
| fused_attn4_m_c0_d2 | VALID | 10.492 | 28.18 | 10409.8 |
- cpu ms per profiled step by family, c1 - c0 (median of two draws, largest change first): matmul -276.1, memcpy +219.9, autograd +6.8, norm_act +4.4, other +2.4, routing +0.8
- cpu ms per profiled step by family, c2 - c0 (median of two draws, largest change first): matmul -3509.8, memcpy +3331.9, other +109.7, norm_act +31.3, autograd +19.2, optimizer -1.0

## Predictions P137 / P138 / P139 / P140 (TC1-PREREG amendment 54: the cuBLASLt cache and the bucket ladder in torch 2.8; scored mechanically)
| prediction | family | verdict | evidence |
|---|---|---|---|
| P137 | qwen3ladder28 | **FALSIFIED** | c2 / c0 1.010 [1.007, 1.013 over 4 cross-draw ratios] vs <= 0.95; s/step c0 10.507 / 10.492, c2 10.584 / 10.632 |
| P138 | qwen3ladder28 | **FALSIFIED** | c1 / c0 0.996 [0.994, 0.999 over 4 cross-draw ratios] vs <= 0.97; s/step c0 10.507 / 10.492, c1 10.477 / 10.446 |
| P139 | qwen3ladder28 | **HELD** | mean held-out at N c0 0.9542; c1 -0.0002, c2 -0.0001 (|.| <= 0.005) |
| P140 | qwen3ladder28 | **HELD** | matched peak c0 28.178 -> c2 28.513 GB (+0.335 vs <= +1.5) |

## Load gate (TC1-PREREG amendment 33): 0 draw(s) voided for host load and run again
Each line: the arm, the attempt, the median host load1 over that attempt's run, the gate. A VOID attempt's files are in loadvoid/; the last attempt of each arm stands whatever its load.

- `qwen3ladder28/e4b/fused_attn4_m_c0 attempt 0 load1_median 4.56 gate 6.0 status ok over 0`
- `qwen3ladder28/e4b/fused_attn4_m_c1 attempt 0 load1_median 3.81 gate 6.0 status ok over 0`
- `qwen3ladder28/e4b/fused_attn4_m_c2 attempt 0 load1_median 3.55 gate 6.0 status ok over 0`
- `qwen3ladder28/e4b/fused_attn4_m_c2_d2 attempt 0 load1_median 3.18 gate 6.0 status ok over 0`
- `qwen3ladder28/e4b/fused_attn4_m_c1_d2 attempt 0 load1_median 3.73 gate 6.0 status ok over 0`
- `qwen3ladder28/e4b/fused_attn4_m_c0_d2 attempt 0 load1_median 4.52 gate 6.0 status ok over 0`
