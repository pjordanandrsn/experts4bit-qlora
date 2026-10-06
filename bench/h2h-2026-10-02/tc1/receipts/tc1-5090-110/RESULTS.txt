# TC1 — Qwen3-30B-A3B on one RTX 5090: e4b vs Unsloth vs HF+PEFT vs axolotl at matched work, init and adapter precision (tc1)
Rule (tc1/TC1-PREREG.md): status per attempt in the vocabulary OK / REFUSED / OOM / INSTALL_FAILED / LOAD_FAULT / HARNESS_ERROR / ALARM / NOT_RUN; VALID/VOID per tp4's predicates plus the matched-set predicates (R3: matched init complete, fp32 adapters, step-0 held-out within 0.005 of e4b/reference_attn4_m); VERDICT exactly one of VALID / VOID / QUALITY_FAIL / OOM / UNSUPPORTED / HARNESS_ERROR / ALARM / NOT_RUN (QUALITY_FAIL = held-out |Δ| at N > 0.05 vs e4b/fused_attn4_m); positions = s/step ratios other/e4b from the medians over both draws (stability |d1-d2|/mean ≤ 5 % e4b / 10 % others, else UNSTABLE), quoted only when both arms are VALID; equivalence vs e4b/reference_attn4_m EQUIVALENT ≤ 0.02 / COMPARABLE ≤ 0.05 / else DIVERGENT; frozen base SAME-BYTES / DIFFERENT / N-A per slot; predictions P1–P10 (+ P1b) scored HELD / FALSIFIED / UNTESTED. VOID never enters a ratio or an equivalence reading. Nothing is licensed; no cross-box number is divided into these.
`versions.txt`
```
e4b 0.48.0 @3c8793f99a26c983046e806315eb19ed917d91ab (GitHub main)
gnf4 0.42.0 @427a771c3e8147f13ffe29bb4cb3528cabe38e9c (GitHub main)
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
e4b(t212) 0.48.0 @3c8793f99a26c983046e806315eb19ed917d91ab
gnf4(t212) 0.42.0 @427a771c3e8147f13ffe29bb4cb3528cabe38e9c
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
 "run_id": "tc1-5090-110",
 "instance_id": "54522186",
 "gpu": "NVIDIA GeForce RTX 5090",
 "driver": "590.48.01",
 "cpu": "AMD Ryzen Threadripper PRO 3955WX 16-Cores",
 "nproc": 32,
 "mem_total_kb": "263771780",
 "cgroup_memory_max": "183350853632",
 "disk_root": "overlay         320G   48M  320G   1% /",
 "hostname": "68244cb013c0",
 "cgroup_cpu_max": "1536000 100000",
 "affinity_cpus": 32,
 "cgroup_cpuset_effective": "0-31",
 "registered_gpu_class": "5090",
 "prereg": "tc1/TC1-PREREG.md",
 "local_box": null,
 "local_snapshot": null
}
```

### Qwen3-30B-A3B (amendment 53: e4b's matched arm at its defaults on packed rows, profiled -- torch 2.12 and torch 2.8, and torch 2.8 with the single block) (`qwen3prof28`, registered n_layers 48, attention census 192)
- model `Qwen/Qwen3-30B-A3B` @ `ad44e777bcd1`; tokens sha `d2a501eba57d`; N=40; fixture template alpaca seq 4096 (packed rows, amendment 39) micro-batch 1 × accum 4 lr 0.0002 r 16 α 16 optimizer adamw_8bit(lr=0.0002, weight_decay=0.001) schedule=linear warmup_steps=5 autocast False; e4b trainable 642514944; box_class RTX 5090 gpu NVIDIA GeForce RTX 5090
| framework | arm | status | validity | **VERDICT** | matched | init / dtype | step-0 class | N | s/step med(11+) | tok/s | peak GB | J/step | train first→last | held-out 0→final | quality Δ | engagement | trainable | regime | reason / why |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| e4b | fused_attn4_m_q212 | **OK** | VALID | **VALID** | yes | matched:3407 (complete 12480/12480) / float32 | SAME-BYTES-CLASS (0.0000) | 40 | 9.924 | 1617.0 | 28.228 | 3268.5 | 1.2644→0.9128 | 1.2893→0.9542 | 0.0000 | patched 48 / kcalls 768 | 642514944 | 4-bit experts (e4b NF4) + NF4 attention |  |
| e4b | fused_attn4_m_q28 | **OK** | VALID | **VALID** | yes | matched:3407 (complete 12480/12480) / float32 | SAME-BYTES-CLASS (0.0021) | 40 | 12.562 | 1254.7 | 28.178 | 3797.4 | 1.2594→0.9131 | 1.2914→0.9544 | 0.0001 | patched 48 / kcalls 768 | 642514944 | 4-bit experts (e4b NF4) + NF4 attention |  |
| e4b | fused_attn4_m_q28k0 | **OK** | VALID | **VALID** | yes | matched:3407 (complete 12480/12480) / float32 | SAME-BYTES-CLASS (0.0021) | 40 | 12.337 | 1288.3 | 32.485 | 4465.7 | 1.2594→0.9126 | 1.2914→0.9541 | -0.0001 | patched 48 / kcalls 768 | 642514944 | 4-bit experts (e4b NF4) + NF4 attention |  |
| e4b | fused_attn4_m_q28k0_d2 | **OK** | VALID | **VALID** | yes | matched:3407 (complete 12480/12480) / float32 | SAME-BYTES-CLASS (0.0021) | 40 | 12.375 | 1286.1 | 32.402 | 4478.2 | 1.2594→0.9126 | 1.2914→0.9545 | 0.0003 | patched 48 / kcalls 768 | 642514944 | 4-bit experts (e4b NF4) + NF4 attention |  |
| e4b | fused_attn4_m_q28_d2 | **OK** | VALID | **VALID** | yes | matched:3407 (complete 12480/12480) / float32 | SAME-BYTES-CLASS (0.0021) | 40 | 12.535 | 1266.6 | 28.178 | 3717.7 | 1.2594→0.9130 | 1.2914→0.9542 | -0.0000 | patched 48 / kcalls 768 | 642514944 | 4-bit experts (e4b NF4) + NF4 attention |  |
| e4b | fused_attn4_m_q212_d2 | **OK** | VALID | **VALID** | yes | matched:3407 (complete 12480/12480) / float32 | SAME-BYTES-CLASS (0.0000) | 40 | 9.914 | 1632.5 | 28.228 | 3116.8 | 1.2644→0.9132 | 1.2893→0.9543 | 0.0001 | patched 48 / kcalls 768 | 642514944 | 4-bit experts (e4b NF4) + NF4 attention |  |
- prologue `e4b/fused_attn4_m_q212` **99.4 s** before step 1 (13% of the arm): c1_before 54.1, load_weights 17.3, eval0 11.7, c1_after 10.5; unattributed 1.826; budget 1260.0
- prologue `e4b/fused_attn4_m_q28` **107.5 s** before step 1 (13% of the arm): c1_before 55.5, load_weights 19.8, eval0 15.2, c1_after 10.6; unattributed 1.283; budget 1260.0
- prologue `e4b/fused_attn4_m_q28k0` **100.3 s** before step 1 (13% of the arm): c1_before 56.3, load_weights 19.4, c1_after 10.6, eval0 7.5; unattributed 1.129; budget 1260.0
- prologue `e4b/fused_attn4_m_q28k0_d2` **98.6 s** before step 1 (13% of the arm): c1_before 55.7, load_weights 19.6, c1_after 10.5, eval0 7.5; unattributed 1.126; budget 1260.0
- prologue `e4b/fused_attn4_m_q28_d2` **99.2 s** before step 1 (12% of the arm): c1_before 55.9, load_weights 19.5, c1_after 10.5, eval0 8.0; unattributed 1.115; budget 1260.0
- prologue `e4b/fused_attn4_m_q212_d2` **95.8 s** before step 1 (13% of the arm): c1_before 55.5, load_weights 17.6, c1_after 10.6, eval0 6.4; unattributed 1.879; budget 1260.0
- draws (R1): `e4b/fused_attn4_m_q212` STABLE (9.924/9.914 s, |Δ|/mean 0.1% vs 5%); `e4b/fused_attn4_m_q28` STABLE (12.562/12.535 s, |Δ|/mean 0.2% vs 5%); `e4b/fused_attn4_m_q28k0` STABLE (12.337/12.375 s, |Δ|/mean 0.3% vs 5%)
- e4b internal parity: NO-ARM
- **NO MATCHED POSITION QUOTED (unsloth)** — not quoted: e4b STABLE / unsloth no receipt
- **NO MATCHED POSITION QUOTED (hf)** — not quoted: e4b STABLE / hf no receipt
- **NO MATCHED POSITION QUOTED (axolotl)** — not quoted: e4b STABLE / axolotl no receipt
- equivalence vs `e4b/fused_attn4_m` (I/H; EQUIVALENT iff both ≤ band = max(0.005, 3 × fused-vs-reference |Δ|) = none (no reference arm: COMPARABLE is the ceiling); COMPARABLE ≤ 0.05; draw-noise floor 0.0004): `e4b/fused_attn4_m_q28` **COMPARABLE** (median step |Δ| 0.0003, |Δ held-out at N| 0.0001, step-0 0.0021 SAME-BYTES-CLASS, |Δ loss at step 2| 0.0000, paired rows mean +0.0001 ± 0.0004 SE over 8, favouring arm 5 / anchor 3) — no reference arm ran: the EQUIVALENT band cannot be set, COMPARABLE is the ceiling; `e4b/fused_attn4_m_q28k0` **COMPARABLE** (median step |Δ| 0.0003, |Δ held-out at N| 0.0001, step-0 0.0021 SAME-BYTES-CLASS, |Δ loss at step 2| 0.0022, paired rows mean -0.0001 ± 0.0003 SE over 8, favouring arm 5 / anchor 3) — no reference arm ran: the EQUIVALENT band cannot be set, COMPARABLE is the ceiling; `e4b/fused_attn4_m_q28k0_d2` **COMPARABLE** (median step |Δ| 0.0003, |Δ held-out at N| 0.0003, step-0 0.0021 SAME-BYTES-CLASS, |Δ loss at step 2| 0.0019, paired rows mean +0.0003 ± 0.0003 SE over 8, favouring arm 2 / anchor 6) — no reference arm ran: the EQUIVALENT band cannot be set, COMPARABLE is the ceiling; `e4b/fused_attn4_m_q28_d2` **COMPARABLE** (median step |Δ| 0.0003, |Δ held-out at N| 0.0000, step-0 0.0021 SAME-BYTES-CLASS, |Δ loss at step 2| 0.0045, paired rows mean -0.0000 ± 0.0002 SE over 8, favouring arm 5 / anchor 3) — no reference arm ran: the EQUIVALENT band cannot be set, COMPARABLE is the ceiling; `e4b/fused_attn4_m_q212_d2` **COMPARABLE** (median step |Δ| 0.0004, |Δ held-out at N| 0.0001, step-0 0.0000 SAME-BYTES-CLASS, |Δ loss at step 2| 0.0014, paired rows mean +0.0001 ± 0.0002 SE over 8, favouring arm 3 / anchor 5) — no reference arm ran: the EQUIVALENT band cannot be set, COMPARABLE is the ceiling
- matched_init_sha (B, name-free, canonical slot order): anchor `f7832488926eda91`; `e4b/fused_attn4_m_q212` same; `e4b/fused_attn4_m_q28` same; `e4b/fused_attn4_m_q28k0` same; `e4b/fused_attn4_m_q28k0_d2` same; `e4b/fused_attn4_m_q28_d2` same; `e4b/fused_attn4_m_q212_d2` same
- frozen base (R5), anchor `e4b/fused_attn4_m`: gate_up nf4/64 sha 8c3b3fd78e89 control detects, down nf4/64 sha 59569d810a1a control detects, q_proj nf4/64+dq sha 5c5e402ce51d control detects
- frozen base `e4b/fused_attn4_m_q28`: gate_up **SAME-BYTES** (nf4/64) control detects, down **SAME-BYTES** (nf4/64) control detects, q_proj **SAME-BYTES** (nf4/64+dq) control detects
- frozen base `e4b/fused_attn4_m_q28k0`: gate_up **SAME-BYTES** (nf4/64) control detects, down **SAME-BYTES** (nf4/64) control detects, q_proj **SAME-BYTES** (nf4/64+dq) control detects
- frozen base `e4b/fused_attn4_m_q28k0_d2`: gate_up **SAME-BYTES** (nf4/64) control detects, down **SAME-BYTES** (nf4/64) control detects, q_proj **SAME-BYTES** (nf4/64+dq) control detects
- frozen base `e4b/fused_attn4_m_q28_d2`: gate_up **SAME-BYTES** (nf4/64) control detects, down **SAME-BYTES** (nf4/64) control detects, q_proj **SAME-BYTES** (nf4/64+dq) control detects
- frozen base `e4b/fused_attn4_m_q212_d2`: gate_up **SAME-BYTES** (nf4/64) control detects, down **SAME-BYTES** (nf4/64) control detects, q_proj **SAME-BYTES** (nf4/64+dq) control detects

## Verdicts
| family | arm | VERDICT | validity | quality | note |
|---|---|---|---|---|---|
| qwen3prof28 | e4b/fused_attn4_m_q212 | **VALID** | VALID | 0.0000 |  |
| qwen3prof28 | e4b/fused_attn4_m_q28 | **VALID** | VALID | 0.0001 |  |
| qwen3prof28 | e4b/fused_attn4_m_q28k0 | **VALID** | VALID | -0.0001 |  |
| qwen3prof28 | e4b/fused_attn4_m_q28k0_d2 | **VALID** | VALID | 0.0003 |  |
| qwen3prof28 | e4b/fused_attn4_m_q28_d2 | **VALID** | VALID | -0.0000 |  |
| qwen3prof28 | e4b/fused_attn4_m_q212_d2 | **VALID** | VALID | 0.0001 |  |

## Amendment 53: e4b's matched arm at its defaults, profiled -- torch 2.12 vs torch 2.8 (descriptive)
| arm | VERDICT | s/step (timed, 11..N) | profiled | wall ms/step (profiled) | device ms/step | device busy | device events/step | CPU ops/step |
|---|---|---|---|---|---|---|---|---|
| fused_attn4_m_q212 | VALID | 9.924 | True | 10648.0 | 9616.1 | 0.9031 | 137438 | 963645 |
| fused_attn4_m_q28 | VALID | 12.562 | True | 14733.4 | 10669.4 | 0.7242 | 137656 | 962394 |
| fused_attn4_m_q28k0 | VALID | 12.337 | True | 12830.4 | 11919.5 | 0.929 | 111643 | 697181 |
| fused_attn4_m_q28k0_d2 | VALID | 12.375 | True | 12858.4 | 11927.0 | 0.9276 | 112823 | 703776 |
| fused_attn4_m_q28_d2 | VALID | 12.535 | True | 14772.3 | 10665.3 | 0.722 | 137777 | 963712 |
| fused_attn4_m_q212_d2 | VALID | 9.914 | True | 10611.9 | 9597.3 | 0.9044 | 137438 | 963761 |
- cpu ms per profiled step by family, q28 - q212 (median of two draws, largest increase first): matmul +4816.5, fused_kernel +21.8, optimizer +1.4, routing -13.8, norm_act -37.1, autograd -79.0, other -99.2, memcpy -512.8
- cpu ms per profiled step by family, q28 - q28k0 (median of two draws, largest increase first): matmul +2959.5, autograd +293.1, other +225.1, norm_act +168.4, fused_kernel +28.2, routing +2.1, optimizer +1.8, memcpy -2001.5
- device ms per profiled step by family, q28 - q212 (median of two draws, largest increase first): fused_kernel +988.5, matmul +67.8, other +29.0, autograd +0.6, routing -0.6, optimizer -2.4, memcpy -10.4, norm_act -11.2
- device ms per profiled step by family, q28 - q28k0 (median of two draws, largest increase first): optimizer +2.0, autograd +0.7, norm_act -1.0, routing -1.3, fused_kernel -10.5, memcpy -13.6, other -20.2, matmul -1210.9

## Predictions P134 / P135 / P136 (TC1-PREREG amendment 53: where torch 2.8's extra time goes at e4b's defaults on packed rows; scored mechanically)
| prediction | family | verdict | evidence |
|---|---|---|---|
| P134 | qwen3prof28 | **HELD** | environment ratio q212 / q28 0.790 [0.789, 0.792 over 4 cross-draw ratios] vs <= 0.92; s/step q212 9.924 / 9.914, q28 12.562 / 12.535 |
| P135 | qwen3prof28 | **HELD** | timed ms/step q28 12548.5 vs q212 9918.9 (D +2629.6); device/step 10667.3 vs 9606.7 ms (+1060.6); share not device time 0.597 vs >= 0.5; profiled wall/step 14752.8 vs 10630.0 (gap +4122.9, reported); largest CPU-self increases (ms/step): matmul +4816.5, fused_kernel +21.8, optimizer +1.4 |
| P136 | qwen3prof28 | **HELD** | device busy fraction (device / timed step) q28 0.850 vs q28k0 0.965 (drop +0.115 vs >= 0.03); timed ms/step 12548.5 vs 12356.1, device/step 10667.3 vs 11923.2 ms; against the profiled wall (reported) 0.723 vs 0.928 |

## Load gate (TC1-PREREG amendment 33): 0 draw(s) voided for host load and run again
Each line: the arm, the attempt, the median host load1 over that attempt's run, the gate. A VOID attempt's files are in loadvoid/; the last attempt of each arm stands whatever its load.

- `qwen3prof28/e4b/fused_attn4_m_q212 attempt 0 load1_median 1.96 gate 6.0 status ok over 0`
- `qwen3prof28/e4b/fused_attn4_m_q28 attempt 0 load1_median 2.07 gate 6.0 status ok over 0`
- `qwen3prof28/e4b/fused_attn4_m_q28k0 attempt 0 load1_median 2.12 gate 6.0 status ok over 0`
- `qwen3prof28/e4b/fused_attn4_m_q28k0_d2 attempt 0 load1_median 2.17 gate 6.0 status ok over 0`
- `qwen3prof28/e4b/fused_attn4_m_q28_d2 attempt 0 load1_median 1.88 gate 6.0 status ok over 0`
- `qwen3prof28/e4b/fused_attn4_m_q212_d2 attempt 0 load1_median 1.97 gate 6.0 status ok over 0`
