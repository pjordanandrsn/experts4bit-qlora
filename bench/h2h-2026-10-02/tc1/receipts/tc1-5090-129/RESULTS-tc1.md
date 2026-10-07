# TC1 — Qwen3-30B-A3B on one RTX 5090: e4b vs Unsloth vs HF+PEFT vs axolotl at matched work, init and adapter precision (tc1)
Rule (tc1/TC1-PREREG.md): status per attempt in the vocabulary OK / REFUSED / OOM / INSTALL_FAILED / LOAD_FAULT / HARNESS_ERROR / ALARM / NOT_RUN; VALID/VOID per tp4's predicates plus the matched-set predicates (R3: matched init complete, fp32 adapters, step-0 held-out within 0.005 of e4b/reference_attn4_m); VERDICT exactly one of VALID / VOID / QUALITY_FAIL / OOM / UNSUPPORTED / HARNESS_ERROR / ALARM / NOT_RUN (QUALITY_FAIL = held-out |Δ| at N > 0.05 vs e4b/fused_attn4_m); positions = s/step ratios other/e4b from the medians over both draws (stability |d1-d2|/mean ≤ 5 % e4b / 10 % others, else UNSTABLE), quoted only when both arms are VALID; equivalence vs e4b/reference_attn4_m EQUIVALENT ≤ 0.02 / COMPARABLE ≤ 0.05 / else DIVERGENT; frozen base SAME-BYTES / DIFFERENT / N-A per slot; predictions P1–P10 (+ P1b) scored HELD / FALSIFIED / UNTESTED. VOID never enters a ratio or an equivalence reading. Nothing is licensed; no cross-box number is divided into these.
`versions.txt`
```
e4b 0.48.0 @2b8c1d468a481b7ac11b89740db53cef2803cd89 (GitHub main)
gnf4 0.42.0 @0e6bff33a56a0277b39a0560c37e5af9bee23cf6 (GitHub main)
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
e4b(t212) 0.48.0 @2b8c1d468a481b7ac11b89740db53cef2803cd89
gnf4(t212) 0.42.0 @0e6bff33a56a0277b39a0560c37e5af9bee23cf6
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
 "run_id": "tc1-5090-129",
 "instance_id": "54652024",
 "gpu": "NVIDIA GeForce RTX 5090",
 "driver": "595.84",
 "cpu": "AMD Ryzen 9 5900XT 16-Core Processor",
 "nproc": 32,
 "mem_total_kb": "131821072",
 "cgroup_memory_max": "129584070656",
 "disk_root": "overlay         320G  2.6M  320G   1% /",
 "hostname": "9362116443ff",
 "cgroup_cpu_max": "3072000 100000",
 "affinity_cpus": 32,
 "cgroup_cpuset_effective": "0-31",
 "registered_gpu_class": "5090",
 "prereg": "tc1/TC1-PREREG.md",
 "local_box": null,
 "local_snapshot": null
}
```

### Qwen3-30B-A3B (amendment 64: the matched arm on packed rows with Hugging Face's checkpoint, the reentrant checkpoint alone, and with its inputs in host memory) (`qwen3ckptre4k`, registered n_layers 48, attention census 192)
- model `Qwen/Qwen3-30B-A3B` @ `ad44e777bcd1`; tokens sha `d2a501eba57d`; N=40; fixture template alpaca seq 4096 (packed rows, amendment 39) micro-batch 1 × accum 4 lr 0.0002 r 16 α 16 optimizer adamw_8bit(lr=0.0002, weight_decay=0.001) schedule=linear warmup_steps=5 autocast False; e4b trainable 642514944; box_class RTX 5090 gpu NVIDIA GeForce RTX 5090
| framework | arm | status | validity | **VERDICT** | matched | init / dtype | step-0 class | N | s/step med(11+) | tok/s | peak GB | J/step | train first→last | held-out 0→final | quality Δ | engagement | trainable | regime | reason / why |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| e4b | fused_attn4_m_r0 | **OK** | VALID | **VALID** | yes | matched:3407 (complete 12480/12480) / float32 | SAME-BYTES-CLASS (0.0000) | 40 | 9.617 | 1678.0 | 26.577 | 4725.0 | 1.2578→0.9124 | 1.2885→0.9542 | 0.0000 | patched 48 / kcalls 768 | 642514944 | 4-bit experts (e4b NF4) + NF4 attention |  |
| e4b | fused_attn4_m_rr | **OK** | VALID | **VALID** | yes | matched:3407 (complete 12480/12480) / float32 | SAME-BYTES-CLASS (0.0000) | 40 | 9.421 | 1727.7 | 26.584 | 4502.1 | 1.2578→0.9130 | 1.2885→0.9545 | 0.0003 | patched 48 / kcalls 768 | 642514944 | 4-bit experts (e4b NF4) + NF4 attention |  |
| e4b | fused_attn4_m_r1 | **OK** | VALID | **VALID** | yes | matched:3407 (complete 12480/12480) / float32 | SAME-BYTES-CLASS (0.0000) | 40 | 9.645 | 1687.1 | 25.837 | 4478.1 | 1.2578→0.9128 | 1.2885→0.9543 | 0.0001 | patched 48 / kcalls 768 | 642514944 | 4-bit experts (e4b NF4) + NF4 attention |  |
| e4b | fused_attn4_m_r1_d2 | **OK** | VALID | **VALID** | yes | matched:3407 (complete 12480/12480) / float32 | SAME-BYTES-CLASS (0.0000) | 40 | 9.637 | 1688.3 | 25.851 | 4539.2 | 1.2578→0.9132 | 1.2885→0.9544 | 0.0002 | patched 48 / kcalls 768 | 642514944 | 4-bit experts (e4b NF4) + NF4 attention |  |
| e4b | fused_attn4_m_rr_d2 | **OK** | VALID | **VALID** | yes | matched:3407 (complete 12480/12480) / float32 | SAME-BYTES-CLASS (0.0000) | 40 | 9.430 | 1727.0 | 26.577 | 4517.5 | 1.2578→0.9125 | 1.2885→0.9546 | 0.0004 | patched 48 / kcalls 768 | 642514944 | 4-bit experts (e4b NF4) + NF4 attention |  |
| e4b | fused_attn4_m_r0_d2 | **OK** | VALID | **VALID** | yes | matched:3407 (complete 12480/12480) / float32 | SAME-BYTES-CLASS (0.0000) | 40 | 9.615 | 1689.7 | 26.584 | 4472.6 | 1.2578→0.9130 | 1.2885→0.9542 | 0.0000 | patched 48 / kcalls 768 | 642514944 | 4-bit experts (e4b NF4) + NF4 attention |  |
- prologue `e4b/fused_attn4_m_r0` **83.8 s** before step 1 (17% of the arm): c1_before 41.0, c1_after 20.5, load_weights 17.2, eval0 9.8; unattributed 1.514; budget 1890.0
- prologue `e4b/fused_attn4_m_rr` **79.1 s** before step 1 (17% of the arm): c1_before 40.3, c1_after 21.2, load_weights 19.0, eval0 6.0; unattributed 1.533; budget 1890.0
- prologue `e4b/fused_attn4_m_r1` **77.1 s** before step 1 (16% of the arm): c1_before 40.5, c1_after 20.9, load_weights 17.0, eval0 6.0; unattributed 1.529; budget 1890.0
- prologue `e4b/fused_attn4_m_r1_d2` **78.0 s** before step 1 (17% of the arm): c1_before 40.9, c1_after 20.2, load_weights 17.1, eval0 6.0; unattributed 1.556; budget 1890.0
- prologue `e4b/fused_attn4_m_rr_d2` **77.8 s** before step 1 (17% of the arm): c1_before 40.9, c1_after 20.9, load_weights 17.1, eval0 6.0; unattributed 1.535; budget 1890.0
- prologue `e4b/fused_attn4_m_r0_d2` **77.2 s** before step 1 (16% of the arm): c1_before 40.4, c1_after 20.9, load_weights 17.0, eval0 6.0; unattributed 1.528; budget 1890.0
- draws (R1): `e4b/fused_attn4_m_r0` STABLE (9.617/9.615 s, |Δ|/mean 0.0% vs 5%); `e4b/fused_attn4_m_rr` STABLE (9.421/9.430 s, |Δ|/mean 0.1% vs 5%); `e4b/fused_attn4_m_r1` STABLE (9.645/9.637 s, |Δ|/mean 0.1% vs 5%)
- e4b internal parity: NO-ARM
- **NO MATCHED POSITION QUOTED (unsloth)** — not quoted: e4b STABLE / unsloth no receipt
- **NO MATCHED POSITION QUOTED (hf)** — not quoted: e4b STABLE / hf no receipt
- **NO MATCHED POSITION QUOTED (axolotl)** — not quoted: e4b STABLE / axolotl no receipt
- equivalence vs `e4b/fused_attn4_m` (I/H; EQUIVALENT iff both ≤ band = max(0.005, 3 × fused-vs-reference |Δ|) = none (no reference arm: COMPARABLE is the ceiling); COMPARABLE ≤ 0.05; draw-noise floor 0.0002): `e4b/fused_attn4_m_rr` **COMPARABLE** (median step |Δ| 0.0003, |Δ held-out at N| 0.0003, step-0 0.0000 SAME-BYTES-CLASS, |Δ loss at step 2| 0.0044, paired rows mean +0.0003 ± 0.0002 SE over 8, favouring arm 1 / anchor 7) — no reference arm ran: the EQUIVALENT band cannot be set, COMPARABLE is the ceiling; `e4b/fused_attn4_m_r1` **COMPARABLE** (median step |Δ| 0.0003, |Δ held-out at N| 0.0001, step-0 0.0000 SAME-BYTES-CLASS, |Δ loss at step 2| 0.0050, paired rows mean +0.0001 ± 0.0002 SE over 8, favouring arm 3 / anchor 5) — no reference arm ran: the EQUIVALENT band cannot be set, COMPARABLE is the ceiling; `e4b/fused_attn4_m_r1_d2` **COMPARABLE** (median step |Δ| 0.0003, |Δ held-out at N| 0.0002, step-0 0.0000 SAME-BYTES-CLASS, |Δ loss at step 2| 0.0086, paired rows mean +0.0002 ± 0.0003 SE over 8, favouring arm 3 / anchor 5) — no reference arm ran: the EQUIVALENT band cannot be set, COMPARABLE is the ceiling; `e4b/fused_attn4_m_rr_d2` **COMPARABLE** (median step |Δ| 0.0003, |Δ held-out at N| 0.0004, step-0 0.0000 SAME-BYTES-CLASS, |Δ loss at step 2| 0.0054, paired rows mean +0.0004 ± 0.0002 SE over 8, favouring arm 1 / anchor 7) — no reference arm ran: the EQUIVALENT band cannot be set, COMPARABLE is the ceiling; `e4b/fused_attn4_m_r0_d2` **COMPARABLE** (median step |Δ| 0.0003, |Δ held-out at N| 0.0000, step-0 0.0000 SAME-BYTES-CLASS, |Δ loss at step 2| 0.0007, paired rows mean +0.0000 ± 0.0003 SE over 8, favouring arm 5 / anchor 3) — no reference arm ran: the EQUIVALENT band cannot be set, COMPARABLE is the ceiling
- matched_init_sha (B, name-free, canonical slot order): anchor `f7832488926eda91`; `e4b/fused_attn4_m_r0` same; `e4b/fused_attn4_m_rr` same; `e4b/fused_attn4_m_r1` same; `e4b/fused_attn4_m_r1_d2` same; `e4b/fused_attn4_m_rr_d2` same; `e4b/fused_attn4_m_r0_d2` same
- frozen base (R5), anchor `e4b/fused_attn4_m`: gate_up nf4/64+dq sha e2bed944143e control detects, down nf4/64+dq sha f2ade29dafdb control detects, q_proj nf4/64+dq sha 5c5e402ce51d control detects
- frozen base `e4b/fused_attn4_m_rr`: gate_up **SAME-BYTES** (nf4/64+dq) control detects, down **SAME-BYTES** (nf4/64+dq) control detects, q_proj **SAME-BYTES** (nf4/64+dq) control detects
- frozen base `e4b/fused_attn4_m_r1`: gate_up **SAME-BYTES** (nf4/64+dq) control detects, down **SAME-BYTES** (nf4/64+dq) control detects, q_proj **SAME-BYTES** (nf4/64+dq) control detects
- frozen base `e4b/fused_attn4_m_r1_d2`: gate_up **SAME-BYTES** (nf4/64+dq) control detects, down **SAME-BYTES** (nf4/64+dq) control detects, q_proj **SAME-BYTES** (nf4/64+dq) control detects
- frozen base `e4b/fused_attn4_m_rr_d2`: gate_up **SAME-BYTES** (nf4/64+dq) control detects, down **SAME-BYTES** (nf4/64+dq) control detects, q_proj **SAME-BYTES** (nf4/64+dq) control detects
- frozen base `e4b/fused_attn4_m_r0_d2`: gate_up **SAME-BYTES** (nf4/64+dq) control detects, down **SAME-BYTES** (nf4/64+dq) control detects, q_proj **SAME-BYTES** (nf4/64+dq) control detects

## Verdicts
| family | arm | VERDICT | validity | quality | note |
|---|---|---|---|---|---|
| qwen3ckptre4k | e4b/fused_attn4_m_r0 | **VALID** | VALID | 0.0000 |  |
| qwen3ckptre4k | e4b/fused_attn4_m_rr | **VALID** | VALID | 0.0003 |  |
| qwen3ckptre4k | e4b/fused_attn4_m_r1 | **VALID** | VALID | 0.0001 |  |
| qwen3ckptre4k | e4b/fused_attn4_m_r1_d2 | **VALID** | VALID | 0.0002 |  |
| qwen3ckptre4k | e4b/fused_attn4_m_rr_d2 | **VALID** | VALID | 0.0004 |  |
| qwen3ckptre4k | e4b/fused_attn4_m_r0_d2 | **VALID** | VALID | 0.0000 |  |

## Amendment 64: Hugging Face's checkpoint, the reentrant checkpoint alone, and with its inputs in host memory -- packed rows (torch 2.12) and the field recipe in torch 2.8 (descriptive)
| family | arm | VERDICT | s/step (11..N) | run peak GB | train | held-out N | checkpoint |
|---|---|---|---|---|---|---|---|
| qwen3ckptre4k | e4b/fused_attn4_m_r0 | VALID | 9.617 | 26.577 | 26.577 | 0.95417 | 0 [] |
| qwen3ckptre4k | e4b/fused_attn4_m_rr | VALID | 9.421 | 26.584 | 26.584 | 0.95445 | reentrant ['reentrant_checkpoint'] |
| qwen3ckptre4k | e4b/fused_attn4_m_r1 | VALID | 9.645 | 25.837 | 25.837 | 0.9543 | 1 ['offloaded_checkpoint'] |
| qwen3ckptre4k | e4b/fused_attn4_m_r1_d2 | VALID | 9.637 | 25.851 | 25.851 | 0.95439 | 1 ['offloaded_checkpoint'] |
| qwen3ckptre4k | e4b/fused_attn4_m_rr_d2 | VALID | 9.430 | 26.577 | 26.577 | 0.95461 | reentrant ['reentrant_checkpoint'] |
| qwen3ckptre4k | e4b/fused_attn4_m_r0_d2 | VALID | 9.615 | 26.584 | 26.584 | 0.95418 | 0 [] |

## Predictions P178-P183 (TC1-PREREG amendment 64: the reentrant checkpoint as the default checkpoint; scored mechanically)
| prediction | family | verdict | evidence |
|---|---|---|---|
| P178 | qwen3ckptre4k | **HELD** | rr / r0 0.980 [0.980, 0.981 over 4 cross-draw ratios] vs <= 1.01; s/step rr 9.421 / 9.430, r0 9.617 / 9.615 |
| P179 | qwen3ckptre4k | **HELD** | training-phase peak r0 26.581, rr 26.581 GB (rr - r0 +0.000 vs |.| <= 0.05) |
| P180 | qwen3ckptre4k | **FALSIFIED** | r1 / rr 1.023 [1.022, 1.024 over 4 cross-draw ratios] vs <= 1.01; s/step r1 9.645 / 9.637, rr 9.421 / 9.430 |
| P181 | qwen3ckptre4k | **HELD** | mean held-out at N r0 0.95417, rr 0.95453, r1 0.95434; largest difference 0.00035 (<= 0.005) |

## Load gate (TC1-PREREG amendment 33): 0 draw(s) voided for host load and run again
Each line: the arm, the attempt, the median host load1 over that attempt's run, the gate. A VOID attempt's files are in loadvoid/; the last attempt of each arm stands whatever its load.

- `qwen3ckptre4k/e4b/fused_attn4_m_r0 attempt 0 load1_median 1.21 gate 6.0 status ok over 0`
- `qwen3ckptre4k/e4b/fused_attn4_m_rr attempt 0 load1_median 1.29 gate 6.0 status ok over 0`
- `qwen3ckptre4k/e4b/fused_attn4_m_r1 attempt 0 load1_median 1.29 gate 6.0 status ok over 0`
- `qwen3ckptre4k/e4b/fused_attn4_m_r1_d2 attempt 0 load1_median 1.25 gate 6.0 status ok over 0`
- `qwen3ckptre4k/e4b/fused_attn4_m_rr_d2 attempt 0 load1_median 1.26 gate 6.0 status ok over 0`
- `qwen3ckptre4k/e4b/fused_attn4_m_r0_d2 attempt 0 load1_median 1.22 gate 6.0 status ok over 0`
