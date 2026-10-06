# TC1 — Qwen3-30B-A3B on one RTX 5090: e4b vs Unsloth vs HF+PEFT vs axolotl at matched work, init and adapter precision (tc1-5090-108)
Rule (tc1/TC1-PREREG.md): status per attempt in the vocabulary OK / REFUSED / OOM / INSTALL_FAILED / LOAD_FAULT / HARNESS_ERROR / ALARM / NOT_RUN; VALID/VOID per tp4's predicates plus the matched-set predicates (R3: matched init complete, fp32 adapters, step-0 held-out within 0.005 of e4b/reference_attn4_m); VERDICT exactly one of VALID / VOID / QUALITY_FAIL / OOM / UNSUPPORTED / HARNESS_ERROR / ALARM / NOT_RUN (QUALITY_FAIL = held-out |Δ| at N > 0.05 vs e4b/fused_attn4_m); positions = s/step ratios other/e4b from the medians over both draws (stability |d1-d2|/mean ≤ 5 % e4b / 10 % others, else UNSTABLE), quoted only when both arms are VALID; equivalence vs e4b/reference_attn4_m EQUIVALENT ≤ 0.02 / COMPARABLE ≤ 0.05 / else DIVERGENT; frozen base SAME-BYTES / DIFFERENT / N-A per slot; predictions P1–P10 (+ P1b) scored HELD / FALSIFIED / UNTESTED. VOID never enters a ratio or an equivalence reading. Nothing is licensed; no cross-box number is divided into these.
`versions.txt`
```
e4b 0.48.0 @7c4bdd7494f6c90312cf0f6ca39ca939bc89d983 (GitHub main)
gnf4 0.41.0 @1cea66140e7547025a33c0147e6e51160b9f0651 (GitHub main)
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
e4b(t212) 0.48.0 @7c4bdd7494f6c90312cf0f6ca39ca939bc89d983
gnf4(t212) 0.41.0 @1cea66140e7547025a33c0147e6e51160b9f0651
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
 "run_id": "tc1-5090-108",
 "instance_id": "54484674",
 "gpu": "NVIDIA GeForce RTX 5090",
 "driver": "595.71.05",
 "cpu": "Intel(R) Xeon(R) W-2145 CPU @ 3.70GHz",
 "nproc": 16,
 "mem_total_kb": "131573680",
 "cgroup_memory_max": "129340801024",
 "disk_root": "overlay         320G  9.5M  320G   1% /",
 "hostname": "466beb794f80",
 "cgroup_cpu_max": "1536000 100000",
 "affinity_cpus": 16,
 "cgroup_cpuset_effective": "0-15",
 "registered_gpu_class": "5090",
 "prereg": "tc1/TC1-PREREG.md",
 "local_box": null,
 "local_snapshot": null
}
```

### Qwen3-30B-A3B (amendment 51: the packed 4,096-token regime on one stack at e4b's defaults -- chunked loss and bucketed padding both auto) (`qwen3samestack4kd`, registered n_layers 48, attention census 192)
- model `Qwen/Qwen3-30B-A3B` @ `ad44e777bcd1`; tokens sha `d2a501eba57d`; N=40; fixture template alpaca seq 4096 (packed rows, amendment 39) micro-batch 1 × accum 4 lr 0.0002 r 16 α 16 optimizer adamw_8bit(lr=0.0002, weight_decay=0.001) schedule=linear warmup_steps=5 autocast False; e4b trainable 642514944; box_class RTX 5090 gpu NVIDIA GeForce RTX 5090
| framework | arm | status | validity | **VERDICT** | matched | init / dtype | step-0 class | N | s/step med(11+) | tok/s | peak GB | J/step | train first→last | held-out 0→final | quality Δ | engagement | trainable | regime | reason / why |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| e4b | fused_attn4_m | **OK** | VALID | **VALID** | yes | matched:3407 (complete 12480/12480) / float32 | SAME-BYTES-CLASS (0.0000) | 40 | 10.070 | 1598.9 | 28.228 | 4780.9 | 1.2644→0.9134 | 1.2893→0.9544 | 0.0000 | patched 48 / kcalls 768 | 642514944 | 4-bit experts (e4b NF4) + NF4 attention |  |
| unsloth | ckpt_unsloth_m | **OK** | VALID | **VALID** | yes | matched:3407 (complete 12480/12480) / float32 | SAME-BYTES-CLASS (0.0022) | 40 | 14.619 | 1103.4 | 24.864 | 4575.0 | 1.2587→0.9130 | 1.2871→0.9543 | -0.0001 | stacks 96 / fwd 384 / u8 48 | 642514944 | 4-bit expert stacks + bnb-4bit attention (Params4bit stacks 96, Linear4bit 384) |  |
| e4b | reference_attn4_m | **NOT_RUN** | — | **NOT_RUN** | yes | — / — | — | 40 | — | — | — | — | —→— | —→— | — | patched — / kcalls — | — | — | skipped by TC1_SKIP |
| e4b | fused_attn4_m_t28 | **OK** | VALID | **VALID** | yes | matched:3407 (complete 12480/12480) / float32 | SAME-BYTES-CLASS (0.0021) | 40 | 13.606 | 1141.7 | 28.178 | 5216.8 | 1.2594→0.9132 | 1.2914→0.9541 | -0.0003 | patched 48 / kcalls 768 | 642514944 | 4-bit experts (e4b NF4) + NF4 attention |  |
| e4b | fused_attn4_m_t28_d2 | **OK** | VALID | **VALID** | yes | matched:3407 (complete 12480/12480) / float32 | SAME-BYTES-CLASS (0.0021) | 40 | 13.636 | 1146.6 | 28.178 | 5245.2 | 1.2594→0.9123 | 1.2914→0.9545 | 0.0001 | patched 48 / kcalls 768 | 642514944 | 4-bit experts (e4b NF4) + NF4 attention |  |
| unsloth | ckpt_unsloth_m_d2 | **OK** | VALID | **VALID** | yes | matched:3407 (complete 12480/12480) / float32 | SAME-BYTES-CLASS (0.0022) | 40 | 14.609 | 1116.3 | 24.864 | 4553.4 | 1.2587→0.9125 | 1.2871→0.9540 | -0.0004 | stacks 96 / fwd 384 / u8 48 | 642514944 | 4-bit expert stacks + bnb-4bit attention (Params4bit stacks 96, Linear4bit 384) |  |
| e4b | fused_attn4_m_d2 | **OK** | VALID | **VALID** | yes | matched:3407 (complete 12480/12480) / float32 | SAME-BYTES-CLASS (0.0000) | 40 | 10.050 | 1614.6 | 28.228 | 4709.8 | 1.2644→0.9132 | 1.2893→0.9542 | -0.0002 | patched 48 / kcalls 768 | 642514944 | 4-bit experts (e4b NF4) + NF4 attention |  |
- prologue `e4b/fused_attn4_m` **149.8 s** before step 1 (26% of the arm): c1_before 95.3, c1_after 47.9, load_weights 20.2, eval0 12.4; unattributed 1.968; budget 1260.0
- prologue `unsloth/ckpt_unsloth_m` **155.5 s** before step 1 (21% of the arm): c1_before 86.9, c1_after 43.2, load_weights 26.1, eval0 21.1; unattributed 9.27; budget 1890.0
- prologue `e4b/fused_attn4_m_t28` **157.0 s** before step 1 (21% of the arm): c1_before 95.5, c1_after 47.9, load_weights 22.9, eval0 17.4; unattributed 1.152; budget 1260.0
- prologue `e4b/fused_attn4_m_t28_d2` **150.9 s** before step 1 (21% of the arm): c1_before 96.1, c1_after 48.0, load_weights 22.9, eval0 9.5; unattributed 1.135; budget 1260.0
- prologue `unsloth/ckpt_unsloth_m_d2` **147.4 s** before step 1 (20% of the arm): c1_before 87.0, c1_after 43.4, load_weights 26.0, eval0 13.1; unattributed 9.262; budget 1890.0
- prologue `e4b/fused_attn4_m_d2` **145.8 s** before step 1 (26% of the arm): c1_before 96.4, c1_after 48.2, load_weights 20.5, eval0 7.0; unattributed 1.94; budget 1260.0
- draws (R1): `e4b/fused_attn4_m` STABLE (10.070/10.050 s, |Δ|/mean 0.2% vs 5%); `unsloth/ckpt_unsloth_m` STABLE (14.619/14.609 s, |Δ|/mean 0.1% vs 5%); `e4b/fused_attn4_m_t28` STABLE (13.606/13.636 s, |Δ|/mean 0.2% vs 5%)
- e4b internal parity: NO-REF — reference arm missing or not OK
- **MATCHED POSITION (micro-batch 1 × accum 4): s/step ratio unsloth/e4b = 1.453** [1.451, 1.455 over 4 cross-draw ratios] (14.614 vs 10.060 s, medians over 2/2 draws; e4b faster per step); peak VRAM unsloth 24.86 vs e4b 28.23 GB (Δ -3.36); J/step unsloth 4564.2 vs e4b 4745.3 (×0.962); tok/s unsloth 1109.8 vs e4b 1606.8; unsloth regime: 4-bit expert stacks + bnb-4bit attention (Params4bit stacks 96, Linear4bit 384)
- quality reading at N=40 (unsloth): held-out e4b 0.9544 / unsloth 0.9543 (Δ -0.0001) → **COMPARABLE** (|Δ| ≤ 0.05 reads COMPARABLE); step-0 Δ -0.0022
- **NO MATCHED POSITION QUOTED (hf)** — not quoted: e4b STABLE / hf no receipt
- **NO MATCHED POSITION QUOTED (axolotl)** — not quoted: e4b STABLE / axolotl no receipt
- equivalence vs `e4b/fused_attn4_m` (I/H; EQUIVALENT iff both ≤ band = max(0.005, 3 × fused-vs-reference |Δ|) = none (no reference arm: COMPARABLE is the ceiling); COMPARABLE ≤ 0.05; draw-noise floor 0.0004): `unsloth/ckpt_unsloth_m` **COMPARABLE** (median step |Δ| 0.0005, |Δ held-out at N| 0.0001, step-0 0.0022 SAME-BYTES-CLASS, |Δ loss at step 2| 0.0046, paired rows mean -0.0002 ± 0.0003 SE over 8, favouring arm 4 / anchor 4) — no reference arm ran: the EQUIVALENT band cannot be set, COMPARABLE is the ceiling; `e4b/reference_attn4_m` **—** — no OK receipt; `e4b/fused_attn4_m_t28` **COMPARABLE** (median step |Δ| 0.0003, |Δ held-out at N| 0.0003, step-0 0.0021 SAME-BYTES-CLASS, |Δ loss at step 2| 0.0055, paired rows mean -0.0003 ± 0.0004 SE over 8, favouring arm 6 / anchor 2) — no reference arm ran: the EQUIVALENT band cannot be set, COMPARABLE is the ceiling; `e4b/fused_attn4_m_t28_d2` **COMPARABLE** (median step |Δ| 0.0004, |Δ held-out at N| 0.0001, step-0 0.0021 SAME-BYTES-CLASS, |Δ loss at step 2| 0.0035, paired rows mean +0.0001 ± 0.0003 SE over 8, favouring arm 4 / anchor 4) — no reference arm ran: the EQUIVALENT band cannot be set, COMPARABLE is the ceiling; `unsloth/ckpt_unsloth_m_d2` **COMPARABLE** (median step |Δ| 0.0005, |Δ held-out at N| 0.0004, step-0 0.0022 SAME-BYTES-CLASS, |Δ loss at step 2| 0.0014, paired rows mean -0.0004 ± 0.0003 SE over 8, favouring arm 5 / anchor 3) — no reference arm ran: the EQUIVALENT band cannot be set, COMPARABLE is the ceiling; `e4b/fused_attn4_m_d2` **COMPARABLE** (median step |Δ| 0.0003, |Δ held-out at N| 0.0002, step-0 0.0000 SAME-BYTES-CLASS, |Δ loss at step 2| 0.0019, paired rows mean -0.0003 ± 0.0003 SE over 8, favouring arm 3 / anchor 5) — no reference arm ran: the EQUIVALENT band cannot be set, COMPARABLE is the ceiling
- matched_init_sha (B, name-free, canonical slot order): anchor `f7832488926eda91`; `e4b/fused_attn4_m` same; `unsloth/ckpt_unsloth_m` same; `e4b/fused_attn4_m_t28` same; `e4b/fused_attn4_m_t28_d2` same; `unsloth/ckpt_unsloth_m_d2` same; `e4b/fused_attn4_m_d2` same
- frozen base (R5), anchor `e4b/fused_attn4_m`: gate_up nf4/64 sha 8c3b3fd78e89 control detects, down nf4/64 sha 59569d810a1a control detects, q_proj nf4/64+dq sha 5c5e402ce51d control detects
- frozen base `unsloth/ckpt_unsloth_m`: gate_up **N-A** (regime differs: anchor nf4/64 vs nf4/64+dq) control detects, down **N-A** (regime differs: anchor nf4/64 vs nf4/64+dq) control detects, q_proj **SAME-BYTES** (nf4/64+dq) control detects
- frozen base `e4b/fused_attn4_m_t28`: gate_up **SAME-BYTES** (nf4/64) control detects, down **SAME-BYTES** (nf4/64) control detects, q_proj **SAME-BYTES** (nf4/64+dq) control detects
- frozen base `e4b/fused_attn4_m_t28_d2`: gate_up **SAME-BYTES** (nf4/64) control detects, down **SAME-BYTES** (nf4/64) control detects, q_proj **SAME-BYTES** (nf4/64+dq) control detects
- frozen base `unsloth/ckpt_unsloth_m_d2`: gate_up **N-A** (regime differs: anchor nf4/64 vs nf4/64+dq) control detects, down **N-A** (regime differs: anchor nf4/64 vs nf4/64+dq) control detects, q_proj **SAME-BYTES** (nf4/64+dq) control detects
- frozen base `e4b/fused_attn4_m_d2`: gate_up **SAME-BYTES** (nf4/64) control detects, down **SAME-BYTES** (nf4/64) control detects, q_proj **SAME-BYTES** (nf4/64+dq) control detects

## Verdicts
| family | arm | VERDICT | validity | quality | note |
|---|---|---|---|---|---|
| qwen3samestack4kd | e4b/fused_attn4_m | **VALID** | VALID | 0.0000 |  |
| qwen3samestack4kd | unsloth/ckpt_unsloth_m | **VALID** | VALID | -0.0001 |  |
| qwen3samestack4kd | e4b/reference_attn4_m | **NOT_RUN** | — | — | skipped by TC1_SKIP |
| qwen3samestack4kd | e4b/fused_attn4_m_t28 | **VALID** | VALID | -0.0003 |  |
| qwen3samestack4kd | e4b/fused_attn4_m_t28_d2 | **VALID** | VALID | 0.0001 |  |
| qwen3samestack4kd | unsloth/ckpt_unsloth_m_d2 | **VALID** | VALID | -0.0004 |  |
| qwen3samestack4kd | e4b/fused_attn4_m_d2 | **VALID** | VALID | -0.0002 |  |

## Predictions P126 / P127 / P128 / P129 (TC1-PREREG amendment 51: the packed 4,096-token regime on one stack at e4b's defaults; scored mechanically)
| prediction | family | verdict | evidence |
|---|---|---|---|
| P126 | qwen3samestack4kd | **HELD** | Unsloth/e4b on one stack 1.453 [1.451, 1.455 over 4 cross-draw ratios] vs [1.25, 1.8]; s/step e4b 10.060 (within 0.2%), Unsloth 14.614 (within 0.1%); peak e4b 28.23 / Unsloth 24.86 GB; held-out at N e4b 0.9544 / Unsloth 0.9543 |
| P127 | qwen3samestack4kd | **FALSIFIED** | e4b matched arm venv-unsloth / venv-e4b 0.739 [0.737, 0.740] vs [0.84, 0.98]; s/step venv-unsloth 10.070 / 10.050, venv-e4b 13.606 / 13.636; amendments 43 / 48: 1.278 chunked without buckets; buckets 0.893 of the step |
| P128 | qwen3samestack4kd | **HELD** | all 4 e4b arms that ran completed resident and VALID: fused_attn4_m VALID peak 28.23 GB; fused_attn4_m_t28 VALID peak 28.18 GB; fused_attn4_m_t28_d2 VALID peak 28.18 GB; fused_attn4_m_d2 VALID peak 28.23 GB |
| P129 | qwen3samestack4kd | **HELD** | e4b matched peak [28.228, 28.228] GB, median 28.228 vs <= 29.5 |

## Load gate (TC1-PREREG amendment 33): 0 draw(s) voided for host load and run again
Each line: the arm, the attempt, the median host load1 over that attempt's run, the gate. A VOID attempt's files are in loadvoid/; the last attempt of each arm stands whatever its load.

- `qwen3samestack4kd/e4b/fused_attn4_m attempt 0 load1_median 1.28 gate 6.0 status ok over 0`
- `qwen3samestack4kd/unsloth/ckpt_unsloth_m attempt 0 load1_median 1.29 gate 6.0 status ok over 0`
- `qwen3samestack4kd/e4b/fused_attn4_m_t28 attempt 0 load1_median 1.31 gate 6.0 status ok over 0`
- `qwen3samestack4kd/e4b/fused_attn4_m_t28_d2 attempt 0 load1_median 1.5 gate 6.0 status ok over 0`
- `qwen3samestack4kd/unsloth/ckpt_unsloth_m_d2 attempt 0 load1_median 1.33 gate 6.0 status ok over 0`
- `qwen3samestack4kd/e4b/fused_attn4_m_d2 attempt 0 load1_median 1.3 gate 6.0 status ok over 0`
